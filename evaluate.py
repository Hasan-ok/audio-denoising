import sys
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.model import UNet


# ============================================================
# Configuration
# ============================================================

TARGET_SR = 16000
N_FFT = 512
HOP_LENGTH = 128

CHUNK_SECONDS = 1.0
CHUNK_LENGTH = int(TARGET_SR * CHUNK_SECONDS)

# 50% overlap between chunks
CHUNK_HOP = CHUNK_LENGTH // 2

NUM_TEST_FILES = 10

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

NOISY_DIR = PROJECT_ROOT / "data" / "noisy_test"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean_test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SNR
# ============================================================

def calculate_snr(clean, signal):
    """
    Calculate SNR between the clean reference and a signal.
    """

    noise = clean - signal

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 1e-12:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


# ============================================================
# Denoise one chunk
# ============================================================

def denoise_chunk(model, noisy_audio):
    """
    Denoise one 1-second audio chunk.

    The model works on log-magnitude spectrograms
    and uses the noisy phase for reconstruction.
    """

    # STFT
    noisy_stft = librosa.stft(
        noisy_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Magnitude + phase
    noisy_magnitude = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)

    # Log magnitude
    noisy_log = np.log1p(noisy_magnitude)

    # Convert to tensor
    input_tensor = torch.from_numpy(
        noisy_log.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    # Model inference
    with torch.no_grad():
        prediction = model(input_tensor)

    # Back to NumPy
    predicted_log = prediction.squeeze().cpu().numpy()

    # Prevent invalid values
    predicted_log = np.nan_to_num(
        predicted_log,
        nan=0.0,
        posinf=20.0,
        neginf=0.0
    )

    # Magnitude cannot be negative
    predicted_log = np.maximum(predicted_log, 0.0)

    # Convert log magnitude back
    predicted_magnitude = np.expm1(predicted_log)

    # Reconstruct complex spectrogram
    denoised_stft = (
        predicted_magnitude *
        np.exp(1j * noisy_phase)
    )

    # ISTFT
    denoised_audio = librosa.istft(
        denoised_stft,
        hop_length=HOP_LENGTH,
        length=len(noisy_audio)
    )

    return denoised_audio


# ============================================================
# Denoise complete audio
# ============================================================

def denoise_full_audio(model, noisy_audio):
    """
    Denoise a complete recording using overlapping
    1-second chunks.
    """

    original_length = len(noisy_audio)

    # If audio is shorter than one chunk,
    # pad it to the required size.
    if original_length < CHUNK_LENGTH:
        padded_audio = np.pad(
            noisy_audio,
            (0, CHUNK_LENGTH - original_length)
        )
    else:
        padded_audio = noisy_audio

    padded_length = len(padded_audio)

    output = np.zeros(
        padded_length,
        dtype=np.float32
    )

    weights = np.zeros(
        padded_length,
        dtype=np.float32
    )

    start = 0

    while start < padded_length:

        end = min(
            start + CHUNK_LENGTH,
            padded_length
        )

        # Extract chunk
        chunk = padded_audio[start:end]

        valid_length = len(chunk)

        # Make sure chunk is exactly 1 second
        if valid_length < CHUNK_LENGTH:
            chunk = np.pad(
                chunk,
                (0, CHUNK_LENGTH - valid_length)
            )

        # Denoise
        denoised_chunk = denoise_chunk(
            model,
            chunk
        )

        # Add only the valid part
        output[start:end] += (
            denoised_chunk[:valid_length]
        )

        # Count how many chunks contribute
        weights[start:end] += 1.0

        # Move forward by 50%
        start += CHUNK_HOP

    # Average overlapping regions
    output /= np.maximum(
        weights,
        1.0
    )

    # Remove padding
    output = output[:original_length]

    return output

# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 60)
    print("AUDIO DENOISING EVALUATION")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(f"\nLoading checkpoint:")
    print(CHECKPOINT_PATH)

    if not CHECKPOINT_PATH.exists():
        print("\nERROR: Checkpoint not found.")
        print("Train the model first.")
        return

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model = UNet().to(DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("✓ Model loaded successfully.")

    # --------------------------------------------------------
    # Find test files
    # --------------------------------------------------------

    noisy_files = sorted(NOISY_DIR.glob("*.wav"))

    print(f"\nTotal noisy test files: {len(noisy_files)}")

    if len(noisy_files) == 0:
        print("ERROR: No test files found.")
        return

    test_files = noisy_files[:NUM_TEST_FILES]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    noisy_snrs = []
    denoised_snrs = []

    print(f"\nEvaluating {len(test_files)} test files...")
    print("-" * 60)

    for index, noisy_path in enumerate(test_files, start=1):

        clean_path = CLEAN_DIR / noisy_path.name

        if not clean_path.exists():
            print(
                f"Skipping {noisy_path.name}: "
                "clean file not found."
            )
            continue

        print(
            f"\n[{index}/{len(test_files)}] "
            f"{noisy_path.name}"
        )

        # Load complete recordings
        noisy_audio, _ = librosa.load(
            noisy_path,
            sr=TARGET_SR,
            mono=True
        )

        clean_audio, _ = librosa.load(
            clean_path,
            sr=TARGET_SR,
            mono=True
        )

        # Make lengths identical
        min_length = min(
            len(noisy_audio),
            len(clean_audio)
        )

        noisy_audio = noisy_audio[:min_length]
        clean_audio = clean_audio[:min_length]

        # Calculate noisy SNR
        noisy_snr = calculate_snr(
            clean_audio,
            noisy_audio
        )

        # Denoise complete recording
        denoised_audio = denoise_full_audio(
            model,
            noisy_audio
        )

        # Calculate denoised SNR
        denoised_snr = calculate_snr(
            clean_audio,
            denoised_audio
        )

        improvement = denoised_snr - noisy_snr

        noisy_snrs.append(noisy_snr)
        denoised_snrs.append(denoised_snr)

        # Save output
        output_path = OUTPUT_DIR / (
            noisy_path.stem + "_denoised.wav"
        )

        sf.write(
            output_path,
            denoised_audio,
            TARGET_SR
        )

        print(f"  Noisy SNR:    {noisy_snr:.2f} dB")
        print(f"  Denoised SNR: {denoised_snr:.2f} dB")
        print(f"  Improvement:  {improvement:.2f} dB")
        print(f"  Saved: {output_path.name}")

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    if len(noisy_snrs) == 0:
        print("\nNo files were successfully evaluated.")
        return

    average_noisy = np.mean(noisy_snrs)
    average_denoised = np.mean(denoised_snrs)
    average_improvement = (
        average_denoised - average_noisy
    )

    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"\nAverage noisy SNR:    "
        f"{average_noisy:.2f} dB"
    )

    print(
        f"Average denoised SNR: "
        f"{average_denoised:.2f} dB"
    )

    print(
        f"Average improvement:  "
        f"{average_improvement:.2f} dB"
    )

    print(
        f"\nDenoised files saved to:"
    )
    print(OUTPUT_DIR)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()