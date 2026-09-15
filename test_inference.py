from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from src.model import UNet


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

noisy_path = PROJECT_ROOT / "data" / "noisy_test" / "p232_001.wav"

output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "p232_001_denoised.wav"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = UNet().to(device)

checkpoint = torch.load(
    PROJECT_ROOT / "checkpoints" / "best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Best model loaded.")


# --------------------------------------------------
# Load noisy audio
# --------------------------------------------------

audio, sample_rate = sf.read(noisy_path)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

print("Original sample rate:", sample_rate)
print("Original samples:", len(audio))


# --------------------------------------------------
# Resample
# --------------------------------------------------

target_sr = 16000

if sample_rate != target_sr:
    audio = librosa.resample(
        audio,
        orig_sr=sample_rate,
        target_sr=target_sr
    )

sample_rate = target_sr

print("Using sample rate:", sample_rate)


# --------------------------------------------------
# Take 1-second segment
# --------------------------------------------------

segment_length = target_sr

if len(audio) < segment_length:
    raise ValueError("Audio is shorter than 1 second.")

audio_segment = audio[:segment_length]


# --------------------------------------------------
# STFT
# --------------------------------------------------

n_fft = 512
hop_length = 128

noisy_stft = librosa.stft(
    audio_segment,
    n_fft=n_fft,
    hop_length=hop_length
)

noisy_magnitude = np.abs(noisy_stft)
noisy_phase = np.angle(noisy_stft)


# --------------------------------------------------
# Convert to tensor
# --------------------------------------------------

noisy_tensor = torch.from_numpy(
    noisy_magnitude
).float()

noisy_tensor = noisy_tensor.unsqueeze(0).unsqueeze(0)

noisy_tensor = noisy_tensor.to(device)


# --------------------------------------------------
# Model inference
# --------------------------------------------------

with torch.no_grad():

    predicted_magnitude = model(
        noisy_tensor
    )

predicted_magnitude = predicted_magnitude.squeeze().cpu().numpy()

print("Predicted spectrogram shape:", predicted_magnitude.shape)


# --------------------------------------------------
# Make sure magnitude is non-negative
# --------------------------------------------------

predicted_magnitude = np.maximum(
    predicted_magnitude,
    0
)


# --------------------------------------------------
# Reconstruct complex spectrogram
# using the noisy phase
# --------------------------------------------------

denoised_stft = (
    predicted_magnitude *
    np.exp(1j * noisy_phase)
)


# --------------------------------------------------
# Inverse STFT
# --------------------------------------------------

denoised_audio = librosa.istft(
    denoised_stft,
    hop_length=hop_length,
    length=len(audio_segment)
)


# --------------------------------------------------
# Normalize if necessary
# --------------------------------------------------

max_value = np.max(
    np.abs(denoised_audio)
)

if max_value > 1.0:
    denoised_audio = (
        denoised_audio / max_value
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

sf.write(
    output_path,
    denoised_audio,
    sample_rate
)

print()
print("Denoised audio saved:")
print(output_path)