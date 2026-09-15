import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import tempfile
from pathlib import Path

from src.model import UNet


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"


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
# Load trained model
# --------------------------------------------------

model = UNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Best model loaded successfully.")


# --------------------------------------------------
# Denoising function
# --------------------------------------------------

def denoise_audio(audio_path):

    if audio_path is None:
        return None

    # ----------------------------------------------
    # Load audio
    # ----------------------------------------------

    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    print("Original sample rate:", sample_rate)
    print("Original duration:", len(audio) / sample_rate, "seconds")

    # ----------------------------------------------
    # Resample to 16 kHz
    # ----------------------------------------------

    target_sr = 16000

    if sample_rate != target_sr:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=target_sr
        )

    sample_rate = target_sr

    # ----------------------------------------------
    # STFT settings
    # ----------------------------------------------

    n_fft = 512
    hop_length = 128

    # ----------------------------------------------
    # Process audio in 1-second segments
    # ----------------------------------------------

    segment_length = target_sr

    output_audio = []

    for start in range(
        0,
        len(audio),
        segment_length
    ):

        segment = audio[
            start:start + segment_length
        ]

        original_length = len(segment)

        # Pad last segment if shorter than 1 second
        if original_length < segment_length:

            segment = np.pad(
                segment,
                (0, segment_length - original_length)
            )

        # ------------------------------------------
        # STFT
        # ------------------------------------------

        noisy_stft = librosa.stft(
            segment,
            n_fft=n_fft,
            hop_length=hop_length
        )

        noisy_magnitude = np.abs(noisy_stft)
        noisy_phase = np.angle(noisy_stft)

        # ------------------------------------------
        # Convert to tensor
        # ------------------------------------------

        noisy_tensor = torch.from_numpy(
            noisy_magnitude
        ).float()

        noisy_tensor = noisy_tensor.unsqueeze(0).unsqueeze(0)

        noisy_tensor = noisy_tensor.to(device)

        # ------------------------------------------
        # Model prediction
        # ------------------------------------------

        with torch.no_grad():

            predicted_magnitude = model(
                noisy_tensor
            )

        predicted_magnitude = (
            predicted_magnitude
            .squeeze()
            .cpu()
            .numpy()
        )

        # ------------------------------------------
        # Ensure magnitude is non-negative
        # ------------------------------------------

        predicted_magnitude = np.maximum(
            predicted_magnitude,
            0
        )

        # ------------------------------------------
        # Reconstruct using noisy phase
        # ------------------------------------------

        denoised_stft = (
            predicted_magnitude *
            np.exp(1j * noisy_phase)
        )

        # ------------------------------------------
        # Inverse STFT
        # ------------------------------------------

        denoised_segment = librosa.istft(
            denoised_stft,
            hop_length=hop_length,
            length=segment_length
        )

        # Remove padding from final segment
        denoised_segment = denoised_segment[
            :original_length
        ]

        output_audio.append(
            denoised_segment
        )

    # ----------------------------------------------
    # Combine all segments
    # ----------------------------------------------

    denoised_audio = np.concatenate(
        output_audio
    )

    # ----------------------------------------------
    # Prevent clipping
    # ----------------------------------------------

    max_value = np.max(
        np.abs(denoised_audio)
    )

    if max_value > 1.0:

        denoised_audio = (
            denoised_audio / max_value
        )

    # ----------------------------------------------
    # Save output
    # ----------------------------------------------

    output_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    output_file.close()

    sf.write(
        output_file.name,
        denoised_audio,
        sample_rate
    )

    print("Denoised audio saved:", output_file.name)

    return output_file.name


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(
    title="Audio Denoising"
) as app:

    gr.Markdown(
        """
        # 🎧 Audio Denoising

        ### Remove background noise from your audio
        Upload an audio file and let the trained AI model
        reduce the unwanted noise.
        """
    )

    noisy_audio = gr.Audio(
        label="Upload Noisy Audio",
        type="filepath"
    )

    denoise_button = gr.Button(
        "✨ Denoise Audio",
        variant="primary"
    )

    denoised_audio = gr.Audio(
        label="Denoised Audio",
        type="filepath"
    )

    denoise_button.click(
        fn=denoise_audio,
        inputs=noisy_audio,
        outputs=denoised_audio
    )


# --------------------------------------------------
# Launch
# --------------------------------------------------

app.launch()