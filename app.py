import gradio as gr
import spaces
import librosa
import numpy as np
import soundfile as sf
import torch
import tempfile
from pathlib import Path

from src.model import UNet


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

TARGET_SAMPLE_RATE = 16000
MAX_DURATION = 60

N_FFT = 512
HOP_LENGTH = 128
SEGMENT_LENGTH = TARGET_SAMPLE_RATE


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load model
# ============================================================

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


# ============================================================
# Audio information
# ============================================================

def get_audio_info(audio_path):

    if audio_path is None:
        return "Select an audio file to continue."

    try:
        info = sf.info(audio_path)

        duration = info.frames / info.samplerate

        if duration >= 60:
            duration_text = f"{duration:.1f} seconds"
        else:
            duration_text = f"{duration:.2f} seconds"

        return (
            f"**Duration:** {duration_text} &nbsp; • &nbsp; "
            f"**Format:** {info.format}"
        )

    except Exception:
        return "Audio information unavailable."


# ============================================================
# Denoising
# ============================================================

@spaces.GPU
def denoise_audio(audio_path):

    if audio_path is None:
        raise gr.Error(
            "Please upload an audio file first."
        )

    try:

        # ----------------------------------------------------
        # Check duration
        # ----------------------------------------------------

        try:
            info = sf.info(audio_path)

            duration = (
                info.frames / info.samplerate
            )

            if duration > MAX_DURATION:
                raise gr.Error(
                    "This audio file is too long. "
                    "Please use a file shorter than 60 seconds."
                )

        except gr.Error:
            raise

        except Exception:
            pass

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sample_rate = librosa.load(
            audio_path,
            sr=None,
            mono=True
        )

        # ----------------------------------------------------
        # Resample
        # ----------------------------------------------------

        if sample_rate != TARGET_SAMPLE_RATE:

            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE
            )

        sample_rate = TARGET_SAMPLE_RATE

        output_audio = []

        # ----------------------------------------------------
        # Process audio in 1-second segments
        # ----------------------------------------------------

        for start in range(
            0,
            len(audio),
            SEGMENT_LENGTH
        ):

            segment = audio[
                start:start + SEGMENT_LENGTH
            ]

            original_length = len(segment)

            # Pad final segment
            if original_length < SEGMENT_LENGTH:

                segment = np.pad(
                    segment,
                    (
                        0,
                        SEGMENT_LENGTH - original_length
                    )
                )

            # ------------------------------------------------
            # Convert audio to spectrogram
            # ------------------------------------------------

            noisy_stft = librosa.stft(
                segment,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )

            noisy_magnitude = np.abs(noisy_stft)
            noisy_phase = np.angle(noisy_stft)

            # ------------------------------------------------
            # Convert to tensor
            # ------------------------------------------------

            noisy_tensor = torch.from_numpy(
                noisy_magnitude
            ).float()

            noisy_tensor = (
                noisy_tensor
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # ------------------------------------------------
            # Model inference
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Prevent negative values
            # ------------------------------------------------

            predicted_magnitude = np.maximum(
                predicted_magnitude,
                0
            )

            # ------------------------------------------------
            # Reconstruct audio
            # ------------------------------------------------

            denoised_stft = (
                predicted_magnitude
                * np.exp(1j * noisy_phase)
            )

            denoised_segment = librosa.istft(
                denoised_stft,
                hop_length=HOP_LENGTH,
                length=SEGMENT_LENGTH
            )

            # Remove padding
            denoised_segment = denoised_segment[
                :original_length
            ]

            output_audio.append(
                denoised_segment
            )

        # ----------------------------------------------------
        # Combine segments
        # ----------------------------------------------------

        denoised_audio = np.concatenate(
            output_audio
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        max_value = np.max(
            np.abs(denoised_audio)
        )

        if max_value > 1.0:

            denoised_audio = (
                denoised_audio / max_value
            )

        # ----------------------------------------------------
        # Save output
        # ----------------------------------------------------

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

        return output_file.name

    except gr.Error:
        raise

    except Exception as error:

        print("Denoising error:", error)

        raise gr.Error(
            "Something went wrong while processing this audio. "
            "Please try another file."
        )


# ============================================================
# Custom CSS
# ============================================================

custom_css = """

/* ---------------------------------------------------------
   Main container
--------------------------------------------------------- */

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    padding: 25px !important;
}


/* ---------------------------------------------------------
   Background
--------------------------------------------------------- */

body {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(99, 102, 241, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 0%,
            rgba(139, 92, 246, 0.08),
            transparent 30%
        ) !important;
}


/* ---------------------------------------------------------
   Hero
--------------------------------------------------------- */

.hero {
    text-align: center;
    padding: 55px 20px 40px 20px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.12);
    color: #6366f1;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.7px;
    margin-bottom: 18px;
}

.hero h1 {
    font-size: 48px !important;
    font-weight: 800 !important;
    letter-spacing: -1.8px;
    margin: 0 0 12px 0 !important;
}

.hero p {
    font-size: 18px;
    color: #6b7280;
    max-width: 650px;
    margin: auto;
    line-height: 1.6;
}


/* ---------------------------------------------------------
   Cards
--------------------------------------------------------- */

.card {
    border: 1px solid rgba(128, 128, 128, 0.16) !important;
    border-radius: 22px !important;
    padding: 28px !important;
    background: rgba(255, 255, 255, 0.72) !important;
    box-shadow:
        0 12px 40px rgba(0, 0, 0, 0.055) !important;
}


/* ---------------------------------------------------------
   Section titles
--------------------------------------------------------- */

.section-title {
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 6px;
}

.section-description {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 20px;
    line-height: 1.5;
}


/* ---------------------------------------------------------
   Upload area
--------------------------------------------------------- */

.upload-area {
    border-radius: 18px !important;
}


/* ---------------------------------------------------------
   File information
--------------------------------------------------------- */

.file-info {
    margin-top: 12px;
    padding: 12px 15px;
    border-radius: 12px;
    background: rgba(99, 102, 241, 0.06);
    font-size: 13px;
    color: #6b7280;
}


/* ---------------------------------------------------------
   Main button
--------------------------------------------------------- */

.denoise-button {
    height: 58px !important;
    border-radius: 15px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}


/* ---------------------------------------------------------
   Success message
--------------------------------------------------------- */

.success-message {
    padding: 15px 18px;
    border-radius: 14px;
    background: rgba(34, 197, 94, 0.09);
    border: 1px solid rgba(34, 197, 94, 0.18);
    color: #15803d;
    font-weight: 600;
    margin-bottom: 18px;
}


/* ---------------------------------------------------------
   How it works
--------------------------------------------------------- */

.info-card {
    text-align: center;
    padding: 26px 18px;
    border-radius: 18px;
    border: 1px solid rgba(128, 128, 128, 0.15);
    background: rgba(255, 255, 255, 0.50);
    min-height: 150px;
}

.info-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

.info-card h3 {
    margin: 5px 0 8px;
    font-size: 17px;
}

.info-card p {
    color: #6b7280;
    font-size: 13px;
    line-height: 1.5;
}


/* ---------------------------------------------------------
   Footer
--------------------------------------------------------- */

.footer {
    text-align: center;
    padding: 40px 10px 15px;
    color: #6b7280;
    font-size: 13px;
    line-height: 1.7;
}


/* ---------------------------------------------------------
   Dark mode
--------------------------------------------------------- */

.dark .card {
    background: rgba(30, 30, 35, 0.75) !important;
}

.dark .hero p,
.dark .section-description,
.dark .info-card p,
.dark .footer {
    color: #a1a1aa;
}

.dark .info-card {
    background: rgba(30, 30, 35, 0.55);
}

.dark .file-info {
    color: #a1a1aa;
}

.dark .success-message {
    color: #86efac;
}


/* ---------------------------------------------------------
   Mobile
--------------------------------------------------------- */

@media (max-width: 700px) {

    .gradio-container {
        padding: 15px !important;
    }

    .hero {
        padding: 35px 10px 25px;
    }

    .hero h1 {
        font-size: 36px !important;
    }

    .hero p {
        font-size: 16px;
    }

    .card {
        padding: 20px !important;
    }

}

"""


# ============================================================
# Interface
# ============================================================

with gr.Blocks(
    title="AudioClean AI",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate"
    )
) as app:

    # ========================================================
    # HERO
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-badge">
                🎧 AI AUDIO ENHANCEMENT
            </div>

            <h1>
                AudioClean AI
            </h1>

            <p>
                Remove unwanted background noise and
                improve speech clarity with AI.
            </p>

        </div>
        """
    )


    # ========================================================
    # STEP 1 — UPLOAD
    # ========================================================

    with gr.Column(elem_classes="card"):

        gr.HTML(
            """
            <div class="section-title">
                <span style="color:#6366f1;">01</span>
                &nbsp; Upload your audio
            </div>

            <div class="section-description">
                Drop a noisy recording below or choose
                a file from your device.
            </div>
            """
        )

        noisy_audio = gr.Audio(
            label="",
            type="filepath",
            sources=["upload"],
            elem_classes="upload-area"
        )

        audio_info = gr.Markdown(
            "Select an audio file to continue.",
            elem_classes="file-info"
        )


    # ========================================================
    # STEP 2 — PROCESS
    # ========================================================

    gr.HTML(
        "<div style='height:25px'></div>"
    )

    gr.HTML(
        """
        <div style="text-align:center;">

            <div class="section-title">
                <span style="color:#6366f1;">02</span>
                &nbsp; Enhance your audio
            </div>

            <div class="section-description">
                Let AI reduce unwanted background noise
                from your recording.
            </div>

        </div>
        """
    )

    denoise_button = gr.Button(
        "✨  Denoise My Audio",
        variant="primary",
        elem_classes="denoise-button"
    )


    # ========================================================
    # STEP 3 — RESULT
    # ========================================================

    gr.HTML(
        "<div style='height:25px'></div>"
    )

    with gr.Column(elem_classes="card"):

        gr.HTML(
            """
            <div class="section-title">
                <span style="color:#6366f1;">03</span>
                &nbsp; Your enhanced audio
            </div>

            <div class="section-description">
                Listen to your cleaned recording below.
            </div>
            """
        )

        success_message = gr.HTML(
            "",
            visible=False
        )

        denoised_audio = gr.Audio(
            label="Denoised Audio",
            type="filepath",
            interactive=False
        )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    gr.HTML(
        "<div style='height:40px'></div>"
    )

    gr.HTML(
        """
        <div style="text-align:center; margin-bottom:22px;">

            <div class="section-title">
                How it works
            </div>

            <div class="section-description">
                Three simple steps from noisy audio to cleaner sound.
            </div>

        </div>
        """
    )

    with gr.Row():

        gr.HTML(
            """
            <div class="info-card">
                <div class="info-icon">📤</div>

                <h3>Upload</h3>

                <p>
                    Select a noisy audio recording
                    from your device.
                </p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="info-card">
                <div class="info-icon">🧠</div>

                <h3>AI Enhancement</h3>

                <p>
                    Our trained AI analyzes your
                    recording and reduces noise.
                </p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="info-card">
                <div class="info-icon">✨</div>

                <h3>Listen & Download</h3>

                <p>
                    Listen to the enhanced audio
                    and download your result.
                </p>
            </div>
            """
        )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">

            AudioClean AI
            <br>

            <strong>AI-powered audio enhancement</strong>

        </div>
        """
    )


    # ========================================================
    # EVENTS
    # ========================================================

    # Show file information
    noisy_audio.change(
        fn=get_audio_info,
        inputs=noisy_audio,
        outputs=audio_info
    )


    # Denoise
    denoise_button.click(
        fn=denoise_audio,
        inputs=noisy_audio,
        outputs=denoised_audio
    ).then(
        fn=lambda: gr.update(
            value="""
            <div class="success-message">
                ✓ Your audio has been successfully enhanced.
            </div>
            """,
            visible=True
        ),
        outputs=success_message
    )


# ============================================================
# Launch
# ============================================================

app.launch()