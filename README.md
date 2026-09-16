---
title: Audio Denoising
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🎧 Audio Denoising

A deep-learning audio denoising system that removes unwanted noise from speech recordings using a **U-Net neural network**.

The model is trained on the **VoiceBank-DEMAND dataset**, using paired noisy and clean speech recordings to learn how to recover cleaner speech from noisy audio.

## 🚀 Live Demo

Try the deployed application:

**[Audio Denoising on Hugging Face](https://huggingface.co/spaces/hok111/audio-denoising)**

Upload a noisy audio file and the application will generate a denoised version.

## 🧠 How It Works

The system processes audio in the following pipeline:

```text
Noisy Audio
     │
     ▼
Audio Preprocessing
     │
     ▼
Spectrogram
     │
     ▼
U-Net Model
     │
     ▼
Denoised Spectrogram
     │
     ▼
Waveform Reconstruction
     │
     ▼
Denoised Audio
```

The audio is converted into a spectrogram representation before being passed through the U-Net model. The model learns to transform the noisy spectrogram into a cleaner representation, which is then converted back into an audio waveform.

## 🗂️ Dataset

The project uses the **VoiceBank-DEMAND** dataset, which contains paired clean and noisy speech recordings.

The dataset is **not included in this repository** because of its large size.

## 🛠️ Technologies

* Python
* PyTorch
* U-Net
* Librosa
* NumPy
* SciPy
* SoundFile
* Gradio

## 📁 Project Structure

```text
audio-denoising/
│
├── app/
│   └── app.py
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── checkpoints/
│
├── app.py
├── check_environment.py
├── test_inference.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 💻 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Hasan-ok/audio-denoising.git
cd audio-denoising
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows:

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

The Gradio interface will start locally.

## 📌 Project Status

This project was developed as a deep-learning audio denoising capstone project.

The trained model has been integrated into a web interface and deployed using Hugging Face Spaces.

## 📄 License

This project is intended for educational and research purposes.
