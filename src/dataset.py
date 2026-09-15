from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class AudioDenoisingDataset(Dataset):

    def __init__(
        self,
        clean_dir,
        noisy_dir,
        target_sr=16000,
        segment_seconds=1.0,
        n_fft=512,
        hop_length=128
    ):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)

        self.target_sr = target_sr
        self.segment_length = int(target_sr * segment_seconds)

        self.n_fft = n_fft
        self.hop_length = hop_length

        # Find all clean WAV files
        self.clean_files = sorted(
            self.clean_dir.glob("*.wav")
        )

        # Check that corresponding noisy files exist
        self.noisy_files = []

        for clean_file in self.clean_files:
            noisy_file = self.noisy_dir / clean_file.name

            if noisy_file.exists():
                self.noisy_files.append(noisy_file)
            else:
                raise FileNotFoundError(
                    f"Missing noisy file: {noisy_file}"
                )

        print(
            f"Dataset loaded: {len(self.clean_files)} pairs"
        )

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):

        clean_path = self.clean_files[index]
        noisy_path = self.noisy_files[index]

        # Load audio
        clean, clean_sr = sf.read(clean_path)
        noisy, noisy_sr = sf.read(noisy_path)

        # Convert stereo to mono if necessary
        if clean.ndim > 1:
            clean = np.mean(clean, axis=1)

        if noisy.ndim > 1:
            noisy = np.mean(noisy, axis=1)

        # Resample to target sample rate
        if clean_sr != self.target_sr:
            clean = librosa.resample(
                clean,
                orig_sr=clean_sr,
                target_sr=self.target_sr
            )

        if noisy_sr != self.target_sr:
            noisy = librosa.resample(
                noisy,
                orig_sr=noisy_sr,
                target_sr=self.target_sr
            )

        # Make sure both have the same length
        min_length = min(
            len(clean),
            len(noisy)
        )

        clean = clean[:min_length]
        noisy = noisy[:min_length]

        # Make sure the recording is long enough
        if len(clean) < self.segment_length:
            raise ValueError(
                f"Audio file is shorter than "
                f"{self.segment_length} samples: {clean_path}"
            )

        # Select a random 1-second segment
        max_start = len(clean) - self.segment_length

        start = np.random.randint(
            0,
            max_start + 1
        )

        clean_segment = clean[
            start:start + self.segment_length
        ]

        noisy_segment = noisy[
            start:start + self.segment_length
        ]

        # STFT
        clean_stft = librosa.stft(
            clean_segment,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        noisy_stft = librosa.stft(
            noisy_segment,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Magnitude
        clean_magnitude = np.abs(
            clean_stft
        )

        noisy_magnitude = np.abs(
            noisy_stft
        )

        # Convert to tensors
        noisy_tensor = torch.from_numpy(
            noisy_magnitude
        ).float()

        clean_tensor = torch.from_numpy(
            clean_magnitude
        ).float()

        # Add channel dimension
        noisy_tensor = noisy_tensor.unsqueeze(0)

        clean_tensor = clean_tensor.unsqueeze(0)

        return noisy_tensor, clean_tensor