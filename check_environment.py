import torch
import librosa
import soundfile as sf

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Librosa:", librosa.__version__)

print("\nDevice:")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CPU")