import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT))

from src.dataset import AudioDenoisingDataset
from src.model import UNet


print("=" * 60)
print("AUDIO DENOISING MODEL TEST")
print("=" * 60)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

clean_dir = PROJECT_ROOT / "data" / "clean_train"
noisy_dir = PROJECT_ROOT / "data" / "noisy_train"

dataset = AudioDenoisingDataset(
    clean_dir=clean_dir,
    noisy_dir=noisy_dir,
    random_crop=True
)

print()
print("Dataset size:", len(dataset))


# --------------------------------------------------
# Load one example
# --------------------------------------------------

print()
print("Loading one training example...")

noisy, clean = dataset[0]

print("Noisy shape:", noisy.shape)
print("Clean shape:", clean.shape)


# --------------------------------------------------
# Add batch dimension
# --------------------------------------------------

noisy_batch = noisy.unsqueeze(0).to(device)
clean_batch = clean.unsqueeze(0).to(device)

print()
print("Input batch shape:", noisy_batch.shape)
print("Target batch shape:", clean_batch.shape)


# --------------------------------------------------
# Create model
# --------------------------------------------------

print()
print("Creating model...")

model = UNet().to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Model parameters: "
    f"{parameter_count:,}"
)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

print()
print("Testing forward pass...")

model.eval()

with torch.inference_mode():
    prediction = model(noisy_batch)

print("Prediction shape:", prediction.shape)


# --------------------------------------------------
# Shape check
# --------------------------------------------------

print()
print("Checking output shape...")

if prediction.shape != clean_batch.shape:

    print("❌ ERROR: Output shape does not match target.")

    print(
        "Prediction:",
        prediction.shape
    )

    print(
        "Target:",
        clean_batch.shape
    )

    sys.exit(1)

else:

    print("✓ Output shape matches target.")


# --------------------------------------------------
# Check for invalid values
# --------------------------------------------------

print()
print("Checking for NaN / Inf values...")

if torch.isnan(prediction).any():

    print("❌ ERROR: Prediction contains NaN values.")

    sys.exit(1)

if torch.isinf(prediction).any():

    print("❌ ERROR: Prediction contains Inf values.")

    sys.exit(1)

print("✓ Prediction contains valid values.")


# --------------------------------------------------
# Test backward pass
# --------------------------------------------------

print()
print("Testing backward pass...")

model.train()

prediction = model(noisy_batch)

loss = torch.mean(
    (prediction - clean_batch) ** 2
)

print(
    f"Test MSE loss: {loss.item():.6f}"
)

loss.backward()

print("✓ Backward pass completed.")


# --------------------------------------------------
# Finished
# --------------------------------------------------

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)

print()
print("The model is ready for GPU training.")