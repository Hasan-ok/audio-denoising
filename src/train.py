import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from src.dataset import AudioDenoisingDataset
from src.model import UNet


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
# Paths
# --------------------------------------------------

clean_dir = PROJECT_ROOT / "data" / "clean_train"
noisy_dir = PROJECT_ROOT / "data" / "noisy_train"

checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = AudioDenoisingDataset(
    clean_dir=clean_dir,
    noisy_dir=noisy_dir
)

print("Full dataset:", len(dataset))


# --------------------------------------------------
# Train / Validation split
# --------------------------------------------------

dataset_size = len(dataset)

validation_size = int(0.10 * dataset_size)
training_size = dataset_size - validation_size

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    dataset_size,
    generator=generator
).tolist()

train_indices = indices[:training_size]
validation_indices = indices[training_size:]

train_dataset = Subset(
    dataset,
    train_indices
)

validation_dataset = Subset(
    dataset,
    validation_indices
)

print("Training examples:", len(train_dataset))
print("Validation examples:", len(validation_dataset))


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet().to(device)

print("Model created.")


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# Training settings
# --------------------------------------------------

epochs = 20

best_validation_loss = float("inf")


# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(epochs):

    print()
    print(
        f"========== Epoch {epoch + 1}/{epochs} =========="
    )

    # ------------------------------
    # Training
    # ------------------------------

    model.train()

    training_loss = 0.0

    for batch_number, (noisy, clean) in enumerate(train_loader):

        noisy = noisy.to(
            device,
            non_blocking=True
        )

        clean = clean.to(
            device,
            non_blocking=True
        )

        prediction = model(noisy)

        loss = criterion(
            prediction,
            clean
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        training_loss += loss.item()

        if (batch_number + 1) % 100 == 0:
            print(
                f"Batch {batch_number + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.6f}"
            )

    training_loss /= len(train_loader)


    # ------------------------------
    # Validation
    # ------------------------------

    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for noisy, clean in validation_loader:

            noisy = noisy.to(
                device,
                non_blocking=True
            )

            clean = clean.to(
                device,
                non_blocking=True
            )

            prediction = model(noisy)

            loss = criterion(
                prediction,
                clean
            )

            validation_loss += loss.item()

    validation_loss /= len(validation_loader)


    # ------------------------------
    # Print results
    # ------------------------------

    print(
        f"Training Loss:   {training_loss:.6f}"
    )

    print(
        f"Validation Loss: {validation_loss:.6f}"
    )


    # ------------------------------
    # Save latest checkpoint
    # ------------------------------

    latest_checkpoint = checkpoint_dir / "latest.pth"

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss": training_loss,
            "validation_loss": validation_loss
        },
        latest_checkpoint
    )


    # ------------------------------
    # Save best checkpoint
    # ------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_checkpoint = checkpoint_dir / "best_model.pth"

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": training_loss,
                "validation_loss": validation_loss
            },
            best_checkpoint
        )

        print(
            "✓ Best model saved."
        )


print()
print("Training complete.")
print(
    f"Best validation loss: {best_validation_loss:.6f}"
)