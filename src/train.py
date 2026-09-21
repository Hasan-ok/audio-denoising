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
# Datasets
# --------------------------------------------------

# Training dataset:
# Random 1-second crops are used so the model sees
# different parts of each recording.
train_full_dataset = AudioDenoisingDataset(
    clean_dir=clean_dir,
    noisy_dir=noisy_dir,
    random_crop=True
)

# Validation dataset:
# Center crops are used so validation is deterministic.
validation_full_dataset = AudioDenoisingDataset(
    clean_dir=clean_dir,
    noisy_dir=noisy_dir,
    random_crop=False
)

print("Full dataset:", len(train_full_dataset))


# --------------------------------------------------
# Train / validation split
# --------------------------------------------------

dataset_size = len(train_full_dataset)

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
    train_full_dataset,
    train_indices
)

validation_dataset = Subset(
    validation_full_dataset,
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
    pin_memory=True,
    persistent_workers=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet().to(device)

print("Model created.")

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Model parameters:", f"{total_parameters:,}")


# --------------------------------------------------
# Loss
# --------------------------------------------------

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


def combined_loss(prediction, target):

    mse = mse_loss(
        prediction,
        target
    )

    l1 = l1_loss(
        prediction,
        target
    )

    return 0.8 * mse + 0.2 * l1


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# Learning-rate scheduler
# --------------------------------------------------

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# --------------------------------------------------
# Mixed precision
# --------------------------------------------------

use_amp = torch.cuda.is_available()

if use_amp:
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None


# --------------------------------------------------
# Training settings
# --------------------------------------------------

epochs = 30

best_validation_loss = float("inf")

early_stopping_patience = 5
epochs_without_improvement = 0


# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(epochs):

    print()
    print(
        f"========== Epoch {epoch + 1}/{epochs} =========="
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    model.train()

    training_loss = 0.0

    for batch_number, (noisy, clean) in enumerate(
        train_loader
    ):

        noisy = noisy.to(
            device,
            non_blocking=True
        )

        clean = clean.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                prediction = model(noisy)

                loss = combined_loss(
                    prediction,
                    clean
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            prediction = model(noisy)

            loss = combined_loss(
                prediction,
                clean
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        training_loss += loss.item()

        if (batch_number + 1) % 100 == 0:

            print(
                f"Batch {batch_number + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.6f}"
            )

    training_loss /= len(train_loader)


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

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

            if use_amp:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    prediction = model(noisy)

                    loss = combined_loss(
                        prediction,
                        clean
                    )

            else:

                prediction = model(noisy)

                loss = combined_loss(
                    prediction,
                    clean
                )

            validation_loss += loss.item()

    validation_loss /= len(validation_loader)


    # --------------------------------------------------
    # Learning-rate update
    # --------------------------------------------------

    scheduler.step(
        validation_loss
    )

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        f"Training Loss:   {training_loss:.6f}"
    )

    print(
        f"Validation Loss: {validation_loss:.6f}"
    )

    print(
        f"Learning Rate:   {current_lr:.8f}"
    )


    # --------------------------------------------------
    # Save latest checkpoint
    # --------------------------------------------------

    latest_checkpoint = (
        checkpoint_dir / "latest.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "learning_rate": current_lr
        },
        latest_checkpoint
    )


    # --------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        epochs_without_improvement = 0

        best_checkpoint = (
            checkpoint_dir / "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_lr
            },
            best_checkpoint
        )

        print("✓ Best model saved.")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )


    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------

    if (
        epochs_without_improvement
        >= early_stopping_patience
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# --------------------------------------------------
# Finished
# --------------------------------------------------

print()
print("Training complete.")

print(
    f"Best validation loss: "
    f"{best_validation_loss:.6f}"
)