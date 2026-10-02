from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GeneralizedManuscriptDataset
from unet import UNet


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-3

PATCH_SIZE = 64
PATCHES_PER_IMAGE_TRAIN = 3
PATCHES_PER_IMAGE_VAL = 1

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "unet_generalized_best.pth"
)

HISTORY_FILE = (
    CHECKPOINT_DIR / "training_history_generalized.json"
)


# ============================================================
# Device
# ============================================================

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Datasets
# ============================================================

print("\nCreating training dataset...")

train_dataset = GeneralizedManuscriptDataset(
    root_dir=DATA_ROOT,
    split="train",
    patch_size=PATCH_SIZE,
    augment=True,
    patches_per_image=PATCHES_PER_IMAGE_TRAIN,
)

print("\nCreating validation dataset...")

val_dataset = GeneralizedManuscriptDataset(
    root_dir=DATA_ROOT,
    split="val",
    patch_size=PATCH_SIZE,
    augment=False,
    patches_per_image=PATCHES_PER_IMAGE_VAL,
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# Model
# ============================================================

model = UNet(
    in_channels=1,
    out_channels=1,
).to(DEVICE)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

history = {
    "train_loss": [],
    "val_loss": [],
}


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(
            DEVICE,
            non_blocking=True,
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()

        optimizer.step()

        running_train_loss += (
            loss.item() * inputs.size(0)
        )

    train_loss = (
        running_train_loss
        / len(train_dataset)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True,
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets,
            )

            running_val_loss += (
                loss.item() * inputs.size(0)
            )

    val_loss = (
        running_val_loss
        / len(val_dataset)
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    with open(
        HISTORY_FILE,
        "w",
    ) as f:
        json.dump(
            history,
            f,
            indent=4,
        )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_CHECKPOINT,
        )

        checkpoint_status = " ← BEST"

    else:

        checkpoint_status = ""


    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
        f"{checkpoint_status}"
    )


# ============================================================
# Finished
# ============================================================

print("\nTraining complete!")

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Best checkpoint saved to:\n"
    f"{BEST_CHECKPOINT}"
)

print(
    f"Training history saved to:\n"
    f"{HISTORY_FILE}"
)