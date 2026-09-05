import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ManuscriptDataset
from unet import UNet


# -----------------------------
# Configuration
# -----------------------------
TRAIN_DIR = "../../data/train"
VAL_DIR = "../../data/val"

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-3

BEST_MODEL_PATH = "unet_best.pth"
HISTORY_PATH = "training_history.json"


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# -----------------------------
# Datasets and DataLoaders
# -----------------------------
train_dataset = ManuscriptDataset(
    TRAIN_DIR,
    patch_size=64,
    augment=True
)

val_dataset = ManuscriptDataset(
    VAL_DIR,
    patch_size=64,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# -----------------------------
# Model
# -----------------------------
model = UNet(
    in_channels=1,
    out_channels=1
).to(device)


# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training history
# -----------------------------
train_losses = []
val_losses = []

best_val_loss = float("inf")
best_epoch = 0


# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):

    # ---- Training ----
    model.train()

    running_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)


    # ---- Validation ----
    model.eval()

    val_running_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            val_running_loss += loss.item()

    val_loss = val_running_loss / len(val_loader)


    # ---- Save history ----
    train_losses.append(train_loss)
    val_losses.append(val_loss)


    # ---- Print results ----
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )


    # ---- Save best model ----
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("  -> Best model saved!")


# -----------------------------
# Save training history
# -----------------------------
with open(HISTORY_PATH, "w") as f:

    json.dump(
        {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss
        },
        f,
        indent=4
    )


# -----------------------------
# Finished
# -----------------------------
print("\nTraining completed.")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss}")
print(f"Best model: {BEST_MODEL_PATH}")
print(f"Training history: {HISTORY_PATH}")