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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BEST_MODEL_PATH = "unet_best.pth"


# -----------------------------
# Datasets
# -----------------------------

train_dataset = ManuscriptDataset(
    root_dir=TRAIN_DIR,
    patch_size=64,
    augment=True
)

val_dataset = ManuscriptDataset(
    root_dir=VAL_DIR,
    patch_size=64,
    augment=False
)


# -----------------------------
# DataLoaders
# -----------------------------

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


# -----------------------------
# Model
# -----------------------------

model = UNet(
    in_channels=1,
    out_channels=1
).to(DEVICE)


# -----------------------------
# Loss and optimizer
# -----------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------

best_val_loss = float("inf")

print("Device:", DEVICE)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print()

for epoch in range(EPOCHS):

    # ----- Training -----

    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ----- Validation -----

    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # ----- Print results -----

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )


    # ----- Save best model -----

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("  -> Best model saved!")


print()
print("Training completed.")
print("Best validation loss:", best_val_loss)
print("Best model:", BEST_MODEL_PATH)