import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "unet"

sys.path.insert(0, str(MODEL_DIR))

from unet import UNet
from dataset import DocumentDataset


# -----------------------------
# Configuration
# -----------------------------
PATCH_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
SEED = 42

TRAIN_DIR = ROOT / "data"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_PATH = CHECKPOINT_DIR / "unet_latest.pth"
BEST_PATH = CHECKPOINT_DIR / "unet_best.pth"


# -----------------------------
# Reproducibility and device
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------
# Dataset and DataLoaders
# -----------------------------
train_dataset = DocumentDataset(TRAIN_DIR, "train", PATCH_SIZE)
val_dataset = DocumentDataset(TRAIN_DIR, "val", PATCH_SIZE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("Training pairs:", len(train_dataset.pairs))
print("Training patches:", len(train_dataset))
print("Validation pairs:", len(val_dataset))


# -----------------------------
# Model, loss and optimizer
# -----------------------------
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
train_history = []
val_history = []


# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"
    )

    for degraded, clean in progress:
        degraded = degraded.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        restored = model(degraded)
        loss = criterion(restored, clean)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.6f}")

    train_loss = train_running_loss / len(train_loader)

    model.eval()
    val_running_loss = 0.0

    with torch.no_grad():
        for degraded, clean in tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"
        ):
            degraded = degraded.to(device)
            clean = clean.to(device)

            height, width = degraded.shape[-2:]
            size = PATCH_SIZE

            if height >= size and width >= size:
                top = (height - size) // 2
                left = (width - size) // 2
                degraded = degraded[:, :, top:top + size, left:left + size]
                clean = clean[:, :, top:top + size, left:left + size]
            else:
                pad_h = max(0, size - height)
                pad_w = max(0, size - width)
                degraded = torch.nn.functional.pad(
                    degraded, (0, pad_w, 0, pad_h), mode="replicate"
                )
                clean = torch.nn.functional.pad(
                    clean, (0, pad_w, 0, pad_h), mode="replicate"
                )

            restored = model(degraded)
            val_running_loss += criterion(restored, clean).item()

    val_loss = val_running_loss / len(val_loader)

    train_history.append(train_loss)
    val_history.append(val_loss)

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

    torch.save(model.state_dict(), LATEST_PATH)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_PATH)
        print("Best model saved!")

    history_path = MODEL_DIR / "training" / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "train_loss": train_history,
                "val_loss": val_history,
                "best_val_loss": best_val_loss
            },
            file,
            indent=4
        )

print("\nTraining completed.")
print("Best validation loss:", best_val_loss)
print("Best checkpoint:", BEST_PATH)
