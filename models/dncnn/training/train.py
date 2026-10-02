import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "dncnn"

sys.path.insert(0, str(MODEL_DIR))
from model import DnCNN
from dataset import DocumentDataset


DATA_ROOT = ROOT / "data"
SAVE_DIR = MODEL_DIR / "checkpoints"

PATCH_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_LAYERS = 20
FEATURES = 64

random.seed(42)
torch.manual_seed(42)


def prepare_validation_image(image, size):
    height, width = image.shape[-2:]

    if height < size or width < size:
        pad_h = max(0, size - height)
        pad_w = max(0, size - width)
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        height, width = image.shape[-2:]

    top = (height - size) // 2
    left = (width - size) // 2

    return image[:, :, top:top + size, left:left + size]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DocumentDataset(DATA_ROOT, "train", PATCH_SIZE)
    val_dataset = DocumentDataset(DATA_ROOT, "val", PATCH_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Training images:", len(train_dataset.pairs))
    print("Training patches:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Device:", device)

    model = DnCNN(
        in_channels=1,
        out_channels=1,
        num_layers=NUM_LAYERS,
        features=FEATURES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5
    )
    criterion = nn.MSELoss()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

        for degraded, clean in progress:
            degraded = degraded.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            restored = model(degraded)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * degraded.size(0)
            progress.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for degraded, clean in val_loader:
                degraded = prepare_validation_image(degraded, PATCH_SIZE).to(device)
                clean = prepare_validation_image(clean, PATCH_SIZE).to(device)

                restored = model(degraded)
                loss = criterion(restored, clean)
                val_loss += loss.item()

        val_loss = val_loss / len(val_dataset)
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch + 1,
            "in_channels": 1,
            "num_layers": NUM_LAYERS,
            "features": FEATURES,
            "patch_size": PATCH_SIZE
        }

        torch.save(checkpoint, SAVE_DIR / "dncnn_latest.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, SAVE_DIR / "dncnn_best.pth")
            print("Saved best model.")

    print("Training complete.")
    print("Best checkpoint:", SAVE_DIR / "dncnn_best.pth")


if __name__ == "__main__":
    main()
