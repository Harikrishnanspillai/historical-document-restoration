from pathlib import Path
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
NAFNET = ROOT / "models" / "nafnet" / "NAFNet"
sys.path.append(str(NAFNET))

from basicsr.models.archs.NAFNet_arch import NAFNet
from dataset import DocumentDataset


DATA_ROOT = ROOT / "data"
SAVE_DIR = ROOT / "models" / "nafnet" / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    random.seed(42)
    torch.manual_seed(42)

    train_data = DocumentDataset(DATA_ROOT, "train", PATCH_SIZE)
    val_data = DocumentDataset(DATA_ROOT, "val", PATCH_SIZE)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    print("Training images:", len(train_data.pairs))
    print("Validation images:", len(val_data.pairs))
    print("Device:", DEVICE)

    # NAFNet uses one output image, so the loss is calculated directly.
    model = NAFNet(
        img_channel=1,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

        for degraded, clean in progress_bar:
            degraded = degraded.to(DEVICE)
            clean = clean.to(DEVICE)

            optimizer.zero_grad()
            restored = model(degraded)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for degraded, clean in val_loader:
                # Crop before moving to GPU to reduce memory use.
                h, w = degraded.shape[-2:]
                size = PATCH_SIZE
                top = max(0, (h - size) // 2)
                left = max(0, (w - size) // 2)
                degraded = degraded[:, :, top:top + size, left:left + size]
                clean = clean[:, :, top:top + size, left:left + size]

                if degraded.shape[-2] < size or degraded.shape[-1] < size:
                    pad_h = size - degraded.shape[-2]
                    pad_w = size - degraded.shape[-1]
                    degraded = torch.nn.functional.pad(degraded, (0, pad_w, 0, pad_h))
                    clean = torch.nn.functional.pad(clean, (0, pad_w, 0, pad_h))

                degraded = degraded.to(DEVICE)
                clean = clean.to(DEVICE)

                restored = model(degraded)
                loss = criterion(restored, clean)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        torch.save(model.state_dict(), SAVE_DIR / "nafnet_latest.pth")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), SAVE_DIR / "nafnet_best.pth")
            print("Saved best model.")

    print("Training complete.")


if __name__ == "__main__":
    main()
