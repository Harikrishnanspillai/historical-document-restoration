from pathlib import Path
import random
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
HINET = ROOT / "models" / "hinet" / "HINet"
sys.path.append(str(HINET / "basicsr" / "models" / "archs"))

from hinet_arch import HINet
from dataset import DocumentDataset


DATA_ROOT = ROOT / "data"
SAVE_DIR = ROOT / "models" / "hinet" / "checkpoints"
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

    # wf=32 is smaller than the original default to fit the 4 GB GPU.
    model = HINet(in_chn=1, wf=32, depth=5).to(DEVICE)
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
            outputs = model(degraded)
            loss = sum(criterion(output, clean) for output in outputs) / len(outputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for degraded, clean in val_loader:
                degraded = degraded.to(DEVICE)
                clean = clean.to(DEVICE)

                # Use a centre crop for validation to keep memory use low.
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

                outputs = model(degraded)
                loss = sum(criterion(output, clean) for output in outputs) / len(outputs)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        torch.save(model.state_dict(), SAVE_DIR / "hinet_latest.pth")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), SAVE_DIR / "hinet_best.pth")
            print("Saved best model.")

    print("Training complete.")


if __name__ == "__main__":
    main()
