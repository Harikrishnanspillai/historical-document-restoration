import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

SWIN_ROOT = Path(__file__).resolve().parents[1]       # models/Swin
PROJECT_ROOT = Path(__file__).resolve().parents[3]    # repository root
SWINIR_DIR = SWIN_ROOT / "SwinIR"
sys.path.insert(0, str(SWIN_ROOT))
sys.path.insert(0, str(SWINIR_DIR))

from training.dataset import HistoricalDocumentDataset, CATEGORIES
from models.network_swinir import SwinIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH_SIZE = 5, 1
PATCH_SIZE, PATCHES_PER_IMAGE = 128, 10
VAL_PATCHES_PER_IMAGE = 1
LEARNING_RATE, NUM_WORKERS, GRAD_CLIP = 2e-5, 2, 1.0
SEED = 42
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_DIR = SWIN_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def create_model():
    return SwinIR(
        upscale=1, in_chans=1, img_size=PATCH_SIZE, window_size=8,
        img_range=1.0, depths=[6,6,6,6,6,6], embed_dim=180,
        num_heads=[6,6,6,6,6,6], mlp_ratio=2,
        upsampler="", resi_connection="1conv"
    )


def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "categories": list(CATEGORIES),
        "patch_size": PATCH_SIZE,
    }, path)


def train():
    set_seed()
    print("Device:", DEVICE, "| Data:", DATA_ROOT)
    print("Training categories:", ", ".join(CATEGORIES))
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True

    train_ds = HistoricalDocumentDataset(
        DATA_ROOT, "train", CATEGORIES, PATCH_SIZE, PATCHES_PER_IMAGE)
    val_ds = HistoricalDocumentDataset(
        DATA_ROOT, "val", CATEGORIES, PATCH_SIZE, VAL_PATCHES_PER_IMAGE)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda", persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda", persistent_workers=NUM_WORKERS > 0)

    model = create_model().to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for degraded, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            degraded, clean = degraded.to(DEVICE), clean.to(DEVICE)
            if DEVICE.type == "cuda":
                degraded = degraded.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                restored = model(degraded)
                loss = criterion(restored, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for degraded, clean in tqdm(val_loader, desc="Validation"):
                degraded, clean = degraded.to(DEVICE), clean.to(DEVICE)
                if DEVICE.type == "cuda":
                    degraded = degraded.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                    enabled=DEVICE.type == "cuda"):
                    loss = criterion(model(degraded), clean)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}, "
              f"lr={optimizer.param_groups[0]['lr']:.8g}")

        save_checkpoint(CHECKPOINT_DIR / "swinir_latest.pth",
                        epoch+1, model, optimizer, scheduler, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(CHECKPOINT_DIR / "swinir_best.pth",
                            epoch+1, model, optimizer, scheduler, val_loss)
            print("Saved best checkpoint.")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("Training finished. Best validation loss:", best_val_loss)


if __name__ == "__main__":
    train()
