import sys
from pathlib import Path

# Allow imports from the SwinIR repository
ROOT = Path(__file__).resolve().parent.parent
SWINIR_DIR = ROOT / "SwinIR"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SWINIR_DIR))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import HistoricalDocumentDataset
from models.network_swinir import SwinIR


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 5

BATCH_SIZE = 1

PATCH_SIZE = 128

PATCHES_PER_IMAGE = 10

LEARNING_RATE = 2e-4

NUM_WORKERS = 2

GRAD_CLIP = 1.0

CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SwinIR model
# ============================================================

def create_model():

    model = SwinIR(
        upscale=1,
        in_chans=1,
        img_size=PATCH_SIZE,
        window_size=8,

        img_range=1.0,

        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,

        num_heads=[
            6, 6, 6,
            6, 6, 6
        ],

        mlp_ratio=2,

        upsampler="",
        resi_connection="1conv"
    )

    return model


# ============================================================
# Main training
# ============================================================

def train():

    print("=" * 60)
    print("SwinIR Historical Document Restoration")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        total_memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"GPU: {gpu_name}")
        print(f"VRAM: {total_memory:.2f} GB")

        # Helps reduce memory fragmentation
        torch.backends.cuda.matmul.allow_tf32 = True

    else:

        print("WARNING: CUDA unavailable.")

    print()

    # ========================================================
    # Dataset
    # ========================================================

    train_dataset = HistoricalDocumentDataset(
        root_dir=ROOT / "dataset",
        split="train",
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE
    )

    val_dataset = HistoricalDocumentDataset(
        root_dir=ROOT / "dataset",
        split="val",
        patch_size=PATCH_SIZE,
        patches_per_image=1
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=1,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=(
            NUM_WORKERS > 0
        )
    )

    # ========================================================
    # Model
    # ========================================================

    print("Creating SwinIR model...")

    model = create_model()

    model = model.to(DEVICE)

    # channels_last can improve convolution performance
    if DEVICE.type == "cuda":

        model = model.to(
            memory_format=torch.channels_last
        )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.L1Loss()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # ========================================================
    # Learning-rate scheduler
    # ========================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    # ========================================================
    # Mixed precision
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    # ========================================================
    # Training
    # ========================================================

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for degraded, gt in progress:

            degraded = degraded.to(
                DEVICE,
                non_blocking=True
            )

            gt = gt.to(
                DEVICE,
                non_blocking=True
            )

            if DEVICE.type == "cuda":

                degraded = degraded.contiguous(
                    memory_format=torch.channels_last
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward pass using mixed precision
            # ------------------------------------------------

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda")
            ):

                restored = model(
                    degraded
                )

                loss = criterion(
                    restored,
                    gt
                )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            scaler.scale(loss).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.5f}"
            )

            # Release references
            del restored
            del loss

        scheduler.step()

        average_train_loss = (
            running_loss /
            len(train_loader)
        )

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for degraded, gt in val_loader:

                degraded = degraded.to(
                    DEVICE,
                    non_blocking=True
                )

                gt = gt.to(
                    DEVICE,
                    non_blocking=True
                )

                if DEVICE.type == "cuda":

                    degraded = degraded.contiguous(
                        memory_format=torch.channels_last
                    )

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=(DEVICE.type == "cuda")
                ):

                    restored = model(
                        degraded
                    )

                    loss = criterion(
                        restored,
                        gt
                    )

                validation_loss += loss.item()

        average_val_loss = (
            validation_loss /
            len(val_loader)
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {average_train_loss:.6f}"
        )

        print(
            f"Val Loss:   {average_val_loss:.6f}"
        )

        print(
            f"LR:         {current_lr:.8f}"
        )

        # ====================================================
        # Save latest checkpoint
        # ====================================================

        latest_path = (
            CHECKPOINT_DIR /
            "swinir_latest.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "val_loss":
                    average_val_loss
            },
            latest_path
        )

        # ====================================================
        # Save best checkpoint
        # ====================================================

        if average_val_loss < best_val_loss:

            best_val_loss = average_val_loss

            best_path = (
                CHECKPOINT_DIR /
                "swinir_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "val_loss":
                        average_val_loss
                },
                best_path
            )

            print(
                f"New best model saved: "
                f"{best_path}"
            )

        # Clear cached CUDA memory

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()

        print()

    print("=" * 60)
    print("Training complete")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_DIR / 'swinir_best.pth'}"
    )


if __name__ == "__main__":
    train()