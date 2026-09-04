"""
Train DnCNN to remove bleed-through / ink-seepage noise from manuscript
images.

Usage:
    python train.py --data_dir data --use_verso false --epochs 50

Expects:
    data/recto/*.tif   -> degraded scans
    data/clean/*.tif   -> matching ground-truth clean scans
    data/verso/*.tif   -> (optional) verso scans, only needed if --use_verso
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DnCNN
from dataset import ManuscriptDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_layers", type=int, default=20)
    p.add_argument("--use_verso", action="store_true", help="feed recto+verso as 2 input channels")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ManuscriptDataset(
        root_dir=args.data_dir,
        patch_size=args.patch_size,
        use_verso=args.use_verso,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    print(f"Loaded {len(dataset)} training image pairs.")

    in_ch = 2 if args.use_verso else 1
    model = DnCNN(in_channels=in_ch, out_channels=1, num_layers=args.num_layers).to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        avg_loss = running_loss / len(dataset)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | loss: {avg_loss:.6f} | {elapsed:.1f}s")

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            ckpt_path = ckpt_dir / f"dncnn_epoch{epoch+1}.pth"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "in_channels": in_ch,
                "num_layers": args.num_layers,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = ckpt_dir / "dncnn_final.pth"
    torch.save({
        "model_state": model.state_dict(),
        "epoch": args.epochs,
        "in_channels": in_ch,
        "num_layers": args.num_layers,
    }, final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
