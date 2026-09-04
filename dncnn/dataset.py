"""
Dataset loader for manuscript bleed-through removal.

Expected folder layout (this matches how you'd organize the ISOS
Bleed-Through Database or DIBCO after download):

    data/
      recto/   -> degraded images (bleed-through visible), e.g. 0001.tif
      clean/   -> ground-truth clean/foreground-only images, same filenames
      verso/   -> (optional) verso side, same filenames, used as extra
                  input channel if you want a 2-channel recto+verso model

If you don't have ground-truth clean images (e.g. only recto/verso pairs
with no manually-cleaned target), see `synthetic_pairs.py` to generate
training pairs by synthetically compositing clean text with a simulated
bleed-through pattern -- this is the same trick Savino & Tonazzini (2024)
use to avoid needing large labelled datasets.
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ManuscriptDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size: int = 64,
        use_verso: bool = False,
        augment: bool = True,
    ):
        self.root = Path(root_dir)
        self.recto_dir = self.root / "recto"
        self.clean_dir = self.root / "clean"
        self.verso_dir = self.root / "verso"
        self.use_verso = use_verso
        self.patch_size = patch_size
        self.augment = augment

        self.filenames = sorted(
            f.name for f in self.recto_dir.iterdir()
            if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        )
        if not self.filenames:
            raise FileNotFoundError(f"No images found in {self.recto_dir}")

        missing = [f for f in self.filenames if not (self.clean_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} recto images have no matching clean/ ground truth "
                f"(e.g. {missing[0]}). Every file in recto/ needs a same-named file "
                f"in clean/."
            )

    def __len__(self):
        return len(self.filenames)

    def _load_gray(self, path):
        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        recto = self._load_gray(self.recto_dir / fname)
        clean = self._load_gray(self.clean_dir / fname)

        if self.use_verso:
            verso_path = self.verso_dir / fname
            verso = self._load_gray(verso_path) if verso_path.exists() else np.zeros_like(recto)

        h, w = recto.shape
        ps = self.patch_size
        if h < ps or w < ps:
            raise ValueError(f"{fname} is smaller than patch_size={ps} ({h}x{w})")

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        recto_patch = recto[top:top + ps, left:left + ps]
        clean_patch = clean[top:top + ps, left:left + ps]
        if self.use_verso:
            verso_patch = verso[top:top + ps, left:left + ps]

        if self.augment:
            if random.random() < 0.5:
                recto_patch = np.fliplr(recto_patch).copy()
                clean_patch = np.fliplr(clean_patch).copy()
                if self.use_verso:
                    verso_patch = np.fliplr(verso_patch).copy()
            if random.random() < 0.5:
                recto_patch = np.flipud(recto_patch).copy()
                clean_patch = np.flipud(clean_patch).copy()
                if self.use_verso:
                    verso_patch = np.flipud(verso_patch).copy()
            k = random.randint(0, 3)
            recto_patch = np.rot90(recto_patch, k).copy()
            clean_patch = np.rot90(clean_patch, k).copy()
            if self.use_verso:
                verso_patch = np.rot90(verso_patch, k).copy()

        if self.use_verso:
            input_patch = np.stack([recto_patch, verso_patch], axis=0)  # (2, H, W)
        else:
            input_patch = recto_patch[None, :, :]  # (1, H, W)

        target_patch = clean_patch[None, :, :]  # (1, H, W)

        return (
            torch.from_numpy(input_patch.astype(np.float32)),
            torch.from_numpy(target_patch.astype(np.float32)),
        )
