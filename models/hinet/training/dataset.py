import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CATEGORIES = ["bleedthrough", "blur", "fading", "noise", "random_degradation", "stains"]


def read_image(path):
    image = Image.open(path).convert("L")
    return np.array(image, dtype=np.float32) / 255.0


class DocumentDataset(Dataset):
    def __init__(self, data_root, split, patch_size=128):
        self.pairs = []
        self.split = split
        self.patch_size = patch_size

        for category in CATEGORIES:
            clean_dir = Path(data_root) / category / split / "clean"
            degraded_dir = Path(data_root) / category / split / "degraded"

            for clean_path in sorted(clean_dir.glob("*")):
                degraded_path = degraded_dir / clean_path.name
                if degraded_path.exists():
                    self.pairs.append((degraded_path, clean_path))

        if not self.pairs:
            raise RuntimeError(f"No image pairs found for {split}")

    def __len__(self):
        return len(self.pairs) * 10 if self.split == "train" else len(self.pairs)

    def __getitem__(self, index):
        if self.split == "train":
            index = index % len(self.pairs)

        degraded_path, clean_path = self.pairs[index]
        degraded = read_image(degraded_path)
        clean = read_image(clean_path)

        if degraded.shape != clean.shape:
            raise ValueError(f"Image sizes do not match: {degraded_path.name}")

        if self.split == "train":
            h, w = clean.shape
            size = self.patch_size

            if h < size or w < size:
                pad_h = max(0, size - h)
                pad_w = max(0, size - w)
                degraded = np.pad(degraded, ((0, pad_h), (0, pad_w)), mode="reflect")
                clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = clean.shape

            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
            degraded = degraded[top:top + size, left:left + size]
            clean = clean[top:top + size, left:left + size]

            if random.random() < 0.5:
                degraded, clean = degraded[:, ::-1], clean[:, ::-1]
            if random.random() < 0.5:
                degraded, clean = degraded[::-1, :], clean[::-1, :]

        degraded = torch.from_numpy(np.ascontiguousarray(degraded)).unsqueeze(0)
        clean = torch.from_numpy(np.ascontiguousarray(clean)).unsqueeze(0)

        return degraded, clean
