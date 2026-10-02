import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CATEGORIES = (
    "bleedthrough",
    "blur",
    "fading",
    "noise",
    "random_degradation",
    "stains"
)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}


def read_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


class DocumentDataset(Dataset):
    def __init__(self, data_root, split, patch_size=128):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.pairs = []

        for category in CATEGORIES:
            folder = self.data_root / category / split
            degraded_dir = folder / "degraded"
            clean_dir = folder / "clean"

            if not degraded_dir.is_dir() or not clean_dir.is_dir():
                raise FileNotFoundError(f"Dataset folders not found: {folder}")

            image_paths = sorted(
                path for path in degraded_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for degraded_path in image_paths:
                clean_path = clean_dir / degraded_path.name

                if not clean_path.is_file():
                    raise FileNotFoundError(
                        f"Matching clean image not found: {clean_path}"
                    )

                self.pairs.append((degraded_path, clean_path))

        if not self.pairs:
            raise RuntimeError(f"No image pairs found for split: {split}")

    def __len__(self):
        if self.split == "train":
            return len(self.pairs) * 10
        return len(self.pairs)

    def __getitem__(self, index):
        degraded_path, clean_path = self.pairs[index % len(self.pairs)]

        degraded = read_image(degraded_path)
        clean = read_image(clean_path)

        if degraded.shape != clean.shape:
            raise ValueError(
                f"Image size mismatch: {degraded_path.name}"
            )

        if self.split == "train":
            size = self.patch_size
            height, width = degraded.shape

            if height < size or width < size:
                pad_h = max(0, size - height)
                pad_w = max(0, size - width)
                degraded = np.pad(
                    degraded, ((0, pad_h), (0, pad_w)), mode="reflect"
                )
                clean = np.pad(
                    clean, ((0, pad_h), (0, pad_w)), mode="reflect"
                )
                height, width = degraded.shape

            top = random.randint(0, height - size)
            left = random.randint(0, width - size)

            degraded = degraded[top:top + size, left:left + size]
            clean = clean[top:top + size, left:left + size]

            if random.random() < 0.5:
                degraded = np.fliplr(degraded).copy()
                clean = np.fliplr(clean).copy()

            if random.random() < 0.5:
                degraded = np.flipud(degraded).copy()
                clean = np.flipud(clean).copy()

        degraded = torch.from_numpy(degraded.copy()).unsqueeze(0).float()
        clean = torch.from_numpy(clean.copy()).unsqueeze(0).float()

        return degraded, clean
