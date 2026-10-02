import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CATEGORIES = (
    "bleedthrough",
    "blur",
    "fading",
    "noise",
    "random_degradation",
    "stains"
)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


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
            degraded_dir = self.data_root / category / split / "degraded"
            clean_dir = self.data_root / category / split / "clean"

            if not degraded_dir.is_dir() or not clean_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing dataset folders for {category}/{split}"
                )

            degraded_files = sorted(
                path for path in degraded_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for degraded_path in degraded_files:
                clean_path = clean_dir / degraded_path.name

                if not clean_path.is_file():
                    raise FileNotFoundError(
                        f"Clean counterpart not found: {clean_path}"
                    )

                self.pairs.append((degraded_path, clean_path, category))

        if not self.pairs:
            raise RuntimeError(f"No image pairs found for split: {split}")

    def __len__(self):
        if self.split == "train":
            return len(self.pairs) * 10
        return len(self.pairs)

    def __getitem__(self, index):
        pair_index = index % len(self.pairs)
        degraded_path, clean_path, category = self.pairs[pair_index]

        degraded = read_image(degraded_path)
        clean = read_image(clean_path)

        if degraded.shape != clean.shape:
            raise ValueError(
                f"Image size mismatch for {category}/{degraded_path.name}: "
                f"{degraded.shape} and {clean.shape}"
            )

        if self.split == "train":
            height, width = degraded.shape
            size = self.patch_size

            pad_h = max(0, size - height)
            pad_w = max(0, size - width)

            if pad_h or pad_w:
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

        degraded = torch.from_numpy(np.ascontiguousarray(degraded)).float().unsqueeze(0)
        clean = torch.from_numpy(np.ascontiguousarray(clean)).float().unsqueeze(0)

        return degraded, clean
