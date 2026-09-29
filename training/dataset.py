from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HistoricalDocumentDataset(Dataset):
    """
    Paired historical-document dataset with random patch extraction.

    Expected structure:

    dataset/
    ├── train/
    │   ├── degraded/
    │   └── gt/
    ├── val/
    │   ├── degraded/
    │   └── gt/
    └── test/
        ├── degraded/
        └── gt/
    """

    def __init__(
        self,
        root_dir,
        split="train",
        patch_size=128,
        patches_per_image=20
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

        self.degraded_dir = self.root_dir / split / "degraded"
        self.gt_dir = self.root_dir / split / "gt"

        if not self.degraded_dir.exists():
            raise FileNotFoundError(
                f"Degraded directory not found: {self.degraded_dir}"
            )

        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"Ground-truth directory not found: {self.gt_dir}"
            )

        self.degraded_files = sorted(
            self.degraded_dir.glob("*.tif")
        )

        if not self.degraded_files:
            raise RuntimeError(
                f"No TIFF images found in {self.degraded_dir}"
            )

        self.pairs = []

        for degraded_path in self.degraded_files:
            gt_path = self.gt_dir / (
                degraded_path.stem + ".gt.tif"
            )

            if not gt_path.exists():
                raise FileNotFoundError(
                    f"GT missing for {degraded_path.name}"
                )

            self.pairs.append(
                (degraded_path, gt_path)
            )

        # Training gets multiple random patches per document.
        # Validation/test will eventually use full-image tiled inference.
        if split in ("train", "val"):
            self.total_samples = (
                len(self.pairs) * patches_per_image
            )
        else:
            self.total_samples = len(self.pairs)

        print(
            f"[{split.upper()}] Loaded {len(self.pairs)} image pairs"
        )

        if split in ("train", "val"):
            print(
                f"[{split.upper()}] "
                f"{patches_per_image} patches per image"
            )
            print(
                f"[{split.upper()}] "
                f"Total training samples: {self.total_samples}"
            )

    def __len__(self):
        return self.total_samples

    def _load_pair(self, index):
        degraded_path, gt_path = self.pairs[index]

        degraded = Image.open(degraded_path).convert("L")
        gt = Image.open(gt_path).convert("L")

        degraded = np.asarray(
            degraded,
            dtype=np.float32
        )

        gt = np.asarray(
            gt,
            dtype=np.float32
        )

        if degraded.shape != gt.shape:
            raise ValueError(
                f"Dimension mismatch:\n"
                f"Degraded: {degraded_path.name} "
                f"{degraded.shape}\n"
                f"GT: {gt_path.name} "
                f"{gt.shape}"
            )

        return degraded, gt

    def __getitem__(self, index):

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        if self.split in ("train", "val"):

            # Map sample index to an actual document.
            image_index = index // self.patches_per_image

            degraded, gt = self._load_pair(image_index)

            height, width = degraded.shape

            if (
                height < self.patch_size
                or width < self.patch_size
            ):
                raise ValueError(
                    f"Image is smaller than patch size: "
                    f"{width}x{height}"
                )

            # Random patch location
            top = random.randint(
                0,
                height - self.patch_size
            )

            left = random.randint(
                0,
                width - self.patch_size
            )

            degraded = degraded[
                top:top + self.patch_size,
                left:left + self.patch_size
            ]

            gt = gt[
                top:top + self.patch_size,
                left:left + self.patch_size
            ]

        # ----------------------------------------------------
        # VALIDATION / TEST
        # ----------------------------------------------------

        else:

            degraded, gt = self._load_pair(index)

        # Normalize [0,255] -> [0,1]
        degraded = degraded / 255.0
        gt = gt / 255.0

        # H x W -> 1 x H x W
        degraded = torch.from_numpy(
            degraded.copy()
        ).unsqueeze(0)

        gt = torch.from_numpy(
            gt.copy()
        ).unsqueeze(0)

        return degraded, gt


if __name__ == "__main__":

    print("=" * 60)
    print("Testing training dataset")
    print("=" * 60)

    dataset = HistoricalDocumentDataset(
        root_dir="dataset",
        split="train",
        patch_size=128,
        patches_per_image=20
    )

    print()
    print(f"Dataset length: {len(dataset)}")

    degraded, gt = dataset[0]

    print(f"Degraded shape: {degraded.shape}")
    print(f"GT shape:       {gt.shape}")

    print(
        f"Degraded range: "
        f"{degraded.min():.4f} - {degraded.max():.4f}"
    )

    print(
        f"GT range:       "
        f"{gt.min():.4f} - {gt.max():.4f}"
    )