import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ManuscriptDataset(Dataset):
    def __init__(
        self,
        root_dir,
        patch_size=64,
        augment=False,
    ):
        self.root = Path(root_dir)

        self.recto_dir = self.root / "recto"
        self.clean_dir = self.root / "clean"

        self.patch_size = patch_size
        self.augment = augment

        if not self.recto_dir.exists():
            raise FileNotFoundError(
                f"Recto directory not found: {self.recto_dir}"
            )

        if not self.clean_dir.exists():
            raise FileNotFoundError(
                f"Clean directory not found: {self.clean_dir}"
            )

        # Find all degraded images
        self.filenames = sorted(
            f.name
            for f in self.recto_dir.iterdir()
            if f.suffix.lower() in (
                ".tif",
                ".tiff",
                ".png",
                ".jpg",
                ".jpeg",
            )
        )

        if not self.filenames:
            raise FileNotFoundError(
                f"No images found in {self.recto_dir}"
            )

        # Make sure every degraded image has a matching GT
        missing = [
            f for f in self.filenames
            if not (self.clean_dir / f).exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no matching "
                f"ground truth. Example: {missing[0]}"
            )

    def __len__(self):
        return len(self.filenames)

    def _load_gray(self, path):
        image = Image.open(path).convert("L")
        return np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    def __getitem__(self, idx):

        filename = self.filenames[idx]

        # Load degraded image
        recto = self._load_gray(
            self.recto_dir / filename
        )

        # Load clean ground truth
        clean = self._load_gray(
            self.clean_dir / filename
        )

        # Make sure dimensions match
        if recto.shape != clean.shape:
            raise ValueError(
                f"Size mismatch for {filename}: "
                f"recto={recto.shape}, clean={clean.shape}"
            )

        h, w = recto.shape
        ps = self.patch_size

        if h < ps or w < ps:
            raise ValueError(
                f"{filename} is smaller than patch size "
                f"{ps}: image size is {h}x{w}"
            )

        # Random crop
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)

        recto_patch = recto[
            top:top + ps,
            left:left + ps
        ]

        clean_patch = clean[
            top:top + ps,
            left:left + ps
        ]

        # Data augmentation
        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:
                recto_patch = np.fliplr(
                    recto_patch
                ).copy()

                clean_patch = np.fliplr(
                    clean_patch
                ).copy()

            # Vertical flip
            if random.random() < 0.5:
                recto_patch = np.flipud(
                    recto_patch
                ).copy()

                clean_patch = np.flipud(
                    clean_patch
                ).copy()

            # Random rotation
            k = random.randint(0, 3)

            recto_patch = np.rot90(
                recto_patch,
                k
            ).copy()

            clean_patch = np.rot90(
                clean_patch,
                k
            ).copy()

        # Add channel dimension
        # (64, 64) -> (1, 64, 64)
        input_patch = recto_patch[None, :, :]
        target_patch = clean_patch[None, :, :]

        return (
            torch.from_numpy(
                input_patch.astype(np.float32)
            ),
            torch.from_numpy(
                target_patch.astype(np.float32)
            ),
        )