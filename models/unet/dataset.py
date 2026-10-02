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

        self.degraded_dir = self.root / "degraded"
        self.clean_dir = self.root / "clean"

        self.patch_size = patch_size
        self.augment = augment

        if not self.degraded_dir.exists():
            raise FileNotFoundError(
                f"Degraded directory not found: {self.degraded_dir}"
            )

        if not self.clean_dir.exists():
            raise FileNotFoundError(
                f"Clean directory not found: {self.clean_dir}"
            )

        self.filenames = sorted(
            f.name
            for f in self.degraded_dir.iterdir()
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
                f"No images found in {self.degraded_dir}"
            )

        missing = [
            f
            for f in self.filenames
            if not (self.clean_dir / f).exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no matching "
                f"ground truth. Example: {missing[0]}"
            )

    def __len__(self):
        return len(self.filenames)

    def _load_patch(self, path, top, left, patch_size):
        image = Image.open(path).convert("L")

        patch = image.crop(
            (
                left,
                top,
                left + patch_size,
                top + patch_size
            )
        )

        return np.asarray(
            patch,
            dtype=np.float32
        ) / 255.0

    def __getitem__(self, idx):

        filename = self.filenames[idx]

        degraded_path = self.degraded_dir / filename
        clean_path = self.clean_dir / filename

        # Get image dimensions without converting the whole image
        with Image.open(degraded_path) as degraded_image:
            degraded_width, degraded_height = degraded_image.size

        with Image.open(clean_path) as clean_image:
            clean_width, clean_height = clean_image.size

        # Make sure dimensions match
        if (
            degraded_width != clean_width
            or degraded_height != clean_height
        ):
            raise ValueError(
                f"Size mismatch for {filename}: "
                f"degraded={degraded_width}x{degraded_height}, "
                f"clean={clean_width}x{clean_height}"
            )

        ps = self.patch_size

        if degraded_height < ps or degraded_width < ps:
            raise ValueError(
                f"{filename} is smaller than patch size "
                f"{ps}: image size is "
                f"{degraded_width}x{degraded_height}"
            )

        # Random crop coordinates
        top = random.randint(
            0,
            degraded_height - ps
        )

        left = random.randint(
            0,
            degraded_width - ps
        )

        # Load only the required patch
        degraded_patch = self._load_patch(
            degraded_path,
            top,
            left,
            ps
        )

        clean_patch = self._load_patch(
            clean_path,
            top,
            left,
            ps
        )

        # Data augmentation
        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:
                degraded_patch = np.fliplr(
                    degraded_patch
                ).copy()

                clean_patch = np.fliplr(
                    clean_patch
                ).copy()

            # Vertical flip
            if random.random() < 0.5:
                degraded_patch = np.flipud(
                    degraded_patch
                ).copy()

                clean_patch = np.flipud(
                    clean_patch
                ).copy()

            # Random rotation
            k = random.randint(0, 3)

            degraded_patch = np.rot90(
                degraded_patch,
                k
            ).copy()

            clean_patch = np.rot90(
                clean_patch,
                k
            ).copy()

        # Add channel dimension
        # (64, 64) -> (1, 64, 64)
        input_patch = degraded_patch[None, :, :]
        target_patch = clean_patch[None, :, :]

        return (
            torch.from_numpy(
                input_patch.astype(np.float32)
            ),
            torch.from_numpy(
                target_patch.astype(np.float32)
            ),
        )