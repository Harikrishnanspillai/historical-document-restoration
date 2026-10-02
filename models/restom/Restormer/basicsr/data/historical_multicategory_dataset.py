import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data


class HistoricalMultiCategoryDataset(data.Dataset):
    """
    Combined dataset for historical document restoration.

    Loads paired clean/degraded images from:
        noise
        blur
        fading
        stains
        bleedthrough
        random_degradation

    Expected structure:

        data/
        ├── noise/
        │   └── train/
        │       ├── clean/
        │       └── degraded/
        ├── blur/
        ├── fading/
        ├── stains/
        ├── bleedthrough/
        └── random_degradation/
    """

    EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.phase = opt.get("phase", "train")
        self.patch_size = opt.get("gt_size", 128)
        self.use_augmentation = opt.get("use_augmentation", True)

        # Six degradation categories
        self.categories = [
            "noise",
            "blur",
            "fading",
            "stains",
            "bleedthrough",
            "random_degradation",
        ]

        self.pairs = []

        # ---------------------------------------------------------
        # Collect all paired images from all six categories
        # ---------------------------------------------------------
        for category in self.categories:

            category_opt = opt["category_roots"][category]

            clean_dir = Path(category_opt["dataroot_gt"])
            degraded_dir = Path(category_opt["dataroot_lq"])

            if not clean_dir.exists():
                raise FileNotFoundError(
                    f"Clean directory not found: {clean_dir}"
                )

            if not degraded_dir.exists():
                raise FileNotFoundError(
                    f"Degraded directory not found: {degraded_dir}"
                )

            clean_files = {
                p.name: p
                for p in clean_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.EXTENSIONS
            }

            degraded_files = {
                p.name: p
                for p in degraded_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.EXTENSIONS
            }

            if set(clean_files.keys()) != set(degraded_files.keys()):
                missing_clean = set(degraded_files) - set(clean_files)
                missing_degraded = set(clean_files) - set(degraded_files)

                raise RuntimeError(
                    f"Mismatch in {category} dataset.\n"
                    f"Missing clean files: {missing_clean}\n"
                    f"Missing degraded files: {missing_degraded}"
                )

            category_pairs = [
                (
                    category,
                    degraded_files[name],
                    clean_files[name],
                )
                for name in sorted(clean_files.keys())
            ]

            self.pairs.extend(category_pairs)

            print(
                f"{category}: {len(category_pairs)} paired images"
            )

        print(
            f"\nTotal paired images across all categories: "
            f"{len(self.pairs)}"
        )

    def __len__(self):
        return len(self.pairs)

    # -------------------------------------------------------------
    # Read grayscale image
    # -------------------------------------------------------------
    def _read_image(self, path):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        image = image.astype(np.float32) / 255.0

        return image

    # -------------------------------------------------------------
    # Random paired crop
    # -------------------------------------------------------------
    def _random_crop(self, degraded, clean):

        h, w = degraded.shape

        patch = self.patch_size

        # If image is smaller than patch size,
        # resize both images to the required size.
        if h < patch or w < patch:

            degraded = cv2.resize(
                degraded,
                (patch, patch),
                interpolation=cv2.INTER_AREA
            )

            clean = cv2.resize(
                clean,
                (patch, patch),
                interpolation=cv2.INTER_AREA
            )

            return degraded, clean

        top = random.randint(0, h - patch)
        left = random.randint(0, w - patch)

        degraded = degraded[
            top:top + patch,
            left:left + patch
        ]

        clean = clean[
            top:top + patch,
            left:left + patch
        ]

        return degraded, clean

    # -------------------------------------------------------------
    # Paired augmentation
    # -------------------------------------------------------------
    def _augment(self, degraded, clean):

        # Horizontal flip
        if random.random() < 0.5:
            degraded = np.fliplr(degraded).copy()
            clean = np.fliplr(clean).copy()

        # Vertical flip
        if random.random() < 0.5:
            degraded = np.flipud(degraded).copy()
            clean = np.flipud(clean).copy()

        # 90 degree rotation
        if random.random() < 0.5:
            degraded = np.rot90(degraded).copy()
            clean = np.rot90(clean).copy()

        return degraded, clean

    # -------------------------------------------------------------
    # Get one sample
    # -------------------------------------------------------------
    def __getitem__(self, index):

        category, degraded_path, clean_path = self.pairs[index]

        degraded = self._read_image(degraded_path)
        clean = self._read_image(clean_path)

        if degraded.shape != clean.shape:
            raise RuntimeError(
                f"Image size mismatch:\n"
                f"Degraded: {degraded_path} {degraded.shape}\n"
                f"Clean: {clean_path} {clean.shape}"
            )

        # Training uses random patches
        if self.phase == "train":

            degraded, clean = self._random_crop(
                degraded,
                clean
            )

            if self.use_augmentation:
                degraded, clean = self._augment(
                    degraded,
                    clean
                )

        # Validation also uses patches so that memory usage
        # stays manageable.
        elif self.phase == "val":

            degraded, clean = self._random_crop(
                degraded,
                clean
            )

        # ---------------------------------------------------------
        # Convert to PyTorch tensors
        # [H,W] -> [1,H,W]
        # ---------------------------------------------------------
        degraded = torch.from_numpy(
            degraded.copy()
        ).float().unsqueeze(0)

        clean = torch.from_numpy(
            clean.copy()
        ).float().unsqueeze(0)

        return {
            "lq": degraded,
            "gt": clean,
            "lq_path": str(degraded_path),
            "gt_path": str(clean_path),
            "category": category,
        }