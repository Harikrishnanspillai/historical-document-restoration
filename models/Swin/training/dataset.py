from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CATEGORIES = ("noise", "blur", "fading", "stains", "bleedthrough", "random_degradation")


class HistoricalDocumentDataset(Dataset):
    """Loads paired clean/degraded TIFFs from all category folders."""

    EXTENSIONS = {".tif", ".tiff"}

    def __init__(self, root_dir, split="train", categories=CATEGORIES,
                 patch_size=128, patches_per_image=10):
        self.root_dir = Path(root_dir)
        self.split = split
        self.categories = tuple(categories)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.pairs = []
        self.category_counts = {}
        for category in self.categories:
            folder = self.root_dir / category / split
            clean_dir, degraded_dir = folder / "clean", folder / "degraded"
            if not clean_dir.is_dir() or not degraded_dir.is_dir():
                raise FileNotFoundError(f"Expected clean/degraded folders under {folder}")

            clean = {p.name: p for p in clean_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in self.EXTENSIONS}
            degraded = {p.name: p for p in degraded_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in self.EXTENSIONS}
            if set(clean) != set(degraded):
                raise RuntimeError(
                    f"Filename mismatch in {category}/{split}; "
                    f"missing clean={sorted(set(degraded)-set(clean))[:5]}, "
                    f"missing degraded={sorted(set(clean)-set(degraded))[:5]}"
                )
            pairs = [(category, degraded[n], clean[n]) for n in sorted(degraded)]
            if not pairs:
                raise RuntimeError(f"No TIFF pairs found in {folder}")
            self.pairs.extend(pairs)
            self.category_counts[category] = len(pairs)

        self.samples_per_image = patches_per_image if split in {"train", "val"} else 1
        self.total_samples = len(self.pairs) * self.samples_per_image
        print(f"[{split.upper()}] {len(self.pairs)} pairs across {len(self.categories)} categories")
        for category, count in self.category_counts.items():
            print(f"  {category}: {count}")
        print(f"[{split.upper()}] Dataset samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    @staticmethod
    def _read(path):
        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)

    def __getitem__(self, index):
        pair_index = index // self.samples_per_image
        category, degraded_path, clean_path = self.pairs[pair_index]
        degraded, clean = self._read(degraded_path), self._read(clean_path)
        if degraded.shape != clean.shape:
            raise ValueError(f"Dimension mismatch: {category}/{degraded_path.name}")

        if self.split in {"train", "val"}:
            h, w = degraded.shape
            pad_h, pad_w = max(0, self.patch_size-h), max(0, self.patch_size-w)
            if pad_h or pad_w:
                mode = "reflect" if h > pad_h and w > pad_w else "edge"
                degraded = np.pad(degraded, ((0,pad_h),(0,pad_w)), mode=mode)
                clean = np.pad(clean, ((0,pad_h),(0,pad_w)), mode=mode)
            h, w = degraded.shape
            top = random.randint(0, h-self.patch_size)
            left = random.randint(0, w-self.patch_size)
            degraded = degraded[top:top+self.patch_size, left:left+self.patch_size]
            clean = clean[top:top+self.patch_size, left:left+self.patch_size]

        degraded = torch.from_numpy((degraded / 255.0).copy()).unsqueeze(0)
        clean = torch.from_numpy((clean / 255.0).copy()).unsqueeze(0)
        return degraded, clean


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    ds = HistoricalDocumentDataset(project_root / "data", "train", patch_size=128, patches_per_image=10)
    x, y = ds[0]
    print("Samples:", len(ds), "| tensors:", tuple(x.shape), tuple(y.shape))
    print("Ranges:", (x.min().item(), x.max().item()), (y.min().item(), y.max().item()))
