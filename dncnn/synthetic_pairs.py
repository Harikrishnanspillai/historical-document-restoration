"""
Generate synthetic (degraded, clean) training pairs by simulating
ink bleed-through on clean text images.

Why this exists: real recto-verso datasets with manually-verified
ground truth are scarce (this is the "research gap" your lit review
identifies). Savino & Tonazzini (2024) get around this by training on
patches generated from a mathematical model of ink penetration rather
than needing a large labelled dataset. This script does the same thing:

    degraded = clean * (1 - alpha) + flipped(other_page) * alpha + noise

Usage:
    python synthetic_pairs.py --clean_dir data/clean_source \
                               --out_dir data --num_pairs 2000

`clean_dir` should contain clean, non-degraded text page images (e.g.
scans of printed or clearly-written text with no bleed-through -- these
become your ground truth). The script pairs them up randomly to
simulate one page's ink bleeding through onto another, and writes
matching recto/ and clean/ folders ready for train.py.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def load_gray(path, size=None):
    img = Image.open(path).convert("L")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def simulate_bleed_through(clean, interference_source, alpha=0.35, blur_sigma=0.6, noise_std=0.02):
    """
    clean: the foreground page (H, W) in [0, 1], 1 = white background, 0 = ink
    interference_source: another text page whose *mirrored* content will
        "bleed through" onto `clean`
    alpha: strength of the bleed-through (0.2-0.5 is realistic)
    """
    h, w = clean.shape
    interference = interference_source
    if interference.shape != clean.shape:
        interference = np.asarray(
            Image.fromarray((interference * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0

    # mirror horizontally to simulate "seeing through the page from the other side"
    interference = np.fliplr(interference)
    interference = gaussian_filter(interference, sigma=blur_sigma)  # ink diffuses through paper fiber

    # blend: darker pixels (ink) on either side pull the composite darker
    degraded = clean * (1 - alpha) + np.minimum(clean, interference) * alpha
    noise = np.random.normal(0, noise_std, size=degraded.shape).astype(np.float32)
    degraded = np.clip(degraded + noise, 0.0, 1.0)
    return degraded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--num_pairs", type=int, default=2000)
    p.add_argument("--alpha_range", type=float, nargs=2, default=[0.2, 0.5])
    args = p.parse_args()

    clean_paths = sorted(
        f for f in Path(args.clean_dir).iterdir()
        if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    )
    if len(clean_paths) < 2:
        raise ValueError("Need at least 2 clean source images to simulate cross-page bleed-through.")

    out_recto = Path(args.out_dir) / "recto"
    out_clean = Path(args.out_dir) / "clean"
    out_recto.mkdir(parents=True, exist_ok=True)
    out_clean.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_pairs):
        fg_path, bg_path = random.sample(clean_paths, 2)
        clean = load_gray(fg_path)
        interference = load_gray(bg_path)
        alpha = random.uniform(*args.alpha_range)

        degraded = simulate_bleed_through(clean, interference, alpha=alpha)

        name = f"synth_{i:05d}.png"
        Image.fromarray((degraded * 255).astype(np.uint8)).save(out_recto / name)
        Image.fromarray((clean * 255).astype(np.uint8)).save(out_clean / name)

        if (i + 1) % 200 == 0:
            print(f"Generated {i+1}/{args.num_pairs} pairs")

    print(f"Done. Wrote {args.num_pairs} pairs to {out_recto} and {out_clean}")


if __name__ == "__main__":
    main()
