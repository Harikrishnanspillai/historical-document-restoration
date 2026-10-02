
from pathlib import Path
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[3]

sys.path.append(
    str(ROOT / "models" / "hinet" / "training")
)

from dataset import CATEGORIES, read_image


def calculate_metrics(predicted, target):
    mse = np.mean((predicted - target) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    ssim = structural_similarity(target, predicted, data_range=1.0)

    return mse, psnr, ssim


def print_average(title, metrics):
    if not metrics:
        print(f"\nNo images evaluated for {title}.")
        return

    avg_mse = np.mean([x[0] for x in metrics])
    avg_psnr = np.mean([x[1] for x in metrics])
    avg_ssim = np.mean([x[2] for x in metrics])

    print(f"\n{title}")
    print("-" * 70)

    print(f"Degraded MSE  : {np.mean([x[0] for x in metrics]):.6f}")
    print(f"Restored MSE  : {np.mean([x[3] for x in metrics]):.6f}")
    print(f"MSE change    : {np.mean([x[3] - x[0] for x in metrics]):+.6f}")

    print(f"\nDegraded PSNR : {np.mean([x[1] for x in metrics]):.2f} dB")
    print(f"Restored PSNR : {np.mean([x[4] for x in metrics]):.2f} dB")
    print(f"PSNR gain     : {np.mean([x[4] - x[1] for x in metrics]):+.2f} dB")

    print(f"\nDegraded SSIM : {np.mean([x[2] for x in metrics]):.4f}")
    print(f"Restored SSIM : {np.mean([x[5] for x in metrics]):.4f}")
    print(f"SSIM gain     : {np.mean([x[5] - x[2] for x in metrics]):+.4f}")


def main():
    data_root = ROOT / "data"
    restored_root = ROOT / "models" / "hinet" / "restored"

    all_metrics = []

    print("=" * 70)
    print("HINet Historical Document Restoration Evaluation")
    print("=" * 70)

    for category in CATEGORIES:
        degraded_folder = data_root / category / "test" / "degraded"
        clean_folder = data_root / category / "test" / "clean"
        restored_folder = restored_root / category

        category_metrics = []

        image_paths = sorted(
            list(degraded_folder.glob("*.tif")) +
            list(degraded_folder.glob("*.tiff")) +
            list(degraded_folder.glob("*.png"))
        )

        print(f"\n{'=' * 70}")
        print(f"{category.upper()} EVALUATION")
        print("=" * 70)
        print(f"Test images found: {len(image_paths)}\n")

        for degraded_path in image_paths:
            clean_path = clean_folder / degraded_path.name

            # restore.py saves images as PNG using the original filename stem.
            restored_path = restored_folder / (degraded_path.stem + ".png")

            # Also allow restored images to retain their original extension.
            if not restored_path.exists():
                restored_path = restored_folder / degraded_path.name

            if not clean_path.exists():
                print(f"Missing clean image: {clean_path.name}")
                continue

            if not restored_path.exists():
                print(f"Missing restored image: {restored_path.name}")
                continue

            degraded = read_image(degraded_path)
            clean = read_image(clean_path)
            restored = read_image(restored_path)

            if degraded.shape != clean.shape or restored.shape != clean.shape:
                print(f"Skipping {degraded_path.name}: image dimensions do not match.")
                continue

            degraded_mse, degraded_psnr, degraded_ssim = calculate_metrics(
                degraded, clean
            )

            restored_mse, restored_psnr, restored_ssim = calculate_metrics(
                restored, clean
            )

            metrics = (
                degraded_mse, degraded_psnr, degraded_ssim,
                restored_mse, restored_psnr, restored_ssim
            )

            category_metrics.append(metrics)
            all_metrics.append(metrics)

            name = degraded_path.stem

            print(
                f"{name:<30} "
                f"MSE: {degraded_mse:.6f} → {restored_mse:.6f}   "
                f"PSNR: {degraded_psnr:.2f} → {restored_psnr:.2f} dB   "
                f"SSIM: {degraded_ssim:.4f} → {restored_ssim:.4f}"
            )

        print_average(f"{category.upper()} AVERAGE RESULTS", category_metrics)

    print(f"\n{'=' * 70}")
    print("OVERALL AVERAGE RESULTS")
    print("=" * 70)

    print(f"Total images evaluated: {len(all_metrics)}")

    print_average("OVERALL RESULTS", all_metrics)

    print("=" * 70)


if __name__ == "__main__":
    main()
