import csv
import sys
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "unet"

sys.path.insert(0, str(MODEL_DIR / "training"))

from dataset import CATEGORIES, read_image


DATA_ROOT = ROOT / "data"
RESTORED_ROOT = MODEL_DIR / "restored"
RESULTS_DIR = MODEL_DIR / "evaluation"

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}


def calculate_metrics(predicted, target):
    mse = np.mean((predicted - target) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    ssim = structural_similarity(target, predicted, data_range=1.0)
    return float(mse), float(psnr), float(ssim)


def print_average(title, metrics):
    if not metrics:
        print(f"\nNo images evaluated for {title}.")
        return

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_metrics = []
    category_summaries = []

    print("=" * 70)
    print("U-Net Historical Document Restoration Evaluation")
    print("=" * 70)

    for category in CATEGORIES:
        degraded_dir = DATA_ROOT / category / "test" / "degraded"
        clean_dir = DATA_ROOT / category / "test" / "clean"
        restored_dir = RESTORED_ROOT / category

        for folder in (degraded_dir, clean_dir, restored_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f"Required directory missing: {folder}")

        image_paths = sorted(
            path for path in degraded_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        category_metrics = []
        category_rows = []

        print(f"\n{'=' * 70}")
        print(f"{category.upper()} EVALUATION")
        print("=" * 70)
        print(f"Test images found: {len(image_paths)}\n")

        for degraded_path in image_paths:
            clean_path = clean_dir / degraded_path.name
            restored_path = restored_dir / f"{degraded_path.stem}.png"

            if not clean_path.is_file():
                raise FileNotFoundError(f"Clean image not found: {clean_path}")
            if not restored_path.is_file():
                raise FileNotFoundError(
                    f"Restored image not found: {restored_path}. Run restore.py first."
                )

            degraded = read_image(degraded_path)
            clean = read_image(clean_path)
            restored = read_image(restored_path)

            if degraded.shape != clean.shape or restored.shape != clean.shape:
                raise ValueError(
                    f"Shape mismatch for {category}/{degraded_path.name}: "
                    f"degraded={degraded.shape}, clean={clean.shape}, "
                    f"restored={restored.shape}"
                )

            before = calculate_metrics(degraded, clean)
            after = calculate_metrics(restored, clean)

            metrics = (*before, *after)
            category_metrics.append(metrics)
            all_metrics.append(metrics)

            row = {
                "category": category,
                "filename": degraded_path.name,
                "degraded_mse": before[0],
                "restored_mse": after[0],
                "degraded_psnr": before[1],
                "restored_psnr": after[1],
                "degraded_ssim": before[2],
                "restored_ssim": after[2],
                "mse_change": after[0] - before[0],
                "psnr_improvement": after[1] - before[1],
                "ssim_improvement": after[2] - before[2]
            }
            category_rows.append(row)
            all_rows.append(row)

            print(
                f"{degraded_path.stem:<30} "
                f"MSE: {before[0]:.6f} -> {after[0]:.6f}   "
                f"PSNR: {before[1]:.2f} -> {after[1]:.2f} dB   "
                f"SSIM: {before[2]:.4f} -> {after[2]:.4f}"
            )

        print_average(f"{category.upper()} AVERAGE RESULTS", category_metrics)

        if category_rows:
            summary = {"category": category, "image_count": len(category_rows)}
            for metric in ("mse", "psnr", "ssim"):
                summary[f"degraded_{metric}"] = float(
                    np.mean([row[f"degraded_{metric}"] for row in category_rows])
                )
                summary[f"restored_{metric}"] = float(
                    np.mean([row[f"restored_{metric}"] for row in category_rows])
                )
                summary[f"{metric}_improvement"] = (
                    summary[f"restored_{metric}"] - summary[f"degraded_{metric}"]
                )
            category_summaries.append(summary)

    if not all_rows:
        raise RuntimeError("No test images were evaluated.")

    with (RESULTS_DIR / "per_image_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with (RESULTS_DIR / "category_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(category_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(category_summaries)

    print(f"\n{'=' * 70}")
    print("OVERALL AVERAGE RESULTS")
    print("=" * 70)
    print("Total images evaluated:", len(all_metrics))
    print_average("OVERALL RESULTS", all_metrics)
    print("=" * 70)
    print("CSV results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
