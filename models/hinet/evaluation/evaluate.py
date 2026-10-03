from pathlib import Path
import sys
import csv

import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[3]

sys.path.append(str(ROOT / "models" / "hinet" / "training"))

from dataset import CATEGORIES, read_image


def calculate_metrics(predicted, target):
    mse = np.mean((predicted - target) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    ssim = structural_similarity(target, predicted, data_range=1.0)
    return mse, psnr, ssim


def get_average_metrics(metrics):
    if not metrics:
        return None

    return {
        "degraded_mse": float(np.mean([x[0] for x in metrics])),
        "restored_mse": float(np.mean([x[3] for x in metrics])),
        "mse_change": float(np.mean([x[3] - x[0] for x in metrics])),
        "degraded_psnr": float(np.mean([x[1] for x in metrics])),
        "restored_psnr": float(np.mean([x[4] for x in metrics])),
        "psnr_gain": float(np.mean([x[4] - x[1] for x in metrics])),
        "degraded_ssim": float(np.mean([x[2] for x in metrics])),
        "restored_ssim": float(np.mean([x[5] for x in metrics])),
        "ssim_gain": float(np.mean([x[5] - x[2] for x in metrics])),
    }


def print_average(title, metrics):
    averages = get_average_metrics(metrics)

    if averages is None:
        print(f"\\nNo images evaluated for {title}.")
        return

    print(f"\\n{title}")
    print("-" * 70)
    print(f"Degraded MSE  : {averages['degraded_mse']:.6f}")
    print(f"Restored MSE  : {averages['restored_mse']:.6f}")
    print(f"MSE change    : {averages['mse_change']:+.6f}")

    print(f"\\nDegraded PSNR : {averages['degraded_psnr']:.2f} dB")
    print(f"Restored PSNR : {averages['restored_psnr']:.2f} dB")
    print(f"PSNR gain     : {averages['psnr_gain']:+.2f} dB")

    print(f"\\nDegraded SSIM : {averages['degraded_ssim']:.4f}")
    print(f"Restored SSIM : {averages['restored_ssim']:.4f}")
    print(f"SSIM gain     : {averages['ssim_gain']:+.4f}")


def save_csv_files(category_rows, image_rows):
    output_dir = Path(__file__).resolve().parent

    category_columns = [
        "category", "images_evaluated",
        "degraded_mse", "restored_mse", "mse_change",
        "degraded_psnr", "restored_psnr", "psnr_gain",
        "degraded_ssim", "restored_ssim", "ssim_gain",
    ]

    image_columns = [
        "category", "image",
        "degraded_mse", "restored_mse", "mse_change",
        "degraded_psnr", "restored_psnr", "psnr_gain",
        "degraded_ssim", "restored_ssim", "ssim_gain",
    ]

    with open(output_dir / "category_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=category_columns)
        writer.writeheader()
        writer.writerows(category_rows)

    with open(output_dir / "per_image_results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=image_columns)
        writer.writeheader()
        writer.writerows(image_rows)

    print("\\nCSV files saved:")
    print(output_dir / "category_summary.csv")
    print(output_dir / "per_image_results.csv")


def main():
    data_root = ROOT / "data"
    restored_root = ROOT / "models" / "hinet" / "restored"

    all_metrics = []
    category_rows = []
    image_rows = []

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

        print(f"\\n{'=' * 70}")
        print(f"{category.upper()} EVALUATION")
        print("=" * 70)
        print(f"Test images found: {len(image_paths)}\\n")

        for degraded_path in image_paths:
            clean_path = clean_folder / degraded_path.name
            restored_path = restored_folder / (degraded_path.stem + ".png")

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

            degraded_mse, degraded_psnr, degraded_ssim = calculate_metrics(degraded, clean)
            restored_mse, restored_psnr, restored_ssim = calculate_metrics(restored, clean)

            metrics = (
                degraded_mse, degraded_psnr, degraded_ssim,
                restored_mse, restored_psnr, restored_ssim
            )

            category_metrics.append(metrics)
            all_metrics.append(metrics)

            image_rows.append({
                "category": category,
                "image": degraded_path.name,
                "degraded_mse": degraded_mse,
                "restored_mse": restored_mse,
                "mse_change": restored_mse - degraded_mse,
                "degraded_psnr": degraded_psnr,
                "restored_psnr": restored_psnr,
                "psnr_gain": restored_psnr - degraded_psnr,
                "degraded_ssim": degraded_ssim,
                "restored_ssim": restored_ssim,
                "ssim_gain": restored_ssim - degraded_ssim,
            })

            print(
                f"{degraded_path.stem:<30} "
                f"MSE: {degraded_mse:.6f} -> {restored_mse:.6f}   "
                f"PSNR: {degraded_psnr:.2f} -> {restored_psnr:.2f} dB   "
                f"SSIM: {degraded_ssim:.4f} -> {restored_ssim:.4f}"
            )

        print_average(f"{category.upper()} AVERAGE RESULTS", category_metrics)

        averages = get_average_metrics(category_metrics)
        if averages is not None:
            category_rows.append({
                "category": category,
                "images_evaluated": len(category_metrics),
                **averages,
            })

    print(f"\\n{'=' * 70}")
    print("OVERALL AVERAGE RESULTS")
    print("=" * 70)
    print(f"Total images evaluated: {len(all_metrics)}")
    print_average("OVERALL RESULTS", all_metrics)

    overall_averages = get_average_metrics(all_metrics)
    if overall_averages is not None:
        category_rows.append({
            "category": "overall",
            "images_evaluated": len(all_metrics),
            **overall_averages,
        })

    save_csv_files(category_rows, image_rows)
    print("=" * 70)


if __name__ == "__main__":
    main()
