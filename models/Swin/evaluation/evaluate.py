import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SWIN_ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = (
    "noise",
    "blur",
    "fading",
    "stains",
    "bleedthrough",
    "random_degradation"
)

DATA_ROOT = PROJECT_ROOT / "data"
RESTORED_ROOT = SWIN_ROOT / "outputs" / "restored"
RESULTS_DIR = SWIN_ROOT / "outputs" / "evaluation"


def load_gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def calculate_metrics(predicted, target):
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: predicted={predicted.shape}, target={target.shape}"
        )

    mse = float(np.mean((predicted - target) ** 2))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(peak_signal_noise_ratio(target, predicted, data_range=1.0))

    ssim = float(structural_similarity(target, predicted, data_range=1.0))

    return mse, psnr, ssim


def print_average(title, metrics):
    degraded_mse, degraded_psnr, degraded_ssim, restored_mse, restored_psnr, restored_ssim = metrics

    print(f"\n{title}")
    print(f"MSE:  {degraded_mse:.6f} -> {restored_mse:.6f}")
    print(f"PSNR: {degraded_psnr:.3f} -> {restored_psnr:.3f}")
    print(f"SSIM: {degraded_ssim:.4f} -> {restored_ssim:.4f}")


def evaluate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    category_summaries = []

    for category in CATEGORIES:
        test_root = DATA_ROOT / category / "test"

        degraded_dir = test_root / "degraded"
        clean_dir = test_root / "clean"
        restored_dir = RESTORED_ROOT / category

        for folder in (degraded_dir, clean_dir, restored_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f"Required directory missing: {folder}")

        image_paths = sorted(
            path for path in degraded_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        )

        if not image_paths:
            raise RuntimeError(f"No test TIFF images found in {degraded_dir}")

        category_rows = []

        print(f"\n--- {category}: {len(image_paths)} images ---")

        for image_path in image_paths:
            name = image_path.name

            clean_path = clean_dir / name
            restored_path = restored_dir / name

            if not clean_path.is_file():
                raise FileNotFoundError(f"Clean image not found: {clean_path}")

            if not restored_path.is_file():
                raise FileNotFoundError(
                    f"Restored image not found: {restored_path}. Run restore.py first."
                )

            degraded = load_gray(image_path)
            clean = load_gray(clean_path)
            restored = load_gray(restored_path)

            if degraded.shape != clean.shape or restored.shape != clean.shape:
                raise ValueError(
                    f"Image shape mismatch for {category}/{name}: "
                    f"degraded={degraded.shape}, clean={clean.shape}, "
                    f"restored={restored.shape}"
                )

            degraded_metrics = calculate_metrics(degraded, clean)
            restored_metrics = calculate_metrics(restored, clean)

            row = {
                "category": category,
                "filename": name,
                "degraded_mse": degraded_metrics[0],
                "restored_mse": restored_metrics[0],
                "degraded_psnr": degraded_metrics[1],
                "restored_psnr": restored_metrics[1],
                "degraded_ssim": degraded_metrics[2],
                "restored_ssim": restored_metrics[2]
            }

            row["mse_change"] = row["restored_mse"] - row["degraded_mse"]
            row["psnr_improvement"] = row["restored_psnr"] - row["degraded_psnr"]
            row["ssim_improvement"] = row["restored_ssim"] - row["degraded_ssim"]

            category_rows.append(row)
            all_rows.append(row)

            print(
                f"{name}: "
                f"MSE {row['degraded_mse']:.5f}->{row['restored_mse']:.5f} | "
                f"PSNR {row['degraded_psnr']:.2f}->{row['restored_psnr']:.2f} | "
                f"SSIM {row['degraded_ssim']:.4f}->{row['restored_ssim']:.4f}"
            )

        category_metrics = []

        for metric in ("mse", "psnr", "ssim"):
            before = float(np.mean([row[f"degraded_{metric}"] for row in category_rows]))
            after = float(np.mean([row[f"restored_{metric}"] for row in category_rows]))
            category_metrics.extend([before, after])

        summary = {
            "category": category,
            "image_count": len(category_rows),
            "degraded_mse": category_metrics[0],
            "restored_mse": category_metrics[1],
            "degraded_psnr": category_metrics[2],
            "restored_psnr": category_metrics[3],
            "degraded_ssim": category_metrics[4],
            "restored_ssim": category_metrics[5]
        }

        summary["mse_improvement"] = summary["degraded_mse"] - summary["restored_mse"]
        summary["psnr_improvement"] = summary["restored_psnr"] - summary["degraded_psnr"]
        summary["ssim_improvement"] = summary["restored_ssim"] - summary["degraded_ssim"]

        category_summaries.append(summary)

        print_average(
            f"{category.upper()} AVERAGE",
            (
                summary["degraded_mse"],
                summary["degraded_psnr"],
                summary["degraded_ssim"],
                summary["restored_mse"],
                summary["restored_psnr"],
                summary["restored_ssim"]
            )
        )

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

    overall_metrics = []

    for metric in ("mse", "psnr", "ssim"):
        before = float(np.mean([row[f"degraded_{metric}"] for row in all_rows]))
        after = float(np.mean([row[f"restored_{metric}"] for row in all_rows]))
        overall_metrics.extend([before, after])

    print_average(
        "OVERALL AVERAGE",
        (
            overall_metrics[0],
            overall_metrics[2],
            overall_metrics[4],
            overall_metrics[1],
            overall_metrics[3],
            overall_metrics[5]
        )
    )

    print("\nImages evaluated:", len(all_rows))
    print("CSV results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    evaluate()
