import csv
import sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SWIN_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("noise", "blur", "fading", "stains", "bleedthrough", "random_degradation")
DATA_ROOT = PROJECT_ROOT / "data"
RESTORED_ROOT = SWIN_ROOT / "outputs" / "restored"
RESULTS_DIR = SWIN_ROOT / "outputs" / "evaluation"


def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def evaluate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_image, summaries = [], []

    for category in CATEGORIES:
        test_root = DATA_ROOT / category / "test"
        degraded_dir, clean_dir = test_root / "degraded", test_root / "clean"
        restored_dir = RESTORED_ROOT / category
        for folder in (degraded_dir, clean_dir, restored_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f"Required directory missing: {folder}")

        degraded_files = sorted(p for p in degraded_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
        if not degraded_files:
            raise RuntimeError(f"No test TIFFs in {degraded_dir}")

        category_rows = []
        print(f"\n--- {category}: {len(degraded_files)} images ---")
        for degraded_path in degraded_files:
            name = degraded_path.name
            clean_path, restored_path = clean_dir / name, restored_dir / name
            if not clean_path.is_file():
                raise FileNotFoundError(f"Clean counterpart missing: {clean_path}")
            if not restored_path.is_file():
                raise FileNotFoundError(f"Restored image missing: {restored_path}; run restore.py")

            degraded, clean, restored = map(load_gray, (degraded_path, clean_path, restored_path))
            if degraded.shape != clean.shape or restored.shape != clean.shape:
                raise ValueError(
                    f"Shape mismatch for {category}/{name}: "
                    f"degraded={degraded.shape}, clean={clean.shape}, restored={restored.shape}"
                )

            dm, rm = calculate_metrics(degraded, clean), calculate_metrics(restored, clean)
            row = {
                "category": category, "filename": name,
                "degraded_mse": float(dm["mse"]), "restored_mse": float(rm["mse"]),
                "degraded_psnr": float(dm["psnr"]), "restored_psnr": float(rm["psnr"]),
                "degraded_ssim": float(dm["ssim"]), "restored_ssim": float(rm["ssim"]),
            }
            row["mse_change"] = row["restored_mse"] - row["degraded_mse"]
            row["psnr_improvement"] = row["restored_psnr"] - row["degraded_psnr"]
            row["ssim_improvement"] = row["restored_ssim"] - row["degraded_ssim"]
            category_rows.append(row)
            per_image.append(row)
            print(f"{name}: MSE {row['degraded_mse']:.5f}->{row['restored_mse']:.5f} | "
                  f"PSNR {row['degraded_psnr']:.2f}->{row['restored_psnr']:.2f} | "
                  f"SSIM {row['degraded_ssim']:.4f}->{row['restored_ssim']:.4f}")

        summary = {"category": category, "image_count": len(category_rows)}
        for metric in ("mse", "psnr", "ssim"):
            before = float(np.mean([r[f"degraded_{metric}"] for r in category_rows]))
            after = float(np.mean([r[f"restored_{metric}"] for r in category_rows]))
            summary[f"degraded_{metric}"] = before
            summary[f"restored_{metric}"] = after
            summary[f"{metric}_improvement"] = after - before
        summaries.append(summary)
        print(f"Category average | MSE {summary['degraded_mse']:.6f}->{summary['restored_mse']:.6f} "
              f"| PSNR {summary['degraded_psnr']:.3f}->{summary['restored_psnr']:.3f} "
              f"| SSIM {summary['degraded_ssim']:.4f}->{summary['restored_ssim']:.4f}")

    with (RESULTS_DIR / "per_image_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    with (RESULTS_DIR / "category_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\n=== OVERALL (mean across all test images) ===")
    print("Images evaluated:", len(per_image))
    for metric in ("mse", "psnr", "ssim"):
        before = np.mean([r[f"degraded_{metric}"] for r in per_image])
        after = np.mean([r[f"restored_{metric}"] for r in per_image])
        print(f"{metric.upper()}: {before:.6f} -> {after:.6f}")
    print("CSV results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    evaluate()
