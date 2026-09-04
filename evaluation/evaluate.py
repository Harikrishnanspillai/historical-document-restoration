from pathlib import Path

import numpy as np
from PIL import Image

from metrics import calculate_metrics


# ============================================================
# Configuration
# ============================================================

TEST_DIR = Path("dataset/test")
DEGRADED_DIR = TEST_DIR / "degraded"
GT_DIR = TEST_DIR / "gt"

# This directory will eventually contain SwinIR outputs.
RESTORED_DIR = Path("outputs/restored")


# ============================================================
# Image loading
# ============================================================

def load_grayscale(path):
    """
    Load an image as grayscale and normalize to [0, 1].
    """

    image = Image.open(path).convert("L")

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image /= 255.0

    return image


# ============================================================
# Main evaluation
# ============================================================

def evaluate():

    print("=" * 70)
    print("Historical Document Restoration Evaluation")
    print("=" * 70)

    if not DEGRADED_DIR.exists():
        raise FileNotFoundError(
            f"Degraded test directory not found: {DEGRADED_DIR}"
        )

    if not GT_DIR.exists():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {GT_DIR}"
        )

    if not RESTORED_DIR.exists():
        print()
        print(
            f"Restored image directory does not exist yet:"
            f"\n{RESTORED_DIR}"
        )
        print()
        print(
            "This is expected until the SwinIR inference step "
            "has been implemented."
        )
        return

    degraded_files = sorted(
        DEGRADED_DIR.glob("*.tif")
    )

    if not degraded_files:
        raise RuntimeError(
            f"No TIFF files found in {DEGRADED_DIR}"
        )

    results = []

    print()
    print(f"Test images found: {len(degraded_files)}")
    print()

    for degraded_path in degraded_files:

        # ----------------------------------------------------
        # File names
        # ----------------------------------------------------

        image_name = degraded_path.stem

        gt_path = GT_DIR / (
            image_name + ".gt.tif"
        )

        restored_path = RESTORED_DIR / (
            image_name + ".png"
        )

        # ----------------------------------------------------
        # Check required files
        # ----------------------------------------------------

        if not gt_path.exists():

            print(
                f"[SKIP] GT missing: "
                f"{gt_path.name}"
            )

            continue

        if not restored_path.exists():

            print(
                f"[SKIP] Restored image missing: "
                f"{restored_path.name}"
            )

            continue

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        degraded = load_grayscale(
            degraded_path
        )

        restored = load_grayscale(
            restored_path
        )

        gt = load_grayscale(
            gt_path
        )

        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        if degraded.shape != gt.shape:

            raise ValueError(
                f"Degraded/GT dimension mismatch for "
                f"{image_name}:\n"
                f"Degraded: {degraded.shape}\n"
                f"GT:       {gt.shape}"
            )

        if restored.shape != gt.shape:

            raise ValueError(
                f"Restored/GT dimension mismatch for "
                f"{image_name}:\n"
                f"Restored: {restored.shape}\n"
                f"GT:       {gt.shape}"
            )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        degraded_metrics = calculate_metrics(
            degraded,
            gt
        )

        restored_metrics = calculate_metrics(
            restored,
            gt
        )

        psnr_improvement = (
            restored_metrics["psnr"]
            - degraded_metrics["psnr"]
        )

        ssim_improvement = (
            restored_metrics["ssim"]
            - degraded_metrics["ssim"]
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({
            "name": image_name,

            "degraded_psnr":
                degraded_metrics["psnr"],

            "degraded_ssim":
                degraded_metrics["ssim"],

            "restored_psnr":
                restored_metrics["psnr"],

            "restored_ssim":
                restored_metrics["ssim"],

            "psnr_improvement":
                psnr_improvement,

            "ssim_improvement":
                ssim_improvement
        })

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"{image_name:<30} "
            f"PSNR: "
            f"{degraded_metrics['psnr']:.2f} → "
            f"{restored_metrics['psnr']:.2f} dB   "
            f"SSIM: "
            f"{degraded_metrics['ssim']:.4f} → "
            f"{restored_metrics['ssim']:.4f}"
        )

    # ========================================================
    # Overall results
    # ========================================================

    if not results:

        print()
        print("No images were evaluated.")
        return

    avg_degraded_psnr = np.mean([
        r["degraded_psnr"]
        for r in results
    ])

    avg_restored_psnr = np.mean([
        r["restored_psnr"]
        for r in results
    ])

    avg_degraded_ssim = np.mean([
        r["degraded_ssim"]
        for r in results
    ])

    avg_restored_ssim = np.mean([
        r["restored_ssim"]
        for r in results
    ])

    print()
    print("=" * 70)
    print("AVERAGE RESULTS")
    print("=" * 70)

    print(
        f"Degraded PSNR : "
        f"{avg_degraded_psnr:.2f} dB"
    )

    print(
        f"Restored PSNR : "
        f"{avg_restored_psnr:.2f} dB"
    )

    print(
        f"PSNR gain     : "
        f"{avg_restored_psnr - avg_degraded_psnr:+.2f} dB"
    )

    print()

    print(
        f"Degraded SSIM : "
        f"{avg_degraded_ssim:.4f}"
    )

    print(
        f"Restored SSIM : "
        f"{avg_restored_ssim:.4f}"
    )

    print(
        f"SSIM gain     : "
        f"{avg_restored_ssim - avg_degraded_ssim:+.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate()