from pathlib import Path

import json
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEGRADATION = "stains"

ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    ROOT.parent.parent.parent
    / "data"
    / DEGRADATION
    / "test"
    / "degraded"
)

RESTORED_DIR = (
    ROOT.parent.parent.parent
    / "data"
    / DEGRADATION
    / "test"
    / "restored"
)

CLEAN_DIR = (
    ROOT.parent.parent.parent
    / "data"
    / DEGRADATION
    / "test"
    / "clean"
)

RESULTS_FILE = (
    ROOT
    / f"evaluation_restormer_{DEGRADATION}.json"
)


# --------------------------------------------------
# Image loading
# --------------------------------------------------

def load_image(path):

    image = Image.open(path).convert("L")

    return (
        np.asarray(
            image,
            dtype=np.float32
        ) / 255.0
    )


# --------------------------------------------------
# Find test images
# --------------------------------------------------

input_files = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in [".tif", ".tiff"]
)

print(f"Found {len(input_files)} test images.")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

results = {}

total_mse = 0.0
total_psnr = 0.0
total_ssim = 0.0


for input_file in input_files:

    restored_file = (
        RESTORED_DIR
        / f"{input_file.stem}_restored.png"
    )

    clean_file = (
        CLEAN_DIR
        / input_file.name
    )

    print(f"\nEvaluating: {input_file.name}")

    if not restored_file.exists():
        print(f"WARNING: Restored image missing: {restored_file}")
        continue

    if not clean_file.exists():
        print(f"WARNING: Clean image missing: {clean_file}")
        continue

    restored = load_image(restored_file)
    clean = load_image(clean_file)

    if restored.shape != clean.shape:
        raise ValueError(
            f"Shape mismatch for {input_file.name}: "
            f"restored={restored.shape}, "
            f"clean={clean.shape}"
        )

    # MSE
    mse = np.mean(
        (clean - restored) ** 2
    )

    # PSNR
    psnr = peak_signal_noise_ratio(
        clean,
        restored,
        data_range=1.0
    )

    # SSIM
    ssim = structural_similarity(
        clean,
        restored,
        data_range=1.0
    )

    results[input_file.name] = {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim)
    }

    total_mse += mse
    total_psnr += psnr
    total_ssim += ssim

    print(f"MSE  : {mse:.6f}")
    print(f"PSNR : {psnr:.4f} dB")
    print(f"SSIM : {ssim:.4f}")


# --------------------------------------------------
# Average metrics
# --------------------------------------------------

num_images = len(results)

if num_images == 0:
    raise RuntimeError(
        "No images were successfully evaluated."
    )

average = {
    "mse": float(total_mse / num_images),
    "psnr": float(total_psnr / num_images),
    "ssim": float(total_ssim / num_images)
}


# --------------------------------------------------
# Save results
# --------------------------------------------------

evaluation_results = {
    "model": "Restormer",
    "degradation": DEGRADATION,
    "num_test_images": num_images,
    "per_image": results,
    "average": average
}

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_results,
        f,
        indent=4
    )


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n" + "=" * 50)
print("RESTORMER EVALUATION RESULTS")
print("=" * 50)

print(f"Degradation : {DEGRADATION}")
print(f"Images      : {num_images}")

print("\nAverage Metrics:")

print(
    f"MSE  : {average['mse']:.6f}"
)

print(
    f"PSNR : {average['psnr']:.4f} dB"
)

print(
    f"SSIM : {average['ssim']:.4f}"
)

print("\nResults saved to:")
print(RESULTS_FILE)