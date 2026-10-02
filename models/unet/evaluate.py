import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# =========================
# Configuration
# =========================

DEGRADATION = "bleedthrough"

INPUT_DIR = Path(f"../../data/{DEGRADATION}/test/degraded")
RESTORED_DIR = Path(f"../../data/{DEGRADATION}/test/restored")
CLEAN_DIR = Path(f"../../data/{DEGRADATION}/test/clean")

RESULTS_FILE = Path(f"evaluation_{DEGRADATION}.json")


# =========================
# Helper function
# =========================

def load_image(path):
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


# =========================
# Evaluation
# =========================

input_files = sorted(INPUT_DIR.glob("*.tif"))

print(f"Test images: {len(input_files)}")

if len(input_files) == 0:
    raise FileNotFoundError(f"No test images found in {INPUT_DIR}")


results = {}

total_mse = 0.0
total_psnr = 0.0
total_ssim = 0.0


for input_file in input_files:

    filename = input_file.name

    restored_file = RESTORED_DIR / f"{input_file.stem}_restored.png"
    clean_file = CLEAN_DIR / filename

    if not restored_file.exists():
        print(f"Skipping {filename}: restored image not found")
        continue

    if not clean_file.exists():
        print(f"Skipping {filename}: clean image not found")
        continue

    # Load images
    restored = load_image(restored_file)
    clean = load_image(clean_file)

    # Check dimensions
    if restored.shape != clean.shape:
        raise ValueError(
            f"Shape mismatch for {filename}: "
            f"restored={restored.shape}, clean={clean.shape}"
        )

    # Calculate metrics
    mse = np.mean((clean - restored) ** 2)

    psnr = peak_signal_noise_ratio(
        clean,
        restored,
        data_range=1.0
    )

    ssim = structural_similarity(
        clean,
        restored,
        data_range=1.0
    )

    # Store per-image results
    results[filename] = {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim)
    }

    # Accumulate
    total_mse += mse
    total_psnr += psnr
    total_ssim += ssim

    print(
        f"{filename}: "
        f"MSE = {mse:.6f}, "
        f"PSNR = {psnr:.4f} dB, "
        f"SSIM = {ssim:.4f}"
    )


# =========================
# Average metrics
# =========================

num_evaluated = len(results)

if num_evaluated == 0:
    raise RuntimeError("No images were successfully evaluated.")

average_mse = total_mse / num_evaluated
average_psnr = total_psnr / num_evaluated
average_ssim = total_ssim / num_evaluated


print("\n--------------------------------")
print(f"Average MSE:  {average_mse:.6f}")
print(f"Average PSNR: {average_psnr:.4f} dB")
print(f"Average SSIM: {average_ssim:.4f}")
print("--------------------------------")


# =========================
# Save results
# =========================

evaluation_results = {
    "degradation": DEGRADATION,
    "num_test_images": num_evaluated,

    "per_image": results,

    "average": {
        "mse": float(average_mse),
        "psnr": float(average_psnr),
        "ssim": float(average_ssim)
    }
}


with open(RESULTS_FILE, "w") as f:
    json.dump(evaluation_results, f, indent=4)


print(f"\nResults saved to: {RESULTS_FILE}")