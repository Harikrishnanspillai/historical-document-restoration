import numpy as np
from pathlib import Path
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# -----------------------------
# Settings
# -----------------------------

INPUT_DIR = "../../data/test/recto"
CLEAN_DIR = "../../data/test/clean"


# -----------------------------
# Get test images
# -----------------------------

input_files = sorted(
    Path(INPUT_DIR).glob("*.tif")
)


# -----------------------------
# Evaluate degraded images
# -----------------------------

total_psnr = 0.0
total_ssim = 0.0

print("Test images:", len(input_files))
print()

for input_file in input_files:

    clean_file = (
        Path(CLEAN_DIR)
        / input_file.name
    )

    degraded = np.asarray(
        Image.open(input_file).convert("L"),
        dtype=np.float32
    ) / 255.0

    clean = np.asarray(
        Image.open(clean_file).convert("L"),
        dtype=np.float32
    ) / 255.0

    psnr = peak_signal_noise_ratio(
        clean,
        degraded,
        data_range=1.0
    )

    ssim = structural_similarity(
        clean,
        degraded,
        data_range=1.0
    )

    total_psnr += psnr
    total_ssim += ssim

    print(
        f"{input_file.name}: "
        f"PSNR = {psnr:.4f} dB, "
        f"SSIM = {ssim:.4f}"
    )


# -----------------------------
# Average metrics
# -----------------------------

average_psnr = total_psnr / len(input_files)
average_ssim = total_ssim / len(input_files)

print()
print("--------------------------------")
print(f"Average PSNR: {average_psnr:.4f} dB")
print(f"Average SSIM: {average_ssim:.4f}")
print("--------------------------------")