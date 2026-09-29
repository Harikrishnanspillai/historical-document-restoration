import numpy as np
from pathlib import Path
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# -----------------------------
# Settings
# -----------------------------

INPUT_DIR = "../../data/test/recto"
RESTORED_DIR = "../../data/test/restored"
CLEAN_DIR = "../../data/test/clean"


# -----------------------------
# Get test images
# -----------------------------

input_files = sorted(
    Path(INPUT_DIR).glob("*.tif")
)


# -----------------------------
# Evaluate
# -----------------------------

total_psnr = 0.0
total_ssim = 0.0

print("Test images:", len(input_files))
print()

for input_file in input_files:

    restored_file = (
        Path(RESTORED_DIR)
        / f"{input_file.stem}_restored.png"
    )

    clean_file = (
        Path(CLEAN_DIR)
        / input_file.name
    )

    # Load images
    restored = np.asarray(
        Image.open(restored_file).convert("L"),
        dtype=np.float32
    ) / 255.0

    clean = np.asarray(
        Image.open(clean_file).convert("L"),
        dtype=np.float32
    ) / 255.0

    # Check dimensions
    if restored.shape != clean.shape:
        raise ValueError(
            f"Size mismatch for {input_file.name}: "
            f"restored={restored.shape}, "
            f"clean={clean.shape}"
        )

    # Calculate metrics
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