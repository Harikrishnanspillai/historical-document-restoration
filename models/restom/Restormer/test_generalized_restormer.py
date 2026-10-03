import os
import csv
import torch
import cv2
import numpy as np

from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from basicsr.models import create_model
from basicsr.utils.options import parse


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

OPT_FILE = ROOT / "Denoising" / "options" / "train_HistoricalGeneralized_Restormer.yml"

CHECKPOINT = (
    ROOT
    / "experiments"
    / "HistoricalGeneralized_Restormer"
    / "models"
    / "net_g_5000.pth"
)

DATA_ROOT = ROOT.parent.parent.parent / "data"

CATEGORIES = [
    "noise",
    "blur",
    "fading",
    "stains",
    "bleedthrough",
    "random_degradation"
]

TILE_SIZE = 256
OVERLAP = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# IMAGE UTILITIES
# =========================================================

def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    image = image.astype(np.float32) / 255.0

    return image


def save_image(path, image):
    image = np.clip(image, 0, 1)

    image = (image * 255.0).round().astype(np.uint8)

    cv2.imwrite(str(path), image)


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(restored, target):

    restored = np.clip(restored, 0, 1)
    target = np.clip(target, 0, 1)

    mse = np.mean((restored - target) ** 2)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(1.0 / mse)

    ssim_value = ssim(
        target,
        restored,
        data_range=1.0
    )

    return mse, psnr, ssim_value


# =========================================================
# TILED RESTORMER INFERENCE
# =========================================================

def restore_image(model, image):

    height, width = image.shape

    pad_h = max(0, TILE_SIZE - height)
    pad_w = max(0, TILE_SIZE - width)

    if pad_h > 0 or pad_w > 0:

        image = np.pad(
            image,
            (
                (0, pad_h),
                (0, pad_w)
            ),
            mode="reflect"
        )

    padded_h, padded_w = image.shape

    stride = TILE_SIZE - OVERLAP

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    weight = np.zeros_like(
        image,
        dtype=np.float32
    )

    for y in range(0, padded_h - TILE_SIZE + 1, stride):

        for x in range(0, padded_w - TILE_SIZE + 1, stride):

            tile = image[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            tensor = torch.from_numpy(tile).float()
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(DEVICE)

            with torch.no_grad():

                restored = model.net_g(tensor)

            restored = restored.squeeze().cpu().numpy()

            output[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += restored

            weight[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += 1.0

    output /= np.maximum(weight, 1e-8)

    output = output[:height, :width]

    return np.clip(output, 0, 1)


# =========================================================
# LOAD RESTORMER
# =========================================================

print("\n==============================================")
print(" GENERALIZED RESTORMER EVALUATION")
print("==============================================")

print("\nCheckpoint:")
print(CHECKPOINT)

print("\nDevice:")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CPU")


opt = parse(
    str(OPT_FILE),
    is_train=False
)

opt["dist"] = False

model = create_model(opt)

print("\nLoading trained checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "params" in checkpoint:
    model.net_g.load_state_dict(
        checkpoint["params"],
        strict=True
    )
elif "state_dict" in checkpoint:
    model.net_g.load_state_dict(
        checkpoint["state_dict"],
        strict=True
    )
else:
    model.net_g.load_state_dict(
        checkpoint,
        strict=True
    )

model.net_g.to(DEVICE)
model.net_g.eval()

print("Checkpoint loaded successfully.")


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

all_results = []

results_root = ROOT / "results_generalized"

results_root.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# PROCESS ALL SIX CATEGORIES
# =========================================================

for category in CATEGORIES:

    print("\n----------------------------------------------")
    print(f"Processing category: {category}")
    print("----------------------------------------------")

    degraded_dir = (
        DATA_ROOT
        / category
        / "test"
        / "degraded"
    )

    clean_dir = (
        DATA_ROOT
        / category
        / "test"
        / "clean"
    )

    restored_dir = (
        DATA_ROOT
        / category
        / "test"
        / "restored_restormer"
    )

    restored_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted(
        list(degraded_dir.glob("*.tif"))
        + list(degraded_dir.glob("*.tiff"))
        + list(degraded_dir.glob("*.png"))
    )

    print(f"Images found: {len(image_files)}")

    category_metrics = []

    for image_path in image_files:

        clean_path = (
            clean_dir
            / image_path.name
        )

        if not clean_path.exists():

            print(
                f"Skipping {image_path.name}: "
                "ground truth not found"
            )

            continue

        print(
            f"  Restoring: {image_path.name}"
        )

        degraded = load_image(
            image_path
        )

        clean = load_image(
            clean_path
        )

        restored = restore_image(
            model,
            degraded
        )

        output_name = (
            image_path.stem
            + "_restored.png"
        )

        output_path = (
            restored_dir
            / output_name
        )

        save_image(
            output_path,
            restored
        )

        mse, psnr, ssim_value = calculate_metrics(
            restored,
            clean
        )

        result = {
            "category": category,
            "image": image_path.name,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim_value
        }

        all_results.append(result)
        category_metrics.append(result)

        print(
            f"    MSE:  {mse:.6f}"
        )

        print(
            f"    PSNR: {psnr:.4f} dB"
        )

        print(
            f"    SSIM: {ssim_value:.4f}"
        )


# =========================================================
# CATEGORY AVERAGES
# =========================================================

category_results = []

for category in CATEGORIES:

    rows = [
        r for r in all_results
        if r["category"] == category
    ]

    if not rows:
        continue

    category_results.append({
        "category": category,
        "image": "AVERAGE",
        "mse": np.mean(
            [r["mse"] for r in rows]
        ),
        "psnr": np.mean(
            [r["psnr"] for r in rows]
        ),
        "ssim": np.mean(
            [r["ssim"] for r in rows]
        )
    })


# =========================================================
# OVERALL AVERAGE
# =========================================================

overall = {
    "category": "OVERALL",
    "image": "AVERAGE",
    "mse": np.mean(
        [r["mse"] for r in all_results]
    ),
    "psnr": np.mean(
        [r["psnr"] for r in all_results]
    ),
    "ssim": np.mean(
        [r["ssim"] for r in all_results]
    )
}


# =========================================================
# SAVE CSV
# =========================================================

csv_path = (
    results_root
    / "restormer_generalized_metrics.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "category",
        "image",
        "mse",
        "psnr",
        "ssim"
    ])

    # Individual image results

    for result in all_results:

        writer.writerow([
            result["category"],
            result["image"],
            f"{result['mse']:.8f}",
            f"{result['psnr']:.6f}",
            f"{result['ssim']:.6f}"
        ])

    # Category averages

    for result in category_results:

        writer.writerow([
            result["category"],
            "AVERAGE",
            f"{result['mse']:.8f}",
            f"{result['psnr']:.6f}",
            f"{result['ssim']:.6f}"
        ])

    # Overall average

    writer.writerow([
        overall["category"],
        overall["image"],
        f"{overall['mse']:.8f}",
        f"{overall['psnr']:.6f}",
        f"{overall['ssim']:.6f}"
    ])


# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n==============================================")
print(" GENERALIZED RESTORMER RESULTS")
print("==============================================")

for result in category_results:

    print(
        f"\n{result['category']}:"
    )

    print(
        f"  MSE  : {result['mse']:.6f}"
    )

    print(
        f"  PSNR : {result['psnr']:.4f} dB"
    )

    print(
        f"  SSIM : {result['ssim']:.4f}"
    )


print("\n----------------------------------------------")
print("OVERALL")
print("----------------------------------------------")

print(
    f"MSE  : {overall['mse']:.6f}"
)

print(
    f"PSNR : {overall['psnr']:.4f} dB"
)

print(
    f"SSIM : {overall['ssim']:.4f}"
)

print("\nCSV saved to:")

print(csv_path)

print("\n==============================================")
print(" EVALUATION COMPLETE")
print("==============================================")