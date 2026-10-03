import cv2
import numpy as np
import torch

from pathlib import Path
from basicsr.models import create_model
from basicsr.utils.options import parse


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent

OPT_FILE = (
    ROOT
    / "Denoising"
    / "options"
    / "train_HistoricalGeneralized_Restormer.yml"
)

CHECKPOINT = (
    ROOT
    / "experiments"
    / "HistoricalGeneralized_Restormer"
    / "models"
    / "net_g_5000.pth"
)

DATA_ROOT = (
    ROOT.parent.parent.parent
    / "data"
)

CATEGORIES = [
    "noise",
    "blur",
    "fading",
    "stains",
    "bleedthrough",
    "random_degradation"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read: {path}"
        )

    return image.astype(
        np.float32
    ) / 255.0


# =========================================================
# RESTORMER INFERENCE
# =========================================================

def restore_image(model, image):

    height, width = image.shape

    TILE_SIZE = 256
    OVERLAP = 32
    STRIDE = TILE_SIZE - OVERLAP

    # -----------------------------------------------------
    # Pad image if smaller than tile
    # -----------------------------------------------------

    padded_height = max(height, TILE_SIZE)
    padded_width = max(width, TILE_SIZE)

    pad_bottom = padded_height - height
    pad_right = padded_width - width

    if pad_bottom > 0 or pad_right > 0:

        image = np.pad(
            image,
            (
                (0, pad_bottom),
                (0, pad_right)
            ),
            mode="reflect"
        )

    padded_height, padded_width = image.shape

    # -----------------------------------------------------
    # Make dimensions divisible by 8
    # -----------------------------------------------------

    extra_h = (
        8 - padded_height % 8
    ) % 8

    extra_w = (
        8 - padded_width % 8
    ) % 8

    if extra_h > 0 or extra_w > 0:

        image = np.pad(
            image,
            (
                (0, extra_h),
                (0, extra_w)
            ),
            mode="reflect"
        )

    padded_height, padded_width = image.shape

    # -----------------------------------------------------
    # Ensure final tile reaches image boundary
    # -----------------------------------------------------

    y_positions = list(
        range(
            0,
            padded_height - TILE_SIZE + 1,
            STRIDE
        )
    )

    x_positions = list(
        range(
            0,
            padded_width - TILE_SIZE + 1,
            STRIDE
        )
    )

    last_y = padded_height - TILE_SIZE
    last_x = padded_width - TILE_SIZE

    if y_positions[-1] != last_y:
        y_positions.append(last_y)

    if x_positions[-1] != last_x:
        x_positions.append(last_x)

    # -----------------------------------------------------
    # Output accumulation
    # -----------------------------------------------------

    output = np.zeros(
        (padded_height, padded_width),
        dtype=np.float32
    )

    weight = np.zeros(
        (padded_height, padded_width),
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Process tiles
    # -----------------------------------------------------

    for y in y_positions:

        for x in x_positions:

            tile = image[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            tensor = torch.from_numpy(
                tile
            ).float()

            tensor = (
                tensor
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():

                restored_tile = model.net_g(
                    tensor
                )

            restored_tile = (
                restored_tile
                .squeeze()
                .cpu()
                .numpy()
            )

            output[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += restored_tile

            weight[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ] += 1.0

            # Release GPU memory after every tile
            del tensor
            del restored_tile

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Average overlapping tiles
    # -----------------------------------------------------

    output /= np.maximum(
        weight,
        1e-8
    )

    # Remove padding
    output = output[
        :height,
        :width
    ]

    return np.clip(
        output,
        0,
        1
    )


# =========================================================
# START
# =========================================================

print(
    "\n=============================================="
)

print(
    " FINAL RESTORMER TEST RESTORATION"
)

print(
    "=============================================="
)

print(
    "\nDevice:",
    (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
)

print(
    "Checkpoint:",
    CHECKPOINT.name
)


# =========================================================
# LOAD CONFIG
# =========================================================

opt = parse(
    str(OPT_FILE),
    is_train=False
)

opt["dist"] = False


# =========================================================
# CREATE MODEL
# =========================================================

model = create_model(opt)

model.net_g.to(DEVICE)
model.net_g.eval()


# =========================================================
# LOAD CHECKPOINT
# =========================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "params" in checkpoint:

    state_dict = checkpoint["params"]

elif "state_dict" in checkpoint:

    state_dict = checkpoint["state_dict"]

else:

    state_dict = checkpoint


model.net_g.load_state_dict(
    state_dict,
    strict=True
)

model.net_g.eval()

print(
    "\nCheckpoint loaded successfully."
)


# =========================================================
# RESTORE TEST IMAGES
# =========================================================

total_images = 0


for category in CATEGORIES:

    input_dir = (
        DATA_ROOT
        / category
        / "test"
        / "degraded"
    )

    output_dir = (
        DATA_ROOT
        / category
        / "test"
        / "restored_restormer"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted(
        list(input_dir.glob("*.tif"))
        +
        list(input_dir.glob("*.tiff"))
        +
        list(input_dir.glob("*.png"))
    )

    print(
        f"\n{category}: {len(image_files)} images"
    )

    for image_path in image_files:

        image = load_image(
            image_path
        )

        restored = restore_image(
            model,
            image
        )

        # Convert back to 8-bit
        restored_uint8 = (
            restored * 255.0
        ).round().astype(
            np.uint8
        )

        output_path = (
            output_dir
            / f"{image_path.stem}_restored.png"
        )

        cv2.imwrite(
            str(output_path),
            restored_uint8
        )

        total_images += 1

        print(
            f"  Restored: {image_path.name}"
        )


# =========================================================
# DONE
# =========================================================

print(
    "\n=============================================="
)

print(
    " RESTORATION COMPLETE"
)

print(
    "=============================================="
)

print(
    "Total images restored:",
    total_images
)

print(
    "\nOutput folders:"
)

for category in CATEGORIES:

    print(
        f"  data/{category}/test/restored_restormer/"
    )

print(
    "\n=============================================="
)