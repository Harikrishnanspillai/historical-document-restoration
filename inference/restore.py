import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "dataset" / "test" / "degraded"
CHECKPOINT_PATH = ROOT / "checkpoints" / "swinir_best.pth"
OUTPUT_DIR = ROOT / "outputs" / "restored"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PATCH_SIZE = 128
OVERLAP = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Load SwinIR
# ------------------------------------------------------------

def create_model():
    """
    Create the same SwinIR architecture used during training.
    """

    sys.path.insert(0, str(ROOT))

    from SwinIR.models.network_swinir import SwinIR

    model = SwinIR(
        upscale=1,
        in_chans=1,
        img_size=PATCH_SIZE,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv"
    )

    return model


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

def load_model():
    print("Creating SwinIR model...")

    model = create_model()

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return model


# ------------------------------------------------------------
# Image loading
# ------------------------------------------------------------

def load_image(path):
    """
    Load grayscale TIFF and convert it to
    a normalized PyTorch tensor.
    """

    image = Image.open(path).convert("L")

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(image)

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


# ------------------------------------------------------------
# Padding
# ------------------------------------------------------------

def pad_image(image):
    """
    Pad image so that it can be processed using
    PATCH_SIZE tiles.
    """

    _, _, h, w = image.shape

    pad_h = (
        PATCH_SIZE - h % PATCH_SIZE
    ) % PATCH_SIZE

    pad_w = (
        PATCH_SIZE - w % PATCH_SIZE
    ) % PATCH_SIZE

    if pad_h > 0 or pad_w > 0:
        image = torch.nn.functional.pad(
            image,
            (0, pad_w, 0, pad_h),
            mode="reflect"
        )

    return image, h, w


# ------------------------------------------------------------
# Tiled inference
# ------------------------------------------------------------

@torch.no_grad()
def restore_image(model, image):
    """
    Restore a full-resolution image using overlapping tiles.
    """

    image, original_h, original_w = pad_image(image)

    _, _, h, w = image.shape

    stride = PATCH_SIZE - OVERLAP

    output = torch.zeros_like(image)

    weight = torch.zeros_like(image)

    positions_y = list(
        range(0, max(1, h - PATCH_SIZE + 1), stride)
    )

    positions_x = list(
        range(0, max(1, w - PATCH_SIZE + 1), stride)
    )

    # Make sure the final tile reaches the image boundary
    if positions_y[-1] != h - PATCH_SIZE:
        positions_y.append(h - PATCH_SIZE)

    if positions_x[-1] != w - PATCH_SIZE:
        positions_x.append(w - PATCH_SIZE)

    total_tiles = len(positions_y) * len(positions_x)

    tile_number = 0

    for y in positions_y:

        for x in positions_x:

            tile = image[
                :,
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            tile = tile.to(DEVICE)

            restored = model(tile)

            output[
                :,
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += restored.cpu()

            weight[
                :,
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += 1.0

            tile_number += 1

    output /= weight

    # Remove padding
    output = output[
        :,
        :,
        :original_h,
        :original_w
    ]

    return output


# ------------------------------------------------------------
# Save image
# ------------------------------------------------------------

def save_image(tensor, path):

    image = tensor.squeeze().numpy()

    image = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(image).save(path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("SwinIR Historical Document Restoration")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found:\n{TEST_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test_images = sorted(
        [
            p for p in TEST_DIR.iterdir()
            if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        ]
    )

    print(f"Test images: {len(test_images)}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Overlap: {OVERLAP}px")
    print(f"Output: {OUTPUT_DIR}")
    print()

    model = load_model()

    print()
    print("Starting restoration...")
    print()

    for image_path in tqdm(
        test_images,
        desc="Restoring"
    ):

        try:

            image = load_image(image_path)

            restored = restore_image(
                model,
                image
            )

            output_path = (
                OUTPUT_DIR /
                f"{image_path.stem}_restored.tif"
            )

            save_image(
                restored,
                output_path
            )

        except Exception as e:

            print(
                f"\nERROR processing {image_path.name}: {e}"
            )

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("RESTORATION COMPLETE")
    print("=" * 70)
    print(f"Restored images: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()