import torch
from PIL import Image
import numpy as np
from pathlib import Path

from unet import UNet


# -----------------------------
# Settings
# -----------------------------

INPUT_DIR = "../../data/test/recto"
OUTPUT_DIR = "../../data/test/restored"

MODEL_PATH = "unet_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load model
# -----------------------------

model = UNet(
    in_channels=1,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# -----------------------------
# Prepare directories
# -----------------------------

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

input_files = sorted(
    Path(INPUT_DIR).glob("*.tif")
)

print("Device:", DEVICE)
print("Test images:", len(input_files))
print()


# -----------------------------
# Process each image
# -----------------------------

tile_size = 256
overlap = 32

for input_file in input_files:

    print("Processing:", input_file.name)

    # -------------------------
    # Load image
    # -------------------------

    image = Image.open(input_file).convert("L")

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    height, width = image_array.shape

    # -------------------------
    # Prepare output arrays
    # -------------------------

    output = np.zeros_like(image_array)
    weight = np.zeros_like(image_array)

    # -------------------------
    # Tiled inference
    # -------------------------

    with torch.no_grad():

        for top in range(
            0,
            height,
            tile_size - overlap
        ):

            for left in range(
                0,
                width,
                tile_size - overlap
            ):

                bottom = min(
                    top + tile_size,
                    height
                )

                right = min(
                    left + tile_size,
                    width
                )

                tile = image_array[
                    top:bottom,
                    left:right
                ]

                tile_h, tile_w = tile.shape

                # Pad edge tiles
                padded = np.zeros(
                    (tile_size, tile_size),
                    dtype=np.float32
                )

                padded[
                    :tile_h,
                    :tile_w
                ] = tile

                # Convert to tensor
                tensor = torch.from_numpy(
                    padded
                ).unsqueeze(0).unsqueeze(0).to(DEVICE)

                # Run U-Net
                prediction = model(tensor)

                prediction = (
                    prediction
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                # Remove padding
                prediction = prediction[
                    :tile_h,
                    :tile_w
                ]

                # Add prediction
                output[
                    top:bottom,
                    left:right
                ] += prediction

                weight[
                    top:bottom,
                    left:right
                ] += 1

    # -------------------------
    # Average overlapping tiles
    # -------------------------

    output /= np.maximum(
        weight,
        1
    )

    # -------------------------
    # Convert to image
    # -------------------------

    output = np.clip(
        output,
        0,
        1
    )

    output_image = Image.fromarray(
        (output * 255).astype(np.uint8)
    )

    # -------------------------
    # Save output
    # -------------------------

    output_path = (
        Path(OUTPUT_DIR)
        / f"{input_file.stem}_restored.png"
    )

    output_image.save(output_path)

    print(
        "Saved:",
        output_path
    )

    print()


print("All test images processed successfully.")