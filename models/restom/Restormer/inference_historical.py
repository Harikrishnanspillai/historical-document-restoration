import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from basicsr.models import create_model
from basicsr.utils.options import parse


# --------------------------------------------------
# Paths
# --------------------------------------------------

OPT_FILE = ROOT / "Denoising" / "Options" / "HistoricalStains_Restormer.yml"

CHECKPOINT = (
    ROOT
    / "experiments"
    / "HistoricalStains_Restormer"
    / "models"
    / "net_g_5000.pth"
)

INPUT_DIR = ROOT.parent.parent.parent / "data" / "stains" / "test" / "degraded"
OUTPUT_DIR = ROOT.parent.parent.parent / "data" / "stains" / "test" / "restored"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Tile settings
# --------------------------------------------------

TILE_SIZE = 256
OVERLAP = 32
STRIDE = TILE_SIZE - OVERLAP


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

opt = parse(str(OPT_FILE), is_train=False)

opt["dist"] = False
opt["rank"] = 0
opt["world_size"] = 1

opt["path"]["pretrain_network_g"] = str(CHECKPOINT)
opt["path"]["strict_load_g"] = True


# --------------------------------------------------
# Build model
# --------------------------------------------------

model = create_model(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Tile size:", TILE_SIZE)
print("Overlap:", OVERLAP)

model.net_g = model.net_g.to(device)
model.net_g.eval()


# --------------------------------------------------
# Tile inference
# --------------------------------------------------

def process_image(image_np):

    original_height, original_width = image_np.shape

    # Pad image to at least TILE_SIZE
    padded_height = max(original_height, TILE_SIZE)
    padded_width = max(original_width, TILE_SIZE)

    # Make dimensions divisible by 8
    padded_height += (8 - padded_height % 8) % 8
    padded_width += (8 - padded_width % 8) % 8

    padded = np.pad(
        image_np,
        (
            (0, padded_height - original_height),
            (0, padded_width - original_width)
        ),
        mode="reflect"
    )

    output = np.zeros_like(padded, dtype=np.float32)
    weight = np.zeros_like(padded, dtype=np.float32)

    # Generate tile start positions
    y_positions = list(range(0, padded_height - TILE_SIZE + 1, STRIDE))
    x_positions = list(range(0, padded_width - TILE_SIZE + 1, STRIDE))

    # Make sure final tile reaches the edge
    if y_positions[-1] != padded_height - TILE_SIZE:
        y_positions.append(padded_height - TILE_SIZE)

    if x_positions[-1] != padded_width - TILE_SIZE:
        x_positions.append(padded_width - TILE_SIZE)

    total_tiles = len(y_positions) * len(x_positions)
    tile_number = 0

    with torch.no_grad():

        for y in y_positions:
            for x in x_positions:

                tile_number += 1

                print(
                    f"    Tile {tile_number}/{total_tiles}",
                    end="\r"
                )

                tile = padded[
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE
                ]

                tensor = (
                    torch.from_numpy(tile)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )

                restored_tile = model.net_g(tensor)

                restored_tile = (
                    restored_tile
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                restored_tile = np.clip(
                    restored_tile,
                    0,
                    1
                )

                output[
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE
                ] += restored_tile

                weight[
                    y:y + TILE_SIZE,
                    x:x + TILE_SIZE
                ] += 1.0

                # Release GPU memory
                del tensor
                del restored_tile

    print()

    # Average overlapping regions
    output /= np.maximum(weight, 1e-8)

    # Remove padding
    output = output[
        :original_height,
        :original_width
    ]

    return output


# --------------------------------------------------
# Run inference
# --------------------------------------------------

input_files = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    ]
)

print(f"Found {len(input_files)} test images.")


for input_file in input_files:

    print(f"\nProcessing: {input_file.name}")

    image = Image.open(input_file).convert("L")

    image_np = (
        np.asarray(
            image,
            dtype=np.float32
        ) / 255.0
    )

    restored = process_image(image_np)

    output_uint8 = (
        restored * 255.0
    ).round().astype(np.uint8)

    output_path = (
        OUTPUT_DIR
        / f"{input_file.stem}_restored.png"
    )

    Image.fromarray(output_uint8).save(output_path)

    print(f"Saved: {output_path}")


print("\nInference completed successfully.")