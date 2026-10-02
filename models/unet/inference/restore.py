import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "unet"

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / "training"))

from unet import UNet
from dataset import CATEGORIES, read_image


CHECKPOINT_PATH = MODEL_DIR / "checkpoints" / "unet_best.pth"
DATA_ROOT = ROOT / "data"
RESTORED_ROOT = MODEL_DIR / "restored"

TILE_SIZE = 256
OVERLAP = 32


def restore_image(model, image, device):
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    step = TILE_SIZE - OVERLAP

    with torch.no_grad():
        for top in range(0, height, step):
            for left in range(0, width, step):
                bottom = min(top + TILE_SIZE, height)
                right = min(left + TILE_SIZE, width)

                tile = image[top:bottom, left:right]
                tile_h, tile_w = tile.shape

                padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                padded[:tile_h, :tile_w] = tile

                tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
                tensor = tensor.to(device)

                prediction = model(tensor)
                prediction = prediction[0, 0].cpu().numpy()
                prediction = prediction[:tile_h, :tile_w]

                output[top:bottom, left:right] += prediction
                weight[top:bottom, left:right] += 1.0

    output /= np.maximum(weight, 1.0)
    return np.clip(output, 0.0, 1.0)


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. Train the model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    for category in CATEGORIES:
        degraded_dir = DATA_ROOT / category / "test" / "degraded"
        output_dir = RESTORED_ROOT / category
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path for path in degraded_dir.iterdir()
            if path.suffix.lower() in {".tif", ".tiff", ".png"}
        )

        print(f"\n{category.upper()}: {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc=category):
            image = read_image(image_path)
            restored = restore_image(model, image, device)

            output_path = output_dir / f"{image_path.stem}.png"
            Image.fromarray(
                (restored * 255.0).round().astype(np.uint8)
            ).save(output_path)

    print("\nAll test categories restored.")


if __name__ == "__main__":
    main()
