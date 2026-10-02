import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "dncnn"

sys.path.insert(0, str(MODEL_DIR))
from model import DnCNN
sys.path.insert(0, str(MODEL_DIR / "training"))
from dataset import CATEGORIES, read_image


DATA_ROOT = ROOT / "data"
CHECKPOINT = MODEL_DIR / "checkpoints" / "dncnn_best.pth"
RESTORE_DIR = MODEL_DIR / "restored"

PATCH_SIZE = 128
OVERLAP = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(checkpoint):
    model = DnCNN(
        in_channels=checkpoint.get("in_channels", 1),
        out_channels=1,
        num_layers=checkpoint.get("num_layers", 20),
        features=checkpoint.get("features", 64)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model


def get_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))

    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)

    return positions


@torch.no_grad()
def restore_image(model, image):
    height, width = image.shape

    pad_h = max(0, PATCH_SIZE - height)
    pad_w = max(0, PATCH_SIZE - width)

    if pad_h or pad_w:
        mode = "reflect" if height > pad_h and width > pad_w else "edge"
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode=mode)

    height, width = image.shape
    stride = PATCH_SIZE - OVERLAP

    y_positions = get_positions(height, PATCH_SIZE, stride)
    x_positions = get_positions(width, PATCH_SIZE, stride)

    output = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    window_1d = np.hanning(PATCH_SIZE).astype(np.float32)
    window_1d[window_1d == 0] = 1e-3
    window = np.outer(window_1d, window_1d)

    for top in y_positions:
        for left in x_positions:
            patch = image[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
            tensor = torch.from_numpy(patch.copy()).float()
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

            restored_patch = model(tensor).squeeze().cpu().numpy()

            output[top:top + PATCH_SIZE, left:left + PATCH_SIZE] += restored_patch * window
            weights[top:top + PATCH_SIZE, left:left + PATCH_SIZE] += window

    output = output / np.maximum(weights, 1e-8)
    output = output[:height - pad_h, :width - pad_w]

    return np.clip(output, 0.0, 1.0)


def save_image(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = create_model(checkpoint)

    print("Loaded checkpoint:", CHECKPOINT)
    print("Device:", DEVICE)

    total_images = 0

    for category in CATEGORIES:
        source_dir = DATA_ROOT / category / "test" / "degraded"
        output_dir = RESTORE_DIR / category

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing test directory: {source_dir}")

        image_paths = sorted(
            path for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png"}
        )

        if not image_paths:
            raise RuntimeError(f"No test images found in {source_dir}")

        print(f"\n{category}: {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc=f"Restoring {category}"):
            image = read_image(image_path)
            restored = restore_image(model, image)

            save_image(restored, output_dir / f"{image_path.stem}.png")
            total_images += 1

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nRestored {total_images} images.")
    print("Outputs saved in:", RESTORE_DIR)


if __name__ == "__main__":
    main()
