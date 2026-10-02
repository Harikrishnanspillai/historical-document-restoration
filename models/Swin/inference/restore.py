import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


SWIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SWINIR_DIR = SWIN_ROOT / "SwinIR"

sys.path.insert(0, str(SWINIR_DIR))
from models.network_swinir import SwinIR


CATEGORIES = (
    "noise",
    "blur",
    "fading",
    "stains",
    "bleedthrough",
    "random_degradation"
)

PATCH_SIZE = 128
OVERLAP = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_PATH = SWIN_ROOT / "checkpoints" / "swinir_best.pth"
OUTPUT_ROOT = SWIN_ROOT / "outputs" / "restored"


def create_model():
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


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')}")
    print("Device:", DEVICE)

    return model


def load_image(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array.copy())
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


def pad_image(image):
    _, _, height, width = image.shape

    pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE

    if pad_h or pad_w:
        if height > pad_h and width > pad_w:
            mode = "reflect"
        else:
            mode = "replicate"

        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)

    return image, height, width


@torch.no_grad()
def restore_image(model, image):
    image, original_h, original_w = pad_image(image)

    _, _, height, width = image.shape
    stride = PATCH_SIZE - OVERLAP

    y_positions = list(range(0, max(1, height - PATCH_SIZE + 1), stride))
    x_positions = list(range(0, max(1, width - PATCH_SIZE + 1), stride))

    if y_positions[-1] != height - PATCH_SIZE:
        y_positions.append(height - PATCH_SIZE)

    if x_positions[-1] != width - PATCH_SIZE:
        x_positions.append(width - PATCH_SIZE)

    output = torch.zeros_like(image)
    weights = torch.zeros_like(image)

    for y in y_positions:
        for x in x_positions:
            patch = image[:, :, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch = patch.to(DEVICE)

            restored_patch = model(patch).float().cpu()

            output[:, :, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += restored_patch
            weights[:, :, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1

    output = output / weights.clamp_min(1)

    return output[:, :, :original_h, :original_w]


def save_image(tensor, path):
    array = tensor.squeeze().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def main():
    model = load_model()
    total_images = 0

    for category in CATEGORIES:
        source_dir = DATA_ROOT / category / "test" / "degraded"
        output_dir = OUTPUT_ROOT / category

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing test directory: {source_dir}")

        image_paths = sorted(
            path for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        )

        if not image_paths:
            raise RuntimeError(f"No TIFF test images found in {source_dir}")

        print(f"\n{category}: {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc=f"Restoring {category}"):
            image = load_image(image_path)
            restored = restore_image(model, image)

            save_image(restored, output_dir / image_path.name)
            total_images += 1

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nRestored {total_images} images.")
    print("Outputs saved in:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
