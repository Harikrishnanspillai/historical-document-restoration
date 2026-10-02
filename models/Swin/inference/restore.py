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

CATEGORIES = ("noise", "blur", "fading", "stains", "bleedthrough", "random_degradation")
PATCH_SIZE, OVERLAP = 128, 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_PATH = SWIN_ROOT / "checkpoints" / "swinir_best.pth"
OUTPUT_ROOT = SWIN_ROOT / "outputs" / "restored"


def create_model():
    return SwinIR(
        upscale=1, in_chans=1, img_size=PATCH_SIZE, window_size=8,
        img_range=1.0, depths=[6,6,6,6,6,6], embed_dim=180,
        num_heads=[6,6,6,6,6,6], mlp_ratio=2,
        upsampler="", resi_connection="1conv"
    )


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')} on {DEVICE}")
    return model


def load_image(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)


def pad_image(image):
    _, _, h, w = image.shape
    ph, pw = (PATCH_SIZE-h % PATCH_SIZE) % PATCH_SIZE, (PATCH_SIZE-w % PATCH_SIZE) % PATCH_SIZE
    if ph or pw:
        mode = "reflect" if h > ph and w > pw else "replicate"
        image = F.pad(image, (0, pw, 0, ph), mode=mode)
    return image, h, w


@torch.no_grad()
def restore_image(model, image):
    image, original_h, original_w = pad_image(image)
    _, _, h, w = image.shape
    stride = PATCH_SIZE - OVERLAP
    ys = list(range(0, max(1, h-PATCH_SIZE+1), stride))
    xs = list(range(0, max(1, w-PATCH_SIZE+1), stride))
    if ys[-1] != h-PATCH_SIZE:
        ys.append(h-PATCH_SIZE)
    if xs[-1] != w-PATCH_SIZE:
        xs.append(w-PATCH_SIZE)

    output, weights = torch.zeros_like(image), torch.zeros_like(image)
    for y in ys:
        for x in xs:
            tile = image[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE].to(DEVICE)
            restored = model(tile).float().cpu()
            output[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += restored
            weights[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
    output = output / weights.clamp_min(1)
    return output[:, :, :original_h, :original_w]


def save_image(tensor, path):
    arr = tensor.squeeze().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def main():
    model = load_model()
    total = 0
    for category in CATEGORIES:
        source_dir = DATA_ROOT / category / "test" / "degraded"
        target_dir = OUTPUT_ROOT / category
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing test directory: {source_dir}")
        paths = sorted(p for p in source_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
        if not paths:
            raise RuntimeError(f"No TIFF test images in {source_dir}")
        print(f"{category}: {len(paths)} images")
        for path in tqdm(paths, desc=f"Restoring {category}"):
            restored = restore_image(model, load_image(path))
            # Retain source filename; category-specific output folders avoid collisions.
            save_image(restored, target_dir / path.name)
            total += 1
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Restored {total} images. Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
