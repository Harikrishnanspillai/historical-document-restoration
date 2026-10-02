from pathlib import Path
import sys

import torch
from PIL import Image
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parents[3]
NAFNET = ROOT / "models" / "nafnet" / "NAFNet"
sys.path.append(str(NAFNET))
sys.path.append(str(ROOT / "models" / "nafnet" / "training"))

from basicsr.models.archs.NAFNet_arch import NAFNet
from dataset import read_image


DATA_ROOT = ROOT / "data"
RESTORE_DIR = ROOT / "models" / "nafnet" / "restored"
CHECKPOINT = ROOT / "models" / "nafnet" / "checkpoints" / "nafnet_best.pth"

CATEGORIES = [
    "bleedthrough",
    "blur",
    "fading",
    "noise",
    "random_degradation",
    "stains"
]

PATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()


def restore_image(model, image_path):
    image = read_image(image_path)
    image = to_tensor(image).unsqueeze(0).to(DEVICE)

    _, _, h, w = image.shape

    # Pad image dimensions to multiples of 128.
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE

    image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
    restored = torch.zeros_like(image)

    with torch.no_grad():
        for top in range(0, image.shape[-2], PATCH_SIZE):
            for left in range(0, image.shape[-1], PATCH_SIZE):
                patch = image[:, :, top:top + PATCH_SIZE, left:left + PATCH_SIZE]
                ph, pw = patch.shape[-2:]

                # Pad boundary tiles to exactly 128x128.
                if ph < PATCH_SIZE or pw < PATCH_SIZE:
                    patch = torch.nn.functional.pad(
                        patch,
                        (0, PATCH_SIZE - pw, 0, PATCH_SIZE - ph)
                    )

                # NAFNet returns one restored tensor.
                restored_patch = model(patch)

                restored[
                    :, :, top:top + ph, left:left + pw
                ] = restored_patch[:, :, :ph, :pw]

    # Remove padding and retain original dimensions.
    restored = restored[:, :, :h, :w]
    restored = restored.squeeze(0).cpu().clamp(0, 1)

    return transforms.ToPILImage()(restored)


def main():
    model = NAFNet(
        img_channel=1,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1]
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    print("Device:", DEVICE)

    for category in CATEGORIES:
        input_dir = DATA_ROOT / category / "test" / "degraded"
        output_dir = RESTORE_DIR / category
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            list(input_dir.glob("*.tif")) +
            list(input_dir.glob("*.tiff")) +
            list(input_dir.glob("*.png"))
        )

        print(f"\nRestoring {category}: {len(image_paths)} images")

        for image_path in image_paths:
            with torch.no_grad():
                restored = restore_image(model, image_path)

            output_path = output_dir / (image_path.stem + ".png")
            restored.save(output_path)

            print("Saved:", output_path.name)

    print("\nRestoration of all categories complete.")


if __name__ == "__main__":
    main()
