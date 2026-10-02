from pathlib import Path
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "models" / "hinet" / "HINet" / "basicsr" / "models" / "archs"))
sys.path.append(str(ROOT / "models" / "hinet" / "training"))

from hinet_arch import HINet
from dataset import CATEGORIES, read_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(predicted, target):
    mse = np.mean((predicted - target) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    ssim = structural_similarity(target, predicted, data_range=1.0)
    return mse, psnr, ssim


def main():
    data_root = ROOT / "data"
    checkpoint = ROOT / "models" / "hinet" / "checkpoints" / "hinet_best.pth"

    model = HINet(in_chn=1, wf=32, depth=5).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    all_metrics = []

    for category in CATEGORIES:
        degraded_folder = data_root / category / "test" / "degraded"
        clean_folder = data_root / category / "test" / "clean"

        for degraded_path in sorted(degraded_folder.glob("*")):
            clean_path = clean_folder / degraded_path.name
            if not clean_path.exists():
                continue

            degraded = read_image(degraded_path)
            clean = read_image(clean_path)
            h, w = degraded.shape

            tensor = torch.from_numpy(degraded).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))

            restored = torch.zeros_like(tensor)
            with torch.no_grad():
                for y in range(0, tensor.shape[-2], 128):
                    for x in range(0, tensor.shape[-1], 128):
                        patch = tensor[:, :, y:y+128, x:x+128]
                        ph, pw = patch.shape[-2:]
                        if ph < 128 or pw < 128:
                            patch = torch.nn.functional.pad(patch, (0,128-pw,0,128-ph))
                        output = model(patch)[-1]
                        restored[:, :, y:y+ph, x:x+pw] = output[:, :, :ph, :pw]

            predicted = restored[:, :, :h, :w].squeeze().cpu().numpy()
            mse, psnr, ssim = calculate_metrics(predicted, clean)
            all_metrics.append((mse, psnr, ssim))
            print(f"{category}/{degraded_path.name}: MSE={mse:.6f}, PSNR={psnr:.3f}, SSIM={ssim:.4f}")

    print("\nAverage test metrics:")
    print("MSE:", np.mean([x[0] for x in all_metrics]))
    print("PSNR:", np.mean([x[1] for x in all_metrics]))
    print("SSIM:", np.mean([x[2] for x in all_metrics]))


if __name__ == "__main__":
    main()
