from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "models" / "hinet" / "HINet" / "basicsr" / "models" / "archs"))
sys.path.append(str(ROOT / "models" / "hinet" / "training"))

from hinet_arch import HINet
from dataset import read_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = ROOT / "models" / "hinet" / "checkpoints" / "hinet_best.pth"


def main():
    input_folder = ROOT / "data" / "random_degradation" / "test" / "degraded"
    output_folder = ROOT / "models" / "hinet" / "restored"
    output_folder.mkdir(parents=True, exist_ok=True)

    model = HINet(in_chn=1, wf=32, depth=5).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    for path in sorted(input_folder.glob("*")):
        image = read_image(path)
        h, w = image.shape

        # Pad dimensions to multiples of 16, required by HINet.
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))

        # Restore in 128x128 tiles to reduce GPU memory use.
        result = torch.zeros_like(tensor)
        with torch.no_grad():
            for y in range(0, tensor.shape[-2], 128):
                for x in range(0, tensor.shape[-1], 128):
                    patch = tensor[:, :, y:y + 128, x:x + 128]
                    ph, pw = patch.shape[-2:]
                    if ph < 128 or pw < 128:
                        patch = torch.nn.functional.pad(patch, (0, 128-pw, 0, 128-ph))
                    output = model(patch)[-1]
                    result[:, :, y:y + ph, x:x + pw] = output[:, :, :ph, :pw]

        result = result[:, :, :h, :w].squeeze().cpu().numpy()
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(result).save(output_folder / path.name)
        print("Restored:", path.name)


if __name__ == "__main__":
    main()
