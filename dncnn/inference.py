"""
Run trained DnCNN on a degraded manuscript image to remove bleed-through,
then run OCR on the cleaned result to extract the actual text.

IMPORTANT: DnCNN is an image-to-image denoiser -- it removes the
bleed-through pattern and gives you back a cleaner image. It does not
itself "read" text. To get actual transcribed text, this script chains
DnCNN's output into Tesseract OCR as a second stage. If your manuscripts
use a historical / non-Latin script, you'll need a Tesseract language
pack for that script (see notes at the bottom of this file).

Usage:
    python inference.py --checkpoint checkpoints/dncnn_final.pth \
                         --input path/to/degraded_page.tif \
                         --output_dir results \
                         --ocr_lang eng

Requires (in addition to torch/pillow):
    pip install pytesseract --break-system-packages
    # plus the tesseract binary itself, e.g.:
    #   sudo apt-get install tesseract-ocr tesseract-ocr-<lang>
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import DnCNN

try:
    import pytesseract
except ImportError:
    pytesseract = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="path to a degraded recto image")
    p.add_argument("--verso", type=str, default=None, help="matching verso image, if model was trained with --use_verso")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--patch_size", type=int, default=256, help="tile size for sliding-window inference")
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--ocr_lang", type=str, default="eng", help="tesseract language code, e.g. eng, san, ara")
    p.add_argument("--ocr_psm", type=int, default=6, help="tesseract page-segmentation mode")
    return p.parse_args()


def load_gray(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def sliding_window_denoise(model, recto, verso, patch_size, overlap, device):
    """Runs the model over the full page in overlapping tiles and blends them,
    since DnCNN is trained on small patches but manuscript pages are large."""
    h, w = recto.shape
    stride = patch_size - overlap
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    # simple raised-cosine-ish blending window to avoid tile seams
    win_1d = np.hanning(patch_size)
    win_1d[win_1d == 0] = 1e-3  # avoid zero weight at edges
    window = np.outer(win_1d, win_1d)

    ys = list(range(0, max(h - patch_size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - patch_size, 0) + 1, stride)) or [0]
    if ys[-1] != h - patch_size and h > patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size and w > patch_size:
        xs.append(w - patch_size)

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                y2, x2 = min(y + patch_size, h), min(x + patch_size, w)
                y1, x1 = y2 - patch_size, x2 - patch_size  # handle edge tiles that are smaller than patch_size
                y1, x1 = max(y1, 0), max(x1, 0)

                recto_tile = recto[y1:y2, x1:x2]
                th, tw = recto_tile.shape
                if verso is not None:
                    verso_tile = verso[y1:y2, x1:x2]
                    tile_in = np.stack([recto_tile, verso_tile], axis=0)[None]  # (1, 2, H, W)
                else:
                    tile_in = recto_tile[None, None]  # (1, 1, H, W)

                tile_tensor = torch.from_numpy(tile_in.astype(np.float32)).to(device)
                out_tile = model(tile_tensor).cpu().numpy()[0, 0]

                w_tile = window[:th, :tw]
                output[y1:y2, x1:x2] += out_tile * w_tile
                weight[y1:y2, x1:x2] += w_tile

    weight[weight == 0] = 1.0
    return np.clip(output / weight, 0.0, 1.0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    in_channels = ckpt.get("in_channels", 1)
    num_layers = ckpt.get("num_layers", 20)
    model = DnCNN(in_channels=in_channels, out_channels=1, num_layers=num_layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint {args.checkpoint} (in_channels={in_channels}, num_layers={num_layers})")

    recto = load_gray(args.input)
    verso = load_gray(args.verso) if (args.verso and in_channels == 2) else None
    if in_channels == 2 and verso is None:
        raise ValueError("This checkpoint was trained with --use_verso; pass --verso <path>.")

    cleaned = sliding_window_denoise(model, recto, verso, args.patch_size, args.overlap, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    cleaned_img = Image.fromarray((cleaned * 255).astype(np.uint8))
    cleaned_path = out_dir / f"{stem}_cleaned.png"
    cleaned_img.save(cleaned_path)
    print(f"Saved cleaned image: {cleaned_path}")

    # --- OCR stage: turn the cleaned image into actual text ---
    if pytesseract is None:
        print("\npytesseract is not installed -- skipping OCR step.")
        print("Install with: pip install pytesseract --break-system-packages")
        print("and make sure the `tesseract` binary is on your system.")
        return

    text = pytesseract.image_to_string(cleaned_img, lang=args.ocr_lang, config=f"--psm {args.ocr_psm}")
    text_path = out_dir / f"{stem}_text.txt"
    text_path.write_text(text, encoding="utf-8")
    print(f"Saved extracted text: {text_path}")
    print("\n--- Extracted text preview ---")
    print(text[:500] if text.strip() else "(no text detected -- check ocr_lang / image quality)")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------
# Notes on OCR language packs:
# - "eng" works out of the box with tesseract-ocr's default install.
# - For historical scripts (e.g. old Latin, Devanagari, Khmer, Coptic),
#   install the matching traineddata file, e.g.:
#     sudo apt-get install tesseract-ocr-san   # Sanskrit/Devanagari
#     sudo apt-get install tesseract-ocr-khm   # Khmer
#   Tesseract is trained on modern typefaces, so accuracy on heavily
#   degraded historical handwriting will be limited even after DnCNN
#   cleanup -- for research-grade results you may eventually want a
#   dedicated HTR model (e.g. TrOCR fine-tuned on your script) instead
#   of general-purpose Tesseract.
# -----------------------------------------------------------------------
