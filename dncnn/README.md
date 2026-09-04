# DnCNN Bleed-Through Removal + OCR Pipeline

Built for the "detect and remove ink bleed-through, preserve foreground text"
project described in your literature review. Matches the approach used in
Savino & Tonazzini (2024) — recto/(optional verso) input, model-based
training pairs, and a downstream text-extraction step.

**Important framing:** DnCNN is an *image denoiser*. It cleans up the page —
it does not read text. To get actual transcribed text, this pipeline chains
DnCNN's cleaned output into Tesseract OCR as a second stage. This has been
tested end-to-end and works, but keep two caveats in mind for a research
writeup:
1. OCR accuracy on historical handwriting (even after cleanup) is limited
   with general-purpose Tesseract — it's trained on modern typefaces. For
   publishable transcription accuracy you'd eventually want a dedicated
   handwritten-text-recognition (HTR) model (e.g. fine-tuned TrOCR) instead
   of Tesseract, especially for non-Latin/historical scripts.
2. If your manuscripts use a historical or non-Latin script, install the
   matching Tesseract language pack (see notes in `inference.py`).

## Files

| File | Purpose |
|---|---|
| `model.py` | DnCNN architecture (residual learning: predicts the bleed-through pattern, subtracts it) |
| `dataset.py` | Loads recto/clean(/verso) image pairs, does patch extraction + augmentation |
| `synthetic_pairs.py` | Generates training pairs from clean pages when you don't have manually-labelled ground truth (mirrors Savino & Tonazzini's model-based training trick) |
| `train.py` | Training loop |
| `inference.py` | Runs the trained model on a full page (sliding-window, since DnCNN trains on small patches) + OCR |
| `requirements.txt` | Dependencies |

## Setup

```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr          # OCR engine
# sudo apt-get install tesseract-ocr-<lang> # for non-English/historical scripts
```

## Step 1 — Get your data into the right shape

Using the **ISOS Bleed-Through Database** (recommended, matches your lit review's benchmark):
```
data/
  recto/   <- degraded scans (bleed-through visible), e.g. 0001.tif
  clean/   <- ground-truth clean images, same filenames
  verso/   <- (optional) verso side, same filenames
```
The ISOS database gives you recto/verso pairs directly. If it doesn't
include a fully "clean" ground truth image, you can either (a) use the
verso as auxiliary input via `--use_verso`, or (b) generate synthetic
ground-truth pairs (Step 1b) to pretrain, then fine-tune on real pairs.

### Step 1b — No ground truth? Generate synthetic pairs

```bash
python synthetic_pairs.py --clean_dir data/clean_source --out_dir data --num_pairs 2000
```
Takes a folder of clean (non-degraded) text page images and synthetically
composites bleed-through onto them, writing matching `recto/` + `clean/`
folders ready for training.

## Step 2 — Train

```bash
# Recto-only model
python train.py --data_dir data --epochs 50 --patch_size 64 --num_layers 20

# Recto+verso model (uses both sides as input, generally more accurate)
python train.py --data_dir data --epochs 50 --use_verso
```
Checkpoints are saved every 10 epochs to `checkpoints/`.

## Step 3 — Run on a real page + extract text

```bash
python inference.py \
  --checkpoint checkpoints/dncnn_final.pth \
  --input path/to/degraded_page.tif \
  --output_dir results \
  --ocr_lang eng
```
Outputs:
- `results/<name>_cleaned.png` — the bleed-through-removed image
- `results/<name>_text.txt` — the OCR-extracted text

## Verified working

This pipeline was smoke-tested end-to-end in this environment: synthetic
data generation → training (loss dropped from 0.445 to 0.017 in 3 epochs)
→ sliding-window inference → OCR text extraction, all completed
successfully. For real results on your manuscripts, train longer (50+
epochs) on real ISOS/DIBCO data rather than the tiny synthetic smoke-test
set.

## Suggested next steps for your project

1. Download the ISOS Bleed-Through Database, arrange into `data/recto` +
   `data/verso` (+ `data/clean` if you hand-align ground truth patches).
2. Train the `--use_verso` variant — using both sides is the core idea
   from your Paper 3 (Savino & Tonazzini) and should outperform recto-only.
3. Evaluate with PSNR/SSIM against ground truth (matches the metrics used
   in Papers 2 and 5 of your lit review) — happy to add an eval script
   that computes these if useful.
4. If OCR accuracy on real manuscripts is poor, consider swapping
   Tesseract for a fine-tuned TrOCR model as a follow-up improvement —
   this is the natural "future work" angle your lit review already flags
   (Paper 5's discussion of transformers and text recovery).
