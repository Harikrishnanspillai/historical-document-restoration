DnCNN centralized scripts

Copy these folders into your existing:
models/dncnn/

Keep your existing models/dncnn/model.py unchanged. The scripts import
the DnCNN class from that file and preserve its residual-learning design.

The new dataset loader uses the common project dataset:
data/<category>/<split>/{clean,degraded}/

It uses all six categories:
bleedthrough, blur, fading, noise, random_degradation, stains.

Training settings:
- Patch size: 128
- Batch size: 2 (conservative for a 4 GB GPU)
- Epochs: 20
- Learning rate: 0.001
- DnCNN: 20 layers, 64 features, one grayscale input channel
- 10 random patches per training image per epoch
- Validation uses the common 30 validation pairs

Run from the project root:
    python models/dncnn/training/train.py
    python models/dncnn/inference/restore.py
    python models/dncnn/evaluation/evaluate.py

Checkpoints:
    models/dncnn/checkpoints/dncnn_latest.pth
    models/dncnn/checkpoints/dncnn_best.pth

Restored images:
    models/dncnn/restored/<category>/

Evaluation CSV files:
    models/dncnn/evaluation/per_image_results.csv
    models/dncnn/evaluation/category_summary.csv

The previous synthetic_pairs.py and OCR inference flow are not used by
this centralized pipeline. Keep or delete those old files as you prefer;
these scripts do not import them.
