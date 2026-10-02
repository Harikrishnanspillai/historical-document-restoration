SwinIR updated restoration and evaluation scripts

Copy these folders into your existing:
models/Swin/

Files:
- inference/restore.py
- evaluation/evaluate.py

The scripts keep the existing SwinIR checkpoint, model settings, paths,
six degradation categories, and overlapping 128x128 tile restoration.

The evaluation script now calculates MSE, PSNR, and SSIM directly.
It no longer imports metrics.py, so metrics.py is not needed by these
two scripts and can be deleted if no other script imports it.

Run from the project root:
    python models/Swin/inference/restore.py
    python models/Swin/evaluation/evaluate.py

Restored images:
    models/Swin/outputs/restored/<category>/

Evaluation CSV files:
    models/Swin/outputs/evaluation/
