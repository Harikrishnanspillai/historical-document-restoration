U-NET CENTRALIZED SIX-CATEGORY PIPELINE
=======================================

This package adapts the uploaded U-Net scripts to the shared six-category
dataset used by HINet, NAFNet and DnCNN.

IMPORTANT
---------
Keep your existing unet.py unchanged. The training and inference scripts import
the UNet class from that file.

Folder structure after extraction:

models/unet/
    unet.py
    training/
        dataset.py
        train.py
    inference/
        restore.py
    evaluation/
        evaluate.py

Dataset expected:

data/
    bleedthrough/{train,val,test}/{clean,degraded}
    blur/{train,val,test}/{clean,degraded}
    fading/{train,val,test}/{clean,degraded}
    noise/{train,val,test}/{clean,degraded}
    random_degradation/{train,val,test}/{clean,degraded}
    stains/{train,val,test}/{clean,degraded}

RUN ORDER
---------

From the project root:

1. Train:
   python models/unet/training/train.py

2. Restore all six test categories:
   python models/unet/inference/restore.py

3. Evaluate:
   python models/unet/evaluation/evaluate.py

SETTINGS
--------
- Patch size: 128 x 128
- Batch size: 1 (conservative for GPUs with 4 GB VRAM)
- Epochs: 20
- Learning rate: 0.001
- Loss: MSE
- Training: 10 random patches per training image
- Validation: center crop to 128 x 128
- Inference: 256 x 256 tiles with 32-pixel overlap

OUTPUTS
-------
models/unet/checkpoints/unet_latest.pth
models/unet/checkpoints/unet_best.pth
models/unet/restored/<category>/*.png
models/unet/evaluation/per_image_results.csv
models/unet/evaluation/category_summary.csv

The original uploaded training setup used one degradation category and 64x64
patches. This centralized version trains one shared U-Net across all six
categories. It retains the uploaded UNet architecture and MSE training loss.

The uploaded baseline.py, plot_comparison.py and plot_training.py are not
required for this centralized train/restore/evaluate pipeline. They are not
included in this package.

NOTE
----
The scripts are syntax-checked, but have not been run against your local GPU,
dataset or trained checkpoint. Start training before running restoration.
