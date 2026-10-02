NAFNet scripts - matching the HINet project code style
======================================================

Extract the models/nafnet folder into your project root.

Expected official repository location:
models/nafnet/NAFNet/

Scripts included:
training/dataset.py
training/train.py
inference/restore.py
evaluation/evaluate.py

The dataset script follows the same paired grayscale image loading, random
128x128 crop, and flip augmentation logic as the HINet dataset script.

Training follows the HINet training structure:
- 128x128 patches
- 10 patches per training image
- batch size 1 (conservative for the 4 GB RTX 3050)
- 20 epochs
- Adam optimizer
- L1 loss
- tqdm progress bar and epoch train/validation loss
- latest and best model checkpoints

NAFNet architecture settings:
- img_channel=1
- width=32
- middle_blk_num=1
- enc_blk_nums=[1, 1, 1, 28]
- dec_blk_nums=[1, 1, 1, 1]

Run these commands from the project root:

python models/nafnet/training/train.py
python models/nafnet/inference/restore.py
python models/nafnet/evaluation/evaluate.py

Restoration processes all six test categories:
bleedthrough, blur, fading, noise, random_degradation, stains

Restored files are saved under:
models/nafnet/restored/<category>/

Checkpoints are saved under:
models/nafnet/checkpoints/

Important:
The official NAFNet repo must be cloned into models/nafnet/NAFNet.
Do not blindly install the official repository's older requirements into your
current Python/PyTorch environment. These files have been syntax-checked,
but still need an import and GPU smoke test on your machine.
