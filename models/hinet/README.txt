HINet - simple training files

1. Extract this ZIP into models/hinet (the HINet folder already there should remain).
   The files should merge into:
   models/hinet/training, models/hinet/inference, models/hinet/evaluation.

2. From the project root, run the one-time grayscale patch:
   python models\hinet\patch_grayscale.py

3. Train:
   python models\hinet\training\train.py

4. Restore:
   python models\hinet\inference\restore.py

5. Evaluate:
   python models\hinet\evaluation\evaluate.py

The scripts use data/<category>/<split>/{clean,degraded}, the existing six
categories, grayscale TIFF images, 128x128 training patches, batch size 1,
HINet wf=32, and the separate models/hinet/checkpoints folder.

If training runs out of GPU memory, change PATCH_SIZE = 64 in train.py.
This is a simplified first implementation. Run the training smoke test/first
epoch and check its output before leaving a long run unattended.
