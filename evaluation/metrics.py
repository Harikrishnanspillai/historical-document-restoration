import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calculate_mse(predicted, target):
    """
    Mean Squared Error.
    Lower is better.
    """
    predicted = predicted.astype(np.float32)
    target = target.astype(np.float32)

    return np.mean((predicted - target) ** 2)

def calculate_psnr(predicted, target):
    """
    Calculate PSNR between two grayscale images.

    Images must have the same dimensions and values in [0, 1].
    """

    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"predicted={predicted.shape}, "
            f"target={target.shape}"
        )

    return peak_signal_noise_ratio(
        target,
        predicted,
        data_range=1.0
    )


def calculate_ssim(predicted, target):
    """
    Calculate SSIM between two grayscale images.

    Images must have the same dimensions and values in [0, 1].
    """

    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"predicted={predicted.shape}, "
            f"target={target.shape}"
        )

    return structural_similarity(
        target,
        predicted,
        data_range=1.0
    )


def calculate_metrics(predicted, target):
    """
    Calculate all currently supported image-quality metrics.
    """

    mse = calculate_mse(predicted, target)
    psnr = calculate_psnr(predicted, target)
    ssim = calculate_ssim(predicted, target)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim
    }