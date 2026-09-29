import cv2
import numpy as np


def add_gaussian_noise(image, sigma=0.20, salt_pepper=0.03):
    result = image.copy()

    # Gaussian noise
    noise = np.random.normal(0, sigma, image.shape)
    result = result + noise

    # Salt-and-pepper noise
    random_map = np.random.random(image.shape)

    result[random_map < salt_pepper / 2] = 0.0
    result[random_map > 1 - salt_pepper / 2] = 1.0

    return np.clip(result, 0, 1)


def add_blur(image, sigma=3.0):
    """
    Apply Gaussian blur to a grayscale image.

    Parameters:
        image: NumPy array with values in [0, 1]
        sigma: Blur strength

    Returns:
        Blurred image
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    return np.clip(blurred, 0, 1)


def add_fading(image, strength=0.4):
    """
    Simulate faded or low-contrast document content.

    Parameters:
        image: NumPy array with values in [0, 1]
        strength: Fading strength between 0 and 1

    Returns:
        Faded image
    """
    mean_intensity = np.mean(image)

    faded = (
        image * (1 - strength)
        + mean_intensity * strength
    )

    return np.clip(faded, 0, 1)


def add_stains(image, num_stains=15, min_radius=20, max_radius=100):
    result = image.copy()
    height, width = image.shape

    for _ in range(num_stains):

        # Random location
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)

        # Random size
        radius = np.random.randint(min_radius, max_radius + 1)

        # Irregular stain mask
        mask = np.zeros_like(image, dtype=np.float32)

        cv2.ellipse(
            mask,
            (center_x, center_y),
            (radius, int(radius * np.random.uniform(0.5, 1.5))),
            np.random.randint(0, 180),
            0,
            360,
            1.0,
            -1
        )

        # Strong Gaussian blur → diffuse stain edges
        blur_radius = max(5, radius // 3)
        if blur_radius % 2 == 0:
            blur_radius += 1

        mask = cv2.GaussianBlur(
            mask,
            (blur_radius, blur_radius),
            blur_radius / 2
        )

        # Random stain darkness
        darkness = np.random.uniform(0.25, 0.75)

        # Apply stain
        result = result * (1 - darkness * mask)

    return np.clip(result, 0, 1)


def add_bleedthrough(
    clean_image,
    verso_image,
    alpha=0.35
):
    """
    Simulate bleed-through by blending information
    from the opposite side of a document.

    Parameters:
        clean_image: Clean front-side image
        verso_image: Opposite-side image
        alpha: Bleed-through intensity

    Returns:
        Image with simulated bleed-through
    """

    degraded = (
        (1 - alpha) * clean_image
        + alpha * verso_image
    )

    return np.clip(degraded, 0, 1)


def apply_mixed_degradation(
    image,
    noise=False,
    blur=False,
    fading=False,
    stains=False,
    noise_sigma=0.15,
    blur_sigma=3.0,
    fading_strength=0.4,
    num_stains=8
):
    """
    Apply multiple synthetic degradations sequentially.

    Parameters:
        image: Normalized grayscale image
        noise: Apply Gaussian noise
        blur: Apply Gaussian blur
        fading: Apply fading
        stains: Apply stains

    Returns:
        Degraded image
    """

    result = image.copy()

    if blur:
        result = add_blur(
            result,
            blur_sigma
        )

    if fading:
        result = add_fading(
            result,
            fading_strength
        )

    if stains:
        result = add_stains(
            result,
            num_stains
        )

    if noise:
        result = add_gaussian_noise(
            result,
            noise_sigma
        )

    return np.clip(result, 0, 1)