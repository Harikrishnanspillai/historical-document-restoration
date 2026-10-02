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
    bleedthrough=False,
    verso_image=None,
    noise_sigma=0.10,
    blur_sigma=1.8,
    fading_strength=0.25,
    num_stains=8,
    bleedthrough_alpha=0.22
):
    """
    Apply multiple synthetic degradations sequentially.

    Parameters:
        image: Normalized grayscale image [0, 1].
        noise: Apply Gaussian and salt-and-pepper noise.
        blur: Apply Gaussian blur.
        fading: Apply fading.
        stains: Apply stains.
        bleedthrough: Apply bleed-through.
        verso_image: Opposite-side image required for bleed-through.

    Returns:
        Degraded image.
    """

    result = image.copy()

    # 1. Bleed-through
    if bleedthrough:
        if verso_image is None:
            raise ValueError(
                "verso_image is required for bleed-through."
            )

        if verso_image.shape != image.shape:
            raise ValueError(
                "verso_image and image must have the same dimensions."
            )

        result = add_bleedthrough(
            result,
            verso_image,
            alpha=bleedthrough_alpha
        )

    # 2. Blur
    if blur:
        result = add_blur(
            result,
            blur_sigma
        )

    # 3. Fading
    if fading:
        result = add_fading(
            result,
            fading_strength
        )

    # 4. Stains
    if stains:
        result = add_stains(
            result,
            num_stains
        )

    # 5. Noise
    if noise:
        result = add_gaussian_noise(
            result,
            noise_sigma
        )

    return np.clip(result, 0, 1)


def random_degradation(
    image,
    verso_image=None,
    min_degradations=2,
    max_degradations=5,
    rng=None,
    return_degradations=False
):
    """
    Randomly select and apply 2-5 degradation types.

    Uses apply_mixed_degradation() to apply the selected
    degradations.

    Parameters:
        image: Normalized grayscale image [0, 1].
        verso_image: Opposite-side image for bleed-through.
        min_degradations: Minimum number of degradations.
        max_degradations: Maximum number of degradations.
        rng: Optional NumPy random generator.
        return_degradations: Also return selected degradation names.

    Returns:
        Degraded image.

        If return_degradations=True:
            (degraded_image, selected_degradations)
    """

    degradations = [
        "noise",
        "blur",
        "fading",
        "stains"
    ]
    
    if verso_image is not None: degradations.append("bleedthrough")

    # Validate input
    if not 2 <= min_degradations <= max_degradations <= 5:
        raise ValueError(
            "Require 2 <= min_degradations "
            "<= max_degradations <= 5."
        )

    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError(
            "image must be a 2D grayscale NumPy array."
        )

    # Initialize random generator
    if rng is None:
        rng = np.random.default_rng()

    # Randomly select number of degradations
    num_selected = int(
        rng.integers(
            min_degradations,
            max_degradations + 1
        )
    )

    # Randomly select unique degradation types
    selected = rng.choice(
        degradations,
        size=num_selected,
        replace=False
    ).tolist()

    # Apply selected degradations
    degraded_image = apply_mixed_degradation(
        image=image,
        noise="noise" in selected,
        blur="blur" in selected,
        fading="fading" in selected,
        stains="stains" in selected,
        bleedthrough="bleedthrough" in selected,
        verso_image=verso_image
    )

    if return_degradations:
        return degraded_image, selected

    return degraded_image
