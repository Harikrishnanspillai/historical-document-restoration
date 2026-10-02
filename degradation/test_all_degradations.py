from pathlib import Path
import numpy as np
from PIL import Image

from degradation import *

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

IMAGE_PATH = BASE_DIR / "test_image.tif"
VERSO_PATH = BASE_DIR / "verso_image.tif"
OUTPUT_DIR = BASE_DIR / "test_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load images
# --------------------------------------------------

image = np.array(
    Image.open(IMAGE_PATH).convert("L"),
    dtype=np.float32
) / 255.0

verso = np.array(
    Image.open(VERSO_PATH).convert("L"),
    dtype=np.float32
) / 255.0


# Make sure verso has the same dimensions
if verso.shape != image.shape:
    verso = np.array(
        Image.fromarray((verso * 255).astype(np.uint8))
        .resize((image.shape[1], image.shape[0]))
    ) / 255.0


# --------------------------------------------------
# Save helper
# --------------------------------------------------

def save_image(array, filename):

    array = np.clip(array, 0, 1)

    output = (array * 255).astype(np.uint8)

    Image.fromarray(output).save(
        OUTPUT_DIR / filename
    )


# --------------------------------------------------
# 1. Original
# --------------------------------------------------

save_image(
    image,
    "original.png"
)


# --------------------------------------------------
# 2. Individual degradations
# --------------------------------------------------

# Noise
noisy = add_gaussian_noise(
    image,
    sigma=0.20
)

# Blur
blurred = add_blur(
    image,
    sigma=3.0
)

# Fading
faded = add_fading(
    image,
    strength=0.40
)

# Stains
stained = add_stains(
    image,
    num_stains=15,
    min_radius=20,
    max_radius=100
)

# Bleed-through
bleedthrough = add_bleedthrough(
    image,
    verso,
    alpha=0.35
)


# Save individual results

save_image(
    noisy,
    "noise.png"
)

save_image(
    blurred,
    "blur.png"
)

save_image(
    faded,
    "fading.png"
)

save_image(
    stained,
    "stains.png"
)

save_image(
    bleedthrough,
    "bleedthrough.png"
)


# --------------------------------------------------
# 3. Mixed degradation - ALL
# --------------------------------------------------
#
# bleed-through
# + blur
# + fading
# + stains
# + noise
#
# Mild degradation so that the original document
# remains visible and readable.
# --------------------------------------------------

# Start with clean image
mixed_all = image.copy()

# 1. Bleed-through
mixed_all = add_bleedthrough(
    mixed_all,
    verso,
    alpha=0.22
)

# 2. Slight blur
mixed_all = add_blur(
    mixed_all,
    sigma=1.8
)

# 3. Mild fading
mixed_all = add_fading(
    mixed_all,
    strength=0.25
)

# 4. Mild stains
mixed_all = add_stains(
    mixed_all,
    num_stains=8,
    min_radius=15,
    max_radius=80
)

# 5. Mild noise
mixed_all = add_gaussian_noise(
    mixed_all,
    sigma=0.10
)

save_image(
    mixed_all,
    "mixed_all.png"
)


# --------------------------------------------------
# 4. Mixed degradation - RANDOM
# --------------------------------------------------

degradations = [
    "noise",
    "blur",
    "fading",
    "stains",
    "bleedthrough"
]

# Fixed seed for reproducibility
rng = np.random.default_rng(42)

print("\nRandom mixed degradation combinations")
print("=" * 50)


# Generate 10 different random combinations

for i in range(10):

    mixed_random = random_degradation(image, verso_image=verso)


    # ----------------------------------------------
    # Save this random combination
    # ----------------------------------------------

    filename = (
        f"mixed_random_{i + 1}.png"
    )

    save_image(
        mixed_random,
        filename
    )


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\n" + "=" * 50)
print("All degradation tests completed.")
print(
    f"Output folder: {OUTPUT_DIR}"
)