from pathlib import Path
import shutil
import random
import numpy as np
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from degradation.degradation import (
    add_gaussian_noise,
    add_blur,
    add_fading,
    add_stains
)


# ============================================================
# CONFIGURATION
# ============================================================

# Original dataset in Downloads
SOURCE_DIR = Path(
    r"C:\Users\PARVATHY\Downloads\BleedThroughDatabase\Bleed-Through Database Images Update"
)
print("SOURCE_DIR:", SOURCE_DIR)
print("SOURCE EXISTS:", SOURCE_DIR.exists())
# Project data directory
OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Reproducibility
SEED = 42

# Dataset split
TRAIN_COUNT = 40
VAL_COUNT = 5
TEST_COUNT = 5


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# FIND VALID DATASET PAIRS
# ============================================================


def find_pairs():
    pairs = []

    all_files = list(SOURCE_DIR.iterdir())

    print(f"Total files found in source folder: {len(all_files)}")

    # Show TIFF files Python can see
    tif_files = [
        f for f in all_files
        if f.suffix.lower() in [".tif", ".tiff"]
    ]

    print(f"Total TIFF files found: {len(tif_files)}")

    print("\nFirst 10 TIFF files:")
    for f in tif_files[:10]:
        print(" ", f.name)

    # Find ground-truth files
    gt_files = [
        f for f in tif_files
        if ".gt." in f.name.lower()
    ]

    print(f"\nGT files found: {len(gt_files)}")

    for gt_file in gt_files:
        # Example:
        # AC.MSBMMM.90r.gt.tif
        # -> AC.MSBMMM.90r.tif

        degraded_name = gt_file.name.replace(".gt.tif", ".tif")

        degraded_file = SOURCE_DIR / degraded_name

        if degraded_file.exists():
            pairs.append((degraded_file, gt_file))

    pairs.sort(key=lambda pair: pair[0].name)

    return pairs


# ============================================================
# CREATE FIXED TRAIN / VAL / TEST SPLIT
# ============================================================

def create_split(pairs):

    pairs = pairs.copy()

    random.Random(SEED).shuffle(pairs)

    train = pairs[:TRAIN_COUNT]

    val = pairs[
        TRAIN_COUNT:
        TRAIN_COUNT + VAL_COUNT
    ]

    test = pairs[
        TRAIN_COUNT + VAL_COUNT:
        TRAIN_COUNT + VAL_COUNT + TEST_COUNT
    ]

    return {
        "train": train,
        "val": val,
        "test": test
    }


# ============================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================

def create_directories(dataset_name):

    dataset_dir = OUTPUT_DIR / dataset_name

    for split in ["train", "val", "test"]:

        (dataset_dir / split / "degraded").mkdir(
            parents=True,
            exist_ok=True
        )

        (dataset_dir / split / "clean").mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_grayscale(path):

    image = Image.open(path).convert("L")

    image = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    return image


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(image, path):

    image = np.clip(
        image,
        0,
        1
    )

    image = (
        image * 255
    ).astype(np.uint8)

    Image.fromarray(image).save(path)


# ============================================================
# APPLY SYNTHETIC DEGRADATION
# ============================================================

def generate_degraded_image(
    clean_image,
    degradation_type
):

    if degradation_type == "noise":

        return add_gaussian_noise(
            clean_image,
            sigma=0.20
        )

    elif degradation_type == "blur":

        return add_blur(
            clean_image,
            sigma=3.0
        )

    elif degradation_type == "fading":

        return add_fading(
            clean_image,
            strength=0.40
        )

    elif degradation_type == "stains":

        return add_stains(
            clean_image,
            num_stains=15,
            min_radius=20,
            max_radius=100
        )

    else:

        raise ValueError(
            f"Unknown degradation type: {degradation_type}"
        )


# ============================================================
# GENERATE SYNTHETIC DATASET
# ============================================================

def generate_synthetic_dataset(
    dataset_name,
    split_data
):

    print(
        f"\nGenerating {dataset_name} dataset..."
    )

    create_directories(
        dataset_name
    )

    for split, pairs in split_data.items():

        print(
            f"  {split}: {len(pairs)} images"
        )

        for degraded_source, gt_source in pairs:

            # Use GT image as the clean source
            clean_image = load_grayscale(
                gt_source
            )

            # Generate degradation
            degraded_image = generate_degraded_image(
                clean_image,
                dataset_name
            )

            filename = gt_source.name

            # Save clean image
            clean_output = (
                OUTPUT_DIR
                / dataset_name
                / split
                / "clean"
                / filename
            )

            save_image(
                clean_image,
                clean_output
            )

            # Save degraded image
            degraded_output = (
                OUTPUT_DIR
                / dataset_name
                / split
                / "degraded"
                / filename
            )

            save_image(
                degraded_image,
                degraded_output
            )


# ============================================================
# GENERATE REAL BLEED-THROUGH DATASET
# ============================================================

def generate_bleedthrough_dataset(
    split_data
):

    dataset_name = "bleedthrough"

    print(
        "\nPreparing bleed-through dataset..."
    )

    create_directories(
        dataset_name
    )

    for split, pairs in split_data.items():

        print(
            f"  {split}: {len(pairs)} images"
        )

        for degraded_source, gt_source in pairs:

            filename = gt_source.name

            # Copy actual degraded image
            degraded_output = (
                OUTPUT_DIR
                / dataset_name
                / split
                / "degraded"
                / filename
            )

            shutil.copy2(
                degraded_source,
                degraded_output
            )

            # Copy actual ground truth
            clean_output = (
                OUTPUT_DIR
                / dataset_name
                / split
                / "clean"
                / filename
            )

            shutil.copy2(
                gt_source,
                clean_output
            )


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_dataset(dataset_name):

    dataset_dir = (
        OUTPUT_DIR / dataset_name
    )

    print(
        f"\nVerifying {dataset_name}..."
    )

    for split, expected_count in [
        ("train", TRAIN_COUNT),
        ("val", VAL_COUNT),
        ("test", TEST_COUNT)
    ]:

        degraded_dir = (
            dataset_dir
            / split
            / "degraded"
        )

        clean_dir = (
            dataset_dir
            / split
            / "clean"
        )

        degraded_files = list(
            degraded_dir.glob("*")
        )

        clean_files = list(
            clean_dir.glob("*")
        )

        print(
            f"  {split}: "
            f"{len(degraded_files)} degraded, "
            f"{len(clean_files)} clean"
        )

        if len(degraded_files) != expected_count:
            raise RuntimeError(
                f"{dataset_name}/{split} "
                f"should contain {expected_count} "
                f"degraded images."
            )

        if len(clean_files) != expected_count:
            raise RuntimeError(
                f"{dataset_name}/{split} "
                f"should contain {expected_count} "
                f"clean images."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HISTORICAL DOCUMENT DATASET GENERATION")
    print("=" * 60)

    print(
        f"\nSource dataset:\n{SOURCE_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Find pairs
    # --------------------------------------------------------

    pairs = find_pairs()

    print(
        f"\nFound {len(pairs)} valid image pairs."
    )

    if len(pairs) != 50:

        raise RuntimeError(
            f"Expected 50 pairs, "
            f"but found {len(pairs)}."
        )

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    split_data = create_split(
        pairs
    )

    print("\nDataset split:")

    for split, items in split_data.items():

        print(
            f"  {split}: {len(items)}"
        )

    # --------------------------------------------------------
    # Generate synthetic datasets
    # --------------------------------------------------------

    synthetic_types = [
        "noise",
        "blur",
        "fading",
        "stains"
    ]

    for degradation_type in synthetic_types:

        generate_synthetic_dataset(
            degradation_type,
            split_data
        )

    # --------------------------------------------------------
    # Prepare real bleed-through
    # --------------------------------------------------------

    generate_bleedthrough_dataset(
        split_data
    )

    # --------------------------------------------------------
    # Verify everything
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "VERIFYING ALL DATASETS"
    )

    print(
        "=" * 60
    )

    all_datasets = [
        "noise",
        "blur",
        "fading",
        "stains",
        "bleedthrough"
    ]

    for dataset_name in all_datasets:

        verify_dataset(
            dataset_name
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "DATASET GENERATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nDatasets created in:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()