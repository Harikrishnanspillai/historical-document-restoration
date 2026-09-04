from pathlib import Path
import shutil
import random
import argparse

from PIL import Image


def find_image_pairs(source_dir):

    source_dir = Path(source_dir)

    degraded_files = []
    pairs = []

    # Search recursively for TIFF files
    for file in source_dir.rglob("*"):
        if not file.is_file():
            continue

        name = file.name.lower()

        # Ignore RGB versions
        if name.endswith(".rgb.tif"):
            continue

        # Ignore ground-truth files
        if name.endswith(".gt.tif"):
            continue

        # Only process TIFF files
        if not name.endswith(".tif"):
            continue

        degraded_files.append(file)

    print(f"Found {len(degraded_files)} possible degraded images.")

    for degraded in degraded_files:

        gt = degraded.with_name(
            degraded.name[:-4] + ".gt.tif"
        )

        if gt.exists():
            pairs.append((degraded, gt))
        else:
            print(f"[WARNING] Ground truth not found for:")
            print(f"          {degraded.name}")

    return pairs


def verify_pair(degraded, gt):
    try:
        with Image.open(degraded) as degraded_img:
            degraded_size = degraded_img.size

        with Image.open(gt) as gt_img:
            gt_size = gt_img.size

        if degraded_size != gt_size:
            print("[WARNING] Dimension mismatch:")
            print(f"          Degraded: {degraded.name} -> {degraded_size}")
            print(f"          GT:       {gt.name} -> {gt_size}")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Could not read image pair:")
        print(f"        {degraded.name}")
        print(f"        {e}")
        return False


def create_directories(output_dir):
    output_dir = Path(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / split / "degraded").mkdir(
            parents=True,
            exist_ok=True
        )

        (output_dir / split / "gt").mkdir(
            parents=True,
            exist_ok=True
        )


def copy_pair(degraded, gt, output_dir, split):
    """
    Copy a degraded/GT pair into the selected split.
    """

    output_dir = Path(output_dir)

    degraded_destination = (
        output_dir / split / "degraded" / degraded.name
    )

    gt_destination = (
        output_dir / split / "gt" / gt.name
    )

    shutil.copy2(degraded, degraded_destination)
    shutil.copy2(gt, gt_destination)


def main():

    parser = argparse.ArgumentParser(
        description="Prepare paired historical document dataset for SwinIR."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="dataset/raw",
        help="Directory containing the original dataset."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset",
        help="Directory where train/val/test will be created."
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Fraction of data used for training."
    )

    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Fraction of data used for validation."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting."
    )

    args = parser.parse_args()

    # Check split percentages
    test_ratio = 1.0 - args.train - args.val

    if test_ratio <= 0:
        raise ValueError(
            "Train + validation percentages must be less than 1.0"
        )

    print("=" * 60)
    print("Historical Document Dataset Preparation")
    print("=" * 60)

    print(f"Source directory : {args.source}")
    print(f"Output directory : {args.output}")
    print(f"Train           : {args.train * 100:.1f}%")
    print(f"Validation      : {args.val * 100:.1f}%")
    print(f"Test            : {test_ratio * 100:.1f}%")
    print()

    # Find paired images
    pairs = find_image_pairs(args.source)

    if len(pairs) == 0:
        print("No valid degraded/GT pairs found.")
        return

    print(f"\nFound {len(pairs)} valid image pairs.")

    # Verify dimensions
    print("\nChecking image dimensions...")

    valid_pairs = []

    for degraded, gt in pairs:
        if verify_pair(degraded, gt):
            valid_pairs.append((degraded, gt))

    print(f"Valid pairs: {len(valid_pairs)}")
    print(f"Invalid pairs: {len(pairs) - len(valid_pairs)}")

    if len(valid_pairs) == 0:
        print("No valid image pairs available.")
        return

    # Shuffle reproducibly
    random.seed(args.seed)
    random.shuffle(valid_pairs)

    # Calculate split sizes
    total = len(valid_pairs)

    train_count = int(total * args.train)
    val_count = int(total * args.val)

    train_pairs = valid_pairs[:train_count]

    val_pairs = valid_pairs[
        train_count:train_count + val_count
    ]

    test_pairs = valid_pairs[
        train_count + val_count:
    ]

    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    # Create directories
    create_directories(args.output)

    # Copy files
    print("\nCopying dataset...")

    for split_name, split_pairs in splits.items():

        print(f"\n{split_name.upper()}: {len(split_pairs)} pairs")

        for degraded, gt in split_pairs:
            copy_pair(
                degraded,
                gt,
                args.output,
                split_name
            )

    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"Total pairs : {total}")
    print(f"Train       : {len(train_pairs)}")
    print(f"Validation  : {len(val_pairs)}")
    print(f"Test        : {len(test_pairs)}")

    print("\nOutput structure:")
    print(f"{args.output}/")
    print("├── train/")
    print("│   ├── degraded/")
    print("│   └── gt/")
    print("├── val/")
    print("│   ├── degraded/")
    print("│   └── gt/")
    print("└── test/")
    print("    ├── degraded/")
    print("    └── gt/")


if __name__ == "__main__":
    main()