import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DATA_ROOT = Path("data")

CATEGORIES = [
    "noise",
    "blur",
    "fading",
    "stains",
    "bleedthrough",
    "random_degradation"
]


def load_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image.astype(np.float32) / 255.0


def calculate_metrics(restored, clean):

    if restored.shape != clean.shape:
        restored = cv2.resize(
            restored,
            (clean.shape[1], clean.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    restored = np.clip(restored, 0, 1)
    clean = np.clip(clean, 0, 1)

    mse = np.mean((restored - clean) ** 2)

    psnr = peak_signal_noise_ratio(
        clean,
        restored,
        data_range=1.0
    )

    ssim = structural_similarity(
        clean,
        restored,
        data_range=1.0
    )

    return mse, psnr, ssim


results = []


print("\n" + "=" * 70)
print("RESTORMER FINAL TEST EVALUATION")
print("=" * 70)


for category in CATEGORIES:

    restored_dir = (
        DATA_ROOT /
        category /
        "test" /
        "restored_restormer"
    )

    clean_dir = (
        DATA_ROOT /
        category /
        "test" /
        "clean"
    )

    if not restored_dir.exists():
        print(f"\nWARNING: Missing {restored_dir}")
        continue

    restored_files = sorted(
        restored_dir.glob("*_restored.png")
    )

    print(f"\n{category.upper()}")
    print("-" * 70)

    for restored_path in restored_files:

        original_stem = restored_path.stem.replace(
            "_restored", ""
        )

        clean_path = None

        for extension in [
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg"
        ]:
            candidate = clean_dir / (
                original_stem + extension
            )

            if candidate.exists():
                clean_path = candidate
                break

        if clean_path is None:
            print(
                f"  CLEAN IMAGE NOT FOUND: "
                f"{original_stem}"
            )
            continue

        restored = load_grayscale(
            restored_path
        )

        clean = load_grayscale(
            clean_path
        )

        mse, psnr, ssim = calculate_metrics(
            restored,
            clean
        )

        results.append({
            "Category": category,
            "Image": original_stem,
            "MSE": mse,
            "PSNR_dB": psnr,
            "SSIM": ssim
        })

        print(
            f"  {original_stem}: "
            f"MSE={mse:.6f}, "
            f"PSNR={psnr:.4f} dB, "
            f"SSIM={ssim:.4f}"
        )


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(results)


# ---------------------------------------------------------
# SAVE IMAGE-LEVEL RESULTS
# ---------------------------------------------------------

image_csv = Path(
    "restormer_final_test_image_metrics.csv"
)

df.to_csv(
    image_csv,
    index=False
)


# ---------------------------------------------------------
# CATEGORY AVERAGES
# ---------------------------------------------------------

category_df = (
    df.groupby("Category")
    .agg({
        "MSE": "mean",
        "PSNR_dB": "mean",
        "SSIM": "mean"
    })
    .reset_index()
)

category_df["Image"] = "AVERAGE"

category_df = category_df[
    [
        "Category",
        "Image",
        "MSE",
        "PSNR_dB",
        "SSIM"
    ]
]


# ---------------------------------------------------------
# OVERALL AVERAGE
# ---------------------------------------------------------

overall = pd.DataFrame([{
    "Category": "OVERALL",
    "Image": "AVERAGE",
    "MSE": df["MSE"].mean(),
    "PSNR_dB": df["PSNR_dB"].mean(),
    "SSIM": df["SSIM"].mean()
}])


# ---------------------------------------------------------
# SAVE SUMMARY CSV
# ---------------------------------------------------------

summary_df = pd.concat(
    [
        category_df,
        overall
    ],
    ignore_index=True
)

summary_csv = Path(
    "restormer_final_test_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)


# ---------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORY AVERAGES")
print("=" * 70)

for _, row in category_df.iterrows():

    print(
        f"{row['Category']:20s} "
        f"MSE={row['MSE']:.6f}  "
        f"PSNR={row['PSNR_dB']:.4f} dB  "
        f"SSIM={row['SSIM']:.4f}"
    )


print("\n" + "=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(
    f"Images evaluated : {len(df)}"
)

print(
    f"MSE              : {df['MSE'].mean():.6f}"
)

print(
    f"PSNR             : {df['PSNR_dB'].mean():.4f} dB"
)

print(
    f"SSIM             : {df['SSIM'].mean():.4f}"
)

print("\nCSV files created:")
print(f"  {image_csv}")
print(f"  {summary_csv}")

print("=" * 70)