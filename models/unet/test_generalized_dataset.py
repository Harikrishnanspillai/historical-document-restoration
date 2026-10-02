from pathlib import Path

from dataset import GeneralizedManuscriptDataset


ROOT = Path(__file__).resolve().parents[2] / "data"


dataset = GeneralizedManuscriptDataset(
    root_dir=ROOT,
    split="train",
    patch_size=64,
    augment=True,
    patches_per_image=10,
)

print("\nDataset length:", len(dataset))

x, y = dataset[0]

print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Input min/max:", x.min().item(), x.max().item())
print("Target min/max:", y.min().item(), y.max().item())