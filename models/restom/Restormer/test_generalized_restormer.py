import torch

from basicsr.data import create_dataset, create_dataloader
from basicsr.models import create_model
from basicsr.utils.options import parse


# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------
opt = parse(
    "Denoising/options/train_HistoricalGeneralized_Restormer.yml",
    is_train=True
)
opt["dist"] = False

# ---------------------------------------------------------
# Create generalized six-category dataset
# ---------------------------------------------------------
dataset_opt = opt["datasets"]["train"]

dataset = create_dataset(dataset_opt)

print("\nDataset size:", len(dataset))


# ---------------------------------------------------------
# Create DataLoader
# ---------------------------------------------------------
loader = create_dataloader(
    dataset,
    dataset_opt,
    num_gpu=1,
    dist=False
)


# ---------------------------------------------------------
# Get one batch
# ---------------------------------------------------------
batch = next(iter(loader))

print("LQ batch:", batch["lq"].shape)
print("GT batch:", batch["gt"].shape)


# ---------------------------------------------------------
# Build Restormer
# ---------------------------------------------------------
model = create_model(opt)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.net_g.to(device)
model.net_g.eval()


# ---------------------------------------------------------
# Move input to GPU
# ---------------------------------------------------------
lq = batch["lq"].to(device)
gt = batch["gt"].to(device)


# ---------------------------------------------------------
# Forward pass
# ---------------------------------------------------------
with torch.no_grad():
    output = model.net_g(lq)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
print("Output:", output.shape)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

print(
    "Output range:",
    output.min().item(),
    "to",
    output.max().item()
)

print("\nGENERALIZED RESTORMER TEST PASSED!")