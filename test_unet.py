import torch
from models.unet.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = UNet(in_channels=1, out_channels=1).to(device)

test_image = torch.randn(1, 1, 256, 256).to(device)

with torch.no_grad():
    output = model(test_image)

print("Input shape :", test_image.shape)
print("Output shape:", output.shape)