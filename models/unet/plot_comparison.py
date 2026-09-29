import matplotlib.pyplot as plt
import numpy as np

images = [
    "AC.MSBMMM.90v",
    "NLI.MSG18.362",
    "NUIM.MSM86.14",
    "NUIM.MSR68.81",
    "UCD.MSA20.127r"
]

baseline_psnr = [14.0748, 10.4550, 11.9446, 9.6153, 10.0679]
unet_psnr = [15.8422, 11.9758, 13.5955, 12.5292, 15.4579]

x = np.arange(len(images))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(x - width/2, baseline_psnr, width, label="Degraded Baseline")
plt.bar(x + width/2, unet_psnr, width, label="U-Net")

plt.xlabel("Test Image")
plt.ylabel("PSNR (dB)")
plt.title("Baseline vs U-Net PSNR")
plt.xticks(x, images, rotation=20)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("psnr_comparison.png", dpi=300)
plt.show()