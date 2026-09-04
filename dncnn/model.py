"""
DnCNN model for manuscript bleed-through / noise removal.

Reference: Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of
Deep CNN for Image Denoising" (DnCNN), IEEE TIP 2017.

The network learns the *residual* (noise / bleed-through pattern) rather
than the clean image directly, which is what makes DnCNN train fast and
generalize well:

    clean_estimate = input - model(input)
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_layers: int = 17,
        features: int = 64,
        residual_channel: int = 0,
    ):
        """
        Args:
            in_channels: 1 for grayscale recto-only input, 2 if you feed
                         recto+verso stacked as two channels.
            out_channels: normally 1 (the cleaned grayscale page).
            num_layers: total conv layers (17 is the standard DnCNN-S depth;
                        use ~20-25 for harder degradations like bleed-through).
            features: number of feature maps in the hidden layers.
            residual_channel: which input channel to treat as the "base"
                image when in_channels != out_channels (e.g. the recto
                channel, index 0), so the network can still learn a
                residual correction instead of the clean image outright.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_channel = residual_channel
        layers = []

        # First layer: Conv + ReLU (no batchnorm on input layer)
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv only, predicts the residual (noise / bleed-through)
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.dncnn(x)
        if self.in_channels == self.out_channels:
            return x - residual  # predicted clean image
        # multi-channel input (e.g. recto+verso), single-channel output:
        # subtract residual from the designated base channel only
        base = x[:, self.residual_channel:self.residual_channel + 1, :, :]
        return base - residual
