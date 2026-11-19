"""Compact acoustic CNN inspired by residual and inverted-bottleneck networks.

No novelty or superiority claim: the experiment tests temporal statistics pooling
with a fixed healthy-ID objective and a separately labelled mean-only ablation.
"""

import torch
from torch import nn


class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.network = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(4, channels // 4), 1),
                                     nn.SiLU(), nn.Conv2d(max(4, channels // 4), channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.network(x)


class AcousticResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=(1, 1)):
        super().__init__()
        expanded = input_channels * 2
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, expanded, 1, bias=False), nn.BatchNorm2d(expanded), nn.SiLU(),
            nn.Conv2d(expanded, expanded, (3, 5), stride=stride, padding=(1, 2), groups=expanded, bias=False),
            nn.BatchNorm2d(expanded), nn.SiLU(), ChannelGate(expanded),
            nn.Conv2d(expanded, output_channels, 1, bias=False), nn.BatchNorm2d(output_channels))
        self.skip = nn.Identity() if input_channels == output_channels and stride == (1, 1) else nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(output_channels))
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class FrequencyStatisticsPool(nn.Module):
    def __init__(self, include_std=True):
        super().__init__()
        self.include_std = include_std

    def forward(self, x):
        # Four frequency regions remain distinct; only time is summarized.
        if x.shape[2] != 4:
            x = nn.functional.adaptive_avg_pool2d(x, (4, None))
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, unbiased=False, keepdim=True) + 1e-6).sqrt() if self.include_std else torch.zeros_like(mean)
        return torch.cat((mean, std), dim=1)


class AcousticResidualStatsCNN(nn.Module):
    def __init__(self, classes=4, include_std=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.SiLU(),
            AcousticResidualBlock(16, 24, (2, 2)), AcousticResidualBlock(24, 24),
            AcousticResidualBlock(24, 48, (2, 2)), AcousticResidualBlock(48, 48),
            AcousticResidualBlock(48, 80, (2, 1)), AcousticResidualBlock(80, 80))
        self.pool = FrequencyStatisticsPool(include_std)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(640, 128), nn.SiLU(), nn.Dropout(.2), nn.Linear(128, classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
