"""DCASE-style dense autoencoder and a compact machine-ID CNN."""

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    """One model per machine type, pooled across IDs, as in DCASE 2020.

    BatchNorm epsilon/momentum and linear initialization follow Keras conventions.
    This PyTorch implementation is not bitwise identical to the historical code.
    """

    def __init__(self, input_dimensions=640, latent_dimensions=8):
        super().__init__()
        layers = []
        previous = input_dimensions
        for width in [128] * 4 + [latent_dimensions] + [128] * 4:
            layers.extend([nn.Linear(previous, width),
                           nn.BatchNorm1d(width, eps=1e-3, momentum=0.01), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, input_dimensions))
        self.network = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class MachineIdCNN(nn.Module):
    """Retain frequency position while pooling time for clip-level ID prediction."""

    def __init__(self, classes=4):
        super().__init__()
        layers = [nn.AvgPool2d(2)]
        previous = 1
        for width in (8, 16, 32):
            layers.extend([nn.Conv2d(previous, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)])
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(64, classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class ResidualBlock(nn.Module):
    """Two convolutions with a channel projection when widths differ."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = nn.Identity() if input_channels == output_channels else nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False), nn.BatchNorm2d(output_channels))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class ResidualMachineIdCNN(nn.Module):
    """Small residual model, trained from scratch with the same ID objective."""

    def __init__(self, classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            ResidualBlock(8, 8), nn.MaxPool2d(2),
            ResidualBlock(8, 16), nn.MaxPool2d(2),
            ResidualBlock(16, 32), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(64, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class DepthwiseMachineIdCNN(nn.Module):
    """Separate spatial filtering from channel mixing to reduce parameters."""

    def __init__(self, classes=4):
        super().__init__()
        layers = [nn.AvgPool2d(2), nn.Conv2d(1, 8, 3, padding=1, bias=False),
                  nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2)]
        previous = 8
        for width in (16, 32):
            layers.extend([
                nn.Conv2d(previous, previous, 3, padding=1, groups=previous, bias=False),
                nn.BatchNorm2d(previous), nn.ReLU(),
                nn.Conv2d(previous, width, 1, bias=False), nn.BatchNorm2d(width),
                nn.ReLU(), nn.MaxPool2d(2),
            ])
            previous = width
        layers.append(nn.AdaptiveAvgPool2d((4, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(64, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class TemporalStatisticsPool(nn.Module):
    """Retain frequency regions and summarize temporal level and variability."""

    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, None))
        mean = x.mean(dim=-1, keepdim=True)
        # Epsilon keeps gradients finite for constant feature maps.
        std = (x.var(dim=-1, keepdim=True, unbiased=False) + 1e-6).sqrt()
        return torch.cat((mean, std), dim=1)


class TemporalStatsMachineIdCNN(nn.Module):
    """Compact baseline with temporal mean/std pooling for fan ID learning.

    Mean pooling represents persistent spectral structure; standard deviation
    retains variation that mean-only pooling discards. This is a hypothesis to
    evaluate, not an established improvement in anomaly detection.
    """

    def __init__(self, classes=4):
        super().__init__()
        layers = [nn.AvgPool2d(2)]
        previous = 1
        for width in (8, 16, 32):
            layers.extend([nn.Conv2d(previous, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)])
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = TemporalStatisticsPool()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(64, classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


CNN_MODELS = {
    "machine_id_cnn": MachineIdCNN,
    "residual_id_cnn": ResidualMachineIdCNN,
    "depthwise_id_cnn": DepthwiseMachineIdCNN,
    "temporal_stats_id_cnn": TemporalStatsMachineIdCNN,
}
