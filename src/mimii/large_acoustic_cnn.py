"""Wider acoustic residual CNN for a matched training capacity comparison."""

from torch import nn

from .acoustic_cnn import AcousticResidualBlock, FrequencyStatisticsPool


class LargeAcousticResidualStatsCNN(nn.Module):
    """Retain small model depth, pooling, kernels and strides; widen channels."""

    def __init__(self, classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            AcousticResidualBlock(32, 64, (2, 2)), AcousticResidualBlock(64, 64),
            AcousticResidualBlock(64, 128, (2, 2)), AcousticResidualBlock(128, 128),
            AcousticResidualBlock(128, 256, (2, 1)), AcousticResidualBlock(256, 256))
        self.pool = FrequencyStatisticsPool(include_std=True)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(2048, 256), nn.SiLU(),
            nn.Dropout(.2), nn.Linear(256, classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
