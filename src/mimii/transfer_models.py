"""ImageNet CNNs with the same healthy-ID head/backbone training interface."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import efficientnet_b0, resnet18


WEIGHTS = {
    "resnet18": ("resnet18-f37072fd.pth", "f37072fd"),
    "efficientnet_b0": ("efficientnet_b0_rwightman-7f5810bc.pth", "7f5810bc"),
}


def pretrained_hash(architecture):
    filename, prefix = WEIGHTS[architecture]
    digest = hashlib.sha256((Path("checkpoints") / filename).read_bytes()).hexdigest()
    if not digest.startswith(prefix):
        raise ValueError("Pretrained checkpoint failed official hash-prefix verification")
    return digest


def grayscale_conv(original):
    layer = nn.Conv2d(1, original.out_channels, original.kernel_size, original.stride,
                      original.padding, dilation=original.dilation, bias=original.bias is not None)
    with torch.no_grad():
        layer.weight.copy_(original.weight.sum(dim=1, keepdim=True))
        if original.bias is not None:
            layer.bias.copy_(original.bias)
    return layer


class ResNetClassifier(nn.Module):
    def __init__(self, backbone, classes):
        super().__init__()
        backbone.conv1 = grayscale_conv(backbone.conv1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(backbone.fc.in_features, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_transfer(architecture, classes=4, pretrained=True):
    constructor = {"resnet18": resnet18, "efficientnet_b0": efficientnet_b0}[architecture]
    model = constructor(weights=None)
    if pretrained:
        pretrained_hash(architecture)
        model.load_state_dict(torch.load(Path("checkpoints") / WEIGHTS[architecture][0], map_location="cpu", weights_only=True))
    if architecture == "resnet18":
        model = ResNetClassifier(model, classes)
    else:
        model.features[0][0] = grayscale_conv(model.features[0][0])
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    nn.init.normal_(model.classifier[-1].weight, 0, 0.01)
    nn.init.zeros_(model.classifier[-1].bias)
    return model
