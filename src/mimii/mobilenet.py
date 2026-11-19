"""ImageNet MobileNetV3-Small adapted to single-channel acoustic features."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
WEIGHTS_HASH_PREFIX = "047dcff4"


def weights_sha256(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not digest.startswith(WEIGHTS_HASH_PREFIX):
        raise ValueError("Pretrained weights do not match the official PyTorch hash prefix")
    return digest


def convert_first_convolution(model):
    """Sum RGB kernels: equivalent to replicating one standardized channel.

    Input uses training-normal per-mel statistics, not RGB ImageNet transforms.
    No resizing or image augmentation is applied to the spectrogram.
    """
    original = model.features[0][0]
    grayscale = nn.Conv2d(1, original.out_channels, original.kernel_size,
                          original.stride, original.padding, bias=False)
    with torch.no_grad():
        grayscale.weight.copy_(original.weight.sum(dim=1, keepdim=True))
    model.features[0][0] = grayscale
    return model


def build_mobilenet(classes=4, pretrained_path=None):
    model = mobilenet_v3_small(weights=None)
    if pretrained_path is not None:
        weights_sha256(pretrained_path)
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
    convert_first_convolution(model)
    previous = model.classifier[-1]
    model.classifier[-1] = nn.Linear(previous.in_features, classes)
    nn.init.normal_(model.classifier[-1].weight, 0, 0.01)
    nn.init.zeros_(model.classifier[-1].bias)
    return model


def set_training_stage(model, warmup):
    """Freeze parameters AND BatchNorm running statistics during head warmup."""
    for parameter in model.features.parameters():
        parameter.requires_grad_(not warmup)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    model.train()
    if warmup:
        model.features.eval()


def make_optimizer(model, config, warmup):
    groups = [{"params": model.classifier.parameters(), "lr": config.head_warmup_lr if warmup else config.head_finetune_lr}]
    if not warmup:
        groups.append({"params": model.features.parameters(), "lr": config.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)
