"""Configurable model zoo for the pilot (M1-M6)."""
from __future__ import annotations

import torch
from torch import nn

import torchvision.models as tv_models


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, double: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if double:
            layers += [
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LightCNN(nn.Module):
    """M1 — lightweight CNN (3 conv blocks + GAP)."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        widths = (16, 32, 64)
        blocks = []
        cin = 3
        for w in widths:
            blocks.append(ConvBlock(cin, w))
            blocks.append(nn.MaxPool2d(2))
            cin = w
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)


class StandardCNN(nn.Module):
    """M2-M5 — standard CNN (4 conv blocks + GAP + MLP head)."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.3, base_width: int = 24):
        super().__init__()
        widths = [base_width * (2 ** i) for i in range(4)]
        blocks = []
        cin = 3
        for w in widths:
            blocks.append(ConvBlock(cin, w))
            blocks.append(nn.MaxPool2d(2))
            cin = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(widths[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


class ResNetBackbone(nn.Module):
    """M6 — torchvision ResNet-18 with configurable pretrained weights."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tv_models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        if dropout > 0:
            net = nn.Sequential(net, nn.Dropout(dropout))
        self.net = net

    def forward(self, x):
        return self.net(x)


def build_model(cfg: dict) -> nn.Module:
    arch = cfg["model"]["architecture"]
    num_classes = cfg["model"].get("num_classes", 5)
    dropout = cfg["model"].get("dropout", 0.3)
    if arch == "light_cnn":
        return LightCNN(num_classes)
    if arch == "standard_cnn":
        return StandardCNN(num_classes, dropout=dropout)
    if arch == "resnet18":
        return ResNetBackbone(num_classes, pretrained=cfg["model"].get("pretrained", True), dropout=dropout)
    raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
