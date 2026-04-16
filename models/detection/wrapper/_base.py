from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

SUPPORTED_RESNET_BACKBONES = {
    "resnet18": ResNet18_Weights,
    "resnet34": ResNet34_Weights,
    "resnet50": ResNet50_Weights,
    "resnet101": ResNet101_Weights,
    "resnet152": ResNet152_Weights,
}

SUPPORTED_MOBILENET_BACKBONES = {
    "mobilenet2": MobileNet_V2_Weights,
    "mobilenet3s": MobileNet_V3_Small_Weights,
    "mobilenet3l": MobileNet_V3_Large_Weights,
}

SUPPORTED_BACKBONES = {
    **SUPPORTED_RESNET_BACKBONES,
    **SUPPORTED_MOBILENET_BACKBONES,
}


def load_cfg(path: str | Path) -> dict:
    """Read a YAML config file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_weights(backbone_name: str, pretrained: str | None):
    """
    Convert a YAML pretrained string into a torchvision WeightsEnum value.

    Example:
        pretrained="DEFAULT" -> ResNet50_Weights.DEFAULT
    """
    if pretrained is None:
        return None
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {backbone_name!r}. "
            f"Choose from {list(SUPPORTED_BACKBONES)}"
        )
    weights_cls = SUPPORTED_BACKBONES[backbone_name]
    if not hasattr(weights_cls, pretrained):
        raise ValueError(f"{weights_cls.__name__} has no attribute {pretrained!r}")
    return getattr(weights_cls, pretrained)


def _build_resnet_fpn(
    name: str,
    weights,
    trainable_layers: int,
    extra_blocks,
) -> "BackboneWithFPN":
    """Build a ResNet backbone with FPN."""
    return resnet_fpn_backbone(
        backbone_name=name,
        weights=weights,
        trainable_layers=trainable_layers,
        extra_blocks=extra_blocks,
    )


def _build_mobilenet_fpn(
    name: str,
    weights,
    trainable_layers: int,
    extra_blocks,
) -> "BackboneWithFPN":
    """Build a MobileNet backbone with FPN."""
    from torchvision.models.detection.backbone_utils import mobilenet_backbone

    mobilenet_name_map = {
        "mobilenet2": "mobilenet_v2",
        "mobilenet3s": "mobilenet_v3_small",
        "mobilenet3l": "mobilenet_v3_large",
    }
    return mobilenet_backbone(
        backbone_name=mobilenet_name_map[name],
        weights=weights,
        fpn=True,
        trainable_layers=trainable_layers,
        extra_blocks=extra_blocks,
    )


def build_backbone_with_fpn(
    cfg: dict,
    extra_blocks=None,
    pre_neck: "nn.Module | None" = None,
    post_neck: "nn.Module | None" = None,
) -> "ExtensibleBackboneWithFPN | BackboneWithFPN":
    """
    Build a BackboneWithFPN from a YAML config.

    If either pre_neck or post_neck is provided, the returned backbone wraps
    torchvision's body/FPN pair and exposes extension points before and after
    the neck.
    """
    backbone_cfg = cfg["backbone"]
    _ = cfg.get("neck", {})
    name = backbone_cfg["name"]
    pretrained = backbone_cfg.get("pretrained")
    trainable_layers = backbone_cfg.get("trainable_layers", 3)

    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {name!r}. "
            f"Choose from {list(SUPPORTED_BACKBONES)}"
        )

    weights = resolve_weights(name, pretrained)

    if name in SUPPORTED_RESNET_BACKBONES:
        raw = _build_resnet_fpn(name, weights, trainable_layers, extra_blocks)
    else:
        raw = _build_mobilenet_fpn(name, weights, trainable_layers, extra_blocks)

    if pre_neck is None and post_neck is None:
        return raw

    return ExtensibleBackboneWithFPN(
        body=raw.body,
        fpn=raw.fpn,
        out_channels=raw.out_channels,
        pre_neck=pre_neck,
        post_neck=post_neck,
    )


class ExtensibleBackboneWithFPN(nn.Module):
    """
    Backbone wrapper with extension points around the FPN.

    pre_neck and post_neck must both be nn.Module instances that accept and
    return dict[str, Tensor].
    """

    out_channels: int

    def __init__(
        self,
        body: nn.Module,
        fpn: nn.Module,
        out_channels: int,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.body = body
        self.fpn = fpn
        self.out_channels = out_channels
        self.pre_neck = pre_neck
        self.post_neck = post_neck

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = self.body(x)

        if self.pre_neck is not None:
            features = self.pre_neck(features)

        features = self.fpn(features)

        if self.post_neck is not None:
            features = self.post_neck(features)

        return features


class BaseDetectionWrapper(nn.Module):
    """Base class for detection model wrappers."""

    model: nn.Module

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        """Match torchvision detection model forward signature."""
        return self.model(images, targets)
