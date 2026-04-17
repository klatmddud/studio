from __future__ import annotations

"""FCOS wrapper: YAML config -> torchvision FCOS."""

import torch.nn as nn
from torchvision.models.detection import FCOS
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


class FCOSWrapper(BaseDetectionWrapper):
    """
    Wrapper that binds a YAML config to torchvision FCOS.

    FCOS uses LastLevelP6P7 to generate the P3-P7 pyramid levels.
    """

    def __init__(
        self,
        cfg: dict,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        backbone = build_backbone_with_fpn(
            cfg,
            # FCOS expects five pyramid levels: P3-P5 from the FPN plus P6-P7.
            extra_blocks=LastLevelP6P7(256, 256),
            pre_neck=pre_neck,
            post_neck=post_neck,
            returned_layers=[2, 3, 4],
        )

        head = cfg.get("head", {})
        tfm = cfg.get("transform", {})

        self.model = FCOS(
            backbone=backbone,
            num_classes=cfg.get("num_classes", 91),
            min_size=tfm.get("min_size", 800),
            max_size=tfm.get("max_size", 1333),
            score_thresh=head.get("score_thresh", 0.2),
            nms_thresh=head.get("nms_thresh", 0.6),
            detections_per_img=head.get("detections_per_img", 100),
            topk_candidates=head.get("topk_candidates", 1000),
            **kwargs,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        **kwargs,
    ) -> "FCOSWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(load_cfg(yaml_path), pre_neck=pre_neck, post_neck=post_neck, **kwargs)
