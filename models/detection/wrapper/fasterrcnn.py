from __future__ import annotations

"""Faster R-CNN wrapper: YAML config -> torchvision FasterRCNN."""

import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from modules.nn.mdmb import MissedDetectionMemoryBank

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


class FasterRCNNWrapper(BaseDetectionWrapper):
    """
    Wrapper that binds a YAML config to torchvision FasterRCNN.

    pre_neck is inserted between the backbone body and FPN.
    post_neck is inserted between the FPN output and the detection heads.
    """

    def __init__(
        self,
        cfg: dict,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mdmb = mdmb

        backbone = build_backbone_with_fpn(
            cfg,
            extra_blocks=LastLevelMaxPool(),
            pre_neck=pre_neck,
            post_neck=post_neck,
        )

        rpn = cfg.get("rpn", {})
        roi = cfg.get("roi_head", {})
        tfm = cfg.get("transform", {})

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=cfg.get("num_classes", 91),
            min_size=tfm.get("min_size", 800),
            max_size=tfm.get("max_size", 1333),
            rpn_pre_nms_top_n_train=rpn.get("pre_nms_top_n_train", 2000),
            rpn_pre_nms_top_n_test=rpn.get("pre_nms_top_n_test", 1000),
            rpn_post_nms_top_n_train=rpn.get("post_nms_top_n_train", 2000),
            rpn_post_nms_top_n_test=rpn.get("post_nms_top_n_test", 1000),
            rpn_nms_thresh=rpn.get("nms_thresh", 0.7),
            rpn_score_thresh=rpn.get("score_thresh", 0.0),
            rpn_fg_iou_thresh=rpn.get("fg_iou_thresh", 0.7),
            rpn_bg_iou_thresh=rpn.get("bg_iou_thresh", 0.3),
            rpn_batch_size_per_image=rpn.get("batch_size_per_image", 256),
            rpn_positive_fraction=rpn.get("positive_fraction", 0.5),
            box_score_thresh=roi.get("box_score_thresh", 0.05),
            box_nms_thresh=roi.get("box_nms_thresh", 0.5),
            box_detections_per_img=roi.get("box_detections_per_img", 100),
            box_fg_iou_thresh=roi.get("box_fg_iou_thresh", 0.5),
            box_bg_iou_thresh=roi.get("box_bg_iou_thresh", 0.5),
            box_batch_size_per_image=roi.get("box_batch_size_per_image", 512),
            box_positive_fraction=roi.get("box_positive_fraction", 0.25),
            **kwargs,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        **kwargs,
    ) -> "FasterRCNNWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            mdmb=mdmb,
            **kwargs,
        )
