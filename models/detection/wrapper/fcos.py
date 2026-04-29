from __future__ import annotations

"""FCOS wrapper: YAML config -> torchvision FCOS."""

from collections import OrderedDict
from typing import Any
import warnings

import torch
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
            extra_blocks=LastLevelP6P7(256, 256),
            pre_neck=pre_neck,
            post_neck=post_neck,
            returned_layers=[2, 3, 4],
        )

        head = cfg.get("head", {})
        transform_cfg = cfg.get("transform", {})

        self.model = FCOS(
            backbone=backbone,
            num_classes=cfg.get("num_classes", 91),
            min_size=transform_cfg.get("min_size", 800),
            max_size=transform_cfg.get("max_size", 1333),
            score_thresh=head.get("score_thresh", 0.2),
            nms_thresh=head.get("nms_thresh", 0.6),
            detections_per_img=head.get("detections_per_img", 100),
            topk_candidates=head.get("topk_candidates", 1000),
            **kwargs,
        )
        self._train_metrics: dict[str, float] = {}

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        """Run FCOS and add MissHead loss when ReMiss MissHead is attached."""
        if getattr(self, "miss_head", None) is None:
            self._train_metrics = {}
            return self.model(images, targets)

        return self._forward_with_miss_head(images, targets)

    def get_training_metrics(self) -> dict[str, float]:
        return dict(self._train_metrics)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        **kwargs,
    ) -> "FCOSWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            **kwargs,
        )

    def _forward_with_miss_head(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
    ):
        fcos = self.model
        miss_head = getattr(self, "miss_head")
        missbank = getattr(self, "missbank", None)
        self._train_metrics = {}

        if fcos.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = fcos.transform(images, targets)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: list[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        f"All bounding boxes should have positive height and width. "
                        f"Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = fcos.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())

        head_outputs = fcos.head(feature_list)
        anchors = fcos.anchor_generator(images, feature_list)
        num_anchors_per_level = [x.size(2) * x.size(3) for x in feature_list]

        losses: dict[str, torch.Tensor] = {}
        detections: list[dict[str, torch.Tensor]] = []
        if fcos.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                losses = fcos.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)
                losses.update(
                    self._compute_miss_head_loss(
                        features=feature_list,
                        targets=targets,
                        miss_head=miss_head,
                        missbank=missbank,
                    )
                )
        else:
            split_head_outputs: dict[str, list[torch.Tensor]] = {}
            for key in head_outputs:
                split_head_outputs[key] = list(head_outputs[key].split(num_anchors_per_level, dim=1))
            split_anchors = [list(anchor.split(num_anchors_per_level)) for anchor in anchors]

            detections = fcos.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = fcos.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not fcos._has_warned:
                warnings.warn("FCOS always returns a (Losses, Detections) tuple in scripting")
                fcos._has_warned = True
            return losses, detections
        return fcos.eager_outputs(losses, detections)

    def _compute_miss_head_loss(
        self,
        *,
        features: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        miss_head: nn.Module,
        missbank: Any,
    ) -> dict[str, torch.Tensor]:
        if missbank is None:
            return {}
        epoch = int(getattr(missbank, "current_epoch", 0))
        is_active = getattr(miss_head, "is_active", None)
        if callable(is_active) and not bool(is_active(epoch)):
            return {}

        region_logits = miss_head(features)
        target_labels = missbank.get_batch_labels(
            targets=targets,
            device=region_logits.device,
        )
        losses = miss_head.compute_loss(region_logits, target_labels)
        metrics = miss_head.compute_metrics(region_logits.detach(), target_labels.detach())
        metrics.update(
            _missed_object_region_metrics(
                missbank=missbank,
                targets=targets,
                predicted_regions=miss_head.predict_region(region_logits.detach()),
            )
        )
        self._train_metrics = metrics
        return losses


def _missed_object_region_metrics(
    *,
    missbank: Any,
    targets: list[dict[str, torch.Tensor]],
    predicted_regions: torch.Tensor,
) -> dict[str, float]:
    get_records = getattr(missbank, "get_records", None)
    if not callable(get_records):
        return {}
    threshold = int(missbank.config.target.miss_threshold)
    predictions = predicted_regions.detach().cpu().tolist()
    total = 0
    correct = 0
    weighted_total = 0.0
    weighted_correct = 0.0
    for target, pred_region in zip(targets, predictions, strict=True):
        image_id = target.get("image_id")
        for record in get_records(image_id):
            if not bool(getattr(record, "is_missed", False)):
                continue
            if int(getattr(record, "miss_count", 0)) < threshold:
                continue
            total += 1
            weight = float(getattr(record, "miss_count", 1))
            weighted_total += weight
            if int(pred_region) == int(getattr(record, "region_id", -1)):
                correct += 1
                weighted_correct += weight
    metrics: dict[str, float] = {}
    if total > 0:
        metrics["missed_object_region_acc"] = correct / float(total)
    if weighted_total > 0.0:
        metrics["missed_object_region_acc_weighted"] = weighted_correct / weighted_total
    return metrics
