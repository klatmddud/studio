from __future__ import annotations

"""FCOS wrapper: YAML config -> torchvision FCOS."""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.detection import FCOS
from torchvision.ops import boxes as box_ops
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
        bcpc: nn.Module | None = None,
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
        self.bcpc = bcpc
        self._train_metrics: dict[str, float] = {}

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        """Run FCOS with optional BCPC calibration."""
        self._train_metrics = {}
        if self.bcpc is not None:
            return self._forward_with_bcpc(images, targets)
        return self.model(images, targets)

    def _forward_with_bcpc(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            for target in targets:
                boxes = target["boxes"]
                torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                torch._assert(
                    len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                    f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                )

        original_image_sizes: list[tuple[int, int]] = []
        for image in images:
            size = image.shape[-2:]
            torch._assert(
                len(size) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {image.shape[-2:]}",
            )
            original_image_sizes.append((size[0], size[1]))

        images, targets = self.model.transform(images, targets)
        if targets is not None:
            for target_index, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if bool(degenerate_boxes.any().item()):
                    box_index = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    invalid_box: list[float] = boxes[box_index].tolist()
                    torch._assert(
                        False,
                        f"All bounding boxes should have positive height and width. "
                        f"Found invalid box {invalid_box} for target at index {target_index}.",
                    )

        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())

        head_outputs, cls_features = _fcos_head_forward_with_cls_features(
            self.model.head,
            feature_list,
        )
        anchors = self.model.anchor_generator(images, feature_list)
        num_anchors_per_level = [feature.size(2) * feature.size(3) for feature in feature_list]

        losses: dict[str, torch.Tensor] = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            matched_idxs = _match_fcos_targets(
                self.model,
                anchors=anchors,
                targets=targets,
                num_anchors_per_level=num_anchors_per_level,
            )
            losses = self.model.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
            pred_boxes = self.model.box_coder.decode(
                head_outputs["bbox_regression"],
                torch.stack(anchors),
            )
            loss_bcpc = self.bcpc.loss_for_fcos(
                cls_features=cls_features,
                cls_logits=head_outputs["cls_logits"],
                bbox_ctrness=head_outputs["bbox_ctrness"],
                pred_boxes=pred_boxes,
                targets=targets,
                matched_idxs=matched_idxs,
            )
            losses["bcpc"] = loss_bcpc * float(self.bcpc.config.lambda_bg)
        else:
            split_head_outputs = {
                name: list(output.split(num_anchors_per_level, dim=1))
                for name, output in head_outputs.items()
            }
            split_cls_features = list(cls_features.split(num_anchors_per_level, dim=1))
            split_anchors = [list(anchor.split(num_anchors_per_level)) for anchor in anchors]
            detections = _postprocess_detections_with_bcpc(
                self.model,
                self.bcpc,
                head_outputs=split_head_outputs,
                cls_features=split_cls_features,
                anchors=split_anchors,
                image_shapes=images.image_sizes,
            )
            detections = self.model.transform.postprocess(
                detections,
                images.image_sizes,
                original_image_sizes,
            )

        return self.model.eager_outputs(losses, detections)

    def get_training_metrics(self) -> dict[str, float]:
        metrics = dict(self._train_metrics)
        if self.bcpc is not None:
            get_metrics = getattr(self.bcpc, "get_training_metrics", None)
            if callable(get_metrics):
                for name, value in get_metrics().items():
                    if isinstance(value, (int, float)):
                        metrics[str(name)] = float(value)
        backbone = getattr(self.model, "backbone", None)
        post_neck = getattr(backbone, "post_neck", None)
        get_metrics = getattr(post_neck, "get_training_metrics", None)
        if callable(get_metrics):
            for name, value in get_metrics().items():
                if isinstance(value, (int, float)):
                    metrics[str(name)] = float(value)
        return metrics

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        bcpc: nn.Module | None = None,
        **kwargs,
    ) -> "FCOSWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            bcpc=bcpc,
            **kwargs,
        )


def _fcos_head_forward_with_cls_features(
    head: nn.Module,
    features: list[torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    classification_head = head.classification_head
    all_cls_logits: list[torch.Tensor] = []
    all_cls_features: list[torch.Tensor] = []
    for feature in features:
        cls_feature = classification_head.conv(feature)
        cls_logits = classification_head.cls_logits(cls_feature)

        batch_size, _, height, width = cls_logits.shape
        cls_logits = cls_logits.view(
            batch_size,
            -1,
            classification_head.num_classes,
            height,
            width,
        )
        cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
        cls_logits = cls_logits.reshape(batch_size, -1, classification_head.num_classes)
        all_cls_logits.append(cls_logits)

        cls_feature = cls_feature.permute(0, 2, 3, 1)
        cls_feature = cls_feature.reshape(batch_size, -1, int(cls_feature.shape[-1]))
        all_cls_features.append(cls_feature)

    bbox_regression, bbox_ctrness = head.regression_head(features)
    return (
        {
            "cls_logits": torch.cat(all_cls_logits, dim=1),
            "bbox_regression": bbox_regression,
            "bbox_ctrness": bbox_ctrness,
        },
        torch.cat(all_cls_features, dim=1),
    )


def _match_fcos_targets(
    model: FCOS,
    *,
    anchors: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    num_anchors_per_level: list[int],
) -> list[torch.Tensor]:
    matched_idxs: list[torch.Tensor] = []
    for anchors_per_image, targets_per_image in zip(anchors, targets, strict=True):
        if len(targets_per_image["labels"]) == 0:
            matched_idxs.append(
                torch.full(
                    (anchors_per_image.size(0),),
                    -1,
                    dtype=torch.int64,
                    device=anchors_per_image.device,
                )
            )
            continue

        gt_boxes = targets_per_image["boxes"]
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2
        anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]
        pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(dim=2).values
        pairwise_match = pairwise_match < model.center_sampling_radius * anchor_sizes[:, None]

        x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
        pairwise_match &= pairwise_dist.min(dim=2).values > 0

        lower_bound = anchor_sizes * 4
        lower_bound[: num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        upper_bound[-num_anchors_per_level[-1] :] = float("inf")
        pairwise_dist = pairwise_dist.max(dim=2).values
        pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
        min_values, matched_idx = pairwise_match.max(dim=1)
        matched_idx[min_values < 1e-5] = -1
        matched_idxs.append(matched_idx)
    return matched_idxs


def _postprocess_detections_with_bcpc(
    model: FCOS,
    bcpc: nn.Module,
    *,
    head_outputs: dict[str, list[torch.Tensor]],
    cls_features: list[torch.Tensor],
    anchors: list[list[torch.Tensor]],
    image_shapes: list[tuple[int, int]],
) -> list[dict[str, torch.Tensor]]:
    class_logits = head_outputs["cls_logits"]
    box_regression = head_outputs["bbox_regression"]
    box_ctrness = head_outputs["bbox_ctrness"]
    detections: list[dict[str, torch.Tensor]] = []

    for image_index, image_shape in enumerate(image_shapes):
        image_boxes: list[torch.Tensor] = []
        image_scores: list[torch.Tensor] = []
        image_labels: list[torch.Tensor] = []
        anchors_per_image = anchors[image_index]
        level_items = zip(
            [level[image_index] for level in class_logits],
            [level[image_index] for level in box_regression],
            [level[image_index] for level in box_ctrness],
            [level[image_index] for level in cls_features],
            anchors_per_image,
            strict=True,
        )
        for (
            logits_per_level,
            box_regression_per_level,
            box_ctrness_per_level,
            features_per_level,
            anchors_per_level,
        ) in level_items:
            num_classes = int(logits_per_level.shape[-1])
            scores_matrix = torch.sqrt(
                torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
            )
            scores_per_level = scores_matrix.flatten()
            keep_idxs = scores_per_level > model.score_thresh
            topk_idxs = torch.where(keep_idxs)[0]
            if topk_idxs.numel() == 0:
                image_boxes.append(anchors_per_level.new_zeros((0, 4)))
                image_scores.append(scores_per_level.new_zeros((0,)))
                image_labels.append(torch.zeros((0,), dtype=torch.long, device=scores_per_level.device))
                continue

            scores_per_level = scores_per_level[keep_idxs]
            num_topk = min(int(topk_idxs.numel()), int(model.topk_candidates))
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes
            candidate_features = features_per_level[anchor_idxs]
            scores_per_level = bcpc.calibrate_scores(
                candidate_features,
                labels_per_level,
                scores_per_level.detach(),
                scores_per_level,
            )
            calibrated_keep = scores_per_level > model.score_thresh
            if not bool(calibrated_keep.any().item()):
                image_boxes.append(anchors_per_level.new_zeros((0, 4)))
                image_scores.append(scores_per_level.new_zeros((0,)))
                image_labels.append(torch.zeros((0,), dtype=torch.long, device=scores_per_level.device))
                continue

            anchor_idxs = anchor_idxs[calibrated_keep]
            labels_per_level = labels_per_level[calibrated_keep]
            scores_per_level = scores_per_level[calibrated_keep]
            boxes_per_level = model.box_coder.decode(
                box_regression_per_level[anchor_idxs],
                anchors_per_level[anchor_idxs],
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)

        boxes = torch.cat(image_boxes, dim=0)
        scores = torch.cat(image_scores, dim=0)
        labels = torch.cat(image_labels, dim=0)
        keep = box_ops.batched_nms(boxes, scores, labels, model.nms_thresh)
        keep = keep[: model.detections_per_img]
        detections.append(
            {
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            }
        )
    return detections
