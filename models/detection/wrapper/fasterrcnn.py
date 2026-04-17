from __future__ import annotations

"""Faster R-CNN wrapper: YAML config -> torchvision FasterRCNN."""

import weakref

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from modules.nn import MDMBObservation, normalize_xyxy_boxes, select_topk_indices
from modules.nn.mdmb import MissedDetectionMemoryBank

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


class MDMBRoIHeads(RoIHeads):
    """RoIHeads variant that exports sampled proposal observations to MDMB."""

    def __init__(self, *args, mdmb: MissedDetectionMemoryBank | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mdmb_ref = weakref.ref(mdmb) if mdmb is not None else None
        self._mdmb_context: dict[str, object] | None = None
        self._last_box_features: torch.Tensor | None = None
        self._box_head_hook = self.box_head.register_forward_hook(self._capture_box_features)
        self._box_predictor_hook = self.box_predictor.register_forward_hook(
            self._capture_box_predictions
        )

    @classmethod
    def from_existing(
        cls,
        roi_heads: RoIHeads,
        mdmb: MissedDetectionMemoryBank | None,
    ) -> "MDMBRoIHeads":
        return cls(
            box_roi_pool=roi_heads.box_roi_pool,
            box_head=roi_heads.box_head,
            box_predictor=roi_heads.box_predictor,
            fg_iou_thresh=roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=getattr(roi_heads.box_coder, "weights", None),
            score_thresh=roi_heads.score_thresh,
            nms_thresh=roi_heads.nms_thresh,
            detections_per_img=roi_heads.detections_per_img,
            mask_roi_pool=roi_heads.mask_roi_pool,
            mask_head=roi_heads.mask_head,
            mask_predictor=roi_heads.mask_predictor,
            keypoint_roi_pool=roi_heads.keypoint_roi_pool,
            keypoint_head=roi_heads.keypoint_head,
            keypoint_predictor=roi_heads.keypoint_predictor,
            mdmb=mdmb,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if self.training:
            self._mdmb_context = {
                "targets": targets,
                "image_shapes": image_shapes,
            }
        try:
            return super().forward(features, proposals, image_shapes, targets)
        finally:
            self._mdmb_context = None
            self._last_box_features = None

    def select_training_samples(
        self,
        proposals: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        proposals, matched_idxs, labels, regression_targets = super().select_training_samples(
            proposals,
            targets,
        )
        if self._mdmb_context is not None:
            self._mdmb_context["sampled_proposals"] = proposals
            self._mdmb_context["matched_idxs"] = matched_idxs
            self._mdmb_context["labels"] = labels
        return proposals, matched_idxs, labels, regression_targets

    def _capture_box_features(self, module, inputs, output) -> None:
        if self.training and self._mdmb_context is not None:
            self._last_box_features = output.detach()

    def _capture_box_predictions(self, module, inputs, output) -> None:
        mdmb = self._get_mdmb()
        if not self.training or mdmb is None or self._mdmb_context is None:
            return
        if self._last_box_features is None:
            return
        if not isinstance(output, tuple) or len(output) != 2:
            return

        proposals = self._mdmb_context.get("sampled_proposals")
        matched_idxs = self._mdmb_context.get("matched_idxs")
        labels = self._mdmb_context.get("labels")
        targets = self._mdmb_context.get("targets")
        image_shapes = self._mdmb_context.get("image_shapes")
        if not all(
            value is not None
            for value in (proposals, matched_idxs, labels, targets, image_shapes)
        ):
            return

        self._update_mdmb(
            mdmb=mdmb,
            proposals=proposals,  # type: ignore[arg-type]
            matched_idxs=matched_idxs,  # type: ignore[arg-type]
            labels=labels,  # type: ignore[arg-type]
            class_logits=output[0].detach(),
            box_features=self._last_box_features,
            targets=targets,  # type: ignore[arg-type]
            image_shapes=image_shapes,  # type: ignore[arg-type]
        )

    def _update_mdmb(
        self,
        *,
        mdmb: MissedDetectionMemoryBank,
        proposals: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        labels: list[torch.Tensor],
        class_logits: torch.Tensor,
        box_features: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> None:
        counts = [proposal.shape[0] for proposal in proposals]
        if not counts or sum(counts) == 0:
            return

        feature_chunks = box_features.split(counts, dim=0)
        logits_chunks = class_logits.split(counts, dim=0)
        observations: list[MDMBObservation] = []
        max_candidates = mdmb.config.max_entries_per_image * 4

        for (
            proposals_per_image,
            matched_idxs_per_image,
            labels_per_image,
            features_per_image,
            logits_per_image,
            target_per_image,
            image_shape,
        ) in zip(
            proposals,
            matched_idxs,
            labels,
            feature_chunks,
            logits_chunks,
            targets,
            image_shapes,
            strict=True,
        ):
            gt_boxes = target_per_image["boxes"]
            gt_labels = target_per_image["labels"]
            if proposals_per_image.numel() == 0 or gt_boxes.numel() == 0:
                continue

            iou_matrix = box_ops.box_iou(gt_boxes, proposals_per_image)
            iou_max, nearest_gt = iou_matrix.max(dim=0)
            gt_classes = gt_labels[nearest_gt]

            probabilities = torch.softmax(logits_per_image, dim=-1)
            proposal_indices = torch.arange(
                probabilities.shape[0],
                device=probabilities.device,
            )
            gt_scores = probabilities[proposal_indices, gt_classes]
            pred_scores, pred_labels = probabilities[:, 1:].max(dim=-1)
            pred_labels = pred_labels + 1

            detected_mask = (
                (labels_per_image > 0)
                & (iou_max >= mdmb.config.iou_high)
                & (pred_labels == gt_classes)
                & (pred_scores >= mdmb.config.detection_score_threshold)
            )
            candidate_mask = (iou_max >= mdmb.config.iou_low) | detected_mask
            candidate_indices = torch.where(candidate_mask)[0]
            if candidate_indices.numel() == 0:
                continue

            priority = iou_max[candidate_indices] + gt_scores[candidate_indices]
            keep_order = select_topk_indices(priority, k=max_candidates)
            candidate_indices = candidate_indices[keep_order]

            normalized_boxes = normalize_xyxy_boxes(
                proposals_per_image[candidate_indices],
                image_shape,
            )
            for local_index, candidate_index in enumerate(candidate_indices.tolist()):
                image_id = int(target_per_image["image_id"].item())
                detected = bool(detected_mask[candidate_index].item())
                effective_iou = float(iou_max[candidate_index].item())
                if detected:
                    effective_iou = max(effective_iou, mdmb.config.iou_high)
                observations.append(
                    MDMBObservation(
                        image_id=image_id,
                        region_coords=normalized_boxes[local_index],
                        iou_max=effective_iou,
                        cls_score=float(gt_scores[candidate_index].item()),
                        feature_vec=features_per_image[candidate_index],
                        gt_class=int(gt_classes[candidate_index].item()),
                        detected=detected,
                        source="proposal",
                    )
                )

        if observations:
            mdmb.update(observations)

    def _get_mdmb(self) -> MissedDetectionMemoryBank | None:
        if self._mdmb_ref is None:
            return None
        return self._mdmb_ref()


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

        if mdmb is not None:
            self.model.roi_heads = MDMBRoIHeads.from_existing(self.model.roi_heads, mdmb)

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
