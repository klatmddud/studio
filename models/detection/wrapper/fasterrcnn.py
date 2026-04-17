from __future__ import annotations

"""Faster R-CNN wrapper: YAML config -> torchvision FasterRCNN."""

import weakref
from typing import Any

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from modules.nn import (
    CounterfactualFeaturePerturbation,
    MDMBObservation,
    normalize_xyxy_boxes,
    select_topk_indices,
)
from modules.nn.mdmb import MissedDetectionMemoryBank
from ops import compute_cfp_loss_dict

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


class MDMBRoIHeads(RoIHeads):
    """RoIHeads variant that exports sampled proposal observations to MDMB."""

    def __init__(
        self,
        *args,
        mdmb: MissedDetectionMemoryBank | None = None,
        cfp: CounterfactualFeaturePerturbation | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mdmb_ref = weakref.ref(mdmb) if mdmb is not None else None
        self._cfp_ref = weakref.ref(cfp) if cfp is not None else None
        self._mdmb_context: dict[str, object] | None = None
        self._last_box_features: torch.Tensor | None = None
        self._cfp_losses: dict[str, torch.Tensor] | None = None
        self._suspend_box_predictor_hook = False
        self._box_head_hook = self.box_head.register_forward_hook(self._capture_box_features)
        self._box_predictor_hook = self.box_predictor.register_forward_hook(
            self._capture_box_predictions
        )

    @classmethod
    def from_existing(
        cls,
        roi_heads: RoIHeads,
        mdmb: MissedDetectionMemoryBank | None,
        cfp: CounterfactualFeaturePerturbation | None,
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
            cfp=cfp,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._cfp_losses = None
        if self.training:
            self._mdmb_context = {
                "targets": targets,
                "image_shapes": image_shapes,
            }
        try:
            result, losses = super().forward(features, proposals, image_shapes, targets)
            if self.training and self._cfp_losses:
                losses.update(self._cfp_losses)
            return result, losses
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
            self._mdmb_context["regression_targets"] = regression_targets
        return proposals, matched_idxs, labels, regression_targets

    def _capture_box_features(self, module, inputs, output) -> None:
        if self.training and self._mdmb_context is not None:
            self._last_box_features = output.detach()

    def _capture_box_predictions(self, module, inputs, output) -> None:
        if self._suspend_box_predictor_hook:
            return
        mdmb = self._get_mdmb()
        cfp = self._get_cfp()
        if not self.training or mdmb is None or self._mdmb_context is None:
            return
        if self._last_box_features is None:
            return
        if not isinstance(output, tuple) or len(output) != 2:
            return

        proposals = self._mdmb_context.get("sampled_proposals")
        matched_idxs = self._mdmb_context.get("matched_idxs")
        labels = self._mdmb_context.get("labels")
        regression_targets = self._mdmb_context.get("regression_targets")
        targets = self._mdmb_context.get("targets")
        image_shapes = self._mdmb_context.get("image_shapes")
        if not all(
            value is not None
            for value in (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
                targets,
                image_shapes,
            )
        ):
            return

        candidate_batches = self._collect_candidate_batches(
            mdmb=mdmb,
            proposals=proposals,  # type: ignore[arg-type]
            matched_idxs=matched_idxs,  # type: ignore[arg-type]
            labels=labels,  # type: ignore[arg-type]
            class_logits=output[0].detach(),
            box_features=self._last_box_features,
            targets=targets,  # type: ignore[arg-type]
            image_shapes=image_shapes,  # type: ignore[arg-type]
        )
        self._update_mdmb(mdmb, candidate_batches)
        if cfp is None:
            return
        self._cfp_losses = self._compute_cfp_losses(cfp, mdmb, candidate_batches)

    def _collect_candidate_batches(
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
    ) -> list[dict[str, Any]]:
        counts = [proposal.shape[0] for proposal in proposals]
        if not counts or sum(counts) == 0:
            return []

        feature_chunks = box_features.split(counts, dim=0)
        logits_chunks = class_logits.split(counts, dim=0)
        max_candidates = mdmb.config.max_entries_per_image * 4
        candidate_batches: list[dict[str, Any]] = []

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
            matched_gt_boxes = gt_boxes[nearest_gt[candidate_indices]]
            gt_classes_selected = gt_classes[candidate_indices]
            gt_scores_selected = gt_scores[candidate_indices]
            features_selected = features_per_image[candidate_indices]
            proposals_selected = proposals_per_image[candidate_indices]
            detected_selected = detected_mask[candidate_indices]
            image_id = int(target_per_image["image_id"].item())
            observations: list[MDMBObservation] = []
            track_ids: list[str] = []
            for local_index, candidate_index in enumerate(candidate_indices.tolist()):
                detected = bool(detected_mask[candidate_index].item())
                effective_iou = float(iou_max[candidate_index].item())
                if detected:
                    effective_iou = max(effective_iou, mdmb.config.iou_high)
                track_id = mdmb.make_track_id(
                    image_id=image_id,
                    region_coords=normalized_boxes[local_index],
                    source="proposal",
                    gt_class=int(gt_classes[candidate_index].item()),
                )
                track_ids.append(track_id)
                observations.append(
                    MDMBObservation(
                        image_id=image_id,
                        region_coords=normalized_boxes[local_index],
                        iou_max=effective_iou,
                        cls_score=float(gt_scores[candidate_index].item()),
                        feature_vec=features_per_image[candidate_index],
                        gt_class=int(gt_classes[candidate_index].item()),
                        detected=detected,
                        track_id=track_id,
                        source="proposal",
                    )
                )
            candidate_batches.append(
                {
                    "features": features_selected,
                    "proposals": proposals_selected,
                    "gt_boxes": matched_gt_boxes,
                    "gt_classes": gt_classes_selected,
                    "base_scores": gt_scores_selected,
                    "detected_mask": detected_selected,
                    "track_ids": track_ids,
                    "observations": observations,
                }
            )

        return candidate_batches

    def _update_mdmb(
        self,
        mdmb: MissedDetectionMemoryBank,
        candidate_batches: list[dict[str, Any]],
    ) -> None:
        observations = [
            observation
            for batch in candidate_batches
            for observation in batch["observations"]
        ]
        if observations:
            mdmb.update(observations)

    def _compute_cfp_losses(
        self,
        cfp: CounterfactualFeaturePerturbation,
        mdmb: MissedDetectionMemoryBank,
        candidate_batches: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor] | None:
        selected_features: list[torch.Tensor] = []
        selected_proposals: list[torch.Tensor] = []
        selected_gt_boxes: list[torch.Tensor] = []
        selected_gt_classes: list[torch.Tensor] = []
        selected_base_scores: list[torch.Tensor] = []

        for batch in candidate_batches:
            chronic_positions = [
                position
                for position, track_id in enumerate(batch["track_ids"])
                if not bool(batch["detected_mask"][position].item())
                and (entry := mdmb.get(track_id)) is not None
                and entry.is_chronic_miss(mdmb.config)
            ]
            if not chronic_positions:
                continue

            device = batch["features"].device
            chronic_indices = torch.tensor(
                chronic_positions,
                device=device,
                dtype=torch.long,
            )
            selected_features.append(batch["features"][chronic_indices])
            selected_proposals.append(batch["proposals"][chronic_indices])
            selected_gt_boxes.append(batch["gt_boxes"][chronic_indices])
            selected_gt_classes.append(batch["gt_classes"][chronic_indices].to(dtype=torch.int64))
            selected_base_scores.append(batch["base_scores"][chronic_indices])

        if not selected_features:
            return None

        chronic_features = torch.cat(selected_features, dim=0)
        chronic_proposals = torch.cat(selected_proposals, dim=0)
        chronic_gt_boxes = torch.cat(selected_gt_boxes, dim=0)
        chronic_gt_classes = torch.cat(selected_gt_classes, dim=0)
        chronic_base_scores = torch.cat(selected_base_scores, dim=0)

        cfp_output = cfp(chronic_features)
        try:
            self._suspend_box_predictor_hook = True
            cf_class_logits, cf_box_regression = self.box_predictor(cfp_output.perturbed_features)
        finally:
            self._suspend_box_predictor_hook = False

        regression_targets = self.box_coder.encode([chronic_gt_boxes], [chronic_proposals])[0]
        loss_classifier, loss_box_reg = fastrcnn_loss(
            cf_class_logits,
            cf_box_regression,
            [chronic_gt_classes],
            [regression_targets],
        )
        proposal_indices = torch.arange(
            chronic_gt_classes.shape[0],
            device=chronic_gt_classes.device,
        )
        cf_scores = torch.softmax(cf_class_logits, dim=-1)[proposal_indices, chronic_gt_classes]
        return compute_cfp_loss_dict(
            detection_loss={
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            },
            delta=cfp_output,
            config=cfp.config,
            cf_scores=cf_scores,
            base_scores=chronic_base_scores,
        )

    def _get_mdmb(self) -> MissedDetectionMemoryBank | None:
        if self._mdmb_ref is None:
            return None
        return self._mdmb_ref()

    def _get_cfp(self) -> CounterfactualFeaturePerturbation | None:
        if self._cfp_ref is None:
            return None
        return self._cfp_ref()


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
        cfp: CounterfactualFeaturePerturbation | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cfp = cfp
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

        if mdmb is not None or cfp is not None:
            self.model.roi_heads = MDMBRoIHeads.from_existing(self.model.roi_heads, mdmb, cfp)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        cfp: CounterfactualFeaturePerturbation | None = None,
        **kwargs,
    ) -> "FasterRCNNWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            mdmb=mdmb,
            cfp=cfp,
            **kwargs,
        )
