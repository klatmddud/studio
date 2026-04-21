from __future__ import annotations

"""FCOS wrapper: YAML config -> torchvision FCOS."""

import math
import weakref
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FCOS
from torchvision.ops import boxes as box_ops
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from modules.nn import (
    CandidateDensifier,
    CanonicalCandidate,
    DensePlan,
    FailureAwareAssignmentRepair,
    FailureAwareNegativeGradientShielding,
    MDMBPlus,
    MissAwareRankingCalibration,
    PerImageCandidateSummary,
    RepairPlan,
)
from modules.nn.mdmb import MissedDetectionMemoryBank, normalize_xyxy_boxes

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


def _is_replay_target(target: dict[str, object]) -> bool:
    raw = target.get("is_replay", False)
    if isinstance(raw, torch.Tensor):
        if raw.numel() == 0:
            return False
        return bool(raw.detach().flatten()[0].item())
    return bool(raw)


def _has_replay_box_weights(targets: list[dict[str, torch.Tensor]] | None) -> bool:
    if not targets:
        return False
    for target in targets:
        weights = target.get("replay_box_weights")
        if not isinstance(weights, torch.Tensor) or weights.numel() == 0:
            continue
        if bool((weights.detach().to(dtype=torch.float32) != 1.0).any().item()):
            return True
    return False


class MDMBFCOS(FCOS):
    """
    FCOS variant with MDMB/MDMB++ memory updates and optional UMR training modules.

    During training the wrapper can repair assignments, add ranking/dense
    auxiliary losses, and apply replay-aware per-GT loss weights. After
    optimizer.step(), it runs an extra no-grad inference pass to refresh MDMB
    and MDMB++ state from final detections.
    """

    def __init__(
        self,
        *args,
        mdmb: MissedDetectionMemoryBank | None = None,
        mdmbpp: MDMBPlus | None = None,
        faar: FailureAwareAssignmentRepair | None = None,
        fang: FailureAwareNegativeGradientShielding | None = None,
        marc: MissAwareRankingCalibration | None = None,
        candidate_densifier: CandidateDensifier | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mdmb_ref = weakref.ref(mdmb) if mdmb is not None else None
        self._mdmbpp_ref = weakref.ref(mdmbpp) if mdmbpp is not None else None
        self._faar_ref = weakref.ref(faar) if faar is not None else None
        self._fang_ref = weakref.ref(fang) if fang is not None else None
        self._marc_ref = weakref.ref(marc) if marc is not None else None
        self._candidate_densifier_ref = (
            weakref.ref(candidate_densifier) if candidate_densifier is not None else None
        )

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            value = img.shape[-2:]
            torch._assert(
                len(value) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((value[0], value[1]))

        images, targets = self.transform(images, targets)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: list[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        f"All bounding boxes should have positive height and width. Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())

        head_outputs = self.head(feature_list)
        anchors = self.anchor_generator(images, feature_list)
        num_anchors_per_level = [x.size(2) * x.size(3) for x in feature_list]

        losses: dict[str, torch.Tensor] = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            matched_idxs = self._match_anchors_to_targets(targets, anchors, num_anchors_per_level)
            mdmbpp = self._get_mdmbpp()
            faar = self._get_faar()
            if faar is not None and faar.should_apply(mdmbpp=mdmbpp):
                repair_plan = faar.plan(
                    mdmbpp=mdmbpp,
                    targets=targets,
                    image_shapes=images.image_sizes,
                )
                (
                    matched_idxs,
                    repaired_points,
                    skipped_no_candidate_points,
                    skipped_existing_positive,
                ) = self._repair_fcos_assignments(
                    repair_plan=repair_plan,
                    targets=targets,
                    anchors=anchors,
                    matched_idxs=matched_idxs,
                    num_anchors_per_level=num_anchors_per_level,
                    faar=faar,
                )
                faar.record_repair_step(
                    repair_plan=repair_plan,
                    repaired_points=repaired_points,
                    skipped_no_candidate_points=skipped_no_candidate_points,
                    skipped_existing_positive=skipped_existing_positive,
                )
            fang = self._get_fang()
            fang_class_weights: list[torch.Tensor] | None = None
            use_fang = False
            if fang is not None and fang.should_apply(mdmbpp=mdmbpp):
                shield_plan = fang.plan(
                    mdmbpp=mdmbpp,
                    targets=targets,
                    image_shapes=images.image_sizes,
                )
                (
                    fang_class_weights,
                    shield_points,
                    skipped_no_candidate_points,
                    shield_weight_sum,
                ) = fang.compute_class_weights(
                    shield_plan=shield_plan,
                    targets=targets,
                    anchors=anchors,
                    matched_idxs=matched_idxs,
                    num_classes=int(head_outputs["cls_logits"].shape[-1]),
                )
                fang.record_shield_step(
                    shield_plan=shield_plan,
                    shield_points=shield_points,
                    skipped_no_candidate_points=skipped_no_candidate_points,
                    shield_weight_sum=shield_weight_sum,
                )
                use_fang = True
            use_weighted = _has_replay_box_weights(targets)
            if not use_weighted and not use_fang:
                losses = self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
            else:
                losses = self._compute_replay_weighted_loss_dict(
                    targets=targets,
                    head_outputs=head_outputs,
                    anchors=anchors,
                    matched_idxs=matched_idxs,
                    class_loss_weights=fang_class_weights,
                )

            marc = self._get_marc()
            if marc is not None and marc.should_apply(mdmbpp=mdmbpp):
                ranking_plan = marc.plan(
                    mdmbpp=mdmbpp,
                    targets=targets,
                    image_shapes=images.image_sizes,
                )
                (
                    marc_loss,
                    rank_losses,
                    rank_negatives,
                    skipped_no_positive,
                    skipped_no_negative,
                ) = marc.compute_loss(
                    ranking_plan=ranking_plan,
                    targets=targets,
                    head_outputs=head_outputs,
                    anchors=anchors,
                    image_shapes=images.image_sizes,
                    decode_boxes_fn=self._decode_boxes,
                )
                marc.record_rank_step(
                    ranking_plan=ranking_plan,
                    rank_losses=rank_losses,
                    rank_negatives=rank_negatives,
                    loss=marc_loss if rank_losses > 0 else None,
                    skipped_no_positive=skipped_no_positive,
                    skipped_no_negative=skipped_no_negative,
                )
                if rank_losses > 0:
                    losses = dict(losses)
                    losses["marc"] = marc_loss

            candidate_densifier = self._get_candidate_densifier()
            if candidate_densifier is not None and candidate_densifier.should_apply(mdmbpp=mdmbpp):
                dense_plan = candidate_densifier.plan(
                    mdmbpp=mdmbpp,
                    targets=targets,
                    image_shapes=images.image_sizes,
                )
                dense_loss, dense_points = self._compute_candidate_dense_loss(
                    dense_plan=dense_plan,
                    targets=targets,
                    head_outputs=head_outputs,
                    anchors=anchors,
                    matched_idxs=matched_idxs,
                    image_shapes=images.image_sizes,
                    candidate_densifier=candidate_densifier,
                )
                candidate_densifier.record_dense_step(
                    num_targets=len(dense_plan),
                    num_points=dense_points,
                    loss=dense_loss if dense_points > 0 else None,
                )
                if dense_points > 0:
                    losses = dict(losses)
                    losses["candidate_dense"] = dense_loss

        else:
            split_head_outputs: dict[str, list[torch.Tensor]] = {}
            for key in head_outputs:
                split_head_outputs[key] = list(head_outputs[key].split(num_anchors_per_level, dim=1))
            split_anchors = [list(anchor.split(num_anchors_per_level)) for anchor in anchors]
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                import warnings

                warnings.warn("FCOS always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

    def _compute_replay_weighted_loss_dict(
        self,
        *,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        class_loss_weights: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_template = head_outputs["cls_logits"]
        cls_sum = cls_template.new_zeros(())
        reg_sum = cls_template.new_zeros(())
        ctr_sum = cls_template.new_zeros(())
        total_pos = 0

        for image_index, target in enumerate(targets):
            raw = self._compute_raw_losses_for_image(
                image_index=image_index,
                target=target,
                head_outputs=head_outputs,
                anchors_per_image=anchors[image_index],
                matched_idxs_per_image=matched_idxs[image_index],
            )
            assignments = raw["assignments"]
            num_points = int(assignments.numel())
            cls_weights = torch.ones(num_points, dtype=torch.float32, device=assignments.device)
            reg_weights = torch.ones(num_points, dtype=torch.float32, device=assignments.device)
            ctr_weights = reg_weights.clone()

            pos_mask = assignments >= 0
            total_pos += int(pos_mask.sum().item())

            cls_weights = self._apply_replay_box_weights(
                base_weights=cls_weights,
                target=target,
                assignments=assignments,
                key="replay_cls_box_weights",
            )
            reg_weights = self._apply_replay_box_weights(
                base_weights=reg_weights,
                target=target,
                assignments=assignments,
                key="replay_reg_box_weights",
            )
            ctr_weights = self._apply_replay_box_weights(
                base_weights=ctr_weights,
                target=target,
                assignments=assignments,
                key="replay_ctr_box_weights",
            )

            cls_losses_by_class = raw["cls_losses_by_class"]
            cls_weights = cls_weights.to(dtype=cls_losses_by_class.dtype)
            reg_weights = reg_weights.to(dtype=cls_losses_by_class.dtype)
            ctr_weights = ctr_weights.to(dtype=cls_losses_by_class.dtype)

            cls_weight_matrix = cls_weights.unsqueeze(1)
            if class_loss_weights is not None:
                image_class_weights = class_loss_weights[image_index].to(
                    device=cls_losses_by_class.device,
                    dtype=cls_losses_by_class.dtype,
                )
                if image_class_weights.shape != cls_losses_by_class.shape:
                    raise ValueError(
                        "FANG class weights must match FCOS class loss shape. "
                        f"Got {tuple(image_class_weights.shape)} and "
                        f"{tuple(cls_losses_by_class.shape)}."
                    )
                cls_weight_matrix = cls_weight_matrix * image_class_weights

            cls_sum = cls_sum + (cls_losses_by_class * cls_weight_matrix).sum()
            if bool(pos_mask.any().item()):
                reg_sum = reg_sum + (raw["reg_losses"][pos_mask] * reg_weights[pos_mask]).sum()
                ctr_sum = ctr_sum + (raw["ctr_losses"][pos_mask] * ctr_weights[pos_mask]).sum()

        normalizer = cls_sum.new_tensor(float(max(total_pos, 1)))
        return {
            "classification": cls_sum / normalizer,
            "bbox_regression": reg_sum / normalizer,
            "bbox_ctrness": ctr_sum / normalizer,
        }

    def _apply_replay_box_weights(
        self,
        *,
        base_weights: torch.Tensor,
        target: dict[str, torch.Tensor],
        assignments: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        box_weights = target.get(key)
        if not isinstance(box_weights, torch.Tensor):
            box_weights = target.get("replay_box_weights")
        if not isinstance(box_weights, torch.Tensor) or box_weights.numel() == 0:
            return base_weights

        pos_mask = assignments >= 0
        if not bool(pos_mask.any().item()):
            return base_weights

        result = base_weights.clone()
        gt_indices = assignments[pos_mask].to(dtype=torch.long)
        valid = gt_indices < box_weights.numel()
        if not bool(valid.any().item()):
            return result

        pos_indices = torch.where(pos_mask)[0][valid]
        selected_gt = gt_indices[valid]
        replay_weights = box_weights.to(device=result.device, dtype=result.dtype)[selected_gt]
        result[pos_indices] = result[pos_indices] * replay_weights
        return result

    def _compute_raw_losses_for_image(
        self,
        *,
        image_index: int,
        target: dict[str, torch.Tensor],
        head_outputs: dict[str, torch.Tensor],
        anchors_per_image: torch.Tensor,
        matched_idxs_per_image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls_logits = head_outputs["cls_logits"][image_index]
        bbox_regression = head_outputs["bbox_regression"][image_index]
        bbox_ctrness = head_outputs["bbox_ctrness"][image_index].flatten()

        if cls_logits.ndim != 2:
            raise ValueError(
                "FCOS cls_logits must have shape [N_points, num_classes] per image. "
                f"Got {tuple(cls_logits.shape)}."
            )
        if bbox_regression.ndim != 2 or bbox_regression.shape[-1] != 4:
            raise ValueError(
                "FCOS bbox_regression must have shape [N_points, 4] per image. "
                f"Got {tuple(bbox_regression.shape)}."
            )
        if bbox_regression.shape[0] != cls_logits.shape[0]:
            raise ValueError("FCOS bbox_regression and cls_logits must share N_points.")
        if bbox_ctrness.shape[0] != cls_logits.shape[0]:
            raise ValueError("FCOS bbox_ctrness and cls_logits must share N_points.")
        if anchors_per_image.shape[0] != cls_logits.shape[0]:
            raise ValueError("FCOS anchors and logits must share N_points.")
        if matched_idxs_per_image.shape[0] != cls_logits.shape[0]:
            raise ValueError("FCOS matched indices and logits must share N_points.")

        gt_boxes = target["boxes"]
        gt_labels = target["labels"].to(dtype=torch.int64)
        gt_classes_targets = torch.zeros_like(cls_logits)

        raw_reg = cls_logits.new_zeros((cls_logits.shape[0],))
        raw_ctr = cls_logits.new_zeros((cls_logits.shape[0],))

        pos_mask = matched_idxs_per_image >= 0
        if bool(pos_mask.any().item()):
            matched_gt_indices = matched_idxs_per_image[pos_mask]
            matched_gt_labels = gt_labels[matched_gt_indices]
            if bool((matched_gt_labels >= cls_logits.shape[1]).any().item()):
                raise ValueError(
                    "FCOS target label exceeds classifier dimension. "
                    f"Max label={int(matched_gt_labels.max().item())}, "
                    f"num_classes={cls_logits.shape[1]}."
                )

            gt_classes_targets[pos_mask, matched_gt_labels] = 1.0

            matched_gt_boxes = gt_boxes[matched_gt_indices]
            pred_boxes = self._decode_boxes(
                box_regression=bbox_regression[pos_mask],
                anchors=anchors_per_image[pos_mask],
            )
            giou = box_ops.generalized_box_iou(pred_boxes, matched_gt_boxes)
            raw_reg[pos_mask] = 1.0 - torch.diagonal(giou)

            ctr_targets = self._compute_centerness_targets(
                anchors=anchors_per_image[pos_mask],
                gt_boxes=matched_gt_boxes,
            )
            raw_ctr[pos_mask] = F.binary_cross_entropy_with_logits(
                bbox_ctrness[pos_mask],
                ctr_targets,
                reduction="none",
            )

        raw_cls_by_class = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="none")
        raw_cls = raw_cls_by_class.sum(dim=-1)
        return {
            "cls_losses_by_class": raw_cls_by_class,
            "cls_losses": raw_cls,
            "reg_losses": raw_reg,
            "ctr_losses": raw_ctr,
            "assignments": matched_idxs_per_image,
            "point_boxes": anchors_per_image,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        }

    def _compute_candidate_dense_loss(
        self,
        *,
        dense_plan: DensePlan,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        candidate_densifier: CandidateDensifier,
    ) -> tuple[torch.Tensor, int]:
        cls_template = head_outputs["cls_logits"]
        dense_sum = cls_template.new_zeros(())
        dense_points = 0

        for image_index, target in enumerate(targets):
            image_id = target.get(
                "image_id",
                torch.tensor(image_index, device=anchors[image_index].device),
            )
            dense_targets = dense_plan.for_image(image_id)
            if not dense_targets:
                continue

            cls_logits = head_outputs["cls_logits"][image_index]
            bbox_regression = head_outputs["bbox_regression"][image_index]
            bbox_ctrness = head_outputs["bbox_ctrness"][image_index].flatten()
            anchors_per_image = anchors[image_index]
            assignments = matched_idxs[image_index]
            gt_boxes = target["boxes"]
            gt_labels = target["labels"].to(device=cls_logits.device, dtype=torch.int64)

            used_points = torch.zeros(
                anchors_per_image.shape[0],
                dtype=torch.bool,
                device=anchors_per_image.device,
            )
            selected_points: list[torch.Tensor] = []
            selected_gt_indices: list[torch.Tensor] = []

            for dense_target in dense_targets:
                if dense_target.gt_index < 0 or dense_target.gt_index >= gt_boxes.shape[0]:
                    continue
                point_indices = self._select_candidate_dense_points(
                    dense_target=dense_target,
                    anchors_per_image=anchors_per_image,
                    assignments=assignments,
                    used_points=used_points,
                    candidate_densifier=candidate_densifier,
                )
                if point_indices.numel() == 0:
                    continue
                used_points[point_indices] = True
                selected_points.append(point_indices)
                selected_gt_indices.append(
                    torch.full(
                        (point_indices.numel(),),
                        int(dense_target.gt_index),
                        dtype=torch.long,
                        device=point_indices.device,
                    )
                )

            if not selected_points:
                continue

            point_indices = torch.cat(selected_points, dim=0)
            gt_indices = torch.cat(selected_gt_indices, dim=0)
            matched_gt_labels = gt_labels[gt_indices]
            valid_label_mask = (
                (matched_gt_labels >= 0)
                & (matched_gt_labels < cls_logits.shape[1])
            )
            if not bool(valid_label_mask.any().item()):
                continue

            point_indices = point_indices[valid_label_mask]
            gt_indices = gt_indices[valid_label_mask]
            matched_gt_labels = matched_gt_labels[valid_label_mask]
            matched_gt_boxes = gt_boxes[gt_indices].to(
                device=cls_logits.device,
                dtype=anchors_per_image.dtype,
            )

            cls_targets = torch.zeros(
                (point_indices.numel(), cls_logits.shape[1]),
                dtype=cls_logits.dtype,
                device=cls_logits.device,
            )
            cls_targets[
                torch.arange(point_indices.numel(), device=cls_logits.device),
                matched_gt_labels,
            ] = 1.0
            cls_losses = sigmoid_focal_loss(
                cls_logits[point_indices],
                cls_targets,
                reduction="none",
            ).sum(dim=-1)

            pred_boxes = self._decode_boxes(
                box_regression=bbox_regression[point_indices],
                anchors=anchors_per_image[point_indices],
            )
            pred_boxes = box_ops.clip_boxes_to_image(pred_boxes, image_shapes[image_index])
            giou = box_ops.generalized_box_iou(pred_boxes, matched_gt_boxes)
            reg_losses = 1.0 - torch.diagonal(giou)

            ctr_targets = self._compute_centerness_targets(
                anchors=anchors_per_image[point_indices],
                gt_boxes=matched_gt_boxes,
            )
            ctr_losses = F.binary_cross_entropy_with_logits(
                bbox_ctrness[point_indices],
                ctr_targets,
                reduction="none",
            )

            dense_sum = dense_sum + cls_losses.sum() + reg_losses.sum() + ctr_losses.sum()
            dense_points += int(point_indices.numel())

        if dense_points <= 0:
            return dense_sum, 0

        normalizer = dense_sum.new_tensor(float(dense_points))
        loss = dense_sum / normalizer
        return loss * float(candidate_densifier.loss_weight()), dense_points

    def _select_candidate_dense_points(
        self,
        *,
        dense_target,
        anchors_per_image: torch.Tensor,
        assignments: torch.Tensor,
        used_points: torch.Tensor,
        candidate_densifier: CandidateDensifier,
    ) -> torch.Tensor:
        budget = int(dense_target.budget)
        if budget <= 0 or anchors_per_image.numel() == 0:
            return anchors_per_image.new_zeros((0,), dtype=torch.long)

        gt_box = dense_target.bbox.to(
            device=anchors_per_image.device,
            dtype=anchors_per_image.dtype,
        )
        gt_center = (gt_box[:2] + gt_box[2:]) * 0.5
        gt_size = (gt_box[2:] - gt_box[:2]).clamp(min=1.0)
        scale = float(candidate_densifier.region_scale_for(dense_target.severity))
        half_size = (gt_size * scale * 0.5).clamp(min=1.0)
        region_min = gt_center - half_size
        region_max = gt_center + half_size

        centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) * 0.5
        candidate_mask = (
            (centers[:, 0] >= region_min[0])
            & (centers[:, 0] <= region_max[0])
            & (centers[:, 1] >= region_min[1])
            & (centers[:, 1] <= region_max[1])
            & (~used_points)
        )
        if candidate_densifier.config.require_unassigned_points:
            candidate_mask &= assignments < 0

        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return candidate_indices

        normalized_offsets = (centers[candidate_indices] - gt_center).abs() / half_size
        distances = normalized_offsets.max(dim=1).values
        count = min(budget, int(candidate_indices.numel()))
        nearest = torch.argsort(distances)[:count]
        return candidate_indices[nearest]

    def _repair_fcos_assignments(
        self,
        *,
        repair_plan: RepairPlan,
        targets: list[dict[str, torch.Tensor]],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        num_anchors_per_level: list[int],
        faar: FailureAwareAssignmentRepair,
    ) -> tuple[list[torch.Tensor], int, int, int]:
        if len(repair_plan) == 0:
            return matched_idxs, 0, 0, 0

        repaired = [assignments.clone() for assignments in matched_idxs]
        repaired_points = 0
        skipped_no_candidate_points = 0
        skipped_existing_positive = 0

        for image_index, target in enumerate(targets):
            image_id = target.get(
                "image_id",
                torch.tensor(image_index, device=anchors[image_index].device),
            )
            repair_targets = repair_plan.for_image(image_id)
            if not repair_targets:
                continue

            anchors_per_image = anchors[image_index]
            assignments = repaired[image_index]
            gt_boxes = target["boxes"].to(device=anchors_per_image.device)
            used_points = torch.zeros(
                anchors_per_image.shape[0],
                dtype=torch.bool,
                device=anchors_per_image.device,
            )

            for repair_target in repair_targets:
                if repair_target.gt_index < 0 or repair_target.gt_index >= gt_boxes.shape[0]:
                    skipped_no_candidate_points += 1
                    continue

                point_indices, positive_skip_count = self._select_faar_repair_points(
                    repair_target=repair_target,
                    anchors_per_image=anchors_per_image,
                    assignments=assignments,
                    used_points=used_points,
                    gt_boxes=gt_boxes,
                    num_anchors_per_level=num_anchors_per_level,
                    faar=faar,
                )
                skipped_existing_positive += positive_skip_count
                if point_indices.numel() == 0:
                    skipped_no_candidate_points += 1
                    continue

                assignments[point_indices] = int(repair_target.gt_index)
                used_points[point_indices] = True
                repaired_points += int(point_indices.numel())

        return repaired, repaired_points, skipped_no_candidate_points, skipped_existing_positive

    def _select_faar_repair_points(
        self,
        *,
        repair_target,
        anchors_per_image: torch.Tensor,
        assignments: torch.Tensor,
        used_points: torch.Tensor,
        gt_boxes: torch.Tensor,
        num_anchors_per_level: list[int],
        faar: FailureAwareAssignmentRepair,
    ) -> tuple[torch.Tensor, int]:
        budget = int(repair_target.budget)
        if budget <= 0 or anchors_per_image.numel() == 0:
            return anchors_per_image.new_zeros((0,), dtype=torch.long), 0

        gt_box = repair_target.bbox.to(
            device=anchors_per_image.device,
            dtype=anchors_per_image.dtype,
        )
        gt_center = (gt_box[:2] + gt_box[2:]) * 0.5
        gt_size = (gt_box[2:] - gt_box[:2]).clamp(min=1.0)
        scale = float(faar.region_scale_for(repair_target.severity))
        half_size = (gt_size * scale * 0.5).clamp(min=1.0)
        region_min = gt_center - half_size
        region_max = gt_center + half_size

        centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) * 0.5
        inside_gt_mask = (
            (centers[:, 0] > gt_box[0])
            & (centers[:, 0] < gt_box[2])
            & (centers[:, 1] > gt_box[1])
            & (centers[:, 1] < gt_box[3])
        )
        region_mask = (
            (centers[:, 0] >= region_min[0])
            & (centers[:, 0] <= region_max[0])
            & (centers[:, 1] >= region_min[1])
            & (centers[:, 1] <= region_max[1])
        )
        candidate_scope_mask = region_mask & inside_gt_mask
        availability_mask, skipped_existing_positive = self._faar_availability_mask(
            repair_target=repair_target,
            assignments=assignments,
            used_points=used_points,
            gt_boxes=gt_boxes,
            count_scope_mask=candidate_scope_mask,
            faar=faar,
        )

        scale_mask = torch.ones_like(region_mask)
        if faar.config.respect_fcos_scale_range:
            scale_mask = self._faar_scale_mask(
                gt_box=gt_box,
                anchors_per_image=anchors_per_image,
                num_anchors_per_level=num_anchors_per_level,
            )

        candidate_mask = candidate_scope_mask & availability_mask & scale_mask
        point_indices = self._take_nearest_faar_points(
            candidate_mask=candidate_mask,
            centers=centers,
            gt_center=gt_center,
            half_size=half_size,
            budget=budget,
        )
        if point_indices.numel() > 0:
            return point_indices, skipped_existing_positive

        if faar.config.allow_adjacent_levels and faar.config.respect_fcos_scale_range:
            adjacent_mask = self._faar_adjacent_level_mask(
                scale_mask=scale_mask,
                gt_box=gt_box,
                anchors_per_image=anchors_per_image,
                num_anchors_per_level=num_anchors_per_level,
            )
            candidate_mask = candidate_scope_mask & availability_mask & adjacent_mask
            point_indices = self._take_nearest_faar_points(
                candidate_mask=candidate_mask,
                centers=centers,
                gt_center=gt_center,
                half_size=half_size,
                budget=budget,
            )
            if point_indices.numel() > 0:
                return point_indices, skipped_existing_positive

        if faar.config.allow_nearest_center_fallback:
            candidate_mask = inside_gt_mask & availability_mask
            point_indices = self._take_nearest_faar_points(
                candidate_mask=candidate_mask,
                centers=centers,
                gt_center=gt_center,
                half_size=half_size,
                budget=budget,
            )
            if point_indices.numel() > 0:
                return point_indices, skipped_existing_positive

        return anchors_per_image.new_zeros((0,), dtype=torch.long), skipped_existing_positive

    def _faar_availability_mask(
        self,
        *,
        repair_target,
        assignments: torch.Tensor,
        used_points: torch.Tensor,
        gt_boxes: torch.Tensor,
        count_scope_mask: torch.Tensor,
        faar: FailureAwareAssignmentRepair,
    ) -> tuple[torch.Tensor, int]:
        base_mask = ~used_points
        positive_mask = assignments >= 0
        skipped_existing_positive = int((base_mask & count_scope_mask & positive_mask).sum().item())

        if faar.config.require_unassigned_points or not faar.config.allow_positive_reassignment:
            return base_mask & (assignments < 0), skipped_existing_positive

        valid_positive = positive_mask & (assignments < gt_boxes.shape[0])
        same_gt = valid_positive & (assignments == int(repair_target.gt_index))
        reassignable_positive = torch.zeros_like(base_mask)
        if bool(valid_positive.any().item()):
            positive_indices = torch.nonzero(valid_positive & (~same_gt), as_tuple=False).flatten()
            if positive_indices.numel() > 0:
                assigned_gt = assignments[positive_indices].to(dtype=torch.long)
                target_box = repair_target.bbox.to(
                    device=gt_boxes.device,
                    dtype=gt_boxes.dtype,
                ).reshape(1, 4)
                ious = box_ops.box_iou(target_box, gt_boxes[assigned_gt])[0]
                reassignable = ious < float(faar.config.protect_existing_positive_iou)
                reassignable_positive[positive_indices[reassignable]] = True

        protected_positive = positive_mask & (~reassignable_positive)
        skipped_existing_positive = int(
            (base_mask & count_scope_mask & protected_positive).sum().item()
        )
        return base_mask & ((assignments < 0) | reassignable_positive), skipped_existing_positive

    def _faar_scale_mask(
        self,
        *,
        gt_box: torch.Tensor,
        anchors_per_image: torch.Tensor,
        num_anchors_per_level: list[int],
    ) -> torch.Tensor:
        centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) * 0.5
        x, y = centers.unbind(dim=1)
        x0, y0, x1, y1 = gt_box
        distances = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=1)
        max_dist = distances.max(dim=1).values

        anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]
        lower_bound = anchor_sizes * 4
        if num_anchors_per_level:
            lower_bound[: num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        if num_anchors_per_level:
            upper_bound[-num_anchors_per_level[-1] :] = float("inf")
        return (max_dist > lower_bound) & (max_dist < upper_bound)

    def _faar_adjacent_level_mask(
        self,
        *,
        scale_mask: torch.Tensor,
        gt_box: torch.Tensor,
        anchors_per_image: torch.Tensor,
        num_anchors_per_level: list[int],
    ) -> torch.Tensor:
        level_ids = self._faar_level_ids(
            num_anchors_per_level=num_anchors_per_level,
            device=anchors_per_image.device,
        )
        num_levels = len(num_anchors_per_level)
        if num_levels == 0:
            return torch.ones_like(scale_mask)

        valid_levels = level_ids[scale_mask]
        if valid_levels.numel() > 0:
            allowed_levels = {
                int(level.item()) + offset
                for level in valid_levels.unique()
                for offset in (-1, 0, 1)
                if 0 <= int(level.item()) + offset < num_levels
            }
        else:
            anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]
            level_sizes = []
            start = 0
            for count in num_anchors_per_level:
                end = start + int(count)
                if end > start:
                    level_sizes.append(float(anchor_sizes[start:end].median().detach().item()))
                else:
                    level_sizes.append(1.0)
                start = end
            gt_extent = float((gt_box[2:] - gt_box[:2]).clamp(min=1.0).max().detach().item())
            target_level = min(
                range(num_levels),
                key=lambda idx: abs(math.log(max(level_sizes[idx] * 6.0, 1e-6) / gt_extent)),
            )
            allowed_levels = {
                level
                for level in (target_level - 1, target_level, target_level + 1)
                if 0 <= level < num_levels
            }

        allowed = torch.zeros_like(scale_mask)
        for level in allowed_levels:
            allowed |= level_ids == int(level)
        return allowed

    def _faar_level_ids(
        self,
        *,
        num_anchors_per_level: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        if not num_anchors_per_level:
            return torch.zeros((0,), dtype=torch.long, device=device)
        return torch.cat(
            [
                torch.full((int(count),), level, dtype=torch.long, device=device)
                for level, count in enumerate(num_anchors_per_level)
            ],
            dim=0,
        )

    def _take_nearest_faar_points(
        self,
        *,
        candidate_mask: torch.Tensor,
        centers: torch.Tensor,
        gt_center: torch.Tensor,
        half_size: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return candidate_indices
        normalized_offsets = (centers[candidate_indices] - gt_center).abs() / half_size
        distances = normalized_offsets.max(dim=1).values
        count = min(int(budget), int(candidate_indices.numel()))
        nearest = torch.argsort(distances)[:count]
        return candidate_indices[nearest]

    def _compute_centerness_targets(
        self,
        *,
        anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        if anchors.numel() == 0:
            return anchors.new_zeros((0,))

        centers = (anchors[:, :2] + anchors[:, 2:]) * 0.5
        left = centers[:, 0] - gt_boxes[:, 0]
        top = centers[:, 1] - gt_boxes[:, 1]
        right = gt_boxes[:, 2] - centers[:, 0]
        bottom = gt_boxes[:, 3] - centers[:, 1]

        eps = torch.finfo(gt_boxes.dtype).eps if gt_boxes.is_floating_point() else 1e-6
        lr = torch.minimum(left, right).clamp(min=0.0) / torch.maximum(left, right).clamp(min=eps)
        tb = torch.minimum(top, bottom).clamp(min=0.0) / torch.maximum(top, bottom).clamp(min=eps)
        return torch.sqrt((lr * tb).clamp(min=0.0))

    def _decode_boxes(
        self,
        *,
        box_regression: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        box_coder = getattr(self, "box_coder", None)
        if box_coder is not None:
            decode_single = getattr(box_coder, "decode_single", None)
            if callable(decode_single):
                try:
                    decoded = decode_single(box_regression, anchors)
                    if isinstance(decoded, torch.Tensor):
                        return decoded
                except TypeError:
                    pass

            decode = getattr(box_coder, "decode", None)
            if callable(decode):
                try:
                    decoded = decode(box_regression, anchors)
                    if isinstance(decoded, torch.Tensor):
                        if decoded.ndim == 3 and decoded.shape[0] == 1:
                            return decoded[0]
                        return decoded
                except TypeError:
                    pass

        centers = (anchors[:, :2] + anchors[:, 2:]) * 0.5
        sizes = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-6)
        left = box_regression[:, 0] * sizes[:, 0]
        top = box_regression[:, 1] * sizes[:, 1]
        right = box_regression[:, 2] * sizes[:, 0]
        bottom = box_regression[:, 3] * sizes[:, 1]
        return torch.stack(
            (
                centers[:, 0] - left,
                centers[:, 1] - top,
                centers[:, 0] + right,
                centers[:, 1] + bottom,
            ),
            dim=-1,
        )

    def _match_anchors_to_targets(
        self,
        targets: list[dict[str, torch.Tensor]],
        anchors: list[torch.Tensor],
        num_anchors_per_level: list[int],
    ) -> list[torch.Tensor]:
        matched_idxs: list[torch.Tensor] = []
        for anchors_per_image, targets_per_image in zip(anchors, targets, strict=True):
            if targets_per_image["boxes"].numel() == 0:
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
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]

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

    @torch.no_grad()
    def _run_post_step_inference(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
    ) -> dict[str, object]:
        original_image_sizes = [tuple(int(dim) for dim in image.shape[-2:]) for image in images]
        cloned_targets = (
            None
            if targets is None
            else [{key: value for key, value in target.items()} for target in targets]
        )
        transformed_images, transformed_targets = self.transform(images, cloned_targets)
        features = self.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        feature_list = list(features.values())
        head_outputs = self.head(feature_list)
        anchors = self.anchor_generator(transformed_images, feature_list)
        num_anchors_per_level = [feature.size(2) * feature.size(3) for feature in feature_list]

        split_head_outputs: dict[str, list[torch.Tensor]] = {}
        for key in head_outputs:
            split_head_outputs[key] = list(head_outputs[key].split(num_anchors_per_level, dim=1))
        split_anchors = [list(anchor.split(num_anchors_per_level)) for anchor in anchors]

        detections = self.postprocess_detections(
            split_head_outputs,
            split_anchors,
            transformed_images.image_sizes,
        )
        detections = self.transform.postprocess(
            detections,
            transformed_images.image_sizes,
            original_image_sizes,
        )

        return {
            "detections": detections,
            "split_head_outputs": split_head_outputs,
            "split_anchors": split_anchors,
            "transformed_targets": transformed_targets,
            "transformed_image_sizes": transformed_images.image_sizes,
        }

    @torch.no_grad()
    def flush_post_step_updates(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        *,
        epoch: int | None = None,
    ) -> None:
        mdmb = self._get_mdmb()
        mdmbpp = self._get_mdmbpp()
        if not targets:
            return

        non_replay = [
            (image, target)
            for image, target in zip(images, targets, strict=True)
            if not _is_replay_target(target)
        ]
        if not non_replay:
            return
        images = [image for image, _ in non_replay]
        targets = [target for _, target in non_replay]

        should_mdmb = mdmb is not None and mdmb.should_update(epoch=epoch)
        should_mdmbpp = mdmbpp is not None and mdmbpp.should_update(epoch=epoch)
        if not (should_mdmb or should_mdmbpp):
            return

        image_ids = [
            target.get("image_id", torch.tensor(index, device=images[index].device))
            for index, target in enumerate(targets)
        ]
        gt_boxes_list = [target["boxes"] for target in targets]
        gt_labels_list = [target["labels"] for target in targets]
        image_shapes = [tuple(int(dim) for dim in image.shape[-2:]) for image in images]

        was_training = self.training
        try:
            self.eval()
            post_step = self._run_post_step_inference(images, targets)
        finally:
            if was_training:
                self.train()

        detections = post_step["detections"]
        pred_boxes_list = [
            detection["boxes"] if "boxes" in detection else image.new_zeros((0, 4))
            for detection, image in zip(detections, images, strict=True)
        ]
        pred_labels_list = [
            detection["labels"] if "labels" in detection else image.new_zeros((0,), dtype=torch.int64)
            for detection, image in zip(detections, images, strict=True)
        ]
        pred_scores_list = [
            detection["scores"] if "scores" in detection else image.new_zeros((0,), dtype=torch.float32)
            for detection, image in zip(detections, images, strict=True)
        ]

        if should_mdmb:
            mdmb.update(
                image_ids=image_ids,
                pred_boxes_list=pred_boxes_list,
                pred_labels_list=pred_labels_list,
                gt_boxes_list=gt_boxes_list,
                gt_labels_list=gt_labels_list,
                image_shapes=image_shapes,
                epoch=epoch,
            )

        transformed_targets = post_step["transformed_targets"]
        transformed_image_sizes = post_step["transformed_image_sizes"]
        if should_mdmbpp and isinstance(transformed_targets, list):
            candidate_summary_list = self._collect_mdmbpp_candidate_summaries(
                split_head_outputs=post_step["split_head_outputs"],
                split_anchors=post_step["split_anchors"],
                image_shapes=transformed_image_sizes,
                targets=transformed_targets,
            )
            mdmbpp.update(
                image_ids=image_ids,
                final_boxes_list=pred_boxes_list,
                final_labels_list=pred_labels_list,
                final_scores_list=pred_scores_list,
                gt_boxes_list=gt_boxes_list,
                gt_labels_list=gt_labels_list,
                image_shapes=image_shapes,
                candidate_summary_list=candidate_summary_list,
                epoch=epoch,
            )

    def _collect_mdmbpp_candidate_summaries(
        self,
        *,
        split_head_outputs: dict[str, list[torch.Tensor]],
        split_anchors: list[list[torch.Tensor]],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[PerImageCandidateSummary]:
        summaries: list[PerImageCandidateSummary] = []
        cls_levels = split_head_outputs["cls_logits"]
        reg_levels = split_head_outputs["bbox_regression"]
        ctr_levels = split_head_outputs["bbox_ctrness"]

        for image_index, target in enumerate(targets):
            image_id = target.get("image_id", torch.tensor(image_index, device=target["boxes"].device))
            gt_boxes = target["boxes"]
            gt_labels = target["labels"].to(dtype=torch.int64)
            if gt_boxes.numel() == 0:
                summaries.append(
                    PerImageCandidateSummary(
                        image_id=self._stringify_image_id(image_id),
                        candidates_by_gt_index={},
                    )
                )
                continue

            image_shape = image_shapes[image_index]
            gt_boxes_norm = normalize_xyxy_boxes(gt_boxes, image_shape).to(device=gt_boxes.device)
            candidates_by_gt_index = {gt_index: [] for gt_index in range(gt_boxes.shape[0])}
            seen_by_gt = {gt_index: set() for gt_index in range(gt_boxes.shape[0])}

            level_states: list[dict[str, object]] = []
            selected_records: list[dict[str, object]] = []

            for level_index, (cls_batch, reg_batch, ctr_batch) in enumerate(
                zip(cls_levels, reg_levels, ctr_levels, strict=True)
            ):
                cls_logits = cls_batch[image_index]
                box_regression = reg_batch[image_index]
                ctrness = ctr_batch[image_index].flatten()
                anchors_per_level = split_anchors[image_index][level_index]
                boxes_per_level = self._decode_boxes(
                    box_regression=box_regression,
                    anchors=anchors_per_level,
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)
                boxes_norm = normalize_xyxy_boxes(boxes_per_level, image_shape).to(device=gt_boxes.device)
                num_classes = int(cls_logits.shape[-1])
                class_scores = torch.sqrt(
                    torch.sigmoid(cls_logits) * torch.sigmoid(ctrness).unsqueeze(-1)
                )
                pred_scores, pred_labels = class_scores.max(dim=1)
                iou_matrix = box_ops.box_iou(gt_boxes_norm, boxes_norm)

                flat_scores = class_scores.flatten()
                selected_flat = flat_scores.new_zeros((0,), dtype=torch.long)
                selected_rank: dict[int, int] = {}
                if flat_scores.numel() > 0:
                    keep_mask = flat_scores > float(self.score_thresh)
                    if bool(keep_mask.any().item()):
                        kept_flat = torch.where(keep_mask)[0]
                        kept_scores = flat_scores[keep_mask]
                        num_topk = min(int(kept_flat.numel()), int(self.topk_candidates))
                        top_scores, top_order = kept_scores.topk(num_topk)
                        selected_flat = kept_flat[top_order]
                        for rank, flat_index in enumerate(selected_flat.tolist()):
                            selected_rank[int(flat_index)] = rank
                        selected_anchor = torch.div(selected_flat, num_classes, rounding_mode="floor")
                        selected_labels = torch.remainder(selected_flat, num_classes)
                        for flat_index, anchor_index, label, score in zip(
                            selected_flat.tolist(),
                            selected_anchor.tolist(),
                            selected_labels.tolist(),
                            top_scores.tolist(),
                            strict=True,
                        ):
                            selected_records.append(
                                {
                                    "level_index": level_index,
                                    "flat_index": int(flat_index),
                                    "anchor_index": int(anchor_index),
                                    "label": int(label),
                                    "score": float(score),
                                    "box": boxes_norm[int(anchor_index)],
                                    "survived_nms": False,
                                }
                            )

                level_states.append(
                    {
                        "level_index": level_index,
                        "level_name": f"p{level_index + 3}",
                        "boxes_norm": boxes_norm,
                        "class_scores": class_scores,
                        "pred_scores": pred_scores,
                        "pred_labels": pred_labels,
                        "selected_flat": selected_flat,
                        "selected_flat_set": set(int(value) for value in selected_flat.tolist()),
                        "selected_rank": selected_rank,
                        "iou_matrix": iou_matrix,
                        "num_classes": num_classes,
                    }
                )

            nms_survival: dict[tuple[int, int], bool] = {}
            if selected_records:
                selected_boxes = torch.stack([record["box"] for record in selected_records], dim=0)
                selected_scores = torch.tensor(
                    [record["score"] for record in selected_records],
                    dtype=torch.float32,
                    device=selected_boxes.device,
                )
                selected_labels = torch.tensor(
                    [record["label"] for record in selected_records],
                    dtype=torch.int64,
                    device=selected_boxes.device,
                )
                keep = box_ops.batched_nms(selected_boxes, selected_scores, selected_labels, self.nms_thresh)
                keep = keep[: self.detections_per_img]
                keep_indices = {int(index.item()) for index in keep}
                for selected_index, record in enumerate(selected_records):
                    survived = selected_index in keep_indices
                    record["survived_nms"] = survived
                    nms_survival[(record["level_index"], record["flat_index"])] = survived

            for gt_index in range(gt_boxes.shape[0]):
                gt_label = int(gt_labels[gt_index].item())
                for level_state in level_states:
                    for candidate in self._build_mdmbpp_candidates_for_gt(
                        level_state=level_state,
                        gt_index=gt_index,
                        gt_label=gt_label,
                        nms_survival=nms_survival,
                    ):
                        self._append_unique_mdmbpp_candidate(
                            candidates_by_gt_index[gt_index],
                            seen_by_gt[gt_index],
                            candidate,
                        )

            summaries.append(
                PerImageCandidateSummary(
                    image_id=self._stringify_image_id(image_id),
                    candidates_by_gt_index=candidates_by_gt_index,
                )
            )

        return summaries

    def _build_mdmbpp_candidates_for_gt(
        self,
        *,
        level_state: dict[str, object],
        gt_index: int,
        gt_label: int,
        nms_survival: dict[tuple[int, int], bool],
    ) -> list[CanonicalCandidate]:
        boxes_norm = level_state["boxes_norm"]
        if boxes_norm.shape[0] == 0:
            return []

        ious = level_state["iou_matrix"][gt_index]
        num_classes = int(level_state["num_classes"])
        level_index = int(level_state["level_index"])
        level_name = str(level_state["level_name"])
        pred_labels = level_state["pred_labels"]
        pred_scores = level_state["pred_scores"]
        class_scores = level_state["class_scores"]
        selected_flat = level_state["selected_flat"]
        selected_flat_set = level_state["selected_flat_set"]
        selected_rank = level_state["selected_rank"]

        candidates: list[CanonicalCandidate] = []

        best_anchor = int(ious.argmax().item())
        best_label = int(pred_labels[best_anchor].item())
        best_flat = best_anchor * num_classes + best_label
        candidates.append(
            self._make_mdmbpp_candidate(
                level_index=level_index,
                level_name=level_name,
                flat_index=best_flat,
                anchor_index=best_anchor,
                label=best_label,
                score=float(pred_scores[best_anchor].item()),
                iou_to_gt=float(ious[best_anchor].item()),
                box=boxes_norm[best_anchor],
                selected_flat_set=selected_flat_set,
                selected_rank=selected_rank,
                nms_survival=nms_survival,
            )
        )

        if 0 <= gt_label < num_classes:
            gt_scores = class_scores[:, gt_label]
            score_mask = gt_scores > float(self.score_thresh)
            if bool(score_mask.any().item()):
                score_indices = torch.nonzero(score_mask, as_tuple=False).flatten()
                threshold_anchor = int(score_indices[ious[score_indices].argmax()].item())
                threshold_flat = threshold_anchor * num_classes + gt_label
                candidates.append(
                    self._make_mdmbpp_candidate(
                        level_index=level_index,
                        level_name=level_name,
                        flat_index=threshold_flat,
                        anchor_index=threshold_anchor,
                        label=gt_label,
                        score=float(gt_scores[threshold_anchor].item()),
                        iou_to_gt=float(ious[threshold_anchor].item()),
                        box=boxes_norm[threshold_anchor],
                        selected_flat_set=selected_flat_set,
                        selected_rank=selected_rank,
                        nms_survival=nms_survival,
                    )
                )

            if selected_flat.numel() > 0:
                selected_gt_mask = torch.remainder(selected_flat, num_classes) == gt_label
                if bool(selected_gt_mask.any().item()):
                    selected_gt_flat = selected_flat[selected_gt_mask]
                    selected_gt_anchor = torch.div(
                        selected_gt_flat,
                        num_classes,
                        rounding_mode="floor",
                    )
                    best_selected_pos = int(ious[selected_gt_anchor].argmax().item())
                    best_selected_anchor = int(selected_gt_anchor[best_selected_pos].item())
                    best_selected_flat = int(selected_gt_flat[best_selected_pos].item())
                    candidates.append(
                        self._make_mdmbpp_candidate(
                            level_index=level_index,
                            level_name=level_name,
                            flat_index=best_selected_flat,
                            anchor_index=best_selected_anchor,
                            label=gt_label,
                            score=float(gt_scores[best_selected_anchor].item()),
                            iou_to_gt=float(ious[best_selected_anchor].item()),
                            box=boxes_norm[best_selected_anchor],
                            selected_flat_set=selected_flat_set,
                            selected_rank=selected_rank,
                            nms_survival=nms_survival,
                        )
                    )

        return candidates

    def _make_mdmbpp_candidate(
        self,
        *,
        level_index: int,
        level_name: str,
        flat_index: int,
        anchor_index: int,
        label: int,
        score: float,
        iou_to_gt: float,
        box: torch.Tensor,
        selected_flat_set: set[int],
        selected_rank: dict[int, int],
        nms_survival: dict[tuple[int, int], bool],
    ) -> CanonicalCandidate:
        survived_selection = flat_index in selected_flat_set
        survived_nms = (
            nms_survival.get((level_index, flat_index))
            if survived_selection
            else None
        )
        return CanonicalCandidate(
            stage=f"fcos_{level_name}",
            box=box.detach().cpu(),
            score=score,
            label=label,
            iou_to_gt=iou_to_gt,
            survived_selection=survived_selection,
            survived_nms=survived_nms,
            rank=selected_rank.get(flat_index),
            level_or_stage_id=level_name,
        )

    def _append_unique_mdmbpp_candidate(
        self,
        candidates: list[CanonicalCandidate],
        seen: set[tuple[object, ...]],
        candidate: CanonicalCandidate,
    ) -> None:
        box_key = tuple(round(float(value), 6) for value in candidate.box.tolist())
        key = (
            candidate.stage,
            candidate.label,
            round(candidate.score, 6),
            round(candidate.iou_to_gt, 6),
            candidate.survived_selection,
            candidate.survived_nms,
            candidate.rank,
            box_key,
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    def _stringify_image_id(self, image_id: torch.Tensor | int | str) -> str:
        if isinstance(image_id, torch.Tensor):
            if image_id.numel() != 1:
                raise ValueError("FCOS image_id tensor must contain a single scalar value.")
            image_id = image_id.item()
        return str(image_id)

    @torch.no_grad()
    def flush_mdmb_update(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        *,
        epoch: int | None = None,
    ) -> None:
        mdmb = self._get_mdmb()
        if mdmb is None or not targets:
            return
        if not mdmb.should_update(epoch=epoch):
            return

        image_ids = [
            target.get("image_id", torch.tensor(index, device=images[index].device))
            for index, target in enumerate(targets)
        ]
        gt_boxes_list = [target["boxes"] for target in targets]
        gt_labels_list = [target["labels"] for target in targets]
        image_shapes = [tuple(int(dim) for dim in image.shape[-2:]) for image in images]

        was_training = self.training
        try:
            self.eval()
            detections = self(images)
        finally:
            if was_training:
                self.train()

        pred_boxes_list = [
            detection["boxes"] if "boxes" in detection else image.new_zeros((0, 4))
            for detection, image in zip(detections, images, strict=True)
        ]
        pred_labels_list = [
            detection["labels"] if "labels" in detection else image.new_zeros((0,), dtype=torch.int64)
            for detection, image in zip(detections, images, strict=True)
        ]
        mdmb.update(
            image_ids=image_ids,
            pred_boxes_list=pred_boxes_list,
            pred_labels_list=pred_labels_list,
            gt_boxes_list=gt_boxes_list,
            gt_labels_list=gt_labels_list,
            image_shapes=image_shapes,
            epoch=epoch,
        )

    def _get_mdmb(self) -> MissedDetectionMemoryBank | None:
        if self._mdmb_ref is None:
            return None
        return self._mdmb_ref()

    def _get_mdmbpp(self) -> MDMBPlus | None:
        if self._mdmbpp_ref is None:
            return None
        return self._mdmbpp_ref()

    def _get_faar(self) -> FailureAwareAssignmentRepair | None:
        if self._faar_ref is None:
            return None
        return self._faar_ref()

    def _get_fang(self) -> FailureAwareNegativeGradientShielding | None:
        if self._fang_ref is None:
            return None
        return self._fang_ref()

    def _get_marc(self) -> MissAwareRankingCalibration | None:
        if self._marc_ref is None:
            return None
        return self._marc_ref()

    def _get_candidate_densifier(self) -> CandidateDensifier | None:
        if self._candidate_densifier_ref is None:
            return None
        return self._candidate_densifier_ref()


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
        mdmb: MissedDetectionMemoryBank | None = None,
        mdmbpp: MDMBPlus | None = None,
        faar: FailureAwareAssignmentRepair | None = None,
        fang: FailureAwareNegativeGradientShielding | None = None,
        marc: MissAwareRankingCalibration | None = None,
        candidate_densifier: CandidateDensifier | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mdmb = mdmb
        self.mdmbpp = mdmbpp
        self.faar = faar
        self.fang = fang
        self.marc = marc
        self.candidate_densifier = candidate_densifier

        backbone = build_backbone_with_fpn(
            cfg,
            extra_blocks=LastLevelP6P7(256, 256),
            pre_neck=pre_neck,
            post_neck=post_neck,
            returned_layers=[2, 3, 4],
        )

        head = cfg.get("head", {})
        tfm = cfg.get("transform", {})

        self.model = MDMBFCOS(
            backbone=backbone,
            num_classes=cfg.get("num_classes", 91),
            min_size=tfm.get("min_size", 800),
            max_size=tfm.get("max_size", 1333),
            score_thresh=head.get("score_thresh", 0.2),
            nms_thresh=head.get("nms_thresh", 0.6),
            detections_per_img=head.get("detections_per_img", 100),
            topk_candidates=head.get("topk_candidates", 1000),
            mdmb=mdmb,
            mdmbpp=mdmbpp,
            faar=faar,
            fang=fang,
            marc=marc,
            candidate_densifier=candidate_densifier,
            **kwargs,
        )

    @torch.no_grad()
    def after_optimizer_step(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        *,
        epoch_index: int | None = None,
    ) -> None:
        if self.mdmb is None and self.mdmbpp is None:
            return
        epoch = None if epoch_index is None else int(epoch_index) + 1
        self.model.flush_post_step_updates(images, targets, epoch=epoch)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        mdmbpp: MDMBPlus | None = None,
        faar: FailureAwareAssignmentRepair | None = None,
        fang: FailureAwareNegativeGradientShielding | None = None,
        marc: MissAwareRankingCalibration | None = None,
        candidate_densifier: CandidateDensifier | None = None,
        **kwargs,
    ) -> "FCOSWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            mdmb=mdmb,
            mdmbpp=mdmbpp,
            faar=faar,
            fang=fang,
            marc=marc,
            candidate_densifier=candidate_densifier,
            **kwargs,
        )
