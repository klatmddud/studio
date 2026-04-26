from __future__ import annotations

"""FCOS wrapper: YAML config -> torchvision FCOS."""

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
    CanonicalCandidate,
    DetectionHysteresisMemory,
    DHMRecord,
    FalseNegativeTransitionDirectionMemory,
    MDMBPlus,
    PerImageCandidateSummary,
    RelapseAwareSupportDistillation,
    TemporalFailureMemory,
    pool_multiscale_box_features,
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
    FCOS variant with MDMB/MDMB++/TFM memory updates and optional UMR training modules.

    During training the wrapper can add RASD support-distillation loss and
    apply replay-aware per-GT loss weights. TFM is refreshed from the normal
    training forward. After optimizer.step(), the wrapper runs an extra no-grad
    inference pass only when MDMB or MDMB++ needs final-detection state.
    """

    def __init__(
        self,
        *args,
        mdmb: MissedDetectionMemoryBank | None = None,
        mdmbpp: MDMBPlus | None = None,
        rasd: RelapseAwareSupportDistillation | None = None,
        tfm: TemporalFailureMemory | None = None,
        fntdm: FalseNegativeTransitionDirectionMemory | None = None,
        dhm: DetectionHysteresisMemory | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mdmb_ref = weakref.ref(mdmb) if mdmb is not None else None
        self._mdmbpp_ref = weakref.ref(mdmbpp) if mdmbpp is not None else None
        self._rasd_ref = weakref.ref(rasd) if rasd is not None else None
        self._tfm_ref = weakref.ref(tfm) if tfm is not None else None
        self._fntdm_ref = weakref.ref(fntdm) if fntdm is not None else None
        self._dhm_ref = weakref.ref(dhm) if dhm is not None else None

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
            tfm = self._get_tfm()
            dhm = self._get_dhm()
            use_tfm_bias = self._should_apply_tfm_assignment_bias(tfm)
            use_dhm_weighting = self._should_apply_dhm_loss_weighting(dhm)
            use_weighted = _has_replay_box_weights(targets) or use_tfm_bias or use_dhm_weighting
            if not use_weighted:
                losses = self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
            else:
                losses = self._compute_weighted_loss_dict(
                    targets=targets,
                    image_shapes=images.image_sizes,
                    head_outputs=head_outputs,
                    anchors=anchors,
                    matched_idxs=matched_idxs,
                    tfm=tfm if use_tfm_bias else None,
                    dhm=dhm if use_dhm_weighting else None,
                )
            self._refresh_tfm_from_training_forward(
                targets=targets,
                image_shapes=images.image_sizes,
                features=features,
                head_outputs=head_outputs,
                anchors=anchors,
                matched_idxs=matched_idxs,
                num_anchors_per_level=num_anchors_per_level,
            )

            rasd = self._get_rasd()
            if rasd is not None and rasd.should_apply(mdmbpp=mdmbpp):
                rasd_plan = rasd.plan(
                    mdmbpp=mdmbpp,
                    targets=targets,
                    image_shapes=images.image_sizes,
                )
                rasd_loss, rasd_stats = rasd.compute_loss(
                    plan=rasd_plan,
                    features=features,
                    image_shapes=images.image_sizes,
                )
                rasd.record_step(plan=rasd_plan, stats=rasd_stats)
                if int(rasd_stats.get("losses", 0)) > 0:
                    losses = dict(losses)
                    losses["rasd"] = rasd_loss

            fntdm = self._get_fntdm()
            if fntdm is not None:
                fntdm_loss = fntdm.compute_loss(
                    targets=targets,
                    image_shapes=images.image_sizes,
                    features=features,
                    head_outputs=head_outputs,
                    anchors=anchors,
                    matched_idxs=matched_idxs,
                )
                if bool(torch.isfinite(fntdm_loss).item()) and float(fntdm_loss.detach().item()) != 0.0:
                    losses = dict(losses)
                    losses["fntdm"] = fntdm_loss

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

    def _compute_weighted_loss_dict(
        self,
        *,
        targets: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        tfm: TemporalFailureMemory | None = None,
        dhm: DetectionHysteresisMemory | None = None,
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
            if tfm is not None:
                cls_weights = self._apply_tfm_assignment_bias(
                    base_weights=cls_weights,
                    tfm=tfm,
                    target=target,
                    image_shape=image_shapes[image_index],
                    image_index=image_index,
                    assignments=assignments,
                    component="cls",
                )
                reg_weights = self._apply_tfm_assignment_bias(
                    base_weights=reg_weights,
                    tfm=tfm,
                    target=target,
                    image_shape=image_shapes[image_index],
                    image_index=image_index,
                    assignments=assignments,
                    component="box",
                )
                ctr_weights = self._apply_tfm_assignment_bias(
                    base_weights=ctr_weights,
                    tfm=tfm,
                    target=target,
                    image_shape=image_shapes[image_index],
                    image_index=image_index,
                    assignments=assignments,
                    component="ctr",
                )
            if dhm is not None:
                cls_weights = self._apply_dhm_loss_weighting(
                    base_weights=cls_weights,
                    dhm=dhm,
                    target=target,
                    image_shape=image_shapes[image_index],
                    image_index=image_index,
                    assignments=assignments,
                    component="cls",
                )
                reg_weights = self._apply_dhm_loss_weighting(
                    base_weights=reg_weights,
                    dhm=dhm,
                    target=target,
                    image_shape=image_shapes[image_index],
                    image_index=image_index,
                    assignments=assignments,
                    component="box",
                )
                ctr_weights = self._apply_dhm_loss_weighting(
                    base_weights=ctr_weights,
                    dhm=dhm,
                    target=target,
                    image_shape=image_shapes[image_index],
                    image_index=image_index,
                    assignments=assignments,
                    component="ctr",
                )

            cls_losses_by_class = raw["cls_losses_by_class"]
            cls_weights = cls_weights.to(dtype=cls_losses_by_class.dtype)
            reg_weights = reg_weights.to(dtype=cls_losses_by_class.dtype)
            ctr_weights = ctr_weights.to(dtype=cls_losses_by_class.dtype)

            cls_sum = cls_sum + (cls_losses_by_class * cls_weights.unsqueeze(1)).sum()
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

    def _should_apply_tfm_assignment_bias(self, tfm: TemporalFailureMemory | None) -> bool:
        if tfm is None:
            return False
        if not bool(tfm.config.assignment_bias.enabled):
            return False
        if len(tfm) == 0:
            return False
        return tfm.should_update()

    def _apply_tfm_assignment_bias(
        self,
        *,
        base_weights: torch.Tensor,
        tfm: TemporalFailureMemory,
        target: dict[str, torch.Tensor],
        image_shape: tuple[int, int],
        image_index: int,
        assignments: torch.Tensor,
        component: str,
    ) -> torch.Tensor:
        if _is_replay_target(target):
            return base_weights

        pos_mask = assignments >= 0
        if not bool(pos_mask.any().item()):
            return base_weights

        risks = self._lookup_tfm_gt_risks(
            tfm=tfm,
            target=target,
            image_shape=image_shape,
            image_index=image_index,
            device=assignments.device,
        )
        if risks.numel() == 0:
            return base_weights

        scale = {
            "cls": tfm.config.assignment_bias.cls_weight,
            "box": tfm.config.assignment_bias.box_weight,
            "ctr": tfm.config.assignment_bias.ctr_weight,
        }[component]
        gt_weights = self._tfm_component_weights(risks, tfm=tfm, scale=float(scale))
        if bool((gt_weights == 1.0).all().item()):
            return base_weights

        result = base_weights.clone()
        gt_indices = assignments[pos_mask].to(dtype=torch.long)
        valid = gt_indices < gt_weights.numel()
        if not bool(valid.any().item()):
            return result

        pos_indices = torch.where(pos_mask)[0][valid]
        selected_gt = gt_indices[valid]
        result[pos_indices] = result[pos_indices] * gt_weights[selected_gt].to(
            device=result.device,
            dtype=result.dtype,
        )
        return result

    def _lookup_tfm_gt_risks(
        self,
        *,
        tfm: TemporalFailureMemory,
        target: dict[str, torch.Tensor],
        image_shape: tuple[int, int],
        image_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        gt_boxes = target["boxes"]
        if gt_boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=device)

        image_id = target.get("image_id", torch.tensor(image_index, device=gt_boxes.device))
        records = tfm.get_image_records(image_id)
        risks = torch.zeros((gt_boxes.shape[0],), dtype=torch.float32, device=device)
        if not records:
            return risks

        gt_labels = target["labels"].to(dtype=torch.int64)
        gt_boxes_norm = normalize_xyxy_boxes(gt_boxes, image_shape).detach().cpu()
        record_boxes = torch.stack([record.bbox for record in records], dim=0)
        ious = box_ops.box_iou(gt_boxes_norm, record_boxes)
        used_records: set[int] = set()
        threshold = float(tfm.config.record_match_threshold)

        for gt_index in range(int(gt_boxes_norm.shape[0])):
            class_id = int(gt_labels[gt_index].item())
            best_record = None
            best_iou = threshold
            for record_index, record in enumerate(records):
                if record_index in used_records or int(record.class_id) != class_id:
                    continue
                iou = float(ious[gt_index, record_index].item())
                if iou > best_iou:
                    best_iou = iou
                    best_record = record_index
            if best_record is None:
                continue
            used_records.add(best_record)
            risks[gt_index] = float(records[best_record].risk)
        return risks

    def _tfm_component_weights(
        self,
        risks: torch.Tensor,
        *,
        tfm: TemporalFailureMemory,
        scale: float,
    ) -> torch.Tensor:
        if scale <= 0.0:
            return torch.ones_like(risks)
        config = tfm.config.assignment_bias
        min_risk = float(config.min_risk)
        if min_risk >= 1.0:
            normalized = (risks >= 1.0).to(dtype=risks.dtype)
        else:
            normalized = ((risks - min_risk).clamp(min=0.0) / max(1.0 - min_risk, 1e-6)).clamp(max=1.0)
        weights = 1.0 + float(scale) * normalized
        return weights.clamp(max=float(config.max_weight))

    def _should_apply_dhm_loss_weighting(self, dhm: DetectionHysteresisMemory | None) -> bool:
        if dhm is None:
            return False
        if not bool(dhm.config.loss_weighting.enabled):
            return False
        if len(dhm) == 0:
            return False
        return True

    def _should_apply_dhm_assignment_expansion(self, dhm: DetectionHysteresisMemory | None) -> bool:
        if dhm is None:
            return False
        if not bool(dhm.config.assignment_expansion.enabled):
            return False
        if int(dhm.config.assignment_expansion.backup_topk) <= 0:
            return False
        if len(dhm) == 0:
            return False
        return True

    def _apply_dhm_loss_weighting(
        self,
        *,
        base_weights: torch.Tensor,
        dhm: DetectionHysteresisMemory,
        target: dict[str, torch.Tensor],
        image_shape: tuple[int, int],
        image_index: int,
        assignments: torch.Tensor,
        component: str,
    ) -> torch.Tensor:
        if _is_replay_target(target):
            return base_weights

        pos_mask = assignments >= 0
        if not bool(pos_mask.any().item()):
            return base_weights

        records = self._lookup_dhm_gt_records(
            dhm=dhm,
            target=target,
            image_shape=image_shape,
            image_index=image_index,
        )
        if not records:
            return base_weights

        gt_weights = torch.ones((len(records),), dtype=torch.float32, device=assignments.device)
        for gt_index, record in enumerate(records):
            gt_weights[gt_index] = float(dhm.loss_weight_for_record(record, component=component))
        if bool((gt_weights == 1.0).all().item()):
            return base_weights

        result = base_weights.clone()
        gt_indices = assignments[pos_mask].to(dtype=torch.long)
        valid = gt_indices < gt_weights.numel()
        if not bool(valid.any().item()):
            return result

        pos_indices = torch.where(pos_mask)[0][valid]
        selected_gt = gt_indices[valid]
        result[pos_indices] = result[pos_indices] * gt_weights[selected_gt].to(
            device=result.device,
            dtype=result.dtype,
        )
        return result

    def _lookup_dhm_gt_records(
        self,
        *,
        dhm: DetectionHysteresisMemory,
        target: dict[str, torch.Tensor],
        image_shape: tuple[int, int],
        image_index: int,
    ) -> list[DHMRecord | None]:
        gt_boxes = target["boxes"]
        num_gt = int(gt_boxes.shape[0])
        if num_gt == 0:
            return []

        image_id = target.get("image_id", torch.tensor(image_index, device=gt_boxes.device))
        image_records = dhm.get_image_records(image_id)
        result: list[DHMRecord | None] = [None for _ in range(num_gt)]
        if not image_records:
            return result

        gt_ids = self._get_tfm_gt_ids(target)
        if isinstance(gt_ids, torch.Tensor) and int(gt_ids.numel()) == num_gt:
            by_ann_id = {
                str(record.ann_id): record
                for record in image_records
                if record.ann_id is not None
            }
            flattened = gt_ids.detach().cpu().flatten().tolist()
            for gt_index, raw_gt_id in enumerate(flattened):
                try:
                    if int(raw_gt_id) < 0:
                        continue
                except (TypeError, ValueError):
                    if str(raw_gt_id) == "":
                        continue
                record = by_ann_id.get(str(raw_gt_id))
                if record is not None:
                    result[gt_index] = record

        unmatched_gt = [index for index, record in enumerate(result) if record is None]
        if not unmatched_gt:
            return result

        gt_labels = target["labels"].to(dtype=torch.int64)
        matched_uids = {record.gt_uid for record in result if record is not None}
        candidate_records = [
            record
            for record in image_records
            if record.gt_uid not in matched_uids
        ]
        if not candidate_records:
            return result
        record_boxes = torch.stack([record.bbox for record in candidate_records], dim=0).to(
            device=gt_boxes.device,
            dtype=gt_boxes.dtype,
        )
        ious = box_ops.box_iou(gt_boxes, record_boxes)
        used_records: set[int] = set()
        threshold = float(dhm.config.record_match_threshold)
        for gt_index in unmatched_gt:
            class_id = int(gt_labels[gt_index].item())
            best_record = None
            best_iou = threshold
            for record_index, record in enumerate(candidate_records):
                if record_index in used_records or int(record.class_id) != class_id:
                    continue
                iou = float(ious[gt_index, record_index].item())
                if iou > best_iou:
                    best_iou = iou
                    best_record = record_index
            if best_record is None:
                continue
            used_records.add(best_record)
            result[gt_index] = candidate_records[best_record]
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

    @torch.no_grad()
    def _refresh_tfm_from_training_forward(
        self,
        *,
        targets: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
        features: OrderedDict[str, torch.Tensor],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        num_anchors_per_level: list[int],
    ) -> None:
        tfm = self._get_tfm()
        if tfm is None or not tfm.should_update():
            return

        flat_features = None
        if bool(tfm.config.support.enabled):
            flat_features = self._flatten_tfm_feature_levels(
                features=features,
                num_anchors_per_level=num_anchors_per_level,
            )

        image_ids = []
        gt_boxes_list = []
        gt_labels_list = []
        kept_image_shapes = []
        gt_states_list = []
        quality_list = []
        gt_ids_list = []
        support_feature_list = [] if flat_features is not None else None

        for image_index, target in enumerate(targets):
            if _is_replay_target(target):
                continue
            image_ids.append(
                target.get("image_id", torch.tensor(image_index, device=target["boxes"].device))
            )
            gt_boxes_list.append(target["boxes"])
            gt_labels_list.append(target["labels"])
            kept_image_shapes.append(image_shapes[image_index])
            gt_ids_list.append(self._get_tfm_gt_ids(target))

            states, qualities, best_point_indices = self._collect_tfm_states_for_image(
                image_index=image_index,
                target=target,
                head_outputs=head_outputs,
                anchors_per_image=anchors[image_index],
                matched_idxs_per_image=matched_idxs[image_index],
                detected_quality_threshold=float(tfm.config.detected_quality_threshold),
            )
            gt_states_list.append(states)
            quality_list.append(qualities)

            if support_feature_list is not None:
                image_features: dict[int, torch.Tensor] = {}
                for gt_index, point_index in best_point_indices.items():
                    image_features[int(gt_index)] = flat_features[image_index, int(point_index)].detach().cpu()
                support_feature_list.append(image_features)

        if not image_ids:
            return

        tfm.update(
            image_ids=image_ids,
            gt_boxes_list=gt_boxes_list,
            gt_labels_list=gt_labels_list,
            image_shapes=kept_image_shapes,
            gt_states_list=gt_states_list,
            quality_list=quality_list,
            gt_ids_list=gt_ids_list,
            support_feature_list=support_feature_list,
            support_quality_list=quality_list if support_feature_list is not None else None,
        )

    def _collect_tfm_states_for_image(
        self,
        *,
        image_index: int,
        target: dict[str, torch.Tensor],
        head_outputs: dict[str, torch.Tensor],
        anchors_per_image: torch.Tensor,
        matched_idxs_per_image: torch.Tensor,
        detected_quality_threshold: float,
    ) -> tuple[list[str], list[float], dict[int, int]]:
        gt_boxes = target["boxes"]
        gt_labels = target["labels"].to(dtype=torch.int64)
        num_gt = int(gt_boxes.shape[0])
        states = ["missing_assignment"] * num_gt
        qualities = [0.0] * num_gt
        best_point_indices: dict[int, int] = {}
        if num_gt == 0:
            return states, qualities, best_point_indices

        cls_logits = head_outputs["cls_logits"][image_index]
        bbox_regression = head_outputs["bbox_regression"][image_index]
        bbox_ctrness = head_outputs["bbox_ctrness"][image_index].flatten()
        pos_mask = matched_idxs_per_image >= 0
        if not bool(pos_mask.any().item()):
            return states, qualities, best_point_indices

        pos_indices = torch.where(pos_mask)[0]
        matched_gt_indices = matched_idxs_per_image[pos_mask].to(dtype=torch.long)
        pred_boxes = self._decode_boxes(
            box_regression=bbox_regression[pos_mask],
            anchors=anchors_per_image[pos_mask],
        )
        matched_gt_boxes = gt_boxes[matched_gt_indices]
        ious = self._aligned_box_iou(pred_boxes, matched_gt_boxes).clamp(min=0.0, max=1.0)
        cls_probs = torch.sigmoid(cls_logits[pos_mask])
        ctr_probs = torch.sigmoid(bbox_ctrness[pos_mask]).clamp(min=0.0, max=1.0)

        for gt_index in range(num_gt):
            gt_pos_mask = matched_gt_indices == gt_index
            if not bool(gt_pos_mask.any().item()):
                continue
            gt_label = int(gt_labels[gt_index].item())
            if gt_label < 0 or gt_label >= cls_probs.shape[1]:
                states[gt_index] = "unknown"
                continue

            local_pos = torch.where(gt_pos_mask)[0]
            gt_cls_scores = cls_probs[local_pos, gt_label]
            gt_ious = ious[local_pos]
            gt_ctr = ctr_probs[local_pos]
            quality_scores = (gt_cls_scores * gt_ious * gt_ctr).detach()
            best_local = int(quality_scores.argmax().item())
            best_quality = float(quality_scores[best_local].item())
            best_point_indices[gt_index] = int(pos_indices[local_pos[best_local]].item())
            qualities[gt_index] = best_quality

            best_cls = float(gt_cls_scores.max().item())
            best_iou = float(gt_ious.max().item())
            competitor = self._max_competing_class_score(cls_probs[local_pos], gt_label)
            if best_quality >= detected_quality_threshold:
                states[gt_index] = "detected_like"
            elif competitor > best_cls and competitor >= 0.25:
                states[gt_index] = "classification_confusion"
            elif best_iou < 0.5:
                states[gt_index] = "localization_weak"
            else:
                states[gt_index] = "score_weak"

        return states, qualities, best_point_indices

    def _flatten_tfm_feature_levels(
        self,
        *,
        features: OrderedDict[str, torch.Tensor],
        num_anchors_per_level: list[int],
    ) -> torch.Tensor | None:
        flattened = []
        for feature, expected_points in zip(features.values(), num_anchors_per_level, strict=True):
            level_features = feature.permute(0, 2, 3, 1).reshape(feature.shape[0], -1, feature.shape[1])
            if int(level_features.shape[1]) != int(expected_points):
                return None
            flattened.append(level_features)
        if not flattened:
            return None
        return torch.cat(flattened, dim=1)

    def _max_competing_class_score(self, scores: torch.Tensor, gt_label: int) -> float:
        if scores.shape[1] <= 1:
            return 0.0
        competitor_scores = scores.clone()
        competitor_scores[:, int(gt_label)] = -1.0
        return float(competitor_scores.max().item())

    def _aligned_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        if boxes1.shape != boxes2.shape:
            raise ValueError(
                "FCOS aligned IoU expects matching box tensors. "
                f"Got {tuple(boxes1.shape)} and {tuple(boxes2.shape)}."
            )
        x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        union = (area1 + area2 - inter).clamp(min=torch.finfo(boxes1.dtype).eps)
        return inter / union

    def _get_tfm_gt_ids(self, target: dict[str, torch.Tensor]) -> torch.Tensor | None:
        for key in ("gt_ids", "annotation_ids", "ann_ids"):
            value = target.get(key)
            if isinstance(value, torch.Tensor):
                return value
        return None

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
        dhm = self._get_dhm()
        use_dhm_expansion = self._should_apply_dhm_assignment_expansion(dhm)
        for image_index, (anchors_per_image, targets_per_image) in enumerate(
            zip(anchors, targets, strict=True)
        ):
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
            if use_dhm_expansion and dhm is not None:
                matched_idx = self._apply_dhm_assignment_expansion(
                    dhm=dhm,
                    target=targets_per_image,
                    matched_idx=matched_idx,
                    anchor_centers=anchor_centers,
                    anchor_sizes=anchor_sizes,
                    num_anchors_per_level=num_anchors_per_level,
                    image_index=image_index,
                )
            matched_idxs.append(matched_idx)
        return matched_idxs

    def _apply_dhm_assignment_expansion(
        self,
        *,
        dhm: DetectionHysteresisMemory,
        target: dict[str, torch.Tensor],
        matched_idx: torch.Tensor,
        anchor_centers: torch.Tensor,
        anchor_sizes: torch.Tensor,
        num_anchors_per_level: list[int],
        image_index: int,
    ) -> torch.Tensor:
        if _is_replay_target(target):
            return matched_idx

        gt_boxes = target["boxes"]
        num_gt = int(gt_boxes.shape[0])
        if num_gt == 0:
            return matched_idx

        records = self._lookup_dhm_gt_records(
            dhm=dhm,
            target=target,
            image_shape=(0, 0),
            image_index=image_index,
        )
        if not records:
            return matched_idx

        eligible: list[tuple[int, DHMRecord, float]] = []
        for gt_index, record in enumerate(records):
            if record is None:
                continue
            radius = float(dhm.assignment_expansion_radius_for_record(record))
            if radius <= 1.0:
                continue
            eligible.append(
                (
                    gt_index,
                    record,
                    max(float(self.center_sampling_radius), radius),
                )
            )
        if not eligible:
            return matched_idx

        config = dhm.config.assignment_expansion
        backup_topk = int(config.backup_topk)
        base_positive_count = int((matched_idx >= 0).sum().item())
        max_extra = max(
            backup_topk,
            int(float(max(base_positive_count, 1)) * float(config.max_extra_positive_ratio) + 0.999),
        )
        if max_extra <= 0:
            return matched_idx

        x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
        inside_box = pairwise_dist.min(dim=2).values > 0

        lower_bound = anchor_sizes * 4
        lower_bound[: num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        upper_bound[-num_anchors_per_level[-1] :] = float("inf")
        max_box_dist = pairwise_dist.max(dim=2).values
        scale_match = (max_box_dist > lower_bound[:, None]) & (max_box_dist < upper_bound[:, None])
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        center_dist = (anchor_centers[:, None, :] - gt_centers[None]).abs().max(dim=2).values

        updated = matched_idx.clone()
        remaining = max_extra
        eligible = sorted(
            eligible,
            key=lambda item: (
                float(item[1].instability_score),
                float(item[1].consecutive_fn),
                float(item[1].fn_count),
            ),
            reverse=True,
        )
        for gt_index, _record, radius in eligible:
            if remaining <= 0:
                break
            candidate_mask = (
                (updated < 0)
                & inside_box[:, gt_index]
                & scale_match[:, gt_index]
                & (center_dist[:, gt_index] < float(radius) * anchor_sizes)
            )
            if not bool(candidate_mask.any().item()):
                continue
            candidate_indices = torch.where(candidate_mask)[0]
            candidate_dist = center_dist[candidate_indices, gt_index]
            add_count = min(backup_topk, remaining, int(candidate_indices.numel()))
            if add_count <= 0:
                continue
            selected_order = torch.argsort(candidate_dist)[:add_count]
            selected_indices = candidate_indices[selected_order]
            updated[selected_indices] = int(gt_index)
            remaining -= add_count
        return updated

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
            "features": features,
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
            support_feature_list = None
            if bool(mdmbpp.config.store_support_feature):
                support_feature_list = self._collect_mdmbpp_support_features(
                    features=post_step["features"],
                    targets=transformed_targets,
                    image_shapes=transformed_image_sizes,
                )
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
                support_feature_list=support_feature_list,
                epoch=epoch,
            )

    @torch.no_grad()
    def mine_fntdm_batch(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        *,
        epoch: int,
    ) -> dict[str, int] | None:
        fntdm = self._get_fntdm()
        if fntdm is None or not targets:
            return None
        if not fntdm.should_mine(epoch=epoch):
            return None

        non_replay = [
            (image, target)
            for image, target in zip(images, targets, strict=True)
            if not _is_replay_target(target)
        ]
        if not non_replay:
            return None
        images = [image for image, _ in non_replay]
        targets = [target for _, target in non_replay]

        was_training = self.training
        try:
            self.eval()
            mined = self._run_post_step_inference(images, targets)
        finally:
            if was_training:
                self.train()

        transformed_targets = mined.get("transformed_targets")
        if not isinstance(transformed_targets, list):
            return None
        return fntdm.mine_batch(
            detections=mined["detections"],
            original_targets=targets,
            transformed_targets=transformed_targets,
            transformed_image_shapes=mined["transformed_image_sizes"],
            features=mined["features"],
            epoch=epoch,
        )

    @torch.no_grad()
    def mine_dhm_batch(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        *,
        epoch: int,
    ) -> dict[str, int] | None:
        dhm = self._get_dhm()
        if dhm is None or not targets:
            return None
        if not dhm.should_mine(epoch=epoch):
            return None

        non_replay = [
            (image, target)
            for image, target in zip(images, targets, strict=True)
            if not _is_replay_target(target)
        ]
        if not non_replay:
            return None
        images = [image for image, _ in non_replay]
        targets = [target for _, target in non_replay]

        was_training = self.training
        try:
            self.eval()
            mined = self._run_post_step_inference(images, targets)
        finally:
            if was_training:
                self.train()

        detections = mined.get("detections")
        if not isinstance(detections, list):
            return None
        return dhm.mine_batch(
            detections=detections,
            original_targets=targets,
            epoch=epoch,
        )

    def _collect_mdmbpp_support_features(
        self,
        *,
        features: OrderedDict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[int, torch.Tensor]]:
        boxes_per_image = [
            target["boxes"].to(dtype=torch.float32).reshape(-1, 4)
            for target in targets
        ]
        total_boxes = sum(int(boxes.shape[0]) for boxes in boxes_per_image)
        if total_boxes == 0:
            return [{} for _ in targets]

        pooled_features = pool_multiscale_box_features(
            features=features,
            boxes_per_image=boxes_per_image,
            image_shapes=image_shapes,
            output_size=7,
            sampling_ratio=2,
            normalize=True,
        )
        support_feature_list: list[dict[int, torch.Tensor]] = []
        offset = 0
        for boxes in boxes_per_image:
            count = int(boxes.shape[0])
            image_features: dict[int, torch.Tensor] = {}
            for gt_index in range(count):
                image_features[gt_index] = pooled_features[offset + gt_index].detach().cpu()
            support_feature_list.append(image_features)
            offset += count
        return support_feature_list

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

    def _get_rasd(self) -> RelapseAwareSupportDistillation | None:
        if self._rasd_ref is None:
            return None
        return self._rasd_ref()

    def _get_tfm(self) -> TemporalFailureMemory | None:
        if self._tfm_ref is None:
            return None
        return self._tfm_ref()

    def _get_fntdm(self) -> FalseNegativeTransitionDirectionMemory | None:
        if self._fntdm_ref is None:
            return None
        return self._fntdm_ref()

    def _get_dhm(self) -> DetectionHysteresisMemory | None:
        if self._dhm_ref is None:
            return None
        return self._dhm_ref()


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
        rasd: RelapseAwareSupportDistillation | None = None,
        tfm: TemporalFailureMemory | None = None,
        fntdm: FalseNegativeTransitionDirectionMemory | None = None,
        dhm: DetectionHysteresisMemory | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mdmb = mdmb
        self.mdmbpp = mdmbpp
        self.rasd = rasd
        self.tfm = tfm
        self.fntdm = fntdm
        self.dhm = dhm

        backbone = build_backbone_with_fpn(
            cfg,
            extra_blocks=LastLevelP6P7(256, 256),
            pre_neck=pre_neck,
            post_neck=post_neck,
            returned_layers=[2, 3, 4],
        )

        head = cfg.get("head", {})
        transform_cfg = cfg.get("transform", {})

        self.model = MDMBFCOS(
            backbone=backbone,
            num_classes=cfg.get("num_classes", 91),
            min_size=transform_cfg.get("min_size", 800),
            max_size=transform_cfg.get("max_size", 1333),
            score_thresh=head.get("score_thresh", 0.2),
            nms_thresh=head.get("nms_thresh", 0.6),
            detections_per_img=head.get("detections_per_img", 100),
            topk_candidates=head.get("topk_candidates", 1000),
            mdmb=mdmb,
            mdmbpp=mdmbpp,
            rasd=rasd,
            tfm=tfm,
            fntdm=fntdm,
            dhm=dhm,
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

    @torch.no_grad()
    def mine_fntdm_batch(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        *,
        epoch: int,
    ) -> dict[str, int] | None:
        if self.fntdm is None:
            return None
        return self.model.mine_fntdm_batch(images, targets, epoch=epoch)

    @torch.no_grad()
    def mine_dhm_batch(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        *,
        epoch: int,
    ) -> dict[str, int] | None:
        if self.dhm is None:
            return None
        return self.model.mine_dhm_batch(images, targets, epoch=epoch)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        mdmbpp: MDMBPlus | None = None,
        rasd: RelapseAwareSupportDistillation | None = None,
        tfm: TemporalFailureMemory | None = None,
        fntdm: FalseNegativeTransitionDirectionMemory | None = None,
        dhm: DetectionHysteresisMemory | None = None,
        **kwargs,
    ) -> "FCOSWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            mdmb=mdmb,
            mdmbpp=mdmbpp,
            rasd=rasd,
            tfm=tfm,
            fntdm=fntdm,
            dhm=dhm,
            **kwargs,
        )
