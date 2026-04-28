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
    DetectionHysteresisMemory,
    DHMRecord,
    DHMRepairModule,
)

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


class DHMFCOS(FCOS):
    """FCOS variant with DHM mining and optional DHM-R border refinement."""

    def __init__(
        self,
        *args,
        dhm: DetectionHysteresisMemory | None = None,
        dhmr: DHMRepairModule | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dhm_ref = weakref.ref(dhm) if dhm is not None else None
        self._dhmr_ref = weakref.ref(dhmr) if dhmr is not None else None

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

        anchors = self.anchor_generator(images, feature_list)
        num_anchors_per_level = [x.size(2) * x.size(3) for x in feature_list]

        losses: dict[str, torch.Tensor] = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            matched_idxs = self._match_anchors_to_targets(targets, anchors, num_anchors_per_level)
            dhm = self._get_dhm()
            dhmr = self._get_dhmr()
            head_outputs = self.head(feature_list)
            if dhm is not None and len(dhm) > 0:
                self._record_dhm_assignment_statistics_for_batch(
                    targets=targets,
                    image_shapes=images.image_sizes,
                    head_outputs=head_outputs,
                    anchors=anchors,
                    matched_idxs=matched_idxs,
                    num_anchors_per_level=num_anchors_per_level,
                    dhm=dhm,
                )
            losses = self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

            if dhmr is not None and dhm is not None:
                losses.update(
                    self._compute_dhmr_border_refinement_losses(
                        dhmr=dhmr,
                        dhm=dhm,
                        targets=targets,
                        image_shapes=images.image_sizes,
                        padded_shape=tuple(int(dim) for dim in images.tensors.shape[-2:]),
                        feature_maps=feature_list,
                        head_outputs=head_outputs,
                        anchors=anchors,
                        matched_idxs=matched_idxs,
                        num_anchors_per_level=num_anchors_per_level,
                    )
                )
        else:
            head_outputs = self.head(feature_list)
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

    def _compute_dhmr_border_refinement_losses(
        self,
        *,
        dhmr: DHMRepairModule,
        dhm: DetectionHysteresisMemory,
        targets: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
        padded_shape: tuple[int, int],
        feature_maps: list[torch.Tensor],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        num_anchors_per_level: list[int],
    ) -> dict[str, torch.Tensor]:
        del image_shapes
        if len(dhm) == 0 or not dhmr.uses_border_refinement():
            return {}
        records_by_image = [
            (
                [None for _ in range(int(target["boxes"].shape[0]))]
                if self._is_replay_crop_target(target)
                else self._lookup_dhm_gt_records(
                    dhm=dhm,
                    target=target,
                    image_shape=(0, 0),
                    image_index=image_index,
                )
            )
            for image_index, target in enumerate(targets)
        ]
        return dhmr.compute_border_refinement_loss(
            targets=targets,
            feature_maps=feature_maps,
            head_outputs=head_outputs,
            anchors=anchors,
            matched_idxs=matched_idxs,
            dhm_records=records_by_image,
            num_anchors_per_level=num_anchors_per_level,
            padded_shape=padded_shape,
            decode_boxes=self._decode_boxes,
        )

    @torch.no_grad()
    def _record_dhm_assignment_statistics_for_batch(
        self,
        *,
        targets: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        num_anchors_per_level: list[int],
        dhm: DetectionHysteresisMemory,
    ) -> None:
        for image_index, target in enumerate(targets):
            if self._is_replay_crop_target(target):
                continue
            raw = self._compute_raw_losses_for_image(
                image_index=image_index,
                target=target,
                head_outputs=head_outputs,
                anchors_per_image=anchors[image_index],
                matched_idxs_per_image=matched_idxs[image_index],
            )
            records = self._lookup_dhm_gt_records(
                dhm=dhm,
                target=target,
                image_shape=image_shapes[image_index],
                image_index=image_index,
            )
            self._record_dhm_assignment_statistics(
                raw=raw,
                dhm=dhm,
                records=records,
                anchors_per_image=anchors[image_index],
                num_anchors_per_level=num_anchors_per_level,
            )

    @torch.no_grad()
    def _record_dhm_assignment_statistics(
        self,
        *,
        raw: dict[str, torch.Tensor],
        dhm: DetectionHysteresisMemory,
        records: list[DHMRecord | None],
        anchors_per_image: torch.Tensor,
        num_anchors_per_level: list[int],
    ) -> None:
        if not records:
            return
        observations = self._build_dhm_assignment_observations(
            raw=raw,
            anchors_per_image=anchors_per_image,
            num_anchors_per_level=num_anchors_per_level,
        )
        if observations:
            dhm.record_assignment_observations(
                records=records,
                observations=observations,
                epoch=dhm.current_epoch,
            )

    @torch.no_grad()
    def _build_dhm_assignment_observations(
        self,
        *,
        raw: dict[str, torch.Tensor],
        anchors_per_image: torch.Tensor,
        num_anchors_per_level: list[int],
    ) -> list[dict[str, object]]:
        assignments = raw["assignments"]
        gt_boxes = raw["gt_boxes"]
        num_gt = int(gt_boxes.shape[0])
        if num_gt == 0:
            return []

        anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) * 0.5
        anchor_sizes = (anchors_per_image[:, 2] - anchors_per_image[:, 0]).clamp_min(1.0e-6)
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        gt_wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp_min(1.0)
        gt_scale = gt_wh.max(dim=1).values.clamp_min(1.0)

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
        center_dist = (anchor_centers[:, None, :] - gt_centers[None]).abs().max(dim=2).values
        near_candidate = (
            inside_box
            & scale_match
            & (center_dist < float(self.center_sampling_radius) * anchor_sizes[:, None])
        )

        level_ids = self._fcos_level_ids(
            num_anchors_per_level=num_anchors_per_level,
            device=assignments.device,
        )
        observations: list[dict[str, object]] = []
        for gt_index in range(num_gt):
            pos_mask = assignments == int(gt_index)
            pos_count = int(pos_mask.sum().item())
            level_counts: dict[str, int] = {}
            if pos_count > 0:
                selected_levels = level_ids[pos_mask]
                for level_index in torch.unique(selected_levels).detach().cpu().tolist():
                    count = int((selected_levels == int(level_index)).sum().item())
                    level_counts[f"P{int(level_index) + 3}"] = count
                normalized_center_dist = (
                    center_dist[pos_mask, gt_index] / gt_scale[gt_index]
                )
                center_dist_value = float(normalized_center_dist.mean().item())
                ctr_target_value = float(raw["ctr_targets"][pos_mask].mean().item())
                cls_loss_value = float(raw["cls_losses"][pos_mask].mean().item())
                box_loss_value = float(raw["reg_losses"][pos_mask].mean().item())
                ctr_loss_value = float(raw["ctr_losses"][pos_mask].mean().item())
            else:
                center_dist_value = 0.0
                ctr_target_value = 0.0
                cls_loss_value = 0.0
                box_loss_value = 0.0
                ctr_loss_value = 0.0

            candidate_mask = near_candidate[:, gt_index]
            near_candidate_count = int(candidate_mask.sum().item())
            near_negative_count = int((candidate_mask & (assignments < 0)).sum().item())
            ambiguous_count = int(
                (candidate_mask & (assignments >= 0) & (assignments != int(gt_index))).sum().item()
            )
            observations.append(
                {
                    "pos_count": pos_count,
                    "level_pos_counts": level_counts,
                    "center_dist": center_dist_value,
                    "centerness_target": ctr_target_value,
                    "cls_loss": cls_loss_value,
                    "box_loss": box_loss_value,
                    "ctr_loss": ctr_loss_value,
                    "near_candidate_count": near_candidate_count,
                    "near_negative_count": near_negative_count,
                    "ambiguous_assigned_elsewhere": ambiguous_count,
                }
            )
        return observations

    def _fcos_level_ids(
        self,
        *,
        num_anchors_per_level: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        return torch.cat(
            [
                torch.full((int(count),), level_index, dtype=torch.int64, device=device)
                for level_index, count in enumerate(num_anchors_per_level)
            ],
            dim=0,
        )

    def _lookup_dhm_gt_records(
        self,
        *,
        dhm: DetectionHysteresisMemory,
        target: dict[str, torch.Tensor],
        image_shape: tuple[int, int],
        image_index: int,
    ) -> list[DHMRecord | None]:
        del image_shape

        gt_boxes = target["boxes"]
        num_gt = int(gt_boxes.shape[0])
        if num_gt == 0:
            return []

        image_id = target.get("image_id", torch.tensor(image_index, device=gt_boxes.device))
        image_records = dhm.get_image_records(image_id)
        result: list[DHMRecord | None] = [None for _ in range(num_gt)]
        if not image_records:
            return result

        gt_ids = self._get_gt_ids(target)
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
        ctr_targets_all = cls_logits.new_zeros((cls_logits.shape[0],))

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
            ctr_targets_all[pos_mask] = ctr_targets
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
            "ctr_targets": ctr_targets_all,
            "bbox_ctrness": bbox_ctrness,
            "bbox_regression": bbox_regression,
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
            "transformed_targets": transformed_targets,
            "transformed_image_sizes": transformed_images.image_sizes,
        }

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

    def _get_gt_ids(self, target: dict[str, torch.Tensor]) -> torch.Tensor | None:
        for key in ("gt_ids", "annotation_ids", "ann_ids"):
            value = target.get(key)
            if isinstance(value, torch.Tensor):
                return value
        return None

    @staticmethod
    def _is_replay_crop_target(target: dict[str, torch.Tensor]) -> bool:
        value = target.get("is_replay_crop")
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            return bool(value.detach().flatten()[0].item())
        return bool(value)

    def _get_dhm(self) -> DetectionHysteresisMemory | None:
        if self._dhm_ref is None:
            return None
        return self._dhm_ref()

    def _get_dhmr(self) -> DHMRepairModule | None:
        if self._dhmr_ref is None:
            return None
        return self._dhmr_ref()


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
        dhm: DetectionHysteresisMemory | None = None,
        dhmr: DHMRepairModule | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dhm = dhm
        self.dhmr = dhmr

        backbone = build_backbone_with_fpn(
            cfg,
            extra_blocks=LastLevelP6P7(256, 256),
            pre_neck=pre_neck,
            post_neck=post_neck,
            returned_layers=[2, 3, 4],
        )

        head = cfg.get("head", {})
        transform_cfg = cfg.get("transform", {})

        self.model = DHMFCOS(
            backbone=backbone,
            num_classes=cfg.get("num_classes", 91),
            min_size=transform_cfg.get("min_size", 800),
            max_size=transform_cfg.get("max_size", 1333),
            score_thresh=head.get("score_thresh", 0.2),
            nms_thresh=head.get("nms_thresh", 0.6),
            detections_per_img=head.get("detections_per_img", 100),
            topk_candidates=head.get("topk_candidates", 1000),
            dhm=dhm,
            dhmr=dhmr,
            **kwargs,
        )

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
        dhm: DetectionHysteresisMemory | None = None,
        dhmr: DHMRepairModule | None = None,
        **kwargs,
    ) -> "FCOSWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            dhm=dhm,
            dhmr=dhmr,
            **kwargs,
        )
