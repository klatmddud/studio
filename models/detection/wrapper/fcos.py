from __future__ import annotations

"""FCOS wrapper: YAML config -> torchvision FCOS."""

import weakref
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from torchvision.models.detection import FCOS
from torchvision.ops import boxes as box_ops, generalized_box_iou_loss, sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from modules.nn import (
    CounterfactualFeaturePerturbation,
    MDMBObservation,
    SoftCounterfactualAssignment,
    normalize_xyxy_boxes,
    select_topk_indices,
)
from modules.nn.mdmb import MissedDetectionMemoryBank
from ops import compute_cfp_loss_dict, compute_sca_loss_dict

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


class MDMBFCOS(FCOS):
    """FCOS variant that records point-level observations into MDMB during training."""

    def __init__(
        self,
        *args,
        mdmb: MissedDetectionMemoryBank | None = None,
        cfp: CounterfactualFeaturePerturbation | None = None,
        sca: SoftCounterfactualAssignment | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mdmb_ref = weakref.ref(mdmb) if mdmb is not None else None
        self._cfp_ref = weakref.ref(cfp) if cfp is not None else None
        self._sca_ref = weakref.ref(sca) if sca is not None else None

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
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
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

        losses = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            matched_idxs = self._match_anchors_to_targets(targets, anchors, num_anchors_per_level)
            losses = self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
            candidate_batches = self._observe_mdmb(
                feature_list=feature_list,
                head_outputs=head_outputs,
                anchors=anchors,
                matched_idxs=matched_idxs,
                targets=targets,
                image_shapes=images.image_sizes,
            )
            sca_losses = self._compute_sca_losses(
                head_outputs=head_outputs,
                candidate_batches=candidate_batches,
            )
            if sca_losses:
                losses.update(sca_losses)
            cfp_losses = self._compute_cfp_losses(
                feature_list=feature_list,
                head_outputs=head_outputs,
                anchors=anchors,
                targets=targets,
                candidate_batches=candidate_batches,
            )
            if cfp_losses:
                losses.update(cfp_losses)
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

    def _observe_mdmb(
        self,
        *,
        feature_list: list[torch.Tensor],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        mdmb = self._get_mdmb()
        if mdmb is None:
            return []

        candidate_batches = self._collect_candidate_batches(
            mdmb=mdmb,
            feature_list=feature_list,
            head_outputs=head_outputs,
            anchors=anchors,
            matched_idxs=matched_idxs,
            targets=targets,
            image_shapes=image_shapes,
        )
        observations = [
            observation
            for batch in candidate_batches
            for observation in batch["observations"]
        ]
        if observations:
            mdmb.update(observations)
        return candidate_batches

    def _collect_candidate_batches(
        self,
        *,
        mdmb: MissedDetectionMemoryBank,
        feature_list: list[torch.Tensor],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        flat_features = self._flatten_feature_points(feature_list)
        cls_logits = head_outputs["cls_logits"].detach()
        bbox_regression = head_outputs["bbox_regression"].detach()
        ctrness = head_outputs["bbox_ctrness"].detach().squeeze(dim=2)
        combined_scores = torch.sqrt(torch.sigmoid(cls_logits) * torch.sigmoid(ctrness.unsqueeze(-1)))
        pred_scores, pred_labels = combined_scores.max(dim=-1)

        max_candidates = mdmb.config.max_entries_per_image * 4
        candidate_batches: list[dict[str, Any]] = []
        for image_index, (
            anchors_per_image,
            matched_idxs_per_image,
            target_per_image,
            image_shape,
        ) in enumerate(zip(anchors, matched_idxs, targets, image_shapes, strict=True)):
            gt_boxes = target_per_image["boxes"]
            gt_labels = target_per_image["labels"]
            if gt_boxes.numel() == 0:
                continue

            pred_boxes = self.box_coder.decode(bbox_regression[image_index], anchors_per_image)
            pred_boxes = box_ops.clip_boxes_to_image(pred_boxes, image_shape)
            iou_matrix = box_ops.box_iou(gt_boxes, pred_boxes)
            iou_max, nearest_gt = iou_matrix.max(dim=0)
            gt_classes = gt_labels[nearest_gt]
            point_indices = torch.arange(
                anchors_per_image.shape[0],
                device=anchors_per_image.device,
            )
            gt_scores = combined_scores[image_index][point_indices, gt_classes]

            detected_mask = (
                (matched_idxs_per_image >= 0)
                & (pred_labels[image_index] == gt_classes)
                & (pred_scores[image_index] >= mdmb.config.detection_score_threshold)
            )
            candidate_mask = (iou_max >= mdmb.config.iou_low) | detected_mask
            candidate_indices = torch.where(candidate_mask)[0]
            if candidate_indices.numel() == 0:
                continue

            priority = iou_max[candidate_indices] + gt_scores[candidate_indices]
            keep_order = select_topk_indices(priority, k=max_candidates)
            candidate_indices = candidate_indices[keep_order]

            normalized_regions = normalize_xyxy_boxes(
                anchors_per_image[candidate_indices],
                image_shape,
            )
            image_id = int(target_per_image["image_id"].item())
            gt_boxes_selected = gt_boxes[nearest_gt[candidate_indices]]
            gt_classes_selected = gt_classes[candidate_indices]
            detected_selected = detected_mask[candidate_indices]
            positive_assigned = matched_idxs_per_image[candidate_indices] >= 0
            point_features = flat_features[image_index][candidate_indices]
            base_scores = gt_scores[candidate_indices]
            anchors_selected = anchors_per_image[candidate_indices]

            observations: list[MDMBObservation] = []
            track_ids: list[str] = []
            for local_index, candidate_index in enumerate(candidate_indices.tolist()):
                detected = bool(detected_mask[candidate_index].item())
                effective_iou = float(iou_max[candidate_index].item())
                if detected:
                    effective_iou = max(effective_iou, mdmb.config.iou_high)
                track_id = mdmb.make_track_id(
                    image_id=image_id,
                    region_coords=normalized_regions[local_index],
                    source="point",
                    gt_class=int(gt_classes[candidate_index].item()),
                )
                track_ids.append(track_id)
                observations.append(
                    MDMBObservation(
                        image_id=image_id,
                        region_coords=normalized_regions[local_index],
                        iou_max=effective_iou,
                        cls_score=float(gt_scores[candidate_index].item()),
                        feature_vec=point_features[local_index],
                        gt_class=int(gt_classes[candidate_index].item()),
                        detected=detected,
                        track_id=track_id,
                        source="point",
                    )
                )

            candidate_batches.append(
                {
                    "image_index": image_index,
                    "candidate_indices": candidate_indices,
                    "features": point_features,
                    "anchors": anchors_selected,
                    "gt_boxes": gt_boxes_selected,
                    "gt_classes": gt_classes_selected.to(dtype=torch.int64),
                    "base_scores": base_scores,
                    "detected_mask": detected_selected,
                    "positive_assigned": positive_assigned,
                    "track_ids": track_ids,
                    "observations": observations,
                }
            )

        return candidate_batches

    def _compute_sca_losses(
        self,
        *,
        head_outputs: dict[str, torch.Tensor],
        candidate_batches: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor] | None:
        mdmb = self._get_mdmb()
        sca = self._get_sca()
        if mdmb is None or sca is None or not candidate_batches:
            return None

        selected_logits: list[torch.Tensor] = []
        selected_ious: list[float] = []
        selected_miss_counts: list[int] = []
        for batch in candidate_batches:
            image_index = int(batch["image_index"])
            for position, track_id in enumerate(batch["track_ids"]):
                entry = mdmb.get(track_id)
                if entry is None or not entry.is_chronic_miss(mdmb.config):
                    continue
                if bool(batch["detected_mask"][position].item()):
                    continue
                if bool(batch["positive_assigned"][position].item()):
                    continue

                flat_index = int(batch["candidate_indices"][position].item())
                gt_class = int(batch["gt_classes"][position].item())
                selected_logits.append(head_outputs["cls_logits"][image_index, flat_index, gt_class])
                selected_ious.append(float(entry.iou_max))
                selected_miss_counts.append(int(entry.miss_count))

        if not selected_logits:
            return None

        logits_tensor = torch.stack(selected_logits, dim=0)
        iou_tensor = logits_tensor.new_tensor(selected_ious)
        miss_count_tensor = logits_tensor.new_tensor(selected_miss_counts)
        soft_targets = sca.compute_soft_weights(
            iou_tensor,
            miss_count_tensor,
            iou_low=mdmb.config.iou_low,
            iou_high=mdmb.config.iou_high,
            max_miss_count=max(selected_miss_counts),
        ).to(dtype=logits_tensor.dtype)
        return compute_sca_loss_dict(
            logits=logits_tensor,
            soft_targets=soft_targets,
            config=sca.config,
        )

    def _compute_cfp_losses(
        self,
        *,
        feature_list: list[torch.Tensor],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        candidate_batches: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor] | None:
        mdmb = self._get_mdmb()
        cfp = self._get_cfp()
        if mdmb is None or cfp is None or not candidate_batches:
            return None

        feature_shapes = [(feature.shape[-2], feature.shape[-1]) for feature in feature_list]
        selected_tasks: list[dict[str, Any]] = []
        selected_features: list[torch.Tensor] = []
        for batch in candidate_batches:
            for position, track_id in enumerate(batch["track_ids"]):
                entry = mdmb.get(track_id)
                if entry is None or not entry.is_chronic_miss(mdmb.config):
                    continue
                if bool(batch["detected_mask"][position].item()):
                    continue
                # CFP in FCOS is meant for chronic near-miss points that still
                # fail the current positive assignment, not for already-positive
                # hard samples.
                if bool(batch["positive_assigned"][position].item()):
                    continue
                selected_tasks.append(
                    {
                        "image_index": int(batch["image_index"]),
                        "flat_index": int(batch["candidate_indices"][position].item()),
                        "anchor": batch["anchors"][position],
                        "gt_box": batch["gt_boxes"][position],
                        "gt_class": batch["gt_classes"][position],
                        "base_score": batch["base_scores"][position],
                    }
                )
                selected_features.append(batch["features"][position])

        if not selected_features:
            return None

        chronic_features = torch.stack(selected_features, dim=0)
        # FCOS CFP should keep the backbone feature graph alive so L_CFP can
        # update the original FPN features instead of only training the branch.
        cfp_output = cfp(chronic_features, detach_input=False)
        cf_feature_list = [feature.clone() for feature in feature_list]
        for task, perturbed_feature in zip(
            selected_tasks,
            cfp_output.perturbed_features,
            strict=True,
        ):
            level, y, x = self._flat_index_to_feature_location(
                task["flat_index"],
                feature_shapes,
            )
            cf_feature_list[level][task["image_index"], :, y, x] = perturbed_feature

        cf_head_outputs = self.head(cf_feature_list)
        cf_cls_logits = []
        cf_bbox_regression = []
        cf_bbox_ctrness = []
        gt_classes = []
        gt_boxes = []
        anchor_boxes = []
        base_scores = []
        for task in selected_tasks:
            image_index = task["image_index"]
            flat_index = task["flat_index"]
            cf_cls_logits.append(cf_head_outputs["cls_logits"][image_index, flat_index])
            cf_bbox_regression.append(cf_head_outputs["bbox_regression"][image_index, flat_index])
            cf_bbox_ctrness.append(cf_head_outputs["bbox_ctrness"][image_index, flat_index])
            gt_classes.append(task["gt_class"])
            gt_boxes.append(task["gt_box"])
            anchor_boxes.append(task["anchor"])
            base_scores.append(task["base_score"])

        cf_cls_logits_tensor = torch.stack(cf_cls_logits, dim=0)
        cf_bbox_regression_tensor = torch.stack(cf_bbox_regression, dim=0)
        cf_bbox_ctrness_tensor = torch.stack(cf_bbox_ctrness, dim=0).squeeze(dim=1)
        gt_classes_tensor = torch.stack(gt_classes, dim=0).to(dtype=torch.int64)
        gt_boxes_tensor = torch.stack(gt_boxes, dim=0)
        anchor_boxes_tensor = torch.stack(anchor_boxes, dim=0)
        base_scores_tensor = torch.stack(base_scores, dim=0)

        num_selected = max(1, gt_classes_tensor.numel())
        gt_class_targets = torch.zeros_like(cf_cls_logits_tensor)
        gt_class_targets[
            torch.arange(num_selected, device=gt_classes_tensor.device),
            gt_classes_tensor,
        ] = 1.0
        loss_cls = sigmoid_focal_loss(
            cf_cls_logits_tensor,
            gt_class_targets,
            reduction="sum",
        ) / num_selected

        pred_boxes = self.box_coder.decode(cf_bbox_regression_tensor, anchor_boxes_tensor)
        loss_bbox_reg = generalized_box_iou_loss(
            pred_boxes,
            gt_boxes_tensor,
            reduction="sum",
        ) / num_selected

        bbox_reg_targets = self.box_coder.encode(anchor_boxes_tensor, gt_boxes_tensor)
        left_right = bbox_reg_targets[:, [0, 2]]
        top_bottom = bbox_reg_targets[:, [1, 3]]
        gt_ctrness_targets = torch.sqrt(
            (left_right.min(dim=-1).values / left_right.max(dim=-1).values)
            * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values)
        )
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            cf_bbox_ctrness_tensor,
            gt_ctrness_targets,
            reduction="sum",
        ) / num_selected

        cf_scores = torch.sqrt(
            torch.sigmoid(
                cf_cls_logits_tensor[
                    torch.arange(num_selected, device=gt_classes_tensor.device),
                    gt_classes_tensor,
                ]
            )
            * torch.sigmoid(cf_bbox_ctrness_tensor)
        )
        return compute_cfp_loss_dict(
            detection_loss={
                "classification": loss_cls,
                "bbox_regression": loss_bbox_reg,
                "bbox_ctrness": loss_bbox_ctrness,
            },
            delta=cfp_output,
            config=cfp.config,
            cf_scores=cf_scores,
            base_scores=base_scores_tensor,
        )

    def _flatten_feature_points(self, feature_list: list[torch.Tensor]) -> torch.Tensor:
        flattened = []
        for features in feature_list:
            flattened.append(features.flatten(start_dim=2).permute(0, 2, 1))
        return torch.cat(flattened, dim=1)

    def _flat_index_to_feature_location(
        self,
        flat_index: int,
        feature_shapes: list[tuple[int, int]],
    ) -> tuple[int, int, int]:
        offset = int(flat_index)
        for level_index, (height, width) in enumerate(feature_shapes):
            num_points = height * width
            if offset < num_points:
                return level_index, offset // width, offset % width
            offset -= num_points
        raise IndexError(f"FCOS flat index {flat_index} is out of range.")

    def _get_mdmb(self) -> MissedDetectionMemoryBank | None:
        if self._mdmb_ref is None:
            return None
        return self._mdmb_ref()

    def _get_cfp(self) -> CounterfactualFeaturePerturbation | None:
        if self._cfp_ref is None:
            return None
        return self._cfp_ref()

    def _get_sca(self) -> SoftCounterfactualAssignment | None:
        if self._sca_ref is None:
            return None
        return self._sca_ref()


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
        cfp: CounterfactualFeaturePerturbation | None = None,
        sca: SoftCounterfactualAssignment | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cfp = cfp
        self.mdmb = mdmb
        self.sca = sca

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
            cfp=cfp,
            sca=sca,
            **kwargs,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        cfp: CounterfactualFeaturePerturbation | None = None,
        sca: SoftCounterfactualAssignment | None = None,
        **kwargs,
    ) -> "FCOSWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            mdmb=mdmb,
            cfp=cfp,
            sca=sca,
            **kwargs,
        )
