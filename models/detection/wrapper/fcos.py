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

from modules.nn import MDMBSelectiveLoss
from modules.nn.far import ForgettingAwareReplay
from modules.nn.mce import MissConditionedEmbedding
from modules.nn.mdmb import MissedDetectionMemoryBank, normalize_xyxy_boxes

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


class MDMBFCOS(FCOS):
    """
    FCOS variant with MDMB-guided loss reweighting and optional FAR/MCE modules.

    During training the wrapper can replace the standard aggregated FCOS loss
    with a per-point loss decomposition, then reweight positives / suppress
    nearby negatives using the MDMB bank. After optimizer.step(), it runs an
    extra no-grad inference pass to refresh the MDMB state from final detections
    and (if enabled) to update the FAR anchor bank.

    MCE (Miss-Conditioned class Embedding) multiplies per-point weights by a
    per-GT factor derived from class prototype similarity and miss streak ratio.
    """

    def __init__(
        self,
        *args,
        mdmb: MissedDetectionMemoryBank | None = None,
        recall: MDMBSelectiveLoss | None = None,
        far: ForgettingAwareReplay | None = None,
        mce: MissConditionedEmbedding | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mdmb_ref = weakref.ref(mdmb) if mdmb is not None else None
        self._recall_ref = weakref.ref(recall) if recall is not None else None
        self._far_ref = weakref.ref(far) if far is not None else None
        self._mce_ref = weakref.ref(mce) if mce is not None else None

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
            mdmb = self._get_mdmb()
            recall = self._get_recall()
            mce = self._get_mce()
            use_weighted = mdmb is not None and (recall is not None or mce is not None)
            if not use_weighted:
                losses = self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
            else:
                losses = self._compute_mdmb_weighted_loss_dict(
                    targets=targets,
                    head_outputs=head_outputs,
                    anchors=anchors,
                    matched_idxs=matched_idxs,
                    image_shapes=images.image_sizes,
                    features=features,
                )

            far = self._get_far()
            if far is not None and mdmb is not None and far.should_apply():
                far_loss = far.compute_loss(
                    image_ids=[
                        target.get(
                            "image_id",
                            torch.tensor(idx, device=images.tensors.device),
                        )
                        for idx, target in enumerate(targets)
                    ],
                    gt_boxes_list=[target["boxes"] for target in targets],
                    gt_labels_list=[target["labels"] for target in targets],
                    features=features,
                    image_shapes=images.image_sizes,
                    mdmb=mdmb,
                )
                if far_loss.requires_grad or float(far_loss.detach().item()) != 0.0:
                    losses = dict(losses)
                    losses["far"] = far_loss
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

    def _compute_mdmb_weighted_loss_dict(
        self,
        *,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        mdmb = self._get_mdmb()
        recall = self._get_recall()
        mce = self._get_mce()
        if mdmb is None:
            raise RuntimeError("MDMB weighted loss requested without MDMB module.")

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
            point_boxes = raw["point_boxes"]
            gt_boxes = raw["gt_boxes"]
            gt_labels = raw["gt_labels"]

            image_id = target.get(
                "image_id",
                torch.tensor(image_index, device=assignments.device),
            )

            # RECALL: per-point weights from miss type
            if recall is not None:
                missed_set = mdmb.get_missed_set_with_type(
                    image_id,
                    gt_boxes,
                    gt_labels,
                    image_shape=image_shapes[image_index],
                )
                weights, valid = recall.compute_weights(
                    point_gt_indices=assignments,
                    point_boxes=point_boxes,
                    gt_boxes=gt_boxes,
                    missed_set=missed_set,
                    device=assignments.device,
                )
            else:
                num_points = int(assignments.numel())
                weights = torch.ones(num_points, dtype=torch.float32, device=assignments.device)
                valid = torch.ones(num_points, dtype=torch.bool, device=assignments.device)

            # MCE: per-GT weights from prototype similarity × streak ratio
            if mce is not None:
                image_key = (
                    str(image_id.item())
                    if isinstance(image_id, torch.Tensor)
                    else str(image_id)
                )
                gt_boxes_norm = normalize_xyxy_boxes(gt_boxes, image_shapes[image_index])
                pooled = mce.pool_features(features, [gt_boxes], [image_shapes[image_index]])
                mce_gt_weights = mce.compute_gt_weights(
                    gt_labels=gt_labels,
                    gt_boxes_norm=gt_boxes_norm,
                    pooled_features=pooled[0],
                    mdmb=mdmb,
                    image_key=image_key,
                )
                pos_indices = assignments.clamp(min=0)
                mce_point_weights = mce_gt_weights[pos_indices].to(
                    device=assignments.device, dtype=weights.dtype
                )
                pos_mask_bool = assignments >= 0
                weights = weights.clone()
                weights[pos_mask_bool] = weights[pos_mask_bool] * mce_point_weights[pos_mask_bool]

            pos_mask = assignments >= 0
            total_pos += int(pos_mask.sum().item())

            weights = weights.to(dtype=raw["cls_losses"].dtype)
            valid_float = valid.to(dtype=raw["cls_losses"].dtype)

            cls_sum = cls_sum + (raw["cls_losses"] * weights * valid_float).sum()
            if bool(pos_mask.any().item()):
                pos_weights = weights[pos_mask]
                reg_sum = reg_sum + (raw["reg_losses"][pos_mask] * pos_weights).sum()
                ctr_sum = ctr_sum + (raw["ctr_losses"][pos_mask] * pos_weights).sum()

        normalizer = cls_sum.new_tensor(float(max(total_pos, 1)))
        return {
            "classification": cls_sum / normalizer,
            "bbox_regression": reg_sum / normalizer,
            "bbox_ctrness": ctr_sum / normalizer,
        }

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

        raw_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="none").sum(dim=-1)
        return {
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

    @torch.no_grad()
    def flush_far_update(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        *,
        epoch: int | None = None,
    ) -> None:
        far = self._get_far()
        mdmb = self._get_mdmb()
        if far is None or mdmb is None or not targets:
            return
        if not far.should_apply(epoch=epoch):
            return

        image_ids = [
            target.get("image_id", torch.tensor(index, device=images[index].device))
            for index, target in enumerate(targets)
        ]

        was_training = self.training
        try:
            self.eval()
            cloned_targets = [
                {key: value for key, value in target.items()} for target in targets
            ]
            transformed_images, transformed_targets = self.transform(images, cloned_targets)
            features = self.backbone(transformed_images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

            gt_boxes_list = [target["boxes"] for target in transformed_targets]
            gt_labels_list = [target["labels"] for target in transformed_targets]

            far.update_anchors(
                image_ids=image_ids,
                gt_boxes_list=gt_boxes_list,
                gt_labels_list=gt_labels_list,
                features=features,
                image_shapes=transformed_images.image_sizes,
                mdmb=mdmb,
                epoch=epoch,
            )
        finally:
            if was_training:
                self.train()

    def _get_mdmb(self) -> MissedDetectionMemoryBank | None:
        if self._mdmb_ref is None:
            return None
        return self._mdmb_ref()

    def _get_recall(self) -> MDMBSelectiveLoss | None:
        if self._recall_ref is None:
            return None
        return self._recall_ref()

    def _get_far(self) -> ForgettingAwareReplay | None:
        if self._far_ref is None:
            return None
        return self._far_ref()

    def _get_mce(self) -> MissConditionedEmbedding | None:
        if self._mce_ref is None:
            return None
        return self._mce_ref()


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
        recall: MDMBSelectiveLoss | None = None,
        far: ForgettingAwareReplay | None = None,
        mce: MissConditionedEmbedding | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mdmb = mdmb
        self.recall = recall
        self.far = far
        self.mce = mce

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
            recall=recall,
            far=far,
            mce=mce,
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
        if self.mdmb is None:
            return
        epoch = None if epoch_index is None else int(epoch_index) + 1
        self.model.flush_mdmb_update(images, targets, epoch=epoch)
        if self.far is not None:
            self.model.flush_far_update(images, targets, epoch=epoch)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        recall: MDMBSelectiveLoss | None = None,
        far: ForgettingAwareReplay | None = None,
        mce: MissConditionedEmbedding | None = None,
        **kwargs,
    ) -> "FCOSWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            mdmb=mdmb,
            recall=recall,
            far=far,
            mce=mce,
            **kwargs,
        )
