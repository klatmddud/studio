from __future__ import annotations

"""FCOS wrapper: YAML config -> torchvision FCOS."""

import weakref
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.detection import FCOS
from torchvision.ops import roi_align
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from modules.nn import (
    CounterfactualFeaturePerturbation,
    MissedObjectDirectSupervision,
    SoftCounterfactualAssignment,
)
from modules.nn.mdmb import MissedDetectionMemoryBank
from ops import compute_mods_loss_dict

from ._base import BaseDetectionWrapper, build_backbone_with_fpn, load_cfg


class MDMBFCOS(FCOS):
    """
    FCOS variant with MDMB v2 support.

    During the training forward pass it computes the standard FCOS loss only.
    After optimizer.step(), the wrapper triggers an extra no-grad inference pass
    to update MDMB from the final post-NMS detections of the current batch.
    """

    def __init__(
        self,
        *args,
        mdmb: MissedDetectionMemoryBank | None = None,
        cfp: CounterfactualFeaturePerturbation | None = None,
        mods: MissedObjectDirectSupervision | None = None,
        sca: SoftCounterfactualAssignment | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mdmb_ref = weakref.ref(mdmb) if mdmb is not None else None
        self._cfp_ref = weakref.ref(cfp) if cfp is not None else None
        self._mods_ref = weakref.ref(mods) if mods is not None else None
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

        losses = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            matched_idxs = self._match_anchors_to_targets(targets, anchors, num_anchors_per_level)
            losses = self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
            mods_losses = self._compute_mods_losses(
                feature_list=feature_list,
                targets=targets,
                image_shapes=images.image_sizes,
            )
            if mods_losses:
                losses.update(mods_losses)
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

    def _compute_mods_losses(
        self,
        *,
        feature_list: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        image_shapes: list[tuple[int, int]],
    ) -> dict[str, torch.Tensor] | None:
        mdmb = self._get_mdmb()
        mods = self._get_mods()
        if mdmb is None or mods is None or not targets:
            return None
        if len(feature_list) != len(mods.config.strides):
            raise ValueError(
                "FCOS MODS expects the configured strides to match the number of FPN levels. "
                f"Got {len(mods.config.strides)} strides and {len(feature_list)} feature maps."
            )

        image_ids = [
            target.get("image_id", torch.tensor(index, device=feature_list[0].device))
            for index, target in enumerate(targets)
        ]
        missed_entries_batch = [mdmb.get(image_id) for image_id in image_ids]
        mods_targets = mods.collect_targets(
            image_ids=image_ids,
            missed_entries_batch=missed_entries_batch,
            image_shapes=image_shapes,
            device=feature_list[0].device,
            training=self.training,
        )
        if mods_targets.is_empty():
            return None

        cls_logits_chunks: list[torch.Tensor] = []
        cls_target_chunks: list[torch.Tensor] = []
        reg_pred_chunks: list[torch.Tensor] = []
        reg_target_chunks: list[torch.Tensor] = []

        for level_index, (feature_map, stride) in enumerate(
            zip(feature_list, mods.config.strides, strict=True)
        ):
            level_mask = mods_targets.level_indices == level_index
            if not bool(level_mask.any().item()):
                continue

            level_image_indices = mods_targets.image_indices[level_mask].to(
                device=feature_map.device,
                dtype=torch.float32,
            )
            level_boxes = mods_targets.boxes_abs[level_mask].to(
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
            rois = torch.cat((level_image_indices.unsqueeze(1), level_boxes), dim=1)
            pooled_features = roi_align(
                feature_map,
                rois,
                output_size=mods.config.roi_output_size,
                spatial_scale=1.0 / float(stride),
                sampling_ratio=mods.config.sampling_ratio,
                aligned=mods.config.aligned,
            )

            cls_logits = self.head.classification_head([pooled_features])
            cls_logits_chunks.append(self._reduce_dense_predictions(cls_logits))
            cls_target_chunks.append(mods_targets.class_ids[level_mask].to(feature_map.device))

            if mods.config.lambda_reg > 0.0:
                bbox_regression, _ = self.head.regression_head([pooled_features])
                reg_pred_chunks.append(self._reduce_dense_predictions(bbox_regression))
                reg_target_chunks.append(
                    mods_targets.boxes_norm[level_mask].to(
                        device=feature_map.device,
                        dtype=feature_map.dtype,
                    )
                )

        if not cls_logits_chunks:
            return None

        return compute_mods_loss_dict(
            cls_logits=torch.cat(cls_logits_chunks, dim=0),
            gt_classes=torch.cat(cls_target_chunks, dim=0),
            reg_pred=None if not reg_pred_chunks else torch.cat(reg_pred_chunks, dim=0),
            reg_target=None if not reg_target_chunks else torch.cat(reg_target_chunks, dim=0),
            config=mods.config,
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
        mdmb.update(
            image_ids=image_ids,
            pred_boxes_list=pred_boxes_list,
            gt_boxes_list=gt_boxes_list,
            gt_labels_list=gt_labels_list,
            image_shapes=image_shapes,
            epoch=epoch,
        )

    def _get_mdmb(self) -> MissedDetectionMemoryBank | None:
        if self._mdmb_ref is None:
            return None
        return self._mdmb_ref()

    def _get_mods(self) -> MissedObjectDirectSupervision | None:
        if self._mods_ref is None:
            return None
        return self._mods_ref()

    def _reduce_dense_predictions(self, tensor: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if isinstance(tensor, (list, tuple)):
            if len(tensor) != 1:
                raise ValueError(
                    "FCOS MODS expected a single pooled feature prediction tensor per level. "
                    f"Got {len(tensor)} tensors."
                )
            tensor = tensor[0]
        if tensor.ndim == 3:
            return tensor.mean(dim=1)
        if tensor.ndim == 2:
            return tensor
        raise ValueError(
            "FCOS MODS expected a dense prediction tensor with shape [N, HWA, C] or [N, C]. "
            f"Got {tuple(tensor.shape)}."
        )


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
        mods: MissedObjectDirectSupervision | None = None,
        sca: SoftCounterfactualAssignment | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cfp = cfp
        self.mdmb = mdmb
        self.mods = mods
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
            mods=mods,
            sca=sca,
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

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        cfp: CounterfactualFeaturePerturbation | None = None,
        mods: MissedObjectDirectSupervision | None = None,
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
            mods=mods,
            sca=sca,
            **kwargs,
        )
