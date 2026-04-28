from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.ops import boxes as box_ops

from .common import normalize_arch
from .dhm import DHMRecord


_BORDER_REFINEMENT_TRANSITIONS = ("FN_LOC->FN_LOC", "TP->FN_LOC")
_BORDER_GEOMETRY_DIM = 9


@dataclass(frozen=True, slots=True)
class BorderRefinementConfig:
    enabled: bool = False
    feature_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    border_points_per_side: int = 5
    max_delta: float = 0.25
    max_gt_per_image: int = 16
    max_points_per_gt: int = 2
    max_points_per_batch: int = 128
    min_observations: int = 2
    min_instability: float = 0.0
    target_transitions: tuple[str, ...] = _BORDER_REFINEMENT_TRANSITIONS
    start_epoch: int = 1
    warmup_epochs: int = 0
    detach_boxes: bool = True
    giou_loss_weight: float = 0.2
    residual_loss_weight: float = 0.1
    quality_loss_weight: float = 0.1
    smooth_l1_beta: float = 0.1
    target_delta_clip: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "BorderRefinementConfig":
        data = dict(raw or {})
        raw_transitions = data.get("target_transitions", _BORDER_REFINEMENT_TRANSITIONS)
        if isinstance(raw_transitions, str):
            target_transitions = (raw_transitions,)
        elif isinstance(raw_transitions, Sequence):
            target_transitions = tuple(str(item) for item in raw_transitions)
        else:
            raise TypeError("DHM-R border_refinement.target_transitions must be a string or sequence.")
        config = cls(
            enabled=bool(data.get("enabled", False)),
            feature_dim=int(data.get("feature_dim", 256)),
            hidden_dim=int(data.get("hidden_dim", 256)),
            num_layers=int(data.get("num_layers", 2)),
            border_points_per_side=int(data.get("border_points_per_side", 5)),
            max_delta=float(data.get("max_delta", 0.25)),
            max_gt_per_image=int(data.get("max_gt_per_image", 16)),
            max_points_per_gt=int(data.get("max_points_per_gt", 2)),
            max_points_per_batch=int(data.get("max_points_per_batch", 128)),
            min_observations=int(data.get("min_observations", 2)),
            min_instability=float(data.get("min_instability", 0.0)),
            target_transitions=target_transitions,
            start_epoch=int(data.get("start_epoch", 1)),
            warmup_epochs=int(data.get("warmup_epochs", 0)),
            detach_boxes=bool(data.get("detach_boxes", True)),
            giou_loss_weight=float(data.get("giou_loss_weight", 0.2)),
            residual_loss_weight=float(data.get("residual_loss_weight", 0.1)),
            quality_loss_weight=float(data.get("quality_loss_weight", 0.1)),
            smooth_l1_beta=float(data.get("smooth_l1_beta", 0.1)),
            target_delta_clip=float(data.get("target_delta_clip", 1.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for field_name in ("feature_dim", "hidden_dim", "num_layers", "border_points_per_side"):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"DHM-R border_refinement.{field_name} must be >= 1.")
        for field_name in ("max_gt_per_image", "max_points_per_gt", "max_points_per_batch"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"DHM-R border_refinement.{field_name} must be >= 0.")
        if int(self.min_observations) < 1:
            raise ValueError("DHM-R border_refinement.min_observations must be >= 1.")
        if not 0.0 <= float(self.min_instability) <= 1.0:
            raise ValueError("DHM-R border_refinement.min_instability must satisfy 0 <= value <= 1.")
        if int(self.start_epoch) < 0:
            raise ValueError("DHM-R border_refinement.start_epoch must be >= 0.")
        if int(self.warmup_epochs) < 0:
            raise ValueError("DHM-R border_refinement.warmup_epochs must be >= 0.")
        for field_name in (
            "max_delta",
            "giou_loss_weight",
            "residual_loss_weight",
            "quality_loss_weight",
            "smooth_l1_beta",
            "target_delta_clip",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"DHM-R border_refinement.{field_name} must be >= 0.")
        if not self.target_transitions:
            raise ValueError("DHM-R border_refinement.target_transitions must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "border_points_per_side": self.border_points_per_side,
            "max_delta": self.max_delta,
            "max_gt_per_image": self.max_gt_per_image,
            "max_points_per_gt": self.max_points_per_gt,
            "max_points_per_batch": self.max_points_per_batch,
            "min_observations": self.min_observations,
            "min_instability": self.min_instability,
            "target_transitions": list(self.target_transitions),
            "start_epoch": self.start_epoch,
            "warmup_epochs": self.warmup_epochs,
            "detach_boxes": self.detach_boxes,
            "giou_loss_weight": self.giou_loss_weight,
            "residual_loss_weight": self.residual_loss_weight,
            "quality_loss_weight": self.quality_loss_weight,
            "smooth_l1_beta": self.smooth_l1_beta,
            "target_delta_clip": self.target_delta_clip,
        }


@dataclass(frozen=True, slots=True)
class DHMRConfig:
    enabled: bool = False
    border_refinement: BorderRefinementConfig = field(default_factory=BorderRefinementConfig)
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "DHMRConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        merged = dict(data)
        for key, value in model_overrides.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value

        return cls(
            enabled=bool(merged.get("enabled", False)),
            border_refinement=BorderRefinementConfig.from_mapping(merged.get("border_refinement")),
            arch=normalized_arch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "border_refinement": self.border_refinement.to_dict(),
            "arch": self.arch,
        }


class DHMRepairModule(nn.Module):
    def __init__(self, config: DHMRConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._stats: Counter[str] = Counter()
        self._border_loss_sums: Counter[str] = Counter()
        if bool(config.border_refinement.enabled):
            border = config.border_refinement
            input_dim = int(border.feature_dim) * 5 + _BORDER_GEOMETRY_DIM
            hidden_dim = int(border.hidden_dim)
            layers: list[nn.Module] = []
            for layer_index in range(int(border.num_layers)):
                in_dim = input_dim if layer_index == 0 else hidden_dim
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, 5))
            self.border_refine_head = nn.Sequential(*layers)
            final = self.border_refine_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        else:
            self.border_refine_head = None

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._stats.clear()
        self._border_loss_sums.clear()

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def border_refinement_warmup_factor(self) -> float:
        border = self.config.border_refinement
        if not self.config.enabled or not border.enabled:
            return 0.0
        if self.current_epoch < int(border.start_epoch):
            return 0.0
        warmup = int(border.warmup_epochs)
        if warmup <= 0:
            return 1.0
        progress = max(0, self.current_epoch - int(border.start_epoch) + 1)
        return min(1.0, float(progress) / float(warmup))

    def uses_border_refinement(self) -> bool:
        border = self.config.border_refinement
        return bool(
            self.border_refine_head is not None
            and self.border_refinement_warmup_factor() > 0.0
            and int(border.max_points_per_gt) > 0
            and int(border.max_points_per_batch) > 0
            and (
                float(border.giou_loss_weight) > 0.0
                or float(border.residual_loss_weight) > 0.0
                or float(border.quality_loss_weight) > 0.0
            )
        )

    def compute_border_refinement_loss(
        self,
        *,
        targets: Sequence[Mapping[str, torch.Tensor]],
        feature_maps: Sequence[torch.Tensor],
        head_outputs: Mapping[str, torch.Tensor],
        anchors: Sequence[torch.Tensor],
        matched_idxs: Sequence[torch.Tensor],
        dhm_records: Sequence[Sequence[DHMRecord | None]],
        num_anchors_per_level: Sequence[int],
        padded_shape: tuple[int, int],
        decode_boxes: Callable[..., torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.uses_border_refinement():
            return {}
        if self.border_refine_head is None or not feature_maps:
            return {}
        border = self.config.border_refinement
        first_feature = feature_maps[0]
        if first_feature.ndim != 4:
            self._stats["border_refine_skipped_bad_feature_shape"] += 1
            return {}
        if int(first_feature.shape[1]) != int(border.feature_dim):
            self._stats["border_refine_skipped_feature_dim"] += 1
            return {}

        selected: list[dict[str, torch.Tensor | int]] = []
        selected_gt = 0
        selected_points = 0
        max_points = int(border.max_points_per_batch)
        level_ids = _level_ids(num_anchors_per_level, device=first_feature.device)
        for image_index, target in enumerate(targets):
            if selected_points >= max_points:
                break
            records = dhm_records[image_index] if image_index < len(dhm_records) else []
            eligible = self._eligible_border_refinement_records(records)
            if not eligible:
                continue
            assignments = matched_idxs[image_index]
            if assignments.numel() == 0:
                continue
            anchors_per_image = anchors[image_index]
            bbox_regression = head_outputs["bbox_regression"][image_index]
            cls_logits = head_outputs["cls_logits"][image_index]
            bbox_ctrness = head_outputs["bbox_ctrness"][image_index].flatten()
            gt_boxes = target["boxes"].to(device=first_feature.device, dtype=first_feature.dtype)
            gt_labels = target["labels"].to(device=first_feature.device, dtype=torch.long)
            for gt_index, _record in eligible:
                if selected_points >= max_points:
                    break
                pos_indices = torch.where(assignments == int(gt_index))[0]
                if pos_indices.numel() == 0:
                    self._stats["border_refine_skipped_no_positive_points"] += 1
                    continue
                pred_boxes = decode_boxes(
                    box_regression=bbox_regression[pos_indices],
                    anchors=anchors_per_image[pos_indices],
                )
                pred_boxes = _clip_boxes(_sanitize_boxes(pred_boxes), padded_shape=padded_shape)
                gt_box = gt_boxes[int(gt_index)].reshape(1, 4).expand_as(pred_boxes)
                with torch.no_grad():
                    giou = torch.diagonal(
                        box_ops.generalized_box_iou(
                            pred_boxes.detach().to(dtype=torch.float32),
                            gt_box.detach().to(dtype=torch.float32),
                        )
                    )
                    order = torch.argsort(1.0 - giou, descending=True)
                remaining = max_points - selected_points
                take = min(int(border.max_points_per_gt), remaining, int(order.numel()))
                if take <= 0:
                    continue
                keep = order[:take]
                point_indices = pos_indices[keep]
                labels = gt_labels[int(gt_index)].reshape(1).expand(take)
                class_scores = cls_logits[point_indices, labels].sigmoid().detach()
                ctr_scores = bbox_ctrness[point_indices].sigmoid().detach()
                selected_boxes = pred_boxes[keep]
                if bool(border.detach_boxes):
                    selected_boxes = selected_boxes.detach()
                selected.append(
                    {
                        "image_index": int(image_index),
                        "point_indices": point_indices,
                        "level_indices": level_ids[point_indices].to(device=first_feature.device),
                        "boxes": selected_boxes.to(device=first_feature.device, dtype=first_feature.dtype),
                        "gt_boxes": gt_box[keep].to(device=first_feature.device, dtype=first_feature.dtype),
                        "class_scores": class_scores.to(device=first_feature.device, dtype=first_feature.dtype),
                        "ctr_scores": ctr_scores.to(device=first_feature.device, dtype=first_feature.dtype),
                    }
                )
                selected_points += int(take)
                selected_gt += 1

        if not selected:
            return {}

        image_indices = torch.cat(
            [
                torch.full(
                    (int(item["point_indices"].numel()),),
                    int(item["image_index"]),
                    dtype=torch.long,
                    device=first_feature.device,
                )
                for item in selected
            ],
            dim=0,
        )
        level_indices = torch.cat(
            [item["level_indices"].to(dtype=torch.long) for item in selected],
            dim=0,
        )
        boxes = torch.cat([item["boxes"] for item in selected], dim=0)
        gt_boxes = torch.cat([item["gt_boxes"] for item in selected], dim=0)
        class_scores = torch.cat([item["class_scores"] for item in selected], dim=0)
        ctr_scores = torch.cat([item["ctr_scores"] for item in selected], dim=0)
        border_features = self._sample_border_refinement_features(
            feature_maps=feature_maps,
            boxes=boxes,
            image_indices=image_indices,
            level_indices=level_indices,
            class_scores=class_scores,
            ctr_scores=ctr_scores,
            padded_shape=padded_shape,
        )
        if border_features.numel() == 0:
            return {}

        outputs = self.border_refine_head(border_features)
        raw_delta = outputs[:, :4]
        iou_logits = outputs[:, 4]
        delta = torch.tanh(raw_delta) * float(border.max_delta)
        refined_boxes = _apply_box_delta(boxes=boxes, delta=delta, padded_shape=padded_shape)
        giou = torch.diagonal(
            box_ops.generalized_box_iou(
                refined_boxes.to(dtype=torch.float32),
                gt_boxes.to(dtype=torch.float32),
            )
        ).to(dtype=refined_boxes.dtype)
        giou_loss = (1.0 - giou).clamp_min(0.0).mean()
        target_delta = _normalized_box_delta(
            boxes=boxes.detach() if bool(border.detach_boxes) else boxes,
            gt_boxes=gt_boxes,
            clip=float(border.target_delta_clip),
        )
        residual_loss = F.smooth_l1_loss(
            delta,
            target_delta,
            beta=float(border.smooth_l1_beta),
            reduction="none",
        ).mean(dim=1).mean()
        iou_targets = _aligned_box_iou(
            refined_boxes.detach().to(dtype=torch.float32),
            gt_boxes.detach().to(dtype=torch.float32),
        ).to(dtype=iou_logits.dtype)
        quality_loss = F.binary_cross_entropy_with_logits(
            iou_logits,
            iou_targets,
            reduction="mean",
        )
        factor = border_features.new_tensor(float(self.border_refinement_warmup_factor()))
        losses = {
            "dhmr_border_giou": giou_loss * float(border.giou_loss_weight) * factor,
            "dhmr_border_residual": residual_loss * float(border.residual_loss_weight) * factor,
            "dhmr_border_quality": quality_loss * float(border.quality_loss_weight) * factor,
        }
        self._record_border_refinement_loss(
            selected_points=int(boxes.shape[0]),
            selected_gt=int(selected_gt),
            giou_loss=giou_loss,
            residual_loss=residual_loss,
            quality_loss=quality_loss,
            refined_iou=iou_targets.mean(),
        )
        return {
            key: value
            for key, value in losses.items()
            if float(value.detach().abs().item()) > 0.0
        }

    def summary(self) -> dict[str, Any]:
        border_losses = int(self._stats.get("border_refine_losses", 0))
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "border_refinement": {
                "enabled": bool(self.config.border_refinement.enabled),
                "warmup_factor": self.border_refinement_warmup_factor(),
                "active": self.uses_border_refinement(),
                "losses": border_losses,
                "selected_points": int(self._stats.get("border_refine_selected_points", 0)),
                "selected_gt": int(self._stats.get("border_refine_selected_gt", 0)),
                "mean_giou_loss": self._border_loss_sums["giou"] / float(max(border_losses, 1)),
                "mean_residual_loss": self._border_loss_sums["residual"] / float(max(border_losses, 1)),
                "mean_quality_loss": self._border_loss_sums["quality"] / float(max(border_losses, 1)),
                "mean_refined_iou": self._border_loss_sums["refined_iou"] / float(max(border_losses, 1)),
                **{
                    key: int(value)
                    for key, value in self._stats.items()
                    if str(key).startswith("border_refine_skipped_")
                },
            },
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 3,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if isinstance(state, Mapping):
            self.current_epoch = int(state.get("current_epoch", self.current_epoch))

    def _sample_border_refinement_features(
        self,
        *,
        feature_maps: Sequence[torch.Tensor],
        boxes: torch.Tensor,
        image_indices: torch.Tensor,
        level_indices: torch.Tensor,
        class_scores: torch.Tensor,
        ctr_scores: torch.Tensor,
        padded_shape: tuple[int, int],
    ) -> torch.Tensor:
        border = self.config.border_refinement
        if boxes.numel() == 0:
            return boxes.new_zeros((0, int(border.feature_dim) * 5 + _BORDER_GEOMETRY_DIM))
        feature_dim = int(border.feature_dim)
        result = boxes.new_zeros((boxes.shape[0], feature_dim * 5))
        for level_index, feature_map in enumerate(feature_maps):
            if feature_map.ndim != 4 or int(feature_map.shape[1]) != feature_dim:
                self._stats["border_refine_skipped_feature_dim"] += 1
                return boxes.new_zeros((0, feature_dim * 5 + _BORDER_GEOMETRY_DIM))
            selected = torch.where(level_indices == int(level_index))[0]
            if selected.numel() == 0:
                continue
            selected_images = image_indices[selected].to(device=feature_map.device, dtype=torch.long)
            sampled = _sample_box_border_features(
                feature_map=feature_map[selected_images],
                boxes=boxes[selected].to(device=feature_map.device, dtype=feature_map.dtype),
                padded_shape=padded_shape,
                points_per_side=int(border.border_points_per_side),
            )
            result[selected] = sampled.to(device=result.device, dtype=result.dtype)
        geometry = _border_geometry_features(
            boxes=boxes,
            level_indices=level_indices,
            num_levels=len(feature_maps),
            class_scores=class_scores,
            ctr_scores=ctr_scores,
            padded_shape=padded_shape,
        )
        return torch.cat((result, geometry.to(dtype=result.dtype)), dim=1)

    def _eligible_border_refinement_records(
        self,
        records: Sequence[DHMRecord | None],
    ) -> list[tuple[int, DHMRecord]]:
        eligible: list[tuple[int, DHMRecord]] = []
        for gt_index, record in enumerate(records):
            if record is None:
                continue
            if self._is_border_refinement_record_active(record, count_stats=False):
                eligible.append((gt_index, record))
        eligible.sort(
            key=lambda item: (
                float(item[1].instability_score),
                float(item[1].state_counts.get("FN_LOC", 0)),
                float(item[1].consecutive_fn),
            ),
            reverse=True,
        )
        max_gt = int(self.config.border_refinement.max_gt_per_image)
        if max_gt > 0:
            eligible = eligible[:max_gt]
        return eligible

    def _is_border_refinement_record_active(
        self,
        record: DHMRecord,
        *,
        count_stats: bool = True,
    ) -> bool:
        border = self.config.border_refinement
        if record.last_transition is None or str(record.last_transition) not in set(border.target_transitions):
            if count_stats:
                self._stats["border_refine_skipped_transition"] += 1
            return False
        if int(record.total_seen) < int(border.min_observations):
            if count_stats:
                self._stats["border_refine_skipped_low_observations"] += 1
            return False
        if float(record.instability_score) < float(border.min_instability):
            if count_stats:
                self._stats["border_refine_skipped_low_instability"] += 1
            return False
        return True

    def _record_border_refinement_loss(
        self,
        *,
        selected_points: int,
        selected_gt: int,
        giou_loss: torch.Tensor,
        residual_loss: torch.Tensor,
        quality_loss: torch.Tensor,
        refined_iou: torch.Tensor,
    ) -> None:
        self._stats["border_refine_losses"] += 1
        self._stats["border_refine_selected_points"] += int(selected_points)
        self._stats["border_refine_selected_gt"] += int(selected_gt)
        self._border_loss_sums["giou"] += float(giou_loss.detach().item())
        self._border_loss_sums["residual"] += float(residual_loss.detach().item())
        self._border_loss_sums["quality"] += float(quality_loss.detach().item())
        self._border_loss_sums["refined_iou"] += float(refined_iou.detach().item())


def load_dhmr_config(path: str | Path, *, arch: str | None = None) -> DHMRConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"DHM-R YAML must contain a mapping at the top level: {config_path}")
    return DHMRConfig.from_mapping(raw, arch=arch)


def build_dhmr_from_config(
    raw_config: Mapping[str, Any] | DHMRConfig,
    *,
    arch: str | None = None,
) -> DHMRepairModule | None:
    config = raw_config if isinstance(raw_config, DHMRConfig) else DHMRConfig.from_mapping(raw_config, arch=arch)
    if not config.enabled:
        return None
    return DHMRepairModule(config)


def build_dhmr_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> DHMRepairModule | None:
    config = load_dhmr_config(path, arch=arch)
    if not config.enabled:
        return None
    return DHMRepairModule(config)


def _sample_box_border_features(
    *,
    feature_map: torch.Tensor,
    boxes: torch.Tensor,
    padded_shape: tuple[int, int],
    points_per_side: int,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return feature_map.new_zeros((0, int(feature_map.shape[1]) * 5))
    n, c, _h, _w = feature_map.shape
    points = _box_border_points(
        boxes=boxes,
        points_per_side=points_per_side,
        padded_shape=padded_shape,
    )
    grid = _points_to_grid(points, padded_shape=padded_shape).reshape(n, -1, 1, 2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(dim=-1).permute(0, 2, 1)
    k = int(points_per_side)
    center = sampled[:, 0, :]
    offset = 1
    left = sampled[:, offset : offset + k, :].mean(dim=1)
    offset += k
    top = sampled[:, offset : offset + k, :].mean(dim=1)
    offset += k
    right = sampled[:, offset : offset + k, :].mean(dim=1)
    offset += k
    bottom = sampled[:, offset : offset + k, :].mean(dim=1)
    return torch.cat((center, left, top, right, bottom), dim=1).reshape(n, c * 5)


def _box_border_points(
    *,
    boxes: torch.Tensor,
    points_per_side: int,
    padded_shape: tuple[int, int],
) -> torch.Tensor:
    boxes = _clip_boxes(_sanitize_boxes(boxes), padded_shape=padded_shape)
    k = int(points_per_side)
    steps = torch.linspace(0.0, 1.0, k, device=boxes.device, dtype=boxes.dtype).reshape(1, k)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    xs = x1[:, None] + (x2 - x1)[:, None] * steps
    ys = y1[:, None] + (y2 - y1)[:, None] * steps
    center = torch.stack((cx, cy), dim=1).reshape(-1, 1, 2)
    left = torch.stack((x1[:, None].expand_as(ys), ys), dim=2)
    top = torch.stack((xs, y1[:, None].expand_as(xs)), dim=2)
    right = torch.stack((x2[:, None].expand_as(ys), ys), dim=2)
    bottom = torch.stack((xs, y2[:, None].expand_as(xs)), dim=2)
    return torch.cat((center, left, top, right, bottom), dim=1)


def _points_to_grid(
    points: torch.Tensor,
    *,
    padded_shape: tuple[int, int],
) -> torch.Tensor:
    height = max(1.0, float(padded_shape[0]))
    width = max(1.0, float(padded_shape[1]))
    x = points[..., 0].clamp(min=0.0, max=width - 1.0)
    y = points[..., 1].clamp(min=0.0, max=height - 1.0)
    grid_x = torch.zeros_like(x) if width <= 1.0 else x / (width - 1.0) * 2.0 - 1.0
    grid_y = torch.zeros_like(y) if height <= 1.0 else y / (height - 1.0) * 2.0 - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)


def _border_geometry_features(
    *,
    boxes: torch.Tensor,
    level_indices: torch.Tensor,
    num_levels: int,
    class_scores: torch.Tensor,
    ctr_scores: torch.Tensor,
    padded_shape: tuple[int, int],
) -> torch.Tensor:
    height = max(1.0, float(padded_shape[0]))
    width = max(1.0, float(padded_shape[1]))
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    box_w = (x2 - x1).clamp_min(1.0)
    box_h = (y2 - y1).clamp_min(1.0)
    cx = (x1 + x2) * 0.5 / width
    cy = (y1 + y2) * 0.5 / height
    area = (box_w * box_h) / max(width * height, 1.0)
    aspect = torch.log((box_w / box_h).clamp_min(1.0e-6))
    level_denominator = float(max(int(num_levels) - 1, 1))
    level = level_indices.to(device=boxes.device, dtype=boxes.dtype) / level_denominator
    return torch.stack(
        (
            cx,
            cy,
            box_w / width,
            box_h / height,
            aspect,
            area,
            class_scores.to(device=boxes.device, dtype=boxes.dtype),
            ctr_scores.to(device=boxes.device, dtype=boxes.dtype),
            level,
        ),
        dim=1,
    )


def _level_ids(
    num_anchors_per_level: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.full((int(count),), index, dtype=torch.long, device=device)
            for index, count in enumerate(num_anchors_per_level)
        ],
        dim=0,
    )


def _normalized_box_delta(
    *,
    boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    scale = torch.stack((width, height, width, height), dim=1)
    delta = (gt_boxes.to(device=boxes.device, dtype=boxes.dtype) - boxes) / scale
    if clip > 0.0:
        delta = delta.clamp(min=-float(clip), max=float(clip))
    return delta


def _apply_box_delta(
    *,
    boxes: torch.Tensor,
    delta: torch.Tensor,
    padded_shape: tuple[int, int],
) -> torch.Tensor:
    width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    scale = torch.stack((width, height, width, height), dim=1)
    refined = boxes + delta.to(device=boxes.device, dtype=boxes.dtype) * scale
    return _clip_boxes(_sanitize_boxes(refined), padded_shape=padded_shape)


def _sanitize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
    y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
    y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
    x2 = torch.maximum(x2, x1 + 1.0)
    y2 = torch.maximum(y2, y1 + 1.0)
    return torch.stack((x1, y1, x2, y2), dim=1)


def _clip_boxes(
    boxes: torch.Tensor,
    *,
    padded_shape: tuple[int, int],
) -> torch.Tensor:
    height = max(1.0, float(padded_shape[0]))
    width = max(1.0, float(padded_shape[1]))
    x1 = boxes[:, 0].clamp(min=0.0, max=width - 1.0)
    y1 = boxes[:, 1].clamp(min=0.0, max=height - 1.0)
    x2 = boxes[:, 2].clamp(min=0.0, max=width - 1.0)
    y2 = boxes[:, 3].clamp(min=0.0, max=height - 1.0)
    x2 = torch.maximum(x2, x1 + 1.0).clamp(max=width)
    y2 = torch.maximum(y2, y1 + 1.0).clamp(max=height)
    return torch.stack((x1, y1, x2, y2), dim=1)


def _aligned_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = (area1 + area2 - inter).clamp(min=torch.finfo(boxes1.dtype).eps)
    return inter / union


DHMR = DHMRepairModule
