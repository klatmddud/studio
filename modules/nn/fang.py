from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torchvision.ops import boxes as box_ops

from .mdmb import normalize_arch, normalize_xyxy_boxes
from .mdmbpp import MDMBPlus


_DEFAULT_FAILURE_TYPES: tuple[str, ...] = (
    "candidate_missing",
    "loc_near_miss",
    "score_suppression",
)

_SUPPORTED_FAILURE_TYPES = set(_DEFAULT_FAILURE_TYPES) | {
    "cls_confusion",
    "nms_suppression",
}


@dataclass(frozen=True, slots=True)
class FANGConfig:
    enabled: bool = False
    warmup_epochs: int = 2
    lambda_fang: float = 0.5
    min_negative_weight: float = 0.25
    min_severity: float = 1.0
    record_match_threshold: float = 0.95
    max_shield_targets_per_image: int = 5
    max_shield_points_per_gt: int = 16
    base_region_scale: float = 1.0
    severity_region_scale: float = 0.1
    max_region_scale: float = 1.5
    severity_weight_scale: float = 0.25
    max_target_weight: float = 3.0
    failure_types: tuple[str, ...] = _DEFAULT_FAILURE_TYPES
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "FANGConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))

        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        def read(name: str, default: Any) -> Any:
            return model_overrides.get(name, data.get(name, default))

        raw_failure_types = read("failure_types", _DEFAULT_FAILURE_TYPES)
        if isinstance(raw_failure_types, str):
            failure_types = (raw_failure_types,)
        else:
            failure_types = tuple(str(value) for value in raw_failure_types)

        config = cls(
            enabled=bool(read("enabled", False)),
            warmup_epochs=int(read("warmup_epochs", 2)),
            lambda_fang=float(read("lambda_fang", 0.5)),
            min_negative_weight=float(read("min_negative_weight", 0.25)),
            min_severity=float(read("min_severity", 1.0)),
            record_match_threshold=float(read("record_match_threshold", 0.95)),
            max_shield_targets_per_image=int(read("max_shield_targets_per_image", 5)),
            max_shield_points_per_gt=int(read("max_shield_points_per_gt", 16)),
            base_region_scale=float(read("base_region_scale", 1.0)),
            severity_region_scale=float(read("severity_region_scale", 0.1)),
            max_region_scale=float(read("max_region_scale", 1.5)),
            severity_weight_scale=float(read("severity_weight_scale", 0.25)),
            max_target_weight=float(read("max_target_weight", 3.0)),
            failure_types=failure_types,
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("FANG warmup_epochs must be >= 0.")
        if self.lambda_fang < 0.0:
            raise ValueError("FANG lambda_fang must be >= 0.")
        if not 0.0 <= self.min_negative_weight <= 1.0:
            raise ValueError("FANG min_negative_weight must satisfy 0 <= value <= 1.")
        if self.min_severity < 0.0:
            raise ValueError("FANG min_severity must be >= 0.")
        if not 0.0 <= self.record_match_threshold <= 1.0:
            raise ValueError("FANG record_match_threshold must satisfy 0 <= value <= 1.")
        if self.max_shield_targets_per_image < 1:
            raise ValueError("FANG max_shield_targets_per_image must be >= 1.")
        if self.max_shield_points_per_gt < 1:
            raise ValueError("FANG max_shield_points_per_gt must be >= 1.")
        if self.base_region_scale <= 0.0:
            raise ValueError("FANG base_region_scale must be > 0.")
        if self.severity_region_scale < 0.0:
            raise ValueError("FANG severity_region_scale must be >= 0.")
        if self.max_region_scale <= 0.0:
            raise ValueError("FANG max_region_scale must be > 0.")
        if self.severity_weight_scale < 0.0:
            raise ValueError("FANG severity_weight_scale must be >= 0.")
        if self.max_target_weight < 1.0:
            raise ValueError("FANG max_target_weight must be >= 1.")
        if not self.failure_types:
            raise ValueError("FANG failure_types must not be empty.")
        unsupported = sorted(set(self.failure_types) - _SUPPORTED_FAILURE_TYPES)
        if unsupported:
            raise ValueError(f"FANG failure_types contains unsupported values: {unsupported}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_epochs": self.warmup_epochs,
            "lambda_fang": self.lambda_fang,
            "min_negative_weight": self.min_negative_weight,
            "min_severity": self.min_severity,
            "record_match_threshold": self.record_match_threshold,
            "max_shield_targets_per_image": self.max_shield_targets_per_image,
            "max_shield_points_per_gt": self.max_shield_points_per_gt,
            "base_region_scale": self.base_region_scale,
            "severity_region_scale": self.severity_region_scale,
            "max_region_scale": self.max_region_scale,
            "severity_weight_scale": self.severity_weight_scale,
            "max_target_weight": self.max_target_weight,
            "failure_types": list(self.failure_types),
            "arch": self.arch,
        }


@dataclass(slots=True)
class ShieldTarget:
    gt_uid: str
    image_id: str
    gt_index: int
    class_id: int
    bbox: torch.Tensor
    failure_type: str
    severity: float
    weight: float

    def __post_init__(self) -> None:
        self.gt_uid = str(self.gt_uid)
        self.image_id = str(self.image_id)
        self.gt_index = int(self.gt_index)
        self.class_id = int(self.class_id)
        self.bbox = torch.as_tensor(self.bbox, dtype=torch.float32).detach()
        if self.bbox.numel() != 4:
            raise ValueError("ShieldTarget bbox must contain exactly four values.")
        self.bbox = self.bbox.reshape(4)
        self.failure_type = str(self.failure_type)
        self.severity = float(self.severity)
        self.weight = float(self.weight)


@dataclass(slots=True)
class ShieldPlan:
    targets: list[ShieldTarget]

    def for_image(self, image_id: Any) -> list[ShieldTarget]:
        image_key = _normalize_image_id(image_id)
        return [target for target in self.targets if target.image_id == image_key]

    def __len__(self) -> int:
        return len(self.targets)


class FailureAwareNegativeGradientShielding(nn.Module):
    def __init__(self, config: FANGConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._epoch_shield_targets = 0
        self._epoch_shield_points = 0
        self._epoch_shield_batches = 0
        self._epoch_shield_weight_sum = 0.0
        self._epoch_failure_counts: Counter[str] = Counter()
        self._epoch_skipped_no_entry_match = 0
        self._epoch_skipped_no_candidate_points = 0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._epoch_shield_targets = 0
        self._epoch_shield_points = 0
        self._epoch_shield_batches = 0
        self._epoch_shield_weight_sum = 0.0
        self._epoch_failure_counts = Counter()
        self._epoch_skipped_no_entry_match = 0
        self._epoch_skipped_no_candidate_points = 0

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def warmup_factor(self) -> float:
        if self.config.warmup_epochs <= 0:
            return 1.0
        if self.current_epoch <= 0:
            return 0.0
        return min(1.0, float(self.current_epoch) / float(self.config.warmup_epochs))

    def loss_weight(self) -> float:
        if not self.config.enabled:
            return 0.0
        return float(self.config.lambda_fang) * self.warmup_factor()

    def should_apply(self, *, mdmbpp: MDMBPlus | None) -> bool:
        return mdmbpp is not None and self.loss_weight() > 0.0

    @torch.no_grad()
    def plan(
        self,
        *,
        mdmbpp: MDMBPlus | None,
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_shapes: Sequence[Sequence[int]],
    ) -> ShieldPlan:
        if mdmbpp is None or not self.config.enabled:
            return ShieldPlan(targets=[])

        shield_targets: list[ShieldTarget] = []
        failure_types = set(self.config.failure_types)

        for image_index, target in enumerate(targets):
            image_id = target.get("image_id", torch.tensor(image_index))
            image_key = _normalize_image_id(image_id)
            entries = [
                entry
                for entry in mdmbpp.get_dense_targets(image_key)
                if entry.failure_type in failure_types
                and float(entry.severity) >= self.config.min_severity
            ]
            if not entries:
                continue

            gt_boxes = target["boxes"]
            gt_labels = target["labels"].to(device=gt_boxes.device, dtype=torch.int64)
            if gt_boxes.numel() == 0:
                continue

            gt_boxes_norm = normalize_xyxy_boxes(gt_boxes, image_shapes[image_index]).to(
                device=gt_boxes.device,
                dtype=torch.float32,
            )
            used_gt_indices: set[int] = set()
            image_targets: list[ShieldTarget] = []

            for entry in entries:
                match = self._match_entry_to_target(
                    entry=entry,
                    gt_boxes=gt_boxes,
                    gt_boxes_norm=gt_boxes_norm,
                    gt_labels=gt_labels,
                    used_gt_indices=used_gt_indices,
                )
                if match is None:
                    self._epoch_skipped_no_entry_match += 1
                    continue

                gt_index = int(match)
                used_gt_indices.add(gt_index)
                image_targets.append(
                    ShieldTarget(
                        gt_uid=entry.gt_uid,
                        image_id=image_key,
                        gt_index=gt_index,
                        class_id=int(gt_labels[gt_index].item()),
                        bbox=gt_boxes[gt_index].detach(),
                        failure_type=entry.failure_type,
                        severity=entry.severity,
                        weight=self.target_weight_for(entry.severity),
                    )
                )
                if len(image_targets) >= self.config.max_shield_targets_per_image:
                    break

            shield_targets.extend(image_targets)

        return ShieldPlan(targets=shield_targets)

    @torch.no_grad()
    def compute_class_weights(
        self,
        *,
        shield_plan: ShieldPlan,
        targets: Sequence[Mapping[str, torch.Tensor]],
        anchors: Sequence[torch.Tensor],
        matched_idxs: Sequence[torch.Tensor],
        num_classes: int,
    ) -> tuple[list[torch.Tensor], int, int, float]:
        class_weights = [
            torch.ones(
                (anchors_per_image.shape[0], int(num_classes)),
                dtype=torch.float32,
                device=anchors_per_image.device,
            )
            for anchors_per_image in anchors
        ]
        if len(shield_plan) == 0 or int(num_classes) <= 0:
            return class_weights, 0, 0, 0.0

        shield_points = 0
        skipped_no_candidate_points = 0
        shield_weight_sum = 0.0
        strength = float(self.loss_weight())

        for image_index, target in enumerate(targets):
            image_id = target.get(
                "image_id",
                torch.tensor(image_index, device=anchors[image_index].device),
            )
            image_targets = shield_plan.for_image(image_id)
            if not image_targets:
                continue

            anchors_per_image = anchors[image_index]
            assignments = matched_idxs[image_index]
            gt_boxes = target["boxes"].to(device=anchors_per_image.device)
            if anchors_per_image.numel() == 0 or gt_boxes.numel() == 0:
                skipped_no_candidate_points += len(image_targets)
                continue

            centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) * 0.5
            negative_mask = assignments < 0

            for shield_target in image_targets:
                if shield_target.gt_index < 0 or shield_target.gt_index >= gt_boxes.shape[0]:
                    skipped_no_candidate_points += 1
                    continue
                if shield_target.class_id < 0 or shield_target.class_id >= int(num_classes):
                    skipped_no_candidate_points += 1
                    continue

                point_indices, distances = self._select_shield_points(
                    shield_target=shield_target,
                    centers=centers,
                    negative_mask=negative_mask,
                    device=anchors_per_image.device,
                    dtype=anchors_per_image.dtype,
                )
                if point_indices.numel() == 0:
                    skipped_no_candidate_points += 1
                    continue

                distance_factor = 1.0 - distances.clamp(min=0.0, max=1.0)
                shield_amount = strength * float(shield_target.weight) * distance_factor
                weights = (1.0 - shield_amount).clamp(
                    min=float(self.config.min_negative_weight),
                    max=1.0,
                )
                class_id = int(shield_target.class_id)
                current = class_weights[image_index][point_indices, class_id]
                class_weights[image_index][point_indices, class_id] = torch.minimum(
                    current,
                    weights.to(device=current.device, dtype=current.dtype),
                )
                shield_points += int(point_indices.numel())
                shield_weight_sum += float(weights.detach().sum().cpu().item())

        return class_weights, shield_points, skipped_no_candidate_points, shield_weight_sum

    def target_weight_for(self, severity: float) -> float:
        weight = 1.0 + float(self.config.severity_weight_scale) * float(severity)
        return min(float(self.config.max_target_weight), max(1.0, weight))

    def region_scale_for(self, severity: float) -> float:
        scale = self.config.base_region_scale + self.config.severity_region_scale * float(severity)
        return min(self.config.max_region_scale, max(self.config.base_region_scale, scale))

    def record_shield_step(
        self,
        *,
        shield_plan: ShieldPlan,
        shield_points: int,
        skipped_no_candidate_points: int = 0,
        shield_weight_sum: float = 0.0,
    ) -> None:
        if len(shield_plan) > 0:
            self._epoch_shield_targets += len(shield_plan)
            self._epoch_failure_counts.update(target.failure_type for target in shield_plan.targets)
        self._epoch_shield_points += int(shield_points)
        self._epoch_skipped_no_candidate_points += int(skipped_no_candidate_points)
        if int(shield_points) > 0:
            self._epoch_shield_batches += 1
            self._epoch_shield_weight_sum += float(shield_weight_sum)

    def summary(self) -> dict[str, Any]:
        mean_weight = self._epoch_shield_weight_sum / float(max(self._epoch_shield_points, 1))
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_factor": self.warmup_factor(),
            "lambda_fang": self.loss_weight(),
            "min_negative_weight": self.config.min_negative_weight,
            "shield_targets": self._epoch_shield_targets,
            "shield_points": self._epoch_shield_points,
            "shield_batches": self._epoch_shield_batches,
            "mean_shield_weight": mean_weight,
            "by_failure_candidate_missing": int(
                self._epoch_failure_counts.get("candidate_missing", 0)
            ),
            "by_failure_loc_near_miss": int(self._epoch_failure_counts.get("loc_near_miss", 0)),
            "by_failure_score_suppression": int(
                self._epoch_failure_counts.get("score_suppression", 0)
            ),
            "by_failure_nms_suppression": int(
                self._epoch_failure_counts.get("nms_suppression", 0)
            ),
            "by_failure_cls_confusion": int(self._epoch_failure_counts.get("cls_confusion", 0)),
            "skipped_no_entry_match": self._epoch_skipped_no_entry_match,
            "skipped_no_candidate_points": self._epoch_skipped_no_candidate_points,
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))

    def _select_shield_points(
        self,
        *,
        shield_target: ShieldTarget,
        centers: torch.Tensor,
        negative_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_box = shield_target.bbox.to(device=device, dtype=dtype)
        gt_center = (gt_box[:2] + gt_box[2:]) * 0.5
        gt_size = (gt_box[2:] - gt_box[:2]).clamp(min=1.0)
        scale = float(self.region_scale_for(shield_target.severity))
        half_size = (gt_size * scale * 0.5).clamp(min=1.0)
        region_min = gt_center - half_size
        region_max = gt_center + half_size
        region_mask = (
            (centers[:, 0] >= region_min[0])
            & (centers[:, 0] <= region_max[0])
            & (centers[:, 1] >= region_min[1])
            & (centers[:, 1] <= region_max[1])
            & negative_mask
        )
        candidate_indices = torch.nonzero(region_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return candidate_indices, centers.new_zeros((0,))

        normalized_offsets = (centers[candidate_indices] - gt_center).abs() / half_size
        distances = normalized_offsets.max(dim=1).values
        count = min(int(self.config.max_shield_points_per_gt), int(candidate_indices.numel()))
        nearest = torch.argsort(distances)[:count]
        return candidate_indices[nearest], distances[nearest]

    def _match_entry_to_target(
        self,
        *,
        entry,
        gt_boxes: torch.Tensor,
        gt_boxes_norm: torch.Tensor,
        gt_labels: torch.Tensor,
        used_gt_indices: set[int],
    ) -> int | None:
        class_mask = gt_labels == int(entry.class_id)
        if used_gt_indices:
            used = torch.tensor(
                sorted(used_gt_indices),
                dtype=torch.long,
                device=gt_labels.device,
            )
            class_mask[used] = False
        if not bool(class_mask.any().item()):
            return None

        candidate_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
        entry_box = entry.bbox.to(device=gt_boxes.device, dtype=gt_boxes_norm.dtype).reshape(1, 4)
        ious = box_ops.box_iou(entry_box, gt_boxes_norm[candidate_indices])[0]
        best_pos = int(ious.argmax().item())
        best_iou = float(ious[best_pos].item())
        if best_iou < self.config.record_match_threshold:
            return None
        return int(candidate_indices[best_pos].item())


def load_fang_config(
    path: str | Path,
    *,
    arch: str | None = None,
) -> FANGConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"FANG YAML must contain a mapping at the top level: {config_path}")
    return FANGConfig.from_mapping(raw, arch=arch)


def build_fang_from_config(
    raw_config: Mapping[str, Any] | FANGConfig,
    *,
    arch: str | None = None,
) -> FailureAwareNegativeGradientShielding | None:
    config = (
        raw_config
        if isinstance(raw_config, FANGConfig)
        else FANGConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return FailureAwareNegativeGradientShielding(config)


def build_fang_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> FailureAwareNegativeGradientShielding | None:
    config = load_fang_config(path, arch=arch)
    if not config.enabled:
        return None
    return FailureAwareNegativeGradientShielding(config)


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("FANG image_id tensor must contain one scalar.")
        value = value.item()
    return str(value)


FANG = FailureAwareNegativeGradientShielding
