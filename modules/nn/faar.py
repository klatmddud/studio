from __future__ import annotations

import math
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
    "nms_suppression",
)

_SUPPORTED_FAILURE_TYPES = set(_DEFAULT_FAILURE_TYPES) | {"cls_confusion"}


@dataclass(frozen=True, slots=True)
class FAARConfig:
    enabled: bool = False
    warmup_epochs: int = 2
    min_severity: float = 1.0
    record_match_threshold: float = 0.95
    max_repair_targets_per_image: int = 5
    base_repair_points: int = 2
    severity_budget_scale: float = 1.0
    max_repair_points_per_gt: int = 8
    base_region_scale: float = 1.0
    severity_region_scale: float = 0.1
    max_region_scale: float = 1.5
    require_unassigned_points: bool = True
    allow_positive_reassignment: bool = False
    protect_existing_positive_iou: float = 0.3
    respect_fcos_scale_range: bool = True
    allow_adjacent_levels: bool = True
    allow_nearest_center_fallback: bool = True
    include_relapse: bool = True
    relapse_budget_bonus: int = 1
    failure_types: tuple[str, ...] = _DEFAULT_FAILURE_TYPES
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "FAARConfig":
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
            min_severity=float(read("min_severity", 1.0)),
            record_match_threshold=float(read("record_match_threshold", 0.95)),
            max_repair_targets_per_image=int(read("max_repair_targets_per_image", 5)),
            base_repair_points=int(read("base_repair_points", 2)),
            severity_budget_scale=float(read("severity_budget_scale", 1.0)),
            max_repair_points_per_gt=int(read("max_repair_points_per_gt", 8)),
            base_region_scale=float(read("base_region_scale", 1.0)),
            severity_region_scale=float(read("severity_region_scale", 0.1)),
            max_region_scale=float(read("max_region_scale", 1.5)),
            require_unassigned_points=bool(read("require_unassigned_points", True)),
            allow_positive_reassignment=bool(read("allow_positive_reassignment", False)),
            protect_existing_positive_iou=float(read("protect_existing_positive_iou", 0.3)),
            respect_fcos_scale_range=bool(read("respect_fcos_scale_range", True)),
            allow_adjacent_levels=bool(read("allow_adjacent_levels", True)),
            allow_nearest_center_fallback=bool(read("allow_nearest_center_fallback", True)),
            include_relapse=bool(read("include_relapse", True)),
            relapse_budget_bonus=int(read("relapse_budget_bonus", 1)),
            failure_types=failure_types,
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("FAAR warmup_epochs must be >= 0.")
        if self.min_severity < 0.0:
            raise ValueError("FAAR min_severity must be >= 0.")
        if not 0.0 <= self.record_match_threshold <= 1.0:
            raise ValueError("FAAR record_match_threshold must satisfy 0 <= value <= 1.")
        if self.max_repair_targets_per_image < 1:
            raise ValueError("FAAR max_repair_targets_per_image must be >= 1.")
        if self.base_repair_points < 0:
            raise ValueError("FAAR base_repair_points must be >= 0.")
        if self.severity_budget_scale < 0.0:
            raise ValueError("FAAR severity_budget_scale must be >= 0.")
        if self.max_repair_points_per_gt < 1:
            raise ValueError("FAAR max_repair_points_per_gt must be >= 1.")
        if self.base_region_scale <= 0.0:
            raise ValueError("FAAR base_region_scale must be > 0.")
        if self.severity_region_scale < 0.0:
            raise ValueError("FAAR severity_region_scale must be >= 0.")
        if self.max_region_scale <= 0.0:
            raise ValueError("FAAR max_region_scale must be > 0.")
        if not 0.0 <= self.protect_existing_positive_iou <= 1.0:
            raise ValueError("FAAR protect_existing_positive_iou must satisfy 0 <= value <= 1.")
        if self.relapse_budget_bonus < 0:
            raise ValueError("FAAR relapse_budget_bonus must be >= 0.")
        if not self.failure_types:
            raise ValueError("FAAR failure_types must not be empty.")
        unsupported = sorted(set(self.failure_types) - _SUPPORTED_FAILURE_TYPES)
        if unsupported:
            raise ValueError(f"FAAR failure_types contains unsupported values: {unsupported}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_epochs": self.warmup_epochs,
            "min_severity": self.min_severity,
            "record_match_threshold": self.record_match_threshold,
            "max_repair_targets_per_image": self.max_repair_targets_per_image,
            "base_repair_points": self.base_repair_points,
            "severity_budget_scale": self.severity_budget_scale,
            "max_repair_points_per_gt": self.max_repair_points_per_gt,
            "base_region_scale": self.base_region_scale,
            "severity_region_scale": self.severity_region_scale,
            "max_region_scale": self.max_region_scale,
            "require_unassigned_points": self.require_unassigned_points,
            "allow_positive_reassignment": self.allow_positive_reassignment,
            "protect_existing_positive_iou": self.protect_existing_positive_iou,
            "respect_fcos_scale_range": self.respect_fcos_scale_range,
            "allow_adjacent_levels": self.allow_adjacent_levels,
            "allow_nearest_center_fallback": self.allow_nearest_center_fallback,
            "include_relapse": self.include_relapse,
            "relapse_budget_bonus": self.relapse_budget_bonus,
            "failure_types": list(self.failure_types),
            "arch": self.arch,
        }


@dataclass(slots=True)
class RepairTarget:
    gt_uid: str
    image_id: str
    gt_index: int
    class_id: int
    bbox: torch.Tensor
    failure_type: str
    severity: float
    budget: int
    relapse: bool = False

    def __post_init__(self) -> None:
        self.gt_uid = str(self.gt_uid)
        self.image_id = str(self.image_id)
        self.gt_index = int(self.gt_index)
        self.class_id = int(self.class_id)
        self.bbox = torch.as_tensor(self.bbox, dtype=torch.float32).detach()
        if self.bbox.numel() != 4:
            raise ValueError("RepairTarget bbox must contain exactly four values.")
        self.bbox = self.bbox.reshape(4)
        self.failure_type = str(self.failure_type)
        self.severity = float(self.severity)
        self.budget = int(self.budget)
        self.relapse = bool(self.relapse)


@dataclass(slots=True)
class RepairPlan:
    targets: list[RepairTarget]

    def for_image(self, image_id: Any) -> list[RepairTarget]:
        image_key = _normalize_image_id(image_id)
        return [target for target in self.targets if target.image_id == image_key]

    def __len__(self) -> int:
        return len(self.targets)


class FailureAwareAssignmentRepair(nn.Module):
    def __init__(self, config: FAARConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._epoch_repair_targets = 0
        self._epoch_repair_points = 0
        self._epoch_repair_images: set[str] = set()
        self._epoch_severity_sum = 0.0
        self._epoch_failure_counts: Counter[str] = Counter()
        self._epoch_skipped_no_entry_match = 0
        self._epoch_skipped_no_candidate_points = 0
        self._epoch_skipped_existing_positive = 0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._epoch_repair_targets = 0
        self._epoch_repair_points = 0
        self._epoch_repair_images = set()
        self._epoch_severity_sum = 0.0
        self._epoch_failure_counts = Counter()
        self._epoch_skipped_no_entry_match = 0
        self._epoch_skipped_no_candidate_points = 0
        self._epoch_skipped_existing_positive = 0

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def warmup_factor(self) -> float:
        if self.config.warmup_epochs <= 0:
            return 1.0
        if self.current_epoch <= 0:
            return 0.0
        return min(1.0, float(self.current_epoch) / float(self.config.warmup_epochs))

    def should_apply(self, *, mdmbpp: MDMBPlus | None) -> bool:
        return mdmbpp is not None and self.config.enabled and self.warmup_factor() > 0.0

    @torch.no_grad()
    def plan(
        self,
        *,
        mdmbpp: MDMBPlus | None,
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_shapes: Sequence[Sequence[int]],
    ) -> RepairPlan:
        if mdmbpp is None or not self.config.enabled:
            return RepairPlan(targets=[])

        repair_targets: list[RepairTarget] = []
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
            image_targets: list[RepairTarget] = []

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

                record = mdmbpp.get_record(entry.gt_uid)
                relapse = bool(
                    self.config.include_relapse
                    and record is not None
                    and getattr(record, "last_detected_epoch", None) is not None
                    and getattr(record, "last_state", None) != "detected"
                )
                gt_index = int(match)
                used_gt_indices.add(gt_index)
                image_targets.append(
                    RepairTarget(
                        gt_uid=entry.gt_uid,
                        image_id=image_key,
                        gt_index=gt_index,
                        class_id=entry.class_id,
                        bbox=gt_boxes[gt_index].detach(),
                        failure_type=entry.failure_type,
                        severity=entry.severity,
                        budget=self.budget_for(entry.severity, relapse=relapse),
                        relapse=relapse,
                    )
                )
                if len(image_targets) >= self.config.max_repair_targets_per_image:
                    break

            repair_targets.extend(image_targets)

        return RepairPlan(targets=repair_targets)

    def budget_for(self, severity: float, *, relapse: bool = False) -> int:
        budget = self.config.base_repair_points + math.floor(
            self.config.severity_budget_scale * float(severity)
        )
        if relapse:
            budget += int(self.config.relapse_budget_bonus)
        return max(1, min(self.config.max_repair_points_per_gt, int(budget)))

    def region_scale_for(self, severity: float) -> float:
        scale = self.config.base_region_scale + self.config.severity_region_scale * float(severity)
        return min(self.config.max_region_scale, max(self.config.base_region_scale, scale))

    def record_repair_step(
        self,
        *,
        repair_plan: RepairPlan,
        repaired_points: int,
        skipped_no_candidate_points: int = 0,
        skipped_existing_positive: int = 0,
    ) -> None:
        if len(repair_plan) > 0:
            self._epoch_repair_targets += len(repair_plan)
            self._epoch_severity_sum += sum(target.severity for target in repair_plan.targets)
            self._epoch_repair_images.update(target.image_id for target in repair_plan.targets)
            self._epoch_failure_counts.update(target.failure_type for target in repair_plan.targets)
        self._epoch_repair_points += int(repaired_points)
        self._epoch_skipped_no_candidate_points += int(skipped_no_candidate_points)
        self._epoch_skipped_existing_positive += int(skipped_existing_positive)

    def summary(self) -> dict[str, Any]:
        mean_severity = self._epoch_severity_sum / float(max(self._epoch_repair_targets, 1))
        summary = {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_factor": self.warmup_factor(),
            "repair_targets": self._epoch_repair_targets,
            "repair_points": self._epoch_repair_points,
            "repair_images": len(self._epoch_repair_images),
            "mean_severity": mean_severity,
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
            "skipped_no_entry_match": self._epoch_skipped_no_entry_match,
            "skipped_no_candidate_points": self._epoch_skipped_no_candidate_points,
            "skipped_existing_positive": self._epoch_skipped_existing_positive,
        }
        summary.update(
            {
                "faar_enabled": summary["enabled"],
                "faar_epoch": summary["current_epoch"],
                "faar_warmup_factor": summary["warmup_factor"],
                "faar_repair_targets": summary["repair_targets"],
                "faar_repair_points": summary["repair_points"],
                "faar_repair_images": summary["repair_images"],
                "faar_mean_severity": summary["mean_severity"],
                "faar_by_failure_candidate_missing": summary[
                    "by_failure_candidate_missing"
                ],
                "faar_by_failure_loc_near_miss": summary["by_failure_loc_near_miss"],
                "faar_by_failure_score_suppression": summary[
                    "by_failure_score_suppression"
                ],
                "faar_by_failure_nms_suppression": summary[
                    "by_failure_nms_suppression"
                ],
                "faar_skipped_no_entry_match": summary["skipped_no_entry_match"],
                "faar_skipped_no_candidate_points": summary[
                    "skipped_no_candidate_points"
                ],
                "faar_skipped_existing_positive": summary["skipped_existing_positive"],
            }
        )
        return summary

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


def load_faar_config(
    path: str | Path,
    *,
    arch: str | None = None,
) -> FAARConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"FAAR YAML must contain a mapping at the top level: {config_path}")
    return FAARConfig.from_mapping(raw, arch=arch)


def build_faar_from_config(
    raw_config: Mapping[str, Any] | FAARConfig,
    *,
    arch: str | None = None,
) -> FailureAwareAssignmentRepair | None:
    config = (
        raw_config
        if isinstance(raw_config, FAARConfig)
        else FAARConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return FailureAwareAssignmentRepair(config)


def build_faar_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> FailureAwareAssignmentRepair | None:
    config = load_faar_config(path, arch=arch)
    if not config.enabled:
        return None
    return FailureAwareAssignmentRepair(config)


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return str(value.detach().cpu().flatten()[0].item())
    return str(value)
