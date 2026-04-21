from __future__ import annotations

import math
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
    "cls_confusion",
    "score_suppression",
    "nms_suppression",
)

_SUPPORTED_FAILURE_TYPES = set(_DEFAULT_FAILURE_TYPES) | {"detected"}


@dataclass(frozen=True, slots=True)
class CandidateDensificationConfig:
    enabled: bool = True
    warmup_epochs: int = 2
    lambda_dense: float = 0.05
    min_severity: float = 1.0
    record_match_threshold: float = 0.95
    max_dense_targets_per_image: int = 5
    base_extra_candidates: int = 2
    severity_budget_scale: float = 1.0
    max_extra_candidates_per_gt: int = 6
    base_region_scale: float = 1.0
    severity_region_scale: float = 0.1
    max_region_scale: float = 1.5
    require_unassigned_points: bool = True
    failure_types: tuple[str, ...] = _DEFAULT_FAILURE_TYPES
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "CandidateDensificationConfig":
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
            enabled=bool(read("enabled", True)),
            warmup_epochs=int(read("warmup_epochs", 2)),
            lambda_dense=float(read("lambda_dense", 0.05)),
            min_severity=float(read("min_severity", 1.0)),
            record_match_threshold=float(read("record_match_threshold", 0.95)),
            max_dense_targets_per_image=int(read("max_dense_targets_per_image", 5)),
            base_extra_candidates=int(read("base_extra_candidates", 2)),
            severity_budget_scale=float(read("severity_budget_scale", 1.0)),
            max_extra_candidates_per_gt=int(read("max_extra_candidates_per_gt", 6)),
            base_region_scale=float(read("base_region_scale", 1.0)),
            severity_region_scale=float(read("severity_region_scale", 0.1)),
            max_region_scale=float(read("max_region_scale", 1.5)),
            require_unassigned_points=bool(read("require_unassigned_points", True)),
            failure_types=failure_types,
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("Candidate Densification warmup_epochs must be >= 0.")
        if self.lambda_dense < 0.0:
            raise ValueError("Candidate Densification lambda_dense must be >= 0.")
        if self.min_severity < 0.0:
            raise ValueError("Candidate Densification min_severity must be >= 0.")
        if not 0.0 <= self.record_match_threshold <= 1.0:
            raise ValueError(
                "Candidate Densification record_match_threshold must satisfy 0 <= value <= 1."
            )
        for name in (
            "max_dense_targets_per_image",
            "base_extra_candidates",
            "max_extra_candidates_per_gt",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"Candidate Densification {name} must be >= 1.")
        if self.severity_budget_scale < 0.0:
            raise ValueError("Candidate Densification severity_budget_scale must be >= 0.")
        if self.base_region_scale <= 0.0:
            raise ValueError("Candidate Densification base_region_scale must be > 0.")
        if self.severity_region_scale < 0.0:
            raise ValueError("Candidate Densification severity_region_scale must be >= 0.")
        if self.max_region_scale <= 0.0:
            raise ValueError("Candidate Densification max_region_scale must be > 0.")
        if not self.failure_types:
            raise ValueError("Candidate Densification failure_types must not be empty.")
        unsupported = sorted(set(self.failure_types) - _SUPPORTED_FAILURE_TYPES)
        if unsupported:
            raise ValueError(
                "Candidate Densification failure_types contains unsupported values: "
                f"{unsupported}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_epochs": self.warmup_epochs,
            "lambda_dense": self.lambda_dense,
            "min_severity": self.min_severity,
            "record_match_threshold": self.record_match_threshold,
            "max_dense_targets_per_image": self.max_dense_targets_per_image,
            "base_extra_candidates": self.base_extra_candidates,
            "severity_budget_scale": self.severity_budget_scale,
            "max_extra_candidates_per_gt": self.max_extra_candidates_per_gt,
            "base_region_scale": self.base_region_scale,
            "severity_region_scale": self.severity_region_scale,
            "max_region_scale": self.max_region_scale,
            "require_unassigned_points": self.require_unassigned_points,
            "failure_types": list(self.failure_types),
            "arch": self.arch,
        }


@dataclass(slots=True)
class DenseTarget:
    gt_uid: str
    image_id: str
    gt_index: int
    class_id: int
    bbox: torch.Tensor
    failure_type: str
    severity: float
    budget: int

    def __post_init__(self) -> None:
        self.gt_uid = str(self.gt_uid)
        self.image_id = str(self.image_id)
        self.gt_index = int(self.gt_index)
        self.class_id = int(self.class_id)
        self.bbox = torch.as_tensor(self.bbox, dtype=torch.float32).detach()
        if self.bbox.numel() != 4:
            raise ValueError("DenseTarget bbox must contain exactly four values.")
        self.bbox = self.bbox.reshape(4)
        self.failure_type = str(self.failure_type)
        self.severity = float(self.severity)
        self.budget = int(self.budget)


@dataclass(slots=True)
class DensePlan:
    targets: list[DenseTarget]

    def for_image(self, image_id: Any) -> list[DenseTarget]:
        image_key = _normalize_image_id(image_id)
        return [target for target in self.targets if target.image_id == image_key]

    def __len__(self) -> int:
        return len(self.targets)


class CandidateDensifier(nn.Module):
    def __init__(self, config: CandidateDensificationConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._epoch_dense_targets = 0
        self._epoch_dense_points = 0
        self._epoch_dense_batches = 0
        self._epoch_dense_loss_sum = 0.0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._epoch_dense_targets = 0
        self._epoch_dense_points = 0
        self._epoch_dense_batches = 0
        self._epoch_dense_loss_sum = 0.0

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
        return float(self.config.lambda_dense) * self.warmup_factor()

    def should_apply(self, *, mdmbpp: MDMBPlus | None) -> bool:
        return mdmbpp is not None and self.loss_weight() > 0.0

    @torch.no_grad()
    def plan(
        self,
        *,
        mdmbpp: MDMBPlus | None,
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_shapes: Sequence[Sequence[int]],
    ) -> DensePlan:
        if mdmbpp is None or not self.config.enabled:
            return DensePlan(targets=[])

        dense_targets: list[DenseTarget] = []
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
            image_targets: list[DenseTarget] = []

            for entry in entries:
                match = self._match_entry_to_target(
                    entry=entry,
                    gt_boxes=gt_boxes,
                    gt_boxes_norm=gt_boxes_norm,
                    gt_labels=gt_labels,
                    used_gt_indices=used_gt_indices,
                )
                if match is None:
                    continue
                gt_index = match
                used_gt_indices.add(gt_index)
                image_targets.append(
                    DenseTarget(
                        gt_uid=entry.gt_uid,
                        image_id=image_key,
                        gt_index=gt_index,
                        class_id=entry.class_id,
                        bbox=gt_boxes[gt_index].detach(),
                        failure_type=entry.failure_type,
                        severity=entry.severity,
                        budget=self.budget_for(entry.severity),
                    )
                )
                if len(image_targets) >= self.config.max_dense_targets_per_image:
                    break

            dense_targets.extend(image_targets)

        return DensePlan(targets=dense_targets)

    def budget_for(self, severity: float) -> int:
        budget = self.config.base_extra_candidates + math.floor(
            self.config.severity_budget_scale * float(severity)
        )
        return max(1, min(self.config.max_extra_candidates_per_gt, int(budget)))

    def region_scale_for(self, severity: float) -> float:
        scale = self.config.base_region_scale + self.config.severity_region_scale * float(severity)
        return min(self.config.max_region_scale, max(self.config.base_region_scale, scale))

    def record_dense_step(
        self,
        *,
        num_targets: int,
        num_points: int,
        loss: torch.Tensor | None,
    ) -> None:
        self._epoch_dense_targets += int(num_targets)
        self._epoch_dense_points += int(num_points)
        self._epoch_dense_batches += 1
        if loss is not None and int(num_points) > 0:
            self._epoch_dense_loss_sum += float(loss.detach().cpu().item())

    def summary(self) -> dict[str, Any]:
        mean_points_per_target = self._epoch_dense_points / float(
            max(self._epoch_dense_targets, 1)
        )
        mean_loss = self._epoch_dense_loss_sum / float(max(self._epoch_dense_batches, 1))
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_factor": self.warmup_factor(),
            "lambda_dense": self.loss_weight(),
            "dense_targets": self._epoch_dense_targets,
            "dense_points": self._epoch_dense_points,
            "dense_batches": self._epoch_dense_batches,
            "mean_dense_points_per_target": mean_points_per_target,
            "mean_candidate_dense_loss": mean_loss,
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


def load_candidate_densification_config(
    path: str | Path,
    *,
    arch: str | None = None,
) -> CandidateDensificationConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            "Candidate Densification YAML must contain a mapping at the top level: "
            f"{config_path}"
        )
    return CandidateDensificationConfig.from_mapping(raw, arch=arch)


def build_candidate_densifier_from_config(
    raw_config: Mapping[str, Any] | CandidateDensificationConfig,
    *,
    arch: str | None = None,
) -> CandidateDensifier | None:
    config = (
        raw_config
        if isinstance(raw_config, CandidateDensificationConfig)
        else CandidateDensificationConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return CandidateDensifier(config)


def build_candidate_densifier_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> CandidateDensifier | None:
    config = load_candidate_densification_config(path, arch=arch)
    if not config.enabled:
        return None
    return CandidateDensifier(config)


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Candidate Densification image_id tensor must contain one scalar.")
        value = value.item()
    return str(value)

