from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from .mdmb import normalize_arch, normalize_xyxy_boxes
from .mdmbpp import MDMBPlus


_DEFAULT_FAILURE_TYPES: tuple[str, ...] = (
    "cls_confusion",
    "score_suppression",
    "nms_suppression",
    "loc_near_miss",
)

_SUPPORTED_FAILURE_TYPES = set(_DEFAULT_FAILURE_TYPES) | {"candidate_missing"}


@dataclass(frozen=True, slots=True)
class RASDConfig:
    enabled: bool = False
    warmup_epochs: int = 5
    lambda_rasd: float = 0.03
    temperature: float = 0.2
    alpha_contrastive: float = 1.0
    min_relapse_streak: int = 1
    min_severity: float = 1.0
    min_support_score: float = 0.2
    max_support_age: int = 15
    record_match_threshold: float = 0.95
    max_targets_per_image: int = 5
    max_targets_per_batch: int = 16
    severity_weight_scale: float = 0.25
    relapse_weight_scale: float = 0.25
    max_target_weight: float = 3.0
    roi_output_size: int = 7
    roi_sampling_ratio: int = 2
    normalize_features: bool = True
    confuser_enabled: bool = False
    confuser_iou_threshold: float = 0.3
    confuser_min_score: float = 0.05
    confuser_max_candidates_per_gt: int = 1
    failure_types: tuple[str, ...] = _DEFAULT_FAILURE_TYPES
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "RASDConfig":
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

        feature_pool = {}
        top_feature_pool = data.get("feature_pool", {})
        if isinstance(top_feature_pool, Mapping):
            feature_pool.update(top_feature_pool)
        override_feature_pool = model_overrides.get("feature_pool", {})
        if isinstance(override_feature_pool, Mapping):
            feature_pool.update(override_feature_pool)

        confuser = {}
        top_confuser = data.get("confuser", {})
        if isinstance(top_confuser, Mapping):
            confuser.update(top_confuser)
        override_confuser = model_overrides.get("confuser", {})
        if isinstance(override_confuser, Mapping):
            confuser.update(override_confuser)

        raw_failure_types = read("failure_types", _DEFAULT_FAILURE_TYPES)
        if isinstance(raw_failure_types, str):
            failure_types = (raw_failure_types,)
        else:
            failure_types = tuple(str(value) for value in raw_failure_types)

        config = cls(
            enabled=bool(read("enabled", False)),
            warmup_epochs=int(read("warmup_epochs", 5)),
            lambda_rasd=float(read("lambda_rasd", 0.03)),
            temperature=float(read("temperature", 0.2)),
            alpha_contrastive=float(read("alpha_contrastive", 1.0)),
            min_relapse_streak=int(read("min_relapse_streak", 1)),
            min_severity=float(read("min_severity", 1.0)),
            min_support_score=float(read("min_support_score", 0.2)),
            max_support_age=int(read("max_support_age", 15)),
            record_match_threshold=float(read("record_match_threshold", 0.95)),
            max_targets_per_image=int(read("max_targets_per_image", 5)),
            max_targets_per_batch=int(read("max_targets_per_batch", 16)),
            severity_weight_scale=float(read("severity_weight_scale", 0.25)),
            relapse_weight_scale=float(read("relapse_weight_scale", 0.25)),
            max_target_weight=float(read("max_target_weight", 3.0)),
            roi_output_size=int(feature_pool.get("output_size", 7)),
            roi_sampling_ratio=int(feature_pool.get("sampling_ratio", 2)),
            normalize_features=bool(feature_pool.get("normalize", True)),
            confuser_enabled=bool(confuser.get("enabled", False)),
            confuser_iou_threshold=float(confuser.get("iou_threshold", 0.3)),
            confuser_min_score=float(confuser.get("min_score", 0.05)),
            confuser_max_candidates_per_gt=int(confuser.get("max_candidates_per_gt", 1)),
            failure_types=failure_types,
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("RASD warmup_epochs must be >= 0.")
        if self.lambda_rasd < 0.0:
            raise ValueError("RASD lambda_rasd must be >= 0.")
        if self.temperature <= 0.0:
            raise ValueError("RASD temperature must be > 0.")
        if self.alpha_contrastive < 0.0:
            raise ValueError("RASD alpha_contrastive must be >= 0.")
        if self.min_relapse_streak < 1:
            raise ValueError("RASD min_relapse_streak must be >= 1.")
        if self.min_severity < 0.0:
            raise ValueError("RASD min_severity must be >= 0.")
        if self.min_support_score < 0.0:
            raise ValueError("RASD min_support_score must be >= 0.")
        if self.max_support_age < 0:
            raise ValueError("RASD max_support_age must be >= 0.")
        for name in (
            "record_match_threshold",
            "confuser_iou_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"RASD {name} must satisfy 0 <= value <= 1.")
        if self.max_targets_per_image < 1:
            raise ValueError("RASD max_targets_per_image must be >= 1.")
        if self.max_targets_per_batch < 1:
            raise ValueError("RASD max_targets_per_batch must be >= 1.")
        if self.severity_weight_scale < 0.0:
            raise ValueError("RASD severity_weight_scale must be >= 0.")
        if self.relapse_weight_scale < 0.0:
            raise ValueError("RASD relapse_weight_scale must be >= 0.")
        if self.max_target_weight < 1.0:
            raise ValueError("RASD max_target_weight must be >= 1.")
        if self.roi_output_size < 1:
            raise ValueError("RASD feature_pool.output_size must be >= 1.")
        if self.roi_sampling_ratio < 0:
            raise ValueError("RASD feature_pool.sampling_ratio must be >= 0.")
        if self.confuser_min_score < 0.0:
            raise ValueError("RASD confuser.min_score must be >= 0.")
        if self.confuser_max_candidates_per_gt < 1:
            raise ValueError("RASD confuser.max_candidates_per_gt must be >= 1.")
        if not self.failure_types:
            raise ValueError("RASD failure_types must not be empty.")
        unsupported = sorted(set(self.failure_types) - _SUPPORTED_FAILURE_TYPES)
        if unsupported:
            raise ValueError(f"RASD failure_types contains unsupported values: {unsupported}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_epochs": self.warmup_epochs,
            "lambda_rasd": self.lambda_rasd,
            "temperature": self.temperature,
            "alpha_contrastive": self.alpha_contrastive,
            "min_relapse_streak": self.min_relapse_streak,
            "min_severity": self.min_severity,
            "min_support_score": self.min_support_score,
            "max_support_age": self.max_support_age,
            "record_match_threshold": self.record_match_threshold,
            "max_targets_per_image": self.max_targets_per_image,
            "max_targets_per_batch": self.max_targets_per_batch,
            "severity_weight_scale": self.severity_weight_scale,
            "relapse_weight_scale": self.relapse_weight_scale,
            "max_target_weight": self.max_target_weight,
            "feature_pool": {
                "output_size": self.roi_output_size,
                "sampling_ratio": self.roi_sampling_ratio,
                "normalize": self.normalize_features,
            },
            "confuser": {
                "enabled": self.confuser_enabled,
                "iou_threshold": self.confuser_iou_threshold,
                "min_score": self.confuser_min_score,
                "max_candidates_per_gt": self.confuser_max_candidates_per_gt,
            },
            "failure_types": list(self.failure_types),
            "arch": self.arch,
        }


@dataclass(slots=True)
class RASDConfuser:
    bbox: torch.Tensor
    label: int
    score: float
    iou_to_gt: float
    stage: str

    def __post_init__(self) -> None:
        self.bbox = torch.as_tensor(self.bbox, dtype=torch.float32).detach().reshape(4).cpu()
        self.label = int(self.label)
        self.score = float(self.score)
        self.iou_to_gt = float(self.iou_to_gt)
        self.stage = str(self.stage)


@dataclass(slots=True)
class RASDTarget:
    gt_uid: str
    image_id: str
    image_index: int
    gt_index: int
    class_id: int
    bbox: torch.Tensor
    failure_type: str
    severity: float
    relapse_count: int
    support_epoch: int
    support_score: float
    support_feature: torch.Tensor
    weight: float
    confusers: list[RASDConfuser] | None = None

    def __post_init__(self) -> None:
        self.gt_uid = str(self.gt_uid)
        self.image_id = str(self.image_id)
        self.image_index = int(self.image_index)
        self.gt_index = int(self.gt_index)
        self.class_id = int(self.class_id)
        self.bbox = torch.as_tensor(self.bbox, dtype=torch.float32).detach().reshape(4)
        self.failure_type = str(self.failure_type)
        self.severity = float(self.severity)
        self.relapse_count = int(self.relapse_count)
        self.support_epoch = int(self.support_epoch)
        self.support_score = float(self.support_score)
        self.support_feature = torch.as_tensor(
            self.support_feature,
            dtype=torch.float32,
        ).detach().flatten().cpu()
        self.weight = float(self.weight)
        self.confusers = [] if self.confusers is None else list(self.confusers)


@dataclass(slots=True)
class RASDPlan:
    targets: list[RASDTarget]

    def for_image(self, image_id: Any) -> list[RASDTarget]:
        image_key = _normalize_image_id(image_id)
        return [target for target in self.targets if target.image_id == image_key]

    def __len__(self) -> int:
        return len(self.targets)


class RelapseAwareSupportDistillation(nn.Module):
    def __init__(self, config: RASDConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._epoch_targets = 0
        self._epoch_losses = 0
        self._epoch_relapse_targets = 0
        self._epoch_confuser_targets = 0
        self._epoch_severity_sum = 0.0
        self._epoch_support_age_sum = 0.0
        self._epoch_weight_sum = 0.0
        self._epoch_support_loss_sum = 0.0
        self._epoch_contrastive_loss_sum = 0.0
        self._epoch_skipped_no_support = 0
        self._epoch_skipped_support_too_old = 0
        self._epoch_skipped_low_support_score = 0
        self._epoch_skipped_no_entry_match = 0
        self._epoch_skipped_no_feature = 0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._epoch_targets = 0
        self._epoch_losses = 0
        self._epoch_relapse_targets = 0
        self._epoch_confuser_targets = 0
        self._epoch_severity_sum = 0.0
        self._epoch_support_age_sum = 0.0
        self._epoch_weight_sum = 0.0
        self._epoch_support_loss_sum = 0.0
        self._epoch_contrastive_loss_sum = 0.0
        self._epoch_skipped_no_support = 0
        self._epoch_skipped_support_too_old = 0
        self._epoch_skipped_low_support_score = 0
        self._epoch_skipped_no_entry_match = 0
        self._epoch_skipped_no_feature = 0

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
        return float(self.config.lambda_rasd) * self.warmup_factor()

    def should_apply(self, *, mdmbpp: MDMBPlus | None) -> bool:
        if not self.config.enabled or self.loss_weight() <= 0.0:
            return False
        if mdmbpp is None:
            raise RuntimeError("RASD requires MDMB++ to be enabled.")
        if not bool(mdmbpp.config.store_support_feature):
            raise RuntimeError("RASD requires mdmbpp.store_support_feature=true.")
        return True

    @torch.no_grad()
    def plan(
        self,
        *,
        mdmbpp: MDMBPlus | None,
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_shapes: Sequence[Sequence[int]],
    ) -> RASDPlan:
        if mdmbpp is None or not self.config.enabled:
            return RASDPlan(targets=[])

        rasd_targets: list[RASDTarget] = []
        failure_types = set(self.config.failure_types)

        for image_index, target in enumerate(targets):
            if len(rasd_targets) >= self.config.max_targets_per_batch:
                break

            image_id = target.get("image_id", torch.tensor(image_index))
            image_key = _normalize_image_id(image_id)
            entries = [
                entry
                for entry in mdmbpp.get_image_entries(image_key)
                if entry.failure_type in failure_types
                and entry.relapse
                and float(entry.severity) >= self.config.min_severity
                and int(entry.consecutive_miss_count) >= self.config.min_relapse_streak
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
            image_targets: list[RASDTarget] = []

            for entry in entries:
                support = entry.support
                if support is None or support.feature is None:
                    self._epoch_skipped_no_support += 1
                    continue
                if float(support.score) < self.config.min_support_score:
                    self._epoch_skipped_low_support_score += 1
                    continue
                support_feature_epoch = getattr(support, "feature_epoch", None)
                if support_feature_epoch is None:
                    support_feature_epoch = support.epoch
                support_age = max(0, int(self.current_epoch) - int(support_feature_epoch))
                if support_age > self.config.max_support_age:
                    self._epoch_skipped_support_too_old += 1
                    continue
                if support.feature.numel() == 0:
                    self._epoch_skipped_no_feature += 1
                    continue

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
                relapse_count = 0 if record is None else int(record.relapse_count)
                gt_index = int(match)
                used_gt_indices.add(gt_index)
                confusers = self._select_confusers(
                    entry=entry,
                    image_shape=image_shapes[image_index],
                )
                image_targets.append(
                    RASDTarget(
                        gt_uid=entry.gt_uid,
                        image_id=image_key,
                        image_index=image_index,
                        gt_index=gt_index,
                        class_id=entry.class_id,
                        bbox=gt_boxes[gt_index].detach(),
                        failure_type=entry.failure_type,
                        severity=entry.severity,
                        relapse_count=relapse_count,
                        support_epoch=support_feature_epoch,
                        support_score=support.score,
                        support_feature=support.feature,
                        weight=self.target_weight_for(
                            severity=entry.severity,
                            relapse_count=relapse_count,
                        ),
                        confusers=confusers,
                    )
                )
                if len(image_targets) >= self.config.max_targets_per_image:
                    break
                if len(rasd_targets) + len(image_targets) >= self.config.max_targets_per_batch:
                    break

            rasd_targets.extend(image_targets)

        return RASDPlan(targets=rasd_targets)

    def compute_loss(
        self,
        *,
        plan: RASDPlan,
        features: Mapping[str, torch.Tensor],
        image_shapes: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        template = _first_feature(features)
        zero = template.new_zeros(())
        if len(plan) == 0:
            return zero, {
                "losses": 0,
                "support_loss_sum": 0.0,
                "contrastive_loss_sum": 0.0,
                "confuser_targets": 0,
                "target_weight_sum": 0.0,
                "skipped_no_feature": 0,
            }

        boxes_per_image: list[torch.Tensor] = []
        ordered_targets: list[RASDTarget] = []
        device = template.device
        for image_index in range(len(image_shapes)):
            image_targets = [
                target for target in plan.targets if target.image_index == image_index
            ]
            if image_targets:
                boxes = torch.stack(
                    [
                        target.bbox.to(device=device, dtype=torch.float32)
                        for target in image_targets
                    ],
                    dim=0,
                )
                ordered_targets.extend(image_targets)
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
            boxes_per_image.append(boxes)

        if not ordered_targets:
            return zero, {
                "losses": 0,
                "support_loss_sum": 0.0,
                "contrastive_loss_sum": 0.0,
                "confuser_targets": 0,
                "target_weight_sum": 0.0,
                "skipped_no_feature": len(plan.targets),
            }

        current_features = pool_multiscale_box_features(
            features=features,
            boxes_per_image=boxes_per_image,
            image_shapes=image_shapes,
            output_size=self.config.roi_output_size,
            sampling_ratio=self.config.roi_sampling_ratio,
            normalize=self.config.normalize_features,
        )
        if current_features.shape[0] != len(ordered_targets):
            return zero, {
                "losses": 0,
                "support_loss_sum": 0.0,
                "contrastive_loss_sum": 0.0,
                "confuser_targets": 0,
                "target_weight_sum": 0.0,
                "skipped_no_feature": len(ordered_targets),
            }

        valid_current: list[torch.Tensor] = []
        valid_support: list[torch.Tensor] = []
        valid_targets: list[RASDTarget] = []
        weights: list[float] = []
        skipped_no_feature = 0
        for index, rasd_target in enumerate(ordered_targets):
            support_feature = rasd_target.support_feature.to(
                device=current_features.device,
                dtype=current_features.dtype,
            )
            if support_feature.numel() != current_features.shape[1]:
                skipped_no_feature += 1
                continue
            valid_current.append(current_features[index])
            valid_support.append(support_feature)
            valid_targets.append(rasd_target)
            weights.append(float(rasd_target.weight))

        if not valid_current:
            return zero, {
                "losses": 0,
                "support_loss_sum": 0.0,
                "contrastive_loss_sum": 0.0,
                "confuser_targets": 0,
                "target_weight_sum": 0.0,
                "skipped_no_feature": skipped_no_feature,
            }

        current = torch.stack(valid_current, dim=0)
        support = torch.stack(valid_support, dim=0)
        if self.config.normalize_features:
            current = F.normalize(current, p=2, dim=1)
            support = F.normalize(support, p=2, dim=1)
        support_losses = 1.0 - (current * support).sum(dim=1).clamp(min=-1.0, max=1.0)
        contrastive_losses, confuser_targets = self._compute_confuser_losses(
            current=current,
            support=support,
            targets=valid_targets,
            features=features,
            image_shapes=image_shapes,
        )
        weight_tensor = torch.tensor(
            weights,
            dtype=support_losses.dtype,
            device=support_losses.device,
        )
        combined_losses = support_losses + float(self.config.alpha_contrastive) * contrastive_losses
        loss = (combined_losses * weight_tensor).sum() / float(max(len(valid_current), 1))
        loss = loss * self.loss_weight()

        return loss, {
            "losses": len(valid_current),
            "support_loss_sum": float(support_losses.detach().sum().cpu().item()),
            "contrastive_loss_sum": float(contrastive_losses.detach().sum().cpu().item()),
            "confuser_targets": confuser_targets,
            "target_weight_sum": float(weight_tensor.detach().sum().cpu().item()),
            "skipped_no_feature": skipped_no_feature,
        }

    def record_step(
        self,
        *,
        plan: RASDPlan,
        stats: Mapping[str, float | int],
    ) -> None:
        self._epoch_targets += len(plan.targets)
        self._epoch_relapse_targets += len(plan.targets)
        self._epoch_severity_sum += sum(float(target.severity) for target in plan.targets)
        self._epoch_support_age_sum += sum(
            max(0, int(self.current_epoch) - int(target.support_epoch))
            for target in plan.targets
        )
        self._epoch_weight_sum += sum(float(target.weight) for target in plan.targets)
        self._epoch_losses += int(stats.get("losses", 0))
        self._epoch_confuser_targets += int(stats.get("confuser_targets", 0))
        self._epoch_support_loss_sum += float(stats.get("support_loss_sum", 0.0))
        self._epoch_contrastive_loss_sum += float(stats.get("contrastive_loss_sum", 0.0))
        self._epoch_skipped_no_feature += int(stats.get("skipped_no_feature", 0))

    def summary(self) -> dict[str, Any]:
        mean_severity = self._epoch_severity_sum / float(max(self._epoch_targets, 1))
        mean_support_age = self._epoch_support_age_sum / float(max(self._epoch_targets, 1))
        mean_target_weight = self._epoch_weight_sum / float(max(self._epoch_targets, 1))
        mean_support_loss = self._epoch_support_loss_sum / float(max(self._epoch_losses, 1))
        mean_contrastive_loss = self._epoch_contrastive_loss_sum / float(
            max(self._epoch_confuser_targets, 1)
        )
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_factor": self.warmup_factor(),
            "lambda_rasd": self.loss_weight(),
            "targets": self._epoch_targets,
            "losses": self._epoch_losses,
            "relapse_targets": self._epoch_relapse_targets,
            "confuser_targets": self._epoch_confuser_targets,
            "mean_severity": mean_severity,
            "mean_support_age": mean_support_age,
            "mean_target_weight": mean_target_weight,
            "mean_support_loss": mean_support_loss,
            "mean_contrastive_loss": mean_contrastive_loss,
            "skipped_no_support": self._epoch_skipped_no_support,
            "skipped_support_too_old": self._epoch_skipped_support_too_old,
            "skipped_low_support_score": self._epoch_skipped_low_support_score,
            "skipped_no_entry_match": self._epoch_skipped_no_entry_match,
            "skipped_no_feature": self._epoch_skipped_no_feature,
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

    def target_weight_for(self, *, severity: float, relapse_count: int) -> float:
        weight = (
            1.0
            + self.config.severity_weight_scale * float(severity)
            + self.config.relapse_weight_scale * float(relapse_count)
        )
        return min(float(self.config.max_target_weight), max(1.0, weight))

    def _select_confusers(
        self,
        *,
        entry,
        image_shape: Sequence[int],
    ) -> list[RASDConfuser]:
        if not self.config.confuser_enabled or str(entry.failure_type) != "cls_confusion":
            return []

        candidates = [
            candidate
            for candidate in entry.topk_candidates
            if int(candidate.label) != int(entry.class_id)
            and float(candidate.iou_to_gt) >= self.config.confuser_iou_threshold
            and float(candidate.score) >= self.config.confuser_min_score
        ]
        if not candidates:
            return []

        candidates.sort(
            key=lambda candidate: (
                -float(candidate.score),
                -float(candidate.iou_to_gt),
                -int(bool(candidate.survived_selection)),
            )
        )

        confusers: list[RASDConfuser] = []
        for candidate in candidates[: self.config.confuser_max_candidates_per_gt]:
            bbox = _normalized_box_to_absolute_xyxy(candidate.box, image_shape)
            if bbox is None:
                continue
            confusers.append(
                RASDConfuser(
                    bbox=bbox,
                    label=int(candidate.label),
                    score=float(candidate.score),
                    iou_to_gt=float(candidate.iou_to_gt),
                    stage=str(candidate.stage),
                )
            )
        return confusers

    def _compute_confuser_losses(
        self,
        *,
        current: torch.Tensor,
        support: torch.Tensor,
        targets: Sequence[RASDTarget],
        features: Mapping[str, torch.Tensor],
        image_shapes: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, int]:
        losses = current.new_zeros((current.shape[0],))
        confuser_boxes_per_image: list[list[torch.Tensor]] = [
            [] for _ in range(len(image_shapes))
        ]
        confuser_owners: list[int] = []

        for target_index, target in enumerate(targets):
            for confuser in target.confusers or ():
                image_index = int(target.image_index)
                if image_index < 0 or image_index >= len(confuser_boxes_per_image):
                    continue
                confuser_boxes_per_image[image_index].append(
                    confuser.bbox.to(device=current.device, dtype=torch.float32)
                )
                confuser_owners.append(target_index)

        if not confuser_owners:
            return losses, 0

        boxes_per_image = [
            torch.stack(boxes, dim=0)
            if boxes
            else torch.empty((0, 4), dtype=torch.float32, device=current.device)
            for boxes in confuser_boxes_per_image
        ]
        confuser_features = pool_multiscale_box_features(
            features=features,
            boxes_per_image=boxes_per_image,
            image_shapes=image_shapes,
            output_size=self.config.roi_output_size,
            sampling_ratio=self.config.roi_sampling_ratio,
            normalize=self.config.normalize_features,
        )
        if confuser_features.shape[0] != len(confuser_owners):
            return losses, 0

        confuser_features = confuser_features.detach()
        if self.config.normalize_features:
            confuser_features = F.normalize(confuser_features, p=2, dim=1)

        by_target: list[list[torch.Tensor]] = [[] for _ in targets]
        for feature_index, target_index in enumerate(confuser_owners):
            if 0 <= target_index < len(by_target):
                by_target[target_index].append(confuser_features[feature_index])

        confuser_targets = 0
        labels = torch.zeros((1,), dtype=torch.long, device=current.device)
        for target_index, negative_features in enumerate(by_target):
            if not negative_features:
                continue
            negatives = torch.stack(negative_features, dim=0).to(
                device=current.device,
                dtype=current.dtype,
            )
            positive_logit = (current[target_index] * support[target_index].detach()).sum().view(1)
            negative_logits = negatives.matmul(current[target_index].unsqueeze(1)).flatten()
            logits = torch.cat((positive_logit, negative_logits), dim=0).unsqueeze(0)
            logits = logits / float(self.config.temperature)
            losses[target_index] = F.cross_entropy(logits, labels, reduction="mean")
            confuser_targets += 1

        return losses, confuser_targets

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


def pool_multiscale_box_features(
    *,
    features: Mapping[str, torch.Tensor],
    boxes_per_image: Sequence[torch.Tensor],
    image_shapes: Sequence[Sequence[int]],
    output_size: int,
    sampling_ratio: int,
    normalize: bool,
) -> torch.Tensor:
    feature_dict = OrderedDict((str(key), value) for key, value in features.items())
    template = _first_feature(feature_dict)
    total_boxes = sum(int(boxes.shape[0]) for boxes in boxes_per_image)
    if total_boxes == 0:
        return template.new_zeros((0, int(template.shape[1])))

    device = template.device
    prepared_boxes = [
        boxes.to(device=device, dtype=torch.float32).reshape(-1, 4)
        for boxes in boxes_per_image
    ]
    prepared_shapes = [
        (int(shape[0]), int(shape[1]))
        for shape in image_shapes
    ]
    pooler = MultiScaleRoIAlign(
        featmap_names=list(feature_dict.keys()),
        output_size=int(output_size),
        sampling_ratio=int(sampling_ratio),
    )
    pooled = pooler(feature_dict, prepared_boxes, prepared_shapes)
    vectors = pooled.flatten(start_dim=2).mean(dim=2)
    if normalize:
        vectors = F.normalize(vectors, p=2, dim=1)
    return vectors


def _normalized_box_to_absolute_xyxy(
    box: torch.Tensor | Sequence[float],
    image_shape: Sequence[int],
) -> torch.Tensor | None:
    values = torch.as_tensor(box, dtype=torch.float32).detach().flatten()
    if values.numel() != 4:
        return None
    if len(image_shape) != 2:
        raise ValueError("RASD image_shape must contain exactly two values: (height, width).")

    height = max(1, int(image_shape[0]))
    width = max(1, int(image_shape[1]))
    scale = values.new_tensor([width, height, width, height], dtype=torch.float32)
    absolute = values.clamp(min=0.0, max=1.0) * scale
    left = torch.minimum(absolute[0], absolute[2]).clamp(min=0.0, max=float(width))
    top = torch.minimum(absolute[1], absolute[3]).clamp(min=0.0, max=float(height))
    right = torch.maximum(absolute[0], absolute[2]).clamp(min=0.0, max=float(width))
    bottom = torch.maximum(absolute[1], absolute[3]).clamp(min=0.0, max=float(height))
    if bool((right <= left).item()) or bool((bottom <= top).item()):
        return None
    return torch.stack((left, top, right, bottom), dim=0).cpu()


def load_rasd_config(path: str | Path, *, arch: str | None = None) -> RASDConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"RASD YAML must contain a mapping at the top level: {config_path}")
    return RASDConfig.from_mapping(raw, arch=arch)


def build_rasd_from_config(
    raw_config: Mapping[str, Any] | RASDConfig,
    *,
    arch: str | None = None,
) -> RelapseAwareSupportDistillation | None:
    config = (
        raw_config
        if isinstance(raw_config, RASDConfig)
        else RASDConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return RelapseAwareSupportDistillation(config)


def build_rasd_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> RelapseAwareSupportDistillation | None:
    config = load_rasd_config(path, arch=arch)
    if not config.enabled:
        return None
    return RelapseAwareSupportDistillation(config)


def _first_feature(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    try:
        return next(iter(features.values()))
    except StopIteration as exc:
        raise ValueError("RASD requires at least one feature map.") from exc


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("RASD image_id tensor must contain one scalar.")
        value = value.item()
    return str(value)
