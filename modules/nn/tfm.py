from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
import yaml

from .mdmb import normalize_arch, normalize_xyxy_boxes


TFMState: TypeAlias = Literal[
    "missing_assignment",
    "classification_confusion",
    "localization_weak",
    "score_weak",
    "quality_ranking_weak",
    "detected_like",
    "recovered",
    "unknown",
]

_TFM_STATES: tuple[str, ...] = (
    "missing_assignment",
    "classification_confusion",
    "localization_weak",
    "score_weak",
    "quality_ranking_weak",
    "detected_like",
    "recovered",
    "unknown",
)

_RECOVERED_STATES = {"detected_like", "recovered"}

_DEFAULT_FAILURE_TYPE_PRIORS = {
    "missing_assignment": 1.0,
    "classification_confusion": 0.8,
    "localization_weak": 0.7,
    "score_weak": 0.6,
    "quality_ranking_weak": 0.5,
    "detected_like": 0.0,
    "recovered": 0.0,
    "unknown": 0.4,
}


@dataclass(frozen=True, slots=True)
class TFMRiskConfig:
    miss_streak_weight: float = 1.0
    total_miss_weight: float = 0.5
    relapse_weight: float = 1.0
    failure_type_weight: float = 0.5
    recent_recovery_weight: float = 1.0
    recovery_decay_epochs: int = 5
    min_risk: float = 0.0
    max_risk: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TFMRiskConfig":
        data = dict(raw or {})
        config = cls(
            miss_streak_weight=float(data.get("miss_streak_weight", 1.0)),
            total_miss_weight=float(data.get("total_miss_weight", 0.5)),
            relapse_weight=float(data.get("relapse_weight", 1.0)),
            failure_type_weight=float(data.get("failure_type_weight", 0.5)),
            recent_recovery_weight=float(data.get("recent_recovery_weight", 1.0)),
            recovery_decay_epochs=int(data.get("recovery_decay_epochs", 5)),
            min_risk=float(data.get("min_risk", 0.0)),
            max_risk=float(data.get("max_risk", 1.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for field_name in (
            "miss_streak_weight",
            "total_miss_weight",
            "relapse_weight",
            "failure_type_weight",
            "recent_recovery_weight",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"TFM risk.{field_name} must be >= 0.")
        if self.recovery_decay_epochs < 0:
            raise ValueError("TFM risk.recovery_decay_epochs must be >= 0.")
        if not 0.0 <= self.min_risk <= 1.0:
            raise ValueError("TFM risk.min_risk must satisfy 0 <= value <= 1.")
        if not 0.0 <= self.max_risk <= 1.0:
            raise ValueError("TFM risk.max_risk must satisfy 0 <= value <= 1.")
        if self.min_risk > self.max_risk:
            raise ValueError("TFM risk.min_risk must be <= risk.max_risk.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "miss_streak_weight": self.miss_streak_weight,
            "total_miss_weight": self.total_miss_weight,
            "relapse_weight": self.relapse_weight,
            "failure_type_weight": self.failure_type_weight,
            "recent_recovery_weight": self.recent_recovery_weight,
            "recovery_decay_epochs": self.recovery_decay_epochs,
            "min_risk": self.min_risk,
            "max_risk": self.max_risk,
        }


@dataclass(frozen=True, slots=True)
class TFMSupportConfig:
    enabled: bool = False
    min_quality: float = 0.5
    quality_margin: float = 0.05
    refresh_age: int = 15
    ema_momentum: float = 0.9
    max_feature_elements: int = 4096

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TFMSupportConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", False)),
            min_quality=float(data.get("min_quality", 0.5)),
            quality_margin=float(data.get("quality_margin", 0.05)),
            refresh_age=int(data.get("refresh_age", 15)),
            ema_momentum=float(data.get("ema_momentum", 0.9)),
            max_feature_elements=int(data.get("max_feature_elements", 4096)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.min_quality < 0.0:
            raise ValueError("TFM support.min_quality must be >= 0.")
        if self.quality_margin < 0.0:
            raise ValueError("TFM support.quality_margin must be >= 0.")
        if self.refresh_age < 0:
            raise ValueError("TFM support.refresh_age must be >= 0.")
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError("TFM support.ema_momentum must satisfy 0 <= value <= 1.")
        if self.max_feature_elements < 1:
            raise ValueError("TFM support.max_feature_elements must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_quality": self.min_quality,
            "quality_margin": self.quality_margin,
            "refresh_age": self.refresh_age,
            "ema_momentum": self.ema_momentum,
            "max_feature_elements": self.max_feature_elements,
        }


@dataclass(frozen=True, slots=True)
class TFMConfig:
    enabled: bool = True
    warmup_epochs: int = 0
    record_match_threshold: float = 0.95
    detected_quality_threshold: float = 0.5
    max_records: int | None = None
    risk: TFMRiskConfig = field(default_factory=TFMRiskConfig)
    support: TFMSupportConfig = field(default_factory=TFMSupportConfig)
    failure_type_priors: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_FAILURE_TYPE_PRIORS)
    )
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "TFMConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))

        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        risk_mapping: dict[str, Any] = {}
        top_risk = data.get("risk", {})
        if isinstance(top_risk, Mapping):
            risk_mapping.update(top_risk)
        override_risk = model_overrides.get("risk", {})
        if isinstance(override_risk, Mapping):
            risk_mapping.update(override_risk)

        support_mapping: dict[str, Any] = {}
        top_support = data.get("support", {})
        if isinstance(top_support, Mapping):
            support_mapping.update(top_support)
        override_support = model_overrides.get("support", {})
        if isinstance(override_support, Mapping):
            support_mapping.update(override_support)

        priors = dict(_DEFAULT_FAILURE_TYPE_PRIORS)
        top_priors = data.get("failure_type_priors", {})
        if isinstance(top_priors, Mapping):
            priors.update({str(key): float(value) for key, value in top_priors.items()})
        override_priors = model_overrides.get("failure_type_priors", {})
        if isinstance(override_priors, Mapping):
            priors.update({str(key): float(value) for key, value in override_priors.items()})

        max_records = model_overrides.get("max_records", data.get("max_records"))
        config = cls(
            enabled=bool(model_overrides.get("enabled", data.get("enabled", True))),
            warmup_epochs=int(model_overrides.get("warmup_epochs", data.get("warmup_epochs", 0))),
            record_match_threshold=float(
                model_overrides.get(
                    "record_match_threshold",
                    data.get("record_match_threshold", 0.95),
                )
            ),
            detected_quality_threshold=float(
                model_overrides.get(
                    "detected_quality_threshold",
                    data.get("detected_quality_threshold", 0.5),
                )
            ),
            max_records=None if max_records is None else int(max_records),
            risk=TFMRiskConfig.from_mapping(risk_mapping),
            support=TFMSupportConfig.from_mapping(support_mapping),
            failure_type_priors=priors,
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("TFM warmup_epochs must be >= 0.")
        if not 0.0 <= self.record_match_threshold <= 1.0:
            raise ValueError("TFM record_match_threshold must satisfy 0 <= value <= 1.")
        if self.detected_quality_threshold < 0.0:
            raise ValueError("TFM detected_quality_threshold must be >= 0.")
        if self.max_records is not None and self.max_records < 1:
            raise ValueError("TFM max_records must be >= 1 when provided.")
        missing = [name for name in _TFM_STATES if name not in self.failure_type_priors]
        if missing:
            raise ValueError(
                "TFM failure_type_priors must include every TFM state. "
                f"Missing: {missing}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_epochs": self.warmup_epochs,
            "record_match_threshold": self.record_match_threshold,
            "detected_quality_threshold": self.detected_quality_threshold,
            "max_records": self.max_records,
            "risk": self.risk.to_dict(),
            "support": self.support.to_dict(),
            "failure_type_priors": dict(self.failure_type_priors),
            "arch": self.arch,
        }


@dataclass(slots=True)
class TFMSupportPrototype:
    epoch: int
    feature: torch.Tensor
    quality: float
    feature_level: str | int | None = None

    def __post_init__(self) -> None:
        self.epoch = int(self.epoch)
        self.feature = _as_feature_vector(self.feature)
        self.quality = float(self.quality)
        if self.feature_level is not None and not isinstance(self.feature_level, (str, int)):
            self.feature_level = str(self.feature_level)

    def to_state(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "feature": self.feature.tolist(),
            "quality": self.quality,
            "feature_level": self.feature_level,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TFMSupportPrototype":
        return cls(
            epoch=int(state.get("epoch", 0)),
            feature=torch.tensor(state["feature"], dtype=torch.float32),
            quality=float(state.get("quality", 0.0)),
            feature_level=state.get("feature_level"),
        )


@dataclass(slots=True)
class TFMRecord:
    gt_uid: str
    image_id: str
    class_id: int
    bbox: torch.Tensor
    first_seen_epoch: int
    last_seen_epoch: int
    last_state: TFMState
    miss_streak: int
    max_miss_streak: int
    total_miss: int
    relapse_count: int
    last_detected_epoch: int | None
    last_failure_epoch: int | None
    last_failure_type: TFMState | None
    risk: float
    support: TFMSupportPrototype | None = None

    def __post_init__(self) -> None:
        self.gt_uid = str(self.gt_uid)
        self.image_id = str(self.image_id)
        self.class_id = int(self.class_id)
        self.bbox = _as_region_tensor(self.bbox)
        self.first_seen_epoch = int(self.first_seen_epoch)
        self.last_seen_epoch = int(self.last_seen_epoch)
        self.last_state = _coerce_state(self.last_state)
        self.miss_streak = int(self.miss_streak)
        self.max_miss_streak = int(self.max_miss_streak)
        self.total_miss = int(self.total_miss)
        self.relapse_count = int(self.relapse_count)
        self.risk = float(self.risk)
        if self.last_failure_type is not None:
            self.last_failure_type = _coerce_state(self.last_failure_type)

    @property
    def is_current_failure(self) -> bool:
        return self.last_state not in _RECOVERED_STATES

    def to_state(self) -> dict[str, Any]:
        return {
            "gt_uid": self.gt_uid,
            "image_id": self.image_id,
            "class_id": self.class_id,
            "bbox": self.bbox.tolist(),
            "first_seen_epoch": self.first_seen_epoch,
            "last_seen_epoch": self.last_seen_epoch,
            "last_state": self.last_state,
            "miss_streak": self.miss_streak,
            "max_miss_streak": self.max_miss_streak,
            "total_miss": self.total_miss,
            "relapse_count": self.relapse_count,
            "last_detected_epoch": self.last_detected_epoch,
            "last_failure_epoch": self.last_failure_epoch,
            "last_failure_type": self.last_failure_type,
            "risk": self.risk,
            "support": None if self.support is None else self.support.to_state(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TFMRecord":
        raw_support = state.get("support")
        return cls(
            gt_uid=str(state["gt_uid"]),
            image_id=str(state["image_id"]),
            class_id=int(state["class_id"]),
            bbox=_as_region_tensor(state["bbox"]),
            first_seen_epoch=int(state.get("first_seen_epoch", 0)),
            last_seen_epoch=int(state.get("last_seen_epoch", 0)),
            last_state=_coerce_state(state.get("last_state", "unknown")),
            miss_streak=int(state.get("miss_streak", 0)),
            max_miss_streak=int(state.get("max_miss_streak", 0)),
            total_miss=int(state.get("total_miss", 0)),
            relapse_count=int(state.get("relapse_count", 0)),
            last_detected_epoch=state.get("last_detected_epoch"),
            last_failure_epoch=state.get("last_failure_epoch"),
            last_failure_type=(
                None
                if state.get("last_failure_type") is None
                else _coerce_state(state["last_failure_type"])
            ),
            risk=float(state.get("risk", 0.0)),
            support=(
                None
                if not isinstance(raw_support, Mapping)
                else TFMSupportPrototype.from_state(raw_support)
            ),
        )


class TemporalFailureMemory(nn.Module):
    def __init__(self, config: TFMConfig) -> None:
        super().__init__()
        self.config = config
        self._records: dict[str, TFMRecord] = {}
        self._image_index: dict[str, set[str]] = {}
        self._global_max_miss_streak = 0
        self.current_epoch = 0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def should_update(self, *, epoch: int | None = None) -> bool:
        if epoch is not None:
            self.current_epoch = int(epoch)
        if not self.config.enabled:
            return False
        return self.current_epoch > self.config.warmup_epochs

    def reset(self) -> None:
        self._records.clear()
        self._image_index.clear()
        self._global_max_miss_streak = 0
        self.current_epoch = 0

    @torch.no_grad()
    def update(
        self,
        *,
        image_ids: Sequence[Any],
        gt_boxes_list: Sequence[torch.Tensor | Sequence[Sequence[float]] | Sequence[float]],
        gt_labels_list: Sequence[torch.Tensor | Sequence[int] | int],
        image_shapes: Sequence[Sequence[int] | torch.Tensor],
        gt_states_list: Sequence[Sequence[Any] | Mapping[int, Any] | None] | None = None,
        quality_list: Sequence[Sequence[float] | torch.Tensor | Mapping[int, float] | None]
        | None = None,
        gt_ids_list: Sequence[torch.Tensor | Sequence[Any] | Any] | None = None,
        support_feature_list: Sequence[Mapping[int, torch.Tensor] | None] | None = None,
        support_quality_list: Sequence[Sequence[float] | torch.Tensor | Mapping[int, float] | None]
        | None = None,
        support_level_list: Sequence[Mapping[int, str | int] | None] | None = None,
        epoch: int | None = None,
    ) -> None:
        if not self.should_update(epoch=epoch):
            return

        batch_size = len(image_ids)
        expected_lengths = (
            len(gt_boxes_list),
            len(gt_labels_list),
            len(image_shapes),
        )
        if any(length != batch_size for length in expected_lengths):
            raise ValueError(
                "TFM update inputs must share the same batch dimension: "
                f"image_ids={batch_size}, gt_boxes={expected_lengths[0]}, "
                f"gt_labels={expected_lengths[1]}, image_shapes={expected_lengths[2]}."
            )

        gt_states_list = _default_batch_items(gt_states_list, batch_size)
        quality_list = _default_batch_items(quality_list, batch_size)
        gt_ids_list = _default_batch_items(gt_ids_list, batch_size)
        support_feature_list = _default_batch_items(support_feature_list, batch_size)
        support_quality_list = _default_batch_items(support_quality_list, batch_size)
        support_level_list = _default_batch_items(support_level_list, batch_size)

        for (
            image_id,
            gt_boxes,
            gt_labels,
            image_shape,
            raw_states,
            raw_quality,
            raw_gt_ids,
            raw_support_features,
            raw_support_quality,
            raw_support_levels,
        ) in zip(
            image_ids,
            gt_boxes_list,
            gt_labels_list,
            image_shapes,
            gt_states_list,
            quality_list,
            gt_ids_list,
            support_feature_list,
            support_quality_list,
            support_level_list,
            strict=True,
        ):
            image_key = _normalize_image_id(image_id)
            gt_boxes_tensor = _as_box_tensor(gt_boxes)
            gt_labels_tensor = _as_label_tensor(gt_labels, device=gt_boxes_tensor.device)
            if gt_boxes_tensor.shape[0] != gt_labels_tensor.shape[0]:
                raise ValueError(
                    "TFM gt_boxes and gt_labels must contain the same number of instances. "
                    f"Got {gt_boxes_tensor.shape[0]} and {gt_labels_tensor.shape[0]} for {image_key!r}."
                )
            if gt_boxes_tensor.numel() == 0:
                self._image_index.setdefault(image_key, set())
                continue

            normalized_gt_boxes = normalize_xyxy_boxes(gt_boxes_tensor, image_shape).cpu()
            count = int(normalized_gt_boxes.shape[0])
            states = _coerce_state_map(raw_states, count=count)
            qualities = _coerce_float_map(raw_quality, count=count)
            support_qualities = _coerce_float_map(raw_support_quality, count=count)
            gt_ids = _coerce_gt_ids(raw_gt_ids, count=count)
            support_features = _coerce_feature_map(raw_support_features)
            support_levels = _coerce_level_map(raw_support_levels)
            seen_uids: set[str] = set()
            matched_existing_uids: set[str] = set()

            for gt_index, (gt_box_norm, gt_label) in enumerate(
                zip(normalized_gt_boxes, gt_labels_tensor, strict=True)
            ):
                quality = qualities.get(gt_index)
                state = states.get(gt_index)
                if state is None:
                    state = _derive_state_from_quality(
                        quality,
                        detected_quality_threshold=self.config.detected_quality_threshold,
                    )
                class_id = int(gt_label.item())
                gt_uid = _resolve_gt_uid(
                    image_id=image_key,
                    class_id=class_id,
                    bbox=gt_box_norm,
                    gt_id=gt_ids[gt_index],
                    records=self._records,
                    image_index=self._image_index,
                    used_uids=matched_existing_uids,
                    record_match_threshold=self.config.record_match_threshold,
                )
                matched_existing_uids.add(gt_uid)
                seen_uids.add(gt_uid)
                previous = self._records.get(gt_uid)
                support = None if previous is None else previous.support
                if state in _RECOVERED_STATES and self.config.support.enabled:
                    support_quality = support_qualities.get(gt_index, quality)
                    support = _maybe_update_support(
                        previous=support,
                        feature=support_features.get(gt_index),
                        quality=support_quality,
                        feature_level=support_levels.get(gt_index),
                        epoch=self.current_epoch,
                        config=self.config.support,
                    )
                record = self._build_record(
                    gt_uid=gt_uid,
                    image_id=image_key,
                    class_id=class_id,
                    bbox=gt_box_norm,
                    state=state,
                    previous=previous,
                    support=support,
                )
                self._records[gt_uid] = record

            self._image_index[image_key] = seen_uids

        self._prune_records()
        self._refresh_global_stats()

    def get_record(self, gt_uid: Any) -> TFMRecord | None:
        return self._records.get(str(gt_uid))

    def get_image_records(self, image_id: Any) -> list[TFMRecord]:
        image_key = _normalize_image_id(image_id)
        records = [self._records[uid] for uid in self._image_index.get(image_key, set()) if uid in self._records]
        return sorted(records, key=lambda item: (-item.risk, -item.miss_streak, item.gt_uid))

    def get_high_risk_records(self, *, min_risk: float = 0.5) -> list[TFMRecord]:
        return sorted(
            [record for record in self._records.values() if record.risk >= float(min_risk)],
            key=lambda item: (-item.risk, -item.miss_streak, item.gt_uid),
        )

    def items(self) -> Iterator[tuple[str, TFMRecord]]:
        for gt_uid, record in self._records.items():
            yield gt_uid, record

    def values(self) -> Iterator[TFMRecord]:
        return iter(self._records.values())

    def summary(self) -> dict[str, Any]:
        warmup_active = self.config.enabled and self.current_epoch <= self.config.warmup_epochs
        state_counts = {state: 0 for state in _TFM_STATES}
        risk_sum = 0.0
        num_current_failures = 0
        num_support = 0
        for record in self._records.values():
            state_counts[record.last_state] += 1
            risk_sum += record.risk
            num_current_failures += int(record.is_current_failure)
            num_support += int(record.support is not None)

        num_records = len(self._records)
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_active": warmup_active,
            "num_records": num_records,
            "num_images": len(self._image_index),
            "num_current_failures": num_current_failures,
            "num_support": num_support,
            "mean_risk": risk_sum / float(max(num_records, 1)),
            "global_max_miss_streak": self._global_max_miss_streak,
            "state_counts": state_counts,
        }

    def __len__(self) -> int:
        return len(self._records)

    def extra_repr(self) -> str:
        return (
            f"enabled={self.config.enabled}, arch={self.config.arch!r}, "
            f"records={len(self._records)}"
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "records": {uid: record.to_state() for uid, record in self._records.items()},
            "image_index": {
                image_id: sorted(gt_uids)
                for image_id, gt_uids in self._image_index.items()
            },
            "global_max_miss_streak": self._global_max_miss_streak,
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            self.reset()
            return

        config_state = state.get("config", {})
        if isinstance(config_state, Mapping):
            try:
                self.config = TFMConfig.from_mapping(config_state, arch=config_state.get("arch"))
            except Exception:
                pass

        self.current_epoch = int(state.get("current_epoch", 0))
        raw_records = state.get("records", {})
        restored_records: dict[str, TFMRecord] = {}
        if isinstance(raw_records, Mapping):
            for uid, raw_record in raw_records.items():
                if not isinstance(raw_record, Mapping):
                    continue
                record = TFMRecord.from_state(raw_record)
                restored_records[str(uid)] = record
        self._records = restored_records

        raw_image_index = state.get("image_index", {})
        restored_index: dict[str, set[str]] = {}
        if isinstance(raw_image_index, Mapping):
            for image_id, raw_uids in raw_image_index.items():
                if not isinstance(raw_uids, Sequence) or isinstance(raw_uids, (str, bytes)):
                    continue
                image_key = _normalize_image_id(image_id)
                restored_index[image_key] = {str(uid) for uid in raw_uids if str(uid) in self._records}
        if not restored_index:
            for uid, record in self._records.items():
                restored_index.setdefault(record.image_id, set()).add(uid)
        self._image_index = restored_index
        self._global_max_miss_streak = int(state.get("global_max_miss_streak", 0))

    def _build_record(
        self,
        *,
        gt_uid: str,
        image_id: str,
        class_id: int,
        bbox: torch.Tensor,
        state: TFMState,
        previous: TFMRecord | None,
        support: TFMSupportPrototype | None,
    ) -> TFMRecord:
        epoch = self.current_epoch
        recovered = state in _RECOVERED_STATES
        relapse_event = bool(
            previous is not None
            and previous.last_state in _RECOVERED_STATES
            and state not in _RECOVERED_STATES
        )
        if recovered:
            miss_streak = 0
            max_miss_streak = 0 if previous is None else previous.max_miss_streak
            total_miss = 0 if previous is None else previous.total_miss
            relapse_count = 0 if previous is None else previous.relapse_count
            last_detected_epoch = epoch
            last_failure_epoch = None if previous is None else previous.last_failure_epoch
            last_failure_type = None if previous is None else previous.last_failure_type
        else:
            previous_streak = 0 if previous is None or previous.last_state in _RECOVERED_STATES else previous.miss_streak
            miss_streak = previous_streak + 1
            max_miss_streak = max(miss_streak, 0 if previous is None else previous.max_miss_streak)
            total_miss = miss_streak if previous is None else previous.total_miss + 1
            relapse_count = (0 if previous is None else previous.relapse_count) + int(relapse_event)
            last_detected_epoch = None if previous is None else previous.last_detected_epoch
            last_failure_epoch = epoch
            last_failure_type = state

        risk = _compute_risk(
            miss_streak=miss_streak,
            max_miss_streak=max_miss_streak,
            total_miss=total_miss,
            relapse_count=relapse_count,
            last_failure_type=last_failure_type,
            last_detected_epoch=last_detected_epoch,
            current_epoch=epoch,
            config=self.config,
            global_max_miss_streak=self._global_max_miss_streak,
        )
        return TFMRecord(
            gt_uid=gt_uid,
            image_id=image_id,
            class_id=class_id,
            bbox=bbox,
            first_seen_epoch=epoch if previous is None else previous.first_seen_epoch,
            last_seen_epoch=epoch,
            last_state=state,
            miss_streak=miss_streak,
            max_miss_streak=max_miss_streak,
            total_miss=total_miss,
            relapse_count=relapse_count,
            last_detected_epoch=last_detected_epoch,
            last_failure_epoch=last_failure_epoch,
            last_failure_type=last_failure_type,
            risk=risk,
            support=support,
        )

    def _prune_records(self) -> None:
        if self.config.max_records is None or len(self._records) <= self.config.max_records:
            return
        keep = {
            record.gt_uid
            for record in sorted(
                self._records.values(),
                key=lambda item: (-item.risk, -item.miss_streak, -item.last_seen_epoch, item.gt_uid),
            )[: self.config.max_records]
        }
        self._records = {uid: record for uid, record in self._records.items() if uid in keep}
        self._image_index = {
            image_id: {uid for uid in gt_uids if uid in keep}
            for image_id, gt_uids in self._image_index.items()
        }

    def _refresh_global_stats(self) -> None:
        self._global_max_miss_streak = max(
            (record.max_miss_streak for record in self._records.values()),
            default=0,
        )


TFM = TemporalFailureMemory


def load_tfm_config(path: str | Path, *, arch: str | None = None) -> TFMConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"TFM YAML must contain a mapping at the top level: {config_path}")
    return TFMConfig.from_mapping(raw, arch=arch)


def build_tfm_from_config(
    raw_config: Mapping[str, Any] | TFMConfig,
    *,
    arch: str | None = None,
) -> TemporalFailureMemory | None:
    config = raw_config if isinstance(raw_config, TFMConfig) else TFMConfig.from_mapping(raw_config, arch=arch)
    if not config.enabled:
        return None
    return TemporalFailureMemory(config)


def build_tfm_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> TemporalFailureMemory | None:
    config = load_tfm_config(path, arch=arch)
    if not config.enabled:
        return None
    return TemporalFailureMemory(config)


def _compute_risk(
    *,
    miss_streak: int,
    max_miss_streak: int,
    total_miss: int,
    relapse_count: int,
    last_failure_type: TFMState | None,
    last_detected_epoch: int | None,
    current_epoch: int,
    config: TFMConfig,
    global_max_miss_streak: int,
) -> float:
    risk_config = config.risk
    streak_scale = max(global_max_miss_streak, max_miss_streak, miss_streak, 1)
    normalized_streak = miss_streak / float(streak_scale)
    failure_prior = 0.0
    if last_failure_type is not None:
        failure_prior = float(config.failure_type_priors.get(last_failure_type, 0.0))

    recent_recovery = 0.0
    if last_detected_epoch is not None and risk_config.recovery_decay_epochs > 0:
        distance = max(int(current_epoch) - int(last_detected_epoch), 0)
        recent_recovery = math.exp(-distance / float(max(risk_config.recovery_decay_epochs, 1)))

    raw = (
        risk_config.miss_streak_weight * normalized_streak
        + risk_config.total_miss_weight * math.log1p(max(total_miss, 0))
        + risk_config.relapse_weight * float(max(relapse_count, 0))
        + risk_config.failure_type_weight * failure_prior
        - risk_config.recent_recovery_weight * recent_recovery
    )
    risk = 1.0 / (1.0 + math.exp(-raw))
    return float(max(risk_config.min_risk, min(risk_config.max_risk, risk)))


def _maybe_update_support(
    *,
    previous: TFMSupportPrototype | None,
    feature: torch.Tensor | None,
    quality: float | None,
    feature_level: str | int | None,
    epoch: int,
    config: TFMSupportConfig,
) -> TFMSupportPrototype | None:
    if not config.enabled or feature is None:
        return previous
    candidate_feature = _as_feature_vector(feature)
    if candidate_feature.numel() > config.max_feature_elements:
        return previous
    candidate_quality = 0.0 if quality is None else float(quality)
    if candidate_quality < config.min_quality:
        return previous
    if previous is None:
        return TFMSupportPrototype(
            epoch=epoch,
            feature=candidate_feature,
            quality=candidate_quality,
            feature_level=feature_level,
        )
    age = int(epoch) - int(previous.epoch)
    should_replace = candidate_quality >= previous.quality + config.quality_margin
    should_refresh = config.refresh_age > 0 and age >= config.refresh_age
    if not should_replace and not should_refresh:
        return previous
    feature_to_store = candidate_feature
    if previous.feature.shape == candidate_feature.shape and config.ema_momentum > 0.0:
        feature_to_store = (
            config.ema_momentum * previous.feature
            + (1.0 - config.ema_momentum) * candidate_feature
        )
    return TFMSupportPrototype(
        epoch=epoch,
        feature=feature_to_store,
        quality=max(previous.quality, candidate_quality),
        feature_level=feature_level,
    )


def _derive_state_from_quality(
    quality: float | None,
    *,
    detected_quality_threshold: float,
) -> TFMState:
    if quality is None:
        return "unknown"
    if float(quality) >= float(detected_quality_threshold):
        return "detected_like"
    return "score_weak"


def _default_batch_items(value: Sequence[Any] | None, batch_size: int) -> Sequence[Any]:
    if value is None:
        return [None] * batch_size
    if len(value) != batch_size:
        raise ValueError(
            f"TFM optional update input must share batch dimension {batch_size}; got {len(value)}."
        )
    return value


def _coerce_state_map(
    value: Sequence[Any] | Mapping[int, Any] | None,
    *,
    count: int,
) -> dict[int, TFMState]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            int(index): _coerce_state(state)
            for index, state in value.items()
            if 0 <= int(index) < count
        }
    if isinstance(value, (str, bytes)):
        raise TypeError("TFM gt state sequence must not be a string.")
    if len(value) != count:
        raise ValueError(f"TFM gt_states must contain {count} states; got {len(value)}.")
    return {index: _coerce_state(state) for index, state in enumerate(value)}


def _coerce_float_map(
    value: Sequence[float] | torch.Tensor | Mapping[int, float] | None,
    *,
    count: int,
) -> dict[int, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            int(index): float(score)
            for index, score in value.items()
            if 0 <= int(index) < count and score is not None
        }
    if isinstance(value, torch.Tensor):
        flattened = value.detach().cpu().flatten().tolist()
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError("TFM float sequence must not be a string.")
        flattened = list(value)
    if len(flattened) != count:
        raise ValueError(f"TFM score sequence must contain {count} values; got {len(flattened)}.")
    return {
        index: float(score)
        for index, score in enumerate(flattened)
        if score is not None
    }


def _coerce_gt_ids(
    value: torch.Tensor | Sequence[Any] | Any,
    *,
    count: int,
) -> list[Any | None]:
    if value is None:
        return [None] * count
    if isinstance(value, torch.Tensor):
        flattened = value.detach().cpu().flatten().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        flattened = list(value)
    else:
        if count != 1:
            raise ValueError(f"TFM scalar gt_id can only be used for one GT, got {count}.")
        flattened = [value]
    if len(flattened) != count:
        raise ValueError(f"TFM gt_ids must align with GT count. Got {len(flattened)} ids for {count} GTs.")
    return flattened


def _coerce_feature_map(value: Mapping[int, torch.Tensor] | None) -> dict[int, torch.Tensor]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("TFM support_feature_list items must be mappings from gt index to tensor.")
    return {int(index): feature for index, feature in value.items() if feature is not None}


def _coerce_level_map(value: Mapping[int, str | int] | None) -> dict[int, str | int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("TFM support_level_list items must be mappings from gt index to level.")
    return {int(index): level for index, level in value.items()}


def _coerce_state(value: Any) -> TFMState:
    candidate = str(value)
    if candidate not in _TFM_STATES:
        raise ValueError(f"Unsupported TFM state {candidate!r}. Supported: {sorted(_TFM_STATES)}.")
    return candidate  # type: ignore[return-value]


def _resolve_gt_uid(
    *,
    image_id: str,
    class_id: int,
    bbox: torch.Tensor,
    gt_id: Any | None,
    records: Mapping[str, TFMRecord],
    image_index: Mapping[str, set[str]],
    used_uids: set[str],
    record_match_threshold: float,
) -> str:
    if gt_id is not None:
        if isinstance(gt_id, torch.Tensor):
            if gt_id.numel() != 1:
                raise ValueError("TFM gt_id tensor must contain a single scalar value.")
            gt_id = gt_id.item()
        return f"ann:{gt_id}"

    matched_uid = _match_existing_gt_uid(
        image_id=image_id,
        class_id=class_id,
        bbox=bbox,
        records=records,
        image_index=image_index,
        used_uids=used_uids,
        record_match_threshold=record_match_threshold,
    )
    if matched_uid is not None:
        return matched_uid

    coords = ",".join(f"{float(value):.6f}" for value in bbox.tolist())
    digest = hashlib.sha1(f"{image_id}|{class_id}|{coords}".encode("utf-8")).hexdigest()[:16]
    return f"{image_id}:{class_id}:{digest}"


def _match_existing_gt_uid(
    *,
    image_id: str,
    class_id: int,
    bbox: torch.Tensor,
    records: Mapping[str, TFMRecord],
    image_index: Mapping[str, set[str]],
    used_uids: set[str],
    record_match_threshold: float,
) -> str | None:
    best_uid = None
    best_iou = float(record_match_threshold)
    candidate_uids = image_index.get(image_id, set())
    for uid in candidate_uids:
        if uid in used_uids:
            continue
        record = records.get(uid)
        if record is None or int(record.class_id) != int(class_id):
            continue
        iou = _single_box_iou(bbox, record.bbox)
        if iou > best_iou:
            best_iou = iou
            best_uid = uid
    return best_uid


def _single_box_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left_box = _as_region_tensor(left)
    right_box = _as_region_tensor(right)
    x1 = max(float(left_box[0]), float(right_box[0]))
    y1 = max(float(left_box[1]), float(right_box[1]))
    x2 = min(float(left_box[2]), float(right_box[2]))
    y2 = min(float(left_box[3]), float(right_box[3]))
    inter_w = max(x2 - x1, 0.0)
    inter_h = max(y2 - y1, 0.0)
    inter = inter_w * inter_h
    left_area = max(float(left_box[2] - left_box[0]), 0.0) * max(float(left_box[3] - left_box[1]), 0.0)
    right_area = max(float(right_box[2] - right_box[0]), 0.0) * max(float(right_box[3] - right_box[1]), 0.0)
    union = left_area + right_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _as_box_tensor(
    value: torch.Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).detach()
    if tensor.numel() == 0:
        return tensor.reshape(-1, 4)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != 4:
        raise ValueError("TFM boxes must have shape [N, 4] or [4].")
    return tensor


def _as_region_tensor(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().flatten()
    if tensor.numel() != 4:
        raise ValueError("TFM bbox must contain exactly four values.")
    return tensor.cpu()


def _as_label_tensor(
    value: torch.Tensor | Sequence[int] | int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.int64, device=device).detach().flatten()


def _as_feature_vector(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach().flatten().cpu()


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("TFM image_id tensor must contain a single scalar value.")
        value = value.item()
    return str(value)
