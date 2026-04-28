from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import yaml
from torchvision.ops import boxes as box_ops

from .common import normalize_arch


DHMState = Literal["TP", "FN_BG", "FN_CLS", "FN_LOC", "FN_MISS"]

_TP_STATE = "TP"
_FN_STATES = ("FN_BG", "FN_CLS", "FN_LOC", "FN_MISS")
_ALL_STATES = (_TP_STATE, *_FN_STATES)
_ASSIGNMENT_MEAN_FIELDS = (
    "pos_count",
    "center_dist",
    "centerness_target",
    "cls_loss",
    "box_loss",
    "ctr_loss",
    "near_candidate_count",
    "near_negative_count",
    "near_negative_ratio",
    "ambiguous_assigned_elsewhere",
    "ambiguous_ratio",
)
_ASSIGNMENT_VALUE_ATTRS = {
    "pos_count": "last_pos_count",
    "center_dist": "ema_center_dist",
    "centerness_target": "ema_centerness_target",
    "cls_loss": "ema_cls_loss",
    "box_loss": "ema_box_loss",
    "ctr_loss": "ema_ctr_loss",
    "near_candidate_count": "last_near_candidate_count",
    "near_negative_count": "last_near_negative_count",
    "near_negative_ratio": "ema_near_negative_ratio",
    "ambiguous_assigned_elsewhere": "last_ambiguous_assigned_elsewhere",
    "ambiguous_ratio": "ema_ambiguous_ratio",
}


@dataclass(frozen=True, slots=True)
class DHMMatchingConfig:
    tau_iou: float = 0.5
    tau_tp: float = 0.3
    tau_near: float = 0.3
    tau_bg_score: float = 0.1
    tau_cls_evidence: float = 0.3
    tau_loc_score: float = 0.3

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "DHMMatchingConfig":
        data = dict(raw or {})
        config = cls(
            tau_iou=float(data.get("tau_iou", 0.5)),
            tau_tp=float(data.get("tau_tp", 0.3)),
            tau_near=float(data.get("tau_near", 0.3)),
            tau_bg_score=float(data.get("tau_bg_score", 0.1)),
            tau_cls_evidence=float(data.get("tau_cls_evidence", 0.3)),
            tau_loc_score=float(data.get("tau_loc_score", 0.3)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for field_name in (
            "tau_iou",
            "tau_tp",
            "tau_near",
            "tau_bg_score",
            "tau_cls_evidence",
            "tau_loc_score",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"DHM matching.{field_name} must satisfy 0 <= value <= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_iou": self.tau_iou,
            "tau_tp": self.tau_tp,
            "tau_near": self.tau_near,
            "tau_bg_score": self.tau_bg_score,
            "tau_cls_evidence": self.tau_cls_evidence,
            "tau_loc_score": self.tau_loc_score,
        }


@dataclass(frozen=True, slots=True)
class DHMMiningConfig:
    enabled: bool = True
    mode: str = "epoch_end_full_train"
    mine_interval: int = 1
    warmup_epochs: int = 0
    matching: DHMMatchingConfig = field(default_factory=DHMMatchingConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "DHMMiningConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", True)),
            mode=str(data.get("mode", "epoch_end_full_train")),
            mine_interval=int(data.get("mine_interval", 1)),
            warmup_epochs=int(data.get("warmup_epochs", 0)),
            matching=DHMMatchingConfig.from_mapping(data.get("matching")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode != "epoch_end_full_train":
            raise ValueError("DHM mining.mode currently supports only 'epoch_end_full_train'.")
        if self.mine_interval < 1:
            raise ValueError("DHM mining.mine_interval must be >= 1.")
        if self.warmup_epochs < 0:
            raise ValueError("DHM mining.warmup_epochs must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "mine_interval": self.mine_interval,
            "warmup_epochs": self.warmup_epochs,
            "matching": self.matching.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DHMScoringConfig:
    min_observations: int = 3
    forgetting_weight: float = 0.35
    fn_rate_weight: float = 0.25
    fn_streak_weight: float = 0.2
    type_switch_weight: float = 0.15
    recent_fn_weight: float = 0.05
    stable_tp_discount: float = 0.2
    streak_norm_epochs: int = 5
    ema_momentum: float = 0.8
    min_instability: float = 0.0
    max_instability: float = 1.0
    stable_tp_rate: float = 0.8
    persistent_fn_rate: float = 0.8
    oscillator_switches: int = 2
    late_learner_min_epoch_gap: int = 3

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "DHMScoringConfig":
        data = dict(raw or {})
        config = cls(
            min_observations=int(data.get("min_observations", 3)),
            forgetting_weight=float(data.get("forgetting_weight", 0.35)),
            fn_rate_weight=float(data.get("fn_rate_weight", 0.25)),
            fn_streak_weight=float(data.get("fn_streak_weight", 0.2)),
            type_switch_weight=float(data.get("type_switch_weight", 0.15)),
            recent_fn_weight=float(data.get("recent_fn_weight", 0.05)),
            stable_tp_discount=float(data.get("stable_tp_discount", 0.2)),
            streak_norm_epochs=int(data.get("streak_norm_epochs", 5)),
            ema_momentum=float(data.get("ema_momentum", 0.8)),
            min_instability=float(data.get("min_instability", 0.0)),
            max_instability=float(data.get("max_instability", 1.0)),
            stable_tp_rate=float(data.get("stable_tp_rate", 0.8)),
            persistent_fn_rate=float(data.get("persistent_fn_rate", 0.8)),
            oscillator_switches=int(data.get("oscillator_switches", 2)),
            late_learner_min_epoch_gap=int(data.get("late_learner_min_epoch_gap", 3)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.min_observations < 1:
            raise ValueError("DHM scoring.min_observations must be >= 1.")
        for field_name in (
            "forgetting_weight",
            "fn_rate_weight",
            "fn_streak_weight",
            "type_switch_weight",
            "recent_fn_weight",
            "stable_tp_discount",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"DHM scoring.{field_name} must be >= 0.")
        if self.streak_norm_epochs < 1:
            raise ValueError("DHM scoring.streak_norm_epochs must be >= 1.")
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError("DHM scoring.ema_momentum must satisfy 0 <= value <= 1.")
        if not 0.0 <= self.min_instability <= 1.0:
            raise ValueError("DHM scoring.min_instability must satisfy 0 <= value <= 1.")
        if not 0.0 <= self.max_instability <= 1.0:
            raise ValueError("DHM scoring.max_instability must satisfy 0 <= value <= 1.")
        if self.min_instability > self.max_instability:
            raise ValueError("DHM scoring.min_instability must be <= max_instability.")
        for field_name in ("stable_tp_rate", "persistent_fn_rate"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"DHM scoring.{field_name} must satisfy 0 <= value <= 1.")
        if self.oscillator_switches < 1:
            raise ValueError("DHM scoring.oscillator_switches must be >= 1.")
        if self.late_learner_min_epoch_gap < 0:
            raise ValueError("DHM scoring.late_learner_min_epoch_gap must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_observations": self.min_observations,
            "forgetting_weight": self.forgetting_weight,
            "fn_rate_weight": self.fn_rate_weight,
            "fn_streak_weight": self.fn_streak_weight,
            "type_switch_weight": self.type_switch_weight,
            "recent_fn_weight": self.recent_fn_weight,
            "stable_tp_discount": self.stable_tp_discount,
            "streak_norm_epochs": self.streak_norm_epochs,
            "ema_momentum": self.ema_momentum,
            "min_instability": self.min_instability,
            "max_instability": self.max_instability,
            "stable_tp_rate": self.stable_tp_rate,
            "persistent_fn_rate": self.persistent_fn_rate,
            "oscillator_switches": self.oscillator_switches,
            "late_learner_min_epoch_gap": self.late_learner_min_epoch_gap,
        }


@dataclass(frozen=True, slots=True)
class DHMConfig:
    enabled: bool = False
    mining: DHMMiningConfig = field(default_factory=DHMMiningConfig)
    scoring: DHMScoringConfig = field(default_factory=DHMScoringConfig)
    record_match_threshold: float = 0.95
    max_records: int | None = None
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "DHMConfig":
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

        config = cls(
            enabled=bool(merged.get("enabled", False)),
            mining=DHMMiningConfig.from_mapping(merged.get("mining")),
            scoring=DHMScoringConfig.from_mapping(merged.get("scoring")),
            record_match_threshold=float(merged.get("record_match_threshold", 0.95)),
            max_records=None if merged.get("max_records") is None else int(merged.get("max_records")),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= self.record_match_threshold <= 1.0:
            raise ValueError("DHM record_match_threshold must satisfy 0 <= value <= 1.")
        if self.max_records is not None and self.max_records < 1:
            raise ValueError("DHM max_records must be null or >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mining": self.mining.to_dict(),
            "scoring": self.scoring.to_dict(),
            "record_match_threshold": self.record_match_threshold,
            "max_records": self.max_records,
            "arch": self.arch,
        }


@dataclass(slots=True)
class DHMRecord:
    gt_uid: str
    image_id: str
    ann_id: str | None
    class_id: int
    bbox: torch.Tensor
    first_seen_epoch: int | None = None
    last_seen_epoch: int | None = None
    first_tp_epoch: int | None = None
    last_tp_epoch: int | None = None
    last_fn_epoch: int | None = None
    last_state: str = "UNSEEN"
    last_score: float = 0.0
    last_iou: float = 0.0
    total_seen: int = 0
    tp_count: int = 0
    fn_count: int = 0
    consecutive_tp: int = 0
    consecutive_fn: int = 0
    max_fn_streak: int = 0
    forgetting_count: int = 0
    recovery_count: int = 0
    fn_type_switch_count: int = 0
    state_change_count: int = 0
    state_counts: dict[str, int] = field(default_factory=dict)
    transition_counts: dict[str, int] = field(default_factory=dict)
    last_transition: str | None = None
    ema_score: float = 0.0
    ema_iou: float = 0.0
    instability_score: float = 0.0
    assignment_seen: int = 0
    last_assignment_epoch: int | None = None
    last_pos_count: int = 0
    zero_pos_count: int = 0
    last_level_pos_counts: dict[str, int] = field(default_factory=dict)
    level_pos_counts: dict[str, int] = field(default_factory=dict)
    last_near_candidate_count: int = 0
    last_near_negative_count: int = 0
    last_ambiguous_assigned_elsewhere: int = 0
    ema_pos_count: float = 0.0
    ema_center_dist: float = 0.0
    ema_centerness_target: float = 0.0
    ema_cls_loss: float = 0.0
    ema_box_loss: float = 0.0
    ema_ctr_loss: float = 0.0
    ema_near_negative_count: float = 0.0
    ema_near_negative_ratio: float = 0.0
    ema_ambiguous_assigned_elsewhere: float = 0.0
    ema_ambiguous_ratio: float = 0.0

    @property
    def dominant_failure_type(self) -> str | None:
        best_state = None
        best_count = 0
        for state in _FN_STATES:
            count = int(self.state_counts.get(state, 0))
            if count > best_count:
                best_state = state
                best_count = count
        return best_state

    def update(
        self,
        *,
        state: str,
        score: float,
        iou: float,
        bbox: torch.Tensor,
        epoch: int,
        scoring: DHMScoringConfig,
    ) -> dict[str, Any]:
        if state not in _ALL_STATES:
            raise ValueError(f"Unsupported DHM state: {state!r}")

        prev_state = self.last_state
        prev_was_tp = prev_state == _TP_STATE
        prev_was_fn = prev_state in _FN_STATES
        current_is_tp = state == _TP_STATE
        current_is_fn = state in _FN_STATES

        relapse = bool(prev_was_tp and current_is_fn)
        recovery = bool(prev_was_fn and current_is_tp)
        type_switch = bool(prev_state in _FN_STATES and state in _FN_STATES and prev_state != state)
        has_previous_state = prev_state in _ALL_STATES
        state_change = bool(has_previous_state and prev_state != state)
        transition = f"{prev_state}->{state}" if has_previous_state else None

        if self.first_seen_epoch is None:
            self.first_seen_epoch = int(epoch)
        self.last_seen_epoch = int(epoch)
        if current_is_tp:
            if self.first_tp_epoch is None:
                self.first_tp_epoch = int(epoch)
            self.last_tp_epoch = int(epoch)
            self.tp_count += 1
            self.consecutive_tp += 1
            self.consecutive_fn = 0
        else:
            self.last_fn_epoch = int(epoch)
            self.fn_count += 1
            self.consecutive_fn += 1
            self.consecutive_tp = 0
            self.max_fn_streak = max(self.max_fn_streak, self.consecutive_fn)

        self.total_seen += 1
        self.state_counts[state] = int(self.state_counts.get(state, 0)) + 1
        if relapse:
            self.forgetting_count += 1
        if recovery:
            self.recovery_count += 1
        if type_switch:
            self.fn_type_switch_count += 1
        if state_change:
            self.state_change_count += 1
        if transition is not None:
            self.transition_counts[transition] = int(self.transition_counts.get(transition, 0)) + 1
            self.last_transition = transition

        momentum = float(scoring.ema_momentum)
        if self.total_seen <= 1:
            self.ema_score = float(score)
            self.ema_iou = float(iou)
        else:
            self.ema_score = momentum * float(self.ema_score) + (1.0 - momentum) * float(score)
            self.ema_iou = momentum * float(self.ema_iou) + (1.0 - momentum) * float(iou)

        self.last_state = state
        self.last_score = float(score)
        self.last_iou = float(iou)
        self.bbox = bbox.detach().cpu().reshape(4)
        self.instability_score = self._compute_instability(scoring)
        return {
            "relapse": relapse,
            "recovery": recovery,
            "type_switch": type_switch,
            "state_change": state_change,
            "transition": transition,
        }

    def update_assignment_statistics(
        self,
        observation: Mapping[str, Any],
        *,
        epoch: int,
        scoring: DHMScoringConfig,
    ) -> None:
        momentum = float(scoring.ema_momentum)
        first_observation = self.assignment_seen <= 0

        pos_count = int(observation.get("pos_count", 0))
        near_candidate_count = int(observation.get("near_candidate_count", 0))
        near_negative_count = int(observation.get("near_negative_count", 0))
        ambiguous_count = int(observation.get("ambiguous_assigned_elsewhere", 0))
        near_negative_ratio = (
            float(near_negative_count) / float(max(near_candidate_count, 1))
            if near_candidate_count > 0
            else 0.0
        )
        ambiguous_ratio = (
            float(ambiguous_count) / float(max(near_candidate_count, 1))
            if near_candidate_count > 0
            else 0.0
        )

        self.assignment_seen += 1
        self.last_assignment_epoch = int(epoch)
        self.last_pos_count = pos_count
        self.last_near_candidate_count = near_candidate_count
        self.last_near_negative_count = near_negative_count
        self.last_ambiguous_assigned_elsewhere = ambiguous_count
        if pos_count <= 0:
            self.zero_pos_count += 1

        raw_level_counts = observation.get("level_pos_counts", {})
        self.last_level_pos_counts = (
            {str(key): int(value) for key, value in dict(raw_level_counts).items()}
            if isinstance(raw_level_counts, Mapping)
            else {}
        )
        for level, count in self.last_level_pos_counts.items():
            self.level_pos_counts[level] = int(self.level_pos_counts.get(level, 0)) + int(count)

        values = {
            "pos_count": float(pos_count),
            "center_dist": float(observation.get("center_dist", 0.0)),
            "centerness_target": float(observation.get("centerness_target", 0.0)),
            "cls_loss": float(observation.get("cls_loss", 0.0)),
            "box_loss": float(observation.get("box_loss", 0.0)),
            "ctr_loss": float(observation.get("ctr_loss", 0.0)),
            "near_negative_count": float(near_negative_count),
            "near_negative_ratio": near_negative_ratio,
            "ambiguous_assigned_elsewhere": float(ambiguous_count),
            "ambiguous_ratio": ambiguous_ratio,
        }
        for field_name, value in values.items():
            attr_name = f"ema_{field_name}"
            if first_observation:
                setattr(self, attr_name, value)
            else:
                previous = float(getattr(self, attr_name))
                setattr(self, attr_name, momentum * previous + (1.0 - momentum) * value)

    def status(self, scoring: DHMScoringConfig) -> str:
        if self.total_seen < int(scoring.min_observations):
            return "warming"
        fn_rate = float(self.fn_count) / float(max(self.total_seen, 1))
        tp_rate = float(self.tp_count) / float(max(self.total_seen, 1))
        if self.fn_type_switch_count >= int(scoring.oscillator_switches):
            return "oscillator"
        if self.forgetting_count > 0:
            return "relapser"
        if fn_rate >= float(scoring.persistent_fn_rate):
            return "persistent_fn"
        if (
            self.first_seen_epoch is not None
            and self.first_tp_epoch is not None
            and int(self.first_tp_epoch) - int(self.first_seen_epoch) >= int(scoring.late_learner_min_epoch_gap)
            and self.consecutive_tp > 0
        ):
            return "late_learner"
        if tp_rate >= float(scoring.stable_tp_rate) and self.forgetting_count == 0 and self.consecutive_fn == 0:
            return "stable_tp"
        return "mixed"

    def to_state(self) -> dict[str, Any]:
        return {
            "gt_uid": self.gt_uid,
            "image_id": self.image_id,
            "ann_id": self.ann_id,
            "class_id": self.class_id,
            "bbox": self.bbox.detach().cpu(),
            "first_seen_epoch": self.first_seen_epoch,
            "last_seen_epoch": self.last_seen_epoch,
            "first_tp_epoch": self.first_tp_epoch,
            "last_tp_epoch": self.last_tp_epoch,
            "last_fn_epoch": self.last_fn_epoch,
            "last_state": self.last_state,
            "last_score": self.last_score,
            "last_iou": self.last_iou,
            "total_seen": self.total_seen,
            "tp_count": self.tp_count,
            "fn_count": self.fn_count,
            "consecutive_tp": self.consecutive_tp,
            "consecutive_fn": self.consecutive_fn,
            "max_fn_streak": self.max_fn_streak,
            "forgetting_count": self.forgetting_count,
            "recovery_count": self.recovery_count,
            "fn_type_switch_count": self.fn_type_switch_count,
            "state_change_count": self.state_change_count,
            "state_counts": dict(self.state_counts),
            "transition_counts": dict(self.transition_counts),
            "last_transition": self.last_transition,
            "ema_score": self.ema_score,
            "ema_iou": self.ema_iou,
            "instability_score": self.instability_score,
            "assignment_seen": self.assignment_seen,
            "last_assignment_epoch": self.last_assignment_epoch,
            "last_pos_count": self.last_pos_count,
            "zero_pos_count": self.zero_pos_count,
            "last_level_pos_counts": dict(self.last_level_pos_counts),
            "level_pos_counts": dict(self.level_pos_counts),
            "last_near_candidate_count": self.last_near_candidate_count,
            "last_near_negative_count": self.last_near_negative_count,
            "last_ambiguous_assigned_elsewhere": self.last_ambiguous_assigned_elsewhere,
            "ema_pos_count": self.ema_pos_count,
            "ema_center_dist": self.ema_center_dist,
            "ema_centerness_target": self.ema_centerness_target,
            "ema_cls_loss": self.ema_cls_loss,
            "ema_box_loss": self.ema_box_loss,
            "ema_ctr_loss": self.ema_ctr_loss,
            "ema_near_negative_count": self.ema_near_negative_count,
            "ema_near_negative_ratio": self.ema_near_negative_ratio,
            "ema_ambiguous_assigned_elsewhere": self.ema_ambiguous_assigned_elsewhere,
            "ema_ambiguous_ratio": self.ema_ambiguous_ratio,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DHMRecord":
        return cls(
            gt_uid=str(state["gt_uid"]),
            image_id=str(state.get("image_id", "")),
            ann_id=None if state.get("ann_id") is None else str(state.get("ann_id")),
            class_id=int(state["class_id"]),
            bbox=torch.as_tensor(state["bbox"], dtype=torch.float32).reshape(4),
            first_seen_epoch=state.get("first_seen_epoch"),
            last_seen_epoch=state.get("last_seen_epoch"),
            first_tp_epoch=state.get("first_tp_epoch"),
            last_tp_epoch=state.get("last_tp_epoch"),
            last_fn_epoch=state.get("last_fn_epoch"),
            last_state=str(state.get("last_state", "UNSEEN")),
            last_score=float(state.get("last_score", 0.0)),
            last_iou=float(state.get("last_iou", 0.0)),
            total_seen=int(state.get("total_seen", 0)),
            tp_count=int(state.get("tp_count", 0)),
            fn_count=int(state.get("fn_count", 0)),
            consecutive_tp=int(state.get("consecutive_tp", 0)),
            consecutive_fn=int(state.get("consecutive_fn", 0)),
            max_fn_streak=int(state.get("max_fn_streak", 0)),
            forgetting_count=int(state.get("forgetting_count", 0)),
            recovery_count=int(state.get("recovery_count", 0)),
            fn_type_switch_count=int(state.get("fn_type_switch_count", 0)),
            state_change_count=int(state.get("state_change_count", 0)),
            state_counts={str(k): int(v) for k, v in dict(state.get("state_counts", {})).items()},
            transition_counts={
                str(k): int(v)
                for k, v in dict(state.get("transition_counts", {})).items()
            },
            last_transition=None
            if state.get("last_transition") is None
            else str(state.get("last_transition")),
            ema_score=float(state.get("ema_score", 0.0)),
            ema_iou=float(state.get("ema_iou", 0.0)),
            instability_score=float(state.get("instability_score", 0.0)),
            assignment_seen=int(state.get("assignment_seen", 0)),
            last_assignment_epoch=state.get("last_assignment_epoch"),
            last_pos_count=int(state.get("last_pos_count", 0)),
            zero_pos_count=int(state.get("zero_pos_count", 0)),
            last_level_pos_counts={
                str(k): int(v)
                for k, v in dict(state.get("last_level_pos_counts", {})).items()
            },
            level_pos_counts={
                str(k): int(v)
                for k, v in dict(state.get("level_pos_counts", {})).items()
            },
            last_near_candidate_count=int(state.get("last_near_candidate_count", 0)),
            last_near_negative_count=int(state.get("last_near_negative_count", 0)),
            last_ambiguous_assigned_elsewhere=int(
                state.get("last_ambiguous_assigned_elsewhere", 0)
            ),
            ema_pos_count=float(state.get("ema_pos_count", 0.0)),
            ema_center_dist=float(state.get("ema_center_dist", 0.0)),
            ema_centerness_target=float(state.get("ema_centerness_target", 0.0)),
            ema_cls_loss=float(state.get("ema_cls_loss", 0.0)),
            ema_box_loss=float(state.get("ema_box_loss", 0.0)),
            ema_ctr_loss=float(state.get("ema_ctr_loss", 0.0)),
            ema_near_negative_count=float(state.get("ema_near_negative_count", 0.0)),
            ema_near_negative_ratio=float(state.get("ema_near_negative_ratio", 0.0)),
            ema_ambiguous_assigned_elsewhere=float(
                state.get("ema_ambiguous_assigned_elsewhere", 0.0)
            ),
            ema_ambiguous_ratio=float(state.get("ema_ambiguous_ratio", 0.0)),
        )

    def _compute_instability(self, scoring: DHMScoringConfig) -> float:
        total = float(max(self.total_seen, 1))
        forgetting_rate = float(self.forgetting_count) / float(max(self.total_seen - 1, 1))
        fn_rate = float(self.fn_count) / total
        fn_streak = min(1.0, float(self.max_fn_streak) / float(max(scoring.streak_norm_epochs, 1)))
        type_switch_rate = float(self.fn_type_switch_count) / float(max(self.fn_count - 1, 1))
        recent_fn = 1.0 if self.last_state in _FN_STATES else 0.0
        stable_tp = float(self.consecutive_tp) / total

        weighted = (
            float(scoring.forgetting_weight) * forgetting_rate
            + float(scoring.fn_rate_weight) * fn_rate
            + float(scoring.fn_streak_weight) * fn_streak
            + float(scoring.type_switch_weight) * type_switch_rate
            + float(scoring.recent_fn_weight) * recent_fn
        )
        total_weight = (
            float(scoring.forgetting_weight)
            + float(scoring.fn_rate_weight)
            + float(scoring.fn_streak_weight)
            + float(scoring.type_switch_weight)
            + float(scoring.recent_fn_weight)
        )
        score = weighted / max(total_weight, 1.0e-6)
        score -= float(scoring.stable_tp_discount) * stable_tp
        return float(max(float(scoring.min_instability), min(float(scoring.max_instability), score)))


class DetectionHysteresisMemory(nn.Module):
    def __init__(self, config: DHMConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._records: dict[str, DHMRecord] = {}
        self._image_index: defaultdict[str, set[str]] = defaultdict(set)
        self._stats: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._records)

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._stats.clear()

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def should_mine(self, *, epoch: int | None = None) -> bool:
        epoch_value = int(self.current_epoch if epoch is None else epoch)
        if not self.config.enabled or not self.config.mining.enabled:
            return False
        if epoch_value <= int(self.config.mining.warmup_epochs):
            return False
        return epoch_value % int(self.config.mining.mine_interval) == 0

    @torch.no_grad()
    def mine_batch(
        self,
        *,
        detections: Sequence[Mapping[str, torch.Tensor]],
        original_targets: Sequence[Mapping[str, torch.Tensor]],
        epoch: int,
    ) -> dict[str, int]:
        if not self.should_mine(epoch=epoch):
            return {}
        self.current_epoch = int(epoch)
        stats = Counter()
        for image_index, (detection, target) in enumerate(zip(detections, original_targets, strict=True)):
            stats.update(self._mine_image(detection=detection, target=target, epoch=epoch, image_index=image_index))
        self._stats.update(stats)
        self._prune_records()
        return {key: int(value) for key, value in stats.items()}

    def get_record(self, gt_uid: str) -> DHMRecord | None:
        return self._records.get(str(gt_uid))

    def get_image_records(self, image_id: torch.Tensor | int | str) -> list[DHMRecord]:
        image_key = _normalize_image_id(image_id)
        return [
            self._records[gt_uid]
            for gt_uid in sorted(self._image_index.get(image_key, ()))
            if gt_uid in self._records
        ]

    def record_assignment_observations(
        self,
        *,
        records: Sequence[DHMRecord | None],
        observations: Sequence[Mapping[str, Any]],
        epoch: int | None = None,
    ) -> dict[str, int]:
        if not self.config.enabled:
            return {}
        epoch_value = int(self.current_epoch if epoch is None else epoch)
        stats = Counter()
        for record, observation in zip(records, observations, strict=True):
            if record is None:
                stats["assignment_missing_record"] += 1
                continue
            record.update_assignment_statistics(
                observation,
                epoch=epoch_value,
                scoring=self.config.scoring,
            )
            stats["assignment_observed"] += 1
            if int(observation.get("pos_count", 0)) <= 0:
                stats["assignment_zero_pos"] += 1
            if int(observation.get("near_negative_count", 0)) > 0:
                stats["assignment_near_negative_gt"] += 1
            if int(observation.get("ambiguous_assigned_elsewhere", 0)) > 0:
                stats["assignment_ambiguous_gt"] += 1
        self._stats.update(stats)
        return {key: int(value) for key, value in stats.items()}

    def summary(self) -> dict[str, Any]:
        status_counts = Counter(record.status(self.config.scoring) for record in self._records.values())
        last_state_counts = Counter(record.last_state for record in self._records.values())
        dominant_failure_counts = Counter(
            record.dominant_failure_type
            for record in self._records.values()
            if record.dominant_failure_type is not None
        )
        instability_sum = sum(float(record.instability_score) for record in self._records.values())
        current_failures = sum(1 for record in self._records.values() if record.last_state in _FN_STATES)
        transition_counts = Counter()
        for record in self._records.values():
            for transition, count in record.transition_counts.items():
                transition_counts[str(transition)] += int(count)
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "num_records": len(self._records),
            "num_images": len(self._image_index),
            "num_current_failures": current_failures,
            "mean_instability": instability_sum / float(max(len(self._records), 1)),
            "global_max_fn_streak": max((record.max_fn_streak for record in self._records.values()), default=0),
            "total_forgetting": sum(record.forgetting_count for record in self._records.values()),
            "total_recovery": sum(record.recovery_count for record in self._records.values()),
            "total_type_switch": sum(record.fn_type_switch_count for record in self._records.values()),
            "last_state_counts": dict(last_state_counts),
            "status_counts": dict(status_counts),
            "dominant_failure_counts": dict(dominant_failure_counts),
            "transition_matrix": _transition_matrix_from_counts(transition_counts),
            "assignment_by_state": _assignment_groups(
                self._records.values(),
                group_by="state",
                epoch=self.current_epoch,
            ),
            "assignment_by_transition": _assignment_groups(
                self._records.values(),
                group_by="transition",
                epoch=self.current_epoch,
            ),
            "mining": {key: int(value) for key, value in self._stats.items()},
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "records": {gt_uid: record.to_state() for gt_uid, record in self._records.items()},
            "stats": dict(self._stats),
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            self._records.clear()
            self._image_index.clear()
            self._stats.clear()
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))
        records: dict[str, DHMRecord] = {}
        raw_records = state.get("records", {})
        if isinstance(raw_records, Mapping):
            for gt_uid, raw_record in raw_records.items():
                if isinstance(raw_record, Mapping):
                    records[str(gt_uid)] = DHMRecord.from_state(raw_record)
        self._records = records
        self._rebuild_image_index()
        raw_stats = state.get("stats", {})
        self._stats = Counter({str(k): int(v) for k, v in raw_stats.items()}) if isinstance(raw_stats, Mapping) else Counter()

    def _mine_image(
        self,
        *,
        detection: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
        epoch: int,
        image_index: int,
    ) -> Counter[str]:
        gt_boxes = target["boxes"].detach()
        gt_labels = target["labels"].to(dtype=torch.int64)
        gt_ids = _extract_gt_ids(target, int(gt_boxes.shape[0]))
        image_id = _normalize_image_id(target.get("image_id", torch.tensor(image_index)))
        pred_boxes = detection.get("boxes", gt_boxes.new_zeros((0, 4))).detach().to(dtype=torch.float32)
        pred_labels = detection.get("labels", gt_labels.new_zeros((0,))).detach().to(dtype=torch.int64)
        pred_scores = detection.get("scores", pred_boxes.new_zeros((0,))).detach().to(dtype=torch.float32)

        stats: Counter[str] = Counter()
        for gt_index in range(int(gt_boxes.shape[0])):
            class_id = int(gt_labels[gt_index].item())
            ann_id = gt_ids[gt_index]
            gt_uid = _gt_uid(
                image_id=image_id,
                class_id=class_id,
                bbox=gt_boxes[gt_index],
                image_shape=None,
                gt_id=ann_id,
            )
            detection_state = _assign_detection_state(
                gt_box=gt_boxes[gt_index],
                gt_label=class_id,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                matching=self.config.mining.matching,
            )
            state = str(detection_state["state"])
            record = self._records.get(gt_uid)
            if record is None:
                record = DHMRecord(
                    gt_uid=gt_uid,
                    image_id=image_id,
                    ann_id=None if ann_id is None else str(ann_id),
                    class_id=class_id,
                    bbox=gt_boxes[gt_index].detach().cpu().reshape(4),
                )
                self._records[gt_uid] = record
                self._image_index[image_id].add(gt_uid)
            flags = record.update(
                state=state,
                score=float(detection_state["score"]),
                iou=float(detection_state["iou"]),
                bbox=gt_boxes[gt_index].detach().cpu().reshape(4),
                epoch=epoch,
                scoring=self.config.scoring,
            )
            stats["gt_seen"] += 1
            if state == _TP_STATE:
                stats["num_tp"] += 1
            else:
                stats["num_fn"] += 1
                stats[f"num_{state.lower()}"] += 1
            if flags["relapse"]:
                stats["relapses"] += 1
            if flags["recovery"]:
                stats["recoveries"] += 1
            if flags["type_switch"]:
                stats["type_switches"] += 1
            if flags["state_change"]:
                stats["state_changes"] += 1
            transition = flags.get("transition")
            if transition is not None:
                safe_transition = str(transition).replace("->", "_to_").lower()
                stats[f"transition_{safe_transition}"] += 1
        return stats

    def _rebuild_image_index(self) -> None:
        image_index: defaultdict[str, set[str]] = defaultdict(set)
        for gt_uid, record in self._records.items():
            if record.image_id:
                image_index[str(record.image_id)].add(gt_uid)
        self._image_index = image_index

    def _prune_records(self) -> None:
        max_records = self.config.max_records
        if max_records is None or len(self._records) <= int(max_records):
            return
        sorted_records = sorted(
            self._records.items(),
            key=lambda item: (
                float(item[1].last_seen_epoch or 0),
                float(item[1].instability_score),
                float(item[1].forgetting_count),
            ),
            reverse=True,
        )
        self._records = dict(sorted_records[: int(max_records)])
        self._rebuild_image_index()


def _transition_matrix_from_counts(counts: Mapping[str, int]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for transition, count in counts.items():
        if "->" not in str(transition):
            continue
        source, target = str(transition).split("->", 1)
        if source not in _ALL_STATES or target not in _ALL_STATES:
            continue
        source_counts = matrix.setdefault(source, {})
        source_counts[target] = int(source_counts.get(target, 0)) + int(count)
    return matrix


def _assignment_groups(
    records: Iterable[DHMRecord],
    *,
    group_by: str,
    epoch: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        if int(record.assignment_seen) <= 0:
            continue
        if record.last_assignment_epoch is None or int(record.last_assignment_epoch) != int(epoch):
            continue
        if group_by == "state":
            group_key = str(record.last_state)
        elif group_by == "transition":
            if record.last_transition is None:
                continue
            group_key = str(record.last_transition)
        else:
            raise ValueError(f"Unsupported DHM assignment summary group: {group_by!r}")
        bucket = groups.setdefault(
            group_key,
            {
                "records": 0,
                "zero_pos": 0,
                "level_pos_counts": Counter(),
                "_sums": defaultdict(float),
            },
        )
        bucket["records"] += 1
        if int(record.last_pos_count) <= 0:
            bucket["zero_pos"] += 1
        level_counts = bucket["level_pos_counts"]
        if isinstance(level_counts, Counter):
            for level, count in record.last_level_pos_counts.items():
                level_counts[str(level)] += int(count)
        sums = bucket["_sums"]
        if isinstance(sums, defaultdict):
            for field_name in _ASSIGNMENT_MEAN_FIELDS:
                attr_name = _ASSIGNMENT_VALUE_ATTRS[field_name]
                sums[field_name] += float(getattr(record, attr_name))

    result: dict[str, dict[str, Any]] = {}
    for group_key, bucket in groups.items():
        count = int(bucket.get("records", 0))
        if count <= 0:
            continue
        sums = bucket.get("_sums", {})
        group_result = {
            "records": count,
            "zero_pos_rate": float(bucket.get("zero_pos", 0)) / float(count),
            "level_pos_counts": dict(bucket.get("level_pos_counts", {})),
        }
        if isinstance(sums, Mapping):
            for field_name in _ASSIGNMENT_MEAN_FIELDS:
                group_result[f"mean_{field_name}"] = float(sums.get(field_name, 0.0)) / float(count)
        result[group_key] = group_result
    return result


def load_dhm_config(path: str | Path, *, arch: str | None = None) -> DHMConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"DHM YAML must contain a mapping at the top level: {config_path}")
    return DHMConfig.from_mapping(raw, arch=arch)


def build_dhm_from_config(
    raw_config: Mapping[str, Any] | DHMConfig,
    *,
    arch: str | None = None,
) -> DetectionHysteresisMemory | None:
    config = raw_config if isinstance(raw_config, DHMConfig) else DHMConfig.from_mapping(raw_config, arch=arch)
    if not config.enabled:
        return None
    return DetectionHysteresisMemory(config)


def build_dhm_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> DetectionHysteresisMemory | None:
    config = load_dhm_config(path, arch=arch)
    if not config.enabled:
        return None
    return DetectionHysteresisMemory(config)


def _assign_detection_state(
    *,
    gt_box: torch.Tensor,
    gt_label: int,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    matching: DHMMatchingConfig,
) -> dict[str, object]:
    if pred_boxes.numel() == 0:
        return {"state": "FN_MISS", "score": 0.0, "iou": 0.0}
    gt_box = gt_box.to(device=pred_boxes.device, dtype=torch.float32).reshape(1, 4)
    ious = box_ops.box_iou(gt_box, pred_boxes.to(dtype=torch.float32))[0].clamp(min=0.0, max=1.0)
    target_mask = pred_labels == int(gt_label)
    tp_mask = target_mask & (ious >= float(matching.tau_iou)) & (pred_scores >= float(matching.tau_tp))
    if bool(tp_mask.any().item()):
        tp_indices = torch.where(tp_mask)[0]
        quality = pred_scores[tp_indices] * ious[tp_indices]
        best = tp_indices[int(torch.argmax(quality).item())]
        return {
            "state": "TP",
            "score": float(pred_scores[best].item()),
            "iou": float(ious[best].item()),
        }

    best_any_iou = float(ious.max().item()) if ious.numel() else 0.0
    best_target_score = 0.0
    best_near_target_score = 0.0
    best_near_wrong_score = 0.0
    best_target_iou = 0.0
    if bool(target_mask.any().item()):
        best_target_score = float(pred_scores[target_mask].max().item())
        evidence_mask = target_mask & (pred_scores >= float(matching.tau_cls_evidence))
        if bool(evidence_mask.any().item()):
            best_target_iou = float(ious[evidence_mask].max().item())
    near_mask = ious >= float(matching.tau_near)
    if bool((near_mask & target_mask).any().item()):
        best_near_target_score = float(pred_scores[near_mask & target_mask].max().item())
    if bool((near_mask & ~target_mask).any().item()):
        best_near_wrong_score = float(pred_scores[near_mask & ~target_mask].max().item())

    if best_any_iou < float(matching.tau_near) and best_target_score < float(matching.tau_bg_score):
        state = "FN_MISS"
    elif best_near_target_score < float(matching.tau_bg_score) and best_near_wrong_score < float(matching.tau_bg_score):
        state = "FN_BG"
    elif best_any_iou >= float(matching.tau_near) and best_near_target_score < float(matching.tau_tp):
        state = "FN_CLS"
    elif best_target_score >= float(matching.tau_loc_score) and best_target_iou < float(matching.tau_iou):
        state = "FN_LOC"
    else:
        state = "FN_BG"

    return {
        "state": state,
        "score": float(max(best_near_target_score, best_target_score)),
        "iou": float(max(best_target_iou, best_any_iou)),
    }


def _extract_gt_ids(target: Mapping[str, Any], count: int) -> list[Any | None]:
    for key in ("gt_ids", "annotation_ids", "ann_ids"):
        value = target.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            flattened = value.detach().cpu().flatten().tolist()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            flattened = list(value)
        else:
            flattened = [value]
        if len(flattened) == count:
            return [None if _is_invalid_gt_id(item) else item for item in flattened]
    return [None for _ in range(count)]


def _is_invalid_gt_id(value: Any) -> bool:
    if value is None:
        return True
    try:
        return int(value) < 0
    except (TypeError, ValueError):
        return str(value) == ""


def _gt_uid(
    *,
    image_id: str,
    class_id: int,
    bbox: torch.Tensor,
    image_shape: Sequence[int] | None,
    gt_id: Any | None,
) -> str:
    if gt_id is not None and not _is_invalid_gt_id(gt_id):
        return f"{image_id}:ann:{gt_id}"
    box = bbox.detach().cpu().to(dtype=torch.float32).flatten()
    if image_shape is not None and len(image_shape) == 2:
        height = max(1.0, float(image_shape[0]))
        width = max(1.0, float(image_shape[1]))
        scale = box.new_tensor([width, height, width, height])
        box = box / scale
    box_key = ",".join(f"{float(value):.6f}" for value in box.tolist())
    digest = hashlib.sha1(f"{image_id}:{class_id}:{box_key}".encode("utf-8")).hexdigest()[:16]
    return f"{image_id}:box:{class_id}:{digest}"


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        if value.numel() == 1:
            raw = value.detach().cpu().flatten()[0].item()
            if isinstance(raw, float) and raw.is_integer():
                return str(int(raw))
            return str(raw)
        return ",".join(str(item) for item in value.detach().cpu().flatten().tolist())
    return str(value)
