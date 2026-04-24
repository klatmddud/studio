from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.ops import boxes as box_ops

from .mdmb import normalize_arch
from .rasd import pool_multiscale_box_features


FNType = Literal["FN_BG", "FN_CLS", "FN_LOC", "FN_MISS"]
RetrievalMode = Literal["last", "topk", "topk_age"]

_ALLOWED_FN_TYPES = ("FN_BG", "FN_CLS", "FN_MISS")
_ALL_FN_TYPES = ("FN_BG", "FN_CLS", "FN_LOC", "FN_MISS")


@dataclass(frozen=True, slots=True)
class HTMMatchingConfig:
    tau_iou: float = 0.5
    tau_tp: float = 0.3
    tau_near: float = 0.3
    tau_bg_score: float = 0.1
    tau_cls_evidence: float = 0.3
    tau_loc_score: float = 0.3

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HTMMatchingConfig":
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
                raise ValueError(f"FN-TDM HTM matching.{field_name} must satisfy 0 <= value <= 1.")

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
class HTMTransitionConfig:
    transition_window: int = 3
    max_transitions_per_gt: int = 1
    allowed_fn_types: tuple[str, ...] = _ALLOWED_FN_TYPES
    min_direction_norm: float = 1.0e-6
    lambda_gap: float = 2.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HTMTransitionConfig":
        data = dict(raw or {})
        allowed = data.get("allowed_fn_types", _ALLOWED_FN_TYPES)
        if isinstance(allowed, str):
            allowed_types = (allowed,)
        else:
            allowed_types = tuple(str(value) for value in allowed)
        config = cls(
            transition_window=int(data.get("transition_window", 3)),
            max_transitions_per_gt=int(data.get("max_transitions_per_gt", 1)),
            allowed_fn_types=allowed_types,
            min_direction_norm=float(data.get("min_direction_norm", 1.0e-6)),
            lambda_gap=float(data.get("lambda_gap", 2.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.transition_window < 1:
            raise ValueError("FN-TDM HTM transition.transition_window must be >= 1.")
        if self.max_transitions_per_gt < 1:
            raise ValueError("FN-TDM HTM transition.max_transitions_per_gt must be >= 1.")
        unsupported = sorted(set(self.allowed_fn_types) - set(_ALL_FN_TYPES))
        if unsupported:
            raise ValueError(f"FN-TDM HTM transition.allowed_fn_types has unsupported values: {unsupported}.")
        if self.min_direction_norm <= 0.0:
            raise ValueError("FN-TDM HTM transition.min_direction_norm must be > 0.")
        if self.lambda_gap <= 0.0:
            raise ValueError("FN-TDM HTM transition.lambda_gap must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_window": self.transition_window,
            "max_transitions_per_gt": self.max_transitions_per_gt,
            "allowed_fn_types": list(self.allowed_fn_types),
            "min_direction_norm": self.min_direction_norm,
            "lambda_gap": self.lambda_gap,
        }


@dataclass(frozen=True, slots=True)
class HTMConfig:
    enabled: bool = True
    mode: str = "epoch_end_full_train"
    mine_interval: int = 1
    warmup_epochs: int = 0
    matching: HTMMatchingConfig = field(default_factory=HTMMatchingConfig)
    transition: HTMTransitionConfig = field(default_factory=HTMTransitionConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HTMConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", True)),
            mode=str(data.get("mode", "epoch_end_full_train")),
            mine_interval=int(data.get("mine_interval", 1)),
            warmup_epochs=int(data.get("warmup_epochs", 0)),
            matching=HTMMatchingConfig.from_mapping(data.get("matching")),
            transition=HTMTransitionConfig.from_mapping(data.get("transition")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode != "epoch_end_full_train":
            raise ValueError("FN-TDM HTM currently supports only mode='epoch_end_full_train'.")
        if self.mine_interval < 1:
            raise ValueError("FN-TDM HTM mine_interval must be >= 1.")
        if self.warmup_epochs < 0:
            raise ValueError("FN-TDM HTM warmup_epochs must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "mine_interval": self.mine_interval,
            "warmup_epochs": self.warmup_epochs,
            "matching": self.matching.to_dict(),
            "transition": self.transition.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TDBStorageConfig:
    max_entries_per_class: int = 128
    max_entries_per_gt: int = 1
    store_z_fn: bool = False
    store_z_tp: bool = False
    store_on_cpu: bool = True
    dtype: str = "float32"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TDBStorageConfig":
        data = dict(raw or {})
        config = cls(
            max_entries_per_class=int(data.get("max_entries_per_class", 128)),
            max_entries_per_gt=int(data.get("max_entries_per_gt", 1)),
            store_z_fn=bool(data.get("store_z_fn", False)),
            store_z_tp=bool(data.get("store_z_tp", False)),
            store_on_cpu=bool(data.get("store_on_cpu", True)),
            dtype=str(data.get("dtype", "float32")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_entries_per_class < 1:
            raise ValueError("FN-TDM TDB storage.max_entries_per_class must be >= 1.")
        if self.max_entries_per_gt < 1:
            raise ValueError("FN-TDM TDB storage.max_entries_per_gt must be >= 1.")
        if self.dtype not in {"float32", "float16"}:
            raise ValueError("FN-TDM TDB storage.dtype must be 'float32' or 'float16'.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_entries_per_class": self.max_entries_per_class,
            "max_entries_per_gt": self.max_entries_per_gt,
            "store_z_fn": self.store_z_fn,
            "store_z_tp": self.store_z_tp,
            "store_on_cpu": self.store_on_cpu,
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class TDBFilteringConfig:
    allowed_fn_types: tuple[str, ...] = _ALLOWED_FN_TYPES
    min_quality: float = 0.0
    min_direction_norm: float = 1.0e-6
    renormalize_direction: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TDBFilteringConfig":
        data = dict(raw or {})
        allowed = data.get("allowed_fn_types", _ALLOWED_FN_TYPES)
        if isinstance(allowed, str):
            allowed_types = (allowed,)
        else:
            allowed_types = tuple(str(value) for value in allowed)
        config = cls(
            allowed_fn_types=allowed_types,
            min_quality=float(data.get("min_quality", 0.0)),
            min_direction_norm=float(data.get("min_direction_norm", 1.0e-6)),
            renormalize_direction=bool(data.get("renormalize_direction", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        unsupported = sorted(set(self.allowed_fn_types) - set(_ALL_FN_TYPES))
        if unsupported:
            raise ValueError(f"FN-TDM TDB filtering.allowed_fn_types has unsupported values: {unsupported}.")
        if self.min_quality < 0.0:
            raise ValueError("FN-TDM TDB filtering.min_quality must be >= 0.")
        if self.min_direction_norm <= 0.0:
            raise ValueError("FN-TDM TDB filtering.min_direction_norm must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_fn_types": list(self.allowed_fn_types),
            "min_quality": self.min_quality,
            "min_direction_norm": self.min_direction_norm,
            "renormalize_direction": self.renormalize_direction,
        }


@dataclass(frozen=True, slots=True)
class TDBPrototypeConfig:
    retrieval: RetrievalMode = "topk"
    tau_proto: float = 0.2
    min_entries_for_query: int = 1
    min_prototype_norm: float = 1.0e-6
    use_age_decay: bool = False
    tau_age: float = 10.0
    use_failure_type_query: bool = False
    min_type_entries_for_query: int = 4

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TDBPrototypeConfig":
        data = dict(raw or {})
        config = cls(
            retrieval=str(data.get("retrieval", "topk")),  # type: ignore[arg-type]
            tau_proto=float(data.get("tau_proto", 0.2)),
            min_entries_for_query=int(data.get("min_entries_for_query", 1)),
            min_prototype_norm=float(data.get("min_prototype_norm", 1.0e-6)),
            use_age_decay=bool(data.get("use_age_decay", False)),
            tau_age=float(data.get("tau_age", 10.0)),
            use_failure_type_query=bool(data.get("use_failure_type_query", False)),
            min_type_entries_for_query=int(data.get("min_type_entries_for_query", 4)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.retrieval not in {"last", "topk", "topk_age"}:
            raise ValueError("FN-TDM TDB prototype.retrieval must be one of: last, topk, topk_age.")
        if self.tau_proto <= 0.0:
            raise ValueError("FN-TDM TDB prototype.tau_proto must be > 0.")
        if self.min_entries_for_query < 1:
            raise ValueError("FN-TDM TDB prototype.min_entries_for_query must be >= 1.")
        if self.min_prototype_norm <= 0.0:
            raise ValueError("FN-TDM TDB prototype.min_prototype_norm must be > 0.")
        if self.tau_age <= 0.0:
            raise ValueError("FN-TDM TDB prototype.tau_age must be > 0.")
        if self.min_type_entries_for_query < 1:
            raise ValueError("FN-TDM TDB prototype.min_type_entries_for_query must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval,
            "tau_proto": self.tau_proto,
            "min_entries_for_query": self.min_entries_for_query,
            "min_prototype_norm": self.min_prototype_norm,
            "use_age_decay": self.use_age_decay,
            "tau_age": self.tau_age,
            "use_failure_type_query": self.use_failure_type_query,
            "min_type_entries_for_query": self.min_type_entries_for_query,
        }


@dataclass(frozen=True, slots=True)
class TDBConfig:
    enabled: bool = True
    storage: TDBStorageConfig = field(default_factory=TDBStorageConfig)
    filtering: TDBFilteringConfig = field(default_factory=TDBFilteringConfig)
    prototype: TDBPrototypeConfig = field(default_factory=TDBPrototypeConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TDBConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", True)),
            storage=TDBStorageConfig.from_mapping(data.get("storage")),
            filtering=TDBFilteringConfig.from_mapping(data.get("filtering")),
            prototype=TDBPrototypeConfig.from_mapping(data.get("prototype")),
        )
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "storage": self.storage.to_dict(),
            "filtering": self.filtering.to_dict(),
            "prototype": self.prototype.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TCSHardnessConfig:
    source: str = "assigned_cls_confidence"
    tau_hard: float = 0.5
    include_centerness: bool = True
    include_missing_assignment: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TCSHardnessConfig":
        data = dict(raw or {})
        config = cls(
            source=str(data.get("source", "assigned_cls_confidence")),
            tau_hard=float(data.get("tau_hard", 0.5)),
            include_centerness=bool(data.get("include_centerness", True)),
            include_missing_assignment=bool(data.get("include_missing_assignment", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.source != "assigned_cls_confidence":
            raise ValueError("FN-TDM TCS hardness.source currently supports only assigned_cls_confidence.")
        if not 0.0 <= self.tau_hard <= 1.0:
            raise ValueError("FN-TDM TCS hardness.tau_hard must satisfy 0 <= value <= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "tau_hard": self.tau_hard,
            "include_centerness": self.include_centerness,
            "include_missing_assignment": self.include_missing_assignment,
        }


@dataclass(frozen=True, slots=True)
class TCSBudgetConfig:
    max_candidates_per_image: int = 8
    max_candidates_per_batch: int = 32
    rank_by: str = "hardness"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TCSBudgetConfig":
        data = dict(raw or {})
        config = cls(
            max_candidates_per_image=int(data.get("max_candidates_per_image", 8)),
            max_candidates_per_batch=int(data.get("max_candidates_per_batch", 32)),
            rank_by=str(data.get("rank_by", "hardness")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_candidates_per_image < 1:
            raise ValueError("FN-TDM TCS budget.max_candidates_per_image must be >= 1.")
        if self.max_candidates_per_batch < 1:
            raise ValueError("FN-TDM TCS budget.max_candidates_per_batch must be >= 1.")
        if self.rank_by != "hardness":
            raise ValueError("FN-TDM TCS budget.rank_by currently supports only hardness.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates_per_image": self.max_candidates_per_image,
            "max_candidates_per_batch": self.max_candidates_per_batch,
            "rank_by": self.rank_by,
        }


@dataclass(frozen=True, slots=True)
class TCSWeightingConfig:
    use_candidate_weight: bool = False
    min_weight: float = 0.25
    max_weight: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TCSWeightingConfig":
        data = dict(raw or {})
        config = cls(
            use_candidate_weight=bool(data.get("use_candidate_weight", False)),
            min_weight=float(data.get("min_weight", 0.25)),
            max_weight=float(data.get("max_weight", 1.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.min_weight < 0.0:
            raise ValueError("FN-TDM TCS weighting.min_weight must be >= 0.")
        if self.max_weight < self.min_weight:
            raise ValueError("FN-TDM TCS weighting.max_weight must be >= min_weight.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_candidate_weight": self.use_candidate_weight,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }


@dataclass(frozen=True, slots=True)
class TCSConfig:
    enabled: bool = True
    selection_mode: str = "current_hard"
    require_tdb_direction: bool = True
    use_failure_type_query: bool = False
    hardness: TCSHardnessConfig = field(default_factory=TCSHardnessConfig)
    budget: TCSBudgetConfig = field(default_factory=TCSBudgetConfig)
    weighting: TCSWeightingConfig = field(default_factory=TCSWeightingConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TCSConfig":
        data = dict(raw or {})
        memory_gate = data.get("memory_gate", {})
        if not isinstance(memory_gate, Mapping):
            memory_gate = {}
        config = cls(
            enabled=bool(data.get("enabled", True)),
            selection_mode=str(data.get("selection_mode", "current_hard")),
            require_tdb_direction=bool(memory_gate.get("require_tdb_direction", True)),
            use_failure_type_query=bool(memory_gate.get("use_failure_type_query", False)),
            hardness=TCSHardnessConfig.from_mapping(data.get("hardness")),
            budget=TCSBudgetConfig.from_mapping(data.get("budget")),
            weighting=TCSWeightingConfig.from_mapping(data.get("weighting")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.selection_mode != "current_hard":
            raise ValueError("FN-TDM TCS currently supports only selection_mode='current_hard'.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "selection_mode": self.selection_mode,
            "memory_gate": {
                "require_tdb_direction": self.require_tdb_direction,
                "use_failure_type_query": self.use_failure_type_query,
            },
            "hardness": self.hardness.to_dict(),
            "budget": self.budget.to_dict(),
            "weighting": self.weighting.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TALConfig:
    enabled: bool = True
    variant: str = "anchor_shift"
    lambda_tal: float = 0.05
    alpha: float = 0.2
    reduction: str = "mean"
    start_epoch: int = 1
    lambda_warmup_epochs: int = 2
    input_dim: int = 256
    projector_dim: int = 256
    roi_output_size: int = 7
    roi_sampling_ratio: int = 2
    normalize: bool = True
    skip_invalid_direction: bool = True
    min_direction_norm: float = 1.0e-6
    skip_invalid_bbox: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TALConfig":
        data = dict(raw or {})
        loss = data.get("loss", {})
        if not isinstance(loss, Mapping):
            loss = {}
        schedule = data.get("schedule", {})
        if not isinstance(schedule, Mapping):
            schedule = {}
        features = data.get("features", {})
        if not isinstance(features, Mapping):
            features = {}
        safety = data.get("safety", {})
        if not isinstance(safety, Mapping):
            safety = {}
        config = cls(
            enabled=bool(data.get("enabled", True)),
            variant=str(data.get("variant", "anchor_shift")),
            lambda_tal=float(loss.get("lambda_tal", 0.05)),
            alpha=float(loss.get("alpha", 0.2)),
            reduction=str(loss.get("reduction", "mean")),
            start_epoch=int(schedule.get("start_epoch", 1)),
            lambda_warmup_epochs=int(schedule.get("lambda_warmup_epochs", 2)),
            input_dim=int(features.get("input_dim", 256)),
            projector_dim=int(features.get("projector_dim", 256)),
            roi_output_size=int(features.get("roi_output_size", 7)),
            roi_sampling_ratio=int(features.get("roi_sampling_ratio", 2)),
            normalize=bool(features.get("normalize", True)),
            skip_invalid_direction=bool(safety.get("skip_invalid_direction", True)),
            min_direction_norm=float(safety.get("min_direction_norm", 1.0e-6)),
            skip_invalid_bbox=bool(safety.get("skip_invalid_bbox", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.variant != "anchor_shift":
            raise ValueError("FN-TDM TAL currently supports only variant='anchor_shift'.")
        if self.lambda_tal < 0.0:
            raise ValueError("FN-TDM TAL loss.lambda_tal must be >= 0.")
        if self.alpha < 0.0:
            raise ValueError("FN-TDM TAL loss.alpha must be >= 0.")
        if self.reduction != "mean":
            raise ValueError("FN-TDM TAL loss.reduction currently supports only mean.")
        if self.start_epoch < 0:
            raise ValueError("FN-TDM TAL schedule.start_epoch must be >= 0.")
        if self.lambda_warmup_epochs < 0:
            raise ValueError("FN-TDM TAL schedule.lambda_warmup_epochs must be >= 0.")
        if self.input_dim < 1 or self.projector_dim < 1:
            raise ValueError("FN-TDM TAL feature dimensions must be >= 1.")
        if self.roi_output_size < 1:
            raise ValueError("FN-TDM TAL features.roi_output_size must be >= 1.")
        if self.roi_sampling_ratio < 0:
            raise ValueError("FN-TDM TAL features.roi_sampling_ratio must be >= 0.")
        if self.min_direction_norm <= 0.0:
            raise ValueError("FN-TDM TAL safety.min_direction_norm must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "variant": self.variant,
            "loss": {
                "lambda_tal": self.lambda_tal,
                "alpha": self.alpha,
                "reduction": self.reduction,
            },
            "schedule": {
                "start_epoch": self.start_epoch,
                "lambda_warmup_epochs": self.lambda_warmup_epochs,
            },
            "features": {
                "input_dim": self.input_dim,
                "projector_dim": self.projector_dim,
                "roi_output_size": self.roi_output_size,
                "roi_sampling_ratio": self.roi_sampling_ratio,
                "normalize": self.normalize,
            },
            "safety": {
                "skip_invalid_direction": self.skip_invalid_direction,
                "min_direction_norm": self.min_direction_norm,
                "skip_invalid_bbox": self.skip_invalid_bbox,
            },
        }


@dataclass(frozen=True, slots=True)
class FNTDMConfig:
    enabled: bool = False
    htm: HTMConfig = field(default_factory=HTMConfig)
    tdb: TDBConfig = field(default_factory=TDBConfig)
    tcs: TCSConfig = field(default_factory=TCSConfig)
    tal: TALConfig = field(default_factory=TALConfig)
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "FNTDMConfig":
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
            htm=HTMConfig.from_mapping(merged.get("htm")),
            tdb=TDBConfig.from_mapping(merged.get("tdb")),
            tcs=TCSConfig.from_mapping(merged.get("tcs")),
            tal=TALConfig.from_mapping(merged.get("tal")),
            arch=normalized_arch,
        )
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "htm": self.htm.to_dict(),
            "tdb": self.tdb.to_dict(),
            "tcs": self.tcs.to_dict(),
            "tal": self.tal.to_dict(),
            "arch": self.arch,
        }


@dataclass(slots=True)
class TransitionEvent:
    gt_uid: str
    image_id: str
    ann_id: str | None
    class_id: int
    bbox: torch.Tensor
    epoch_fn: int
    epoch_tp: int
    fn_type: str
    z_fn: torch.Tensor
    z_tp: torch.Tensor
    direction: torch.Tensor
    score_fn: float
    score_tp: float
    iou_fn: float
    iou_tp: float
    quality: float


@dataclass(slots=True)
class TDBEntry:
    entry_id: str
    gt_uid: str
    image_id: str
    ann_id: str | None
    class_id: int
    bbox: torch.Tensor | None
    fn_type: str
    epoch_fn: int
    epoch_tp: int
    direction: torch.Tensor
    quality: float
    score_fn: float
    score_tp: float
    iou_fn: float
    iou_tp: float
    z_fn: torch.Tensor | None = None
    z_tp: torch.Tensor | None = None

    @property
    def gt_key(self) -> tuple[str, str]:
        return (self.gt_uid, self.fn_type)

    def to_state(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "gt_uid": self.gt_uid,
            "image_id": self.image_id,
            "ann_id": self.ann_id,
            "class_id": self.class_id,
            "bbox": None if self.bbox is None else self.bbox.detach().cpu(),
            "fn_type": self.fn_type,
            "epoch_fn": self.epoch_fn,
            "epoch_tp": self.epoch_tp,
            "direction": self.direction.detach().cpu(),
            "quality": self.quality,
            "score_fn": self.score_fn,
            "score_tp": self.score_tp,
            "iou_fn": self.iou_fn,
            "iou_tp": self.iou_tp,
            "z_fn": None if self.z_fn is None else self.z_fn.detach().cpu(),
            "z_tp": None if self.z_tp is None else self.z_tp.detach().cpu(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TDBEntry":
        return cls(
            entry_id=str(state["entry_id"]),
            gt_uid=str(state["gt_uid"]),
            image_id=str(state.get("image_id", "")),
            ann_id=None if state.get("ann_id") is None else str(state.get("ann_id")),
            class_id=int(state["class_id"]),
            bbox=None if state.get("bbox") is None else torch.as_tensor(state["bbox"], dtype=torch.float32).reshape(4),
            fn_type=str(state["fn_type"]),
            epoch_fn=int(state["epoch_fn"]),
            epoch_tp=int(state["epoch_tp"]),
            direction=torch.as_tensor(state["direction"], dtype=torch.float32).flatten(),
            quality=float(state["quality"]),
            score_fn=float(state.get("score_fn", 0.0)),
            score_tp=float(state.get("score_tp", 0.0)),
            iou_fn=float(state.get("iou_fn", 0.0)),
            iou_tp=float(state.get("iou_tp", 0.0)),
            z_fn=None if state.get("z_fn") is None else torch.as_tensor(state["z_fn"], dtype=torch.float32).flatten(),
            z_tp=None if state.get("z_tp") is None else torch.as_tensor(state["z_tp"], dtype=torch.float32).flatten(),
        )


@dataclass(slots=True)
class HTMHistory:
    gt_uid: str
    image_id: str
    ann_id: str | None
    class_id: int
    bbox: torch.Tensor
    last_state: str = "UNSEEN"
    last_epoch: int | None = None
    last_fn_epoch: int | None = None
    last_fn_z: torch.Tensor | None = None
    last_fn_score: float | None = None
    last_fn_iou: float | None = None
    last_fn_type: str | None = None
    last_tp_epoch: int | None = None
    last_tp_z: torch.Tensor | None = None
    last_tp_score: float | None = None
    last_tp_iou: float | None = None
    fn_count: int = 0
    tp_count: int = 0
    transition_count: int = 0
    last_emitted_epoch: int | None = None

    def to_state(self) -> dict[str, Any]:
        return {
            "gt_uid": self.gt_uid,
            "image_id": self.image_id,
            "ann_id": self.ann_id,
            "class_id": self.class_id,
            "bbox": self.bbox.detach().cpu(),
            "last_state": self.last_state,
            "last_epoch": self.last_epoch,
            "last_fn_epoch": self.last_fn_epoch,
            "last_fn_z": None if self.last_fn_z is None else self.last_fn_z.detach().cpu(),
            "last_fn_score": self.last_fn_score,
            "last_fn_iou": self.last_fn_iou,
            "last_fn_type": self.last_fn_type,
            "last_tp_epoch": self.last_tp_epoch,
            "last_tp_z": None if self.last_tp_z is None else self.last_tp_z.detach().cpu(),
            "last_tp_score": self.last_tp_score,
            "last_tp_iou": self.last_tp_iou,
            "fn_count": self.fn_count,
            "tp_count": self.tp_count,
            "transition_count": self.transition_count,
            "last_emitted_epoch": self.last_emitted_epoch,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "HTMHistory":
        return cls(
            gt_uid=str(state["gt_uid"]),
            image_id=str(state.get("image_id", "")),
            ann_id=None if state.get("ann_id") is None else str(state.get("ann_id")),
            class_id=int(state["class_id"]),
            bbox=torch.as_tensor(state["bbox"], dtype=torch.float32).reshape(4),
            last_state=str(state.get("last_state", "UNSEEN")),
            last_epoch=state.get("last_epoch"),
            last_fn_epoch=state.get("last_fn_epoch"),
            last_fn_z=None if state.get("last_fn_z") is None else torch.as_tensor(state["last_fn_z"], dtype=torch.float32).flatten(),
            last_fn_score=None if state.get("last_fn_score") is None else float(state.get("last_fn_score")),
            last_fn_iou=None if state.get("last_fn_iou") is None else float(state.get("last_fn_iou")),
            last_fn_type=None if state.get("last_fn_type") is None else str(state.get("last_fn_type")),
            last_tp_epoch=state.get("last_tp_epoch"),
            last_tp_z=None if state.get("last_tp_z") is None else torch.as_tensor(state["last_tp_z"], dtype=torch.float32).flatten(),
            last_tp_score=None if state.get("last_tp_score") is None else float(state.get("last_tp_score")),
            last_tp_iou=None if state.get("last_tp_iou") is None else float(state.get("last_tp_iou")),
            fn_count=int(state.get("fn_count", 0)),
            tp_count=int(state.get("tp_count", 0)),
            transition_count=int(state.get("transition_count", 0)),
            last_emitted_epoch=state.get("last_emitted_epoch"),
        )


@dataclass(slots=True)
class TCSCandidate:
    gt_uid: str | None
    image_id: str
    ann_id: str | None
    batch_index: int
    target_index: int
    class_id: int
    bbox: torch.Tensor
    hardness: float
    confidence: float
    selection_reason: str
    direction: torch.Tensor
    direction_source: str = "class_prototype"
    fn_type_hint: str | None = None
    weight: float = 1.0


class TransitionDirectionBank:
    def __init__(self, config: TDBConfig) -> None:
        self.config = config
        self.current_epoch = 0
        self._bank: dict[int, list[TDBEntry]] = defaultdict(list)
        self._revision = 0
        self._stats: Counter[str] = Counter()

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._stats.clear()

    def update(self, events: Sequence[TransitionEvent], *, epoch: int | None = None) -> dict[str, int]:
        if epoch is not None:
            self.current_epoch = int(epoch)
        stats = Counter()
        for event in events:
            stats["received"] += 1
            entry = self._event_to_entry(event)
            if entry is None:
                stats["rejected"] += 1
                continue
            inserted, replaced, duplicate = self._insert(entry)
            if inserted:
                stats["stored"] += 1
            if replaced:
                stats["replaced"] += 1
            if duplicate:
                stats["duplicates"] += 1
        self._stats.update(stats)
        return dict(stats)

    def get_prototype(
        self,
        class_id: int,
        fn_type: str | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        entries = self._entries_for_query(int(class_id), fn_type=fn_type)
        if not entries:
            return None
        if len(entries) < int(self.config.prototype.min_entries_for_query):
            return None

        mode = self.config.prototype.retrieval
        if mode == "last":
            selected = max(entries, key=lambda item: (int(item.epoch_tp), float(item.quality), item.entry_id))
            direction = selected.direction.detach()
            return direction.to(device=device) if device is not None else direction

        weights = []
        directions = []
        for entry in entries:
            quality = float(entry.quality)
            if mode == "topk_age" or bool(self.config.prototype.use_age_decay):
                age = max(0, int(self.current_epoch) - int(entry.epoch_tp))
                quality *= math.exp(-float(age) / float(self.config.prototype.tau_age))
            weights.append(quality)
            directions.append(entry.direction.detach())
        if not directions:
            return None

        direction_tensor = torch.stack(directions, dim=0)
        weight_tensor = torch.tensor(
            weights,
            dtype=direction_tensor.dtype,
            device=direction_tensor.device,
        )
        weight_tensor = torch.softmax(weight_tensor / float(self.config.prototype.tau_proto), dim=0)
        prototype = (direction_tensor * weight_tensor.unsqueeze(1)).sum(dim=0)
        norm = torch.linalg.vector_norm(prototype)
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) < float(self.config.prototype.min_prototype_norm):
            return None
        prototype = prototype / norm.clamp(min=float(self.config.prototype.min_prototype_norm))
        return prototype.to(device=device) if device is not None else prototype

    def sample(
        self,
        class_id: int,
        *,
        k: int = 1,
        fn_type: str | None = None,
    ) -> list[TDBEntry]:
        entries = self._entries_for_query(int(class_id), fn_type=fn_type)
        if not entries:
            return []
        return list(entries[: max(1, int(k))])

    def has_entries(self, class_id: int) -> bool:
        return bool(self._bank.get(int(class_id)))

    def summary(self) -> dict[str, Any]:
        per_class_counts = {str(class_id): len(entries) for class_id, entries in self._bank.items()}
        total_entries = sum(per_class_counts.values())
        mean_entries = total_entries / float(max(len(per_class_counts), 1))
        return {
            "total_entries": total_entries,
            "num_classes_with_entries": len(per_class_counts),
            "entries_per_class_mean": mean_entries,
            "entries_per_class_max": max(per_class_counts.values(), default=0),
            "events_received": int(self._stats.get("received", 0)),
            "events_stored": int(self._stats.get("stored", 0)),
            "events_rejected": int(self._stats.get("rejected", 0)),
            "duplicate_events": int(self._stats.get("duplicates", 0)),
            "replaced_entries": int(self._stats.get("replaced", 0)),
            "per_class_counts": per_class_counts,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "bank": {
                str(class_id): [entry.to_state() for entry in entries]
                for class_id, entries in self._bank.items()
            },
            "stats": dict(self._stats),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            self._bank.clear()
            self._stats.clear()
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))
        bank: dict[int, list[TDBEntry]] = defaultdict(list)
        raw_bank = state.get("bank", {})
        if isinstance(raw_bank, Mapping):
            for raw_class, raw_entries in raw_bank.items():
                if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
                    continue
                class_id = int(raw_class)
                restored = []
                for raw_entry in raw_entries:
                    if isinstance(raw_entry, Mapping):
                        restored.append(TDBEntry.from_state(raw_entry))
                restored.sort(key=lambda item: (-float(item.quality), -int(item.epoch_tp), item.entry_id))
                bank[class_id] = restored[: int(self.config.storage.max_entries_per_class)]
        self._bank = bank
        raw_stats = state.get("stats", {})
        self._stats = Counter({str(k): int(v) for k, v in raw_stats.items()}) if isinstance(raw_stats, Mapping) else Counter()
        self._revision += 1

    def _entries_for_query(self, class_id: int, *, fn_type: str | None) -> list[TDBEntry]:
        entries = list(self._bank.get(int(class_id), ()))
        if fn_type and bool(self.config.prototype.use_failure_type_query):
            typed = [entry for entry in entries if entry.fn_type == str(fn_type)]
            if len(typed) >= int(self.config.prototype.min_type_entries_for_query):
                entries = typed
        entries.sort(key=lambda item: (-float(item.quality), -int(item.epoch_tp), item.entry_id))
        return entries

    def _event_to_entry(self, event: TransitionEvent) -> TDBEntry | None:
        if not self.config.enabled:
            return None
        if event.fn_type not in set(self.config.filtering.allowed_fn_types):
            return None
        if not math.isfinite(float(event.quality)) or float(event.quality) <= float(self.config.filtering.min_quality):
            return None
        direction = torch.as_tensor(event.direction, dtype=self._storage_dtype()).detach().flatten()
        if direction.numel() == 0 or not bool(torch.isfinite(direction).all().item()):
            return None
        norm = torch.linalg.vector_norm(direction.to(dtype=torch.float32))
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) < float(self.config.filtering.min_direction_norm):
            return None
        if self.config.filtering.renormalize_direction:
            direction = direction / norm.to(dtype=direction.dtype).clamp(min=float(self.config.filtering.min_direction_norm))
        if self.config.storage.store_on_cpu:
            direction = direction.cpu()

        entry_id = _entry_id(event.gt_uid, event.epoch_fn, event.epoch_tp, event.fn_type)
        bbox = event.bbox.detach().cpu().reshape(4) if event.bbox is not None else None
        return TDBEntry(
            entry_id=entry_id,
            gt_uid=str(event.gt_uid),
            image_id=str(event.image_id),
            ann_id=event.ann_id,
            class_id=int(event.class_id),
            bbox=bbox,
            fn_type=str(event.fn_type),
            epoch_fn=int(event.epoch_fn),
            epoch_tp=int(event.epoch_tp),
            direction=direction,
            quality=float(event.quality),
            score_fn=float(event.score_fn),
            score_tp=float(event.score_tp),
            iou_fn=float(event.iou_fn),
            iou_tp=float(event.iou_tp),
            z_fn=_maybe_store_vector(event.z_fn, self.config.storage.store_z_fn, self._storage_dtype()),
            z_tp=_maybe_store_vector(event.z_tp, self.config.storage.store_z_tp, self._storage_dtype()),
        )

    def _insert(self, entry: TDBEntry) -> tuple[bool, bool, bool]:
        entries = list(self._bank.get(entry.class_id, ()))
        duplicate = False
        replaced = False

        for index, current in enumerate(entries):
            if current.entry_id == entry.entry_id:
                duplicate = True
                if float(entry.quality) > float(current.quality):
                    entries[index] = entry
                    replaced = True
                    self._bank[entry.class_id] = self._trim(entries)
                    self._revision += 1
                    return True, replaced, duplicate
                return False, replaced, duplicate

        if self.config.storage.max_entries_per_gt == 1:
            for index, current in enumerate(entries):
                if current.gt_key == entry.gt_key:
                    duplicate = True
                    if float(entry.quality) > float(current.quality):
                        entries[index] = entry
                        replaced = True
                        self._bank[entry.class_id] = self._trim(entries)
                        self._revision += 1
                        return True, replaced, duplicate
                    return False, replaced, duplicate

        entries.append(entry)
        self._bank[entry.class_id] = self._trim(entries)
        self._revision += 1
        return True, replaced, duplicate

    def _trim(self, entries: list[TDBEntry]) -> list[TDBEntry]:
        entries.sort(key=lambda item: (-float(item.quality), -int(item.epoch_tp), item.entry_id))
        return entries[: int(self.config.storage.max_entries_per_class)]

    def _storage_dtype(self) -> torch.dtype:
        return torch.float16 if self.config.storage.dtype == "float16" else torch.float32


class FalseNegativeTransitionDirectionMemory(nn.Module):
    def __init__(self, config: FNTDMConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self.tdb = TransitionDirectionBank(config.tdb)
        self.projector = nn.Linear(int(config.tal.input_dim), int(config.tal.projector_dim))
        self._histories: dict[str, HTMHistory] = {}
        self._htm_stats: Counter[str] = Counter()
        self._tcs_stats: Counter[str] = Counter()
        self._tal_stats: Counter[str] = Counter()
        self._tal_loss_sum = 0.0
        self._tal_cosine_sum = 0.0
        self._tal_hardness_sum = 0.0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._htm_stats.clear()
        self._tcs_stats.clear()
        self._tal_stats.clear()
        self._tal_loss_sum = 0.0
        self._tal_cosine_sum = 0.0
        self._tal_hardness_sum = 0.0
        self.tdb.start_epoch(epoch)

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def should_mine(self, *, epoch: int | None = None) -> bool:
        if epoch is not None:
            epoch_value = int(epoch)
        else:
            epoch_value = int(self.current_epoch)
        if not self.config.enabled or not self.config.htm.enabled or not self.config.tdb.enabled:
            return False
        if epoch_value <= int(self.config.htm.warmup_epochs):
            return False
        return epoch_value % int(self.config.htm.mine_interval) == 0

    def loss_weight(self) -> float:
        if not self.config.enabled or not self.config.tal.enabled:
            return 0.0
        if self.current_epoch < int(self.config.tal.start_epoch):
            return 0.0
        warmup = int(self.config.tal.lambda_warmup_epochs)
        if warmup <= 0:
            factor = 1.0
        else:
            progress = max(0, self.current_epoch - int(self.config.tal.start_epoch) + 1)
            factor = min(1.0, float(progress) / float(warmup))
        return float(self.config.tal.lambda_tal) * factor

    def compute_loss(
        self,
        *,
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_shapes: Sequence[Sequence[int]],
        features: Mapping[str, torch.Tensor],
        head_outputs: Mapping[str, torch.Tensor],
        anchors: Sequence[torch.Tensor],
        matched_idxs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        template = _first_feature(features)
        zero = template.new_zeros(())
        weight = self.loss_weight()
        if weight <= 0.0 or not self.config.tcs.enabled:
            return zero

        candidates = self.select_candidates(
            targets=targets,
            head_outputs=head_outputs,
            matched_idxs=matched_idxs,
            device=template.device,
        )
        if not candidates:
            return zero

        loss, stats = self._compute_anchor_shift_loss(
            candidates=candidates,
            features=features,
            image_shapes=image_shapes,
        )
        self._record_tal_step(candidates=candidates, loss=loss, stats=stats)
        return loss * float(weight)

    @torch.no_grad()
    def mine_batch(
        self,
        *,
        detections: Sequence[Mapping[str, torch.Tensor]],
        original_targets: Sequence[Mapping[str, torch.Tensor]],
        transformed_targets: Sequence[Mapping[str, torch.Tensor]],
        transformed_image_shapes: Sequence[Sequence[int]],
        features: Mapping[str, torch.Tensor],
        epoch: int,
    ) -> dict[str, int]:
        if not self.should_mine(epoch=epoch):
            return {}
        self.current_epoch = int(epoch)
        events = self._mine_transition_events(
            detections=detections,
            original_targets=original_targets,
            transformed_targets=transformed_targets,
            transformed_image_shapes=transformed_image_shapes,
            features=features,
            epoch=epoch,
        )
        update_stats = self.tdb.update(events, epoch=epoch)
        self._htm_stats["events_emitted"] += len(events)
        for key, value in update_stats.items():
            self._htm_stats[f"tdb_{key}"] += int(value)
        return {"events": len(events), **{f"tdb_{key}": int(value) for key, value in update_stats.items()}}

    def select_candidates(
        self,
        *,
        targets: Sequence[Mapping[str, torch.Tensor]],
        head_outputs: Mapping[str, torch.Tensor],
        matched_idxs: Sequence[torch.Tensor],
        device: torch.device,
    ) -> list[TCSCandidate]:
        candidates: list[TCSCandidate] = []
        cls_logits_batch = head_outputs["cls_logits"]
        ctr_batch = head_outputs.get("bbox_ctrness")
        per_image_selected: list[list[TCSCandidate]] = []

        for image_index, target in enumerate(targets):
            self._tcs_stats["gt_seen"] += int(target["boxes"].shape[0])
            if _is_replay_target(target):
                per_image_selected.append([])
                continue
            image_candidates: list[TCSCandidate] = []
            boxes = target["boxes"]
            labels = target["labels"].to(dtype=torch.int64)
            if boxes.numel() == 0:
                per_image_selected.append([])
                continue
            assignments = matched_idxs[image_index]
            cls_logits = cls_logits_batch[image_index]
            ctr_logits = ctr_batch[image_index].flatten() if ctr_batch is not None else None
            pos_mask = assignments >= 0
            image_id = _normalize_image_id(target.get("image_id", torch.tensor(image_index)))
            gt_ids = _extract_gt_ids(target, int(boxes.shape[0]))

            for gt_index in range(int(boxes.shape[0])):
                class_id = int(labels[gt_index].item())
                direction = self.tdb.get_prototype(class_id, device=device)
                if direction is None:
                    self._tcs_stats["skipped_no_memory"] += 1
                    continue
                self._tcs_stats["memory_covered_gt"] += 1

                confidence, reason = _confidence_for_gt(
                    gt_index=gt_index,
                    class_id=class_id,
                    cls_logits=cls_logits,
                    ctr_logits=ctr_logits,
                    assignments=assignments,
                    pos_mask=pos_mask,
                    include_centerness=bool(self.config.tcs.hardness.include_centerness),
                )
                if reason == "missing_assignment" and not bool(self.config.tcs.hardness.include_missing_assignment):
                    self._tcs_stats["skipped_missing_assignment"] += 1
                    continue
                hardness = 1.0 - float(confidence)
                if hardness < float(self.config.tcs.hardness.tau_hard):
                    self._tcs_stats["skipped_easy"] += 1
                    continue
                self._tcs_stats["hard_gt"] += 1
                ann_id = gt_ids[gt_index]
                gt_uid = _gt_uid(
                    image_id=image_id,
                    class_id=class_id,
                    bbox=boxes[gt_index],
                    image_shape=None,
                    gt_id=ann_id,
                )
                weight = 1.0
                if bool(self.config.tcs.weighting.use_candidate_weight):
                    weight = min(
                        float(self.config.tcs.weighting.max_weight),
                        max(float(self.config.tcs.weighting.min_weight), hardness),
                    )
                image_candidates.append(
                    TCSCandidate(
                        gt_uid=gt_uid,
                        image_id=image_id,
                        ann_id=None if ann_id is None else str(ann_id),
                        batch_index=image_index,
                        target_index=gt_index,
                        class_id=class_id,
                        bbox=boxes[gt_index].detach(),
                        hardness=hardness,
                        confidence=float(confidence),
                        selection_reason=reason,
                        direction=direction.detach(),
                        weight=weight,
                    )
                )

            image_candidates.sort(key=lambda item: (-item.hardness, _box_area(item.bbox), item.gt_uid or ""))
            per_image_selected.append(image_candidates[: int(self.config.tcs.budget.max_candidates_per_image)])

        for image_candidates in per_image_selected:
            candidates.extend(image_candidates)
        candidates.sort(key=lambda item: (-item.hardness, item.batch_index, item.target_index))
        candidates = candidates[: int(self.config.tcs.budget.max_candidates_per_batch)]
        self._tcs_stats["selected_candidates"] += len(candidates)
        return candidates

    def summary(self) -> dict[str, Any]:
        tdb_summary = self.tdb.summary()
        loss_count = int(self._tal_stats.get("losses", 0))
        candidate_count = int(self._tal_stats.get("valid_candidates", 0))
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "loss_weight": self.loss_weight(),
            "htm": {
                "histories": len(self._histories),
                **{key: int(value) for key, value in self._htm_stats.items()},
            },
            "tdb": tdb_summary,
            "tcs": {key: int(value) for key, value in self._tcs_stats.items()},
            "tal": {
                **{key: int(value) for key, value in self._tal_stats.items()},
                "mean_loss": self._tal_loss_sum / float(max(loss_count, 1)),
                "mean_cosine": self._tal_cosine_sum / float(max(candidate_count, 1)),
                "mean_hardness": self._tal_hardness_sum / float(max(candidate_count, 1)),
            },
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "tdb": self.tdb.state_dict(),
            "htm": {
                "histories": {uid: history.to_state() for uid, history in self._histories.items()},
                "stats": dict(self._htm_stats),
            },
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            self._histories.clear()
            self.tdb.load_state_dict(None)
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))
        self.tdb.load_state_dict(state.get("tdb"))
        raw_htm = state.get("htm", {})
        histories: dict[str, HTMHistory] = {}
        if isinstance(raw_htm, Mapping):
            raw_histories = raw_htm.get("histories", {})
            if isinstance(raw_histories, Mapping):
                for uid, raw_history in raw_histories.items():
                    if isinstance(raw_history, Mapping):
                        histories[str(uid)] = HTMHistory.from_state(raw_history)
            raw_stats = raw_htm.get("stats", {})
            self._htm_stats = Counter({str(k): int(v) for k, v in raw_stats.items()}) if isinstance(raw_stats, Mapping) else Counter()
        self._histories = histories

    def _compute_anchor_shift_loss(
        self,
        *,
        candidates: Sequence[TCSCandidate],
        features: Mapping[str, torch.Tensor],
        image_shapes: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        template = _first_feature(features)
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (int(item.batch_index), int(item.target_index), item.gt_uid or ""),
        )
        boxes_per_image = _candidate_boxes_per_image(
            ordered_candidates,
            len(image_shapes),
            template.device,
        )
        pooled = pool_multiscale_box_features(
            features=features,
            boxes_per_image=boxes_per_image,
            image_shapes=image_shapes,
            output_size=int(self.config.tal.roi_output_size),
            sampling_ratio=int(self.config.tal.roi_sampling_ratio),
            normalize=False,
        )
        if pooled.shape[0] != len(ordered_candidates):
            return template.new_zeros(()), {"valid": 0, "skipped": len(ordered_candidates), "cosine_sum": 0.0}
        if pooled.shape[1] != int(self.config.tal.input_dim):
            raise ValueError(
                "FN-TDM TAL projector input dimension mismatch: "
                f"pooled={pooled.shape[1]} config={self.config.tal.input_dim}."
            )
        z = self.projector(pooled)
        if bool(self.config.tal.normalize):
            z = F.normalize(z, p=2, dim=1)

        losses = []
        weights = []
        cosine_sum = 0.0
        skipped = 0
        for index, candidate in enumerate(ordered_candidates):
            direction = candidate.direction.to(device=z.device, dtype=z.dtype).detach().flatten()
            if direction.numel() != z.shape[1]:
                skipped += 1
                continue
            direction_norm = torch.linalg.vector_norm(direction)
            if (
                not bool(torch.isfinite(direction_norm).item())
                or float(direction_norm.item()) < float(self.config.tal.min_direction_norm)
            ):
                skipped += 1
                continue
            direction = direction / direction_norm.clamp(min=float(self.config.tal.min_direction_norm))
            z_anchor = z[index].detach()
            z_target = F.normalize(
                z_anchor + float(self.config.tal.alpha) * direction,
                p=2,
                dim=0,
            ).detach()
            cosine = (z[index] * z_target).sum().clamp(min=-1.0, max=1.0)
            losses.append(1.0 - cosine)
            weights.append(float(candidate.weight))
            cosine_sum += float(cosine.detach().cpu().item())

        if not losses:
            return template.new_zeros(()), {"valid": 0, "skipped": skipped, "cosine_sum": 0.0}
        loss_tensor = torch.stack(losses, dim=0)
        weight_tensor = torch.tensor(weights, dtype=loss_tensor.dtype, device=loss_tensor.device)
        loss = (loss_tensor * weight_tensor).sum() / float(max(len(losses), 1))
        return loss, {
            "valid": len(losses),
            "skipped": skipped,
            "cosine_sum": cosine_sum,
        }

    def _record_tal_step(
        self,
        *,
        candidates: Sequence[TCSCandidate],
        loss: torch.Tensor,
        stats: Mapping[str, float | int],
    ) -> None:
        valid = int(stats.get("valid", 0))
        self._tal_stats["candidates"] += len(candidates)
        self._tal_stats["valid_candidates"] += valid
        self._tal_stats["skipped_candidates"] += int(stats.get("skipped", 0))
        if valid > 0:
            self._tal_stats["losses"] += 1
            self._tal_loss_sum += float(loss.detach().cpu().item())
            self._tal_cosine_sum += float(stats.get("cosine_sum", 0.0))
            self._tal_hardness_sum += sum(float(candidate.hardness) for candidate in candidates[:valid])

    @torch.no_grad()
    def _mine_transition_events(
        self,
        *,
        detections: Sequence[Mapping[str, torch.Tensor]],
        original_targets: Sequence[Mapping[str, torch.Tensor]],
        transformed_targets: Sequence[Mapping[str, torch.Tensor]],
        transformed_image_shapes: Sequence[Sequence[int]],
        features: Mapping[str, torch.Tensor],
        epoch: int,
    ) -> list[TransitionEvent]:
        template = _first_feature(features)
        boxes_per_image = [
            target["boxes"].to(device=template.device, dtype=torch.float32).reshape(-1, 4)
            for target in transformed_targets
        ]
        pooled = pool_multiscale_box_features(
            features=features,
            boxes_per_image=boxes_per_image,
            image_shapes=transformed_image_shapes,
            output_size=int(self.config.tal.roi_output_size),
            sampling_ratio=int(self.config.tal.roi_sampling_ratio),
            normalize=False,
        )
        if pooled.numel() == 0:
            embeddings = pooled.new_zeros((0, int(self.config.tal.projector_dim)))
        else:
            if pooled.shape[1] != int(self.config.tal.input_dim):
                raise ValueError(
                    "FN-TDM HTM projector input dimension mismatch: "
                    f"pooled={pooled.shape[1]} config={self.config.tal.input_dim}."
                )
            embeddings = self.projector(pooled)
            if bool(self.config.tal.normalize):
                embeddings = F.normalize(embeddings, p=2, dim=1)
        events: list[TransitionEvent] = []
        offset = 0
        for image_index, (detection, original_target, transformed_target) in enumerate(
            zip(detections, original_targets, transformed_targets, strict=True)
        ):
            gt_count = int(transformed_target["boxes"].shape[0])
            image_embeddings = embeddings[offset : offset + gt_count].detach().cpu()
            offset += gt_count
            events.extend(
                self._mine_image_events(
                    detection=detection,
                    original_target=original_target,
                    transformed_target=transformed_target,
                    transformed_image_shape=transformed_image_shapes[image_index],
                    embeddings=image_embeddings,
                    epoch=epoch,
                    image_index=image_index,
                )
            )
        return events

    @torch.no_grad()
    def _mine_image_events(
        self,
        *,
        detection: Mapping[str, torch.Tensor],
        original_target: Mapping[str, torch.Tensor],
        transformed_target: Mapping[str, torch.Tensor],
        transformed_image_shape: Sequence[int],
        embeddings: torch.Tensor,
        epoch: int,
        image_index: int,
    ) -> list[TransitionEvent]:
        gt_boxes_orig = original_target["boxes"].detach()
        gt_boxes_transformed = transformed_target["boxes"].detach()
        gt_labels = original_target["labels"].to(dtype=torch.int64)
        gt_ids = _extract_gt_ids(original_target, int(gt_boxes_orig.shape[0]))
        image_id = _normalize_image_id(original_target.get("image_id", torch.tensor(image_index)))
        pred_boxes = detection.get("boxes", gt_boxes_orig.new_zeros((0, 4))).detach().to(dtype=torch.float32)
        pred_labels = detection.get("labels", gt_labels.new_zeros((0,))).detach().to(dtype=torch.int64)
        pred_scores = detection.get("scores", pred_boxes.new_zeros((0,))).detach().to(dtype=torch.float32)

        events: list[TransitionEvent] = []
        states = Counter()
        for gt_index in range(int(gt_boxes_orig.shape[0])):
            class_id = int(gt_labels[gt_index].item())
            ann_id = gt_ids[gt_index]
            gt_uid = _gt_uid(
                image_id=image_id,
                class_id=class_id,
                bbox=gt_boxes_orig[gt_index],
                image_shape=None,
                gt_id=ann_id,
            )
            state = _assign_detection_state(
                gt_box=gt_boxes_orig[gt_index],
                gt_label=class_id,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                matching=self.config.htm.matching,
            )
            states[state["state"]] += 1
            if state["state"] == "FN":
                states[str(state["fn_type"])] += 1
            history = self._histories.get(gt_uid)
            if history is None:
                history = HTMHistory(
                    gt_uid=gt_uid,
                    image_id=image_id,
                    ann_id=None if ann_id is None else str(ann_id),
                    class_id=class_id,
                    bbox=gt_boxes_orig[gt_index].detach().cpu().reshape(4),
                )
                self._histories[gt_uid] = history
            event = self._update_history(
                history=history,
                current_state=str(state["state"]),
                fn_type=None if state["fn_type"] is None else str(state["fn_type"]),
                z_current=embeddings[gt_index],
                score=float(state["score"]),
                iou=float(state["iou"]),
                bbox=gt_boxes_orig[gt_index].detach().cpu().reshape(4),
                epoch=epoch,
            )
            if event is not None:
                events.append(event)

        self._htm_stats["gt_seen"] += int(gt_boxes_orig.shape[0])
        self._htm_stats["num_tp"] += int(states.get("TP", 0))
        self._htm_stats["num_fn"] += int(states.get("FN", 0))
        for fn_type in _ALL_FN_TYPES:
            self._htm_stats[f"num_{fn_type.lower()}"] += int(states.get(fn_type, 0))
        return events

    def _update_history(
        self,
        *,
        history: HTMHistory,
        current_state: str,
        fn_type: str | None,
        z_current: torch.Tensor,
        score: float,
        iou: float,
        bbox: torch.Tensor,
        epoch: int,
    ) -> TransitionEvent | None:
        prev_state = history.last_state
        event = None
        z_current = z_current.detach().cpu().flatten()
        if current_state == "TP":
            if (
                prev_state == "FN"
                and history.last_fn_epoch is not None
                and history.last_fn_z is not None
                and history.last_fn_type in set(self.config.htm.transition.allowed_fn_types)
                and int(epoch) - int(history.last_fn_epoch) <= int(self.config.htm.transition.transition_window)
                and int(history.transition_count) < int(self.config.htm.transition.max_transitions_per_gt)
            ):
                raw_direction = z_current - history.last_fn_z
                direction_norm = torch.linalg.vector_norm(raw_direction)
                if bool(torch.isfinite(direction_norm).item()) and float(direction_norm.item()) >= float(self.config.htm.transition.min_direction_norm):
                    direction = raw_direction / direction_norm.clamp(min=float(self.config.htm.transition.min_direction_norm))
                    gap = max(0, int(epoch) - int(history.last_fn_epoch) - 1)
                    quality = (
                        float(score)
                        * (1.0 - float(history.last_fn_score or 0.0))
                        * math.exp(-float(gap) / float(self.config.htm.transition.lambda_gap))
                    )
                    event = TransitionEvent(
                        gt_uid=history.gt_uid,
                        image_id=history.image_id,
                        ann_id=history.ann_id,
                        class_id=history.class_id,
                        bbox=bbox.detach().cpu().reshape(4),
                        epoch_fn=int(history.last_fn_epoch),
                        epoch_tp=int(epoch),
                        fn_type=str(history.last_fn_type),
                        z_fn=history.last_fn_z.detach().cpu(),
                        z_tp=z_current.detach().cpu(),
                        direction=direction.detach().cpu(),
                        score_fn=float(history.last_fn_score or 0.0),
                        score_tp=float(score),
                        iou_fn=float(history.last_fn_iou or 0.0),
                        iou_tp=float(iou),
                        quality=float(max(0.0, quality)),
                    )
                    history.transition_count += 1
                    history.last_emitted_epoch = int(epoch)
            history.last_tp_epoch = int(epoch)
            history.last_tp_z = z_current
            history.last_tp_score = float(score)
            history.last_tp_iou = float(iou)
            history.tp_count += 1
        else:
            history.last_fn_epoch = int(epoch)
            history.last_fn_z = z_current
            history.last_fn_score = float(score)
            history.last_fn_iou = float(iou)
            history.last_fn_type = fn_type
            history.fn_count += 1
        history.bbox = bbox.detach().cpu().reshape(4)
        history.last_state = current_state
        history.last_epoch = int(epoch)
        return event


def load_fntdm_config(path: str | Path, *, arch: str | None = None) -> FNTDMConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"FN-TDM YAML must contain a mapping at the top level: {config_path}")
    return FNTDMConfig.from_mapping(raw, arch=arch)


def build_fntdm_from_config(
    raw_config: Mapping[str, Any] | FNTDMConfig,
    *,
    arch: str | None = None,
) -> FalseNegativeTransitionDirectionMemory | None:
    config = (
        raw_config
        if isinstance(raw_config, FNTDMConfig)
        else FNTDMConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return FalseNegativeTransitionDirectionMemory(config)


def build_fntdm_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> FalseNegativeTransitionDirectionMemory | None:
    config = load_fntdm_config(path, arch=arch)
    if not config.enabled:
        return None
    return FalseNegativeTransitionDirectionMemory(config)


def _assign_detection_state(
    *,
    gt_box: torch.Tensor,
    gt_label: int,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    matching: HTMMatchingConfig,
) -> dict[str, object]:
    if pred_boxes.numel() == 0:
        return {"state": "FN", "fn_type": "FN_MISS", "score": 0.0, "iou": 0.0}
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
            "fn_type": None,
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
        fn_type = "FN_MISS"
    elif best_near_target_score < float(matching.tau_bg_score) and best_near_wrong_score < float(matching.tau_bg_score):
        fn_type = "FN_BG"
    elif best_any_iou >= float(matching.tau_near) and best_near_target_score < float(matching.tau_tp):
        fn_type = "FN_CLS"
    elif best_target_score >= float(matching.tau_loc_score) and best_target_iou < float(matching.tau_iou):
        fn_type = "FN_LOC"
    else:
        fn_type = "FN_BG"

    return {
        "state": "FN",
        "fn_type": fn_type,
        "score": float(max(best_near_target_score, best_target_score)),
        "iou": float(max(best_target_iou, best_any_iou)),
    }


def _confidence_for_gt(
    *,
    gt_index: int,
    class_id: int,
    cls_logits: torch.Tensor,
    ctr_logits: torch.Tensor | None,
    assignments: torch.Tensor,
    pos_mask: torch.Tensor,
    include_centerness: bool,
) -> tuple[float, str]:
    if not bool(pos_mask.any().item()):
        return 0.0, "missing_assignment"
    matched_gt = assignments[pos_mask].to(dtype=torch.long)
    gt_pos_mask = matched_gt == int(gt_index)
    if not bool(gt_pos_mask.any().item()):
        return 0.0, "missing_assignment"
    local_indices = torch.where(pos_mask)[0][gt_pos_mask]
    if class_id < 0 or class_id >= cls_logits.shape[1]:
        return 0.0, "invalid_class"
    cls_scores = torch.sigmoid(cls_logits[local_indices, int(class_id)])
    if include_centerness and ctr_logits is not None:
        scores = cls_scores * torch.sigmoid(ctr_logits[local_indices]).clamp(min=0.0, max=1.0)
    else:
        scores = cls_scores
    confidence = float(scores.max().detach().cpu().item()) if scores.numel() else 0.0
    return confidence, "low_confidence"


def _candidate_boxes_per_image(
    candidates: Sequence[TCSCandidate],
    num_images: int,
    device: torch.device,
) -> list[torch.Tensor]:
    by_image: list[list[torch.Tensor]] = [[] for _ in range(num_images)]
    for candidate in candidates:
        if 0 <= int(candidate.batch_index) < num_images:
            by_image[int(candidate.batch_index)].append(
                candidate.bbox.to(device=device, dtype=torch.float32).reshape(4)
            )
    return [
        torch.stack(boxes, dim=0) if boxes else torch.empty((0, 4), dtype=torch.float32, device=device)
        for boxes in by_image
    ]


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


def _entry_id(gt_uid: str, epoch_fn: int, epoch_tp: int, fn_type: str) -> str:
    digest = hashlib.sha1(f"{gt_uid}:{epoch_fn}:{epoch_tp}:{fn_type}".encode("utf-8")).hexdigest()
    return digest[:20]


def _maybe_store_vector(value: torch.Tensor, enabled: bool, dtype: torch.dtype) -> torch.Tensor | None:
    if not enabled:
        return None
    return torch.as_tensor(value, dtype=dtype).detach().flatten().cpu()


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("FN-TDM image_id tensor must contain one scalar.")
        value = value.detach().cpu().item()
    return str(value)


def _is_replay_target(target: Mapping[str, Any]) -> bool:
    raw = target.get("is_replay", False)
    if isinstance(raw, torch.Tensor):
        if raw.numel() == 0:
            return False
        return bool(raw.detach().flatten()[0].item())
    return bool(raw)


def _first_feature(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    try:
        return next(iter(features.values()))
    except StopIteration as exc:
        raise ValueError("FN-TDM requires at least one feature map.") from exc


def _box_area(box: torch.Tensor) -> float:
    values = box.detach().cpu().flatten()
    if values.numel() != 4:
        return 0.0
    return float(max(0.0, float(values[2] - values[0])) * max(0.0, float(values[3] - values[1])))
