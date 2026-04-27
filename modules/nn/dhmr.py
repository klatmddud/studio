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


_EDGE_NAMES = ("left", "top", "right", "bottom")
_TYPED_FILM_STATES = ("FN_LOC", "FN_CLS", "FN_BG")
_BORDER_REFINEMENT_TRANSITIONS = ("FN_LOC->FN_LOC", "TP->FN_LOC")
_BORDER_GEOMETRY_DIM = 9
_TYPED_FILM_STATE_TO_INDEX = {
    state: index
    for index, state in enumerate(_TYPED_FILM_STATES)
}


@dataclass(frozen=True, slots=True)
class HLRTResidualMemoryConfig:
    enabled: bool = True
    ema_momentum: float = 0.8
    min_edge_observations: int = 1
    max_points_per_gt: int = 4
    max_points_per_batch: int = 128
    target_delta_clip: float = 2.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HLRTResidualMemoryConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", True)),
            ema_momentum=float(data.get("ema_momentum", 0.8)),
            min_edge_observations=int(data.get("min_edge_observations", 1)),
            max_points_per_gt=int(data.get("max_points_per_gt", 4)),
            max_points_per_batch=int(data.get("max_points_per_batch", 128)),
            target_delta_clip=float(data.get("target_delta_clip", 2.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= float(self.ema_momentum) <= 1.0:
            raise ValueError("DHM-R hlrt.residual_memory.ema_momentum must satisfy 0 <= value <= 1.")
        if int(self.min_edge_observations) < 1:
            raise ValueError("DHM-R hlrt.residual_memory.min_edge_observations must be >= 1.")
        for field_name in ("max_points_per_gt", "max_points_per_batch"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"DHM-R hlrt.residual_memory.{field_name} must be >= 0.")
        if float(self.target_delta_clip) < 0.0:
            raise ValueError("DHM-R hlrt.residual_memory.target_delta_clip must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ema_momentum": self.ema_momentum,
            "min_edge_observations": self.min_edge_observations,
            "max_points_per_gt": self.max_points_per_gt,
            "max_points_per_batch": self.max_points_per_batch,
            "target_delta_clip": self.target_delta_clip,
        }


@dataclass(frozen=True, slots=True)
class HLRTResidualReplayConfig:
    enabled: bool = False
    max_points_per_gt: int = 2
    max_points_per_batch: int = 64
    tau_near: float = 0.3
    tau_iou: float = 0.5
    residual_scale: float = 1.0
    max_residual_scale: float = 3.0
    scale_step: float = 1.25
    center_radius_multiplier: float = 1.5
    require_iou_window: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HLRTResidualReplayConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", False)),
            max_points_per_gt=int(data.get("max_points_per_gt", 2)),
            max_points_per_batch=int(data.get("max_points_per_batch", 64)),
            tau_near=float(data.get("tau_near", 0.3)),
            tau_iou=float(data.get("tau_iou", 0.5)),
            residual_scale=float(data.get("residual_scale", 1.0)),
            max_residual_scale=float(data.get("max_residual_scale", 3.0)),
            scale_step=float(data.get("scale_step", 1.25)),
            center_radius_multiplier=float(data.get("center_radius_multiplier", 1.5)),
            require_iou_window=bool(data.get("require_iou_window", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for field_name in ("max_points_per_gt", "max_points_per_batch"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"DHM-R hlrt.residual_replay.{field_name} must be >= 0.")
        for field_name in ("tau_near", "tau_iou"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"DHM-R hlrt.residual_replay.{field_name} must satisfy 0 <= value <= 1.")
        if float(self.tau_near) > float(self.tau_iou):
            raise ValueError("DHM-R hlrt.residual_replay.tau_near must be <= tau_iou.")
        for field_name in ("residual_scale", "max_residual_scale", "scale_step", "center_radius_multiplier"):
            if float(getattr(self, field_name)) <= 0.0:
                raise ValueError(f"DHM-R hlrt.residual_replay.{field_name} must be > 0.")
        if float(self.max_residual_scale) < float(self.residual_scale):
            raise ValueError("DHM-R hlrt.residual_replay.max_residual_scale must be >= residual_scale.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_points_per_gt": self.max_points_per_gt,
            "max_points_per_batch": self.max_points_per_batch,
            "tau_near": self.tau_near,
            "tau_iou": self.tau_iou,
            "residual_scale": self.residual_scale,
            "max_residual_scale": self.max_residual_scale,
            "scale_step": self.scale_step,
            "center_radius_multiplier": self.center_radius_multiplier,
            "require_iou_window": self.require_iou_window,
        }


@dataclass(frozen=True, slots=True)
class HLRTIoULossWeightingConfig:
    enabled: bool = False
    gamma: float = 0.25
    max_weight: float = 2.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HLRTIoULossWeightingConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", False)),
            gamma=float(data.get("gamma", 0.25)),
            max_weight=float(data.get("max_weight", 2.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if float(self.gamma) < 0.0:
            raise ValueError("DHM-R hlrt.iou_loss_weighting.gamma must be >= 0.")
        if float(self.max_weight) < 1.0:
            raise ValueError("DHM-R hlrt.iou_loss_weighting.max_weight must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "gamma": self.gamma,
            "max_weight": self.max_weight,
        }


@dataclass(frozen=True, slots=True)
class HLRTSideAwareLossConfig:
    enabled: bool = False
    loss_weight: float = 0.1
    smooth_l1_beta: float = 0.1
    use_residual_edge_weights: bool = True
    use_dominant_edge_mask: bool = True
    non_dominant_edge_weight: float = 0.5
    min_edge_weight: float = 0.25
    max_edge_weight: float = 2.0
    instability_gamma: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HLRTSideAwareLossConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", False)),
            loss_weight=float(data.get("loss_weight", 0.1)),
            smooth_l1_beta=float(data.get("smooth_l1_beta", 0.1)),
            use_residual_edge_weights=bool(data.get("use_residual_edge_weights", True)),
            use_dominant_edge_mask=bool(data.get("use_dominant_edge_mask", True)),
            non_dominant_edge_weight=float(data.get("non_dominant_edge_weight", 0.5)),
            min_edge_weight=float(data.get("min_edge_weight", 0.25)),
            max_edge_weight=float(data.get("max_edge_weight", 2.0)),
            instability_gamma=float(data.get("instability_gamma", 1.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for field_name in (
            "loss_weight",
            "smooth_l1_beta",
            "non_dominant_edge_weight",
            "min_edge_weight",
            "max_edge_weight",
            "instability_gamma",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"DHM-R hlrt.side_aware_loss.{field_name} must be >= 0.")
        if float(self.max_edge_weight) < float(self.min_edge_weight):
            raise ValueError("DHM-R hlrt.side_aware_loss.max_edge_weight must be >= min_edge_weight.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loss_weight": self.loss_weight,
            "smooth_l1_beta": self.smooth_l1_beta,
            "use_residual_edge_weights": self.use_residual_edge_weights,
            "use_dominant_edge_mask": self.use_dominant_edge_mask,
            "non_dominant_edge_weight": self.non_dominant_edge_weight,
            "min_edge_weight": self.min_edge_weight,
            "max_edge_weight": self.max_edge_weight,
            "instability_gamma": self.instability_gamma,
        }


@dataclass(frozen=True, slots=True)
class HLRTQualityGateConfig:
    enabled: bool = False
    blend: float = 0.5
    min_quality: float = 0.05
    max_quality: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HLRTQualityGateConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", False)),
            blend=float(data.get("blend", 0.5)),
            min_quality=float(data.get("min_quality", 0.05)),
            max_quality=float(data.get("max_quality", 1.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for field_name in ("blend", "min_quality", "max_quality"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"DHM-R hlrt.quality_gate.{field_name} must satisfy 0 <= value <= 1.")
        if float(self.min_quality) > float(self.max_quality):
            raise ValueError("DHM-R hlrt.quality_gate.min_quality must be <= max_quality.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "blend": self.blend,
            "min_quality": self.min_quality,
            "max_quality": self.max_quality,
        }


@dataclass(frozen=True, slots=True)
class HLRTConfig:
    enabled: bool = False
    min_observations: int = 3
    min_fn_loc_count: int = 2
    min_instability: float = 0.25
    max_gt_per_image: int = 16
    start_epoch: int = 1
    warmup_epochs: int = 2
    residual_memory: HLRTResidualMemoryConfig = field(default_factory=HLRTResidualMemoryConfig)
    residual_replay: HLRTResidualReplayConfig = field(default_factory=HLRTResidualReplayConfig)
    iou_loss_weighting: HLRTIoULossWeightingConfig = field(default_factory=HLRTIoULossWeightingConfig)
    side_aware_loss: HLRTSideAwareLossConfig = field(default_factory=HLRTSideAwareLossConfig)
    quality_gate: HLRTQualityGateConfig = field(default_factory=HLRTQualityGateConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HLRTConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", False)),
            min_observations=int(data.get("min_observations", 3)),
            min_fn_loc_count=int(data.get("min_fn_loc_count", 2)),
            min_instability=float(data.get("min_instability", 0.25)),
            max_gt_per_image=int(data.get("max_gt_per_image", 16)),
            start_epoch=int(data.get("start_epoch", 1)),
            warmup_epochs=int(data.get("warmup_epochs", 2)),
            residual_memory=HLRTResidualMemoryConfig.from_mapping(data.get("residual_memory")),
            residual_replay=HLRTResidualReplayConfig.from_mapping(data.get("residual_replay")),
            iou_loss_weighting=HLRTIoULossWeightingConfig.from_mapping(data.get("iou_loss_weighting")),
            side_aware_loss=HLRTSideAwareLossConfig.from_mapping(data.get("side_aware_loss")),
            quality_gate=HLRTQualityGateConfig.from_mapping(data.get("quality_gate")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.min_observations) < 1:
            raise ValueError("DHM-R hlrt.min_observations must be >= 1.")
        if int(self.min_fn_loc_count) < 1:
            raise ValueError("DHM-R hlrt.min_fn_loc_count must be >= 1.")
        if not 0.0 <= float(self.min_instability) <= 1.0:
            raise ValueError("DHM-R hlrt.min_instability must satisfy 0 <= value <= 1.")
        if int(self.max_gt_per_image) < 0:
            raise ValueError("DHM-R hlrt.max_gt_per_image must be >= 0.")
        if int(self.start_epoch) < 0:
            raise ValueError("DHM-R hlrt.start_epoch must be >= 0.")
        if int(self.warmup_epochs) < 0:
            raise ValueError("DHM-R hlrt.warmup_epochs must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_observations": self.min_observations,
            "min_fn_loc_count": self.min_fn_loc_count,
            "min_instability": self.min_instability,
            "max_gt_per_image": self.max_gt_per_image,
            "start_epoch": self.start_epoch,
            "warmup_epochs": self.warmup_epochs,
            "residual_memory": self.residual_memory.to_dict(),
            "residual_replay": self.residual_replay.to_dict(),
            "iou_loss_weighting": self.iou_loss_weighting.to_dict(),
            "side_aware_loss": self.side_aware_loss.to_dict(),
            "quality_gate": self.quality_gate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TypedFiLMConfig:
    enabled: bool = False
    embedding_dim: int = 32
    feature_dim: int = 256
    target_states: tuple[str, ...] = _TYPED_FILM_STATES
    min_observations: int = 3
    min_instability: float = 0.25
    start_epoch: int = 1
    warmup_epochs: int = 2
    scale: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TypedFiLMConfig":
        data = dict(raw or {})
        raw_states = data.get("target_states", _TYPED_FILM_STATES)
        if isinstance(raw_states, str):
            target_states = (raw_states,)
        elif isinstance(raw_states, Sequence):
            target_states = tuple(str(state) for state in raw_states)
        else:
            raise TypeError("DHM-R typed_film.target_states must be a string or sequence.")
        config = cls(
            enabled=bool(data.get("enabled", False)),
            embedding_dim=int(data.get("embedding_dim", 32)),
            feature_dim=int(data.get("feature_dim", 256)),
            target_states=target_states,
            min_observations=int(data.get("min_observations", 3)),
            min_instability=float(data.get("min_instability", 0.25)),
            start_epoch=int(data.get("start_epoch", 1)),
            warmup_epochs=int(data.get("warmup_epochs", 2)),
            scale=float(data.get("scale", 1.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.embedding_dim) < 1:
            raise ValueError("DHM-R typed_film.embedding_dim must be >= 1.")
        if int(self.feature_dim) < 1:
            raise ValueError("DHM-R typed_film.feature_dim must be >= 1.")
        if int(self.min_observations) < 1:
            raise ValueError("DHM-R typed_film.min_observations must be >= 1.")
        if not 0.0 <= float(self.min_instability) <= 1.0:
            raise ValueError("DHM-R typed_film.min_instability must satisfy 0 <= value <= 1.")
        if int(self.start_epoch) < 0:
            raise ValueError("DHM-R typed_film.start_epoch must be >= 0.")
        if int(self.warmup_epochs) < 0:
            raise ValueError("DHM-R typed_film.warmup_epochs must be >= 0.")
        if float(self.scale) < 0.0:
            raise ValueError("DHM-R typed_film.scale must be >= 0.")
        if not self.target_states:
            raise ValueError("DHM-R typed_film.target_states must not be empty.")
        unsupported = sorted(set(self.target_states) - set(_TYPED_FILM_STATES))
        if unsupported:
            raise ValueError(
                "DHM-R typed_film.target_states has unsupported states: "
                f"{unsupported}. Supported states: {list(_TYPED_FILM_STATES)}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "embedding_dim": self.embedding_dim,
            "feature_dim": self.feature_dim,
            "target_states": list(self.target_states),
            "min_observations": self.min_observations,
            "min_instability": self.min_instability,
            "start_epoch": self.start_epoch,
            "warmup_epochs": self.warmup_epochs,
            "scale": self.scale,
        }


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
    hlrt: HLRTConfig = field(default_factory=HLRTConfig)
    typed_film: TypedFiLMConfig = field(default_factory=TypedFiLMConfig)
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
            hlrt=HLRTConfig.from_mapping(merged.get("hlrt")),
            typed_film=TypedFiLMConfig.from_mapping(merged.get("typed_film")),
            border_refinement=BorderRefinementConfig.from_mapping(merged.get("border_refinement")),
            arch=normalized_arch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hlrt": self.hlrt.to_dict(),
            "typed_film": self.typed_film.to_dict(),
            "border_refinement": self.border_refinement.to_dict(),
            "arch": self.arch,
        }


@dataclass(slots=True)
class HLRTResidualRecord:
    gt_uid: str
    class_id: int
    observations: int = 0
    edge_error_ema: torch.Tensor = field(default_factory=lambda: torch.zeros(4, dtype=torch.float32))
    edge_abs_ema: torch.Tensor = field(default_factory=lambda: torch.zeros(4, dtype=torch.float32))
    edge_sq_ema: torch.Tensor = field(default_factory=lambda: torch.zeros(4, dtype=torch.float32))
    last_epoch: int = 0

    @property
    def dominant_edge(self) -> str:
        index = int(torch.argmax(self.edge_abs_ema).item()) if self.edge_abs_ema.numel() else 0
        return _EDGE_NAMES[max(0, min(index, len(_EDGE_NAMES) - 1))]

    @property
    def edge_var_ema(self) -> torch.Tensor:
        return (self.edge_sq_ema - self.edge_error_ema.square()).clamp_min(0.0)

    def update(self, edge_delta: torch.Tensor, *, epoch: int, momentum: float) -> None:
        edge_delta = edge_delta.detach().cpu().to(dtype=torch.float32).reshape(4)
        edge_abs = edge_delta.abs()
        edge_sq = edge_delta.square()
        if self.observations <= 0:
            self.edge_error_ema = edge_delta
            self.edge_abs_ema = edge_abs
            self.edge_sq_ema = edge_sq
        else:
            self.edge_error_ema = momentum * self.edge_error_ema + (1.0 - momentum) * edge_delta
            self.edge_abs_ema = momentum * self.edge_abs_ema + (1.0 - momentum) * edge_abs
            self.edge_sq_ema = momentum * self.edge_sq_ema + (1.0 - momentum) * edge_sq
        self.observations += 1
        self.last_epoch = int(epoch)

    def to_state(self) -> dict[str, Any]:
        return {
            "gt_uid": self.gt_uid,
            "class_id": self.class_id,
            "observations": self.observations,
            "edge_error_ema": self.edge_error_ema.detach().cpu(),
            "edge_abs_ema": self.edge_abs_ema.detach().cpu(),
            "edge_sq_ema": self.edge_sq_ema.detach().cpu(),
            "last_epoch": self.last_epoch,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "HLRTResidualRecord":
        return cls(
            gt_uid=str(state["gt_uid"]),
            class_id=int(state.get("class_id", -1)),
            observations=int(state.get("observations", 0)),
            edge_error_ema=torch.as_tensor(state.get("edge_error_ema", [0, 0, 0, 0]), dtype=torch.float32).reshape(4),
            edge_abs_ema=torch.as_tensor(state.get("edge_abs_ema", [0, 0, 0, 0]), dtype=torch.float32).reshape(4),
            edge_sq_ema=torch.as_tensor(state.get("edge_sq_ema", [0, 0, 0, 0]), dtype=torch.float32).reshape(4),
            last_epoch=int(state.get("last_epoch", 0)),
        )


class DHMRepairModule(nn.Module):
    def __init__(self, config: DHMRConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._residual_records: dict[str, HLRTResidualRecord] = {}
        self._stats: Counter[str] = Counter()
        self._hlrt_side_loss_sum = 0.0
        self._border_loss_sums: Counter[str] = Counter()
        if bool(config.typed_film.enabled):
            typed = config.typed_film
            self.typed_film_embeddings = nn.Embedding(
                len(_TYPED_FILM_STATES),
                int(typed.embedding_dim),
            )
            self.typed_film_projection = nn.Linear(
                int(typed.embedding_dim),
                int(typed.feature_dim) * 2,
            )
            nn.init.zeros_(self.typed_film_projection.weight)
            nn.init.zeros_(self.typed_film_projection.bias)
        else:
            self.typed_film_embeddings = None
            self.typed_film_projection = None
        if bool(config.border_refinement.enabled):
            border = config.border_refinement
            input_dim = int(border.feature_dim) * 5 + _BORDER_GEOMETRY_DIM
            layers: list[nn.Module] = []
            hidden_dim = int(border.hidden_dim)
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
        self._hlrt_side_loss_sum = 0.0
        self._border_loss_sums.clear()

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def hlrt_warmup_factor(self) -> float:
        hlrt = self.config.hlrt
        if not self.config.enabled or not hlrt.enabled:
            return 0.0
        if self.current_epoch < int(hlrt.start_epoch):
            return 0.0
        warmup = int(hlrt.warmup_epochs)
        if warmup <= 0:
            return 1.0
        progress = max(0, self.current_epoch - int(hlrt.start_epoch) + 1)
        return min(1.0, float(progress) / float(warmup))

    def typed_film_warmup_factor(self) -> float:
        typed = self.config.typed_film
        if not self.config.enabled or not typed.enabled:
            return 0.0
        if self.current_epoch < int(typed.start_epoch):
            return 0.0
        warmup = int(typed.warmup_epochs)
        if warmup <= 0:
            return 1.0
        progress = max(0, self.current_epoch - int(typed.start_epoch) + 1)
        return min(1.0, float(progress) / float(warmup))

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

    def uses_typed_film(self) -> bool:
        typed = self.config.typed_film
        return bool(
            self.typed_film_embeddings is not None
            and self.typed_film_projection is not None
            and self.typed_film_warmup_factor() > 0.0
            and float(typed.scale) > 0.0
            and bool(typed.target_states)
        )

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

    def uses_native_loss_hooks(self) -> bool:
        if self.hlrt_warmup_factor() <= 0.0:
            return False
        hlrt = self.config.hlrt
        return bool(
            hlrt.iou_loss_weighting.enabled
            or hlrt.side_aware_loss.enabled
            or hlrt.quality_gate.enabled
        )

    def uses_assignment_replay(self) -> bool:
        if self.hlrt_warmup_factor() <= 0.0:
            return False
        replay = self.config.hlrt.residual_replay
        return bool(
            replay.enabled
            and int(replay.max_points_per_gt) > 0
            and int(replay.max_points_per_batch) > 0
        )

    def uses_residual_memory(self) -> bool:
        hlrt = self.config.hlrt
        memory = hlrt.residual_memory
        return bool(
            self.config.enabled
            and hlrt.enabled
            and memory.enabled
            and int(memory.max_points_per_gt) > 0
            and int(memory.max_points_per_batch) > 0
        )

    def is_hlrt_record_active(self, record: DHMRecord | None) -> bool:
        return bool(
            self.hlrt_warmup_factor() > 0.0
            and record is not None
            and self._is_hlrt_eligible_record(record, count_stats=False)
        )

    def hlrt_iou_weight_for_record(self, record: DHMRecord | None) -> float:
        config = self.config.hlrt.iou_loss_weighting
        if self.hlrt_warmup_factor() <= 0.0 or not bool(config.enabled):
            return 1.0
        if not self.is_hlrt_record_active(record):
            return 1.0
        weight = 1.0 + float(config.gamma) * float(record.instability_score) * self.hlrt_warmup_factor()
        weight = min(float(config.max_weight), max(1.0, weight))
        self._stats["hlrt_iou_weight_hits"] += 1
        return float(weight)

    def hlrt_quality_targets(
        self,
        *,
        record: DHMRecord | None,
        base_targets: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config.hlrt.quality_gate
        if self.hlrt_warmup_factor() <= 0.0 or not bool(config.enabled):
            return base_targets
        if not self.is_hlrt_record_active(record):
            return base_targets
        quality = float(record.ema_iou) if float(record.ema_iou) > 0.0 else float(record.last_iou)
        quality = min(float(config.max_quality), max(float(config.min_quality), quality))
        blend = float(config.blend) * self.hlrt_warmup_factor()
        self._stats["hlrt_quality_gate_hits"] += int(base_targets.numel())
        return base_targets * (1.0 - blend) + base_targets.new_full(base_targets.shape, quality) * blend

    def hlrt_side_weights_for_record(
        self,
        *,
        record: DHMRecord | None,
        current_delta: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config.hlrt.side_aware_loss
        if not self.is_hlrt_record_active(record):
            return torch.ones(4, dtype=torch.float32)
        return self._edge_prior_from_config(
            record=record,
            current_delta=current_delta,
            use_residual_edge_weights=bool(config.use_residual_edge_weights),
            min_edge_observations=int(self.config.hlrt.residual_memory.min_edge_observations),
            use_dominant_edge_mask=bool(config.use_dominant_edge_mask),
            non_dominant_edge_weight=float(config.non_dominant_edge_weight),
            min_edge_weight=float(config.min_edge_weight),
            max_edge_weight=float(config.max_edge_weight),
            stat_key="hlrt_residual_weight_hits",
        )

    def hlrt_side_sample_weight(self, record: DHMRecord | None) -> float:
        config = self.config.hlrt.side_aware_loss
        if not self.is_hlrt_record_active(record):
            return 1.0
        return float(1.0 + float(config.instability_gamma) * float(record.instability_score))

    def hlrt_side_loss_weight(self) -> float:
        config = self.config.hlrt.side_aware_loss
        if self.hlrt_warmup_factor() <= 0.0 or not bool(config.enabled):
            return 0.0
        return float(config.loss_weight) * self.hlrt_warmup_factor()

    def apply_typed_film(
        self,
        *,
        feature_maps: Sequence[torch.Tensor],
        matched_idxs: Sequence[torch.Tensor],
        dhm_records: Sequence[Sequence[DHMRecord | None]],
        num_anchors_per_level: Sequence[int],
    ) -> list[torch.Tensor]:
        if not self.uses_typed_film():
            return list(feature_maps)
        if not feature_maps:
            return list(feature_maps)
        typed = self.config.typed_film
        first = feature_maps[0]
        if first.ndim != 4:
            self._stats["typed_film_skipped_bad_feature_shape"] += 1
            return list(feature_maps)
        batch_size = int(first.shape[0])
        feature_dim = int(first.shape[1])
        if feature_dim != int(typed.feature_dim):
            self._stats["typed_film_skipped_feature_dim"] += 1
            return list(feature_maps)

        flat_chunks: list[torch.Tensor] = []
        shapes: list[tuple[int, int, int, int]] = []
        for feature in feature_maps:
            if feature.ndim != 4:
                self._stats["typed_film_skipped_bad_feature_shape"] += 1
                return list(feature_maps)
            n, c, h, w = (int(dim) for dim in feature.shape)
            if n != batch_size or c != feature_dim:
                self._stats["typed_film_skipped_feature_dim"] += 1
                return list(feature_maps)
            shapes.append((n, c, h, w))
            flat_chunks.append(feature.permute(0, 2, 3, 1).reshape(n, h * w, c))

        expected_points = sum(int(value) for value in num_anchors_per_level)
        flat_features = torch.cat(flat_chunks, dim=1)
        if int(flat_features.shape[1]) != expected_points:
            self._stats["typed_film_skipped_anchor_mismatch"] += 1
            return list(feature_maps)

        image_indices: list[torch.Tensor] = []
        point_indices: list[torch.Tensor] = []
        state_indices: list[torch.Tensor] = []
        target_states = set(str(state) for state in typed.target_states)
        for image_index, assignments in enumerate(matched_idxs):
            if image_index >= batch_size:
                break
            if assignments.numel() == 0:
                continue
            if int(assignments.numel()) != int(flat_features.shape[1]):
                self._stats["typed_film_skipped_assignment_mismatch"] += 1
                continue
            records = dhm_records[image_index] if image_index < len(dhm_records) else []
            for gt_index, record in enumerate(records):
                if not self._is_typed_film_record_active(
                    record,
                    target_states=target_states,
                ):
                    continue
                indices = torch.where(assignments == int(gt_index))[0]
                if indices.numel() == 0:
                    self._stats["typed_film_skipped_no_positive_points"] += 1
                    continue
                state = str(record.last_state)
                state_index = _TYPED_FILM_STATE_TO_INDEX[state]
                image_indices.append(torch.full_like(indices, int(image_index)))
                point_indices.append(indices)
                state_indices.append(torch.full_like(indices, int(state_index)))
                self._stats["typed_film_selected_gt"] += 1
                self._stats["typed_film_selected_points"] += int(indices.numel())
                self._stats[f"typed_film_state_{state}"] += 1

        if not point_indices:
            return list(feature_maps)

        selected_images = torch.cat(image_indices, dim=0).to(device=flat_features.device, dtype=torch.long)
        selected_points = torch.cat(point_indices, dim=0).to(device=flat_features.device, dtype=torch.long)
        selected_states = torch.cat(state_indices, dim=0).to(device=flat_features.device, dtype=torch.long)
        embeddings = self.typed_film_embeddings(selected_states)
        gamma_beta = self.typed_film_projection(embeddings).to(dtype=flat_features.dtype)
        gamma_delta, beta = gamma_beta.chunk(2, dim=1)
        factor = flat_features.new_tensor(float(typed.scale) * self.typed_film_warmup_factor())
        selected_features = flat_features[selected_images, selected_points]
        modulated = selected_features * (1.0 + factor * torch.tanh(gamma_delta)) + factor * beta

        updated_flat = flat_features.clone()
        updated_flat[selected_images, selected_points] = modulated

        result: list[torch.Tensor] = []
        offset = 0
        for n, c, h, w in shapes:
            count = h * w
            chunk = updated_flat[:, offset : offset + count, :]
            result.append(chunk.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous())
            offset += count
        return result

    @torch.no_grad()
    def update_hlrt_residual_memory(
        self,
        *,
        targets: Sequence[Mapping[str, torch.Tensor]],
        head_outputs: Mapping[str, torch.Tensor],
        anchors: Sequence[torch.Tensor],
        matched_idxs: Sequence[torch.Tensor],
        dhm_records: Sequence[Sequence[DHMRecord | None]],
        decode_boxes: Callable[..., torch.Tensor],
    ) -> None:
        if not self.uses_residual_memory():
            return
        memory = self.config.hlrt.residual_memory
        selected_total = 0
        for image_index, target in enumerate(targets):
            self._stats["hlrt_memory_gt_seen"] += int(target["boxes"].shape[0])
            if _is_replay_target(target):
                self._stats["hlrt_memory_skipped_replay_target"] += 1
                continue
            assignments = matched_idxs[image_index]
            if assignments.numel() == 0:
                continue
            boxes = target["boxes"].to(dtype=torch.float32)
            records = dhm_records[image_index] if image_index < len(dhm_records) else []
            eligible = self._eligible_hlrt_records(records)
            self._stats["hlrt_memory_eligible_gt"] += len(eligible)
            if not eligible:
                continue

            bbox_regression = head_outputs["bbox_regression"][image_index]
            anchors_per_image = anchors[image_index]
            for gt_index, record in eligible:
                if selected_total >= int(memory.max_points_per_batch):
                    break
                pos_indices = torch.where(assignments == int(gt_index))[0]
                if pos_indices.numel() == 0:
                    self._stats["hlrt_memory_skipped_no_positive_points"] += 1
                    continue
                pred_boxes = decode_boxes(
                    box_regression=bbox_regression[pos_indices],
                    anchors=anchors_per_image[pos_indices],
                ).detach()
                gt_box = boxes[int(gt_index)].to(device=pred_boxes.device, dtype=pred_boxes.dtype).reshape(1, 4)
                target_delta = _normalized_edge_delta(
                    pred_boxes=pred_boxes,
                    gt_box=gt_box,
                    clip=float(memory.target_delta_clip),
                )
                finite_mask = torch.isfinite(target_delta).all(dim=1)
                if not bool(finite_mask.any().item()):
                    self._stats["hlrt_memory_skipped_invalid_delta"] += 1
                    continue
                target_delta = target_delta[finite_mask]
                edge_magnitude = target_delta.abs().sum(dim=1)
                order = torch.argsort(edge_magnitude, descending=True)
                remaining = int(memory.max_points_per_batch) - selected_total
                order = order[: min(int(memory.max_points_per_gt), remaining)]
                if order.numel() == 0:
                    continue
                selected_delta = target_delta[order]
                self._record_residual_updates([(record, selected_delta.detach().mean(dim=0))])
                selected_total += int(selected_delta.shape[0])
                self._stats["hlrt_memory_selected_gt"] += 1
                self._stats["hlrt_memory_selected_points"] += int(selected_delta.shape[0])
            if selected_total >= int(memory.max_points_per_batch):
                break

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
        level_ids = _level_ids(
            num_anchors_per_level,
            device=first_feature.device,
        )
        for image_index, target in enumerate(targets):
            if selected_points >= max_points:
                break
            if _is_replay_target(target):
                self._stats["border_refine_skipped_replay_target"] += 1
                continue
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
                pred_boxes = _clip_boxes(
                    _sanitize_boxes(pred_boxes),
                    padded_shape=padded_shape,
                )
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
        refined_boxes = _apply_box_delta(
            boxes=boxes,
            delta=delta,
            padded_shape=padded_shape,
        )
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
        side_losses = int(self._stats.get("hlrt_side_losses", 0))
        border_losses = int(self._stats.get("border_refine_losses", 0))
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "hlrt_warmup_factor": self.hlrt_warmup_factor(),
            "num_residual_records": len(self._residual_records),
            "hlrt": {
                "enabled": bool(self.config.hlrt.enabled),
                "native_loss_hooks": self.uses_native_loss_hooks(),
                "assignment_replay": self.uses_assignment_replay(),
                "residual_memory": self.uses_residual_memory(),
                "mean_side_loss": self._hlrt_side_loss_sum / float(max(side_losses, 1)),
                **{
                    key: int(value)
                    for key, value in self._stats.items()
                    if str(key).startswith("hlrt_")
                },
            },
            "typed_film": {
                "enabled": bool(self.config.typed_film.enabled),
                "warmup_factor": self.typed_film_warmup_factor(),
                "selected_points": int(self._stats.get("typed_film_selected_points", 0)),
                "selected_gt": int(self._stats.get("typed_film_selected_gt", 0)),
                "state_counts": {
                    state: int(self._stats.get(f"typed_film_state_{state}", 0))
                    for state in _TYPED_FILM_STATES
                },
                **{
                    key: int(value)
                    for key, value in self._stats.items()
                    if str(key).startswith("typed_film_skipped_")
                },
            },
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
            "version": 2,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "residual_records": {
                gt_uid: record.to_state()
                for gt_uid, record in self._residual_records.items()
            },
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            self._residual_records.clear()
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))
        records: dict[str, HLRTResidualRecord] = {}
        raw_records = state.get("residual_records", {})
        if isinstance(raw_records, Mapping):
            for gt_uid, raw_record in raw_records.items():
                if isinstance(raw_record, Mapping):
                    records[str(gt_uid)] = HLRTResidualRecord.from_state(raw_record)
        self._residual_records = records

    @torch.no_grad()
    def record_hlrt_side_loss(
        self,
        *,
        loss: torch.Tensor,
        selected_points: int,
    ) -> None:
        self._stats["hlrt_side_losses"] += 1
        self._stats["hlrt_side_points"] += int(selected_points)
        self._hlrt_side_loss_sum += float(loss.detach().item())

    def apply_hlrt_assignment_replay(
        self,
        *,
        target: Mapping[str, torch.Tensor],
        matched_idx: torch.Tensor,
        anchor_centers: torch.Tensor,
        anchor_sizes: torch.Tensor,
        num_anchors_per_level: Sequence[int],
        dhm_records: Sequence[DHMRecord | None],
    ) -> torch.Tensor:
        if not self.uses_assignment_replay():
            return matched_idx
        replay = self.config.hlrt.residual_replay
        if int(replay.max_points_per_gt) <= 0 or int(replay.max_points_per_batch) <= 0:
            return matched_idx

        gt_boxes = target["boxes"].to(device=matched_idx.device, dtype=anchor_centers.dtype)
        if gt_boxes.numel() == 0:
            return matched_idx

        eligible = self._eligible_hlrt_records(dhm_records)
        if not eligible:
            return matched_idx

        x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
        inside_gt = pairwise_dist.min(dim=2).values > 0

        lower_bound = anchor_sizes * 4
        if len(num_anchors_per_level) > 0 and int(num_anchors_per_level[0]) > 0:
            lower_bound[: int(num_anchors_per_level[0])] = 0
        upper_bound = anchor_sizes * 8
        if len(num_anchors_per_level) > 0 and int(num_anchors_per_level[-1]) > 0:
            upper_bound[-int(num_anchors_per_level[-1]) :] = float("inf")
        max_box_dist = pairwise_dist.max(dim=2).values
        scale_match = (max_box_dist > lower_bound[:, None]) & (max_box_dist < upper_bound[:, None])

        updated = matched_idx.clone()
        remaining = int(replay.max_points_per_batch)
        for gt_index, record in eligible:
            if remaining <= 0:
                break
            residual_record = self._residual_records.get(record.gt_uid)
            if residual_record is None or int(residual_record.observations) < int(self.config.hlrt.residual_memory.min_edge_observations):
                self._stats["hlrt_replay_skipped_no_residual"] += 1
                continue
            residual = residual_record.edge_error_ema.to(device=gt_boxes.device, dtype=gt_boxes.dtype)
            replay_box, replay_iou = _make_replay_box(
                gt_box=gt_boxes[int(gt_index)],
                residual=residual,
                residual_scale=float(replay.residual_scale),
                max_residual_scale=float(replay.max_residual_scale),
                scale_step=float(replay.scale_step),
                tau_iou=float(replay.tau_iou),
            )
            if bool(replay.require_iou_window):
                if not (float(replay.tau_near) <= float(replay_iou.item()) < float(replay.tau_iou)):
                    self._stats["hlrt_replay_skipped_iou_window"] += 1
                    continue

            replay_center = (replay_box[:2] + replay_box[2:]) * 0.5
            replay_dist = (anchor_centers - replay_center.reshape(1, 2)).abs().max(dim=1).values
            rx0, ry0, rx1, ry1 = replay_box.reshape(1, 4).unbind(dim=1)
            inside_replay = (
                (anchor_centers[:, 0] > rx0)
                & (anchor_centers[:, 0] < rx1)
                & (anchor_centers[:, 1] > ry0)
                & (anchor_centers[:, 1] < ry1)
            )
            radius_match = replay_dist < float(replay.center_radius_multiplier) * anchor_sizes
            candidate_mask = (
                (updated < 0)
                & inside_gt[:, int(gt_index)]
                & scale_match[:, int(gt_index)]
                & (inside_replay | radius_match)
            )
            if not bool(candidate_mask.any().item()):
                self._stats["hlrt_replay_skipped_no_candidates"] += 1
                continue

            candidate_indices = torch.where(candidate_mask)[0]
            candidate_dist = replay_dist[candidate_indices]
            add_count = min(int(replay.max_points_per_gt), remaining, int(candidate_indices.numel()))
            if add_count <= 0:
                continue
            selected_order = torch.argsort(candidate_dist)[:add_count]
            selected_indices = candidate_indices[selected_order]
            updated[selected_indices] = int(gt_index)
            remaining -= add_count
            self._stats["hlrt_replay_gt"] += 1
            self._stats["hlrt_replay_points"] += int(selected_indices.numel())
        return updated

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

    def _eligible_hlrt_records(
        self,
        records: Sequence[DHMRecord | None],
    ) -> list[tuple[int, DHMRecord]]:
        eligible: list[tuple[int, DHMRecord]] = []
        for gt_index, record in enumerate(records):
            if record is None:
                continue
            if self._is_hlrt_eligible_record(record, count_stats=False):
                eligible.append((gt_index, record))
        eligible.sort(
            key=lambda item: (
                float(item[1].instability_score),
                float(item[1].state_counts.get("FN_LOC", 0)),
                float(item[1].consecutive_fn),
            ),
            reverse=True,
        )
        max_gt = int(self.config.hlrt.max_gt_per_image)
        if max_gt > 0:
            eligible = eligible[:max_gt]
        return eligible

    def _is_hlrt_eligible_record(self, record: DHMRecord, *, count_stats: bool = True) -> bool:
        hlrt = self.config.hlrt
        if str(record.last_state) != "FN_LOC":
            if count_stats:
                self._stats["hlrt_skipped_not_fn_loc"] += 1
            return False
        if int(record.total_seen) < int(hlrt.min_observations):
            if count_stats:
                self._stats["hlrt_skipped_low_observations"] += 1
            return False
        if int(record.state_counts.get("FN_LOC", 0)) < int(hlrt.min_fn_loc_count):
            if count_stats:
                self._stats["hlrt_skipped_low_fn_loc_count"] += 1
            return False
        if float(record.instability_score) < float(hlrt.min_instability):
            if count_stats:
                self._stats["hlrt_skipped_low_instability"] += 1
            return False
        return True

    def _is_typed_film_record_active(
        self,
        record: DHMRecord | None,
        *,
        target_states: set[str],
    ) -> bool:
        typed = self.config.typed_film
        if record is None:
            return False
        if str(record.last_state) not in target_states:
            return False
        if int(record.total_seen) < int(typed.min_observations):
            self._stats["typed_film_skipped_low_observations"] += 1
            return False
        if float(record.instability_score) < float(typed.min_instability):
            self._stats["typed_film_skipped_low_instability"] += 1
            return False
        return True

    def _edge_prior_from_config(
        self,
        *,
        record: DHMRecord,
        current_delta: torch.Tensor,
        use_residual_edge_weights: bool,
        min_edge_observations: int,
        use_dominant_edge_mask: bool,
        non_dominant_edge_weight: float,
        min_edge_weight: float,
        max_edge_weight: float,
        stat_key: str,
    ) -> torch.Tensor:
        residual_record = self._residual_records.get(record.gt_uid)
        if (
            bool(use_residual_edge_weights)
            and residual_record is not None
            and int(residual_record.observations) >= int(min_edge_observations)
        ):
            abs_prior = residual_record.edge_abs_ema.detach().clone()
            self._stats[stat_key] += 1
        else:
            abs_prior = current_delta.detach().abs().mean(dim=0).cpu()
        if bool(use_dominant_edge_mask):
            weights = torch.full_like(abs_prior, float(non_dominant_edge_weight))
            if bool((abs_prior > 0).any().item()):
                weights[int(torch.argmax(abs_prior).item())] = 1.0
        else:
            mean = abs_prior.mean().clamp_min(1.0e-6)
            weights = abs_prior / mean
        return weights.clamp(min=float(min_edge_weight), max=float(max_edge_weight))

    @torch.no_grad()
    def _record_residual_updates(
        self,
        update_items: Sequence[tuple[DHMRecord, torch.Tensor]],
    ) -> None:
        if not update_items:
            return
        for record, delta in update_items:
            residual_record = self._residual_records.get(record.gt_uid)
            if residual_record is None:
                residual_record = HLRTResidualRecord(
                    gt_uid=record.gt_uid,
                    class_id=int(record.class_id),
                )
                self._residual_records[record.gt_uid] = residual_record
            residual_record.update(
                delta,
                epoch=self.current_epoch,
                momentum=float(self.config.hlrt.residual_memory.ema_momentum),
            )
            self._stats["hlrt_memory_updates"] += 1


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
    if width <= 1.0:
        grid_x = torch.zeros_like(x)
    else:
        grid_x = x / (width - 1.0) * 2.0 - 1.0
    if height <= 1.0:
        grid_y = torch.zeros_like(y)
    else:
        grid_y = y / (height - 1.0) * 2.0 - 1.0
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


def _normalized_edge_delta(
    *,
    pred_boxes: torch.Tensor,
    gt_box: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    gt_box = gt_box.to(device=pred_boxes.device, dtype=pred_boxes.dtype)
    width = (gt_box[:, 2] - gt_box[:, 0]).clamp_min(1.0)
    height = (gt_box[:, 3] - gt_box[:, 1]).clamp_min(1.0)
    scale = torch.stack((width, height, width, height), dim=1)
    delta = (gt_box.expand_as(pred_boxes) - pred_boxes) / scale
    if clip > 0.0:
        delta = delta.clamp(min=-float(clip), max=float(clip))
    return delta.detach()


def _make_replay_box(
    *,
    gt_box: torch.Tensor,
    residual: torch.Tensor,
    residual_scale: float,
    max_residual_scale: float,
    scale_step: float,
    tau_iou: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_value = float(residual_scale)
    replay_box = _box_from_residual(gt_box=gt_box, residual=residual, residual_scale=scale_value)
    replay_iou = _aligned_box_iou(replay_box.reshape(1, 4), gt_box.reshape(1, 4))[0]
    while float(replay_iou.item()) >= float(tau_iou) and scale_value < float(max_residual_scale):
        scale_value = min(float(max_residual_scale), scale_value * float(scale_step))
        replay_box = _box_from_residual(gt_box=gt_box, residual=residual, residual_scale=scale_value)
        replay_iou = _aligned_box_iou(replay_box.reshape(1, 4), gt_box.reshape(1, 4))[0]
    return replay_box, replay_iou


def _box_from_residual(
    *,
    gt_box: torch.Tensor,
    residual: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    width = (gt_box[2] - gt_box[0]).clamp_min(1.0)
    height = (gt_box[3] - gt_box[1]).clamp_min(1.0)
    scale = torch.stack((width, height, width, height))
    raw = gt_box - residual.to(device=gt_box.device, dtype=gt_box.dtype).reshape(4) * scale * float(residual_scale)
    x1 = torch.minimum(raw[0], raw[2] - 1.0)
    y1 = torch.minimum(raw[1], raw[3] - 1.0)
    x2 = torch.maximum(raw[2], x1 + 1.0)
    y2 = torch.maximum(raw[3], y1 + 1.0)
    return torch.stack((x1, y1, x2, y2))


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


def _is_replay_target(target: Mapping[str, object]) -> bool:
    raw = target.get("is_replay", False)
    if isinstance(raw, torch.Tensor):
        if raw.numel() == 0:
            return False
        return bool(raw.detach().flatten()[0].item())
    return bool(raw)


DHMR = DHMRepairModule
