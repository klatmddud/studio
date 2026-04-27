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

from .dhm import DHMRecord
from .common import normalize_arch


_EDGE_NAMES = ("left", "top", "right", "bottom")


@dataclass(frozen=True, slots=True)
class TemporalEdgeRepairConfig:
    enabled: bool = True
    input_dim: int = 256
    hidden_dim: int = 128
    min_observations: int = 3
    min_fn_loc_count: int = 2
    min_instability: float = 0.25
    max_gt_per_image: int = 16
    max_points_per_gt: int = 4
    max_points_per_batch: int = 128
    target_delta_clip: float = 2.0
    smooth_l1_beta: float = 0.1
    edge_ema_momentum: float = 0.8
    min_edge_observations: int = 1
    use_temporal_edge_weights: bool = True
    use_dominant_edge_mask: bool = True
    non_dominant_edge_weight: float = 0.25
    min_edge_weight: float = 0.25
    max_edge_weight: float = 2.0
    consistency_weight: float = 0.1
    direction_weight: float = 0.05
    direction_min_abs_delta: float = 0.02
    direction_margin: float = 0.0
    loss_weight: float = 0.25
    start_epoch: int = 1
    warmup_epochs: int = 2

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TemporalEdgeRepairConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", True)),
            input_dim=int(data.get("input_dim", 256)),
            hidden_dim=int(data.get("hidden_dim", 128)),
            min_observations=int(data.get("min_observations", 3)),
            min_fn_loc_count=int(data.get("min_fn_loc_count", 2)),
            min_instability=float(data.get("min_instability", 0.25)),
            max_gt_per_image=int(data.get("max_gt_per_image", 16)),
            max_points_per_gt=int(data.get("max_points_per_gt", 4)),
            max_points_per_batch=int(data.get("max_points_per_batch", 128)),
            target_delta_clip=float(data.get("target_delta_clip", 2.0)),
            smooth_l1_beta=float(data.get("smooth_l1_beta", 0.1)),
            edge_ema_momentum=float(data.get("edge_ema_momentum", 0.8)),
            min_edge_observations=int(data.get("min_edge_observations", 1)),
            use_temporal_edge_weights=bool(data.get("use_temporal_edge_weights", True)),
            use_dominant_edge_mask=bool(data.get("use_dominant_edge_mask", True)),
            non_dominant_edge_weight=float(data.get("non_dominant_edge_weight", 0.25)),
            min_edge_weight=float(data.get("min_edge_weight", 0.25)),
            max_edge_weight=float(data.get("max_edge_weight", 2.0)),
            consistency_weight=float(data.get("consistency_weight", 0.1)),
            direction_weight=float(data.get("direction_weight", 0.05)),
            direction_min_abs_delta=float(data.get("direction_min_abs_delta", 0.02)),
            direction_margin=float(data.get("direction_margin", 0.0)),
            loss_weight=float(data.get("loss_weight", 0.25)),
            start_epoch=int(data.get("start_epoch", 1)),
            warmup_epochs=int(data.get("warmup_epochs", 2)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for field_name in ("input_dim", "hidden_dim", "min_observations", "min_edge_observations"):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"DHM-R temporal_edge_repair.{field_name} must be >= 1.")
        for field_name in ("max_gt_per_image", "max_points_per_gt", "max_points_per_batch"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"DHM-R temporal_edge_repair.{field_name} must be >= 0.")
        for field_name in (
            "min_instability",
            "edge_ema_momentum",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"DHM-R temporal_edge_repair.{field_name} must satisfy 0 <= value <= 1.")
        for field_name in (
            "target_delta_clip",
            "smooth_l1_beta",
            "non_dominant_edge_weight",
            "min_edge_weight",
            "max_edge_weight",
            "consistency_weight",
            "direction_weight",
            "direction_min_abs_delta",
            "loss_weight",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"DHM-R temporal_edge_repair.{field_name} must be >= 0.")
        if self.max_edge_weight < self.min_edge_weight:
            raise ValueError("DHM-R temporal_edge_repair.max_edge_weight must be >= min_edge_weight.")
        if self.start_epoch < 0:
            raise ValueError("DHM-R temporal_edge_repair.start_epoch must be >= 0.")
        if self.warmup_epochs < 0:
            raise ValueError("DHM-R temporal_edge_repair.warmup_epochs must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "min_observations": self.min_observations,
            "min_fn_loc_count": self.min_fn_loc_count,
            "min_instability": self.min_instability,
            "max_gt_per_image": self.max_gt_per_image,
            "max_points_per_gt": self.max_points_per_gt,
            "max_points_per_batch": self.max_points_per_batch,
            "target_delta_clip": self.target_delta_clip,
            "smooth_l1_beta": self.smooth_l1_beta,
            "edge_ema_momentum": self.edge_ema_momentum,
            "min_edge_observations": self.min_edge_observations,
            "use_temporal_edge_weights": self.use_temporal_edge_weights,
            "use_dominant_edge_mask": self.use_dominant_edge_mask,
            "non_dominant_edge_weight": self.non_dominant_edge_weight,
            "min_edge_weight": self.min_edge_weight,
            "max_edge_weight": self.max_edge_weight,
            "consistency_weight": self.consistency_weight,
            "direction_weight": self.direction_weight,
            "direction_min_abs_delta": self.direction_min_abs_delta,
            "direction_margin": self.direction_margin,
            "loss_weight": self.loss_weight,
            "start_epoch": self.start_epoch,
            "warmup_epochs": self.warmup_epochs,
        }


@dataclass(frozen=True, slots=True)
class DHMRConfig:
    enabled: bool = False
    temporal_edge_repair: TemporalEdgeRepairConfig = field(default_factory=TemporalEdgeRepairConfig)
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
            temporal_edge_repair=TemporalEdgeRepairConfig.from_mapping(
                merged.get("temporal_edge_repair")
            ),
            arch=normalized_arch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "temporal_edge_repair": self.temporal_edge_repair.to_dict(),
            "arch": self.arch,
        }


@dataclass(slots=True)
class TemporalEdgeRecord:
    gt_uid: str
    class_id: int
    observations: int = 0
    edge_error_ema: torch.Tensor = field(default_factory=lambda: torch.zeros(4, dtype=torch.float32))
    edge_abs_ema: torch.Tensor = field(default_factory=lambda: torch.zeros(4, dtype=torch.float32))
    last_epoch: int = 0

    @property
    def dominant_edge(self) -> str:
        index = int(torch.argmax(self.edge_abs_ema).item()) if self.edge_abs_ema.numel() else 0
        return _EDGE_NAMES[max(0, min(index, len(_EDGE_NAMES) - 1))]

    def update(self, edge_delta: torch.Tensor, *, epoch: int, momentum: float) -> None:
        edge_delta = edge_delta.detach().cpu().to(dtype=torch.float32).reshape(4)
        edge_abs = edge_delta.abs()
        if self.observations <= 0:
            self.edge_error_ema = edge_delta
            self.edge_abs_ema = edge_abs
        else:
            self.edge_error_ema = momentum * self.edge_error_ema + (1.0 - momentum) * edge_delta
            self.edge_abs_ema = momentum * self.edge_abs_ema + (1.0 - momentum) * edge_abs
        self.observations += 1
        self.last_epoch = int(epoch)

    def to_state(self) -> dict[str, Any]:
        return {
            "gt_uid": self.gt_uid,
            "class_id": self.class_id,
            "observations": self.observations,
            "edge_error_ema": self.edge_error_ema.detach().cpu(),
            "edge_abs_ema": self.edge_abs_ema.detach().cpu(),
            "last_epoch": self.last_epoch,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TemporalEdgeRecord":
        return cls(
            gt_uid=str(state["gt_uid"]),
            class_id=int(state.get("class_id", -1)),
            observations=int(state.get("observations", 0)),
            edge_error_ema=torch.as_tensor(state.get("edge_error_ema", [0, 0, 0, 0]), dtype=torch.float32).reshape(4),
            edge_abs_ema=torch.as_tensor(state.get("edge_abs_ema", [0, 0, 0, 0]), dtype=torch.float32).reshape(4),
            last_epoch=int(state.get("last_epoch", 0)),
        )


class DHMRepairModule(nn.Module):
    def __init__(self, config: DHMRConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        edge_config = config.temporal_edge_repair
        self.edge_head = nn.Sequential(
            nn.Linear(int(edge_config.input_dim), int(edge_config.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(edge_config.hidden_dim), 4),
        )
        self._edge_records: dict[str, TemporalEdgeRecord] = {}
        self._stats: Counter[str] = Counter()
        self._loss_sum = 0.0
        self._edge_loss_sum = 0.0
        self._consistency_loss_sum = 0.0
        self._direction_loss_sum = 0.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.edge_head(features)

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._stats.clear()
        self._loss_sum = 0.0
        self._edge_loss_sum = 0.0
        self._consistency_loss_sum = 0.0
        self._direction_loss_sum = 0.0

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def loss_weight(self) -> float:
        edge_config = self.config.temporal_edge_repair
        if not self.config.enabled or not edge_config.enabled:
            return 0.0
        if self.current_epoch < int(edge_config.start_epoch):
            return 0.0
        warmup = int(edge_config.warmup_epochs)
        if warmup <= 0:
            factor = 1.0
        else:
            progress = max(0, self.current_epoch - int(edge_config.start_epoch) + 1)
            factor = min(1.0, float(progress) / float(warmup))
        return float(edge_config.loss_weight) * factor

    def compute_loss(
        self,
        *,
        targets: Sequence[Mapping[str, torch.Tensor]],
        flat_features: torch.Tensor,
        head_outputs: Mapping[str, torch.Tensor],
        anchors: Sequence[torch.Tensor],
        matched_idxs: Sequence[torch.Tensor],
        dhm_records: Sequence[Sequence[DHMRecord | None]],
        decode_boxes: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        zero = flat_features.new_zeros(())
        edge_config = self.config.temporal_edge_repair
        weight = self.loss_weight()
        if weight <= 0.0:
            return zero
        if int(flat_features.shape[-1]) != int(edge_config.input_dim):
            self._stats["skipped_feature_dim"] += 1
            return zero
        if int(edge_config.max_points_per_batch) <= 0 or int(edge_config.max_points_per_gt) <= 0:
            return zero

        features: list[torch.Tensor] = []
        target_deltas: list[torch.Tensor] = []
        edge_weights: list[torch.Tensor] = []
        pred_boxes_for_loss: list[torch.Tensor] = []
        gt_boxes_for_loss: list[torch.Tensor] = []
        update_items: list[tuple[DHMRecord, torch.Tensor]] = []

        for image_index, target in enumerate(targets):
            self._stats["gt_seen"] += int(target["boxes"].shape[0])
            if _is_replay_target(target):
                self._stats["skipped_replay"] += 1
                continue
            assignments = matched_idxs[image_index]
            if assignments.numel() == 0:
                continue
            boxes = target["boxes"].to(dtype=torch.float32)
            records = dhm_records[image_index] if image_index < len(dhm_records) else []
            eligible: list[tuple[int, DHMRecord]] = []
            for gt_index, record in enumerate(records):
                if record is None:
                    self._stats["skipped_no_dhm_record"] += 1
                    continue
                if not self._is_eligible_record(record):
                    continue
                eligible.append((gt_index, record))
            eligible.sort(
                key=lambda item: (
                    float(item[1].instability_score),
                    float(item[1].state_counts.get("FN_LOC", 0)),
                    float(item[1].consecutive_fn),
                ),
                reverse=True,
            )
            if edge_config.max_gt_per_image > 0:
                eligible = eligible[: int(edge_config.max_gt_per_image)]
            self._stats["eligible_gt"] += len(eligible)

            bbox_regression = head_outputs["bbox_regression"][image_index]
            anchors_per_image = anchors[image_index]
            image_features = flat_features[image_index]
            for gt_index, record in eligible:
                pos_indices = torch.where(assignments == int(gt_index))[0]
                if pos_indices.numel() == 0:
                    self._stats["skipped_no_positive_points"] += 1
                    continue
                pred_boxes = decode_boxes(
                    box_regression=bbox_regression[pos_indices],
                    anchors=anchors_per_image[pos_indices],
                ).detach()
                gt_box = boxes[int(gt_index)].to(device=pred_boxes.device, dtype=pred_boxes.dtype).reshape(1, 4)
                target_delta = _normalized_edge_delta(
                    pred_boxes=pred_boxes,
                    gt_box=gt_box,
                    clip=float(edge_config.target_delta_clip),
                )
                finite_mask = torch.isfinite(target_delta).all(dim=1)
                if not bool(finite_mask.any().item()):
                    self._stats["skipped_invalid_delta"] += 1
                    continue
                pos_indices = pos_indices[finite_mask]
                pred_boxes = pred_boxes[finite_mask]
                target_delta = target_delta[finite_mask]
                edge_magnitude = target_delta.abs().sum(dim=1)
                order = torch.argsort(edge_magnitude, descending=True)
                order = order[: int(edge_config.max_points_per_gt)]
                selected_pos = pos_indices[order]
                selected_delta = target_delta[order]
                selected_pred_boxes = pred_boxes[order]
                gt_boxes_expanded = gt_box.expand_as(selected_pred_boxes)
                prior = self._edge_prior(record, selected_delta)
                if selected_pos.numel() == 0:
                    continue
                features.append(image_features[selected_pos])
                target_deltas.append(selected_delta)
                edge_weights.append(prior.to(device=selected_delta.device, dtype=selected_delta.dtype).expand_as(selected_delta))
                pred_boxes_for_loss.append(selected_pred_boxes)
                gt_boxes_for_loss.append(gt_boxes_expanded)
                update_items.append((record, selected_delta.detach().mean(dim=0)))
                self._stats["selected_gt"] += 1
                self._stats["selected_points"] += int(selected_pos.numel())

                if sum(int(chunk.shape[0]) for chunk in features) >= int(edge_config.max_points_per_batch):
                    break
            if sum(int(chunk.shape[0]) for chunk in features) >= int(edge_config.max_points_per_batch):
                break

        if not features:
            return zero

        selected_features = torch.cat(features, dim=0)[: int(edge_config.max_points_per_batch)]
        selected_targets = torch.cat(target_deltas, dim=0)[: selected_features.shape[0]]
        selected_edge_weights = torch.cat(edge_weights, dim=0)[: selected_features.shape[0]]
        selected_pred_boxes = torch.cat(pred_boxes_for_loss, dim=0)[: selected_features.shape[0]]
        selected_gt_boxes = torch.cat(gt_boxes_for_loss, dim=0)[: selected_features.shape[0]]

        predicted_delta = self.edge_head(selected_features)
        edge_loss_raw = F.smooth_l1_loss(
            predicted_delta,
            selected_targets,
            beta=float(edge_config.smooth_l1_beta),
            reduction="none",
        )
        edge_loss = (edge_loss_raw * selected_edge_weights).sum(dim=1) / selected_edge_weights.sum(dim=1).clamp_min(1.0e-6)
        edge_loss = edge_loss.mean()

        consistency_loss = _consistency_loss(
            predicted_delta=predicted_delta,
            pred_boxes=selected_pred_boxes,
            gt_boxes=selected_gt_boxes,
            margin=0.0,
        )
        direction_loss = _direction_loss(
            predicted_delta=predicted_delta,
            target_delta=selected_targets,
            min_abs_delta=float(edge_config.direction_min_abs_delta),
            margin=float(edge_config.direction_margin),
        )
        total = (
            edge_loss
            + float(edge_config.consistency_weight) * consistency_loss
            + float(edge_config.direction_weight) * direction_loss
        )
        scaled = total * float(weight)
        self._record_step(
            total=scaled,
            edge_loss=edge_loss,
            consistency_loss=consistency_loss,
            direction_loss=direction_loss,
            update_items=update_items,
        )
        return scaled

    def summary(self) -> dict[str, Any]:
        losses = int(self._stats.get("losses", 0))
        selected_points = int(self._stats.get("selected_points", 0))
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "loss_weight": self.loss_weight(),
            "edge_records": len(self._edge_records),
            "temporal_edge_repair": {
                **{key: int(value) for key, value in self._stats.items()},
                "mean_loss": self._loss_sum / float(max(losses, 1)),
                "mean_edge_loss": self._edge_loss_sum / float(max(losses, 1)),
                "mean_consistency_loss": self._consistency_loss_sum / float(max(losses, 1)),
                "mean_direction_loss": self._direction_loss_sum / float(max(losses, 1)),
                "mean_points_per_loss": float(selected_points) / float(max(losses, 1)),
            },
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "edge_records": {gt_uid: record.to_state() for gt_uid, record in self._edge_records.items()},
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            self._edge_records.clear()
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))
        records: dict[str, TemporalEdgeRecord] = {}
        raw_records = state.get("edge_records", {})
        if isinstance(raw_records, Mapping):
            for gt_uid, raw_record in raw_records.items():
                if isinstance(raw_record, Mapping):
                    records[str(gt_uid)] = TemporalEdgeRecord.from_state(raw_record)
        self._edge_records = records

    def _is_eligible_record(self, record: DHMRecord) -> bool:
        edge_config = self.config.temporal_edge_repair
        if str(record.last_state) != "FN_LOC":
            self._stats["skipped_not_fn_loc"] += 1
            return False
        if int(record.total_seen) < int(edge_config.min_observations):
            self._stats["skipped_low_observations"] += 1
            return False
        if int(record.state_counts.get("FN_LOC", 0)) < int(edge_config.min_fn_loc_count):
            self._stats["skipped_low_fn_loc_count"] += 1
            return False
        if float(record.instability_score) < float(edge_config.min_instability):
            self._stats["skipped_low_instability"] += 1
            return False
        return True

    def _edge_prior(self, record: DHMRecord, current_delta: torch.Tensor) -> torch.Tensor:
        edge_config = self.config.temporal_edge_repair
        temporal = self._edge_records.get(record.gt_uid)
        if (
            bool(edge_config.use_temporal_edge_weights)
            and temporal is not None
            and int(temporal.observations) >= int(edge_config.min_edge_observations)
        ):
            abs_prior = temporal.edge_abs_ema.detach().clone()
            self._stats["temporal_edge_weight_hits"] += 1
        else:
            abs_prior = current_delta.detach().abs().mean(dim=0).cpu()
        if bool(edge_config.use_dominant_edge_mask):
            weights = torch.full_like(abs_prior, float(edge_config.non_dominant_edge_weight))
            if bool((abs_prior > 0).any().item()):
                weights[int(torch.argmax(abs_prior).item())] = 1.0
        else:
            mean = abs_prior.mean().clamp_min(1.0e-6)
            weights = abs_prior / mean
        return weights.clamp(min=float(edge_config.min_edge_weight), max=float(edge_config.max_edge_weight))

    @torch.no_grad()
    def _record_step(
        self,
        *,
        total: torch.Tensor,
        edge_loss: torch.Tensor,
        consistency_loss: torch.Tensor,
        direction_loss: torch.Tensor,
        update_items: Sequence[tuple[DHMRecord, torch.Tensor]],
    ) -> None:
        edge_config = self.config.temporal_edge_repair
        self._stats["losses"] += 1
        self._loss_sum += float(total.detach().item())
        self._edge_loss_sum += float(edge_loss.detach().item())
        self._consistency_loss_sum += float(consistency_loss.detach().item())
        self._direction_loss_sum += float(direction_loss.detach().item())
        for record, delta in update_items:
            edge_record = self._edge_records.get(record.gt_uid)
            if edge_record is None:
                edge_record = TemporalEdgeRecord(
                    gt_uid=record.gt_uid,
                    class_id=int(record.class_id),
                )
                self._edge_records[record.gt_uid] = edge_record
            edge_record.update(
                delta,
                epoch=self.current_epoch,
                momentum=float(edge_config.edge_ema_momentum),
            )


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


def _consistency_loss(
    *,
    predicted_delta: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    scale = torch.stack(
        (
            (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp_min(1.0),
            (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp_min(1.0),
            (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp_min(1.0),
            (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp_min(1.0),
        ),
        dim=1,
    ).to(device=predicted_delta.device, dtype=predicted_delta.dtype)
    repaired = pred_boxes.to(device=predicted_delta.device, dtype=predicted_delta.dtype) + predicted_delta * scale
    original_iou = _aligned_box_iou(pred_boxes.to(device=repaired.device, dtype=repaired.dtype), gt_boxes.to(device=repaired.device, dtype=repaired.dtype)).detach()
    repaired_iou = _aligned_box_iou(repaired, gt_boxes.to(device=repaired.device, dtype=repaired.dtype))
    return F.relu(original_iou + float(margin) - repaired_iou).mean()


def _direction_loss(
    *,
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    min_abs_delta: float,
    margin: float,
) -> torch.Tensor:
    mask = target_delta.abs() >= float(min_abs_delta)
    if not bool(mask.any().item()):
        return predicted_delta.new_zeros(())
    target_sign = target_delta.sign()
    agreement = predicted_delta * target_sign
    return F.relu(float(margin) - agreement[mask]).mean()


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
