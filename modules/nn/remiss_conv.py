from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .common import normalize_arch


@dataclass(frozen=True)
class ReMissConvConfig:
    enabled: bool = False
    arch: str | None = None
    grid_size: int = 2
    conv_enabled: bool = False
    start_epoch: int = 18
    in_channels: int = 256
    hidden_channels: int = 256
    use_coord: bool = True
    loss_weight: float = 0.1
    pos_weight: float | str | None = "auto"
    prototype_shared_across_levels: bool = True
    prototype_init_std: float = 0.02
    injection_enabled: bool = True
    injection_mode: str = "gated_additive"
    alpha: float = 0.1
    soft_injection: bool = True
    gate_threshold: float = 0.5
    levels: str | tuple[str, ...] = "all"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        arch: str | None = None,
    ) -> "ReMissConvConfig":
        merged = _merge_model_overrides(raw or {}, arch=arch)
        conv = dict(merged.get("conv") or {})
        prototype = dict(merged.get("prototype") or {})
        injection = dict(merged.get("injection") or {})

        levels = injection.get("levels", "all")
        if isinstance(levels, list):
            levels = tuple(str(level) for level in levels)
        elif isinstance(levels, tuple):
            levels = tuple(str(level) for level in levels)
        else:
            levels = str(levels)

        config = cls(
            enabled=bool(merged.get("enabled", False)),
            arch=normalize_arch(arch or merged.get("arch")) if (arch or merged.get("arch")) else None,
            grid_size=int(merged.get("grid_size", 2)),
            conv_enabled=bool(conv.get("enabled", False)),
            start_epoch=int(conv.get("start_epoch", 18)),
            in_channels=int(conv.get("in_channels", 256)),
            hidden_channels=int(conv.get("hidden_channels", 256)),
            use_coord=bool(conv.get("use_coord", True)),
            loss_weight=float(conv.get("loss_weight", 0.1)),
            pos_weight=_parse_pos_weight(conv.get("pos_weight", "auto")),
            prototype_shared_across_levels=bool(prototype.get("shared_across_levels", True)),
            prototype_init_std=float(prototype.get("init_std", 0.02)),
            injection_enabled=bool(injection.get("enabled", True)),
            injection_mode=str(injection.get("mode", "gated_additive")).lower(),
            alpha=float(injection.get("alpha", 0.1)),
            soft_injection=bool(injection.get("soft_injection", True)),
            gate_threshold=float(injection.get("gate_threshold", 0.5)),
            levels=levels,
        )
        config.validate()
        return config

    @property
    def num_regions(self) -> int:
        return int(self.grid_size) * int(self.grid_size)

    def validate(self) -> None:
        if self.grid_size < 1:
            raise ValueError("ReMissConv grid_size must be >= 1.")
        if self.start_epoch < 0:
            raise ValueError("ReMissConv conv.start_epoch must be >= 0.")
        if self.in_channels < 1:
            raise ValueError("ReMissConv conv.in_channels must be >= 1.")
        if self.hidden_channels < 0:
            raise ValueError("ReMissConv conv.hidden_channels must be >= 0.")
        if self.loss_weight < 0:
            raise ValueError("ReMissConv conv.loss_weight must be >= 0.")
        if not self.prototype_shared_across_levels:
            raise ValueError("ReMissConv currently supports only shared_across_levels: true.")
        if self.prototype_init_std < 0:
            raise ValueError("ReMissConv prototype.init_std must be >= 0.")
        if self.injection_mode not in {"additive", "gated_additive"}:
            raise ValueError("ReMissConv injection.mode must be 'additive' or 'gated_additive'.")
        if self.alpha < 0:
            raise ValueError("ReMissConv injection.alpha must be >= 0.")
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("ReMissConv injection.gate_threshold must satisfy 0 <= value <= 1.")
        if isinstance(self.pos_weight, (int, float)) and float(self.pos_weight) < 0.0:
            raise ValueError("ReMissConv conv.pos_weight must be >= 0, null, or 'auto'.")
        if self.levels != "all" and not isinstance(self.levels, tuple):
            raise ValueError("ReMissConv injection.levels must be 'all' or a list of level names.")


class ReMissConv(nn.Module):
    """Convolutional miss-aware spatial modulation block for ReMiss."""

    def __init__(self, config: ReMissConvConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, ReMissConvConfig)
            else ReMissConvConfig.from_mapping(config or {})
        )
        gate_in_channels = int(self.config.in_channels) + (2 if bool(self.config.use_coord) else 0)
        if int(self.config.hidden_channels) > 0:
            self.gate_head = nn.Sequential(
                nn.Conv2d(gate_in_channels, int(self.config.hidden_channels), kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(self.config.hidden_channels), 1, kernel_size=1),
            )
        else:
            self.gate_head = nn.Conv2d(gate_in_channels, 1, kernel_size=1)

        self.prototype = nn.Parameter(
            torch.empty(1, int(self.config.in_channels), int(self.config.grid_size), int(self.config.grid_size))
        )
        nn.init.normal_(self.prototype, mean=0.0, std=float(self.config.prototype_init_std))

    def is_active(self, epoch: int | None = None) -> bool:
        epoch_value = 0 if epoch is None else int(epoch)
        return bool(
            self.config.enabled
            and self.config.conv_enabled
            and epoch_value >= int(self.config.start_epoch)
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    ) -> dict[str, Any]:
        tensors, restore = _feature_tensors_and_restorer(features)
        if not tensors:
            raise ValueError("ReMissConv requires at least one feature tensor.")

        logits_per_level: list[torch.Tensor] = []
        gates_per_level: list[torch.Tensor] = []
        modulated: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        delta_ratios: list[torch.Tensor] = []

        for tensor in tensors:
            if tensor.ndim != 4:
                raise ValueError(
                    "ReMissConv expects feature tensors with shape [B, C, H, W]; "
                    f"got {tuple(tensor.shape)}."
                )
            if int(tensor.shape[1]) != int(self.config.in_channels):
                raise ValueError(
                    f"ReMissConv expected {self.config.in_channels} channels, got {int(tensor.shape[1])}."
                )

            gate_input = tensor
            if bool(self.config.use_coord):
                gate_input = torch.cat([tensor, _coord_channels_like(tensor)], dim=1)
            pooled = F.adaptive_avg_pool2d(
                gate_input,
                output_size=(int(self.config.grid_size), int(self.config.grid_size)),
            )
            logits = self.gate_head(pooled)
            gate_grid = torch.sigmoid(logits)
            logits_per_level.append(logits)
            gates_per_level.append(gate_grid)

            if bool(self.config.injection_enabled):
                injection_gate = gate_grid
                if not bool(self.config.soft_injection):
                    injection_gate = (injection_gate >= float(self.config.gate_threshold)).to(dtype=tensor.dtype)
                gate_map = F.interpolate(injection_gate, size=tensor.shape[-2:], mode="nearest")
                prototype_map = F.interpolate(
                    self.prototype.to(device=tensor.device, dtype=tensor.dtype),
                    size=tensor.shape[-2:],
                    mode="nearest",
                )
                delta = float(self.config.alpha) * gate_map * prototype_map
                updated = tensor + delta
            else:
                delta = torch.zeros_like(tensor)
                updated = tensor

            modulated.append(updated)
            delta_norm = delta.flatten(1).norm(p=2, dim=1).mean()
            feature_norm = tensor.flatten(1).norm(p=2, dim=1).mean().clamp_min(1e-12)
            delta_norms.append(delta_norm)
            delta_ratios.append(delta_norm / feature_norm)

        return {
            "features": restore(modulated),
            "miss_logits": logits_per_level,
            "gate_grids": gates_per_level,
            "delta_norm": torch.stack(delta_norms).mean(),
            "delta_ratio": torch.stack(delta_ratios).mean(),
        }

    def make_target_maps(
        self,
        missbank: Any,
        targets: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        get_records = getattr(missbank, "get_records", None)
        if not callable(get_records):
            raise TypeError("ReMissConv target generation requires a MissBank-like object with get_records().")
        threshold = int(missbank.config.target.miss_threshold)
        grid_size = int(self.config.grid_size)
        target_maps = torch.zeros(
            (len(targets), 1, grid_size, grid_size),
            dtype=torch.float32,
            device=device,
        )
        for batch_index, target in enumerate(targets):
            image_id = target.get("image_id")
            for record in get_records(image_id):
                if not bool(getattr(record, "is_missed", False)):
                    continue
                if int(getattr(record, "miss_count", 0)) < threshold:
                    continue
                region_id = int(getattr(record, "region_id", 0))
                if region_id <= 0 or region_id > grid_size * grid_size:
                    continue
                row = (region_id - 1) // grid_size
                col = (region_id - 1) % grid_size
                target_maps[batch_index, 0, row, col] = 1.0
        return target_maps

    def compute_loss(
        self,
        output: Mapping[str, Any],
        target_maps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits_per_level = _require_logits(output)
        targets = target_maps.to(device=logits_per_level[0].device, dtype=logits_per_level[0].dtype)
        pos_weight = self._pos_weight(targets, dtype=logits_per_level[0].dtype, device=logits_per_level[0].device)
        losses = [
            F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            for logits in logits_per_level
        ]
        loss = torch.stack(losses).mean() * float(self.config.loss_weight)
        return {"remiss_conv_loss": loss}

    def compute_metrics(
        self,
        output: Mapping[str, Any],
        target_maps: torch.Tensor,
    ) -> dict[str, float]:
        probabilities = self.aggregate_probabilities(output)
        targets = target_maps.to(device=probabilities.device, dtype=probabilities.dtype)
        target_positive = targets >= 0.5
        pred_positive = probabilities >= float(self.config.gate_threshold)
        target_negative = ~target_positive
        pred_negative = ~pred_positive

        total = target_positive.numel()
        correct = pred_positive.eq(target_positive).float().mean()
        metrics: dict[str, float] = {
            "remiss_conv_cell_acc": float(correct.item()),
            "remiss_conv_target_pos_ratio": float(target_positive.float().mean().item()),
            "remiss_conv_pred_pos_ratio": float(pred_positive.float().mean().item()),
            "remiss_conv_gate_positive_ratio": float(pred_positive.float().mean().item()),
            "remiss_conv_gate_mean": float(probabilities.mean().item()),
        }

        pred_pos_count = int(pred_positive.sum().item())
        target_pos_count = int(target_positive.sum().item())
        true_pos = int((pred_positive & target_positive).sum().item())
        if pred_pos_count > 0:
            metrics["remiss_conv_pos_precision"] = true_pos / float(pred_pos_count)
        if target_pos_count > 0:
            metrics["remiss_conv_pos_recall"] = true_pos / float(target_pos_count)
        precision = metrics.get("remiss_conv_pos_precision")
        recall = metrics.get("remiss_conv_pos_recall")
        if precision is not None and recall is not None and precision + recall > 0.0:
            metrics["remiss_conv_pos_f1"] = 2.0 * precision * recall / (precision + recall)

        target_neg_count = int(target_negative.sum().item())
        if target_neg_count > 0:
            true_neg = int((pred_negative & target_negative).sum().item())
            metrics["remiss_conv_neg_recall"] = true_neg / float(target_neg_count)

        intersection = (pred_positive & target_positive).flatten(1).sum(dim=1).float()
        union = (pred_positive | target_positive).flatten(1).sum(dim=1).float()
        map_iou = torch.where(union > 0, intersection / union.clamp_min(1.0), torch.ones_like(union))
        metrics["remiss_conv_map_iou"] = float(map_iou.mean().item()) if total > 0 else 0.0
        positive_images = target_positive.flatten(1).any(dim=1)
        if bool(positive_images.any().item()):
            metrics["remiss_conv_map_iou_pos_images"] = float(map_iou[positive_images].mean().item())
        empty_images = ~positive_images
        if bool(empty_images.any().item()):
            pred_empty = ~pred_positive.flatten(1).any(dim=1)
            metrics["remiss_conv_empty_map_acc"] = float(pred_empty[empty_images].float().mean().item())

        clamped = probabilities.clamp(1e-12, 1.0 - 1e-12)
        entropy = -(clamped * clamped.log() + (1.0 - clamped) * (1.0 - clamped).log()).mean()
        metrics["remiss_conv_gate_entropy"] = float(entropy.item())

        delta_norm = output.get("delta_norm")
        delta_ratio = output.get("delta_ratio")
        if isinstance(delta_norm, torch.Tensor):
            metrics["remiss_conv_delta_norm"] = float(delta_norm.detach().item())
        if isinstance(delta_ratio, torch.Tensor):
            metrics["remiss_conv_delta_ratio"] = float(delta_ratio.detach().item())
        return metrics

    def compute_missed_object_metrics(
        self,
        output: Mapping[str, Any],
        missbank: Any,
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, float]:
        get_records = getattr(missbank, "get_records", None)
        if not callable(get_records):
            return {}
        probabilities = self.aggregate_probabilities(output).detach()
        pred_positive = probabilities >= float(self.config.gate_threshold)
        threshold = int(missbank.config.target.miss_threshold)
        grid_size = int(self.config.grid_size)
        total = 0
        correct = 0
        weighted_total = 0.0
        weighted_correct = 0.0
        for batch_index, target in enumerate(targets):
            image_id = target.get("image_id")
            for record in get_records(image_id):
                if not bool(getattr(record, "is_missed", False)):
                    continue
                if int(getattr(record, "miss_count", 0)) < threshold:
                    continue
                region_id = int(getattr(record, "region_id", 0))
                if region_id <= 0 or region_id > grid_size * grid_size:
                    continue
                row = (region_id - 1) // grid_size
                col = (region_id - 1) % grid_size
                total += 1
                weight = float(getattr(record, "miss_count", 1))
                weighted_total += weight
                if bool(pred_positive[batch_index, 0, row, col].item()):
                    correct += 1
                    weighted_correct += weight
        metrics: dict[str, float] = {}
        if total > 0:
            metrics["remiss_conv_missed_object_cell_acc"] = correct / float(total)
        if weighted_total > 0.0:
            metrics["remiss_conv_missed_object_cell_acc_weighted"] = weighted_correct / weighted_total
        return metrics

    def aggregate_probabilities(self, output: Mapping[str, Any]) -> torch.Tensor:
        logits_per_level = _require_logits(output)
        return torch.stack([torch.sigmoid(logits) for logits in logits_per_level], dim=0).mean(dim=0)

    def output_device(self, output: Mapping[str, Any]) -> torch.device:
        return _require_logits(output)[0].device

    def _pos_weight(
        self,
        targets: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        value = self.config.pos_weight
        if value is None:
            return None
        if value == "auto":
            positives = targets.sum()
            negatives = targets.numel() - positives
            if not bool((positives > 0).item()):
                return None
            value_tensor = (negatives / positives.clamp_min(1.0)).to(device=device, dtype=dtype)
            return value_tensor.reshape(1)
        return torch.tensor([float(value)], dtype=dtype, device=device)


def load_remiss_conv_config(path: str | Path, *, arch: str | None = None) -> ReMissConvConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"ReMissConv YAML must contain a mapping at the top level: {config_path}")
    return ReMissConvConfig.from_mapping(raw, arch=arch)


def build_remiss_conv_from_config(
    raw_config: Mapping[str, Any] | ReMissConvConfig,
    *,
    arch: str | None = None,
    remiss_enabled: bool = True,
) -> ReMissConv | None:
    config = raw_config if isinstance(raw_config, ReMissConvConfig) else ReMissConvConfig.from_mapping(raw_config, arch=arch)
    if not (config.enabled and config.conv_enabled):
        return None
    if not bool(remiss_enabled):
        raise ValueError("ReMissConv requires a ReMissConv MissBank to be enabled.")
    return ReMissConv(config)


def build_remiss_conv_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    remiss_enabled: bool = True,
) -> ReMissConv | None:
    config = load_remiss_conv_config(path, arch=arch)
    return build_remiss_conv_from_config(config, arch=arch, remiss_enabled=remiss_enabled)


def _parse_pos_weight(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if normalized == "auto":
            return "auto"
        return float(normalized)
    return float(value)


def _require_logits(output: Mapping[str, Any]) -> list[torch.Tensor]:
    logits = output.get("miss_logits")
    if not isinstance(logits, list) or not logits:
        raise KeyError("ReMissConv output must contain a non-empty 'miss_logits' list.")
    if not all(isinstance(item, torch.Tensor) for item in logits):
        raise TypeError("ReMissConv output 'miss_logits' must be a list of tensors.")
    return logits


def _coord_channels_like(tensor: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = tensor.shape
    y = torch.linspace(-1.0, 1.0, steps=height, device=tensor.device, dtype=tensor.dtype)
    x = torch.linspace(-1.0, 1.0, steps=width, device=tensor.device, dtype=tensor.dtype)
    yy = y.view(1, 1, height, 1).expand(batch, 1, height, width)
    xx = x.view(1, 1, 1, width).expand(batch, 1, height, width)
    return torch.cat([xx, yy], dim=1)


def _feature_tensors_and_restorer(
    features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
) -> tuple[list[torch.Tensor], Any]:
    if isinstance(features, torch.Tensor):
        return [features], lambda values: values[0]
    if isinstance(features, Mapping):
        keys = sorted(features)
        return [features[key] for key in keys], lambda values: {key: value for key, value in zip(keys, values, strict=True)}
    if isinstance(features, Sequence):
        original_type = tuple if isinstance(features, tuple) else list
        return list(features), lambda values: tuple(values) if original_type is tuple else list(values)
    raise TypeError(f"Unsupported ReMissConv feature input type: {type(features).__name__}")


def _merge_model_overrides(
    raw: Mapping[str, Any],
    *,
    arch: str | None,
) -> dict[str, Any]:
    data = dict(raw)
    normalized_arch = normalize_arch(arch or data.get("arch")) if (arch or data.get("arch")) else None
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
    return merged
