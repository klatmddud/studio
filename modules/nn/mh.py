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


@dataclass(frozen=True, slots=True)
class MissHeadConfig:
    enabled: bool = False
    start_epoch: int = 1
    grid_size: int = 2
    in_channels: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    pooling: str = "gap"
    level_aggregation: str = "mean"
    loss_weight: float = 0.1
    ignore_none_loss: bool = False
    none_loss_weight: float = 1.0
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "MissHeadConfig":
        merged = _merge_model_overrides(raw or {}, arch=arch)
        head = dict(merged.get("miss_head") or {})
        if "grid_size" not in head:
            head["grid_size"] = merged.get("grid_size", 2)
        if "start_epoch" not in head:
            head["start_epoch"] = merged.get("start_epoch", 1)

        config = cls(
            enabled=bool(head.get("enabled", False)),
            start_epoch=int(head.get("start_epoch", 1)),
            grid_size=int(head.get("grid_size", 2)),
            in_channels=int(head.get("in_channels", 256)),
            hidden_dim=int(head.get("hidden_dim", 256)),
            num_layers=int(head.get("num_layers", 2)),
            dropout=float(head.get("dropout", 0.0)),
            pooling=str(head.get("pooling", "gap")),
            level_aggregation=str(head.get("level_aggregation", "mean")),
            loss_weight=float(head.get("loss_weight", 0.1)),
            ignore_none_loss=bool(head.get("ignore_none_loss", False)),
            none_loss_weight=float(head.get("none_loss_weight", 1.0)),
            arch=normalize_arch(arch or merged.get("arch")),
        )
        config.validate()
        return config

    @property
    def num_regions(self) -> int:
        return int(self.grid_size) * int(self.grid_size)

    @property
    def num_labels(self) -> int:
        return int(self.num_regions) + 1

    def validate(self) -> None:
        if int(self.start_epoch) < 0:
            raise ValueError("MissHead start_epoch must be >= 0.")
        if int(self.grid_size) < 1:
            raise ValueError("MissHead grid_size must be >= 1.")
        if int(self.in_channels) < 1:
            raise ValueError("MissHead in_channels must be >= 1.")
        if int(self.hidden_dim) < 1:
            raise ValueError("MissHead hidden_dim must be >= 1.")
        if int(self.num_layers) < 1:
            raise ValueError("MissHead num_layers must be >= 1.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("MissHead dropout must satisfy 0 <= value < 1.")
        if self.pooling != "gap":
            raise ValueError("MissHead pooling currently supports only 'gap'.")
        if self.level_aggregation != "mean":
            raise ValueError("MissHead level_aggregation currently supports only 'mean'.")
        if float(self.loss_weight) < 0.0:
            raise ValueError("MissHead loss_weight must be >= 0.")
        if float(self.none_loss_weight) < 0.0:
            raise ValueError("MissHead none_loss_weight must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "start_epoch": self.start_epoch,
            "grid_size": self.grid_size,
            "in_channels": self.in_channels,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "pooling": self.pooling,
            "level_aggregation": self.level_aggregation,
            "loss_weight": self.loss_weight,
            "ignore_none_loss": self.ignore_none_loss,
            "none_loss_weight": self.none_loss_weight,
            "arch": self.arch,
        }


class MissHead(nn.Module):
    """Image-level region classifier for ReMiss missed-object targets."""

    def __init__(self, config: MissHeadConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, MissHeadConfig)
            else MissHeadConfig.from_mapping(config or {})
        )
        layers: list[nn.Module] = []
        if int(self.config.num_layers) == 1:
            layers.append(nn.Linear(self.config.in_channels, self.config.num_labels))
        else:
            layers.append(nn.Linear(self.config.in_channels, self.config.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if float(self.config.dropout) > 0.0:
                layers.append(nn.Dropout(p=float(self.config.dropout)))
            for _ in range(int(self.config.num_layers) - 2):
                layers.append(nn.Linear(self.config.hidden_dim, self.config.hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                if float(self.config.dropout) > 0.0:
                    layers.append(nn.Dropout(p=float(self.config.dropout)))
            layers.append(nn.Linear(self.config.hidden_dim, self.config.num_labels))
        self.classifier = nn.Sequential(*layers)

    @property
    def num_regions(self) -> int:
        return int(self.config.num_regions)

    @property
    def num_labels(self) -> int:
        return int(self.config.num_labels)

    def is_active(self, epoch: int | None = None) -> bool:
        epoch_value = 0 if epoch is None else int(epoch)
        return bool(self.config.enabled and epoch_value >= int(self.config.start_epoch))

    def forward(
        self,
        features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        pooled = self._pool_features(features)
        return self.classifier(pooled)

    def compute_loss(
        self,
        region_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        targets = target_labels.to(device=region_logits.device, dtype=torch.long)
        if bool(self.config.ignore_none_loss):
            valid = targets != 0
            if not bool(valid.any().item()):
                return {"miss_head_ce": region_logits.sum() * 0.0}
            ce = F.cross_entropy(region_logits[valid], targets[valid])
        else:
            weight = None
            if float(self.config.none_loss_weight) != 1.0:
                none_weight_is_zero = float(self.config.none_loss_weight) == 0.0
                has_non_none_target = bool((targets != 0).any().item())
                if none_weight_is_zero and not has_non_none_target:
                    return {"miss_head_ce": region_logits.sum() * 0.0}
                weight = torch.ones(
                    (self.num_labels,),
                    dtype=region_logits.dtype,
                    device=region_logits.device,
                )
                weight[0] = float(self.config.none_loss_weight)
            ce = F.cross_entropy(region_logits, targets, weight=weight)
        return {"miss_head_ce": ce * float(self.config.loss_weight)}

    @torch.no_grad()
    def compute_metrics(
        self,
        region_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> dict[str, float]:
        targets = target_labels.to(device=region_logits.device, dtype=torch.long)
        preds = self.predict_region(region_logits)
        metrics: dict[str, float] = {}
        total = int(targets.numel())
        if total <= 0:
            return metrics

        correct = preds.eq(targets)
        metrics["miss_head_acc"] = float(correct.float().mean().item())

        non_none = targets != 0
        if bool(non_none.any().item()):
            metrics["miss_head_non_none_acc"] = float(correct[non_none].float().mean().item())

        pred_none = preds == 0
        target_none = targets == 0
        metrics["miss_head_none_ratio"] = float(target_none.float().mean().item())
        metrics["miss_head_pred_none_ratio"] = float(pred_none.float().mean().item())
        if bool(pred_none.any().item()):
            metrics["miss_head_none_precision"] = float(target_none[pred_none].float().mean().item())
        if bool(target_none.any().item()):
            metrics["miss_head_none_recall"] = float(pred_none[target_none].float().mean().item())

        probabilities = region_logits.softmax(dim=1).clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
        metrics["miss_head_entropy"] = float(entropy.item())
        return metrics

    def predict_region(self, region_logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(region_logits, dim=1)

    def _pool_features(
        self,
        features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        tensors = _feature_tensors(features)
        if not tensors:
            raise ValueError("MissHead requires at least one feature tensor.")

        pooled_levels = []
        for tensor in tensors:
            if tensor.ndim != 4:
                raise ValueError(
                    "MissHead expects feature tensors with shape [B, C, H, W]; "
                    f"got {tuple(tensor.shape)}."
                )
            pooled_levels.append(F.adaptive_avg_pool2d(tensor, output_size=1).flatten(1))
        return torch.stack(pooled_levels, dim=0).mean(dim=0)


def load_misshead_config(path: str | Path, *, arch: str | None = None) -> MissHeadConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"ReMiss YAML must contain a mapping at the top level: {config_path}")
    return MissHeadConfig.from_mapping(raw, arch=arch)


def build_misshead_from_config(
    raw_config: Mapping[str, Any] | MissHeadConfig,
    *,
    arch: str | None = None,
    remiss_enabled: bool = True,
) -> MissHead | None:
    config = raw_config if isinstance(raw_config, MissHeadConfig) else MissHeadConfig.from_mapping(raw_config, arch=arch)
    if not config.enabled:
        return None
    if not remiss_enabled:
        raise ValueError("MissHead requires ReMiss/MissBank to be enabled.")
    return MissHead(config)


def build_misshead_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    remiss_enabled: bool = True,
) -> MissHead | None:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"ReMiss YAML must contain a mapping at the top level: {config_path}")
    config = MissHeadConfig.from_mapping(raw, arch=arch)
    if not config.enabled:
        return None
    if not bool(remiss_enabled):
        raise ValueError("MissHead requires ReMiss/MissBank to be enabled.")
    return MissHead(config)


def _merge_model_overrides(
    raw: Mapping[str, Any],
    *,
    arch: str | None,
) -> dict[str, Any]:
    data = dict(raw)
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
    return merged


def _feature_tensors(
    features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
) -> list[torch.Tensor]:
    if isinstance(features, torch.Tensor):
        return [features]
    if isinstance(features, Mapping):
        return [features[key] for key in sorted(features)]
    if isinstance(features, Sequence):
        return list(features)
    raise TypeError(f"Unsupported MissHead feature input type: {type(features).__name__}")
