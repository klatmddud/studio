from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .common import normalize_arch


@dataclass(frozen=True, slots=True)
class QGAFPConfig:
    enabled: bool = False
    in_channels: int = 256
    topk: int = 128
    hidden_dim: int = 256
    max_levels: int = 8
    gate_temperature: float = 1.0
    residual_scale_init: float = 0.0
    min_score: float | None = None
    levels: tuple[str, ...] | None = None
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "QGAFPConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        merged = _merge_model_overrides(data, normalized_arch)

        config = cls(
            enabled=bool(merged.get("enabled", False)),
            in_channels=int(merged.get("in_channels", 256)),
            topk=int(merged.get("topk", 128)),
            hidden_dim=int(merged.get("hidden_dim", 256)),
            max_levels=int(merged.get("max_levels", 8)),
            gate_temperature=float(merged.get("gate_temperature", 1.0)),
            residual_scale_init=float(merged.get("residual_scale_init", 0.0)),
            min_score=_optional_float(merged.get("min_score")),
            levels=_parse_levels(merged.get("levels")),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.in_channels < 1:
            raise ValueError("QG-AFP in_channels must be >= 1.")
        if self.topk < 1:
            raise ValueError("QG-AFP topk must be >= 1.")
        if self.hidden_dim < 1:
            raise ValueError("QG-AFP hidden_dim must be >= 1.")
        if self.max_levels < 1:
            raise ValueError("QG-AFP max_levels must be >= 1.")
        if self.gate_temperature <= 0.0:
            raise ValueError("QG-AFP gate_temperature must be > 0.")
        if self.min_score is not None and not 0.0 <= self.min_score <= 1.0:
            raise ValueError("QG-AFP min_score must be null or satisfy 0 <= value <= 1.")
        if self.levels is not None and len(self.levels) < 1:
            raise ValueError("QG-AFP levels must be null or contain at least one feature key.")
        if self.levels is not None and len(self.levels) > self.max_levels:
            raise ValueError("QG-AFP levels length cannot exceed max_levels.")

    def with_in_channels(self, in_channels: int) -> "QGAFPConfig":
        config = replace(self, in_channels=int(in_channels))
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "in_channels": self.in_channels,
            "topk": self.topk,
            "hidden_dim": self.hidden_dim,
            "max_levels": self.max_levels,
            "gate_temperature": self.gate_temperature,
            "residual_scale_init": self.residual_scale_init,
            "min_score": self.min_score,
            "levels": list(self.levels) if self.levels is not None else None,
            "arch": self.arch,
        }


class QueryGuidedScaleGate(nn.Module):
    """
    QG-AFP v0 post-neck gate.

    The module keeps the incoming feature dict contract intact. It mines top-k
    proxy objectness points from the FPN outputs, predicts query-conditioned
    level gates, aggregates them per batch, and applies a residual level scale.
    """

    def __init__(self, config: QGAFPConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, QGAFPConfig) else QGAFPConfig.from_mapping(config or {})

        self.proxy_score = nn.Conv2d(self.config.in_channels, 1, kernel_size=1)
        self.level_embedding = nn.Embedding(self.config.max_levels, self.config.hidden_dim)
        query_input_dim = self.config.in_channels + self.config.hidden_dim + 3
        self.query_mlp = nn.Sequential(
            nn.Linear(query_input_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gate_head = nn.Linear(self.config.hidden_dim, self.config.max_levels)
        self.residual_scale = nn.Parameter(torch.tensor(float(self.config.residual_scale_init)))
        self._last_metrics: dict[str, float] = {}

    def forward(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_items = list(features.items())
        active_items = [
            (name, feature)
            for name, feature in feature_items
            if self._uses_level(name)
        ]
        if not active_items:
            return _same_mapping_type(features, dict(feature_items))
        if len(active_items) > self.config.max_levels:
            raise ValueError(
                f"QG-AFP received {len(active_items)} active levels, "
                f"but max_levels={self.config.max_levels}."
            )

        first = active_items[0][1]
        batch_size = int(first.shape[0])
        if batch_size == 0:
            return _same_mapping_type(features, dict(feature_items))

        flat_features: list[torch.Tensor] = []
        flat_scores: list[torch.Tensor] = []
        flat_positions: list[torch.Tensor] = []
        flat_level_ids: list[torch.Tensor] = []
        for level_index, (_, feature) in enumerate(active_items):
            if feature.ndim != 4:
                raise ValueError("QG-AFP expects each feature map to have shape [B, C, H, W].")
            if int(feature.shape[1]) != self.config.in_channels:
                raise ValueError(
                    f"QG-AFP expected {self.config.in_channels} channels, "
                    f"but received {int(feature.shape[1])}."
                )
            _, _, height, width = feature.shape
            flat_features.append(feature.flatten(2).transpose(1, 2))
            flat_scores.append(self.proxy_score(feature).flatten(1))
            positions = _normalized_grid(
                height=int(height),
                width=int(width),
                device=feature.device,
                dtype=feature.dtype,
            )
            flat_positions.append(positions)
            flat_level_ids.append(
                torch.full(
                    (int(height) * int(width),),
                    level_index,
                    device=feature.device,
                    dtype=torch.long,
                )
            )

        all_features = torch.cat(flat_features, dim=1)
        all_scores = torch.cat(flat_scores, dim=1)
        all_positions = torch.cat(flat_positions, dim=0)
        all_level_ids = torch.cat(flat_level_ids, dim=0)
        total_points = int(all_scores.shape[1])
        query_count = min(int(self.config.topk), total_points)

        topk_logits, topk_indices = torch.topk(all_scores, k=query_count, dim=1)
        selected_features = _batched_gather(all_features, topk_indices)
        selected_positions = _batched_gather(
            all_positions.unsqueeze(0).expand(batch_size, -1, -1),
            topk_indices,
        )
        selected_levels = torch.gather(
            all_level_ids.unsqueeze(0).expand(batch_size, -1),
            dim=1,
            index=topk_indices,
        )
        selected_scores = topk_logits.sigmoid().unsqueeze(-1)
        level_embedding = self.level_embedding(selected_levels)

        query_input = torch.cat(
            [selected_features, level_embedding, selected_positions, selected_scores],
            dim=-1,
        )
        query_hidden = self.query_mlp(query_input)
        num_active_levels = len(active_items)
        gate_logits = self.gate_head(query_hidden)[..., :num_active_levels]
        gate_logits = gate_logits / float(self.config.gate_temperature)
        gates = torch.softmax(gate_logits, dim=-1)

        active_query_mask: torch.Tensor | None = None
        if self.config.min_score is not None:
            active_query_mask = (selected_scores >= float(self.config.min_score)).to(gates.dtype)
            gates = gates * active_query_mask
            denominator = active_query_mask.sum(dim=1).clamp_min(1.0)
        else:
            denominator = torch.full(
                (batch_size, 1),
                float(query_count),
                device=gates.device,
                dtype=gates.dtype,
            )
        alpha = gates.sum(dim=1) / denominator

        output = dict(feature_items)
        scale = self.residual_scale.to(dtype=first.dtype)
        for level_index, (name, feature) in enumerate(active_items):
            level_alpha = alpha[:, level_index].to(dtype=feature.dtype).view(batch_size, 1, 1, 1)
            output[name] = feature * (1.0 + scale * level_alpha)

        self._last_metrics = _summarize_metrics(
            gates=gates,
            alpha=alpha,
            query_count=query_count,
            residual_scale=self.residual_scale,
            active_query_mask=active_query_mask,
        )
        return _same_mapping_type(features, output)

    def get_training_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def _uses_level(self, name: str) -> bool:
        if self.config.levels is None:
            return True
        return str(name) in self.config.levels


def load_qg_afp_config(path: str | Path, *, arch: str | None = None) -> QGAFPConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"QG-AFP config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("QG-AFP config YAML must contain a mapping at the top level.")
    return QGAFPConfig.from_mapping(raw, arch=arch)


def build_qg_afp_from_config(
    raw_config: Mapping[str, Any] | QGAFPConfig,
    *,
    arch: str | None = None,
    in_channels: int | None = None,
) -> QueryGuidedScaleGate | None:
    config = raw_config if isinstance(raw_config, QGAFPConfig) else QGAFPConfig.from_mapping(raw_config, arch=arch)
    if in_channels is not None:
        config = config.with_in_channels(int(in_channels))
    if not config.enabled:
        return None
    return QueryGuidedScaleGate(config)


def build_qg_afp_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    in_channels: int | None = None,
) -> QueryGuidedScaleGate | None:
    config = load_qg_afp_config(path, arch=arch)
    return build_qg_afp_from_config(config, arch=arch, in_channels=in_channels)


def _merge_model_overrides(data: Mapping[str, Any], arch: str | None) -> dict[str, Any]:
    merged = dict(data)
    if arch is None:
        return merged
    per_model = data.get("models", {})
    if not isinstance(per_model, Mapping):
        return merged
    selected = per_model.get(arch, {})
    if not isinstance(selected, Mapping):
        return merged
    for key, value in selected.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _parse_levels(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    raise TypeError("QG-AFP levels must be null, a string, or a list of strings.")


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    return float(raw)


def _normalized_grid(
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)


def _batched_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    index = indices.unsqueeze(-1).expand(-1, -1, int(values.shape[-1]))
    return torch.gather(values, dim=1, index=index)


def _same_mapping_type(
    original: Mapping[str, torch.Tensor],
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if isinstance(original, OrderedDict):
        return OrderedDict((name, output[name]) for name in original.keys())
    return output


def _summarize_metrics(
    *,
    gates: torch.Tensor,
    alpha: torch.Tensor,
    query_count: int,
    residual_scale: torch.Tensor,
    active_query_mask: torch.Tensor | None,
) -> dict[str, float]:
    with torch.no_grad():
        num_levels = int(gates.shape[-1])
        if active_query_mask is not None:
            active_mask = active_query_mask.squeeze(-1).to(dtype=torch.bool)
        else:
            active_mask = torch.ones(
                gates.shape[:-1],
                device=gates.device,
                dtype=torch.bool,
            )
        active_count = int(active_mask.sum().detach().cpu().item())

        if num_levels > 1:
            entropy = -(gates.clamp_min(1e-12) * gates.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy / math.log(float(num_levels))
            entropy = entropy[active_mask].mean() if active_count > 0 else entropy.new_zeros(())
        else:
            entropy = torch.zeros((), device=gates.device, dtype=gates.dtype)

        gate_max = gates.max(dim=-1).values
        gate_max_mean = gate_max[active_mask].mean() if active_count > 0 else gate_max.new_zeros(())

        top1_levels = gates.argmax(dim=-1)
        if active_count > 0:
            level_counts = torch.bincount(
                top1_levels[active_mask].reshape(-1),
                minlength=num_levels,
            ).to(dtype=gates.dtype)
            level_probs = level_counts / level_counts.sum().clamp_min(1.0)
            nonzero_probs = level_probs[level_probs > 0]
            if num_levels > 1 and int(nonzero_probs.numel()) > 0:
                usage_entropy = -(nonzero_probs * nonzero_probs.log()).sum()
                usage_entropy = usage_entropy / math.log(float(num_levels))
            else:
                usage_entropy = level_probs.new_zeros(())
            top1_share = level_probs.max()
        else:
            usage_entropy = gates.new_zeros(())
            top1_share = gates.new_zeros(())

        metrics = {
            "qg_afp_gate_entropy": float(entropy.detach().cpu().item()),
            "qg_afp_gate_max_mean": float(gate_max_mean.detach().cpu().item()),
            "qg_afp_level_usage_entropy": float(usage_entropy.detach().cpu().item()),
            "qg_afp_level_top1_share": float(top1_share.detach().cpu().item()),
            "qg_afp_alpha_mean": float(alpha.mean().detach().cpu().item()),
            "qg_afp_residual_scale": float(residual_scale.detach().cpu().item()),
            "qg_afp_query_count": float(query_count),
        }
        alpha_mean = alpha.mean(dim=0)
        for level_index in range(num_levels):
            metrics[f"qg_afp_alpha_l{level_index}"] = float(
                alpha_mean[level_index].detach().cpu().item()
            )
        if active_query_mask is not None:
            metrics["qg_afp_active_query_fraction"] = float(
                active_query_mask.mean().detach().cpu().item()
            )
        return metrics
