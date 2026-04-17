from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .mdmb import normalize_arch


@dataclass(frozen=True, slots=True)
class SCAConfig:
    enabled: bool = True
    lambda_sca: float = 0.2
    reduction: str = "mean"
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "SCAConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        config = cls(
            enabled=bool(model_overrides.get("enabled", data.get("enabled", True))),
            lambda_sca=float(model_overrides.get("lambda_sca", data.get("lambda_sca", 0.2))),
            reduction=str(model_overrides.get("reduction", data.get("reduction", "mean"))),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.lambda_sca < 0.0:
            raise ValueError("SCA lambda_sca must be >= 0.")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("SCA reduction must be either 'mean' or 'sum'.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lambda_sca": self.lambda_sca,
            "reduction": self.reduction,
            "arch": self.arch,
        }


class SoftCounterfactualAssignment(nn.Module):
    """Computes soft chronic-miss targets from IoU and miss-count history."""

    def __init__(self, config: SCAConfig) -> None:
        super().__init__()
        self.config = config

    def compute_soft_weights(
        self,
        iou_scores: torch.Tensor,
        miss_counts: torch.Tensor,
        *,
        iou_low: float,
        iou_high: float,
        max_miss_count: int | float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        iou_tensor = torch.as_tensor(iou_scores, dtype=torch.float32)
        if not torch.is_floating_point(iou_tensor):
            iou_tensor = iou_tensor.to(dtype=torch.float32)

        miss_tensor = torch.as_tensor(
            miss_counts,
            device=iou_tensor.device,
            dtype=iou_tensor.dtype,
        )
        if iou_tensor.shape != miss_tensor.shape:
            raise ValueError(
                "SCA iou_scores and miss_counts must share the same shape. "
                f"Got {tuple(iou_tensor.shape)} and {tuple(miss_tensor.shape)}."
            )
        if iou_tensor.numel() == 0:
            return iou_tensor

        if not 0.0 <= float(iou_low) < float(iou_high) <= 1.0:
            raise ValueError("SCA IoU bounds must satisfy 0 <= iou_low < iou_high <= 1.")

        if max_miss_count is None:
            max_miss_tensor = miss_tensor.max()
        else:
            max_miss_tensor = torch.as_tensor(
                max_miss_count,
                device=iou_tensor.device,
                dtype=iou_tensor.dtype,
            )
        max_miss_tensor = max_miss_tensor.clamp_min(1.0)

        scale = max(float(iou_high) - float(iou_low), torch.finfo(iou_tensor.dtype).eps)
        iou_weight = torch.sigmoid((iou_tensor - float(iou_low)) / scale)
        miss_weight = miss_tensor / max_miss_tensor
        return (iou_weight * miss_weight).clamp_(min=0.0, max=1.0)

    def extra_repr(self) -> str:
        return f"arch={self.config.arch!r}, reduction={self.config.reduction!r}"


SCA = SoftCounterfactualAssignment


def load_sca_config(path: str | Path, *, arch: str | None = None) -> SCAConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"SCA YAML must contain a mapping at the top level: {config_path}")
    return SCAConfig.from_mapping(raw, arch=arch)


def build_sca_from_config(
    raw_config: Mapping[str, Any] | SCAConfig,
    *,
    arch: str | None = None,
) -> SoftCounterfactualAssignment | None:
    config = (
        raw_config
        if isinstance(raw_config, SCAConfig)
        else SCAConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return SoftCounterfactualAssignment(config)


def build_sca_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> SoftCounterfactualAssignment | None:
    config = load_sca_config(path, arch=arch)
    if not config.enabled:
        return None
    return SoftCounterfactualAssignment(config)
