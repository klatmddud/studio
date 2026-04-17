from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .mdmb import normalize_arch


DEFAULT_FEATURE_DIMS = {
    "fasterrcnn": 1024,
    "fcos": 256,
    "dino": 256,
}


@dataclass(frozen=True, slots=True)
class CFPConfig:
    enabled: bool = True
    alpha_init: float = 0.1
    lambda_reg: float = 0.01
    lambda_margin: float = 0.5
    margin: float = 0.3
    hidden_dim: int | None = None
    hidden_ratio: float = 0.5
    detach_input: bool = True
    arch: str | None = None
    feature_dim: int | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "CFPConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        feature_dim = model_overrides.get(
            "feature_dim",
            data.get("feature_dim", DEFAULT_FEATURE_DIMS.get(normalized_arch)),
        )
        hidden_dim = model_overrides.get("hidden_dim", data.get("hidden_dim"))

        config = cls(
            enabled=bool(model_overrides.get("enabled", data.get("enabled", True))),
            alpha_init=float(model_overrides.get("alpha_init", data.get("alpha_init", 0.1))),
            lambda_reg=float(model_overrides.get("lambda_reg", data.get("lambda_reg", 0.01))),
            lambda_margin=float(
                model_overrides.get("lambda_margin", data.get("lambda_margin", 0.5))
            ),
            margin=float(model_overrides.get("margin", data.get("margin", 0.3))),
            hidden_dim=None if hidden_dim is None else int(hidden_dim),
            hidden_ratio=float(model_overrides.get("hidden_ratio", data.get("hidden_ratio", 0.5))),
            detach_input=bool(model_overrides.get("detach_input", data.get("detach_input", True))),
            arch=normalized_arch,
            feature_dim=None if feature_dim is None else int(feature_dim),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.alpha_init < 0.0:
            raise ValueError("CFP alpha_init must be >= 0.")
        if self.lambda_reg < 0.0:
            raise ValueError("CFP lambda_reg must be >= 0.")
        if self.lambda_margin < 0.0:
            raise ValueError("CFP lambda_margin must be >= 0.")
        if self.margin < 0.0:
            raise ValueError("CFP margin must be >= 0.")
        if self.feature_dim is not None and self.feature_dim < 1:
            raise ValueError("CFP feature_dim must be >= 1.")
        if self.hidden_dim is not None and self.hidden_dim < 1:
            raise ValueError("CFP hidden_dim must be >= 1.")
        if self.hidden_dim is None and self.hidden_ratio <= 0.0:
            raise ValueError("CFP hidden_ratio must be > 0 when hidden_dim is not set.")

    def resolved_feature_dim(self) -> int:
        if self.feature_dim is None:
            raise ValueError(
                "CFP feature_dim is required. Provide it in modules/cfg/cfp.yaml or per-model overrides."
            )
        return self.feature_dim

    def resolved_hidden_dim(self) -> int:
        if self.hidden_dim is not None:
            return self.hidden_dim
        return max(1, int(round(self.resolved_feature_dim() * self.hidden_ratio)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha_init": self.alpha_init,
            "lambda_reg": self.lambda_reg,
            "lambda_margin": self.lambda_margin,
            "margin": self.margin,
            "hidden_dim": self.hidden_dim,
            "hidden_ratio": self.hidden_ratio,
            "detach_input": self.detach_input,
            "arch": self.arch,
            "feature_dim": self.feature_dim,
        }


@dataclass(slots=True)
class CFPOutput:
    base_features: torch.Tensor
    delta: torch.Tensor
    perturbed_features: torch.Tensor
    alpha: torch.Tensor


class CounterfactualFeaturePerturbation(nn.Module):
    """Training-only branch that perturbs chronic-miss features with a small MLP."""

    def __init__(self, config: CFPConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_dim = config.resolved_feature_dim()
        self.hidden_dim = config.resolved_hidden_dim()
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(config.alpha_init), dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor,
        *,
        detach_input: bool | None = None,
    ) -> CFPOutput:
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim < 1:
            raise ValueError("CFP features must have at least one dimension.")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"CFP expected feature_dim={self.feature_dim}, got last_dim={features.shape[-1]}."
            )
        if not torch.is_floating_point(features):
            features = features.to(dtype=torch.float32)

        use_detach = self.config.detach_input if detach_input is None else bool(detach_input)
        base_features = features.detach() if use_detach else features
        delta = self.mlp(base_features)
        alpha = self.alpha.to(device=base_features.device, dtype=base_features.dtype)
        perturbed_features = base_features + alpha * delta
        return CFPOutput(
            base_features=base_features,
            delta=delta,
            perturbed_features=perturbed_features,
            alpha=alpha,
        )

    def extra_repr(self) -> str:
        return (
            f"arch={self.config.arch!r}, feature_dim={self.feature_dim}, "
            f"hidden_dim={self.hidden_dim}, detach_input={self.config.detach_input}"
        )


CFP = CounterfactualFeaturePerturbation


def load_cfp_config(path: str | Path, *, arch: str | None = None) -> CFPConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"CFP YAML must contain a mapping at the top level: {config_path}")
    return CFPConfig.from_mapping(raw, arch=arch)


def build_cfp_from_config(
    raw_config: Mapping[str, Any] | CFPConfig,
    *,
    arch: str | None = None,
) -> CounterfactualFeaturePerturbation | None:
    config = (
        raw_config
        if isinstance(raw_config, CFPConfig)
        else CFPConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return CounterfactualFeaturePerturbation(config)


def build_cfp_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> CounterfactualFeaturePerturbation | None:
    config = load_cfp_config(path, arch=arch)
    if not config.enabled:
        return None
    return CounterfactualFeaturePerturbation(config)
