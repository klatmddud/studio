from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from modules.nn.sca import SCAConfig


def sca_classification_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    logits_tensor = torch.as_tensor(logits)
    if not torch.is_floating_point(logits_tensor):
        logits_tensor = logits_tensor.to(dtype=torch.float32)

    target_tensor = torch.as_tensor(
        soft_targets,
        device=logits_tensor.device,
        dtype=logits_tensor.dtype,
    )
    if logits_tensor.shape != target_tensor.shape:
        raise ValueError(
            "SCA logits and soft_targets must share the same shape. "
            f"Got {tuple(logits_tensor.shape)} and {tuple(target_tensor.shape)}."
        )
    if logits_tensor.numel() == 0:
        return logits_tensor.new_zeros(())

    losses = F.binary_cross_entropy_with_logits(
        logits_tensor,
        target_tensor,
        reduction="none",
    )
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported SCA reduction: {reduction!r}")


def compute_sca_loss_dict(
    *,
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    config: SCAConfig | Mapping[str, Any],
    prefix: str = "sca",
) -> dict[str, torch.Tensor]:
    resolved_config = _resolve_config(config)
    raw_loss = sca_classification_loss(
        logits,
        soft_targets,
        reduction=resolved_config.reduction,
    )
    return {f"loss_{prefix}": raw_loss * resolved_config.lambda_sca}


def _resolve_config(config: SCAConfig | Mapping[str, Any]) -> SCAConfig:
    if isinstance(config, SCAConfig):
        return config
    return SCAConfig.from_mapping(config)
