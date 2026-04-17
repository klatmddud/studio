from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from modules.nn.cfp import CFPConfig, CFPOutput


def cfp_regularization_loss(
    delta: torch.Tensor | CFPOutput,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    delta_tensor = _as_delta_tensor(delta)
    penalties = delta_tensor.pow(2).sum(dim=-1)
    return _reduce_loss(penalties, reduction)


def cfp_margin_loss(
    cf_scores: torch.Tensor,
    base_scores: torch.Tensor,
    *,
    margin: float,
    reduction: str = "mean",
) -> torch.Tensor:
    cf_scores = torch.as_tensor(cf_scores)
    base_scores = torch.as_tensor(
        base_scores,
        device=cf_scores.device,
        dtype=cf_scores.dtype,
    )
    penalties = (float(margin) - (cf_scores - base_scores)).clamp_min(0.0)
    return _reduce_loss(penalties, reduction)


def reduce_detection_loss(
    detection_loss: torch.Tensor | Mapping[str, Any],
    *,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(detection_loss, torch.Tensor):
        if detection_loss.ndim != 0:
            raise ValueError("CFP detection loss tensor must be scalar.")
        return detection_loss

    scalar_losses = [
        value
        for value in detection_loss.values()
        if isinstance(value, torch.Tensor) and value.ndim == 0
    ]
    if scalar_losses:
        total = scalar_losses[0]
        for value in scalar_losses[1:]:
            total = total + value
        return total

    if reference is None:
        return torch.zeros(())
    return reference.new_zeros(())


def compute_cfp_loss_dict(
    *,
    detection_loss: torch.Tensor | Mapping[str, Any],
    delta: torch.Tensor | CFPOutput,
    config: CFPConfig | Mapping[str, Any],
    cf_scores: torch.Tensor | None = None,
    base_scores: torch.Tensor | None = None,
    prefix: str = "cfp",
) -> dict[str, torch.Tensor]:
    resolved_config = _resolve_config(config)
    delta_tensor = _as_delta_tensor(delta)
    reference = _reference_tensor(delta_tensor, cf_scores, base_scores)

    det_loss = reduce_detection_loss(detection_loss, reference=reference)
    reg_loss = cfp_regularization_loss(delta_tensor) * resolved_config.lambda_reg
    if cf_scores is None or base_scores is None:
        margin_loss = det_loss.new_zeros(())
    else:
        margin_loss = (
            cfp_margin_loss(cf_scores, base_scores, margin=resolved_config.margin)
            * resolved_config.lambda_margin
        )

    total_loss = det_loss + reg_loss + margin_loss
    return {
        f"loss_{prefix}": total_loss,
        f"loss_{prefix}_det": det_loss,
        f"loss_{prefix}_reg": reg_loss,
        f"loss_{prefix}_margin": margin_loss,
    }


def _resolve_config(config: CFPConfig | Mapping[str, Any]) -> CFPConfig:
    if isinstance(config, CFPConfig):
        return config
    return CFPConfig.from_mapping(config)


def _as_delta_tensor(delta: torch.Tensor | CFPOutput) -> torch.Tensor:
    if isinstance(delta, CFPOutput):
        return delta.delta
    return torch.as_tensor(delta)


def _reference_tensor(*values: torch.Tensor | None) -> torch.Tensor:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
    return torch.zeros(())


def _reduce_loss(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if values.numel() == 0:
        return values.new_zeros(())
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"Unsupported CFP loss reduction: {reduction!r}")
