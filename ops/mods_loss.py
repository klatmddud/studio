from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from modules.nn.mods import MODSConfig


def mods_classification_loss(
    cls_logits: torch.Tensor,
    gt_classes: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = torch.as_tensor(cls_logits)
    targets = torch.as_tensor(gt_classes, device=logits.device, dtype=torch.int64)
    if logits.ndim != 2:
        raise ValueError(
            "MODS classification logits must have shape [N, C]. "
            f"Got {tuple(logits.shape)}."
        )
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "MODS classification logits and targets must share the same batch dimension. "
            f"Got {logits.shape[0]} and {targets.shape[0]}."
        )
    if logits.shape[0] == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits, targets, reduction=reduction)


def mods_regression_loss(
    reg_pred: torch.Tensor,
    reg_target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    pred = torch.as_tensor(reg_pred)
    target = torch.as_tensor(reg_target, device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        raise ValueError(
            "MODS regression predictions and targets must share the same shape. "
            f"Got {tuple(pred.shape)} and {tuple(target.shape)}."
        )
    if pred.numel() == 0:
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred, target, reduction=reduction)


def compute_mods_loss_dict(
    *,
    cls_logits: torch.Tensor,
    gt_classes: torch.Tensor,
    config: MODSConfig | Mapping[str, Any],
    reg_pred: torch.Tensor | None = None,
    reg_target: torch.Tensor | None = None,
    prefix: str = "mods",
) -> dict[str, torch.Tensor]:
    resolved_config = _resolve_config(config)
    reference = torch.as_tensor(cls_logits)
    cls_loss = mods_classification_loss(
        reference,
        gt_classes,
        reduction=resolved_config.reduction,
    )
    losses = {
        f"loss_{prefix}_cls": cls_loss * resolved_config.lambda_mods,
    }
    if (
        reg_pred is not None
        and reg_target is not None
        and resolved_config.lambda_reg > 0.0
    ):
        reg_loss = mods_regression_loss(
            reg_pred,
            reg_target,
            reduction=resolved_config.reduction,
        )
        losses[f"loss_{prefix}_reg"] = (
            reg_loss
            * resolved_config.lambda_mods
            * resolved_config.lambda_reg
        )
    return losses


def _resolve_config(config: MODSConfig | Mapping[str, Any]) -> MODSConfig:
    if isinstance(config, MODSConfig):
        return config
    return MODSConfig.from_mapping(config)
