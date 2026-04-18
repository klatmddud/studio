from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torchvision.ops import boxes as box_ops

from .mdmb import normalize_arch


@dataclass(frozen=True, slots=True)
class RECALLConfig:
    enabled: bool = True
    amp_type_a: float = 2.5
    amp_type_b: float = 1.5
    ignore_iou_threshold: float = 0.3
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "RECALLConfig":
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
            amp_type_a=float(model_overrides.get("amp_type_a", data.get("amp_type_a", 2.5))),
            amp_type_b=float(model_overrides.get("amp_type_b", data.get("amp_type_b", 1.5))),
            ignore_iou_threshold=float(
                model_overrides.get(
                    "ignore_iou_threshold",
                    data.get("ignore_iou_threshold", 0.3),
                )
            ),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.amp_type_a < 0.0:
            raise ValueError("RECALL amp_type_a must be >= 0.")
        if self.amp_type_b < 0.0:
            raise ValueError("RECALL amp_type_b must be >= 0.")
        if not 0.0 <= self.ignore_iou_threshold <= 1.0:
            raise ValueError("RECALL ignore_iou_threshold must satisfy 0 <= threshold <= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "amp_type_a": self.amp_type_a,
            "amp_type_b": self.amp_type_b,
            "ignore_iou_threshold": self.ignore_iou_threshold,
            "arch": self.arch,
        }


class MDMBSelectiveLoss(nn.Module):
    """
    MDMB-guided selective loss reweighting.

    This helper has no trainable parameters. It only rescales existing per-sample
    detection losses based on MDMB miss history.
    """

    def __init__(self, config: RECALLConfig) -> None:
        super().__init__()
        self.config = config

    def compute_weights(
        self,
        *,
        point_gt_indices: torch.Tensor,
        point_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        missed_set: Mapping[int, str],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assignments = torch.as_tensor(point_gt_indices, device=device, dtype=torch.int64).flatten()
        boxes = torch.as_tensor(point_boxes, device=device, dtype=torch.float32)
        gt_boxes_tensor = torch.as_tensor(gt_boxes, device=device, dtype=torch.float32)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(
                "RECALL point_boxes must have shape [N, 4]. "
                f"Got {tuple(boxes.shape)}."
            )
        if gt_boxes_tensor.numel() == 0:
            gt_boxes_tensor = gt_boxes_tensor.reshape(-1, 4)
        elif gt_boxes_tensor.ndim != 2 or gt_boxes_tensor.shape[-1] != 4:
            raise ValueError(
                "RECALL gt_boxes must have shape [M, 4]. "
                f"Got {tuple(gt_boxes_tensor.shape)}."
            )

        num_points = int(assignments.numel())
        weights = torch.ones((num_points,), dtype=torch.float32, device=device)
        valid = torch.ones((num_points,), dtype=torch.bool, device=device)

        pos_mask = assignments >= 0
        neg_mask = ~pos_mask

        for point_index in pos_mask.nonzero(as_tuple=True)[0].tolist():
            gt_index = int(assignments[point_index].item())
            miss_type = missed_set.get(gt_index)
            if miss_type is None:
                continue
            weights[point_index] = self._get_amp(miss_type)

        if missed_set and bool(neg_mask.any().item()) and self.config.ignore_iou_threshold > 0.0:
            missed_indices = torch.as_tensor(
                sorted(missed_set),
                dtype=torch.int64,
                device=device,
            )
            missed_gt_boxes = gt_boxes_tensor[missed_indices]
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            neg_boxes = boxes[neg_indices]
            if neg_boxes.numel() > 0 and missed_gt_boxes.numel() > 0:
                ious = box_ops.box_iou(neg_boxes, missed_gt_boxes)
                near_missed = ious.max(dim=1).values >= self.config.ignore_iou_threshold
                valid[neg_indices[near_missed]] = False

        return weights, valid

    def apply(
        self,
        *,
        cls_losses: torch.Tensor,
        reg_losses: torch.Tensor,
        ctr_losses: torch.Tensor,
        point_gt_indices: torch.Tensor,
        point_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        missed_set: Mapping[int, str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_tensor = torch.as_tensor(cls_losses)
        reg_tensor = torch.as_tensor(reg_losses, device=cls_tensor.device, dtype=cls_tensor.dtype)
        ctr_tensor = torch.as_tensor(ctr_losses, device=cls_tensor.device, dtype=cls_tensor.dtype)
        assignments = torch.as_tensor(
            point_gt_indices,
            device=cls_tensor.device,
            dtype=torch.int64,
        ).flatten()

        if cls_tensor.ndim != 1:
            raise ValueError(
                "RECALL cls_losses must be a 1D tensor of per-point losses. "
                f"Got {tuple(cls_tensor.shape)}."
            )
        for name, tensor in (("reg_losses", reg_tensor), ("ctr_losses", ctr_tensor)):
            if tensor.shape != cls_tensor.shape:
                raise ValueError(
                    f"RECALL {name} must match cls_losses shape {tuple(cls_tensor.shape)}. "
                    f"Got {tuple(tensor.shape)}."
                )
        if assignments.shape != cls_tensor.shape:
            raise ValueError(
                "RECALL point_gt_indices must match cls_losses shape. "
                f"Got {tuple(assignments.shape)} and {tuple(cls_tensor.shape)}."
            )

        weights, valid = self.compute_weights(
            point_gt_indices=assignments,
            point_boxes=point_boxes,
            gt_boxes=gt_boxes,
            missed_set=missed_set,
            device=cls_tensor.device,
        )

        pos_mask = assignments >= 0
        num_pos = pos_mask.sum().clamp(min=1).to(dtype=cls_tensor.dtype)

        cls_loss = (cls_tensor * weights.to(dtype=cls_tensor.dtype) * valid.to(dtype=cls_tensor.dtype)).sum() / num_pos
        if bool(pos_mask.any().item()):
            pos_weights = weights[pos_mask].to(dtype=cls_tensor.dtype)
            reg_loss = (reg_tensor[pos_mask] * pos_weights).sum() / num_pos
            ctr_loss = (ctr_tensor[pos_mask] * pos_weights).sum() / num_pos
        else:
            reg_loss = cls_tensor.new_zeros(())
            ctr_loss = cls_tensor.new_zeros(())
        return cls_loss, reg_loss, ctr_loss

    def _get_amp(self, miss_type: str) -> float:
        return self.config.amp_type_a if miss_type == "type_a" else self.config.amp_type_b

    def extra_repr(self) -> str:
        return (
            f"arch={self.config.arch!r}, amp_type_a={self.config.amp_type_a}, "
            f"amp_type_b={self.config.amp_type_b}, "
            f"ignore_iou_threshold={self.config.ignore_iou_threshold}"
        )


RECALL = MDMBSelectiveLoss


def load_recall_config(path: str | Path, *, arch: str | None = None) -> RECALLConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"RECALL YAML must contain a mapping at the top level: {config_path}")
    return RECALLConfig.from_mapping(raw, arch=arch)


def build_recall_from_config(
    raw_config: Mapping[str, Any] | RECALLConfig,
    *,
    arch: str | None = None,
) -> MDMBSelectiveLoss | None:
    config = (
        raw_config
        if isinstance(raw_config, RECALLConfig)
        else RECALLConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return MDMBSelectiveLoss(config)


def build_recall_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> MDMBSelectiveLoss | None:
    config = load_recall_config(path, arch=arch)
    if not config.enabled:
        return None
    return MDMBSelectiveLoss(config)
