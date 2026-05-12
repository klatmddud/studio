from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.ops import boxes as box_ops

from .common import normalize_arch


@dataclass(frozen=True, slots=True)
class BCPCConfig:
    enabled: bool = False
    start_epoch: int = 1
    num_classes: int = 91
    in_channels: int = 256
    prototype_dim: int = 256
    prototypes_per_class: int = 16
    momentum: float = 0.95
    tau_bg: float = 0.2
    tau_cls: float = 0.4
    lambda_bg: float = 0.5
    gamma: float = 0.5
    rho: float = 1.0
    max_hard_bg_per_image: int = 256
    max_pos_per_image: int = 256
    max_updates_per_class: int = 16
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "BCPCConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        merged = _merge_model_overrides(data, normalized_arch)
        config = cls(
            enabled=bool(merged.get("enabled", False)),
            start_epoch=int(merged.get("start_epoch", 1)),
            num_classes=int(merged.get("num_classes", 91)),
            in_channels=int(merged.get("in_channels", 256)),
            prototype_dim=int(merged.get("prototype_dim", 256)),
            prototypes_per_class=int(merged.get("prototypes_per_class", 16)),
            momentum=float(merged.get("momentum", 0.95)),
            tau_bg=float(merged.get("tau_bg", 0.2)),
            tau_cls=float(merged.get("tau_cls", 0.4)),
            lambda_bg=float(merged.get("lambda_bg", 0.5)),
            gamma=float(merged.get("gamma", 0.5)),
            rho=float(merged.get("rho", 1.0)),
            max_hard_bg_per_image=int(merged.get("max_hard_bg_per_image", 256)),
            max_pos_per_image=int(merged.get("max_pos_per_image", 256)),
            max_updates_per_class=int(merged.get("max_updates_per_class", 16)),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.start_epoch) < 0:
            raise ValueError("BCPC start_epoch must be >= 0.")
        if int(self.num_classes) < 1:
            raise ValueError("BCPC num_classes must be >= 1.")
        if int(self.in_channels) < 1:
            raise ValueError("BCPC in_channels must be >= 1.")
        if int(self.prototype_dim) < 1:
            raise ValueError("BCPC prototype_dim must be >= 1.")
        if int(self.prototypes_per_class) < 1:
            raise ValueError("BCPC prototypes_per_class must be >= 1.")
        if not 0.0 <= float(self.momentum) <= 1.0:
            raise ValueError("BCPC momentum must satisfy 0 <= value <= 1.")
        for name in ("tau_bg", "tau_cls"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"BCPC {name} must satisfy 0 <= value <= 1.")
        if float(self.lambda_bg) < 0.0:
            raise ValueError("BCPC lambda_bg must be >= 0.")
        if float(self.gamma) < 0.0:
            raise ValueError("BCPC gamma must be >= 0.")
        if float(self.rho) < 0.0:
            raise ValueError("BCPC rho must be >= 0.")
        for name in ("max_hard_bg_per_image", "max_pos_per_image", "max_updates_per_class"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"BCPC {name} must be >= 0.")

    def with_runtime(
        self,
        *,
        num_classes: int | None = None,
        in_channels: int | None = None,
    ) -> "BCPCConfig":
        config = replace(
            self,
            num_classes=self.num_classes if num_classes is None else int(num_classes),
            in_channels=self.in_channels if in_channels is None else int(in_channels),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "start_epoch": self.start_epoch,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "prototype_dim": self.prototype_dim,
            "prototypes_per_class": self.prototypes_per_class,
            "momentum": self.momentum,
            "tau_bg": self.tau_bg,
            "tau_cls": self.tau_cls,
            "lambda_bg": self.lambda_bg,
            "gamma": self.gamma,
            "rho": self.rho,
            "max_hard_bg_per_image": self.max_hard_bg_per_image,
            "max_pos_per_image": self.max_pos_per_image,
            "max_updates_per_class": self.max_updates_per_class,
            "arch": self.arch,
        }


class BackgroundConfuserPrototypeCalibration(nn.Module):
    """Class-conditioned background confuser prototype calibration."""

    def __init__(self, config: BCPCConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, BCPCConfig) else BCPCConfig.from_mapping(config or {})
        self.current_epoch = 0
        self.proj = nn.Linear(self.config.in_channels, self.config.prototype_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(self.config.prototype_dim + 2, self.config.prototype_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.prototype_dim, 1),
        )
        self.register_buffer(
            "prototypes",
            torch.zeros(
                self.config.num_classes,
                self.config.prototypes_per_class,
                self.config.prototype_dim,
            ),
        )
        self.register_buffer(
            "prototype_counts",
            torch.zeros(
                self.config.num_classes,
                self.config.prototypes_per_class,
                dtype=torch.long,
            ),
        )
        self._last_metrics: dict[str, float] = {}

    def start_epoch(self, epoch: int) -> None:
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def is_active(self, epoch: int | None = None) -> bool:
        epoch_value = self.current_epoch if epoch is None else int(epoch)
        return bool(self.config.enabled and int(epoch_value) >= int(self.config.start_epoch))

    def project(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features.new_zeros((0, self.config.prototype_dim))
        return F.normalize(self.proj(features), dim=-1)

    def background_similarity_pairs(
        self,
        z: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        if z.numel() == 0:
            return z.new_zeros((0,))
        class_ids = class_ids.to(device=z.device, dtype=torch.long).clamp(
            min=0,
            max=int(self.config.num_classes) - 1,
        )
        memory = self.prototypes.detach().to(device=z.device, dtype=z.dtype)[class_ids]
        counts = self.prototype_counts.detach().to(device=z.device)[class_ids]
        active = counts > 0
        normalized_memory = F.normalize(memory, dim=-1)
        sim = torch.einsum("nd,nkd->nk", F.normalize(z, dim=-1), normalized_memory)
        sim = sim.masked_fill(~active, float("-inf"))
        values = sim.max(dim=-1).values
        return torch.where(torch.isfinite(values), values, torch.zeros_like(values))

    def risk_logits_for_pairs(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        class_scores: torch.Tensor,
    ) -> torch.Tensor:
        z = self.project(features)
        similarity = self.background_similarity_pairs(z, class_ids)
        score_column = class_scores.to(device=z.device, dtype=z.dtype).reshape(-1, 1)
        risk_input = torch.cat([z, similarity.unsqueeze(-1), score_column], dim=-1)
        return self.risk_head(risk_input).squeeze(-1)

    def calibrate_scores(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        class_scores: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.enabled or features.numel() == 0:
            return base_scores
        class_ids = class_ids.to(device=base_scores.device, dtype=torch.long)
        active = (class_ids > 0) & (class_ids < int(self.config.num_classes))
        if not bool(active.any().item()):
            return base_scores
        counts_per_class = self.prototype_counts.sum(dim=1).to(device=base_scores.device)
        active = active & (counts_per_class[class_ids.clamp(min=0, max=int(self.config.num_classes) - 1)] > 0)
        if not bool(active.any().item()):
            return base_scores

        calibrated = base_scores.clone()
        logits = self.risk_logits_for_pairs(
            features[active],
            class_ids[active],
            class_scores[active].detach(),
        )
        risk = torch.sigmoid(logits).clamp(min=0.0, max=1.0)
        factor = (1.0 - risk).clamp_min(1e-6).pow(float(self.config.gamma))
        calibrated[active] = base_scores[active] * factor.to(dtype=base_scores.dtype)
        return calibrated

    def loss_for_fcos(
        self,
        *,
        cls_features: torch.Tensor,
        cls_logits: torch.Tensor,
        bbox_ctrness: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        zero = cls_logits.sum() * 0.0
        if not self.is_active():
            self._last_metrics = {"bcpc_active": 0.0}
            return zero

        selected_features: list[torch.Tensor] = []
        selected_classes: list[torch.Tensor] = []
        selected_scores: list[torch.Tensor] = []
        selected_targets: list[torch.Tensor] = []
        selected_weights: list[torch.Tensor] = []
        hard_update_features: list[torch.Tensor] = []
        hard_update_classes: list[torch.Tensor] = []
        stats: Counter[str] = Counter()

        quality_scores = torch.sqrt(
            torch.sigmoid(cls_logits) * torch.sigmoid(bbox_ctrness).clamp_min(0.0)
        )
        for image_index, target in enumerate(targets):
            scores_per_image = quality_scores[image_index]
            features_per_image = cls_features[image_index]
            boxes_per_image = pred_boxes[image_index].detach()
            matched_per_image = matched_idxs[image_index].to(device=scores_per_image.device)
            gt_boxes = target["boxes"].to(device=scores_per_image.device, dtype=torch.float32)
            gt_labels = target["labels"].to(device=scores_per_image.device, dtype=torch.long)

            max_iou = _max_iou_per_box(boxes_per_image, gt_boxes)
            hard_indices, hard_classes, hard_scores = self._select_hard_background(
                scores=scores_per_image.detach(),
                max_iou=max_iou,
                matched_idxs=matched_per_image,
            )
            if hard_indices.numel() > 0:
                hard_features = features_per_image[hard_indices]
                selected_features.append(hard_features)
                selected_classes.append(hard_classes)
                selected_scores.append(hard_scores)
                selected_targets.append(torch.ones_like(hard_scores))
                selected_weights.append(hard_scores.clamp(min=0.0, max=1.0).pow(float(self.config.rho)))
                hard_update_features.append(hard_features.detach())
                hard_update_classes.append(hard_classes.detach())
                stats["hard_bg"] += int(hard_indices.numel())

            pos_indices, pos_classes, pos_scores = self._select_positives(
                scores=scores_per_image.detach(),
                matched_idxs=matched_per_image,
                gt_labels=gt_labels,
            )
            if pos_indices.numel() > 0:
                selected_features.append(features_per_image[pos_indices])
                selected_classes.append(pos_classes)
                selected_scores.append(pos_scores)
                selected_targets.append(torch.zeros_like(pos_scores))
                selected_weights.append(torch.ones_like(pos_scores))
                stats["positive"] += int(pos_indices.numel())

        if not selected_features:
            self._last_metrics = {
                "bcpc_active": 1.0,
                "bcpc_loss_raw": 0.0,
                "bcpc_hard_bg": 0.0,
                "bcpc_positive": 0.0,
                "bcpc_memory_filled": float((self.prototype_counts > 0).sum().detach().cpu().item()),
            }
            return zero

        pair_features = torch.cat(selected_features, dim=0)
        pair_classes = torch.cat(selected_classes, dim=0)
        pair_scores = torch.cat(selected_scores, dim=0)
        pair_targets = torch.cat(selected_targets, dim=0)
        pair_weights = torch.cat(selected_weights, dim=0)

        logits = self.risk_logits_for_pairs(pair_features, pair_classes, pair_scores.detach())
        loss = F.binary_cross_entropy_with_logits(logits, pair_targets, reduction="none")
        weighted_loss = (loss * pair_weights).sum() / pair_weights.sum().clamp_min(1.0)
        if hard_update_features:
            self.update_memory(
                torch.cat(hard_update_features, dim=0),
                torch.cat(hard_update_classes, dim=0),
            )
        with torch.no_grad():
            risk = torch.sigmoid(logits)
            self._last_metrics = {
                "bcpc_active": 1.0,
                "bcpc_loss_raw": float(weighted_loss.detach().cpu().item()),
                "bcpc_hard_bg": float(stats["hard_bg"]),
                "bcpc_positive": float(stats["positive"]),
                "bcpc_risk_mean": float(risk.mean().detach().cpu().item()),
                "bcpc_memory_filled": float((self.prototype_counts > 0).sum().detach().cpu().item()),
            }
        return weighted_loss

    @torch.no_grad()
    def update_memory(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> None:
        if features.numel() == 0:
            return
        z = self.project(features).detach().to(device=self.prototypes.device, dtype=self.prototypes.dtype)
        class_ids = class_ids.detach().to(device=self.prototypes.device, dtype=torch.long)
        per_class_updates: Counter[int] = Counter()
        for feature, raw_class_id in zip(z, class_ids, strict=True):
            class_id = int(raw_class_id.item())
            if class_id <= 0 or class_id >= int(self.config.num_classes):
                continue
            if per_class_updates[class_id] >= int(self.config.max_updates_per_class):
                continue
            self._update_class_memory(class_id, feature)
            per_class_updates[class_id] += 1

    def get_training_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def summary(self) -> dict[str, Any]:
        filled_per_class = (self.prototype_counts > 0).sum(dim=1)
        active_classes = int((filled_per_class > 0).sum().detach().cpu().item())
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "num_classes": self.config.num_classes,
            "prototype_dim": self.config.prototype_dim,
            "prototypes_per_class": self.config.prototypes_per_class,
            "active_classes": active_classes,
            "filled_slots": int(filled_per_class.sum().detach().cpu().item()),
        }

    def _select_hard_background(
        self,
        *,
        scores: torch.Tensor,
        max_iou: torch.Tensor,
        matched_idxs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(self.config.max_hard_bg_per_image) == 0 or scores.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=scores.device)
            return empty, empty, scores.new_zeros((0,))
        if scores.shape[-1] <= 1:
            empty = torch.zeros((0,), dtype=torch.long, device=scores.device)
            return empty, empty, scores.new_zeros((0,))
        object_scores = scores[:, 1:]
        max_scores, local_classes = object_scores.max(dim=1)
        class_ids = local_classes + 1
        hard_mask = (
            (matched_idxs < 0)
            & (max_iou.to(device=scores.device) < float(self.config.tau_bg))
            & (max_scores > float(self.config.tau_cls))
        )
        indices = torch.where(hard_mask)[0]
        if indices.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=scores.device)
            return empty, empty, scores.new_zeros((0,))
        count = min(int(indices.numel()), int(self.config.max_hard_bg_per_image))
        selected_scores, order = max_scores[indices].topk(count)
        selected_indices = indices[order]
        return selected_indices, class_ids[selected_indices].to(dtype=torch.long), selected_scores

    def _select_positives(
        self,
        *,
        scores: torch.Tensor,
        matched_idxs: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(self.config.max_pos_per_image) == 0 or gt_labels.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=scores.device)
            return empty, empty, scores.new_zeros((0,))
        pos_indices = torch.where(matched_idxs >= 0)[0]
        if pos_indices.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=scores.device)
            return empty, empty, scores.new_zeros((0,))
        matched_gt = matched_idxs[pos_indices].clamp(min=0)
        class_ids = gt_labels[matched_gt].to(device=scores.device, dtype=torch.long)
        valid = (class_ids > 0) & (class_ids < int(scores.shape[-1]))
        pos_indices = pos_indices[valid]
        class_ids = class_ids[valid]
        if pos_indices.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=scores.device)
            return empty, empty, scores.new_zeros((0,))
        pos_scores = scores[pos_indices, class_ids]
        count = min(int(pos_indices.numel()), int(self.config.max_pos_per_image))
        selected_scores, order = pos_scores.topk(count)
        selected_indices = pos_indices[order]
        return selected_indices, class_ids[order], selected_scores

    @torch.no_grad()
    def _update_class_memory(self, class_id: int, feature: torch.Tensor) -> None:
        counts = self.prototype_counts[class_id]
        empty_slots = torch.where(counts <= 0)[0]
        if empty_slots.numel() > 0:
            slot = int(empty_slots[0].item())
            self.prototypes[class_id, slot] = F.normalize(feature, dim=0)
            self.prototype_counts[class_id, slot] += 1
            return

        proto = F.normalize(self.prototypes[class_id], dim=-1)
        slot = int(torch.argmax(proto @ F.normalize(feature, dim=0)).item())
        updated = float(self.config.momentum) * self.prototypes[class_id, slot]
        updated = updated + (1.0 - float(self.config.momentum)) * feature
        self.prototypes[class_id, slot] = F.normalize(updated, dim=0)
        self.prototype_counts[class_id, slot] += 1


def load_bcpc_config(path: str | Path, *, arch: str | None = None) -> BCPCConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"BCPC config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("BCPC config YAML must contain a mapping at the top level.")
    return BCPCConfig.from_mapping(raw, arch=arch)


def build_bcpc_from_config(
    raw_config: Mapping[str, Any] | BCPCConfig,
    *,
    arch: str | None = None,
    num_classes: int | None = None,
    in_channels: int | None = None,
) -> BackgroundConfuserPrototypeCalibration | None:
    config = raw_config if isinstance(raw_config, BCPCConfig) else BCPCConfig.from_mapping(raw_config, arch=arch)
    config = config.with_runtime(num_classes=num_classes, in_channels=in_channels)
    if not config.enabled:
        return None
    return BackgroundConfuserPrototypeCalibration(config)


def build_bcpc_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    num_classes: int | None = None,
    in_channels: int | None = None,
) -> BackgroundConfuserPrototypeCalibration | None:
    config = load_bcpc_config(path, arch=arch)
    return build_bcpc_from_config(
        config,
        arch=arch,
        num_classes=num_classes,
        in_channels=in_channels,
    )


def _max_iou_per_box(boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    if gt_boxes.numel() == 0:
        return boxes.new_zeros((boxes.shape[0],))
    ious = box_ops.box_iou(
        boxes.to(dtype=torch.float32),
        gt_boxes.to(device=boxes.device, dtype=torch.float32),
    )
    return ious.max(dim=1).values.to(device=boxes.device)


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
