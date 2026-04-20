from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.ops import MultiScaleRoIAlign

from .far import _match_boxes_to_records
from .mdmb import MissedDetectionMemoryBank, _GTRecord, normalize_arch


@dataclass(frozen=True, slots=True)
class MCEConfig:
    """Configuration for the Miss-Conditioned class Embedding module."""

    enabled: bool = False
    embed_dim: int = 256
    lambda_mce: float = 1.0
    min_miss_streak: int = 1
    match_threshold: float = 0.95
    feature_keys: tuple[str, ...] = ("0", "1", "2", "p6", "p7")
    roi_output_size: int = 7
    roi_sampling_ratio: int = 2
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "MCEConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    overrides = dict(selected)

        def _pick(key: str, default: Any) -> Any:
            return overrides.get(key, data.get(key, default))

        feature_keys_raw = _pick("feature_keys", ("0", "1", "2", "p6", "p7"))
        if isinstance(feature_keys_raw, str):
            feature_keys_raw = (feature_keys_raw,)
        feature_keys = tuple(str(k) for k in feature_keys_raw)

        config = cls(
            enabled=bool(_pick("enabled", False)),
            embed_dim=int(_pick("embed_dim", 256)),
            lambda_mce=float(_pick("lambda_mce", 1.0)),
            min_miss_streak=int(_pick("min_miss_streak", 1)),
            match_threshold=float(_pick("match_threshold", 0.95)),
            feature_keys=feature_keys,
            roi_output_size=int(_pick("roi_output_size", 7)),
            roi_sampling_ratio=int(_pick("roi_sampling_ratio", 2)),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.embed_dim < 1:
            raise ValueError("MCE embed_dim must be >= 1.")
        if self.lambda_mce < 0.0:
            raise ValueError("MCE lambda_mce must be >= 0.")
        if self.min_miss_streak < 1:
            raise ValueError("MCE min_miss_streak must be >= 1.")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("MCE match_threshold must satisfy 0 <= threshold <= 1.")
        if not self.feature_keys:
            raise ValueError("MCE feature_keys must contain at least one key.")
        if self.roi_output_size < 1:
            raise ValueError("MCE roi_output_size must be >= 1.")
        if self.roi_sampling_ratio < 0:
            raise ValueError("MCE roi_sampling_ratio must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "embed_dim": self.embed_dim,
            "lambda_mce": self.lambda_mce,
            "min_miss_streak": self.min_miss_streak,
            "match_threshold": self.match_threshold,
            "feature_keys": list(self.feature_keys),
            "roi_output_size": self.roi_output_size,
            "roi_sampling_ratio": self.roi_sampling_ratio,
            "arch": self.arch,
        }


class MissConditionedEmbedding(nn.Module):
    """
    MCE — Miss-Conditioned class Embedding.

    Maintains one learnable prototype embedding per class. When a GT has been
    consecutively missed for at least `min_miss_streak` epochs, its loss weight
    is scaled by `1 + lambda_mce * (1 - alpha) * (n / w)`, where:
      - alpha = sigmoid(dot(e_c, f_gt) / sqrt(D))
        measures how close the current feature is to the class prototype.
      - n / w is the streak ratio (consecutive miss count / global max).

    Prototypes are trained purely through the detection loss — no auxiliary
    objective. Features are not modified, so there is no train/inference
    distribution shift.

    Depends on MDMB being enabled (reads `mdmb._gt_records`).
    """

    def __init__(self, config: MCEConfig, num_classes: int) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(num_classes, config.embed_dim)
        nn.init.normal_(self.embeddings.weight, std=0.01)
        self._pool = MultiScaleRoIAlign(
            featmap_names=list(config.feature_keys),
            output_size=config.roi_output_size,
            sampling_ratio=config.roi_sampling_ratio,
        )

    @property
    def num_classes(self) -> int:
        return self.embeddings.num_embeddings

    def forward(self, features):
        return features

    def compute_gt_weights(
        self,
        *,
        gt_labels: torch.Tensor,
        gt_boxes_norm: torch.Tensor,
        pooled_features: torch.Tensor,
        mdmb: MissedDetectionMemoryBank,
        image_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-GT loss weight multipliers for one image, split by loss type.

        Args:
            gt_labels:       [N_gt] int64 class IDs.
            gt_boxes_norm:   [N_gt, 4] normalized xyxy GT boxes.
            pooled_features: [N_gt, D] ROI-pooled neck features per GT.
            mdmb:            MDMB instance to read _gt_records and _bank from.
            image_key:       Normalized image ID string.

        Returns:
            (cls_weights, reg_weights): each [N_gt] float32, >= 1.0.
            - type_a miss (localization failure): reg_weights amplified, cls_weights=1.
            - type_b miss (class failure):        cls_weights amplified, reg_weights=1.
            - miss_type unknown:                  both amplified.
        """
        n_gt = int(gt_labels.shape[0])
        device = pooled_features.device
        cls_weights = torch.ones(n_gt, dtype=torch.float32, device=device)
        reg_weights = torch.ones(n_gt, dtype=torch.float32, device=device)

        global_max = int(getattr(mdmb, "_global_max_consecutive_miss", 0))
        if global_max <= 0:
            return cls_weights, reg_weights

        records_map = getattr(mdmb, "_gt_records", None)
        if not isinstance(records_map, Mapping):
            return cls_weights, reg_weights
        records: list[_GTRecord] = list(records_map.get(image_key, ()))
        if not records:
            return cls_weights, reg_weights

        matches = _match_boxes_to_records(
            gt_boxes_norm=gt_boxes_norm.cpu(),
            gt_labels=gt_labels.cpu(),
            records=records,
            iou_thresh=self.config.match_threshold,
        )

        # Match to bank entries to retrieve miss_type
        bank_map = getattr(mdmb, "_bank", None)
        bank_entries = list(bank_map.get(image_key, ())) if isinstance(bank_map, Mapping) else []
        if bank_entries:
            bank_matches = _match_boxes_to_records(
                gt_boxes_norm=gt_boxes_norm.cpu(),
                gt_labels=gt_labels.cpu(),
                records=bank_entries,
                iou_thresh=self.config.match_threshold,
            )
        else:
            bank_matches: list[int | None] = [None] * n_gt

        scale = math.sqrt(float(self.config.embed_dim))
        for i in range(n_gt):
            record_idx = matches[i]
            if record_idx is None:
                continue
            record = records[record_idx]
            n = record.consecutive_miss_count
            if n < self.config.min_miss_streak:
                continue

            streak_ratio = float(n) / float(global_max)
            class_id = int(gt_labels[i].item())
            if class_id >= self.num_classes:
                continue

            e_c = self.embeddings.weight[class_id]
            f_gt = F.normalize(pooled_features[i], dim=-1, eps=1e-6)
            alpha = torch.sigmoid(torch.dot(e_c, f_gt) / scale)
            amp = 1.0 + self.config.lambda_mce * (1.0 - alpha) * streak_ratio

            miss_type: str | None = None
            bank_idx = bank_matches[i]
            if bank_idx is not None:
                miss_type = getattr(bank_entries[bank_idx], "miss_type", None)

            if miss_type == "type_a":
                reg_weights[i] = amp
            elif miss_type == "type_b":
                cls_weights[i] = amp
            else:
                cls_weights[i] = amp
                reg_weights[i] = amp

        return cls_weights, reg_weights

    def pool_features(
        self,
        features: Mapping[str, torch.Tensor],
        boxes_per_image: Sequence[torch.Tensor],
        image_shapes: Sequence[tuple[int, int]],
    ) -> list[torch.Tensor]:
        """ROI-pool neck features at GT box locations. Returns [N_gt, D] per image."""
        feature_map: dict[str, torch.Tensor] = {}
        for key in self.config.feature_keys:
            if key not in features:
                raise KeyError(
                    f"MCE expected feature map {key!r} in backbone output. "
                    f"Available keys: {list(features.keys())}"
                )
            feature_map[key] = features[key]

        counts = [int(b.shape[0]) for b in boxes_per_image]
        total = sum(counts)
        any_feat = next(iter(feature_map.values()))
        if total == 0:
            empty = any_feat.new_zeros((0, any_feat.shape[1]))
            return [empty for _ in boxes_per_image]

        pooled = self._pool(feature_map, list(boxes_per_image), list(image_shapes))
        pooled_flat = pooled.flatten(2).mean(dim=-1)  # [total, C]
        return list(pooled_flat.split(counts, dim=0))

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "num_classes": self.num_classes,
            "embed_dim": self.config.embed_dim,
            "lambda_mce": self.config.lambda_mce,
            "min_miss_streak": self.config.min_miss_streak,
        }

    def extra_repr(self) -> str:
        return (
            f"enabled={self.config.enabled}, arch={self.config.arch!r}, "
            f"num_classes={self.num_classes}, embed_dim={self.config.embed_dim}"
        )


MCE = MissConditionedEmbedding


def load_mce_config(path: str | Path, *, arch: str | None = None) -> MCEConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"MCE YAML must contain a mapping at the top level: {config_path}")
    return MCEConfig.from_mapping(raw, arch=arch)


def build_mce_from_config(
    raw_config: Mapping[str, Any] | MCEConfig,
    *,
    arch: str | None = None,
    num_classes: int = 91,
) -> MissConditionedEmbedding | None:
    config = (
        raw_config
        if isinstance(raw_config, MCEConfig)
        else MCEConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return MissConditionedEmbedding(config, num_classes=num_classes)


def build_mce_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    num_classes: int = 91,
) -> MissConditionedEmbedding | None:
    config = load_mce_config(path, arch=arch)
    if not config.enabled:
        return None
    return MissConditionedEmbedding(config, num_classes=num_classes)
