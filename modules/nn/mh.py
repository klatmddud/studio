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
    class_balanced_loss: bool = False
    has_miss_head: bool = False
    has_miss_loss_weight: float = 0.1
    region_loss_weight: float = 0.1
    has_miss_pos_weight: float | str | None = "auto"
    has_miss_threshold: float = 0.5
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
            class_balanced_loss=bool(head.get("class_balanced_loss", False)),
            has_miss_head=bool(head.get("has_miss_head", False)),
            has_miss_loss_weight=float(
                head.get("has_miss_loss_weight", head.get("loss_weight", 0.1))
            ),
            region_loss_weight=float(head.get("region_loss_weight", head.get("loss_weight", 0.1))),
            has_miss_pos_weight=_parse_pos_weight(head.get("has_miss_pos_weight", "auto")),
            has_miss_threshold=float(head.get("has_miss_threshold", 0.5)),
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
        if float(self.has_miss_loss_weight) < 0.0:
            raise ValueError("MissHead has_miss_loss_weight must be >= 0.")
        if float(self.region_loss_weight) < 0.0:
            raise ValueError("MissHead region_loss_weight must be >= 0.")
        if not 0.0 <= float(self.has_miss_threshold) <= 1.0:
            raise ValueError("MissHead has_miss_threshold must satisfy 0 <= value <= 1.")
        if isinstance(self.has_miss_pos_weight, str):
            if self.has_miss_pos_weight != "auto":
                raise ValueError("MissHead has_miss_pos_weight must be a non-negative number, null, or 'auto'.")
        elif self.has_miss_pos_weight is not None and float(self.has_miss_pos_weight) < 0.0:
            raise ValueError("MissHead has_miss_pos_weight must be >= 0.")

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
            "class_balanced_loss": self.class_balanced_loss,
            "has_miss_head": self.has_miss_head,
            "has_miss_loss_weight": self.has_miss_loss_weight,
            "region_loss_weight": self.region_loss_weight,
            "has_miss_pos_weight": self.has_miss_pos_weight,
            "has_miss_threshold": self.has_miss_threshold,
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
        if bool(self.config.has_miss_head):
            self.trunk, output_dim = self._build_trunk()
            self.has_miss_classifier = nn.Linear(output_dim, 1)
            self.region_classifier = nn.Linear(output_dim, self.config.num_regions)
        else:
            self.classifier = self._build_legacy_classifier()

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
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        pooled = self._pool_features(features)
        if bool(self.config.has_miss_head):
            representation = self.trunk(pooled)
            return {
                "has_miss_logits": self.has_miss_classifier(representation).squeeze(1),
                "region_logits": self.region_classifier(representation),
            }
        return self.classifier(pooled)

    def compute_loss(
        self,
        miss_head_output: torch.Tensor | Mapping[str, torch.Tensor],
        target_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if bool(self.config.has_miss_head):
            return self._compute_split_loss(miss_head_output, target_labels)

        region_logits = _require_tensor_output(miss_head_output)
        targets = target_labels.to(device=region_logits.device, dtype=torch.long)
        if bool(self.config.ignore_none_loss):
            valid = targets != 0
            if not bool(valid.any().item()):
                return {"miss_head_ce": region_logits.sum() * 0.0}
            logits = region_logits[valid]
            filtered_targets = targets[valid]
            ce = self._cross_entropy(logits, filtered_targets, include_none_weight=True)
        else:
            ce = self._cross_entropy(region_logits, targets, include_none_weight=True)
        return {"miss_head_ce": ce * float(self.config.loss_weight)}

    def _cross_entropy(
        self,
        region_logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        include_none_weight: bool,
    ) -> torch.Tensor:
        weight = self._loss_weight(
            targets,
            dtype=region_logits.dtype,
            device=region_logits.device,
            num_classes=int(region_logits.shape[1]),
            include_none_weight=include_none_weight,
        )
        if weight is not None:
            target_weights = weight.gather(0, targets)
            if not bool((target_weights > 0).any().item()):
                return region_logits.sum() * 0.0
        return F.cross_entropy(region_logits, targets, weight=weight)

    def _loss_weight(
        self,
        targets: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
        num_classes: int,
        include_none_weight: bool,
    ) -> torch.Tensor | None:
        use_class_balance = bool(self.config.class_balanced_loss)
        use_none_weight = include_none_weight and float(self.config.none_loss_weight) != 1.0
        if not use_class_balance and not use_none_weight:
            return None

        if use_class_balance:
            counts = torch.bincount(
                targets.detach(),
                minlength=num_classes,
            ).to(device=device, dtype=dtype)
            present = counts > 0
            num_present = present.sum().to(dtype=dtype).clamp_min(1.0)
            total = counts.sum().clamp_min(1.0)
            weight = torch.zeros((num_classes,), dtype=dtype, device=device)
            weight[present] = total / (num_present * counts[present].clamp_min(1.0))
        else:
            weight = torch.ones((num_classes,), dtype=dtype, device=device)

        if use_none_weight:
            weight[0] = weight[0] * float(self.config.none_loss_weight)
        return weight

    def _compute_split_loss(
        self,
        miss_head_output: torch.Tensor | Mapping[str, torch.Tensor],
        target_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = _require_split_output(miss_head_output)
        has_miss_logits = output["has_miss_logits"]
        region_logits = output["region_logits"]
        targets = target_labels.to(device=has_miss_logits.device, dtype=torch.long)
        has_miss_targets = (targets > 0).to(dtype=has_miss_logits.dtype)
        pos_weight = self._has_miss_pos_weight(
            has_miss_targets,
            dtype=has_miss_logits.dtype,
            device=has_miss_logits.device,
        )
        has_miss_loss = F.binary_cross_entropy_with_logits(
            has_miss_logits,
            has_miss_targets,
            pos_weight=pos_weight,
        )

        positive = targets > 0
        if bool(positive.any().item()):
            region_targets = targets[positive] - 1
            region_loss = self._cross_entropy(
                region_logits[positive],
                region_targets,
                include_none_weight=False,
            )
        else:
            region_loss = region_logits.sum() * 0.0

        return {
            "miss_head_has_miss_bce": has_miss_loss * float(self.config.has_miss_loss_weight),
            "miss_head_region_ce": region_loss * float(self.config.region_loss_weight),
        }

    def _has_miss_pos_weight(
        self,
        targets: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        value = self.config.has_miss_pos_weight
        if value is None:
            return None
        if isinstance(value, str):
            positives = targets.detach().sum().to(dtype=dtype)
            total = torch.as_tensor(targets.numel(), dtype=dtype, device=device)
            negatives = total - positives
            if float(positives.item()) <= 0.0 or float(negatives.item()) <= 0.0:
                return torch.ones((1,), dtype=dtype, device=device)
            return (negatives / positives.clamp_min(1.0)).reshape(1)
        return torch.as_tensor([float(value)], dtype=dtype, device=device)

    @torch.no_grad()
    def compute_metrics(
        self,
        miss_head_output: torch.Tensor | Mapping[str, torch.Tensor],
        target_labels: torch.Tensor,
    ) -> dict[str, float]:
        device = self.output_device(miss_head_output)
        targets = target_labels.to(device=device, dtype=torch.long)
        preds = self.predict_region(miss_head_output)
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

        if bool(self.config.has_miss_head):
            output = _require_split_output(miss_head_output)
            has_miss_logits = output["has_miss_logits"]
            region_logits = output["region_logits"]
            has_miss_prob = has_miss_logits.sigmoid()
            has_miss_pred = has_miss_prob >= float(self.config.has_miss_threshold)
            has_miss_target = targets > 0
            has_miss_correct = has_miss_pred.eq(has_miss_target)
            metrics["miss_head_has_miss_acc"] = float(has_miss_correct.float().mean().item())
            metrics["miss_head_has_miss_target_ratio"] = float(has_miss_target.float().mean().item())
            metrics["miss_head_has_miss_pred_ratio"] = float(has_miss_pred.float().mean().item())
            if bool(has_miss_pred.any().item()):
                metrics["miss_head_has_miss_precision"] = float(
                    has_miss_target[has_miss_pred].float().mean().item()
                )
            if bool(has_miss_target.any().item()):
                metrics["miss_head_has_miss_recall"] = float(
                    has_miss_pred[has_miss_target].float().mean().item()
                )
                region_preds = torch.argmax(region_logits, dim=1) + 1
                metrics["miss_head_region_acc"] = float(
                    region_preds[has_miss_target].eq(targets[has_miss_target]).float().mean().item()
                )
            precision = metrics.get("miss_head_has_miss_precision")
            recall = metrics.get("miss_head_has_miss_recall")
            if precision is not None and recall is not None and precision + recall > 0.0:
                metrics["miss_head_has_miss_f1"] = 2.0 * precision * recall / (precision + recall)

            binary_probs = torch.stack((1.0 - has_miss_prob, has_miss_prob), dim=1).clamp_min(1e-12)
            binary_entropy = -(binary_probs * binary_probs.log()).sum(dim=1).mean()
            metrics["miss_head_has_miss_entropy"] = float(binary_entropy.item())

            probabilities = region_logits.softmax(dim=1).clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
            metrics["miss_head_entropy"] = float(entropy.item())
            metrics["miss_head_region_entropy"] = float(entropy.item())
        else:
            region_logits = _require_tensor_output(miss_head_output)
            probabilities = region_logits.softmax(dim=1).clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
            metrics["miss_head_entropy"] = float(entropy.item())
        return metrics

    def predict_region(self, miss_head_output: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if bool(self.config.has_miss_head):
            output = _require_split_output(miss_head_output)
            has_miss = output["has_miss_logits"].sigmoid() >= float(self.config.has_miss_threshold)
            regions = torch.argmax(output["region_logits"], dim=1) + 1
            return torch.where(has_miss, regions, torch.zeros_like(regions))
        return torch.argmax(_require_tensor_output(miss_head_output), dim=1)

    def output_device(self, miss_head_output: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.device:
        if isinstance(miss_head_output, torch.Tensor):
            return miss_head_output.device
        return _require_split_output(miss_head_output)["region_logits"].device

    def _build_legacy_classifier(self) -> nn.Sequential:
        layers, output_dim = self._build_hidden_layers()
        layers.append(nn.Linear(output_dim, self.config.num_labels))
        return nn.Sequential(*layers)

    def _build_trunk(self) -> tuple[nn.Module, int]:
        layers, output_dim = self._build_hidden_layers()
        if not layers:
            return nn.Identity(), output_dim
        return nn.Sequential(*layers), output_dim

    def _build_hidden_layers(self) -> tuple[list[nn.Module], int]:
        if int(self.config.num_layers) == 1:
            return [], int(self.config.in_channels)

        layers: list[nn.Module] = []
        layers.append(nn.Linear(self.config.in_channels, self.config.hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if float(self.config.dropout) > 0.0:
            layers.append(nn.Dropout(p=float(self.config.dropout)))
        for _ in range(int(self.config.num_layers) - 2):
            layers.append(nn.Linear(self.config.hidden_dim, self.config.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if float(self.config.dropout) > 0.0:
                layers.append(nn.Dropout(p=float(self.config.dropout)))
        return layers, int(self.config.hidden_dim)

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


def _parse_pos_weight(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if normalized == "auto":
            return "auto"
        return float(normalized)
    return float(value)


def _require_tensor_output(output: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError("Legacy MissHead mode expects a tensor output.")


def _require_split_output(output: torch.Tensor | Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    if not isinstance(output, Mapping):
        raise TypeError("Split MissHead mode expects a mapping output.")
    if "has_miss_logits" not in output or "region_logits" not in output:
        raise KeyError("Split MissHead output must contain 'has_miss_logits' and 'region_logits'.")
    return output


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
