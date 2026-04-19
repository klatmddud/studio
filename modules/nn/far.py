from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from .mdmb import (
    MissedDetectionMemoryBank,
    _GTRecord,
    normalize_arch,
    normalize_xyxy_boxes,
)


@dataclass(frozen=True, slots=True)
class FARConfig:
    """Configuration for the Forgetting-Aware feature Replay module."""

    enabled: bool = False
    lambda_far: float = 0.1
    anchor_ema_mu: float = 0.9
    persistence_gamma: float = 1.0
    min_relapse_streak: int = 1
    match_threshold: float = 0.95
    warmup_epochs: int = 1
    max_anchors_per_image: int | None = None
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
    ) -> "FARConfig":
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
        feature_keys = tuple(str(key) for key in feature_keys_raw)

        max_anchors = _pick("max_anchors_per_image", None)
        config = cls(
            enabled=bool(_pick("enabled", False)),
            lambda_far=float(_pick("lambda_far", 0.1)),
            anchor_ema_mu=float(_pick("anchor_ema_mu", 0.9)),
            persistence_gamma=float(_pick("persistence_gamma", 1.0)),
            min_relapse_streak=int(_pick("min_relapse_streak", 1)),
            match_threshold=float(_pick("match_threshold", 0.95)),
            warmup_epochs=int(_pick("warmup_epochs", 1)),
            max_anchors_per_image=None if max_anchors is None else int(max_anchors),
            feature_keys=feature_keys,
            roi_output_size=int(_pick("roi_output_size", 7)),
            roi_sampling_ratio=int(_pick("roi_sampling_ratio", 2)),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.lambda_far < 0.0:
            raise ValueError("FAR lambda_far must be >= 0.")
        if not 0.0 <= self.anchor_ema_mu <= 1.0:
            raise ValueError("FAR anchor_ema_mu must satisfy 0 <= mu <= 1.")
        if self.persistence_gamma < 0.0:
            raise ValueError("FAR persistence_gamma must be >= 0.")
        if self.min_relapse_streak < 1:
            raise ValueError("FAR min_relapse_streak must be >= 1.")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("FAR match_threshold must satisfy 0 <= threshold <= 1.")
        if self.warmup_epochs < 0:
            raise ValueError("FAR warmup_epochs must be >= 0.")
        if self.max_anchors_per_image is not None and self.max_anchors_per_image < 1:
            raise ValueError("FAR max_anchors_per_image must be >= 1 when provided.")
        if not self.feature_keys:
            raise ValueError("FAR feature_keys must contain at least one key.")
        if self.roi_output_size < 1:
            raise ValueError("FAR roi_output_size must be >= 1.")
        if self.roi_sampling_ratio < 0:
            raise ValueError("FAR roi_sampling_ratio must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lambda_far": self.lambda_far,
            "anchor_ema_mu": self.anchor_ema_mu,
            "persistence_gamma": self.persistence_gamma,
            "min_relapse_streak": self.min_relapse_streak,
            "match_threshold": self.match_threshold,
            "warmup_epochs": self.warmup_epochs,
            "max_anchors_per_image": self.max_anchors_per_image,
            "feature_keys": list(self.feature_keys),
            "roi_output_size": self.roi_output_size,
            "roi_sampling_ratio": self.roi_sampling_ratio,
            "arch": self.arch,
        }


@dataclass(slots=True)
class _AnchorRecord:
    """Per-GT feature anchor with freeze state for relapse handling."""

    class_id: int
    bbox: torch.Tensor          # normalized xyxy, shape [4], cpu
    anchor: torch.Tensor        # [D], L2-normalized, cpu
    last_updated_epoch: int
    frozen: bool

    def to_state(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "bbox": self.bbox.cpu().tolist(),
            "anchor": self.anchor.cpu().tolist(),
            "last_updated_epoch": self.last_updated_epoch,
            "frozen": self.frozen,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "_AnchorRecord":
        return cls(
            class_id=int(state["class_id"]),
            bbox=torch.tensor(state["bbox"], dtype=torch.float32),
            anchor=torch.tensor(state["anchor"], dtype=torch.float32),
            last_updated_epoch=int(state["last_updated_epoch"]),
            frozen=bool(state.get("frozen", False)),
        )


class ForgettingAwareReplay(nn.Module):
    """
    FAR — Forgetting-Aware feature Replay.

    Stores a per-GT feature anchor captured when the object was last successfully
    detected. On relapse (the object was previously detected but is currently
    being missed) the anchor is frozen and a cosine-distance consistency loss
    pulls the current feature back toward the frozen anchor. Anchor updates,
    freezing, and loss contributions are all training-only; inference is
    unaffected.

    FAR depends on MDMB's temporal tracking (`_gt_records`) to determine the
    detected/missed/relapse state of each GT across epochs.
    """

    def __init__(self, config: FARConfig) -> None:
        super().__init__()
        self.config = config
        self._anchors: dict[str, list[_AnchorRecord]] = {}
        self.current_epoch = 0
        self._pool = MultiScaleRoIAlign(
            featmap_names=list(config.feature_keys),
            output_size=config.roi_output_size,
            sampling_ratio=config.roi_sampling_ratio,
        )

    def forward(self, features):
        return features

    # ---------------------------------------------------------- epoch lifecycle

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def should_apply(self, *, epoch: int | None = None) -> bool:
        if epoch is not None:
            self.current_epoch = int(epoch)
        if not self.config.enabled:
            return False
        return self.current_epoch > self.config.warmup_epochs

    def reset(self) -> None:
        self._anchors.clear()
        self.current_epoch = 0

    # --------------------------------------------------------- training-time loss

    def compute_loss(
        self,
        *,
        image_ids: Sequence[Any],
        gt_boxes_list: Sequence[torch.Tensor],
        gt_labels_list: Sequence[torch.Tensor],
        features: Mapping[str, torch.Tensor],
        image_shapes: Sequence[Sequence[int]],
        mdmb: MissedDetectionMemoryBank | None,
    ) -> torch.Tensor:
        """
        Compute the FAR consistency loss for the current training batch.

        Returns a scalar tensor on the feature device. Zero when FAR is in
        warmup, when there are no relapsed GTs with a frozen anchor, or when
        MDMB has no temporal records yet.
        """
        feature_device = self._infer_device(features)
        zero = torch.zeros((), device=feature_device, dtype=torch.float32)
        if not self.should_apply() or mdmb is None:
            return zero
        if not self._anchors:
            return zero

        batch_size = len(image_ids)
        _validate_batch_lengths(
            batch_size=batch_size,
            lengths={
                "gt_boxes_list": len(gt_boxes_list),
                "gt_labels_list": len(gt_labels_list),
                "image_shapes": len(image_shapes),
            },
        )

        pooled_per_image = self._pool_features(
            features=features,
            boxes_per_image=[_as_float_tensor(boxes) for boxes in gt_boxes_list],
            image_shapes=[tuple(int(v) for v in shape) for shape in image_shapes],
        )

        total_contribution = zero
        num_relapse = 0

        for image_id, gt_boxes, gt_labels, pooled, image_shape in zip(
            image_ids,
            gt_boxes_list,
            gt_labels_list,
            pooled_per_image,
            image_shapes,
            strict=True,
        ):
            if pooled.numel() == 0:
                continue
            image_key = _normalize_image_id(image_id)
            anchor_records = self._anchors.get(image_key)
            if not anchor_records:
                continue
            records_for_image = _gt_records_for_image(mdmb, image_key)
            if not records_for_image:
                continue

            gt_boxes_norm = normalize_xyxy_boxes(gt_boxes, image_shape)
            gt_labels_tensor = _as_int_tensor(gt_labels)
            anchor_matches = _match_boxes_to_records(
                gt_boxes_norm=gt_boxes_norm,
                gt_labels=gt_labels_tensor,
                records=anchor_records,
                iou_thresh=self.config.match_threshold,
            )
            mdmb_matches = _match_boxes_to_records(
                gt_boxes_norm=gt_boxes_norm,
                gt_labels=gt_labels_tensor,
                records=records_for_image,
                iou_thresh=self.config.match_threshold,
            )

            for gt_index in range(gt_boxes_norm.shape[0]):
                anchor_idx = anchor_matches[gt_index]
                mdmb_idx = mdmb_matches[gt_index]
                if anchor_idx is None or mdmb_idx is None:
                    continue
                mdmb_record = records_for_image[mdmb_idx]
                if not self._is_relapse(mdmb_record):
                    continue
                anchor_record = anchor_records[anchor_idx]
                if not anchor_record.frozen:
                    continue

                feature_vec = pooled[gt_index]
                feature_norm = F.normalize(feature_vec, dim=-1, eps=1e-6)
                anchor_vec = anchor_record.anchor.to(
                    device=feature_norm.device, dtype=feature_norm.dtype
                )
                cos_sim = torch.dot(feature_norm, anchor_vec)
                streak_weight = 1.0 + self.config.persistence_gamma * self._streak_ratio(
                    mdmb_record.consecutive_miss_count, mdmb
                )
                total_contribution = total_contribution + streak_weight * (1.0 - cos_sim)
                num_relapse += 1

        if num_relapse == 0:
            return zero
        return self.config.lambda_far * total_contribution / float(num_relapse)

    # ---------------------------------------------------- anchor bank maintenance

    @torch.no_grad()
    def update_anchors(
        self,
        *,
        image_ids: Sequence[Any],
        gt_boxes_list: Sequence[torch.Tensor],
        gt_labels_list: Sequence[torch.Tensor],
        features: Mapping[str, torch.Tensor],
        image_shapes: Sequence[Sequence[int]],
        mdmb: MissedDetectionMemoryBank | None,
        epoch: int | None = None,
    ) -> None:
        """
        Update the anchor bank from the post-optimizer-step view of the batch.

        Called after MDMB has been updated for the same batch so that
        `_gt_records` reflects the latest detected/miss state.
        """
        if epoch is not None:
            self.current_epoch = int(epoch)
        if not self.should_apply() or mdmb is None:
            return

        batch_size = len(image_ids)
        _validate_batch_lengths(
            batch_size=batch_size,
            lengths={
                "gt_boxes_list": len(gt_boxes_list),
                "gt_labels_list": len(gt_labels_list),
                "image_shapes": len(image_shapes),
            },
        )

        pooled_per_image = self._pool_features(
            features=features,
            boxes_per_image=[_as_float_tensor(boxes) for boxes in gt_boxes_list],
            image_shapes=[tuple(int(v) for v in shape) for shape in image_shapes],
        )

        for image_id, gt_boxes, gt_labels, pooled, image_shape in zip(
            image_ids,
            gt_boxes_list,
            gt_labels_list,
            pooled_per_image,
            image_shapes,
            strict=True,
        ):
            image_key = _normalize_image_id(image_id)
            gt_boxes_tensor = _as_float_tensor(gt_boxes)
            if gt_boxes_tensor.numel() == 0:
                self._anchors.pop(image_key, None)
                continue

            records_for_image = _gt_records_for_image(mdmb, image_key)
            if not records_for_image:
                self._anchors.pop(image_key, None)
                continue
            gt_boxes_norm = normalize_xyxy_boxes(gt_boxes_tensor, image_shape)
            gt_labels_tensor = _as_int_tensor(gt_labels)
            mdmb_matches = _match_boxes_to_records(
                gt_boxes_norm=gt_boxes_norm,
                gt_labels=gt_labels_tensor,
                records=records_for_image,
                iou_thresh=self.config.match_threshold,
            )

            existing_records = self._anchors.get(image_key, [])
            anchor_matches = _match_boxes_to_records(
                gt_boxes_norm=gt_boxes_norm,
                gt_labels=gt_labels_tensor,
                records=existing_records,
                iou_thresh=self.config.match_threshold,
            )

            new_records: list[_AnchorRecord] = []
            for gt_index in range(gt_boxes_norm.shape[0]):
                mdmb_idx = mdmb_matches[gt_index]
                if mdmb_idx is None:
                    anchor_idx = anchor_matches[gt_index]
                    if anchor_idx is not None:
                        new_records.append(existing_records[anchor_idx])
                    continue
                mdmb_record = records_for_image[mdmb_idx]
                feature_vec = pooled[gt_index]
                feature_norm = F.normalize(feature_vec, dim=-1, eps=1e-6).detach().cpu()

                anchor_idx = anchor_matches[gt_index]
                prev_record = existing_records[anchor_idx] if anchor_idx is not None else None
                updated = self._step_anchor(
                    class_id=int(gt_labels_tensor[gt_index].item()),
                    bbox_norm=gt_boxes_norm[gt_index].detach().cpu(),
                    current_feature=feature_norm,
                    mdmb_record=mdmb_record,
                    prev_record=prev_record,
                )
                if updated is not None:
                    new_records.append(updated)

            if self.config.max_anchors_per_image is not None:
                new_records = new_records[: self.config.max_anchors_per_image]

            if new_records:
                self._anchors[image_key] = new_records
            else:
                self._anchors.pop(image_key, None)

    # ---------------------------------------------------------- helpers (public)

    def summary(self) -> dict[str, Any]:
        num_anchors = 0
        num_frozen = 0
        for records in self._anchors.values():
            for record in records:
                num_anchors += 1
                if record.frozen:
                    num_frozen += 1
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_active": self.config.enabled
            and self.current_epoch <= self.config.warmup_epochs,
            "num_images": len(self._anchors),
            "num_anchors": num_anchors,
            "num_frozen": num_frozen,
        }

    def items(self) -> Iterator[tuple[str, list[_AnchorRecord]]]:
        for image_id, records in self._anchors.items():
            yield image_id, list(records)

    def __len__(self) -> int:
        return sum(len(records) for records in self._anchors.values())

    def extra_repr(self) -> str:
        return (
            f"enabled={self.config.enabled}, arch={self.config.arch!r}, "
            f"images={len(self._anchors)}, anchors={len(self)}"
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "anchors": {
                image_id: [record.to_state() for record in records]
                for image_id, records in self._anchors.items()
            },
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            self.reset()
            return

        config_state = state.get("config", {})
        if isinstance(config_state, Mapping):
            try:
                self.config = FARConfig.from_mapping(config_state, arch=config_state.get("arch"))
            except Exception:
                pass
        self.current_epoch = int(state.get("current_epoch", 0))

        if int(state.get("version", 0)) != 1:
            self._anchors = {}
            return

        raw_anchors = state.get("anchors", {})
        if not isinstance(raw_anchors, Mapping):
            self._anchors = {}
            return

        restored: dict[str, list[_AnchorRecord]] = {}
        for image_id, raw_records in raw_anchors.items():
            if not isinstance(raw_records, Sequence):
                continue
            image_key = _normalize_image_id(image_id)
            records: list[_AnchorRecord] = []
            for raw_record in raw_records:
                if not isinstance(raw_record, Mapping):
                    continue
                records.append(_AnchorRecord.from_state(raw_record))
            if records:
                restored[image_key] = records
        self._anchors = restored

    # ---------------------------------------------------------- helpers (private)

    def _is_relapse(self, record: _GTRecord) -> bool:
        if record.last_detected_epoch is None:
            return False
        return record.consecutive_miss_count >= self.config.min_relapse_streak

    def _streak_ratio(
        self,
        streak: int,
        mdmb: MissedDetectionMemoryBank,
    ) -> float:
        global_max = int(getattr(mdmb, "_global_max_consecutive_miss", 0))
        if global_max <= 0:
            return 0.0
        return float(streak) / float(global_max)

    def _step_anchor(
        self,
        *,
        class_id: int,
        bbox_norm: torch.Tensor,
        current_feature: torch.Tensor,
        mdmb_record: _GTRecord,
        prev_record: _AnchorRecord | None,
    ) -> _AnchorRecord | None:
        detected = mdmb_record.consecutive_miss_count == 0
        missed = mdmb_record.consecutive_miss_count > 0

        if detected:
            if prev_record is None:
                return _AnchorRecord(
                    class_id=class_id,
                    bbox=bbox_norm,
                    anchor=current_feature,
                    last_updated_epoch=self.current_epoch,
                    frozen=False,
                )

            if prev_record.frozen:
                # Recovery: drop the frozen anchor and re-seed from the current
                # successful detection. The model has re-learned the object, so
                # the frozen historical representation is no longer the target.
                return _AnchorRecord(
                    class_id=class_id,
                    bbox=bbox_norm,
                    anchor=current_feature,
                    last_updated_epoch=self.current_epoch,
                    frozen=False,
                )

            mu = self.config.anchor_ema_mu
            blended = mu * prev_record.anchor + (1.0 - mu) * current_feature
            blended = F.normalize(blended, dim=-1, eps=1e-6)
            return _AnchorRecord(
                class_id=class_id,
                bbox=bbox_norm,
                anchor=blended,
                last_updated_epoch=self.current_epoch,
                frozen=False,
            )

        if missed:
            if prev_record is None:
                # We have no remembered representation for this GT, so there is
                # nothing to pull back toward. Stay silent until it is detected
                # at least once.
                return None

            should_freeze = (
                mdmb_record.last_detected_epoch is not None
                and mdmb_record.consecutive_miss_count >= self.config.min_relapse_streak
            )
            return _AnchorRecord(
                class_id=class_id,
                bbox=prev_record.bbox,
                anchor=prev_record.anchor,
                last_updated_epoch=prev_record.last_updated_epoch,
                frozen=bool(should_freeze or prev_record.frozen),
            )

        return prev_record

    def _pool_features(
        self,
        *,
        features: Mapping[str, torch.Tensor],
        boxes_per_image: Sequence[torch.Tensor],
        image_shapes: Sequence[tuple[int, int]],
    ) -> list[torch.Tensor]:
        feature_map: dict[str, torch.Tensor] = {}
        for key in self.config.feature_keys:
            if key not in features:
                raise KeyError(
                    f"FAR expected feature map {key!r} in backbone output. "
                    f"Available keys: {list(features.keys())}"
                )
            feature_map[key] = features[key]

        counts = [int(boxes.shape[0]) for boxes in boxes_per_image]
        total = sum(counts)
        any_feature = next(iter(feature_map.values()))
        if total == 0:
            empty = any_feature.new_zeros((0, any_feature.shape[1]))
            return [
                empty.new_zeros((0, any_feature.shape[1])) for _ in boxes_per_image
            ]

        pooled = self._pool(feature_map, list(boxes_per_image), list(image_shapes))
        if pooled.ndim != 4:
            raise RuntimeError(
                "FAR expected MultiScaleRoIAlign to return a 4D tensor, "
                f"got shape {tuple(pooled.shape)}."
            )
        pooled_flat = pooled.flatten(2).mean(dim=-1)  # [total, C]
        return list(pooled_flat.split(counts, dim=0))

    def _infer_device(
        self, features: Mapping[str, torch.Tensor]
    ) -> torch.device:
        for tensor in features.values():
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        return torch.device("cpu")


FAR = ForgettingAwareReplay


# ------------------------------------------------------------------- factories


def load_far_config(path: str | Path, *, arch: str | None = None) -> FARConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"FAR YAML must contain a mapping at the top level: {config_path}")
    return FARConfig.from_mapping(raw, arch=arch)


def build_far_from_config(
    raw_config: Mapping[str, Any] | FARConfig,
    *,
    arch: str | None = None,
) -> ForgettingAwareReplay | None:
    config = (
        raw_config
        if isinstance(raw_config, FARConfig)
        else FARConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return ForgettingAwareReplay(config)


def build_far_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> ForgettingAwareReplay | None:
    config = load_far_config(path, arch=arch)
    if not config.enabled:
        return None
    return ForgettingAwareReplay(config)


# ------------------------------------------------------------------- utilities


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("FAR image_id tensor must contain a single scalar value.")
        value = value.item()
    return str(value)


def _as_float_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor.reshape(-1, 4)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != 4:
        raise ValueError("FAR boxes must have shape [N, 4] or [4].")
    return tensor


def _as_int_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.int64).flatten()


def _validate_batch_lengths(*, batch_size: int, lengths: Mapping[str, int]) -> None:
    mismatches = {name: length for name, length in lengths.items() if length != batch_size}
    if mismatches:
        raise ValueError(
            "FAR batch inputs must share the same batch dimension. "
            f"Expected {batch_size}, got {mismatches}."
        )


def _gt_records_for_image(
    mdmb: MissedDetectionMemoryBank | None,
    image_key: str,
) -> list[_GTRecord]:
    if mdmb is None:
        return []
    records_map = getattr(mdmb, "_gt_records", None)
    if not isinstance(records_map, Mapping):
        return []
    return list(records_map.get(image_key, ()))


def _match_boxes_to_records(
    *,
    gt_boxes_norm: torch.Tensor,
    gt_labels: torch.Tensor,
    records: Sequence[_GTRecord | _AnchorRecord],
    iou_thresh: float,
) -> list[int | None]:
    num_gts = int(gt_boxes_norm.shape[0])
    matches: list[int | None] = [None] * num_gts
    if num_gts == 0 or not records:
        return matches

    record_boxes = torch.stack(
        [record.bbox.to(dtype=gt_boxes_norm.dtype) for record in records],
        dim=0,
    )
    record_classes = torch.as_tensor(
        [int(record.class_id) for record in records],
        dtype=torch.int64,
    )

    ious = box_ops.box_iou(gt_boxes_norm.cpu(), record_boxes.cpu())
    gt_labels_cpu = gt_labels.detach().cpu()
    claimed = torch.zeros((len(records),), dtype=torch.bool)
    for gt_index in range(num_gts):
        row = ious[gt_index]
        class_mask = record_classes == gt_labels_cpu[gt_index].item()
        available_mask = class_mask & ~claimed
        if not bool(available_mask.any().item()):
            continue
        masked = row.clone()
        masked[~available_mask] = -1.0
        best_value, best_idx = masked.max(dim=0)
        if float(best_value.item()) < float(iou_thresh):
            continue
        record_index = int(best_idx.item())
        matches[gt_index] = record_index
        claimed[record_index] = True

    return matches
