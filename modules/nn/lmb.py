from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torchvision.ops import boxes as box_ops

from .common import normalize_arch

MISSING_STATE = "missing"
LOW_IOU_STATE = "low_iou"
GOOD_STATE = "good"
UNKNOWN_STATE = "unknown"


@dataclass(frozen=True, slots=True)
class LMBMatchingConfig:
    score_threshold: float | str = "auto"
    low_iou_threshold: float = 0.5
    good_iou_threshold: float = 0.75
    class_aware: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "LMBMatchingConfig":
        data = dict(raw or {})
        config = cls(
            score_threshold=_parse_threshold(data.get("score_threshold", "auto")),
            low_iou_threshold=float(data.get("low_iou_threshold", 0.5)),
            good_iou_threshold=float(data.get("good_iou_threshold", 0.75)),
            class_aware=bool(data.get("class_aware", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if _is_auto_threshold(self.score_threshold):
            pass
        elif not 0.0 <= float(self.score_threshold) <= 1.0:
            raise ValueError("LMB matching.score_threshold must be 'auto' or satisfy 0 <= value <= 1.")
        if not 0.0 <= float(self.low_iou_threshold) <= 1.0:
            raise ValueError("LMB matching.low_iou_threshold must satisfy 0 <= value <= 1.")
        if not 0.0 <= float(self.good_iou_threshold) <= 1.0:
            raise ValueError("LMB matching.good_iou_threshold must satisfy 0 <= value <= 1.")
        if float(self.low_iou_threshold) >= float(self.good_iou_threshold):
            raise ValueError("LMB matching.low_iou_threshold must be smaller than good_iou_threshold.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_threshold": self.score_threshold,
            "low_iou_threshold": self.low_iou_threshold,
            "good_iou_threshold": self.good_iou_threshold,
            "class_aware": self.class_aware,
        }


@dataclass(frozen=True, slots=True)
class LMBStabilityConfig:
    stable_epochs: int = 3
    hotspot_top_k: int = 10

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "LMBStabilityConfig":
        data = dict(raw or {})
        config = cls(
            stable_epochs=int(data.get("stable_epochs", 3)),
            hotspot_top_k=int(data.get("hotspot_top_k", 10)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.stable_epochs) < 1:
            raise ValueError("LMB stability.stable_epochs must be >= 1.")
        if int(self.hotspot_top_k) < 1:
            raise ValueError("LMB stability.hotspot_top_k must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_epochs": self.stable_epochs,
            "hotspot_top_k": self.hotspot_top_k,
        }


@dataclass(frozen=True, slots=True)
class LMBConfig:
    enabled: bool = False
    grid_size: int = 2
    start_epoch: int = 1
    matching: LMBMatchingConfig = field(default_factory=LMBMatchingConfig)
    stability: LMBStabilityConfig = field(default_factory=LMBStabilityConfig)
    max_records: int | None = None
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "LMBConfig":
        data = dict(raw or {})
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

        config = cls(
            enabled=bool(merged.get("enabled", False)),
            grid_size=int(merged.get("grid_size", 2)),
            start_epoch=int(merged.get("start_epoch", 1)),
            matching=LMBMatchingConfig.from_mapping(merged.get("matching")),
            stability=LMBStabilityConfig.from_mapping(merged.get("stability")),
            max_records=None if merged.get("max_records") is None else int(merged.get("max_records")),
            arch=normalized_arch,
        )
        config.validate()
        return config

    @property
    def num_regions(self) -> int:
        return int(self.grid_size) * int(self.grid_size)

    def validate(self) -> None:
        if int(self.grid_size) < 1:
            raise ValueError("LMB grid_size must be >= 1.")
        if int(self.start_epoch) < 0:
            raise ValueError("LMB start_epoch must be >= 0.")
        if self.max_records is not None and int(self.max_records) < 1:
            raise ValueError("LMB max_records must be null or >= 1.")

    def resolve_detector_thresholds(
        self,
        *,
        detector_score_threshold: float | None,
    ) -> "LMBConfig":
        score_threshold = self.matching.score_threshold
        if _is_auto_threshold(score_threshold):
            if detector_score_threshold is None:
                raise ValueError("LMB matching.score_threshold='auto' requires the detector final score threshold.")
            score_threshold = float(detector_score_threshold)
        if score_threshold == self.matching.score_threshold:
            return self
        matching = replace(self.matching, score_threshold=score_threshold)
        matching.validate()
        return replace(self, matching=matching)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "grid_size": self.grid_size,
            "start_epoch": self.start_epoch,
            "matching": self.matching.to_dict(),
            "stability": self.stability.to_dict(),
            "max_records": self.max_records,
            "arch": self.arch,
        }


@dataclass(slots=True)
class LMBRecord:
    record_key: str
    image_id: str
    gt_id: str | None
    gt_class: int
    bbox_xyxy: tuple[float, float, float, float]
    region_id: int
    state: str = UNKNOWN_STATE
    previous_state: str | None = None
    last_epoch: int = 0
    last_step: int = 0
    best_iou: float | None = None
    best_iou_score: float | None = None
    best_score: float | None = None
    best_score_iou: float | None = None
    total_seen: int = 0
    total_missing: int = 0
    total_low_iou: int = 0
    total_good: int = 0
    current_state_streak: int = 0
    low_iou_streak: int = 0
    max_low_iou_streak: int = 0
    missing_streak: int = 0
    max_missing_streak: int = 0
    good_streak: int = 0
    max_good_streak: int = 0
    low_iou_deficit_sum: float = 0.0

    def update(
        self,
        *,
        state: str,
        bbox_xyxy: tuple[float, float, float, float],
        region_id: int,
        epoch: int,
        step: int,
        best_iou: float | None,
        best_iou_score: float | None,
        best_score: float | None,
        best_score_iou: float | None,
        good_iou_threshold: float,
    ) -> None:
        previous = self.state
        self.previous_state = None if previous == UNKNOWN_STATE else previous
        self.state = _normalize_state(state)
        self.bbox_xyxy = bbox_xyxy
        self.region_id = int(region_id)
        self.last_epoch = int(epoch)
        self.last_step = int(step)
        self.best_iou = None if best_iou is None else float(best_iou)
        self.best_iou_score = None if best_iou_score is None else float(best_iou_score)
        self.best_score = None if best_score is None else float(best_score)
        self.best_score_iou = None if best_score_iou is None else float(best_score_iou)
        self.total_seen += 1
        self.current_state_streak = self.current_state_streak + 1 if previous == self.state else 1

        if self.state == MISSING_STATE:
            self.total_missing += 1
            self.missing_streak = self.missing_streak + 1 if previous == MISSING_STATE else 1
            self.low_iou_streak = 0
            self.good_streak = 0
            self.max_missing_streak = max(self.max_missing_streak, self.missing_streak)
        elif self.state == LOW_IOU_STATE:
            self.total_low_iou += 1
            self.low_iou_streak = self.low_iou_streak + 1 if previous == LOW_IOU_STATE else 1
            self.missing_streak = 0
            self.good_streak = 0
            self.max_low_iou_streak = max(self.max_low_iou_streak, self.low_iou_streak)
            if best_iou is not None:
                self.low_iou_deficit_sum += max(0.0, float(good_iou_threshold) - float(best_iou))
        elif self.state == GOOD_STATE:
            self.total_good += 1
            self.good_streak = self.good_streak + 1 if previous == GOOD_STATE else 1
            self.missing_streak = 0
            self.low_iou_streak = 0
            self.max_good_streak = max(self.max_good_streak, self.good_streak)

    def is_stable_low_iou(self, *, stable_epochs: int) -> bool:
        return self.state == LOW_IOU_STATE and int(self.low_iou_streak) >= int(stable_epochs)

    def to_state(self) -> dict[str, Any]:
        return {
            "record_key": self.record_key,
            "image_id": self.image_id,
            "gt_id": self.gt_id,
            "gt_class": self.gt_class,
            "bbox_xyxy": list(self.bbox_xyxy),
            "region_id": self.region_id,
            "state": self.state,
            "previous_state": self.previous_state,
            "last_epoch": self.last_epoch,
            "last_step": self.last_step,
            "best_iou": self.best_iou,
            "best_iou_score": self.best_iou_score,
            "best_score": self.best_score,
            "best_score_iou": self.best_score_iou,
            "total_seen": self.total_seen,
            "total_missing": self.total_missing,
            "total_low_iou": self.total_low_iou,
            "total_good": self.total_good,
            "current_state_streak": self.current_state_streak,
            "low_iou_streak": self.low_iou_streak,
            "max_low_iou_streak": self.max_low_iou_streak,
            "missing_streak": self.missing_streak,
            "max_missing_streak": self.max_missing_streak,
            "good_streak": self.good_streak,
            "max_good_streak": self.max_good_streak,
            "low_iou_deficit_sum": self.low_iou_deficit_sum,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "LMBRecord":
        bbox = state.get("bbox_xyxy", (0.0, 0.0, 0.0, 0.0))
        bbox_values = tuple(float(value) for value in list(bbox)[:4])
        if len(bbox_values) != 4:
            bbox_values = (0.0, 0.0, 0.0, 0.0)
        return cls(
            record_key=str(state.get("record_key", "")),
            image_id=str(state.get("image_id", "")),
            gt_id=None if state.get("gt_id") is None else str(state.get("gt_id")),
            gt_class=int(state.get("gt_class", 0)),
            bbox_xyxy=bbox_values,
            region_id=int(state.get("region_id", 1)),
            state=_normalize_state(str(state.get("state", UNKNOWN_STATE))),
            previous_state=None if state.get("previous_state") is None else _normalize_state(str(state.get("previous_state"))),
            last_epoch=int(state.get("last_epoch", 0)),
            last_step=int(state.get("last_step", 0)),
            best_iou=None if state.get("best_iou") is None else float(state.get("best_iou")),
            best_iou_score=None if state.get("best_iou_score") is None else float(state.get("best_iou_score")),
            best_score=None if state.get("best_score") is None else float(state.get("best_score")),
            best_score_iou=None if state.get("best_score_iou") is None else float(state.get("best_score_iou")),
            total_seen=int(state.get("total_seen", 0)),
            total_missing=int(state.get("total_missing", 0)),
            total_low_iou=int(state.get("total_low_iou", 0)),
            total_good=int(state.get("total_good", 0)),
            current_state_streak=int(state.get("current_state_streak", 0)),
            low_iou_streak=int(state.get("low_iou_streak", 0)),
            max_low_iou_streak=int(state.get("max_low_iou_streak", 0)),
            missing_streak=int(state.get("missing_streak", 0)),
            max_missing_streak=int(state.get("max_missing_streak", 0)),
            good_streak=int(state.get("good_streak", 0)),
            max_good_streak=int(state.get("max_good_streak", 0)),
            low_iou_deficit_sum=float(state.get("low_iou_deficit_sum", 0.0)),
        )


class LocalizationMemoryBank(nn.Module):
    """Train-time memory for recurrent low-quality localization GT instances."""

    def __init__(self, config: LMBConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, LMBConfig) else LMBConfig.from_mapping(config or {})
        if _is_auto_threshold(self.config.matching.score_threshold):
            raise ValueError("LMB matching auto thresholds must be resolved before constructing LocalizationMemoryBank.")
        self.current_epoch = 0
        self._records: dict[str, LMBRecord] = {}
        self._image_index: dict[str, set[str]] = defaultdict(set)
        self._stats: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._records)

    @property
    def num_regions(self) -> int:
        return int(self.config.num_regions)

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def is_active(self, epoch: int | None = None) -> bool:
        epoch_value = self.current_epoch if epoch is None else int(epoch)
        return bool(self.config.enabled and int(epoch_value) >= int(self.config.start_epoch))

    def reset(self) -> None:
        self._records.clear()
        self._image_index.clear()
        self._stats.clear()

    def update(
        self,
        *,
        targets: Sequence[Mapping[str, Any]],
        detections: Sequence[Mapping[str, torch.Tensor]],
        epoch: int | None = None,
        step: int = 0,
        image_sizes: Sequence[Sequence[int | float]] | None = None,
    ) -> dict[str, int]:
        if not self.config.enabled:
            return {"enabled": 0}
        if len(targets) != len(detections):
            raise ValueError("LMB update requires the same number of targets and detections.")

        epoch_value = int(self.current_epoch if epoch is None else epoch)
        stats: Counter[str] = Counter()
        for image_index, (target, detection) in enumerate(zip(targets, detections, strict=True)):
            image_size = None if image_sizes is None else image_sizes[image_index]
            stats.update(
                self._update_image(
                    target=target,
                    detection=detection,
                    image_index=image_index,
                    image_size=image_size,
                    epoch=epoch_value,
                    step=int(step),
                )
            )
        self._stats.update(stats)
        self._enforce_max_records()
        return {key: int(value) for key, value in stats.items()}

    def get_records(self, image_id: Any | None = None) -> list[LMBRecord]:
        if image_id is None:
            return list(self._records.values())
        normalized = _normalize_image_id(image_id)
        return [
            self._records[key]
            for key in sorted(self._image_index.get(normalized, set()))
            if key in self._records
        ]

    def summary(self) -> dict[str, Any]:
        records = list(self._records.values())
        low_iou_records = [record for record in records if record.state == LOW_IOU_STATE]
        stable_records = [
            record
            for record in low_iou_records
            if record.is_stable_low_iou(stable_epochs=self.config.stability.stable_epochs)
        ]
        missing_records = [record for record in records if record.state == MISSING_STATE]
        good_records = [record for record in records if record.state == GOOD_STATE]
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "grid_size": self.config.grid_size,
            "num_regions": self.num_regions,
            "num_records": len(records),
            "num_images": len(self._image_index),
            "num_missing": len(missing_records),
            "num_low_iou": len(low_iou_records),
            "num_stable_low_iou": len(stable_records),
            "num_good": len(good_records),
            "low_iou_region_histogram": _string_counter(Counter(record.region_id for record in low_iou_records)),
            "stable_low_iou_region_histogram": _string_counter(Counter(record.region_id for record in stable_records)),
            "class_histogram": _string_counter(Counter(record.gt_class for record in stable_records)),
            "stats": {key: int(value) for key, value in self._stats.items()},
        }

    def epoch_snapshot(
        self,
        epoch: int | None = None,
        *,
        stable_epochs: int | None = None,
        hotspot_top_k: int | None = None,
    ) -> dict[str, Any]:
        epoch_value = int(self.current_epoch if epoch is None else epoch)
        stable_threshold = int(self.config.stability.stable_epochs if stable_epochs is None else stable_epochs)
        top_k = int(self.config.stability.hotspot_top_k if hotspot_top_k is None else hotspot_top_k)
        records = [record for record in self._records.values() if int(record.last_epoch) == epoch_value]
        missing_records = [record for record in records if record.state == MISSING_STATE]
        low_iou_records = [record for record in records if record.state == LOW_IOU_STATE]
        good_records = [record for record in records if record.state == GOOD_STATE]
        stable_low_iou_records = [
            record for record in low_iou_records if record.is_stable_low_iou(stable_epochs=stable_threshold)
        ]
        low_iou_deficits = [
            max(0.0, float(self.config.matching.good_iou_threshold) - float(record.best_iou))
            for record in low_iou_records
            if record.best_iou is not None
        ]
        transition_histogram = Counter(
            f"{record.previous_state or UNKNOWN_STATE}->{record.state}"
            for record in records
        )
        low_iou_streaks = [int(record.low_iou_streak) for record in low_iou_records]
        low_iou_region_histogram = Counter(int(record.region_id) for record in low_iou_records)
        stable_region_histogram = Counter(int(record.region_id) for record in stable_low_iou_records)
        low_iou_hotspots = Counter(_hotspot_key(record.image_id, int(record.region_id)) for record in low_iou_records)
        stable_hotspots = Counter(_hotspot_key(record.image_id, int(record.region_id)) for record in stable_low_iou_records)
        return {
            "epoch": epoch_value,
            "grid_size": self.config.grid_size,
            "num_regions": self.num_regions,
            "score_threshold": self.config.matching.score_threshold,
            "low_iou_threshold": self.config.matching.low_iou_threshold,
            "good_iou_threshold": self.config.matching.good_iou_threshold,
            "stable_epochs": stable_threshold,
            "hotspot_top_k": top_k,
            "num_seen_gts": len(records),
            "num_missing_gts": len(missing_records),
            "num_low_iou_gts": len(low_iou_records),
            "num_stable_low_iou_gts": len(stable_low_iou_records),
            "num_good_gts": len(good_records),
            "missing_gt_ratio": _safe_ratio(len(missing_records), len(records)),
            "low_iou_gt_ratio": _safe_ratio(len(low_iou_records), len(records)),
            "stable_low_iou_gt_ratio": _safe_ratio(len(stable_low_iou_records), len(records)),
            "good_gt_ratio": _safe_ratio(len(good_records), len(records)),
            "low_iou_streak_mean": _mean(low_iou_streaks),
            "low_iou_streak_p50": _percentile(low_iou_streaks, 0.50),
            "low_iou_streak_p90": _percentile(low_iou_streaks, 0.90),
            "low_iou_streak_max": max(low_iou_streaks, default=0),
            "low_iou_streak_values": list(low_iou_streaks),
            "low_iou_deficit_mean": _mean(low_iou_deficits),
            "low_iou_deficit_sum": float(sum(low_iou_deficits)),
            "low_iou_deficit_values": list(low_iou_deficits),
            "state_transition_histogram": _string_counter(transition_histogram),
            "missing_gt_keys": sorted(record.record_key for record in missing_records),
            "low_iou_gt_keys": sorted(record.record_key for record in low_iou_records),
            "stable_low_iou_gt_keys": sorted(record.record_key for record in stable_low_iou_records),
            "good_gt_keys": sorted(record.record_key for record in good_records),
            "low_iou_region_histogram": _string_counter(low_iou_region_histogram),
            "stable_low_iou_region_histogram": _string_counter(stable_region_histogram),
            "low_iou_image_region_hotspots": _string_counter(low_iou_hotspots),
            "stable_low_iou_image_region_hotspots": _string_counter(stable_hotspots),
            "top_low_iou_hotspots": _topk_items(low_iou_hotspots, k=top_k),
            "top_stable_low_iou_hotspots": _topk_items(stable_hotspots, k=top_k),
        }

    def stability_metrics(
        self,
        current_snapshot: Mapping[str, Any],
        previous_snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return compute_lmb_stability_metrics(
            current_snapshot,
            previous_snapshot,
            hotspot_top_k=self.config.stability.hotspot_top_k,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "current_epoch": self.current_epoch,
            "records": [record.to_state() for record in self._records.values()],
            "stats": dict(self._stats),
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        self.reset()
        if not isinstance(state, Mapping):
            return
        self.current_epoch = int(state.get("current_epoch", 0))
        self._stats.update(Counter({str(key): int(value) for key, value in dict(state.get("stats", {})).items()}))
        records: dict[str, LMBRecord] = {}
        for raw_record in state.get("records", []):
            if not isinstance(raw_record, Mapping):
                continue
            record = LMBRecord.from_state(raw_record)
            if not record.record_key:
                continue
            records[record.record_key] = record
            self._image_index[record.image_id].add(record.record_key)
        self._records = records

    def _update_image(
        self,
        *,
        target: Mapping[str, Any],
        detection: Mapping[str, torch.Tensor],
        image_index: int,
        image_size: Sequence[int | float] | None,
        epoch: int,
        step: int,
    ) -> Counter[str]:
        stats: Counter[str] = Counter()
        gt_boxes = _as_boxes_tensor(target.get("boxes"))
        gt_labels = _as_int_tensor(target.get("labels"), length=gt_boxes.shape[0])
        height, width = _resolve_image_size(target=target, image_size=image_size, boxes=gt_boxes)
        image_id = _normalize_image_id(target.get("image_id", image_index))
        gt_ids = _extract_gt_ids(target, int(gt_boxes.shape[0]))

        pred_boxes = _as_boxes_tensor(detection.get("boxes"))
        pred_labels = _as_int_tensor(detection.get("labels"), length=pred_boxes.shape[0]).to(device=pred_boxes.device)
        pred_scores = _as_float_tensor(detection.get("scores"), length=pred_boxes.shape[0], fill=1.0).to(device=pred_boxes.device)

        stats["images"] += 1
        stats["gts"] += int(gt_boxes.shape[0])
        for gt_index, gt_box in enumerate(gt_boxes):
            if not _valid_box(gt_box):
                stats["invalid_gt_boxes"] += 1
                continue
            gt_label = int(gt_labels[gt_index].item()) if gt_index < gt_labels.numel() else 0
            clamped = _clamp_box(gt_box, height=height, width=width)
            bbox_xyxy = _box_to_tuple(clamped)
            region_id = _region_id_for_box(bbox_xyxy, height=height, width=width, grid_size=self.config.grid_size)
            record_key = _record_key(
                image_id=image_id,
                gt_id=gt_ids[gt_index] if gt_index < len(gt_ids) else None,
                gt_class=gt_label,
                bbox_xyxy=bbox_xyxy,
                height=height,
                width=width,
            )
            match = _match_gt(
                gt_box=clamped,
                gt_label=gt_label,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                score_threshold=float(self.config.matching.score_threshold),
                class_aware=self.config.matching.class_aware,
            )
            state = _state_from_iou(
                match["best_iou"],
                low_iou_threshold=self.config.matching.low_iou_threshold,
                good_iou_threshold=self.config.matching.good_iou_threshold,
            )
            record = self._records.get(record_key)
            if record is None:
                record = LMBRecord(
                    record_key=record_key,
                    image_id=image_id,
                    gt_id=None if gt_ids[gt_index] is None else str(gt_ids[gt_index]),
                    gt_class=gt_label,
                    bbox_xyxy=bbox_xyxy,
                    region_id=region_id,
                )
                self._records[record_key] = record
                self._image_index[image_id].add(record_key)
                stats["new_records"] += 1
            record.update(
                state=state,
                bbox_xyxy=bbox_xyxy,
                region_id=region_id,
                epoch=epoch,
                step=step,
                best_iou=match["best_iou"],
                best_iou_score=match["best_iou_score"],
                best_score=match["best_score"],
                best_score_iou=match["best_score_iou"],
                good_iou_threshold=float(self.config.matching.good_iou_threshold),
            )
            stats[f"state_{state}"] += 1
        return stats

    def _enforce_max_records(self) -> None:
        if self.config.max_records is None:
            return
        max_records = int(self.config.max_records)
        if len(self._records) <= max_records:
            return
        ordered = sorted(
            self._records.values(),
            key=lambda record: (
                int(record.last_epoch),
                int(record.last_step),
                int(record.max_low_iou_streak),
                str(record.record_key),
            ),
        )
        for record in ordered[: max(0, len(self._records) - max_records)]:
            self._records.pop(record.record_key, None)
            indexed = self._image_index.get(record.image_id)
            if indexed is not None:
                indexed.discard(record.record_key)
                if not indexed:
                    self._image_index.pop(record.image_id, None)


def load_lmb_config(path: str | Path, *, arch: str | None = None) -> LMBConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"LMB config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("LMB config YAML must contain a mapping at the top level.")
    return LMBConfig.from_mapping(raw, arch=arch)


def build_lmb_from_config(
    raw_config: Mapping[str, Any] | LMBConfig,
    *,
    arch: str | None = None,
    detector_score_threshold: float | None = None,
) -> LocalizationMemoryBank | None:
    config = raw_config if isinstance(raw_config, LMBConfig) else LMBConfig.from_mapping(raw_config, arch=arch)
    if not config.enabled:
        return None
    config = config.resolve_detector_thresholds(detector_score_threshold=detector_score_threshold)
    return LocalizationMemoryBank(config)


def build_lmb_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    detector_score_threshold: float | None = None,
) -> LocalizationMemoryBank | None:
    config = load_lmb_config(path, arch=arch)
    if not config.enabled:
        return None
    config = config.resolve_detector_thresholds(detector_score_threshold=detector_score_threshold)
    return LocalizationMemoryBank(config)


def merge_lmb_epoch_snapshots(
    snapshots: Sequence[Mapping[str, Any] | None],
    *,
    hotspot_top_k: int = 10,
) -> dict[str, Any] | None:
    valid_snapshots = [snapshot for snapshot in snapshots if isinstance(snapshot, Mapping)]
    if not valid_snapshots:
        return None

    missing_gt_keys: set[str] = set()
    low_iou_gt_keys: set[str] = set()
    stable_low_iou_gt_keys: set[str] = set()
    good_gt_keys: set[str] = set()
    low_iou_region_histogram: Counter[str] = Counter()
    stable_low_iou_region_histogram: Counter[str] = Counter()
    low_iou_hotspots: Counter[str] = Counter()
    stable_low_iou_hotspots: Counter[str] = Counter()
    state_transition_histogram: Counter[str] = Counter()
    low_iou_streak_values: list[float] = []
    low_iou_deficit_values: list[float] = []

    merged: dict[str, Any] = {
        "epoch": max(_optional_int(snapshot.get("epoch")) or 0 for snapshot in valid_snapshots),
        "grid_size": valid_snapshots[0].get("grid_size"),
        "num_regions": valid_snapshots[0].get("num_regions"),
        "score_threshold": valid_snapshots[0].get("score_threshold"),
        "low_iou_threshold": valid_snapshots[0].get("low_iou_threshold"),
        "good_iou_threshold": valid_snapshots[0].get("good_iou_threshold"),
        "stable_epochs": valid_snapshots[0].get("stable_epochs"),
        "hotspot_top_k": int(hotspot_top_k),
        "num_seen_gts": 0,
    }
    for snapshot in valid_snapshots:
        missing_gt_keys.update(_as_string_list(snapshot.get("missing_gt_keys")))
        low_iou_gt_keys.update(_as_string_list(snapshot.get("low_iou_gt_keys")))
        stable_low_iou_gt_keys.update(_as_string_list(snapshot.get("stable_low_iou_gt_keys")))
        good_gt_keys.update(_as_string_list(snapshot.get("good_gt_keys")))
        low_iou_region_histogram.update(_counter_from_mapping(snapshot.get("low_iou_region_histogram")))
        stable_low_iou_region_histogram.update(
            _counter_from_mapping(snapshot.get("stable_low_iou_region_histogram"))
        )
        low_iou_hotspots.update(_counter_from_mapping(snapshot.get("low_iou_image_region_hotspots")))
        stable_low_iou_hotspots.update(
            _counter_from_mapping(snapshot.get("stable_low_iou_image_region_hotspots"))
        )
        state_transition_histogram.update(_counter_from_mapping(snapshot.get("state_transition_histogram")))
        low_iou_streak_values.extend(_as_float_list(snapshot.get("low_iou_streak_values")))
        low_iou_deficit_values.extend(_as_float_list(snapshot.get("low_iou_deficit_values")))
        merged["num_seen_gts"] += _optional_int(snapshot.get("num_seen_gts")) or 0

    num_seen = int(merged["num_seen_gts"])
    merged.update(
        {
            "num_missing_gts": len(missing_gt_keys),
            "num_low_iou_gts": len(low_iou_gt_keys),
            "num_stable_low_iou_gts": len(stable_low_iou_gt_keys),
            "num_good_gts": len(good_gt_keys),
            "missing_gt_ratio": _safe_ratio(len(missing_gt_keys), num_seen),
            "low_iou_gt_ratio": _safe_ratio(len(low_iou_gt_keys), num_seen),
            "stable_low_iou_gt_ratio": _safe_ratio(len(stable_low_iou_gt_keys), num_seen),
            "good_gt_ratio": _safe_ratio(len(good_gt_keys), num_seen),
            "low_iou_streak_mean": _mean(low_iou_streak_values),
            "low_iou_streak_p50": _percentile(low_iou_streak_values, 0.50),
            "low_iou_streak_p90": _percentile(low_iou_streak_values, 0.90),
            "low_iou_streak_max": max(low_iou_streak_values, default=0),
            "low_iou_streak_values": low_iou_streak_values,
            "low_iou_deficit_mean": _mean(low_iou_deficit_values),
            "low_iou_deficit_sum": float(sum(low_iou_deficit_values)),
            "low_iou_deficit_values": low_iou_deficit_values,
            "state_transition_histogram": _string_counter(state_transition_histogram),
            "missing_gt_keys": sorted(missing_gt_keys),
            "low_iou_gt_keys": sorted(low_iou_gt_keys),
            "stable_low_iou_gt_keys": sorted(stable_low_iou_gt_keys),
            "good_gt_keys": sorted(good_gt_keys),
            "low_iou_region_histogram": dict(sorted(low_iou_region_histogram.items())),
            "stable_low_iou_region_histogram": dict(sorted(stable_low_iou_region_histogram.items())),
            "low_iou_image_region_hotspots": dict(sorted(low_iou_hotspots.items())),
            "stable_low_iou_image_region_hotspots": dict(sorted(stable_low_iou_hotspots.items())),
            "top_low_iou_hotspots": _topk_items(low_iou_hotspots, k=int(hotspot_top_k)),
            "top_stable_low_iou_hotspots": _topk_items(stable_low_iou_hotspots, k=int(hotspot_top_k)),
        }
    )
    return merged


def compute_lmb_stability_metrics(
    current_snapshot: Mapping[str, Any],
    previous_snapshot: Mapping[str, Any] | None = None,
    *,
    hotspot_top_k: int = 10,
) -> dict[str, Any]:
    current_low = set(_as_string_list(current_snapshot.get("low_iou_gt_keys")))
    current_stable = set(_as_string_list(current_snapshot.get("stable_low_iou_gt_keys")))
    current_region_histogram = _counter_from_mapping(current_snapshot.get("low_iou_region_histogram"))
    current_hotspots = _counter_from_mapping(current_snapshot.get("low_iou_image_region_hotspots"))

    has_previous = isinstance(previous_snapshot, Mapping)
    previous_low: set[str] = set()
    previous_stable: set[str] = set()
    previous_region_histogram: Counter[str] = Counter()
    previous_hotspots: Counter[str] = Counter()
    if has_previous and previous_snapshot is not None:
        previous_low = set(_as_string_list(previous_snapshot.get("low_iou_gt_keys")))
        previous_stable = set(_as_string_list(previous_snapshot.get("stable_low_iou_gt_keys")))
        previous_region_histogram = _counter_from_mapping(previous_snapshot.get("low_iou_region_histogram"))
        previous_hotspots = _counter_from_mapping(previous_snapshot.get("low_iou_image_region_hotspots"))

    low_iou_jaccard = None
    stable_low_iou_jaccard = None
    low_iou_churn_rate = None
    new_low_iou_rate = None
    region_js_divergence = None
    hotspot_overlap_at_k = None
    if has_previous:
        low_iou_jaccard = _jaccard(current_low, previous_low)
        stable_low_iou_jaccard = _jaccard(current_stable, previous_stable)
        low_iou_churn_rate = None if low_iou_jaccard is None else 1.0 - low_iou_jaccard
        new_low_iou_rate = _new_item_rate(current_low, previous_low)
        region_js_divergence = _js_divergence(current_region_histogram, previous_region_histogram)
        hotspot_overlap_at_k = _topk_overlap(current_hotspots, previous_hotspots, k=int(hotspot_top_k))

    num_seen = _optional_int(current_snapshot.get("num_seen_gts")) or 0
    num_low = len(current_low)
    num_stable = len(current_stable)
    return {
        "epoch": _optional_int(current_snapshot.get("epoch")),
        "num_seen_gts": num_seen,
        "num_missing_gts": _optional_int(current_snapshot.get("num_missing_gts")),
        "num_low_iou_gts": num_low,
        "num_stable_low_iou_gts": num_stable,
        "num_good_gts": _optional_int(current_snapshot.get("num_good_gts")),
        "low_iou_gt_ratio": _safe_ratio(num_low, num_seen),
        "stable_low_iou_gt_ratio": _safe_ratio(num_stable, num_seen),
        "stable_among_low_iou_ratio": _safe_ratio(num_stable, num_low),
        "low_iou_jaccard_stability": low_iou_jaccard,
        "stable_low_iou_jaccard_stability": stable_low_iou_jaccard,
        "low_iou_churn_rate": low_iou_churn_rate,
        "new_low_iou_rate": new_low_iou_rate,
        "low_iou_region_js_divergence": region_js_divergence,
        "top1_low_iou_region_share": _top1_share(current_region_histogram),
        "low_iou_region_entropy": _normalized_entropy(
            current_region_histogram,
            num_bins=_optional_int(current_snapshot.get("num_regions")),
        ),
        "low_iou_hotspot_overlap_at_k": hotspot_overlap_at_k,
        "low_iou_hotspot_k": int(hotspot_top_k),
        "low_iou_streak_mean": current_snapshot.get("low_iou_streak_mean"),
        "low_iou_streak_p50": current_snapshot.get("low_iou_streak_p50"),
        "low_iou_streak_p90": current_snapshot.get("low_iou_streak_p90"),
        "low_iou_streak_max": current_snapshot.get("low_iou_streak_max"),
        "low_iou_deficit_mean": current_snapshot.get("low_iou_deficit_mean"),
        "low_iou_deficit_sum": current_snapshot.get("low_iou_deficit_sum"),
        "state_transition_histogram": dict(current_snapshot.get("state_transition_histogram", {})),
        "low_iou_region_histogram": dict(sorted(current_region_histogram.items())),
        "stable_low_iou_region_histogram": dict(
            sorted(_counter_from_mapping(current_snapshot.get("stable_low_iou_region_histogram")).items())
        ),
        "top_low_iou_hotspots": _topk_items(current_hotspots, k=int(hotspot_top_k)),
    }


def _match_gt(
    *,
    gt_box: torch.Tensor,
    gt_label: int,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    score_threshold: float,
    class_aware: bool,
) -> dict[str, Any]:
    if pred_boxes.numel() == 0:
        return {
            "best_iou": None,
            "best_iou_score": None,
            "best_score": None,
            "best_score_iou": None,
        }
    gt_box = gt_box.to(device=pred_boxes.device, dtype=torch.float32).reshape(1, 4)
    pred_boxes = pred_boxes.to(device=gt_box.device, dtype=torch.float32)
    pred_labels = pred_labels.to(device=gt_box.device, dtype=torch.long)
    pred_scores = pred_scores.to(device=gt_box.device, dtype=torch.float32)
    candidate_mask = pred_scores >= float(score_threshold)
    if bool(class_aware):
        candidate_mask = candidate_mask & (pred_labels == int(gt_label))
    if not bool(candidate_mask.any().item()):
        return {
            "best_iou": None,
            "best_iou_score": None,
            "best_score": None,
            "best_score_iou": None,
        }
    candidate_indices = torch.where(candidate_mask)[0]
    candidate_boxes = pred_boxes[candidate_indices]
    candidate_scores = pred_scores[candidate_indices]
    ious = box_ops.box_iou(gt_box, candidate_boxes)[0].clamp(min=0.0, max=1.0)
    best_iou_local = int(torch.argmax(ious).item())
    best_iou = float(ious[best_iou_local].item())
    best_iou_score = float(candidate_scores[best_iou_local].item())
    best_score_local = int(torch.argmax(candidate_scores).item())
    best_score = float(candidate_scores[best_score_local].item())
    best_score_iou = float(ious[best_score_local].item())
    return {
        "best_iou": best_iou,
        "best_iou_score": best_iou_score,
        "best_score": best_score,
        "best_score_iou": best_score_iou,
    }


def _state_from_iou(
    best_iou: float | None,
    *,
    low_iou_threshold: float,
    good_iou_threshold: float,
) -> str:
    if best_iou is None or float(best_iou) < float(low_iou_threshold):
        return MISSING_STATE
    if float(best_iou) < float(good_iou_threshold):
        return LOW_IOU_STATE
    return GOOD_STATE


def _parse_threshold(value: Any) -> float | str:
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return float(value)


def _is_auto_threshold(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


def _normalize_state(value: str) -> str:
    normalized = str(value).lower()
    if normalized in {MISSING_STATE, LOW_IOU_STATE, GOOD_STATE}:
        return normalized
    return UNKNOWN_STATE


def _region_id_for_box(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    height: float,
    width: float,
    grid_size: int,
) -> int:
    x1, y1, x2, y2 = bbox_xyxy
    height = max(float(height), 1.0)
    width = max(float(width), 1.0)
    grid_size = max(int(grid_size), 1)
    cell_w = width / float(grid_size)
    cell_h = height / float(grid_size)
    best_region = 1
    best_area = -1.0
    for row in range(grid_size):
        for col in range(grid_size):
            rx1 = float(col) * cell_w
            ry1 = float(row) * cell_h
            rx2 = width if col == grid_size - 1 else float(col + 1) * cell_w
            ry2 = height if row == grid_size - 1 else float(row + 1) * cell_h
            ix1 = max(float(x1), rx1)
            iy1 = max(float(y1), ry1)
            ix2 = min(float(x2), rx2)
            iy2 = min(float(y2), ry2)
            area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            region_id = row * grid_size + col + 1
            if area > best_area:
                best_area = area
                best_region = region_id
    return int(best_region)


def _resolve_image_size(
    *,
    target: Mapping[str, Any],
    image_size: Sequence[int | float] | None,
    boxes: torch.Tensor,
) -> tuple[float, float]:
    if image_size is not None and len(image_size) >= 2:
        return float(image_size[0]), float(image_size[1])
    for key in ("image_size", "orig_size", "size"):
        value = target.get(key)
        parsed = _parse_size(value)
        if parsed is not None:
            return parsed
    if boxes.numel() > 0:
        max_xy = boxes.detach().cpu().to(dtype=torch.float32).reshape(-1, 4)[:, 2:].max(dim=0).values
        width = max(float(max_xy[0].item()), 1.0)
        height = max(float(max_xy[1].item()), 1.0)
        return height, width
    return 1.0, 1.0


def _parse_size(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() < 2:
            return None
        flat = value.detach().cpu().flatten().tolist()
        return float(flat[0]), float(flat[1])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _as_boxes_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.float32, device=value.device)
        return value.detach().to(dtype=torch.float32).reshape(-1, 4)
    if value is None:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32).reshape(-1, 4)


def _as_int_tensor(value: Any, *, length: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.long).flatten()
    if value is None:
        return torch.zeros((length,), dtype=torch.long)
    return torch.tensor(value, dtype=torch.long).flatten()


def _as_float_tensor(value: Any, *, length: int, fill: float) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).flatten()
    if value is None:
        return torch.full((length,), float(fill), dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32).flatten()


def _valid_box(box: torch.Tensor) -> bool:
    values = box.detach().cpu().to(dtype=torch.float32).flatten()
    if values.numel() != 4:
        return False
    return bool((values[2] > values[0]).item() and (values[3] > values[1]).item())


def _clamp_box(box: torch.Tensor, *, height: float, width: float) -> torch.Tensor:
    values = box.detach().cpu().to(dtype=torch.float32).flatten()
    x1 = torch.minimum(values[0], values[2]).clamp(min=0.0, max=max(float(width), 1.0))
    y1 = torch.minimum(values[1], values[3]).clamp(min=0.0, max=max(float(height), 1.0))
    x2 = torch.maximum(values[0], values[2]).clamp(min=0.0, max=max(float(width), 1.0))
    y2 = torch.maximum(values[1], values[3]).clamp(min=0.0, max=max(float(height), 1.0))
    return torch.stack([x1, y1, x2, y2])


def _box_to_tuple(box: torch.Tensor) -> tuple[float, float, float, float]:
    values = box.detach().cpu().to(dtype=torch.float32).flatten().tolist()
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _extract_gt_ids(target: Mapping[str, Any], count: int) -> list[Any | None]:
    for key in ("gt_ids", "annotation_ids", "ann_ids"):
        value = target.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            flattened = value.detach().cpu().flatten().tolist()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            flattened = list(value)
        else:
            flattened = [value]
        if len(flattened) == count:
            return [None if _is_invalid_gt_id(item) else item for item in flattened]
    return [None for _ in range(count)]


def _is_invalid_gt_id(value: Any) -> bool:
    if value is None:
        return True
    try:
        return int(value) < 0
    except (TypeError, ValueError):
        return str(value) == ""


def _record_key(
    *,
    image_id: str,
    gt_id: Any | None,
    gt_class: int,
    bbox_xyxy: tuple[float, float, float, float],
    height: float,
    width: float,
) -> str:
    if gt_id is not None and not _is_invalid_gt_id(gt_id):
        return f"{image_id}:ann:{gt_id}"
    scale = (max(float(width), 1.0), max(float(height), 1.0))
    normalized_box = (
        float(bbox_xyxy[0]) / scale[0],
        float(bbox_xyxy[1]) / scale[1],
        float(bbox_xyxy[2]) / scale[0],
        float(bbox_xyxy[3]) / scale[1],
    )
    box_key = ",".join(f"{value:.6f}" for value in normalized_box)
    digest = hashlib.sha1(f"{image_id}:{gt_class}:{box_key}".encode("utf-8")).hexdigest()[:16]
    return f"{image_id}:box:{gt_class}:{digest}"


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        if value.numel() == 1:
            raw = value.detach().cpu().flatten()[0].item()
            if isinstance(raw, float) and raw.is_integer():
                return str(int(raw))
            return str(raw)
        return ",".join(str(item) for item in value.detach().cpu().flatten().tolist())
    return str(value)


def _hotspot_key(image_id: Any, region_id: int) -> str:
    return f"{_normalize_image_id(image_id)}::{int(region_id)}"


def _string_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    parsed: list[float] = []
    for item in value:
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            continue
    return parsed


def _counter_from_mapping(value: Any) -> Counter[str]:
    if not isinstance(value, Mapping):
        return Counter()
    return Counter({str(key): int(count) for key, count in value.items()})


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) for value in values)) / float(len(values))


def _percentile(values: Sequence[int | float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, float(q))) * float(len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - float(lower)
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _jaccard(current: set[str], previous: set[str]) -> float:
    union = current | previous
    if not union:
        return 1.0
    return len(current & previous) / float(len(union))


def _new_item_rate(current: set[str], previous: set[str]) -> float:
    if not current:
        return 0.0
    return len(current - previous) / float(len(current))


def _top1_share(counter: Counter[str]) -> float | None:
    total = sum(counter.values())
    if total <= 0:
        return None
    return max(counter.values()) / float(total)


def _normalized_entropy(counter: Counter[str], *, num_bins: int | None = None) -> float | None:
    total = sum(counter.values())
    if total <= 0:
        return None
    nonzero = [count for count in counter.values() if count > 0]
    bins = int(num_bins or len(nonzero))
    if bins <= 1:
        return 0.0
    entropy = 0.0
    for count in nonzero:
        probability = count / float(total)
        entropy -= probability * math.log(probability)
    return entropy / math.log(float(bins))


def _js_divergence(current: Counter[str], previous: Counter[str]) -> float:
    keys = sorted(set(current) | set(previous))
    current_total = sum(current.values())
    previous_total = sum(previous.values())
    if current_total <= 0 and previous_total <= 0:
        return 0.0
    if current_total <= 0 or previous_total <= 0:
        return 1.0
    p = [current[key] / float(current_total) for key in keys]
    q = [previous[key] / float(previous_total) for key in keys]
    m = [(p_value + q_value) * 0.5 for p_value, q_value in zip(p, q, strict=True)]
    divergence = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
    return divergence / math.log(2.0)


def _kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    total = 0.0
    for p_value, q_value in zip(p, q, strict=True):
        if p_value <= 0.0:
            continue
        total += p_value * math.log(p_value / max(q_value, 1e-12))
    return total


def _topk_overlap(current: Counter[str], previous: Counter[str], *, k: int) -> float | None:
    if k <= 0:
        return None
    if not current and not previous:
        return 1.0
    if not current or not previous:
        return 0.0
    current_top = {key for key, _ in _topk_pairs(current, k=k)}
    previous_top = {key for key, _ in _topk_pairs(previous, k=k)}
    return len(current_top & previous_top) / float(k)


def _topk_items(counter: Counter[str], *, k: int) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": int(count)}
        for key, count in _topk_pairs(counter, k=k)
    ]


def _topk_pairs(counter: Counter[str], *, k: int) -> list[tuple[str, int]]:
    if k <= 0:
        return []
    return sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))[:k]
