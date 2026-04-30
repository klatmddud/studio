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


@dataclass(frozen=True, slots=True)
class MissBankMatchingConfig:
    score_threshold: float | str = 0.2
    iou_threshold: float | str = 0.5

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
    ) -> "MissBankMatchingConfig":
        data = dict(raw or {})
        config = cls(
            score_threshold=_parse_threshold(data.get("score_threshold", 0.2)),
            iou_threshold=_parse_threshold(data.get("iou_threshold", 0.5)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if _is_auto_threshold(self.score_threshold):
            pass
        elif not 0.0 <= float(self.score_threshold) <= 1.0:
            raise ValueError("MissBank matching.score_threshold must be 'auto' or satisfy 0 <= value <= 1.")
        if _is_auto_threshold(self.iou_threshold):
            pass
        elif not 0.0 <= float(self.iou_threshold) <= 1.0:
            raise ValueError("MissBank matching.iou_threshold must be 'auto' or satisfy 0 <= value <= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold,
        }


@dataclass(frozen=True, slots=True)
class MissBankTargetConfig:
    miss_threshold: int = 2
    aggregation: str = "miss_count"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
    ) -> "MissBankTargetConfig":
        data = dict(raw or {})
        config = cls(
            miss_threshold=int(data.get("miss_threshold", 2)),
            aggregation=str(data.get("aggregation", "miss_count")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.miss_threshold) < 1:
            raise ValueError("MissBank target.miss_threshold must be >= 1.")
        if self.aggregation != "miss_count":
            raise ValueError("MissBank target.aggregation currently supports only 'miss_count'.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "miss_threshold": self.miss_threshold,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True, slots=True)
class MissBankMiningConfig:
    type: str = "online"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
    ) -> "MissBankMiningConfig":
        data = dict(raw or {})
        config = cls(type=str(data.get("type", "online")).lower())
        config.validate()
        return config

    def validate(self) -> None:
        if self.type not in {"online", "offline"}:
            raise ValueError("MissBank mining.type must be either 'online' or 'offline'.")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type}


@dataclass(frozen=True, slots=True)
class MissBankConfig:
    enabled: bool = False
    grid_size: int = 2
    start_epoch: int = 1
    mining: MissBankMiningConfig = field(default_factory=MissBankMiningConfig)
    matching: MissBankMatchingConfig = field(default_factory=MissBankMatchingConfig)
    target: MissBankTargetConfig = field(default_factory=MissBankTargetConfig)
    max_records: int | None = None
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "MissBankConfig":
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
            mining=MissBankMiningConfig.from_mapping(merged.get("mining")),
            matching=MissBankMatchingConfig.from_mapping(merged.get("matching")),
            target=MissBankTargetConfig.from_mapping(merged.get("target")),
            max_records=None if merged.get("max_records") is None else int(merged.get("max_records")),
            arch=normalized_arch,
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
        if int(self.grid_size) < 1:
            raise ValueError("MissBank grid_size must be >= 1.")
        if int(self.start_epoch) < 0:
            raise ValueError("MissBank start_epoch must be >= 0.")
        if self.max_records is not None and int(self.max_records) < 1:
            raise ValueError("MissBank max_records must be null or >= 1.")

    def resolve_detector_thresholds(
        self,
        *,
        detector_score_threshold: float | None,
        detector_iou_threshold: float | None,
    ) -> "MissBankConfig":
        score_threshold = self.matching.score_threshold
        iou_threshold = self.matching.iou_threshold
        if _is_auto_threshold(score_threshold):
            if detector_score_threshold is None:
                raise ValueError(
                    "MissBank matching.score_threshold='auto' requires the detector final score threshold."
                )
            score_threshold = float(detector_score_threshold)
        if _is_auto_threshold(iou_threshold):
            if detector_iou_threshold is None:
                raise ValueError(
                    "MissBank matching.iou_threshold='auto' requires the detector final IoU threshold."
                )
            iou_threshold = float(detector_iou_threshold)
        if (
            score_threshold == self.matching.score_threshold
            and iou_threshold == self.matching.iou_threshold
        ):
            return self
        matching = replace(
            self.matching,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        matching.validate()
        return replace(self, matching=matching)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "grid_size": self.grid_size,
            "start_epoch": self.start_epoch,
            "mining": self.mining.to_dict(),
            "matching": self.matching.to_dict(),
            "target": self.target.to_dict(),
            "max_records": self.max_records,
            "arch": self.arch,
        }


@dataclass(slots=True)
class MissBankRecord:
    record_key: str
    image_id: str
    gt_id: str | None
    gt_class: int
    bbox_xyxy: tuple[float, float, float, float]
    region_id: int
    miss_count: int = 0
    is_missed: bool = False
    last_epoch: int = 0
    last_step: int = 0
    last_iou: float | None = None
    last_score: float | None = None
    best_iou: float | None = None
    best_iou_score: float | None = None
    best_score: float | None = None
    best_score_iou: float | None = None
    total_seen: int = 0
    total_missed: int = 0
    max_miss_count: int = 0

    def update(
        self,
        *,
        is_missed: bool,
        bbox_xyxy: tuple[float, float, float, float],
        region_id: int,
        epoch: int,
        step: int,
        matched_iou: float | None,
        matched_score: float | None,
        best_iou: float | None,
        best_iou_score: float | None,
        best_score: float | None,
        best_score_iou: float | None,
    ) -> None:
        self.bbox_xyxy = bbox_xyxy
        self.region_id = int(region_id)
        self.is_missed = bool(is_missed)
        self.last_epoch = int(epoch)
        self.last_step = int(step)
        self.best_iou = None if best_iou is None else float(best_iou)
        self.best_iou_score = None if best_iou_score is None else float(best_iou_score)
        self.best_score = None if best_score is None else float(best_score)
        self.best_score_iou = None if best_score_iou is None else float(best_score_iou)
        self.total_seen += 1
        if bool(is_missed):
            self.miss_count += 1
            self.total_missed += 1
            self.max_miss_count = max(int(self.max_miss_count), int(self.miss_count))
            self.last_iou = None
            self.last_score = None
        else:
            self.miss_count = 0
            self.last_iou = None if matched_iou is None else float(matched_iou)
            self.last_score = None if matched_score is None else float(matched_score)

    @property
    def is_target(self) -> bool:
        return bool(self.is_missed)

    def to_state(self) -> dict[str, Any]:
        return {
            "record_key": self.record_key,
            "image_id": self.image_id,
            "gt_id": self.gt_id,
            "gt_class": self.gt_class,
            "bbox_xyxy": list(self.bbox_xyxy),
            "region_id": self.region_id,
            "miss_count": self.miss_count,
            "is_missed": self.is_missed,
            "last_epoch": self.last_epoch,
            "last_step": self.last_step,
            "last_iou": self.last_iou,
            "last_score": self.last_score,
            "best_iou": self.best_iou,
            "best_iou_score": self.best_iou_score,
            "best_score": self.best_score,
            "best_score_iou": self.best_score_iou,
            "total_seen": self.total_seen,
            "total_missed": self.total_missed,
            "max_miss_count": self.max_miss_count,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "MissBankRecord":
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
            miss_count=int(state.get("miss_count", 0)),
            is_missed=bool(state.get("is_missed", False)),
            last_epoch=int(state.get("last_epoch", 0)),
            last_step=int(state.get("last_step", 0)),
            last_iou=None if state.get("last_iou") is None else float(state.get("last_iou")),
            last_score=None if state.get("last_score") is None else float(state.get("last_score")),
            best_iou=None if state.get("best_iou") is None else float(state.get("best_iou")),
            best_iou_score=None if state.get("best_iou_score") is None else float(state.get("best_iou_score")),
            best_score=None if state.get("best_score") is None else float(state.get("best_score")),
            best_score_iou=None if state.get("best_score_iou") is None else float(state.get("best_score_iou")),
            total_seen=int(state.get("total_seen", 0)),
            total_missed=int(state.get("total_missed", 0)),
            max_miss_count=int(state.get("max_miss_count", 0)),
        )


class MissBank(nn.Module):
    """Train-time memory for recurrent missed GT instances."""

    def __init__(self, config: MissBankConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, MissBankConfig)
            else MissBankConfig.from_mapping(config or {})
        )
        if (
            _is_auto_threshold(self.config.matching.score_threshold)
            or _is_auto_threshold(self.config.matching.iou_threshold)
        ):
            raise ValueError(
                "MissBank matching auto thresholds must be resolved before constructing MissBank."
            )
        self.current_epoch = 0
        self._records: dict[str, MissBankRecord] = {}
        self._image_index: dict[str, set[str]] = defaultdict(set)
        self._stats: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._records)

    @property
    def num_regions(self) -> int:
        return int(self.config.num_regions)

    @property
    def num_labels(self) -> int:
        return int(self.config.num_labels)

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
            raise ValueError("MissBank update requires the same number of targets and detections.")

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

    def get_image_targets(
        self,
        image_ids: Sequence[Any],
        *,
        miss_threshold: int | None = None,
    ) -> dict[str, int]:
        threshold = int(self.config.target.miss_threshold if miss_threshold is None else miss_threshold)
        return {
            _normalize_image_id(image_id): self._target_for_image(
                _normalize_image_id(image_id),
                miss_threshold=threshold,
            )
            for image_id in image_ids
        }

    def get_batch_labels(
        self,
        targets: Sequence[Mapping[str, Any]] | None = None,
        *,
        image_ids: Sequence[Any] | None = None,
        miss_threshold: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if image_ids is None:
            if targets is None:
                raise ValueError("MissBank get_batch_labels requires either targets or image_ids.")
            image_ids = [
                target.get("image_id", torch.tensor(index))
                for index, target in enumerate(targets)
            ]
        target_map = self.get_image_targets(image_ids, miss_threshold=miss_threshold)
        labels = [int(target_map[_normalize_image_id(image_id)]) for image_id in image_ids]
        return torch.tensor(labels, dtype=torch.long, device=device)

    def get_records(
        self,
        image_id: Any | None = None,
    ) -> list[MissBankRecord]:
        if image_id is None:
            return list(self._records.values())
        normalized = _normalize_image_id(image_id)
        return [
            self._records[key]
            for key in sorted(self._image_index.get(normalized, set()))
            if key in self._records
        ]

    def summary(self) -> dict[str, Any]:
        missed_records = [record for record in self._records.values() if record.is_missed]
        target_records = [
            record
            for record in missed_records
            if int(record.miss_count) >= int(self.config.target.miss_threshold)
        ]
        region_histogram = Counter(
            int(record.region_id)
            for record in target_records
            if int(record.region_id) > 0
        )
        class_histogram = Counter(int(record.gt_class) for record in target_records)
        miss_sum = sum(int(record.miss_count) for record in missed_records)
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "grid_size": self.config.grid_size,
            "mining_type": self.config.mining.type,
            "num_regions": self.num_regions,
            "num_labels": self.num_labels,
            "num_records": len(self._records),
            "num_images": len(self._image_index),
            "num_missed": len(missed_records),
            "num_target_gts": len(target_records),
            "mean_miss_count": miss_sum / float(max(len(missed_records), 1)),
            "max_miss_count": max((record.max_miss_count for record in self._records.values()), default=0),
            "region_histogram": dict(region_histogram),
            "class_histogram": dict(class_histogram),
            "stats": {key: int(value) for key, value in self._stats.items()},
        }

    def epoch_snapshot(
        self,
        epoch: int | None = None,
        *,
        miss_threshold: int | None = None,
        hotspot_top_k: int = 10,
    ) -> dict[str, Any]:
        epoch_value = int(self.current_epoch if epoch is None else epoch)
        threshold = int(self.config.target.miss_threshold if miss_threshold is None else miss_threshold)
        records = [
            record
            for record in self._records.values()
            if int(record.last_epoch) == epoch_value
        ]
        missed_records = [record for record in records if record.is_missed]
        target_records = [
            record
            for record in missed_records
            if int(record.miss_count) >= threshold
        ]

        missed_region_histogram = Counter(int(record.region_id) for record in missed_records)
        target_region_histogram = Counter(int(record.region_id) for record in target_records)
        image_region_hotspots = Counter(
            _hotspot_key(record.image_id, int(record.region_id))
            for record in missed_records
        )
        target_image_region_hotspots = Counter(
            _hotspot_key(record.image_id, int(record.region_id))
            for record in target_records
        )

        return {
            "epoch": epoch_value,
            "grid_size": self.config.grid_size,
            "mining_type": self.config.mining.type,
            "num_regions": self.num_regions,
            "miss_threshold": threshold,
            "hotspot_top_k": int(hotspot_top_k),
            "num_seen_gts": len(records),
            "num_missed_gts": len(missed_records),
            "num_target_gts": len(target_records),
            "num_images_seen": len({record.image_id for record in records}),
            "num_images_with_miss": len({record.image_id for record in missed_records}),
            "num_images_with_target": len({record.image_id for record in target_records}),
            "missed_gt_keys": sorted(record.record_key for record in missed_records),
            "target_gt_keys": sorted(record.record_key for record in target_records),
            "missed_region_histogram": _string_counter(missed_region_histogram),
            "target_region_histogram": _string_counter(target_region_histogram),
            "image_region_hotspots": _string_counter(image_region_hotspots),
            "target_image_region_hotspots": _string_counter(target_image_region_hotspots),
        }

    def stability_metrics(
        self,
        previous_snapshot: Mapping[str, Any] | None = None,
        *,
        epoch: int | None = None,
        miss_threshold: int | None = None,
        hotspot_top_k: int = 10,
    ) -> dict[str, Any]:
        current_snapshot = self.epoch_snapshot(
            epoch=epoch,
            miss_threshold=miss_threshold,
            hotspot_top_k=hotspot_top_k,
        )
        return compute_missbank_stability_metrics(
            current_snapshot,
            previous_snapshot=previous_snapshot,
            hotspot_top_k=hotspot_top_k,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "records": {key: record.to_state() for key, record in self._records.items()},
            "stats": dict(self._stats),
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            self.reset()
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))
        records: dict[str, MissBankRecord] = {}
        raw_records = state.get("records", {})
        if isinstance(raw_records, Mapping):
            for key, raw_record in raw_records.items():
                if isinstance(raw_record, Mapping):
                    record = MissBankRecord.from_state(raw_record)
                    record_key = str(key)
                    record.record_key = record_key
                    records[record.record_key] = record
        self._records = records
        self._rebuild_image_index()
        raw_stats = state.get("stats", {})
        self._stats = Counter({str(k): int(v) for k, v in raw_stats.items()}) if isinstance(raw_stats, Mapping) else Counter()

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
        gt_labels = _as_int_tensor(target.get("labels"), length=int(gt_boxes.shape[0]))
        if gt_boxes.numel() == 0:
            stats["images_without_gt"] += 1
            return stats

        height, width = _resolve_image_size(
            target=target,
            image_size=image_size,
            boxes=gt_boxes,
        )
        image_id = _normalize_image_id(target.get("image_id", torch.tensor(image_index)))
        gt_ids = _extract_gt_ids(target, int(gt_boxes.shape[0]))
        pred_boxes = _as_boxes_tensor(detection.get("boxes"))
        pred_labels = _as_int_tensor(detection.get("labels"), length=int(pred_boxes.shape[0]))
        pred_scores = _as_float_tensor(
            detection.get("scores"),
            length=int(pred_boxes.shape[0]),
            fill=1.0,
        )

        stats["images_seen"] += 1
        for gt_index, gt_box in enumerate(gt_boxes):
            if not _valid_box(gt_box):
                stats["invalid_gt"] += 1
                continue
            gt_label = int(gt_labels[gt_index].item())
            clamped_box = _clamp_box(gt_box, height=height, width=width)
            if not _valid_box(clamped_box):
                stats["invalid_gt_clamped"] += 1
                continue
            bbox_tuple = _box_to_tuple(clamped_box)
            region_id = _region_id_for_box(
                bbox_tuple,
                height=height,
                width=width,
                grid_size=int(self.config.grid_size),
            )
            gt_id = gt_ids[gt_index]
            record_key = _record_key(
                image_id=image_id,
                gt_id=gt_id,
                gt_class=gt_label,
                bbox_xyxy=bbox_tuple,
                height=height,
                width=width,
            )
            match = _match_gt(
                gt_box=torch.tensor(bbox_tuple, dtype=torch.float32),
                gt_label=gt_label,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                score_threshold=float(self.config.matching.score_threshold),
                iou_threshold=float(self.config.matching.iou_threshold),
            )
            is_missed = not bool(match["matched"])
            record = self._records.get(record_key)
            if record is None:
                record = MissBankRecord(
                    record_key=record_key,
                    image_id=image_id,
                    gt_id=None if gt_id is None else str(gt_id),
                    gt_class=gt_label,
                    bbox_xyxy=bbox_tuple,
                    region_id=region_id,
                )
                self._records[record_key] = record
                self._image_index[image_id].add(record_key)
                stats["records_created"] += 1
            record.update(
                is_missed=is_missed,
                bbox_xyxy=bbox_tuple,
                region_id=region_id,
                epoch=epoch,
                step=step,
                matched_iou=match["iou"],
                matched_score=match["score"],
                best_iou=match["best_iou"],
                best_iou_score=match["best_iou_score"],
                best_score=match["best_score"],
                best_score_iou=match["best_score_iou"],
            )
            stats["gt_seen"] += 1
            if is_missed:
                stats["gt_missed"] += 1
            else:
                stats["gt_detected"] += 1
        return stats

    def _target_for_image(
        self,
        image_id: str,
        *,
        miss_threshold: int,
    ) -> int:
        region_scores: Counter[int] = Counter()
        for record_key in self._image_index.get(image_id, set()):
            record = self._records.get(record_key)
            if record is None:
                continue
            if not record.is_missed:
                continue
            if int(record.miss_count) < int(miss_threshold):
                continue
            region_scores[int(record.region_id)] += int(record.miss_count)
        if not region_scores:
            return 0
        max_score = max(region_scores.values())
        return min(region for region, score in region_scores.items() if score == max_score)

    def _enforce_max_records(self) -> None:
        max_records = self.config.max_records
        if max_records is None or len(self._records) <= int(max_records):
            return
        ordered = sorted(
            self._records.values(),
            key=lambda record: (
                int(record.last_epoch),
                int(record.last_step),
                int(record.total_seen),
                record.record_key,
            ),
        )
        remove_count = len(self._records) - int(max_records)
        for record in ordered[:remove_count]:
            self._records.pop(record.record_key, None)
        self._rebuild_image_index()

    def _rebuild_image_index(self) -> None:
        image_index: dict[str, set[str]] = defaultdict(set)
        for key, record in self._records.items():
            image_index[str(record.image_id)].add(str(key))
        self._image_index = image_index


def load_remiss_config(path: str | Path, *, arch: str | None = None) -> MissBankConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"ReMiss YAML must contain a mapping at the top level: {config_path}")
    return MissBankConfig.from_mapping(raw, arch=arch)


def build_missbank_from_config(
    raw_config: Mapping[str, Any] | MissBankConfig,
    *,
    arch: str | None = None,
    detector_score_threshold: float | None = None,
    detector_iou_threshold: float | None = None,
) -> MissBank | None:
    config = raw_config if isinstance(raw_config, MissBankConfig) else MissBankConfig.from_mapping(raw_config, arch=arch)
    if not config.enabled:
        return None
    config = config.resolve_detector_thresholds(
        detector_score_threshold=detector_score_threshold,
        detector_iou_threshold=detector_iou_threshold,
    )
    return MissBank(config)


def build_missbank_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    detector_score_threshold: float | None = None,
    detector_iou_threshold: float | None = None,
) -> MissBank | None:
    config = load_remiss_config(path, arch=arch)
    if not config.enabled:
        return None
    config = config.resolve_detector_thresholds(
        detector_score_threshold=detector_score_threshold,
        detector_iou_threshold=detector_iou_threshold,
    )
    return MissBank(config)


def compute_missbank_stability_metrics(
    current_snapshot: Mapping[str, Any],
    *,
    previous_snapshot: Mapping[str, Any] | None = None,
    hotspot_top_k: int = 10,
) -> dict[str, Any]:
    current_missed = set(_as_string_list(current_snapshot.get("missed_gt_keys")))
    current_target = set(_as_string_list(current_snapshot.get("target_gt_keys")))
    current_region_histogram = _counter_from_mapping(
        current_snapshot.get("missed_region_histogram")
    )
    current_hotspots = _counter_from_mapping(
        current_snapshot.get("image_region_hotspots")
    )

    has_previous = isinstance(previous_snapshot, Mapping)
    previous_missed: set[str] = set()
    previous_region_histogram: Counter[str] = Counter()
    previous_hotspots: Counter[str] = Counter()
    if has_previous and previous_snapshot is not None:
        previous_missed = set(_as_string_list(previous_snapshot.get("missed_gt_keys")))
        previous_region_histogram = _counter_from_mapping(
            previous_snapshot.get("missed_region_histogram")
        )
        previous_hotspots = _counter_from_mapping(
            previous_snapshot.get("image_region_hotspots")
        )

    jaccard = None
    churn_rate = None
    new_miss_rate = None
    region_js_divergence = None
    hotspot_overlap_at_k = None
    if has_previous:
        jaccard = _jaccard(current_missed, previous_missed)
        churn_rate = None if jaccard is None else 1.0 - jaccard
        new_miss_rate = _new_item_rate(current_missed, previous_missed)
        region_js_divergence = _js_divergence(
            current_region_histogram,
            previous_region_histogram,
        )
        hotspot_overlap_at_k = _topk_overlap(
            current_hotspots,
            previous_hotspots,
            k=int(hotspot_top_k),
        )

    num_missed = len(current_missed)
    persistent_miss_ratio = (
        len(current_target) / float(num_missed)
        if num_missed > 0
        else None
    )

    return {
        "epoch": _optional_int(current_snapshot.get("epoch")),
        "num_seen_gts": _optional_int(current_snapshot.get("num_seen_gts")),
        "num_missed_gts": num_missed,
        "num_target_gts": len(current_target),
        "num_images_seen": _optional_int(current_snapshot.get("num_images_seen")),
        "num_images_with_miss": _optional_int(current_snapshot.get("num_images_with_miss")),
        "num_images_with_target": _optional_int(current_snapshot.get("num_images_with_target")),
        "miss_gt_jaccard_stability": jaccard,
        "miss_gt_churn_rate": churn_rate,
        "new_miss_rate": new_miss_rate,
        "persistent_miss_ratio": persistent_miss_ratio,
        "miss_region_js_divergence": region_js_divergence,
        "top1_miss_region_share": _top1_share(current_region_histogram),
        "miss_region_entropy": _normalized_entropy(
            current_region_histogram,
            num_bins=_optional_int(current_snapshot.get("num_regions")),
        ),
        "miss_hotspot_overlap_at_k": hotspot_overlap_at_k,
        "miss_hotspot_k": int(hotspot_top_k),
        "missed_region_histogram": dict(sorted(current_region_histogram.items())),
        "target_region_histogram": dict(
            sorted(_counter_from_mapping(current_snapshot.get("target_region_histogram")).items())
        ),
        "top_miss_hotspots": _topk_items(current_hotspots, k=int(hotspot_top_k)),
    }


def merge_missbank_epoch_snapshots(
    snapshots: Sequence[Mapping[str, Any] | None],
    *,
    hotspot_top_k: int = 10,
) -> dict[str, Any] | None:
    valid_snapshots = [snapshot for snapshot in snapshots if isinstance(snapshot, Mapping)]
    if not valid_snapshots:
        return None

    missed_gt_keys: set[str] = set()
    target_gt_keys: set[str] = set()
    missed_region_histogram: Counter[str] = Counter()
    target_region_histogram: Counter[str] = Counter()
    image_region_hotspots: Counter[str] = Counter()
    target_image_region_hotspots: Counter[str] = Counter()
    merged: dict[str, Any] = {
        "epoch": max(_optional_int(snapshot.get("epoch")) or 0 for snapshot in valid_snapshots),
        "grid_size": valid_snapshots[0].get("grid_size"),
        "mining_type": valid_snapshots[0].get("mining_type"),
        "num_regions": valid_snapshots[0].get("num_regions"),
        "miss_threshold": valid_snapshots[0].get("miss_threshold"),
        "hotspot_top_k": int(hotspot_top_k),
        "num_seen_gts": 0,
        "num_images_seen": 0,
        "num_images_with_miss": 0,
        "num_images_with_target": 0,
    }
    for snapshot in valid_snapshots:
        missed_gt_keys.update(_as_string_list(snapshot.get("missed_gt_keys")))
        target_gt_keys.update(_as_string_list(snapshot.get("target_gt_keys")))
        missed_region_histogram.update(_counter_from_mapping(snapshot.get("missed_region_histogram")))
        target_region_histogram.update(_counter_from_mapping(snapshot.get("target_region_histogram")))
        image_region_hotspots.update(_counter_from_mapping(snapshot.get("image_region_hotspots")))
        target_image_region_hotspots.update(_counter_from_mapping(snapshot.get("target_image_region_hotspots")))
        merged["num_seen_gts"] += _optional_int(snapshot.get("num_seen_gts")) or 0
        merged["num_images_seen"] += _optional_int(snapshot.get("num_images_seen")) or 0
        merged["num_images_with_miss"] += _optional_int(snapshot.get("num_images_with_miss")) or 0
        merged["num_images_with_target"] += _optional_int(snapshot.get("num_images_with_target")) or 0

    merged.update(
        {
            "num_missed_gts": len(missed_gt_keys),
            "num_target_gts": len(target_gt_keys),
            "missed_gt_keys": sorted(missed_gt_keys),
            "target_gt_keys": sorted(target_gt_keys),
            "missed_region_histogram": dict(sorted(missed_region_histogram.items())),
            "target_region_histogram": dict(sorted(target_region_histogram.items())),
            "image_region_hotspots": dict(sorted(image_region_hotspots.items())),
            "target_image_region_hotspots": dict(sorted(target_image_region_hotspots.items())),
        }
    )
    return merged


def _match_gt(
    *,
    gt_box: torch.Tensor,
    gt_label: int,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    if pred_boxes.numel() == 0:
        return {
            "matched": False,
            "iou": None,
            "score": None,
            "best_iou": None,
            "best_iou_score": None,
            "best_score": None,
            "best_score_iou": None,
        }
    gt_box = gt_box.to(device=pred_boxes.device, dtype=torch.float32).reshape(1, 4)
    pred_boxes = pred_boxes.to(device=gt_box.device, dtype=torch.float32)
    pred_labels = pred_labels.to(device=gt_box.device, dtype=torch.long)
    pred_scores = pred_scores.to(device=gt_box.device, dtype=torch.float32)
    ious = box_ops.box_iou(gt_box, pred_boxes)[0].clamp(min=0.0, max=1.0)
    same_class_mask = pred_labels == int(gt_label)
    best_iou = None
    best_iou_score = None
    best_score = None
    best_score_iou = None
    if bool(same_class_mask.any().item()):
        same_class_indices = torch.where(same_class_mask)[0]
        same_class_ious = ious[same_class_indices]
        best_iou_local = int(torch.argmax(same_class_ious).item())
        best_iou_index = int(same_class_indices[best_iou_local].item())
        best_score_local = int(torch.argmax(pred_scores[same_class_indices]).item())
        best_score_index = int(same_class_indices[best_score_local].item())
        best_iou = float(ious[best_iou_index].item())
        best_iou_score = float(pred_scores[best_iou_index].item())
        best_score = float(pred_scores[best_score_index].item())
        best_score_iou = float(ious[best_score_index].item())
    candidate_mask = (
        same_class_mask
        & (pred_scores >= float(score_threshold))
        & (ious >= float(iou_threshold))
    )
    if not bool(candidate_mask.any().item()):
        return {
            "matched": False,
            "iou": None,
            "score": None,
            "best_iou": best_iou,
            "best_iou_score": best_iou_score,
            "best_score": best_score,
            "best_score_iou": best_score_iou,
        }
    candidate_indices = torch.where(candidate_mask)[0]
    candidate_ious = ious[candidate_indices]
    best_local = int(torch.argmax(candidate_ious).item())
    best_index = int(candidate_indices[best_local].item())
    return {
        "matched": True,
        "iou": float(ious[best_index].item()),
        "score": float(pred_scores[best_index].item()),
        "best_iou": best_iou,
        "best_iou_score": best_iou_score,
        "best_score": best_score,
        "best_score_iou": best_score_iou,
    }


def _parse_threshold(value: Any) -> float | str:
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return float(value)


def _is_auto_threshold(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "auto"


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
