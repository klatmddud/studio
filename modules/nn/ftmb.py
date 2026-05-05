from __future__ import annotations

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
from .mb import (
    _as_boxes_tensor,
    _as_float_tensor,
    _as_int_tensor,
    _box_to_tuple,
    _clamp_box,
    _extract_gt_ids,
    _is_auto_threshold,
    _record_key,
    _resolve_image_size,
    _valid_box,
)

LOCALIZATION = "localization"
CLASSIFICATION = "classification"
BOTH = "both"
MISSED = "missed"
DUPLICATE = "duplicate"
BACKGROUND = "background"

FAILURE_TYPES = (LOCALIZATION, CLASSIFICATION, BOTH, MISSED, DUPLICATE, BACKGROUND)
GT_FAILURE_TYPES = (LOCALIZATION, CLASSIFICATION, BOTH, MISSED)
PRED_FAILURE_TYPES = (DUPLICATE, BACKGROUND)


@dataclass(frozen=True, slots=True)
class FTMBConfig:
    enabled: bool = False
    start_epoch: int = 1
    mining_type: str = "online"
    score_threshold: float | str = "auto"
    iou_threshold: float | str = "auto"
    background_iou_threshold: float = 0.1
    max_records: int | None = None
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "FTMBConfig":
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

        mining = _mapping_or_empty(merged.get("mining"))
        matching = _mapping_or_empty(merged.get("matching"))
        background = _mapping_or_empty(merged.get("background"))
        config = cls(
            enabled=bool(merged.get("enabled", False)),
            start_epoch=int(merged.get("start_epoch", 1)),
            mining_type=str(mining.get("type", merged.get("mining_type", "online"))).lower(),
            score_threshold=matching.get("score_threshold", merged.get("score_threshold", "auto")),
            iou_threshold=matching.get("iou_threshold", merged.get("iou_threshold", "auto")),
            background_iou_threshold=float(
                background.get("iou_threshold", merged.get("background_iou_threshold", 0.1))
            ),
            max_records=None if merged.get("max_records") is None else int(merged.get("max_records")),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.start_epoch) < 0:
            raise ValueError("FTMB start_epoch must be >= 0.")
        if self.mining_type not in {"online", "offline"}:
            raise ValueError("FTMB mining_type must be either 'online' or 'offline'.")
        for name in ("score_threshold", "iou_threshold"):
            value = getattr(self, name)
            if _is_auto_threshold(value):
                continue
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"FTMB {name} must satisfy 0 <= value <= 1.")
        background_iou_threshold = float(self.background_iou_threshold)
        if not 0.0 <= background_iou_threshold <= 1.0:
            raise ValueError("FTMB background_iou_threshold must satisfy 0 <= value <= 1.")
        if (
            not _is_auto_threshold(self.iou_threshold)
            and background_iou_threshold > float(self.iou_threshold)
        ):
            raise ValueError("FTMB background_iou_threshold must be <= iou_threshold.")
        if self.max_records is not None and int(self.max_records) < 1:
            raise ValueError("FTMB max_records must be null or >= 1.")

    def resolve_detector_thresholds(
        self,
        *,
        detector_score_threshold: float | None,
        detector_iou_threshold: float | None,
    ) -> "FTMBConfig":
        score_threshold = self.score_threshold
        iou_threshold = self.iou_threshold
        if _is_auto_threshold(score_threshold):
            if detector_score_threshold is None:
                raise ValueError(
                    "FTMB matching.score_threshold='auto' requires the detector final score threshold."
                )
            score_threshold = float(detector_score_threshold)
        if _is_auto_threshold(iou_threshold):
            if detector_iou_threshold is None:
                raise ValueError(
                    "FTMB matching.iou_threshold='auto' requires the detector final IoU threshold."
                )
            iou_threshold = float(detector_iou_threshold)
        if score_threshold == self.score_threshold and iou_threshold == self.iou_threshold:
            return self
        config = replace(
            self,
            score_threshold=float(score_threshold),
            iou_threshold=float(iou_threshold),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "start_epoch": self.start_epoch,
            "mining_type": self.mining_type,
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold,
            "background_iou_threshold": self.background_iou_threshold,
            "max_records": self.max_records,
            "arch": self.arch,
        }


@dataclass(slots=True)
class FTMBGTRecord:
    record_key: str
    image_id: str
    gt_id: str | None
    gt_class: int
    bbox_xyxy: tuple[float, float, float, float]
    failure_type: str | None = None
    last_epoch: int = 0
    last_step: int = 0
    total_seen: int = 0
    total_failed: int = 0
    type_counts: dict[str, int] = field(default_factory=dict)
    consecutive_type: str | None = None
    consecutive_count: int = 0
    best_iou: float | None = None
    best_score: float | None = None
    assigned_pred_class: int | None = None
    assigned_pred_iou: float | None = None
    assigned_pred_score: float | None = None

    def update(
        self,
        *,
        failure_type: str | None,
        bbox_xyxy: tuple[float, float, float, float],
        epoch: int,
        step: int,
        best_iou: float | None,
        best_score: float | None,
        assigned_pred_class: int | None,
        assigned_pred_iou: float | None,
        assigned_pred_score: float | None,
    ) -> None:
        self.bbox_xyxy = bbox_xyxy
        self.failure_type = failure_type
        self.last_epoch = int(epoch)
        self.last_step = int(step)
        self.total_seen += 1
        self.best_iou = _optional_float(best_iou)
        self.best_score = _optional_float(best_score)
        self.assigned_pred_class = None if assigned_pred_class is None else int(assigned_pred_class)
        self.assigned_pred_iou = _optional_float(assigned_pred_iou)
        self.assigned_pred_score = _optional_float(assigned_pred_score)

        if failure_type is None:
            self.consecutive_type = None
            self.consecutive_count = 0
            return

        if failure_type not in GT_FAILURE_TYPES:
            raise ValueError(f"Unsupported FTMB GT failure type: {failure_type}")
        self.total_failed += 1
        self.type_counts[failure_type] = int(self.type_counts.get(failure_type, 0)) + 1
        if self.consecutive_type == failure_type:
            self.consecutive_count += 1
        else:
            self.consecutive_type = failure_type
            self.consecutive_count = 1

    def to_state(self) -> dict[str, Any]:
        return {
            "record_key": self.record_key,
            "image_id": self.image_id,
            "gt_id": self.gt_id,
            "gt_class": self.gt_class,
            "bbox_xyxy": list(self.bbox_xyxy),
            "failure_type": self.failure_type,
            "last_epoch": self.last_epoch,
            "last_step": self.last_step,
            "total_seen": self.total_seen,
            "total_failed": self.total_failed,
            "type_counts": dict(self.type_counts),
            "consecutive_type": self.consecutive_type,
            "consecutive_count": self.consecutive_count,
            "best_iou": self.best_iou,
            "best_score": self.best_score,
            "assigned_pred_class": self.assigned_pred_class,
            "assigned_pred_iou": self.assigned_pred_iou,
            "assigned_pred_score": self.assigned_pred_score,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FTMBGTRecord":
        bbox = state.get("bbox_xyxy", (0.0, 0.0, 0.0, 0.0))
        bbox_values = tuple(float(value) for value in list(bbox)[:4])
        if len(bbox_values) != 4:
            bbox_values = (0.0, 0.0, 0.0, 0.0)
        raw_type_counts = state.get("type_counts", {})
        type_counts = (
            {str(key): int(value) for key, value in raw_type_counts.items()}
            if isinstance(raw_type_counts, Mapping)
            else {}
        )
        return cls(
            record_key=str(state.get("record_key", "")),
            image_id=str(state.get("image_id", "")),
            gt_id=None if state.get("gt_id") is None else str(state.get("gt_id")),
            gt_class=int(state.get("gt_class", 0)),
            bbox_xyxy=bbox_values,
            failure_type=None if state.get("failure_type") is None else str(state.get("failure_type")),
            last_epoch=int(state.get("last_epoch", 0)),
            last_step=int(state.get("last_step", 0)),
            total_seen=int(state.get("total_seen", 0)),
            total_failed=int(state.get("total_failed", 0)),
            type_counts=type_counts,
            consecutive_type=None if state.get("consecutive_type") is None else str(state.get("consecutive_type")),
            consecutive_count=int(state.get("consecutive_count", 0)),
            best_iou=_optional_float(state.get("best_iou")),
            best_score=_optional_float(state.get("best_score")),
            assigned_pred_class=None
            if state.get("assigned_pred_class") is None
            else int(state.get("assigned_pred_class")),
            assigned_pred_iou=_optional_float(state.get("assigned_pred_iou")),
            assigned_pred_score=_optional_float(state.get("assigned_pred_score")),
        )


class FailureTypeMemoryBank(nn.Module):
    """Train-time memory for failure-type-aware hard replay analysis."""

    def __init__(self, config: FTMBConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._records: dict[str, FTMBGTRecord] = {}
        self._image_index: dict[str, set[str]] = defaultdict(set)
        self._stats: Counter[str] = Counter()
        self._epoch_step_summaries: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._prediction_events: list[dict[str, Any]] = []

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def reset(self) -> None:
        self._records.clear()
        self._image_index.clear()
        self._stats.clear()
        self._epoch_step_summaries.clear()
        self._prediction_events.clear()

    def is_active(self, epoch: int | None = None) -> bool:
        epoch_value = self.current_epoch if epoch is None else int(epoch)
        return bool(self.config.enabled and epoch_value >= int(self.config.start_epoch))

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
            raise ValueError("FTMB update requires the same number of targets and detections.")

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
        self._epoch_step_summaries[epoch_value].append(_step_summary(epoch_value, int(step), stats))
        self._enforce_max_records()
        return {key: int(value) for key, value in stats.items()}

    def get_records(self, image_id: Any | None = None) -> list[FTMBGTRecord]:
        if image_id is None:
            return list(self._records.values())
        normalized = _normalize_image_id(image_id)
        return [
            self._records[key]
            for key in sorted(self._image_index.get(normalized, set()))
            if key in self._records
        ]

    def get_prediction_events(self, epoch: int | None = None) -> list[dict[str, Any]]:
        if epoch is None:
            return [dict(event) for event in self._prediction_events]
        epoch_value = int(epoch)
        return [
            dict(event)
            for event in self._prediction_events
            if int(event.get("epoch", 0)) == epoch_value
        ]

    def epoch_snapshot(self, epoch: int | None = None) -> dict[str, Any]:
        epoch_value = int(self.current_epoch if epoch is None else epoch)
        step_summaries = list(self._epoch_step_summaries.get(epoch_value, []))
        counts = Counter()
        for summary in step_summaries:
            for failure_type in FAILURE_TYPES:
                counts[failure_type] += int(summary.get(f"{failure_type}_count", 0))
        return {
            "epoch": epoch_value,
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "mining_type": self.config.mining_type,
            "score_threshold": self.config.score_threshold,
            "iou_threshold": self.config.iou_threshold,
            "background_iou_threshold": self.config.background_iou_threshold,
            "num_images_seen": sum(int(summary.get("images_seen", 0)) for summary in step_summaries),
            "num_gt_seen": sum(int(summary.get("gt_seen", 0)) for summary in step_summaries),
            "num_gt_detected": sum(int(summary.get("gt_detected", 0)) for summary in step_summaries),
            "num_predictions_seen": sum(int(summary.get("predictions_seen", 0)) for summary in step_summaries),
            **{f"{failure_type}_count": int(counts[failure_type]) for failure_type in FAILURE_TYPES},
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "records": {key: record.to_state() for key, record in self._records.items()},
            "stats": dict(self._stats),
            "prediction_events": list(self._prediction_events),
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            self.reset()
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))
        records: dict[str, FTMBGTRecord] = {}
        raw_records = state.get("records", {})
        if isinstance(raw_records, Mapping):
            for key, raw_record in raw_records.items():
                if isinstance(raw_record, Mapping):
                    record = FTMBGTRecord.from_state(raw_record)
                    record.record_key = str(key)
                    records[record.record_key] = record
        self._records = records
        self._rebuild_image_index()
        raw_stats = state.get("stats", {})
        self._stats = Counter({str(k): int(v) for k, v in raw_stats.items()}) if isinstance(raw_stats, Mapping) else Counter()
        self._prediction_events = [
            dict(event)
            for event in _as_mapping_list(state.get("prediction_events"))
        ]

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
        pred_boxes = _as_boxes_tensor(detection.get("boxes"))
        pred_labels = _as_int_tensor(detection.get("labels"), length=int(pred_boxes.shape[0]))
        pred_scores = _as_float_tensor(
            detection.get("scores"),
            length=int(pred_boxes.shape[0]),
            fill=1.0,
        )
        image_id = _normalize_image_id(target.get("image_id", torch.tensor(image_index)))
        stats["images_seen"] += 1
        stats["predictions_seen"] += int((pred_scores >= float(self.config.score_threshold)).sum().item())

        if gt_boxes.numel() == 0:
            stats.update(
                self._record_prediction_failures(
                    image_id=image_id,
                    gt_items=[],
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                    matched_pred_indices=set(),
                    epoch=epoch,
                    step=step,
                )
            )
            return stats

        height, width = _resolve_image_size(
            target=target,
            image_size=image_size,
            boxes=gt_boxes,
        )
        gt_ids = _extract_gt_ids(target, int(gt_boxes.shape[0]))
        gt_items: list[dict[str, Any]] = []
        for gt_index, gt_box in enumerate(gt_boxes):
            if not _valid_box(gt_box):
                stats["invalid_gt"] += 1
                continue
            clamped_box = _clamp_box(gt_box, height=height, width=width)
            if not _valid_box(clamped_box):
                stats["invalid_gt_clamped"] += 1
                continue
            bbox_tuple = _box_to_tuple(clamped_box)
            gt_label = int(gt_labels[gt_index].item())
            gt_id = gt_ids[gt_index]
            gt_items.append(
                {
                    "source_index": gt_index,
                    "image_id": image_id,
                    "gt_id": gt_id,
                    "gt_class": gt_label,
                    "bbox_xyxy": bbox_tuple,
                    "record_key": _record_key(
                        image_id=image_id,
                        gt_id=gt_id,
                        gt_class=gt_label,
                        bbox_xyxy=bbox_tuple,
                        height=height,
                        width=width,
                    ),
                }
            )

        if not gt_items:
            return stats

        gt_tensor = torch.tensor([item["bbox_xyxy"] for item in gt_items], dtype=torch.float32)
        if pred_boxes.numel() > 0:
            gt_tensor = gt_tensor.to(device=pred_boxes.device)
        pred_boxes = pred_boxes.to(device=gt_tensor.device, dtype=torch.float32)
        pred_labels = pred_labels.to(device=gt_tensor.device, dtype=torch.long)
        pred_scores = pred_scores.to(device=gt_tensor.device, dtype=torch.float32)
        active_mask = pred_scores >= float(self.config.score_threshold)
        active_indices = torch.where(active_mask)[0]
        iou_matrix = (
            box_ops.box_iou(gt_tensor, pred_boxes).clamp(min=0.0, max=1.0)
            if pred_boxes.numel() > 0
            else torch.zeros((len(gt_items), 0), dtype=torch.float32, device=gt_tensor.device)
        )
        matches = _match_correct_pairs(
            gt_items=gt_items,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            iou_matrix=iou_matrix,
            active_indices=active_indices,
            iou_threshold=float(self.config.iou_threshold),
        )
        matched_pred_indices = set(matches.values())

        for local_gt_index, item in enumerate(gt_items):
            stats["gt_seen"] += 1
            matched_pred_index = matches.get(local_gt_index)
            if matched_pred_index is not None:
                stats["gt_detected"] += 1
                self._update_gt_record(
                    item=item,
                    failure_type=None,
                    epoch=epoch,
                    step=step,
                    pred_index=matched_pred_index,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                    iou_row=iou_matrix[local_gt_index],
                )
                continue

            failure_type, pred_index = _classify_gt_failure(
                gt_class=int(item["gt_class"]),
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                iou_row=iou_matrix[local_gt_index],
                active_indices=active_indices,
                iou_threshold=float(self.config.iou_threshold),
                background_iou_threshold=float(self.config.background_iou_threshold),
            )
            stats[f"{failure_type}_count"] += 1
            self._update_gt_record(
                item=item,
                failure_type=failure_type,
                epoch=epoch,
                step=step,
                pred_index=pred_index,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                iou_row=iou_matrix[local_gt_index],
            )

        stats.update(
            self._record_prediction_failures(
                image_id=image_id,
                gt_items=gt_items,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                matched_pred_indices=matched_pred_indices,
                epoch=epoch,
                step=step,
                iou_matrix=iou_matrix,
            )
        )
        return stats

    def _update_gt_record(
        self,
        *,
        item: Mapping[str, Any],
        failure_type: str | None,
        epoch: int,
        step: int,
        pred_index: int | None,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        iou_row: torch.Tensor,
    ) -> None:
        record_key = str(item["record_key"])
        record = self._records.get(record_key)
        if record is None:
            record = FTMBGTRecord(
                record_key=record_key,
                image_id=str(item["image_id"]),
                gt_id=None if item.get("gt_id") is None else str(item.get("gt_id")),
                gt_class=int(item["gt_class"]),
                bbox_xyxy=item["bbox_xyxy"],
            )
            self._records[record_key] = record
            self._image_index[str(item["image_id"])].add(record_key)

        active_best = _best_prediction(iou_row=iou_row, pred_scores=pred_scores)
        assigned_class = None
        assigned_iou = None
        assigned_score = None
        if pred_index is not None:
            assigned_class = int(pred_labels[pred_index].item())
            assigned_iou = float(iou_row[pred_index].item())
            assigned_score = float(pred_scores[pred_index].item())
        record.update(
            failure_type=failure_type,
            bbox_xyxy=item["bbox_xyxy"],
            epoch=epoch,
            step=step,
            best_iou=active_best["best_iou"],
            best_score=active_best["best_score"],
            assigned_pred_class=assigned_class,
            assigned_pred_iou=assigned_iou,
            assigned_pred_score=assigned_score,
        )

    def _record_prediction_failures(
        self,
        *,
        image_id: str,
        gt_items: Sequence[Mapping[str, Any]],
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        matched_pred_indices: set[int],
        epoch: int,
        step: int,
        iou_matrix: torch.Tensor | None = None,
    ) -> Counter[str]:
        stats: Counter[str] = Counter()
        if pred_boxes.numel() == 0:
            return stats
        if iou_matrix is None:
            if gt_items:
                gt_tensor = torch.tensor(
                    [item["bbox_xyxy"] for item in gt_items],
                    dtype=torch.float32,
                    device=pred_boxes.device,
                )
                iou_matrix = box_ops.box_iou(gt_tensor, pred_boxes).clamp(min=0.0, max=1.0)
            else:
                iou_matrix = torch.zeros((0, pred_boxes.shape[0]), dtype=torch.float32, device=pred_boxes.device)

        active_indices = torch.where(pred_scores >= float(self.config.score_threshold))[0].tolist()
        for pred_index in active_indices:
            if int(pred_index) in matched_pred_indices:
                continue
            failure_type, gt_index, iou_value = _classify_prediction_failure(
                pred_index=int(pred_index),
                gt_items=gt_items,
                pred_label=int(pred_labels[pred_index].item()),
                iou_matrix=iou_matrix,
                iou_threshold=float(self.config.iou_threshold),
                background_iou_threshold=float(self.config.background_iou_threshold),
            )
            if failure_type is None:
                continue
            stats[f"{failure_type}_count"] += 1
            matched_gt = None if gt_index is None else gt_items[gt_index]
            event = {
                "epoch": int(epoch),
                "step": int(step),
                "image_id": image_id,
                "failure_type": failure_type,
                "pred_class": int(pred_labels[pred_index].item()),
                "pred_score": float(pred_scores[pred_index].item()),
                "pred_bbox_xyxy": _tensor_box_to_list(pred_boxes[pred_index]),
                "matched_gt_id": None if matched_gt is None or matched_gt.get("gt_id") is None else str(matched_gt.get("gt_id")),
                "matched_gt_class": None if matched_gt is None else int(matched_gt.get("gt_class", 0)),
                "matched_iou": iou_value,
            }
            self._prediction_events.append(event)
        return stats

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


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def build_ftmb_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    detector_score_threshold: float | None = None,
    detector_iou_threshold: float | None = None,
) -> FailureTypeMemoryBank | None:
    config = load_ftmb_config(path, arch=arch)
    if not config.enabled:
        return None
    config = config.resolve_detector_thresholds(
        detector_score_threshold=detector_score_threshold,
        detector_iou_threshold=detector_iou_threshold,
    )
    return FailureTypeMemoryBank(config)


def load_ftmb_config(path: str | Path, *, arch: str | None = None) -> FTMBConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"FTMB YAML must contain a mapping at the top level: {config_path}")
    return FTMBConfig.from_mapping(raw, arch=arch)


def build_ftmb_from_config(
    raw_config: Mapping[str, Any] | FTMBConfig,
    *,
    arch: str | None = None,
    detector_score_threshold: float | None = None,
    detector_iou_threshold: float | None = None,
) -> FailureTypeMemoryBank | None:
    config = raw_config if isinstance(raw_config, FTMBConfig) else FTMBConfig.from_mapping(raw_config, arch=arch)
    if not config.enabled:
        return None
    config = config.resolve_detector_thresholds(
        detector_score_threshold=detector_score_threshold,
        detector_iou_threshold=detector_iou_threshold,
    )
    return FailureTypeMemoryBank(config)


def merge_ftmb_epoch_snapshots(
    snapshots: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any] | None:
    valid_snapshots = [snapshot for snapshot in snapshots if isinstance(snapshot, Mapping)]
    if not valid_snapshots:
        return None

    counts = Counter()
    merged: dict[str, Any] = {
        "epoch": max(_optional_int(snapshot.get("epoch")) or 0 for snapshot in valid_snapshots),
        "enabled": bool(valid_snapshots[0].get("enabled", True)),
        "arch": valid_snapshots[0].get("arch"),
        "mining_type": valid_snapshots[0].get("mining_type"),
        "score_threshold": valid_snapshots[0].get("score_threshold"),
        "iou_threshold": valid_snapshots[0].get("iou_threshold"),
        "background_iou_threshold": valid_snapshots[0].get("background_iou_threshold"),
        "num_images_seen": 0,
        "num_gt_seen": 0,
        "num_gt_detected": 0,
        "num_predictions_seen": 0,
    }
    for snapshot in valid_snapshots:
        for key in ("num_images_seen", "num_gt_seen", "num_gt_detected", "num_predictions_seen"):
            merged[key] += _optional_int(snapshot.get(key)) or 0
        for failure_type in FAILURE_TYPES:
            counts[failure_type] += _optional_int(snapshot.get(f"{failure_type}_count")) or 0

    merged.update({f"{failure_type}_count": int(counts[failure_type]) for failure_type in FAILURE_TYPES})
    return merged


def _match_correct_pairs(
    *,
    gt_items: Sequence[Mapping[str, Any]],
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_matrix: torch.Tensor,
    active_indices: torch.Tensor,
    iou_threshold: float,
) -> dict[int, int]:
    pairs: list[tuple[float, float, int, int]] = []
    active_set = {int(index.item()) for index in active_indices}
    for gt_index, item in enumerate(gt_items):
        gt_class = int(item["gt_class"])
        for pred_index in active_set:
            if int(pred_labels[pred_index].item()) != gt_class:
                continue
            iou = float(iou_matrix[gt_index, pred_index].item())
            if iou < float(iou_threshold):
                continue
            pairs.append((iou, float(pred_scores[pred_index].item()), gt_index, pred_index))
    pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: dict[int, int] = {}
    for _, _, gt_index, pred_index in pairs:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches[gt_index] = pred_index
    return matches


def _classify_gt_failure(
    *,
    gt_class: int,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_row: torch.Tensor,
    active_indices: torch.Tensor,
    iou_threshold: float,
    background_iou_threshold: float,
) -> tuple[str, int | None]:
    active = [int(index.item()) for index in active_indices]
    if not active:
        return MISSED, None

    wrong_high = _best_index(
        active,
        iou_row=iou_row,
        pred_scores=pred_scores,
        predicate=lambda index: int(pred_labels[index].item()) != int(gt_class)
        and float(iou_row[index].item()) >= float(iou_threshold),
    )
    if wrong_high is not None:
        return CLASSIFICATION, wrong_high

    same_near = _best_index(
        active,
        iou_row=iou_row,
        pred_scores=pred_scores,
        predicate=lambda index: int(pred_labels[index].item()) == int(gt_class)
        and float(iou_row[index].item()) >= float(background_iou_threshold),
    )
    if same_near is not None:
        return LOCALIZATION, same_near

    wrong_near = _best_index(
        active,
        iou_row=iou_row,
        pred_scores=pred_scores,
        predicate=lambda index: int(pred_labels[index].item()) != int(gt_class)
        and float(iou_row[index].item()) >= float(background_iou_threshold),
    )
    if wrong_near is not None:
        return BOTH, wrong_near

    return MISSED, None


def _classify_prediction_failure(
    *,
    pred_index: int,
    gt_items: Sequence[Mapping[str, Any]],
    pred_label: int,
    iou_matrix: torch.Tensor,
    iou_threshold: float,
    background_iou_threshold: float,
) -> tuple[str | None, int | None, float | None]:
    if not gt_items or iou_matrix.shape[0] == 0:
        return BACKGROUND, None, None

    column = iou_matrix[:, int(pred_index)]
    best_iou, best_gt_index = torch.max(column, dim=0)
    best_iou_value = float(best_iou.item())
    best_gt = int(best_gt_index.item())
    same_class_duplicate = (
        best_iou_value >= float(iou_threshold)
        and int(gt_items[best_gt]["gt_class"]) == int(pred_label)
    )
    if same_class_duplicate:
        return DUPLICATE, best_gt, best_iou_value
    if best_iou_value < float(background_iou_threshold):
        return BACKGROUND, best_gt, best_iou_value
    return None, None, None


def _best_index(
    indices: Sequence[int],
    *,
    iou_row: torch.Tensor,
    pred_scores: torch.Tensor,
    predicate,
) -> int | None:
    candidates = [index for index in indices if predicate(index)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda index: (
            float(iou_row[index].item()),
            float(pred_scores[index].item()),
        ),
    )


def _best_prediction(
    *,
    iou_row: torch.Tensor,
    pred_scores: torch.Tensor,
) -> dict[str, float | None]:
    if iou_row.numel() == 0:
        return {"best_iou": None, "best_score": None}
    best_iou_index = int(torch.argmax(iou_row).item())
    best_score_index = int(torch.argmax(pred_scores).item())
    return {
        "best_iou": float(iou_row[best_iou_index].item()),
        "best_score": float(pred_scores[best_score_index].item()),
    }


def _step_summary(epoch: int, step: int, stats: Counter[str]) -> dict[str, Any]:
    summary = {
        "epoch": int(epoch),
        "step": int(step),
        "images_seen": int(stats.get("images_seen", 0)),
        "gt_seen": int(stats.get("gt_seen", 0)),
        "gt_detected": int(stats.get("gt_detected", 0)),
        "predictions_seen": int(stats.get("predictions_seen", 0)),
    }
    summary.update({f"{failure_type}_count": int(stats.get(f"{failure_type}_count", 0)) for failure_type in FAILURE_TYPES})
    return summary


def _tensor_box_to_list(box: torch.Tensor) -> list[float]:
    return [float(value) for value in box.detach().cpu().to(dtype=torch.float32).flatten().tolist()[:4]]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _as_mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]
