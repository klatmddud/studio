from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Sampler

from modules.nn import MDMBPlus, MDMBPlusEntry, normalize_arch

from .config import load_yaml_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARD_REPLAY_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "hard_replay.yaml"


@dataclass(frozen=True, slots=True)
class FCDRConfig:
    enabled: bool = False
    counterfactual_ratio: float = 0.5
    min_severity: float = 0.0
    max_crops_per_gt_per_epoch: int = 1
    crop_context_scale: float = 1.0
    min_crop_context_px: int = 16
    overlap_threshold: float = 0.1
    copy_paste_prob: float = 0.0
    pair_replay_prob: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "FCDRConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", False)),
            counterfactual_ratio=float(data.get("counterfactual_ratio", 0.5)),
            min_severity=float(data.get("min_severity", 0.0)),
            max_crops_per_gt_per_epoch=int(data.get("max_crops_per_gt_per_epoch", 1)),
            crop_context_scale=float(data.get("crop_context_scale", 1.0)),
            min_crop_context_px=int(data.get("min_crop_context_px", 16)),
            overlap_threshold=float(data.get("overlap_threshold", 0.1)),
            copy_paste_prob=float(data.get("copy_paste_prob", 0.0)),
            pair_replay_prob=float(data.get("pair_replay_prob", 0.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= self.counterfactual_ratio <= 1.0:
            raise ValueError("FCDR counterfactual_ratio must satisfy 0 <= value <= 1.")
        if self.min_severity < 0.0:
            raise ValueError("FCDR min_severity must be >= 0.")
        if self.max_crops_per_gt_per_epoch < 1:
            raise ValueError("FCDR max_crops_per_gt_per_epoch must be >= 1.")
        if self.crop_context_scale < 0.0:
            raise ValueError("FCDR crop_context_scale must be >= 0.")
        if self.min_crop_context_px < 0:
            raise ValueError("FCDR min_crop_context_px must be >= 0.")
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("FCDR overlap_threshold must satisfy 0 <= value <= 1.")
        if not 0.0 <= self.copy_paste_prob <= 1.0:
            raise ValueError("FCDR copy_paste_prob must satisfy 0 <= value <= 1.")
        if not 0.0 <= self.pair_replay_prob <= 1.0:
            raise ValueError("FCDR pair_replay_prob must satisfy 0 <= value <= 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "counterfactual_ratio": self.counterfactual_ratio,
            "min_severity": self.min_severity,
            "max_crops_per_gt_per_epoch": self.max_crops_per_gt_per_epoch,
            "crop_context_scale": self.crop_context_scale,
            "min_crop_context_px": self.min_crop_context_px,
            "overlap_threshold": self.overlap_threshold,
            "copy_paste_prob": self.copy_paste_prob,
            "pair_replay_prob": self.pair_replay_prob,
        }


@dataclass(frozen=True, slots=True)
class HardReplayConfig:
    enabled: bool = False
    beta: float = 1.0
    temperature: float = 1.0
    replay_ratio: float = 0.25
    max_image_weight: float = 5.0
    min_replay_weight: float = 1.0
    replacement: bool = True
    max_replays_per_gt_per_epoch: int = 4
    replay_recency_window: int = 3
    arch: str | None = None
    fcdr: FCDRConfig = field(default_factory=FCDRConfig)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "HardReplayConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))

        overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    overrides = dict(selected)

        fcdr_data: dict[str, Any] = {}
        top_fcdr = data.get("fcdr", {})
        if isinstance(top_fcdr, Mapping):
            fcdr_data.update(top_fcdr)
        override_fcdr = overrides.get("fcdr", {})
        if isinstance(override_fcdr, Mapping):
            fcdr_data.update(override_fcdr)

        config = cls(
            enabled=bool(overrides.get("enabled", data.get("enabled", False))),
            beta=float(overrides.get("beta", data.get("beta", 1.0))),
            temperature=float(overrides.get("temperature", data.get("temperature", 1.0))),
            replay_ratio=float(overrides.get("replay_ratio", data.get("replay_ratio", 0.25))),
            max_image_weight=float(
                overrides.get("max_image_weight", data.get("max_image_weight", 5.0))
            ),
            min_replay_weight=float(
                overrides.get("min_replay_weight", data.get("min_replay_weight", 1.0))
            ),
            replacement=bool(overrides.get("replacement", data.get("replacement", True))),
            max_replays_per_gt_per_epoch=int(
                overrides.get(
                    "max_replays_per_gt_per_epoch",
                    data.get("max_replays_per_gt_per_epoch", 4),
                )
            ),
            replay_recency_window=int(
                overrides.get("replay_recency_window", data.get("replay_recency_window", 3))
            ),
            arch=normalized_arch,
            fcdr=FCDRConfig.from_mapping(fcdr_data),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.beta < 0.0:
            raise ValueError("Hard Replay beta must be >= 0.")
        if self.temperature <= 0.0:
            raise ValueError("Hard Replay temperature must be > 0.")
        if not 0.0 <= self.replay_ratio < 1.0:
            raise ValueError("Hard Replay replay_ratio must satisfy 0 <= value < 1.")
        if self.max_image_weight <= 0.0:
            raise ValueError("Hard Replay max_image_weight must be > 0.")
        if self.min_replay_weight <= 0.0:
            raise ValueError("Hard Replay min_replay_weight must be > 0.")
        if self.max_replays_per_gt_per_epoch < 1:
            raise ValueError("Hard Replay max_replays_per_gt_per_epoch must be >= 1.")
        if self.replay_recency_window < 0:
            raise ValueError("Hard Replay replay_recency_window must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "beta": self.beta,
            "temperature": self.temperature,
            "replay_ratio": self.replay_ratio,
            "max_image_weight": self.max_image_weight,
            "min_replay_weight": self.min_replay_weight,
            "replacement": self.replacement,
            "max_replays_per_gt_per_epoch": self.max_replays_per_gt_per_epoch,
            "replay_recency_window": self.replay_recency_window,
            "arch": self.arch,
            "fcdr": self.fcdr.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReplayCrop:
    dataset_index: int
    gt_uid: str
    image_id: str
    class_id: int
    failure_type: str
    mode: str
    crop_box_abs: tuple[int, int, int, int]
    source_bbox_abs: tuple[int, int, int, int]
    severity: float
    support_box_abs: tuple[int, int, int, int] | None
    sampling_weight: float
    replay_cap: int


@dataclass(frozen=True, slots=True)
class ReplayIndex:
    enabled: bool
    image_weights: dict[str, float] = field(default_factory=dict)
    replay_crops: list[ReplayCrop] = field(default_factory=list)
    replay_gt_ids: set[str] = field(default_factory=set)
    replay_dataset_indices: list[int] = field(default_factory=list)
    replay_sampling_weights: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.float32)
    )
    active_gt_counts: dict[int, int] = field(default_factory=dict)
    replay_caps: dict[int, int] = field(default_factory=dict)
    summary: dict[str, float | int | bool] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        *,
        enabled: bool,
        epoch: int = 0,
        warmup_active: bool = False,
    ) -> "ReplayIndex":
        return cls(
            enabled=enabled,
            summary={
                "enabled": enabled,
                "epoch": int(epoch),
                "warmup_active": bool(warmup_active),
                "replay_num_images": 0,
                "replay_num_crops": 0,
                "replay_num_active_gt": 0,
                "replay_mean_image_weight": 0.0,
                "replay_mean_gt_severity": 0.0,
                "replay_ratio_requested": 0.0,
                "replay_ratio_effective": 0.0,
                "replay_exposure_per_gt": 0.0,
                "fcdr_enabled": False,
                "fcdr_num_crops": 0,
                "fcdr_counterfactual_ratio_requested": 0.0,
                "fcdr_ratio_effective": 0.0,
                "fcdr_samples": 0,
                "fcdr_unique_crops": 0,
            },
        )


class HardReplayPlanner:
    def __init__(self, config: HardReplayConfig) -> None:
        self.config = config

    def build_epoch_index(
        self,
        *,
        mdmbpp: MDMBPlus | None,
        dataset: Any,
        epoch: int,
    ) -> ReplayIndex:
        if not self.config.enabled:
            return self._empty_index(enabled=False, epoch=epoch, warmup_active=False)
        if mdmbpp is None:
            raise ValueError("Hard Replay requires an initialized mdmbpp module when enabled.")

        should_update = mdmbpp.should_update(epoch=epoch)
        if not should_update:
            return self._empty_index(enabled=True, epoch=epoch, warmup_active=True)

        image_ids = getattr(dataset, "image_ids", None)
        if not isinstance(image_ids, Sequence):
            raise TypeError("Hard Replay requires a dataset exposing a sequence-like image_ids field.")

        image_weights: dict[str, float] = {}
        replay_dataset_indices: list[int] = []
        replay_sampling_weights: list[float] = []
        replay_gt_ids: set[str] = set()
        active_gt_counts: dict[int, int] = {}
        replay_caps: dict[int, int] = {}
        severity_sum = 0.0
        replay_crops: list[ReplayCrop] = []
        fcdr_failure_counts: Counter[str] = Counter()
        fcdr_mode_counts: Counter[str] = Counter()

        for dataset_index, image_id in enumerate(image_ids):
            image_key = str(image_id)
            entries = self._select_replay_entries(mdmbpp, image_key=image_key, epoch=epoch)
            if not entries:
                continue

            image_severity = float(sum(entry.severity for entry in entries))
            raw_weight = 1.0 + self.config.beta * image_severity
            clipped_weight = min(self.config.max_image_weight, raw_weight)
            clipped_weight = max(self.config.min_replay_weight, clipped_weight)
            sampling_weight = clipped_weight**self.config.temperature

            image_weights[image_key] = clipped_weight
            replay_dataset_indices.append(dataset_index)
            replay_sampling_weights.append(float(sampling_weight))
            active_gt_counts[dataset_index] = len(entries)
            replay_caps[dataset_index] = self.config.max_replays_per_gt_per_epoch
            severity_sum += image_severity

            for entry in entries:
                replay_gt_ids.add(entry.gt_uid)

            image_replay_crops = self._build_replay_crops(
                mdmbpp=mdmbpp,
                dataset=dataset,
                dataset_index=dataset_index,
                image_id=image_id,
                entries=entries,
            )
            replay_crops.extend(image_replay_crops)
            for crop in image_replay_crops:
                fcdr_failure_counts[crop.failure_type] += 1
                fcdr_mode_counts[crop.mode] += 1

        active_images = len(replay_dataset_indices)
        mean_image_weight = 0.0
        if active_images > 0:
            mean_image_weight = sum(image_weights.values()) / float(active_images)

        mean_gt_severity = 0.0
        if replay_gt_ids:
            mean_gt_severity = severity_sum / float(len(replay_gt_ids))

        fcdr_summary = self._build_fcdr_summary(
            replay_crops=replay_crops,
            failure_counts=fcdr_failure_counts,
            mode_counts=fcdr_mode_counts,
        )

        return ReplayIndex(
            enabled=True,
            image_weights=image_weights,
            replay_crops=replay_crops,
            replay_gt_ids=replay_gt_ids,
            replay_dataset_indices=replay_dataset_indices,
            replay_sampling_weights=torch.tensor(replay_sampling_weights, dtype=torch.float32),
            active_gt_counts=active_gt_counts,
            replay_caps=replay_caps,
            summary={
                "enabled": True,
                "epoch": int(epoch),
                "warmup_active": False,
                "replay_num_images": active_images,
                "replay_num_crops": len(replay_crops),
                "replay_num_active_gt": len(replay_gt_ids),
                "replay_mean_image_weight": mean_image_weight,
                "replay_mean_gt_severity": mean_gt_severity,
                "replay_ratio_requested": self.config.replay_ratio,
                "replay_ratio_effective": 0.0,
                "replay_exposure_per_gt": 0.0,
                **fcdr_summary,
            },
        )

    def _empty_index(
        self,
        *,
        enabled: bool,
        epoch: int,
        warmup_active: bool,
    ) -> ReplayIndex:
        replay_index = ReplayIndex.empty(
            enabled=enabled,
            epoch=epoch,
            warmup_active=warmup_active,
        )
        replay_index.summary.update(
            {
                "fcdr_enabled": bool(enabled and self.config.fcdr.enabled),
                "fcdr_counterfactual_ratio_requested": (
                    self.config.fcdr.counterfactual_ratio
                    if enabled and self.config.fcdr.enabled
                    else 0.0
                ),
                "fcdr_copy_paste_prob": (
                    self.config.fcdr.copy_paste_prob
                    if enabled and self.config.fcdr.enabled
                    else 0.0
                ),
                "fcdr_pair_replay_prob": (
                    self.config.fcdr.pair_replay_prob
                    if enabled and self.config.fcdr.enabled
                    else 0.0
                ),
            }
        )
        return replay_index

    def _select_replay_entries(
        self,
        mdmbpp: MDMBPlus,
        *,
        image_key: str,
        epoch: int,
    ) -> list[MDMBPlusEntry]:
        entries = list(mdmbpp.get_dense_targets(image_key))
        if self.config.replay_recency_window <= 0:
            return entries

        selected: list[MDMBPlusEntry] = []
        for entry in entries:
            record = mdmbpp.get_record(entry.gt_uid)
            if record is None or record.last_failure_epoch is None:
                selected.append(entry)
                continue
            if epoch - int(record.last_failure_epoch) <= self.config.replay_recency_window:
                selected.append(entry)
        return selected

    def _build_replay_crops(
        self,
        *,
        mdmbpp: MDMBPlus,
        dataset: Any,
        dataset_index: int,
        image_id: Any,
        entries: Sequence[MDMBPlusEntry],
    ) -> list[ReplayCrop]:
        if not self.config.fcdr.enabled:
            return []

        coco = getattr(dataset, "coco", None)
        image_ids = getattr(dataset, "image_ids", None)
        if coco is None or not isinstance(image_ids, Sequence):
            return []

        image_info = coco.imgs.get(image_id)
        if not isinstance(image_info, Mapping):
            return []

        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        if width <= 0 or height <= 0:
            return []

        annotations = []
        get_ann_ids = getattr(coco, "getAnnIds", None)
        load_anns = getattr(coco, "loadAnns", None)
        if callable(get_ann_ids) and callable(load_anns):
            annotations = list(load_anns(get_ann_ids(imgIds=[image_id])))

        crops: list[ReplayCrop] = []
        for entry in entries:
            if entry.failure_type == "detected":
                continue
            if entry.severity < self.config.fcdr.min_severity:
                continue

            source_box = _normalized_box_to_abs(entry.bbox, width=width, height=height)
            if source_box is None:
                continue

            record = mdmbpp.get_record(entry.gt_uid)
            support = entry.support if entry.support is not None else getattr(record, "support", None)
            support_box = None
            if support is not None:
                support_box = _normalized_box_to_abs(support.box, width=width, height=height)

            mode = _select_fcdr_mode(entry=entry, record=record)
            base_box = source_box
            if entry.failure_type == "nms_suppression":
                base_box = _union_with_overlapping_annotations(
                    source_box=source_box,
                    annotations=annotations,
                    width=width,
                    height=height,
                    overlap_threshold=self.config.fcdr.overlap_threshold,
                )

            crop_box = _expand_box_abs(
                base_box,
                image_width=width,
                image_height=height,
                context_scale=self.config.fcdr.crop_context_scale,
                min_context_px=self.config.fcdr.min_crop_context_px,
            )
            if crop_box is None:
                continue

            raw_weight = 1.0 + self.config.beta * float(entry.severity)
            clipped_weight = min(self.config.max_image_weight, raw_weight)
            clipped_weight = max(self.config.min_replay_weight, clipped_weight)
            sampling_weight = clipped_weight**self.config.temperature

            crops.append(
                ReplayCrop(
                    dataset_index=int(dataset_index),
                    gt_uid=entry.gt_uid,
                    image_id=str(image_id),
                    class_id=entry.class_id,
                    failure_type=str(entry.failure_type),
                    mode=mode,
                    crop_box_abs=crop_box,
                    source_bbox_abs=source_box,
                    severity=entry.severity,
                    support_box_abs=support_box,
                    sampling_weight=float(sampling_weight),
                    replay_cap=self.config.fcdr.max_crops_per_gt_per_epoch,
                )
            )

        return crops

    def _build_fcdr_summary(
        self,
        *,
        replay_crops: Sequence[ReplayCrop],
        failure_counts: Counter[str],
        mode_counts: Counter[str],
    ) -> dict[str, float | int | bool]:
        summary: dict[str, float | int | bool] = {
            "fcdr_enabled": bool(self.config.fcdr.enabled),
            "fcdr_num_crops": int(len(replay_crops)),
            "fcdr_counterfactual_ratio_requested": (
                self.config.fcdr.counterfactual_ratio if self.config.fcdr.enabled else 0.0
            ),
            "fcdr_ratio_effective": 0.0,
            "fcdr_samples": 0,
            "fcdr_unique_crops": 0,
            "fcdr_copy_paste_prob": self.config.fcdr.copy_paste_prob,
            "fcdr_pair_replay_prob": self.config.fcdr.pair_replay_prob,
        }
        for failure_type, count in failure_counts.items():
            summary[f"fcdr_failure_{failure_type}"] = int(count)
        for mode, count in mode_counts.items():
            summary[f"fcdr_policy_{mode}"] = int(count)
        return summary


def _select_fcdr_mode(*, entry: MDMBPlusEntry, record: Any) -> str:
    relapse_count = int(getattr(record, "relapse_count", 0)) if record is not None else 0
    last_detected_epoch = getattr(record, "last_detected_epoch", None) if record is not None else None
    if entry.relapse or relapse_count > 0 or last_detected_epoch is not None:
        return "support_guided_crop" if entry.support is not None else "relapse_crop"
    if entry.failure_type == "candidate_missing":
        return "zoom_context_crop"
    if entry.failure_type == "loc_near_miss":
        return "boundary_context_crop"
    if entry.failure_type == "cls_confusion":
        return "class_context_crop"
    if entry.failure_type == "score_suppression":
        return "weak_context_crop"
    if entry.failure_type == "nms_suppression":
        return "overlap_preserving_crop"
    return "context_crop"


def _normalized_box_to_abs(
    box: torch.Tensor,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    values = torch.as_tensor(box, dtype=torch.float32).detach().cpu().flatten()
    if values.numel() != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in values.tolist()]
    x1 = min(max(x1, 0.0), 1.0) * float(width)
    y1 = min(max(y1, 0.0), 1.0) * float(height)
    x2 = min(max(x2, 0.0), 1.0) * float(width)
    y2 = min(max(y2, 0.0), 1.0) * float(height)
    epsilon = 1e-4
    left = int(math.floor(min(x1, x2) + epsilon))
    top = int(math.floor(min(y1, y2) + epsilon))
    right = int(math.ceil(max(x1, x2) - epsilon))
    bottom = int(math.ceil(max(y1, y2) - epsilon))
    left = min(max(left, 0), width)
    right = min(max(right, 0), width)
    top = min(max(top, 0), height)
    bottom = min(max(bottom, 0), height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _expand_box_abs(
    box: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    context_scale: float,
    min_context_px: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    context_x = max(float(box_width) * context_scale, float(min_context_px))
    context_y = max(float(box_height) * context_scale, float(min_context_px))

    left = int(math.floor(float(x1) - context_x))
    top = int(math.floor(float(y1) - context_y))
    right = int(math.ceil(float(x2) + context_x))
    bottom = int(math.ceil(float(y2) + context_y))

    left = min(max(left, 0), image_width)
    right = min(max(right, 0), image_width)
    top = min(max(top, 0), image_height)
    bottom = min(max(bottom, 0), image_height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _union_with_overlapping_annotations(
    *,
    source_box: tuple[int, int, int, int],
    annotations: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    overlap_threshold: float,
) -> tuple[int, int, int, int]:
    boxes = [source_box]
    for annotation in annotations:
        annotation_box = _annotation_bbox_to_abs(annotation, width=width, height=height)
        if annotation_box is None:
            continue
        if _box_iou_abs(source_box, annotation_box) >= overlap_threshold:
            boxes.append(annotation_box)

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return left, top, right, bottom


def _annotation_bbox_to_abs(
    annotation: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    raw_bbox = annotation.get("bbox", None)
    if not isinstance(raw_bbox, Sequence) or len(raw_bbox) != 4:
        return None
    x, y, w, h = [float(value) for value in raw_bbox]
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + max(w, 0.0), 0.0), float(width))
    y2 = min(max(y + max(h, 0.0), 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))


def _box_iou_abs(
    left_box: tuple[int, int, int, int],
    right_box: tuple[int, int, int, int],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left_box
    right_x1, right_y1, right_x2, right_y2 = right_box
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0
    left_area = max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1)
    right_area = max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


class MixedReplayBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        shuffle: bool,
        replay_ratio: float,
        counterfactual_ratio: float,
        replacement: bool,
        seed: int,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.replay_ratio = float(replay_ratio)
        self.counterfactual_ratio = float(counterfactual_ratio)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self._replay_index = ReplayIndex.empty(enabled=False)
        self._last_summary: dict[str, float | int | bool] = dict(self._replay_index.summary)

        self.replay_count = int(math.floor(self.batch_size * self.replay_ratio))
        self.base_count = self.batch_size - self.replay_count
        self._active_replay_count = 0
        self._active_base_count = self.batch_size
        if self.batch_size < 1:
            raise ValueError("Hard Replay batch_size must be >= 1.")
        if self.base_count < 1:
            raise ValueError(
                "Hard Replay replay_ratio is too large for the configured batch size. "
                "At least one base sample must remain in each batch."
            )

    def set_replay_index(self, replay_index: ReplayIndex, *, epoch: int) -> None:
        self._replay_index = replay_index
        self.epoch = int(epoch)
        replay_is_active = self.replay_count > 0 and (
            bool(replay_index.replay_dataset_indices) or bool(replay_index.replay_crops)
        )
        self._active_replay_count = self.replay_count if replay_is_active else 0
        self._active_base_count = self.batch_size - self._active_replay_count
        self._last_summary = {
            **dict(replay_index.summary),
            "epoch": int(epoch),
            "replay_ratio_requested": self.replay_ratio if self._active_replay_count > 0 else 0.0,
        }

    def summary(self) -> dict[str, float | int | bool]:
        return dict(self._last_summary)

    def __len__(self) -> int:
        if self.dataset_size <= 0:
            return 0
        return int(math.ceil(self.dataset_size / float(self._active_base_count)))

    def __iter__(self):
        if self.dataset_size <= 0:
            self._finalize_summary(Counter(), total_replay_samples=0)
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            base_indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            base_indices = list(range(self.dataset_size))

        num_batches = len(self)
        replay_schedule = self._build_replay_schedule(
            total_replay_samples=num_batches * self._active_replay_count,
            generator=generator,
        )
        replay_cursor = 0
        replay_counts: Counter[int] = Counter()

        for batch_start in range(0, self.dataset_size, self._active_base_count):
            batch = list(base_indices[batch_start : batch_start + self._active_base_count])
            if self._active_replay_count > 0 and replay_cursor < len(replay_schedule):
                replay_slice = replay_schedule[
                    replay_cursor : replay_cursor + self._active_replay_count
                ]
                replay_cursor += len(replay_slice)
                batch.extend(replay_slice)
                replay_counts.update(replay_slice)

            if self.shuffle and len(batch) > 1:
                order = torch.randperm(len(batch), generator=generator).tolist()
                batch = [batch[index] for index in order]

            yield batch

        self._finalize_summary(replay_counts, total_replay_samples=len(replay_schedule))

    def _build_replay_schedule(
        self,
        *,
        total_replay_samples: int,
        generator: torch.Generator,
    ) -> list[int]:
        if total_replay_samples <= 0:
            return []

        crop_samples = 0
        if self.counterfactual_ratio > 0.0 and self._replay_index.replay_crops:
            crop_samples = int(round(total_replay_samples * self.counterfactual_ratio))
            crop_samples = max(1, min(total_replay_samples, crop_samples))

        image_samples = total_replay_samples - crop_samples
        schedule = self._build_image_replay_schedule(
            total_replay_samples=image_samples,
            generator=generator,
        )
        schedule.extend(
            self._build_crop_replay_schedule(
                total_replay_samples=crop_samples,
                generator=generator,
            )
        )

        if len(schedule) > 1:
            order = torch.randperm(len(schedule), generator=generator).tolist()
            schedule = [schedule[index] for index in order]
        return schedule

    def _build_image_replay_schedule(
        self,
        *,
        total_replay_samples: int,
        generator: torch.Generator,
    ) -> list[int]:
        if total_replay_samples <= 0:
            return []

        candidate_indices = list(self._replay_index.replay_dataset_indices)
        if not candidate_indices:
            return []

        weights = self._replay_index.replay_sampling_weights
        if weights.numel() != len(candidate_indices):
            return []

        if not self.replacement:
            sample_count = min(total_replay_samples, len(candidate_indices))
            chosen = torch.multinomial(weights, sample_count, replacement=False, generator=generator)
            return [candidate_indices[index] for index in chosen.tolist()]

        expanded_indices: list[int] = []
        expanded_weights: list[float] = []
        for position, dataset_index in enumerate(candidate_indices):
            cap = int(self._replay_index.replay_caps.get(dataset_index, 0))
            if cap <= 0:
                continue
            expanded_indices.extend([dataset_index] * cap)
            expanded_weights.extend([float(weights[position].item())] * cap)

        if not expanded_indices:
            return []

        sample_count = min(total_replay_samples, len(expanded_indices))
        expanded_tensor = torch.tensor(expanded_weights, dtype=torch.float32)
        chosen = torch.multinomial(
            expanded_tensor,
            sample_count,
            replacement=False,
            generator=generator,
        )
        return [expanded_indices[index] for index in chosen.tolist()]

    def _build_crop_replay_schedule(
        self,
        *,
        total_replay_samples: int,
        generator: torch.Generator,
    ) -> list[int]:
        if total_replay_samples <= 0:
            return []
        replay_crops = list(self._replay_index.replay_crops)
        if not replay_crops:
            return []

        virtual_offset = self.dataset_size
        if not self.replacement:
            sample_count = min(total_replay_samples, len(replay_crops))
            weights = torch.tensor(
                [max(float(crop.sampling_weight), 1e-6) for crop in replay_crops],
                dtype=torch.float32,
            )
            chosen = torch.multinomial(weights, sample_count, replacement=False, generator=generator)
            return [virtual_offset + index for index in chosen.tolist()]

        expanded_indices: list[int] = []
        expanded_weights: list[float] = []
        for crop_index, crop in enumerate(replay_crops):
            cap = max(1, int(crop.replay_cap))
            expanded_indices.extend([virtual_offset + crop_index] * cap)
            expanded_weights.extend([max(float(crop.sampling_weight), 1e-6)] * cap)

        if not expanded_indices:
            return []

        sample_count = min(total_replay_samples, len(expanded_indices))
        expanded_tensor = torch.tensor(expanded_weights, dtype=torch.float32)
        chosen = torch.multinomial(
            expanded_tensor,
            sample_count,
            replacement=False,
            generator=generator,
        )
        return [expanded_indices[index] for index in chosen.tolist()]

    def _finalize_summary(
        self,
        replay_counts: Counter[int],
        *,
        total_replay_samples: int,
    ) -> None:
        total_base_samples = self.dataset_size
        total_samples = total_base_samples + total_replay_samples
        image_replay_counts = Counter(
            {
                dataset_index: count
                for dataset_index, count in replay_counts.items()
                if dataset_index < self.dataset_size
            }
        )
        crop_replay_counts = Counter(
            {
                dataset_index - self.dataset_size: count
                for dataset_index, count in replay_counts.items()
                if dataset_index >= self.dataset_size
            }
        )

        replay_exposure_per_gt = 0.0
        num_active_gt = int(self._replay_index.summary.get("replay_num_active_gt", 0))
        if num_active_gt > 0:
            gt_exposures = 0
            for dataset_index, count in image_replay_counts.items():
                gt_exposures += count * int(self._replay_index.active_gt_counts.get(dataset_index, 0))
            gt_exposures += sum(crop_replay_counts.values())
            replay_exposure_per_gt = gt_exposures / float(num_active_gt)

        effective_ratio = 0.0
        if total_samples > 0:
            effective_ratio = total_replay_samples / float(total_samples)

        fcdr_samples = int(sum(crop_replay_counts.values()))
        fcdr_ratio_effective = 0.0
        if total_replay_samples > 0:
            fcdr_ratio_effective = fcdr_samples / float(total_replay_samples)

        self._last_summary = {
            **dict(self._replay_index.summary),
            "replay_ratio_requested": self.replay_ratio if self._active_replay_count > 0 else 0.0,
            "replay_ratio_effective": effective_ratio,
            "replay_exposure_per_gt": replay_exposure_per_gt,
            "replay_samples": int(total_replay_samples),
            "replay_unique_images": int(len(image_replay_counts)),
            "fcdr_samples": fcdr_samples,
            "fcdr_ratio_effective": fcdr_ratio_effective,
            "fcdr_unique_crops": int(len(crop_replay_counts)),
        }


class HardReplayController:
    def __init__(
        self,
        *,
        config: HardReplayConfig,
        dataset: Any,
        batch_sampler: MixedReplayBatchSampler,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.planner = HardReplayPlanner(config)
        self._latest_index = ReplayIndex.empty(enabled=config.enabled)
        self._replay_dataset: Any | None = None

    def attach_replay_dataset(self, dataset: Any) -> None:
        self._replay_dataset = dataset

    def start_epoch(self, *, mdmbpp: MDMBPlus | None, epoch: int) -> None:
        replay_index = self.planner.build_epoch_index(
            mdmbpp=mdmbpp,
            dataset=self.dataset,
            epoch=epoch,
        )
        self._latest_index = replay_index
        self.batch_sampler.set_replay_index(replay_index, epoch=epoch)
        if self._replay_dataset is not None:
            set_replay_index = getattr(self._replay_dataset, "set_replay_index", None)
            if callable(set_replay_index):
                set_replay_index(replay_index)

    def summary(self) -> dict[str, float | int | bool]:
        return self.batch_sampler.summary()


def load_hard_replay_config(path: str | Path, *, arch: str | None = None) -> HardReplayConfig:
    raw = load_yaml_file(path)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Hard Replay YAML must contain a mapping at the top level: {path}")
    return HardReplayConfig.from_mapping(raw, arch=arch)


def build_hard_replay_controller_from_yaml(
    path: str | Path,
    *,
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
    arch: str | None = None,
) -> HardReplayController | None:
    config = load_hard_replay_config(path, arch=arch)
    if not config.enabled:
        return None
    batch_sampler = MixedReplayBatchSampler(
        dataset_size=len(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        replay_ratio=config.replay_ratio,
        counterfactual_ratio=(
            config.fcdr.counterfactual_ratio if config.fcdr.enabled else 0.0
        ),
        replacement=config.replacement,
        seed=seed,
    )
    return HardReplayController(
        config=config,
        dataset=dataset,
        batch_sampler=batch_sampler,
    )
