from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Sampler

from modules.nn import BACKGROUND, BOTH, CLASSIFICATION, DUPLICATE, FAILURE_TYPES, LOCALIZATION, MISSED, normalize_arch

_TYPE_ALIASES = {
    "loc": LOCALIZATION,
    "localization": LOCALIZATION,
    "cls": CLASSIFICATION,
    "classification": CLASSIFICATION,
    "both": BOTH,
    "miss": MISSED,
    "missed": MISSED,
    "duplicate": DUPLICATE,
    "dupe": DUPLICATE,
    "background": BACKGROUND,
    "bkg": BACKGROUND,
}

DEFAULT_TYPE_RATIOS = {
    LOCALIZATION: 0.25,
    CLASSIFICATION: 0.15,
    BOTH: 0.25,
    MISSED: 0.25,
    DUPLICATE: 0.05,
    BACKGROUND: 0.05,
}


@dataclass(frozen=True, slots=True)
class TARConfig:
    enabled: bool = False
    start_epoch: int = 2
    warmup_epochs: int = 0
    replay_ratio: float = 0.25
    type_ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TYPE_RATIOS))
    max_replays_per_batch: int = 0
    replacement: bool = True
    replay_recency_window: int = 1
    max_replays_per_record_per_epoch: int = 4
    min_consecutive_count: int = 1
    min_total_failed: int = 1
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "TARConfig":
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
            start_epoch=int(merged.get("start_epoch", 2)),
            warmup_epochs=int(merged.get("warmup_epochs", 0)),
            replay_ratio=float(merged.get("replay_ratio", 0.25)),
            type_ratios=_normalize_type_ratios(merged.get("type_ratios")),
            max_replays_per_batch=int(merged.get("max_replays_per_batch", 0)),
            replacement=bool(merged.get("replacement", True)),
            replay_recency_window=int(merged.get("replay_recency_window", 1)),
            max_replays_per_record_per_epoch=int(merged.get("max_replays_per_record_per_epoch", 4)),
            min_consecutive_count=int(merged.get("min_consecutive_count", 1)),
            min_total_failed=int(merged.get("min_total_failed", 1)),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.start_epoch) < 1:
            raise ValueError("TAR start_epoch must be >= 1.")
        if int(self.warmup_epochs) < 0:
            raise ValueError("TAR warmup_epochs must be >= 0.")
        if not 0.0 <= float(self.replay_ratio) < 1.0:
            raise ValueError("TAR replay_ratio must satisfy 0 <= value < 1.")
        if sum(float(value) for value in self.type_ratios.values()) <= 0.0:
            raise ValueError("TAR type_ratios must contain at least one positive value.")
        if int(self.max_replays_per_batch) < 0:
            raise ValueError("TAR max_replays_per_batch must be >= 0.")
        if int(self.replay_recency_window) < 0:
            raise ValueError("TAR replay_recency_window must be >= 0.")
        if int(self.max_replays_per_record_per_epoch) < 1:
            raise ValueError("TAR max_replays_per_record_per_epoch must be >= 1.")
        if int(self.min_consecutive_count) < 1:
            raise ValueError("TAR min_consecutive_count must be >= 1.")
        if int(self.min_total_failed) < 1:
            raise ValueError("TAR min_total_failed must be >= 1.")

    def scheduled_ratio(self, epoch: int) -> float:
        if not self.enabled:
            return 0.0
        if int(epoch) < int(self.start_epoch):
            return 0.0
        if int(self.warmup_epochs) <= 0:
            return float(self.replay_ratio)
        active_epoch = int(epoch) - int(self.start_epoch) + 1
        progress = min(1.0, max(0.0, active_epoch / float(self.warmup_epochs)))
        return float(self.replay_ratio) * progress

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "start_epoch": self.start_epoch,
            "warmup_epochs": self.warmup_epochs,
            "replay_ratio": self.replay_ratio,
            "type_ratios": dict(self.type_ratios),
            "max_replays_per_batch": self.max_replays_per_batch,
            "replacement": self.replacement,
            "replay_recency_window": self.replay_recency_window,
            "max_replays_per_record_per_epoch": self.max_replays_per_record_per_epoch,
            "min_consecutive_count": self.min_consecutive_count,
            "min_total_failed": self.min_total_failed,
            "arch": self.arch,
        }


@dataclass(frozen=True, slots=True)
class TARSampleRef:
    dataset_index: int
    image_id: str
    failure_type: str
    source: str
    gt_uid: str | None = None
    gt_id: str | None = None
    class_id: int | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class TARCandidate:
    dataset_index: int
    image_id: str
    failure_type: str
    source: str
    weight: float
    cap: int
    gt_uid: str | None = None
    gt_id: str | None = None
    class_id: int | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None

    def to_sample(self) -> TARSampleRef:
        return TARSampleRef(
            dataset_index=int(self.dataset_index),
            image_id=str(self.image_id),
            failure_type=str(self.failure_type),
            source=str(self.source),
            gt_uid=self.gt_uid,
            gt_id=self.gt_id,
            class_id=self.class_id,
            bbox_xyxy=self.bbox_xyxy,
        )


@dataclass(slots=True)
class TARIndex:
    enabled: bool
    epoch: int = 0
    requested_ratio: float = 0.0
    candidates_by_type: dict[str, list[TARCandidate]] = field(default_factory=lambda: {
        failure_type: [] for failure_type in FAILURE_TYPES
    })
    summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        *,
        enabled: bool,
        epoch: int = 0,
        requested_ratio: float = 0.0,
        reason: str = "inactive",
    ) -> "TARIndex":
        return cls(
            enabled=bool(enabled),
            epoch=int(epoch),
            requested_ratio=float(requested_ratio),
            summary=_empty_summary(enabled=enabled, epoch=epoch, requested_ratio=requested_ratio, reason=reason),
        )


class TARPlanner:
    def __init__(self, config: TARConfig) -> None:
        self.config = config

    def build_epoch_index(
        self,
        *,
        ftmb: Any,
        dataset: Any,
        epoch: int,
    ) -> TARIndex:
        requested_ratio = self.config.scheduled_ratio(epoch)
        if not self.config.enabled:
            return TARIndex.empty(enabled=False, epoch=epoch, requested_ratio=0.0, reason="disabled")
        if requested_ratio <= 0.0:
            return TARIndex.empty(enabled=True, epoch=epoch, requested_ratio=0.0, reason="warmup")
        if ftmb is None:
            return TARIndex.empty(enabled=True, epoch=epoch, requested_ratio=requested_ratio, reason="missing_ftmb")

        image_ids = getattr(dataset, "image_ids", None)
        if not isinstance(image_ids, Sequence):
            raise TypeError("TAR requires a dataset exposing a sequence-like image_ids field.")
        image_to_index = {_normalize_image_id(image_id): index for index, image_id in enumerate(image_ids)}
        candidates_by_type: dict[str, list[TARCandidate]] = {failure_type: [] for failure_type in FAILURE_TYPES}

        for record in _ftmb_records(ftmb):
            failure_type = _normalize_failure_type(getattr(record, "failure_type", None))
            if failure_type is None:
                continue
            if not self._record_is_eligible(record, epoch=epoch):
                continue
            image_id = _normalize_image_id(getattr(record, "image_id", ""))
            dataset_index = image_to_index.get(image_id)
            if dataset_index is None:
                continue
            candidates_by_type[failure_type].append(
                TARCandidate(
                    dataset_index=int(dataset_index),
                    image_id=image_id,
                    failure_type=failure_type,
                    source="gt",
                    weight=_record_priority(record),
                    cap=int(self.config.max_replays_per_record_per_epoch),
                    gt_uid=str(getattr(record, "record_key", "")),
                    gt_id=None if getattr(record, "gt_id", None) is None else str(getattr(record, "gt_id")),
                    class_id=int(getattr(record, "gt_class", 0)),
                    bbox_xyxy=_record_bbox(record),
                )
            )

        for event in _ftmb_prediction_events(ftmb):
            failure_type = _normalize_failure_type(event.get("failure_type"))
            if failure_type not in {DUPLICATE, BACKGROUND}:
                continue
            if not self._event_is_eligible(event, epoch=epoch):
                continue
            image_id = _normalize_image_id(event.get("image_id", ""))
            dataset_index = image_to_index.get(image_id)
            if dataset_index is None:
                continue
            candidates_by_type[failure_type].append(
                TARCandidate(
                    dataset_index=int(dataset_index),
                    image_id=image_id,
                    failure_type=failure_type,
                    source="prediction",
                    weight=_event_priority(event),
                    cap=1,
                    class_id=int(event.get("pred_class", 0)),
                    bbox_xyxy=_event_bbox(event),
                )
            )

        if not any(candidates_by_type.values()):
            return TARIndex.empty(enabled=True, epoch=epoch, requested_ratio=requested_ratio, reason="no_failure_candidates")

        summary = _empty_summary(enabled=True, epoch=epoch, requested_ratio=requested_ratio, reason="active")
        for failure_type in FAILURE_TYPES:
            candidates = candidates_by_type[failure_type]
            summary[f"{failure_type}_candidate_count"] = len(candidates)
            summary[f"{failure_type}_unique_images"] = len({candidate.image_id for candidate in candidates})
        summary["replay_num_candidates"] = sum(len(candidates) for candidates in candidates_by_type.values())
        summary["replay_num_images"] = len(
            {
                candidate.image_id
                for candidates in candidates_by_type.values()
                for candidate in candidates
            }
        )
        return TARIndex(
            enabled=True,
            epoch=int(epoch),
            requested_ratio=float(requested_ratio),
            candidates_by_type=candidates_by_type,
            summary=summary,
        )

    def _record_is_eligible(self, record: Any, *, epoch: int) -> bool:
        if int(getattr(record, "total_failed", 0)) < int(self.config.min_total_failed):
            return False
        if int(getattr(record, "consecutive_count", 0)) < int(self.config.min_consecutive_count):
            return False
        return self._recent_enough(getattr(record, "last_epoch", 0), epoch=epoch)

    def _event_is_eligible(self, event: Mapping[str, Any], *, epoch: int) -> bool:
        return self._recent_enough(event.get("epoch", 0), epoch=epoch)

    def _recent_enough(self, last_epoch: Any, *, epoch: int) -> bool:
        if int(self.config.replay_recency_window) <= 0:
            return True
        try:
            value = int(last_epoch)
        except (TypeError, ValueError):
            return False
        return int(epoch) - value <= int(self.config.replay_recency_window)


class TARBatchSampler(Sampler[list[int | TARSampleRef]]):
    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        shuffle: bool,
        type_ratios: Mapping[str, float],
        max_replays_per_batch: int,
        replacement: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.type_ratios = dict(type_ratios)
        self.max_replays_per_batch = int(max_replays_per_batch)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.base_only = False
        self._tar_index = TARIndex.empty(enabled=False)
        self._last_summary = dict(self._tar_index.summary)

        if self.dataset_size < 0:
            raise ValueError("TAR dataset_size must be >= 0.")
        if self.batch_size < 1:
            raise ValueError("TAR batch_size must be >= 1.")
        if self.max_replays_per_batch < 0:
            raise ValueError("TAR max_replays_per_batch must be >= 0.")
        if self.world_size < 1:
            raise ValueError("TAR world_size must be >= 1.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("TAR rank must satisfy 0 <= rank < world_size.")

    @property
    def last_summary(self) -> dict[str, Any]:
        return dict(self._last_summary)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_base_only(self, enabled: bool) -> None:
        self.base_only = bool(enabled)

    def set_tar_index(self, tar_index: TARIndex, *, epoch: int) -> None:
        self._tar_index = tar_index
        self.set_epoch(epoch)
        self._last_summary = self._summary_with_slot_plan(tar_index)

    def __len__(self) -> int:
        base_count = max(1, self.batch_size - self._active_replay_slots())
        return int(math.ceil(self._rank_base_sample_count() / float(base_count)))

    def __iter__(self):
        base_indices = self._build_base_indices()
        replay_slots = self._active_replay_slots()
        if replay_slots <= 0:
            for batch in _chunks(base_indices, self.batch_size):
                yield batch
            if not self.base_only:
                self._finalize_summary(total_base_samples=len(base_indices), replay_samples=[])
            return

        base_count = self.batch_size - replay_slots
        num_batches = int(math.ceil(len(base_indices) / float(base_count)))
        slot_plan = self._type_slot_plan(replay_slots)
        schedules = {
            failure_type: self._build_replay_schedule(
                self._tar_index.candidates_by_type.get(failure_type, []),
                total_samples=num_batches * slots,
                seed_offset=101 + position * 997,
            )
            for position, (failure_type, slots) in enumerate(slot_plan.items())
            if slots > 0
        }
        cursors = {failure_type: 0 for failure_type in slot_plan}
        fallback_schedule = self._build_replay_schedule(
            [
                candidate
                for candidates in self._tar_index.candidates_by_type.values()
                for candidate in candidates
            ],
            total_samples=num_batches * replay_slots,
            seed_offset=9001,
        )
        fallback_cursor = 0
        replay_samples: list[TARSampleRef] = []
        for batch_number, base_batch in enumerate(_chunks(base_indices, base_count), start=1):
            replay_slice: list[TARSampleRef] = []
            for failure_type, slots in slot_plan.items():
                schedule = schedules.get(failure_type, [])
                cursor = cursors.get(failure_type, 0)
                selected = schedule[cursor : cursor + slots]
                cursors[failure_type] = cursor + len(selected)
                replay_slice.extend(selected)

            while len(replay_slice) < replay_slots and fallback_cursor < len(fallback_schedule):
                replay_slice.append(fallback_schedule[fallback_cursor])
                fallback_cursor += 1

            batch: list[int | TARSampleRef] = [*base_batch, *replay_slice]
            random.Random(self.seed + self.epoch * 1009 + batch_number).shuffle(batch)
            replay_samples.extend(replay_slice)
            yield batch

        self._finalize_summary(total_base_samples=len(base_indices), replay_samples=replay_samples)

    def _active_replay_slots(self) -> int:
        if self.base_only:
            return 0
        if not self._tar_index.enabled:
            return 0
        if not any(self._tar_index.candidates_by_type.values()):
            return 0
        requested_ratio = float(self._tar_index.requested_ratio)
        if requested_ratio <= 0.0:
            return 0
        replay_slots = int(math.floor(self.batch_size * requested_ratio))
        if self.max_replays_per_batch > 0:
            replay_slots = min(replay_slots, int(self.max_replays_per_batch))
        replay_slots = max(0, replay_slots)
        if replay_slots >= self.batch_size:
            raise ValueError(
                "TAR replay_ratio leaves no base samples in a batch. "
                "Reduce replay_ratio or max_replays_per_batch."
            )
        return replay_slots

    def _type_slot_plan(self, replay_slots: int) -> dict[str, int]:
        available = [
            failure_type
            for failure_type in FAILURE_TYPES
            if self._tar_index.candidates_by_type.get(failure_type)
        ]
        if replay_slots <= 0 or not available:
            return {failure_type: 0 for failure_type in FAILURE_TYPES}
        weights = {
            failure_type: max(0.0, float(self.type_ratios.get(failure_type, 0.0)))
            for failure_type in available
        }
        if sum(weights.values()) <= 0.0:
            weights = {failure_type: 1.0 for failure_type in available}
        weight_sum = sum(weights.values())
        raw_slots = {failure_type: replay_slots * weights[failure_type] / weight_sum for failure_type in available}
        plan = {failure_type: int(math.floor(raw_slots[failure_type])) for failure_type in available}
        remaining = replay_slots - sum(plan.values())
        remainders = sorted(
            available,
            key=lambda failure_type: (raw_slots[failure_type] - plan[failure_type], weights[failure_type]),
            reverse=True,
        )
        for failure_type in remainders[:remaining]:
            plan[failure_type] += 1
        return {failure_type: int(plan.get(failure_type, 0)) for failure_type in FAILURE_TYPES}

    def _build_base_indices(self) -> list[int]:
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        if self.world_size <= 1:
            return indices
        num_samples = int(math.ceil(self.dataset_size / float(self.world_size)))
        total_size = num_samples * self.world_size
        if len(indices) < total_size:
            needed = total_size - len(indices)
            repeats = int(math.ceil(needed / float(max(len(indices), 1))))
            indices.extend((indices * repeats)[:needed])
        indices = indices[:total_size]
        return indices[self.rank : total_size : self.world_size]

    def _rank_base_sample_count(self) -> int:
        if self.world_size <= 1:
            return self.dataset_size
        return int(math.ceil(self.dataset_size / float(self.world_size)))

    def _build_replay_schedule(
        self,
        candidates: Sequence[TARCandidate],
        *,
        total_samples: int,
        seed_offset: int,
    ) -> list[TARSampleRef]:
        if total_samples <= 0 or not candidates:
            return []

        expanded_positions: list[int] = []
        expanded_weights: list[float] = []
        for position, candidate in enumerate(candidates):
            cap = int(candidate.cap)
            if not self.replacement:
                cap = min(cap, 1)
            for _ in range(max(cap, 0)):
                expanded_positions.append(position)
                expanded_weights.append(max(0.0, float(candidate.weight)))

        if not expanded_positions:
            return []
        if sum(expanded_weights) <= 0.0:
            expanded_weights = [1.0 for _ in expanded_weights]

        sample_count = min(int(total_samples), len(expanded_positions))
        generator = torch.Generator()
        generator.manual_seed(int(self.seed + self.epoch * 1000003 + self.rank * 9176 + seed_offset))
        selected = torch.multinomial(
            torch.tensor(expanded_weights, dtype=torch.float32),
            num_samples=sample_count,
            replacement=False,
            generator=generator,
        ).tolist()

        return [
            candidates[expanded_positions[int(expanded_index)]].to_sample()
            for expanded_index in selected
        ]

    def _summary_with_slot_plan(self, tar_index: TARIndex) -> dict[str, Any]:
        summary = dict(tar_index.summary)
        replay_slots = self._active_replay_slots()
        slot_plan = self._type_slot_plan(replay_slots)
        summary.update(
            {
                "active": False,
                "replay_slots_per_batch": int(replay_slots),
                "base_slots_per_batch": int(self.batch_size - replay_slots),
                **{f"{failure_type}_slots_per_batch": int(slot_plan.get(failure_type, 0)) for failure_type in FAILURE_TYPES},
            }
        )
        return summary

    def _finalize_summary(
        self,
        *,
        total_base_samples: int,
        replay_samples: Sequence[TARSampleRef],
    ) -> None:
        type_counts = Counter(sample.failure_type for sample in replay_samples)
        unique_images = {sample.image_id for sample in replay_samples}
        total_replay_samples = len(replay_samples)
        total_samples = int(total_base_samples) + int(total_replay_samples)
        effective_ratio = (
            total_replay_samples / float(total_samples)
            if total_samples > 0
            else 0.0
        )
        replay_slots = self._active_replay_slots()
        slot_plan = self._type_slot_plan(replay_slots)
        self._last_summary = {
            **dict(self._tar_index.summary),
            "active": bool(total_replay_samples > 0),
            "replay_ratio_effective": effective_ratio,
            "replay_samples": int(total_replay_samples),
            "replay_unique_images": int(len(unique_images)),
            "replay_slots_per_batch": int(replay_slots),
            "base_slots_per_batch": int(self.batch_size - replay_slots),
            **{f"{failure_type}_samples": int(type_counts.get(failure_type, 0)) for failure_type in FAILURE_TYPES},
            **{f"{failure_type}_slots_per_batch": int(slot_plan.get(failure_type, 0)) for failure_type in FAILURE_TYPES},
        }


class TARController:
    def __init__(
        self,
        *,
        config: TARConfig,
        dataset: Any,
        batch_sampler: TARBatchSampler,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.planner = TARPlanner(config)
        self._latest_index = TARIndex.empty(enabled=config.enabled)

    def refresh(self, *, ftmb: Any, epoch: int) -> None:
        tar_index = self.planner.build_epoch_index(
            ftmb=ftmb,
            dataset=self.dataset,
            epoch=int(epoch),
        )
        self._latest_index = tar_index
        self.batch_sampler.set_tar_index(tar_index, epoch=int(epoch))

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(int(epoch))

    def set_base_only(self, enabled: bool) -> None:
        self.batch_sampler.set_base_only(bool(enabled))

    def summary(self) -> dict[str, Any]:
        return self.batch_sampler.last_summary


def load_tar_config(path: str | Path, *, arch: str | None = None) -> TARConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"TAR YAML must contain a mapping at the top level: {config_path}")
    return TARConfig.from_mapping(raw, arch=arch)


def build_tar_controller_from_yaml(
    path: str | Path,
    *,
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    arch: str | None = None,
) -> TARController | None:
    config = load_tar_config(path, arch=arch)
    if not config.enabled:
        return None
    batch_sampler = TARBatchSampler(
        dataset_size=len(dataset),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        type_ratios=config.type_ratios,
        max_replays_per_batch=int(config.max_replays_per_batch),
        replacement=bool(config.replacement),
        seed=int(seed),
        rank=int(rank),
        world_size=int(world_size),
    )
    return TARController(config=config, dataset=dataset, batch_sampler=batch_sampler)


def _normalize_type_ratios(raw: Any) -> dict[str, float]:
    ratios = {failure_type: 0.0 for failure_type in FAILURE_TYPES}
    if raw is None:
        ratios.update(DEFAULT_TYPE_RATIOS)
        return ratios
    if not isinstance(raw, Mapping):
        raise TypeError("TAR type_ratios must be a mapping.")
    for key, value in raw.items():
        failure_type = _normalize_failure_type(key)
        if failure_type is None:
            raise KeyError(f"Unsupported TAR failure type ratio key: {key!r}")
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError("TAR type_ratios values must be >= 0.")
        ratios[failure_type] = numeric
    return ratios


def _empty_summary(
    *,
    enabled: bool,
    epoch: int,
    requested_ratio: float,
    reason: str,
) -> dict[str, Any]:
    summary = {
        "enabled": bool(enabled),
        "active": False,
        "epoch": int(epoch),
        "reason": str(reason),
        "replay_ratio_requested": float(requested_ratio),
        "replay_ratio_effective": 0.0,
        "replay_num_candidates": 0,
        "replay_num_images": 0,
        "replay_samples": 0,
        "replay_unique_images": 0,
        "replay_slots_per_batch": 0,
        "base_slots_per_batch": 0,
    }
    for failure_type in FAILURE_TYPES:
        summary[f"{failure_type}_candidate_count"] = 0
        summary[f"{failure_type}_unique_images"] = 0
        summary[f"{failure_type}_samples"] = 0
        summary[f"{failure_type}_slots_per_batch"] = 0
    return summary


def _ftmb_records(ftmb: Any) -> list[Any]:
    get_records = getattr(ftmb, "get_records", None)
    if callable(get_records):
        return list(get_records())
    return []


def _ftmb_prediction_events(ftmb: Any) -> list[Mapping[str, Any]]:
    get_events = getattr(ftmb, "get_prediction_events", None)
    if not callable(get_events):
        return []
    return [event for event in get_events() if isinstance(event, Mapping)]


def _record_priority(record: Any) -> float:
    consecutive_count = max(1, int(getattr(record, "consecutive_count", 0)))
    total_failed = max(0, int(getattr(record, "total_failed", 0)))
    assigned_iou = getattr(record, "assigned_pred_iou", None)
    severity = 1.0
    if assigned_iou is not None:
        severity += max(0.0, 1.0 - float(assigned_iou))
    return float(consecutive_count) + 0.25 * float(total_failed) + severity


def _event_priority(event: Mapping[str, Any]) -> float:
    try:
        return 1.0 + max(0.0, float(event.get("pred_score", 0.0)))
    except (TypeError, ValueError):
        return 1.0


def _record_bbox(record: Any) -> tuple[float, float, float, float] | None:
    raw = getattr(record, "bbox_xyxy", None)
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        values = raw.detach().cpu().flatten().tolist()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        return None
    if len(values) < 4:
        return None
    return tuple(float(value) for value in values[:4])  # type: ignore[return-value]


def _event_bbox(event: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = event.get("pred_bbox_xyxy")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 4:
        return None
    return tuple(float(value) for value in list(raw)[:4])  # type: ignore[return-value]


def _normalize_failure_type(value: Any) -> str | None:
    if value is None:
        return None
    return _TYPE_ALIASES.get(str(value).lower())


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


def _chunks(values: Sequence[int], size: int):
    chunk_size = max(1, int(size))
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])
