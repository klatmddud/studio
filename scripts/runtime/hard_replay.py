from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Sampler

from modules.nn import normalize_arch


@dataclass(frozen=True, slots=True)
class HardReplayConfig:
    enabled: bool = False
    start_epoch: int = 2
    warmup_epochs: int = 0
    replay_ratio: float = 0.25
    max_replays_per_batch: int = 0
    beta: float = 1.0
    temperature: float = 1.0
    max_image_weight: float = 5.0
    min_image_weight: float = 1.0
    replacement: bool = True
    min_miss_count: int = 1
    min_observations: int = 1
    replay_recency_window: int = 1
    current_epoch_only: bool = False
    max_replays_per_gt_per_epoch: int = 4
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "HardReplayConfig":
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
            max_replays_per_batch=int(merged.get("max_replays_per_batch", 0)),
            beta=float(merged.get("beta", 1.0)),
            temperature=float(merged.get("temperature", 1.0)),
            max_image_weight=float(merged.get("max_image_weight", 5.0)),
            min_image_weight=float(merged.get("min_image_weight", 1.0)),
            replacement=bool(merged.get("replacement", True)),
            min_miss_count=int(merged.get("min_miss_count", 1)),
            min_observations=int(merged.get("min_observations", 1)),
            replay_recency_window=int(merged.get("replay_recency_window", 1)),
            current_epoch_only=bool(merged.get("current_epoch_only", False)),
            max_replays_per_gt_per_epoch=int(merged.get("max_replays_per_gt_per_epoch", 4)),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.start_epoch) < 1:
            raise ValueError("Hard Replay start_epoch must be >= 1.")
        if int(self.warmup_epochs) < 0:
            raise ValueError("Hard Replay warmup_epochs must be >= 0.")
        if not 0.0 <= float(self.replay_ratio) < 1.0:
            raise ValueError("Hard Replay replay_ratio must satisfy 0 <= value < 1.")
        if int(self.max_replays_per_batch) < 0:
            raise ValueError("Hard Replay max_replays_per_batch must be >= 0.")
        if float(self.beta) < 0.0:
            raise ValueError("Hard Replay beta must be >= 0.")
        if float(self.temperature) <= 0.0:
            raise ValueError("Hard Replay temperature must be > 0.")
        if float(self.max_image_weight) <= 0.0:
            raise ValueError("Hard Replay max_image_weight must be > 0.")
        if float(self.min_image_weight) <= 0.0:
            raise ValueError("Hard Replay min_image_weight must be > 0.")
        if float(self.min_image_weight) > float(self.max_image_weight):
            raise ValueError("Hard Replay min_image_weight must be <= max_image_weight.")
        if int(self.min_miss_count) < 1:
            raise ValueError("Hard Replay min_miss_count must be >= 1.")
        if int(self.min_observations) < 1:
            raise ValueError("Hard Replay min_observations must be >= 1.")
        if int(self.replay_recency_window) < 0:
            raise ValueError("Hard Replay replay_recency_window must be >= 0.")
        if int(self.max_replays_per_gt_per_epoch) < 1:
            raise ValueError("Hard Replay max_replays_per_gt_per_epoch must be >= 1.")

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
            "max_replays_per_batch": self.max_replays_per_batch,
            "beta": self.beta,
            "temperature": self.temperature,
            "max_image_weight": self.max_image_weight,
            "min_image_weight": self.min_image_weight,
            "replacement": self.replacement,
            "min_miss_count": self.min_miss_count,
            "min_observations": self.min_observations,
            "replay_recency_window": self.replay_recency_window,
            "current_epoch_only": self.current_epoch_only,
            "max_replays_per_gt_per_epoch": self.max_replays_per_gt_per_epoch,
            "arch": self.arch,
        }


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    dataset_index: int
    weight: float
    cap: int
    active_gt_count: int
    priority: float


@dataclass(slots=True)
class ReplayIndex:
    enabled: bool
    epoch: int = 0
    requested_ratio: float = 0.0
    image_weights: dict[str, float] = field(default_factory=dict)
    replay_gt_ids: set[str] = field(default_factory=set)
    active_gt_counts: dict[int, int] = field(default_factory=dict)
    image_candidates: list[ReplayCandidate] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        *,
        enabled: bool,
        epoch: int = 0,
        requested_ratio: float = 0.0,
        reason: str = "inactive",
    ) -> "ReplayIndex":
        return cls(
            enabled=bool(enabled),
            epoch=int(epoch),
            requested_ratio=float(requested_ratio),
            summary=_empty_summary(
                enabled=enabled,
                epoch=epoch,
                requested_ratio=requested_ratio,
                reason=reason,
            ),
        )


class HardReplayPlanner:
    def __init__(self, config: HardReplayConfig) -> None:
        self.config = config

    def build_epoch_index(
        self,
        *,
        missbank: Any,
        dataset: Any,
        epoch: int,
    ) -> ReplayIndex:
        requested_ratio = self.config.scheduled_ratio(epoch)
        if not self.config.enabled:
            return ReplayIndex.empty(
                enabled=False,
                epoch=epoch,
                requested_ratio=0.0,
                reason="disabled",
            )
        if requested_ratio <= 0.0:
            return ReplayIndex.empty(
                enabled=True,
                epoch=epoch,
                requested_ratio=0.0,
                reason="warmup",
            )
        if missbank is None:
            return ReplayIndex.empty(
                enabled=True,
                epoch=epoch,
                requested_ratio=requested_ratio,
                reason="missing_missbank",
            )

        image_ids = getattr(dataset, "image_ids", None)
        if not isinstance(image_ids, Sequence):
            raise TypeError("Hard Replay requires a dataset exposing a sequence-like image_ids field.")

        image_weights: dict[str, float] = {}
        replay_gt_ids: set[str] = set()
        active_gt_counts: dict[int, int] = {}
        image_candidates: list[ReplayCandidate] = []
        priority_sum = 0.0

        score_threshold, iou_threshold = _missbank_thresholds(missbank)
        for dataset_index, image_id in enumerate(image_ids):
            records = self._select_records(
                missbank=missbank,
                image_id=image_id,
                epoch=epoch,
            )
            if not records:
                continue

            image_priority = sum(
                _record_priority(
                    record,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )
                for record in records
            )
            if image_priority <= 0.0:
                continue
            for record in records:
                replay_gt_ids.add(_record_uid(record))

            image_key = _normalize_image_id(image_id)
            clipped_weight = self._clipped_weight(image_priority)
            sampling_weight = clipped_weight ** float(self.config.temperature)
            cap = int(self.config.max_replays_per_gt_per_epoch) * len(records)

            image_weights[image_key] = float(clipped_weight)
            active_gt_counts[int(dataset_index)] = len(records)
            image_candidates.append(
                ReplayCandidate(
                    dataset_index=int(dataset_index),
                    weight=float(sampling_weight),
                    cap=int(cap),
                    active_gt_count=len(records),
                    priority=float(image_priority),
                )
            )
            priority_sum += image_priority

        if not image_candidates:
            return ReplayIndex.empty(
                enabled=True,
                epoch=epoch,
                requested_ratio=requested_ratio,
                reason="no_missed_gt",
            )

        mean_image_weight = sum(image_weights.values()) / float(max(len(image_weights), 1))
        mean_gt_priority = priority_sum / float(max(len(replay_gt_ids), 1))
        summary = _empty_summary(
            enabled=True,
            epoch=epoch,
            requested_ratio=requested_ratio,
            reason="active",
        )
        summary.update(
            {
                "active": False,
                "replay_num_images": len(image_weights),
                "replay_num_active_gt": len(replay_gt_ids),
                "replay_mean_image_weight": mean_image_weight,
                "replay_mean_gt_priority": mean_gt_priority,
            }
        )
        return ReplayIndex(
            enabled=True,
            epoch=int(epoch),
            requested_ratio=float(requested_ratio),
            image_weights=image_weights,
            replay_gt_ids=replay_gt_ids,
            active_gt_counts=active_gt_counts,
            image_candidates=image_candidates,
            summary=summary,
        )

    def _select_records(self, *, missbank: Any, image_id: Any, epoch: int) -> list[Any]:
        get_records = getattr(missbank, "get_records", None)
        if not callable(get_records):
            return []

        selected: list[Any] = []
        for record in get_records(image_id):
            if not self._record_is_eligible(record, epoch=epoch):
                continue
            selected.append(record)
        selected.sort(
            key=lambda record: _record_priority(
                record,
                score_threshold=_missbank_thresholds(missbank)[0],
                iou_threshold=_missbank_thresholds(missbank)[1],
            ),
            reverse=True,
        )
        return selected

    def _record_is_eligible(self, record: Any, *, epoch: int) -> bool:
        if not bool(getattr(record, "is_missed", False)):
            return False
        if int(getattr(record, "miss_count", 0)) < int(self.config.min_miss_count):
            return False
        if int(getattr(record, "total_seen", 0)) < int(self.config.min_observations):
            return False
        last_epoch = int(getattr(record, "last_epoch", 0))
        if bool(self.config.current_epoch_only):
            return last_epoch == int(epoch)
        if int(self.config.replay_recency_window) <= 0:
            return True
        return int(epoch) - last_epoch <= int(self.config.replay_recency_window)

    def _clipped_weight(self, priority: float) -> float:
        raw_weight = 1.0 + float(self.config.beta) * float(priority)
        clipped = min(float(self.config.max_image_weight), raw_weight)
        return max(float(self.config.min_image_weight), clipped)


class MixedReplayBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        shuffle: bool,
        max_replays_per_batch: int,
        replacement: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.max_replays_per_batch = int(max_replays_per_batch)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.base_only = False
        self._replay_index = ReplayIndex.empty(enabled=False)
        self._last_summary: dict[str, Any] = dict(self._replay_index.summary)

        if self.dataset_size < 0:
            raise ValueError("Hard Replay dataset_size must be >= 0.")
        if self.batch_size < 1:
            raise ValueError("Hard Replay batch_size must be >= 1.")
        if self.max_replays_per_batch < 0:
            raise ValueError("Hard Replay max_replays_per_batch must be >= 0.")
        if self.world_size < 1:
            raise ValueError("Hard Replay world_size must be >= 1.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Hard Replay rank must satisfy 0 <= rank < world_size.")

    @property
    def last_summary(self) -> dict[str, Any]:
        return dict(self._last_summary)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_base_only(self, enabled: bool) -> None:
        self.base_only = bool(enabled)

    def set_replay_index(self, replay_index: ReplayIndex, *, epoch: int) -> None:
        self._replay_index = replay_index
        self.set_epoch(epoch)
        self._last_summary = self._summary_with_slot_plan(replay_index)

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
                self._finalize_summary(
                    total_base_samples=len(base_indices),
                    replay_samples=[],
                )
            return

        base_count = self.batch_size - replay_slots
        num_batches = int(math.ceil(len(base_indices) / float(base_count)))
        replay_schedule = self._build_replay_schedule(
            self._replay_index.image_candidates,
            total_samples=num_batches * replay_slots,
            seed_offset=17,
        )

        replay_cursor = 0
        replay_samples: list[int] = []
        for batch_number, base_batch in enumerate(_chunks(base_indices, base_count), start=1):
            replay_slice = replay_schedule[replay_cursor : replay_cursor + replay_slots]
            replay_cursor += len(replay_slice)

            batch = [*base_batch, *replay_slice]
            random.Random(self.seed + self.epoch * 1009 + batch_number).shuffle(batch)
            replay_samples.extend(replay_slice)
            yield batch

        self._finalize_summary(
            total_base_samples=len(base_indices),
            replay_samples=replay_samples,
        )

    def _active_replay_slots(self) -> int:
        if self.base_only:
            return 0
        if not self._replay_index.enabled:
            return 0
        if not self._has_candidates():
            return 0
        requested_ratio = float(self._replay_index.requested_ratio)
        if requested_ratio <= 0.0:
            return 0
        replay_slots = int(math.floor(self.batch_size * requested_ratio))
        if self.max_replays_per_batch > 0:
            replay_slots = min(replay_slots, int(self.max_replays_per_batch))
        replay_slots = max(0, replay_slots)
        if replay_slots >= self.batch_size:
            raise ValueError(
                "Hard Replay replay_ratio leaves no base samples in a batch. "
                "Reduce replay_ratio or max_replays_per_batch."
            )
        return replay_slots

    def _has_candidates(self) -> bool:
        return bool(self._replay_index.image_candidates)

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
        candidates: Sequence[ReplayCandidate],
        *,
        total_samples: int,
        seed_offset: int,
    ) -> list[int]:
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
        generator.manual_seed(
            int(self.seed + self.epoch * 1000003 + self.rank * 9176 + seed_offset)
        )
        selected = torch.multinomial(
            torch.tensor(expanded_weights, dtype=torch.float32),
            num_samples=sample_count,
            replacement=False,
            generator=generator,
        ).tolist()

        samples: list[int] = []
        for expanded_index in selected:
            candidate = candidates[expanded_positions[int(expanded_index)]]
            samples.append(int(candidate.dataset_index))
        return samples

    def _summary_with_slot_plan(self, replay_index: ReplayIndex) -> dict[str, Any]:
        summary = dict(replay_index.summary)
        replay_slots = self._active_replay_slots()
        summary.update(
            {
                "active": False,
                "replay_slots_per_batch": int(replay_slots),
                "base_slots_per_batch": int(self.batch_size - replay_slots),
            }
        )
        return summary

    def _finalize_summary(
        self,
        *,
        total_base_samples: int,
        replay_samples: Sequence[int],
    ) -> None:
        replay_counts: dict[int, int] = {}
        gt_exposure_count = 0
        for sample in replay_samples:
            dataset_index = int(sample)
            replay_counts[dataset_index] = replay_counts.get(dataset_index, 0) + 1
            gt_exposure_count += self._sample_active_gt_count(sample)

        total_replay_samples = len(replay_samples)
        total_samples = int(total_base_samples) + int(total_replay_samples)
        effective_ratio = (
            total_replay_samples / float(total_samples)
            if total_samples > 0
            else 0.0
        )
        num_active_gt = int(self._replay_index.summary.get("replay_num_active_gt", 0))
        replay_exposure_per_gt = (
            gt_exposure_count / float(num_active_gt)
            if num_active_gt > 0
            else 0.0
        )
        replay_slots = self._active_replay_slots()
        self._last_summary = {
            **dict(self._replay_index.summary),
            "active": bool(total_replay_samples > 0),
            "replay_ratio_effective": effective_ratio,
            "replay_samples": int(total_replay_samples),
            "replay_unique_images": int(len(replay_counts)),
            "replay_exposure_per_gt": replay_exposure_per_gt,
            "replay_slots_per_batch": int(replay_slots),
            "base_slots_per_batch": int(self.batch_size - replay_slots),
        }

    def _sample_active_gt_count(self, sample: int) -> int:
        return int(self._replay_index.active_gt_counts.get(int(sample), 0))


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

    def refresh(self, *, missbank: Any, epoch: int) -> None:
        replay_index = self.planner.build_epoch_index(
            missbank=missbank,
            dataset=self.dataset,
            epoch=int(epoch),
        )
        self._latest_index = replay_index
        self.batch_sampler.set_replay_index(replay_index, epoch=int(epoch))

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(int(epoch))

    def set_base_only(self, enabled: bool) -> None:
        self.batch_sampler.set_base_only(bool(enabled))

    def summary(self) -> dict[str, Any]:
        return self.batch_sampler.last_summary


def load_hard_replay_config(path: str | Path, *, arch: str | None = None) -> HardReplayConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Hard Replay YAML must contain a mapping at the top level: {config_path}")
    return HardReplayConfig.from_mapping(raw, arch=arch)


def build_hard_replay_controller_from_yaml(
    path: str | Path,
    *,
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    arch: str | None = None,
) -> HardReplayController | None:
    config = load_hard_replay_config(path, arch=arch)
    if not config.enabled:
        return None
    batch_sampler = MixedReplayBatchSampler(
        dataset_size=len(dataset),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        max_replays_per_batch=int(config.max_replays_per_batch),
        replacement=bool(config.replacement),
        seed=int(seed),
        rank=int(rank),
        world_size=int(world_size),
    )
    return HardReplayController(
        config=config,
        dataset=dataset,
        batch_sampler=batch_sampler,
    )


def _empty_summary(
    *,
    enabled: bool,
    epoch: int,
    requested_ratio: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "active": False,
        "epoch": int(epoch),
        "reason": str(reason),
        "replay_ratio_requested": float(requested_ratio),
        "replay_ratio_effective": 0.0,
        "replay_num_images": 0,
        "replay_num_active_gt": 0,
        "replay_mean_image_weight": 0.0,
        "replay_mean_gt_priority": 0.0,
        "replay_samples": 0,
        "replay_unique_images": 0,
        "replay_exposure_per_gt": 0.0,
        "replay_slots_per_batch": 0,
        "base_slots_per_batch": 0,
    }


def _record_priority(
    record: Any,
    *,
    score_threshold: float,
    iou_threshold: float,
) -> float:
    miss_count = max(1, int(getattr(record, "miss_count", 0)))
    total_missed = max(0, int(getattr(record, "total_missed", 0)))
    max_miss_count = max(0, int(getattr(record, "max_miss_count", 0)))
    best_iou = getattr(record, "best_iou", None)
    best_score = getattr(record, "best_score", None)
    iou_gap = 0.0 if best_iou is None else max(0.0, float(iou_threshold) - float(best_iou))
    score_gap = 0.0 if best_score is None else max(0.0, float(score_threshold) - float(best_score))
    return (
        float(miss_count)
        + 0.25 * float(total_missed)
        + 0.1 * float(max_miss_count)
        + float(iou_gap)
        + 0.25 * float(score_gap)
    )


def _missbank_thresholds(missbank: Any) -> tuple[float, float]:
    config = getattr(missbank, "config", None)
    matching = getattr(config, "matching", None)
    score = float(getattr(matching, "score_threshold", 0.0))
    iou = float(getattr(matching, "iou_threshold", 0.5))
    return score, iou


def _record_uid(record: Any) -> str:
    value = getattr(record, "record_key", None)
    if value is not None:
        return str(value)
    image_id = getattr(record, "image_id", "")
    gt_id = getattr(record, "gt_id", "")
    gt_class = getattr(record, "gt_class", "")
    return f"{image_id}:{gt_id}:{gt_class}"


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return str(int(value.detach().cpu().flatten()[0].item()))
    return str(value)


def _chunks(values: Sequence[int], size: int):
    chunk_size = max(1, int(size))
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])
