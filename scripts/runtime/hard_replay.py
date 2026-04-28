from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Sampler


_DEFAULT_TRANSITIONS = (
    "FN_BG->FN_BG",
    "FN_CLS->FN_CLS",
    "FN_LOC->FN_LOC",
    "FN_MISS->FN_MISS",
    "TP->FN_BG",
    "TP->FN_CLS",
    "TP->FN_LOC",
    "TP->FN_MISS",
)
_DEFAULT_STATES = ("FN_BG", "FN_CLS", "FN_LOC", "FN_MISS")


@dataclass(frozen=True, slots=True)
class HardReplayConfig:
    enabled: bool = False
    start_epoch: int = 3
    warmup_epochs: int = 0
    max_ratio: float = 0.25
    max_replays_per_batch: int = 0
    beta: float = 1.0
    temperature: float = 1.0
    max_image_weight: float = 5.0
    min_replay_weight: float = 1.0
    replacement: bool = True
    max_replays_per_gt_per_epoch: int = 4
    replay_recency_window: int = 3
    target_transitions: tuple[str, ...] = _DEFAULT_TRANSITIONS
    persistent_states: tuple[str, ...] = _DEFAULT_STATES
    min_observations: int = 2
    min_fn_streak: int = 2

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "HardReplayConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", False)),
            start_epoch=int(data.get("start_epoch", 3)),
            warmup_epochs=int(data.get("warmup_epochs", 0)),
            max_ratio=float(data.get("max_ratio", 0.25)),
            max_replays_per_batch=int(data.get("max_replays_per_batch", 0)),
            beta=float(data.get("beta", 1.0)),
            temperature=float(data.get("temperature", 1.0)),
            max_image_weight=float(data.get("max_image_weight", 5.0)),
            min_replay_weight=float(data.get("min_replay_weight", 1.0)),
            replacement=bool(data.get("replacement", True)),
            max_replays_per_gt_per_epoch=int(data.get("max_replays_per_gt_per_epoch", 4)),
            replay_recency_window=int(data.get("replay_recency_window", 3)),
            target_transitions=_coerce_string_tuple(
                data.get("target_transitions", _DEFAULT_TRANSITIONS)
            ),
            persistent_states=_coerce_string_tuple(data.get("persistent_states", _DEFAULT_STATES)),
            min_observations=int(data.get("min_observations", 2)),
            min_fn_streak=int(data.get("min_fn_streak", 2)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.start_epoch < 1:
            raise ValueError("Hard Replay start_epoch must be >= 1.")
        if self.warmup_epochs < 0:
            raise ValueError("Hard Replay warmup_epochs must be >= 0.")
        if not 0.0 <= self.max_ratio < 1.0:
            raise ValueError("Hard Replay max_ratio must satisfy 0 <= value < 1.")
        if self.max_replays_per_batch < 0:
            raise ValueError("Hard Replay max_replays_per_batch must be >= 0.")
        if self.beta < 0.0:
            raise ValueError("Hard Replay beta must be >= 0.")
        if self.temperature <= 0.0:
            raise ValueError("Hard Replay temperature must be > 0.")
        if self.max_image_weight <= 0.0:
            raise ValueError("Hard Replay max_image_weight must be > 0.")
        if self.min_replay_weight <= 0.0:
            raise ValueError("Hard Replay min_replay_weight must be > 0.")
        if self.max_replays_per_gt_per_epoch < 1:
            raise ValueError("Hard Replay max_replays_per_gt_per_epoch must be >= 1.")
        if self.replay_recency_window < 0:
            raise ValueError("Hard Replay replay_recency_window must be >= 0.")
        if self.min_observations < 1:
            raise ValueError("Hard Replay min_observations must be >= 1.")
        if self.min_fn_streak < 0:
            raise ValueError("Hard Replay min_fn_streak must be >= 0.")
        if not self.target_transitions:
            raise ValueError("Hard Replay target_transitions must not be empty.")
        if not self.persistent_states:
            raise ValueError("Hard Replay persistent_states must not be empty.")

    def scheduled_ratio(self, *, epoch: int) -> float:
        if not self.enabled or int(epoch) < self.start_epoch:
            return 0.0
        if self.max_ratio <= 0.0:
            return 0.0
        if self.warmup_epochs <= 1:
            return self.max_ratio
        progress = float(int(epoch) - self.start_epoch + 1) / float(self.warmup_epochs)
        return self.max_ratio * min(max(progress, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class ReplayIndex:
    enabled: bool
    image_weights: dict[str, float] = field(default_factory=dict)
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
        requested_ratio: float = 0.0,
        warmup_active: bool = False,
    ) -> "ReplayIndex":
        return cls(
            enabled=enabled,
            summary={
                "enabled": enabled,
                "active": False,
                "epoch": int(epoch),
                "warmup_active": bool(warmup_active),
                "replay_num_images": 0,
                "replay_num_active_gt": 0,
                "replay_mean_image_weight": 0.0,
                "replay_mean_gt_severity": 0.0,
                "replay_ratio_requested": float(requested_ratio),
                "replay_ratio_effective": 0.0,
                "replay_exposure_per_gt": 0.0,
                "replay_samples": 0,
                "replay_unique_images": 0,
            },
        )


class HardReplayPlanner:
    def __init__(self, config: HardReplayConfig) -> None:
        self.config = config

    def build_epoch_index(
        self,
        *,
        dhm: Any | None,
        dataset: Any,
        epoch: int,
    ) -> ReplayIndex:
        requested_ratio = self.config.scheduled_ratio(epoch=epoch)
        warmup_active = self.config.enabled and requested_ratio <= 0.0
        if not self.config.enabled or requested_ratio <= 0.0:
            return ReplayIndex.empty(
                enabled=self.config.enabled,
                epoch=epoch,
                requested_ratio=requested_ratio,
                warmup_active=warmup_active,
            )
        if dhm is None or len(dhm) == 0:
            return ReplayIndex.empty(
                enabled=True,
                epoch=epoch,
                requested_ratio=requested_ratio,
                warmup_active=False,
            )

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

        for dataset_index, image_id in enumerate(image_ids):
            records = self._select_replay_records(dhm=dhm, image_id=image_id, epoch=epoch)
            if not records:
                continue

            image_severity = sum(_record_priority(record) for record in records)
            raw_weight = 1.0 + self.config.beta * image_severity
            clipped_weight = min(self.config.max_image_weight, raw_weight)
            clipped_weight = max(self.config.min_replay_weight, clipped_weight)
            sampling_weight = clipped_weight**self.config.temperature

            image_key = str(image_id)
            image_weights[image_key] = clipped_weight
            replay_dataset_indices.append(int(dataset_index))
            replay_sampling_weights.append(float(sampling_weight))
            active_gt_counts[int(dataset_index)] = len(records)
            replay_caps[int(dataset_index)] = (
                self.config.max_replays_per_gt_per_epoch * len(records)
            )
            severity_sum += image_severity
            for record in records:
                replay_gt_ids.add(str(getattr(record, "gt_uid", "")))

        if not replay_dataset_indices:
            return ReplayIndex.empty(
                enabled=True,
                epoch=epoch,
                requested_ratio=requested_ratio,
                warmup_active=False,
            )

        mean_image_weight = sum(image_weights.values()) / float(len(image_weights))
        mean_gt_severity = severity_sum / float(max(len(replay_gt_ids), 1))
        return ReplayIndex(
            enabled=True,
            image_weights=image_weights,
            replay_gt_ids=replay_gt_ids,
            replay_dataset_indices=replay_dataset_indices,
            replay_sampling_weights=torch.tensor(replay_sampling_weights, dtype=torch.float32),
            active_gt_counts=active_gt_counts,
            replay_caps=replay_caps,
            summary={
                "enabled": True,
                "active": True,
                "epoch": int(epoch),
                "warmup_active": False,
                "replay_num_images": len(replay_dataset_indices),
                "replay_num_active_gt": len(replay_gt_ids),
                "replay_mean_image_weight": mean_image_weight,
                "replay_mean_gt_severity": mean_gt_severity,
                "replay_ratio_requested": requested_ratio,
                "replay_ratio_effective": 0.0,
                "replay_exposure_per_gt": 0.0,
                "replay_samples": 0,
                "replay_unique_images": 0,
            },
        )

    def _select_replay_records(
        self,
        *,
        dhm: Any,
        image_id: int | str,
        epoch: int,
    ) -> list[Any]:
        records = dhm.get_image_records(image_id)
        selected: list[Any] = []
        for record in records:
            if int(getattr(record, "total_seen", 0)) < self.config.min_observations:
                continue
            if self.config.replay_recency_window > 0:
                last_fn_epoch = getattr(record, "last_fn_epoch", None)
                if last_fn_epoch is None:
                    continue
                if int(epoch) - int(last_fn_epoch) > self.config.replay_recency_window:
                    continue

            transition_hit = str(getattr(record, "last_transition", "")) in set(
                self.config.target_transitions
            )
            persistent_hit = (
                str(getattr(record, "last_state", "")) in set(self.config.persistent_states)
                and int(getattr(record, "consecutive_fn", 0)) >= self.config.min_fn_streak
            )
            if transition_hit or persistent_hit:
                selected.append(record)
        selected.sort(key=_record_priority, reverse=True)
        return selected


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
        self._replay_index = ReplayIndex.empty(enabled=False)
        self._last_summary: dict[str, float | int | bool] = dict(self._replay_index.summary)
        self._active_replay_count = 0
        self._active_base_count = self.batch_size

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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_replay_index(self, replay_index: ReplayIndex, *, epoch: int) -> None:
        self._replay_index = replay_index
        self.epoch = int(epoch)

        requested_ratio = float(replay_index.summary.get("replay_ratio_requested", 0.0))
        replay_count = int(math.floor(self.batch_size * requested_ratio))
        if self.max_replays_per_batch > 0:
            replay_count = min(replay_count, self.max_replays_per_batch)
        replay_is_active = replay_count > 0 and bool(replay_index.replay_dataset_indices)

        self._active_replay_count = replay_count if replay_is_active else 0
        self._active_base_count = self.batch_size - self._active_replay_count
        if self._active_base_count < 1:
            raise ValueError(
                "Hard Replay max_ratio is too large for the configured batch size. "
                "At least one base sample must remain in each batch."
            )

        self._last_summary = {
            **dict(replay_index.summary),
            "epoch": int(epoch),
            "active": bool(replay_is_active),
            "replay_ratio_requested": requested_ratio if replay_is_active else 0.0,
            "base_slots_per_batch": self._active_base_count,
            "replay_slots_per_batch": self._active_replay_count,
            "batch_size": self.batch_size,
        }

    def summary(self) -> dict[str, float | int | bool]:
        return dict(self._last_summary)

    def __len__(self) -> int:
        if self.dataset_size <= 0:
            return 0
        return int(math.ceil(self._num_base_samples() / float(self._active_base_count)))

    def __iter__(self):
        if self.dataset_size <= 0:
            self._finalize_summary(Counter(), total_replay_samples=0)
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + self.rank * 100_003)

        base_indices = self._build_base_indices(generator)
        num_batches = len(self)
        replay_schedule = self._build_replay_schedule(
            total_replay_samples=num_batches * self._active_replay_count,
            generator=generator,
        )
        replay_cursor = 0
        replay_counts: Counter[int] = Counter()

        for batch_start in range(0, len(base_indices), self._active_base_count):
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

    def _num_base_samples(self) -> int:
        if self.world_size <= 1:
            return self.dataset_size
        return int(math.ceil(self.dataset_size / float(self.world_size)))

    def _build_base_indices(self, generator: torch.Generator) -> list[int]:
        if self.shuffle:
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        if self.world_size <= 1:
            return indices

        total_size = self._num_base_samples() * self.world_size
        if len(indices) < total_size:
            padding = indices[: total_size - len(indices)]
            indices = [*indices, *padding]
        return indices[self.rank : total_size : self.world_size]

    def _build_replay_schedule(
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

    def _finalize_summary(
        self,
        replay_counts: Counter[int],
        *,
        total_replay_samples: int,
    ) -> None:
        total_base_samples = self._num_base_samples()
        total_samples = total_base_samples + total_replay_samples

        replay_exposure_per_gt = 0.0
        num_active_gt = int(self._replay_index.summary.get("replay_num_active_gt", 0))
        if num_active_gt > 0:
            gt_exposures = 0
            for dataset_index, count in replay_counts.items():
                gt_exposures += count * int(self._replay_index.active_gt_counts.get(dataset_index, 0))
            replay_exposure_per_gt = gt_exposures / float(num_active_gt)

        effective_ratio = 0.0
        if total_samples > 0:
            effective_ratio = total_replay_samples / float(total_samples)

        self._last_summary = {
            **dict(self._replay_index.summary),
            "active": bool(total_replay_samples > 0),
            "replay_ratio_requested": (
                float(self._replay_index.summary.get("replay_ratio_requested", 0.0))
                if total_replay_samples > 0
                else 0.0
            ),
            "replay_ratio_effective": effective_ratio,
            "replay_exposure_per_gt": replay_exposure_per_gt,
            "replay_samples": int(total_replay_samples),
            "replay_unique_images": int(len(replay_counts)),
            "base_slots_per_batch": self._active_base_count,
            "replay_slots_per_batch": self._active_replay_count,
            "batch_size": self.batch_size,
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

    def start_epoch(self, *, dhm: Any | None, epoch: int) -> None:
        replay_index = self.planner.build_epoch_index(
            dhm=dhm,
            dataset=self.dataset,
            epoch=epoch,
        )
        self._latest_index = replay_index
        self.batch_sampler.set_replay_index(replay_index, epoch=epoch)

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(epoch)

    def summary(self) -> dict[str, float | int | bool]:
        return self.batch_sampler.summary()


def build_hard_replay_controller(
    config: Mapping[str, Any],
    *,
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> HardReplayController | None:
    replay_config = HardReplayConfig.from_mapping(config)
    if not replay_config.enabled:
        return None
    batch_sampler = MixedReplayBatchSampler(
        dataset_size=len(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        max_replays_per_batch=replay_config.max_replays_per_batch,
        replacement=replay_config.replacement,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    return HardReplayController(
        config=replay_config,
        dataset=dataset,
        batch_sampler=batch_sampler,
    )


def _record_priority(record: Any) -> float:
    return (
        float(getattr(record, "instability_score", 0.0))
        + float(getattr(record, "consecutive_fn", 0))
        + 2.0 * float(getattr(record, "forgetting_count", 0))
        + 0.25 * float(getattr(record, "fn_count", 0))
        + 0.1 * float(getattr(record, "zero_pos_count", 0))
    )


def _coerce_string_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence):
        return tuple(str(item) for item in raw if str(item))
    return ()
