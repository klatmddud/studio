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
        }


@dataclass(frozen=True, slots=True)
class ReplayCrop:
    gt_uid: str
    image_id: str
    class_id: int
    crop_box_abs: tuple[int, int, int, int]
    source_bbox_abs: tuple[int, int, int, int]
    severity: float
    support_box_abs: tuple[int, int, int, int] | None


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
            return ReplayIndex.empty(enabled=False, epoch=epoch, warmup_active=False)
        if mdmbpp is None:
            raise ValueError("Hard Replay requires an initialized mdmbpp module when enabled.")

        should_update = mdmbpp.should_update(epoch=epoch)
        if not should_update:
            return ReplayIndex.empty(enabled=True, epoch=epoch, warmup_active=True)

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

        active_images = len(replay_dataset_indices)
        mean_image_weight = 0.0
        if active_images > 0:
            mean_image_weight = sum(image_weights.values()) / float(active_images)

        mean_gt_severity = 0.0
        if replay_gt_ids:
            mean_gt_severity = severity_sum / float(len(replay_gt_ids))

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
                "epoch": int(epoch),
                "warmup_active": False,
                "replay_num_images": active_images,
                "replay_num_crops": 0,
                "replay_num_active_gt": len(replay_gt_ids),
                "replay_mean_image_weight": mean_image_weight,
                "replay_mean_gt_severity": mean_gt_severity,
                "replay_ratio_requested": self.config.replay_ratio,
                "replay_ratio_effective": 0.0,
                "replay_exposure_per_gt": 0.0,
            },
        )

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


class MixedReplayBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        shuffle: bool,
        replay_ratio: float,
        replacement: bool,
        seed: int,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.replay_ratio = float(replay_ratio)
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
        replay_is_active = self.replay_count > 0 and bool(replay_index.replay_dataset_indices)
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
        total_base_samples = self.dataset_size
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
            "replay_ratio_requested": self.replay_ratio if self._active_replay_count > 0 else 0.0,
            "replay_ratio_effective": effective_ratio,
            "replay_exposure_per_gt": replay_exposure_per_gt,
            "replay_samples": int(total_replay_samples),
            "replay_unique_images": int(len(replay_counts)),
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

    def start_epoch(self, *, mdmbpp: MDMBPlus | None, epoch: int) -> None:
        replay_index = self.planner.build_epoch_index(
            mdmbpp=mdmbpp,
            dataset=self.dataset,
            epoch=epoch,
        )
        self._latest_index = replay_index
        self.batch_sampler.set_replay_index(replay_index, epoch=epoch)

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
        replacement=config.replacement,
        seed=seed,
    )
    return HardReplayController(
        config=config,
        dataset=dataset,
        batch_sampler=batch_sampler,
    )
