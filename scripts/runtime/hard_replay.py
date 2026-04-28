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
_DEFAULT_LOC_TRANSITIONS = ("FN_LOC->FN_LOC", "TP->FN_LOC")
_DEFAULT_LOC_STATES = ("FN_LOC",)
_IMAGE_POLICY = "image"
_LOC_CROP_POLICY = "loc_crop"


@dataclass(frozen=True, slots=True)
class LocalizationRepairConfig:
    enabled: bool = True
    replay_fraction: float = 0.6
    context_scale: float = 2.0
    context_scale_jitter: float = 0.25
    center_jitter: float = 0.10
    min_crop_size: int = 128
    min_visible_ratio: float = 0.50
    focus_min_visible_ratio: float = 0.90
    include_other_gt: bool = True
    target_transitions: tuple[str, ...] = _DEFAULT_LOC_TRANSITIONS
    persistent_states: tuple[str, ...] = _DEFAULT_LOC_STATES

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
    ) -> "LocalizationRepairConfig":
        data = dict(raw or {})
        config = cls(
            enabled=bool(data.get("enabled", True)),
            replay_fraction=float(data.get("replay_fraction", 0.6)),
            context_scale=float(data.get("context_scale", 2.0)),
            context_scale_jitter=float(data.get("context_scale_jitter", 0.25)),
            center_jitter=float(data.get("center_jitter", 0.10)),
            min_crop_size=int(data.get("min_crop_size", 128)),
            min_visible_ratio=float(data.get("min_visible_ratio", 0.50)),
            focus_min_visible_ratio=float(data.get("focus_min_visible_ratio", 0.90)),
            include_other_gt=bool(data.get("include_other_gt", True)),
            target_transitions=_coerce_string_tuple(
                data.get("target_transitions", _DEFAULT_LOC_TRANSITIONS)
            ),
            persistent_states=_coerce_string_tuple(
                data.get("persistent_states", _DEFAULT_LOC_STATES)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= self.replay_fraction <= 1.0:
            raise ValueError("Hard Replay loc_repair.replay_fraction must satisfy 0 <= value <= 1.")
        if self.context_scale <= 0.0:
            raise ValueError("Hard Replay loc_repair.context_scale must be > 0.")
        if self.context_scale_jitter < 0.0:
            raise ValueError("Hard Replay loc_repair.context_scale_jitter must be >= 0.")
        if self.center_jitter < 0.0:
            raise ValueError("Hard Replay loc_repair.center_jitter must be >= 0.")
        if self.min_crop_size < 1:
            raise ValueError("Hard Replay loc_repair.min_crop_size must be >= 1.")
        for field_name in ("min_visible_ratio", "focus_min_visible_ratio"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Hard Replay loc_repair.{field_name} must satisfy 0 <= value <= 1."
                )
        if not self.target_transitions:
            raise ValueError("Hard Replay loc_repair.target_transitions must not be empty.")
        if not self.persistent_states:
            raise ValueError("Hard Replay loc_repair.persistent_states must not be empty.")


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
    loc_repair: LocalizationRepairConfig = field(default_factory=LocalizationRepairConfig)

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
            loc_repair=LocalizationRepairConfig.from_mapping(data.get("loc_repair")),
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
        self.loc_repair.validate()

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
class ReplaySampleRef:
    dataset_index: int
    policy: str = _LOC_CROP_POLICY
    gt_uid: str = ""
    ann_id: str | None = None
    class_id: int = 0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    state: str = ""
    transition: str = ""
    priority: float = 0.0
    seed: int = 0
    context_scale: float = 2.0
    context_scale_jitter: float = 0.25
    center_jitter: float = 0.10
    min_crop_size: int = 128
    min_visible_ratio: float = 0.50
    focus_min_visible_ratio: float = 0.90
    include_other_gt: bool = True


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    dataset_index: int
    policy: str
    weight: float
    cap: int
    active_gt_count: int
    priority: float
    gt_uid: str = ""
    ann_id: str | None = None
    class_id: int = 0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    state: str = ""
    transition: str = ""
    loc_repair: LocalizationRepairConfig | None = None

    def to_sample(self, *, seed: int) -> int | ReplaySampleRef:
        if self.policy != _LOC_CROP_POLICY:
            return int(self.dataset_index)
        loc_repair = self.loc_repair or LocalizationRepairConfig()
        return ReplaySampleRef(
            dataset_index=int(self.dataset_index),
            policy=_LOC_CROP_POLICY,
            gt_uid=self.gt_uid,
            ann_id=self.ann_id,
            class_id=int(self.class_id),
            bbox=self.bbox,
            state=self.state,
            transition=self.transition,
            priority=float(self.priority),
            seed=int(seed),
            context_scale=float(loc_repair.context_scale),
            context_scale_jitter=float(loc_repair.context_scale_jitter),
            center_jitter=float(loc_repair.center_jitter),
            min_crop_size=int(loc_repair.min_crop_size),
            min_visible_ratio=float(loc_repair.min_visible_ratio),
            focus_min_visible_ratio=float(loc_repair.focus_min_visible_ratio),
            include_other_gt=bool(loc_repair.include_other_gt),
        )


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
    image_candidates: list[ReplayCandidate] = field(default_factory=list)
    loc_candidates: list[ReplayCandidate] = field(default_factory=list)
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
                "replay_sample_budget": 0,
                "replay_samples": 0,
                "replay_unique_images": 0,
                "loc_repair_enabled": False,
                "loc_repair_candidates": 0,
                "loc_repair_num_images": 0,
                "loc_repair_num_active_gt": 0,
                "loc_repair_mean_priority": 0.0,
                "loc_repair_samples": 0,
                "loc_repair_unique_images": 0,
                "loc_repair_unique_gt": 0,
                "loc_repair_slots_per_batch": 0,
                "image_replay_samples": 0,
            },
        )


class HardReplayPlanner:
    def __init__(self, config: HardReplayConfig) -> None:
        self.config = config
        self._target_transitions = set(config.target_transitions)
        self._persistent_states = set(config.persistent_states)
        self._loc_target_transitions = set(config.loc_repair.target_transitions)
        self._loc_persistent_states = set(config.loc_repair.persistent_states)

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
        replay_gt_ids: set[str] = set()
        image_candidates: list[ReplayCandidate] = []
        loc_candidates: list[ReplayCandidate] = []
        active_gt_counts: dict[int, int] = {}
        replay_caps: dict[int, int] = {}
        severity_sum = 0.0

        tau_iou = _dhm_tau_iou(dhm)
        for dataset_index, image_id in enumerate(image_ids):
            records = self._select_replay_records(dhm=dhm, image_id=image_id, epoch=epoch)
            loc_records = self._select_loc_repair_records(
                dhm=dhm,
                image_id=image_id,
                epoch=epoch,
            )
            if not records and not loc_records:
                continue

            if records:
                image_priority = sum(_record_priority(record) for record in records)
                sampling_weight = self._sampling_weight(image_priority)
                clipped_weight = self._clipped_weight(image_priority)
                image_key = str(image_id)
                image_weights[image_key] = clipped_weight
                cap = self.config.max_replays_per_gt_per_epoch * len(records)
                image_candidates.append(
                    ReplayCandidate(
                        dataset_index=int(dataset_index),
                        policy=_IMAGE_POLICY,
                        weight=float(sampling_weight),
                        cap=int(cap),
                        active_gt_count=len(records),
                        priority=float(image_priority),
                    )
                )
                active_gt_counts[int(dataset_index)] = len(records)
                replay_caps[int(dataset_index)] = int(cap)
                severity_sum += image_priority
                for record in records:
                    replay_gt_ids.add(str(getattr(record, "gt_uid", "")))

            if self.config.loc_repair.enabled:
                for record in loc_records:
                    loc_priority = _loc_record_priority(record, tau_iou=tau_iou)
                    loc_candidates.append(
                        ReplayCandidate(
                            dataset_index=int(dataset_index),
                            policy=_LOC_CROP_POLICY,
                            weight=float(self._sampling_weight(loc_priority)),
                            cap=int(self.config.max_replays_per_gt_per_epoch),
                            active_gt_count=1,
                            priority=float(loc_priority),
                            gt_uid=str(getattr(record, "gt_uid", "")),
                            ann_id=_record_ann_id(record),
                            class_id=int(getattr(record, "class_id", 0)),
                            bbox=_record_bbox_tuple(record),
                            state=str(getattr(record, "last_state", "")),
                            transition=str(getattr(record, "last_transition", "") or ""),
                            loc_repair=self.config.loc_repair,
                        )
                    )
                    replay_gt_ids.add(str(getattr(record, "gt_uid", "")))

        all_candidates = [*image_candidates, *loc_candidates]
        if not all_candidates:
            return ReplayIndex.empty(
                enabled=True,
                epoch=epoch,
                requested_ratio=requested_ratio,
                warmup_active=False,
            )

        replay_dataset_indices = sorted({candidate.dataset_index for candidate in all_candidates})
        replay_sampling_weights = torch.tensor(
            [candidate.weight for candidate in image_candidates],
            dtype=torch.float32,
        )
        mean_image_weight = (
            sum(image_weights.values()) / float(len(image_weights)) if image_weights else 0.0
        )
        mean_gt_severity = severity_sum / float(max(len(replay_gt_ids), 1))
        loc_priorities = [candidate.priority for candidate in loc_candidates]
        loc_images = {candidate.dataset_index for candidate in loc_candidates}
        loc_gt_ids = {candidate.gt_uid for candidate in loc_candidates if candidate.gt_uid}

        return ReplayIndex(
            enabled=True,
            image_weights=image_weights,
            replay_gt_ids=replay_gt_ids,
            replay_dataset_indices=replay_dataset_indices,
            replay_sampling_weights=replay_sampling_weights,
            active_gt_counts=active_gt_counts,
            replay_caps=replay_caps,
            image_candidates=image_candidates,
            loc_candidates=loc_candidates,
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
                "replay_sample_budget": 0,
                "replay_samples": 0,
                "replay_unique_images": 0,
                "loc_repair_enabled": bool(self.config.loc_repair.enabled),
                "loc_repair_candidates": len(loc_candidates),
                "loc_repair_num_images": len(loc_images),
                "loc_repair_num_active_gt": len(loc_gt_ids),
                "loc_repair_mean_priority": (
                    sum(loc_priorities) / float(len(loc_priorities)) if loc_priorities else 0.0
                ),
                "loc_repair_samples": 0,
                "loc_repair_unique_images": 0,
                "loc_repair_unique_gt": 0,
                "loc_repair_slots_per_batch": 0,
                "image_replay_samples": 0,
            },
        )

    def _sampling_weight(self, priority: float) -> float:
        return self._clipped_weight(priority) ** self.config.temperature

    def _clipped_weight(self, priority: float) -> float:
        raw_weight = 1.0 + self.config.beta * float(priority)
        clipped_weight = min(self.config.max_image_weight, raw_weight)
        return max(self.config.min_replay_weight, clipped_weight)

    def _select_replay_records(
        self,
        *,
        dhm: Any,
        image_id: int | str,
        epoch: int,
    ) -> list[Any]:
        selected: list[Any] = []
        for record in dhm.get_image_records(image_id):
            if not self._record_is_eligible(record, epoch=epoch):
                continue
            transition_hit = str(getattr(record, "last_transition", "")) in self._target_transitions
            persistent_hit = (
                str(getattr(record, "last_state", "")) in self._persistent_states
                and int(getattr(record, "consecutive_fn", 0)) >= self.config.min_fn_streak
            )
            if transition_hit or persistent_hit:
                selected.append(record)
        selected.sort(key=_record_priority, reverse=True)
        return selected

    def _select_loc_repair_records(
        self,
        *,
        dhm: Any,
        image_id: int | str,
        epoch: int,
    ) -> list[Any]:
        if not self.config.loc_repair.enabled:
            return []
        selected: list[Any] = []
        for record in dhm.get_image_records(image_id):
            if not self._record_is_eligible(record, epoch=epoch):
                continue
            transition_hit = (
                str(getattr(record, "last_transition", "")) in self._loc_target_transitions
            )
            persistent_hit = (
                str(getattr(record, "last_state", "")) in self._loc_persistent_states
                and int(getattr(record, "consecutive_fn", 0)) >= self.config.min_fn_streak
            )
            if transition_hit or persistent_hit:
                selected.append(record)
        selected.sort(key=lambda record: _loc_record_priority(record), reverse=True)
        return selected

    def _record_is_eligible(self, record: Any, *, epoch: int) -> bool:
        if int(getattr(record, "total_seen", 0)) < self.config.min_observations:
            return False
        if self.config.replay_recency_window <= 0:
            return True
        last_fn_epoch = getattr(record, "last_fn_epoch", None)
        if last_fn_epoch is None:
            return False
        return int(epoch) - int(last_fn_epoch) <= self.config.replay_recency_window


class MixedReplayBatchSampler(Sampler[list[Any]]):
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
        self._active_loc_replay_count = 0
        self._active_base_count = self.batch_size
        self._planned_replay_samples = 0

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

        candidate_capacity = self._candidate_capacity(replay_index.image_candidates)
        candidate_capacity += self._candidate_capacity(replay_index.loc_candidates)
        replay_is_active = replay_count > 0 and candidate_capacity > 0

        self._active_replay_count = replay_count if replay_is_active else 0
        self._active_base_count = (
            self.batch_size - self._active_replay_count if replay_is_active else self.batch_size
        )
        if self._active_base_count < 1:
            raise ValueError(
                "Hard Replay max_ratio is too large for the configured batch size. "
                "At least one base sample must remain in each replay-active batch."
            )

        loc_repair_enabled = bool(replay_index.summary.get("loc_repair_enabled", False))
        loc_fraction = 0.0
        if loc_repair_enabled and replay_index.loc_candidates and self._active_replay_count > 0:
            first_candidate = replay_index.loc_candidates[0]
            if first_candidate.loc_repair is not None:
                loc_fraction = float(first_candidate.loc_repair.replay_fraction)
        self._active_loc_replay_count = (
            min(self._active_replay_count, int(round(self._active_replay_count * loc_fraction)))
            if replay_is_active
            else 0
        )
        self._planned_replay_samples = (
            int(len(self) * self._active_replay_count) if replay_is_active else 0
        )

        self._last_summary = {
            **dict(replay_index.summary),
            "epoch": int(epoch),
            "active": bool(replay_is_active),
            "replay_ratio_requested": requested_ratio if replay_is_active else 0.0,
            "replay_sample_budget": self._planned_replay_samples,
            "base_slots_per_batch": self._active_base_count,
            "replay_slots_per_batch": self._active_replay_count,
            "loc_repair_slots_per_batch": self._active_loc_replay_count,
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
            self._finalize_summary(
                Counter(),
                Counter(),
                Counter(),
                set(),
                set(),
                total_replay_samples=0,
                gt_exposure_count=0,
            )
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + self.rank * 100_003)

        base_indices = self._build_base_indices(generator)
        planned_replay_samples = len(self) * self._active_replay_count
        self._planned_replay_samples = int(planned_replay_samples)
        loc_schedule = self._build_candidate_schedule(
            self._replay_index.loc_candidates,
            total_replay_samples=planned_replay_samples,
            generator=generator,
        )
        image_schedule = self._build_candidate_schedule(
            self._replay_index.image_candidates,
            total_replay_samples=planned_replay_samples,
            generator=generator,
        )

        base_cursor = 0
        loc_cursor = 0
        image_cursor = 0
        replay_used = 0
        replay_counts: Counter[int] = Counter()
        policy_counts: Counter[str] = Counter()
        state_counts: Counter[str] = Counter()
        loc_gt_ids: set[str] = set()
        loc_image_ids: set[int] = set()
        gt_exposure_count = 0

        while base_cursor < len(base_indices):
            desired_replay = 0
            if self._active_replay_count > 0:
                desired_replay = min(
                    self._active_replay_count,
                    planned_replay_samples - replay_used,
                )
            loc_desired = min(self._active_loc_replay_count, desired_replay)
            image_desired = desired_replay - loc_desired

            loc_slice, loc_cursor = _take(loc_schedule, loc_cursor, loc_desired)
            image_slice, image_cursor = _take(image_schedule, image_cursor, image_desired)

            if len(loc_slice) < loc_desired:
                fill, image_cursor = _take(
                    image_schedule,
                    image_cursor,
                    loc_desired - len(loc_slice),
                )
                image_slice.extend(fill)
            if len(image_slice) < image_desired:
                fill, loc_cursor = _take(
                    loc_schedule,
                    loc_cursor,
                    image_desired - len(image_slice),
                )
                loc_slice.extend(fill)

            replay_slice: list[int | ReplaySampleRef] = [*loc_slice, *image_slice]
            if desired_replay > len(replay_slice):
                replay_slice = replay_slice[:desired_replay]
            replay_used += len(replay_slice)

            base_take = (
                self._active_base_count if self._active_replay_count > 0 else self.batch_size
            )
            batch: list[Any] = list(base_indices[base_cursor : base_cursor + base_take])
            base_cursor += len(batch)
            batch.extend(replay_slice)

            if not batch:
                break
            if self.shuffle and len(batch) > 1:
                order = torch.randperm(len(batch), generator=generator).tolist()
                batch = [batch[index] for index in order]

            for sample in replay_slice:
                dataset_index = _sample_dataset_index(sample)
                replay_counts[dataset_index] += 1
                policy = _sample_policy(sample)
                policy_counts[policy] += 1
                state = _sample_state(sample)
                if state:
                    state_counts[state] += 1
                gt_exposure_count += _sample_active_gt_count(
                    sample,
                    active_gt_counts=self._replay_index.active_gt_counts,
                )
                if isinstance(sample, ReplaySampleRef) and sample.gt_uid:
                    loc_gt_ids.add(sample.gt_uid)
                    loc_image_ids.add(int(sample.dataset_index))
            yield batch

        self._finalize_summary(
            replay_counts,
            policy_counts,
            state_counts,
            loc_gt_ids,
            loc_image_ids,
            total_replay_samples=replay_used,
            gt_exposure_count=gt_exposure_count,
        )

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

    def _candidate_capacity(self, candidates: Sequence[ReplayCandidate]) -> int:
        if not candidates:
            return 0
        if not self.replacement:
            return len(candidates)
        return sum(max(int(candidate.cap), 0) for candidate in candidates)

    def _build_candidate_schedule(
        self,
        candidates: Sequence[ReplayCandidate],
        *,
        total_replay_samples: int,
        generator: torch.Generator,
    ) -> list[int | ReplaySampleRef]:
        if total_replay_samples <= 0 or not candidates:
            return []

        weights = torch.tensor([candidate.weight for candidate in candidates], dtype=torch.float32)
        if weights.numel() != len(candidates):
            return []

        if not self.replacement:
            sample_count = min(total_replay_samples, len(candidates))
            chosen = torch.multinomial(weights, sample_count, replacement=False, generator=generator)
            return [
                candidates[index].to_sample(seed=_next_seed(generator))
                for index in chosen.tolist()
            ]

        expanded_positions: list[int] = []
        expanded_weights: list[float] = []
        for position, candidate in enumerate(candidates):
            cap = max(int(candidate.cap), 0)
            if cap <= 0:
                continue
            expanded_positions.extend([position] * cap)
            expanded_weights.extend([float(candidate.weight)] * cap)

        if not expanded_positions:
            return []

        sample_count = min(total_replay_samples, len(expanded_positions))
        expanded_tensor = torch.tensor(expanded_weights, dtype=torch.float32)
        chosen = torch.multinomial(
            expanded_tensor,
            sample_count,
            replacement=False,
            generator=generator,
        )
        return [
            candidates[expanded_positions[index]].to_sample(seed=_next_seed(generator))
            for index in chosen.tolist()
        ]

    def _finalize_summary(
        self,
        replay_counts: Counter[int],
        policy_counts: Counter[str],
        state_counts: Counter[str],
        loc_gt_ids: set[str],
        loc_image_ids: set[int],
        *,
        total_replay_samples: int,
        gt_exposure_count: int,
    ) -> None:
        total_base_samples = self._num_base_samples()
        total_samples = total_base_samples + total_replay_samples

        replay_exposure_per_gt = 0.0
        num_active_gt = int(self._replay_index.summary.get("replay_num_active_gt", 0))
        if num_active_gt > 0:
            replay_exposure_per_gt = float(gt_exposure_count) / float(num_active_gt)

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
            "replay_sample_budget": int(self._planned_replay_samples),
            "replay_samples": int(total_replay_samples),
            "replay_unique_images": int(len(replay_counts)),
            "base_slots_per_batch": self._active_base_count,
            "replay_slots_per_batch": self._active_replay_count,
            "loc_repair_slots_per_batch": self._active_loc_replay_count,
            "batch_size": self.batch_size,
            "image_replay_samples": int(policy_counts.get(_IMAGE_POLICY, 0)),
            "loc_repair_samples": int(policy_counts.get(_LOC_CROP_POLICY, 0)),
            "loc_repair_unique_images": int(len(loc_image_ids)),
            "loc_repair_unique_gt": int(len(loc_gt_ids)),
            "replay_samples_by_policy": dict(policy_counts),
            "replay_samples_by_state": dict(state_counts),
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


def _loc_record_priority(record: Any, *, tau_iou: float = 0.5) -> float:
    iou_gap = max(0.0, float(tau_iou) - float(getattr(record, "last_iou", 0.0)))
    return (
        2.0 * float(getattr(record, "ema_box_loss", 0.0))
        + iou_gap
        + 0.8 * float(getattr(record, "ema_center_dist", 0.0))
        + 0.5 * max(0.0, 1.0 - float(getattr(record, "ema_centerness_target", 0.0)))
        + 0.5 * float(getattr(record, "consecutive_fn", 0))
        + float(getattr(record, "forgetting_count", 0))
        + 0.3 * float(getattr(record, "zero_pos_count", 0))
    )


def _record_bbox_tuple(record: Any) -> tuple[float, float, float, float]:
    bbox = getattr(record, "bbox", None)
    if isinstance(bbox, torch.Tensor):
        values = bbox.detach().cpu().to(dtype=torch.float32).flatten().tolist()
    elif isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)):
        values = [float(value) for value in bbox]
    else:
        values = [0.0, 0.0, 0.0, 0.0]
    padded = [*values[:4], 0.0, 0.0, 0.0, 0.0]
    return tuple(float(value) for value in padded[:4])  # type: ignore[return-value]


def _record_ann_id(record: Any) -> str | None:
    ann_id = getattr(record, "ann_id", None)
    if ann_id is None:
        return None
    text = str(ann_id)
    return text if text else None


def _dhm_tau_iou(dhm: Any) -> float:
    config = getattr(dhm, "config", None)
    mining = getattr(config, "mining", None)
    matching = getattr(mining, "matching", None)
    return float(getattr(matching, "tau_iou", 0.5))


def _coerce_string_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence):
        return tuple(str(item) for item in raw if str(item))
    return ()


def _take(
    values: Sequence[int | ReplaySampleRef],
    cursor: int,
    count: int,
) -> tuple[list[int | ReplaySampleRef], int]:
    if count <= 0 or cursor >= len(values):
        return [], cursor
    end = min(cursor + count, len(values))
    return list(values[cursor:end]), end


def _next_seed(generator: torch.Generator) -> int:
    return int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())


def _sample_dataset_index(sample: int | ReplaySampleRef) -> int:
    if isinstance(sample, ReplaySampleRef):
        return int(sample.dataset_index)
    return int(sample)


def _sample_policy(sample: int | ReplaySampleRef) -> str:
    if isinstance(sample, ReplaySampleRef):
        return str(sample.policy)
    return _IMAGE_POLICY


def _sample_state(sample: int | ReplaySampleRef) -> str:
    if isinstance(sample, ReplaySampleRef):
        return str(sample.state)
    return ""


def _sample_active_gt_count(
    sample: int | ReplaySampleRef,
    *,
    active_gt_counts: Mapping[int, int],
) -> int:
    if isinstance(sample, ReplaySampleRef):
        return 1
    return int(active_gt_counts.get(int(sample), 1))
