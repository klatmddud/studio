from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
import yaml
from torchvision.ops import boxes as box_ops

from .mdmb import normalize_arch, normalize_xyxy_boxes, select_topk_indices


FailureType: TypeAlias = Literal[
    "candidate_missing",
    "loc_near_miss",
    "cls_confusion",
    "score_suppression",
    "nms_suppression",
    "detected",
]

_FAILURE_TYPES: tuple[str, ...] = (
    "candidate_missing",
    "loc_near_miss",
    "cls_confusion",
    "score_suppression",
    "nms_suppression",
    "detected",
)

_DEFAULT_FAILURE_TYPE_PRIORS = {
    "candidate_missing": 1.0,
    "score_suppression": 0.8,
    "nms_suppression": 0.7,
    "cls_confusion": 0.6,
    "loc_near_miss": 0.5,
    "detected": 0.0,
}


@dataclass(frozen=True, slots=True)
class MDMBPlusConfig:
    enabled: bool = True
    warmup_epochs: int = 1
    detected_iou_threshold: float = 0.5
    near_iou_threshold: float = 0.1
    class_iou_threshold: float = 0.5
    record_match_threshold: float = 0.95
    max_per_image: int | None = None
    candidate_topk: int = 5
    store_topk_candidates: bool = True
    store_support_feature: bool = False
    severity_lambda_streak: float = 1.0
    severity_lambda_relapse: float = 1.0
    severity_lambda_failure_type: float = 1.0
    severity_lambda_coverage: float = 1.0
    failure_type_priors: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_FAILURE_TYPE_PRIORS)
    )
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "MDMBPlusConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))

        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        severity = {}
        top_severity = data.get("severity", {})
        if isinstance(top_severity, Mapping):
            severity.update(top_severity)
        override_severity = model_overrides.get("severity", {})
        if isinstance(override_severity, Mapping):
            severity.update(override_severity)

        priors = dict(_DEFAULT_FAILURE_TYPE_PRIORS)
        top_priors = data.get("failure_type_priors", {})
        if isinstance(top_priors, Mapping):
            priors.update({str(key): float(value) for key, value in top_priors.items()})
        override_priors = model_overrides.get("failure_type_priors", {})
        if isinstance(override_priors, Mapping):
            priors.update({str(key): float(value) for key, value in override_priors.items()})

        max_per_image = model_overrides.get("max_per_image", data.get("max_per_image"))
        config = cls(
            enabled=bool(model_overrides.get("enabled", data.get("enabled", True))),
            warmup_epochs=int(model_overrides.get("warmup_epochs", data.get("warmup_epochs", 1))),
            detected_iou_threshold=float(
                model_overrides.get(
                    "detected_iou_threshold",
                    data.get("detected_iou_threshold", 0.5),
                )
            ),
            near_iou_threshold=float(
                model_overrides.get("near_iou_threshold", data.get("near_iou_threshold", 0.1))
            ),
            class_iou_threshold=float(
                model_overrides.get("class_iou_threshold", data.get("class_iou_threshold", 0.5))
            ),
            record_match_threshold=float(
                model_overrides.get(
                    "record_match_threshold",
                    data.get("record_match_threshold", 0.95),
                )
            ),
            max_per_image=None if max_per_image is None else int(max_per_image),
            candidate_topk=int(
                model_overrides.get("candidate_topk", data.get("candidate_topk", 5))
            ),
            store_topk_candidates=bool(
                model_overrides.get(
                    "store_topk_candidates",
                    data.get("store_topk_candidates", True),
                )
            ),
            store_support_feature=bool(
                model_overrides.get(
                    "store_support_feature",
                    data.get("store_support_feature", False),
                )
            ),
            severity_lambda_streak=float(
                severity.get("lambda_streak", data.get("severity_lambda_streak", 1.0))
            ),
            severity_lambda_relapse=float(
                severity.get("lambda_relapse", data.get("severity_lambda_relapse", 1.0))
            ),
            severity_lambda_failure_type=float(
                severity.get(
                    "lambda_failure_type",
                    data.get("severity_lambda_failure_type", 1.0),
                )
            ),
            severity_lambda_coverage=float(
                severity.get("lambda_coverage", data.get("severity_lambda_coverage", 1.0))
            ),
            failure_type_priors=priors,
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        for field_name in (
            "detected_iou_threshold",
            "near_iou_threshold",
            "class_iou_threshold",
            "record_match_threshold",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"MDMB++ {field_name} must satisfy 0 <= value <= 1.")
        if self.max_per_image is not None and self.max_per_image < 1:
            raise ValueError("MDMB++ max_per_image must be >= 1 when provided.")
        if self.candidate_topk < 1:
            raise ValueError("MDMB++ candidate_topk must be >= 1.")
        if self.warmup_epochs < 0:
            raise ValueError("MDMB++ warmup_epochs must be >= 0.")
        missing = [name for name in _FAILURE_TYPES if name not in self.failure_type_priors]
        if missing:
            raise ValueError(
                "MDMB++ failure_type_priors must include every failure type. "
                f"Missing: {missing}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_epochs": self.warmup_epochs,
            "detected_iou_threshold": self.detected_iou_threshold,
            "near_iou_threshold": self.near_iou_threshold,
            "class_iou_threshold": self.class_iou_threshold,
            "record_match_threshold": self.record_match_threshold,
            "max_per_image": self.max_per_image,
            "candidate_topk": self.candidate_topk,
            "store_topk_candidates": self.store_topk_candidates,
            "store_support_feature": self.store_support_feature,
            "severity": {
                "lambda_streak": self.severity_lambda_streak,
                "lambda_relapse": self.severity_lambda_relapse,
                "lambda_failure_type": self.severity_lambda_failure_type,
                "lambda_coverage": self.severity_lambda_coverage,
            },
            "failure_type_priors": dict(self.failure_type_priors),
            "arch": self.arch,
        }


@dataclass(slots=True)
class CanonicalCandidate:
    stage: str
    box: torch.Tensor
    score: float
    label: int
    iou_to_gt: float
    survived_selection: bool
    survived_nms: bool | None
    rank: int | None
    level_or_stage_id: str | int | None

    def __post_init__(self) -> None:
        self.box = _as_region_tensor(self.box)
        self.stage = str(self.stage)
        self.score = float(self.score)
        self.label = int(self.label)
        self.iou_to_gt = float(self.iou_to_gt)
        self.survived_selection = bool(self.survived_selection)
        if self.survived_nms is not None:
            self.survived_nms = bool(self.survived_nms)
        if self.rank is not None:
            self.rank = int(self.rank)
        if self.level_or_stage_id is not None and not isinstance(
            self.level_or_stage_id, (str, int)
        ):
            self.level_or_stage_id = str(self.level_or_stage_id)

    def to_state(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "box": self.box.tolist(),
            "score": self.score,
            "label": self.label,
            "iou_to_gt": self.iou_to_gt,
            "survived_selection": self.survived_selection,
            "survived_nms": self.survived_nms,
            "rank": self.rank,
            "level_or_stage_id": self.level_or_stage_id,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CanonicalCandidate":
        return cls(
            stage=str(state["stage"]),
            box=_as_region_tensor(state["box"]),
            score=float(state.get("score", 0.0)),
            label=int(state.get("label", -1)),
            iou_to_gt=float(state.get("iou_to_gt", 0.0)),
            survived_selection=bool(state.get("survived_selection", False)),
            survived_nms=state.get("survived_nms"),
            rank=None if state.get("rank") is None else int(state["rank"]),
            level_or_stage_id=state.get("level_or_stage_id"),
        )


@dataclass(slots=True)
class SupportSnapshot:
    epoch: int
    box: torch.Tensor
    score: float
    feature: torch.Tensor | None
    feature_level: str | int | None

    def __post_init__(self) -> None:
        self.epoch = int(self.epoch)
        self.box = _as_region_tensor(self.box)
        self.score = float(self.score)
        if self.feature is not None:
            self.feature = torch.as_tensor(self.feature, dtype=torch.float32).detach().cpu()
        if self.feature_level is not None and not isinstance(self.feature_level, (str, int)):
            self.feature_level = str(self.feature_level)

    def to_state(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "box": self.box.tolist(),
            "score": self.score,
            "feature": None if self.feature is None else self.feature.tolist(),
            "feature_level": self.feature_level,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "SupportSnapshot":
        feature = state.get("feature")
        return cls(
            epoch=int(state.get("epoch", 0)),
            box=_as_region_tensor(state["box"]),
            score=float(state.get("score", 0.0)),
            feature=None if feature is None else torch.tensor(feature, dtype=torch.float32),
            feature_level=state.get("feature_level"),
        )


@dataclass(slots=True)
class GTFailureRecord:
    gt_uid: str
    image_id: str
    class_id: int
    bbox: torch.Tensor
    first_seen_epoch: int
    last_seen_epoch: int
    last_state: FailureType
    consecutive_miss_count: int
    max_consecutive_miss_count: int
    total_miss_count: int
    relapse_count: int
    last_detected_epoch: int | None
    last_failure_epoch: int | None
    last_failure_type: FailureType | None
    severity: float
    support: SupportSnapshot | None

    def __post_init__(self) -> None:
        self.gt_uid = str(self.gt_uid)
        self.image_id = str(self.image_id)
        self.class_id = int(self.class_id)
        self.bbox = _as_region_tensor(self.bbox)
        self.first_seen_epoch = int(self.first_seen_epoch)
        self.last_seen_epoch = int(self.last_seen_epoch)
        self.consecutive_miss_count = int(self.consecutive_miss_count)
        self.max_consecutive_miss_count = int(self.max_consecutive_miss_count)
        self.total_miss_count = int(self.total_miss_count)
        self.relapse_count = int(self.relapse_count)
        self.severity = float(self.severity)

    def to_state(self) -> dict[str, Any]:
        return {
            "gt_uid": self.gt_uid,
            "image_id": self.image_id,
            "class_id": self.class_id,
            "bbox": self.bbox.tolist(),
            "first_seen_epoch": self.first_seen_epoch,
            "last_seen_epoch": self.last_seen_epoch,
            "last_state": self.last_state,
            "consecutive_miss_count": self.consecutive_miss_count,
            "max_consecutive_miss_count": self.max_consecutive_miss_count,
            "total_miss_count": self.total_miss_count,
            "relapse_count": self.relapse_count,
            "last_detected_epoch": self.last_detected_epoch,
            "last_failure_epoch": self.last_failure_epoch,
            "last_failure_type": self.last_failure_type,
            "severity": self.severity,
            "support": None if self.support is None else self.support.to_state(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "GTFailureRecord":
        raw_support = state.get("support")
        return cls(
            gt_uid=str(state["gt_uid"]),
            image_id=str(state["image_id"]),
            class_id=int(state["class_id"]),
            bbox=_as_region_tensor(state["bbox"]),
            first_seen_epoch=int(state.get("first_seen_epoch", 0)),
            last_seen_epoch=int(state.get("last_seen_epoch", 0)),
            last_state=_coerce_failure_type(state.get("last_state", "candidate_missing")),
            consecutive_miss_count=int(state.get("consecutive_miss_count", 0)),
            max_consecutive_miss_count=int(state.get("max_consecutive_miss_count", 0)),
            total_miss_count=int(state.get("total_miss_count", 0)),
            relapse_count=int(state.get("relapse_count", 0)),
            last_detected_epoch=state.get("last_detected_epoch"),
            last_failure_epoch=state.get("last_failure_epoch"),
            last_failure_type=(
                None
                if state.get("last_failure_type") is None
                else _coerce_failure_type(state["last_failure_type"])
            ),
            severity=float(state.get("severity", 0.0)),
            support=(
                None
                if not isinstance(raw_support, Mapping)
                else SupportSnapshot.from_state(raw_support)
            ),
        )


@dataclass(slots=True)
class MDMBPlusEntry:
    gt_uid: str
    image_id: str
    class_id: int
    bbox: torch.Tensor
    failure_type: FailureType
    consecutive_miss_count: int
    max_consecutive_miss_count: int
    total_miss_count: int
    relapse: bool
    severity: float
    best_candidate: CanonicalCandidate | None
    topk_candidates: list[CanonicalCandidate]
    support: SupportSnapshot | None

    def __post_init__(self) -> None:
        self.gt_uid = str(self.gt_uid)
        self.image_id = str(self.image_id)
        self.class_id = int(self.class_id)
        self.bbox = _as_region_tensor(self.bbox)
        self.consecutive_miss_count = int(self.consecutive_miss_count)
        self.max_consecutive_miss_count = int(self.max_consecutive_miss_count)
        self.total_miss_count = int(self.total_miss_count)
        self.relapse = bool(self.relapse)
        self.severity = float(self.severity)
        self.topk_candidates = list(self.topk_candidates)

    def to_state(self) -> dict[str, Any]:
        return {
            "gt_uid": self.gt_uid,
            "image_id": self.image_id,
            "class_id": self.class_id,
            "bbox": self.bbox.tolist(),
            "failure_type": self.failure_type,
            "consecutive_miss_count": self.consecutive_miss_count,
            "max_consecutive_miss_count": self.max_consecutive_miss_count,
            "total_miss_count": self.total_miss_count,
            "relapse": self.relapse,
            "severity": self.severity,
            "best_candidate": None if self.best_candidate is None else self.best_candidate.to_state(),
            "topk_candidates": [candidate.to_state() for candidate in self.topk_candidates],
            "support": None if self.support is None else self.support.to_state(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "MDMBPlusEntry":
        raw_best = state.get("best_candidate")
        raw_topk = state.get("topk_candidates", ())
        raw_support = state.get("support")
        return cls(
            gt_uid=str(state["gt_uid"]),
            image_id=str(state["image_id"]),
            class_id=int(state["class_id"]),
            bbox=_as_region_tensor(state["bbox"]),
            failure_type=_coerce_failure_type(state.get("failure_type", "candidate_missing")),
            consecutive_miss_count=int(state.get("consecutive_miss_count", 1)),
            max_consecutive_miss_count=int(state.get("max_consecutive_miss_count", 1)),
            total_miss_count=int(state.get("total_miss_count", 1)),
            relapse=bool(state.get("relapse", False)),
            severity=float(state.get("severity", 0.0)),
            best_candidate=(
                None
                if not isinstance(raw_best, Mapping)
                else CanonicalCandidate.from_state(raw_best)
            ),
            topk_candidates=[
                CanonicalCandidate.from_state(item)
                for item in raw_topk
                if isinstance(item, Mapping)
            ],
            support=(
                None
                if not isinstance(raw_support, Mapping)
                else SupportSnapshot.from_state(raw_support)
            ),
        )


@dataclass(slots=True)
class PerImageCandidateSummary:
    image_id: str
    candidates_by_gt_index: dict[int, list[CanonicalCandidate]]

    def __post_init__(self) -> None:
        self.image_id = str(self.image_id)
        self.candidates_by_gt_index = {
            int(gt_index): list(candidates)
            for gt_index, candidates in self.candidates_by_gt_index.items()
        }

    def get_candidates(self, gt_index: int) -> list[CanonicalCandidate]:
        return list(self.candidates_by_gt_index.get(int(gt_index), ()))


class MDMBPlus(nn.Module):
    def __init__(self, config: MDMBPlusConfig) -> None:
        super().__init__()
        self.config = config
        self._bank: dict[str, list[MDMBPlusEntry]] = {}
        self._entry_index: dict[str, MDMBPlusEntry] = {}
        self._persistent_records: dict[str, GTFailureRecord] = {}
        self._global_max_consecutive_miss: int = 0
        self._class_statistics: dict[str, dict[str, float | int]] = {}
        self._epoch_recovery_candidates: int = 0
        self._epoch_recovery_success: int = 0
        self._epoch_relapses: int = 0
        self.current_epoch = 0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._epoch_recovery_candidates = 0
        self._epoch_recovery_success = 0
        self._epoch_relapses = 0

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def should_update(self, *, epoch: int | None = None) -> bool:
        if epoch is not None:
            self.current_epoch = int(epoch)
        if not self.config.enabled:
            return False
        return self.current_epoch > self.config.warmup_epochs

    def reset(self) -> None:
        self._bank.clear()
        self._entry_index.clear()
        self._persistent_records.clear()
        self._global_max_consecutive_miss = 0
        self._class_statistics = {}
        self._epoch_recovery_candidates = 0
        self._epoch_recovery_success = 0
        self._epoch_relapses = 0
        self.current_epoch = 0

    @torch.no_grad()
    def update(
        self,
        *,
        image_ids: Sequence[Any],
        final_boxes_list: Sequence[torch.Tensor | Sequence[Sequence[float]] | Sequence[float]],
        final_labels_list: Sequence[torch.Tensor | Sequence[int] | int],
        gt_boxes_list: Sequence[torch.Tensor | Sequence[Sequence[float]] | Sequence[float]],
        gt_labels_list: Sequence[torch.Tensor | Sequence[int] | int],
        image_shapes: Sequence[Sequence[int] | torch.Tensor],
        candidate_summary_list: Sequence[
            PerImageCandidateSummary
            | Mapping[int, Sequence[CanonicalCandidate | Mapping[str, Any]]]
            | Sequence[Sequence[CanonicalCandidate | Mapping[str, Any]]]
            | None
        ]
        | None = None,
        final_scores_list: Sequence[torch.Tensor | Sequence[float] | float] | None = None,
        gt_ids_list: Sequence[torch.Tensor | Sequence[Any] | Any] | None = None,
        epoch: int | None = None,
    ) -> None:
        if not self.should_update(epoch=epoch):
            return

        batch_size = len(image_ids)
        expected_lengths = (
            len(final_boxes_list),
            len(final_labels_list),
            len(gt_boxes_list),
            len(gt_labels_list),
            len(image_shapes),
        )
        if any(length != batch_size for length in expected_lengths):
            raise ValueError(
                "MDMB++ update inputs must share the same batch dimension: "
                f"image_ids={batch_size}, final_boxes={expected_lengths[0]}, "
                f"final_labels={expected_lengths[1]}, gt_boxes={expected_lengths[2]}, "
                f"gt_labels={expected_lengths[3]}, image_shapes={expected_lengths[4]}."
            )

        if candidate_summary_list is None:
            candidate_summary_list = [None] * batch_size
        elif len(candidate_summary_list) != batch_size:
            raise ValueError(
                "MDMB++ candidate_summary_list must share the same batch dimension as image_ids."
            )

        if final_scores_list is None:
            final_scores_list = [None] * batch_size
        elif len(final_scores_list) != batch_size:
            raise ValueError(
                "MDMB++ final_scores_list must share the same batch dimension as image_ids."
            )

        if gt_ids_list is None:
            gt_ids_list = [None] * batch_size
        elif len(gt_ids_list) != batch_size:
            raise ValueError("MDMB++ gt_ids_list must share the same batch dimension as image_ids.")

        new_entry_index = dict(self._entry_index)

        for (
            image_id,
            final_boxes,
            final_labels,
            gt_boxes,
            gt_labels,
            image_shape,
            raw_summary,
            final_scores,
            gt_ids,
        ) in zip(
            image_ids,
            final_boxes_list,
            final_labels_list,
            gt_boxes_list,
            gt_labels_list,
            image_shapes,
            candidate_summary_list,
            final_scores_list,
            gt_ids_list,
            strict=True,
        ):
            image_key = _normalize_image_id(image_id)
            gt_boxes_tensor = _as_box_tensor(gt_boxes)
            gt_labels_tensor = _as_label_tensor(gt_labels, device=gt_boxes_tensor.device)
            if gt_boxes_tensor.shape[0] != gt_labels_tensor.shape[0]:
                raise ValueError(
                    "MDMB++ gt_boxes and gt_labels must contain the same number of instances. "
                    f"Got {gt_boxes_tensor.shape[0]} and {gt_labels_tensor.shape[0]} for {image_key!r}."
                )

            if gt_boxes_tensor.numel() == 0:
                stale_entries = self._bank.pop(image_key, [])
                for entry in stale_entries:
                    new_entry_index.pop(entry.gt_uid, None)
                continue

            normalized_gt_boxes = normalize_xyxy_boxes(gt_boxes_tensor, image_shape).cpu()
            final_boxes_tensor = normalize_xyxy_boxes(final_boxes, image_shape).to(
                device=gt_boxes_tensor.device
            )
            final_labels_tensor = _as_label_tensor(final_labels, device=gt_boxes_tensor.device)
            final_scores_tensor = _as_score_tensor(final_scores, device=gt_boxes_tensor.device)
            if final_boxes_tensor.shape[0] != final_labels_tensor.shape[0]:
                raise ValueError(
                    "MDMB++ final_boxes and final_labels must contain the same number of predictions. "
                    f"Got {final_boxes_tensor.shape[0]} and {final_labels_tensor.shape[0]} for {image_key!r}."
                )
            if final_scores_tensor.shape[0] not in {0, final_boxes_tensor.shape[0]}:
                raise ValueError(
                    "MDMB++ final_scores must either be empty or match final_boxes in length. "
                    f"Got {final_scores_tensor.shape[0]} and {final_boxes_tensor.shape[0]} for {image_key!r}."
                )

            gt_ids_seq = _coerce_gt_ids(gt_ids, count=gt_boxes_tensor.shape[0])
            candidate_map = _coerce_candidate_summary(
                image_id=image_key,
                raw_summary=raw_summary,
                num_gt=gt_boxes_tensor.shape[0],
            )

            stale_entries = self._bank.get(image_key, [])
            for entry in stale_entries:
                new_entry_index.pop(entry.gt_uid, None)

            image_entries: list[MDMBPlusEntry] = []
            epoch = self.current_epoch

            for gt_index, (gt_box_norm, gt_label) in enumerate(
                zip(normalized_gt_boxes, gt_labels_tensor, strict=True)
            ):
                gt_uid = _make_gt_uid(
                    image_id=image_key,
                    class_id=int(gt_label.item()),
                    bbox=gt_box_norm,
                    gt_id=gt_ids_seq[gt_index],
                )
                previous_record = self._persistent_records.get(gt_uid)
                all_candidates = _coerce_candidates(candidate_map.get(gt_index, ()))
                stored_candidates = _select_stored_candidates(
                    all_candidates,
                    candidate_topk=self.config.candidate_topk,
                    store_topk=self.config.store_topk_candidates,
                )
                best_candidate = _select_best_candidate(all_candidates)
                final_match = _find_best_final_detection(
                    gt_box=gt_box_norm,
                    gt_label=int(gt_label.item()),
                    final_boxes=final_boxes_tensor,
                    final_labels=final_labels_tensor,
                    final_scores=final_scores_tensor,
                    detected_iou_threshold=self.config.detected_iou_threshold,
                )
                failure_type = classify_failure(
                    gt_box=gt_box_norm,
                    gt_label=int(gt_label.item()),
                    final_match=final_match,
                    candidates=all_candidates,
                    near_iou_threshold=self.config.near_iou_threshold,
                    class_iou_threshold=self.config.class_iou_threshold,
                )

                if (
                    previous_record is not None
                    and previous_record.last_state != "detected"
                    and previous_record.last_seen_epoch == epoch - 1
                ):
                    self._epoch_recovery_candidates += 1
                    if failure_type == "detected":
                        self._epoch_recovery_success += 1

                relapse_event = bool(
                    previous_record is not None
                    and previous_record.last_state == "detected"
                    and previous_record.last_detected_epoch is not None
                    and failure_type != "detected"
                )
                relapse_flag = bool(
                    previous_record is not None
                    and previous_record.last_detected_epoch is not None
                    and failure_type != "detected"
                )
                if relapse_event:
                    self._epoch_relapses += 1

                previous_support = None if previous_record is None else previous_record.support
                if failure_type == "detected":
                    record = GTFailureRecord(
                        gt_uid=gt_uid,
                        image_id=image_key,
                        class_id=int(gt_label.item()),
                        bbox=gt_box_norm,
                        first_seen_epoch=epoch if previous_record is None else previous_record.first_seen_epoch,
                        last_seen_epoch=epoch,
                        last_state="detected",
                        consecutive_miss_count=0,
                        max_consecutive_miss_count=(
                            0
                            if previous_record is None
                            else previous_record.max_consecutive_miss_count
                        ),
                        total_miss_count=0 if previous_record is None else previous_record.total_miss_count,
                        relapse_count=0 if previous_record is None else previous_record.relapse_count,
                        last_detected_epoch=epoch,
                        last_failure_epoch=(
                            None
                            if previous_record is None
                            else previous_record.last_failure_epoch
                        ),
                        last_failure_type=(
                            None
                            if previous_record is None
                            else previous_record.last_failure_type
                        ),
                        severity=0.0,
                        support=_build_support_snapshot(
                            epoch=epoch,
                            final_match=final_match,
                            fallback_candidate=best_candidate,
                            previous_support=previous_support,
                            keep_feature=self.config.store_support_feature,
                        ),
                    )
                    self._persistent_records[gt_uid] = record
                    continue

                previous_consecutive = (
                    0
                    if previous_record is None or previous_record.last_state == "detected"
                    else previous_record.consecutive_miss_count
                )
                consecutive_miss_count = previous_consecutive + 1
                max_consecutive_miss_count = max(
                    consecutive_miss_count,
                    0
                    if previous_record is None
                    else previous_record.max_consecutive_miss_count,
                )
                total_miss_count = consecutive_miss_count
                if previous_record is not None:
                    total_miss_count = previous_record.total_miss_count + 1

                coverage = _max_candidate_iou(all_candidates)
                severity = self._compute_severity(
                    consecutive_miss_count=consecutive_miss_count,
                    failure_type=failure_type,
                    relapse=relapse_flag,
                    coverage=coverage,
                )

                record = GTFailureRecord(
                    gt_uid=gt_uid,
                    image_id=image_key,
                    class_id=int(gt_label.item()),
                    bbox=gt_box_norm,
                    first_seen_epoch=epoch if previous_record is None else previous_record.first_seen_epoch,
                    last_seen_epoch=epoch,
                    last_state=failure_type,
                    consecutive_miss_count=consecutive_miss_count,
                    max_consecutive_miss_count=max_consecutive_miss_count,
                    total_miss_count=total_miss_count,
                    relapse_count=(
                        (0 if previous_record is None else previous_record.relapse_count)
                        + int(relapse_event)
                    ),
                    last_detected_epoch=(
                        None
                        if previous_record is None
                        else previous_record.last_detected_epoch
                    ),
                    last_failure_epoch=epoch,
                    last_failure_type=failure_type,
                    severity=severity,
                    support=previous_support,
                )
                self._persistent_records[gt_uid] = record

                image_entries.append(
                    MDMBPlusEntry(
                        gt_uid=gt_uid,
                        image_id=image_key,
                        class_id=int(gt_label.item()),
                        bbox=gt_box_norm,
                        failure_type=failure_type,
                        consecutive_miss_count=consecutive_miss_count,
                        max_consecutive_miss_count=max_consecutive_miss_count,
                        total_miss_count=total_miss_count,
                        relapse=relapse_flag,
                        severity=severity,
                        best_candidate=best_candidate,
                        topk_candidates=stored_candidates,
                        support=previous_support,
                    )
                )

            image_entries.sort(key=lambda item: (-item.severity, -item.consecutive_miss_count, item.gt_uid))
            if self.config.max_per_image is not None:
                image_entries = image_entries[: self.config.max_per_image]

            if image_entries:
                self._bank[image_key] = image_entries
                for entry in image_entries:
                    new_entry_index[entry.gt_uid] = entry
            else:
                self._bank.pop(image_key, None)

        self._entry_index = new_entry_index
        self._refresh_statistics()

    def get(self, image_id: Any) -> list[MDMBPlusEntry]:
        return list(self._bank.get(_normalize_image_id(image_id), ()))

    def items(self) -> Iterator[tuple[str, list[MDMBPlusEntry]]]:
        for image_id, entries in self._bank.items():
            yield image_id, list(entries)

    def values(self) -> Iterator[list[MDMBPlusEntry]]:
        for entries in self._bank.values():
            yield list(entries)

    def get_image_entries(self, image_id: Any) -> list[MDMBPlusEntry]:
        return self.get(image_id)

    def get_entry(self, gt_uid: Any) -> MDMBPlusEntry | None:
        return self._entry_index.get(str(gt_uid))

    def get_record(self, gt_uid: Any) -> GTFailureRecord | None:
        return self._persistent_records.get(str(gt_uid))

    def get_replay_priority(self, image_id: Any) -> float:
        return float(sum(entry.severity for entry in self.get_image_entries(image_id)))

    def get_dense_targets(self, image_id: Any) -> list[MDMBPlusEntry]:
        entries = self.get_image_entries(image_id)
        return sorted(entries, key=lambda item: (-item.severity, -item.consecutive_miss_count))

    def summary(self) -> dict[str, Any]:
        num_entries = len(self)
        num_images = len(self._bank)
        warmup_active = self.config.enabled and self.current_epoch <= self.config.warmup_epochs

        counts = {
            "candidate_missing": 0,
            "loc_near_miss": 0,
            "cls_confusion": 0,
            "score_suppression": 0,
            "nms_suppression": 0,
        }
        relapse_count = 0
        severity_sum = 0.0
        for entries in self._bank.values():
            for entry in entries:
                if entry.failure_type in counts:
                    counts[entry.failure_type] += 1
                relapse_count += int(entry.relapse)
                severity_sum += entry.severity

        recovery_rate = 0.0
        if self._epoch_recovery_candidates > 0:
            recovery_rate = self._epoch_recovery_success / float(self._epoch_recovery_candidates)

        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_active": warmup_active,
            "num_images": num_images,
            "num_entries": num_entries,
            "num_relapse": relapse_count,
            "num_candidate_missing": counts["candidate_missing"],
            "num_loc_near_miss": counts["loc_near_miss"],
            "num_cls_confusion": counts["cls_confusion"],
            "num_score_suppression": counts["score_suppression"],
            "num_nms_suppression": counts["nms_suppression"],
            "global_max_consecutive_miss": self._global_max_consecutive_miss,
            "mean_severity": severity_sum / float(max(num_entries, 1)),
            "recovery_rate_last_1_epoch": recovery_rate,
            "relapses_this_epoch": self._epoch_relapses,
        }

    def __len__(self) -> int:
        return sum(len(entries) for entries in self._bank.values())

    def extra_repr(self) -> str:
        return (
            f"enabled={self.config.enabled}, arch={self.config.arch!r}, "
            f"images={len(self._bank)}, entries={len(self)}"
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "bank": {
                image_id: [entry.to_state() for entry in entries]
                for image_id, entries in self._bank.items()
            },
            "persistent_records": {
                gt_uid: record.to_state()
                for gt_uid, record in self._persistent_records.items()
            },
            "global_max_consecutive_miss": self._global_max_consecutive_miss,
            "class_statistics": dict(self._class_statistics),
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            self.reset()
            return

        config_state = state.get("config", {})
        if isinstance(config_state, Mapping):
            try:
                self.config = MDMBPlusConfig.from_mapping(config_state, arch=config_state.get("arch"))
            except Exception:
                pass

        self.current_epoch = int(state.get("current_epoch", 0))

        raw_bank = state.get("bank", {})
        restored_bank: dict[str, list[MDMBPlusEntry]] = {}
        restored_index: dict[str, MDMBPlusEntry] = {}
        if isinstance(raw_bank, Mapping):
            for image_id, raw_entries in raw_bank.items():
                if not isinstance(raw_entries, Sequence):
                    continue
                image_key = _normalize_image_id(image_id)
                entries: list[MDMBPlusEntry] = []
                for raw_entry in raw_entries:
                    if not isinstance(raw_entry, Mapping):
                        continue
                    entry = MDMBPlusEntry.from_state(raw_entry)
                    entries.append(entry)
                    restored_index[entry.gt_uid] = entry
                if entries:
                    restored_bank[image_key] = entries
        self._bank = restored_bank
        self._entry_index = restored_index

        raw_records = state.get("persistent_records", {})
        restored_records: dict[str, GTFailureRecord] = {}
        if isinstance(raw_records, Mapping):
            for gt_uid, raw_record in raw_records.items():
                if not isinstance(raw_record, Mapping):
                    continue
                restored_records[str(gt_uid)] = GTFailureRecord.from_state(raw_record)
        self._persistent_records = restored_records
        self._global_max_consecutive_miss = int(state.get("global_max_consecutive_miss", 0))

        raw_class_stats = state.get("class_statistics", {})
        restored_class_stats: dict[str, dict[str, float | int]] = {}
        if isinstance(raw_class_stats, Mapping):
            for class_id, raw_stats in raw_class_stats.items():
                if not isinstance(raw_stats, Mapping):
                    continue
                restored_class_stats[str(class_id)] = {
                    str(key): _coerce_numeric(value)
                    for key, value in raw_stats.items()
                }
        self._class_statistics = restored_class_stats
        self._epoch_recovery_candidates = 0
        self._epoch_recovery_success = 0
        self._epoch_relapses = 0

    def _compute_severity(
        self,
        *,
        consecutive_miss_count: int,
        failure_type: FailureType,
        relapse: bool,
        coverage: float,
    ) -> float:
        global_scale = max(self._global_max_consecutive_miss, consecutive_miss_count, 1)
        streak_term = consecutive_miss_count / float(global_scale)
        failure_prior = float(self.config.failure_type_priors.get(failure_type, 0.0))
        coverage_term = 1.0 - float(max(0.0, min(coverage, 1.0)))
        return float(
            self.config.severity_lambda_streak * streak_term
            + self.config.severity_lambda_relapse * float(relapse)
            + self.config.severity_lambda_failure_type * failure_prior
            + self.config.severity_lambda_coverage * coverage_term
        )

    def _refresh_statistics(self) -> None:
        self._global_max_consecutive_miss = max(
            (record.max_consecutive_miss_count for record in self._persistent_records.values()),
            default=0,
        )

        stats: dict[str, dict[str, float | int]] = {}
        severity_sums: dict[str, float] = {}
        for record in self._persistent_records.values():
            key = str(record.class_id)
            class_stats = stats.setdefault(
                key,
                {
                    "num_records": 0,
                    "num_entries": 0,
                    "total_miss_count": 0,
                    "max_consecutive_miss_count": 0,
                    "mean_severity": 0.0,
                },
            )
            class_stats["num_records"] = int(class_stats["num_records"]) + 1
            class_stats["total_miss_count"] = int(class_stats["total_miss_count"]) + record.total_miss_count
            class_stats["max_consecutive_miss_count"] = max(
                int(class_stats["max_consecutive_miss_count"]),
                record.max_consecutive_miss_count,
            )
            severity_sums[key] = severity_sums.get(key, 0.0) + record.severity

        for entries in self._bank.values():
            for entry in entries:
                key = str(entry.class_id)
                class_stats = stats.setdefault(
                    key,
                    {
                        "num_records": 0,
                        "num_entries": 0,
                        "total_miss_count": 0,
                        "max_consecutive_miss_count": 0,
                        "mean_severity": 0.0,
                    },
                )
                class_stats["num_entries"] = int(class_stats["num_entries"]) + 1

        for key, class_stats in stats.items():
            num_records = int(class_stats["num_records"])
            class_stats["mean_severity"] = severity_sums.get(key, 0.0) / float(max(num_records, 1))

        self._class_statistics = stats


MDMBPP = MDMBPlus


def classify_failure(
    *,
    gt_box: torch.Tensor | Sequence[float],
    gt_label: int,
    final_match: CanonicalCandidate | None,
    candidates: Sequence[CanonicalCandidate],
    near_iou_threshold: float,
    class_iou_threshold: float,
) -> FailureType:
    if final_match is not None and final_match.label == int(gt_label):
        return "detected"

    coverage = _max_candidate_iou(candidates)
    if coverage < float(near_iou_threshold):
        return "candidate_missing"

    same_class_candidates = [candidate for candidate in candidates if candidate.label == int(gt_label)]
    if _max_candidate_iou(same_class_candidates) < float(class_iou_threshold):
        return "cls_confusion"

    selected_same_class = [
        candidate
        for candidate in same_class_candidates
        if candidate.survived_selection and candidate.iou_to_gt >= float(class_iou_threshold)
    ]
    if any(candidate.survived_nms is False for candidate in selected_same_class):
        return "nms_suppression"
    if not selected_same_class and any(
        candidate.iou_to_gt >= float(class_iou_threshold) and not candidate.survived_selection
        for candidate in same_class_candidates
    ):
        return "score_suppression"
    return "loc_near_miss"


def load_mdmbpp_config(path: str | Path, *, arch: str | None = None) -> MDMBPlusConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"MDMB++ YAML must contain a mapping at the top level: {config_path}")
    return MDMBPlusConfig.from_mapping(raw, arch=arch)


def build_mdmbpp_from_config(
    raw_config: Mapping[str, Any] | MDMBPlusConfig,
    *,
    arch: str | None = None,
) -> MDMBPlus | None:
    config = (
        raw_config
        if isinstance(raw_config, MDMBPlusConfig)
        else MDMBPlusConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return MDMBPlus(config)


def build_mdmbpp_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> MDMBPlus | None:
    config = load_mdmbpp_config(path, arch=arch)
    if not config.enabled:
        return None
    return MDMBPlus(config)


def _coerce_candidates(
    candidates: Sequence[CanonicalCandidate | Mapping[str, Any]],
) -> list[CanonicalCandidate]:
    return sorted(
        [
            candidate
            if isinstance(candidate, CanonicalCandidate)
            else CanonicalCandidate.from_state(candidate)
            for candidate in candidates
        ],
        key=lambda item: (
            -item.iou_to_gt,
            -item.score,
            item.rank if item.rank is not None else 10**9,
        ),
    )


def _select_stored_candidates(
    candidates: Sequence[CanonicalCandidate],
    *,
    candidate_topk: int,
    store_topk: bool,
) -> list[CanonicalCandidate]:
    if not candidates or not store_topk:
        return []
    scores = torch.tensor([candidate.score for candidate in candidates], dtype=torch.float32)
    keep = select_topk_indices(scores, k=candidate_topk).tolist()
    selected = [candidates[index] for index in keep]
    selected.sort(
        key=lambda item: (
            -item.iou_to_gt,
            -item.score,
            item.rank if item.rank is not None else 10**9,
        )
    )
    return selected


def _select_best_candidate(candidates: Sequence[CanonicalCandidate]) -> CanonicalCandidate | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.iou_to_gt,
            item.score,
            1 if item.survived_selection else 0,
            0 if item.rank is None else -item.rank,
        ),
    )


def _find_best_final_detection(
    *,
    gt_box: torch.Tensor,
    gt_label: int,
    final_boxes: torch.Tensor,
    final_labels: torch.Tensor,
    final_scores: torch.Tensor,
    detected_iou_threshold: float,
) -> CanonicalCandidate | None:
    if final_boxes.numel() == 0:
        return None

    ious = box_ops.box_iou(gt_box.unsqueeze(0), final_boxes)[0]
    label_mask = final_labels == int(gt_label)
    if not bool(label_mask.any().item()):
        return None

    candidate_indices = torch.nonzero(label_mask, as_tuple=False).flatten()
    best_index = None
    best_key: tuple[float, float] | None = None
    for index in candidate_indices.tolist():
        iou = float(ious[index].item())
        if iou < float(detected_iou_threshold):
            continue
        score = 0.0
        if final_scores.numel() == final_boxes.shape[0]:
            score = float(final_scores[index].item())
        key = (iou, score)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index

    if best_index is None:
        return None

    score = 0.0
    if final_scores.numel() == final_boxes.shape[0]:
        score = float(final_scores[best_index].item())
    return CanonicalCandidate(
        stage="final_detection",
        box=final_boxes[best_index].detach().cpu(),
        score=score,
        label=int(final_labels[best_index].item()),
        iou_to_gt=float(ious[best_index].item()),
        survived_selection=True,
        survived_nms=True,
        rank=best_index,
        level_or_stage_id=None,
    )


def _build_support_snapshot(
    *,
    epoch: int,
    final_match: CanonicalCandidate | None,
    fallback_candidate: CanonicalCandidate | None,
    previous_support: SupportSnapshot | None,
    keep_feature: bool,
) -> SupportSnapshot | None:
    source = final_match or fallback_candidate
    if source is None:
        return previous_support

    feature = None
    if keep_feature and previous_support is not None:
        feature = previous_support.feature
    return SupportSnapshot(
        epoch=epoch,
        box=source.box,
        score=source.score,
        feature=feature,
        feature_level=source.level_or_stage_id,
    )


def _coerce_candidate_summary(
    *,
    image_id: str,
    raw_summary: PerImageCandidateSummary
    | Mapping[int, Sequence[CanonicalCandidate | Mapping[str, Any]]]
    | Sequence[Sequence[CanonicalCandidate | Mapping[str, Any]]]
    | None,
    num_gt: int,
) -> dict[int, list[CanonicalCandidate]]:
    if raw_summary is None:
        return {}

    if isinstance(raw_summary, PerImageCandidateSummary):
        if raw_summary.image_id != image_id:
            raise ValueError(
                "MDMB++ candidate summary image_id does not match update image_id: "
                f"{raw_summary.image_id!r} != {image_id!r}."
            )
        raw_map = raw_summary.candidates_by_gt_index
    elif isinstance(raw_summary, Mapping):
        raw_map = raw_summary
    else:
        raw_map = {index: candidates for index, candidates in enumerate(raw_summary)}

    coerced: dict[int, list[CanonicalCandidate]] = {}
    for raw_gt_index, raw_candidates in raw_map.items():
        gt_index = int(raw_gt_index)
        if gt_index < 0 or gt_index >= num_gt:
            raise IndexError(
                f"MDMB++ candidate summary gt_index={gt_index} is out of range for {num_gt} GTs."
            )
        candidates: list[CanonicalCandidate] = []
        for raw_candidate in raw_candidates:
            if isinstance(raw_candidate, CanonicalCandidate):
                candidate = raw_candidate
            elif isinstance(raw_candidate, Mapping):
                candidate = CanonicalCandidate.from_state(raw_candidate)
            else:
                raise TypeError(
                    "MDMB++ candidates must be CanonicalCandidate instances or mappings."
                )
            candidates.append(candidate)
        coerced[gt_index] = candidates
    return coerced


def _coerce_gt_ids(
    value: torch.Tensor | Sequence[Any] | Any,
    *,
    count: int,
) -> list[Any | None]:
    if value is None:
        return [None] * count
    if isinstance(value, torch.Tensor):
        flattened = value.detach().cpu().flatten().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        flattened = list(value)
    else:
        flattened = [value]
    if count == 1 and len(flattened) == 1:
        return flattened
    if len(flattened) != count:
        raise ValueError(
            f"MDMB++ gt_ids must align with GT count. Got {len(flattened)} ids for {count} GTs."
        )
    return flattened


def _make_gt_uid(
    *,
    image_id: str,
    class_id: int,
    bbox: torch.Tensor,
    gt_id: Any | None,
) -> str:
    if gt_id is not None:
        if isinstance(gt_id, torch.Tensor):
            if gt_id.numel() != 1:
                raise ValueError("MDMB++ gt_id tensor must contain a single scalar value.")
            gt_id = gt_id.item()
        return f"ann:{gt_id}"

    coords = ",".join(f"{float(value):.6f}" for value in bbox.tolist())
    digest = hashlib.sha1(f"{image_id}|{class_id}|{coords}".encode("utf-8")).hexdigest()[:16]
    return f"{image_id}:{class_id}:{digest}"


def _max_candidate_iou(candidates: Sequence[CanonicalCandidate]) -> float:
    if not candidates:
        return 0.0
    return max(float(candidate.iou_to_gt) for candidate in candidates)


def _coerce_failure_type(value: Any) -> FailureType:
    candidate = str(value)
    if candidate not in _FAILURE_TYPES:
        raise ValueError(
            f"Unsupported MDMB++ failure type {candidate!r}. Supported: {sorted(_FAILURE_TYPES)}."
        )
    return candidate  # type: ignore[return-value]


def _coerce_numeric(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return float(value)


def _as_box_tensor(
    value: torch.Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).detach()
    if tensor.numel() == 0:
        return tensor.reshape(-1, 4)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != 4:
        raise ValueError("MDMB++ boxes must have shape [N, 4] or [4].")
    return tensor


def _as_region_tensor(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().flatten()
    if tensor.numel() != 4:
        raise ValueError("MDMB++ bbox must contain exactly four values.")
    return tensor.cpu()


def _as_label_tensor(
    value: torch.Tensor | Sequence[int] | int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.int64, device=device).detach().flatten()


def _as_score_tensor(
    value: torch.Tensor | Sequence[float] | float | None,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if value is None:
        return torch.empty((0,), dtype=torch.float32, device=device)
    return torch.as_tensor(value, dtype=torch.float32, device=device).detach().flatten()


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("MDMB++ image_id tensor must contain a single scalar value.")
        value = value.item()
    return str(value)
