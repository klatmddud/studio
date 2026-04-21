from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.ops import boxes as box_ops

from .mdmb import normalize_arch, normalize_xyxy_boxes
from .mdmbpp import MDMBPlus


_DEFAULT_FAILURE_TYPES: tuple[str, ...] = (
    "score_suppression",
    "nms_suppression",
    "cls_confusion",
    "loc_near_miss",
)

_SUPPORTED_FAILURE_TYPES = set(_DEFAULT_FAILURE_TYPES) | {"candidate_missing"}


@dataclass(frozen=True, slots=True)
class MARCConfig:
    enabled: bool = False
    warmup_epochs: int = 2
    lambda_rank: float = 0.05
    temperature: float = 0.1
    min_severity: float = 1.0
    record_match_threshold: float = 0.95
    max_rank_targets_per_image: int = 5
    max_negatives_per_gt: int = 8
    positive_iou_threshold: float = 0.3
    near_positive_iou_threshold: float = 0.1
    confuser_iou_threshold: float = 0.3
    same_class_iou_gap: float = 0.1
    region_scale: float = 1.5
    allow_near_positive_fallback: bool = True
    severity_weight_scale: float = 0.25
    max_target_weight: float = 3.0
    failure_types: tuple[str, ...] = _DEFAULT_FAILURE_TYPES
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "MARCConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))

        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        def read(name: str, default: Any) -> Any:
            return model_overrides.get(name, data.get(name, default))

        raw_failure_types = read("failure_types", _DEFAULT_FAILURE_TYPES)
        if isinstance(raw_failure_types, str):
            failure_types = (raw_failure_types,)
        else:
            failure_types = tuple(str(value) for value in raw_failure_types)

        config = cls(
            enabled=bool(read("enabled", False)),
            warmup_epochs=int(read("warmup_epochs", 2)),
            lambda_rank=float(read("lambda_rank", 0.05)),
            temperature=float(read("temperature", 0.1)),
            min_severity=float(read("min_severity", 1.0)),
            record_match_threshold=float(read("record_match_threshold", 0.95)),
            max_rank_targets_per_image=int(read("max_rank_targets_per_image", 5)),
            max_negatives_per_gt=int(read("max_negatives_per_gt", 8)),
            positive_iou_threshold=float(read("positive_iou_threshold", 0.3)),
            near_positive_iou_threshold=float(read("near_positive_iou_threshold", 0.1)),
            confuser_iou_threshold=float(read("confuser_iou_threshold", 0.3)),
            same_class_iou_gap=float(read("same_class_iou_gap", 0.1)),
            region_scale=float(read("region_scale", 1.5)),
            allow_near_positive_fallback=bool(read("allow_near_positive_fallback", True)),
            severity_weight_scale=float(read("severity_weight_scale", 0.25)),
            max_target_weight=float(read("max_target_weight", 3.0)),
            failure_types=failure_types,
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("MARC warmup_epochs must be >= 0.")
        if self.lambda_rank < 0.0:
            raise ValueError("MARC lambda_rank must be >= 0.")
        if self.temperature <= 0.0:
            raise ValueError("MARC temperature must be > 0.")
        if self.min_severity < 0.0:
            raise ValueError("MARC min_severity must be >= 0.")
        for name in (
            "record_match_threshold",
            "positive_iou_threshold",
            "near_positive_iou_threshold",
            "confuser_iou_threshold",
            "same_class_iou_gap",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"MARC {name} must satisfy 0 <= value <= 1.")
        if self.max_rank_targets_per_image < 1:
            raise ValueError("MARC max_rank_targets_per_image must be >= 1.")
        if self.max_negatives_per_gt < 1:
            raise ValueError("MARC max_negatives_per_gt must be >= 1.")
        if self.region_scale <= 0.0:
            raise ValueError("MARC region_scale must be > 0.")
        if self.severity_weight_scale < 0.0:
            raise ValueError("MARC severity_weight_scale must be >= 0.")
        if self.max_target_weight < 1.0:
            raise ValueError("MARC max_target_weight must be >= 1.")
        if not self.failure_types:
            raise ValueError("MARC failure_types must not be empty.")
        unsupported = sorted(set(self.failure_types) - _SUPPORTED_FAILURE_TYPES)
        if unsupported:
            raise ValueError(f"MARC failure_types contains unsupported values: {unsupported}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_epochs": self.warmup_epochs,
            "lambda_rank": self.lambda_rank,
            "temperature": self.temperature,
            "min_severity": self.min_severity,
            "record_match_threshold": self.record_match_threshold,
            "max_rank_targets_per_image": self.max_rank_targets_per_image,
            "max_negatives_per_gt": self.max_negatives_per_gt,
            "positive_iou_threshold": self.positive_iou_threshold,
            "near_positive_iou_threshold": self.near_positive_iou_threshold,
            "confuser_iou_threshold": self.confuser_iou_threshold,
            "same_class_iou_gap": self.same_class_iou_gap,
            "region_scale": self.region_scale,
            "allow_near_positive_fallback": self.allow_near_positive_fallback,
            "severity_weight_scale": self.severity_weight_scale,
            "max_target_weight": self.max_target_weight,
            "failure_types": list(self.failure_types),
            "arch": self.arch,
        }


@dataclass(slots=True)
class RankingTarget:
    gt_uid: str
    image_id: str
    gt_index: int
    class_id: int
    bbox: torch.Tensor
    failure_type: str
    severity: float
    weight: float

    def __post_init__(self) -> None:
        self.gt_uid = str(self.gt_uid)
        self.image_id = str(self.image_id)
        self.gt_index = int(self.gt_index)
        self.class_id = int(self.class_id)
        self.bbox = torch.as_tensor(self.bbox, dtype=torch.float32).detach()
        if self.bbox.numel() != 4:
            raise ValueError("RankingTarget bbox must contain exactly four values.")
        self.bbox = self.bbox.reshape(4)
        self.failure_type = str(self.failure_type)
        self.severity = float(self.severity)
        self.weight = float(self.weight)


@dataclass(slots=True)
class RankingPlan:
    targets: list[RankingTarget]

    def for_image(self, image_id: Any) -> list[RankingTarget]:
        image_key = _normalize_image_id(image_id)
        return [target for target in self.targets if target.image_id == image_key]

    def __len__(self) -> int:
        return len(self.targets)


class MissAwareRankingCalibration(nn.Module):
    def __init__(self, config: MARCConfig) -> None:
        super().__init__()
        self.config = config
        self.current_epoch = 0
        self._epoch_rank_targets = 0
        self._epoch_rank_losses = 0
        self._epoch_rank_negatives = 0
        self._epoch_rank_loss_sum = 0.0
        self._epoch_failure_counts: Counter[str] = Counter()
        self._epoch_skipped_no_entry_match = 0
        self._epoch_skipped_no_positive = 0
        self._epoch_skipped_no_negative = 0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._epoch_rank_targets = 0
        self._epoch_rank_losses = 0
        self._epoch_rank_negatives = 0
        self._epoch_rank_loss_sum = 0.0
        self._epoch_failure_counts = Counter()
        self._epoch_skipped_no_entry_match = 0
        self._epoch_skipped_no_positive = 0
        self._epoch_skipped_no_negative = 0

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def warmup_factor(self) -> float:
        if self.config.warmup_epochs <= 0:
            return 1.0
        if self.current_epoch <= 0:
            return 0.0
        return min(1.0, float(self.current_epoch) / float(self.config.warmup_epochs))

    def loss_weight(self) -> float:
        if not self.config.enabled:
            return 0.0
        return float(self.config.lambda_rank) * self.warmup_factor()

    def should_apply(self, *, mdmbpp: MDMBPlus | None) -> bool:
        return mdmbpp is not None and self.loss_weight() > 0.0

    @torch.no_grad()
    def plan(
        self,
        *,
        mdmbpp: MDMBPlus | None,
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_shapes: Sequence[Sequence[int]],
    ) -> RankingPlan:
        if mdmbpp is None or not self.config.enabled:
            return RankingPlan(targets=[])

        ranking_targets: list[RankingTarget] = []
        failure_types = set(self.config.failure_types)

        for image_index, target in enumerate(targets):
            image_id = target.get("image_id", torch.tensor(image_index))
            image_key = _normalize_image_id(image_id)
            entries = [
                entry
                for entry in mdmbpp.get_dense_targets(image_key)
                if entry.failure_type in failure_types
                and float(entry.severity) >= self.config.min_severity
            ]
            if not entries:
                continue

            gt_boxes = target["boxes"]
            gt_labels = target["labels"].to(device=gt_boxes.device, dtype=torch.int64)
            if gt_boxes.numel() == 0:
                continue

            gt_boxes_norm = normalize_xyxy_boxes(gt_boxes, image_shapes[image_index]).to(
                device=gt_boxes.device,
                dtype=torch.float32,
            )
            used_gt_indices: set[int] = set()
            image_targets: list[RankingTarget] = []

            for entry in entries:
                match = self._match_entry_to_target(
                    entry=entry,
                    gt_boxes=gt_boxes,
                    gt_boxes_norm=gt_boxes_norm,
                    gt_labels=gt_labels,
                    used_gt_indices=used_gt_indices,
                )
                if match is None:
                    self._epoch_skipped_no_entry_match += 1
                    continue

                gt_index = int(match)
                used_gt_indices.add(gt_index)
                image_targets.append(
                    RankingTarget(
                        gt_uid=entry.gt_uid,
                        image_id=image_key,
                        gt_index=gt_index,
                        class_id=entry.class_id,
                        bbox=gt_boxes[gt_index].detach(),
                        failure_type=entry.failure_type,
                        severity=entry.severity,
                        weight=self.target_weight_for(entry.severity),
                    )
                )
                if len(image_targets) >= self.config.max_rank_targets_per_image:
                    break

            ranking_targets.extend(image_targets)

        return RankingPlan(targets=ranking_targets)

    def compute_loss(
        self,
        *,
        ranking_plan: RankingPlan,
        targets: Sequence[Mapping[str, torch.Tensor]],
        head_outputs: Mapping[str, torch.Tensor],
        anchors: Sequence[torch.Tensor],
        image_shapes: Sequence[Sequence[int]],
        decode_boxes_fn: Callable[..., torch.Tensor],
    ) -> tuple[torch.Tensor, int, int, int, int]:
        cls_template = head_outputs["cls_logits"]
        rank_sum = cls_template.new_zeros(())
        rank_losses = 0
        rank_negatives = 0
        skipped_no_positive = 0
        skipped_no_negative = 0

        for image_index, target in enumerate(targets):
            image_id = target.get(
                "image_id",
                torch.tensor(image_index, device=anchors[image_index].device),
            )
            ranking_targets = ranking_plan.for_image(image_id)
            if not ranking_targets:
                continue

            cls_logits = head_outputs["cls_logits"][image_index]
            bbox_regression = head_outputs["bbox_regression"][image_index]
            bbox_ctrness = head_outputs["bbox_ctrness"][image_index].flatten()
            anchors_per_image = anchors[image_index]
            if cls_logits.numel() == 0 or anchors_per_image.numel() == 0:
                skipped_no_positive += len(ranking_targets)
                continue

            decoded_boxes = decode_boxes_fn(
                box_regression=bbox_regression,
                anchors=anchors_per_image,
            )
            decoded_boxes = box_ops.clip_boxes_to_image(decoded_boxes, image_shapes[image_index])
            gt_boxes = target["boxes"].to(device=decoded_boxes.device, dtype=decoded_boxes.dtype)
            if gt_boxes.numel() == 0:
                skipped_no_positive += len(ranking_targets)
                continue

            rank_scores = self._ranking_scores(
                cls_logits=cls_logits,
                bbox_ctrness=bbox_ctrness,
            )
            num_classes = int(rank_scores.shape[1])
            centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) * 0.5

            for ranking_target in ranking_targets:
                if ranking_target.gt_index < 0 or ranking_target.gt_index >= gt_boxes.shape[0]:
                    skipped_no_positive += 1
                    continue
                if ranking_target.class_id < 0 or ranking_target.class_id >= num_classes:
                    skipped_no_positive += 1
                    continue

                gt_box = gt_boxes[int(ranking_target.gt_index)]
                ious = box_ops.box_iou(gt_box.reshape(1, 4), decoded_boxes)[0]
                positive_index = self._select_positive_index(
                    ious=ious,
                    class_scores=rank_scores[:, int(ranking_target.class_id)],
                )
                if positive_index is None:
                    skipped_no_positive += 1
                    continue

                pos_score = rank_scores[positive_index, int(ranking_target.class_id)]
                pos_iou = ious[positive_index]
                region_mask = self._region_mask(
                    gt_box=gt_box,
                    centers=centers,
                    scale=float(self.config.region_scale),
                )
                negative_scores = self._select_negative_scores(
                    rank_scores=rank_scores,
                    ious=ious,
                    region_mask=region_mask,
                    class_id=int(ranking_target.class_id),
                    positive_index=int(positive_index),
                    positive_score=pos_score,
                    positive_iou=pos_iou,
                )
                if negative_scores.numel() == 0:
                    skipped_no_negative += 1
                    continue

                logits = torch.cat([pos_score.reshape(1), negative_scores], dim=0)
                logits = logits / float(self.config.temperature)
                target_index = torch.zeros((1,), dtype=torch.long, device=logits.device)
                rank_loss = F.cross_entropy(logits.reshape(1, -1), target_index)
                rank_loss = rank_loss * float(ranking_target.weight)
                rank_sum = rank_sum + rank_loss
                rank_losses += 1
                rank_negatives += int(negative_scores.numel())

        if rank_losses <= 0:
            return rank_sum, 0, 0, skipped_no_positive, skipped_no_negative

        loss = rank_sum / rank_sum.new_tensor(float(rank_losses))
        return (
            loss * float(self.loss_weight()),
            rank_losses,
            rank_negatives,
            skipped_no_positive,
            skipped_no_negative,
        )

    def target_weight_for(self, severity: float) -> float:
        weight = 1.0 + float(self.config.severity_weight_scale) * float(severity)
        return min(float(self.config.max_target_weight), max(1.0, weight))

    def record_rank_step(
        self,
        *,
        ranking_plan: RankingPlan,
        rank_losses: int,
        rank_negatives: int,
        loss: torch.Tensor | None,
        skipped_no_positive: int = 0,
        skipped_no_negative: int = 0,
    ) -> None:
        if len(ranking_plan) > 0:
            self._epoch_rank_targets += len(ranking_plan)
            self._epoch_failure_counts.update(target.failure_type for target in ranking_plan.targets)
        self._epoch_rank_losses += int(rank_losses)
        self._epoch_rank_negatives += int(rank_negatives)
        self._epoch_skipped_no_positive += int(skipped_no_positive)
        self._epoch_skipped_no_negative += int(skipped_no_negative)
        if loss is not None and int(rank_losses) > 0:
            self._epoch_rank_loss_sum += float(loss.detach().cpu().item()) * float(rank_losses)

    def summary(self) -> dict[str, Any]:
        mean_loss = self._epoch_rank_loss_sum / float(max(self._epoch_rank_losses, 1))
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_factor": self.warmup_factor(),
            "lambda_rank": self.loss_weight(),
            "rank_targets": self._epoch_rank_targets,
            "rank_losses": self._epoch_rank_losses,
            "rank_negatives": self._epoch_rank_negatives,
            "mean_rank_loss": mean_loss,
            "by_failure_score_suppression": int(
                self._epoch_failure_counts.get("score_suppression", 0)
            ),
            "by_failure_nms_suppression": int(
                self._epoch_failure_counts.get("nms_suppression", 0)
            ),
            "by_failure_cls_confusion": int(self._epoch_failure_counts.get("cls_confusion", 0)),
            "by_failure_loc_near_miss": int(self._epoch_failure_counts.get("loc_near_miss", 0)),
            "skipped_no_entry_match": self._epoch_skipped_no_entry_match,
            "skipped_no_positive": self._epoch_skipped_no_positive,
            "skipped_no_negative": self._epoch_skipped_no_negative,
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            return
        self.current_epoch = int(state.get("current_epoch", self.current_epoch))

    def _ranking_scores(
        self,
        *,
        cls_logits: torch.Tensor,
        bbox_ctrness: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            F.logsigmoid(cls_logits) + F.logsigmoid(bbox_ctrness).unsqueeze(-1)
        )

    def _select_positive_index(
        self,
        *,
        ious: torch.Tensor,
        class_scores: torch.Tensor,
    ) -> int | None:
        pos_mask = ious >= float(self.config.positive_iou_threshold)
        if bool(pos_mask.any().item()):
            pos_indices = torch.nonzero(pos_mask, as_tuple=False).flatten()
            best = int(class_scores[pos_indices].argmax().item())
            return int(pos_indices[best].item())

        if not self.config.allow_near_positive_fallback:
            return None
        near_mask = ious >= float(self.config.near_positive_iou_threshold)
        if not bool(near_mask.any().item()):
            return None
        near_indices = torch.nonzero(near_mask, as_tuple=False).flatten()
        best = int(ious[near_indices].argmax().item())
        return int(near_indices[best].item())

    def _select_negative_scores(
        self,
        *,
        rank_scores: torch.Tensor,
        ious: torch.Tensor,
        region_mask: torch.Tensor,
        class_id: int,
        positive_index: int,
        positive_score: torch.Tensor,
        positive_iou: torch.Tensor,
    ) -> torch.Tensor:
        region_indices = torch.nonzero(region_mask, as_tuple=False).flatten()
        if region_indices.numel() == 0:
            return rank_scores.new_zeros((0,))

        num_classes = int(rank_scores.shape[1])
        region_scores = rank_scores[region_indices]
        class_ids = torch.arange(num_classes, device=rank_scores.device).view(1, -1)
        class_ids = class_ids.expand(region_indices.numel(), num_classes)
        region_ious = ious[region_indices].view(-1, 1).expand_as(region_scores)
        point_ids = region_indices.view(-1, 1).expand_as(region_scores)

        wrong_class = class_ids != int(class_id)
        wrong_confuser = wrong_class & (region_ious >= float(self.config.confuser_iou_threshold))
        local_distractor = wrong_class & (region_scores >= positive_score.detach())
        same_class_suppressor = (
            (class_ids == int(class_id))
            & (region_ious <= positive_iou - float(self.config.same_class_iou_gap))
        )
        negative_mask = wrong_confuser | local_distractor | same_class_suppressor
        negative_mask &= ~((point_ids == int(positive_index)) & (class_ids == int(class_id)))
        if not bool(negative_mask.any().item()):
            return rank_scores.new_zeros((0,))

        negative_scores = region_scores[negative_mask]
        count = min(int(self.config.max_negatives_per_gt), int(negative_scores.numel()))
        if count <= 0:
            return rank_scores.new_zeros((0,))
        if negative_scores.numel() > count:
            negative_scores, _ = torch.topk(negative_scores, k=count)
        return negative_scores

    def _region_mask(
        self,
        *,
        gt_box: torch.Tensor,
        centers: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        gt_center = (gt_box[:2] + gt_box[2:]) * 0.5
        gt_size = (gt_box[2:] - gt_box[:2]).clamp(min=1.0)
        half_size = (gt_size * float(scale) * 0.5).clamp(min=1.0)
        region_min = gt_center - half_size
        region_max = gt_center + half_size
        return (
            (centers[:, 0] >= region_min[0])
            & (centers[:, 0] <= region_max[0])
            & (centers[:, 1] >= region_min[1])
            & (centers[:, 1] <= region_max[1])
        )

    def _match_entry_to_target(
        self,
        *,
        entry,
        gt_boxes: torch.Tensor,
        gt_boxes_norm: torch.Tensor,
        gt_labels: torch.Tensor,
        used_gt_indices: set[int],
    ) -> int | None:
        class_mask = gt_labels == int(entry.class_id)
        if used_gt_indices:
            used = torch.tensor(
                sorted(used_gt_indices),
                dtype=torch.long,
                device=gt_labels.device,
            )
            class_mask[used] = False
        if not bool(class_mask.any().item()):
            return None

        candidate_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
        entry_box = entry.bbox.to(device=gt_boxes.device, dtype=gt_boxes_norm.dtype).reshape(1, 4)
        ious = box_ops.box_iou(entry_box, gt_boxes_norm[candidate_indices])[0]
        best_pos = int(ious.argmax().item())
        best_iou = float(ious[best_pos].item())
        if best_iou < self.config.record_match_threshold:
            return None
        return int(candidate_indices[best_pos].item())


def load_marc_config(
    path: str | Path,
    *,
    arch: str | None = None,
) -> MARCConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"MARC YAML must contain a mapping at the top level: {config_path}")
    return MARCConfig.from_mapping(raw, arch=arch)


def build_marc_from_config(
    raw_config: Mapping[str, Any] | MARCConfig,
    *,
    arch: str | None = None,
) -> MissAwareRankingCalibration | None:
    config = (
        raw_config
        if isinstance(raw_config, MARCConfig)
        else MARCConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return MissAwareRankingCalibration(config)


def build_marc_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> MissAwareRankingCalibration | None:
    config = load_marc_config(path, arch=arch)
    if not config.enabled:
        return None
    return MissAwareRankingCalibration(config)


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return str(value.detach().cpu().flatten()[0].item())
    return str(value)
