from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .common import normalize_arch


@dataclass(frozen=True)
class MPDConfig:
    enabled: bool = False
    arch: str | None = None
    start_epoch: int = 18
    radius: float = 1.0
    radius_mode: str = "expand"
    require_current_missed: bool = True
    require_inside_gt: bool = True
    respect_scale_range: bool = True
    override_existing_positive: bool = False

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        arch: str | None = None,
    ) -> "MPDConfig":
        merged = _merge_model_overrides(raw or {}, arch=arch)
        mpd = dict(merged.get("mpd") or {})
        config = cls(
            enabled=bool(mpd.get("enabled", False)),
            arch=normalize_arch(arch or merged.get("arch")) if (arch or merged.get("arch")) else None,
            start_epoch=int(mpd.get("start_epoch", 18)),
            radius=float(mpd.get("radius", 1.0)),
            radius_mode=str(mpd.get("radius_mode", "expand")).lower(),
            require_current_missed=bool(mpd.get("require_current_missed", True)),
            require_inside_gt=bool(mpd.get("require_inside_gt", True)),
            respect_scale_range=bool(mpd.get("respect_scale_range", True)),
            override_existing_positive=bool(mpd.get("override_existing_positive", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.start_epoch) < 0:
            raise ValueError("MPD mpd.start_epoch must be >= 0.")
        if float(self.radius) < 0.0:
            raise ValueError("MPD mpd.radius must be >= 0.")
        if self.radius_mode not in {"expand", "absolute"}:
            raise ValueError("MPD mpd.radius_mode must be either 'expand' or 'absolute'.")


@dataclass(frozen=True)
class MPDMatchResult:
    matched_idxs: list[torch.Tensor]
    metrics: dict[str, float]


class MPD(nn.Module):
    """Miss-Guided Positive Densification for FCOS target assignment."""

    def __init__(self, config: MPDConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, MPDConfig)
            else MPDConfig.from_mapping(config or {})
        )

    def is_active(self, epoch: int | None = None) -> bool:
        epoch_value = 0 if epoch is None else int(epoch)
        return bool(self.config.enabled and epoch_value >= int(self.config.start_epoch))

    def build_fcos_matched_idxs(
        self,
        *,
        targets: Sequence[Mapping[str, Any]],
        anchors: Sequence[torch.Tensor],
        num_anchors_per_level: Sequence[int],
        missbank: Any,
        center_sampling_radius: float,
    ) -> MPDMatchResult:
        matched_idxs: list[torch.Tensor] = []
        stats: Counter[str] = Counter()
        for anchors_per_image, targets_per_image in zip(anchors, targets, strict=True):
            result = self._match_single_image(
                targets_per_image=targets_per_image,
                anchors_per_image=anchors_per_image,
                num_anchors_per_level=num_anchors_per_level,
                missbank=missbank,
                center_sampling_radius=float(center_sampling_radius),
            )
            matched_idxs.append(result.matched_idx)
            stats.update(result.stats)
        return MPDMatchResult(matched_idxs=matched_idxs, metrics=_stats_to_metrics(stats))

    def _match_single_image(
        self,
        *,
        targets_per_image: Mapping[str, Any],
        anchors_per_image: torch.Tensor,
        num_anchors_per_level: Sequence[int],
        missbank: Any,
        center_sampling_radius: float,
    ) -> "_SingleImageMatch":
        gt_boxes = targets_per_image["boxes"]
        if gt_boxes.numel() == 0:
            matched_idx = torch.full(
                (anchors_per_image.size(0),),
                -1,
                dtype=torch.int64,
                device=anchors_per_image.device,
            )
            return _SingleImageMatch(matched_idx=matched_idx, stats=Counter(images=1, images_without_gt=1))

        match_data = _fcos_match_data(
            gt_boxes=gt_boxes,
            anchors_per_image=anchors_per_image,
            num_anchors_per_level=num_anchors_per_level,
            center_sampling_radius=center_sampling_radius,
        )
        matched_idx = _select_smallest_area_match(match_data.base_pairwise_match, match_data.gt_areas)
        stats: Counter[str] = Counter(images=1)
        stats["base_positive_count"] += int((matched_idx >= 0).sum().item())

        hard_gt_indices = self._hard_gt_indices(
            targets_per_image=targets_per_image,
            missbank=missbank,
        )
        stats["target_gt_count"] += len(hard_gt_indices)
        if hard_gt_indices:
            stats["images_with_target"] += 1
        if not hard_gt_indices:
            stats["final_positive_count"] += int((matched_idx >= 0).sum().item())
            return _SingleImageMatch(matched_idx=matched_idx, stats=stats)

        hard_indices = torch.tensor(hard_gt_indices, dtype=torch.int64, device=gt_boxes.device)
        extra_pairwise_match = self._extra_pairwise_match(
            hard_indices=hard_indices,
            match_data=match_data,
            center_sampling_radius=center_sampling_radius,
        )
        stats["candidate_positive_count"] += int(extra_pairwise_match.sum().item())
        if not bool(self.config.override_existing_positive):
            existing_positive = matched_idx >= 0
            if bool(existing_positive.any().item()):
                stats["existing_positive_candidate_count"] += int(extra_pairwise_match[existing_positive].sum().item())
                extra_pairwise_match = extra_pairwise_match.clone()
                extra_pairwise_match[existing_positive] = False

        if not bool(extra_pairwise_match.any().item()):
            stats["final_positive_count"] += int((matched_idx >= 0).sum().item())
            return _SingleImageMatch(matched_idx=matched_idx, stats=stats)

        selected_hard_idx = _select_smallest_area_hard_match(
            extra_pairwise_match,
            hard_indices=hard_indices,
            gt_areas=match_data.gt_areas,
        )
        assign_mask = selected_hard_idx >= 0
        before_positive = matched_idx >= 0
        densified = matched_idx.clone()
        densified[assign_mask] = selected_hard_idx[assign_mask]

        after_positive = densified >= 0
        added_mask = after_positive & ~before_positive
        reassigned_mask = assign_mask & before_positive & (densified != matched_idx)
        stats["added_positive_count"] += int(added_mask.sum().item())
        stats["reassigned_positive_count"] += int(reassigned_mask.sum().item())
        stats["final_positive_count"] += int(after_positive.sum().item())

        covered_gt_count = 0
        for gt_index in hard_gt_indices:
            was_added = bool((added_mask & (densified == int(gt_index))).any().item())
            was_reassigned = bool((reassigned_mask & (densified == int(gt_index))).any().item())
            if was_added or was_reassigned:
                covered_gt_count += 1
        stats["covered_target_gt_count"] += covered_gt_count
        return _SingleImageMatch(matched_idx=densified, stats=stats)

    def _hard_gt_indices(
        self,
        *,
        targets_per_image: Mapping[str, Any],
        missbank: Any,
    ) -> list[int]:
        get_records = getattr(missbank, "get_records", None)
        if not callable(get_records):
            return []
        image_id = targets_per_image.get("image_id")
        id_to_index = _target_id_to_index(targets_per_image)
        labels = targets_per_image.get("labels")
        threshold = int(missbank.config.target.miss_threshold)
        hard_indices: set[int] = set()
        for record in get_records(image_id):
            if bool(self.config.require_current_missed) and not bool(getattr(record, "is_missed", False)):
                continue
            if int(getattr(record, "miss_count", 0)) < threshold:
                continue
            gt_id = _normalize_optional_id(getattr(record, "gt_id", None))
            gt_index = id_to_index.get(gt_id) if gt_id is not None else None
            if gt_index is None:
                continue
            if labels is not None and int(labels[gt_index].item()) != int(getattr(record, "gt_class", -1)):
                continue
            hard_indices.add(int(gt_index))
        return sorted(hard_indices)

    def _extra_pairwise_match(
        self,
        *,
        hard_indices: torch.Tensor,
        match_data: "_FCOSMatchData",
        center_sampling_radius: float,
    ) -> torch.Tensor:
        hard_centers = match_data.gt_centers[hard_indices]
        max_center_delta = (
            match_data.anchor_centers[:, None, :] - hard_centers[None, :, :]
        ).abs().max(dim=2).values
        center_radius = self._effective_center_radius(center_sampling_radius)
        extra_pairwise_match = max_center_delta < center_radius * match_data.anchor_sizes[:, None]

        if bool(self.config.require_inside_gt):
            extra_pairwise_match &= match_data.inside_gt[:, hard_indices]
        if bool(self.config.respect_scale_range):
            extra_pairwise_match &= match_data.scale_match[:, hard_indices]
        return extra_pairwise_match

    def _effective_center_radius(self, center_sampling_radius: float) -> float:
        if self.config.radius_mode == "expand":
            return float(center_sampling_radius) + float(self.config.radius)
        return float(self.config.radius) + 0.5


@dataclass(frozen=True)
class _SingleImageMatch:
    matched_idx: torch.Tensor
    stats: Counter[str]


@dataclass(frozen=True)
class _FCOSMatchData:
    gt_centers: torch.Tensor
    gt_areas: torch.Tensor
    anchor_centers: torch.Tensor
    anchor_sizes: torch.Tensor
    inside_gt: torch.Tensor
    scale_match: torch.Tensor
    base_pairwise_match: torch.Tensor


def load_mpd_config(path: str | Path, *, arch: str | None = None) -> MPDConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"MPD YAML must contain a mapping at the top level: {config_path}")
    return MPDConfig.from_mapping(raw, arch=arch)


def build_mpd_from_config(
    raw_config: Mapping[str, Any] | MPDConfig,
    *,
    arch: str | None = None,
    missbank_enabled: bool = True,
) -> MPD | None:
    config = raw_config if isinstance(raw_config, MPDConfig) else MPDConfig.from_mapping(raw_config, arch=arch)
    if not config.enabled:
        return None
    if not missbank_enabled:
        raise ValueError("MPD requires its MissBank to be enabled.")
    return MPD(config)


def build_mpd_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
    missbank_enabled: bool = True,
) -> MPD | None:
    config = load_mpd_config(path, arch=arch)
    return build_mpd_from_config(config, arch=arch, missbank_enabled=missbank_enabled)


def _fcos_match_data(
    *,
    gt_boxes: torch.Tensor,
    anchors_per_image: torch.Tensor,
    num_anchors_per_level: Sequence[int],
    center_sampling_radius: float,
) -> _FCOSMatchData:
    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2
    anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]

    center_match = (
        (anchor_centers[:, None, :] - gt_centers[None, :, :])
        .abs()
        .max(dim=2)
        .values
        < float(center_sampling_radius) * anchor_sizes[:, None]
    )

    x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)
    x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)
    pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
    inside_gt = pairwise_dist.min(dim=2).values > 0

    lower_bound = anchor_sizes * 4
    if num_anchors_per_level:
        lower_bound[: int(num_anchors_per_level[0])] = 0
    upper_bound = anchor_sizes * 8
    if num_anchors_per_level:
        upper_bound[-int(num_anchors_per_level[-1]) :] = float("inf")
    pairwise_extent = pairwise_dist.max(dim=2).values
    scale_match = (pairwise_extent > lower_bound[:, None]) & (pairwise_extent < upper_bound[:, None])
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    return _FCOSMatchData(
        gt_centers=gt_centers,
        gt_areas=gt_areas,
        anchor_centers=anchor_centers,
        anchor_sizes=anchor_sizes,
        inside_gt=inside_gt,
        scale_match=scale_match,
        base_pairwise_match=center_match & inside_gt & scale_match,
    )


def _select_smallest_area_match(
    pairwise_match: torch.Tensor,
    gt_areas: torch.Tensor,
) -> torch.Tensor:
    scores = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
    min_values, matched_idx = scores.max(dim=1)
    matched_idx[min_values < 1e-5] = -1
    return matched_idx


def _select_smallest_area_hard_match(
    pairwise_match: torch.Tensor,
    *,
    hard_indices: torch.Tensor,
    gt_areas: torch.Tensor,
) -> torch.Tensor:
    hard_areas = gt_areas[hard_indices]
    scores = pairwise_match.to(torch.float32) * (1e8 - hard_areas[None, :])
    min_values, selected_col = scores.max(dim=1)
    selected = hard_indices[selected_col]
    selected = selected.clone()
    selected[min_values < 1e-5] = -1
    return selected


def _target_id_to_index(target: Mapping[str, Any]) -> dict[str, int]:
    count = int(target["boxes"].shape[0])
    for key in ("gt_ids", "annotation_ids", "ann_ids"):
        if key not in target:
            continue
        raw_ids = target[key]
        if isinstance(raw_ids, torch.Tensor):
            flat = raw_ids.detach().flatten().tolist()
        elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
            flat = list(raw_ids)
        else:
            continue
        if len(flat) != count:
            continue
        mapping: dict[str, int] = {}
        for index, raw_id in enumerate(flat):
            normalized = _normalize_optional_id(raw_id)
            if normalized is not None:
                mapping[normalized] = int(index)
        if mapping:
            return mapping
    return {}


def _normalize_optional_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value)
    if text in {"", "-1", "None", "none", "nan"}:
        return None
    return text


def _stats_to_metrics(stats: Counter[str]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "mpd_target_gt_count": float(stats.get("target_gt_count", 0)),
        "mpd_base_positive_count": float(stats.get("base_positive_count", 0)),
        "mpd_candidate_positive_count": float(stats.get("candidate_positive_count", 0)),
        "mpd_added_positive_count": float(stats.get("added_positive_count", 0)),
        "mpd_reassigned_positive_count": float(stats.get("reassigned_positive_count", 0)),
        "mpd_final_positive_count": float(stats.get("final_positive_count", 0)),
        "mpd_existing_positive_candidate_count": float(stats.get("existing_positive_candidate_count", 0)),
    }
    images = float(max(stats.get("images", 0), 1))
    target_gts = float(max(stats.get("target_gt_count", 0), 1))
    base_positive = float(max(stats.get("base_positive_count", 0), 1))
    final_positive = float(max(stats.get("final_positive_count", 0), 1))
    metrics["mpd_active_image_ratio"] = float(stats.get("images_with_target", 0)) / images
    metrics["mpd_added_positive_ratio"] = float(stats.get("added_positive_count", 0)) / final_positive
    metrics["mpd_positive_increase_ratio"] = float(stats.get("added_positive_count", 0)) / base_positive
    metrics["mpd_target_gt_covered_ratio"] = float(stats.get("covered_target_gt_count", 0)) / target_gts
    return metrics


def _merge_model_overrides(
    raw: Mapping[str, Any],
    *,
    arch: str | None,
) -> dict[str, Any]:
    normalized_arch = normalize_arch(arch or raw.get("arch"))
    model_overrides: dict[str, Any] = {}
    if normalized_arch is not None:
        per_model = raw.get("models", {})
        if isinstance(per_model, Mapping):
            selected = per_model.get(normalized_arch, {})
            if isinstance(selected, Mapping):
                model_overrides = dict(selected)

    merged = dict(raw)
    for key, value in model_overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged

