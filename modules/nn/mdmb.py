from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


ARCH_ALIASES = {
    "faster_rcnn": "fasterrcnn",
    "faster-rcnn": "fasterrcnn",
    "fasterrcnn": "fasterrcnn",
    "fcos": "fcos",
    "dino": "dino",
}

DEFAULT_TRACK_UNITS = {
    "fasterrcnn": "proposal",
    "fcos": "point",
    "dino": "query",
}


@dataclass(frozen=True, slots=True)
class MDMBConfig:
    enabled: bool = True
    tau: int = 3
    iou_low: float = 0.3
    iou_high: float = 0.5
    momentum: float = 0.99
    stale_ttl: int = 5
    max_entries_per_image: int = 50
    coord_precision: int = 4
    storage_device: str = "cpu"
    prune_on_epoch_end: bool = True
    detection_score_threshold: float = 0.5
    arch: str | None = None
    track_unit: str = "proposal"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "MDMBConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        model_overrides = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        track_unit = model_overrides.get(
            "track_unit",
            data.get("track_unit", DEFAULT_TRACK_UNITS.get(normalized_arch, "proposal")),
        )
        config = cls(
            enabled=bool(model_overrides.get("enabled", data.get("enabled", True))),
            tau=int(model_overrides.get("tau", data.get("tau", 3))),
            iou_low=float(model_overrides.get("iou_low", data.get("iou_low", 0.3))),
            iou_high=float(model_overrides.get("iou_high", data.get("iou_high", 0.5))),
            momentum=float(model_overrides.get("momentum", data.get("momentum", 0.99))),
            stale_ttl=int(model_overrides.get("stale_ttl", data.get("stale_ttl", 5))),
            max_entries_per_image=int(
                model_overrides.get(
                    "max_entries_per_image",
                    data.get("max_entries_per_image", 50),
                )
            ),
            coord_precision=int(
                model_overrides.get("coord_precision", data.get("coord_precision", 4))
            ),
            storage_device=str(
                model_overrides.get("storage_device", data.get("storage_device", "cpu"))
            ),
            prune_on_epoch_end=bool(
                model_overrides.get(
                    "prune_on_epoch_end",
                    data.get("prune_on_epoch_end", True),
                )
            ),
            detection_score_threshold=float(
                model_overrides.get(
                    "detection_score_threshold",
                    data.get("detection_score_threshold", 0.5),
                )
            ),
            arch=normalized_arch,
            track_unit=str(track_unit),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.tau < 1:
            raise ValueError("MDMB tau must be >= 1.")
        if not 0.0 <= self.iou_low < self.iou_high <= 1.0:
            raise ValueError("MDMB IoU thresholds must satisfy 0 <= iou_low < iou_high <= 1.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("MDMB momentum must satisfy 0 <= momentum < 1.")
        if not 0.0 <= self.detection_score_threshold <= 1.0:
            raise ValueError("MDMB detection_score_threshold must satisfy 0 <= threshold <= 1.")
        if self.stale_ttl < 1:
            raise ValueError("MDMB stale_ttl must be >= 1.")
        if self.max_entries_per_image < 1:
            raise ValueError("MDMB max_entries_per_image must be >= 1.")
        if self.coord_precision < 0:
            raise ValueError("MDMB coord_precision must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tau": self.tau,
            "iou_low": self.iou_low,
            "iou_high": self.iou_high,
            "momentum": self.momentum,
            "stale_ttl": self.stale_ttl,
            "max_entries_per_image": self.max_entries_per_image,
            "coord_precision": self.coord_precision,
            "storage_device": self.storage_device,
            "prune_on_epoch_end": self.prune_on_epoch_end,
            "detection_score_threshold": self.detection_score_threshold,
            "arch": self.arch,
            "track_unit": self.track_unit,
        }


@dataclass(slots=True)
class MDMBObservation:
    image_id: int
    region_coords: torch.Tensor | Sequence[float]
    iou_max: float
    cls_score: float
    feature_vec: torch.Tensor | Sequence[float] | None = None
    gt_class: int | None = None
    detected: bool = False
    track_id: str | None = None
    source: str | None = None


@dataclass(slots=True)
class MDMBEntry:
    track_id: str
    image_id: int
    region_coords: torch.Tensor
    iou_max: float
    cls_score: float
    miss_count: int
    feature_vec: torch.Tensor | None
    gt_class: int | None
    last_updated: int
    source: str | None = None

    def is_chronic_miss(self, config: MDMBConfig) -> bool:
        return (
            config.iou_low <= self.iou_max < config.iou_high
            and self.miss_count >= config.tau
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "image_id": self.image_id,
            "region_coords": self.region_coords.cpu(),
            "iou_max": self.iou_max,
            "cls_score": self.cls_score,
            "miss_count": self.miss_count,
            "feature_vec": None if self.feature_vec is None else self.feature_vec.cpu(),
            "gt_class": self.gt_class,
            "last_updated": self.last_updated,
            "source": self.source,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "MDMBEntry":
        return cls(
            track_id=str(state["track_id"]),
            image_id=int(state["image_id"]),
            region_coords=torch.as_tensor(state["region_coords"], dtype=torch.float32),
            iou_max=float(state["iou_max"]),
            cls_score=float(state["cls_score"]),
            miss_count=int(state["miss_count"]),
            feature_vec=_as_feature_tensor(state.get("feature_vec")),
            gt_class=_optional_int(state.get("gt_class")),
            last_updated=int(state["last_updated"]),
            source=_optional_str(state.get("source")),
        )


class MissedDetectionMemoryBank(nn.Module):
    """
    Epoch-level memory bank for tracking persistent near-miss regions.

    The bank is deliberately detector-agnostic. Model-specific code is expected
    to provide stable observations keyed by image id plus normalized region
    coordinates, or an explicit track_id when a better proposal/query identity
    is available.
    """

    def __init__(self, config: MDMBConfig) -> None:
        super().__init__()
        self.config = config
        self._entries: dict[str, MDMBEntry] = {}
        self.current_epoch = 0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)
        if self.config.prune_on_epoch_end:
            self.prune_stale(self.current_epoch)

    def update(
        self,
        observations: Iterable[MDMBObservation | Mapping[str, Any]],
        *,
        epoch: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return

        update_epoch = self.current_epoch if epoch is None else int(epoch)
        if update_epoch < 0:
            raise ValueError("MDMB epoch must be >= 0.")
        self.current_epoch = update_epoch

        touched_images: set[int] = set()
        for observation in observations:
            normalized = self._normalize_observation(observation)
            touched_images.add(normalized.image_id)
            track_id = normalized.track_id or self.make_track_id(
                image_id=normalized.image_id,
                region_coords=normalized.region_coords,
                source=normalized.source,
                gt_class=normalized.gt_class,
            )

            if self._is_positive_detected(normalized):
                entry = self._entries.get(track_id)
                if entry is None:
                    continue
                entry.miss_count = 0
                entry.iou_max = normalized.iou_max
                entry.cls_score = normalized.cls_score
                entry.region_coords = normalized.region_coords
                feature_vec = _as_feature_tensor(
                    normalized.feature_vec,
                    device=self.config.storage_device,
                )
                if feature_vec is not None:
                    entry.feature_vec = feature_vec
                entry.gt_class = normalized.gt_class
                entry.last_updated = update_epoch
                entry.source = normalized.source
                continue

            if self._is_near_miss(normalized):
                self._update_near_miss(track_id, normalized, update_epoch)

        for image_id in touched_images:
            self._enforce_capacity(image_id)

    def observe(
        self,
        *,
        image_id: int,
        region_coords: torch.Tensor | Sequence[float],
        iou_max: float,
        cls_score: float,
        feature_vec: torch.Tensor | Sequence[float] | None = None,
        gt_class: int | None = None,
        detected: bool = False,
        track_id: str | None = None,
        source: str | None = None,
        epoch: int | None = None,
    ) -> None:
        self.update(
            [
                MDMBObservation(
                    image_id=image_id,
                    region_coords=region_coords,
                    iou_max=iou_max,
                    cls_score=cls_score,
                    feature_vec=feature_vec,
                    gt_class=gt_class,
                    detected=detected,
                    track_id=track_id,
                    source=source,
                )
            ],
            epoch=epoch,
        )

    def prune_stale(self, epoch: int | None = None) -> int:
        current_epoch = self.current_epoch if epoch is None else int(epoch)
        stale_keys = [
            key
            for key, entry in self._entries.items()
            if current_epoch - entry.last_updated >= self.config.stale_ttl
        ]
        for key in stale_keys:
            del self._entries[key]
        return len(stale_keys)

    def reset(self) -> None:
        self._entries.clear()
        self.current_epoch = 0

    def get(self, track_id: str) -> MDMBEntry | None:
        return self._entries.get(track_id)

    def items(self) -> Iterator[tuple[str, MDMBEntry]]:
        return iter(self._entries.items())

    def values(self) -> Iterator[MDMBEntry]:
        return iter(self._entries.values())

    def get_image_entries(self, image_id: int) -> list[MDMBEntry]:
        return [entry for entry in self._entries.values() if entry.image_id == int(image_id)]

    def get_chronic_misses(self, image_id: int | None = None) -> list[MDMBEntry]:
        entries = self._entries.values()
        if image_id is not None:
            entries = (entry for entry in entries if entry.image_id == int(image_id))
        chronic = [entry for entry in entries if entry.is_chronic_miss(self.config)]
        return sorted(
            chronic,
            key=lambda entry: (entry.miss_count, entry.iou_max, entry.last_updated),
            reverse=True,
        )

    def summary(self) -> dict[str, Any]:
        chronic = self.get_chronic_misses()
        max_miss = max((entry.miss_count for entry in self._entries.values()), default=0)
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "track_unit": self.config.track_unit,
            "current_epoch": self.current_epoch,
            "num_entries": len(self._entries),
            "num_chronic_misses": len(chronic),
            "num_images": len({entry.image_id for entry in self._entries.values()}),
            "max_miss_count": max_miss,
        }

    def make_track_id(
        self,
        *,
        image_id: int,
        region_coords: torch.Tensor | Sequence[float],
        source: str | None = None,
        gt_class: int | None = None,
    ) -> str:
        coords = _as_region_tensor(region_coords)
        quantized = [
            f"{round(float(value), self.config.coord_precision):.{self.config.coord_precision}f}"
            for value in coords.tolist()
        ]
        source_value = source or self.config.track_unit
        label_value = "na" if gt_class is None else str(int(gt_class))
        return (
            f"img:{int(image_id)}"
            f"|src:{source_value}"
            f"|cls:{label_value}"
            f"|box:{','.join(quantized)}"
        )

    def __len__(self) -> int:
        return len(self._entries)

    def extra_repr(self) -> str:
        return (
            f"enabled={self.config.enabled}, arch={self.config.arch!r}, "
            f"track_unit={self.config.track_unit!r}, entries={len(self._entries)}"
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "entries": [entry.to_state() for entry in self._entries.values()],
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            self.reset()
            return

        config_state = state.get("config", {})
        self.config = MDMBConfig.from_mapping(config_state, arch=config_state.get("arch"))
        self.current_epoch = int(state.get("current_epoch", 0))
        self._entries = {}
        for entry_state in state.get("entries", []):
            entry = MDMBEntry.from_state(entry_state)
            self._entries[entry.track_id] = entry

    def _update_near_miss(
        self,
        track_id: str,
        observation: MDMBObservation,
        epoch: int,
    ) -> None:
        feature_vec = _as_feature_tensor(
            observation.feature_vec,
            device=self.config.storage_device,
        )
        entry = self._entries.get(track_id)
        if entry is None:
            self._entries[track_id] = MDMBEntry(
                track_id=track_id,
                image_id=observation.image_id,
                region_coords=observation.region_coords,
                iou_max=observation.iou_max,
                cls_score=observation.cls_score,
                miss_count=1,
                feature_vec=feature_vec,
                gt_class=observation.gt_class,
                last_updated=epoch,
                source=observation.source,
            )
            return

        entry.image_id = observation.image_id
        entry.region_coords = observation.region_coords
        entry.iou_max = observation.iou_max
        entry.cls_score = observation.cls_score
        entry.gt_class = observation.gt_class
        entry.last_updated = epoch
        entry.source = observation.source
        entry.miss_count += 1

        if feature_vec is None:
            return
        if entry.feature_vec is None:
            entry.feature_vec = feature_vec
            return
        entry.feature_vec = (
            self.config.momentum * entry.feature_vec
            + (1.0 - self.config.momentum) * feature_vec
        )

    def _enforce_capacity(self, image_id: int) -> None:
        entries = [
            entry
            for entry in self._entries.values()
            if entry.image_id == int(image_id)
        ]
        if len(entries) <= self.config.max_entries_per_image:
            return

        entries.sort(
            key=lambda entry: (entry.iou_max, entry.miss_count, entry.last_updated),
            reverse=True,
        )
        for entry in entries[self.config.max_entries_per_image :]:
            self._entries.pop(entry.track_id, None)

    def _normalize_observation(
        self,
        observation: MDMBObservation | Mapping[str, Any],
    ) -> MDMBObservation:
        if isinstance(observation, MDMBObservation):
            raw = observation
        else:
            raw = MDMBObservation(
                image_id=int(observation["image_id"]),
                region_coords=observation["region_coords"],
                iou_max=float(observation["iou_max"]),
                cls_score=float(observation.get("cls_score", 0.0)),
                feature_vec=observation.get("feature_vec"),
                gt_class=_optional_int(observation.get("gt_class")),
                detected=bool(observation.get("detected", False)),
                track_id=_optional_str(observation.get("track_id")),
                source=_optional_str(observation.get("source")),
            )

        return MDMBObservation(
            image_id=int(raw.image_id),
            region_coords=_as_region_tensor(raw.region_coords),
            iou_max=float(raw.iou_max),
            cls_score=float(raw.cls_score),
            feature_vec=raw.feature_vec,
            gt_class=_optional_int(raw.gt_class),
            detected=bool(raw.detected),
            track_id=_optional_str(raw.track_id),
            source=_optional_str(raw.source) or self.config.track_unit,
        )

    def _is_positive_detected(self, observation: MDMBObservation) -> bool:
        return observation.detected and observation.iou_max >= self.config.iou_high

    def _is_near_miss(self, observation: MDMBObservation) -> bool:
        return self.config.iou_low <= observation.iou_max < self.config.iou_high


MDMB = MissedDetectionMemoryBank


def normalize_arch(raw_arch: str | None) -> str | None:
    if raw_arch is None:
        return None
    return ARCH_ALIASES.get(str(raw_arch).lower(), str(raw_arch).lower())


def load_mdmb_config(path: str | Path, *, arch: str | None = None) -> MDMBConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"MDMB YAML must contain a mapping at the top level: {config_path}")
    return MDMBConfig.from_mapping(raw, arch=arch)


def build_mdmb_from_config(
    raw_config: Mapping[str, Any] | MDMBConfig,
    *,
    arch: str | None = None,
) -> MissedDetectionMemoryBank | None:
    config = (
        raw_config
        if isinstance(raw_config, MDMBConfig)
        else MDMBConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return MissedDetectionMemoryBank(config)


def build_mdmb_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> MissedDetectionMemoryBank | None:
    config = load_mdmb_config(path, arch=arch)
    if not config.enabled:
        return None
    return MissedDetectionMemoryBank(config)


def normalize_xyxy_boxes(
    boxes: torch.Tensor | Sequence[Sequence[float]] | Sequence[float],
    image_shape: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    tensor = torch.as_tensor(boxes, dtype=torch.float32).detach()
    if tensor.numel() == 0:
        return tensor.reshape(-1, 4)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != 4:
        raise ValueError("Boxes must have shape [N, 4] or [4].")

    if isinstance(image_shape, torch.Tensor):
        height = int(image_shape[-2].item())
        width = int(image_shape[-1].item())
    else:
        if len(image_shape) != 2:
            raise ValueError("image_shape must contain exactly two values: (height, width).")
        height = int(image_shape[0])
        width = int(image_shape[1])

    scale = tensor.new_tensor([width, height, width, height]).clamp(min=1.0)
    normalized = tensor / scale
    return normalized.clamp_(min=0.0, max=1.0)


def cxcywh_to_xyxy(boxes: torch.Tensor | Sequence[Sequence[float]] | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(boxes, dtype=torch.float32).detach()
    if tensor.numel() == 0:
        return tensor.reshape(-1, 4)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != 4:
        raise ValueError("Boxes must have shape [N, 4] or [4].")

    cx, cy, width, height = tensor.unbind(dim=-1)
    half_width = width * 0.5
    half_height = height * 0.5
    return torch.stack(
        (cx - half_width, cy - half_height, cx + half_width, cy + half_height),
        dim=-1,
    )


def select_topk_indices(scores: torch.Tensor, *, k: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_zeros((0,), dtype=torch.long)
    if k <= 0 or scores.numel() <= k:
        return torch.argsort(scores, descending=True)
    return torch.topk(scores, k=k).indices


def _as_region_tensor(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().flatten()
    if tensor.numel() != 4:
        raise ValueError("MDMB region_coords must contain exactly four values.")
    return tensor.cpu()


def _as_feature_tensor(
    value: torch.Tensor | Sequence[float] | None,
    *,
    device: str = "cpu",
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().flatten()
    return tensor.to(device=device)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
