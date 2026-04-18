from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .mdmb import MDMBEntry, normalize_arch


DEFAULT_FCOS_STRIDES = (8, 16, 32, 64, 128)
DEFAULT_SCALE_RANGES = (
    (0.0, 64.0),
    (64.0, 128.0),
    (128.0, 256.0),
    (256.0, 512.0),
    (512.0, 1_000_000.0),
)


@dataclass(frozen=True, slots=True)
class MODSConfig:
    enabled: bool = True
    lambda_mods: float = 0.1
    lambda_reg: float = 0.0
    jitter: float = 0.0
    roi_output_size: int = 1
    sampling_ratio: int = 2
    aligned: bool = True
    canonical_image_size: int = 640
    max_samples_per_image: int | None = None
    reduction: str = "mean"
    arch: str | None = None
    strides: tuple[int, ...] = DEFAULT_FCOS_STRIDES
    scale_ranges: tuple[tuple[float, float], ...] = DEFAULT_SCALE_RANGES

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "MODSConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        max_samples_per_image = model_overrides.get(
            "max_samples_per_image",
            data.get("max_samples_per_image"),
        )
        config = cls(
            enabled=bool(model_overrides.get("enabled", data.get("enabled", True))),
            lambda_mods=float(model_overrides.get("lambda_mods", data.get("lambda_mods", 0.1))),
            lambda_reg=float(model_overrides.get("lambda_reg", data.get("lambda_reg", 0.0))),
            jitter=float(model_overrides.get("jitter", data.get("jitter", 0.0))),
            roi_output_size=int(
                model_overrides.get("roi_output_size", data.get("roi_output_size", 1))
            ),
            sampling_ratio=int(
                model_overrides.get("sampling_ratio", data.get("sampling_ratio", 2))
            ),
            aligned=bool(model_overrides.get("aligned", data.get("aligned", True))),
            canonical_image_size=int(
                model_overrides.get(
                    "canonical_image_size",
                    data.get("canonical_image_size", 640),
                )
            ),
            max_samples_per_image=(
                None if max_samples_per_image is None else int(max_samples_per_image)
            ),
            reduction=str(model_overrides.get("reduction", data.get("reduction", "mean"))),
            arch=normalized_arch,
            strides=_parse_strides(model_overrides.get("strides", data.get("strides"))),
            scale_ranges=_parse_scale_ranges(
                model_overrides.get("scale_ranges", data.get("scale_ranges"))
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.lambda_mods < 0.0:
            raise ValueError("MODS lambda_mods must be >= 0.")
        if self.lambda_reg < 0.0:
            raise ValueError("MODS lambda_reg must be >= 0.")
        if self.jitter < 0.0:
            raise ValueError("MODS jitter must be >= 0.")
        if self.roi_output_size < 1:
            raise ValueError("MODS roi_output_size must be >= 1.")
        if self.sampling_ratio < -1:
            raise ValueError("MODS sampling_ratio must be >= -1.")
        if self.canonical_image_size < 1:
            raise ValueError("MODS canonical_image_size must be >= 1.")
        if self.max_samples_per_image is not None and self.max_samples_per_image < 1:
            raise ValueError("MODS max_samples_per_image must be >= 1 when provided.")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("MODS reduction must be either 'mean' or 'sum'.")
        if not self.strides:
            raise ValueError("MODS strides must not be empty.")
        if len(self.strides) != len(self.scale_ranges):
            raise ValueError("MODS strides and scale_ranges must have the same length.")
        for lower, upper in self.scale_ranges:
            if lower < 0.0 or upper <= lower:
                raise ValueError("Each MODS scale range must satisfy 0 <= lower < upper.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lambda_mods": self.lambda_mods,
            "lambda_reg": self.lambda_reg,
            "jitter": self.jitter,
            "roi_output_size": self.roi_output_size,
            "sampling_ratio": self.sampling_ratio,
            "aligned": self.aligned,
            "canonical_image_size": self.canonical_image_size,
            "max_samples_per_image": self.max_samples_per_image,
            "reduction": self.reduction,
            "arch": self.arch,
            "strides": list(self.strides),
            "scale_ranges": [list(pair) for pair in self.scale_ranges],
        }


@dataclass(slots=True)
class MODSTargetBatch:
    image_indices: torch.Tensor
    class_ids: torch.Tensor
    boxes_norm: torch.Tensor
    boxes_abs: torch.Tensor
    level_indices: torch.Tensor

    def __len__(self) -> int:
        return int(self.class_ids.numel())

    def is_empty(self) -> bool:
        return len(self) == 0


class MissedObjectDirectSupervision(nn.Module):
    """Utility module that prepares MODS targets for architecture-specific wrappers."""

    def __init__(self, config: MODSConfig) -> None:
        super().__init__()
        self.config = config

    def collect_targets(
        self,
        *,
        image_ids: Sequence[Any],
        missed_entries_batch: Sequence[Sequence[MDMBEntry | Mapping[str, Any]]],
        image_shapes: Sequence[Sequence[int] | torch.Tensor],
        device: torch.device | str,
        training: bool | None = None,
    ) -> MODSTargetBatch:
        if len(image_ids) != len(missed_entries_batch) or len(image_ids) != len(image_shapes):
            raise ValueError(
                "MODS collect_targets expects image_ids, missed_entries_batch, and image_shapes "
                "to share the same batch dimension."
            )

        use_jitter = self.training if training is None else bool(training)
        image_indices: list[torch.Tensor] = []
        class_ids: list[torch.Tensor] = []
        boxes_norm: list[torch.Tensor] = []
        boxes_abs: list[torch.Tensor] = []
        level_indices: list[torch.Tensor] = []

        for batch_index, (entries, image_shape) in enumerate(
            zip(missed_entries_batch, image_shapes, strict=True)
        ):
            if not entries:
                continue
            normalized_boxes, entry_class_ids = _entries_to_tensors(entries, device=device)
            if self.config.max_samples_per_image is not None:
                normalized_boxes = normalized_boxes[: self.config.max_samples_per_image]
                entry_class_ids = entry_class_ids[: self.config.max_samples_per_image]
            if normalized_boxes.numel() == 0:
                continue

            if use_jitter and self.config.jitter > 0.0:
                normalized_boxes = apply_gt_jitter(normalized_boxes, self.config.jitter)

            boxes_in_pixels = denormalize_xyxy_boxes(normalized_boxes, image_shape)
            assigned_levels = assign_fpn_level_for_gt(
                normalized_boxes,
                image_shape=image_shape,
                scale_ranges=self.config.scale_ranges,
                canonical_image_size=self.config.canonical_image_size,
            ).to(device=device)

            image_indices.append(
                torch.full(
                    (normalized_boxes.shape[0],),
                    batch_index,
                    dtype=torch.int64,
                    device=device,
                )
            )
            class_ids.append(entry_class_ids)
            boxes_norm.append(normalized_boxes)
            boxes_abs.append(boxes_in_pixels)
            level_indices.append(assigned_levels)

        if not class_ids:
            return MODSTargetBatch(
                image_indices=torch.zeros((0,), dtype=torch.int64, device=device),
                class_ids=torch.zeros((0,), dtype=torch.int64, device=device),
                boxes_norm=torch.zeros((0, 4), dtype=torch.float32, device=device),
                boxes_abs=torch.zeros((0, 4), dtype=torch.float32, device=device),
                level_indices=torch.zeros((0,), dtype=torch.int64, device=device),
            )

        return MODSTargetBatch(
            image_indices=torch.cat(image_indices, dim=0),
            class_ids=torch.cat(class_ids, dim=0),
            boxes_norm=torch.cat(boxes_norm, dim=0),
            boxes_abs=torch.cat(boxes_abs, dim=0),
            level_indices=torch.cat(level_indices, dim=0),
        )

    def extra_repr(self) -> str:
        return (
            f"arch={self.config.arch!r}, lambda_mods={self.config.lambda_mods}, "
            f"lambda_reg={self.config.lambda_reg}, jitter={self.config.jitter}"
        )


MODS = MissedObjectDirectSupervision


def apply_gt_jitter(
    boxes: torch.Tensor | Sequence[Sequence[float]] | Sequence[float],
    jitter: float = 0.05,
) -> torch.Tensor:
    tensor = _as_box_tensor(boxes)
    if tensor.numel() == 0 or float(jitter) <= 0.0:
        return tensor

    x1, y1, x2, y2 = tensor.unbind(dim=-1)
    width = (x2 - x1).clamp_min(1e-6)
    height = (y2 - y1).clamp_min(1e-6)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    offset_scale = float(jitter)
    size_scale = float(jitter)
    center_x = center_x + ((torch.rand_like(center_x) * 2.0) - 1.0) * offset_scale * width
    center_y = center_y + ((torch.rand_like(center_y) * 2.0) - 1.0) * offset_scale * height
    width = width * (1.0 + ((torch.rand_like(width) * 2.0) - 1.0) * size_scale)
    height = height * (1.0 + ((torch.rand_like(height) * 2.0) - 1.0) * size_scale)

    jittered = torch.stack(
        (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ),
        dim=-1,
    )
    return jittered.clamp_(min=0.0, max=1.0)


def assign_fpn_level_for_gt(
    boxes: torch.Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    image_shape: Sequence[int] | torch.Tensor | None = None,
    scale_ranges: Sequence[Sequence[float]] = DEFAULT_SCALE_RANGES,
    canonical_image_size: int = 640,
) -> torch.Tensor:
    tensor = _as_box_tensor(boxes)
    if tensor.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64, device=tensor.device)

    widths = (tensor[:, 2] - tensor[:, 0]).clamp_min(0.0)
    heights = (tensor[:, 3] - tensor[:, 1]).clamp_min(0.0)
    if image_shape is None:
        box_sizes = torch.maximum(widths, heights) * float(canonical_image_size)
    else:
        height, width = _resolve_hw(image_shape)
        box_sizes = torch.maximum(widths * float(width), heights * float(height))

    resolved_ranges = tuple((float(lower), float(upper)) for lower, upper in scale_ranges)
    levels = torch.full(
        (tensor.shape[0],),
        len(resolved_ranges) - 1,
        dtype=torch.int64,
        device=tensor.device,
    )
    for level_index, (lower, upper) in enumerate(resolved_ranges):
        mask = (box_sizes >= lower) & (box_sizes < upper)
        levels[mask] = level_index
    return levels


def denormalize_xyxy_boxes(
    boxes: torch.Tensor | Sequence[Sequence[float]] | Sequence[float],
    image_shape: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    tensor = _as_box_tensor(boxes)
    if tensor.numel() == 0:
        return tensor
    height, width = _resolve_hw(image_shape)
    scale = tensor.new_tensor([width, height, width, height]).clamp(min=1.0)
    return tensor * scale


def load_mods_config(path: str | Path, *, arch: str | None = None) -> MODSConfig:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"MODS YAML must contain a mapping at the top level: {config_path}")
    return MODSConfig.from_mapping(raw, arch=arch)


def build_mods_from_config(
    raw_config: Mapping[str, Any] | MODSConfig,
    *,
    arch: str | None = None,
) -> MissedObjectDirectSupervision | None:
    config = (
        raw_config
        if isinstance(raw_config, MODSConfig)
        else MODSConfig.from_mapping(raw_config, arch=arch)
    )
    if not config.enabled:
        return None
    return MissedObjectDirectSupervision(config)


def build_mods_from_yaml(
    path: str | Path,
    *,
    arch: str | None = None,
) -> MissedObjectDirectSupervision | None:
    config = load_mods_config(path, arch=arch)
    if not config.enabled:
        return None
    return MissedObjectDirectSupervision(config)


def _entries_to_tensors(
    entries: Sequence[MDMBEntry | Mapping[str, Any]],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes: list[torch.Tensor] = []
    class_ids: list[int] = []
    for entry in entries:
        if isinstance(entry, MDMBEntry):
            boxes.append(torch.as_tensor(entry.bbox, dtype=torch.float32, device=device))
            class_ids.append(int(entry.class_id))
            continue
        if not isinstance(entry, Mapping):
            raise TypeError("MODS entries must be MDMBEntry or mappings.")
        boxes.append(torch.as_tensor(entry["bbox"], dtype=torch.float32, device=device))
        class_ids.append(int(entry["class_id"]))

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.int64, device=device),
        )

    return (
        torch.stack(boxes, dim=0),
        torch.as_tensor(class_ids, dtype=torch.int64, device=device),
    )


def _as_box_tensor(
    boxes: torch.Tensor | Sequence[Sequence[float]] | Sequence[float],
) -> torch.Tensor:
    tensor = torch.as_tensor(boxes, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor.reshape(-1, 4)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != 4:
        raise ValueError("MODS boxes must have shape [N, 4] or [4].")
    return tensor


def _resolve_hw(image_shape: Sequence[int] | torch.Tensor) -> tuple[int, int]:
    if isinstance(image_shape, torch.Tensor):
        return int(image_shape[-2].item()), int(image_shape[-1].item())
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain exactly two values: (height, width).")
    return int(image_shape[0]), int(image_shape[1])


def _parse_strides(value: Any) -> tuple[int, ...]:
    raw = DEFAULT_FCOS_STRIDES if value is None else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("MODS strides must be a sequence of integers.")
    return tuple(int(stride) for stride in raw)


def _parse_scale_ranges(value: Any) -> tuple[tuple[float, float], ...]:
    raw = DEFAULT_SCALE_RANGES if value is None else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("MODS scale_ranges must be a sequence of [lower, upper] pairs.")

    parsed: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise TypeError("Each MODS scale range must contain exactly two numeric values.")
        parsed.append((float(item[0]), float(item[1])))
    return tuple(parsed)
