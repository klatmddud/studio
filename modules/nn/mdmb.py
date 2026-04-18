from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torchvision.ops import boxes as box_ops


ARCH_ALIASES = {
    "faster_rcnn": "fasterrcnn",
    "faster-rcnn": "fasterrcnn",
    "fasterrcnn": "fasterrcnn",
    "fcos": "fcos",
    "dino": "dino",
}


@dataclass(frozen=True, slots=True)
class MDMBConfig:
    enabled: bool = True
    match_threshold: float = 0.5
    max_per_image: int | None = None
    warmup_epochs: int = 1
    arch: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None = None,
        *,
        arch: str | None = None,
    ) -> "MDMBConfig":
        data = dict(raw or {})
        normalized_arch = normalize_arch(arch or data.get("arch"))
        model_overrides: dict[str, Any] = {}
        if normalized_arch is not None:
            per_model = data.get("models", {})
            if isinstance(per_model, Mapping):
                selected = per_model.get(normalized_arch, {})
                if isinstance(selected, Mapping):
                    model_overrides = dict(selected)

        max_per_image = model_overrides.get("max_per_image", data.get("max_per_image"))
        config = cls(
            enabled=bool(model_overrides.get("enabled", data.get("enabled", True))),
            match_threshold=float(
                model_overrides.get("match_threshold", data.get("match_threshold", 0.5))
            ),
            max_per_image=None if max_per_image is None else int(max_per_image),
            warmup_epochs=int(model_overrides.get("warmup_epochs", data.get("warmup_epochs", 1))),
            arch=normalized_arch,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("MDMB match_threshold must satisfy 0 <= threshold <= 1.")
        if self.max_per_image is not None and self.max_per_image < 1:
            raise ValueError("MDMB max_per_image must be >= 1 when provided.")
        if self.warmup_epochs < 0:
            raise ValueError("MDMB warmup_epochs must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "match_threshold": self.match_threshold,
            "max_per_image": self.max_per_image,
            "warmup_epochs": self.warmup_epochs,
            "arch": self.arch,
        }


@dataclass(slots=True)
class MDMBObservation:
    """
    Legacy v1 observation payload kept only to preserve the import surface.

    MDMB v2 no longer updates from point/proposal/query observations.
    """

    image_id: int | str
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
    image_id: str
    class_id: int
    bbox: torch.Tensor

    def to_state(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "class_id": self.class_id,
            "bbox": self.bbox.cpu(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "MDMBEntry":
        return cls(
            image_id=_normalize_image_id(state["image_id"]),
            class_id=int(state["class_id"]),
            bbox=_as_region_tensor(state["bbox"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "class_id": self.class_id,
            "bbox": self.bbox.tolist(),
        }


class MissedDetectionMemoryBank(nn.Module):
    """
    MDMB v2 bank keyed by image id.

    Each image stores the current step's missed ground-truth boxes only:
    image_id -> [MDMBEntry(class_id, normalized_bbox)].
    """

    def __init__(self, config: MDMBConfig) -> None:
        super().__init__()
        self.config = config
        self._bank: dict[str, list[MDMBEntry]] = {}
        self.current_epoch = 0

    def forward(self, features):
        return features

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def end_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def should_update(self, *, epoch: int | None = None) -> bool:
        if epoch is not None:
            self.current_epoch = int(epoch)
        if not self.config.enabled:
            return False
        # current_epoch is tracked as 1-based in the runtime loop.
        return self.current_epoch > self.config.warmup_epochs

    @torch.no_grad()
    def update(
        self,
        *,
        image_ids: Sequence[Any],
        pred_boxes_list: Sequence[torch.Tensor | Sequence[Sequence[float]] | Sequence[float]],
        gt_boxes_list: Sequence[torch.Tensor | Sequence[Sequence[float]] | Sequence[float]],
        gt_labels_list: Sequence[torch.Tensor | Sequence[int] | int],
        image_shapes: Sequence[Sequence[int] | torch.Tensor],
        epoch: int | None = None,
    ) -> None:
        if not self.should_update(epoch=epoch):
            return

        batch_size = len(image_ids)
        expected_lengths = (
            len(pred_boxes_list),
            len(gt_boxes_list),
            len(gt_labels_list),
            len(image_shapes),
        )
        if any(length != batch_size for length in expected_lengths):
            raise ValueError(
                "MDMB update inputs must share the same batch dimension: "
                f"image_ids={batch_size}, pred_boxes={expected_lengths[0]}, "
                f"gt_boxes={expected_lengths[1]}, gt_labels={expected_lengths[2]}, "
                f"image_shapes={expected_lengths[3]}."
            )

        for image_id, pred_boxes, gt_boxes, gt_labels, image_shape in zip(
            image_ids,
            pred_boxes_list,
            gt_boxes_list,
            gt_labels_list,
            image_shapes,
            strict=True,
        ):
            image_key = _normalize_image_id(image_id)
            gt_boxes_tensor = _as_box_tensor(gt_boxes)
            gt_labels_tensor = _as_label_tensor(gt_labels, device=gt_boxes_tensor.device)
            if gt_boxes_tensor.shape[0] != gt_labels_tensor.shape[0]:
                raise ValueError(
                    "MDMB gt_boxes and gt_labels must contain the same number of instances. "
                    f"Got {gt_boxes_tensor.shape[0]} and {gt_labels_tensor.shape[0]} for {image_key!r}."
                )

            if gt_boxes_tensor.numel() == 0:
                self._bank.pop(image_key, None)
                continue

            pred_boxes_tensor = _as_box_tensor(pred_boxes, device=gt_boxes_tensor.device)
            if pred_boxes_tensor.numel() == 0:
                max_ious = gt_boxes_tensor.new_zeros((gt_boxes_tensor.shape[0],))
            else:
                max_ious = box_ops.box_iou(gt_boxes_tensor, pred_boxes_tensor).max(dim=1).values

            normalized_gt_boxes = normalize_xyxy_boxes(gt_boxes_tensor, image_shape)
            missed_entries: list[MDMBEntry] = []
            for gt_index in torch.where(max_ious < self.config.match_threshold)[0].tolist():
                missed_entries.append(
                    MDMBEntry(
                        image_id=image_key,
                        class_id=int(gt_labels_tensor[gt_index].item()),
                        bbox=normalized_gt_boxes[gt_index].cpu(),
                    )
                )

            if self.config.max_per_image is not None:
                missed_entries = missed_entries[: self.config.max_per_image]

            if missed_entries:
                self._bank[image_key] = missed_entries
            else:
                self._bank.pop(image_key, None)

    def observe(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "MDMB v2 no longer supports observation-based updates. "
            "Use update(...) with final prediction boxes and GT boxes after optimizer.step()."
        )

    def reset(self) -> None:
        self._bank.clear()
        self.current_epoch = 0

    def get(self, image_id: Any) -> list[MDMBEntry]:
        return list(self._bank.get(_normalize_image_id(image_id), ()))

    def items(self) -> Iterator[tuple[str, list[MDMBEntry]]]:
        for image_id, entries in self._bank.items():
            yield image_id, list(entries)

    def values(self) -> Iterator[list[MDMBEntry]]:
        for entries in self._bank.values():
            yield list(entries)

    def get_image_entries(self, image_id: Any) -> list[MDMBEntry]:
        return self.get(image_id)

    def summary(self) -> dict[str, Any]:
        num_entries = len(self)
        num_images = len(self._bank)
        warmup_active = self.config.enabled and self.current_epoch <= self.config.warmup_epochs
        return {
            "enabled": self.config.enabled,
            "arch": self.config.arch,
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.config.warmup_epochs,
            "warmup_active": warmup_active,
            "num_images": num_images,
            "num_entries": num_entries,
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
            "version": 2,
            "current_epoch": self.current_epoch,
            "config": self.config.to_dict(),
            "bank": {
                image_id: [entry.to_state() for entry in entries]
                for image_id, entries in self._bank.items()
            },
        }

    def set_extra_state(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            self.reset()
            return

        config_state = state.get("config", {})
        if isinstance(config_state, Mapping):
            try:
                self.config = MDMBConfig.from_mapping(config_state, arch=config_state.get("arch"))
            except Exception:
                pass
        self.current_epoch = int(state.get("current_epoch", 0))

        # Gracefully ignore legacy v1 checkpoints instead of failing the whole load.
        if int(state.get("version", 0)) != 2:
            self._bank = {}
            return

        raw_bank = state.get("bank", {})
        if not isinstance(raw_bank, Mapping):
            self._bank = {}
            return

        restored_bank: dict[str, list[MDMBEntry]] = {}
        for image_id, raw_entries in raw_bank.items():
            if not isinstance(raw_entries, Sequence):
                continue
            image_key = _normalize_image_id(image_id)
            entries: list[MDMBEntry] = []
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, Mapping):
                    continue
                entries.append(MDMBEntry.from_state(raw_entry))
            if entries:
                restored_bank[image_key] = entries
        self._bank = restored_bank


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


def cxcywh_to_xyxy(
    boxes: torch.Tensor | Sequence[Sequence[float]] | Sequence[float],
) -> torch.Tensor:
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
        raise ValueError("MDMB boxes must have shape [N, 4] or [4].")
    return tensor


def _as_region_tensor(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().flatten()
    if tensor.numel() != 4:
        raise ValueError("MDMB bbox must contain exactly four values.")
    return tensor.cpu()


def _as_label_tensor(
    value: torch.Tensor | Sequence[int] | int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int64, device=device).detach().flatten()
    return tensor


def _normalize_image_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("MDMB image_id tensor must contain a single scalar value.")
        value = value.item()
    return str(value)
