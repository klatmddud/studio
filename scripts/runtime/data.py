from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from .hard_replay import (
    HARD_REPLAY_CONFIG_PATH,
    HardReplayController,
    ReplayCrop,
    ReplayIndex,
    build_hard_replay_controller_from_yaml,
)


class CocoDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, images_dir: str | Path, annotations_path: str | Path) -> None:
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.annotations_path = Path(annotations_path).expanduser().resolve()
        self.coco = COCO(str(self.annotations_path))
        self.image_ids = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = _resolve_image_path(self.images_dir, image_info["file_name"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            image_tensor = F.to_tensor(image)

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(ann_ids)
        target = _build_target(image_id, width, height, annotations)
        return image_tensor, target


class CounterfactualReplayDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    requires_fresh_workers_per_epoch = True

    def __init__(self, base_dataset: CocoDetectionDataset) -> None:
        self.base_dataset = base_dataset
        self.images_dir = base_dataset.images_dir
        self.annotations_path = base_dataset.annotations_path
        self.coco = base_dataset.coco
        self.image_ids = base_dataset.image_ids
        self._replay_index = ReplayIndex.empty(enabled=False)

    def __len__(self) -> int:
        return len(self.base_dataset) + len(self._replay_index.replay_crops)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        base_size = len(self.base_dataset)
        if index < base_size:
            return self.base_dataset[index]

        crop_index = index - base_size
        replay_crops = self._replay_index.replay_crops
        if crop_index < 0 or crop_index >= len(replay_crops):
            raise IndexError(f"FCDR replay crop index out of range: {crop_index}")
        return self._build_replay_crop_sample(replay_crops[crop_index], crop_index=crop_index)

    def set_replay_index(self, replay_index: ReplayIndex) -> None:
        self._replay_index = replay_index

    def _build_replay_crop_sample(
        self,
        replay_crop: ReplayCrop,
        *,
        crop_index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        source_image_id = self.image_ids[replay_crop.dataset_index]
        image_info = self.coco.loadImgs([source_image_id])[0]
        image_path = _resolve_image_path(self.images_dir, image_info["file_name"])
        crop_x1, crop_y1, crop_x2, crop_y2 = replay_crop.crop_box_abs

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_width, crop_height = crop.size
            image_tensor = F.to_tensor(crop)

        ann_ids = self.coco.getAnnIds(imgIds=[source_image_id])
        annotations = self.coco.loadAnns(ann_ids)
        crop_annotations = _clip_annotations_to_crop(
            annotations,
            crop_box_abs=replay_crop.crop_box_abs,
        )

        synthetic_image_id = -(int(crop_index) + 1)
        target = _build_target(
            synthetic_image_id,
            crop_width,
            crop_height,
            crop_annotations,
        )
        target["is_replay"] = torch.tensor(True, dtype=torch.bool)
        target["source_image_id"] = _source_image_id_tensor(source_image_id)
        target["replay_gt_uid"] = replay_crop.gt_uid
        target["replay_failure_type"] = replay_crop.failure_type
        target["replay_mode"] = replay_crop.mode
        target["replay_severity"] = torch.tensor(replay_crop.severity, dtype=torch.float32)
        return image_tensor, target


def build_train_dataloaders(
    config: dict[str, Any],
    *,
    arch: str | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any] | None]:
    train_dataset = CocoDetectionDataset(
        config["data"]["train_images"],
        config["data"]["train_annotations"],
    )
    hard_replay = _build_hard_replay_controller(
        config,
        dataset=train_dataset,
        arch=arch,
    )
    loader_dataset: Dataset[Any] = train_dataset
    if hard_replay is not None and hard_replay.config.fcdr.enabled:
        replay_dataset = CounterfactualReplayDataset(train_dataset)
        hard_replay.attach_replay_dataset(replay_dataset)
        loader_dataset = replay_dataset

    train_loader = _build_loader(
        loader_dataset,
        batch_size=config["loader"]["batch_size"],
        num_workers=config["loader"]["num_workers"],
        pin_memory=config["loader"]["pin_memory"],
        shuffle=config["loader"]["shuffle"],
        batch_sampler=None if hard_replay is None else hard_replay.batch_sampler,
    )
    if hard_replay is not None:
        train_loader.hard_replay = hard_replay

    val_images = config["data"].get("val_images")
    val_annotations = config["data"].get("val_annotations")
    if not val_images or not val_annotations:
        return train_loader, None

    val_dataset = CocoDetectionDataset(val_images, val_annotations)
    val_loader = _build_loader(
        val_dataset,
        batch_size=config["loader"]["batch_size"],
        num_workers=config["loader"]["num_workers"],
        pin_memory=config["loader"]["pin_memory"],
        shuffle=False,
    )
    return train_loader, val_loader


def build_eval_dataloader(config: dict[str, Any]) -> DataLoader[Any]:
    dataset = CocoDetectionDataset(
        config["data"]["images"],
        config["data"]["annotations"],
    )
    return _build_loader(
        dataset,
        batch_size=config["loader"]["batch_size"],
        num_workers=config["loader"]["num_workers"],
        pin_memory=config["loader"]["pin_memory"],
        shuffle=False,
    )


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    if not batch:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)


def _build_loader(
    dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    batch_sampler=None,
) -> DataLoader[Any]:
    loader_kwargs = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0
        and not bool(getattr(dataset, "requires_fresh_workers_per_epoch", False)),
    }
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
    return DataLoader(
        **loader_kwargs,
    )


def _build_hard_replay_controller(
    config: dict[str, Any],
    *,
    dataset: Dataset[Any],
    arch: str | None,
) -> HardReplayController | None:
    if not HARD_REPLAY_CONFIG_PATH.is_file():
        return None
    return build_hard_replay_controller_from_yaml(
        HARD_REPLAY_CONFIG_PATH,
        dataset=dataset,
        batch_size=config["loader"]["batch_size"],
        shuffle=config["loader"]["shuffle"],
        seed=int(config["seed"]),
        arch=arch,
    )


def _resolve_image_path(images_dir: Path, file_name: str) -> Path:
    image_path = Path(file_name)
    if image_path.is_absolute():
        return image_path.resolve()

    for relative_path in _candidate_relative_image_paths(images_dir, image_path):
        candidate = (images_dir / relative_path).resolve()
        if candidate.is_file():
            return candidate

    return (images_dir / image_path).resolve()


def _candidate_relative_image_paths(images_dir: Path, image_path: Path) -> list[Path]:
    candidates: list[Path] = [image_path]

    deduped = _strip_images_dir_prefix(images_dir, image_path)
    if deduped is not None and deduped not in candidates:
        candidates.append(deduped)

    parts = image_path.parts
    for index in range(1, len(parts)):
        trimmed = Path(*parts[index:])
        if trimmed not in candidates:
            candidates.append(trimmed)

    return candidates


def _strip_images_dir_prefix(images_dir: Path, image_path: Path) -> Path | None:
    image_parts = image_path.parts
    images_dir_parts = images_dir.parts
    max_prefix_len = min(len(image_parts), len(images_dir_parts))
    for prefix_len in range(max_prefix_len, 0, -1):
        if image_parts[:prefix_len] != images_dir_parts[-prefix_len:]:
            continue
        remaining_parts = image_parts[prefix_len:]
        if not remaining_parts:
            return None
        return Path(*remaining_parts)
    return None


def _build_target(
    image_id: int,
    width: int,
    height: int,
    annotations: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    areas: list[float] = []
    crowds: list[int] = []

    for annotation in annotations:
        x, y, w, h = annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x1 = max(0.0, float(x))
        y1 = max(0.0, float(y))
        x2 = min(float(width), x1 + max(0.0, float(w)))
        y2 = min(float(height), y1 + max(0.0, float(h)))
        if x2 <= x1 or y2 <= y1:
            continue

        category_id = int(annotation["category_id"])
        if category_id < 1:
            continue

        boxes.append([x1, y1, x2, y2])
        labels.append(category_id)
        areas.append(float(annotation.get("area", (x2 - x1) * (y2 - y1))))
        crowds.append(int(annotation.get("iscrowd", 0)))

    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area_tensor = torch.tensor(areas, dtype=torch.float32)
        crowd_tensor = torch.tensor(crowds, dtype=torch.int64)
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)
        area_tensor = torch.zeros((0,), dtype=torch.float32)
        crowd_tensor = torch.zeros((0,), dtype=torch.int64)

    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "image_id": torch.tensor(image_id, dtype=torch.int64),
        "area": area_tensor,
        "iscrowd": crowd_tensor,
    }


def _clip_annotations_to_crop(
    annotations: list[dict[str, Any]],
    *,
    crop_box_abs: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box_abs
    clipped: list[dict[str, Any]] = []
    for annotation in annotations:
        raw_bbox = annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            continue
        x, y, w, h = [float(value) for value in raw_bbox]
        box_x1 = x
        box_y1 = y
        box_x2 = x + max(0.0, w)
        box_y2 = y + max(0.0, h)

        inter_x1 = max(box_x1, float(crop_x1))
        inter_y1 = max(box_y1, float(crop_y1))
        inter_x2 = min(box_x2, float(crop_x2))
        inter_y2 = min(box_y2, float(crop_y2))
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue

        clipped_annotation = dict(annotation)
        clipped_width = inter_x2 - inter_x1
        clipped_height = inter_y2 - inter_y1
        clipped_annotation["bbox"] = [
            inter_x1 - float(crop_x1),
            inter_y1 - float(crop_y1),
            clipped_width,
            clipped_height,
        ]
        clipped_annotation["area"] = clipped_width * clipped_height
        clipped.append(clipped_annotation)
    return clipped


def _source_image_id_tensor(image_id: Any) -> torch.Tensor:
    try:
        value = int(image_id)
    except (TypeError, ValueError):
        value = abs(hash(str(image_id))) % (2**31)
    return torch.tensor(value, dtype=torch.int64)
