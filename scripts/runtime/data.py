from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as F

from .hard_replay import HardReplayController, build_hard_replay_controller_from_yaml
from .tar import FAILURE_AWARE_REPLAY, TARController, TARSampleRef, build_tar_controller_from_yaml

_LOCALIZATION = "localization"
_BACKGROUND = "background"
_LOCALIZATION_CROP_SCALE = 2.0
_BACKGROUND_CROP_SCALE = 2.0
_MIN_VISIBLE_FRACTION = 0.5


class CocoDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, images_dir: str | Path, annotations_path: str | Path) -> None:
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.annotations_path = Path(annotations_path).expanduser().resolve()
        self.coco = COCO(str(self.annotations_path))
        self.image_ids = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int | TARSampleRef) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(index, TARSampleRef):
            return self._get_tar_sample(index)
        return self._get_full_sample(int(index))

    def _get_full_sample(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id, image, annotations = self._load_sample(int(index))
        width, height = image.size
        image_tensor = F.to_tensor(image)
        target = _build_target(image_id, width, height, annotations)
        return image_tensor, target

    def _get_tar_sample(self, sample: TARSampleRef) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if sample.replay_mode != FAILURE_AWARE_REPLAY:
            return self._get_full_sample(int(sample.dataset_index))
        if sample.failure_type not in {_LOCALIZATION, _BACKGROUND} or sample.bbox_xyxy is None:
            return self._get_full_sample(int(sample.dataset_index))

        image_id, image, annotations = self._load_sample(int(sample.dataset_index))
        width, height = image.size
        crop_scale = _LOCALIZATION_CROP_SCALE if sample.failure_type == _LOCALIZATION else _BACKGROUND_CROP_SCALE
        crop_box = _expanded_crop_box(sample.bbox_xyxy, width=width, height=height, scale=crop_scale)
        if crop_box is None:
            image_tensor = F.to_tensor(image)
            target = _build_target(image_id, width, height, annotations)
            return image_tensor, target

        force_gt_id = sample.gt_id if sample.failure_type == _LOCALIZATION else None
        cropped_annotations = _crop_annotations(
            annotations,
            crop_box=crop_box,
            min_visible_fraction=_MIN_VISIBLE_FRACTION,
            force_gt_id=force_gt_id,
        )
        if sample.failure_type == _LOCALIZATION and not cropped_annotations:
            image_tensor = F.to_tensor(image)
            target = _build_target(image_id, width, height, annotations)
            return image_tensor, target

        left, top, right, bottom = crop_box
        cropped_image = image.crop((left, top, right, bottom))
        crop_width, crop_height = cropped_image.size
        image_tensor = F.to_tensor(cropped_image)
        target = _build_target(image_id, crop_width, crop_height, cropped_annotations)
        return image_tensor, target

    def _load_sample(self, index: int) -> tuple[int, Image.Image, list[dict[str, Any]]]:
        image_id = self.image_ids[int(index)]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = _resolve_image_path(self.images_dir, image_info["file_name"])
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(ann_ids)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return int(image_id), image, annotations


def build_train_dataloaders(
    config: dict[str, Any],
    *,
    arch: str | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    module_config_paths: dict[str, str | Path] | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any] | None]:
    train_dataset = CocoDetectionDataset(
        config["data"]["train_images"],
        config["data"]["train_annotations"],
    )
    tar = _build_tar_controller(
        dataset=train_dataset,
        config=config,
        arch=arch,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        module_config_paths=module_config_paths,
    )
    hard_replay = None if tar is not None else _build_hard_replay_controller(
        dataset=train_dataset,
        config=config,
        arch=arch,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        module_config_paths=module_config_paths,
    )
    sampler = None
    replay_batch_sampler = tar.batch_sampler if tar is not None else None
    if replay_batch_sampler is None and hard_replay is not None:
        replay_batch_sampler = hard_replay.batch_sampler

    if distributed and replay_batch_sampler is None:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(config["loader"]["shuffle"]),
            seed=int(config["seed"]),
            drop_last=False,
        )

    train_loader = _build_loader(
        train_dataset,
        batch_size=config["loader"]["batch_size"],
        num_workers=config["loader"]["num_workers"],
        pin_memory=config["loader"]["pin_memory"],
        shuffle=config["loader"]["shuffle"] if sampler is None else False,
        sampler=sampler,
        batch_sampler=replay_batch_sampler,
    )
    if tar is not None:
        train_loader.tar_replay = tar
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
    sampler=None,
    batch_sampler=None,
) -> DataLoader[Any]:
    if batch_sampler is not None:
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )


def _build_hard_replay_controller(
    *,
    dataset: CocoDetectionDataset,
    config: dict[str, Any],
    arch: str | None,
    distributed: bool,
    rank: int,
    world_size: int,
    module_config_paths: dict[str, str | Path] | None,
) -> HardReplayController | None:
    if not module_config_paths:
        return None
    hard_replay_path = module_config_paths.get("hard_replay")
    if hard_replay_path is None:
        return None
    path = Path(hard_replay_path).expanduser().resolve()
    if not path.is_file():
        return None
    return build_hard_replay_controller_from_yaml(
        path,
        dataset=dataset,
        batch_size=int(config["loader"]["batch_size"]),
        shuffle=bool(config["loader"]["shuffle"]),
        seed=int(config["seed"]),
        rank=int(rank) if distributed else 0,
        world_size=int(world_size) if distributed else 1,
        arch=arch,
    )


def _build_tar_controller(
    *,
    dataset: CocoDetectionDataset,
    config: dict[str, Any],
    arch: str | None,
    distributed: bool,
    rank: int,
    world_size: int,
    module_config_paths: dict[str, str | Path] | None,
) -> TARController | None:
    if not module_config_paths:
        return None
    tar_path = module_config_paths.get("tar")
    if tar_path is None:
        return None
    path = Path(tar_path).expanduser().resolve()
    if not path.is_file():
        return None
    return build_tar_controller_from_yaml(
        path,
        dataset=dataset,
        batch_size=int(config["loader"]["batch_size"]),
        shuffle=bool(config["loader"]["shuffle"]),
        seed=int(config["seed"]),
        rank=int(rank) if distributed else 0,
        world_size=int(world_size) if distributed else 1,
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
        remaining_parts = image_path.parts[prefix_len:]
        if not remaining_parts:
            return None
        return Path(*remaining_parts)
    return None


def _expanded_crop_box(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int] | None:
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    box_width = x2 - x1
    box_height = y2 - y1
    crop_width = max(1.0, box_width * float(scale))
    crop_height = max(1.0, box_height * float(scale))
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    left = max(0.0, center_x - 0.5 * crop_width)
    top = max(0.0, center_y - 0.5 * crop_height)
    right = min(float(width), center_x + 0.5 * crop_width)
    bottom = min(float(height), center_y + 0.5 * crop_height)

    left_i = int(math.floor(left))
    top_i = int(math.floor(top))
    right_i = int(math.ceil(right))
    bottom_i = int(math.ceil(bottom))
    if right_i <= left_i or bottom_i <= top_i:
        return None
    return left_i, top_i, right_i, bottom_i


def _crop_annotations(
    annotations: list[dict[str, Any]],
    *,
    crop_box: tuple[int, int, int, int],
    min_visible_fraction: float,
    force_gt_id: str | None,
) -> list[dict[str, Any]]:
    left, top, right, bottom = crop_box
    cropped: list[dict[str, Any]] = []
    for annotation in annotations:
        x, y, width, height = annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x1 = float(x)
        y1 = float(y)
        x2 = x1 + max(0.0, float(width))
        y2 = y1 + max(0.0, float(height))
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if original_area <= 0.0:
            continue

        ix1 = max(x1, float(left))
        iy1 = max(y1, float(top))
        ix2 = min(x2, float(right))
        iy2 = min(y2, float(bottom))
        visible_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if visible_area <= 0.0:
            continue

        forced = force_gt_id is not None and str(annotation.get("id")) == str(force_gt_id)
        visible_fraction = visible_area / original_area
        if not forced and visible_fraction < float(min_visible_fraction):
            continue

        cropped_annotation = dict(annotation)
        cropped_annotation["bbox"] = [
            ix1 - float(left),
            iy1 - float(top),
            ix2 - ix1,
            iy2 - iy1,
        ]
        cropped_annotation["area"] = visible_area
        cropped.append(cropped_annotation)
    return cropped


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
    annotation_ids: list[int] = []

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
        try:
            annotation_ids.append(int(annotation.get("id", -1)))
        except (TypeError, ValueError):
            annotation_ids.append(-1)

    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area_tensor = torch.tensor(areas, dtype=torch.float32)
        crowd_tensor = torch.tensor(crowds, dtype=torch.int64)
        annotation_ids_tensor = torch.tensor(annotation_ids, dtype=torch.int64)
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)
        area_tensor = torch.zeros((0,), dtype=torch.float32)
        crowd_tensor = torch.zeros((0,), dtype=torch.int64)
        annotation_ids_tensor = torch.zeros((0,), dtype=torch.int64)

    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "image_id": torch.tensor(image_id, dtype=torch.int64),
        "area": area_tensor,
        "iscrowd": crowd_tensor,
        "annotation_ids": annotation_ids_tensor,
        "gt_ids": annotation_ids_tensor,
    }
