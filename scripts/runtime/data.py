from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as F

from .hard_replay import HardReplayController, ReplaySampleRef, build_hard_replay_controller_from_yaml


class CocoDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, images_dir: str | Path, annotations_path: str | Path) -> None:
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.annotations_path = Path(annotations_path).expanduser().resolve()
        self.coco = COCO(str(self.annotations_path))
        self.image_ids = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int | ReplaySampleRef) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(index, ReplaySampleRef):
            return self._get_replay_sample(index)

        image_id = self.image_ids[int(index)]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = _resolve_image_path(self.images_dir, image_info["file_name"])
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(ann_ids)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            image_tensor = F.to_tensor(image)

        target = _build_target(image_id, width, height, annotations)
        return image_tensor, target

    def _get_replay_sample(
        self,
        replay_ref: ReplaySampleRef,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if replay_ref.policy != "crop":
            return self[int(replay_ref.dataset_index)]

        image_id = self.image_ids[int(replay_ref.dataset_index)]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = _resolve_image_path(self.images_dir, image_info["file_name"])
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(ann_ids)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            crop_box = _build_replay_crop_box(
                replay_ref,
                width=width,
                height=height,
            )
            cropped = image.crop(crop_box)
            image_tensor = F.to_tensor(cropped)

        target = _build_crop_target(
            image_id=image_id,
            crop_box=crop_box,
            annotations=annotations,
            replay_ref=replay_ref,
        )
        return image_tensor, target


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
    hard_replay = _build_hard_replay_controller(
        dataset=train_dataset,
        config=config,
        arch=arch,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        module_config_paths=module_config_paths,
    )
    sampler = None
    if distributed and hard_replay is None:
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


def _build_replay_crop_box(
    replay_ref: ReplaySampleRef,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _clip_box(replay_ref.bbox_xyxy, width=width, height=height)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    rng = random.Random(int(replay_ref.seed))

    scale = max(float(replay_ref.crop.context_scale), 1e-6)
    if replay_ref.crop.context_scale_jitter > 0.0:
        jitter = rng.uniform(
            -float(replay_ref.crop.context_scale_jitter),
            float(replay_ref.crop.context_scale_jitter),
        )
        scale = max(1e-6, scale * (1.0 + jitter))

    crop_w = min(float(width), max(box_w * scale, float(replay_ref.crop.min_crop_size)))
    crop_h = min(float(height), max(box_h * scale, float(replay_ref.crop.min_crop_size)))
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    if replay_ref.crop.center_jitter > 0.0:
        center_x += rng.uniform(-1.0, 1.0) * float(replay_ref.crop.center_jitter) * box_w
        center_y += rng.uniform(-1.0, 1.0) * float(replay_ref.crop.center_jitter) * box_h

    left = _clamp_float(center_x - crop_w * 0.5, 0.0, max(0.0, float(width) - crop_w))
    top = _clamp_float(center_y - crop_h * 0.5, 0.0, max(0.0, float(height) - crop_h))
    right = min(float(width), left + crop_w)
    bottom = min(float(height), top + crop_h)

    left_i = int(math.floor(left))
    top_i = int(math.floor(top))
    right_i = int(math.ceil(right))
    bottom_i = int(math.ceil(bottom))
    right_i = min(int(width), max(left_i + 1, right_i))
    bottom_i = min(int(height), max(top_i + 1, bottom_i))
    return left_i, top_i, right_i, bottom_i


def _build_crop_target(
    *,
    image_id: int,
    crop_box: tuple[int, int, int, int],
    annotations: list[dict[str, Any]],
    replay_ref: ReplaySampleRef,
) -> dict[str, torch.Tensor]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    boxes: list[list[float]] = []
    labels: list[int] = []
    areas: list[float] = []
    crowds: list[int] = []
    annotation_ids: list[int] = []

    for annotation in annotations:
        original_box = _annotation_box_xyxy(annotation)
        if original_box is None:
            continue
        is_focus = _annotation_matches_replay_focus(annotation, original_box, replay_ref)
        intersection = _intersect_boxes(original_box, crop_box)
        if intersection is None:
            continue
        original_area = _box_area(original_box)
        if original_area <= 0.0:
            continue
        visible_ratio = _box_area(intersection) / original_area
        if is_focus:
            if visible_ratio < float(replay_ref.crop.focus_min_visible_ratio):
                continue
        else:
            if not replay_ref.crop.include_other_gt:
                continue
            if visible_ratio < float(replay_ref.crop.min_visible_ratio):
                continue

        local_box = [
            float(intersection[0] - crop_x1),
            float(intersection[1] - crop_y1),
            float(intersection[2] - crop_x1),
            float(intersection[3] - crop_y1),
        ]
        if local_box[2] <= local_box[0] or local_box[3] <= local_box[1]:
            continue

        boxes.append(local_box)
        labels.append(int(annotation["category_id"]))
        areas.append(float(_box_area(intersection)))
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

    focus_gt_id = _focus_gt_id(replay_ref)
    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "image_id": torch.tensor(image_id, dtype=torch.int64),
        "area": area_tensor,
        "iscrowd": crowd_tensor,
        "annotation_ids": annotation_ids_tensor,
        "gt_ids": annotation_ids_tensor,
        "is_replay_crop": torch.tensor(1, dtype=torch.int64),
        "replay_policy_id": torch.tensor(1, dtype=torch.int64),
        "replay_focus_gt_id": torch.tensor(focus_gt_id, dtype=torch.int64),
    }


def _annotation_box_xyxy(annotation: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x, y, w, h = bbox[:4]
    x1 = float(x)
    y1 = float(y)
    x2 = x1 + max(0.0, float(w))
    y2 = y1 + max(0.0, float(h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _annotation_matches_replay_focus(
    annotation: Mapping[str, Any],
    box: tuple[float, float, float, float],
    replay_ref: ReplaySampleRef,
) -> bool:
    ann_id = annotation.get("id")
    if replay_ref.gt_id is not None and ann_id is not None:
        if str(ann_id) == str(replay_ref.gt_id):
            return True
    if int(annotation.get("category_id", -1)) != int(replay_ref.class_id):
        return False
    return _box_iou(box, replay_ref.bbox_xyxy) >= 0.95


def _focus_gt_id(replay_ref: ReplaySampleRef) -> int:
    if replay_ref.gt_id is None:
        return -1
    try:
        return int(replay_ref.gt_id)
    except ValueError:
        return -1


def _clip_box(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        _clamp_float(float(x1), 0.0, float(width)),
        _clamp_float(float(y1), 0.0, float(height)),
        _clamp_float(float(x2), 0.0, float(width)),
        _clamp_float(float(y2), 0.0, float(height)),
    )


def _intersect_boxes(
    box: tuple[float, float, float, float],
    crop_box: tuple[float, float, float, float] | tuple[int, int, int, int],
) -> tuple[float, float, float, float] | None:
    x1 = max(float(box[0]), float(crop_box[0]))
    y1 = max(float(box[1]), float(crop_box[1]))
    x2 = min(float(box[2]), float(crop_box[2]))
    y2 = min(float(box[3]), float(crop_box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _box_iou(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> float:
    intersection = _intersect_boxes(lhs, rhs)
    if intersection is None:
        return 0.0
    union = _box_area(lhs) + _box_area(rhs) - _box_area(intersection)
    if union <= 0.0:
        return 0.0
    return _box_area(intersection) / union


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, float(value)))
