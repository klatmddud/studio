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
    ReplayIndex,
    ReplaySampleSpec,
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


class HardReplayDatasetWrapper(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    requires_fresh_workers_per_epoch = True

    def __init__(self, base_dataset: CocoDetectionDataset) -> None:
        self.base_dataset = base_dataset
        self.images_dir = base_dataset.images_dir
        self.annotations_path = base_dataset.annotations_path
        self.coco = base_dataset.coco
        self.image_ids = base_dataset.image_ids
        self._replay_index = ReplayIndex.empty(enabled=False)

    def __len__(self) -> int:
        return len(self.base_dataset) + len(self._replay_samples())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        base_size = len(self.base_dataset)
        if index < base_size:
            return self.base_dataset[index]

        crop_index = index - base_size
        replay_samples = self._replay_samples()
        if crop_index < 0 or crop_index >= len(replay_samples):
            raise IndexError(f"Hard Replay virtual index out of range: {crop_index}")
        replay_sample = replay_samples[crop_index]
        if replay_sample.kind == "copy_paste":
            return self._build_copy_paste_sample(replay_sample, sample_index=crop_index)
        return self._build_replay_crop_sample(replay_sample, sample_index=crop_index)

    def set_replay_index(self, replay_index: ReplayIndex) -> None:
        self._replay_index = replay_index

    def _replay_samples(self) -> list[ReplaySampleSpec]:
        return list(self._replay_index.replay_samples or self._replay_index.replay_crops)

    def _build_replay_crop_sample(
        self,
        replay_sample: ReplaySampleSpec,
        *,
        sample_index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        source_image_id = self.image_ids[replay_sample.dataset_index]
        image_info = self.coco.loadImgs([source_image_id])[0]
        image_path = _resolve_image_path(self.images_dir, image_info["file_name"])
        crop_x1, crop_y1, crop_x2, crop_y2 = replay_sample.crop_box_abs

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_width, crop_height = crop.size
            image_tensor = F.to_tensor(crop)

        ann_ids = self.coco.getAnnIds(imgIds=[source_image_id])
        annotations = self.coco.loadAnns(ann_ids)
        crop_annotations = _clip_annotations_to_crop(
            annotations,
            crop_box_abs=replay_sample.crop_box_abs,
        )

        synthetic_image_id = -(int(sample_index) + 1)
        target = _build_target(
            synthetic_image_id,
            crop_width,
            crop_height,
            crop_annotations,
        )
        source_box_crop = _shift_box_to_crop(
            replay_sample.source_bbox_abs,
            crop_box_abs=replay_sample.crop_box_abs,
        )
        box_weights = _weights_for_matching_box(
            target["boxes"],
            source_box_crop,
            weight=replay_sample.loss_weight,
        )
        _attach_replay_metadata(
            target,
            replay_sample,
            source_image_id=source_image_id,
            sample_index=sample_index,
            box_weights=box_weights,
        )
        return image_tensor, target

    def _build_copy_paste_sample(
        self,
        replay_sample: ReplaySampleSpec,
        *,
        sample_index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if replay_sample.target_dataset_index is None or replay_sample.paste_box_abs is None:
            raise ValueError("copy_paste replay sample requires target_dataset_index and paste_box_abs.")

        source_image_id = self.image_ids[replay_sample.dataset_index]
        target_image_id = self.image_ids[replay_sample.target_dataset_index]
        source_info = self.coco.loadImgs([source_image_id])[0]
        target_info = self.coco.loadImgs([target_image_id])[0]
        source_path = _resolve_image_path(self.images_dir, source_info["file_name"])
        target_path = _resolve_image_path(self.images_dir, target_info["file_name"])

        with Image.open(source_path) as source_image, Image.open(target_path) as target_image:
            source_image = source_image.convert("RGB")
            target_image = target_image.convert("RGB")
            object_crop = source_image.crop(replay_sample.source_bbox_abs)
            paste_x1, paste_y1, paste_x2, paste_y2 = replay_sample.paste_box_abs
            paste_width = paste_x2 - paste_x1
            paste_height = paste_y2 - paste_y1
            if object_crop.size != (paste_width, paste_height):
                object_crop = object_crop.resize((paste_width, paste_height))
            target_image.paste(object_crop, (paste_x1, paste_y1))
            width, height = target_image.size
            image_tensor = F.to_tensor(target_image)

        ann_ids = self.coco.getAnnIds(imgIds=[target_image_id])
        annotations = list(self.coco.loadAnns(ann_ids))
        pasted_annotation = {
            "id": -int(sample_index) - 1,
            "image_id": target_image_id,
            "category_id": replay_sample.class_id,
            "bbox": [
                float(replay_sample.paste_box_abs[0]),
                float(replay_sample.paste_box_abs[1]),
                float(replay_sample.paste_box_abs[2] - replay_sample.paste_box_abs[0]),
                float(replay_sample.paste_box_abs[3] - replay_sample.paste_box_abs[1]),
            ],
            "area": float(
                (replay_sample.paste_box_abs[2] - replay_sample.paste_box_abs[0])
                * (replay_sample.paste_box_abs[3] - replay_sample.paste_box_abs[1])
            ),
            "iscrowd": 0,
        }
        annotations.append(pasted_annotation)
        target = _build_target(-(int(sample_index) + 1), width, height, annotations)
        box_weights = torch.ones((target["boxes"].shape[0],), dtype=torch.float32)
        if box_weights.numel() > 0:
            box_weights[-1] = float(replay_sample.loss_weight)
        _attach_replay_metadata(
            target,
            replay_sample,
            source_image_id=source_image_id,
            sample_index=sample_index,
            box_weights=box_weights,
        )
        target["target_image_id"] = _source_image_id_tensor(target_image_id)
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
    if hard_replay is not None and hard_replay.config.object_replay.enabled:
        replay_dataset = HardReplayDatasetWrapper(train_dataset)
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


def _shift_box_to_crop(
    box_abs: tuple[int, int, int, int],
    *,
    crop_box_abs: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    crop_x1, crop_y1, _, _ = crop_box_abs
    return (
        float(box_abs[0] - crop_x1),
        float(box_abs[1] - crop_y1),
        float(box_abs[2] - crop_x1),
        float(box_abs[3] - crop_y1),
    )


def _weights_for_matching_box(
    boxes: torch.Tensor,
    match_box: tuple[float, float, float, float],
    *,
    weight: float,
) -> torch.Tensor:
    weights = torch.ones((boxes.shape[0],), dtype=torch.float32)
    if boxes.numel() == 0:
        return weights
    match = torch.tensor(match_box, dtype=boxes.dtype).reshape(1, 4)
    ious = _box_iou_tensor(boxes.to(dtype=torch.float32), match.to(dtype=torch.float32)).flatten()
    if ious.numel() == 0:
        return weights
    best_index = int(torch.argmax(ious).item())
    if float(ious[best_index].item()) > 0.0:
        weights[best_index] = float(weight)
    return weights


def _box_iou_tensor(boxes: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0 or query.numel() == 0:
        return boxes.new_zeros((boxes.shape[0], query.shape[0]))
    inter_x1 = torch.maximum(boxes[:, None, 0], query[None, :, 0])
    inter_y1 = torch.maximum(boxes[:, None, 1], query[None, :, 1])
    inter_x2 = torch.minimum(boxes[:, None, 2], query[None, :, 2])
    inter_y2 = torch.minimum(boxes[:, None, 3], query[None, :, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    query_area = (query[:, 2] - query[:, 0]).clamp(min=0) * (
        query[:, 3] - query[:, 1]
    ).clamp(min=0)
    union = boxes_area[:, None] + query_area[None, :] - intersection
    return intersection / union.clamp(min=1e-6)


def _attach_replay_metadata(
    target: dict[str, Any],
    replay_sample: ReplaySampleSpec,
    *,
    source_image_id: Any,
    sample_index: int,
    box_weights: torch.Tensor,
) -> None:
    target["is_replay"] = torch.tensor(True, dtype=torch.bool)
    target["source_image_id"] = _source_image_id_tensor(source_image_id)
    target["replay_kind"] = replay_sample.kind
    target["replay_gt_uid"] = replay_sample.gt_uid
    target["replay_pair_id"] = replay_sample.pair_id or ""
    target["replay_role"] = replay_sample.role or ""
    target["replay_failure_type"] = replay_sample.failure_type
    target["replay_mode"] = replay_sample.mode
    target["replay_severity"] = torch.tensor(replay_sample.severity, dtype=torch.float32)
    target["replay_loss_weight"] = torch.tensor(replay_sample.loss_weight, dtype=torch.float32)
    target["replay_sample_index"] = torch.tensor(int(sample_index), dtype=torch.int64)
    target["replay_box_weights"] = box_weights.to(dtype=torch.float32)
    target["replay_cls_box_weights"] = _component_box_weights(
        box_weights,
        replay_sample.cls_loss_weight,
    )
    target["replay_reg_box_weights"] = _component_box_weights(
        box_weights,
        replay_sample.reg_loss_weight,
    )
    target["replay_ctr_box_weights"] = _component_box_weights(
        box_weights,
        replay_sample.ctr_loss_weight,
    )


def _component_box_weights(box_weights: torch.Tensor, component_weight: float) -> torch.Tensor:
    if box_weights.numel() == 0:
        return box_weights.to(dtype=torch.float32)
    result = torch.ones_like(box_weights, dtype=torch.float32)
    hard_mask = box_weights > 1.0
    result[hard_mask] = float(component_weight)
    return result


def _source_image_id_tensor(image_id: Any) -> torch.Tensor:
    try:
        value = int(image_id)
    except (TypeError, ValueError):
        value = abs(hash(str(image_id))) % (2**31)
    return torch.tensor(value, dtype=torch.int64)
