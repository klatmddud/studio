from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


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


def build_train_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader[Any], DataLoader[Any] | None]:
    train_dataset = CocoDetectionDataset(
        config["data"]["train_images"],
        config["data"]["train_annotations"],
    )
    train_loader = _build_loader(
        train_dataset,
        batch_size=config["loader"]["batch_size"],
        num_workers=config["loader"]["num_workers"],
        pin_memory=config["loader"]["pin_memory"],
        shuffle=config["loader"]["shuffle"],
    )

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
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )


def _resolve_image_path(images_dir: Path, file_name: str) -> Path:
    image_path = Path(file_name)
    if not image_path.is_absolute():
        image_path = images_dir / image_path
    return image_path.resolve()


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
