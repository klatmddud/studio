from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import torch
from pycocotools.cocoeval import COCOeval

BOX_METRIC_NAMES = (
    "bbox_mAP_50_95",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "bbox_mAP_small",
    "bbox_mAP_medium",
    "bbox_mAP_large",
    "bbox_mAR_1",
    "bbox_mAR_10",
    "bbox_mAR_100",
    "bbox_mAR_small",
    "bbox_mAR_medium",
    "bbox_mAR_large",
)


def evaluate_coco_detection(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    dataset = data_loader.dataset
    if not hasattr(dataset, "coco"):
        raise TypeError("COCO evaluation requires a dataset with a .coco attribute.")

    model_was_training = model.training
    model.eval()

    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = _to_image_id(target["image_id"])
                predictions.extend(_to_coco_predictions(image_id, output))

    metrics = _compute_box_metrics(dataset.coco, predictions)
    metrics["num_images"] = float(len(dataset))
    metrics["num_predictions"] = float(len(predictions))

    if model_was_training:
        model.train()

    return metrics, predictions


def save_predictions(path: str | Path, predictions: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_to_json(predictions), encoding="utf-8")


def _to_image_id(raw: torch.Tensor | int) -> int:
    if isinstance(raw, torch.Tensor):
        if raw.ndim == 0:
            return int(raw.item())
        return int(raw.reshape(-1)[0].item())
    return int(raw)


def _to_coco_predictions(
    image_id: int,
    output: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()

    results: list[dict[str, Any]] = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        results.append(
            {
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [
                    float(x1),
                    float(y1),
                    float(max(0.0, x2 - x1)),
                    float(max(0.0, y2 - y1)),
                ],
                "score": float(score.item()),
            }
        )
    return results


def _compute_box_metrics(coco_gt, predictions: list[dict[str, Any]]) -> dict[str, float]:
    if not predictions:
        return {name: 0.0 for name in BOX_METRIC_NAMES}

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()

    return {
        name: float(value)
        for name, value in zip(BOX_METRIC_NAMES, evaluator.stats, strict=True)
    }


def _to_json(payload: list[dict[str, Any]]) -> str:
    import json

    return json.dumps(payload, indent=2)
