from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
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

VOC_METRIC_NAMES = (
    "voc_mAP_50",
    "voc_mAP_50_integral",
)


def evaluate_coco_detection(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool,
    log_interval: int = 0,
    stage_label: str = "eval",
    epoch_index: int | None = None,
    total_epochs: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    dataset = data_loader.dataset
    if not hasattr(dataset, "coco"):
        raise TypeError("COCO evaluation requires a dataset with a .coco attribute.")

    model_was_training = model.training
    model.eval()

    predictions: list[dict[str, Any]] = []
    total_steps = len(data_loader)
    start_time = time.perf_counter()
    processed_images = 0
    with torch.inference_mode():
        for step, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = _to_image_id(target["image_id"])
                predictions.extend(_to_coco_predictions(image_id, output))

            processed_images += len(images)

            should_log = log_interval > 0 and (step % log_interval == 0 or step == total_steps)
            if should_log:
                elapsed = time.perf_counter() - start_time
                avg_step_time = elapsed / max(step, 1)
                remaining_steps = max(total_steps - step, 0)
                eta_label = "epoch_eta" if epoch_index is not None and total_epochs is not None else "eval_eta"
                prefix = _build_progress_prefix(
                    stage_label=stage_label,
                    step=step,
                    total_steps=total_steps,
                    eta_label=eta_label,
                    eta_seconds=avg_step_time * remaining_steps,
                    epoch_index=epoch_index,
                    total_epochs=total_epochs,
                )
                print(
                    f"{prefix} images={processed_images} predictions={len(predictions)}"
                )

    metrics = _compute_box_metrics(dataset.coco, predictions)
    metrics["num_images"] = float(len(dataset))
    metrics["num_predictions"] = float(len(predictions))

    if model_was_training:
        model.train()

    return metrics, predictions


def evaluate_voc_detection(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool,
    log_interval: int = 0,
    stage_label: str = "eval",
    epoch_index: int | None = None,
    total_epochs: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    dataset = data_loader.dataset
    if not hasattr(dataset, "coco"):
        raise TypeError("VOC evaluation requires a dataset with a .coco attribute.")

    model_was_training = model.training
    model.eval()

    predictions: list[dict[str, Any]] = []
    total_steps = len(data_loader)
    start_time = time.perf_counter()
    processed_images = 0
    with torch.inference_mode():
        for step, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = _to_image_id(target["image_id"])
                predictions.extend(_to_coco_predictions(image_id, output))

            processed_images += len(images)

            should_log = log_interval > 0 and (step % log_interval == 0 or step == total_steps)
            if should_log:
                elapsed = time.perf_counter() - start_time
                avg_step_time = elapsed / max(step, 1)
                remaining_steps = max(total_steps - step, 0)
                eta_label = "epoch_eta" if epoch_index is not None and total_epochs is not None else "eval_eta"
                prefix = _build_progress_prefix(
                    stage_label=stage_label,
                    step=step,
                    total_steps=total_steps,
                    eta_label=eta_label,
                    eta_seconds=avg_step_time * remaining_steps,
                    epoch_index=epoch_index,
                    total_epochs=total_epochs,
                )
                print(
                    f"{prefix} images={processed_images} predictions={len(predictions)}"
                )

    metrics = _compute_voc_box_metrics(dataset.coco, predictions)
    metrics["num_images"] = float(len(dataset))
    metrics["num_predictions"] = float(len(predictions))

    if model_was_training:
        model.train()

    return metrics, predictions


def evaluate_detection(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool,
    metrics_type: str,
    log_interval: int = 0,
    stage_label: str = "eval",
    epoch_index: int | None = None,
    total_epochs: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if metrics_type == "coco_detection":
        return evaluate_coco_detection(
            model=model,
            data_loader=data_loader,
            device=device,
            amp=amp,
            log_interval=log_interval,
            stage_label=stage_label,
            epoch_index=epoch_index,
            total_epochs=total_epochs,
        )
    if metrics_type == "voc_detection":
        return evaluate_voc_detection(
            model=model,
            data_loader=data_loader,
            device=device,
            amp=amp,
            log_interval=log_interval,
            stage_label=stage_label,
            epoch_index=epoch_index,
            total_epochs=total_epochs,
        )
    raise ValueError(f"Unsupported metrics.type: {metrics_type!r}")


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


def _compute_voc_box_metrics(
    coco_gt,
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    category_ids = sorted(coco_gt.cats.keys())
    predictions_by_category: dict[int, list[dict[str, Any]]] = {
        category_id: [] for category_id in category_ids
    }
    for prediction in predictions:
        category_id = int(prediction["category_id"])
        if category_id in predictions_by_category:
            predictions_by_category[category_id].append(prediction)

    ap_11_values: list[float] = []
    ap_integral_values: list[float] = []
    metrics: dict[str, float] = {}

    for category_id in category_ids:
        category_name = str(coco_gt.cats[category_id].get("name", category_id))
        records, positive_count = _build_voc_class_records(coco_gt, category_id)
        ap_11, ap_integral = _evaluate_voc_class(
            records,
            positive_count=positive_count,
            predictions=predictions_by_category[category_id],
            iou_threshold=float(iou_threshold),
        )
        metrics[f"voc_AP_50_{_metric_name_slug(category_name)}"] = ap_11
        if positive_count > 0:
            ap_11_values.append(ap_11)
            ap_integral_values.append(ap_integral)

    metrics["voc_mAP_50"] = float(np.mean(ap_11_values)) if ap_11_values else 0.0
    metrics["voc_mAP_50_integral"] = (
        float(np.mean(ap_integral_values)) if ap_integral_values else 0.0
    )
    metrics["voc_num_classes"] = float(len(ap_11_values))
    return metrics


def _build_voc_class_records(coco_gt, category_id: int) -> tuple[dict[int, dict[str, Any]], int]:
    records: dict[int, dict[str, Any]] = {}
    positive_count = 0
    for image_id in coco_gt.imgs:
        ann_ids = coco_gt.getAnnIds(imgIds=[image_id], catIds=[category_id])
        annotations = coco_gt.loadAnns(ann_ids)
        boxes: list[list[float]] = []
        difficult: list[bool] = []
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            boxes.append([x, y, x + width, y + height])
            is_difficult = bool(annotation.get("difficult", 0)) or bool(
                annotation.get("iscrowd", 0)
            )
            difficult.append(is_difficult)
            if not is_difficult:
                positive_count += 1
        records[int(image_id)] = {
            "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
            "difficult": np.asarray(difficult, dtype=bool),
            "detected": np.zeros(len(boxes), dtype=bool),
        }
    return records, positive_count


def _evaluate_voc_class(
    records: dict[int, dict[str, Any]],
    *,
    positive_count: int,
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[float, float]:
    if positive_count <= 0:
        return 0.0, 0.0

    sorted_predictions = sorted(
        predictions,
        key=lambda prediction: float(prediction.get("score", 0.0)),
        reverse=True,
    )
    tp = np.zeros(len(sorted_predictions), dtype=np.float64)
    fp = np.zeros(len(sorted_predictions), dtype=np.float64)

    for index, prediction in enumerate(sorted_predictions):
        record = records.get(int(prediction["image_id"]))
        if record is None:
            fp[index] = 1.0
            continue

        boxes = record["boxes"]
        if boxes.size == 0:
            fp[index] = 1.0
            continue

        pred_box = _coco_bbox_to_xyxy(prediction["bbox"])
        overlaps = _box_iou_np(pred_box, boxes)
        max_index = int(np.argmax(overlaps))
        max_overlap = float(overlaps[max_index])

        if max_overlap < float(iou_threshold):
            fp[index] = 1.0
            continue
        if bool(record["difficult"][max_index]):
            continue
        if not bool(record["detected"][max_index]):
            tp[index] = 1.0
            record["detected"][max_index] = True
        else:
            fp[index] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    recall = tp / float(positive_count)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return _voc_ap_11_point(recall, precision), _voc_ap_integral(recall, precision)


def _coco_bbox_to_xyxy(bbox: list[float]) -> np.ndarray:
    x, y, width, height = [float(value) for value in bbox]
    return np.asarray([x, y, x + width, y + height], dtype=np.float64)


def _box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])
    intersections = np.maximum(x_max - x_min, 0.0) * np.maximum(y_max - y_min, 0.0)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1],
        0.0,
    )
    unions = np.maximum(box_area + boxes_area - intersections, np.finfo(np.float64).eps)
    return intersections / unions


def _voc_ap_11_point(recall: np.ndarray, precision: np.ndarray) -> float:
    ap = 0.0
    for threshold in np.arange(0.0, 1.1, 0.1):
        if np.any(recall >= threshold):
            ap += float(np.max(precision[recall >= threshold]))
    return ap / 11.0


def _voc_ap_integral(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    change_indices = np.where(mrec[1:] != mrec[:-1])[0]
    areas = (mrec[change_indices + 1] - mrec[change_indices]) * mpre[
        change_indices + 1
    ]
    return float(np.sum(areas))


def _metric_name_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value.lower()
    ).strip("_")


def _to_json(payload: list[dict[str, Any]]) -> str:
    import json

    return json.dumps(payload, indent=2)


def _build_progress_prefix(
    stage_label: str,
    step: int,
    total_steps: int,
    eta_label: str,
    eta_seconds: float,
    epoch_index: int | None,
    total_epochs: int | None,
) -> str:
    parts = [f"[{stage_label}]"]
    if epoch_index is not None and total_epochs is not None:
        parts.append(f"epoch {epoch_index + 1}/{total_epochs}")
    parts.append(f"step {step}/{total_steps}")
    parts.append(f"{eta_label}={_format_eta(eta_seconds)}")
    return " ".join(parts)


def _format_eta(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
