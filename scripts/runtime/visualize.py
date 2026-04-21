from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torchvision.ops import box_iou

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


_LOSS_SKIP = frozenset({"lr", "epoch_time_sec"})

_MAP_KEYS = [
    ("bbox_mAP_50_95", "mAP50:95"),
    ("bbox_mAP_50", "mAP50"),
    ("bbox_mAP_75", "mAP75"),
]

_MDMB_KEYS = [
    ("num_entries", "Miss entries"),
    ("num_images", "Miss images"),
]


def plot_loss_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    records = [record for record in history if "train" in record]
    if not records:
        return

    all_keys: set[str] = set()
    for record in records:
        all_keys.update(key for key in record["train"] if key not in _LOSS_SKIP)
    sorted_keys = sorted(all_keys, key=lambda key: (key != "loss", key))

    epochs = [record["epoch"] for record in records]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in sorted_keys:
        values = [record["train"].get(key, float("nan")) for record in records]
        label = "total_loss" if key == "loss" else key
        ax.plot(epochs, values, label=label, linewidth=2.0 if key == "loss" else 1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir / "figures" / "loss.png")


def plot_map_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    records = [record for record in history if "val" in record]
    if not records:
        return

    epochs = [record["epoch"] for record in records]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in _MAP_KEYS:
        values = [record["val"].get(key, float("nan")) for record in records]
        ax.plot(epochs, values, label=label, linewidth=1.8, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir / "figures" / "map.png")


def plot_mdmb_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    records = [record for record in history if record.get("mdmb") is not None]
    if not records:
        return

    epochs = [record["epoch"] for record in records]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in _MDMB_KEYS:
        values = [record["mdmb"].get(key, 0) for record in records]
        ax.plot(epochs, values, label=label, linewidth=1.8, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax.set_title("MDMB: Missed Detection Bank Stats")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir / "figures" / "mdmb.png")


def build_confusion_matrix(
    predictions: list[dict[str, Any]],
    data_loader,
    iou_thresh: float = 0.45,
    conf_thresh: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
    """Build a YOLO-compatible confusion matrix from COCO-format predictions.

    Returns (cm, class_names) where cm has shape [nc+1, nc+1] indexed as [pred, true].
    Index nc is background (FP column / FN row).
    """
    coco = data_loader.dataset.coco
    cat_ids = sorted(coco.cats.keys())
    class_names = [coco.cats[cid]["name"] for cid in cat_ids] + ["background"]
    cat_to_idx = {cid: index for index, cid in enumerate(cat_ids)}
    num_classes = len(cat_ids)
    background = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        preds_by_image[int(prediction["image_id"])].append(prediction)

    for image_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        annotations = coco.loadAnns(ann_ids)

        gt_boxes: list[list[float]] = []
        gt_labels: list[int] = []
        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id not in cat_to_idx:
                continue
            x, y, width, height = annotation["bbox"]
            gt_boxes.append([x, y, x + width, y + height])
            gt_labels.append(cat_to_idx[category_id])

        image_predictions = sorted(
            (
                prediction
                for prediction in preds_by_image.get(image_id, [])
                if float(prediction.get("score", 0.0)) > float(conf_thresh)
            ),
            key=lambda prediction: -float(prediction.get("score", 0.0)),
        )
        pred_boxes: list[list[float]] = []
        pred_labels: list[int] = []
        for prediction in image_predictions:
            x, y, width, height = prediction["bbox"]
            pred_boxes.append([x, y, x + width, y + height])
            pred_labels.append(cat_to_idx.get(prediction["category_id"], background))

        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        if gt_boxes and pred_boxes:
            iou_matrix = box_iou(
                torch.tensor(gt_boxes, dtype=torch.float32),
                torch.tensor(pred_boxes, dtype=torch.float32),
            ).numpy()  # [G, P]

            for gt_index, pred_index in _match_yolo_confusion_pairs(iou_matrix, iou_thresh):
                gt_matched[gt_index] = True
                pred_matched[pred_index] = True
                cm[pred_labels[pred_index], gt_labels[gt_index]] += 1

        for gt_index, matched in enumerate(gt_matched):
            if not matched:
                cm[background, gt_labels[gt_index]] += 1

        for pred_index, matched in enumerate(pred_matched):
            if not matched:
                cm[pred_labels[pred_index], background] += 1

    return cm, class_names


def _match_yolo_confusion_pairs(
    iou_matrix: np.ndarray,
    iou_thresh: float,
) -> np.ndarray:
    gt_indices, pred_indices = np.where(iou_matrix > float(iou_thresh))
    if gt_indices.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    matches = np.stack(
        (
            gt_indices,
            pred_indices,
            iou_matrix[gt_indices, pred_indices],
        ),
        axis=1,
    )
    if matches.shape[0] > 1:
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return matches[:, :2].astype(np.int64)


def plot_confusion_matrices(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"

    _plot_single_cm(
        cm,
        class_names,
        figures_dir / "confusion_matrix.png",
        title="Confusion Matrix",
        normalize=False,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_norm = np.where(col_sums > 0, cm / col_sums.clip(min=1), 0.0)

    _plot_single_cm(
        cm_norm,
        class_names,
        figures_dir / "confusion_matrix_normalized.png",
        title="Confusion Matrix Normalized",
        normalize=True,
    )


def plot_mdmb_per_class(
    model: torch.nn.Module,
    data_loader,
    output_dir: Path,
) -> None:
    mdmb = getattr(model, "mdmb", None)
    if mdmb is None:
        return

    coco = data_loader.dataset.coco
    cat_ids = sorted(coco.cats.keys())
    cat_to_idx = {cid: index for index, cid in enumerate(cat_ids)}
    class_names = [coco.cats[cid]["name"] for cid in cat_ids]

    counts = np.zeros(len(cat_ids), dtype=np.int64)
    for _, entries in mdmb.items():
        for entry in entries:
            index = cat_to_idx.get(int(entry.class_id))
            if index is not None:
                counts[index] += 1

    if counts.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), 6))
    x = np.arange(len(class_names))
    ax.bar(x, counts, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("MDMB: Miss Detections per Class (best.pt)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir / "figures" / "mdmb_per_class.png")


def _plot_single_cm(
    cm: np.ndarray,
    class_names: list[str],
    path: Path,
    title: str,
    normalize: bool,
) -> None:
    num_classes = len(class_names)
    cell_size = max(0.55, 8.0 / num_classes)
    fig_size = max(8, num_classes * cell_size)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    if normalize:
        annot = np.where(cm > 0, np.round(cm, 2).astype(str), "")
    else:
        annot = np.where(cm > 0, cm.astype(str), "")

    sns.heatmap(
        cm.astype(float),
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.0,
        vmin=0,
        vmax=1.0 if normalize else None,
    )
    ax.set_xlabel("True", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, path)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved -> {path}")
