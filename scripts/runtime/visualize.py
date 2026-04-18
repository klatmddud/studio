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


# ─── training curves ──────────────────────────────────────────────────────────

def plot_loss_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    records = [r for r in history if "train" in r]
    if not records:
        return

    all_keys: set[str] = set()
    for r in records:
        all_keys.update(k for k in r["train"] if k not in _LOSS_SKIP)
    sorted_keys = sorted(all_keys, key=lambda k: (k != "loss", k))

    epochs = [r["epoch"] for r in records]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in sorted_keys:
        values = [r["train"].get(key, float("nan")) for r in records]
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
    records = [r for r in history if "val" in r]
    if not records:
        return

    epochs = [r["epoch"] for r in records]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in _MAP_KEYS:
        values = [r["val"].get(key, float("nan")) for r in records]
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
    records = [r for r in history if r.get("mdmb") is not None]
    if not records:
        return

    epochs = [r["epoch"] for r in records]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in _MDMB_KEYS:
        values = [r["mdmb"].get(key, 0) for r in records]
        ax.plot(epochs, values, label=label, linewidth=1.8, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax.set_title("MDMB: Missed Detection Bank Stats")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_dir / "figures" / "mdmb.png")


# ─── confusion matrix ─────────────────────────────────────────────────────────

def build_confusion_matrix(
    predictions: list[dict[str, Any]],
    data_loader,
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, list[str]]:
    """Build confusion matrix from COCO-format predictions and a CocoDetectionDataset.

    Returns (cm, class_names) where cm has shape [nc+1, nc+1] indexed as [pred, true].
    Index nc is background (FP column / FN row).
    """
    coco = data_loader.dataset.coco
    cat_ids = sorted(coco.cats.keys())
    class_names = [coco.cats[cid]["name"] for cid in cat_ids] + ["background"]
    cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    nc = len(cat_ids)
    bg = nc
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        preds_by_image[int(p["image_id"])].append(p)

    for img_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        gt_boxes: list[list[float]] = []
        gt_labels: list[int] = []
        for ann in anns:
            cid = ann["category_id"]
            if cid not in cat_to_idx:
                continue
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(cat_to_idx[cid])

        img_preds = sorted(preds_by_image.get(img_id, []), key=lambda p: -p["score"])
        pred_boxes: list[list[float]] = []
        pred_labels: list[int] = []
        for p in img_preds:
            x, y, w, h = p["bbox"]
            pred_boxes.append([x, y, x + w, y + h])
            pred_labels.append(cat_to_idx.get(p["category_id"], bg))

        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        if gt_boxes and pred_boxes:
            iou_mat = box_iou(
                torch.tensor(pred_boxes, dtype=torch.float32),
                torch.tensor(gt_boxes, dtype=torch.float32),
            ).numpy()  # [P, G]

            for pi in range(len(pred_boxes)):
                best_gi, best_iou = -1, iou_thresh
                for gi in range(len(gt_boxes)):
                    if not gt_matched[gi] and iou_mat[pi, gi] > best_iou:
                        best_iou = iou_mat[pi, gi]
                        best_gi = gi
                if best_gi >= 0:
                    gt_matched[best_gi] = True
                    pred_matched[pi] = True
                    cm[pred_labels[pi], gt_labels[best_gi]] += 1

        for gi, matched in enumerate(gt_matched):
            if not matched:
                cm[bg, gt_labels[gi]] += 1

        for pi, matched in enumerate(pred_matched):
            if not matched:
                cm[pred_labels[pi], bg] += 1

    return cm, class_names


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


# ─── MDMB per-class bar chart ─────────────────────────────────────────────────

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
    cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    class_names = [coco.cats[cid]["name"] for cid in cat_ids]

    counts = np.zeros(len(cat_ids), dtype=np.int64)
    for _, entries in mdmb.items():
        for entry in entries:
            idx = cat_to_idx.get(int(entry.class_id))
            if idx is not None:
                counts[idx] += 1

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


# ─── internal helpers ─────────────────────────────────────────────────────────

def _plot_single_cm(
    cm: np.ndarray,
    class_names: list[str],
    path: Path,
    title: str,
    normalize: bool,
) -> None:
    nc = len(class_names)
    cell_size = max(0.55, 8.0 / nc)
    fig_size = max(8, nc * cell_size)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Annotate only non-zero cells
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
    print(f"[visualize] saved → {path}")
