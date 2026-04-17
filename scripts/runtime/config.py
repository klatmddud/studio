from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from .metrics import BOX_METRIC_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "amp": True,
    "output_dir": "runs/train",
    "data": {
        "train_images": "data/coco/train2017",
        "train_annotations": "data/coco/annotations/instances_train2017.json",
        "val_images": "data/coco/val2017",
        "val_annotations": "data/coco/annotations/instances_val2017.json",
    },
    "loader": {
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,
    },
    "optimizer": {
        "name": "sgd",
        "lr": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nesterov": False,
    },
    "scheduler": {
        "name": "multistep",
        "milestones": [8, 11],
        "gamma": 0.1,
    },
    "train": {
        "epochs": 12,
        "grad_clip_norm": None,
        "log_interval": 20,
        "eval_every_epochs": 1,
    },
    "checkpoint": {
        "dir": "checkpoints",
        "resume": None,
        "save_last": True,
        "save_best": True,
        "monitor": "bbox_mAP_50_95",
        "mode": "max",
    },
    "metrics": {
        "type": "coco_detection",
        "iou_types": ["bbox"],
        "primary": "bbox_mAP_50_95",
    },
}

EVAL_DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "amp": False,
    "output_dir": "runs/eval",
    "data": {
        "images": "data/coco/val2017",
        "annotations": "data/coco/annotations/instances_val2017.json",
    },
    "loader": {
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": False,
    },
    "checkpoint": {
        "path": "runs/train/checkpoints/best.pt",
    },
    "metrics": {
        "type": "coco_detection",
        "iou_types": ["bbox"],
        "primary": "bbox_mAP_50_95",
    },
    "eval": {
        "log_interval": 20,
        "save_predictions": False,
        "predictions_path": None,
    },
}


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML config must contain a mapping at the top level: {config_path}")
    return data


def dump_yaml_file(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_device(raw: str) -> torch.device:
    name = raw.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS was requested but is not available in this environment.")
    return device


def load_runtime_config(path: str | Path, mode: str) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    raw = load_yaml_file(config_path)

    if mode == "train":
        config = deep_merge(TRAIN_DEFAULTS, raw)
        _resolve_train_paths(config)
        _validate_train_config(config)
    elif mode == "eval":
        config = deep_merge(EVAL_DEFAULTS, raw)
        _resolve_eval_paths(config)
        _validate_eval_config(config)
    else:
        raise ValueError(f"Unsupported runtime mode: {mode!r}")

    config["_config_path"] = str(config_path)
    config["_mode"] = mode
    return config, config_path


def _resolve_train_paths(config: dict[str, Any]) -> None:
    output_dir = _resolve_path(config["output_dir"], PROJECT_ROOT)
    config["output_dir"] = str(output_dir)

    data = config["data"]
    for key in ("train_images", "train_annotations", "val_images", "val_annotations"):
        if data.get(key):
            data[key] = str(_resolve_path(data[key], PROJECT_ROOT))

    checkpoint = config["checkpoint"]
    checkpoint["dir"] = str(_resolve_path(checkpoint["dir"], output_dir))
    if checkpoint.get("resume"):
        checkpoint["resume"] = str(_resolve_path(checkpoint["resume"], PROJECT_ROOT))


def _resolve_eval_paths(config: dict[str, Any]) -> None:
    output_dir = _resolve_path(config["output_dir"], PROJECT_ROOT)
    config["output_dir"] = str(output_dir)

    data = config["data"]
    for key in ("images", "annotations"):
        if data.get(key):
            data[key] = str(_resolve_path(data[key], PROJECT_ROOT))

    checkpoint = config["checkpoint"]
    if checkpoint.get("path"):
        checkpoint["path"] = str(_resolve_path(checkpoint["path"], PROJECT_ROOT))

    eval_config = config["eval"]
    if eval_config.get("predictions_path"):
        eval_config["predictions_path"] = str(
            _resolve_path(eval_config["predictions_path"], output_dir)
        )


def _validate_train_config(config: dict[str, Any]) -> None:
    _validate_common_config(config)

    data = config["data"]
    required_pairs = (
        ("train_images", "train_annotations"),
        ("val_images", "val_annotations"),
    )
    for image_key, annotation_key in required_pairs:
        image_path = data.get(image_key)
        annotation_path = data.get(annotation_key)
        if bool(image_path) != bool(annotation_path):
            raise ValueError(f"{image_key} and {annotation_key} must be set together.")

    if not data.get("train_images") or not data.get("train_annotations"):
        raise ValueError("train_images and train_annotations are required for training.")

    if config["train"]["epochs"] < 1:
        raise ValueError("train.epochs must be >= 1.")
    if config["train"]["eval_every_epochs"] < 1:
        raise ValueError("train.eval_every_epochs must be >= 1.")

    checkpoint = config["checkpoint"]
    metrics = config["metrics"]
    if checkpoint["mode"] not in {"max", "min"}:
        raise ValueError("checkpoint.mode must be either 'max' or 'min'.")
    if checkpoint["monitor"] not in BOX_METRIC_NAMES:
        raise ValueError(
            f"checkpoint.monitor must be one of {list(BOX_METRIC_NAMES)}; "
            f"got {checkpoint['monitor']!r}."
        )
    if metrics["primary"] not in BOX_METRIC_NAMES:
        raise ValueError(
            f"metrics.primary must be one of {list(BOX_METRIC_NAMES)}; "
            f"got {metrics['primary']!r}."
        )
    if checkpoint["save_best"] and (
        not data.get("val_images") or not data.get("val_annotations")
    ):
        raise ValueError("Validation data is required when checkpoint.save_best is true.")


def _validate_eval_config(config: dict[str, Any]) -> None:
    _validate_common_config(config)

    data = config["data"]
    if not data.get("images") or not data.get("annotations"):
        raise ValueError("data.images and data.annotations are required for evaluation.")

    checkpoint_path = config["checkpoint"].get("path")
    if not checkpoint_path:
        raise ValueError(
            "checkpoint.path is required for evaluation to avoid scoring a randomly initialized model."
        )


def _validate_common_config(config: dict[str, Any]) -> None:
    loader = config["loader"]
    if loader["batch_size"] < 1:
        raise ValueError("loader.batch_size must be >= 1.")
    if loader["num_workers"] < 0:
        raise ValueError("loader.num_workers must be >= 0.")

    metrics = config["metrics"]
    if metrics["type"] != "coco_detection":
        raise ValueError("Only metrics.type='coco_detection' is supported right now.")

    iou_types = tuple(metrics.get("iou_types", []))
    if iou_types != ("bbox",):
        raise ValueError("Only metrics.iou_types=['bbox'] is supported right now.")


def _resolve_path(raw_path: str | Path, base_dir: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()
