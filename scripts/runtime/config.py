from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any

import torch
import yaml
from dotenv import load_dotenv

from .metrics import BOX_METRIC_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-|-)([^}]*))?\}")
_DOTENV_LOADED = False
_DATA_ENV_SUFFIXES = (
    "TRAIN_ANNOTATIONS",
    "TRAIN_IMAGES",
    "VAL_ANNOTATIONS",
    "VAL_IMAGES",
    "ANNOTATIONS",
    "IMAGES",
)
_DATA_PATH_KEYS = {
    "train_images",
    "train_annotations",
    "val_images",
    "val_annotations",
    "images",
    "annotations",
}

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
        "hard_crop_second_view": {
            "enabled": False,
            "start_epoch": 3,
            "loss_weight": 0.25,
            "max_views_per_image": 1,
            "max_views_per_batch": 8,
            "crop_scale_min": 1.6,
            "crop_scale_max": 2.4,
            "jitter": 0.15,
            "min_crop_size": 96,
            "include_other_gt": True,
            "min_box_size": 2.0,
            "target_transitions": ["FN_LOC->FN_LOC", "TP->FN_LOC"],
            "persistent_states": ["FN_LOC"],
            "min_observations": 2,
            "min_fn_streak": 2,
        },
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
    _load_project_dotenv()
    config_path = Path(path).expanduser().resolve()
    data = _read_yaml_file(config_path)
    data = _expand_env_placeholders(data, source=config_path)
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
    if device.type == "cuda" and device.index is not None:
        _validate_cuda_index(device.index)
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS was requested but is not available in this environment.")
    return device


def resolve_devices(raw: str | Sequence[str]) -> list[torch.device]:
    values = _coerce_device_values(raw)
    if len(values) == 1:
        return [resolve_device(values[0])]

    devices = [torch.device(value) for value in values]
    if any(device.type != "cuda" for device in devices):
        raise ValueError("Multiple --device values are only supported for CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("Multiple CUDA devices were requested but CUDA is not available.")

    indices: list[int] = []
    for device in devices:
        if device.index is None:
            raise ValueError("Multi-GPU training requires explicit device ids like cuda:0 cuda:1.")
        _validate_cuda_index(device.index)
        indices.append(int(device.index))

    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate CUDA devices are not allowed: {values}")
    return devices


def format_device_name(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return str(device)


def load_runtime_config(
    path: str | Path,
    mode: str,
    dataset: str | None = None,
) -> tuple[dict[str, Any], Path]:
    _load_project_dotenv()
    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml_file(config_path)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML config must contain a mapping at the top level: {config_path}")
    normalized_dataset = None
    if dataset:
        normalized_dataset = _normalize_dataset_name(dataset)
        raw = _apply_dataset_env_overrides(raw, dataset=normalized_dataset, source=config_path)
    raw = _expand_env_placeholders(raw, source=config_path)

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
    if normalized_dataset is not None:
        config["_dataset"] = normalized_dataset
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
    _validate_hard_crop_second_view_config(config["train"].get("hard_crop_second_view", {}))

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


def _validate_hard_crop_second_view_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise ValueError("train.hard_crop_second_view must be a mapping.")
    if not bool(config.get("enabled", False)):
        return

    for key in (
        "start_epoch",
        "max_views_per_image",
        "max_views_per_batch",
        "min_observations",
        "min_fn_streak",
    ):
        if int(config.get(key, 0)) < 0:
            raise ValueError(f"train.hard_crop_second_view.{key} must be >= 0.")
    for key in (
        "loss_weight",
        "crop_scale_min",
        "crop_scale_max",
        "jitter",
        "min_crop_size",
        "min_box_size",
    ):
        if float(config.get(key, 0.0)) < 0.0:
            raise ValueError(f"train.hard_crop_second_view.{key} must be >= 0.")
    if int(config.get("max_views_per_image", 0)) < 1:
        raise ValueError("train.hard_crop_second_view.max_views_per_image must be >= 1 when enabled.")
    if int(config.get("max_views_per_batch", 0)) < 1:
        raise ValueError("train.hard_crop_second_view.max_views_per_batch must be >= 1 when enabled.")
    if float(config.get("loss_weight", 0.0)) <= 0.0:
        raise ValueError("train.hard_crop_second_view.loss_weight must be > 0 when enabled.")
    if float(config.get("crop_scale_max", 0.0)) < float(config.get("crop_scale_min", 0.0)):
        raise ValueError("train.hard_crop_second_view.crop_scale_max must be >= crop_scale_min.")
    _validate_string_sequence(
        config.get("target_transitions", []),
        "train.hard_crop_second_view.target_transitions",
    )
    _validate_string_sequence(
        config.get("persistent_states", []),
        "train.hard_crop_second_view.persistent_states",
    )


def _validate_string_sequence(value: Any, name: str) -> None:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} must not contain empty strings.")
        return
    if not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a string or a sequence of strings.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain only non-empty strings.")


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


def _coerce_device_values(raw: str | Sequence[str]) -> list[str]:
    if isinstance(raw, str):
        values = [part.strip() for part in re.split(r"[\s,]+", raw) if part.strip()]
        if len(values) == 1:
            return values
        if values:
            return values
        raise ValueError("device must not be empty.")
    values = [str(value).strip() for value in raw if str(value).strip()]
    if not values:
        raise ValueError("device list must not be empty.")
    if len(values) == 1 and "," in values[0]:
        return _coerce_device_values(values[0])
    return values


def _validate_cuda_index(index: int) -> None:
    if index < 0:
        raise ValueError(f"CUDA device index must be >= 0; got cuda:{index}.")
    device_count = torch.cuda.device_count()
    if device_count > 0 and index >= device_count:
        raise RuntimeError(
            f"CUDA device cuda:{index} was requested, but only {device_count} CUDA "
            "device(s) are visible."
        )


def _resolve_path(raw_path: str | Path, base_dir: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def _load_project_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    _DOTENV_LOADED = True


def _read_yaml_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_dataset_name(dataset: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", dataset).lower()
    if not normalized:
        raise ValueError("dataset must contain at least one ASCII letter or digit.")
    return normalized


def _apply_dataset_env_overrides(
    raw: dict[str, Any],
    *,
    dataset: str,
    source: Path,
) -> dict[str, Any]:
    updated = deepcopy(raw)
    data = updated.get("data")
    if not isinstance(data, dict):
        return updated

    dataset_prefix = dataset.upper()
    for key, value in data.items():
        if key in _DATA_PATH_KEYS and isinstance(value, str):
            data[key] = _rewrite_dataset_env_placeholder(
                value,
                dataset_prefix=dataset_prefix,
                source=source,
                data_key=key,
            )
    return updated


def _rewrite_dataset_env_placeholder(
    value: str,
    *,
    dataset_prefix: str,
    source: Path,
    data_key: str,
) -> str:
    match = _ENV_VAR_PATTERN.fullmatch(value)
    if match is None:
        return value

    env_name = match.group(1)
    suffix = _extract_data_env_suffix(env_name)
    if suffix is None:
        raise ValueError(
            f"Unsupported data placeholder {env_name!r} for data.{data_key} in {source}. "
            f"Expected a placeholder ending with one of: {', '.join(_DATA_ENV_SUFFIXES)}."
        )

    operator = match.group(2) or ""
    default_value = match.group(3)
    rewritten_name = f"{dataset_prefix}_{suffix}"
    if default_value is None:
        return f"${{{rewritten_name}}}"
    return f"${{{rewritten_name}{operator}{default_value}}}"


def _extract_data_env_suffix(env_name: str) -> str | None:
    upper_name = env_name.upper()
    for suffix in _DATA_ENV_SUFFIXES:
        if upper_name == suffix or upper_name.endswith(f"_{suffix}"):
            return suffix
    return None


def _expand_env_placeholders(value: Any, source: Path) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(item, source) for item in value]
    if isinstance(value, str):
        return _expand_env_string(value, source)
    return value


def _expand_env_string(value: str, source: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default_value = match.group(3)
        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value
        if default_value is not None:
            return default_value
        raise KeyError(
            f"Environment variable {name!r} referenced in {source} is not set."
        )

    return _ENV_VAR_PATTERN.sub(replace, value)
