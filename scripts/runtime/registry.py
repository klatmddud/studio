from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch.nn as nn

from models.detection.wrapper import DINOWrapper, FCOSWrapper, FasterRCNNWrapper
from modules.nn import build_missbank_from_yaml, build_misshead_from_yaml, build_remiss_conv_from_yaml

from .config import load_yaml_file
from .dataset_meta import infer_num_classes_from_runtime_config

ARCH_ALIASES = {
    "faster_rcnn": "fasterrcnn",
    "faster-rcnn": "fasterrcnn",
    "fasterrcnn": "fasterrcnn",
    "fcos": "fcos",
    "dino": "dino",
}

MODEL_BUILDERS = {
    "dino": DINOWrapper,
    "fasterrcnn": FasterRCNNWrapper,
    "fcos": FCOSWrapper,
}

def normalize_arch(raw_arch: str) -> str:
    return ARCH_ALIASES.get(raw_arch.lower(), raw_arch.lower())


def infer_arch(model_config: dict[str, Any], model_config_path: str | Path) -> str:
    explicit = model_config.get("arch")
    candidate = explicit if explicit else Path(model_config_path).stem
    return normalize_arch(candidate)


def build_model_from_config(
    model_config: dict[str, Any],
    arch: str,
    *,
    module_config_paths: dict[str, str | Path] | None = None,
) -> nn.Module:
    normalized_arch = normalize_arch(arch)
    builder = MODEL_BUILDERS.get(normalized_arch)
    if builder is None:
        supported = ", ".join(sorted(MODEL_BUILDERS))
        raise NotImplementedError(
            f"Model arch {arch!r} is not implemented. Supported arches: {supported}. "
            "If your YAML filename does not match the arch name, add an explicit 'arch:' field."
        )
    model = builder(model_config)
    _attach_remiss_modules(
        model,
        model_config=model_config,
        arch=normalized_arch,
        module_config_paths=module_config_paths,
    )
    _attach_remiss_conv_modules(
        model,
        model_config=model_config,
        arch=normalized_arch,
        module_config_paths=module_config_paths,
    )
    return model


def build_model_from_path(
    model_config_path: str | Path,
    runtime_config: dict[str, Any] | None = None,
    *,
    module_config_paths: dict[str, str | Path] | None = None,
) -> tuple[nn.Module, dict[str, Any], str, Path]:
    resolved_path = Path(model_config_path).expanduser().resolve()
    model_config = load_yaml_file(resolved_path)
    if runtime_config is not None:
        inferred_num_classes = infer_num_classes_from_runtime_config(runtime_config)
        if inferred_num_classes is not None:
            model_config = dict(model_config)
            model_config["num_classes"] = inferred_num_classes
    arch = infer_arch(model_config, resolved_path)
    model = build_model_from_config(
        model_config,
        arch,
        module_config_paths=module_config_paths,
    )
    return model, model_config, arch, resolved_path


def _attach_remiss_modules(
    model: nn.Module,
    *,
    model_config: dict[str, Any],
    arch: str,
    module_config_paths: dict[str, str | Path] | None,
) -> None:
    if arch != "fcos":
        return
    if not module_config_paths:
        return
    remiss_path = module_config_paths.get("remiss")
    if remiss_path is None:
        return
    detector_thresholds = _detector_thresholds(model_config, arch=arch)
    missbank = build_missbank_from_yaml(
        remiss_path,
        arch=arch,
        detector_score_threshold=detector_thresholds.get("score"),
        detector_iou_threshold=detector_thresholds.get("iou"),
    )
    misshead = build_misshead_from_yaml(
        remiss_path,
        arch=arch,
        remiss_enabled=missbank is not None,
    )
    if missbank is None:
        if misshead is not None:
            raise ValueError("MissHead requires MissBank to be enabled.")
        return
    if misshead is not None and int(misshead.config.grid_size) != int(missbank.config.grid_size):
        raise ValueError("MissHead grid_size must match MissBank grid_size.")
    model.missbank = missbank
    if misshead is not None:
        model.miss_head = misshead


def _attach_remiss_conv_modules(
    model: nn.Module,
    *,
    model_config: dict[str, Any],
    arch: str,
    module_config_paths: dict[str, str | Path] | None,
) -> None:
    if arch != "fcos":
        return
    if not module_config_paths:
        return
    remiss_conv_path = module_config_paths.get("remiss_conv")
    if remiss_conv_path is None:
        return
    detector_thresholds = _detector_thresholds(model_config, arch=arch)
    remiss_conv_bank = build_missbank_from_yaml(
        remiss_conv_path,
        arch=arch,
        detector_score_threshold=detector_thresholds.get("score"),
        detector_iou_threshold=detector_thresholds.get("iou"),
    )
    remiss_conv = build_remiss_conv_from_yaml(
        remiss_conv_path,
        arch=arch,
        remiss_enabled=remiss_conv_bank is not None,
    )
    if remiss_conv_bank is None:
        if remiss_conv is not None:
            raise ValueError("ReMissConv requires its MissBank to be enabled.")
        return
    if remiss_conv is not None and int(remiss_conv.config.grid_size) != int(remiss_conv_bank.config.grid_size):
        raise ValueError("ReMissConv grid_size must match its MissBank grid_size.")
    model.remiss_conv_bank = remiss_conv_bank
    if remiss_conv is not None:
        model.remiss_conv = remiss_conv


def _detector_thresholds(
    model_config: Mapping[str, Any],
    *,
    arch: str,
) -> dict[str, float | None]:
    if arch == "fcos":
        head = model_config.get("head", {})
        if isinstance(head, Mapping):
            return {
                "score": float(head.get("score_thresh", 0.2)),
                "iou": float(head.get("nms_thresh", 0.6)),
            }
        return {"score": 0.2, "iou": 0.6}
    return {"score": None, "iou": None}
