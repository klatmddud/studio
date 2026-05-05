from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch.nn as nn

from models.detection.wrapper import DINOWrapper, FCOSWrapper, FasterRCNNWrapper
from modules.nn import build_ftmb_from_yaml, build_lmb_from_yaml, build_missbank_from_yaml, build_qg_afp_from_yaml

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
    post_neck = _build_post_neck_modules(
        arch=normalized_arch,
        module_config_paths=module_config_paths,
    )
    model = builder(model_config, post_neck=post_neck)
    _attach_remiss_modules(
        model,
        model_config=model_config,
        arch=normalized_arch,
        module_config_paths=module_config_paths,
    )
    _attach_ftmb_modules(
        model,
        model_config=model_config,
        arch=normalized_arch,
        module_config_paths=module_config_paths,
    )
    _attach_lmb_modules(
        model,
        model_config=model_config,
        arch=normalized_arch,
        module_config_paths=module_config_paths,
    )
    return model


def _build_post_neck_modules(
    *,
    arch: str,
    module_config_paths: dict[str, str | Path] | None,
) -> nn.Module | None:
    if not module_config_paths:
        return None
    modules: list[nn.Module] = []
    qg_afp_path = module_config_paths.get("qg_afp")
    if qg_afp_path is not None and arch == "fcos":
        qg_afp = build_qg_afp_from_yaml(
            qg_afp_path,
            arch=arch,
        )
        if qg_afp is not None:
            modules.append(qg_afp)
    if not modules:
        return None
    if len(modules) == 1:
        return modules[0]
    return _FeatureDictSequential(*modules)


class _FeatureDictSequential(nn.Sequential):
    def forward(self, features):  # type: ignore[override]
        for module in self:
            features = module(features)
        return features

    def get_training_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for module in self:
            get_metrics = getattr(module, "get_training_metrics", None)
            if not callable(get_metrics):
                continue
            for name, value in get_metrics().items():
                if isinstance(value, (int, float)):
                    metrics[str(name)] = float(value)
        return metrics

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
    if missbank is None:
        return
    model.missbank = missbank


def _attach_ftmb_modules(
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
    ftmb_path = module_config_paths.get("ftmb")
    if ftmb_path is None:
        return
    detector_thresholds = _detector_thresholds(model_config, arch=arch)
    ftmb = build_ftmb_from_yaml(
        ftmb_path,
        arch=arch,
        detector_score_threshold=detector_thresholds.get("score"),
        detector_iou_threshold=detector_thresholds.get("iou"),
    )
    if ftmb is not None:
        model.ftmb = ftmb


def _attach_lmb_modules(
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
    lmb_path = module_config_paths.get("lmb")
    if lmb_path is None:
        return
    detector_thresholds = _detector_thresholds(model_config, arch=arch)
    lmb = build_lmb_from_yaml(
        lmb_path,
        arch=arch,
        detector_score_threshold=detector_thresholds.get("score"),
    )
    if lmb is None:
        return
    model.lmb = lmb


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
