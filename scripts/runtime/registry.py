from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from modules.nn import (
    build_candidate_densifier_from_yaml,
    build_faar_from_yaml,
    build_far_from_yaml,
    build_marc_from_yaml,
    build_mce_from_yaml,
    build_mdmb_from_yaml,
    build_mdmbpp_from_yaml,
    build_recall_from_yaml,
)
from models.detection.wrapper import DINOWrapper, FCOSWrapper, FasterRCNNWrapper

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DENSIFICATION_CONFIG_PATH = (
    PROJECT_ROOT / "modules" / "cfg" / "candidate_densification.yaml"
)
FAAR_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "faar.yaml"
FAR_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "far.yaml"
MARC_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "marc.yaml"
MCE_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "mce.yaml"
MDMB_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "mdmb.yaml"
MDMBPP_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "mdmbpp.yaml"
RECALL_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "recall.yaml"


def normalize_arch(raw_arch: str) -> str:
    return ARCH_ALIASES.get(raw_arch.lower(), raw_arch.lower())


def infer_arch(model_config: dict[str, Any], model_config_path: str | Path) -> str:
    explicit = model_config.get("arch")
    candidate = explicit if explicit else Path(model_config_path).stem
    return normalize_arch(candidate)


def build_model_from_config(model_config: dict[str, Any], arch: str) -> nn.Module:
    normalized_arch = normalize_arch(arch)
    builder = MODEL_BUILDERS.get(normalized_arch)
    if builder is None:
        supported = ", ".join(sorted(MODEL_BUILDERS))
        raise NotImplementedError(
            f"Model arch {arch!r} is not implemented. Supported arches: {supported}. "
            "If your YAML filename does not match the arch name, add an explicit 'arch:' field."
        )
    num_classes = int(model_config.get("num_classes", 91))
    mdmb = _build_mdmb(normalized_arch)
    mdmbpp = _build_mdmbpp(normalized_arch)
    recall = _build_recall(normalized_arch)
    far = _build_far(normalized_arch)
    faar = _build_faar(normalized_arch)
    marc = _build_marc(normalized_arch)
    mce = _build_mce(normalized_arch, num_classes=num_classes)
    candidate_densifier = _build_candidate_densifier(normalized_arch)
    if normalized_arch == "fcos":
        return builder(
            model_config,
            mdmb=mdmb,
            mdmbpp=mdmbpp,
            recall=recall,
            far=far,
            faar=faar,
            marc=marc,
            mce=mce,
            candidate_densifier=candidate_densifier,
        )
    return builder(model_config, mdmb=mdmb)


def build_model_from_path(
    model_config_path: str | Path,
    runtime_config: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any], str, Path]:
    resolved_path = Path(model_config_path).expanduser().resolve()
    model_config = load_yaml_file(resolved_path)
    if runtime_config is not None:
        inferred_num_classes = infer_num_classes_from_runtime_config(runtime_config)
        if inferred_num_classes is not None:
            model_config = dict(model_config)
            model_config["num_classes"] = inferred_num_classes
    arch = infer_arch(model_config, resolved_path)
    model = build_model_from_config(model_config, arch)
    return model, model_config, arch, resolved_path


def _build_mdmb(arch: str) -> nn.Module | None:
    if arch != "fcos":
        return None
    if not MDMB_CONFIG_PATH.is_file():
        return None
    return build_mdmb_from_yaml(MDMB_CONFIG_PATH, arch=arch)


def _build_mdmbpp(arch: str) -> nn.Module | None:
    if arch != "fcos":
        return None
    if not MDMBPP_CONFIG_PATH.is_file():
        return None
    return build_mdmbpp_from_yaml(MDMBPP_CONFIG_PATH, arch=arch)


def _build_recall(arch: str) -> nn.Module | None:
    if arch != "fcos":
        return None
    if not RECALL_CONFIG_PATH.is_file():
        return None
    return build_recall_from_yaml(RECALL_CONFIG_PATH, arch=arch)


def _build_far(arch: str) -> nn.Module | None:
    if arch != "fcos":
        return None
    if not FAR_CONFIG_PATH.is_file():
        return None
    return build_far_from_yaml(FAR_CONFIG_PATH, arch=arch)


def _build_faar(arch: str) -> nn.Module | None:
    if arch != "fcos":
        return None
    if not FAAR_CONFIG_PATH.is_file():
        return None
    return build_faar_from_yaml(FAAR_CONFIG_PATH, arch=arch)


def _build_marc(arch: str) -> nn.Module | None:
    if arch != "fcos":
        return None
    if not MARC_CONFIG_PATH.is_file():
        return None
    return build_marc_from_yaml(MARC_CONFIG_PATH, arch=arch)


def _build_mce(arch: str, *, num_classes: int) -> nn.Module | None:
    if arch != "fcos":
        return None
    if not MCE_CONFIG_PATH.is_file():
        return None
    return build_mce_from_yaml(MCE_CONFIG_PATH, arch=arch, num_classes=num_classes)


def _build_candidate_densifier(arch: str) -> nn.Module | None:
    if arch != "fcos":
        return None
    if not CANDIDATE_DENSIFICATION_CONFIG_PATH.is_file():
        return None
    return build_candidate_densifier_from_yaml(
        CANDIDATE_DENSIFICATION_CONFIG_PATH,
        arch=arch,
    )
