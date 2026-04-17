from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from modules.nn import build_cfp_from_yaml, build_mdmb_from_yaml, build_sca_from_yaml
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
CFP_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "cfp.yaml"
MDMB_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "mdmb.yaml"
SCA_CONFIG_PATH = PROJECT_ROOT / "modules" / "cfg" / "sca.yaml"


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
    cfp = _build_cfp(normalized_arch)
    mdmb = _build_mdmb(normalized_arch)
    sca = _build_sca(normalized_arch)
    if normalized_arch == "fcos":
        return builder(model_config, mdmb=mdmb, cfp=cfp, sca=sca)
    return builder(model_config, mdmb=mdmb, cfp=cfp)


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
    if not MDMB_CONFIG_PATH.is_file():
        return None
    return build_mdmb_from_yaml(MDMB_CONFIG_PATH, arch=arch)


def _build_cfp(arch: str) -> nn.Module | None:
    if not CFP_CONFIG_PATH.is_file():
        return None
    return build_cfp_from_yaml(CFP_CONFIG_PATH, arch=arch)


def _build_sca(arch: str) -> nn.Module | None:
    if not SCA_CONFIG_PATH.is_file():
        return None
    return build_sca_from_yaml(SCA_CONFIG_PATH, arch=arch)
