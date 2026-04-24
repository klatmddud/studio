from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from modules.nn import (
    build_mdmb_from_yaml,
    build_mdmbpp_from_yaml,
    build_rasd_from_yaml,
    build_tfm_from_yaml,
)
from models.detection.wrapper import DINOWrapper, FCOSWrapper, FasterRCNNWrapper

from .config import load_yaml_file
from .dataset_meta import infer_num_classes_from_runtime_config
from .module_configs import DEFAULT_MODULE_CONFIG_PATHS, resolve_module_config_paths

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
MDMB_CONFIG_PATH = DEFAULT_MODULE_CONFIG_PATHS["mdmb"]
MDMBPP_CONFIG_PATH = DEFAULT_MODULE_CONFIG_PATHS["mdmbpp"]
RASD_CONFIG_PATH = DEFAULT_MODULE_CONFIG_PATHS["rasd"]
TFM_CONFIG_PATH = DEFAULT_MODULE_CONFIG_PATHS["tfm"]


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
    module_paths = resolve_module_config_paths(module_config_paths, require_exists=False)
    builder = MODEL_BUILDERS.get(normalized_arch)
    if builder is None:
        supported = ", ".join(sorted(MODEL_BUILDERS))
        raise NotImplementedError(
            f"Model arch {arch!r} is not implemented. Supported arches: {supported}. "
            "If your YAML filename does not match the arch name, add an explicit 'arch:' field."
        )
    mdmb = _build_mdmb(normalized_arch, module_paths["mdmb"])
    mdmbpp = _build_mdmbpp(normalized_arch, module_paths["mdmbpp"])
    rasd = _build_rasd(normalized_arch, module_paths["rasd"])
    tfm = _build_tfm(normalized_arch, module_paths["tfm"])
    if normalized_arch == "fcos":
        if rasd is not None and mdmbpp is None:
            raise ValueError("RASD requires MDMB++ to be enabled for FCOS.")
        if rasd is not None and not bool(mdmbpp.config.store_support_feature):
            raise ValueError(
                "RASD requires the active MDMB++ config to set store_support_feature: true."
            )
        return builder(
            model_config,
            mdmb=mdmb,
            mdmbpp=mdmbpp,
            rasd=rasd,
            tfm=tfm,
        )
    return builder(model_config, mdmb=mdmb)


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


def _build_mdmb(arch: str, config_path: str | Path) -> nn.Module | None:
    if arch != "fcos":
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    return build_mdmb_from_yaml(path, arch=arch)


def _build_mdmbpp(arch: str, config_path: str | Path) -> nn.Module | None:
    if arch != "fcos":
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    return build_mdmbpp_from_yaml(path, arch=arch)


def _build_rasd(arch: str, config_path: str | Path) -> nn.Module | None:
    if arch != "fcos":
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    return build_rasd_from_yaml(path, arch=arch)


def _build_tfm(arch: str, config_path: str | Path) -> nn.Module | None:
    if arch != "fcos":
        return None
    path = Path(config_path)
    if not path.is_file():
        return None
    return build_tfm_from_yaml(path, arch=arch)
