from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from modules.nn import (
    load_dhm_config,
    load_dhmr_config,
    normalize_arch,
)

from .config import load_yaml_file
from .module_configs import DEFAULT_MODULE_CONFIG_PATHS, resolve_module_config_paths

MODULE_CONFIG_PATHS = DEFAULT_MODULE_CONFIG_PATHS

_CONFIG_LOADERS: dict[str, Callable[..., Any]] = {
    "dhm": load_dhm_config,
    "dhmr": load_dhmr_config,
}

_ARCH_SUPPORT: dict[str, set[str] | None] = {
    "dhm": {"fcos"},
    "dhmr": {"fcos"},
}


def collect_enabled_module_configs(
    arch: str | None,
    *,
    config_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    normalized_arch = normalize_arch(arch)
    paths = resolve_module_config_paths(config_paths, require_exists=False)

    snapshots: dict[str, Any] = {}
    for name in ("dhm", "dhmr"):
        supported_arches = _ARCH_SUPPORT[name]
        if supported_arches is not None and normalized_arch not in supported_arches:
            continue
        path = Path(paths[name]).expanduser().resolve()
        if not path.is_file():
            continue
        config = _CONFIG_LOADERS[name](path, arch=normalized_arch)
        if not bool(getattr(config, "enabled", False)):
            continue
        raw_config = load_yaml_file(path)
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"Module config must be a mapping: {path}")
        snapshots[name] = deepcopy(dict(raw_config))
    return snapshots
