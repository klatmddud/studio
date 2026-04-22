from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from modules.nn import (
    load_mdmb_config,
    load_mdmbpp_config,
    load_rasd_config,
    normalize_arch,
)

from .config import load_yaml_file
from .hard_replay import load_hard_replay_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODULE_CONFIG_PATHS = {
    "mdmb": PROJECT_ROOT / "modules" / "cfg" / "mdmb.yaml",
    "mdmbpp": PROJECT_ROOT / "modules" / "cfg" / "mdmbpp.yaml",
    "rasd": PROJECT_ROOT / "modules" / "cfg" / "rasd.yaml",
    "hard_replay": PROJECT_ROOT / "modules" / "cfg" / "hard_replay.yaml",
}

_CONFIG_LOADERS: dict[str, Callable[..., Any]] = {
    "mdmb": load_mdmb_config,
    "mdmbpp": load_mdmbpp_config,
    "rasd": load_rasd_config,
    "hard_replay": load_hard_replay_config,
}

_ARCH_SUPPORT: dict[str, set[str] | None] = {
    "mdmb": {"fcos"},
    "mdmbpp": {"fcos"},
    "rasd": {"fcos"},
    "hard_replay": None,
}


def collect_enabled_module_configs(
    arch: str | None,
    *,
    config_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    normalized_arch = normalize_arch(arch)
    paths = dict(MODULE_CONFIG_PATHS)
    if config_paths is not None:
        for name, path in config_paths.items():
            if name not in _CONFIG_LOADERS:
                raise KeyError(f"Unsupported module config key: {name!r}")
            paths[name] = Path(path)

    snapshots: dict[str, Any] = {}
    for name in ("mdmb", "mdmbpp", "rasd", "hard_replay"):
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
