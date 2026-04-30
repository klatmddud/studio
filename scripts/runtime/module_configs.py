from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODULE_CONFIG_KEYS = ("remiss",)

DEFAULT_MODULE_CONFIG_PATHS = {
    "remiss": PROJECT_ROOT / "modules" / "cfg" / "remiss.yaml",
}


def resolve_module_config_paths(
    overrides: Mapping[str, str | Path | None] | None = None,
    *,
    require_exists: bool = True,
) -> dict[str, Path]:
    paths = dict(DEFAULT_MODULE_CONFIG_PATHS)
    for name, raw_path in (overrides or {}).items():
        if name not in paths:
            raise KeyError(f"Unsupported module config key: {name!r}")
        if raw_path is None:
            continue
        paths[name] = Path(raw_path).expanduser().resolve()

    if require_exists:
        missing = [
            f"{name}={path}"
            for name, path in paths.items()
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Module config file was not found: " + ", ".join(missing)
            )
    return paths


def serialize_module_config_paths(
    paths: Mapping[str, str | Path],
) -> dict[str, str]:
    return {
        name: str(Path(path).expanduser().resolve())
        for name, path in paths.items()
        if name in DEFAULT_MODULE_CONFIG_PATHS
    }
