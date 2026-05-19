from __future__ import annotations

"""IDEA-DINO backend adapter for ``DINOWrapper``."""

import importlib
import sys
from argparse import Namespace
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DINO_ROOT = PROJECT_ROOT / "third-party" / "DINO"
DEFAULT_CONFIG_FILE = Path("config") / "DINO" / "DINO_4scale.py"


def build_idea_dino_components(
    cfg: Mapping[str, Any],
    num_classes: int,
    *,
    dino_root: str | Path | None = None,
    config_file: str | Path | None = None,
    checkpoint: str | Path | None = None,
    pretrain_model_path: str | Path | None = None,
    checkpoint_key: str = "model",
    finetune_ignore: Sequence[str] | None = None,
    args_overrides: Mapping[str, Any] | None = None,
    device: str = "cpu",
    disable_torchvision_pretrained_backbone: bool = True,
    pre_neck: torch.nn.Module | None = None,
    post_neck: torch.nn.Module | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """
    Build IDEA-DINO components for ``models.detection.wrapper.DINOWrapper``.

    The adapter intentionally keeps IDEA-DINO isolated behind this function because
    that repository uses top-level package names such as ``models`` and ``util``.
    """
    _ = cfg, extra
    if pre_neck is not None or post_neck is not None:
        raise ValueError("IDEA-DINO backend does not support pre_neck or post_neck hooks.")

    resolved_dino_root = _resolve_dino_root(dino_root)
    resolved_config = _resolve_path(
        config_file or DEFAULT_CONFIG_FILE,
        search_roots=(resolved_dino_root, PROJECT_ROOT),
    )
    args = _load_dino_args(
        resolved_dino_root=resolved_dino_root,
        config_file=resolved_config,
        num_classes=int(num_classes),
        device=device,
        args_overrides=args_overrides,
    )

    model, criterion, postprocessors = _build_dino_from_args(
        args,
        dino_root=resolved_dino_root,
        disable_torchvision_pretrained_backbone=disable_torchvision_pretrained_backbone,
    )

    checkpoint_path = checkpoint or pretrain_model_path
    if checkpoint_path is not None:
        load_summary = _load_partial_checkpoint(
            model,
            checkpoint_path,
            checkpoint_key=checkpoint_key,
            ignore_keywords=finetune_ignore,
        )
        model.idea_dino_checkpoint = load_summary

    return {
        "model": model,
        "criterion": criterion,
        "postprocessors": postprocessors,
    }


def _resolve_dino_root(path: str | Path | None) -> Path:
    root = Path(path).expanduser() if path is not None else DEFAULT_DINO_ROOT
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"IDEA-DINO root does not exist: {root}")
    return root


def _resolve_path(raw_path: str | Path, *, search_roots: Sequence[Path]) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"File does not exist: {resolved}")
        return resolved

    for root in search_roots:
        candidate = (root / path).resolve()
        if candidate.is_file():
            return candidate

    fallback = (search_roots[0] / path).resolve()
    raise FileNotFoundError(f"File does not exist: {fallback}")


def _load_dino_args(
    *,
    resolved_dino_root: Path,
    config_file: Path,
    num_classes: int,
    device: str,
    args_overrides: Mapping[str, Any] | None,
) -> Namespace:
    with _idea_dino_import_context(resolved_dino_root):
        from util.slconfig import SLConfig

        dino_config = SLConfig.fromfile(str(config_file))
        config_dict = dino_config._cfg_dict.to_dict()

    config_dict.update(
        {
            "config_file": str(config_file),
            "dataset_file": config_dict.get("dataset_file", "coco"),
            "device": str(device),
            "distributed": False,
            "frozen_weights": config_dict.get("frozen_weights"),
            "num_classes": int(num_classes),
            "output_dir": "",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        }
    )
    if args_overrides:
        config_dict.update(dict(args_overrides))
    return Namespace(**config_dict)


def _build_dino_from_args(
    args: Namespace,
    *,
    dino_root: Path,
    disable_torchvision_pretrained_backbone: bool,
):
    with _idea_dino_import_context(dino_root):
        try:
            dino_module = importlib.import_module("models.dino.dino")
        except ModuleNotFoundError as exc:
            if exc.name == "MultiScaleDeformableAttention":
                raise RuntimeError(
                    "IDEA-DINO requires its MultiScaleDeformableAttention extension. "
                    "Build it from third-party/DINO/models/dino/ops before using this backend."
                ) from exc
            raise

        if disable_torchvision_pretrained_backbone:
            backbone_module = importlib.import_module("models.dino.backbone")
            backbone_module.is_main_process = lambda: False

        return dino_module.build_dino(args)


def _load_partial_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    checkpoint_key: str,
    ignore_keywords: Sequence[str] | None,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"IDEA-DINO checkpoint does not exist: {path}")

    with torch.serialization.safe_globals([Namespace]):
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint
    if isinstance(checkpoint, Mapping) and checkpoint_key in checkpoint:
        state = checkpoint[checkpoint_key]
    if not isinstance(state, Mapping):
        raise TypeError(f"IDEA-DINO checkpoint state must be a mapping: {path}")

    ignore = tuple(ignore_keywords or ())
    model_state = model.state_dict()
    selected: OrderedDict[str, torch.Tensor] = OrderedDict()
    skipped_ignored: list[str] = []
    skipped_missing: list[str] = []
    skipped_shape: list[str] = []

    for raw_key, value in state.items():
        key = _strip_module_prefix(str(raw_key))
        if any(token in key for token in ignore):
            skipped_ignored.append(key)
            continue
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if not isinstance(value, torch.Tensor):
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append(key)
            continue
        selected[key] = value

    incompatible = model.load_state_dict(selected, strict=False)
    return {
        "path": str(path),
        "loaded_keys": len(selected),
        "skipped_ignored": skipped_ignored,
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


@contextmanager
def _idea_dino_import_context(dino_root: Path):
    """Temporarily make IDEA-DINO's top-level packages win import resolution."""
    original_path = list(sys.path)
    ops_root = dino_root / "models" / "dino" / "ops"
    saved_models = sys.modules.get("models")
    saved_registry = sys.modules.get("models.registry")
    saved_util = sys.modules.get("util")
    saved_util_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("util.")
    }
    saved_dino_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "models.dino" or name.startswith("models.dino.")
    }

    sys.modules.pop("models", None)
    sys.modules.pop("models.registry", None)
    sys.modules.pop("util", None)
    for name in saved_util_modules:
        sys.modules.pop(name, None)
    for name in saved_dino_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(ops_root))
    sys.path.insert(0, str(dino_root))
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == "models.dino" or name.startswith("models.dino."):
                sys.modules.pop(name, None)
            if name == "util" or name.startswith("util."):
                sys.modules.pop(name, None)
        sys.modules.pop("models.registry", None)
        sys.modules.pop("models", None)
        if saved_models is not None:
            sys.modules["models"] = saved_models
        if saved_registry is not None:
            sys.modules["models.registry"] = saved_registry
        if saved_util is not None:
            sys.modules["util"] = saved_util
        sys.modules.update(saved_util_modules)
        sys.modules.update(saved_dino_modules)


__all__ = ["build_idea_dino_components"]
