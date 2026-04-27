from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from .config import dump_yaml_file
from .data import build_train_mining_dataloader
from .distributed import (
    DistributedContext,
    all_gather_object,
    barrier,
    is_distributed,
    is_main_process,
    synchronize_extra_state,
    unwrap_model,
)
from .metrics import evaluate_coco_detection, save_predictions
from .module_metadata import collect_enabled_module_configs
from .visualize import (
    build_confusion_matrix,
    plot_confusion_matrices,
    plot_loss_curves,
    plot_map_curves,
)


def fit(
    model: torch.nn.Module,
    model_config: dict[str, Any],
    model_config_path: str | Path,
    runtime_config: dict[str, Any],
    runtime_config_path: str | Path,
    train_loader,
    val_loader,
    device: torch.device,
    arch: str,
    distributed: DistributedContext | None = None,
    module_config_paths: Mapping[str, str | Path] | None = None,
) -> list[dict[str, Any]]:
    distributed = distributed or DistributedContext(enabled=False, device=device)
    main_process = is_main_process(distributed)
    output_dir = Path(runtime_config["output_dir"])
    checkpoint_dir = Path(runtime_config["checkpoint"]["dir"])
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _probe_writable_dir(checkpoint_dir)
    barrier(distributed)

    seed = int(runtime_config["seed"])
    seed_everything(seed + (distributed.rank if is_distributed(distributed) else 0))
    model.to(device)
    if is_distributed(distributed):
        if device.type != "cuda" or device.index is None:
            raise RuntimeError("DDP training requires an explicit CUDA device per rank.")
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )

    optimizer = build_optimizer(model, runtime_config["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        runtime_config["scheduler"],
        runtime_config["train"]["epochs"],
    )
    scaler = _build_grad_scaler(
        enabled=runtime_config["amp"] and device.type == "cuda",
        device_type=device.type,
    )

    history: list[dict[str, Any]] = []
    best_metric = _initial_best(runtime_config["checkpoint"]["mode"])
    start_epoch = 0

    resume_path = runtime_config["checkpoint"].get("resume")
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        best_metric = float(
            checkpoint.get("best_metric", _initial_best(runtime_config["checkpoint"]["mode"]))
        )

    if main_process:
        persist_run_metadata(
            output_dir=output_dir,
            arch=arch,
            model_config=model_config,
            model_config_path=model_config_path,
            runtime_config=runtime_config,
            runtime_config_path=runtime_config_path,
            module_configs=collect_enabled_module_configs(
                arch,
                config_paths=module_config_paths or runtime_config.get("_module_config_paths"),
            ),
        )
    barrier(distributed)

    total_epochs = runtime_config["train"]["epochs"]
    eval_every = runtime_config["train"]["eval_every_epochs"]
    monitor = runtime_config["checkpoint"]["monitor"]
    mode = runtime_config["checkpoint"]["mode"]

    for epoch in range(start_epoch, total_epochs):
        _call_model_hook(model, "dhm", "start_epoch", epoch + 1)
        _call_model_hook(model, "dhmr", "start_epoch", epoch + 1)
        _set_data_loader_epoch(train_loader, epoch + 1)

        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            amp=runtime_config["amp"],
            scaler=scaler,
            log_interval=runtime_config["train"]["log_interval"] if main_process else 0,
            grad_clip_norm=runtime_config["train"]["grad_clip_norm"],
            epoch_index=epoch,
            total_epochs=total_epochs,
            distributed=distributed,
        )
        _run_dhm_epoch_mining(
            model=model,
            runtime_config=runtime_config,
            device=device,
            amp=runtime_config["amp"],
            log_interval=runtime_config["train"]["log_interval"] if main_process else 0,
            epoch=epoch + 1,
            total_epochs=total_epochs,
            distributed=distributed,
        )

        _call_model_hook(model, "dhm", "end_epoch", epoch + 1)
        _call_model_hook(model, "dhmr", "end_epoch", epoch + 1)

        local_summaries = {
            "dhm": _get_module_summary(model, "dhm"),
            "dhmr": _get_module_summary(model, "dhmr"),
        }
        merged_epoch_summaries = _merge_epoch_summaries(local_summaries, distributed)
        _synchronize_research_memory(model, distributed)
        dhm_summary = merged_epoch_summaries.get("dhm")
        dhmr_summary = merged_epoch_summaries.get("dhmr")

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train": train_metrics,
        }
        if dhm_summary is not None:
            record["dhm"] = dhm_summary
        if dhmr_summary is not None:
            record["dhmr"] = dhmr_summary

        should_eval = (
            val_loader is not None
            and ((epoch + 1) % eval_every == 0 or (epoch + 1) == total_epochs)
        )
        if should_eval and main_process:
            val_metrics, _ = evaluate(
                model=unwrap_model(model),
                runtime_config=runtime_config,
                data_loader=val_loader,
                device=device,
                output_dir=None,
                log_interval=runtime_config["train"]["log_interval"],
                stage_label="val",
                epoch_index=epoch,
                total_epochs=total_epochs,
            )
            record["val"] = val_metrics

            current_value = val_metrics.get(monitor)
            if current_value is None:
                raise KeyError(f"Monitored metric {monitor!r} was not found in validation output.")
            if _is_better(current_value, best_metric, mode):
                best_metric = current_value
                if runtime_config["checkpoint"]["save_best"]:
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        best_metric=best_metric,
                    )
        barrier(distributed)

        if main_process and runtime_config["checkpoint"]["save_last"]:
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_metric=best_metric,
            )

        if scheduler is not None:
            scheduler.step()

        if main_process:
            history.append(record)
            _write_json(output_dir / "history.json", history)

            summary = format_metrics(record, runtime_config["metrics"]["primary"])
            print(f"[epoch {epoch + 1}/{total_epochs}] {summary}")

    best_pt = checkpoint_dir / "best.pt"
    if main_process and val_loader is not None and best_pt.is_file():
        print(f"[best_val] Loading best checkpoint: {best_pt}")
        load_checkpoint(best_pt, model=unwrap_model(model), map_location=device)
        best_val_metrics, best_val_predictions = evaluate_coco_detection(
            model=unwrap_model(model),
            data_loader=val_loader,
            device=device,
            amp=runtime_config["amp"],
            log_interval=runtime_config["train"]["log_interval"],
            stage_label="best_val",
        )
        _write_json(output_dir / "best_val_metrics.json", best_val_metrics)
        primary = runtime_config["metrics"]["primary"]
        print(
            f"[best_val] {primary}={best_val_metrics.get(primary, float('nan')):.4f} "
            f"saved to {output_dir / 'best_val_metrics.json'}"
        )

        print("[visualize] Generating training figures ...")
        plot_loss_curves(history, output_dir)
        plot_map_curves(history, output_dir)

        print("[visualize] Building confusion matrix ...")
        cm, class_names = build_confusion_matrix(best_val_predictions, val_loader)
        plot_confusion_matrices(cm, class_names, output_dir)

    barrier(distributed)
    return history


def evaluate(
    model: torch.nn.Module,
    runtime_config: dict[str, Any],
    data_loader,
    device: torch.device,
    output_dir: str | Path | None,
    log_interval: int = 0,
    stage_label: str = "eval",
    epoch_index: int | None = None,
    total_epochs: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    checkpoint_path = runtime_config["checkpoint"].get("path")
    if checkpoint_path:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model=model, map_location=device)

    model.to(device)
    metrics, predictions = evaluate_coco_detection(
        model=model,
        data_loader=data_loader,
        device=device,
        amp=runtime_config["amp"],
        log_interval=log_interval,
        stage_label=stage_label,
        epoch_index=epoch_index,
        total_epochs=total_epochs,
    )

    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(resolved_output_dir / "metrics.json", metrics)

        eval_config = runtime_config.get("eval", {})
        if eval_config.get("save_predictions"):
            predictions_path = eval_config.get("predictions_path")
            if predictions_path:
                destination = Path(predictions_path)
            else:
                destination = resolved_output_dir / "predictions.json"
            save_predictions(destination, predictions)

        print("[visualize] Building confusion matrix ...")
        cm, class_names = build_confusion_matrix(predictions, data_loader)
        plot_confusion_matrices(cm, class_names, resolved_output_dir)

    return metrics, predictions


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    data_loader,
    device: torch.device,
    amp: bool,
    scaler: Any,
    log_interval: int,
    grad_clip_norm: float | None,
    epoch_index: int,
    total_epochs: int,
    distributed: DistributedContext | None = None,
) -> dict[str, float]:
    model.train()
    loss_sums: defaultdict[str, float] = defaultdict(float)
    num_batches = 0
    start_time = time.perf_counter()
    total_steps = len(data_loader)

    for step, (images, targets) in enumerate(data_loader, start=1):
        images = [image.to(device) for image in images]
        targets = [_move_target_to_device(target, device) for target in targets]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            loss_dict = model(images, targets)
            if not loss_dict:
                raise RuntimeError("Model returned an empty loss dict during training.")
            total_loss = sum(loss_dict.values())

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss encountered: {float(total_loss.item())}")

        if scaler.is_enabled():
            scaler.scale(total_loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        num_batches += 1
        loss_sums["loss"] += float(total_loss.detach().item())
        for name, value in loss_dict.items():
            loss_sums[name] += float(value.detach().item())

        should_log = log_interval > 0 and (step % log_interval == 0 or step == total_steps)
        if should_log:
            elapsed = time.perf_counter() - start_time
            avg_step_time = elapsed / max(step, 1)
            remaining_steps = max(total_steps - step, 0)
            epoch_eta = _format_eta(avg_step_time * remaining_steps)
            mean_total_loss = loss_sums["loss"] / num_batches
            component_summary = ", ".join(
                f"{name}={loss_sums[name] / num_batches:.4f}"
                for name in sorted(loss_dict)
            )
            message = (
                f"[train] epoch {epoch_index + 1}/{total_epochs} "
                f"step {step}/{total_steps} "
                f"epoch_eta={epoch_eta} "
                f"total_loss={mean_total_loss:.4f}"
            )
            if component_summary:
                message = f"{message} {component_summary}"
            print(message)

    duration = time.perf_counter() - start_time
    if is_distributed(distributed):
        return _reduce_train_metrics(
            loss_sums=loss_sums,
            num_batches=num_batches,
            duration=duration,
            lr=float(optimizer.param_groups[0]["lr"]),
            distributed=distributed,
        )
    metrics = {
        key: value / max(num_batches, 1)
        for key, value in loss_sums.items()
    }
    metrics["lr"] = float(optimizer.param_groups[0]["lr"])
    metrics["epoch_time_sec"] = duration
    return metrics


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = config["name"].lower()

    if name == "sgd":
        return optim.SGD(
            params,
            lr=config["lr"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),
            nesterov=config.get("nesterov", False),
        )
    if name == "adamw":
        return optim.AdamW(
            params,
            lr=config["lr"],
            betas=tuple(config.get("betas", (0.9, 0.999))),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.01),
        )

    raise ValueError(f"Unsupported optimizer.name: {config['name']!r}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: dict[str, Any],
    total_epochs: int,
):
    name = config["name"].lower()
    if name == "none":
        return None
    if name == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["milestones"],
            gamma=config.get("gamma", 0.1),
        )
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config.get("gamma", 0.1),
        )
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("t_max", total_epochs),
            eta_min=config.get("eta_min", 0.0),
        )

    raise ValueError(f"Unsupported scheduler.name: {config['name']!r}")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler=None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    state_result = unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing_keys = [key for key in state_result.missing_keys if not _is_optional_research_key(key)]
    unexpected_keys = [
        key for key in state_result.unexpected_keys if not _is_optional_research_key(key)
    ]
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"missing_keys={missing_keys}")
        if unexpected_keys:
            details.append(f"unexpected_keys={unexpected_keys}")
        raise RuntimeError(
            f"Checkpoint state_dict is incompatible with the current model: {'; '.join(details)}"
        )

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f"{target.stem}-",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def persist_run_metadata(
    output_dir: Path,
    arch: str,
    model_config: dict[str, Any],
    model_config_path: str | Path,
    runtime_config: dict[str, Any],
    runtime_config_path: str | Path,
    module_configs: dict[str, Any] | None = None,
) -> None:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml_file(metadata_dir / "model.yaml", model_config)
    dump_yaml_file(
        metadata_dir / f"{runtime_config['_mode']}.yaml",
        _strip_internal_keys(runtime_config),
    )
    if module_configs is not None:
        dump_yaml_file(metadata_dir / "modules.yaml", module_configs)

    metadata = {
        "arch": arch,
        "model_config_path": str(Path(model_config_path).resolve()),
        "runtime_config_path": str(Path(runtime_config_path).resolve()),
    }
    module_config_paths = runtime_config.get("_module_config_paths")
    if isinstance(module_config_paths, Mapping):
        metadata["module_config_paths"] = {
            str(name): str(Path(path).expanduser().resolve())
            for name, path in module_config_paths.items()
        }
    _write_json(metadata_dir / "run.json", metadata)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_metrics(record: dict[str, Any], primary_metric: str = "bbox_mAP_50_95") -> str:
    parts = []
    train_metrics = record.get("train", {})
    if "loss" in train_metrics:
        parts.append(f"train_loss={train_metrics['loss']:.4f}")

    dhm_summary = record.get("dhm")
    if isinstance(dhm_summary, dict):
        parts.append(f"dhm_records={int(dhm_summary.get('num_records', 0))}")
        parts.append(f"dhm_mean_instability={float(dhm_summary.get('mean_instability', 0.0)):.3f}")

    dhmr_summary = record.get("dhmr")
    if isinstance(dhmr_summary, dict):
        hlrt = dhmr_summary.get("hlrt", {})
        typed_film = dhmr_summary.get("typed_film", {})
        parts.append(f"dhmr_residual_records={int(dhmr_summary.get('num_residual_records', 0))}")
        if isinstance(hlrt, Mapping) and bool(hlrt.get("enabled", False)):
            parts.append(f"hlrt_replay_points={int(hlrt.get('hlrt_replay_points', 0))}")
            parts.append(f"hlrt_side_points={int(hlrt.get('hlrt_side_points', 0))}")
        if isinstance(typed_film, Mapping) and bool(typed_film.get("enabled", False)):
            parts.append(f"typed_film_points={int(typed_film.get('selected_points', 0))}")

    val_metrics = record.get("val")
    if isinstance(val_metrics, dict):
        primary = val_metrics.get(primary_metric)
        if primary is not None:
            parts.append(f"val_{primary_metric}={primary:.4f}")

    return " ".join(parts)


def _reduce_train_metrics(
    *,
    loss_sums: Mapping[str, float],
    num_batches: int,
    duration: float,
    lr: float,
    distributed: DistributedContext | None,
) -> dict[str, float]:
    gathered = all_gather_object(
        {
            "loss_sums": dict(loss_sums),
            "num_batches": int(num_batches),
            "duration": float(duration),
            "lr": float(lr),
        },
        distributed,
    )
    total_batches = sum(int(item.get("num_batches", 0)) for item in gathered if isinstance(item, Mapping))
    combined: defaultdict[str, float] = defaultdict(float)
    max_duration = 0.0
    for item in gathered:
        if not isinstance(item, Mapping):
            continue
        max_duration = max(max_duration, float(item.get("duration", 0.0)))
        raw_losses = item.get("loss_sums", {})
        if not isinstance(raw_losses, Mapping):
            continue
        for key, value in raw_losses.items():
            combined[str(key)] += float(value)

    metrics = {
        key: value / float(max(total_batches, 1))
        for key, value in combined.items()
    }
    metrics["lr"] = float(lr)
    metrics["epoch_time_sec"] = max_duration
    return metrics


def _merge_epoch_summaries(
    local_summaries: Mapping[str, dict[str, Any] | None],
    distributed: DistributedContext | None,
) -> dict[str, dict[str, Any] | None]:
    gathered = all_gather_object(dict(local_summaries), distributed)
    merged: dict[str, dict[str, Any] | None] = {}
    for name in ("dhm", "dhmr"):
        summaries = [
            item.get(name)
            for item in gathered
            if isinstance(item, Mapping) and isinstance(item.get(name), Mapping)
        ]
        merged[name] = _merge_summary_group(name, summaries)
    return merged


def _merge_summary_group(
    name: str,
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not summaries:
        return None
    result = dict(summaries[0])
    if name == "dhm":
        active_summaries = [
            summary
            for summary in summaries
            if isinstance(summary.get("mining"), Mapping)
            and int(summary.get("mining", {}).get("gt_seen", 0)) > 0
        ]
        source_summaries = active_summaries or [
            max(summaries, key=lambda summary: int(summary.get("num_records", 0)))
        ]
        result = dict(source_summaries[0])
        for key in (
            "num_records",
            "num_images",
            "num_current_failures",
            "global_max_fn_streak",
            "total_forgetting",
            "total_recovery",
            "total_type_switch",
        ):
            result[key] = max(int(summary.get(key, 0)) for summary in source_summaries)
        _merge_weighted_mean(result, source_summaries, "mean_instability", "num_records")

        for nested_key in ("last_state_counts", "status_counts", "dominant_failure_counts"):
            counts: defaultdict[str, int] = defaultdict(int)
            for summary in source_summaries:
                raw_counts = summary.get(nested_key, {})
                if not isinstance(raw_counts, Mapping):
                    continue
                for state, count in raw_counts.items():
                    counts[str(state)] += int(count)
            result[nested_key] = dict(counts)

        result["transition_matrix"] = _merge_nested_count_dicts(
            source_summaries,
            "transition_matrix",
        )
        result["assignment_by_state"] = _merge_assignment_summary_dicts(
            source_summaries,
            "assignment_by_state",
        )
        result["assignment_by_transition"] = _merge_assignment_summary_dicts(
            source_summaries,
            "assignment_by_transition",
        )

        mining_counts: defaultdict[str, int] = defaultdict(int)
        for summary in summaries:
            raw_mining = summary.get("mining", {})
            if not isinstance(raw_mining, Mapping):
                continue
            for key, value in raw_mining.items():
                mining_counts[str(key)] += int(value)
        result["mining"] = dict(mining_counts)
        return result
    if name == "dhmr":
        hlrt_counts: defaultdict[str, int] = defaultdict(int)
        typed_counts: defaultdict[str, int] = defaultdict(int)
        typed_state_counts: defaultdict[str, int] = defaultdict(int)
        side_loss_weighted = 0.0
        side_loss_count = 0
        result["num_residual_records"] = max(
            int(summary.get("num_residual_records", 0))
            for summary in summaries
        )
        result["hlrt_warmup_factor"] = max(
            float(summary.get("hlrt_warmup_factor", 0.0))
            for summary in summaries
        )
        for summary in summaries:
            hlrt = summary.get("hlrt", {})
            if not isinstance(hlrt, Mapping):
                continue
            for key, value in hlrt.items():
                if isinstance(value, (int, float)) and not str(key).startswith("mean_"):
                    hlrt_counts[str(key)] += int(value)
            losses = int(hlrt.get("hlrt_side_losses", 0))
            side_loss_weighted += float(hlrt.get("mean_side_loss", 0.0)) * float(losses)
            side_loss_count += losses
            typed_film = summary.get("typed_film", {})
            if isinstance(typed_film, Mapping):
                for key, value in typed_film.items():
                    if str(key) in {"enabled", "warmup_factor", "state_counts"}:
                        continue
                    if isinstance(value, (int, float)):
                        typed_counts[str(key)] += int(value)
                raw_state_counts = typed_film.get("state_counts", {})
                if isinstance(raw_state_counts, Mapping):
                    for state, count in raw_state_counts.items():
                        typed_state_counts[str(state)] += int(count)
        hlrt_result = dict(hlrt_counts)
        hlrt_result["enabled"] = any(
            bool(summary.get("hlrt", {}).get("enabled", False))
            for summary in summaries
            if isinstance(summary.get("hlrt"), Mapping)
        )
        hlrt_result["native_loss_hooks"] = any(
            bool(summary.get("hlrt", {}).get("native_loss_hooks", False))
            for summary in summaries
            if isinstance(summary.get("hlrt"), Mapping)
        )
        hlrt_result["assignment_replay"] = any(
            bool(summary.get("hlrt", {}).get("assignment_replay", False))
            for summary in summaries
            if isinstance(summary.get("hlrt"), Mapping)
        )
        hlrt_result["residual_memory"] = any(
            bool(summary.get("hlrt", {}).get("residual_memory", False))
            for summary in summaries
            if isinstance(summary.get("hlrt"), Mapping)
        )
        hlrt_result["mean_side_loss"] = side_loss_weighted / float(max(side_loss_count, 1))
        result["hlrt"] = hlrt_result
        typed_result = dict(typed_counts)
        typed_result["enabled"] = any(
            bool(summary.get("typed_film", {}).get("enabled", False))
            for summary in summaries
            if isinstance(summary.get("typed_film"), Mapping)
        )
        typed_result["warmup_factor"] = max(
            float(summary.get("typed_film", {}).get("warmup_factor", 0.0))
            for summary in summaries
            if isinstance(summary.get("typed_film"), Mapping)
        ) if any(isinstance(summary.get("typed_film"), Mapping) for summary in summaries) else 0.0
        typed_result["state_counts"] = dict(typed_state_counts)
        result["typed_film"] = typed_result
        return result
    return result


def _merge_nested_count_dicts(
    summaries: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, int]]:
    merged: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for summary in summaries:
        raw = summary.get(key, {})
        if not isinstance(raw, Mapping):
            continue
        for outer_key, inner in raw.items():
            if not isinstance(inner, Mapping):
                continue
            for inner_key, value in inner.items():
                merged[str(outer_key)][str(inner_key)] += int(value)
    return {
        outer_key: dict(inner_counts)
        for outer_key, inner_counts in merged.items()
    }


def _merge_assignment_summary_dicts(
    summaries: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        raw = summary.get(key, {})
        if not isinstance(raw, Mapping):
            continue
        for group_key, group_value in raw.items():
            if not isinstance(group_value, Mapping):
                continue
            count = int(group_value.get("records", 0))
            if count <= 0:
                continue
            bucket = buckets.setdefault(
                str(group_key),
                {
                    "records": 0,
                    "level_pos_counts": defaultdict(int),
                    "_weighted": defaultdict(float),
                },
            )
            bucket["records"] += count
            level_counts = group_value.get("level_pos_counts", {})
            if isinstance(level_counts, Mapping):
                for level, value in level_counts.items():
                    bucket["level_pos_counts"][str(level)] += int(value)
            weighted = bucket["_weighted"]
            for metric, value in group_value.items():
                if metric in {"records", "level_pos_counts"}:
                    continue
                if isinstance(value, (int, float)):
                    weighted[str(metric)] += float(value) * float(count)

    result: dict[str, dict[str, Any]] = {}
    for group_key, bucket in buckets.items():
        count = int(bucket.get("records", 0))
        if count <= 0:
            continue
        group_result: dict[str, Any] = {
            "records": count,
            "level_pos_counts": dict(bucket.get("level_pos_counts", {})),
        }
        weighted = bucket.get("_weighted", {})
        if isinstance(weighted, Mapping):
            for metric, value in weighted.items():
                group_result[str(metric)] = float(value) / float(count)
        result[group_key] = group_result
    return result


def _merge_weighted_mean(
    result: dict[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    value_key: str,
    weight_key: str,
) -> None:
    total_weight = sum(float(summary.get(weight_key, 0.0)) for summary in summaries)
    if total_weight <= 0.0:
        result[value_key] = 0.0
        return
    weighted = sum(
        float(summary.get(value_key, 0.0)) * float(summary.get(weight_key, 0.0))
        for summary in summaries
    )
    result[value_key] = weighted / total_weight


def _synchronize_research_memory(
    model: torch.nn.Module,
    distributed: DistributedContext | None,
) -> None:
    if not is_distributed(distributed):
        return
    base_model = unwrap_model(model)
    sync_specs = (
        ("dhm", _merge_dhm_states),
        ("dhmr", _merge_dhmr_states),
    )
    for attribute_name, merge_fn in sync_specs:
        module = getattr(base_model, attribute_name, None)
        if module is None:
            continue
        synchronize_extra_state(module, distributed, merge_fn)


def _merge_dhm_states(states: list[Any]) -> dict[str, Any] | None:
    valid = [state for state in states if isinstance(state, Mapping)]
    if not valid:
        return None
    merged = dict(valid[0])
    merged["current_epoch"] = max(int(state.get("current_epoch", 0)) for state in valid)

    records: dict[str, Mapping[str, Any]] = {}
    stats: defaultdict[str, int] = defaultdict(int)
    for state in valid:
        raw_stats = state.get("stats", {})
        if isinstance(raw_stats, Mapping):
            for key, value in raw_stats.items():
                stats[str(key)] += int(value)

        raw_records = state.get("records", {})
        if not isinstance(raw_records, Mapping):
            continue
        for gt_uid, raw_record in raw_records.items():
            if not isinstance(raw_record, Mapping):
                continue
            key = str(gt_uid)
            current = records.get(key)
            candidate = dict(raw_record)
            if current is None:
                records[key] = candidate
                continue
            records[key] = _merge_dhm_record(current, candidate)

    merged["records"] = records
    merged["stats"] = dict(stats)
    return merged


def _merge_dhmr_states(states: list[Any]) -> dict[str, Any] | None:
    valid = [state for state in states if isinstance(state, Mapping)]
    if not valid:
        return None
    merged = dict(valid[0])
    merged["current_epoch"] = max(int(state.get("current_epoch", 0)) for state in valid)
    records: dict[str, Mapping[str, Any]] = {}
    for state in valid:
        raw_records = state.get("residual_records", {})
        if not isinstance(raw_records, Mapping):
            continue
        for gt_uid, raw_record in raw_records.items():
            if not isinstance(raw_record, Mapping):
                continue
            key = str(gt_uid)
            candidate = dict(raw_record)
            current = records.get(key)
            if current is None:
                records[key] = candidate
                continue
            records[key] = _select_dhmr_residual_record(current, candidate)
    merged["residual_records"] = records
    return merged


def _merge_dhm_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_priority = _dhm_record_priority(left)
    right_priority = _dhm_record_priority(right)
    selected = dict(left if left_priority >= right_priority else right)
    assignment_source = _select_dhm_assignment_record(left, right)
    for field_name in (
        "assignment_seen",
        "last_assignment_epoch",
        "last_pos_count",
        "zero_pos_count",
        "last_level_pos_counts",
        "level_pos_counts",
        "last_near_candidate_count",
        "last_near_negative_count",
        "last_ambiguous_assigned_elsewhere",
        "ema_pos_count",
        "ema_center_dist",
        "ema_centerness_target",
        "ema_cls_loss",
        "ema_box_loss",
        "ema_ctr_loss",
        "ema_near_negative_count",
        "ema_near_negative_ratio",
        "ema_ambiguous_assigned_elsewhere",
        "ema_ambiguous_ratio",
    ):
        if field_name in assignment_source:
            selected[field_name] = assignment_source[field_name]
    return selected


def _select_dhm_assignment_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Mapping[str, Any]:
    left_priority = (
        float(left.get("last_assignment_epoch") or 0),
        float(left.get("assignment_seen", 0)),
    )
    right_priority = (
        float(right.get("last_assignment_epoch") or 0),
        float(right.get("assignment_seen", 0)),
    )
    return left if left_priority >= right_priority else right


def _dhm_record_priority(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(record.get("last_seen_epoch") or 0),
        float(record.get("instability_score", 0.0)),
        float(record.get("forgetting_count", 0)),
        float(record.get("fn_count", 0)),
        float(record.get("max_fn_streak", 0)),
        float(record.get("state_change_count", 0)),
    )


def _select_dhmr_residual_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_priority = _dhmr_residual_record_priority(left)
    right_priority = _dhmr_residual_record_priority(right)
    return dict(left if left_priority >= right_priority else right)


def _dhmr_residual_record_priority(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(record.get("last_epoch") or 0),
        float(record.get("observations", 0)),
    )


def _set_data_loader_epoch(data_loader, epoch: int) -> None:
    sampler = getattr(data_loader, "sampler", None)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))
    batch_sampler = getattr(data_loader, "batch_sampler", None)
    set_epoch = getattr(batch_sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))


def _move_target_to_device(
    target: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _initial_best(mode: str) -> float:
    return float("-inf") if mode == "max" else float("inf")


def _is_better(current: float, best: float, mode: str) -> bool:
    if mode == "max":
        return current > best
    return current < best


def _strip_internal_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _build_grad_scaler(enabled: bool, device_type: str):
    grad_scaler = getattr(torch.amp, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _probe_writable_dir(path: Path) -> None:
    probe_path = path / ".write_probe"
    probe_path.write_text("ok", encoding="utf-8")
    probe_path.unlink()


def _format_eta(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _call_model_hook(
    model: torch.nn.Module,
    attribute_name: str,
    hook_name: str,
    *args,
) -> None:
    model = unwrap_model(model)
    attribute = getattr(model, attribute_name, None)
    if attribute is None:
        return
    hook = getattr(attribute, hook_name, None)
    if callable(hook):
        hook(*args)


def _is_optional_research_key(key: str) -> bool:
    return (
        key == "dhm._extra_state"
        or key == "dhmr._extra_state"
        or key.startswith("dhm.")
        or key.startswith("dhmr.")
    )


def _get_module_summary(
    model: torch.nn.Module,
    attribute_name: str,
) -> dict[str, Any] | None:
    model = unwrap_model(model)
    module = getattr(model, attribute_name, None)
    if module is None:
        return None
    summary = getattr(module, "summary", None)
    if not callable(summary):
        return None
    return summary()


def _run_dhm_epoch_mining(
    *,
    model: torch.nn.Module,
    runtime_config: dict[str, Any],
    device: torch.device,
    amp: bool,
    log_interval: int,
    epoch: int,
    total_epochs: int,
    distributed: DistributedContext | None,
) -> None:
    base_model = unwrap_model(model)
    dhm = getattr(base_model, "dhm", None)
    should_mine = getattr(dhm, "should_mine", None)
    if dhm is None or not callable(should_mine) or not bool(should_mine(epoch=epoch)):
        barrier(distributed)
        return

    if not is_main_process(distributed):
        barrier(distributed)
        return

    hook = getattr(base_model, "mine_dhm_batch", None)
    if not callable(hook):
        barrier(distributed)
        return

    mining_loader = build_train_mining_dataloader(runtime_config)
    start_time = time.perf_counter()
    total_batches = len(mining_loader)
    total_gt = 0
    total_fn = 0
    total_relapses = 0
    total_recoveries = 0
    total_state_changes = 0
    was_training = base_model.training
    try:
        base_model.eval()
        for step, (images, targets) in enumerate(mining_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [_move_target_to_device(target, device) for target in targets]
            total_gt += sum(int(target["boxes"].shape[0]) for target in targets)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                stats = hook(images, targets, epoch=epoch)
            if isinstance(stats, Mapping):
                total_fn += int(stats.get("num_fn", 0))
                total_relapses += int(stats.get("relapses", 0))
                total_recoveries += int(stats.get("recoveries", 0))
                total_state_changes += int(stats.get("state_changes", 0))

            should_log = log_interval > 0 and (step % log_interval == 0 or step == total_batches)
            if should_log:
                elapsed = time.perf_counter() - start_time
                avg_step_time = elapsed / max(step, 1)
                remaining_steps = max(total_batches - step, 0)
                epoch_eta = _format_eta(avg_step_time * remaining_steps)
                print(
                    f"[dhm] epoch {epoch}/{total_epochs} "
                    f"mining_step {step}/{total_batches} "
                    f"epoch_eta={epoch_eta} fn={total_fn} "
                    f"relapses={total_relapses} recoveries={total_recoveries} "
                    f"state_changes={total_state_changes} gt={total_gt}"
                )
    finally:
        if was_training:
            base_model.train()
    barrier(distributed)
