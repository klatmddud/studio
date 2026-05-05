from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from modules.nn import (
    compute_lmb_stability_metrics,
    merge_ftmb_epoch_snapshots,
    merge_lmb_epoch_snapshots,
)

from .config import dump_yaml_file
from .distributed import (
    DistributedContext,
    all_gather_object,
    barrier,
    is_distributed,
    is_main_process,
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
    ftmb_failure_history: list[dict[str, Any]] = []
    lmb_stability_history: list[dict[str, Any]] = []
    previous_lmb_snapshot: dict[str, Any] | None = None
    best_metric = _initial_best(runtime_config["checkpoint"]["mode"])
    start_epoch = 0

    resume_path = runtime_config["checkpoint"].get("resume")
    if resume_path:
        checkpoint_config = runtime_config["checkpoint"]
        resume_optimizer = bool(checkpoint_config.get("resume_optimizer", True))
        resume_scheduler = bool(checkpoint_config.get("resume_scheduler", True))
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer if resume_optimizer else None,
            scheduler=scheduler if resume_scheduler else None,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        best_metric = float(
            checkpoint.get("best_metric", _initial_best(runtime_config["checkpoint"]["mode"]))
        )
        if bool(checkpoint_config.get("reset_optimizer_lr", False)):
            _reset_optimizer_lr(
                optimizer,
                lr=float(runtime_config["optimizer"]["lr"]),
            )
        if scheduler is not None and not resume_scheduler:
            _align_scheduler_to_epoch(
                scheduler,
                epoch=start_epoch,
            )

    if main_process:
        if resume_path:
            ftmb_failure_history = _read_ftmb_failure_history(
                output_dir / "ftmb",
            )
            lmb_stability_history = _read_lmb_stability_history(
                output_dir / "lmb",
            )
            previous_lmb_snapshot = _read_lmb_stability_state(
                output_dir / "lmb" / "lmb_stability_state.json"
            )
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
        _set_data_loader_epoch(train_loader, epoch + 1)
        _set_missbank_epoch(model, epoch + 1)
        _set_ftmb_epoch(model, epoch + 1)
        _set_lmb_epoch(model, epoch + 1)
        _refresh_tar_replay(train_loader, model, epoch + 1)
        _refresh_hard_replay(train_loader, model, epoch + 1)

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
        _run_missbank_offline_mining(
            model=model,
            data_loader=train_loader,
            device=device,
            amp=runtime_config["amp"],
            epoch=epoch + 1,
            total_epochs=total_epochs,
            log_interval=runtime_config["train"]["log_interval"] if main_process else 0,
            distributed=distributed,
        )
        _run_lmb_offline_mining(
            model=model,
            data_loader=train_loader,
            device=device,
            amp=runtime_config["amp"],
            epoch=epoch + 1,
            total_epochs=total_epochs,
            log_interval=runtime_config["train"]["log_interval"] if main_process else 0,
            distributed=distributed,
        )
        current_ftmb_snapshot = _collect_ftmb_epoch_snapshot(
            model=model,
            epoch=epoch + 1,
            distributed=distributed,
        )
        if main_process and current_ftmb_snapshot is not None:
            ftmb_failure_history.append(current_ftmb_snapshot)
            _write_ftmb_failure_outputs(
                output_dir=output_dir,
                history=ftmb_failure_history,
                snapshot=current_ftmb_snapshot,
            )
        current_lmb_snapshot = _collect_lmb_epoch_snapshot(
            model=model,
            epoch=epoch + 1,
            distributed=distributed,
        )
        if main_process and current_lmb_snapshot is not None:
            lmb_metrics = compute_lmb_stability_metrics(
                current_lmb_snapshot,
                previous_snapshot=previous_lmb_snapshot,
                hotspot_top_k=int(current_lmb_snapshot.get("hotspot_top_k", 10)),
            )
            lmb_stability_history.append(lmb_metrics)
            _write_lmb_stability_outputs(
                output_dir=output_dir,
                subdir="lmb",
                history=lmb_stability_history,
                snapshot=current_lmb_snapshot,
            )
            previous_lmb_snapshot = current_lmb_snapshot

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train": train_metrics,
        }
        hard_replay_summary = _get_hard_replay_summary(train_loader)
        if hard_replay_summary is not None:
            record["hard_replay"] = hard_replay_summary
        tar_summary = _get_tar_summary(train_loader)
        if tar_summary is not None:
            record["tar"] = tar_summary

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
        save_every_epochs = runtime_config["checkpoint"].get("save_every_epochs")
        if (
            main_process
            and save_every_epochs is not None
            and (epoch + 1) % int(save_every_epochs) == 0
        ):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
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
            _write_history_outputs(output_dir, history)

            summary = format_metrics(record, runtime_config["metrics"]["primary"])
            print(f"[epoch {epoch + 1}/{total_epochs}] {summary}")

    best_pt = checkpoint_dir / "best.pt"
    if main_process and val_loader is not None and best_pt.is_file():
        print(f"[best_val] Loading best checkpoint: {best_pt}")
        best_checkpoint = load_checkpoint(best_pt, model=unwrap_model(model), map_location=device)
        best_epoch = int(best_checkpoint.get("epoch", 0))
        best_val_metrics, best_val_predictions = evaluate_coco_detection(
            model=unwrap_model(model),
            data_loader=val_loader,
            device=device,
            amp=runtime_config["amp"],
            log_interval=runtime_config["train"]["log_interval"],
            stage_label="best_val",
        )
        best_val_payload: dict[str, Any] = {"epoch": best_epoch}
        best_val_payload.update(best_val_metrics)
        _write_json(output_dir / "best_val_metrics.json", best_val_payload)
        primary = runtime_config["metrics"]["primary"]
        print(
            f"[best_val] epoch={best_epoch} "
            f"{primary}={best_val_metrics.get(primary, float('nan')):.4f} "
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
    metric_sums: defaultdict[str, float] = defaultdict(float)
    metric_counts: defaultdict[str, int] = defaultdict(int)
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
        _accumulate_model_train_metrics(
            model=model,
            metric_sums=metric_sums,
            metric_counts=metric_counts,
        )

        _update_missbank_for_batch(
            model=model,
            images=images,
            targets=targets,
            device=device,
            amp=amp,
            epoch=epoch_index + 1,
            step=step,
        )

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
            metric_sums=metric_sums,
            metric_counts=metric_counts,
            num_batches=num_batches,
            duration=duration,
            lr=float(optimizer.param_groups[0]["lr"]),
            distributed=distributed,
        )
    metrics = {
        key: value / max(num_batches, 1)
        for key, value in loss_sums.items()
    }
    for key, value in metric_sums.items():
        count = int(metric_counts.get(key, 0))
        if count > 0:
            metrics[key] = value / float(count)
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


def _reset_optimizer_lr(optimizer: optim.Optimizer, *, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _align_scheduler_to_epoch(scheduler, *, epoch: int) -> None:
    """Align a fresh scheduler to the global completed epoch after resume."""
    epoch_value = int(max(epoch, 0))
    optimizer = getattr(scheduler, "optimizer", None)
    param_groups = [] if optimizer is None else list(getattr(optimizer, "param_groups", []))
    base_lrs = list(getattr(scheduler, "base_lrs", []))
    if not param_groups or not base_lrs:
        return

    lrs = _scheduler_lrs_at_epoch(scheduler, epoch=epoch_value, base_lrs=base_lrs)
    for group, base_lr, lr in zip(param_groups, base_lrs, lrs, strict=False):
        group["initial_lr"] = float(base_lr)
        group["lr"] = float(lr)
    scheduler.last_epoch = epoch_value
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [float(lr) for lr in lrs]


def _scheduler_lrs_at_epoch(
    scheduler,
    *,
    epoch: int,
    base_lrs: list[float],
) -> list[float]:
    if isinstance(scheduler, optim.lr_scheduler.MultiStepLR):
        milestones = getattr(scheduler, "milestones", {})
        gamma = float(getattr(scheduler, "gamma", 0.1))
        decay_count = sum(
            int(count)
            for milestone, count in milestones.items()
            if int(milestone) <= int(epoch)
        )
        factor = gamma ** decay_count
        return [float(base_lr) * factor for base_lr in base_lrs]
    if isinstance(scheduler, optim.lr_scheduler.StepLR):
        step_size = int(getattr(scheduler, "step_size", 1))
        gamma = float(getattr(scheduler, "gamma", 0.1))
        decay_count = int(epoch) // max(step_size, 1)
        factor = gamma ** decay_count
        return [float(base_lr) * factor for base_lr in base_lrs]
    if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
        t_max = max(int(getattr(scheduler, "T_max", 1)), 1)
        eta_min = float(getattr(scheduler, "eta_min", 0.0))
        cosine = (1.0 + math.cos(math.pi * float(epoch) / float(t_max))) * 0.5
        return [eta_min + (float(base_lr) - eta_min) * cosine for base_lr in base_lrs]
    return [float(group["lr"]) for group in scheduler.optimizer.param_groups]


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler=None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    state_result = unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing_keys = [
        key for key in state_result.missing_keys
        if not _is_optional_checkpoint_state_key(str(key))
    ]
    unexpected_keys = [
        key for key in state_result.unexpected_keys
        if not _is_optional_checkpoint_state_key(str(key))
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


def _is_optional_checkpoint_state_key(key: str) -> bool:
    return key in {"missbank._extra_state", "ftmb._extra_state", "lmb._extra_state"}


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

    val_metrics = record.get("val")
    if isinstance(val_metrics, dict):
        primary = val_metrics.get(primary_metric)
        if primary is not None:
            parts.append(f"val_{primary_metric}={primary:.4f}")

    hard_replay = record.get("hard_replay")
    if isinstance(hard_replay, Mapping) and bool(hard_replay.get("enabled", False)):
        parts.append(f"replay_images={int(hard_replay.get('replay_num_images', 0))}")
        parts.append(f"replay_ratio={float(hard_replay.get('replay_ratio_effective', 0.0)):.3f}")
    tar = record.get("tar")
    if isinstance(tar, Mapping) and bool(tar.get("enabled", False)):
        parts.append(f"tar_images={int(tar.get('replay_num_images', 0))}")
        parts.append(f"tar_ratio={float(tar.get('replay_ratio_effective', 0.0)):.3f}")

    return " ".join(parts)


def _reduce_train_metrics(
    *,
    loss_sums: Mapping[str, float],
    metric_sums: Mapping[str, float],
    metric_counts: Mapping[str, int],
    num_batches: int,
    duration: float,
    lr: float,
    distributed: DistributedContext | None,
) -> dict[str, float]:
    gathered = all_gather_object(
        {
            "loss_sums": dict(loss_sums),
            "metric_sums": dict(metric_sums),
            "metric_counts": dict(metric_counts),
            "num_batches": int(num_batches),
            "duration": float(duration),
            "lr": float(lr),
        },
        distributed,
    )
    total_batches = sum(int(item.get("num_batches", 0)) for item in gathered if isinstance(item, Mapping))
    combined: defaultdict[str, float] = defaultdict(float)
    combined_metric_sums: defaultdict[str, float] = defaultdict(float)
    combined_metric_counts: defaultdict[str, int] = defaultdict(int)
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
        raw_metric_sums = item.get("metric_sums", {})
        if isinstance(raw_metric_sums, Mapping):
            for key, value in raw_metric_sums.items():
                combined_metric_sums[str(key)] += float(value)
        raw_metric_counts = item.get("metric_counts", {})
        if isinstance(raw_metric_counts, Mapping):
            for key, value in raw_metric_counts.items():
                combined_metric_counts[str(key)] += int(value)

    metrics = {
        key: value / float(max(total_batches, 1))
        for key, value in combined.items()
    }
    for key, value in combined_metric_sums.items():
        count = int(combined_metric_counts.get(key, 0))
        if count > 0:
            metrics[key] = value / float(count)
    metrics["lr"] = float(lr)
    metrics["epoch_time_sec"] = max_duration
    return metrics


def _set_data_loader_epoch(data_loader, epoch: int) -> None:
    sampler = getattr(data_loader, "sampler", None)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))
    batch_sampler = getattr(data_loader, "batch_sampler", None)
    set_epoch = getattr(batch_sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))


def _refresh_hard_replay(data_loader, model: torch.nn.Module, epoch: int) -> None:
    controller = getattr(data_loader, "hard_replay", None)
    refresh = getattr(controller, "refresh", None)
    if not callable(refresh):
        return
    refresh(
        missbank=_get_missbank(model),
        epoch=int(epoch),
    )


def _refresh_tar_replay(data_loader, model: torch.nn.Module, epoch: int) -> None:
    controller = getattr(data_loader, "tar_replay", None)
    refresh = getattr(controller, "refresh", None)
    if not callable(refresh):
        return
    refresh(
        ftmb=_get_ftmb(model),
        epoch=int(epoch),
    )


def _get_hard_replay_summary(data_loader) -> dict[str, Any] | None:
    controller = getattr(data_loader, "hard_replay", None)
    summary = getattr(controller, "summary", None)
    if not callable(summary):
        return None
    value = summary()
    return dict(value) if isinstance(value, Mapping) else None


def _get_tar_summary(data_loader) -> dict[str, Any] | None:
    controller = getattr(data_loader, "tar_replay", None)
    summary = getattr(controller, "summary", None)
    if not callable(summary):
        return None
    value = summary()
    return dict(value) if isinstance(value, Mapping) else None


def _set_replay_base_only(data_loader, enabled: bool) -> dict[str, bool]:
    states: dict[str, bool] = {}
    for attr in ("tar_replay", "hard_replay"):
        controller = getattr(data_loader, attr, None)
        batch_sampler = getattr(controller, "batch_sampler", None)
        if batch_sampler is None:
            continue
        states[attr] = bool(getattr(batch_sampler, "base_only", False))
        set_base_only = getattr(controller, "set_base_only", None)
        if callable(set_base_only):
            set_base_only(bool(enabled))
    return states


def _restore_replay_base_only(data_loader, states: Mapping[str, bool]) -> None:
    for attr, enabled in states.items():
        controller = getattr(data_loader, attr, None)
        set_base_only = getattr(controller, "set_base_only", None)
        if callable(set_base_only):
            set_base_only(bool(enabled))


def _move_target_to_device(
    target: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def _set_missbank_epoch(model: torch.nn.Module, epoch: int) -> None:
    for missbank in _iter_missbanks(model):
        start_epoch = getattr(missbank, "start_epoch", None)
        if callable(start_epoch):
            start_epoch(int(epoch))


def _set_ftmb_epoch(model: torch.nn.Module, epoch: int) -> None:
    for ftmb in _iter_ftmbs(model):
        start_epoch = getattr(ftmb, "start_epoch", None)
        if callable(start_epoch):
            start_epoch(int(epoch))


def _set_lmb_epoch(model: torch.nn.Module, epoch: int) -> None:
    for lmb in _iter_lmbs(model):
        start_epoch = getattr(lmb, "start_epoch", None)
        if callable(start_epoch):
            start_epoch(int(epoch))


def _accumulate_model_train_metrics(
    *,
    model: torch.nn.Module,
    metric_sums: defaultdict[str, float],
    metric_counts: defaultdict[str, int],
) -> None:
    get_metrics = getattr(unwrap_model(model), "get_training_metrics", None)
    if not callable(get_metrics):
        return
    metrics = get_metrics()
    if not isinstance(metrics, Mapping):
        return
    for name, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        metric_sums[str(name)] += numeric
        metric_counts[str(name)] += 1


def _update_missbank_for_batch(
    *,
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
    amp: bool,
    epoch: int,
    step: int,
) -> None:
    missbanks = [
        missbank
        for missbank in _iter_missbanks(model)
        if _missbank_enabled(missbank) and _missbank_mining_type(missbank) == "online"
    ]
    ftmbs = [
        ftmb
        for ftmb in _iter_ftmbs(model)
        if _ftmb_enabled(ftmb) and _ftmb_mining_type(ftmb) == "online"
    ]
    if not missbanks and not ftmbs:
        return

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            detections = model(images)
    finally:
        if was_training:
            model.train()

    for missbank in missbanks:
        _update_missbank_from_detections(
            missbank=missbank,
            targets=targets,
            detections=detections,
            epoch=epoch,
            step=step,
        )
    for ftmb in ftmbs:
        _update_ftmb_from_detections(
            ftmb=ftmb,
            targets=targets,
            detections=detections,
            epoch=epoch,
            step=step,
        )


def _run_missbank_offline_mining(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool,
    epoch: int,
    total_epochs: int,
    log_interval: int,
    distributed: DistributedContext | None,
) -> None:
    missbanks = [
        missbank
        for missbank in _iter_missbanks(model)
        if _missbank_enabled(missbank) and _missbank_mining_type(missbank) == "offline"
    ]
    ftmbs = [
        ftmb
        for ftmb in _iter_ftmbs(model)
        if _ftmb_enabled(ftmb) and _ftmb_mining_type(ftmb) == "offline"
    ]
    if not missbanks and not ftmbs:
        return

    was_training = model.training
    previous_base_only = _set_replay_base_only(data_loader, True)
    model.eval()
    start_time = time.perf_counter()
    total_steps = len(data_loader)
    try:
        for step, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [_move_target_to_device(target, device) for target in targets]
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                detections = model(images)

            for missbank in missbanks:
                _update_missbank_from_detections(
                    missbank=missbank,
                    targets=targets,
                    detections=detections,
                    epoch=epoch,
                    step=step,
                )
            for ftmb in ftmbs:
                _update_ftmb_from_detections(
                    ftmb=ftmb,
                    targets=targets,
                    detections=detections,
                    epoch=epoch,
                    step=step,
                )

            should_log = log_interval > 0 and (step % log_interval == 0 or step == total_steps)
            if should_log:
                elapsed = time.perf_counter() - start_time
                avg_step_time = elapsed / max(step, 1)
                remaining_steps = max(total_steps - step, 0)
                epoch_eta = _format_eta(avg_step_time * remaining_steps)
                print(
                    f"[remiss-mining] epoch {epoch}/{total_epochs} "
                    f"step {step}/{total_steps} epoch_eta={epoch_eta}"
                )
    finally:
        _restore_replay_base_only(data_loader, previous_base_only)
        if was_training:
            model.train()
    barrier(distributed)


def _run_lmb_offline_mining(
    *,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool,
    epoch: int,
    total_epochs: int,
    log_interval: int,
    distributed: DistributedContext | None,
) -> None:
    lmbs = [
        lmb
        for lmb in _iter_lmbs(model)
        if _lmb_active(lmb, epoch)
    ]
    if not lmbs:
        return

    was_training = model.training
    previous_base_only = _set_replay_base_only(data_loader, True)
    model.eval()
    start_time = time.perf_counter()
    total_steps = len(data_loader)
    try:
        for step, (images, targets) in enumerate(data_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [_move_target_to_device(target, device) for target in targets]
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                detections = model(images)

            for lmb in lmbs:
                _update_lmb_from_detections(
                    lmb=lmb,
                    targets=targets,
                    detections=detections,
                    epoch=epoch,
                    step=step,
                )

            should_log = log_interval > 0 and (step % log_interval == 0 or step == total_steps)
            if should_log:
                elapsed = time.perf_counter() - start_time
                avg_step_time = elapsed / max(step, 1)
                remaining_steps = max(total_steps - step, 0)
                epoch_eta = _format_eta(avg_step_time * remaining_steps)
                print(
                    f"[lmb-mining] epoch {epoch}/{total_epochs} "
                    f"step {step}/{total_steps} epoch_eta={epoch_eta}"
                )
    finally:
        _restore_replay_base_only(data_loader, previous_base_only)
        if was_training:
            model.train()
    barrier(distributed)


def _update_missbank_from_detections(
    *,
    missbank: Any,
    targets: list[dict[str, torch.Tensor]],
    detections: Any,
    epoch: int,
    step: int,
) -> None:
    update = getattr(missbank, "update", None)
    if callable(update):
        update(
            targets=targets,
            detections=detections,
            epoch=int(epoch),
            step=int(step),
        )


def _update_ftmb_from_detections(
    *,
    ftmb: Any,
    targets: list[dict[str, torch.Tensor]],
    detections: Any,
    epoch: int,
    step: int,
) -> None:
    update = getattr(ftmb, "update", None)
    if callable(update):
        update(
            targets=targets,
            detections=detections,
            epoch=int(epoch),
            step=int(step),
        )


def _update_lmb_from_detections(
    *,
    lmb: Any,
    targets: list[dict[str, torch.Tensor]],
    detections: Any,
    epoch: int,
    step: int,
) -> None:
    update = getattr(lmb, "update", None)
    if callable(update):
        update(
            targets=targets,
            detections=detections,
            epoch=int(epoch),
            step=int(step),
        )


def _collect_ftmb_epoch_snapshot(
    *,
    model: torch.nn.Module,
    epoch: int,
    distributed: DistributedContext | None,
) -> dict[str, Any] | None:
    ftmb = _get_ftmb(model)
    snapshot = None
    if _ftmb_enabled(ftmb):
        epoch_snapshot = getattr(ftmb, "epoch_snapshot", None)
        if callable(epoch_snapshot):
            snapshot = epoch_snapshot(epoch=int(epoch))
    gathered = all_gather_object(snapshot, distributed)
    return merge_ftmb_epoch_snapshots(gathered)


def _collect_lmb_epoch_snapshot(
    *,
    model: torch.nn.Module,
    epoch: int,
    distributed: DistributedContext | None,
) -> dict[str, Any] | None:
    lmb = _get_lmb(model)
    snapshot = None
    if _lmb_active(lmb, epoch):
        epoch_snapshot = getattr(lmb, "epoch_snapshot", None)
        if callable(epoch_snapshot):
            snapshot = epoch_snapshot(epoch=int(epoch))
    gathered = all_gather_object(snapshot, distributed)
    return merge_lmb_epoch_snapshots(gathered)


def _get_missbank(model: torch.nn.Module, *, attr: str = "missbank"):
    return getattr(unwrap_model(model), attr, None)


def _get_ftmb(model: torch.nn.Module, *, attr: str = "ftmb"):
    return getattr(unwrap_model(model), attr, None)


def _get_lmb(model: torch.nn.Module, *, attr: str = "lmb"):
    return getattr(unwrap_model(model), attr, None)


def _iter_missbanks(model: torch.nn.Module):
    unwrapped = unwrap_model(model)
    for attr in ("missbank",):
        missbank = getattr(unwrapped, attr, None)
        if missbank is not None:
            yield missbank


def _iter_ftmbs(model: torch.nn.Module):
    unwrapped = unwrap_model(model)
    for attr in ("ftmb",):
        ftmb = getattr(unwrapped, attr, None)
        if ftmb is not None:
            yield ftmb


def _iter_lmbs(model: torch.nn.Module):
    unwrapped = unwrap_model(model)
    for attr in ("lmb",):
        lmb = getattr(unwrapped, attr, None)
        if lmb is not None:
            yield lmb


def _missbank_enabled(missbank: Any) -> bool:
    if missbank is None:
        return False
    config = getattr(missbank, "config", None)
    return bool(getattr(config, "enabled", False))


def _missbank_mining_type(missbank: Any) -> str:
    config = getattr(missbank, "config", None)
    mining = getattr(config, "mining", None)
    return str(getattr(mining, "type", "online")).lower()


def _ftmb_enabled(ftmb: Any) -> bool:
    if ftmb is None:
        return False
    config = getattr(ftmb, "config", None)
    return bool(getattr(config, "enabled", False))


def _ftmb_mining_type(ftmb: Any) -> str:
    config = getattr(ftmb, "config", None)
    return str(getattr(config, "mining_type", "online")).lower()


def _lmb_enabled(lmb: Any) -> bool:
    if lmb is None:
        return False
    config = getattr(lmb, "config", None)
    return bool(getattr(config, "enabled", False))


def _lmb_active(lmb: Any, epoch: int) -> bool:
    if not _lmb_enabled(lmb):
        return False
    is_active = getattr(lmb, "is_active", None)
    if callable(is_active):
        return bool(is_active(int(epoch)))
    return True


def _write_ftmb_failure_outputs(
    *,
    output_dir: Path,
    history: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> None:
    ftmb_dir = output_dir / "ftmb"
    _write_json(ftmb_dir / "failure_type_epoch.json", history)
    _write_history_csv(ftmb_dir / "failure_type_epoch.csv", history)
    _write_json(ftmb_dir / "failure_type_state.json", {"snapshot": snapshot})


def _write_lmb_stability_outputs(
    *,
    output_dir: Path,
    subdir: str,
    history: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> None:
    lmb_dir = output_dir / subdir
    _write_json(lmb_dir / "lmb_stability_epoch.json", history)
    _write_history_csv(lmb_dir / "lmb_stability_epoch.csv", history)
    _write_json(lmb_dir / "lmb_stability_state.json", {"snapshot": snapshot})


def _read_ftmb_failure_history(ftmb_dir: Path) -> list[dict[str, Any]]:
    return _read_json_list(ftmb_dir / "failure_type_epoch.json")


def _read_lmb_stability_history(lmb_dir: Path) -> list[dict[str, Any]]:
    return _read_json_list(lmb_dir / "lmb_stability_epoch.json")


def _read_json_list(path: str | Path) -> list[dict[str, Any]]:
    json_path = Path(path)
    if not json_path.is_file():
        return []
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list in {json_path}.")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _read_missbank_stability_state(path: str | Path) -> dict[str, Any] | None:
    json_path = Path(path)
    if not json_path.is_file():
        return None
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        return None
    snapshot = payload.get("snapshot")
    return dict(snapshot) if isinstance(snapshot, Mapping) else None


def _read_lmb_stability_state(path: str | Path) -> dict[str, Any] | None:
    return _read_missbank_stability_state(path)


def _write_history_outputs(output_dir: Path, history: list[dict[str, Any]]) -> None:
    _write_json(output_dir / "history.json", history)
    _write_history_csv(output_dir / "results.csv", history)


def _write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    rows = [_flatten_history_record(record) for record in history]
    if not rows:
        return
    fieldnames = _history_csv_fieldnames(rows)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _flatten_history_record(
    record: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        flat_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_history_record(value, prefix=flat_key))
        else:
            flattened[flat_key] = value
    return flattened


def _history_csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys = sorted({key for row in rows for key in row})
    if "epoch" in keys:
        keys.remove("epoch")
        return ["epoch", *keys]
    return keys


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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
