from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim

from .config import dump_yaml_file
from .metrics import evaluate_coco_detection, save_predictions


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
) -> list[dict[str, Any]]:
    output_dir = Path(runtime_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(runtime_config["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _probe_writable_dir(checkpoint_dir)

    seed_everything(runtime_config["seed"])
    model.to(device)

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

    persist_run_metadata(
        output_dir=output_dir,
        arch=arch,
        model_config=model_config,
        model_config_path=model_config_path,
        runtime_config=runtime_config,
        runtime_config_path=runtime_config_path,
    )

    total_epochs = runtime_config["train"]["epochs"]
    eval_every = runtime_config["train"]["eval_every_epochs"]
    monitor = runtime_config["checkpoint"]["monitor"]
    mode = runtime_config["checkpoint"]["mode"]

    for epoch in range(start_epoch, total_epochs):
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            amp=runtime_config["amp"],
            scaler=scaler,
            log_interval=runtime_config["train"]["log_interval"],
            grad_clip_norm=runtime_config["train"]["grad_clip_norm"],
            epoch_index=epoch,
            total_epochs=total_epochs,
        )

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train": train_metrics,
        }

        should_eval = (
            val_loader is not None
            and ((epoch + 1) % eval_every == 0 or (epoch + 1) == total_epochs)
        )
        if should_eval:
            val_metrics, _ = evaluate(
                model=model,
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

        if runtime_config["checkpoint"]["save_last"]:
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

        history.append(record)
        _write_json(output_dir / "history.json", history)

        summary = format_metrics(record, runtime_config["metrics"]["primary"])
        print(f"[epoch {epoch + 1}/{total_epochs}] {summary}")

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
    model.load_state_dict(checkpoint["model_state_dict"])

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
        "model_state_dict": model.state_dict(),
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
) -> None:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml_file(metadata_dir / "model.yaml", model_config)
    dump_yaml_file(
        metadata_dir / f"{runtime_config['_mode']}.yaml",
        _strip_internal_keys(runtime_config),
    )

    metadata = {
        "arch": arch,
        "model_config_path": str(Path(model_config_path).resolve()),
        "runtime_config_path": str(Path(runtime_config_path).resolve()),
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

    return " ".join(parts)


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
