from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from collections import defaultdict
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
    plot_mdmb_curves,
    plot_mdmb_per_class,
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
        _call_model_hook(model, "mdmb", "start_epoch", epoch + 1)
        _call_model_hook(model, "mdmbpp", "start_epoch", epoch + 1)
        _call_model_hook(model, "rasd", "start_epoch", epoch + 1)
        _call_model_hook(model, "tfm", "start_epoch", epoch + 1)
        _call_model_hook(model, "fntdm", "start_epoch", epoch + 1)
        _call_model_hook(model, "dhm", "start_epoch", epoch + 1)
        _set_data_loader_epoch(train_loader, epoch + 1)
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
        _run_fntdm_epoch_mining(
            model=model,
            runtime_config=runtime_config,
            device=device,
            amp=runtime_config["amp"],
            log_interval=runtime_config["train"]["log_interval"] if main_process else 0,
            epoch=epoch + 1,
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
        _call_model_hook(model, "mdmb", "end_epoch", epoch + 1)
        _call_model_hook(model, "mdmbpp", "end_epoch", epoch + 1)
        _call_model_hook(model, "rasd", "end_epoch", epoch + 1)
        _call_model_hook(model, "tfm", "end_epoch", epoch + 1)
        _call_model_hook(model, "fntdm", "end_epoch", epoch + 1)
        _call_model_hook(model, "dhm", "end_epoch", epoch + 1)
        local_summaries = {
            "mdmb": _get_mdmb_summary(model),
            "mdmbpp": _get_module_summary(model, "mdmbpp"),
            "rasd": _get_module_summary(model, "rasd"),
            "hard_replay": _get_hard_replay_summary(train_loader),
            "tfm": _get_module_summary(model, "tfm"),
            "fntdm": _get_module_summary(model, "fntdm"),
            "dhm": _get_module_summary(model, "dhm"),
        }
        merged_epoch_summaries = _merge_epoch_summaries(local_summaries, distributed)
        _synchronize_research_memory(model, distributed)
        mdmb_summary = _overlay_epoch_summary(
            _get_mdmb_summary(model),
            merged_epoch_summaries.get("mdmb"),
            attribute_name="mdmb",
        )
        mdmbpp_summary = _overlay_epoch_summary(
            _get_module_summary(model, "mdmbpp"),
            merged_epoch_summaries.get("mdmbpp"),
            attribute_name="mdmbpp",
        )
        rasd_summary = merged_epoch_summaries.get("rasd")
        hard_replay_summary = merged_epoch_summaries.get("hard_replay")
        tfm_summary = _get_module_summary(model, "tfm")
        fntdm_summary = merged_epoch_summaries.get("fntdm")
        dhm_summary = merged_epoch_summaries.get("dhm")

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train": train_metrics,
        }
        if mdmb_summary is not None:
            record["mdmb"] = mdmb_summary
        if mdmbpp_summary is not None:
            record["mdmbpp"] = mdmbpp_summary
        if rasd_summary is not None:
            record["rasd"] = rasd_summary
        if hard_replay_summary is not None:
            record["hard_replay"] = hard_replay_summary
        if tfm_summary is not None:
            record["tfm"] = tfm_summary
        if fntdm_summary is not None:
            record["fntdm"] = fntdm_summary
        if dhm_summary is not None:
            record["dhm"] = dhm_summary

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
            f"→ saved to {output_dir / 'best_val_metrics.json'}"
        )

        print("[visualize] Generating training figures ...")
        plot_loss_curves(history, output_dir)
        plot_map_curves(history, output_dir)
        plot_mdmb_curves(history, output_dir)

        print("[visualize] Building confusion matrix ...")
        cm, class_names = build_confusion_matrix(best_val_predictions, val_loader)
        plot_confusion_matrices(cm, class_names, output_dir)
        plot_mdmb_per_class(unwrap_model(model), val_loader, output_dir)

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
        plot_mdmb_per_class(model, data_loader, resolved_output_dir)

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

        _call_after_optimizer_step(
            model,
            images,
            targets,
            epoch_index=epoch_index,
        )

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
    missing_keys = [key for key in state_result.missing_keys if not _is_optional_mdmb_key(key)]
    unexpected_keys = [
        key for key in state_result.unexpected_keys if not _is_optional_mdmb_key(key)
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

    mdmb_summary = record.get("mdmb")
    if isinstance(mdmb_summary, dict):
        parts.append(f"mdmb_entries={int(mdmb_summary.get('num_entries', 0))}")
        parts.append(f"mdmb_images={int(mdmb_summary.get('num_images', 0))}")

    mdmbpp_summary = record.get("mdmbpp")
    if isinstance(mdmbpp_summary, dict):
        parts.append(f"mdmbpp_entries={int(mdmbpp_summary.get('num_entries', 0))}")
        parts.append(f"mdmbpp_mean_severity={float(mdmbpp_summary.get('mean_severity', 0.0)):.3f}")

    hard_replay_summary = record.get("hard_replay")
    if isinstance(hard_replay_summary, dict):
        parts.append(f"replay_images={int(hard_replay_summary.get('replay_num_images', 0))}")
        parts.append(
            f"replay_ratio={float(hard_replay_summary.get('replay_ratio_effective', 0.0)):.3f}"
        )

    rasd_summary = record.get("rasd")
    if isinstance(rasd_summary, dict):
        parts.append(f"rasd_losses={int(rasd_summary.get('losses', 0))}")
        parts.append(f"rasd_targets={int(rasd_summary.get('targets', 0))}")

    tfm_summary = record.get("tfm")
    if isinstance(tfm_summary, dict):
        parts.append(f"tfm_records={int(tfm_summary.get('num_records', 0))}")
        parts.append(f"tfm_mean_risk={float(tfm_summary.get('mean_risk', 0.0)):.3f}")

    fntdm_summary = record.get("fntdm")
    if isinstance(fntdm_summary, dict):
        tdb = fntdm_summary.get("tdb", {})
        tal = fntdm_summary.get("tal", {})
        if isinstance(tdb, Mapping):
            parts.append(f"fntdm_tdb_entries={int(tdb.get('total_entries', 0))}")
        if isinstance(tal, Mapping):
            parts.append(f"fntdm_candidates={int(tal.get('valid_candidates', 0))}")

    dhm_summary = record.get("dhm")
    if isinstance(dhm_summary, dict):
        parts.append(f"dhm_records={int(dhm_summary.get('num_records', 0))}")
        parts.append(f"dhm_mean_instability={float(dhm_summary.get('mean_instability', 0.0)):.3f}")

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
    for name in ("mdmb", "mdmbpp", "rasd", "hard_replay", "tfm", "fntdm", "dhm"):
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
    if name == "rasd":
        _merge_count_keys(
            result,
            summaries,
            (
                "targets",
                "losses",
                "relapse_targets",
                "confuser_targets",
                "skipped_no_support",
                "skipped_support_too_old",
                "skipped_low_support_score",
                "skipped_no_entry_match",
                "skipped_no_feature",
            ),
        )
        _merge_weighted_mean(result, summaries, "mean_severity", "targets")
        _merge_weighted_mean(result, summaries, "mean_support_age", "targets")
        _merge_weighted_mean(result, summaries, "mean_target_weight", "targets")
        _merge_weighted_mean(result, summaries, "mean_support_loss", "losses")
        _merge_weighted_mean(result, summaries, "mean_contrastive_loss", "losses")
        return result
    if name == "mdmbpp":
        _merge_count_keys(
            result,
            summaries,
            (
                "relapses_this_epoch",
                "support_updates_this_epoch",
                "support_kept_this_epoch",
            ),
        )
        return result
    if name == "tfm":
        _merge_count_keys(
            result,
            summaries,
            (
                "num_records",
                "num_images",
                "num_current_failures",
                "num_support",
            ),
        )
        _merge_weighted_mean(result, summaries, "mean_risk", "num_records")
        result["global_max_miss_streak"] = max(
            int(summary.get("global_max_miss_streak", 0)) for summary in summaries
        )
        state_counts: defaultdict[str, int] = defaultdict(int)
        for summary in summaries:
            raw_counts = summary.get("state_counts", {})
            if not isinstance(raw_counts, Mapping):
                continue
            for state, count in raw_counts.items():
                state_counts[str(state)] += int(count)
        result["state_counts"] = dict(state_counts)
        return result
    if name == "fntdm":
        htm_counts: defaultdict[str, int] = defaultdict(int)
        tcs_counts: defaultdict[str, int] = defaultdict(int)
        tal_counts: defaultdict[str, int] = defaultdict(int)
        tdb_total_entries = 0
        tdb_classes = 0
        tal_loss_weighted = 0.0
        tal_cosine_weighted = 0.0
        tal_hardness_weighted = 0.0
        tal_loss_weight = 0
        tal_candidate_weight = 0
        for summary in summaries:
            htm = summary.get("htm", {})
            if isinstance(htm, Mapping):
                for key, value in htm.items():
                    if isinstance(value, (int, float)):
                        htm_counts[str(key)] += int(value)
            tcs = summary.get("tcs", {})
            if isinstance(tcs, Mapping):
                for key, value in tcs.items():
                    if isinstance(value, (int, float)):
                        tcs_counts[str(key)] += int(value)
            tal = summary.get("tal", {})
            if isinstance(tal, Mapping):
                for key, value in tal.items():
                    if isinstance(value, (int, float)) and not str(key).startswith("mean_"):
                        tal_counts[str(key)] += int(value)
                losses = int(tal.get("losses", 0))
                candidates = int(tal.get("valid_candidates", 0))
                tal_loss_weighted += float(tal.get("mean_loss", 0.0)) * float(losses)
                tal_cosine_weighted += float(tal.get("mean_cosine", 0.0)) * float(candidates)
                tal_hardness_weighted += float(tal.get("mean_hardness", 0.0)) * float(candidates)
                tal_loss_weight += losses
                tal_candidate_weight += candidates
            tdb = summary.get("tdb", {})
            if isinstance(tdb, Mapping):
                tdb_total_entries = max(tdb_total_entries, int(tdb.get("total_entries", 0)))
                tdb_classes = max(tdb_classes, int(tdb.get("num_classes_with_entries", 0)))
        result["htm"] = dict(htm_counts)
        result["tcs"] = dict(tcs_counts)
        result["tal"] = dict(tal_counts)
        result["tal"]["mean_loss"] = tal_loss_weighted / float(max(tal_loss_weight, 1))
        result["tal"]["mean_cosine"] = tal_cosine_weighted / float(max(tal_candidate_weight, 1))
        result["tal"]["mean_hardness"] = tal_hardness_weighted / float(max(tal_candidate_weight, 1))
        result["tdb"] = {
            "total_entries": tdb_total_entries,
            "num_classes_with_entries": tdb_classes,
        }
        return result
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

        mining_counts: defaultdict[str, int] = defaultdict(int)
        for summary in summaries:
            raw_mining = summary.get("mining", {})
            if not isinstance(raw_mining, Mapping):
                continue
            for key, value in raw_mining.items():
                mining_counts[str(key)] += int(value)
        result["mining"] = dict(mining_counts)
        return result
    if name == "hard_replay":
        _merge_count_keys(
            result,
            summaries,
            (
                "replay_samples",
                "replay_crop_samples",
                "replay_copy_paste_samples",
                "replay_pair_samples",
            ),
        )
        total_base = 0.0
        total_replay = float(result.get("replay_samples", 0.0))
        for summary in summaries:
            effective = float(summary.get("replay_ratio_effective", 0.0))
            replay_samples = float(summary.get("replay_samples", 0.0))
            if effective > 0.0:
                total_base += replay_samples * (1.0 - effective) / effective
        if total_base + total_replay > 0.0:
            result["replay_ratio_effective"] = total_replay / (total_base + total_replay)
        world_size = max(int(summary.get("ddp_world_size", 1)) for summary in summaries)
        if world_size > 1:
            result["ddp_world_size"] = world_size
            result["ddp_rank"] = 0
        return result
    return result


def _merge_count_keys(
    result: dict[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> None:
    for key in keys:
        result[key] = sum(int(summary.get(key, 0)) for summary in summaries)


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


def _overlay_epoch_summary(
    base_summary: dict[str, Any] | None,
    epoch_summary: dict[str, Any] | None,
    *,
    attribute_name: str,
) -> dict[str, Any] | None:
    if base_summary is None:
        return epoch_summary
    if epoch_summary is None:
        return base_summary
    merged = dict(base_summary)
    if attribute_name == "mdmbpp":
        for key in (
            "relapses_this_epoch",
            "support_updates_this_epoch",
            "support_kept_this_epoch",
        ):
            if key in epoch_summary:
                merged[key] = epoch_summary[key]
    return merged


def _synchronize_research_memory(
    model: torch.nn.Module,
    distributed: DistributedContext | None,
) -> None:
    if not is_distributed(distributed):
        return
    base_model = unwrap_model(model)
    sync_specs = (
        ("mdmb", _merge_mdmb_states),
        ("mdmbpp", _merge_mdmbpp_states),
        ("tfm", _merge_tfm_states),
        ("fntdm", _merge_fntdm_states),
        ("dhm", _merge_dhm_states),
    )
    for attribute_name, merge_fn in sync_specs:
        module = getattr(base_model, attribute_name, None)
        if module is None:
            continue
        synchronize_extra_state(module, distributed, merge_fn)


def _merge_mdmb_states(states: list[Any]) -> dict[str, Any] | None:
    valid = [state for state in states if isinstance(state, Mapping)]
    if not valid:
        return None
    merged = dict(valid[0])
    merged["current_epoch"] = max(int(state.get("current_epoch", 0)) for state in valid)
    merged["global_max_consecutive_miss"] = max(
        int(state.get("global_max_consecutive_miss", 0)) for state in valid
    )

    bank: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    for state in valid:
        raw_bank = state.get("bank", {})
        if not isinstance(raw_bank, Mapping):
            continue
        for image_id, entries in raw_bank.items():
            if not isinstance(entries, Sequence):
                continue
            image_key = str(image_id)
            image_entries = bank.setdefault(image_key, {})
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                key = _mdmb_entry_key(entry)
                current = image_entries.get(key)
                if current is None or _mdmb_entry_priority(entry) > _mdmb_entry_priority(current):
                    image_entries[key] = dict(entry)
    merged["bank"] = {
        image_id: list(entries.values())
        for image_id, entries in bank.items()
        if entries
    }

    gt_records: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    for state in valid:
        raw_records = state.get("gt_records", {})
        if not isinstance(raw_records, Mapping):
            continue
        for image_id, records in raw_records.items():
            if not isinstance(records, Sequence):
                continue
            image_key = str(image_id)
            image_records = gt_records.setdefault(image_key, {})
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                key = _mdmb_record_key(record)
                current = image_records.get(key)
                if current is None or _mdmb_record_priority(record) > _mdmb_record_priority(current):
                    image_records[key] = dict(record)
    merged["gt_records"] = {
        image_id: list(records.values())
        for image_id, records in gt_records.items()
        if records
    }
    return merged


def _merge_mdmbpp_states(states: list[Any]) -> dict[str, Any] | None:
    valid = [state for state in states if isinstance(state, Mapping)]
    if not valid:
        return None
    merged = dict(valid[0])
    merged["current_epoch"] = max(int(state.get("current_epoch", 0)) for state in valid)
    merged["global_max_consecutive_miss"] = max(
        int(state.get("global_max_consecutive_miss", 0)) for state in valid
    )

    records: dict[str, dict[str, Any]] = {}
    for state in valid:
        raw_records = state.get("persistent_records", {})
        if not isinstance(raw_records, Mapping):
            continue
        for gt_uid, record in raw_records.items():
            if not isinstance(record, Mapping):
                continue
            key = str(gt_uid)
            current = records.get(key)
            candidate = dict(record)
            if current is None:
                records[key] = candidate
                continue
            selected = _select_mdmbpp_record(current, candidate)
            selected["support"] = _select_support_snapshot(
                current.get("support"),
                candidate.get("support"),
            )
            records[key] = selected
    merged["persistent_records"] = records

    entries: dict[str, dict[str, Any]] = {}
    for state in valid:
        raw_bank = state.get("bank", {})
        if not isinstance(raw_bank, Mapping):
            continue
        for raw_entries in raw_bank.values():
            if not isinstance(raw_entries, Sequence):
                continue
            for entry in raw_entries:
                if not isinstance(entry, Mapping):
                    continue
                gt_uid = str(entry.get("gt_uid", ""))
                if not gt_uid:
                    continue
                candidate = dict(entry)
                current = entries.get(gt_uid)
                if current is None:
                    entries[gt_uid] = candidate
                    continue
                selected = _select_mdmbpp_entry(current, candidate, records)
                selected["support"] = _select_support_snapshot(
                    current.get("support"),
                    candidate.get("support"),
                )
                entries[gt_uid] = selected

    bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries.values():
        bank[str(entry.get("image_id", ""))].append(entry)
    merged["bank"] = {image_id: values for image_id, values in bank.items() if image_id}
    merged["class_statistics"] = _build_mdmbpp_class_statistics(records, entries)
    return merged


def _merge_tfm_states(states: list[Any]) -> dict[str, Any] | None:
    valid = [state for state in states if isinstance(state, Mapping)]
    if not valid:
        return None
    merged = dict(valid[0])
    merged["current_epoch"] = max(int(state.get("current_epoch", 0)) for state in valid)
    merged["global_max_miss_streak"] = max(
        int(state.get("global_max_miss_streak", 0)) for state in valid
    )

    records: dict[str, dict[str, Any]] = {}
    for state in valid:
        raw_records = state.get("records", {})
        if not isinstance(raw_records, Mapping):
            continue
        for gt_uid, record in raw_records.items():
            if not isinstance(record, Mapping):
                continue
            key = str(gt_uid)
            candidate = dict(record)
            current = records.get(key)
            if current is None:
                records[key] = candidate
                continue
            selected = _select_tfm_record(current, candidate)
            selected["support"] = _select_support_snapshot(
                current.get("support"),
                candidate.get("support"),
            )
            records[key] = selected

    image_index: defaultdict[str, list[str]] = defaultdict(list)
    for gt_uid, record in records.items():
        image_id = str(record.get("image_id", ""))
        if image_id:
            image_index[image_id].append(gt_uid)
    merged["records"] = records
    merged["image_index"] = {
        image_id: sorted(gt_uids)
        for image_id, gt_uids in image_index.items()
    }
    return merged


def _merge_fntdm_states(states: list[Any]) -> dict[str, Any] | None:
    valid = [state for state in states if isinstance(state, Mapping)]
    if not valid:
        return None
    merged = dict(valid[0])
    merged["current_epoch"] = max(int(state.get("current_epoch", 0)) for state in valid)

    config = merged.get("config", {})
    max_entries_per_class = 128
    if isinstance(config, Mapping):
        tdb_cfg = config.get("tdb", {})
        if isinstance(tdb_cfg, Mapping):
            storage = tdb_cfg.get("storage", {})
            if isinstance(storage, Mapping):
                max_entries_per_class = int(storage.get("max_entries_per_class", 128))

    entries_by_class: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    tdb_stats: defaultdict[str, int] = defaultdict(int)
    for state in valid:
        raw_tdb = state.get("tdb", {})
        if not isinstance(raw_tdb, Mapping):
            continue
        raw_stats = raw_tdb.get("stats", {})
        if isinstance(raw_stats, Mapping):
            for key, value in raw_stats.items():
                tdb_stats[str(key)] += int(value)
        raw_bank = raw_tdb.get("bank", {})
        if not isinstance(raw_bank, Mapping):
            continue
        for class_id, raw_entries in raw_bank.items():
            if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
                continue
            class_entries = entries_by_class[str(class_id)]
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, Mapping):
                    continue
                entry_id = str(raw_entry.get("entry_id", ""))
                if not entry_id:
                    continue
                current = class_entries.get(entry_id)
                if current is None or _fntdm_entry_priority(raw_entry) > _fntdm_entry_priority(current):
                    class_entries[entry_id] = dict(raw_entry)

    merged_bank = {}
    for class_id, entries in entries_by_class.items():
        sorted_entries = sorted(
            entries.values(),
            key=lambda item: _fntdm_entry_priority(item),
            reverse=True,
        )
        merged_bank[class_id] = sorted_entries[:max_entries_per_class]

    histories: dict[str, Mapping[str, Any]] = {}
    htm_stats: defaultdict[str, int] = defaultdict(int)
    for state in valid:
        raw_htm = state.get("htm", {})
        if not isinstance(raw_htm, Mapping):
            continue
        raw_stats = raw_htm.get("stats", {})
        if isinstance(raw_stats, Mapping):
            for key, value in raw_stats.items():
                htm_stats[str(key)] += int(value)
        raw_histories = raw_htm.get("histories", {})
        if not isinstance(raw_histories, Mapping):
            continue
        for uid, raw_history in raw_histories.items():
            if not isinstance(raw_history, Mapping):
                continue
            key = str(uid)
            current = histories.get(key)
            if current is None or _fntdm_history_priority(raw_history) > _fntdm_history_priority(current):
                histories[key] = dict(raw_history)

    merged["tdb"] = {
        "version": 1,
        "current_epoch": merged["current_epoch"],
        "bank": merged_bank,
        "stats": dict(tdb_stats),
    }
    merged["htm"] = {
        "histories": histories,
        "stats": dict(htm_stats),
    }
    return merged


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
            records[key] = _select_dhm_record(current, candidate)

    merged["records"] = records
    merged["stats"] = dict(stats)
    return merged


def _mdmb_entry_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(entry.get("image_id", "")),
        int(entry.get("class_id", -1)),
        _box_key(entry.get("bbox")),
    )


def _mdmb_record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(record.get("class_id", -1)),
        _box_key(record.get("bbox")),
    )


def _mdmb_entry_priority(entry: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(entry.get("consecutive_miss_count", 0)),
        float(entry.get("max_consecutive_miss_count", 0)),
        float(entry.get("last_detected_epoch") or -1),
    )


def _mdmb_record_priority(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(record.get("consecutive_miss_count", 0)),
        float(record.get("max_consecutive_miss_count", 0)),
        float(record.get("last_detected_epoch") or -1),
    )


def _select_mdmbpp_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    selected = left if _mdmbpp_record_priority(left) >= _mdmbpp_record_priority(right) else right
    return dict(selected)


def _select_mdmbpp_entry(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    left_priority = _mdmbpp_entry_priority(left, records)
    right_priority = _mdmbpp_entry_priority(right, records)
    return dict(left if left_priority >= right_priority else right)


def _select_tfm_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_priority = _tfm_record_priority(left)
    right_priority = _tfm_record_priority(right)
    return dict(left if left_priority >= right_priority else right)


def _mdmbpp_record_priority(record: Mapping[str, Any]) -> tuple[float, ...]:
    support = record.get("support")
    return (
        float(record.get("last_seen_epoch", 0)),
        float(record.get("severity", 0.0)),
        float(record.get("total_miss_count", 0)),
        float(record.get("relapse_count", 0)),
        _support_priority(support)[1],
    )


def _mdmbpp_entry_priority(
    entry: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[float, ...]:
    record = records.get(str(entry.get("gt_uid", "")), {})
    return (
        float(record.get("last_seen_epoch", 0)),
        float(entry.get("severity", 0.0)),
        float(entry.get("consecutive_miss_count", 0)),
        float(entry.get("total_miss_count", 0)),
        _support_priority(entry.get("support"))[1],
    )


def _tfm_record_priority(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(record.get("last_seen_epoch", 0)),
        float(record.get("risk", 0.0)),
        float(record.get("miss_streak", 0)),
        float(record.get("total_miss", 0)),
        float(record.get("relapse_count", 0)),
        _support_priority(record.get("support"))[1],
    )


def _fntdm_entry_priority(entry: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(entry.get("quality", 0.0)),
        float(entry.get("epoch_tp", 0)),
        float(entry.get("score_tp", 0.0)),
        float(entry.get("iou_tp", 0.0)),
    )


def _fntdm_history_priority(history: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(history.get("last_epoch") or 0),
        float(history.get("transition_count", 0)),
        float(history.get("tp_count", 0)),
        float(history.get("fn_count", 0)),
    )


def _select_dhm_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_priority = _dhm_record_priority(left)
    right_priority = _dhm_record_priority(right)
    return dict(left if left_priority >= right_priority else right)


def _dhm_record_priority(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(record.get("last_seen_epoch") or 0),
        float(record.get("instability_score", 0.0)),
        float(record.get("forgetting_count", 0)),
        float(record.get("fn_count", 0)),
        float(record.get("max_fn_streak", 0)),
        float(record.get("state_change_count", 0)),
    )


def _select_support_snapshot(left: Any, right: Any) -> Any:
    if not isinstance(left, Mapping):
        return dict(right) if isinstance(right, Mapping) else None
    if not isinstance(right, Mapping):
        return dict(left)
    return dict(left if _support_priority(left) >= _support_priority(right) else right)


def _support_priority(support: Any) -> tuple[float, ...]:
    if not isinstance(support, Mapping):
        return (0.0, 0.0, -1.0, 0.0, 0.0)
    has_feature = 1.0 if support.get("feature") is not None else 0.0
    feature_epoch = support.get("feature_epoch")
    if feature_epoch is None:
        feature_epoch = support.get("epoch", -1)
    return (
        has_feature,
        float(support.get("quality", 0.0)),
        float(feature_epoch if feature_epoch is not None else -1),
        float(support.get("score", 0.0)),
        float(support.get("iou_to_gt", 0.0)),
    )


def _build_mdmbpp_class_statistics(
    records: Mapping[str, Mapping[str, Any]],
    entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    severity_sums: dict[str, float] = defaultdict(float)
    for record in records.values():
        key = str(record.get("class_id", -1))
        class_stats = stats.setdefault(
            key,
            {
                "num_records": 0,
                "num_entries": 0,
                "total_miss_count": 0,
                "max_consecutive_miss_count": 0,
                "mean_severity": 0.0,
            },
        )
        class_stats["num_records"] = int(class_stats["num_records"]) + 1
        class_stats["total_miss_count"] = int(class_stats["total_miss_count"]) + int(
            record.get("total_miss_count", 0)
        )
        class_stats["max_consecutive_miss_count"] = max(
            int(class_stats["max_consecutive_miss_count"]),
            int(record.get("max_consecutive_miss_count", 0)),
        )
        severity_sums[key] += float(record.get("severity", 0.0))

    for entry in entries.values():
        key = str(entry.get("class_id", -1))
        class_stats = stats.setdefault(
            key,
            {
                "num_records": 0,
                "num_entries": 0,
                "total_miss_count": 0,
                "max_consecutive_miss_count": 0,
                "mean_severity": 0.0,
            },
        )
        class_stats["num_entries"] = int(class_stats["num_entries"]) + 1

    for key, class_stats in stats.items():
        num_records = int(class_stats["num_records"])
        class_stats["mean_severity"] = severity_sums.get(key, 0.0) / float(max(num_records, 1))
    return stats


def _box_key(raw_box: Any) -> tuple[float, ...]:
    if raw_box is None:
        return ()
    if isinstance(raw_box, torch.Tensor):
        values = raw_box.detach().cpu().flatten().tolist()
    elif isinstance(raw_box, Sequence) and not isinstance(raw_box, (str, bytes)):
        values = list(raw_box)
    else:
        values = [raw_box]
    return tuple(round(float(value), 6) for value in values)


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


def _call_after_optimizer_step(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    epoch_index: int,
) -> None:
    model = unwrap_model(model)
    hook = getattr(model, "after_optimizer_step", None)
    if callable(hook):
        hook(images, targets, epoch_index=epoch_index)


def _is_optional_mdmb_key(key: str) -> bool:
    return (
        key == "mdmb._extra_state"
        or key == "mdmbpp._extra_state"
        or key == "rasd._extra_state"
        or key == "tfm._extra_state"
        or key == "fntdm._extra_state"
        or key == "dhm._extra_state"
        or key.startswith("mdmb.")
        or key.startswith("mdmbpp.")
        or key.startswith("rasd.")
        or key.startswith("tfm.")
        or key.startswith("fntdm.")
        or key.startswith("dhm.")
    )


def _get_mdmb_summary(model: torch.nn.Module) -> dict[str, Any] | None:
    return _get_module_summary(model, "mdmb")


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


def _refresh_hard_replay(
    train_loader,
    model: torch.nn.Module,
    epoch: int,
) -> None:
    model = unwrap_model(model)
    controller = getattr(train_loader, "hard_replay", None)
    if controller is None:
        return
    start_epoch = getattr(controller, "start_epoch", None)
    if not callable(start_epoch):
        return
    start_epoch(mdmbpp=getattr(model, "mdmbpp", None), epoch=epoch)


def _run_fntdm_epoch_mining(
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
    fntdm = getattr(base_model, "fntdm", None)
    should_mine = getattr(fntdm, "should_mine", None)
    if fntdm is None or not callable(should_mine) or not bool(should_mine(epoch=epoch)):
        barrier(distributed)
        return

    if not is_main_process(distributed):
        barrier(distributed)
        return

    hook = getattr(base_model, "mine_fntdm_batch", None)
    if not callable(hook):
        barrier(distributed)
        return

    mining_loader = build_train_mining_dataloader(runtime_config)
    start_time = time.perf_counter()
    total_batches = len(mining_loader)
    total_events = 0
    total_gt = 0
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
                total_events += int(stats.get("events", 0))

            should_log = log_interval > 0 and (step % log_interval == 0 or step == total_batches)
            if should_log:
                elapsed = time.perf_counter() - start_time
                avg_step_time = elapsed / max(step, 1)
                remaining_steps = max(total_batches - step, 0)
                epoch_eta = _format_eta(avg_step_time * remaining_steps)
                print(
                    f"[fntdm] epoch {epoch}/{total_epochs} "
                    f"mining_step {step}/{total_batches} "
                    f"epoch_eta={epoch_eta} events={total_events} gt={total_gt}"
                )
    finally:
        if was_training:
            base_model.train()
    barrier(distributed)


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


def _get_hard_replay_summary(train_loader) -> dict[str, Any] | None:
    controller = getattr(train_loader, "hard_replay", None)
    if controller is None:
        return None
    summary = getattr(controller, "summary", None)
    if not callable(summary):
        return None
    return summary()
