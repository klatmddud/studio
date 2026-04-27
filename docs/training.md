# Training

## Entry Point

```bash
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti
```

Optional overrides:

```bash
uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --seed 42 \
  --device cuda:0 cuda:1 \
  --dhm-config modules/cfg/dhm.yaml \
  --dhmr-config modules/cfg/dhmr.yaml
```

Baseline seed-mean training script:

```bash
bash scripts/bash/baseline/train.bash
```

## Runtime Flow

`scripts/train.py`:

1. Loads runtime YAML with `load_runtime_config()`.
2. Applies CLI overrides for dataset selector, seed, device, output directory, and module config paths.
3. Builds the model through `scripts/runtime/registry.py`.
4. Builds train and validation DataLoaders through `scripts/runtime/data.py`.
5. Calls `engine.fit()`.

`engine.fit()`:

1. Seeds Python, NumPy, and PyTorch.
2. Moves the model to the selected device and wraps it in DDP when multiple CUDA devices are requested.
3. Builds optimizer, scheduler, and AMP scaler.
4. Resumes checkpoint state when `checkpoint.resume` is set.
5. For each epoch:
   - Calls `start_epoch()` on enabled `dhm` and `dhmr` modules.
   - Runs `train_one_epoch()`.
   - Runs DHM epoch-end mining when DHM config conditions allow it.
   - Calls `end_epoch()` on enabled `dhm` and `dhmr` modules.
   - Merges module summaries and synchronizes module state under DDP.
   - Runs validation when scheduled.
   - Writes `history.json` and checkpoints.

## Module Configs

Only DHM and DHM-R are currently wired:

| CLI flag | Default path |
|---|---|
| `--dhm-config` | `modules/cfg/dhm.yaml` |
| `--dhmr-config` | `modules/cfg/dhmr.yaml` |

Both configs are disabled by default. Enabled config snapshots are persisted to `metadata/modules.yaml`.

## DHM Interaction

DHM epoch-end mining uses `build_train_mining_dataloader()` with `shuffle: false`.

Mining behavior:

- Runs after the normal training epoch.
- Uses no-grad evaluation-style FCOS predictions over the train set.
- Updates per-GT states: `TP`, `FN_BG`, `FN_CLS`, `FN_LOC`, `FN_MISS`.
- Logs counts such as FN total, relapses, recoveries, state changes, and state-transition counts.

Training behavior:

- When DHM records exist, FCOS logs compact per-GT assignment statistics from the current training forward: positive count, FPN level histogram, centerness/loss means, near-candidate/near-negative count, and ambiguous points assigned to another GT.
- If `dhm.loss_weighting.enabled` is true and DHM has records, FCOS reweights raw classification, box, and centerness losses by GT state.
- If `dhm.assignment_expansion.enabled` is true and DHM has eligible records, FCOS adds backup positive points before loss computation.

`history.json` stores DHM assignment aggregates under `dhm.assignment_by_state` and
`dhm.assignment_by_transition`. These summaries are intended to diagnose whether an FN type is
driven by missing positive assignment, poor localization/centerness quality, or ambiguous GT
competition before enabling a repair intervention.

## DHM-R Interaction

DHM-R depends on DHM. If DHM-R is enabled while DHM is disabled, registry construction raises an error.

When enabled, DHM-R uses HLRT for eligible previous DHM `FN_LOC` records. HLRT adds independently configurable hooks under `dhmr.hlrt`: residual memory, residual replay, IoU loss weighting, side-aware loss, and a centerness quality gate.

HLRT residual replay changes only the FCOS training assignment by adding capped extra positive points. It does not duplicate images, change the dataset, or alter evaluation postprocessing. HLRT side-aware supervision is logged as `dhmr_hlrt_side`.

DHM-R can also enable `typed_film`, a training-only feature conditioning path for DHM `FN_LOC`, `FN_CLS`, and `FN_BG` records. It applies trainable FiLM embeddings to matched positive feature locations before the FCOS head, but it does not add an auxiliary loss or change evaluation postprocessing.

## Checkpointing

Checkpoint fields:

| Field | Description |
|---|---|
| `epoch` | Last completed epoch |
| `best_metric` | Best monitored metric so far |
| `model_state_dict` | Model and enabled module state |
| `optimizer_state_dict` | Optimizer state |
| `scheduler_state_dict` | Scheduler state, or `null` |

`checkpoint.save_last` writes `last.pt`. `checkpoint.save_best` writes `best.pt` when the monitored metric improves.

## Metrics And Outputs

Common outputs under `output_dir`:

| Path | Description |
|---|---|
| `history.json` | Epoch-level train, validation, DHM, and DHM-R summaries |
| `best_val_metrics.json` | Best-checkpoint validation metrics |
| `figures/loss.png` | Training loss curves |
| `figures/map.png` | Validation mAP curves |
| `figures/confusion_matrix.png` | COCO prediction confusion matrix |
| `metadata/run.json` | Resolved run metadata |
| `metadata/modules.yaml` | Enabled module config snapshots |

## Distributed Training

Multiple `--device` values trigger PyTorch DDP through `torch.multiprocessing.spawn`.

Rules:

- DDP requires CUDA devices and NCCL.
- Each rank trains independently during the epoch.
- DHM mining runs on rank 0.
- DHM and DHM-R `extra_state` payloads are merged and broadcast once per epoch.
