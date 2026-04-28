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
- If `dhmr.border_refinement.enabled` is true and DHM-R has active DHM transition targets, FCOS adds training-only `dhmr_border_giou`, `dhmr_border_residual`, and `dhmr_border_quality` losses. Phase 1 uses dense positive points for `FN_LOC->FN_LOC` and `TP->FN_LOC` records only; inference-time refinement is not applied yet.
- If `train.hard_replay.enabled` is true and DHM records exist, the trainer duplicates current-batch images that contain persistent FN or relapse GTs. Replayed images are appended to the normal batch before the detector forward, so they use the same detection and DHM-R losses as the original view.

`history.json` stores DHM assignment aggregates under `dhm.assignment_by_state` and
`dhm.assignment_by_transition`. These summaries are intended to diagnose whether an FN type is
driven by missing positive assignment, poor localization/centerness quality, or ambiguous GT
competition before enabling a repair intervention.

`history.json` also stores DHM-R border-refinement aggregates under
`dhmr.border_refinement` when the module is enabled. These include selected point counts,
selected GT counts, and mean raw GIoU, residual, quality, and refined-IoU values for the
training-only auxiliary head.

## Hard Replay

`train.hard_replay` is a training-only replay policy driven by DHM state. It scans the current batch for GTs whose DHM record is a persistent FN or relapse transition, ranks matching images by DHM priority, and duplicates the top images into the same training batch.

`max_ratio` is the maximum replayed-image ratio relative to the original batch size. For example, `max_ratio: 0.25` allows at most 8 replay images for a 32-image batch. `warmup_epochs` can ramp the active ratio from zero to `max_ratio` after `start_epoch`.

Default disabled configuration:

```yaml
train:
  hard_replay:
    enabled: false
    start_epoch: 3
    warmup_epochs: 0
    max_ratio: 0.25
    max_replays_per_batch: 0
    target_transitions:
      - FN_BG->FN_BG
      - FN_CLS->FN_CLS
      - FN_LOC->FN_LOC
      - FN_MISS->FN_MISS
      - TP->FN_BG
      - TP->FN_CLS
      - TP->FN_LOC
      - TP->FN_MISS
    persistent_states:
      - FN_BG
      - FN_CLS
      - FN_LOC
      - FN_MISS
    min_observations: 2
    min_fn_streak: 2
```

`max_replays_per_batch: 0` means no explicit per-batch cap beyond `max_ratio`. The trainer logs `hard_replay_images`, `hard_replay_gt`, and `hard_replay_ratio` as non-loss training metrics in `history.json`.

## DHM-R Interaction

DHM-R depends on DHM. If DHM-R is enabled while DHM is disabled, registry construction raises an error.

When `dhmr.border_refinement.enabled` is true, DHM-R uses previous DHM transition records to select dense FCOS positive points for localization repair. Phase 1 is training-only: it adds `dhmr_border_giou`, `dhmr_border_residual`, and `dhmr_border_quality` auxiliary losses, but does not change image sampling, inference postprocessing, or evaluation behavior.

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
