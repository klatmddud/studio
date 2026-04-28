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
- If `dhmr.border_refinement.enabled` is true and DHM-R has active DHM transition targets, FCOS adds training-only `dhmr_border_giou`, `dhmr_border_residual`, and `dhmr_border_quality` losses. Phase 1 uses dense positive points for `FN_LOC->FN_LOC` and `TP->FN_LOC` records only; inference-time refinement is not applied yet.
- If `train.hard_crop_second_view.enabled` is true and DHM records exist, the trainer adds GT-centered crop second views for hard DHM records. The second-view forward uses the normal detector losses with a configurable weight, but marks crop targets as `_second_view` so FCOS skips DHM and DHM-R logging/losses for those crop-coordinate targets.

`history.json` stores DHM assignment aggregates under `dhm.assignment_by_state` and
`dhm.assignment_by_transition`. These summaries are intended to diagnose whether an FN type is
driven by missing positive assignment, poor localization/centerness quality, or ambiguous GT
competition before enabling a repair intervention.

`history.json` also stores DHM-R border-refinement aggregates under
`dhmr.border_refinement` when the module is enabled. These include selected point counts,
selected GT counts, and mean raw GIoU, residual, quality, and refined-IoU values for the
training-only auxiliary head.

## Hard-Crop Second View

`train.hard_crop_second_view` is a training-only hard-view augmentation driven by DHM state. It selects GTs from the current batch whose DHM record is either persistent `FN_LOC` or has a target transition such as `FN_LOC->FN_LOC` or `TP->FN_LOC`, then creates an enlarged crop around that GT and runs a second detector forward on the crop.

Default disabled configuration:

```yaml
train:
  hard_crop_second_view:
    enabled: false
    start_epoch: 3
    loss_weight: 0.25
    max_views_per_image: 1
    max_views_per_batch: 8
    crop_scale_min: 1.6
    crop_scale_max: 2.4
    jitter: 0.15
    min_crop_size: 96
    include_other_gt: true
    min_box_size: 2.0
    target_transitions:
      - FN_LOC->FN_LOC
      - TP->FN_LOC
    persistent_states:
      - FN_LOC
    min_observations: 2
    min_fn_streak: 2
```

When enabled, the crop target keeps all valid GT boxes inside the crop by default (`include_other_gt: true`) to avoid teaching false negatives. The second-view loss components are logged with the `second_view_` prefix, and `second_view_images` / `second_view_targets` report the average crop views and crop GTs per training batch.

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
