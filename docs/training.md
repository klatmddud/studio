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
   - Refreshes the Hard Replay index from DHM when `train.hard_replay.enabled` is true.
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
- If `train.hard_replay.enabled` is true and DHM records exist, the train DataLoader uses a mixed batch sampler. Replay can add DHM-hard full-image samples and, for `FN_LOC`, DHM-guided localization repair crops while keeping each emitted batch at the configured batch size when enough base samples remain.

`history.json` stores DHM assignment aggregates under `dhm.assignment_by_state` and
`dhm.assignment_by_transition`. These summaries are intended to diagnose whether an FN type is
driven by missing positive assignment, poor localization/centerness quality, or ambiguous GT
competition before enabling a repair intervention.

`history.json` also stores DHM-R border-refinement aggregates under
`dhmr.border_refinement` when the module is enabled. These include selected point counts,
selected GT counts, and mean raw GIoU, residual, quality, and refined-IoU values for the
training-only auxiliary head.

## Hard Replay

`train.hard_replay` is a training-only replay policy driven by DHM state. At epoch start, the trainer builds a replay index from DHM records, ranks images by the severity of persistent FN or relapse GTs, and lets the DataLoader yield mixed batches with normal base samples plus replay samples.

`max_ratio` is the maximum replay slot ratio inside each configured batch. For example, batch size 32 and `max_ratio: 0.25` gives 24 base slots and 8 replay slots. The epoch length becomes approximately `ceil(num_images / 24)` when replay is active, rather than `ceil(num_images / 32)`. `warmup_epochs` can ramp the active ratio from zero to `max_ratio` after `start_epoch`.

When `loc_repair.enabled` is true, `FN_LOC` replay candidates can be emitted as GT-centered crops instead of full images. `loc_repair.replay_fraction` is applied to the per-batch replay slots with Python-style `round()`. For example, batch size 32, `max_ratio: 0.25`, and `replay_fraction: 0.6` gives 8 replay slots and `round(8 * 0.6) = 5` preferred localization-repair crop slots; remaining replay slots use full-image hard replay. If FN_LOC crop candidates are exhausted, full-image replay fills the remaining replay capacity.

Default disabled configuration:

```yaml
train:
  hard_replay:
    enabled: false
    start_epoch: 3
    warmup_epochs: 0
    max_ratio: 0.25
    max_replays_per_batch: 0
    beta: 1.0
    temperature: 1.0
    max_image_weight: 5.0
    min_replay_weight: 1.0
    replacement: true
    max_replays_per_gt_per_epoch: 4
    replay_recency_window: 3
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
    loc_repair:
      enabled: true
      replay_fraction: 0.6
      context_scale: 2.0
      context_scale_jitter: 0.25
      center_jitter: 0.10
      min_crop_size: 128
      min_visible_ratio: 0.50
      focus_min_visible_ratio: 0.90
      include_other_gt: true
      target_transitions:
        - FN_LOC->FN_LOC
        - TP->FN_LOC
      persistent_states:
        - FN_LOC
```

`max_replays_per_batch: 0` means no explicit per-batch cap beyond `max_ratio`. Replay candidates are filtered by `target_transitions`, `persistent_states`, `min_observations`, `min_fn_streak`, and `replay_recency_window`. `loc_repair` uses the same DHM recency/observation gates, then narrows candidates to localization failures through `FN_LOC->FN_LOC`, `TP->FN_LOC`, and persistent `FN_LOC` records.

Localization repair crops preserve the original image ID and annotation IDs, shift boxes into crop-local coordinates, and include other GTs whose visible area exceeds `min_visible_ratio`. Crop replay targets are excluded from DHM assignment-stat logging and DHM-R auxiliary selection so DHM memory remains anchored to full-image mining.

The trainer logs replay statistics under the top-level `hard_replay` key in `history.json`, including `replay_num_images`, `replay_num_active_gt`, `replay_ratio_requested`, `replay_ratio_effective`, `replay_sample_budget`, `replay_samples`, `replay_unique_images`, `image_replay_samples`, `loc_repair_samples`, `loc_repair_unique_images`, and `loc_repair_unique_gt`.

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
