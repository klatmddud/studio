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
  --remiss-config modules/cfg/remiss.yaml \
  --hard-replay-config modules/cfg/hard_replay.yaml
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
5. For each epoch, runs `train_one_epoch()`, scheduled validation, `history.json` updates, and checkpoint writes. If `train.max_iterations` is set, training stops after that many optimizer updates even if `train.epochs` has not been exhausted.

## Iteration-Budget Training

The training loop supports both epoch-budget and iteration-budget runs.

| Field | Description |
|---|---|
| `train.epochs` | Maximum epoch count. Still acts as a safety cap when `max_iterations` is set. |
| `train.max_iterations` | Optional global optimizer-step budget. Set to `null` for pure epoch-based training. |
| `train.eval_every_epochs` | Optional epoch validation interval. Set to `null` when validation should be iteration-only. |
| `train.eval_every_iterations` | Optional global optimizer-step validation interval. |
| `scheduler.unit` | `epoch` steps the scheduler after each epoch; `iteration` steps it after each optimizer update. |

For iteration-based PASCAL-style comparisons, use iteration milestones:

```yaml
scheduler:
  name: multistep
  unit: iteration
  milestones: [20000, 27000]
  gamma: 0.1

train:
  epochs: 200
  max_iterations: 30000
  eval_every_epochs: null
  eval_every_iterations: 1000
```

`history.json` and `results.csv` include an `iteration` column when iteration-budget training is active. Mid-epoch validation records are written at the triggering iteration, while epoch-end records continue to capture epoch-level train summaries and module outputs.

## Module Configs

When ReMiss is enabled, `scripts/runtime/registry.py` attaches MissBank to FCOS or Faster R-CNN. `matching.score_threshold: auto` and `matching.iou_threshold: auto` resolve matching from the detector's final post-processing config: FCOS uses `head.score_thresh` and `head.nms_thresh`, while Faster R-CNN uses `roi_head.box_score_thresh` and `roi_head.box_nms_thresh`. `modules/cfg/remiss.yaml` controls mining with `mining.type: online` or `mining.type: offline`. Online mining runs an eval-style no-grad detection pass after each optimization step. Offline mining skips per-step updates and runs one additional no-grad pass over the training loader on epochs that satisfy `mining.start_epoch` and `mining.interval_epoch`. On epochs where offline MissBank mining runs, `history.json` and `results.csv` include top-level `remiss_mining_time_sec`. MissBank writes epoch-level summaries under `output_dir/missbank/`. When `loss_weight.enabled: true` for FCOS, MissBank `miss_count` reweights positive classification, box regression, and centerness losses; Faster R-CNN remains logging/replay-only. Set `loss_weight.hard_replay_only: true` to apply weights only to GTs in the current Hard Replay replay-slot sample.

When Hard Replay is enabled for FCOS or Faster R-CNN, `scripts/runtime/data.py` replaces the normal train sampler with a mixed replay batch sampler. The replay index is refreshed at epoch start from ReMiss MissBank records. A GT is a replay target when MissBank says it is currently missed under the detector's final class/score/IoU matching thresholds, with no FN subtype split. Images containing eligible missed GTs are sampled into replay slots with priority based on missed-GT count and streak diagnostics. Replay slot samples carry `hard_replay` metadata and the active MissBank GT keys so FCOS loss weighting can distinguish replay occurrences from the base pass. `replay_epochs_after_mining > 0` limits Hard Replay to the first N epochs after the latest MissBank mining epoch, while `0` leaves it unlimited. Hard Replay epoch summaries are written under `output_dir/hard-replay/` instead of being mixed into the main `results.csv`.

MissBank offline mining temporarily switches the train loader to base-only iteration, so mining still sees the base training set once rather than replay-augmented batches.

| CLI flag | Default path |
|---|---|
| `--remiss-config` | `modules/cfg/remiss.yaml` |
| `--hard-replay-config` | `modules/cfg/hard_replay.yaml` |

Enabled module config snapshots are persisted to `metadata/modules.yaml`.

## Checkpointing

Checkpoint fields:

| Field | Description |
|---|---|
| `epoch` | Last completed epoch |
| `iteration` | Last completed optimizer update |
| `best_metric` | Best monitored metric so far |
| `model_state_dict` | Model state |
| `optimizer_state_dict` | Optimizer state |
| `scheduler_state_dict` | Scheduler state, or `null` |

`checkpoint.save_last` writes `last.pt`. `checkpoint.save_best` writes `best.pt` when the monitored metric improves. `checkpoint.save_every_epochs` can be set to a positive integer to additionally write periodic checkpoints such as `epoch_0020.pt`; set it to `null` to disable periodic checkpointing.

Resume controls:

| Field | Default | Description |
|---|---:|---|
| `checkpoint.resume_optimizer` | `true` | Load optimizer momentum/state from the checkpoint |
| `checkpoint.resume_scheduler` | `true` | Load scheduler state from the checkpoint |
| `checkpoint.reset_optimizer_lr` | `false` | After loading optimizer state, reset optimizer param-group LR to `optimizer.lr` from the YAML |

To resume model weights while making the LR schedule follow the current YAML milestones, set:

```yaml
checkpoint:
  resume: /path/to/checkpoint.pt
  resume_optimizer: true
  resume_scheduler: false
  reset_optimizer_lr: true
```

When `resume_scheduler: false`, the fresh scheduler is aligned to the resumed global epoch for `scheduler.unit: epoch`, or to the resumed global optimizer update for `scheduler.unit: iteration`.

When resuming a baseline checkpoint with ReMiss MissBank newly enabled, `missbank._extra_state` is allowed to be missing. MissBank starts from its configured initial state. Detector weights, epoch, and `best_metric` are restored from the checkpoint; optimizer and scheduler state follow the resume controls above.

## Metrics And Outputs

`metrics.type` controls the validation metric family while the dataset loader remains
COCO-format:

| `metrics.type` | Primary metric | Description |
|---|---|---|
| `voc_detection` | `voc_mAP_50` | PASCAL VOC-style mAP at IoU 0.5 using the VOC 2007 11-point AP rule. Per-class AP is written as `voc_AP_50_<class_name>`. |
| `coco_detection` | `bbox_mAP_50_95` | COCOeval bbox metrics including AP50:95, AP50, AP75, and size-specific AP/AR. |

For PASCAL VOC 2007 runs, set both the monitored checkpoint metric and the primary
metric to VOC mAP:

```yaml
checkpoint:
  monitor: voc_mAP_50

metrics:
  type: voc_detection
  iou_types: [bbox]
  primary: voc_mAP_50
```

Common outputs under `output_dir`:

| Path | Description |
|---|---|
| `history.json` | Train and validation metrics by epoch and, when enabled, global iteration; includes `remiss_mining_time_sec` on epochs where offline MissBank mining runs |
| `results.csv` | Flattened CSV view of `history.json` for spreadsheet-style analysis |
| `checkpoints/last.pt` | Last checkpoint when `checkpoint.save_last` is enabled |
| `checkpoints/best.pt` | Best monitored checkpoint when `checkpoint.save_best` is enabled |
| `checkpoints/epoch_0020.pt` | Periodic checkpoint example written by `checkpoint.save_every_epochs: 20` |
| `hard-replay/hard_replay_epoch.json` | Epoch-level Hard Replay candidate and exposure summaries |
| `hard-replay/hard_replay_epoch.csv` | Flattened CSV view of `hard_replay_epoch.json` |
| `hard-replay/hard_replay_state.json` | Last Hard Replay replay summary |
| `missbank/missbank_epoch.json` | Epoch-level ReMiss MissBank summary metrics accumulated as a JSON list |
| `missbank/missbank_epoch.csv` | Flattened CSV view of `missbank_epoch.json` |
| `last_val_metrics.json` | Last-checkpoint epoch, iteration, and validation metrics |
| `best_val_metrics.json` | Best-checkpoint epoch, iteration, and validation metrics |
| `figures/loss.png` | Training loss curves |
| `figures/map.png` | Validation mAP curves |
| `figures/confusion_matrix.png` | COCO-format prediction confusion matrix |
| `metadata/run.json` | Resolved run metadata |
| `metadata/modules.yaml` | Enabled module config snapshots |

## Distributed Training

Multiple `--device` values trigger PyTorch DDP through `torch.multiprocessing.spawn`.

Rules:

- DDP requires CUDA devices and NCCL.
- Each rank trains independently during the epoch.
- Train metrics are reduced across ranks before logging.
