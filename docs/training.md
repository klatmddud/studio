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
  --remiss-config modules/cfg/remiss.yaml
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
5. For each epoch, runs `train_one_epoch()`, scheduled validation, `history.json` updates, and checkpoint writes.

## Module Configs

When ReMiss is enabled, `scripts/runtime/registry.py` attaches MissBank to the model. `matching.score_threshold: auto` and `matching.iou_threshold: auto` resolve MissBank matching from the detector's final post-processing config, so FCOS uses `head.score_thresh` and `head.nms_thresh`. `modules/cfg/remiss.yaml` controls MissBank mining with `mining.type: online` or `mining.type: offline`. Online mining runs an eval-style no-grad detection pass after each optimization step. Offline mining skips per-step updates and runs one additional no-grad pass over the training loader after each epoch, before ReMiss stability metrics are written. MissBank does not add detector losses or feature injection.

| CLI flag | Default path |
|---|---|
| `--remiss-config` | `modules/cfg/remiss.yaml` |

Enabled module config snapshots are persisted to `metadata/modules.yaml`.

## Checkpointing

Checkpoint fields:

| Field | Description |
|---|---|
| `epoch` | Last completed epoch |
| `best_metric` | Best monitored metric so far |
| `model_state_dict` | Model state |
| `optimizer_state_dict` | Optimizer state |
| `scheduler_state_dict` | Scheduler state, or `null` |

`checkpoint.save_last` writes `last.pt`. `checkpoint.save_best` writes `best.pt` when the monitored metric improves. `checkpoint.save_every_epochs` can be set to a positive integer to additionally write periodic checkpoints such as `epoch_0020.pt`; set it to `null` to disable periodic checkpointing.

When resuming a baseline checkpoint with ReMiss MissBank newly enabled, `missbank._extra_state` is allowed to be missing and MissBank starts from an empty state. Detector weights, optimizer state, scheduler state, epoch, and `best_metric` are still restored from the checkpoint.

## Metrics And Outputs

Common outputs under `output_dir`:

| Path | Description |
|---|---|
| `history.json` | Epoch-level train and validation metrics |
| `results.csv` | Flattened CSV view of `history.json` for spreadsheet-style analysis |
| `checkpoints/last.pt` | Last checkpoint when `checkpoint.save_last` is enabled |
| `checkpoints/best.pt` | Best monitored checkpoint when `checkpoint.save_best` is enabled |
| `checkpoints/epoch_0020.pt` | Periodic checkpoint example written by `checkpoint.save_every_epochs: 20` |
| `remiss/miss_stability_epoch.json` | Epoch-level MissBank stability metrics accumulated as a JSON list |
| `remiss/miss_stability_epoch.csv` | Flattened CSV view of `miss_stability_epoch.json` |
| `remiss/miss_stability_state.json` | Last MissBank snapshot used for next-epoch comparison |
| `best_val_metrics.json` | Best-checkpoint epoch and validation metrics |
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
- Train metrics are reduced across ranks before logging.
