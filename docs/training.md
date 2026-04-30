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
  --remiss-conv-config modules/cfg/remiss_conv.yaml \
  --mpd-config modules/cfg/mpd.yaml
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

When ReMiss is enabled, `scripts/runtime/registry.py` attaches MissBank to the model. `matching.score_threshold: auto` and `matching.iou_threshold: auto` resolve MissBank matching from the detector's final post-processing config, so FCOS uses `head.score_thresh` and `head.nms_thresh`. `modules/cfg/remiss.yaml` controls MissBank mining with `mining.type: online` or `mining.type: offline`. Online mining runs an eval-style no-grad detection pass after each optimization step. Offline mining skips per-step updates and runs one additional no-grad pass over the training loader after each epoch, before ReMiss stability metrics are written. When `miss_head.enabled` is true, FCOS training also adds MissHead loss after `miss_head.start_epoch`: `miss_head_ce` in legacy 5-way mode, or `miss_head_has_miss_bce` plus `miss_head_region_ce` when `miss_head.has_miss_head: true`.

ReMissConv is configured separately through `modules/cfg/remiss_conv.yaml`. When enabled, runtime attaches a separate `remiss_conv_bank` and `remiss_conv` module. The bank uses the same matching and mining semantics as MissBank but writes outputs under `remiss_conv/`. The convolutional module predicts grid-level miss maps from FCOS FPN features, adds `remiss_conv_loss` after `conv.start_epoch`, and applies gated additive prototype modulation before the FCOS detector head.

MPD is configured separately through `modules/cfg/mpd.yaml`. When enabled, runtime attaches a separate `mpd_bank` and `mpd` module. The bank uses the same matching and mining semantics as MissBank but writes outputs under `mpd/`. MPD changes FCOS training assignment only after `mpd.start_epoch`: repeated missed GTs are used to densify positive matched locations, then the original FCOS head loss is computed. MPD does not change inference or add an auxiliary loss.

| CLI flag | Default path |
|---|---|
| `--remiss-config` | `modules/cfg/remiss.yaml` |
| `--remiss-conv-config` | `modules/cfg/remiss_conv.yaml` |
| `--mpd-config` | `modules/cfg/mpd.yaml` |

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

`checkpoint.save_last` writes `last.pt`. `checkpoint.save_best` writes `best.pt` when the monitored metric improves.

## Metrics And Outputs

Common outputs under `output_dir`:

| Path | Description |
|---|---|
| `history.json` | Epoch-level train and validation metrics |
| `results.csv` | Flattened CSV view of `history.json` for spreadsheet-style analysis |
| `remiss/miss_stability_epoch.json` | Epoch-level MissBank stability metrics accumulated as a JSON list |
| `remiss/miss_stability_epoch.csv` | Flattened CSV view of `miss_stability_epoch.json` |
| `remiss/miss_stability_state.json` | Last MissBank snapshot used for next-epoch comparison |
| `remiss/miss_head_epoch.json` | Epoch-level MissHead train loss and metrics accumulated separately from `history.json` |
| `remiss/miss_head_epoch.csv` | Flattened CSV view of `miss_head_epoch.json` |
| `remiss_conv/miss_stability_epoch.json` | Epoch-level stability metrics for the separate ReMissConv MissBank |
| `remiss_conv/miss_stability_epoch.csv` | Flattened CSV view of ReMissConv stability metrics |
| `remiss_conv/miss_stability_state.json` | Last ReMissConv MissBank snapshot used for next-epoch comparison |
| `remiss_conv/miss_map_epoch.json` | Epoch-level ReMissConv miss-map loss and metrics accumulated separately from `history.json` |
| `remiss_conv/miss_map_epoch.csv` | Flattened CSV view of `miss_map_epoch.json` |
| `mpd/miss_stability_epoch.json` | Epoch-level stability metrics for the separate MPD MissBank |
| `mpd/miss_stability_epoch.csv` | Flattened CSV view of MPD stability metrics |
| `mpd/miss_stability_state.json` | Last MPD MissBank snapshot used for next-epoch comparison |
| `mpd/mpd_epoch.json` | Epoch-level MPD assignment metrics accumulated separately from `history.json` |
| `mpd/mpd_epoch.csv` | Flattened CSV view of `mpd_epoch.json` |
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
- Train metrics are reduced across ranks before logging.
