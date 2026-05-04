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
  --lmb-config modules/cfg/lmb.yaml \
  --qg-afp-config modules/cfg/qg_afp.yaml \
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
5. For each epoch, runs `train_one_epoch()`, scheduled validation, `history.json` updates, and checkpoint writes.

## Module Configs

When ReMiss is enabled, `scripts/runtime/registry.py` attaches MissBank to the model. `matching.score_threshold: auto` and `matching.iou_threshold: auto` resolve MissBank matching from the detector's final post-processing config, so FCOS uses `head.score_thresh` and `head.nms_thresh`. `modules/cfg/remiss.yaml` controls MissBank mining with `mining.type: online` or `mining.type: offline`. Online mining runs an eval-style no-grad detection pass after each optimization step. Offline mining skips per-step updates and runs one additional no-grad pass over the training loader after each epoch, before ReMiss stability metrics are written. MissBank does not add detector losses or feature injection.

When LMB is enabled, `scripts/runtime/registry.py` attaches `LocalizationMemoryBank` to the model. `matching.score_threshold: auto` resolves from the detector's final score threshold. Starting at `start_epoch`, LMB runs one no-grad training-set mining pass after each epoch and writes localization-quality stability metrics under `output_dir/lmb/`. LMB does not add detector losses or change inference.

When QG-AFP v0 is enabled, `scripts/runtime/registry.py` builds it before the FCOS wrapper and inserts it as a `post_neck` module. It mines top-k proxy-objectness feature locations from FPN outputs, predicts query-conditioned level gates, and applies an identity-biased residual scale to the pyramid features. QG-AFP v0 changes FCOS forward features but does not add auxiliary losses.

QG-AFP v0 metrics are aggregated through the standard train metric path, so `history.json` and `results.csv` include fields such as `train_qg_afp_gate_entropy`, `train_qg_afp_gate_max_mean`, `train_qg_afp_level_usage_entropy`, `train_qg_afp_level_top1_share`, and `train_qg_afp_alpha_l0`.

When Hard Replay is enabled, `scripts/runtime/data.py` replaces the normal train sampler with a mixed replay batch sampler. The replay index is refreshed at epoch start from ReMiss MissBank records. A GT is a replay target when MissBank says it is currently missed under the detector's final class/score/IoU matching thresholds, with no FN subtype split. The initial path is image-level replay: images containing eligible missed GTs are sampled into replay slots with priority based on missed-GT count and streak diagnostics. `crop_replay` is disabled by default but can be enabled to emit GT-centered crop replay samples from the same missed-GT records.

MissBank and LMB offline mining temporarily switch the train loader to base-only iteration, so mining still sees the base training set once rather than replay-augmented batches.

| CLI flag | Default path |
|---|---|
| `--remiss-config` | `modules/cfg/remiss.yaml` |
| `--lmb-config` | `modules/cfg/lmb.yaml` |
| `--qg-afp-config` | `modules/cfg/qg_afp.yaml` |
| `--hard-replay-config` | `modules/cfg/hard_replay.yaml` |

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

When `resume_scheduler: false`, the fresh scheduler is aligned to the resumed global epoch, so a `multistep` milestone `143` still takes effect after epoch 143 and affects epoch 144 onward.

When resuming a baseline checkpoint with ReMiss MissBank or LMB newly enabled, `missbank._extra_state` and `lmb._extra_state` are allowed to be missing and the corresponding memory bank starts from an empty state. Detector weights, epoch, and `best_metric` are restored from the checkpoint; optimizer and scheduler state follow the resume controls above.

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
| `lmb/lmb_stability_epoch.json` | Epoch-level LMB low-IoU stability metrics accumulated as a JSON list |
| `lmb/lmb_stability_epoch.csv` | Flattened CSV view of `lmb_stability_epoch.json` |
| `lmb/lmb_stability_state.json` | Last LMB snapshot used for next-epoch comparison |
| `history.json` `hard_replay` key | Epoch-level replay candidate and exposure summary when Hard Replay is enabled |
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
