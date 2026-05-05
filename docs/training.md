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
  --ftmb-config modules/cfg/ftmb.yaml \
  --lmb-config modules/cfg/lmb.yaml \
  --qg-afp-config modules/cfg/qg_afp.yaml \
  --hard-replay-config modules/cfg/hard_replay.yaml \
  --tar-config modules/cfg/tar.yaml
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

When ReMiss is enabled, `scripts/runtime/registry.py` attaches MissBank to the model. `matching.score_threshold: auto` and `matching.iou_threshold: auto` resolve matching from the detector's final post-processing config, so FCOS uses `head.score_thresh` and `head.nms_thresh`. `modules/cfg/remiss.yaml` controls mining with `mining.type: online` or `mining.type: offline`. Online mining runs an eval-style no-grad detection pass after each optimization step. Offline mining skips per-step updates and runs one additional no-grad pass over the training loader after each epoch. MissBank does not add detector losses or feature injection.

When FTMB is enabled, `scripts/runtime/registry.py` attaches `FailureTypeMemoryBank` from `modules/cfg/ftmb.yaml` independently from ReMiss MissBank. `matching.score_threshold: auto` and `matching.iou_threshold: auto` resolve from the detector's final post-processing config. FTMB uses `mining.type: online` or `offline` to record epoch-level `localization`, `classification`, `both`, `missed`, `duplicate`, and `background` counts. Runtime outputs under `output_dir/ftmb/` are count-only summaries and do not persist detailed GT records or prediction events.

When LMB is enabled, `scripts/runtime/registry.py` attaches `LocalizationMemoryBank` to the model. `matching.score_threshold: auto` resolves from the detector's final score threshold. Starting at `start_epoch`, LMB runs one no-grad training-set mining pass after each epoch and writes localization-quality stability metrics under `output_dir/lmb/`. LMB does not add detector losses or change inference.

When QG-AFP v0 is enabled, `scripts/runtime/registry.py` builds it before the FCOS wrapper and inserts it as a `post_neck` module. It mines top-k proxy-objectness feature locations from FPN outputs, predicts query-conditioned level gates, and applies an identity-biased residual scale to the pyramid features. QG-AFP v0 changes FCOS forward features but does not add auxiliary losses.

QG-AFP v0 metrics are aggregated through the standard train metric path, so `history.json` and `results.csv` include fields such as `train_qg_afp_gate_entropy`, `train_qg_afp_gate_max_mean`, `train_qg_afp_level_usage_entropy`, `train_qg_afp_level_top1_share`, and `train_qg_afp_alpha_l0`.

When Hard Replay is enabled, `scripts/runtime/data.py` replaces the normal train sampler with a mixed replay batch sampler. The replay index is refreshed at epoch start from ReMiss MissBank records. A GT is a replay target when MissBank says it is currently missed under the detector's final class/score/IoU matching thresholds, with no FN subtype split. The initial path is image-level replay: images containing eligible missed GTs are sampled into replay slots with priority based on missed-GT count and streak diagnostics. `crop_replay` is disabled by default but can be enabled to emit GT-centered crop replay samples from the same missed-GT records.

When TAR is enabled, `scripts/runtime/data.py` uses the TAR batch sampler instead of Hard Replay. The TAR index is refreshed at epoch start from FTMB records and prediction events. `modules/cfg/tar.yaml` controls the total `replay_ratio` and the split of replay slots across `loc`, `cls`, `both`, `missed`, `duplicate`, and `background` failure types through `type_ratios`. TAR currently replays full images by failure type and keeps `gt_id`, class, bbox, and failure type on the replay sample reference for later crop/negative replay policies.

MissBank, FTMB, TAR, Hard Replay, and LMB offline mining temporarily switch the train loader to base-only iteration, so mining still sees the base training set once rather than replay-augmented batches.

| CLI flag | Default path |
|---|---|
| `--remiss-config` | `modules/cfg/remiss.yaml` |
| `--ftmb-config` | `modules/cfg/ftmb.yaml` |
| `--lmb-config` | `modules/cfg/lmb.yaml` |
| `--qg-afp-config` | `modules/cfg/qg_afp.yaml` |
| `--hard-replay-config` | `modules/cfg/hard_replay.yaml` |
| `--tar-config` | `modules/cfg/tar.yaml` |

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

When resuming a baseline checkpoint with ReMiss MissBank, FTMB, or LMB newly enabled, `missbank._extra_state`, `ftmb._extra_state`, and `lmb._extra_state` are allowed to be missing and the corresponding memory bank starts from an empty state. Detector weights, epoch, and `best_metric` are restored from the checkpoint; optimizer and scheduler state follow the resume controls above.

## Metrics And Outputs

Common outputs under `output_dir`:

| Path | Description |
|---|---|
| `history.json` | Epoch-level train and validation metrics |
| `results.csv` | Flattened CSV view of `history.json` for spreadsheet-style analysis |
| `checkpoints/last.pt` | Last checkpoint when `checkpoint.save_last` is enabled |
| `checkpoints/best.pt` | Best monitored checkpoint when `checkpoint.save_best` is enabled |
| `checkpoints/epoch_0020.pt` | Periodic checkpoint example written by `checkpoint.save_every_epochs: 20` |
| `ftmb/failure_type_epoch.json` | Epoch-level FTMB failure-type counts accumulated as a JSON list |
| `ftmb/failure_type_epoch.csv` | Flattened CSV view of `failure_type_epoch.json` |
| `ftmb/failure_type_state.json` | Last count-only FTMB failure-type snapshot |
| `lmb/lmb_stability_epoch.json` | Epoch-level LMB low-IoU stability metrics accumulated as a JSON list |
| `lmb/lmb_stability_epoch.csv` | Flattened CSV view of `lmb_stability_epoch.json` |
| `lmb/lmb_stability_state.json` | Last LMB snapshot used for next-epoch comparison |
| `history.json` `hard_replay` key | Epoch-level replay candidate and exposure summary when Hard Replay is enabled |
| `history.json` `tar` key | Epoch-level type-aware replay candidate, slot, and exposure summary when TAR is enabled |
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
