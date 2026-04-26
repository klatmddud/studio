# Training System

## Config Schema

Two YAML files are always required.

### Runtime config (`scripts/cfg/train.yaml`)

```yaml
seed: 42
device: auto          # auto | cpu | cuda | cuda:0 | mps
amp: false
output_dir: runs/train

data:
  train_images: ${KITTI_TRAIN_IMAGES}
  train_annotations: ${KITTI_TRAIN_ANNOTATIONS}
  val_images: ${KITTI_VAL_IMAGES}
  val_annotations: ${KITTI_VAL_ANNOTATIONS}

loader:
  batch_size: 16
  num_workers: 8
  pin_memory: true
  shuffle: true

optimizer:
  name: sgd           # sgd | adamw
  lr: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  nesterov: false

scheduler:
  name: multistep     # multistep | step | cosine | none
  milestones: [8, 11]
  gamma: 0.1

train:
  epochs: 50
  grad_clip_norm: 1.0
  log_interval: 20
  eval_every_epochs: 1

checkpoint:
  dir: checkpoints
  resume: null
  save_last: true
  save_best: true
  monitor: bbox_mAP_50_95
  mode: max

metrics:
  type: coco_detection
  iou_types: [bbox]
  primary: bbox_mAP_50_95
```

### Model config (`models/detection/cfg/*.yaml`)

```yaml
backbone:
  name: resnet18
  pretrained: DEFAULT
  trainable_layers: 3
neck:
  out_channels: 256
head:
  score_thresh: 0.2
  nms_thresh: 0.6
  detections_per_img: 100
  topk_candidates: 1000
num_classes: 9
transform:
  min_size: 800
  max_size: 1333
```

## Engine (`scripts/runtime/engine.py`)

### `fit()`

Main training loop:

1. Build optimizer, scheduler, and grad scaler.
2. Optionally resume from `checkpoint.resume`.
3. For each epoch:
   - Call `start_epoch()` on enabled `mdmb`, `mdmbpp`, `rasd`, `tfm`, `fntdm`, and `dhm` modules.
   - Refresh the Hard Replay `ReplayIndex` from `model.mdmbpp` if the train loader has a replay controller.
   - Run `train_one_epoch()`.
   - Run FN-TDM and DHM epoch-end mining when their configs request full train-set mining.
   - Call `end_epoch()` on enabled `mdmb`, `mdmbpp`, `rasd`, `tfm`, `fntdm`, and `dhm`.
   - Optionally evaluate on the validation loader.
   - Save `best.pt` and `last.pt` according to checkpoint settings.
   - Append the epoch record to `history.json`.

Inside `train_one_epoch()`, FCOS may run `after_optimizer_step()` after every optimizer step.
That post-step hook performs one no-grad inference pass and refreshes `mdmb` and `mdmbpp`
state from the updated model.

When TFM is enabled, FCOS refreshes `tfm` inside the normal training forward from assignment,
classification, localization, and centerness signals. TFM does not require the post-step inference
pass and does not change inference behavior.

When `tfm.assignment_bias.enabled` is true, FCOS uses the previous TFM record for each matched GT to
increase positive-point loss weights for high-risk GTs. The current implementation only reweights
classification, box regression, and centerness losses; it does not change center sampling or add
backup positives.

When RASD is enabled, FCOS reads relapse entries from `mdmbpp`, pools current GT features from the
FPN, and appends a `rasd` auxiliary loss when matching support features exist. RASD is training-only
and does not change inference, NMS, or score thresholds.

When DHM is enabled, `engine.fit()` can run an epoch-end full train-set inference pass to update
per-GT detection-state hysteresis records. When `dhm.loss_weighting.enabled` is true, FCOS uses the
previous DHM instability score and dominant failure type to scale the existing classification, box,
and centerness losses for matched positive points. DHM does not add an auxiliary loss.
When `dhm.assignment_expansion.enabled` is true, FCOS applies HLAE before loss computation:
`FN_BG` GTs with sufficient history can receive a small capped set of additional center-nearest
positive points from positions that were negative under the standard FCOS assignment.

### Hard Replay Interaction

Hard Replay is configured through `modules/cfg/hard_replay.yaml`, but it runs in the runtime/data
layer rather than as an `nn.Module`.

- Epoch start: `engine.fit()` asks the train loader's replay controller to rebuild its
  `ReplayIndex` from the current `model.mdmbpp`.
- Epoch iteration: `MixedReplayBatchSampler` yields mixed batches with
  `batch_size - floor(batch_size * replay_ratio)` base samples and
  `floor(batch_size * replay_ratio)` replay samples.
- Epoch logging: replay statistics are stored under the `hard_replay` key in `history.json`.

### Per-Run Research Module Configs

By default, training reads research module configs from `modules/cfg/*.yaml`. For concurrent
experiments, pass per-run config files on the CLI so one run does not mutate or depend on global
module YAML:

```bash
uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --device cuda:0 \
  --mdmb-config scripts/bash/mdmbpp_only/cfg/mdmb_disabled.yaml \
  --mdmbpp-config scripts/bash/mdmbpp_only/cfg/mdmbpp_only.yaml \
  --rasd-config scripts/bash/mdmbpp_only/cfg/rasd_disabled.yaml \
  --hard-replay-config scripts/bash/mdmbpp_only/cfg/hard_replay_disabled.yaml \
  --tfm-config modules/cfg/tfm.yaml \
  --fntdm-config modules/cfg/fntdm.yaml \
  --dhm-config modules/cfg/dhm.yaml
```

The same resolved paths are used for model construction, Hard Replay DataLoader setup, DDP worker
spawn, and `metadata/modules.yaml` snapshots.

### FN-TDM Interaction

FN-TDM is configured through `modules/cfg/fntdm.yaml` and is disabled by default.

When enabled for FCOS:

- FCOS training forward can add an `fntdm` auxiliary loss after TCS selects hard current GTs.
- After each training epoch satisfying `fntdm.htm.mine_interval` and `fntdm.htm.warmup_epochs`,
  `engine.fit()` builds a deterministic train mining loader and runs one full train-set inference
  pass to mine `FN -> TP` transitions.
- TDB state is stored inside the model `state_dict` through `fntdm._extra_state`.
- Under DDP, epoch-end HTM mining runs on rank 0 and FN-TDM state is synchronized before the next
  epoch.

This baseline is intentionally expensive and is meant as the clean reference implementation before
adding online or subset mining variants.

### DHM Interaction

DHM is configured through `modules/cfg/dhm.yaml` and is disabled by default.

When enabled for FCOS:

- After each training epoch satisfying `dhm.mining.mine_interval` and
  `dhm.mining.warmup_epochs`, `engine.fit()` builds a deterministic train mining loader and runs
  one full train-set inference pass to update per-GT `TP`/`FN_*` hysteresis records.
- `dhm._extra_state` stores the memory records in the model `state_dict`.
- If `dhm.loss_weighting.enabled` is true, FCOS reweights existing positive-point detection losses
  from the previous memory state; no new loss key is added.
- If `dhm.assignment_expansion.enabled` is true, HLAE preserves standard FCOS positives and adds up
  to `dhm.assignment_expansion.backup_topk` extra positives per eligible `FN_BG` GT, capped by
  `dhm.assignment_expansion.max_extra_positive_ratio` per image.
- Under DDP, epoch-end mining runs on rank 0 and DHM state is synchronized before the next epoch.

This is the clean full-mining baseline. Subset mining or online mining should be added as separate
variants after the full baseline is measured.

### Multi-GPU DDP

Training supports single-process single-device execution and internal DDP spawn for multiple CUDA
devices:

```bash
uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --device cuda:0 cuda:1
```

When more than one `--device` value is provided, `train.py` starts one worker process per CUDA
device and wraps the model with `DistributedDataParallel`. Checkpoints, metadata, history, figures,
and progress logs are written only by rank 0. Checkpoints are saved from the unwrapped model, so the
`model_state_dict` keeps the same keys as single-device training.

DDP currently uses the NCCL backend and requires explicit CUDA ids such as `cuda:0 cuda:1`. CPU/MPS
multi-device training and multi-device evaluation are not supported. Validation during training runs
on rank 0 only while the other workers wait at a barrier.

Research module state is synchronized once per epoch. MDMB/MDMB++/TFM local memory states are
gathered after the epoch, merged on rank 0, and broadcast back before the next epoch. RASD summary
counters are reduced for logging, while RASD itself remains training-only and stateless apart from
epoch bookkeeping.

Current replay statistics include:

- `replay_num_images`
- `replay_num_crops`
- `replay_num_object_samples`
- `replay_num_active_gt`
- `replay_mean_image_weight`
- `replay_mean_gt_severity`
- `replay_ratio_requested`
- `replay_ratio_effective`
- `replay_exposure_per_gt`
- `replay_loss_weight_mean`

### `evaluate()`

- Called both during training and from `eval.py`
- Loads `checkpoint.path` only for standalone evaluation
- Runs `evaluate_coco_detection()` and returns COCO metrics

### Checkpoint format

```python
{
  "epoch": int,
  "best_metric": float,
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "scheduler_state_dict": ...,
}
```

Files are written to `{checkpoint.dir}/best.pt` and `{checkpoint.dir}/last.pt` using
`tempfile + os.replace` for atomic writes.

## Available Monitor Metrics

COCO bbox metrics available for `checkpoint.monitor` and `metrics.primary`:

- `bbox_mAP_50_95`
- `bbox_mAP_50`
- `bbox_mAP_75`
- `bbox_mAP_small`
- `bbox_mAP_medium`
- `bbox_mAP_large`
- `bbox_mAR_1`
- `bbox_mAR_10`
- `bbox_mAR_100`

## Output Artifacts

`{output_dir}/history.json` stores per-epoch training and validation records. When enabled, the
record may include `mdmb`, `mdmbpp`, `rasd`, `tfm`, `fntdm`, `dhm`, and `hard_replay` summaries
alongside `train` and `val`.

Additional outputs:

- `{output_dir}/best_val_metrics.json`
- `{output_dir}/checkpoints/best.pt`
- `{output_dir}/checkpoints/last.pt`
- `{output_dir}/figures/`
- `{output_dir}/metadata/model.yaml`
- `{output_dir}/metadata/train.yaml`
- `{output_dir}/metadata/modules.yaml` for train runs, containing enabled research module YAML snapshots
- `{output_dir}/metadata/run.json`

`metadata/modules.yaml` is written during training only. It stores the parsed YAML mapping for each
research module that is effectively enabled for the selected architecture, using keys such as
`mdmbpp`, `rasd`, `hard_replay`, `tfm`, `fntdm`, and `dhm`. If no research module is enabled, the
file contains an empty mapping. `metadata/run.json` also records the resolved module config paths
used by that run.

Confusion matrix figures use Ultralytics YOLO-compatible detection matching: predictions are
filtered with `score > 0.25`, GT/prediction pairs use global one-to-one matching at `IoU > 0.45`,
and `confusion_matrix_normalized.png` is normalized by true-class column.

## CLI Flags

```bash
uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --output-dir runs/exp1 \
  --seed 42 \
  --device cuda

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --output-dir runs/exp1_ddp \
  --seed 42 \
  --device cuda:0 cuda:1

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --output-dir runs/mdmbpp_only \
  --seed 42 \
  --device cuda:0 \
  --mdmb-config scripts/bash/mdmbpp_only/cfg/mdmb_disabled.yaml \
  --mdmbpp-config scripts/bash/mdmbpp_only/cfg/mdmbpp_only.yaml \
  --rasd-config scripts/bash/mdmbpp_only/cfg/rasd_disabled.yaml \
  --hard-replay-config scripts/bash/mdmbpp_only/cfg/hard_replay_disabled.yaml \
  --tfm-config modules/cfg/tfm.yaml \
  --fntdm-config modules/cfg/fntdm.yaml \
  --dhm-config modules/cfg/dhm.yaml

uv run scripts/eval.py \
  --config scripts/cfg/eval.yaml \
  --model models/detection/cfg/fcos.yaml \
  --checkpoint runs/train/checkpoints/best.pt \
  --data kitti \
  --output-dir runs/eval_exp1
```

`--seed` is a train-only override for `runtime.seed`. It is applied before model/data construction,
DDP worker spawn, and metadata persistence, so `metadata/train.yaml` records the effective seed.
