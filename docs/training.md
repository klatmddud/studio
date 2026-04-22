# Training System

## Config Schema

Two YAML files are always required.

### Runtime config (`scripts/cfg/train.yaml`)

```yaml
seed: 42
device: auto          # auto | cpu | cuda | mps
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
   - Call `start_epoch()` on enabled `mdmb`, `mdmbpp`, and `rasd` modules.
   - Refresh the Hard Replay `ReplayIndex` from `model.mdmbpp` if the train loader has a replay controller.
   - Run `train_one_epoch()`.
   - Call `end_epoch()` on enabled `mdmb`, `mdmbpp`, and `rasd`.
   - Optionally evaluate on the validation loader.
   - Save `best.pt` and `last.pt` according to checkpoint settings.
   - Append the epoch record to `history.json`.

Inside `train_one_epoch()`, FCOS may run `after_optimizer_step()` after every optimizer step.
That post-step hook performs one no-grad inference pass and refreshes `mdmb` and `mdmbpp`
state from the updated model.

When RASD is enabled, FCOS reads relapse entries from `mdmbpp`, pools current GT features from the
FPN, and appends a `rasd` auxiliary loss when matching support features exist. RASD is training-only
and does not change inference, NMS, or score thresholds.

### Hard Replay Interaction

Hard Replay is configured through `modules/cfg/hard_replay.yaml`, but it runs in the runtime/data
layer rather than as an `nn.Module`.

- Epoch start: `engine.fit()` asks the train loader's replay controller to rebuild its
  `ReplayIndex` from the current `model.mdmbpp`.
- Epoch iteration: `MixedReplayBatchSampler` yields mixed batches with
  `batch_size - floor(batch_size * replay_ratio)` base samples and
  `floor(batch_size * replay_ratio)` replay samples.
- Epoch logging: replay statistics are stored under the `hard_replay` key in `history.json`.

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
record may include `mdmb`, `mdmbpp`, `rasd`, and `hard_replay` summaries alongside `train` and
`val`.

Additional outputs:

- `{output_dir}/best_val_metrics.json`
- `{output_dir}/checkpoints/best.pt`
- `{output_dir}/checkpoints/last.pt`
- `{output_dir}/figures/`
- `{output_dir}/metadata/model.yaml`
- `{output_dir}/metadata/train.yaml`
- `{output_dir}/metadata/run.json`

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
  --device cuda

uv run scripts/eval.py \
  --config scripts/cfg/eval.yaml \
  --model models/detection/cfg/fcos.yaml \
  --checkpoint runs/train/checkpoints/best.pt \
  --data kitti \
  --output-dir runs/eval_exp1
```
