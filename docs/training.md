# Training System

## Config Schema

Two YAML files are always required:

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
  grad_clip_norm: 1.0   # null to disable
  log_interval: 20
  eval_every_epochs: 1

checkpoint:
  dir: checkpoints       # saves best.pt and last.pt here
  resume: null           # path to resume from
  save_last: true
  save_best: true
  monitor: bbox_mAP_50_95
  mode: max              # max | min

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

### `fit()` — main training loop

1. Builds optimizer + scheduler + grad scaler (AMP)
2. Optionally resumes from `checkpoint.resume` (restores epoch, best_metric, optimizer/scheduler state)
3. Per epoch:
   - Calls `train_one_epoch()` → logs loss components + ETA
   - If `(epoch+1) % eval_every_epochs == 0`: calls `evaluate()` on val set
   - Saves `best.pt` when monitor metric improves
   - Always saves `last.pt` (if `save_last: true`)
   - Appends record to `history.json`

### `evaluate()` — standalone evaluation

- Called both during training (val) and from `eval.py`
- When called from `eval.py`: loads checkpoint from `checkpoint.path` before inference
- When called during training (in-loop val): uses current model weights (no checkpoint load)
- Runs `evaluate_coco_detection()` → returns COCO metrics dict

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

Files: `{checkpoint.dir}/best.pt`, `{checkpoint.dir}/last.pt`
Written atomically via `tempfile + os.replace`.

## Available Monitor Metrics

COCO bbox metrics available for `checkpoint.monitor` / `metrics.primary`:

| Key | Description |
|---|---|
| `bbox_mAP_50_95` | mAP @ IoU 0.50:0.95 (primary COCO metric) |
| `bbox_mAP_50` | mAP @ IoU 0.50 |
| `bbox_mAP_75` | mAP @ IoU 0.75 |
| `bbox_mAP_small` | mAP for small objects |
| `bbox_mAP_medium` | mAP for medium objects |
| `bbox_mAP_large` | mAP for large objects |
| `bbox_mAR_1` | mAR with max 1 detection |
| `bbox_mAR_10` | mAR with max 10 detections |
| `bbox_mAR_100` | mAR with max 100 detections |

## Output Artifacts

```
{output_dir}/
├── history.json               # per-epoch train/val metrics
├── best_val_metrics.json      # final COCO metrics from best.pt validation
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── figures/                   # 학습 종료 후 자동 생성 (best.pt 기준)
│   ├── loss.png               # epoch별 train loss (total + 각 컴포넌트 자동 감지)
│   ├── map.png                # epoch별 val mAP (mAP50:95 / mAP50 / mAP75)
│   ├── mdmb.png               # MDMB bank 통계 (entries / images) — MDMB 활성 시만 생성
│   ├── confusion_matrix.png              # 클래스별 raw count confusion matrix
│   ├── confusion_matrix_normalized.png   # 정규화 confusion matrix (열 합 기준)
│   └── mdmb_per_class.png     # 클래스별 miss detection 개수 — MDMB 활성 시만 생성
└── metadata/
    ├── model.yaml             # model config snapshot
    ├── train.yaml             # runtime config snapshot
    └── run.json               # arch + config paths
```

- `best_val_metrics.json`과 `figures/`는 `val_loader`가 있고 `best.pt`가 존재할 때만 생성된다 (`save_best: false`이거나 val set 미설정 시 스킵).
- Confusion matrix: IoU 0.5 기준 greedy matching, x축=True class, y축=Predicted class, 마지막 행/열=background (FN/FP).
- Loss 곡선: `history.json`의 `train` 키에서 `lr`, `epoch_time_sec`을 제외한 모든 loss 키를 자동 감지하므로 커스텀 loss 추가 시 별도 수정 불필요.
- 시각화 구현: `scripts/runtime/visualize.py`

## CLI Flags

```bash
# Training
uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \              # dataset selector (sets env var prefix)
  --output-dir runs/exp1 \   # overrides output_dir
  --device cuda               # overrides device

# Evaluation
uv run scripts/eval.py \
  --config scripts/cfg/eval.yaml \
  --model models/detection/cfg/fcos.yaml \
  --checkpoint runs/train/checkpoints/best.pt \
  --data kitti \
  --output-dir runs/eval_exp1
```
