# Data Pipeline

## Dataset Format

COCO JSON format is required. Each split needs:

- An image directory
- A COCO-format annotation JSON containing `images`, `annotations`, and `categories`

`num_classes` is inferred as `max(category_id) + 1`, with background at index `0`.

## Environment Variables

Dataset paths are stored in a project-root `.env` file and referenced from YAML via `${VAR_NAME}`.

Example:

```text
KITTI_TRAIN_IMAGES=/data/kitti/train/images
KITTI_TRAIN_ANNOTATIONS=/data/kitti/train/annotations.json
KITTI_VAL_IMAGES=/data/kitti/val/images
KITTI_VAL_ANNOTATIONS=/data/kitti/val/annotations.json

BDD100K_TRAIN_IMAGES=/data/bdd100k/train/images
BDD100K_TRAIN_ANNOTATIONS=/data/bdd100k/train/annotations.json
```

The `--data kitti` CLI flag selects `KITTI_*` variables. `--data bdd100k` selects `BDD100K_*`.

## DataLoader

Built by `scripts/runtime/data.py`:

- `build_train_dataloaders(runtime_config, arch=...)` -> `(train_loader, val_loader)`
- `build_eval_dataloader(runtime_config)` -> `eval_loader`

`CocoDetectionDataset` per-item output:

- Image: `Tensor[C, H, W]`, `float32`, RGB
- Target: `{boxes, labels, image_id, area, iscrowd}`

`boxes` use absolute `xyxy` coordinates. Degenerate boxes are filtered. Missing images raise
errors.

## Hard Replay Loader Path

When `modules/cfg/hard_replay.yaml` is enabled for the active architecture:

- `build_train_dataloaders()` loads a `HardReplayController`
- The training `DataLoader` switches from `batch_size + shuffle` to a custom
  `MixedReplayBatchSampler`
- Each epoch still walks the full base dataset once, then injects replay samples into each batch
  according to `replay_ratio`

The current implementation is image-level only. `ReplayCrop` exists as a placeholder, but crop
replay and copy-paste replay are not enabled yet.

## DataLoader Config

```yaml
loader:
  batch_size: 16
  num_workers: 8
  pin_memory: true
  shuffle: true
```

Custom `collate_fn` preserves variable-length image lists, so no padding is required for the
TorchVision detectors in this project.

## Format Conversion

YOLO-to-COCO converter: `scripts/utils/yolo2coco.py`
