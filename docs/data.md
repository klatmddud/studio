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
- `build_train_mining_dataloader(runtime_config)` -> deterministic train loader for epoch-end mining
- `build_eval_dataloader(runtime_config)` -> `eval_loader`

`CocoDetectionDataset` per-item output:

- Image: `Tensor[C, H, W]`, `float32`, RGB
- Target: `{boxes, labels, image_id, area, iscrowd, annotation_ids, gt_ids}`

`boxes` use absolute `xyxy` coordinates. Degenerate boxes are filtered. Missing images raise
errors.
`annotation_ids` and `gt_ids` mirror the COCO annotation `id` field for each kept box. Temporal
modules use them for stable GT identity when available. Missing annotation IDs are stored as `-1`,
which lets modules fall back to image/class/box identity.

## Hard Replay Loader Path

When `modules/cfg/hard_replay.yaml` is enabled for the active architecture:

- `build_train_dataloaders()` loads a `HardReplayController`
- The training `DataLoader` switches from `batch_size + shuffle` to a custom
  `MixedReplayBatchSampler`
- Each epoch still walks the full base dataset once, then injects replay samples into each batch
  according to `replay_ratio`

When top-level Hard Replay `enabled: true` and `object_replay.enabled: true`, the train dataset is
wrapped by `HardReplayDatasetWrapper`.
The wrapper keeps original dataset indices unchanged and maps virtual indices after the base
dataset range to replay samples.

Supported replay sample kinds:

- `crop`: crop a hard GT with context and remap boxes to crop coordinates
- `copy_paste`: paste a rectangular hard-object crop into a sampled target image
- `pair_miss` / `pair_support`: expose current miss and support crops for the same `gt_uid`

Replay targets include `is_replay`, `replay_kind`, `source_image_id`, `replay_gt_uid`,
`replay_pair_id`, `replay_role`, `replay_loss_weight`, and `replay_box_weights` metadata.

Replay indices are regenerated every epoch from MDMB++. Because worker dataset copies would
otherwise keep stale replay indices, the loader disables persistent workers for this wrapper.

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

## Train Mining Loader

`build_train_mining_dataloader(runtime_config)` builds a deterministic train-set loader with
`shuffle: false`. FN-TDM uses this loader for epoch-end HTM mining so transition detection runs on
the original COCO train set rather than replay samples or shuffled batch order.

## Format Conversion

YOLO-to-COCO converter: `scripts/utils/yolo2coco.py`
