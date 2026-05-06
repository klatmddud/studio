# Data

## Dataset Format

The project uses COCO-format detection datasets.

Required runtime config fields:

| Field | Description |
|---|---|
| `data.train_images` | Directory containing train images |
| `data.train_annotations` | COCO JSON for train annotations |
| `data.val_images` | Optional validation image directory |
| `data.val_annotations` | Optional validation COCO JSON |
| `data.images` | Evaluation image directory |
| `data.annotations` | Evaluation COCO JSON |

`scripts/runtime/config.py` resolves `${ENV_VAR}` placeholders from `.env` and can select dataset-specific environment variables through `--data`.

## DataLoader Path

`scripts/runtime/data.py` provides:

- `CocoDetectionDataset`: loads images and COCO annotations.
- `build_train_dataloaders()`: returns train and optional validation loaders.
- `build_eval_dataloader()`: returns the evaluation loader.
- `collate_fn()`: returns `list[Tensor]` images and `list[dict]` targets for TorchVision detection models.

## Hard Replay Loader Path

When `modules/cfg/hard_replay.yaml` is enabled, `build_train_dataloaders()` attaches a `HardReplayController` and uses `MixedReplayBatchSampler` instead of the normal sampler. Each epoch still walks the base training dataset once, then fills configured replay slots with images selected from ReMiss MissBank records.

Replay eligibility is GT-level: a GT must be currently missed by MissBank under the model's final matching thresholds, satisfy `min_miss_count` and `min_observations`, and pass `replay_recency_window`. Set `current_epoch_only: true` to require the MissBank record's `last_epoch` to equal the replay refresh epoch. The sampler weights each image by the summed priority of its eligible missed GTs, clips the image weight, and mixes replay samples according to `replay_ratio`.

## Target Fields

Each target dict contains:

| Key | Type | Description |
|---|---|---|
| `boxes` | `FloatTensor[N, 4]` | Absolute `xyxy` boxes clipped to image bounds |
| `labels` | `LongTensor[N]` | COCO category IDs |
| `image_id` | `LongTensor[1]` | COCO image ID |
| `area` | `FloatTensor[N]` | COCO area |
| `iscrowd` | `LongTensor[N]` | COCO crowd flag |
| `annotation_ids` | `LongTensor[N]` | COCO annotation IDs, or `-1` when absent |
| `gt_ids` | `LongTensor[N]` | Alias for per-GT modules such as ReMiss MissBank |

## Dataset Selector

`--data kitti` selects dataset-specific environment variables such as `KITTI_TRAIN_IMAGES` and `KITTI_TRAIN_ANNOTATIONS`. Omit `--data` when runtime YAML contains direct paths.
