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
- `build_train_mining_dataloader()`: deterministic train-set loader used by DHM epoch-end mining.
- `build_eval_dataloader()`: evaluation loader.
- `collate_fn()`: returns `list[Tensor]` images and `list[dict]` targets for TorchVision detection models.

When `train.hard_replay.enabled` is true, `build_train_dataloaders()` attaches a
DHM-driven mixed replay batch sampler. The sampler keeps the configured batch size fixed by
reserving replay slots according to `train.hard_replay.max_ratio`. Replay samples can be normal
full-image replays or, when `loc_repair.enabled` is true, `FN_LOC` localization-repair crops.

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
| `gt_ids` | `LongTensor[N]` | Alias used by DHM/DHM-R for per-GT lookup |

Localization-repair crop targets additionally contain:

| Key | Type | Description |
|---|---|---|
| `is_replay_crop` | `LongTensor[]` | `1` for DHM-guided crop replay targets |
| `replay_policy_id` | `LongTensor[]` | Numeric replay policy ID; `1` means localization crop |
| `focus_gt_id` | `LongTensor[]` | Annotation ID of the focused FN_LOC GT, or `-1` |

## Dataset Selector

`--data kitti` selects dataset-specific environment variables such as `KITTI_TRAIN_IMAGES` and `KITTI_TRAIN_ANNOTATIONS`. Omit `--data` when runtime YAML contains direct paths.
