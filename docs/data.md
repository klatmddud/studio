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

## Dataset Selector

`--data kitti` selects dataset-specific environment variables such as `KITTI_TRAIN_IMAGES` and `KITTI_TRAIN_ANNOTATIONS`. Omit `--data` when runtime YAML contains direct paths.
