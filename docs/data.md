# Data Pipeline

## Dataset Format

COCO JSON format required. Each split needs:
- An image directory
- A COCO-format annotation JSON (`images`, `annotations`, `categories`)

`num_classes` is inferred as `max(category_id) + 1` (background at index 0).

## Environment Variables

Dataset paths are stored in a `.env` file at project root and referenced in YAML via `${VAR_NAME}`.

Example `.env`:
```
KITTI_TRAIN_IMAGES=/data/kitti/train/images
KITTI_TRAIN_ANNOTATIONS=/data/kitti/train/annotations.json
KITTI_VAL_IMAGES=/data/kitti/val/images
KITTI_VAL_ANNOTATIONS=/data/kitti/val/annotations.json

BDD100K_TRAIN_IMAGES=/data/bdd100k/train/images
BDD100K_TRAIN_ANNOTATIONS=/data/bdd100k/train/annotations.json
...
```

The `--data kitti` CLI flag selects `KITTI_*` vars; `--data bdd100k` selects `BDD100K_*` vars.

## DataLoader

Built by `scripts/runtime/data.py`:
- `build_train_dataloaders(runtime_config)` → `(train_loader, val_loader)`
- `build_eval_dataloader(runtime_config)` → `eval_loader`

`CocoDetectionDataset` per-item output:
- Image: `Tensor[C, H, W]` float32, RGB
- Target: `{boxes: Tensor[N,4] xyxy, labels: Tensor[N], image_id: int, area: Tensor[N], iscrowd: Tensor[N]}`

Degenerate boxes (zero area) are filtered. Missing images raise errors.

## DataLoader Config

```yaml
loader:
  batch_size: 16
  num_workers: 8
  pin_memory: true
  shuffle: true        # false for val/eval
```

Custom `collate_fn` preserves variable-length image lists (no padding required — TorchVision models handle variable sizes internally via `GeneralizedRCNNTransform`).

## Format Conversion

YOLO → COCO converter: `scripts/utils/yolo2coco.py`
