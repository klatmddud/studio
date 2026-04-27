# Models

## Architecture Wrappers (`models/detection/wrapper/`)

All wrappers inherit from `_base.py`, which builds the backbone + FPN via TorchVision.

### FCOS (`fcos.py` -> `DHMFCOS`)

- Single-stage anchor-free detector.
- Integrates DHM and DHM-R when enabled in `modules/cfg/`.
- DHM mines per-GT detection-state hysteresis at epoch end.
- DHM can reweight existing FCOS classification, box, and centerness losses.
- DHM can apply HLAE training-only positive assignment expansion for eligible hard GTs.
- DHM-R reads previous DHM records and can add a `dhmr_edge` training-only auxiliary loss for `FN_LOC` GTs through the Temporal Edge Repair branch.
- `mine_dhm_batch()` supports epoch-end full train-set hysteresis mining when DHM is enabled.
- Forward in train mode returns a `loss_dict`; forward in eval mode returns predictions.

### Faster R-CNN (`fasterrcnn.py`)

- Two-stage region proposal detector.
- Plain TorchVision FasterRCNN; no research modules are currently wired up.
- Compatible backbones: ResNet50/101 with FPN, MobileNetV2/V3.

### DINO (`dino.py` -> `DINOWrapper`)

- Transformer-based detector adapter.
- Uses an external backend builder via the `backend.builder` YAML field.
- No research modules are currently wired up.

## Model Config Fields

| Field | Description |
|---|---|
| `backbone.name` | `resnet18/34/50/101/152`, `mobilenet_v2`, `mobilenet_v3_large` |
| `backbone.pretrained` | `DEFAULT` or `null` |
| `backbone.trainable_layers` | Number of trainable backbone layers |
| `neck.out_channels` | FPN output channels (default: 256) |
| `head.score_thresh` | Score threshold for predictions |
| `head.nms_thresh` | NMS IoU threshold |
| `head.detections_per_img` | Max detections per image |
| `num_classes` | Auto-inferred from COCO JSON if not set |
| `transform.min_size` | Shorter edge resize target |
| `transform.max_size` | Longer edge cap |

## Model Building (`scripts/runtime/registry.py`)

`build_model_from_path(model_yaml, runtime_config)`:

1. Loads model YAML.
2. Infers `arch` from filename stem.
3. Dispatches to the appropriate wrapper class.
4. Loads DHM/DHM-R configs from `modules/cfg/` and attaches enabled modules for FCOS.
5. Returns `(model, model_config, arch, model_config_path)`.

`num_classes` is resolved at build time: model YAML value takes precedence; otherwise it is inferred from `train_annotations` via `dataset_meta.py`.
