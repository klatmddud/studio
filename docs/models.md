# Models

## Architecture Wrappers (`models/detection/wrapper/`)

All wrappers inherit from `_base.py` which builds the backbone + FPN via TorchVision.

### FCOS (`fcos.py` → `MDMBFCOS`)

- Single-stage anchor-free detector
- Integrates MDMB, RECALL, SCA modules (if enabled in `modules/cfg/`)
- `after_optimizer_step()` hook: passes images+targets to MDMB for missed detection tracking
- Forward in train mode: returns `loss_dict`; in eval mode: returns predictions

### Faster R-CNN (`fasterrcnn.py`)

- Two-stage region proposal detector
- Integrates MDMB, CFP modules
- Compatible backbones: ResNet50/101 with FPN, MobileNetV2/V3

### DINO (`dino.py` → `DINOWrapper`)

- Transformer-based detector (DEtection with INtroduced queries + Optimized)
- No custom modules by default
- Requires ResNet50 backbone

## Model Config Fields

| Field | Description |
|---|---|
| `backbone.name` | `resnet18/34/50/101/152`, `mobilenet_v2`, `mobilenet_v3_large` |
| `backbone.pretrained` | `DEFAULT` or `null` |
| `backbone.trainable_layers` | Number of trainable backbone layers (0–5) |
| `neck.out_channels` | FPN output channels (default: 256) |
| `head.score_thresh` | Score threshold for predictions |
| `head.nms_thresh` | NMS IoU threshold |
| `head.detections_per_img` | Max detections per image |
| `num_classes` | Auto-inferred from COCO JSON if not set |
| `transform.min_size` | Shorter edge resize target |
| `transform.max_size` | Longer edge cap |

## Model Building (`scripts/runtime/registry.py`)

`build_model_from_path(model_yaml, runtime_config)`:
1. Loads model YAML
2. Infers `arch` from filename stem
3. Dispatches to the appropriate wrapper class
4. Loads module configs from `modules/cfg/` and attaches enabled modules
5. Returns `(model, model_config, arch, model_config_path)`

`num_classes` is resolved at build time: model YAML value takes precedence; otherwise inferred from `train_annotations` via `dataset_meta.py`.
