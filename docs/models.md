# Models

## Architecture Wrappers (`models/detection/wrapper/`)

All wrappers inherit from `_base.py`, which builds the backbone + FPN via TorchVision.

### FCOS (`fcos.py` -> `FCOSWrapper`)

- Single-stage anchor-free detector.
- Uses TorchVision `FCOS` directly.
- Forward in train mode returns a `loss_dict`; forward in eval mode returns predictions.
- Optional MissBank, Hard Replay, and LMB runtime modules sit outside the detector forward path and do not change FCOS losses or inference.
- Optional QG-AFP v0 is inserted as a `post_neck` module between TorchVision FPN outputs and the FCOS head. It preserves feature keys and shapes but changes forward features when enabled.

### Faster R-CNN (`fasterrcnn.py`)

- Two-stage region proposal detector.
- Plain TorchVision FasterRCNN.
- Compatible backbones: ResNet50/101 with FPN, MobileNetV2/V3.

### DINO (`dino.py` -> `DINOWrapper`)

- Transformer-based detector adapter.
- Uses an external backend builder via the `backend.builder` YAML field.

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

## Post-Neck Modules

`models/detection/wrapper/_base.py` exposes `pre_neck` and `post_neck` extension points around TorchVision FPN. QG-AFP v0 uses the `post_neck` slot for FCOS, so it receives the FPN feature dict and returns an updated feature dict before the detection head runs.

## Model Building (`scripts/runtime/registry.py`)

`build_model_from_path(model_yaml, runtime_config)`:

1. Loads model YAML.
2. Infers `arch` from filename stem.
3. Dispatches to the appropriate wrapper class.
4. Returns `(model, model_config, arch, model_config_path)`.

`num_classes` is resolved at build time: model YAML value takes precedence; otherwise it is inferred from `train_annotations` via `dataset_meta.py`.
