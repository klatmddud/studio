# Architecture

## Directory Layout

```text
scripts/
  train.py                  # Entry point: training
  eval.py                   # Entry point: evaluation
  cfg/
    train.yaml              # Runtime config template (train)
    eval.yaml               # Runtime config template (eval)
  runtime/
    config.py               # Config loading, env-var substitution, validation
    data.py                 # COCO dataset + DataLoader builders, Hard Replay/crop replay wiring
    engine.py               # fit(), evaluate(), train_one_epoch(), checkpointing, DHM mining/replay
    hard_replay.py          # DHM-driven mixed replay and FN_LOC localization repair sampler
    metrics.py              # COCO evaluation via pycocotools
    registry.py             # Builds model from YAML (arch dispatch)
    dataset_meta.py         # Infers num_classes from COCO JSON
  utils/
    yolo2coco.py            # YOLO-to-COCO format converter

models/detection/
  cfg/                      # Per-architecture YAML (fcos/fasterrcnn/dino)
  wrapper/
    _base.py                # Backbone + FPN builder
    fcos.py                 # FCOS wrapper with DHM/DHM-R hooks
    fasterrcnn.py           # FasterRCNN wrapper
    dino.py                 # DINO wrapper

modules/
  cfg/                      # DHM/DHM-R module configs, disabled by default
  nn/                       # DHM/DHM-R implementations and shared helpers

ops/                        # Reserved for custom ops (currently empty)
```

## Tech Stack

| Component | Library |
|---|---|
| Deep learning | PyTorch 2.11+, TorchVision 0.26+ |
| Evaluation | pycocotools (COCOeval) |
| Config | PyYAML + python-dotenv |
| Package manager | uv (`uv sync` to install) |
| Python | 3.12+ |

## Data Flow

```text
train.py
  -> load_runtime_config()       # merge YAML + env vars
  -> build_model_from_path()     # registry.py -> wrapper
  -> build_train_dataloaders()   # data.py -> CocoDetectionDataset
  -> fit()                       # engine.py -> training loop + optional DHM mining/replay
```

## Supported Architectures

| arch key | Wrapper | Backbone options |
|---|---|---|
| `fcos` | `DHMFCOS` | ResNet18/50, MobileNetV2/V3 |
| `fasterrcnn` | `FasterRCNN` | ResNet50/101, MobileNetV2/V3 |
| `dino` | `DINOWrapper` | External backend-defined |

`arch` is inferred from the model YAML filename, for example `fcos.yaml` -> `fcos`.
