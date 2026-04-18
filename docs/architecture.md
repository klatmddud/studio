# Architecture

## Directory Layout

```
├── scripts/
│   ├── train.py                  # Entry point: training
│   ├── eval.py                   # Entry point: evaluation
│   ├── cfg/
│   │   ├── train.yaml            # Runtime config template (train)
│   │   └── eval.yaml             # Runtime config template (eval)
│   ├── runtime/
│   │   ├── config.py             # Config loading, env-var substitution, validation
│   │   ├── data.py               # COCO dataset + DataLoader builders
│   │   ├── engine.py             # fit(), evaluate(), train_one_epoch(), checkpointing
│   │   ├── metrics.py            # COCO evaluation via pycocotools
│   │   ├── registry.py           # Builds model from YAML (arch dispatch)
│   │   └── dataset_meta.py       # Infers num_classes from COCO JSON
│   └── utils/
│       └── yolo2coco.py          # YOLO→COCO format converter
│
├── models/detection/
│   ├── cfg/                      # Per-architecture YAML (fcos/fasterrcnn/dino)
│   └── wrapper/
│       ├── _base.py              # Backbone + FPN builder
│       ├── fcos.py               # MDMBFCOS wrapper
│       ├── fasterrcnn.py         # FasterRCNN wrapper
│       └── dino.py               # DINO wrapper
│
├── modules/
│   ├── cfg/                      # Module configs (all disabled by default)
│   └── nn/                       # Module implementations (mdmb/recall/cfp/mods/sca)
│
└── ops/                          # Custom loss functions (cfp/mods/sca)
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

```
train.py
  └─ load_runtime_config()       # merge YAML + env vars
  └─ build_model_from_path()     # registry.py → wrapper
  └─ build_train_dataloaders()   # data.py → CocoDetectionDataset
  └─ fit()                       # engine.py → training loop
```

## Supported Architectures

| arch key | Wrapper | Backbone options |
|---|---|---|
| `fcos` | `MDMBFCOS` | ResNet18/50, MobileNetV2/V3 |
| `fasterrcnn` | `FasterRCNN` | ResNet50/101, MobileNetV2/V3 |
| `dino` | `DINOWrapper` | ResNet50 |

`arch` is inferred from the model YAML filename (e.g., `fcos.yaml` → `fcos`).
