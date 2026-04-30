# AGENTS.md

PyTorch object detection training framework supporting FCOS, Faster R-CNN, and DINO with COCO-format datasets. Current research-module work is centered on ReMiss and MPD.

## Quick Reference

| Task | Docs |
|---|---|
| Project layout, entry points, tech stack | [docs/architecture.md](docs/architecture.md) |
| Training config, engine, checkpointing | [docs/training.md](docs/training.md) |
| Model architectures and wrappers | [docs/models.md](docs/models.md) |
| Research modules (ReMiss, MPD) | [docs/modules.md](docs/modules.md) |
| Dataset format, DataLoader, env vars | [docs/data.md](docs/data.md) |

## Common Commands

```bash
# Install
uv sync

# Train
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti --seed 42 --device cuda:0 cuda:1
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti --remiss-config modules/cfg/remiss.yaml
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti --remiss-conv-config modules/cfg/remiss_conv.yaml
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti --mpd-config modules/cfg/mpd.yaml
bash scripts/bash/baseline/train.bash

# Evaluate
uv run scripts/eval.py --config scripts/cfg/eval.yaml --model models/detection/cfg/fcos.yaml --checkpoint runs/train/checkpoints/best.pt
```

## Documentation Policy

When code changes affect training, model wrappers, modules, data loading, or commands, update the related docs in the same change.

| Change | Update |
|---|---|
| Training/evaluation flow, config schema, checkpointing | [docs/training.md](docs/training.md) |
| Project structure, entry points, dependencies | [docs/architecture.md](docs/architecture.md) |
| Model architecture, wrappers, config fields | [docs/models.md](docs/models.md) |
| Research modules | [docs/modules.md](docs/modules.md) |
| Dataset format, DataLoader, environment variables | [docs/data.md](docs/data.md) |
| Common commands or repository rules | This file |

## Key Conventions

- Config uses `${ENV_VAR}` placeholders resolved from `.env` at runtime.
- `--data kitti` selects dataset-specific env vars (`KITTI_*`); omit if using raw paths in YAML.
- `checkpoint.dir` defaults to `checkpoints/` relative to cwd; `output_dir` defaults to `runs/train`.
- Research modules are disabled by default in `modules/cfg/*.yaml`.
