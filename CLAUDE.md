# CLAUDE.md

PyTorch object detection training framework supporting FCOS, Faster R-CNN, and DINO with COCO-format datasets. Includes research modules (MDMB, RECALL, CFP, MODS, SCA) for improving detection performance.

## Quick Reference

| Task | Docs |
|---|---|
| Project layout, entry points, tech stack | [docs/architecture.md](docs/architecture.md) |
| Training config, engine, checkpointing | [docs/training.md](docs/training.md) |
| Model architectures and wrappers | [docs/models.md](docs/models.md) |
| Research modules (MDMB, RECALL, CFP, …) | [docs/modules.md](docs/modules.md) |
| Dataset format, DataLoader, env vars | [docs/data.md](docs/data.md) |

## Common Commands

```bash
# Install
uv sync

# Train
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti

# Evaluate
uv run scripts/eval.py --config scripts/cfg/eval.yaml --model models/detection/cfg/fcos.yaml --checkpoint runs/train/checkpoints/best.pt
```

## Documentation Policy

코드를 수정하거나 추가할 때 관련 문서도 함께 업데이트한다.

| 변경 내용 | 업데이트할 문서 |
|---|---|
| 학습/평가 흐름, config 스키마, 체크포인트 | [docs/training.md](docs/training.md) |
| 프로젝트 구조, 진입점, 의존성 | [docs/architecture.md](docs/architecture.md) |
| 모델 아키텍처, 래퍼, config 필드 | [docs/models.md](docs/models.md) |
| 연구 모듈 추가/수정/삭제 | [docs/modules.md](docs/modules.md) |
| 데이터셋 형식, DataLoader, 환경변수 | [docs/data.md](docs/data.md) |
| 자주 쓰는 커맨드, 규칙 변경 | 이 파일 (CLAUDE.md) |

## Key Conventions

- Config uses `${ENV_VAR}` placeholders resolved from `.env` at runtime.
- `--data kitti` selects dataset-specific env vars (`KITTI_*`); omit if using raw paths in YAML.
- `checkpoint.dir` defaults to `checkpoints/` (relative to cwd); `output_dir` defaults to `runs/train`.
- All research modules are **disabled by default** in `modules/cfg/*.yaml`.
