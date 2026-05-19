# Detection Training / Evaluation Guide

이 저장소는 COCO 형식 annotation JSON을 사용하는 객체 탐지 학습/평가 스크립트를 제공합니다.
현재 모델 설정 예시는 아래 3가지를 포함합니다.

- `models/detection/cfg/fasterrcnn.yaml`
- `models/detection/cfg/fcos.yaml`
- `models/detection/cfg/dino.yaml`

## 1. 환경 준비

### uv

의존성 설치:

```powershell
uv sync
```

이후 명령은 README 예시처럼 `uv run ...` 형식으로 실행합니다.

### pip

가상환경 생성 및 의존성 설치: CUDA 버전은 본인의 환경에 맞게 조절해야 합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

`pip` 환경에서는 이후 명령의 `uv run`을 빼고 `python`으로 실행합니다.

```powershell
python scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti
```

### conda

Conda 환경 생성 및 의존성 설치: CUDA 버전은 본인의 환경에 맞게 조절해야 합니다.

```powershell
conda create -n studio python=3.12 -y
conda activate studio
python -m pip install --upgrade pip
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

`conda` 환경에서도 이후 명령의 `uv run`을 빼고 `python`으로 실행합니다.

학습/평가 런타임 설정은 프로젝트 루트의 `.env` 파일을 자동으로 읽습니다.
`scripts/cfg/train.yaml`, `scripts/cfg/eval.yaml`에서 `${VAR_NAME}` 형태로 참조한 경로는 `.env`에 정의해 두면 됩니다.

예시 `.env`:

```dotenv
KITTI_TRAIN_IMAGES=C:/datasets/kitti/images/train
KITTI_TRAIN_ANNOTATIONS=C:/datasets/kitti/annotations/train.json
KITTI_VAL_IMAGES=C:/datasets/kitti/images/val
KITTI_VAL_ANNOTATIONS=C:/datasets/kitti/annotations/val.json
```

참고:

- `.env` 파일 위치는 프로젝트 루트입니다.
- annotation 파일은 COCO 형식 JSON이어야 합니다.
- `.env`에 없는 변수를 YAML에서 참조하면 실행 시 에러가 발생합니다.

## 2. `scripts/cfg/train.yaml` / `scripts/cfg/eval.yaml` 데이터 경로 설정

기본 예시는 `.env` 변수를 사용하도록 작성되어 있습니다.

`scripts/cfg/train.yaml`

```yaml
data:
  train_images: ${KITTI_TRAIN_IMAGES}
  train_annotations: ${KITTI_TRAIN_ANNOTATIONS}
  val_images: ${KITTI_VAL_IMAGES}
  val_annotations: ${KITTI_VAL_ANNOTATIONS}
```

`scripts/cfg/eval.yaml`

```yaml
data:
  images: ${KITTI_VAL_IMAGES}
  annotations: ${KITTI_VAL_ANNOTATIONS}
```

필요하면 `${...}` 대신 직접 경로를 적어도 됩니다.

```yaml
data:
  train_images: C:/datasets/kitti/images/train
  train_annotations: C:/datasets/kitti/annotations/train.json
  val_images: C:/datasets/kitti/images/val
  val_annotations: C:/datasets/kitti/annotations/val.json
```

경로 해석 규칙:

- 상대 경로는 프로젝트 루트 기준으로 해석됩니다.
- `train.yaml`의 `checkpoint.dir`는 `output_dir` 기준으로 해석됩니다.
- `eval.yaml`의 `eval.predictions_path`는 `output_dir` 기준으로 해석됩니다.

## 3. 학습 실행

기본 실행:

```powershell
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data kitti
```

### 학습 CLI 옵션

| 옵션 | 설명 |
| --- | --- |
| `--config` | 학습 런타임 YAML 경로. 필수 |
| `--model` | 모델 YAML 경로. 필수 |
| `--data` | 데이터셋 env prefix 선택. 예: `kitti`, `pascal`, `visdrone`, `bdd100k`, `bdd10k` |
| `--output-dir` | `train.yaml`의 `output_dir`를 CLI에서 덮어씀 |
| `--seed` | `train.yaml`의 `seed`를 CLI에서 덮어씀. 0 이상의 정수 |
| `--device` | `train.yaml`의 `device`를 CLI에서 덮어씀. 단일 장치 또는 DDP용 복수 CUDA 장치 지정. 예: `auto`, `cpu`, `cuda`, `mps`, `cuda:0 cuda:1` |
| `--remiss-config` | ReMiss YAML 설정 경로를 CLI에서 덮어씀 |
| `--hard-replay-config` | Hard Replay YAML 설정 경로를 CLI에서 덮어씀 |

### 학습 커맨드 예시

FCOS 학습:

```powershell
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fcos.yaml --data bdd100k
```

Faster R-CNN 학습 결과를 별도 폴더에 저장:

```powershell
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/fasterrcnn.yaml --data kitti --output-dir runs/fasterrcnn_kitti --device cuda
```

DINO 학습: DINO는 아직 미구현

```powershell
uv run scripts/train.py --config scripts/cfg/train.yaml --model models/detection/cfg/dino.yaml --data bdd10k
```

### `scripts/cfg/train.yaml` 주요 파라미터

| 섹션 | 주요 키 | 설명 |
| --- | --- | --- |
| 공통 | `seed`, `device`, `amp`, `output_dir` | 시드, 디바이스, AMP 사용 여부, 결과 저장 위치 |
| `data` | `train_images`, `train_annotations`, `val_images`, `val_annotations` | 학습/검증 이미지 폴더와 COCO annotation 경로 |
| `loader` | `batch_size`, `num_workers`, `pin_memory`, `shuffle` | DataLoader 설정 |
| `optimizer` | `name`, `lr`, `momentum`, `weight_decay`, `nesterov`, `betas`, `eps` | 현재 `sgd`, `adamw` 지원 |
| `scheduler` | `name`, `unit`, `milestones`, `gamma`, `step_size`, `t_max`, `eta_min` | 현재 `none`, `multistep`, `step`, `cosine` 지원. `unit`은 `epoch` 또는 `iteration` |
| `train` | `epochs`, `max_iterations`, `grad_clip_norm`, `log_interval`, `eval_every_epochs`, `eval_every_iterations` | 학습 epoch 수, iteration budget, gradient clipping, 로그 주기, 검증 주기 |
| `checkpoint` | `dir`, `resume`, `save_last`, `save_best`, `monitor`, `mode` | 체크포인트 저장 경로, 재시작 경로, best/last 저장 여부 |
| `metrics` | `type`, `iou_types`, `primary` | 현재 `coco_detection`, `bbox`만 지원 |

Iteration 기준 스케줄을 사용하려면 `scheduler.unit: iteration`으로 설정하고, `scheduler.milestones`에는 global optimizer step 기준 값을 넣습니다. `train.max_iterations`가 설정되어 있으면 해당 iteration 수에서 학습이 종료됩니다.

```yaml
scheduler:
  name: multistep
  unit: iteration
  milestones: [60000, 80000]
  gamma: 0.1

train:
  epochs: 300
  max_iterations: 90000
  eval_every_epochs: 1
  eval_every_iterations: 3000
```

체크포인트 재시작은 CLI 옵션이 아니라 YAML에서 설정합니다.

```yaml
checkpoint:
  dir: checkpoints
  resume: runs/prev_experiment/checkpoints/last.pt
  save_last: true
  save_best: true
  monitor: bbox_mAP_50_95
  mode: max
```

`checkpoint.monitor`와 `metrics.primary`에 사용할 수 있는 metric 이름:

- `bbox_mAP_50_95`
- `bbox_mAP_50`
- `bbox_mAP_75`
- `bbox_mAP_small`
- `bbox_mAP_medium`
- `bbox_mAP_large`
- `bbox_mAR_1`
- `bbox_mAR_10`
- `bbox_mAR_100`
- `bbox_mAR_small`
- `bbox_mAR_medium`
- `bbox_mAR_large`

학습 결과물:

- `{output_dir}/history.json`
- `{output_dir}/metadata/model.yaml`
- `{output_dir}/metadata/train.yaml`
- `{output_dir}/metadata/run.json`
- `{output_dir}/checkpoints/last.pt`
- `{output_dir}/checkpoints/best.pt`

## 4. 평가 실행

기본 실행:

```powershell
uv run scripts/eval.py --config scripts/cfg/eval.yaml --model models/detection/cfg/fcos.yaml --data kitti
```

실무에서는 보통 평가할 체크포인트를 함께 지정합니다.

```powershell
uv run scripts/eval.py --config scripts/cfg/eval.yaml --model models/detection/cfg/fcos.yaml --data kitti --checkpoint runs/train/checkpoints/best.pt
```

### 평가 CLI 옵션

| 옵션 | 설명 |
| --- | --- |
| `--config` | 평가 런타임 YAML 경로. 필수 |
| `--model` | 모델 YAML 경로. 필수 |
| `--data` | 데이터셋 env prefix 선택. 예: `kitti`, `bdd100k`, `bdd10k` |
| `--output-dir` | `eval.yaml`의 `output_dir`를 CLI에서 덮어씀 |
| `--device` | `runtime.device`를 CLI에서 덮어씀 |
| `--checkpoint` | `eval.yaml`의 `checkpoint.path`를 CLI에서 덮어씀 |

### 평가 커맨드 예시

FCOS best 체크포인트 평가:

```powershell
uv run scripts/eval.py --config scripts/cfg/eval.yaml --model models/detection/cfg/fcos.yaml --data bdd100k --checkpoint runs/fcos_exp/checkpoints/best.pt
```

평가 결과를 별도 폴더에 저장:

```powershell
uv run scripts/eval.py --config scripts/cfg/eval.yaml --model models/detection/cfg/dino.yaml --data bdd10k --checkpoint runs/dino_exp/checkpoints/best.pt --output-dir runs/dino_eval --device cuda
```

### `scripts/cfg/eval.yaml` 주요 파라미터

| 섹션 | 주요 키 | 설명 |
| --- | --- | --- |
| 공통 | `device`, `amp`, `output_dir` | 평가 디바이스, AMP 사용 여부, 결과 저장 위치 |
| `data` | `images`, `annotations` | 평가 이미지 폴더와 COCO annotation 경로 |
| `loader` | `batch_size`, `num_workers`, `pin_memory`, `shuffle` | 평가용 DataLoader 설정 |
| `checkpoint` | `path` | 평가에 사용할 `.pt` 체크포인트 경로 |
| `metrics` | `type`, `iou_types`, `primary` | 현재 `coco_detection`, `bbox`만 지원 |
| `eval` | `log_interval`, `save_predictions`, `predictions_path` | 로그 주기, 예측 저장 여부, 저장 경로 |

예측 결과까지 저장하려면 `eval.yaml`에서 설정합니다.

```yaml
eval:
  log_interval: 20
  save_predictions: true
  predictions_path: predictions.json
```

`predictions_path`를 비워두면 기본값은 `{output_dir}/predictions.json`입니다.

평가 결과물:

- `{output_dir}/metrics.json`
- `{output_dir}/metadata/model.yaml`
- `{output_dir}/metadata/eval.yaml`
- `{output_dir}/metadata/run.json`
- `{output_dir}/predictions.json` 또는 `eval.predictions_path`에 지정한 파일

## 5. 빠른 체크

CLI 도움말:

```powershell
uv run scripts/train.py --help
uv run scripts/eval.py --help
```

자주 확인할 항목:

- `--model`에 지정한 YAML 파일과 체크포인트가 같은 아키텍처인지 확인
- `train.yaml`, `eval.yaml`의 annotation 경로가 COCO JSON인지 확인
- `checkpoint.save_best: true`를 사용할 때는 validation 경로가 반드시 필요
- `--output-dir`를 바꾸면 학습 체크포인트 저장 위치도 함께 바뀜
