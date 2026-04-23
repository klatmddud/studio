#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="scripts/bash/mdmbpp_only"
CFG_DIR="${SCRIPT_DIR}/cfg"
LOG_DIR="${SCRIPT_DIR}/logs"

MODEL_CFG="models/detection/cfg/fcos.yaml"
DATASET="kitti"
DEVICE="${DEVICE:-cuda:0}"

DEFAULT_MDMB_CFG="modules/cfg/mdmb.yaml"
DEFAULT_MDMBPP_CFG="modules/cfg/mdmbpp.yaml"
DEFAULT_RASD_CFG="modules/cfg/rasd.yaml"
DEFAULT_HARD_REPLAY_CFG="modules/cfg/hard_replay.yaml"

RUN_MDMB_CFG="${CFG_DIR}/mdmb_disabled.yaml"
RUN_MDMBPP_CFG="${CFG_DIR}/mdmbpp_only.yaml"
RUN_RASD_CFG="${CFG_DIR}/rasd_disabled.yaml"
RUN_HARD_REPLAY_CFG="${CFG_DIR}/hard_replay_disabled.yaml"

SEEDS=(42 43 44)

read -r -a DEVICE_ARGS <<< "$DEVICE"
DEVICE_LABEL="${DEVICE_ARGS[*]}"

mkdir -p "$CFG_DIR" "$LOG_DIR"

uv run python - \
  "$DEFAULT_MDMB_CFG" "$DEFAULT_MDMBPP_CFG" "$DEFAULT_RASD_CFG" "$DEFAULT_HARD_REPLAY_CFG" \
  "$RUN_MDMB_CFG" "$RUN_MDMBPP_CFG" "$RUN_RASD_CFG" "$RUN_HARD_REPLAY_CFG" <<'PY'
from pathlib import Path
import sys

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return data


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def set_enabled(data: dict, enabled: bool) -> dict:
    data = dict(data)
    data["enabled"] = bool(enabled)
    models = data.setdefault("models", {})
    if not isinstance(models, dict):
        models = {}
        data["models"] = models
    fcos = models.setdefault("fcos", {})
    if not isinstance(fcos, dict):
        fcos = {}
        models["fcos"] = fcos
    fcos["enabled"] = bool(enabled)
    return data


(
    default_mdmb,
    default_mdmbpp,
    default_rasd,
    default_hard_replay,
    run_mdmb,
    run_mdmbpp,
    run_rasd,
    run_hard_replay,
) = map(Path, sys.argv[1:])

write_yaml(run_mdmb, set_enabled(load_yaml(default_mdmb), False))
write_yaml(run_rasd, set_enabled(load_yaml(default_rasd), False))
write_yaml(run_hard_replay, set_enabled(load_yaml(default_hard_replay), False))

mdmbpp = set_enabled(load_yaml(default_mdmbpp), True)
mdmbpp["warmup_epochs"] = 0
mdmbpp["store_topk_candidates"] = True
mdmbpp["store_support_feature"] = True
write_yaml(run_mdmbpp, mdmbpp)
PY
CONFIG_STATUS=$?
if [[ "$CONFIG_STATUS" -ne 0 ]]; then
  echo "Failed to prepare MDMB++ only module YAMLs."
  exit "$CONFIG_STATUS"
fi

FAILED=()

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="seed_${SEED}"
  OUTPUT_DIR="runs/train/kitti/fcos/resnet18/mdmbpp_only/${RUN_NAME}"
  TRAIN_CFG="${CFG_DIR}/train_${RUN_NAME}.yaml"
  LOG_PATH="${LOG_DIR}/mdmbpp_only_${RUN_NAME}.log"

  cat > "$TRAIN_CFG" <<EOF
seed: ${SEED}
device: ${DEVICE_ARGS[0]}
amp: false
output_dir: ${OUTPUT_DIR}

data:
  train_images: \${KITTI_TRAIN_IMAGES}
  train_annotations: \${KITTI_TRAIN_ANNOTATIONS}
  val_images: \${KITTI_VAL_IMAGES}
  val_annotations: \${KITTI_VAL_ANNOTATIONS}

loader:
  batch_size: 32
  num_workers: 8
  pin_memory: true
  shuffle: true

optimizer:
  name: sgd
  lr: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  nesterov: false

scheduler:
  name: multistep
  milestones: [8, 11]
  gamma: 0.1

train:
  epochs: 50
  grad_clip_norm: 1.0
  log_interval: 20
  eval_every_epochs: 1

checkpoint:
  dir: checkpoints
  resume: null
  save_last: true
  save_best: true
  monitor: bbox_mAP_50_95
  mode: max

metrics:
  type: coco_detection
  iou_types: [bbox]
  primary: bbox_mAP_50_95
EOF

  echo "============================================================" | tee "$LOG_PATH"
  echo "[mdmbpp_only] seed=${SEED} output=${OUTPUT_DIR} device=${DEVICE_LABEL}" | tee -a "$LOG_PATH"
  echo "============================================================" | tee -a "$LOG_PATH"

  uv run scripts/train.py \
    --config "$TRAIN_CFG" \
    --model "$MODEL_CFG" \
    --data "$DATASET" \
    --seed "$SEED" \
    --device "${DEVICE_ARGS[@]}" \
    --mdmb-config "$RUN_MDMB_CFG" \
    --mdmbpp-config "$RUN_MDMBPP_CFG" \
    --rasd-config "$RUN_RASD_CFG" \
    --hard-replay-config "$RUN_HARD_REPLAY_CFG" \
    2>&1 | tee -a "$LOG_PATH"

  STATUS=${PIPESTATUS[0]}
  if [[ "$STATUS" -ne 0 ]]; then
    echo "[FAILED] seed=${SEED} status=${STATUS}" | tee -a "$LOG_PATH"
    FAILED+=("${SEED}")
    continue
  fi

  echo "[DONE] seed=${SEED}" | tee -a "$LOG_PATH"
done

echo "============================================================"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Completed with failed seeds: ${FAILED[*]}"
  exit 1
fi

echo "All MDMB++ only seed runs completed successfully."
