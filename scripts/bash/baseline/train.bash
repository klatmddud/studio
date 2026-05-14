#!/usr/bin/env bash

set -u
set -o pipefail
export PYTHONUNBUFFERED=1

MOD="baseline"
DATA="kitti"
MODEL="fcos"
BACKBONE="resnet50"
MODEL_CFG="models/detection/cfg/$MODEL.yaml"

DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="scripts/bash/$MOD/logs"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_42" \
  --remiss-config "scripts/bash/$MOD/cfg/remiss.yaml" \
  --hard-replay-config "scripts/bash/$MOD/cfg/hard_replay.yaml" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_42.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 43 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_43" \
  --remiss-config "scripts/bash/$MOD/cfg/remiss.yaml" \
  --hard-replay-config "scripts/bash/$MOD/cfg/hard_replay.yaml" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_43.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 44 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_44" \
  --remiss-config "scripts/bash/$MOD/cfg/remiss.yaml" \
  --hard-replay-config "scripts/bash/$MOD/cfg/hard_replay.yaml" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_44.log"
