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

mkdir -p "$LOG_DIR"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_42" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_42.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 43 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_43" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_43.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 44 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_44" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_44.log"
