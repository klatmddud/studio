#!/usr/bin/env bash

set -u
set -o pipefail
export PYTHONUNBUFFERED=1

DATA="pascal"
MODEL="fcos"
BACKBONE="resnet50"
PWD="scripts/bash/fcos_HR_interval"

MODEL_CFG="models/detection/cfg/$MODEL.yaml"

DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$LOG_DIR"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-I1" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I1.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-I1/train.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-I2" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I2.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-I2/train.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-I3" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I3.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-I3/train.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-I4" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I4.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-I4/train.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-I5" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I5.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-I5/train.log"