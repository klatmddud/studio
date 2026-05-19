#!/usr/bin/env bash

set -u
set -o pipefail
export PYTHONUNBUFFERED=1

DATA="${DATA:-kitti}"
MODEL="fcos"
BACKBONE="resnet50"
PWD="scripts/bash/fcos_HR_interval"
MODEL_CFG="models/detection/cfg/$MODEL.yaml"
TRAIN_CFG="scripts/bash/cfg/fcos_train.yaml"
DEVICE="${DEVICE:-cuda:0 cuda:1}"
MODE="${MODE:-90K}"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I1"

uv run scripts/train.py \
  --config "$TRAIN_CFG" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device "${DEVICE}" \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I1" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I1.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I1/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I2"

uv run scripts/train.py \
  --config "$TRAIN_CFG" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device "${DEVICE}" \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I2" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I2.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I2/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I3"

uv run scripts/train.py \
  --config "$TRAIN_CFG" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device "${DEVICE}" \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I3" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I3.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I3/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/HR-I4"

uv run scripts/train.py \
  --config "$PWD/cfg/train.yaml" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device "${DEVICE}" \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I4" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I4.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I4/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I5"

uv run scripts/train.py \
  --config "$PWD/cfg/train.yaml" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device "${DEVICE}" \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I5" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_I5.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/$MODE/HR-I5/train.log"