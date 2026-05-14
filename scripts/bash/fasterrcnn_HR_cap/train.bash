#!/usr/bin/env bash

set -u
set -o pipefail
export PYTHONUNBUFFERED=1

DATA="kitti"
MODEL="fasterrcnn"
BACKBONE="resnet50"
PWD="scripts/bash/fasterrcnn_HR_cap"
MODEL_CFG="models/detection/cfg/$MODEL.yaml"
DEVICE="${DEVICE:-cuda:1}"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap2"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap2" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_2.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap2/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap4"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap4" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_4.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap4/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap8"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap8" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_8.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-cap8/train.log"
