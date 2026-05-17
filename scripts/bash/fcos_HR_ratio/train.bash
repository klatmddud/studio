#!/usr/bin/env bash

set -u
set -o pipefail
export PYTHONUNBUFFERED=1

DATA="${DATA:-kitti}"
MODEL="fcos"
BACKBONE="resnet50"
PWD="scripts/bash/fcos_HR_ratio"
MODEL_CFG="models/detection/cfg/$MODEL.yaml"
DEVICE="${DEVICE:-cuda:0 cuda:1}"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/HR-R0125"

uv run scripts/train.py \
  --config "$PWD/cfg/train.yaml" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-R0125" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_0125.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-R0125/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/HR-R025"

uv run scripts/train.py \
  --config "$PWD/cfg/train.yaml" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-R025" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_025.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-R025/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/HR-R05"

uv run scripts/train.py \
  --config "$PWD/cfg/train.yaml" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/HR-R05" \
  --remiss-config "$PWD/cfg/remiss.yaml" \
  --hard-replay-config "$PWD/cfg/HR_05.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/HR-R05/train.log"
