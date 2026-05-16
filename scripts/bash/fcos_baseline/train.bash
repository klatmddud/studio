#!/usr/bin/env bash

set -u
set -o pipefail
export PYTHONUNBUFFERED=1

DATA="pascal"
MODEL="fcos"
BACKBONE="resnet50"
PWD="scripts/bash/fcos_baseline"
MODEL_CFG="models/detection/cfg/$MODEL.yaml"
DEVICE="cuda:0 cuda:1"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/baseline"

uv run scripts/train.py \
  --config "$PWD/cfg/train.yaml" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/baseline" \
  --remiss-config "$PWD/cfg/remiss_off.yaml" \
  --hard-replay-config "$PWD/cfg/hard_replay.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/baseline/train.log"

mkdir -p "runs/train/$DATA/$MODEL/$BACKBONE/missbank"

uv run scripts/train.py \
  --config "$PWD/cfg/train.yaml" \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/missbank" \
  --remiss-config "$PWD/cfg/remiss_on.yaml" \
  --hard-replay-config "$PWD/cfg/hard_replay.yaml" \
  2>&1 | tee -a "runs/train/$DATA/$MODEL/$BACKBONE/missbank/train.log"