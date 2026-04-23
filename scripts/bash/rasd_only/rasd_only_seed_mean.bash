#!/usr/bin/env bash

set -u
set -o pipefail
export PYTHONUNBUFFERED=1

MOD="rasd_only"
DATA="kitti"
MODEL="fcos"
BACKBONE="resnet18"

MODEL_CFG="models/detection/cfg/$MODEL.yaml"
MODULE_BASE_PATH="scripts/bash/$MOD/cfg"
MDMB_CFG="$MODULE_BASE_PATH/mdmb_off.yaml"
MDMBPP_CFG="$MODULE_BASE_PATH/mdmbpp_on.yaml"
HARD_REPLAY_CFG="$MODULE_BASE_PATH/hard_replay_off.yaml"
RASD_CFG="$MODULE_BASE_PATH/rasd_on.yaml"

DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="scripts/bash/$MOD/logs"

mkdir -p "$LOG_DIR"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 42 \
  --device ${DEVICE} \
  --mdmb-config ${MDMB_CFG}$ \
  --mdmbpp-config ${MDMBPP_CFG} \
  --hard-replay-config ${HARD_REPLAY_CFG} \
  --rasd-config ${RASD_CFG} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_42" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_42.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 43 \
  --device ${DEVICE} \
  --mdmb-config ${MDMB_CFG}$ \
  --mdmbpp-config ${MDMBPP_CFG} \
  --hard-replay-config ${HARD_REPLAY_CFG} \
  --rasd-config ${RASD_CFG} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_43" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_43.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$MODEL_CFG" \
  --data "$DATA" \
  --seed 44 \
  --device ${DEVICE} \
  --mdmb-config ${MDMB_CFG}$ \
  --mdmbpp-config ${MDMBPP_CFG} \
  --hard-replay-config ${HARD_REPLAY_CFG} \
  --rasd-config ${RASD_CFG} \
  --output-dir "runs/train/$DATA/$MODEL/$BACKBONE/${MOD}_mean/seed_44" \
  2>&1 | tee -a "$LOG_DIR/${MOD}_seed_44.log"
