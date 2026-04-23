#!/usr/bin/env bash

set -u
set -o pipefail

FCOS="models/detection/cfg/fcos.yaml"
MODULE_BASE_PATH="scripts/bash/baseline/cfg"

DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="scripts/bash/baseline/logs"

mkdir -p "$LOG_DIR"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$FCOS" \
  --data kitti \
  --seed 42 \
  --device ${DEVICE} \
  --mdmb-config "$MODULE_BASE_PATH/mdmb_off.yaml" \
  --mdmbpp-config "$MODULE_BASE_PATH/mdmbpp_off.yaml" \
  --rasd-config "$MODULE_BASE_PATH/rasd_off.yaml" \
  --hard-replay-config "$MODULE_BASE_PATH/hard_replay_off.yaml" \
  --output-dir runs/train/kitti/fcos/resnet18/baseline_mean/seed_42 \
  2>&1 | tee -a "$LOG_DIR/baseline_seed_42.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$FCOS" \
  --data kitti \
  --seed 43 \
  --device ${DEVICE} \
  --mdmb-config "$MODULE_BASE_PATH/mdmb_off.yaml" \
  --mdmbpp-config "$MODULE_BASE_PATH/mdmbpp_off.yaml" \
  --rasd-config "$MODULE_BASE_PATH/rasd_off.yaml" \
  --hard-replay-config "$MODULE_BASE_PATH/hard_replay_off.yaml" \
  --output-dir runs/train/kitti/fcos/resnet18/baseline_mean/seed_43 \
  2>&1 | tee -a "$LOG_DIR/baseline_seed_43.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model "$FCOS" \
  --data kitti \
  --seed 44 \
  --device ${DEVICE} \
  --mdmb-config "$MODULE_BASE_PATH/mdmb_off.yaml" \
  --mdmbpp-config "$MODULE_BASE_PATH/mdmbpp_off.yaml" \
  --rasd-config "$MODULE_BASE_PATH/rasd_off.yaml" \
  --hard-replay-config "$MODULE_BASE_PATH/hard_replay_off.yaml" \
  --output-dir runs/train/kitti/fcos/resnet18/baseline_mean/seed_44 \
  2>&1 | tee -a "$LOG_DIR/baseline_seed_44.log"
