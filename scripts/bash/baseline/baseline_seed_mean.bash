#!/usr/bin/env bash

set -u
set -o pipefail

DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="scripts/bash/baseline/logs"

mkdir -p "$LOG_DIR"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --seed 42 \
  --device ${DEVICE} \
  --mdmb-config scripts/bash/baseline/cfg/mdmb_off.yaml \
  --mdmbpp-config scripts/bash/baseline/cfg/mdmbpp_off.yaml \
  --rasd-config scripts/bash/baseline/cfg/rasd_off.yaml \
  --hard-replay-config scripts/bash/baseline/cfg/hard_replay_off.yaml \
  --output-dir runs/train/kitti/fcos/resnet18/baseline_mean/seed_42 \
  2>&1 | tee -a "$LOG_DIR/baseline_seed_42.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --seed 43 \
  --device ${DEVICE} \
  --mdmb-config scripts/bash/baseline/cfg/mdmb_off.yaml \
  --mdmbpp-config scripts/bash/baseline/cfg/mdmbpp_off.yaml \
  --rasd-config scripts/bash/baseline/cfg/rasd_off.yaml \
  --hard-replay-config scripts/bash/baseline/cfg/hard_replay_off.yaml \
  --output-dir runs/train/kitti/fcos/resnet18/baseline_mean/seed_43 \
  2>&1 | tee -a "$LOG_DIR/baseline_seed_43.log"

uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --seed 44 \
  --device ${DEVICE} \
  --mdmb-config scripts/bash/baseline/cfg/mdmb_off.yaml \
  --mdmbpp-config scripts/bash/baseline/cfg/mdmbpp_off.yaml \
  --rasd-config scripts/bash/baseline/cfg/rasd_off.yaml \
  --hard-replay-config scripts/bash/baseline/cfg/hard_replay_off.yaml \
  --output-dir runs/train/kitti/fcos/resnet18/baseline_mean/seed_44 \
  2>&1 | tee -a "$LOG_DIR/baseline_seed_44.log"
