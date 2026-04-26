# Research Modules

Research features are configured from `modules/cfg/*.yaml`.
`mdmb`, `mdmbpp`, `rasd`, `tfm`, `fntdm`, and `dhm` live in `modules/nn/`.
`hard_replay` is configured in the same folder but implemented in
`scripts/runtime/hard_replay.py` because it operates at the data-loading layer.

**All research modules are disabled by default.**

## Enabling a Module

Edit the corresponding YAML file in `modules/cfg/`:

```yaml
enabled: true
```

Per-architecture overrides use the `models:` mapping:

```yaml
models:
  fcos:
    enabled: true
  fasterrcnn:
    enabled: false
  dino:
    enabled: false
```

For parallel experiments, prefer per-run config files instead of editing the global YAML files.
`scripts/train.py` accepts `--mdmb-config`, `--mdmbpp-config`, `--rasd-config`,
`--hard-replay-config`, `--tfm-config`, `--fntdm-config`, and `--dhm-config`; those paths override
the defaults for that run only and are also used when writing `metadata/modules.yaml`.

## Module Overview

### MDMB - Missed Detection Memory Bank (`modules/nn/mdmb.py`)

Tracks false-negative GTs over time.

- Hooks: `start_epoch()`, `end_epoch()`
- FCOS hook: `after_optimizer_step(images, targets, epoch_index)`
- Summary: `mdmb.summary()`
- Config: `modules/cfg/mdmb.yaml`
- Arch support: FCOS

Key state:

- `MDMBEntry`: unresolved miss entry stored in the bank
- `_GTRecord`: persistent per-GT temporal record

### MDMB++ - Structured Failure Memory (`modules/nn/mdmbpp.py`)

Stores structured miss state for each unresolved GT rather than only a miss flag.

- Hooks: `start_epoch()`, `end_epoch()`
- Update API: `mdmbpp.update(...)`
- Summary: `mdmbpp.summary()`
- Config: `modules/cfg/mdmbpp.yaml`
- Arch support: FCOS

Key state:

- `CanonicalCandidate`: detector-specific candidate normalized into a common schema
- `SupportSnapshot`: last successful support state
- `GTFailureRecord`: persistent per-GT history
- `MDMBPlusEntry`: unresolved miss entry with `failure_type`, `severity`, and candidate context

Useful read APIs for downstream modules:

- `mdmbpp.get_image_entries(image_id)`
- `mdmbpp.get_replay_priority(image_id)`
- `mdmbpp.get_dense_targets(image_id)`
- `mdmbpp.get_record(gt_uid)`

Implementation note: MDMB++ stores persistent memory tensors on CPU, but transient IoU checks align
GT, label, and score tensors to the final-detection device before calling TorchVision box ops.
When `store_support_feature: true`, FCOS post-step updates also store object-level support feature
vectors in `SupportSnapshot.feature` for downstream temporal distillation modules such as RASD.
MDMB++ uses quality-gated support memory by default: a newer detected support replaces the old
teacher only when its score/IoU quality clears the configured margin, the old teacher is stale, or
the old teacher lacks a feature.

### Hard Replay (`scripts/runtime/hard_replay.py`)

Data-layer replay that redistributes training exposure toward images whose GTs remain unresolved in
`MDMB++`.

- Planner: `HardReplayPlanner.build_epoch_index(...)`
- Sampler: `MixedReplayBatchSampler`
- Controller: `HardReplayController`
- Config: `modules/cfg/hard_replay.yaml`
- Arch support: data-layer replay is model-agnostic; replay-aware loss weighting is FCOS-only

Current scope:

- Epoch-level `ReplayIndex`
- Image-level replay
- Object-level crop replay through `object_replay.crop`
- Rectangular copy-paste replay through `object_replay.copy_paste`
- Support/miss pair replay through `object_replay.pair`
- Mixed batch composition
- FCOS replay-aware per-GT loss weighting through replay target metadata

Important behavior:

- Replay weight uses `1 + beta * sum(severity)` per image, clipped by `max_image_weight`
- Sampling weight applies `temperature` as an exponent
- Replay candidates are filtered by `replay_recency_window`
- Per-image replay repeats are capped by `max_replays_per_gt_per_epoch`
- Object replay creates virtual indices after the base dataset range
- Replay targets set `is_replay: true` and are skipped by FCOS MDMB/MDMB++ memory updates
- Pair replay keeps `pair_miss` and `pair_support` in the same mini-batch when replay slots allow it
- FCOS applies `replay_box_weights` only to positive points matched to replay-weighted GTs
- Under DDP, each rank builds the same replay schedule from the synchronized MDMB++ state, then
  consumes a disjoint padded slice of global batch numbers so every rank runs the same number of
  optimizer steps.

### TFM - Temporal Failure Memory (`modules/nn/tfm.py`)

Lightweight per-GT temporal memory for TAMR experiments. TFM is intended to keep the useful
longitudinal state from MDMB++ while avoiding post-step inference and dense candidate-summary
storage.

- Hooks: `start_epoch()`, `end_epoch()`
- Update API: `tfm.update(...)`
- Summary: `tfm.summary()`
- Config: `modules/cfg/tfm.yaml`
- Arch support: FCOS

Key state:

- `TFMRecord`: persistent per-GT state with miss streak, relapse count, last failure type, and risk
- `TFMSupportPrototype`: optional compact feature teacher for anti-relapse distillation
- `TFMRiskConfig`: bounded temporal risk scoring from miss/recovery/relapse history
- `TFMAssignmentBiasConfig`: optional risk-gated FCOS positive-point loss weighting

TFM is refreshed inside the normal FCOS training forward from assignment, classification,
localization, and centerness signals. It does not trigger the MDMB/MDMB++ post-step inference pass.
When `assignment_bias.enabled: true`, FCOS looks up each matched GT's previous TFM risk and
multiplies the matched positive point's classification, box, and centerness losses by bounded
component weights. Assignment radius expansion and backup positives are not implemented yet.

### RASD - Relapse-Aware Support Distillation (`modules/nn/rasd.py`)

Training-time support distillation for relapse GTs stored in `MDMB++`.

- Depends on MDMB++ with `store_support_feature: true`
- Hooks: `start_epoch()`, `end_epoch()`
- Planning API: `rasd.plan(mdmbpp, targets, image_shapes)`
- Training path: FCOS adds a `rasd` auxiliary loss after the base FCOS loss
- Summary: `rasd.summary()`
- Config: `modules/cfg/rasd.yaml`
- Arch support: FCOS

Current scope:

- FCOS only
- Training-only; inference, NMS, and score thresholds are unchanged
- Relapse entries with stored support features are selected from MDMB++
- Current GT features are pooled from FPN features with MultiScaleRoIAlign
- The v1 loss is support attraction: current GT feature is pulled toward its previous successful support feature
- When `confuser.enabled` is true, `cls_confusion` relapse targets also use wrong-class
  `MDMBPlusEntry.topk_candidates` as detached contrastive negatives

### FN-TDM - False-Negative Transition Direction Memory (`modules/nn/fntdm.py`)

Training-time direction memory for false-negative recovery transitions.

- Hooks: `start_epoch()`, `end_epoch()`
- Epoch-end mining hook: `FCOSWrapper.mine_fntdm_batch(...)`
- Training path: FCOS adds an `fntdm` auxiliary loss after the base FCOS loss
- Config: `modules/cfg/fntdm.yaml`
- Arch support: FCOS

Implemented V0 components:

- HTM: epoch-end full-train mining of `FN -> TP` GT transitions from post-processed predictions
- TDB: class-wise quality-gated top-K transition direction bank with `last`, `topk`, and `topk_age`
  retrieval variants
- TCS: current hard-GT selection from FCOS assigned positive classification/centerness confidence
- TAL: anchor-shift cosine auxiliary loss using GT ROIAlign features and a shared projection head

Current scope:

- FCOS only
- Training-only; inference, NMS, score thresholds, and FCOS assignment are unchanged
- GT identity prefers `annotation_ids` / `gt_ids` supplied by the COCO data loader
- HTM mining is intentionally expensive: one extra full train-set inference pass per mining epoch
  on rank 0 under DDP, followed by FN-TDM state synchronization

### DHM - Detection Hysteresis Memory (`modules/nn/dhm.py`)

Per-GT detection-state hysteresis memory for identifying unstable GTs without adding a new
auxiliary loss.

- Hooks: `start_epoch()`, `end_epoch()`
- Epoch-end mining hook: `FCOSWrapper.mine_dhm_batch(...)`
- Training path: optional FCOS positive-point reweighting of the existing classification, box, and
  centerness losses
- Config: `modules/cfg/dhm.yaml`
- Arch support: FCOS

Implemented V0 components:

- Full train-set epoch-end mining from post-processed FCOS predictions
- Per-GT state tracking for `TP`, `FN_BG`, `FN_CLS`, `FN_LOC`, and `FN_MISS`
- Hysteresis statistics: forgetting, recovery, failure-type switching, state changes, streaks, and
  bounded instability score
- Optional failure-type-aware loss weighting through `loss_weighting.enabled`
- Optional HLAE assignment expansion through `assignment_expansion.enabled`

Current scope:

- FCOS only
- Training-only; inference, NMS, and score thresholds are unchanged
- No auxiliary loss is introduced; DHM only scales the existing FCOS loss terms for matched positive
  points when loss weighting is enabled
- HLAE V0 preserves existing FCOS positive assignments and only converts a capped number of
  currently negative points into positives for GTs whose last DHM state is `FN_BG`
- GT identity prefers `annotation_ids` / `gt_ids`; box/class fallback matching is available but less
  stable when strong train-time resizing is used
- Mining is intentionally expensive: one extra full train-set inference pass per mining epoch on
  rank 0 under DDP, followed by DHM state synchronization

## Runtime Integration

FCOS currently wires the following path:

1. `registry.py` builds `mdmb`, `mdmbpp`, `rasd`, `tfm`, `fntdm`, and `dhm` from
   `modules/cfg/*.yaml` or per-run CLI config overrides.
2. FCOS forward computes the base detection loss, refreshes TFM when enabled, and optionally adds
   `rasd` and `fntdm`; DHM can reweight the base FCOS loss or apply HLAE assignment expansion when
   enabled.
3. FCOS applies replay-aware per-GT weights when Hard Replay metadata is present in the batch.
4. `FCOSWrapper.after_optimizer_step()` runs one no-grad post-step inference pass.
5. That pass refreshes `mdmb` and `mdmbpp` state, including MDMB++ support feature snapshots when enabled.
6. `engine.fit()` calls module epoch hooks, runs FN-TDM and DHM epoch-end mining when enabled, and
   refreshes Hard Replay from `model.mdmbpp`.

In multi-GPU DDP training, FCOS/MDMB++/RASD/TFM/FN-TDM/DHM training losses run independently on
each rank during the epoch. At the epoch boundary, MDMB, MDMB++, TFM, FN-TDM, and DHM `extra_state`
payloads are gathered, merged by image/GT identity on rank 0, and broadcast back to every rank.
FN-TDM HTM mining and DHM mining run on rank 0 in V0. This keeps the next epoch's Hard Replay
planner, RASD teacher lookup, TFM risk lookup, FN-TDM direction bank, and DHM instability lookup
based on the same global memory while avoiding per-step memory synchronization.

Train metadata also snapshots active research-module YAML into
`{output_dir}/metadata/modules.yaml`. The file includes only modules that are effectively enabled
after architecture-specific overrides are applied.

## Compatibility

| Module | FCOS | Faster R-CNN | DINO |
|---|:---:|:---:|:---:|
| MDMB | yes | no | no |
| MDMB++ | yes | no | no |
| Hard Replay | yes | no | no |
| RASD | yes | no | no |
| TFM | yes | no | no |
| FN-TDM | yes | no | no |
| DHM | yes | no | no |
