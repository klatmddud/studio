# Research Modules

Research features are configured from `modules/cfg/*.yaml`.
`mdmb`, `mdmbpp`, `candidate_densification`, `faar`, `fang`, and `marc` live in `modules/nn/`.
`hard_replay` is configured in the same folder but implemented in `scripts/runtime/hard_replay.py`
because it operates at the data-loading layer.

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

### Hard Replay (`scripts/runtime/hard_replay.py`)

Data-layer replay that redistributes training exposure toward images whose GTs remain unresolved in
`MDMB++`.

- Planner: `HardReplayPlanner.build_epoch_index(...)`
- Sampler: `MixedReplayBatchSampler`
- Controller: `HardReplayController`
- Config: `modules/cfg/hard_replay.yaml`
- Arch support: data-layer replay is model-agnostic; replay-aware loss weighting is FCOS-only

Current scope is the minimal first version:

- Epoch-level `ReplayIndex`
- Image-level replay
- Object-level crop replay through `object_replay.crop`
- Rectangular copy-paste replay through `object_replay.copy_paste`
- Support/miss pair replay through `object_replay.pair`
- FCDR crop replay when `fcdr.enabled: true`
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
- FCDR is retained as a failure-conditioned policy preset over the object replay path
- FCOS applies `replay_box_weights` only to positive points matched to replay-weighted GTs

### Candidate Densification (`modules/nn/candidate_densification.py`)

Training-time auxiliary positive point densification for hard GTs stored in `MDMB++`.

- Depends on MDMB++
- Hooks: `start_epoch()`, `end_epoch()`
- Planning API: `candidate_densifier.plan(mdmbpp, targets, image_shapes)`
- Training path: FCOS adds a `candidate_dense` auxiliary loss when dense points are selected
- Summary: `candidate_densifier.summary()`
- Config: `modules/cfg/candidate_densification.yaml`
- Arch support: FCOS

Current scope is the minimal first version:

- FCOS only
- Base FCOS assignment is unchanged
- Dense positives are selected from points near hard GT centers
- Default behavior uses only points that were background under the base FCOS assignment
- Loss weight uses linear warmup through `lambda_dense`

### FAAR - Failure-Aware Assignment Repair (`modules/nn/faar.py`)

Training-time assignment repair for hard GTs stored in `MDMB++`.

- Depends on MDMB++
- Hooks: `start_epoch()`, `end_epoch()`
- Planning API: `faar.plan(mdmbpp, targets, image_shapes)`
- Training path: FCOS repairs `matched_idxs` after the base assignment and before loss computation
- Summary: `faar.summary()`
- Config: `modules/cfg/faar.yaml`
- Arch support: FCOS

Current scope is the minimal first version:

- FCOS only
- No auxiliary loss is added
- Repair targets are selected by MDMB++ `failure_type`, `severity`, and relapse state
- Default behavior only converts unassigned FCOS points into positives for the hard GT
- Repaired FCOS points must keep their centers inside the target GT box so bbox regression and centerness targets stay finite
- Existing positive assignments are not stolen unless `allow_positive_reassignment: true`
- If Candidate Densification is also enabled, FAAR runs first so dense supervision sees the repaired assignment

### MARC - Miss-Aware Ranking Calibration (`modules/nn/marc.py`)

Training-time ranking calibration for hard GTs stored in `MDMB++`.

- Depends on MDMB++
- Hooks: `start_epoch()`, `end_epoch()`
- Planning API: `marc.plan(mdmbpp, targets, image_shapes)`
- Training path: FCOS adds a `marc` auxiliary ranking loss after FAAR and before Candidate Densification
- Summary: `marc.summary()`
- Config: `modules/cfg/marc.yaml`
- Arch support: FCOS

Current scope is the minimal first version:

- FCOS only
- No assignment, data sampling, or inference behavior is changed
- Ranking targets are selected from MDMB++ `score_suppression`, `nms_suppression`, `cls_confusion`, and `loc_near_miss`
- `candidate_missing` is excluded by default because there may be no meaningful candidate to rank
- Positive candidates use GT-class score and IoU; negatives use wrong-class confusers, same-class suppressors, and high-score local distractors

### FANG - Failure-Aware Negative Gradient Shielding (`modules/nn/fang.py`)

Training-time negative-gradient shielding for hard GTs stored in `MDMB++`.

- Depends on MDMB++
- Hooks: `start_epoch()`, `end_epoch()`
- Planning API: `fang.plan(mdmbpp, targets, image_shapes)`
- Weighting API: `fang.compute_class_weights(...)`
- Training path: FCOS lowers selected class-wise negative focal-loss terms after FAAR and before loss aggregation
- Summary: `fang.summary()`
- Config: `modules/cfg/fang.yaml`
- Arch support: FCOS

Current scope is the minimal first version:

- FCOS only
- No auxiliary loss is added
- Only `matched_idxs < 0` negative points are eligible
- Only the hard GT true-class column is shielded; other class losses, bbox regression, and centerness are unchanged
- Shield targets are selected by MDMB++ `failure_type` and `severity`
- Overlapping shield targets use the lowest class weight for the same point/class pair

## Runtime Integration

FCOS currently wires the following path:

1. `registry.py` builds `mdmb`, `mdmbpp`, `candidate_densifier`, `faar`, `fang`, and `marc` from `modules/cfg/*.yaml`.
2. FCOS forward reads `model.mdmbpp` through FAAR, FANG, MARC, and Candidate Densification when enabled.
3. FAAR repairs `matched_idxs`; FANG can shield class-wise negative focal terms; MARC can add `marc` ranking loss; Candidate Densification can add `candidate_dense` auxiliary loss.
4. `FCOSWrapper.after_optimizer_step()` runs one no-grad post-step inference pass.
5. That pass refreshes `mdmb` and `mdmbpp` state.
6. `engine.fit()` calls module epoch hooks and refreshes Hard Replay from `model.mdmbpp`.

## Compatibility

| Module | FCOS | Faster R-CNN | DINO |
|---|:---:|:---:|:---:|
| MDMB | yes | no | no |
| MDMB++ | yes | no | no |
| Hard Replay | yes | no | no |
| Candidate Densification | yes | no | no |
| FAAR | yes | no | no |
| FANG | yes | no | no |
| MARC | yes | no | no |
