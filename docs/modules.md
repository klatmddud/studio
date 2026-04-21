# Research Modules

Research features are configured from `modules/cfg/*.yaml`.
`mdmb`, `mdmbpp`, `recall`, `far`, and `mce` live in `modules/nn/`.
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
- Arch support: FCOS

Current scope is the minimal first version:

- Epoch-level `ReplayIndex`
- Image-level replay only
- Mixed batch composition

Current implementation does **not** enable:

- Crop replay
- Copy-paste replay
- Pair replay

Important behavior:

- Replay weight uses `1 + beta * sum(severity)` per image, clipped by `max_image_weight`
- Sampling weight applies `temperature` as an exponent
- Replay candidates are filtered by `replay_recency_window`
- Per-image replay repeats are capped by `max_replays_per_gt_per_epoch`

### RECALL - Selective Loss Reweighting (`modules/nn/recall.py`)

Uses MDMB observations to upweight losses on hard GTs.

- Depends on MDMB
- Config: `modules/cfg/recall.yaml`
- Arch support: FCOS

### FAR - Forgetting-Aware Replay (`modules/nn/far.py`)

Applies a feature-level consistency loss toward a frozen anchor captured when a GT was last
detected successfully.

- Depends on MDMB
- Hooks: `start_epoch()`, `end_epoch()`
- Training loss: `far.compute_loss(...)`
- Summary: `far.summary()`
- Config: `modules/cfg/far.yaml`
- Arch support: FCOS

### MCE - Miss-Conditioned class Embedding (`modules/nn/mce.py`)

Uses a learnable class prototype embedding to amplify loss on GTs that remain hard under MDMB
streak statistics.

- Depends on MDMB
- Training path: integrated into FCOS loss computation
- Config: `modules/cfg/mce.yaml`
- Arch support: FCOS

## Runtime Integration

FCOS currently wires the following path:

1. `registry.py` builds `mdmb`, `mdmbpp`, `recall`, `far`, and `mce` from `modules/cfg/*.yaml`.
2. `FCOSWrapper.after_optimizer_step()` runs one no-grad post-step inference pass.
3. That pass refreshes `mdmb`, `mdmbpp`, and FAR state.
4. `engine.fit()` calls module epoch hooks and refreshes Hard Replay from `model.mdmbpp`.

## Compatibility

| Module | FCOS | Faster R-CNN | DINO |
|---|:---:|:---:|:---:|
| MDMB | yes | no | no |
| MDMB++ | yes | no | no |
| Hard Replay | yes | no | no |
| RECALL | yes | no | no |
| FAR | yes | no | no |
| MCE | yes | no | no |
