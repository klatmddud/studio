# Research Modules

Current module support is limited to DHM and DHM-R. Both modules are disabled by default in `modules/cfg/*.yaml`.

## Config Resolution

`scripts/train.py` accepts:

- `--dhm-config`
- `--dhmr-config`

`scripts/runtime/module_configs.py` resolves those paths. `scripts/runtime/module_metadata.py` persists enabled module snapshots to `metadata/modules.yaml` for reproducibility.

## DHM - Detection Hysteresis Memory (`modules/nn/dhm.py`)

DHM stores per-GT detection-state history mined from full train-set inference at the end of selected epochs.

Key concepts:

- States: `TP`, `FN_BG`, `FN_CLS`, `FN_LOC`, `FN_MISS`.
- Mining: `DetectionHysteresisMemory.mine_batch(...)` compares detections against GT boxes and updates per-GT records.
- Records: `DHMRecord` stores last state, FN streaks, recovery/forgetting counts, dominant failure type, instability score, and annotation ID.
- Loss weighting: optional training-only weights for FCOS classification, box, and centerness loss terms.
- Assignment expansion: optional training-only positive-point expansion for eligible hard GTs.
- Summary: `dhm.summary()` is logged under the `dhm` key in `history.json`.
- Config: `modules/cfg/dhm.yaml`.

## DHM-R - Detection Hysteresis Memory Repair (`modules/nn/dhmr.py`)

DHM-R currently implements Temporal Edge Repair for localization false negatives.

Key concepts:

- Depends on DHM because it needs previous per-GT DHM records.
- Selects DHM records whose recent state matches `FN_LOC`.
- Adds a training-only `dhmr_edge` auxiliary loss in FCOS.
- Uses current FPN point features, FCOS logits, anchor points, matched GT indices, and DHM records.
- Maintains compact temporal edge records for state synchronization and summary logging.
- Summary: `dhmr.summary()` is logged under the `dhmr` key in `history.json`.
- Config: `modules/cfg/dhmr.yaml`.

## Runtime Flow

1. `registry.py` builds enabled DHM/DHM-R modules for FCOS.
2. FCOS forward computes the base detection loss.
3. If DHM loss weighting is enabled and DHM has records, FCOS reweights raw per-point losses.
4. If DHM assignment expansion is enabled and DHM has eligible records, FCOS adds extra positive points before loss computation.
5. If DHM-R is enabled, FCOS adds `dhmr_edge` for eligible DHM `FN_LOC` records.
6. `engine.fit()` runs DHM epoch-end mining when `dhm.mining.enabled` and interval/warmup conditions allow it.
7. Under DDP, DHM and DHM-R `extra_state` payloads are merged and synchronized once per epoch.

## Support Matrix

| Module | FCOS | Faster R-CNN | DINO |
|---|---:|---:|---:|
| DHM | yes | no | no |
| DHM-R | yes | no | no |
