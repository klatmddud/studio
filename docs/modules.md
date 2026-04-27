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

DHM-R implements HLRT (Hysteretic Localization Residual Transport) for DHM
`FN_LOC` records in FCOS.

Key concepts:

- Depends on DHM because it needs previous per-GT DHM records.
- Selects DHM records whose recent state matches `FN_LOC`.
- HLRT can independently enable residual memory, residual replay, native IoU loss weighting, side-aware boundary loss, and a centerness quality gate through `modules/cfg/dhmr.yaml`.
- HLRT residual replay adds extra FCOS positive points for eligible `FN_LOC` GTs without changing the image batch or dataset annotations.
- HLRT side-aware loss is logged as `dhmr_hlrt_side`.
- `typed_film` can learn one FiLM embedding each for `FN_LOC`, `FN_CLS`, and `FN_BG`; during training it modulates matched positive feature locations before the FCOS head while keeping the standard detection losses.
- `typed_film` logs `enabled`, `warmup_factor`, `selected_points`, `selected_gt`, and per-state selected-GT counts under `state_counts`.
- Uses FCOS logits, anchor points, matched GT indices, and DHM records.
- Maintains compact residual records for state synchronization and summary logging.
- Summary: `dhmr.summary()` is logged under the `dhmr` key in `history.json`.
- Config: `modules/cfg/dhmr.yaml`.

## Runtime Flow

1. `registry.py` builds enabled DHM/DHM-R modules for FCOS.
2. FCOS forward computes the base detection loss.
3. If DHM loss weighting is enabled and DHM has records, FCOS reweights raw per-point losses.
4. If DHM assignment expansion is enabled and DHM has eligible records, FCOS adds extra positive points before loss computation.
5. If DHM-R HLRT residual replay is enabled, FCOS adds replay positive points from temporal residual memory.
6. If DHM-R `typed_film` is enabled, FCOS applies training-only FiLM modulation to positive points matched to `FN_LOC`, `FN_CLS`, or `FN_BG` records before running the head.
7. If DHM-R HLRT native hooks are enabled, FCOS applies IoU loss weighting, optional centerness target gating, and optional `dhmr_hlrt_side`.
8. DHM-R updates HLRT residual memory from eligible `FN_LOC` positive points.
9. `engine.fit()` runs DHM epoch-end mining when `dhm.mining.enabled` and interval/warmup conditions allow it.
10. Under DDP, DHM and DHM-R `extra_state` payloads are merged and synchronized once per epoch.

## Support Matrix

| Module | FCOS | Faster R-CNN | DINO |
|---|---:|---:|---:|
| DHM | yes | no | no |
| DHM-R | yes | no | no |
