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
- Records: `DHMRecord` stores last state, FN streaks, recovery/forgetting counts, dominant failure type, transition counts, instability score, annotation ID, and compact FCOS assignment statistics.
- Assignment statistics: FCOS logs per-GT positive counts, FPN level histograms, centerness/loss means, near-candidate/near-negative counts, and ambiguous points assigned to other GTs.
- Hard replay: `train.hard_replay` can reuse DHM records to build a replay index for images containing persistent FN or relapse GTs, then mix replay samples into DataLoader batches. `loc_repair` further converts selected `FN_LOC` records into GT-centered crop replay samples.
- Summary: `dhm.summary()` is logged under the `dhm` key in `history.json` with `transition_matrix`, `assignment_by_state`, and `assignment_by_transition` aggregates.
- Config: `modules/cfg/dhm.yaml`.

## DHM-R - Detection Hysteresis Memory Repair (`modules/nn/dhmr.py`)

DHM-R currently implements DHM-guided training-only localization repair paths for FCOS.

Key concepts:

- Depends on DHM because it needs previous per-GT DHM records.
- `border_refinement` is a training-only Phase 1 implementation. It selects dense FCOS positive points for `FN_LOC->FN_LOC` and `TP->FN_LOC` GT transitions, samples center/border FPN features, and adds residual box, GIoU, and IoU-quality auxiliary losses.
- `border_refinement` logs `dhmr_border_giou`, `dhmr_border_residual`, and `dhmr_border_quality` in the training loss dict when active, plus selected-point and mean-loss summaries under `dhmr.border_refinement`.
- `counterfactual_repair` is a training-only DCLR Phase 1 implementation. It uses the same DHM transition targets, selects dense positive points whose initial IoU remains below `tau_iou + tau_margin`, and adds residual, GIoU, IoU-quality, and threshold-crossing losses to optimize `FN_LOC -> TP` repair.
- `counterfactual_repair` logs `dhmr_dclr_giou`, `dhmr_dclr_residual`, `dhmr_dclr_crossing`, and `dhmr_dclr_quality` in the training loss dict when active, plus selected-point, initial-IoU, refined-IoU, and mean-loss summaries under `dhmr.counterfactual_repair`.
- Uses FCOS logits, anchor points, matched GT indices, and DHM records.
- Summary: `dhmr.summary()` is logged under the `dhmr` key in `history.json`.
- Config: `modules/cfg/dhmr.yaml`.

## Runtime Flow

1. `registry.py` builds enabled DHM/DHM-R modules for FCOS.
2. `engine.fit()` refreshes the Hard Replay index from DHM at epoch start when `train.hard_replay.enabled` is true.
3. The train DataLoader's mixed replay batch sampler yields normal base samples plus DHM-hard full-image replay and optional `FN_LOC` localization-repair crop replay.
4. FCOS forward computes the base detection loss and, when DHM records exist, logs compact per-GT assignment statistics for full-image targets. Crop replay targets are skipped for DHM memory logging so full-image mining remains the source of DHM state.
5. If DHM-R `border_refinement` or `counterfactual_repair` is enabled, FCOS adds training-only localization repair losses for dense positive points matched to DHM transition targets.
6. `engine.fit()` runs DHM epoch-end mining when `dhm.mining.enabled` and interval/warmup conditions allow it.
7. Under DDP, DHM and DHM-R `extra_state` payloads are merged and synchronized once per epoch.

## Support Matrix

| Module | FCOS | Faster R-CNN | DINO |
|---|---:|---:|---:|
| DHM | yes | no | no |
| DHM-R | yes | no | no |
