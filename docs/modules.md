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
- Hard replay: `train.hard_replay` can reuse DHM records to duplicate current-batch images containing persistent FN or relapse GTs, bounded by a configured replay ratio.
- Summary: `dhm.summary()` is logged under the `dhm` key in `history.json` with `transition_matrix`, `assignment_by_state`, and `assignment_by_transition` aggregates.
- Config: `modules/cfg/dhm.yaml`.

## DHM-R - Detection Hysteresis Memory Repair (`modules/nn/dhmr.py`)

DHM-R currently implements the DHM-guided border-aware residual refinement path for
FCOS localization repair.

Key concepts:

- Depends on DHM because it needs previous per-GT DHM records.
- `border_refinement` is a training-only Phase 1 implementation. It selects dense FCOS positive points for `FN_LOC->FN_LOC` and `TP->FN_LOC` GT transitions, samples center/border FPN features, and adds residual box, GIoU, and IoU-quality auxiliary losses.
- `border_refinement` logs `dhmr_border_giou`, `dhmr_border_residual`, and `dhmr_border_quality` in the training loss dict when active, plus selected-point and mean-loss summaries under `dhmr.border_refinement`.
- Uses FCOS logits, anchor points, matched GT indices, and DHM records.
- Summary: `dhmr.summary()` is logged under the `dhmr` key in `history.json`.
- Config: `modules/cfg/dhmr.yaml`.

## Runtime Flow

1. `registry.py` builds enabled DHM/DHM-R modules for FCOS.
2. `engine.train_one_epoch()` can append hard replay images selected from current-batch DHM records when `train.hard_replay.enabled` is true.
3. FCOS forward computes the base detection loss and, when DHM records exist, logs compact per-GT assignment statistics.
4. If DHM-R `border_refinement` is enabled, FCOS adds training-only border residual and IoU-quality losses for dense positive points matched to DHM transition targets.
5. `engine.fit()` runs DHM epoch-end mining when `dhm.mining.enabled` and interval/warmup conditions allow it.
6. Under DDP, DHM and DHM-R `extra_state` payloads are merged and synchronized once per epoch.

## Support Matrix

| Module | FCOS | Faster R-CNN | DINO |
|---|---:|---:|---:|
| DHM | yes | no | no |
| DHM-R | yes | no | no |
