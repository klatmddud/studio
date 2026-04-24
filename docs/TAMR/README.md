# TAMR - Temporal Anti-Miss Regularization

TAMR is a candidate research direction for reducing recurrent false negatives in object detection
without adding inference-time cost. It keeps the useful part of MDMB++ - per-GT temporal memory -
but avoids the expensive post-step inference and dense candidate-summary storage that currently
make MDMB++ slow.

## Motivation

The current UMR stack shows that Hard Replay is the strongest practical axis, while MDMB++ and RASD
add useful diagnostics and temporal support signals but carry a large training-time overhead.
MDMB++ is especially expensive because it performs an extra post-step inference pass and scans dense
FCOS candidates to build structured failure context.

TAMR reframes the memory module as a lightweight training prior:

- keep only compact per-GT temporal state
- read signals already produced by the normal training forward
- apply auxiliary losses only during training
- remove all branches at inference

## Design Constraints

- No inference-time overhead.
- Low training overhead: no extra detector forward pass, no dense candidate summary pass.
- Per-GT state must be compact enough for checkpointing and DDP synchronization.
- The method should stand independently from Hard Replay, but remain compatible with it.
- Novelty should come from temporal failure state, not only from hard-sample reweighting.

## Proposed Components

TAMR is best treated as a family of methods. Each component can be ablated independently.

1. Lightweight Temporal Failure Memory
   - Stores compact per-GT miss/recovery/relapse state and optional support prototypes.
   - Replaces MDMB++ dense `topk_candidates` with scalar risk state.

2. Temporal Assignment Bias
   - Uses per-GT temporal risk to adapt positive assignment or loss weight for historically hard
     GTs.
   - Targets repeated misses without replaying the whole image distribution.

3. Anti-Relapse Prototype Distillation
   - Stores a compact feature prototype from a past successful detection.
   - Pulls current positive GT features toward the stable historical prototype.

4. Failure-Type Conditional Margin
   - Uses the remembered failure type to choose a targeted correction loss.
   - Different failures receive different supervision instead of one generic hard weight.

## Positioning Against Related Work

Existing hard-mining and assignment methods mostly operate on the current mini-batch or current
forward pass:

- OHEM mines hard examples online.
- Focal Loss down-weights easy examples.
- Libra R-CNN and PISA rebalance sample/objective importance.
- ATSS, OTA, and PAA adapt assignment based on current statistics or costs.
- GFL improves quality and localization representation for dense detection.

TAMR's intended novelty is longitudinal: each GT carries a training history, and that history
conditions assignment, weighting, margin, and feature regularization in later epochs.

## Suggested Ablation Ladder

1. Baseline detector.
2. Baseline + Lightweight Temporal Failure Memory only.
3. Add Temporal Assignment Bias.
4. Add Anti-Relapse Prototype Distillation.
5. Add Failure-Type Conditional Margin.
6. Combine TAMR with Hard Replay.

The key comparison is not only final AP. Track false-negative recovery, relapse count,
per-class miss streak, AP_small/AP_medium/AP_large, training time, and checkpoint state size.

## Files

- [01_lightweight_temporal_failure_memory.md](01_lightweight_temporal_failure_memory.md)
- [02_temporal_assignment_bias.md](02_temporal_assignment_bias.md)
- [03_anti_relapse_prototype_distillation.md](03_anti_relapse_prototype_distillation.md)
- [04_failure_type_conditional_margin.md](04_failure_type_conditional_margin.md)
- [05_experiment_plan.md](05_experiment_plan.md)

