# UMR Current Scope

UMR currently keeps a compact memory-and-replay stack:

1. MDMB: records missed detections over training time.
2. MDMB++: records structured GT failure state, support snapshots, candidate context, severity, and relapse state.
3. Hard Replay: increases exposure for unresolved MDMB++ GTs through image and object replay.
4. RASD: uses stored MDMB++ support features as temporal teachers for relapse GTs.

All modules are disabled by default through `modules/cfg/*.yaml`.

## Runtime Flow

```text
train batch
  -> FCOS forward
  -> base detection loss
  -> optional replay-aware per-GT loss weighting
  -> optional RASD support-distillation loss
  -> optimizer step
  -> post-step inference
  -> MDMB / MDMB++ memory refresh
  -> optional support feature snapshot storage
```

Hard Replay runs outside the model in the data-loading layer. At the start of each epoch,
`engine.fit()` refreshes the replay controller from `model.mdmbpp`; the mixed sampler then injects
replay samples into training batches.

## Experiment Ladder

Use this sequence when measuring the current stack:

1. Baseline
2. Baseline + MDMB++
3. MDMB++ + RASD
4. MDMB++ + Hard Replay
5. MDMB++ + Hard Replay + RASD

This separates memory tracking, temporal support distillation, data replay, and their combination.

## Notes

- RASD requires `mdmbpp.store_support_feature: true`.
- A GT that has never been successfully detected has no support teacher, so RASD skips it.
- Hard Replay can still operate on unresolved GTs without support teachers.
- Inference behavior is unchanged by the current UMR modules.
