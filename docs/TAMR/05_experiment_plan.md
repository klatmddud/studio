# TAMR Experiment Plan

## Primary Questions

1. Can temporal GT history improve detection without Hard Replay?
2. Can TAMR reduce relapse and repeated miss events with less overhead than MDMB++?
3. Which TAMR component contributes most: risk weighting, assignment bias, prototype
   distillation, or failure-type margin?
4. Does TAMR combine additively with Hard Replay?

## Baselines

Run these in order:

```text
Baseline
Baseline + current MDMB++ only
Baseline + Hard Replay only
Baseline + TAMR memory only
Baseline + TAMR assignment bias
Baseline + TAMR assignment bias + prototype distillation
Baseline + full TAMR
Baseline + Hard Replay + best TAMR variant
```

## Required Metrics

Detection:

```text
mAP
AP50
AP75
AP_small
AP_medium
AP_large
per-class AP
```

Temporal failure:

```text
num_high_risk_gt
mean_risk
mean_miss_streak
max_miss_streak
total_relapse
relapse_this_epoch
recovery_rate_last_epoch
failure_type_counts
failure_type_transition_matrix
```

Efficiency:

```text
train_seconds_per_epoch
images_per_second
GPU memory peak
checkpoint_size
memory_state_size
DDP sync time if distributed
```

## Minimal First Implementation

Start with the cheapest path:

1. Add lightweight per-GT state.
2. Compute GT risk from current training-forward statistics.
3. Apply risk-gated loss reweighting only.
4. Log temporal metrics.

Do not implement prototypes or conditional margins until the risk signal is proven useful.

## Acceptance Criteria

A TAMR variant is worth keeping if it satisfies:

```text
training_time <= baseline_time * 1.15
inference_time == baseline_time
checkpoint_size increase is acceptable
relapse count decreases
mean miss streak decreases
mAP or AP_small improves
```

For research value, prioritize a clear temporal-metric improvement even if mAP gains are modest.
The paper story can be built around reducing recurrent false negatives, not only aggregate AP.

## Negative Results to Watch

- mAP improves only when combined with Hard Replay.
- risk weighting improves recall but hurts AP75.
- prototype loss helps small objects but hurts class confusion.
- failure-type margins are too noisy without post-NMS diagnosis.

These outcomes are still useful because they define which parts of the temporal memory are actually
valuable.

