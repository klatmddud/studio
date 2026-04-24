# Lightweight Temporal Failure Memory

## Goal

Replace the heavy MDMB++ storage path with a compact per-GT temporal state that can drive
training-only regularization.

The current MDMB++ design stores rich candidate context, including `best_candidate`,
`topk_candidates`, and support snapshots. That is useful for analysis, but expensive because FCOS
must build dense candidate summaries after the optimizer step. TAMR should avoid that path.

## Stored State

Store one record per stable GT identity:

```text
gt_uid
image_id
class_id
bbox_norm
first_seen_epoch
last_seen_epoch
last_state
miss_streak
max_miss_streak
total_miss
relapse_count
last_detected_epoch
last_failure_epoch
last_failure_type
risk
support_proto optional
support_quality optional
support_epoch optional
```

The state intentionally excludes dense candidates and full ROI features.

## GT Identity

Prefer dataset annotation IDs when available. If annotation IDs are unavailable, use the current
MDMB++ style fallback:

```text
image_id + class_id + normalized bbox hash
```

The fallback is acceptable for static COCO-style datasets, but annotation IDs are more stable when
augmentation or box transforms become more complex.

## Update Signal

The memory should be refreshed from signals already available in the normal training forward:

- assigned positive locations per GT
- classification target and prediction at those locations
- regression quality or IoU for assigned boxes
- centerness or quality score if available
- current training loss terms

Avoid running a second no-grad inference pass. The update can classify each GT into coarse states:

```text
detected_like
weak_positive
classification_confusion
localization_weak
missing_assignment
```

These states do not need to exactly match final post-NMS detection outcomes. TAMR needs a useful
training prior, not a perfect evaluator.

## Risk Score

Use a bounded scalar risk so it can safely condition assignment and loss weights:

```text
risk = sigmoid(
    a * normalized_miss_streak
  + b * log1p(total_miss)
  + c * relapse_count
  + d * failure_type_prior
  - e * recent_recovery
)
```

Recommended initial values:

```text
a = 1.0
b = 0.5
c = 1.0
d = 0.5
e = 1.0
```

`recent_recovery` can decay with epoch distance from `last_detected_epoch`.

## Support Prototype

If prototype distillation is enabled, store a compact vector rather than a full 7x7 ROI tensor:

```text
support_proto: float16 or float32 vector with shape [C] or [C_reduced]
support_quality: scalar
support_epoch: int
```

The prototype can be an EMA of pooled positive-location features:

```text
proto <- normalize(momentum * proto + (1 - momentum) * current_proto)
```

Only update the prototype when current quality is high enough.

## Expected Overhead

The target overhead should be close to:

- one small dictionary update per GT
- optional feature pooling over already selected positive locations
- no post-step inference
- no dense candidate summary
- no NMS inside the memory path

This should be far cheaper than current MDMB++.

## Failure Modes

- Training-forward states may not perfectly predict final detection failures.
- Risk can over-focus noisy labels if not clipped.
- Bbox-hash identity can become unstable under aggressive data transforms.
- Storing prototypes for every GT may still grow memory on large datasets; use optional top-risk
  retention if needed.

