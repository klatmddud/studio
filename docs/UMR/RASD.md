# RASD - Relapse-Aware Support Distillation

## Summary

RASD is a training-only auxiliary module for FCOS. It uses MDMB++ support snapshots as temporal
teachers for GTs that were detected before but later relapsed into missed detections.

The first implemented version is attraction-only:

```text
loss_rasd = lambda_rasd * warmup * mean(weight_g * (1 - cosine(current_gt_feature, support_feature_g)))
```

RASD does not change inference, NMS, score thresholds, target assignment, or detector outputs.

## Motivation

Hard Replay gives unresolved GTs more exposure, but exposure alone does not preserve the
object-specific representation that previously made the GT detectable. MDMB++ already records a
successful support state for each GT when it is detected. RASD turns that stored support feature into
a temporal teacher when the same GT later relapses.

This makes the module selective:

- It only acts on MDMB++ entries that are currently failures.
- It requires `entry.relapse == true`.
- It requires `SupportSnapshot.feature` to exist.
- It pulls the current GT feature toward the last successful support feature.

If a GT has never been detected, MDMB++ cannot create a support teacher for it, so RASD skips that
GT. Hard Replay can still sample the image/object, but RASD needs at least one previous successful
detection to provide a support feature.

## Target Selection

`RelapseAwareSupportDistillation.plan(...)` reads `mdmbpp.get_image_entries(image_id)` and matches
entries to the current transformed targets by class and normalized box IoU.

A target is selected only when all required filters pass:

- `entry.relapse == true`
- `entry.support is not None`
- `entry.support.feature is not None`
- `entry.consecutive_miss_count >= min_relapse_streak`
- `entry.severity >= min_severity`
- `entry.failure_type` is in `failure_types`
- `support.score >= min_support_score`
- `current_epoch - support.epoch <= max_support_age`
- the MDMB++ entry matches a current GT by `record_match_threshold`

Target weight:

```text
weight_g = min(
  max_target_weight,
  1 + severity_weight_scale * severity + relapse_weight_scale * relapse_count
)
```

## Feature Path

MDMB++ support features are produced after `optimizer.step()`:

1. `FCOSWrapper.after_optimizer_step()` calls `MDMBFCOS.flush_post_step_updates(...)`.
2. FCOS runs a no-grad inference pass and also exposes backbone FPN features.
3. If `mdmbpp.config.store_support_feature` is true, transformed GT boxes are pooled from FPN
   features using MultiScaleRoIAlign.
4. Pooled features are GAP-reduced, L2-normalized, moved to CPU, and passed to
   `MDMBPlus.update(..., support_feature_list=...)`.
5. When a GT is detected, MDMB++ stores the feature in `SupportSnapshot.feature`.

During training, RASD pools current GT features from the same FPN feature maps, normalizes them, and
computes the cosine attraction loss against the stored support feature.

## Config

Default config: `modules/cfg/rasd.yaml`

Key fields:

```yaml
enabled: false
warmup_epochs: 5
lambda_rasd: 0.03
min_relapse_streak: 1
min_severity: 1.0
min_support_score: 0.2
max_support_age: 15
max_targets_per_batch: 16

failure_types:
  - cls_confusion
  - score_suppression
  - nms_suppression
  - loc_near_miss

confuser:
  enabled: false
```

RASD has two fail-fast requirements when enabled:

- MDMB++ must also be enabled.
- `modules/cfg/mdmbpp.yaml` must set `store_support_feature: true`.

## Runtime Integration

Current FCOS training order:

```text
transform images/targets
backbone + FCOS head
base FCOS target matching
base FCOS loss, optionally replay-weighted
optional RASD loss
```

The loss key is `losses["rasd"]`. Engine history stores RASD summary under `record["rasd"]`.

## Summary Metrics

`rasd.summary()` reports:

- `targets`
- `losses`
- `relapse_targets`
- `mean_severity`
- `mean_support_age`
- `mean_target_weight`
- `mean_support_loss`
- `skipped_no_support`
- `skipped_support_too_old`
- `skipped_low_support_score`
- `skipped_no_entry_match`
- `skipped_no_feature`

The config also reserves confuser-related fields, but contrastive loss is disabled by default in the
current version.

## Recommended Ablation

Use the following sequence:

1. Baseline
2. Baseline + MDMB++
3. MDMB++ + RASD
4. MDMB++ + Hard Replay
5. MDMB++ + Hard Replay + RASD

This isolates whether RASD improves temporal relapse recovery beyond memory tracking alone and
whether it adds value on top of replay exposure.
