# Failure-Type Conditional Margin

## Hypothesis

Not all misses are the same. A classification confusion, localization near miss, score suppression,
and assignment miss should not receive the same correction signal.

Failure-Type Conditional Margin uses the remembered failure type to choose a targeted auxiliary
loss for each high-risk GT.

## Failure Types

Use a lightweight taxonomy aligned with MDMB++ but computed from training-forward signals:

```text
missing_assignment
classification_confusion
localization_weak
score_weak
quality_ranking_weak
recovered
```

If exact diagnosis is unavailable, store the best available coarse type. The method does not depend
on perfect post-NMS attribution.

## Conditional Corrections

### Classification Confusion

Increase the margin between the GT class and the strongest competing class:

```text
L_cls_margin = risk * max(0, m_cls - logit_gt + logit_confuser)
```

Use only positives assigned to the GT.

### Localization Weak

Increase localization pressure on positives for that GT:

```text
L_loc_margin = risk * max(0, m_iou - IoU(pred_box, gt_box))
```

This should be capped and applied only to reasonable positive locations.

### Score Weak

Raise the GT class confidence floor:

```text
L_score_margin = risk * max(0, m_score - p_gt)
```

This is useful when localization is acceptable but confidence remains too low.

### Quality Ranking Weak

Encourage the best GT positive to outrank nearby lower-quality positives:

```text
L_rank = risk * max(0, m_rank - score_best_gt + score_neighbor)
```

This should be used carefully because dense detectors already rely on many correlated positives.

### Missing Assignment

Do not add a margin first. Assignment expansion or backup positives are safer:

```text
center_radius <- center_radius * (1 + alpha * risk)
```

Then apply standard detection losses to the newly admitted positives.

## Loss Composition

Use one conditional loss per GT per step:

```text
L_ftcm = sum_g risk_g * L_condition(last_failure_type_g)
```

Keep the weight small initially:

```text
lambda_ftcm = 0.05 to 0.2
```

## Novelty Angle

This moves beyond hard-example weighting. The model remembers the failure mode and applies a
different corrective pressure later. This is closer to temporal diagnosis-driven training than
generic mining.

## Metrics

Track failure-type transition matrices:

```text
classification_confusion -> recovered
localization_weak -> recovered
score_weak -> recovered
missing_assignment -> recovered
recovered -> relapse
```

These transitions are more diagnostic than AP alone.

## Risks

- Incorrect failure typing can apply the wrong margin.
- Multiple auxiliary losses can destabilize training.
- Margin losses can hurt calibration if over-weighted.

Mitigation:

- use one dominant failure type per GT
- warm up after base detector stabilizes
- clip risk and auxiliary loss
- ablate each failure type correction independently

