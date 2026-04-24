# Temporal Assignment Bias

## Hypothesis

Some GTs repeatedly fail because the detector does not assign enough useful positive training
locations to them, or because the assigned positives are too weak to dominate the loss. Static or
current-batch assignment does not know that a GT has repeatedly relapsed.

Temporal Assignment Bias uses the GT's historical risk to slightly bias assignment and weighting
during training.

## Method

For each GT, read `risk(gt)` from Lightweight Temporal Failure Memory. Use that risk to adapt one or
more training knobs:

```text
assignment_radius(gt) = base_radius * (1 + alpha * risk(gt))
positive_topk(gt)    = base_topk + round(beta * risk(gt))
loss_weight(gt)      = 1 + gamma * risk(gt)
```

Start with loss reweighting because it is least invasive. Assignment expansion should be added only
after verifying that weights alone are insufficient.

## FCOS-Oriented Variant

For FCOS, apply risk to the existing positive-location path:

- increase classification weight for positives assigned to high-risk GTs
- increase regression and centerness weight for high-risk GTs
- optionally expand center sampling radius for high-risk GTs
- optionally allow a small number of near-center backup positives

The backup positives should be capped to avoid flooding the batch with ambiguous samples.

## Loss Form

Let `G+` be positive locations assigned to a GT:

```text
L_tabi = sum_{g in GT} (1 + gamma * risk_g) * L_det(G+_g)
```

Use separate caps for classification and localization:

```text
w_cls = clamp(1 + gamma_cls * risk, 1, w_cls_max)
w_box = clamp(1 + gamma_box * risk, 1, w_box_max)
```

Reasonable initial caps:

```text
w_cls_max = 2.0
w_box_max = 2.0
```

## Novelty Angle

This differs from Focal Loss and OHEM because the weight is not only a function of current loss or
confidence. It is conditioned on per-instance temporal history: repeated miss, recovery, and
relapse.

It differs from ATSS/OTA/PAA because assignment is not recalculated only from the current forward
statistics. Historical failure risk acts as a prior.

## Metrics

Track:

- false-negative recovery rate
- relapse count
- mean miss streak
- number of high-risk GTs
- positive locations per high-risk GT
- AP_small and per-class AP
- training time delta

## Risks

- Overweighting can amplify bad annotations.
- Expanding positives can harm localization if the radius is too large.
- If risk decays too slowly, the model may keep over-focusing already recovered GTs.

Mitigation:

- clip risk and weights
- decay risk after recovery
- require repeated failures before applying assignment expansion

