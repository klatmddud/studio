# Anti-Relapse Prototype Distillation

## Hypothesis

A GT that was once detected and later missed has lost a useful object representation. A compact
historical support prototype can act as a temporal teacher and stabilize the current representation
for that same GT.

The goal is to prevent relapse without using a second model, a teacher detector, or inference-time
branches.

## Stored Teacher

Store one compact prototype per GT:

```text
support_proto
support_quality
support_epoch
support_level optional
```

The prototype should come from features already used in training. For FCOS, this can be a pooled or
averaged feature over positive locations assigned to the GT.

Avoid full ROIAlign tensors unless needed. A `[C]` vector is the preferred first version.

## Prototype Update

Update the teacher only when the current GT state is reliable:

```text
if current_quality >= support_quality + margin:
    support_proto = current_proto
elif support_age >= refresh_age and current_quality >= min_quality:
    support_proto = ema(support_proto, current_proto)
```

Quality can be approximated from training-forward signals:

```text
quality = q_cls * cls_confidence + q_iou * regression_iou + q_ctr * centerness
```

This avoids requiring post-NMS final detections.

## Distillation Loss

Apply only to GTs that have a support prototype and non-trivial temporal risk:

```text
L_proto = risk(gt) * (1 - cosine(current_proto, stopgrad(support_proto)))
```

Optional temperature-normalized version:

```text
L_proto = risk(gt) * || normalize(P(current_proto)) - stopgrad(normalize(support_proto)) ||_2^2
```

`P` can be a tiny projection head used only during training.

## When to Apply

Apply to:

- relapsed GTs
- GTs with high miss streak and an existing support prototype
- recently recovered GTs for a short stabilization window

Skip:

- GTs without support prototypes
- low-quality current positive features
- ambiguous assignments

## Novelty Angle

This is not generic self-distillation. The teacher is not another model or a larger model. It is a
per-GT temporal support prototype from the same detector's past successful state. The loss is
activated by failure history, not uniformly over all objects.

Compared with the current RASD idea, this version should be cheaper because it avoids post-step
feature extraction and full support tensors.

## Expected Overhead

Low:

- feature aggregation over assigned positives
- one cosine loss per selected GT
- optional small projection head during training
- compact prototype storage

No inference overhead.

## Risks

- Old prototypes can become stale as the backbone changes.
- A bad early prototype can anchor the model to a weak representation.
- Class-level collapse is possible if prototypes are too generic.

Mitigation:

- quality-gated updates
- max prototype age
- warmup before storing prototypes
- risk-gated loss activation
- optional class-specific normalization

