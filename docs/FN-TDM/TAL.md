# TAL - Transition Alignment Loss

## Goal

TAL is the auxiliary loss component of FN-TDM.

HTM discovers feature directions that accompanied historical `FN -> TP` recovery. TDB stores and
aggregates those directions. TCS selects current hard GT candidates. TAL uses the retrieved
directions to gently pull the current candidate embedding toward a recovery direction.

```text
selected hard GT g
current embedding z_g
TDB direction D_c
TAL encourages z_g to move toward D_c
```

TAL is training-only. It must not change detector inference.

## Position in FN-TDM

```text
HTM: mine historical FN -> TP directions
TDB: store/retrieve direction priors
TCS: select current hard candidates
TAL: apply direction alignment loss to selected candidates
```

Total training objective:

```text
L_total = L_det + lambda_tal * L_TAL
```

TAL should be a small auxiliary term, not a replacement for the base detection loss.

## Inputs

TAL consumes candidates from TCS:

```python
TCSCandidate = {
    "batch_index": int,
    "target_index": int,
    "class_id": int,
    "bbox": Tensor[4],
    "hardness": float,
    "direction": Tensor[D],
    "direction_source": str,
}
```

TAL also needs current feature maps from the detector:

```text
FPN feature maps or detector feature maps
```

The current candidate embedding is extracted from GT boxes:

```text
z_g = normalize(projector(pool(ROIAlign(F, bbox_g))))
```

## V0 Main Method: Anchor-Shift Alignment

The main V0 method is **Anchor-Shift Alignment**.

TAL receives one current embedding `z_g` and one historical transition direction `D_c`.
Because a single forward pass does not directly provide a current movement vector, TAL builds a
detached target by shifting the current embedding along the historical recovery direction.

```text
z_anchor = stopgrad(z_g)
z_target = stopgrad(normalize(z_anchor + alpha * D_c))
L_TAL(g) = 1 - cos(z_g, z_target)
```

Equivalent compact form:

```text
L_TAL(g) = 1 - cos(z_g, stopgrad(normalize(stopgrad(z_g) + alpha D_c)))
```

Where:

```text
z_g: current trainable GT embedding
D_c: detached transition direction from TDB
alpha: direction step size
```

Recommended defaults:

```text
alpha: 0.2
lambda_tal: 0.05
```

Search range:

```text
alpha: [0.1, 0.2, 0.5]
lambda_tal: [0.02, 0.05, 0.1, 0.2]
```

## Why Anchor-Shift

The ideal idea is to align the current feature movement with historical recovery directions.
However, the current batch gives only one embedding unless we store previous features for each GT.

Anchor-Shift is the simplest direction-based approximation:

```text
current feature z_g
historical recovery direction D_c
desired local target z_g + alpha * D_c
```

Benefits:

- Uses TDB directions directly.
- Does not require a per-GT previous-feature cache.
- Works with GT ROIAlign features even when detector assignment is missing.
- Keeps gradients only on current embeddings.
- Keeps TDB directions detached and stable.

Risks:

- The target is self-anchored, so the loss is local and relatively weak.
- If `alpha` is too large, the target may leave the local valid feature neighborhood.
- If applied to easy GTs, it can over-regularize. TCS should prevent this.

## Feature Extraction

Use the same projection space as HTM.

```text
FPN feature maps -> MultiScaleRoIAlign(gt_bbox) -> GAP -> projection head -> normalize
```

The projection head should be shared by HTM and TAL:

```text
HTM stores directions in projector space
TAL computes current embeddings in the same projector space
```

Recommended defaults:

```text
roi_output_size: 7
projector_dim: 256
normalize_embedding: true
```

If the detector has multiple FPN levels, use the same `MultiScaleRoIAlign` settings as HTM.

## Loss Aggregation

For selected candidates:

```text
L_TAL = mean_g w_g * L_TAL(g)
```

V0 candidate weight:

```text
w_g = 1.0
```

Optional hardness weighting:

```text
w_g = clamp(hardness(g), min_weight, max_weight)
```

Recommended optional values:

```text
min_weight: 0.25
max_weight: 1.0
```

If no candidates are selected:

```text
L_TAL = 0
```

The zero loss must be safe for distributed training and logging.

## Stop-Gradient Rules

TAL should follow strict detach rules.

Gradient should flow through:

```text
current detector features
projection head, if trainable
```

Gradient should not flow through:

```text
TDB directions
z_anchor used to construct target
z_target
HTM stored embeddings
```

Implementation:

```python
z = normalize(projector(roi_feat))
D = candidate.direction.detach()
z_anchor = z.detach()
z_target = normalize(z_anchor + alpha * D).detach()
loss = 1.0 - cosine_similarity(z, z_target)
```

## Projection Head

TAL requires a projection head.

Minimal projector:

```text
Linear(C -> D)
LayerNorm or BatchNorm optional
ReLU optional
Linear(D -> D) optional
L2 normalize output
```

Recommended V0:

```text
projector: Linear(C -> 256)
normalize output: true
```

Keep the projector simple for the first experiment. Extra MLP capacity can make it harder to know
whether gains come from FN-TDM or a new representation head.

## Scheduling

TAL should not start before TDB has useful entries.

Recommended conditions:

```text
tal_enabled_epoch >= htm_warmup_epochs + 1
TDB has at least one valid entry for the candidate class
```

Optional loss weight warmup:

```text
lambda_tal(e) = lambda_tal_max * min(1, (e - tal_start_epoch) / warmup_epochs)
```

Recommended defaults:

```text
tal_start_epoch: 1 or 2
lambda_warmup_epochs: 2
```

If HTM starts later, TAL effectively remains inactive until TDB is populated.

## TAL Variants for Ablation

### TAL-Target / Anchor-Shift

Main V0 method:

```text
z_target = stopgrad(normalize(stopgrad(z_g) + alpha D_c))
L = 1 - cos(z_g, z_target)
```

Pros:

- Simple.
- No previous-feature cache.
- Uses transition direction directly.

Cons:

- Local/self-anchored approximation.

### TAL-Delta

Align actual current feature movement with TDB direction.

```text
delta_g = normalize(z_g_current - stopgrad(z_g_prev))
L = 1 - cos(delta_g, D_c)
```

Pros:

- Closest to the original feature-trajectory idea.
- Directly aligns movement direction.

Cons:

- Requires a previous-feature cache per GT.
- Sensitive to augmentation/view differences.
- Needs careful stale-feature handling.

This is a strong V1 candidate after V0 is working.

### TAL-SuccessProto

Attract current embedding to stored successful embeddings instead of directions.

```text
z_success_proto = weighted_mean(z_tp_i)
L = 1 - cos(z_g, z_success_proto)
```

Pros:

- Stable and easy to optimize.
- Similar to prototype distillation.

Cons:

- Weaker novelty because it uses successful feature prototypes rather than transition directions.
- Requires TDB to store or aggregate `z_tp`.

Use as an ablation, not the main FN-TDM claim.

## Interaction With TCS

TAL should only run on candidates selected by TCS.

```text
all GTs -> TCS -> selected candidates -> TAL
```

TAL should not repeat TCS logic except for safety checks:

```text
candidate has direction
candidate bbox is valid
candidate class is valid
```

This keeps candidate-selection ablations separate from loss-design ablations.

## Interaction With TDB Retrieval

TAL is independent of how TDB builds the direction.

Supported retrieval variants:

```text
TDB-Last
TDB-TopK
TDB-TopK+Age
```

TAL consumes only:

```text
D_c or D_{c,t}
```

The TDB config decides whether the direction came from the latest entry, quality top-K prototype,
or age-decayed top-K prototype.

## FCOS Integration Notes

For FCOS V0:

1. Use FPN features from the normal training forward.
2. Use GT boxes from current targets.
3. Use `MultiScaleRoIAlign` to extract GT features.
4. Use the shared FN-TDM projection head.
5. Use TCS candidates selected from FCOS assignment/classification signals.
6. Add `lambda_tal * loss_tal` to FCOS training losses.

TAL should not modify:

```text
FCOS target assignment
classification logits
box regression targets
centerness targets
post-processing / NMS
```

## Configuration Sketch

Future config under `modules/cfg/fntdm.yaml`:

```yaml
tal:
  enabled: true
  variant: anchor_shift

  loss:
    lambda_tal: 0.05
    alpha: 0.2
    reduction: mean
    use_candidate_weight: false
    min_weight: 0.25
    max_weight: 1.0

  schedule:
    start_epoch: 1
    lambda_warmup_epochs: 2

  features:
    source: fpn_roi_align_gt
    roi_output_size: 7
    projector_dim: 256
    normalize: true

  safety:
    skip_invalid_direction: true
    min_direction_norm: 1.0e-6
    skip_invalid_bbox: true

  logging:
    save_summary: true
```

## Logging

Iteration or epoch summary:

```text
num_candidates
num_valid_candidates
num_skipped_invalid_direction
loss_tal
mean_cosine
mean_candidate_hardness
lambda_tal_current
```

Optional per-class summary:

```text
class_id
num_candidates
loss_tal_mean
cosine_mean
```

Do not log embeddings or full direction vectors by default.

## Edge Cases

- No candidates: return zero loss.
- Candidate direction is `None`: skip.
- Direction norm invalid: skip.
- GT bbox invalid or empty: skip.
- ROIAlign returns NaN/Inf: skip candidate and log.
- Projection output norm too small: skip candidate.
- TDB empty early in training: TAL remains inactive.

## V0 Implementation Checklist

1. Add `TransitionAlignmentLoss` module.
2. Add GT ROIAlign feature extraction shared with HTM.
3. Add simple projection head.
4. Implement Anchor-Shift target construction.
5. Implement cosine loss with safe zero-candidate behavior.
6. Add lambda scheduling.
7. Wire TCS candidates into TAL.
8. Add TAL loss into detector training loss dict.
9. Add summary logging.
10. Add unit tests for detach and no-op behavior.

## Minimal Unit Tests

Loss behavior:

```text
aligned z and target gives smaller loss than opposite direction
empty candidates returns zero scalar loss
invalid direction is skipped
```

Gradient:

```text
gradient flows to current z/projector
gradient does not flow to candidate.direction
gradient does not flow to z_target
```

Scheduling:

```text
lambda is zero before start_epoch
lambda warms up to lambda_tal
```

Device:

```text
candidate directions and ROI features are on the same device
```

## Research Claim

TAL should be described as:

```text
Transition Alignment Loss converts historical false-negative recovery directions
into a training-time auxiliary objective by locally shifting selected hard-instance
embeddings toward class-wise transition direction priors.
```

TAL is the component that turns FN-TDM from an analysis memory into an active regularizer.
