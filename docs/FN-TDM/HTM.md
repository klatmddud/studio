# HTM - Hard Transition Mining

## Goal

HTM is the transition-mining component of FN-TDM.

Its job is to find reliable ground-truth instances that move from a false-negative state to
a true-positive state during training, then emit the feature transition needed by the
Transition Direction Bank (TDB).

```text
FN-like GT at epoch k  ->  TP-like GT at epoch e
d = normalize(z_e - z_k)
```

The first implementation should prioritize correctness and interpretability over speed.
Therefore, the baseline design uses deterministic full-train inference after every epoch.

## Scope

HTM does not train the model directly. It only produces transition events.

Responsibilities:

- Run epoch-end mining over the training set.
- Assign each GT instance a detection state.
- Extract a GT-aligned embedding for each GT.
- Maintain rolling per-GT history.
- Emit `FN -> TP` transition events.
- Log enough metadata for analysis and ablation.

Non-responsibilities:

- Storing the final direction memory policy. That belongs to TDB.
- Applying auxiliary losses. That belongs to TAL or the detector wrapper.
- Changing inference behavior.
- Replacing COCO evaluation.

## Baseline Mining Mode

HTM baseline runs after each training epoch.

```text
for epoch e:
    train one epoch

    model.eval()
    for image, targets in deterministic_train_loader:
        predictions, feature_maps = model(image, return_features=True)
        update HTM with predictions, feature_maps, targets, epoch=e
```

The deterministic train loader must disable random training augmentation. It should use the
same image scaling and normalization each epoch so that feature differences mostly reflect
model learning rather than view noise.

Recommended default:

```text
mine_interval: 1
warmup_epochs: 0 or 1
deterministic_view: resize + normalize only
random_flip: false
strong_augmentation: false
```

`warmup_epochs` may be increased if early detector predictions are too noisy.

## GT Identity

HTM is GT-centric. Every record is keyed by a stable GT identity.

Preferred key:

```text
gt_uid = (image_id, ann_id)
```

Fallback when annotation ID is unavailable:

```text
gt_uid = (image_id, class_id, normalized_bbox_hash, instance_index)
```

The fallback is acceptable for static COCO-style datasets, but annotation IDs are strongly
preferred because HTM compares the same instance across epochs.

## Detection State

The minimal state is:

```text
state_e(g) in {TP, FN}
```

For analysis and transition filtering, HTM should also store a failure subtype when
`state_e(g) = FN`.

```text
FN_BG
    The detector has low target-class/object confidence around the GT.
    This is the primary target for FN-TDM.

FN_CLS
    A candidate overlaps the GT, but the predicted class is wrong or the target-class score is low.

FN_LOC
    The target class has a confident candidate, but localization quality is insufficient.

FN_MISS
    There is no meaningful candidate near the GT.
```

Initial transition sources:

```text
use for TDB: FN_BG, FN_CLS, FN_MISS
log only or downweight: FN_LOC
```

The reason is that FN-TDM is aimed at background confusion and missed foreground
recognition. `FN_LOC` may be dominated by box regression or assignment issues rather than
feature movement from background to object.

## TP Assignment

For each GT `g`, define TP using post-processed predictions.

```text
TP_e(g) =
    exists prediction p such that:
        class(p) == class(g)
        IoU(p.box, g.box) >= tau_iou
        score(p) >= tau_tp
```

Recommended initial values:

```text
tau_iou: 0.5
tau_tp: 0.3
```

If multiple predictions match the same GT, use the one with highest quality:

```text
quality(p, g) = score(p) * IoU(p.box, g.box)
```

The selected prediction provides:

```text
matched_score
matched_iou
matched_class
matched_box
```

## FN Subtype Assignment

When no valid TP exists, HTM should compute coarse evidence from nearby predictions.

Definitions:

```text
best_any_iou:
    max IoU between any prediction box and GT box.

best_target_score:
    max score among predictions with class == class(g), regardless of IoU.

best_near_target_score:
    max target-class score among predictions with IoU >= tau_near.

best_near_wrong_score:
    max wrong-class score among predictions with IoU >= tau_near.

best_target_iou:
    max IoU among predictions with class == class(g) and score >= tau_cls_evidence.
```

Recommended initial values:

```text
tau_near: 0.3
tau_cls_evidence: 0.3
tau_bg_score: 0.1
tau_loc_score: 0.3
```

Subtype heuristic:

```text
if best_any_iou < tau_near and best_target_score < tau_bg_score:
    fn_type = FN_MISS

elif best_near_target_score < tau_bg_score and best_near_wrong_score < tau_bg_score:
    fn_type = FN_BG

elif best_any_iou >= tau_near and best_near_target_score < tau_tp:
    fn_type = FN_CLS

elif best_target_score >= tau_loc_score and best_target_iou < tau_iou:
    fn_type = FN_LOC

else:
    fn_type = FN_BG
```

These subtypes are diagnostic, not a replacement for evaluation. Keep them simple for V0.

## Feature Extraction

HTM should extract a GT-aligned feature even when the detector missed the object.

Baseline method:

```text
FPN feature maps -> MultiScaleRoIAlign(gt_bbox) -> GAP -> projection head -> normalize
```

Notation:

```text
z_e(g) = normalize(projector(pool(ROIAlign(F_e, bbox_g))))
```

Recommended defaults:

```text
roi_output_size: 7
projector_dim: 256
normalize_embedding: true
detach_embedding: true
store_on_cpu: true
```

The projection head is part of FN-TDM training state. It should be shared by HTM and TAL so
the stored directions and current training embeddings live in the same space.

Important: HTM should store detached embeddings only. It must not keep computation graphs
from the mining pass.

## Rolling History

HTM should not stack every epoch's GT embedding by default. It maintains rolling state per
GT and emits events when a transition occurs.

Per-GT state:

```python
HTMHistory = {
    "gt_uid": str,
    "image_id": Any,
    "ann_id": Any,
    "class_id": int,
    "bbox": Tensor[4],

    "last_state": "UNSEEN" | "FN" | "TP",
    "last_epoch": int | None,

    "last_fn_epoch": int | None,
    "last_fn_z": Tensor[D] | None,
    "last_fn_score": float | None,
    "last_fn_iou": float | None,
    "last_fn_type": str | None,

    "last_tp_epoch": int | None,
    "last_tp_z": Tensor[D] | None,
    "last_tp_score": float | None,
    "last_tp_iou": float | None,

    "fn_count": int,
    "tp_count": int,
    "transition_count": int,
    "last_emitted_epoch": int | None,
}
```

When a GT is FN, update the latest FN snapshot:

```text
last_fn_epoch = current_epoch
last_fn_z = z_current
last_fn_score = matched_or_evidence_score
last_fn_iou = matched_or_evidence_iou
last_fn_type = fn_type
```

When a GT is TP, update the latest TP snapshot:

```text
last_tp_epoch = current_epoch
last_tp_z = z_current
last_tp_score = matched_score
last_tp_iou = matched_iou
```

## Transition Emission

Baseline transition rule:

```text
emit when previous state is FN and current state is TP
```

More generally, allow a recent FN within a short window:

```text
k = last_fn_epoch
if current_state == TP
and last_state == FN
and k is not None
and current_epoch - k <= transition_window:
    emit transition
```

Recommended defaults:

```text
transition_window: 3
max_transitions_per_gt: 1
allowed_fn_types: [FN_BG, FN_CLS, FN_MISS]
```

Behavior examples:

```text
epoch 9  FN  -> update last_fn
epoch 10 TP  -> emit transition d = normalize(z10 - z9)
epoch 11 TP  -> no new transition
```

If repeated transitions are enabled:

```text
epoch 9  FN
epoch 10 TP  -> transition A
epoch 11 TP  -> no transition
epoch 12 FN
epoch 13 TP  -> transition B
```

For V0, use `max_transitions_per_gt = 1` to reduce noise and memory dominance from unstable
instances.

## Transition Event

HTM emits an event to TDB.

```python
TransitionEvent = {
    "gt_uid": str,
    "image_id": Any,
    "ann_id": Any,
    "class_id": int,
    "bbox": Tensor[4],

    "epoch_fn": int,
    "epoch_tp": int,
    "fn_type": str,

    "z_fn": Tensor[D],
    "z_tp": Tensor[D],
    "direction": Tensor[D],

    "score_fn": float,
    "score_tp": float,
    "iou_fn": float,
    "iou_tp": float,

    "quality": float,
}
```

Direction:

```text
direction = normalize(z_tp - z_fn)
```

If the norm is too small, skip the event.

```text
min_direction_norm: 1e-6
```

## Event Quality

Quality is used by TDB for ranking, replacement, or sampling.

Baseline score:

```text
quality = score_tp * (1 - score_fn) * exp(-(epoch_tp - epoch_fn - 1) / lambda_gap)
```

Recommended default:

```text
lambda_gap: 2.0
```

Optional subtype weighting:

```text
FN_BG:   1.0
FN_MISS: 1.0
FN_CLS:  0.8
FN_LOC:  0.3
```

Then:

```text
quality = quality * type_weight[fn_type]
```

For V0, either exclude `FN_LOC` or keep it with low quality for analysis only.

## State Update Order

Use the previous state for transition checks before overwriting it.

```python
def update_gt(history, current, epoch):
    prev_state = history.last_state

    if current.state == "TP":
        if should_emit(history, current, prev_state, epoch):
            event = build_event(history.last_fn, current)
            emit(event)

        history.last_tp_epoch = epoch
        history.last_tp_z = current.z
        history.last_tp_score = current.score
        history.last_tp_iou = current.iou
        history.tp_count += 1

    else:
        history.last_fn_epoch = epoch
        history.last_fn_z = current.z
        history.last_fn_score = current.score
        history.last_fn_iou = current.iou
        history.last_fn_type = current.fn_type
        history.fn_count += 1

    history.last_state = current.state
    history.last_epoch = epoch
```

After emitting a transition, do not emit again on `TP -> TP`. Either leave `last_fn_*` in
history for debugging or mark it consumed. The transition condition must still require
`prev_state == FN`.

## Configuration Sketch

Add a future config under `modules/cfg/fntdm.yaml` or `modules/cfg/htm.yaml`.

```yaml
htm:
  enabled: false
  mode: epoch_end_full_train
  mine_interval: 1
  warmup_epochs: 0

  matching:
    tau_iou: 0.5
    tau_tp: 0.3
    tau_near: 0.3
    tau_bg_score: 0.1
    tau_cls_evidence: 0.3
    tau_loc_score: 0.3

  transition:
    transition_window: 3
    max_transitions_per_gt: 1
    allowed_fn_types: [FN_BG, FN_CLS, FN_MISS]
    min_direction_norm: 1.0e-6
    lambda_gap: 2.0

  features:
    source: fpn_roi_align_gt
    roi_output_size: 7
    projector_dim: 256
    normalize: true
    store_on_cpu: true

  logging:
    save_events_csv: true
    save_epoch_summary: true
```

## Logging

HTM should log both compact summaries and transition events.

Epoch summary:

```text
epoch
num_gt
num_tp
num_fn
num_fn_bg
num_fn_cls
num_fn_loc
num_fn_miss
num_transitions
num_skipped_transition
```

Event CSV:

```text
epoch_fn,epoch_tp,gt_uid,image_id,ann_id,class_id,fn_type,
score_fn,score_tp,iou_fn,iou_tp,quality,direction_norm
```

Do not write large embedding tensors to CSV. If event embeddings need to be persisted, use a
separate tensor checkpoint owned by TDB.

## DDP Behavior

For the first baseline, prefer running epoch-end HTM on rank 0 only after the epoch is
complete. Rank 0 can build the deterministic train loader and mine transitions.

Implementation options:

1. Rank 0 mines full train set and broadcasts TDB state before the next epoch.
2. Every rank mines a disjoint train subset, then states/events are gathered and merged.

Option 1 is simpler and preferred for V0. It may be slower but avoids duplicate events and
merge complexity.

## Expected Overhead

This baseline adds roughly one extra full training-set detector inference per mining epoch.
It is intentionally expensive.

Use it as the clean reference implementation before developing cheaper online HTM variants.

Mitigations when needed:

- Increase `mine_interval`.
- Start after `warmup_epochs`.
- Mine a fixed subset for debugging.
- Disable embedding persistence except selected events.

## Edge Cases

- No predictions for an image: all GTs become `FN_MISS`.
- Multiple predictions match one GT: choose highest `score * IoU`.
- One prediction matches multiple GTs: for HTM state, allow independent GT matching in V0.
  Evaluation-style one-to-one matching can be added later if needed.
- Direction norm is near zero: skip event.
- GT appears TP at first observation: update TP state, emit no transition.
- GT alternates frequently: `max_transitions_per_gt` controls event spam.
- Annotation ID missing: use bbox hash fallback and log a warning once.

## V0 Implementation Checklist

1. Add HTM config with all defaults disabled.
2. Build stable GT UID helper.
3. Add deterministic train mining loader.
4. Add detector inference path that returns post-processed predictions and FPN features.
5. Add GT-state assignment from predictions.
6. Add GT ROIAlign embedding extractor and projection head.
7. Add rolling `HTMHistory` store.
8. Add transition emission and quality scoring.
9. Add event handoff API to TDB.
10. Add epoch summary and event CSV logging.
11. Add unit tests for state transitions.

## Minimal Unit Tests

State transition:

```text
FN -> TP emits one event
TP -> TP emits zero events
TP -> FN -> TP emits one new event if allowed
```

Limits:

```text
max_transitions_per_gt = 1 prevents repeated events
transition_window rejects stale FN snapshots
disallowed fn_type does not emit
```

Quality:

```text
higher score_tp increases quality
higher score_fn decreases quality
larger epoch gap decreases quality
```

Feature:

```text
direction has unit norm
near-zero direction is skipped
stored z tensors are detached and on CPU when configured
```

## Research Claim

HTM should be described as:

```text
Hard Transition Mining identifies reliable false-negative-to-true-positive turning
points of hard ground-truth instances and extracts the discriminative feature direction
that accompanied the recovery.
```

This distinguishes FN-TDM from ordinary hard example mining: HTM does not only ask which
instances are hard; it asks which hard instances became detectable and what feature
direction marked that transition.
