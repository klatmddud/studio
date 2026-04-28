# DHM-Guided Counterfactual Localization Repair

## 1. Motivation

DHM analysis of the ResNet50 FCOS runs suggests that many `FN_LOC` cases are not caused by a
complete lack of positive assignment. The detector often has same-class evidence and FCOS positive
points, but the decoded box does not cross the IoU threshold required to become a true positive.

This makes `FN_LOC` different from `FN_BG`, `FN_CLS`, and `FN_MISS`:

- `FN_BG`: the detector has weak foreground evidence around the GT.
- `FN_CLS`: the detector localizes a nearby object but class evidence is wrong or weak.
- `FN_MISS`: no useful candidate exists.
- `FN_LOC`: same-class evidence exists, but localization quality is insufficient.

The key observation is that many `FN_LOC` boxes may be close to the decision boundary:

```text
same-class candidate exists
score >= tau_loc_score
IoU(candidate, GT) < tau_iou
```

Instead of treating these samples only as hard examples, DCLR asks a counterfactual question:

```text
What minimal box intervention would have converted this FN_LOC into a TP?
```

The proposed method converts DHM-recorded localization failures into explicit repair supervision.

## 2. Core Idea

DCLR uses DHM temporal memory to identify persistent and relapse localization failures, then
constructs a counterfactual target for each selected failure:

```text
GT box: g
best same-class predicted box: b
current state: FN_LOC

counterfactual target:
  edge action and residual that would move b across the TP IoU threshold
```

The repair head does not merely regress from `b` to `g`. It learns the action needed to cross the
localization decision boundary:

```text
IoU(repair(b), g) >= tau_iou + margin
```

This makes the objective more specific than generic bounding-box refinement.

## 3. Relation to Prior Work

DCLR is related to, but distinct from, several existing detection ideas:

| Area | Representative methods | What they do |
|---|---|---|
| Stage-wise box refinement | Cascade R-CNN | Refine boxes through cascaded detectors |
| Localization quality estimation | IoU-Net, GFL, GFLV2, VarifocalNet | Predict or encode localization quality in the score |
| Dense assignment improvement | ATSS, PAA, TOOD | Improve positive/negative assignment or task alignment |
| Border-aware detection | BorderDet | Use boundary features to improve localization |
| Hard example mining | OHEM and variants | Reweight or replay difficult samples |

DCLR's novelty is not simply using border features or adding another refinement head. The novelty is
the connection:

```text
DHM temporal GT memory
-> persistent/relapse FN_LOC identification
-> counterfactual edge-action label generation
-> IoU threshold-crossing repair objective
-> selective localization rescue
```

Existing methods usually operate on the current candidate distribution. DCLR uses per-GT temporal
failure history to generate a different kind of target: the repair action that would have changed
the detector's state from `FN_LOC` to `TP`.

## 4. Target Failure Modes

### 4.1 Persistent Localization Failure

Transition:

```text
FN_LOC -> FN_LOC
```

This means the same GT repeatedly remains a localization failure. Possible causes:

- systematic edge bias in one or more directions;
- predicted box is repeatedly too small or shifted;
- object boundary is ambiguous at the selected FPN level;
- center point feature is insufficient for boundary localization;
- localization loss improves slowly even though class evidence exists.

For this case, DCLR emphasizes action and residual supervision.

### 4.2 Localization Relapse

Transition:

```text
TP -> FN_LOC
```

This means the GT was previously detected but later fell below the localization threshold. Possible
causes:

- localization forgetting;
- unstable candidate ranking;
- quality score and IoU mismatch;
- NMS selects a worse localized candidate;
- class confidence stays high while box quality drops.

For this case, DCLR emphasizes quality calibration and safe rescue.

## 5. Counterfactual Label Construction

### 5.1 Candidate Selection

During DHM mining, for each GT `g`, the detector predictions are analyzed. A GT becomes a DCLR
candidate when the DHM state is `FN_LOC` and there is sufficient same-class evidence:

```text
predicted label == GT label
predicted score >= tau_loc_score
IoU(predicted box, GT box) < tau_iou
```

The best same-class candidate can be selected by one of the following criteria:

```text
best by score * IoU
best by IoU among same-class candidates above tau_loc_score
best by score among same-class candidates with IoU >= tau_near
```

Recommended default:

```text
choose the same-class candidate with highest IoU among candidates whose score >= tau_loc_score
```

This keeps the counterfactual target focused on localization rather than classification.

### 5.2 Edge Residual Target

Let:

```text
b = (b_x1, b_y1, b_x2, b_y2)
g = (g_x1, g_y1, g_x2, g_y2)
w = b_x2 - b_x1
h = b_y2 - b_y1
```

The edge residual target is:

```text
r_l = (g_x1 - b_x1) / w
r_t = (g_y1 - b_y1) / h
r_r = (g_x2 - b_x2) / w
r_b = (g_y2 - b_y2) / h
```

Interpretation:

```text
r_l < 0: move left edge outward
r_l > 0: move left edge inward
r_r > 0: move right edge outward
r_r < 0: move right edge inward
r_t < 0: move top edge outward
r_t > 0: move top edge inward
r_b > 0: move bottom edge outward
r_b < 0: move bottom edge inward
```

The residual can be clipped to avoid unstable targets:

```text
r = clip(r, -target_delta_clip, target_delta_clip)
```

### 5.3 Discrete Action Target

DCLR can also convert the residual into a discrete action:

```text
A = {
  expand_left,
  shrink_left,
  expand_right,
  shrink_right,
  expand_top,
  shrink_top,
  expand_bottom,
  shrink_bottom,
  shift_left,
  shift_right,
  shift_up,
  shift_down,
  expand_all,
  shrink_all,
  no_op
}
```

Two label-generation strategies are possible.

#### Max-IoU Action

Apply each action with a fixed magnitude and choose the action that maximizes IoU:

```text
a* = argmax_a IoU(apply_action(b, a, magnitude), g)
```

This is simple and robust.

#### Minimal Crossing Action

Search for the smallest action magnitude that crosses the TP threshold:

```text
a*, m* = argmin_action_magnitude
         subject to IoU(apply_action(b, a, m), g) >= tau_iou + margin
```

This is more faithful to the counterfactual question but more expensive.

Recommended MVP:

```text
use edge residual target first;
add discrete action classification after residual repair is stable.
```

## 6. Temporal Error Memory

DCLR can extend DHM records with localization error statistics:

```text
last_pred_box
last_candidate_score
last_candidate_iou
last_edge_residual_ltrb
ema_edge_residual_ltrb
ema_abs_edge_residual_ltrb
edge_error_count
counterfactual_crossable_count
```

For each FN_LOC record:

```text
ema_edge_residual = momentum * ema_edge_residual
                  + (1 - momentum) * current_edge_residual
```

This allows the method to distinguish:

- random localization noise;
- persistent left/right/top/bottom bias;
- shrinking boxes;
- expanding boxes;
- scale-specific failure patterns.

The temporal error memory is optional for the MVP, but it strengthens the novelty claim because the
method uses DHM not only for sample selection but also for target shaping.

## 7. Model Architecture

### 7.1 Repair Candidate Feature

For each selected candidate box `b`, construct a feature vector:

```text
z = concat(
  f_center,
  f_left,
  f_right,
  f_top,
  f_bottom,
  box_geometry,
  class_score,
  centerness,
  fpn_level_embedding,
  transition_embedding,
  optional_ema_edge_residual
)
```

Feature components:

- `f_center`: feature sampled at the candidate center;
- `f_left`, `f_right`, `f_top`, `f_bottom`: pooled border features;
- `box_geometry`: width, height, aspect ratio, normalized area, normalized position;
- `class_score`: candidate class confidence;
- `centerness`: FCOS centerness prediction;
- `fpn_level_embedding`: selected FPN level;
- `transition_embedding`: `FN_LOC->FN_LOC`, `TP->FN_LOC`, or other selected transition;
- `optional_ema_edge_residual`: DHM temporal edge-error prior.

### 7.2 Repair Head

The repair head predicts:

```text
pred_delta = (delta_l, delta_t, delta_r, delta_b)
pred_action_logits
pred_iou_quality
pred_rescue_score
```

MVP output:

```text
pred_delta
pred_iou_quality
```

Expanded output:

```text
pred_delta
pred_action_logits
pred_iou_quality
pred_rescue_score
```

### 7.3 Box Application

The refined box is:

```text
delta = max_delta * tanh(raw_delta)

x1' = x1 + delta_l * w
y1' = y1 + delta_t * h
x2' = x2 + delta_r * w
y2' = y2 + delta_b * h
```

Then clip to the image boundary:

```text
b' = clip_box((x1', y1', x2', y2'), image_shape)
```

## 8. Loss Function

The full DCLR objective:

```text
L_DCLR =
  lambda_giou     * L_giou(b', g)
+ lambda_residual * SmoothL1(pred_delta, target_delta)
+ lambda_cross    * max(0, tau_iou + margin - IoU(b', g))
+ lambda_quality  * BCE(pred_iou_quality, IoU(b', g))
+ lambda_action   * CE(pred_action, target_action)
+ lambda_rescue   * BCE(pred_rescue_score, rescue_label)
```

MVP objective:

```text
L_DCLR_MVP =
  lambda_giou     * L_giou(b', g)
+ lambda_residual * SmoothL1(pred_delta, target_delta)
+ lambda_cross    * max(0, tau_iou + margin - IoU(b', g))
+ lambda_quality  * BCE(pred_iou_quality, IoU(b', g))
```

The threshold-crossing loss is the method-specific term:

```text
L_cross = max(0, tau_iou + margin - IoU(b', g))
```

This directly optimizes the desired state transition:

```text
FN_LOC -> TP
```

## 9. Transition-Aware Weighting

Different transitions should not necessarily receive the same loss weights.

Recommended initial weighting:

```yaml
transition_weights:
  FN_LOC->FN_LOC:
    giou: 1.0
    residual: 2.0
    crossing: 2.0
    quality: 1.0
    action: 1.0
    rescue: 1.0

  TP->FN_LOC:
    giou: 1.0
    residual: 1.0
    crossing: 1.0
    quality: 2.0
    action: 0.5
    rescue: 2.0

  TP->TP:
    giou: 0.25
    residual: 0.25
    crossing: 0.0
    quality: 0.5
    action: 0.0
    rescue: 1.0
```

Rationale:

- `FN_LOC->FN_LOC` needs stronger edge correction.
- `TP->FN_LOC` may involve ranking or quality mismatch, so quality calibration matters more.
- `TP->TP` can be used as a preservation sample to prevent the repair head from damaging stable TPs.

## 10. Training Procedure

### 10.1 Epoch-End Mining

At epoch end:

1. Run DHM mining over the train set.
2. For each GT, record state, transition, best same-class candidate, candidate IoU, and candidate score.
3. For `FN_LOC` records, compute counterfactual residual/action targets.
4. Store selected repair metadata in DHM or a sidecar repair memory.

### 10.2 Next Epoch Training

During the next training epoch:

1. FCOS computes normal detection losses.
2. DCLR reads DHM records for each target GT.
3. Eligible `FN_LOC->FN_LOC` and `TP->FN_LOC` records are selected.
4. Candidate positive points or dense predictions are decoded.
5. Border/geometry/score features are built.
6. The repair head predicts residual and quality.
7. DCLR losses are added to the normal FCOS losses.

### 10.3 Candidate Source Options

#### Dense Positive Points

Use assigned FCOS positive points.

Pros:

- easy to integrate with the training forward;
- directly connected to FCOS assignment;
- lower computational overhead.

Cons:

- may not perfectly match inference-time ranking/NMS behavior.

#### Pre-NMS Detection Candidates

Use dense predictions after score filtering but before NMS.

Pros:

- closer to inference failure;
- better for quality calibration.

Cons:

- more expensive;
- candidate matching is more complex.

Recommended MVP:

```text
start with dense positive points;
add pre-NMS candidates only after the training-only loss is stable.
```

## 11. Inference Procedure

DHM records are train-set specific and cannot be directly used for unseen validation/test images.
Inference therefore uses the learned repair/risk heads.

Proposed inference:

1. Run normal FCOS forward.
2. Build pre-NMS candidates from class score and centerness.
3. Select rescue candidates:

```text
candidate if:
  class_score >= tau_cls
  and (
    centerness <= tau_ctr
    or predicted_iou_quality <= tau_quality
    or predicted_rescue_score >= tau_rescue
  )
```

4. Apply DCLR repair to selected candidates only.
5. Recompute refined score:

```text
score_refined = class_score * predicted_iou_quality
```

or:

```text
score_refined = class_score * sqrt(centerness * predicted_iou_quality)
```

6. Run NMS over original and refined boxes.

Safe initial setting:

```yaml
inference:
  enabled: true
  keep_original_boxes: true
  rescue_topk: 100
  max_delta: 0.25
  class_score_threshold: 0.05
  rescue_score_threshold: 0.3
```

`keep_original_boxes: true` is important for early experiments because a bad repair prediction should
not remove the original candidate.

## 12. Expected Benefits

DCLR is expected to help when `FN_LOC` candidates are close to the TP boundary.

Primary expected improvements:

- lower final DHM `FN_LOC` count;
- more `FN_LOC->TP` recoveries;
- lower `FN_LOC->FN_LOC` persistence;
- higher mAP75;
- higher mAP50:95;
- higher mAR100 if rescued boxes survive NMS.

The method may not help if most `FN_LOC` cases have very low IoU and cannot be repaired by a small
edge intervention.

## 13. Diagnostic Checks Before Implementation

Before full implementation, run these diagnostics on DHM logs and saved predictions:

1. FN_LOC best same-class IoU histogram.
2. Percentage of FN_LOC candidates in IoU ranges:

```text
[0.0, 0.1)
[0.1, 0.3)
[0.3, 0.5)
[0.5, 1.0)
```

3. Counterfactual crossing rate:

```text
percentage of FN_LOC candidates that can cross tau_iou with max_delta <= 0.25
```

4. Edge residual direction consistency for persistent `FN_LOC->FN_LOC` records.
5. Residual variance by class, FPN level, and object scale.
6. Difference between persistent FN_LOC and relapse FN_LOC residual patterns.
7. Correlation between centerness, predicted IoU quality, and actual IoU.

If many FN_LOC candidates are in the 0.3 to 0.5 IoU range and are crossable with small deltas, DCLR
has a strong chance of improving AP75 and mAP50:95.

## 14. Minimal Implementation Plan

### Phase 0: Diagnostics

- Store best same-class candidate box for each FN_LOC GT during DHM mining.
- Compute edge residuals and crossing feasibility.
- Report histograms in `history.json` or a sidecar JSON.

### Phase 1: Training-Only Residual Repair

- Implementation status: implemented as disabled-by-default `dhmr.counterfactual_repair` in `modules/cfg/dhmr.yaml`.
- Add DCLR config.
- Add repair candidate selection from DHM records.
- Use dense positive points.
- Add residual, GIoU, crossing, and IoU-quality losses.
- Do not change inference yet.

Goal:

```text
verify that the repair head learns meaningful residuals without destabilizing FCOS training
```

### Phase 2: Transition-Aware DCLR

- Add transition embeddings.
- Add transition-specific loss weights.
- Separate persistent and relapse statistics.

Goal:

```text
show that FN_LOC->FN_LOC and TP->FN_LOC benefit from different supervision emphasis
```

### Phase 3: Inference-Time Selective Repair

- Apply repair to top-K selected candidates.
- Keep original boxes.
- Score refined boxes with predicted IoU quality.
- Run joint NMS.

Goal:

```text
convert training-time repair ability into validation mAP and mAR improvements
```

### Phase 4: Counterfactual Action Head

- Add discrete action labels.
- Add action classification loss.
- Compare residual-only versus residual-plus-action.

Goal:

```text
test whether explicit edge-action supervision improves repair generalization
```

## 15. Ablation Plan

| Experiment | Description |
|---|---|
| Baseline | FCOS + DHM logging only |
| Hard Replay | DHM-guided full-image replay |
| FN_LOC Crop Repair | DHM-guided crop replay for FN_LOC |
| Generic Refinement | Refinement head without DHM selection |
| DHM Selective Refinement | Refine only DHM-selected FN_LOC candidates |
| DCLR without Crossing Loss | Residual + GIoU + quality only |
| DCLR with Crossing Loss | Add threshold-crossing objective |
| DCLR without Temporal Memory | Use current FN_LOC only |
| DCLR with Temporal Memory | Use EMA edge residuals |
| Transition-Agnostic DCLR | Treat all FN_LOC transitions the same |
| Transition-Aware DCLR | Separate FN_LOC->FN_LOC and TP->FN_LOC |
| Residual Only | Predict continuous residual only |
| Residual + Action | Predict residual and discrete edge action |
| Training-Only | Add auxiliary loss but no inference repair |
| Inference Repair | Apply selective repair at inference |

Metrics:

- COCO mAP50:95;
- mAP75;
- mAP50;
- mAR100;
- final DHM `FN_LOC`;
- `FN_LOC->FN_LOC` count;
- `TP->FN_LOC` count;
- `FN_LOC->TP` recovery count;
- stable TP degradation rate;
- number of predictions after NMS;
- AP by object scale.

## 16. Risks

Main risks:

- If the selected FN_LOC candidate is not close to the GT, repair may hallucinate boxes.
- If `max_delta` is too large, refined boxes may create false positives.
- If quality calibration is poor, NMS ranking may degrade.
- If inference selection is too broad, stable TPs may be damaged.
- If most FN_LOC cases are not counterfactually crossable, the method may not improve AP.

Mitigations:

- keep `max_delta` small at first;
- use `keep_original_boxes: true`;
- cap `rescue_topk`;
- include stable `TP->TP` preservation samples;
- report crossing feasibility before claiming the method should work.

## 17. Novelty Claim

Suggested claim:

```text
We propose DHM-Guided Counterfactual Localization Repair, a selective localization rescue method
that converts temporally persistent and relapse localization failures into counterfactual edge
intervention targets. Instead of only replaying hard images or globally refining all detections,
DCLR learns the minimal box repair needed to move DHM-identified FN_LOC candidates across the TP IoU
threshold.
```

Short version:

```text
DCLR uses temporal GT-level failure memory to generate counterfactual localization actions for
FN_LOC candidates, directly optimizing the state transition from FN_LOC to TP.
```

## 18. References

- FCOS: Fully Convolutional One-Stage Object Detection
- Cascade R-CNN: Delving Into High Quality Object Detection
- IoU-Net: Acquisition of Localization Confidence for Accurate Object Detection
- Generalized Focal Loss and Generalized Focal Loss V2
- VarifocalNet: An IoU-aware Dense Object Detector
- BorderDet: Border Feature for Dense Object Detection
- TOOD: Task-Aligned One-Stage Object Detection
- ATSS: Bridging the Gap Between Anchor-based and Anchor-free Detection
- PAA: Probabilistic Anchor Assignment with IoU Prediction
