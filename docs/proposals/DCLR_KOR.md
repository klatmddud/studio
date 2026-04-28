# DHM-Guided Counterfactual Localization Repair

## 1. 동기

ResNet50 FCOS 실험에 대한 DHM 분석 결과를 보면, 많은 `FN_LOC` 사례는 positive assignment가 완전히 부족해서 생긴 문제가 아니다. 모델은 같은 class에 대한 evidence와 FCOS positive point를 가지고 있지만, decoded box가 true positive가 되기 위한 IoU threshold를 넘지 못한다.

따라서 `FN_LOC`은 `FN_BG`, `FN_CLS`, `FN_MISS`와 다르게 해석해야 한다.

- `FN_BG`: GT 주변에서 foreground evidence가 약하다.
- `FN_CLS`: 근처 객체를 localization했지만 class evidence가 틀렸거나 약하다.
- `FN_MISS`: 유의미한 candidate가 없다.
- `FN_LOC`: 같은 class evidence는 있지만 localization quality가 부족하다.

핵심 관찰은 많은 `FN_LOC` box가 decision boundary 근처에 있을 수 있다는 점이다.

```text
same-class candidate exists
score >= tau_loc_score
IoU(candidate, GT) < tau_iou
```

DCLR은 이런 sample을 단순 hard example로만 다루지 않고, 다음 counterfactual question을 던진다.

```text
어떤 최소 box intervention이 있었으면 이 FN_LOC이 TP가 되었는가?
```

제안 방법은 DHM이 기록한 localization failure를 명시적인 repair supervision으로 변환한다.

## 2. 핵심 아이디어

DCLR은 DHM temporal memory를 사용해 persistent localization failure와 relapse localization failure를 식별한 뒤, 선택된 각 failure에 대해 counterfactual target을 만든다.

```text
GT box: g
best same-class predicted box: b
current state: FN_LOC

counterfactual target:
  b를 TP IoU threshold 너머로 이동시키는 edge action과 residual
```

repair head는 단순히 `b`에서 `g`로 회귀하는 것만 학습하지 않는다. localization decision boundary를 넘기기 위해 필요한 action을 학습한다.

```text
IoU(repair(b), g) >= tau_iou + margin
```

이 점 때문에 DCLR의 objective는 일반적인 bounding-box refinement보다 더 구체적이다.

## 3. 기존 연구와의 관계

DCLR은 여러 기존 object detection 아이디어와 관련이 있지만, 동일하지 않다.

| 영역 | 대표 방법 | 핵심 내용 |
|---|---|---|
| Stage-wise box refinement | Cascade R-CNN | 여러 detector stage를 거쳐 box를 refine |
| Localization quality estimation | IoU-Net, GFL, GFLV2, VarifocalNet | localization quality를 예측하거나 score에 반영 |
| Dense assignment improvement | ATSS, PAA, TOOD | positive/negative assignment 또는 task alignment 개선 |
| Border-aware detection | BorderDet | boundary feature를 사용해 localization 개선 |
| Hard example mining | OHEM 및 변형 | 어려운 sample을 reweight 또는 replay |

DCLR의 novelty는 단순히 border feature를 쓰거나 refinement head를 하나 더 붙이는 데 있지 않다. novelty는 다음 연결 구조에 있다.

```text
DHM temporal GT memory
-> persistent/relapse FN_LOC identification
-> counterfactual edge-action label generation
-> IoU threshold-crossing repair objective
-> selective localization rescue
```

기존 방법들은 대체로 현재 candidate distribution에서 동작한다. 반면 DCLR은 GT별 temporal failure history를 사용해 다른 종류의 target을 만든다. 즉 detector state를 `FN_LOC`에서 `TP`로 바꾸기 위해 필요했던 repair action을 학습 target으로 만든다.

## 4. 대상 실패 유형

### 4.1 Persistent Localization Failure

Transition:

```text
FN_LOC -> FN_LOC
```

이는 같은 GT가 반복적으로 localization failure로 남는다는 뜻이다. 가능한 원인은 다음과 같다.

- 특정 방향으로 box edge bias가 반복된다.
- predicted box가 계속 너무 작거나 이동되어 있다.
- 선택된 FPN level에서 object boundary가 모호하다.
- center point feature만으로 boundary localization이 충분하지 않다.
- class evidence는 있지만 localization loss가 느리게 개선된다.

이 경우 DCLR은 action과 residual supervision을 강하게 둔다.

### 4.2 Localization Relapse

Transition:

```text
TP -> FN_LOC
```

이는 이전에는 GT가 검출되었지만 이후 localization threshold 아래로 떨어졌다는 뜻이다. 가능한 원인은 다음과 같다.

- localization forgetting;
- 불안정한 candidate ranking;
- quality score와 IoU의 mismatch;
- NMS가 더 나쁜 localization candidate를 선택;
- class confidence는 유지되지만 box quality가 흔들림.

이 경우 DCLR은 quality calibration과 안전한 rescue를 더 중요하게 둔다.

## 5. Counterfactual Label 생성

### 5.1 Candidate Selection

DHM mining 중 각 GT `g`에 대해 detector prediction을 분석한다. DHM state가 `FN_LOC`이고 같은 class evidence가 충분하면 해당 GT를 DCLR candidate로 삼는다.

```text
predicted label == GT label
predicted score >= tau_loc_score
IoU(predicted box, GT box) < tau_iou
```

best same-class candidate는 다음 기준 중 하나로 고를 수 있다.

```text
score * IoU가 가장 큰 candidate
tau_loc_score 이상인 same-class candidate 중 IoU가 가장 큰 candidate
tau_near 이상인 same-class candidate 중 score가 가장 큰 candidate
```

권장 기본값:

```text
score >= tau_loc_score를 만족하는 same-class candidate 중 IoU가 가장 큰 candidate를 선택
```

이렇게 하면 counterfactual target이 classification 문제가 아니라 localization 문제에 집중한다.

### 5.2 Edge Residual Target

다음과 같이 둔다.

```text
b = (b_x1, b_y1, b_x2, b_y2)
g = (g_x1, g_y1, g_x2, g_y2)
w = b_x2 - b_x1
h = b_y2 - b_y1
```

edge residual target은 다음과 같다.

```text
r_l = (g_x1 - b_x1) / w
r_t = (g_y1 - b_y1) / h
r_r = (g_x2 - b_x2) / w
r_b = (g_y2 - b_y2) / h
```

해석:

```text
r_l < 0: left edge를 바깥쪽으로 이동
r_l > 0: left edge를 안쪽으로 이동
r_r > 0: right edge를 바깥쪽으로 이동
r_r < 0: right edge를 안쪽으로 이동
r_t < 0: top edge를 바깥쪽으로 이동
r_t > 0: top edge를 안쪽으로 이동
r_b > 0: bottom edge를 바깥쪽으로 이동
r_b < 0: bottom edge를 안쪽으로 이동
```

불안정한 target을 막기 위해 residual은 clip할 수 있다.

```text
r = clip(r, -target_delta_clip, target_delta_clip)
```

### 5.3 Discrete Action Target

DCLR은 residual을 discrete action으로 변환할 수도 있다.

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

label 생성 전략은 두 가지가 가능하다.

#### Max-IoU Action

각 action을 고정 magnitude로 적용한 뒤 IoU를 가장 크게 만드는 action을 고른다.

```text
a* = argmax_a IoU(apply_action(b, a, magnitude), g)
```

이 방식은 단순하고 robust하다.

#### Minimal Crossing Action

TP threshold를 넘기는 가장 작은 action magnitude를 찾는다.

```text
a*, m* = argmin_action_magnitude
         subject to IoU(apply_action(b, a, m), g) >= tau_iou + margin
```

이 방식은 counterfactual question에 더 충실하지만 계산 비용이 더 크다.

권장 MVP:

```text
먼저 edge residual target만 사용한다.
residual repair가 안정화된 뒤 discrete action classification을 추가한다.
```

## 6. Temporal Error Memory

DCLR은 DHM record에 localization error statistics를 확장해 저장할 수 있다.

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

각 FN_LOC record에 대해 다음처럼 EMA를 업데이트한다.

```text
ema_edge_residual = momentum * ema_edge_residual
                  + (1 - momentum) * current_edge_residual
```

이를 통해 다음 패턴을 구분할 수 있다.

- random localization noise;
- 지속적인 left/right/top/bottom bias;
- box가 계속 shrink되는 패턴;
- box가 계속 expand되는 패턴;
- scale-specific failure pattern.

Temporal error memory는 MVP에서는 선택사항이지만, 방법론의 novelty를 강화한다. DHM을 sample selection뿐 아니라 target shaping에도 사용하기 때문이다.

## 7. Model Architecture

### 7.1 Repair Candidate Feature

선택된 candidate box `b`마다 feature vector를 구성한다.

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

Feature 구성 요소:

- `f_center`: candidate center에서 sampling한 feature;
- `f_left`, `f_right`, `f_top`, `f_bottom`: pooled border feature;
- `box_geometry`: width, height, aspect ratio, normalized area, normalized position;
- `class_score`: candidate class confidence;
- `centerness`: FCOS centerness prediction;
- `fpn_level_embedding`: 선택된 FPN level;
- `transition_embedding`: `FN_LOC->FN_LOC`, `TP->FN_LOC`, 기타 선택 transition;
- `optional_ema_edge_residual`: DHM temporal edge-error prior.

### 7.2 Repair Head

repair head는 다음 값을 예측한다.

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

확장 output:

```text
pred_delta
pred_action_logits
pred_iou_quality
pred_rescue_score
```

### 7.3 Box Application

refined box는 다음처럼 계산한다.

```text
delta = max_delta * tanh(raw_delta)

x1' = x1 + delta_l * w
y1' = y1 + delta_t * h
x2' = x2 + delta_r * w
y2' = y2 + delta_b * h
```

이후 image boundary로 clip한다.

```text
b' = clip_box((x1', y1', x2', y2'), image_shape)
```

## 8. Loss Function

전체 DCLR objective는 다음과 같다.

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

threshold-crossing loss가 이 방법의 핵심 term이다.

```text
L_cross = max(0, tau_iou + margin - IoU(b', g))
```

이는 원하는 state transition을 직접 최적화한다.

```text
FN_LOC -> TP
```

## 9. Transition-Aware Weighting

서로 다른 transition에 동일한 loss weight를 줄 필요는 없다.

권장 초기 weighting:

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

근거:

- `FN_LOC->FN_LOC`에는 더 강한 edge correction이 필요하다.
- `TP->FN_LOC`은 ranking 또는 quality mismatch일 수 있으므로 quality calibration이 더 중요하다.
- `TP->TP`는 repair head가 stable TP를 망가뜨리지 않도록 preservation sample로 사용할 수 있다.

## 10. Training Procedure

### 10.1 Epoch-End Mining

epoch 종료 시:

1. train set 전체에 대해 DHM mining을 실행한다.
2. 각 GT에 대해 state, transition, best same-class candidate, candidate IoU, candidate score를 기록한다.
3. `FN_LOC` record에 대해 counterfactual residual/action target을 계산한다.
4. 선택된 repair metadata를 DHM 또는 sidecar repair memory에 저장한다.

### 10.2 Next Epoch Training

다음 training epoch 동안:

1. FCOS가 일반 detection loss를 계산한다.
2. DCLR은 각 target GT에 대한 DHM record를 읽는다.
3. eligible `FN_LOC->FN_LOC`, `TP->FN_LOC` record를 선택한다.
4. candidate positive point 또는 dense prediction을 decode한다.
5. border/geometry/score feature를 만든다.
6. repair head가 residual과 quality를 예측한다.
7. DCLR loss를 일반 FCOS loss에 더한다.

### 10.3 Candidate Source Options

#### Dense Positive Points

assigned FCOS positive point를 사용한다.

장점:

- training forward에 통합하기 쉽다.
- FCOS assignment와 직접 연결된다.
- 계산 비용이 낮다.

단점:

- inference-time ranking/NMS behavior를 완전히 반영하지 못할 수 있다.

#### Pre-NMS Detection Candidates

score filtering 이후, NMS 이전의 dense prediction candidate를 사용한다.

장점:

- 실제 inference failure에 더 가깝다.
- quality calibration에 더 적합하다.

단점:

- 계산 비용이 더 크다.
- candidate matching이 더 복잡하다.

권장 MVP:

```text
dense positive point로 시작한다.
training-only loss가 안정화된 뒤 pre-NMS candidate로 확장한다.
```

## 11. Inference Procedure

DHM record는 train-set specific하므로 validation/test image에 직접 사용할 수 없다. 따라서 inference에서는 학습된 repair/risk head를 사용한다.

제안 inference:

1. 일반 FCOS forward를 실행한다.
2. class score와 centerness로 pre-NMS candidate를 만든다.
3. rescue candidate를 선택한다.

```text
candidate if:
  class_score >= tau_cls
  and (
    centerness <= tau_ctr
    or predicted_iou_quality <= tau_quality
    or predicted_rescue_score >= tau_rescue
  )
```

4. 선택된 candidate에만 DCLR repair를 적용한다.
5. refined score를 다시 계산한다.

```text
score_refined = class_score * predicted_iou_quality
```

또는:

```text
score_refined = class_score * sqrt(centerness * predicted_iou_quality)
```

6. original box와 refined box를 함께 NMS한다.

안전한 초기 설정:

```yaml
inference:
  enabled: true
  keep_original_boxes: true
  rescue_topk: 100
  max_delta: 0.25
  class_score_threshold: 0.05
  rescue_score_threshold: 0.3
```

초기 실험에서는 `keep_original_boxes: true`가 중요하다. repair prediction이 나쁘더라도 original candidate를 제거하지 않기 때문이다.

## 12. Expected Benefits

DCLR은 `FN_LOC` candidate가 TP boundary 근처에 있을 때 효과가 클 것으로 예상된다.

주요 기대 효과:

- final DHM `FN_LOC` count 감소;
- `FN_LOC->TP` recovery 증가;
- `FN_LOC->FN_LOC` persistence 감소;
- mAP75 상승;
- mAP50:95 상승;
- rescued box가 NMS에서 살아남는 경우 mAR100 상승.

대부분의 `FN_LOC`이 IoU가 매우 낮고 작은 edge intervention으로 repair될 수 없다면 효과는 제한적일 수 있다.

## 13. 구현 전 진단

전체 구현 전에 DHM log와 저장된 prediction으로 다음 진단을 수행한다.

1. FN_LOC best same-class IoU histogram.
2. IoU 구간별 FN_LOC candidate 비율:

```text
[0.0, 0.1)
[0.1, 0.3)
[0.3, 0.5)
[0.5, 1.0)
```

3. Counterfactual crossing rate:

```text
max_delta <= 0.25에서 tau_iou를 넘길 수 있는 FN_LOC candidate 비율
```

4. Persistent `FN_LOC->FN_LOC` record의 edge residual direction consistency.
5. class, FPN level, object scale별 residual variance.
6. persistent FN_LOC과 relapse FN_LOC의 residual pattern 차이.
7. centerness, predicted IoU quality, actual IoU 사이의 correlation.

많은 FN_LOC candidate가 IoU 0.3~0.5 구간에 있고 작은 delta로 threshold를 넘을 수 있다면, DCLR은 AP75와 mAP50:95를 개선할 가능성이 높다.

## 14. Minimal Implementation Plan

### Phase 0: Diagnostics

- DHM mining 중 각 FN_LOC GT에 대한 best same-class candidate box를 저장한다.
- edge residual과 crossing feasibility를 계산한다.
- histogram을 `history.json` 또는 sidecar JSON에 기록한다.

### Phase 1: Training-Only Residual Repair

- Implementation status: disabled-by-default `dhmr.counterfactual_repair`로 `modules/cfg/dhmr.yaml`에 구현됨.
- DCLR config를 추가한다.
- DHM record 기반 repair candidate selection을 추가한다.
- dense positive point를 사용한다.
- residual, GIoU, crossing, IoU-quality loss를 추가한다.
- inference는 아직 변경하지 않는다.

목표:

```text
repair head가 FCOS training을 불안정하게 만들지 않고 의미 있는 residual을 학습하는지 확인
```

### Phase 2: Transition-Aware DCLR

- transition embedding을 추가한다.
- transition-specific loss weight를 추가한다.
- persistent와 relapse statistics를 분리한다.

목표:

```text
FN_LOC->FN_LOC과 TP->FN_LOC이 서로 다른 supervision emphasis에서 이득을 보는지 확인
```

### Phase 3: Inference-Time Selective Repair

- top-K selected candidate에 repair를 적용한다.
- original box를 유지한다.
- predicted IoU quality로 refined box score를 계산한다.
- joint NMS를 수행한다.

목표:

```text
training-time repair 능력을 validation mAP/mAR 개선으로 연결
```

### Phase 4: Counterfactual Action Head

- discrete action label을 추가한다.
- action classification loss를 추가한다.
- residual-only와 residual-plus-action을 비교한다.

목표:

```text
명시적인 edge-action supervision이 repair generalization을 개선하는지 검증
```

## 15. Ablation Plan

| Experiment | Description |
|---|---|
| Baseline | FCOS + DHM logging only |
| Hard Replay | DHM-guided full-image replay |
| FN_LOC Crop Repair | FN_LOC에 대한 DHM-guided crop replay |
| Generic Refinement | DHM selection 없이 refinement head 사용 |
| DHM Selective Refinement | DHM-selected FN_LOC candidate만 refine |
| DCLR without Crossing Loss | residual + GIoU + quality만 사용 |
| DCLR with Crossing Loss | threshold-crossing objective 추가 |
| DCLR without Temporal Memory | 현재 FN_LOC만 사용 |
| DCLR with Temporal Memory | EMA edge residual 사용 |
| Transition-Agnostic DCLR | 모든 FN_LOC transition을 동일하게 처리 |
| Transition-Aware DCLR | FN_LOC->FN_LOC과 TP->FN_LOC 분리 |
| Residual Only | continuous residual만 예측 |
| Residual + Action | residual과 discrete edge action 예측 |
| Training-Only | auxiliary loss만 추가하고 inference repair 없음 |
| Inference Repair | inference에서 selective repair 적용 |

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
- NMS 이후 prediction 수;
- object scale별 AP.

## 16. Risks

주요 위험:

- 선택된 FN_LOC candidate가 GT와 충분히 가깝지 않으면 repair가 box를 hallucinate할 수 있다.
- `max_delta`가 너무 크면 refined box가 false positive를 만들 수 있다.
- quality calibration이 부정확하면 NMS ranking이 악화될 수 있다.
- inference selection이 너무 넓으면 stable TP가 손상될 수 있다.
- 대부분의 FN_LOC이 counterfactually crossable하지 않다면 성능이 개선되지 않을 수 있다.

완화 방법:

- 초기에는 `max_delta`를 작게 둔다.
- `keep_original_boxes: true`를 사용한다.
- `rescue_topk`를 제한한다.
- stable `TP->TP` preservation sample을 포함한다.
- method가 작동해야 한다고 주장하기 전에 crossing feasibility를 보고한다.

## 17. Novelty Claim

제안 claim:

```text
We propose DHM-Guided Counterfactual Localization Repair, a selective localization rescue method
that converts temporally persistent and relapse localization failures into counterfactual edge
intervention targets. Instead of only replaying hard images or globally refining all detections,
DCLR learns the minimal box repair needed to move DHM-identified FN_LOC candidates across the TP IoU
threshold.
```

짧은 버전:

```text
DCLR uses temporal GT-level failure memory to generate counterfactual localization actions for
FN_LOC candidates, directly optimizing the state transition from FN_LOC to TP.
```

한국어 버전:

```text
DCLR은 GT-level temporal failure memory를 사용해 FN_LOC candidate에 대한 counterfactual
localization action을 생성하고, FN_LOC에서 TP로 넘어가는 state transition 자체를 직접 최적화하는
selective localization repair 방법이다.
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
