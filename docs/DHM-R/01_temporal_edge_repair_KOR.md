# Temporal Edge Repair Module

## 개요

Temporal Edge Repair Module은 DHM-R의 `FN_LOC` 전용 보정 모듈이다. DHM mining에서
`FN_LOC`로 판정된 GT는 모델이 정답 class 신호는 냈지만, box IoU가 TP 기준을 넘지 못한
case다. 따라서 이 모듈의 목표는 classification confidence를 더 키우는 것이 아니라, 반복적으로
틀리는 box boundary를 시간축으로 추적하고 localization feature가 해당 edge를 더 잘 보정하도록
training-only auxiliary task를 주는 것이다.

핵심 아이디어는 다음과 같다.

- DHM이 GT별 `FN_LOC` 이력을 제공한다.
- 각 `FN_LOC` GT에 대해 예측 box와 GT box의 edge 오차를 `left`, `top`, `right`, `bottom`
  단위로 분해한다.
- epoch 간 반복되는 edge 오차 방향과 크기를 memory에 저장한다.
- training 중 hard GT 주변 positive point 또는 후보 point에서 edge repair branch가 보정 방향을
  예측하도록 학습한다.
- inference graph는 유지하거나, 선택적으로 lightweight refine head를 남길 수 있다.

## 문제 정의

DHM 기준에서 `FN_LOC`는 다음 조건에 해당한다.

- 같은 class prediction score가 충분히 존재한다.
- 하지만 high-confidence same-class prediction의 IoU가 `tau_iou`보다 낮다.

즉 모델은 물체와 class를 완전히 놓친 것이 아니라, box 위치나 크기를 잘못 회귀한 것이다. HLAE
결과처럼 `AP50`은 높지만 `AP75`가 낮고, current failure 중 `FN_LOC` 비중이 크면 localization
quality가 주요 병목이라고 해석할 수 있다.

일반적인 reweighting은 `FN_LOC` GT의 box loss를 더 크게 만들 수는 있지만, 어떤 edge가 왜 반복적으로
틀리는지 알려주지 않는다. Temporal Edge Repair Module은 hard GT마다 누적된 edge error pattern을
명시적으로 학습 신호로 바꾼다.

## 모듈 구성

### 1. Temporal Edge Memory

GT 단위로 edge 오차 이력을 저장한다.

```text
EdgeRecord
  gt_uid
  image_id
  class_id
  last_state
  total_seen
  fn_loc_count
  last_iou
  ema_iou
  edge_error_ema: [l, t, r, b]
  edge_error_abs_ema: [l, t, r, b]
  dominant_edge
  edge_flip_count
  last_seen_epoch
```

edge 오차는 GT box와 best same-class candidate box를 비교해서 계산한다.

```text
e_l = pred_x1 - gt_x1
e_t = pred_y1 - gt_y1
e_r = pred_x2 - gt_x2
e_b = pred_y2 - gt_y2
```

부호는 보정 방향을 보존하기 위해 유지한다. 예를 들어 `e_r < 0`이면 오른쪽 edge가 GT보다 안쪽에
있어서 box가 좁게 예측된 것이다.

### 2. Edge Repair Head

FCOS head feature에서 training-only edge correction을 예측한다.

입력:

- matched positive point의 FPN feature
- optional: point-to-GT geometry embedding
- optional: DHM instability scalar
- optional: dominant edge one-hot

출력:

```text
delta_edge: [dl, dt, dr, db]
edge_confidence: [cl, ct, cr, cb]
```

`delta_edge`는 현재 predicted box를 GT 방향으로 보정하기 위한 normalized edge correction이다.
`edge_confidence`는 어떤 edge를 신뢰하고 학습할지 조절하는 auxiliary confidence다.

### 3. Hard GT Selector

모든 GT에 적용하지 않고 DHM memory에서 다음 조건을 만족하는 GT만 대상으로 한다.

```text
last_state == FN_LOC
total_seen >= min_observations
fn_loc_count >= min_fn_loc_count
instability_score >= min_instability
```

이렇게 해야 easy GT의 정상적인 localization 학습을 방해하지 않는다.

## 학습 목표

### Edge Repair Loss

현재 box prediction `b_pred`와 GT `b_gt`의 edge residual을 target으로 사용한다.

```text
target_delta = normalize(b_gt - b_pred)
L_edge = SmoothL1(delta_edge, target_delta)
```

단, 모든 edge를 동일하게 학습하지 않고 Temporal Edge Memory의 dominant edge 또는 edge error magnitude로
masking한다.

```text
edge_weight_i = normalize(edge_error_abs_ema_i)
L_edge = sum_i edge_weight_i * SmoothL1(delta_i, target_delta_i)
```

### Consistency Loss

보정 후 box가 원래 box보다 GT에 가까워지도록 제한한다.

```text
b_repaired = apply_delta(b_pred, delta_edge)
L_consistency = max(0, IoU(b_pred, b_gt) - IoU(b_repaired, b_gt) + margin)
```

이 loss는 repair branch가 의미 없는 보정을 만들지 않도록 한다.

### Direction Agreement Loss

반복적으로 같은 방향으로 틀리는 edge에 대해 correction 방향을 맞춘다.

```text
sign(delta_edge_i) == sign(-edge_error_ema_i)
```

예를 들어 오른쪽 edge가 반복적으로 작게 예측되면 `delta_right`는 양수 방향이어야 한다.

## FCOS와의 통합

V0에서는 inference path를 바꾸지 않는 training-only 모듈로 시작한다.

1. epoch-end DHM mining이 `FN_LOC` GT를 기록한다.
2. 다음 epoch training forward에서 현재 mini-batch GT와 DHM record를 매칭한다.
3. `FN_LOC` hard GT에 matched된 positive point를 선택한다.
4. 해당 point feature에서 Edge Repair Head를 실행한다.
5. `L_edge`, `L_consistency`, `L_direction`을 base detection loss에 auxiliary loss로 더한다.

전체 loss는 다음처럼 둔다.

```text
L_total = L_fcos + lambda_edge * L_edge
        + lambda_consistency * L_consistency
        + lambda_direction * L_direction
```

기본 inference에서는 Edge Repair Head를 제거한다. 만약 실험적으로 효과가 크면, V1에서 lightweight
box refinement head로 inference에 남기는 변형을 별도 ablation으로 둔다.

현재 구현 상태:

- Config: `modules/cfg/dhmr.yaml`
- Module: `modules/nn/dhmr.py`
- FCOS loss key: `dhmr_edge`
- Dependency: `modules/cfg/dhm.yaml`의 DHM mining을 함께 켜야 한다.
- V0 구현은 이전 DHM record의 `last_state == FN_LOC` GT를 선택하고, 현재 training forward의 matched
  positive point에서 detached prediction-to-GT edge residual을 target으로 사용한다.
- 모듈 내부에는 GT별 temporal edge EMA가 저장되며, 반복적으로 큰 edge 오차를 보이는 dominant edge에
  더 큰 auxiliary supervision을 준다.

## DHM-R 내 역할

DHM-R은 failure type별로 서로 다른 repair path를 둔다.

- `FN_LOC`: Temporal Edge Repair Module
- `FN_CLS`: Confusion Prototype Memory Module
- `FN_BG`: Latent Foregroundness Branch

Temporal Edge Repair Module은 이 중 localization 전용 path다. 따라서 class confidence나 foreground
objectness를 직접 올리는 방식이 아니라, hard GT의 boundary geometry를 안정화하는 데 집중한다.

## Novelty 포인트

기존 box loss 개선은 대개 현재 mini-batch의 box와 GT만 본다. Temporal Edge Repair Module은 다음
차이가 있다.

- GT instance별 temporal failure memory를 사용한다.
- `FN_LOC`로 진단된 GT에만 localization auxiliary task를 적용한다.
- box 오차를 edge 단위로 분해하고, 반복적으로 틀리는 edge를 선택적으로 보정한다.
- inference 구조를 바꾸지 않고도 backbone/FPN/head에 boundary-sensitive representation을 주입할 수 있다.

따라서 단순 hard example mining이나 loss reweighting이 아니라, detection hysteresis를 이용한
failure-type-aware localization repair로 정의할 수 있다.

## 기대 효과

주요 개선 목표:

- `FN_LOC` count 감소
- `bbox_mAP_75` 증가
- `bbox_mAP_50_95` 증가
- `AP50` 대비 `AP75` gap 축소
- DHM `last_state_counts.FN_LOC` 감소
- DHM `dominant_failure_counts.FN_LOC` 감소

부수적으로 기대할 수 있는 변화:

- `mean_instability` 감소
- `relapses` 감소
- `state_changes` 감소

## Ablation 계획

최소 ablation은 다음 순서로 둔다.

| 실험 | 구성 | 목적 |
|---|---|---|
| Baseline | FCOS + DHM mining only | 기준선 |
| Edge loss only | `L_edge`만 추가 | edge regression auxiliary 효과 확인 |
| + dominant edge mask | 반복 edge만 학습 | temporal edge memory 효과 확인 |
| + consistency | repair 후 IoU 개선 constraint | 안정성 확인 |
| + direction agreement | correction 방향 supervision | edge 방향성 효과 확인 |
| inference refine V1 | repair head를 inference에 사용 | training-only 대비 상한 확인 |

## 실패 가능성과 점검 항목

### Capacity 부족

ResNet18 + FCOS head가 hard localization pattern을 담기에 부족할 수 있다. 이 경우 edge repair loss를
추가해도 train loss만 증가하고 validation AP75가 오르지 않을 수 있다. ResNet50 또는 head width/depth
증가 ablation이 필요하다.

### Noisy FN_LOC

`FN_LOC` 판정이 score threshold나 NMS 결과에 민감하면 잘못된 GT가 repair 대상이 될 수 있다.
`min_observations`, `min_instability`, `fn_loc_count`를 보수적으로 둔다.

### Easy GT 간섭

auxiliary branch가 전체 localization feature를 과도하게 바꾸면 easy GT AP가 떨어질 수 있다. hard GT
selector와 loss weight warmup이 필요하다.

### Annotation noise

box annotation 자체가 모호하면 edge repair가 오히려 overfit될 수 있다. `edge_error_abs_ema`가 큰데
IoU가 계속 낮은 persistent outlier는 loss cap 또는 ignore bucket으로 보낸다.

## V0 구현 스케치

```text
for each mining epoch:
  detections = model(train_images)
  for each GT:
    state = DHM.assign_detection_state(GT, detections)
    if state == FN_LOC:
      candidate = best_same_class_candidate(GT, detections)
      edge_error = candidate.box - GT.box
      edge_memory.update(gt_uid, edge_error, iou, score)

for each training batch:
  matched_idxs = FCOS.match_anchors_to_targets(...)
  hard_gt_records = edge_memory.lookup(batch_gt)
  selected_points = positives matched to hard FN_LOC GTs
  delta_edge, edge_conf = edge_repair_head(features[selected_points])
  losses["dhmr_edge"] = edge_repair_loss(delta_edge, hard_gt_records, current_boxes, gt_boxes)
```

## 기본 설정 초안

```yaml
dhmr:
  enabled: true
  temporal_edge_repair:
    enabled: true
    min_observations: 3
    min_fn_loc_count: 2
    min_instability: 0.25
    use_dominant_edge_mask: true
    edge_ema_momentum: 0.8
    loss:
      edge_weight: 0.25
      consistency_weight: 0.1
      direction_weight: 0.05
      warmup_epochs: 2
      max_gt_per_image: 16
```

## 한 줄 요약

Temporal Edge Repair Module은 DHM이 찾아낸 `FN_LOC` hard GT에 대해 반복적으로 틀리는 box edge를
기억하고, training-only edge correction task로 localization representation을 보정하는 DHM-R의
localization repair path다.
