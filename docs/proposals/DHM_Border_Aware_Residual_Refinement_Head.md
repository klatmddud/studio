# DHM-Guided Border-Aware Residual Refinement Head

## 1. Motivation

`runs/DHM-ResNet18`와 `runs/DHM-ResNet50` 분석 결과에서 FN_LOC는 positive assignment 부족보다는 localization quality 문제에 가깝다.

핵심 관찰은 다음과 같다.

- FN_LOC GT도 평균 7개 이상의 FCOS positive point를 가진다.
- FN_LOC의 zero-positive 비율은 매우 낮다.
- FN_LOC는 TP보다 `box loss`가 크게 높고, `centerness target`은 낮다.
- `FN_LOC -> FN_LOC` transition 비율이 높아, 반복적으로 localization 실패를 겪는 GT가 존재한다.
- `TP -> FN_LOC`는 이전에는 검출되던 GT가 localization 문제로 relapse된 경우다.

따라서 단순히 positive를 더 만들거나 hard GT에 loss weight를 주는 방식보다, 이미 생성된 detection candidate의 box boundary를 직접 보정하고, refined box의 localization quality를 재평가하는 방식이 더 적합하다.

## 2. Core Idea

Border-Aware Residual Refinement Head는 FCOS가 생성한 candidate box 중 FN_LOC 위험이 높은 box만 골라 추가 보정하는 selective rescue module이다.

기본 detector는 그대로 둔다.

```text
FCOS backbone + FPN
        |
   FCOS head
        |
 initial boxes, class scores, centerness
        |
 FN_LOC-risk selector
        |
 border-aware residual refinement head
        |
 refined boxes, IoU/quality scores
        |
 rescoring + NMS
```

이 방법은 모든 box에 refinement를 적용하지 않는다. DHM이 기록한 GT-level transition을 이용해 다음 두 유형을 우선 rescue 대상으로 삼는다.

- `FN_LOC -> FN_LOC`: persistent localization failure
- `TP -> FN_LOC`: localization relapse

## 3. Relation to Prior Work

이 방법은 기존 localization refinement 및 localization quality estimation 연구와 연결된다.

- FCOS는 center sampling과 centerness로 anchor-free dense detection을 구성한다.
- Cascade R-CNN은 stage-wise refinement로 high-IoU detection을 만든다.
- IoU-Net, GFLV2, VarifocalNet, PAA는 localization quality를 score에 반영한다.
- BorderDet과 VarifocalNet은 border/star-shaped feature를 활용해 box refinement를 강화한다.
- TOOD는 classification과 localization의 spatial misalignment를 줄인다.

차별점은 다음이다.

기존 방법은 대부분 모든 sample 또는 모든 detection에 전역적으로 적용된다. 반면 이 proposal은 DHM의 temporal transition memory를 사용해 반복적 FN_LOC와 relapse FN_LOC를 구분하고, 해당 risk candidate에만 border-aware refinement와 quality calibration을 적용한다.

즉 novelty는 border feature 자체가 아니라, `DHM transition -> FN_LOC-risk routing -> selective border refinement -> quality recalibration`의 연결 구조에 있다.

## 4. Target Failure Modes

### 4.1 Persistent FN_LOC: `FN_LOC -> FN_LOC`

같은 GT가 여러 epoch 동안 계속 FN_LOC로 남는 경우다.

가능한 원인은 다음과 같다.

- box edge가 특정 방향으로 계속 치우침
- point feature만으로 object boundary를 충분히 표현하지 못함
- class evidence는 있지만 localization regression이 불안정함
- predicted box가 IoU threshold 근처에서 반복적으로 탈락함

이 경우에는 border feature 기반 residual correction이 핵심이다.

### 4.2 Relapse FN_LOC: `TP -> FN_LOC`

이전 epoch에서는 TP였지만 이후 FN_LOC로 떨어진 경우다.

가능한 원인은 다음과 같다.

- localization forgetting
- box score와 localization quality의 ranking mismatch
- NMS에서 더 낮은 quality의 box가 선택됨
- class confidence는 유지되지만 box quality가 흔들림

이 경우에는 box refinement와 함께 IoU/centerness quality calibration이 중요하다.

## 5. Architecture

### 5.1 FN_LOC-Risk Selector

추론 시 모든 detection에 refinement를 적용하면 기존 stable TP까지 손상될 수 있다. 따라서 먼저 rescue candidate를 선택한다.

입력 후보는 FCOS postprocess 전의 dense candidate 또는 post-NMS 전 candidate를 사용할 수 있다.

추천 selector 조건은 다음과 같다.

```text
candidate if:
  class_score >= tau_cls
  and (
    centerness <= tau_ctr
    or predicted_iou_quality <= tau_iou_quality
    or risk_score >= tau_risk
  )
```

MVP에서는 별도 risk head 없이 다음 휴리스틱으로 시작할 수 있다.

```text
class_score high, centerness medium/low, box size valid, top-K per image
```

이후 DHM transition label을 사용해 `q_rescue` risk head를 학습한다.

### 5.2 Border Feature Sampler

초기 predicted box를 다음과 같이 둔다.

```text
b = (x1, y1, x2, y2)
w = x2 - x1
h = y2 - y1
```

각 box의 네 변에서 K개 point를 샘플링한다.

```text
left edge:   (x1, y_i)
right edge:  (x2, y_i)
top edge:    (x_i, y1)
bottom edge: (x_i, y2)
```

예를 들어 `K=5`이면 box 하나당 20개의 border point를 사용한다.

각 point는 해당 FPN level feature map에서 bilinear sampling으로 feature를 추출한다. 그 뒤 edge별로 pooling한다.

```text
f_left   = pool(sample(left edge))
f_right  = pool(sample(right edge))
f_top    = pool(sample(top edge))
f_bottom = pool(sample(bottom edge))
f_center = sample(box center)
```

최종 refinement 입력은 다음처럼 구성한다.

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
  optional_transition_embedding
)
```

`box_geometry`에는 width, height, aspect ratio, area, normalized FCOS distances 등을 포함할 수 있다.

### 5.3 Residual Box Head

head는 absolute box를 새로 예측하지 않고 edge-wise residual을 예측한다.

```text
delta = (delta_l, delta_t, delta_r, delta_b)
```

refined box는 다음처럼 계산한다.

```text
x1' = x1 + delta_l * w
y1' = y1 + delta_t * h
x2' = x2 + delta_r * w
y2' = y2 + delta_b * h
```

과도한 보정을 막기 위해 residual은 제한한다.

```text
delta = max_delta * tanh(raw_delta)
```

추천 초기값:

```yaml
max_delta: 0.25
border_points_per_side: 5
hidden_dim: 256
num_layers: 2
```

### 5.4 Quality Calibration Head

box만 보정해도 NMS ranking에서 살아남지 못하면 recall rescue가 되지 않는다. 따라서 refined box의 quality도 함께 예측한다.

출력은 다음과 같다.

```text
q_iou: refined box의 predicted IoU
q_ctr: refined centerness-like quality
q_rescue: FN_LOC rescue 대상일 확률
```

최종 score는 다음 중 하나로 계산한다.

```text
score_refined = class_score * q_iou
score_refined = class_score * sqrt(q_iou * q_ctr)
score_refined = class_score * centerness * q_rescue
```

MVP에서는 `class_score * q_iou`를 우선 사용한다.

## 6. Training Target

초기 predicted box `b`와 matched GT box `g`가 있을 때 residual target은 다음과 같다.

```text
delta_l* = (g_x1 - b_x1) / w
delta_t* = (g_y1 - b_y1) / h
delta_r* = (g_x2 - b_x2) / w
delta_b* = (g_y2 - b_y2) / h
```

refined box는 predicted residual을 적용해 얻는다.

```text
b' = refine(b, delta)
```

기본 loss는 다음과 같다.

```text
L = lambda_giou * GIoU(b', g)
  + lambda_residual * SmoothL1(delta, delta*)
  + lambda_iou * BCE(q_iou, IoU(b', g))
  + lambda_rescue * BCE(q_rescue, rescue_label)
```

`rescue_label`은 DHM transition으로 정의한다.

```text
rescue_label = 1 for FN_LOC -> FN_LOC
rescue_label = 1 for TP -> FN_LOC
rescue_label = 0 for stable TP -> TP
```

transition별 weighting은 다르게 둔다.

```yaml
transition_weights:
  FN_LOC->FN_LOC:
    giou: 2.0
    residual: 2.0
    iou_quality: 1.0
    rescue: 1.0
  TP->FN_LOC:
    giou: 1.0
    residual: 1.0
    iou_quality: 2.0
    rescue: 2.0
  TP->TP:
    giou: 0.25
    residual: 0.25
    iou_quality: 0.5
    rescue: 1.0
```

## 7. Data Construction

학습 데이터는 DHM mining 이후 생성한다.

1. Epoch 종료 후 DHM이 GT별 state와 transition을 기록한다.
2. 다음 training forward에서 각 GT에 대응되는 FCOS candidate를 수집한다.
3. candidate와 GT를 class, image_id, annotation_id 또는 IoU로 매칭한다.
4. transition이 `FN_LOC->FN_LOC` 또는 `TP->FN_LOC`이면 rescue positive로 사용한다.
5. stable `TP->TP`는 negative 또는 preservation sample로 사용한다.

candidate 선택 방법은 두 가지가 가능하다.

### Dense-point candidate

FCOS positive point의 raw box prediction을 사용한다.

장점:

- assignment와 직접 연결된다.
- training forward에서 계산하기 쉽다.

단점:

- 실제 NMS ranking 문제를 완전히 반영하지 못할 수 있다.

### Detection candidate

postprocess 전 top-K candidate 또는 post-NMS 전 candidate를 사용한다.

장점:

- 실제 추론 failure와 더 가깝다.
- score calibration과 NMS 개선에 직접 연결된다.

단점:

- 구현이 더 복잡하다.
- 학습 중 candidate 수가 많아질 수 있다.

MVP는 dense-point candidate로 시작하고, 이후 detection candidate로 확장한다.

## 8. Inference Procedure

추론 절차는 다음과 같다.

1. 기존 FCOS forward로 dense predictions를 생성한다.
2. class score와 centerness로 top-K candidate를 만든다.
3. FN_LOC-risk selector로 rescue 대상만 선택한다.
4. 선택된 box의 border feature를 FPN에서 샘플링한다.
5. residual head로 refined box를 생성한다.
6. quality head로 `q_iou`, `q_rescue`를 예측한다.
7. refined score로 detection score를 재계산한다.
8. 기존 candidate와 refined candidate를 함께 NMS한다.

추천 초기 설정:

```yaml
inference:
  enabled: true
  pre_nms_topk: 1000
  rescue_topk: 100
  class_score_threshold: 0.05
  rescue_score_threshold: 0.3
  max_delta: 0.25
  keep_original_boxes: true
```

`keep_original_boxes`를 true로 두면 refined box가 실패하더라도 원래 box 후보를 유지할 수 있다. recall rescue 관점에서는 초기 실험에서 이 설정이 안전하다.

## 9. Implementation Plan

### 9.1 New Module

추가 후보 파일:

```text
modules/nn/dhm_border_refine.py
modules/cfg/dhm_border_refine.yaml
```

주요 class:

```python
class DHMBorderRefineConfig:
    ...

class BorderFeatureSampler(nn.Module):
    ...

class DHMBorderResidualRefinementHead(nn.Module):
    ...
```

### 9.2 FCOS Wrapper Integration

수정 후보 파일:

```text
models/detection/wrapper/fcos.py
```

필요한 hook:

- training forward에서 rescue candidate 수집
- FPN feature와 candidate box를 refinement head에 전달
- refinement loss를 기존 loss dict에 추가
- evaluation/inference path에서 rescue refinement 적용

loss dict 예시:

```python
losses["dhm_border_refine_giou"] = ...
losses["dhm_border_refine_residual"] = ...
losses["dhm_border_refine_quality"] = ...
losses["dhm_border_refine_rescue"] = ...
```

### 9.3 DHM State Extension

기존 DHM에는 transition과 assignment statistics가 있으므로, 추가로 다음 정보를 저장하면 좋다.

```text
last_pred_box
last_matched_iou
last_box_residual
ema_residual_ltrb
ema_edge_error_ltrb
```

이 정보는 persistent FN_LOC의 systematic edge bias를 분석하고, 이후 temporal localization prior로 확장하는 데 사용할 수 있다.

### 9.4 Runtime and Checkpoint

수정 후보:

```text
scripts/runtime/registry.py
scripts/runtime/engine.py
docs/modules.md
docs/training.md
```

필요 작업:

- module config loading 추가
- checkpoint extra state 저장 여부 결정
- `history.json`에 border refinement summary 기록
- docs 업데이트

## 10. Minimal MVP

처음부터 전체 구조를 구현하지 말고 다음 순서로 진행한다.

### Phase 1: Training-only refinement loss

- Implementation status: implemented as disabled-by-default `dhmr.border_refinement` in `modules/cfg/dhmr.yaml`.
- dense positive point 기반 candidate 사용
- `FN_LOC->FN_LOC`, `TP->FN_LOC`만 positive rescue sample로 사용
- border feature sampler 구현
- single border residual head + IoU quality head 구현
- residual + GIoU + IoU-quality auxiliary loss 추가
- inference에는 아직 적용하지 않음

목표:

- refinement head가 FN_LOC box residual을 학습할 수 있는지 확인
- quality head가 refined box IoU를 예측할 수 있는지 확인
- training loss가 안정적으로 감소하는지 확인

### Phase 2: Inference-time selective refinement

- top-K candidate에만 refinement 적용
- refined box와 original box를 함께 NMS
- `class_score * q_iou` scoring 적용

목표:

- FN_LOC 감소
- mAR 증가
- mAP75 증가 여부 확인

### Phase 3: Quality calibration

- `q_iou`, `q_rescue` head 추가
- stable TP preservation sample 추가
- score fusion ablation 수행

목표:

- false positive 증가 억제
- 기존 TP degradation 완화

### Phase 4: Transition-specific expert

- `FN_LOC->FN_LOC` expert와 `TP->FN_LOC` expert 분리
- 또는 transition embedding 기반 shared head 사용

목표:

- persistent localization failure와 relapse localization failure의 대응 분리

## 11. Ablation Plan

필수 ablation은 다음과 같다.

| Experiment | Description |
|---|---|
| Baseline | FCOS + DHM logging only |
| Global refine | 모든 candidate에 border refinement 적용 |
| Selective refine | FN_LOC-risk candidate에만 refinement 적용 |
| No border feature | center feature만 사용 |
| Border feature | center + border feature 사용 |
| No quality head | refined box만 사용 |
| IoU quality head | `class_score * q_iou` 사용 |
| Transition-agnostic | FN_LOC 전체를 하나로 처리 |
| Transition-aware | `FN_LOC->FN_LOC`, `TP->FN_LOC` 분리 |

주요 metric:

- COCO mAP50:95
- mAP75
- mAR100
- DHM final `FN_LOC` count
- DHM `FN_LOC->FN_LOC` transition count
- DHM `TP->FN_LOC` transition count
- 기존 stable TP의 degradation rate
- per-class FN_LOC 변화

## 12. Expected Outcome

기대 효과는 다음과 같다.

- FN_LOC 감소
- mAP75 상승
- mAR100 상승
- persistent FN_LOC GT의 일부 rescue
- class confidence는 있지만 IoU가 낮아 탈락하던 candidate의 회복

주의할 점은 다음이다.

- rescue candidate를 너무 많이 refine하면 false positive가 증가할 수 있다.
- quality calibration이 부정확하면 NMS ranking이 악화될 수 있다.
- stable TP에 refinement가 적용되면 box가 망가질 수 있다.
- small object에서는 border sampling이 noisy할 수 있다.

따라서 초기 실험은 `keep_original_boxes=true`, `rescue_topk` 제한, `max_delta` 제한을 두고 진행하는 것이 좋다.

## 13. Novelty Claim

제안 방법의 novelty는 다음 문장으로 정리할 수 있다.

> We propose a DHM-guided selective localization rescue module that uses GT-level temporal failure transitions to identify persistent and relapse localization failures, then applies border-aware residual refinement and localization quality recalibration only to FN_LOC-risk detections.

한국어로는 다음과 같다.

> 본 방법은 DHM이 기록한 GT-level temporal transition을 이용해 반복적 FN_LOC와 relapse FN_LOC를 식별하고, 해당 위험 candidate에만 border-aware residual refinement와 localization quality recalibration을 선택적으로 적용하는 localization rescue 방법론이다.

## 14. References

- [FCOS: Fully Convolutional One-Stage Object Detection](https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html)
- [Cascade R-CNN: Delving Into High Quality Object Detection](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html)
- [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.html)
- [VarifocalNet: An IoU-aware Dense Object Detector](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf)
- [BorderDet: Border Feature for Dense Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460528.pdf)
- [TOOD: Task-Aligned One-Stage Object Detection](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html)
- [Probabilistic Anchor Assignment with IoU Prediction for Object Detection](https://link.springer.com/chapter/10.1007/978-3-030-58595-2_22)
