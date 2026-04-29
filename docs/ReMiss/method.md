# ReMiss: Recurrent Missed-Object Memory for Spatial Prototype Injection

## 개요

ReMiss는 객체 검출 모델이 반복적으로 놓치는 Ground Truth(GT)를 기억하고, 해당 실패 패턴을 이용해 detector feature를 공간적으로 보정하는 방법론이다. 핵심 목표는 단순히 어려운 샘플의 loss를 키우는 것이 아니라, 연속 미검출되는 객체의 공간적 분포를 학습 신호로 사용해 모델이 취약한 이미지 영역을 인식하고 해당 영역의 feature 표현을 보강하도록 만드는 것이다.

기본 설정은 이미지를 `2x2` 영역으로 나누는 coarse spatial prior에서 시작한다. 이후 동일한 구조를 `4x4`, `8x8` 등 더 세밀한 grid로 확장할 수 있도록 설계한다.

## 핵심 아이디어

1. 학습 중 각 GT 객체가 연속으로 미검출되는지 memory module에 기록한다.
2. 미검출 GT가 이미지 내 어느 공간 영역에 주로 위치하는지 계산한다.
3. detector head에 미검출이 집중되는 영역을 예측하는 auxiliary head를 추가한다.
4. auxiliary head가 예측한 영역에 대응하는 learnable embedding prototype을 backbone 또는 neck feature에 additive 방식으로 주입한다.
5. 초기 실험은 hard label, hard injection, additive injection으로 시작하고, 이후 soft label, soft injection, gated additive로 확장한다.

## Memory 정의

Memory는 이미지와 GT 단위의 반복 실패 이력을 저장한다. 기본적으로 다음 정보를 가진다.

| 필드 | 설명 |
|---|---|
| `image_id` | 이미지 식별자 |
| `gt_id` | 이미지 내 GT 객체 식별자 |
| `gt_class` | GT 클래스 |
| `miss_count` | 해당 GT가 연속으로 미검출된 횟수 |
| `region_id` | GT가 가장 많이 걸쳐 있는 grid 영역 |
| `last_state` | 직전 평가에서의 검출 또는 미검출 상태 |

`miss_count`는 GT가 현재 epoch 또는 iteration의 평가에서 미검출되면 증가하고, 정상 검출되면 초기화한다. 이렇게 하면 일시적인 실패보다 반복적으로 회복되지 않는 실패를 더 강하게 구분할 수 있다.

## 미검출 판정

GT 객체는 detector의 최종 예측 결과를 기준으로 검출 여부를 판정한다. 최종 예측은 일반적으로 score threshold, NMS, detector별 post-processing이 적용된 결과를 의미한다.

GT가 검출된 것으로 보려면 다음 조건을 만족하는 예측 box가 존재해야 한다.

1. 예측 class가 GT class와 일치한다.
2. 예측 score가 detector의 최종 검출 threshold 이상이다.
3. 예측 box와 GT box의 IoU가 GT 매칭 기준 이상이다.

위 조건을 만족하는 예측이 없으면 해당 GT는 현재 상태에서 미검출로 판정한다.

## 공간 영역 라벨

기본 baseline은 `2x2` grid를 사용한다. 라벨은 5-way hard label이다.

| Label | 의미 |
|---|---|
| `0` | none, 학습 target 없음 |
| `1` | 좌상단 |
| `2` | 우상단 |
| `3` | 좌하단 |
| `4` | 우하단 |

GT box가 여러 영역에 걸쳐 있으면 각 영역과 GT box의 겹치는 면적을 계산하고, 가장 많이 겹치는 영역을 `region_id`로 선택한다.

향후 `NxN` grid로 확장할 경우 label 공간은 `0`부터 `N^2`까지 확장한다. `0`은 none으로 유지하고, `1..N^2`는 row-major 순서로 공간 영역을 나타낸다.

## 학습 타깃 선택

Auxiliary region head의 학습 target은 모든 GT에서 만들지 않는다. 현재 상태가 미검출이고, 연속 미검출 횟수가 threshold 이상인 GT만 target으로 사용한다.

조건은 다음과 같다.

```text
is_target(gt) =
    current_state(gt) == missed
    and miss_count(gt) >= miss_threshold
```

이미지 내에 조건을 만족하는 GT가 없으면 해당 이미지의 region label은 `0`으로 둔다. 조건을 만족하는 GT가 여러 개 있으면 region별 target score를 누적한 뒤 가장 큰 region을 hard label로 선택한다. 초기 버전에서는 hard label만 사용한다.

예시:

```text
region_score[q] = sum(miss_count(gt) for gt in target_gts if region_id(gt) == q)
target_region = argmax_q region_score[q]
```

## Auxiliary Region Head

Detector head에 미검출 취약 영역을 예측하는 auxiliary head를 추가한다. 이 head는 이미지 또는 feature map 단위로 `0..4` 중 하나를 예측한다.

초기 버전의 손실 함수는 softmax cross entropy를 사용한다.

```text
L_region = CE(region_logits, target_region)
```

최종 학습 loss는 detector 기본 loss와 region loss를 가중합한다.

```text
L_total = L_detector + lambda_region * L_region
```

Region head와 prototype injection module은 `start_epoch` 이후부터 학습에 참여한다. Memory update는 학습 초기부터 계속 수행할 수 있지만, auxiliary loss와 injection은 warm-up 이후에 켜서 초기 불안정한 detector 예측이 module 학습을 오염시키는 문제를 줄인다.

## Learnable Region Prototype

각 spatial region에는 학습 가능한 embedding prototype을 하나씩 둔다. `2x2` baseline에서는 네 개의 prototype을 사용한다.

```text
P_1: 좌상단 prototype
P_2: 우상단 prototype
P_3: 좌하단 prototype
P_4: 우하단 prototype
```

`0: none`은 injection을 수행하지 않는 상태로 둔다. 필요하면 향후 none prototype을 따로 둘 수 있지만, 초기 실험에서는 none일 때 feature를 변경하지 않는다.

## Hard Injection

초기 버전은 hard injection을 사용한다. Region head의 예측값에서 argmax를 취해 하나의 영역을 선택하고, 해당 영역에 대응하는 prototype을 feature map의 같은 공간 영역에 additive 방식으로 주입한다.

```text
pred_region = argmax(region_logits)

if pred_region == 0:
    F_out = F
else:
    F_out = F + mask(pred_region) * P_pred_region
```

여기서 `F`는 backbone 또는 neck feature이고, `mask(pred_region)`은 선택된 grid 영역에만 값이 1인 spatial mask이다. Prototype은 channel dimension에 맞춰 broadcast된다.

Injection 위치는 ablation 대상으로 둔다.

| 위치 | 설명 |
|---|---|
| Backbone output | 더 이른 feature 표현을 보정 |
| Neck output | detector head 직전의 multi-scale feature를 보정 |
| 특정 FPN level | 작은 객체 또는 큰 객체에 특화된 보정 가능 |
| 전체 FPN level | 모든 scale에 동일한 region prior 주입 |

## 학습 절차

1. 기본 detector가 입력 이미지를 forward한다.
2. 최종 검출 결과를 GT와 매칭해 GT별 검출 또는 미검출 상태를 계산한다.
3. Memory의 `miss_count`, `region_id`, `last_state`를 업데이트한다.
4. `epoch < start_epoch`이면 기본 detector loss만 사용한다.
5. `epoch >= start_epoch`이면 memory에서 조건을 만족하는 GT를 찾아 5-way region target을 생성한다.
6. Auxiliary region head를 cross entropy로 학습한다.
7. Region head의 예측 영역에 대응하는 prototype을 feature에 hard additive injection한다.
8. Detector loss와 region loss를 함께 최적화한다.

## 추론 절차

추론 시에는 GT가 없으므로 memory update나 GT 기반 target 생성은 수행하지 않는다. Region head가 입력 이미지의 취약 영역을 직접 예측하고, 예측된 region에 해당하는 prototype을 feature에 주입한다.

추론 흐름은 다음과 같다.

1. 입력 이미지에서 backbone 또는 neck feature를 추출한다.
2. Region head가 `0..4` region logits를 예측한다.
3. Hard injection 설정에서는 argmax region을 선택한다.
4. `0`이면 injection 없이 detector를 실행한다.
5. `1..4`이면 선택 영역에 해당 prototype을 additive injection한 뒤 detector head를 실행한다.

## 확장 방향

초기 실험 이후 다음 확장을 고려한다.

| 확장 | 설명 |
|---|---|
| Soft label | region별 누적 miss score를 확률 분포로 정규화해 CE 대신 KL 또는 soft CE 사용 |
| Soft injection | `argmax` 대신 `softmax(region_logits)`로 prototype mixture를 만들어 주입 |
| Gated additive | feature 또는 region confidence로 injection 강도를 조절 |
| `4x4`, `8x8` grid | 더 세밀한 failure location prior 학습 |
| Class-aware prototype | region뿐 아니라 class별 missed pattern을 반영 |
| Scale-aware prototype | FPN level 또는 GT 크기에 따라 다른 prototype 사용 |

## 주요 Ablation

ReMiss의 효과를 검증하려면 다음 ablation이 필요하다.

| 실험 | 목적 |
|---|---|
| Baseline detector | 기준 성능 |
| Memory만 사용, injection 없음 | 반복 미검출 통계 자체의 효과 확인 |
| Region head만 사용, injection 없음 | auxiliary supervision 효과 확인 |
| GT region injection | region 예측 오류를 제거한 upper-bound 확인 |
| Predicted hard injection | 초기 제안 방식의 실제 효과 확인 |
| Backbone vs neck injection | injection 위치 영향 확인 |
| Additive vs gated additive | injection 방식 영향 확인 |
| `2x2` vs `4x4` vs `8x8` | grid 해상도 영향 확인 |

## 평가 지표

일반적인 mAP 외에 미검출 회복 여부를 직접 볼 수 있는 지표가 필요하다.

| 지표 | 설명 |
|---|---|
| mAP / AP50 / AP75 | 표준 검출 성능 |
| AP_small / AP_medium / AP_large | 객체 크기별 성능 |
| Recall | 미검출 개선 여부 확인 |
| FN count | 전체 false negative 수 |
| Recurrent FN recovery rate | 연속 미검출 GT가 이후 검출되는 비율 |
| Region-wise recall | grid 영역별 recall 변화 |
| Class-wise FN reduction | 클래스별 미검출 감소량 |

## 기대 효과

ReMiss는 반복 미검출 객체가 특정 공간, 크기, 클래스, 장면 조건에 몰리는 경우에 효과가 있을 것으로 기대된다. 특히 작은 객체, 부분 가림 객체, 이미지 경계 부근 객체처럼 기존 detector가 반복적으로 놓치는 GT가 존재할 때 memory 기반 target이 의미 있는 학습 신호가 될 수 있다.

## 리스크

초기 detector가 불안정할 때 생성된 미검출 memory가 잘못된 bias를 만들 수 있다. 이를 줄이기 위해 `start_epoch`, `miss_threshold`, score threshold, IoU matching threshold를 신중히 설정해야 한다.

또한 hard injection은 예측 region이 틀렸을 때 잘못된 영역의 feature를 보정할 수 있다. 따라서 초기에는 hard injection으로 단순성을 확보하되, 이후 soft injection이나 gated additive로 확장해 과도한 feature 변경을 완화하는 것이 바람직하다.
