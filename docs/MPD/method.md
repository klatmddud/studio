# MPD: Miss-Guided Positive Densification

## 개요

MPD(Miss-Guided Positive Densification)는 detector가 반복적으로 놓치는 GT를 MissBank에 기억한 뒤, 다음 학습 epoch에서 해당 GT에 대한 positive supervision을 더 촘촘하게 제공하는 방법론이다.

ReMiss 계열의 기존 접근은 missed GT가 많이 발생하는 region을 예측하고, 해당 region의 feature를 embedding prototype 또는 convolutional modulation으로 보강하는 방식이었다. 반면 MPD는 feature를 직접 바꾸기보다 detector의 학습 target assignment를 보정한다.

핵심 아이디어는 다음과 같다.

```text
반복적으로 missed 되는 GT
  -> detector가 현재 positive로 충분히 학습하지 못하는 객체
  -> 다음 epoch에서 해당 GT 주변의 positive location을 확장
  -> 기존 detector loss를 통해 직접 재학습
```

즉, MPD는 "어려운 객체의 feature를 강화하자"가 아니라 "계속 놓치는 객체에 대해 positive 학습 신호를 더 많이 주자"에 가깝다.

## 문제 정의

Object detection 학습에서 일부 GT는 여러 epoch 동안 반복적으로 미검출될 수 있다. 이러한 GT는 다음과 같은 특성을 가질 수 있다.

- 크기가 작거나 멀리 있는 객체
- occlusion 또는 truncation이 심한 객체
- 배경과 구분이 어려운 객체
- detector의 현재 positive assignment에서 충분히 학습되지 않는 객체
- score threshold 기준으로 계속 최종 검출에 도달하지 못하는 객체

기존 detector는 일반적으로 현재 batch의 GT와 prediction을 기준으로 loss를 계산한다. 따라서 특정 GT가 이전 epoch들에서도 계속 missed 되었는지에 대한 history는 직접 사용하지 않는다.

MPD는 이 history를 MissBank에 저장하고, 반복 missed GT를 다음 학습에서 더 강한 positive target으로 사용한다.

## 핵심 구성

MPD는 세 가지 구성으로 이루어진다.

| 구성 | 역할 |
|---|---|
| MissBank | image ID, GT ID, class, bbox, region, consecutive miss count를 저장 |
| Missed GT Selector | `miss_count >= miss_threshold`인 GT를 hard missed GT로 선택 |
| Positive Densification | 선택된 GT 주변의 positive location을 확장 |

## 동작 흐름

MPD의 전체 흐름은 다음과 같다.

```text
1. 일반 detector 학습 수행
2. epoch 종료 후 training dataset에 대해 missed GT mining
3. MissBank 업데이트
4. 다음 epoch에서 MissBank의 hard missed GT 조회
5. 해당 GT에 대해 positive assignment 영역 확장
6. detector의 기존 cls/reg/centerness loss로 학습
7. inference 때는 MPD 모듈을 사용하지 않음
```

MPD는 학습 단계에서만 target assignment를 보정한다. inference 구조, detector head, post-processing, threshold, NMS는 변경하지 않는다.

## FCOS 기준 설계

FCOS는 feature map의 각 location을 기준으로 객체를 예측한다. 일반적으로 GT 내부에 있는 location 중 center sampling, regression range 등의 조건을 만족하는 지점이 positive로 지정된다.

MPD는 반복 missed GT에 대해 기존 positive location 주변을 추가 positive로 확장한다.

예시는 다음과 같다.

```text
기존 FCOS positive:
  GT 중심 또는 center sampling 조건을 만족하는 일부 location

MPD positive:
  기존 positive location
  + MissBank에서 hard missed GT로 선택된 객체의 중심 주변 r-neighborhood location
```

`radius = 1`이면 feature map 상에서 중심 주변 `3x3` 영역이 후보가 된다.

```text
radius = 0: center 1개 cell
radius = 1: 3x3 cells
radius = 2: 5x5 cells
```

현재 구현에서는 FCOS의 기본 center sampling radius를 고려해 `mpd.radius_mode`를 둔다.

```text
radius_mode: expand
  effective_radius = detector.center_sampling_radius + mpd.radius

radius_mode: absolute
  effective_radius = mpd.radius + 0.5
```

TorchVision FCOS의 기본 `center_sampling_radius`는 1.5이므로, `radius_mode: expand`, `radius: 1`은 기존 center sampling보다 한 ring 더 넓은 영역을 후보로 삼는다. 즉 구현 기준의 기본값은 단순 3x3 재현이 아니라, 반복 missed GT에 대해 기존 FCOS positive 후보보다 더 넓은 positive 후보를 추가하는 설정이다.

단, 추가 positive는 다음 조건을 만족하는 경우에만 사용하는 것이 적절하다.

- 해당 GT가 MissBank에서 현재 missed 상태일 것
- `miss_count >= miss_threshold`일 것
- 추가 location이 GT 내부 또는 center sampling 허용 범위에 있을 것
- 기존 다른 GT의 positive와 심하게 충돌하지 않을 것
- regression target이 유효한 범위일 것

## Target Assignment 방식

MPD의 target assignment는 detector의 기존 방식을 최대한 유지한다.

추가 positive location에 대해서도 기존 detector와 동일하게 target을 부여한다.

| Target | 부여 방식 |
|---|---|
| classification | missed GT의 class label |
| box regression | 해당 location에서 GT box까지의 거리 |
| centerness | 기존 FCOS centerness 계산식 사용 |

따라서 MPD는 별도의 auxiliary loss를 만들지 않는다. 추가된 positive location은 기존 detector loss 안으로 들어간다.

```text
L_total = L_cls + L_reg + L_centerness
```

이 점이 MissHead, MissInjection, ReMiss-Conv와 가장 큰 차이다.

## 왜 단순한가

MPD는 새로운 feature branch나 prediction head를 추가하지 않는다.

- 새로운 embedding prototype 없음
- gate calibration 없음
- auxiliary CE/BCE loss 없음
- `none` class imbalance 문제 없음
- inference overhead 없음
- detector output format 변화 없음

주요 hyperparameter는 `start_epoch`, `miss_threshold`, `radius` 정도다.

## 기존 연구와의 비교

### OHEM / Hard Example Mining

OHEM은 loss가 큰 sample을 선택해 학습을 집중한다. Focal Loss도 easy negative의 영향을 줄이고 hard example에 더 집중하도록 loss를 재가중한다.

MPD도 hard sample에 집중한다는 점은 유사하다. 그러나 MPD는 현재 batch loss가 아니라 최종 detector output 기준의 missed GT history를 사용한다.

차이점은 다음과 같다.

| 항목 | OHEM / Focal Loss | MPD |
|---|---|---|
| 기준 | 현재 loss 또는 confidence | 최종 검출 실패 여부 |
| memory | 없음 | MissBank 사용 |
| 반복 miss 고려 | 없음 | consecutive miss count 사용 |
| 조치 | loss reweighting 또는 sample selection | positive assignment 확장 |

### ATSS / PAA / OTA

ATSS, PAA, OTA는 positive/negative assignment를 개선하는 방법이다. 이들은 object detection에서 어떤 anchor/location을 positive로 볼 것인지가 성능에 중요하다는 점을 보여준다.

MPD도 assignment를 조정한다는 점에서 이 계열과 가깝다. 하지만 기존 assignment 방법들은 대부분 현재 GT와 현재 prediction 또는 anchor 통계에 기반한다. MPD는 epoch-level miss history를 사용한다는 점이 다르다.

| 항목 | ATSS / PAA / OTA | MPD |
|---|---|---|
| assignment 기준 | 현재 sample의 통계, loss, matching cost | 반복 missed GT |
| history 사용 | 일반적으로 없음 | 있음 |
| 목적 | 더 좋은 positive/negative 구분 | 반복 미검출 객체 재학습 |
| 적용 방식 | assignment rule 자체 개선 | hard missed GT에 한정한 positive densification |

### Attention / Feature Modulation 계열

CBAM, Dynamic Head, DCNv2 같은 방법은 attention 또는 deformable convolution을 통해 feature representation을 개선한다.

MPD는 feature를 직접 modulate하지 않는다. 대신 어떤 위치가 detector loss를 받아야 하는지를 보정한다.

| 항목 | Attention / Modulation | MPD |
|---|---|---|
| 변경 대상 | feature map | training target assignment |
| 추가 파라미터 | 있음 | 원칙적으로 없음 |
| inference overhead | 있을 수 있음 | 없음 |
| missed GT history | 사용하지 않음 | 직접 사용 |

### ReMiss / ReMiss-Conv와의 차이

ReMiss와 ReMiss-Conv는 MissBank를 사용해 missed region을 예측하고 feature map에 embedding 또는 modulation을 주입한다.

MPD는 MissBank를 사용하지만, feature injection을 하지 않는다. missed GT를 기억한 뒤 해당 GT의 positive supervision을 늘린다.

| 항목 | ReMiss / ReMiss-Conv | MPD |
|---|---|---|
| MissBank 사용 | 사용 | 사용 |
| 추가 head | 있음 | 없음 |
| feature injection | 있음 | 없음 |
| auxiliary metric/loss | 있음 | 없음 |
| inference 변경 | 설정에 따라 가능 | 없음 |
| 핵심 조치 | feature repair | assignment repair |

## 장점

MPD의 장점은 다음과 같다.

1. 구현이 단순하다.
2. inference cost가 증가하지 않는다.
3. 기존 detector loss를 그대로 사용한다.
4. feature injection 계열보다 hyperparameter가 적다.
5. MissBank의 정보를 detector 학습에 직접 연결한다.
6. 반복 missed GT의 recall 개선 여부를 명확하게 측정할 수 있다.

## 잠재적 위험

MPD에도 주의할 점은 있다.

첫째, positive를 너무 많이 늘리면 localization 품질이 떨어질 수 있다. 특히 GT 경계 근처 location까지 positive로 확장하면 regression target이 불안정해질 수 있다.

둘째, annotation noise 또는 매우 어려운 GT에 과도하게 집중할 수 있다. 이 경우 전체 mAP보다 특정 noisy sample에 loss가 끌릴 수 있다.

셋째, 작은 객체에서는 `radius = 1`도 과도할 수 있다. feature stride가 큰 level에서는 3x3 확장이 실제 image 공간에서 넓은 영역을 의미할 수 있다.

따라서 MPD는 작은 radius에서 시작하고, GT 내부 또는 center sampling 조건을 유지하는 것이 안전하다.

## 1차 실험 설정

초기 실험은 다음 설정을 권장한다.

```yaml
enabled: true
grid_size: 2

mining:
  type: offline

matching:
  score_threshold: auto
  iou_threshold: auto

target:
  miss_threshold: 2
  aggregation: miss_count

mpd:
  enabled: true
  start_epoch: 18
  radius: 1
  radius_mode: expand
  require_current_missed: true
  require_inside_gt: true
  respect_scale_range: true
  override_existing_positive: false
```

의미는 다음과 같다.

| 설정 | 의미 |
|---|---|
| `start_epoch` | MissBank가 충분히 안정된 뒤 MPD를 시작 |
| `target.miss_threshold` | 연속 miss count가 이 값 이상인 GT만 사용 |
| `radius` | missed GT 중심 주변 positive 확장 반경 |
| `radius_mode` | `expand`이면 detector 기본 center sampling보다 추가 확장, `absolute`이면 MPD radius를 직접 사용 |
| `require_current_missed` | 현재 epoch mining 결과에서도 missed 상태인 GT만 사용 |
| `require_inside_gt` | 추가 positive 후보가 GT 내부에 있을 때만 사용 |
| `respect_scale_range` | FCOS level별 regression range 조건을 유지 |
| `override_existing_positive` | 다른 GT에 이미 할당된 positive를 덮어쓰지 않음 |

처음에는 `start_epoch = 18`, `miss_threshold = 2`, `radius = 1`이 적절하다. 이후 실험에서 `start_epoch = 12`, `miss_threshold = 3`, `radius = 0/1/2`를 비교할 수 있다.

## 평가 지표

MPD는 전체 mAP만 보면 효과를 해석하기 어렵다. 반드시 MissBank 기반 지표를 함께 봐야 한다.

주요 지표는 다음과 같다.

| 지표 | 의미 |
|---|---|
| `missed_gt_recall` | MPD target GT 중 다음 epoch에서 검출된 비율 |
| `persistent_miss_count` | 계속 missed 상태로 남은 GT 수 |
| `recovered_gt_count` | 이전에는 missed였지만 현재 검출된 GT 수 |
| `mpd_positive_count` | MPD로 추가된 positive location 수 |
| `mpd_positive_ratio` | 전체 positive 중 MPD positive 비율 |
| `mpd_region_histogram` | MPD가 집중한 region 분포 |
| `mAP`, `mAP50`, `mAR` | detector 전체 성능 |

가장 중요한 판단 기준은 다음과 같다.

```text
1. missed_gt_recall이 증가하는가?
2. persistent_miss_count가 감소하는가?
3. 전체 mAP가 유지되거나 상승하는가?
4. MPD positive가 과도하게 많아지지 않는가?
```

## Ablation 계획

초기 ablation은 복잡하게 가져갈 필요가 없다.

| 실험 | 목적 |
|---|---|
| baseline | MPD 없이 기존 detector |
| MPD r0 | center cell만 추가 |
| MPD r1 | `radius_mode: expand` 기준 기존 FCOS center sampling보다 1 ring 추가 확장 |
| MPD r1, threshold 3 | 더 신뢰도 높은 hard GT만 사용 |
| MPD start 12 vs 18 | MPD 시작 시점 비교 |

처음부터 attention, convolution, gate, extra loss를 추가하지 않는다. MPD의 기본 가설은 positive assignment 보정만으로 missed GT recall을 개선할 수 있는지 확인하는 것이다.

## 기대 효과

MPD가 잘 동작한다면 다음 현상이 나타나야 한다.

- 반복 missed GT의 검출 회복률 증가
- missed GT가 몰린 region의 recall 개선
- 작은 객체 또는 어려운 객체의 recall 개선
- 전체 mAP 유지 또는 상승
- inference 속도 변화 없음

특히 KITTI처럼 특정 위치나 특정 객체 유형에서 반복 miss가 발생하는 데이터셋에서는 MissBank history를 target assignment에 직접 반영하는 MPD가 효과적일 가능성이 있다.

## 요약

MPD는 MissBank에 저장된 반복 missed GT를 활용해 detector의 positive assignment를 조밀하게 만드는 방법이다.

```text
Missed GT memory
  -> hard missed GT selection
  -> positive location densification
  -> original detector loss
  -> inference unchanged
```

ReMiss가 feature repair라면, MPD는 assignment repair다. 복잡한 auxiliary head나 feature injection 없이 missed GT memory를 직접 detector 학습에 연결한다는 점에서 단순하고 실험 가치가 높은 방법론이다.
