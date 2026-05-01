# Stable Miss Pinning

## 개요

Stable Miss Pinning(SMP)은 학습 후반부에 detector가 반복적으로 놓치는 GT의 위치를 MissBank로 식별하고, 해당 위치를 feature map 위의 pin으로 변환한 뒤, pin 주변에 학습 가능한 embedding prototype을 주입하는 방법론이다.

기존 ReMiss의 MissHead 방식은 이미지 단위로 "어느 region에 missed GT가 많은가"를 예측하려고 한다. 그러나 KITTI 실험에서 전체 5985장 중 missed object가 존재하는 이미지가 약 77장뿐이라면 image-level head를 안정적으로 학습시키기 어렵다. SMP는 이 문제를 우회한다. 예측 head를 먼저 학습하지 않고, 이미 MissBank가 알고 있는 안정적인 missed GT 위치를 직접 feature 보정 위치로 사용한다.

핵심 아이디어는 다음과 같다.

1. MissBank stability가 충분히 높아진 시점 이후의 missed GT만 stable miss로 간주한다.
2. stable miss GT의 bbox 중심 또는 GT 내부 대표점을 normalized coordinate로 저장한다.
3. FPN 각 level의 해상도에 맞춰 normalized coordinate를 feature map 좌표로 변환한다.
4. 해당 위치에 pin을 찍고, pin 주변에 learnable prototype을 additive 방식으로 주입한다.
5. 초기 버전에서는 inference에는 사용하지 않고 training-time feature correction으로만 사용한다.
6. 효과가 검증되면 pin 위치를 예측하는 lightweight head로 확장한다.

## 문제 정의

현재 관측된 MissBank 패턴은 다음과 같다.

- FCOS + ResNet50 + KITTI baseline의 최고 성능은 약 140 epoch 부근에서 발생했다.
- 130 epoch checkpoint에서 MissBank를 켜고 166 epoch까지 추가 학습한다.
- milestone은 `[133, 143]`으로 설정한다.
- 135 epoch까지 `jaccard_stability`는 약 `0.5 ~ 0.6`이다.
- 136 epoch 이후 `jaccard_stability`가 `0.8` 이상으로 상승하며 점차 안정화된다.
- region histogram은 대략 `1: 2, 2: 10, 3: 33, 4: 35`로 하단 region에 missed GT가 집중된다.
- 전체 5985장 중 missed object가 있는 이미지는 약 77장뿐이다.

이 상황에서 MissHead처럼 image-level classification target을 학습시키면 none sample이 압도적으로 많아지고, non-none sample은 매우 희소하다. 따라서 head가 none으로 붕괴하거나, 특정 region prior만 외울 가능성이 크다.

SMP는 missed sample 수가 적은 상황을 학습 문제로 직접 풀지 않고, stable miss 위치를 sparse supervision으로 사용한다. 즉, "어디가 어려운지 예측"하기 전에 "이미 안정적으로 어려운 위치를 보정했을 때 detector가 좋아지는지"를 먼저 검증한다.

## Stable Miss 정의

SMP에서 stable miss는 다음 조건을 만족하는 GT record로 정의한다.

- 현재 epoch에서 missed 상태이다.
- `miss_count >= miss_threshold`이다.
- epoch-level miss set의 `jaccard_stability >= stability_threshold`인 시점 이후에 관측된다.
- 선택적으로 같은 image/GT/region이 최근 `K` epoch 중 `M`회 이상 반복 missed 상태이다.

초기 실험에서는 다음 기준을 권장한다.

```yaml
stability_threshold: 0.8
miss_threshold: 2
stable_window: 3
stable_min_hits: 2
```

`jaccard_stability`가 높다는 것은 missed GT set이 epoch 간 크게 바뀌지 않는다는 의미다. 이 시점 이후의 miss는 단순한 학습 노이즈보다 모델이 계속 해결하지 못하는 persistent false negative일 가능성이 높다.

## Pin 생성

각 stable miss GT에 대해 pin coordinate를 만든다.

기본 pin 위치:

```text
px = (x1 + x2) / 2 / image_width
py = (y1 + y2) / 2 / image_height
```

여기서 `(px, py)`는 `0..1` 범위의 normalized coordinate이다. FPN level `l`의 feature map 크기가 `(Hl, Wl)`이면 실제 feature 좌표는 다음과 같다.

```text
fx_l = round(px * (Wl - 1))
fy_l = round(py * (Hl - 1))
```

객체 크기가 FCOS level assignment와 맞는 경우, 모든 FPN level에 pin을 찍기보다 GT scale에 대응되는 level에만 pin을 찍는 것이 더 안전하다. 초기 실험에서는 두 가지를 비교한다.

| 방식 | 설명 |
|---|---|
| all-level pin | 모든 FPN level의 대응 좌표에 prototype 주입 |
| scale-aware pin | GT bbox 크기에 맞는 FPN level에만 prototype 주입 |

## Prototype Injection

각 pin에는 learnable embedding prototype을 주입한다. FPN feature `F_l`의 channel 수가 `C`라면 prototype은 `R^C` 벡터다.

가장 단순한 방식은 additive injection이다.

```text
F_l[:, :, fy, fx] = F_l[:, :, fy, fx] + alpha * p
```

단일 점만 보정하면 영향이 너무 약할 수 있으므로, 초기 구현에서는 작은 spatial mask를 함께 고려한다.

```text
F'_l(c, y, x) = F_l(c, y, x) + alpha * G(y, x; fy, fx, sigma) * p_c
```

여기서 `G`는 pin 중심의 Gaussian 또는 작은 fixed window mask이다.

권장 초기값:

```yaml
injection:
  mode: additive
  alpha: 0.1
  radius: 1
  mask: gaussian
  sigma: 1.0
```

처음부터 강하게 주입하면 detector가 prototype에 의존하거나 feature distribution이 흔들릴 수 있으므로 `alpha`는 작게 시작한다.

## Prototype 공유 방식

pin마다 독립 prototype을 만들면 77장 수준의 sparse miss set에서는 memorization 가능성이 크다. 따라서 prototype은 공유하는 것이 낫다.

후보는 다음과 같다.

| 방식 | 장점 | 리스크 |
|---|---|---|
| global prototype | 가장 단순하고 파라미터 적음 | class/scale 차이를 반영하지 못함 |
| class-wise prototype | missed class별 보정 가능 | class별 sample이 더 희소해짐 |
| level-wise prototype | FPN level별 feature 분포 차이 반영 | 위치/클래스 정보는 약함 |
| class x level prototype | 표현력 가장 좋음 | 데이터가 적으면 과적합 가능 |
| region x level prototype | 현재 region histogram과 잘 맞음 | KITTI 위치 prior를 외울 수 있음 |

초기 실험에서는 `level-wise prototype` 또는 `global prototype`을 추천한다. 성능이 확인되면 `class x level`로 확장한다.

## Training-only 사용

초기 SMP는 inference에 사용하지 않는다. 목적은 pin prototype이 missed GT 주변 feature를 보정하여 detector 학습을 돕는지 확인하는 것이다.

다만 training에서만 prototype을 넣고 inference에서 제거하면 train-test mismatch가 생긴다. 따라서 단순히 pinned feature로 detector loss만 계산하는 방식은 위험하다.

더 안전한 구조는 clean forward와 pinned forward를 함께 사용하는 것이다.

```text
clean feature -> detector loss
pinned feature -> detector loss on stable miss targets
pinned prediction -> clean prediction consistency or distillation
```

이렇게 하면 pin branch는 training-time booster 역할을 하고, clean branch가 그 효과를 흡수하도록 유도할 수 있다.

초기 버전의 loss 구성:

```text
L = L_det_clean
  + lambda_pin * L_det_pinned
  + lambda_consistency * L_consistency(clean, pinned)
```

`L_det_clean`은 기존 detector 성능을 유지하기 위한 기본 loss다. `L_det_pinned`는 pin이 찍힌 stable miss GT가 더 잘 학습되도록 돕는다. `L_consistency`는 inference에 pin을 쓰지 않는 조건에서 clean branch가 pinned branch의 개선된 신호를 따라가도록 한다.

## 기존 연구와의 차별점

### OHEM / Focal Loss와의 비교

OHEM과 Focal Loss는 hard example에 더 집중한다. 하지만 이들은 반복적으로 놓친 특정 GT의 위치를 기억하지 않는다.

SMP는 hard sample을 loss 크기로만 선택하지 않고, MissBank가 누적한 persistent false negative를 기반으로 spatial memory를 만든다. 즉, hard mining보다 더 구체적인 failure memory이다.

### Deformable Conv / Deformable DETR와의 비교

Deformable Conv는 sampling offset을 학습하고, Deformable DETR은 reference point 주변의 sparse sampling point에 attention을 집중한다.

SMP도 sparse spatial location에 집중한다는 점은 유사하다. 그러나 offset이나 reference point를 feature에서 직접 학습하는 대신, MissBank가 발견한 stable failure coordinate를 pin으로 사용한다. 따라서 초기 버전은 위치 예측 문제를 제거하고, feature 보정 효과만 분리해서 검증할 수 있다.

### DETR Query / DAB-DETR와의 비교

DAB-DETR은 anchor box coordinate를 query로 사용하여 positional prior를 제공한다. 이는 모든 객체 검출을 위한 일반적인 query formulation이다.

SMP의 pin은 전체 객체를 찾기 위한 일반 query가 아니라, 특정 detector가 학습 후반에도 반복적으로 놓치는 GT에만 생성되는 failure-conditioned spatial prompt이다.

### Visual Prompt Tuning과의 비교

Visual Prompt Tuning은 learnable prompt token으로 pretrained vision model을 조정한다. SMP도 learnable embedding을 feature에 주입한다는 점에서는 prompt tuning과 유사하다.

차이점은 SMP의 prompt가 고정된 전역 token이 아니라 MissBank의 stable miss coordinate에 결합된 spatial prompt라는 점이다.

## 기대 효과

SMP가 성공하면 다음 효과를 기대할 수 있다.

1. missed GT 주변 feature의 object evidence 강화
2. 반복 false negative에 대한 detector recall 개선
3. MissHead 없이도 sparse miss sample을 활용 가능
4. inference-time module 없이 training-time 보정 효과 검증 가능
5. 향후 pin prediction head로 확장할 수 있는 중간 단계 제공

특히 KITTI처럼 missed object가 특정 하단 region에 집중되는 경우, SMP는 해당 위치 bias를 직접 feature supervision으로 활용할 수 있다.

## 주요 리스크

### 과적합

missed image가 77장뿐이면 pin prototype이 특정 이미지나 위치를 외울 수 있다. 이를 줄이려면 prototype을 공유하고, pin 위치에 jitter를 주며, 일부 step에서만 injection을 켜는 것이 좋다.

### Train-test mismatch

초기 버전은 inference에 pin을 사용하지 않는다. 따라서 pinned branch만 좋아지고 clean inference 성능은 개선되지 않을 수 있다. clean branch consistency loss가 필요하다.

### 위치 prior 과의존

region histogram이 하단에 몰려 있으므로, 모델이 KITTI의 spatial bias를 외울 가능성이 있다. validation set의 stable miss recall과 전체 mAP를 함께 봐야 한다.

### Feature distribution shift

prototype을 feature에 더하면 FPN feature distribution이 바뀐다. `alpha`를 작게 시작하고 delta norm을 로깅해야 한다.

## 평가 지표

SMP는 전체 mAP만 보면 효과를 해석하기 어렵다. 다음 지표를 함께 본다.

| 지표 | 의미 |
|---|---|
| `smp_target_gt_count` | pin 대상 stable miss GT 수 |
| `smp_active_image_count` | pin이 적용된 이미지 수 |
| `smp_pin_count_per_level` | FPN level별 pin 수 |
| `smp_delta_norm` | injection으로 feature가 얼마나 변했는지 |
| `smp_delta_ratio` | 원본 feature norm 대비 delta 비율 |
| `smp_stable_miss_recall` | stable miss GT가 validation에서 검출되는 비율 |
| `smp_stable_miss_recall_delta` | baseline 대비 stable miss recall 개선량 |
| `smp_false_positive_delta` | pin 주변 또는 전체 FP 증가량 |
| `bbox_mAP_50_95` | 전체 detector 성능 |
| `bbox_mAP_75` | localization quality 변화 확인 |

초기 성공 조건은 전체 mAP보다 stable miss recall 개선을 우선 본다. 단, stable miss recall이 오르더라도 전체 mAP가 하락하면 prototype이 과도한 feature shift를 만든 것으로 해석한다.

## Ablation

필수 ablation은 다음과 같다.

| 실험 | 목적 |
|---|---|
| baseline | 원래 detector 성능 |
| MissBank only | mining/logging 자체 영향 확인 |
| SMP additive | pin prototype 주입 효과 확인 |
| SMP random pin | 위치가 중요한지 확인 |
| SMP random prototype | learnable prototype이 중요한지 확인 |
| all-level vs scale-aware | FPN level 선택 영향 확인 |
| point vs gaussian mask | spatial injection 범위 영향 확인 |
| alpha sweep | injection strength 민감도 확인 |
| clean-only vs clean+pinned consistency | train-test mismatch 완화 효과 확인 |

## 확장 방향

SMP 1차 목표는 "stable miss 위치에 prototype을 주입하면 detector가 놓친 객체를 더 잘 학습하는가"를 확인하는 것이다.

성능 향상이 검증되면 다음 단계로 확장한다.

1. pin prediction head를 추가하여 inference에서도 pin 위치를 예측한다.
2. fixed pin 대신 soft pin heatmap을 예측한다.
3. class-aware 또는 level-aware prototype을 사용한다.
4. prototype injection을 gated additive로 바꾼다.
5. pin 위치를 bbox center가 아니라 best-IoU failure 위치, FCOS positive center, 또는 GT 내부 hard point로 바꾼다.

## 요약

SMP는 MissBank가 발견한 stable false negative를 feature-space spatial prompt로 변환하는 방법론이다. MissHead처럼 희소한 image-level label을 학습하는 대신, 학습 후반에 안정적으로 반복되는 missed GT 위치를 직접 pin으로 사용한다.

이 방법의 핵심 novelty는 prototype injection 자체가 아니라, detector의 persistent failure memory를 FPN coordinate에 정렬된 sparse pin으로 바꾸고, 이를 training-time feature correction에 사용하는 데 있다.
