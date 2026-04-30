# ReMiss-Conv: Convolutional Miss-Aware Spatial Modulation

## 개요

ReMiss-Conv는 기존 ReMiss의 `MissHead -> MissInjection` 흐름을 하나의 convolutional feature modulation block으로 통합하는 확장 방법론이다. 기존 ReMiss는 MissBank가 누적한 반복 미검출 정보를 이용해 이미지 단위의 취약 region을 예측하고, 해당 region에 learnable embedding prototype을 additive 방식으로 주입한다.

ReMiss-Conv는 이 과정을 더 직접적인 spatial feature 보정 문제로 재정의한다.

```text
기존 ReMiss:
MissBank target
  -> MissHead image-level region prediction
  -> hard/soft region selection
  -> prototype injection
  -> detector head

ReMiss-Conv:
MissBank target
  -> convolutional miss map supervision
  -> spatial gate / prototype modulation
  -> detector head
```

핵심 아이디어는 neck feature map을 `2x2`, `4x4`, `8x8` 등의 grid로 나누고, 각 grid cell이 반복 미검출이 발생하는 취약 영역인지 convolutional head가 직접 예측하게 만드는 것이다. 이후 예측된 miss map을 이용해 해당 위치의 feature를 residual additive 또는 gated additive 방식으로 보정한다.

## 동기

현재 MissHead는 neck feature map을 global average pooling한 뒤 image-level label을 예측한다. 이 방식은 구현이 단순하지만, 좌상단, 우상단, 좌하단, 우하단과 같은 spatial region 정보를 직접 보존하지 못한다. ReMiss가 실제로 해결하려는 문제는 "이미지의 어느 위치에서 반복적으로 GT를 놓치는가"이므로, spatial feature를 유지한 상태에서 miss-prone region을 예측하는 구조가 더 자연스럽다.

ReMiss-Conv는 다음 문제를 해결하기 위한 확장이다.

1. image-level hard label 하나로는 여러 region에 동시에 발생하는 miss를 표현하기 어렵다.
2. global pooling 기반 MissHead는 region 위치 정보를 약하게만 학습한다.
3. `has-miss head`와 `region head`를 분리해도 최종 injection은 gate 품질에 크게 의존한다.
4. MissInjection을 별도 후처리처럼 두는 대신, detector feature path 안의 trainable modulation block으로 통합할 수 있다.

## 기본 구조

ReMiss-Conv block은 FPN 또는 neck feature map `F_l`을 입력으로 받는다.

```text
F_l: [B, C, H_l, W_l]
```

각 level의 feature map에 대해 다음 절차를 수행한다.

```text
1. F_l에서 S x S grid-level descriptor 추출
2. grid별 miss probability 또는 miss logit 예측
3. grid별 learnable prototype 또는 prototype map 생성
4. miss probability를 spatial mask로 upsample
5. feature map에 residual modulation 적용
```

기본 수식은 다음과 같다.

```text
F'_l = F_l + alpha * A_l(F_l) * P_l
```

여기서:

| 기호 | 의미 |
|---|---|
| `F_l` | l번째 neck/FPN feature map |
| `F'_l` | modulation 이후 feature map |
| `A_l(F_l)` | convolutional miss gate 또는 miss probability map |
| `P_l` | region prototype map |
| `alpha` | modulation strength |

`A_l(F_l)`는 `[B, 1, H_l, W_l]` 또는 `[B, R, H_l, W_l]` 형태가 될 수 있다. `R = S * S`는 grid region 수이다.

## Grid Miss Map Head

가장 단순한 구현은 `S x S` grid miss map을 예측하는 head이다.

```text
F_l
  -> AdaptiveAvgPool2d(S, S)
  -> Conv1x1 / Conv3x3
  -> miss_logits_l: [B, 1, S, S]
```

예를 들어 `S=2`이면 다음과 같은 map을 예측한다.

```text
miss_logits_l.shape = [B, 1, 2, 2]
```

각 cell은 해당 quadrant에 반복 미검출 target이 있는지를 나타낸다.

```text
0: 좌상단
1: 우상단
2: 좌하단
3: 우하단
```

기존 5-way hard label과 달리, 이 방식은 multi-label target을 사용할 수 있다. 하나의 이미지에서 좌하단과 우하단에 모두 반복 미검출 GT가 있으면 두 cell을 동시에 positive로 둘 수 있다.

```text
target_miss_map: [B, 1, S, S]
target_miss_map[b, 0, row, col] = 1 if 해당 cell에 miss_count >= threshold GT 존재
```

## MissBank Target 변환

MissBank는 기존과 동일하게 GT별 반복 미검출 정보를 관리한다.

```text
image_id
gt_id
gt_class
miss_count
region_id
current_state
```

ReMiss-Conv에서는 image-level hard label 대신 grid-level binary target을 생성한다.

```text
target_cell(gt) =
    region_id(gt)
    if current_state(gt) == missed
    and miss_count(gt) >= miss_threshold
```

각 이미지에 대해 target grid를 다음과 같이 만든다.

```text
Y[b, row, col] = 1
    if image b에 속한 target GT 중 하나 이상이 해당 cell에 존재
```

miss count를 단순 binary가 아니라 soft score로 사용할 수도 있다.

```text
Y[b, row, col] = normalize(sum miss_count of target GTs in cell)
```

초기 실험에서는 binary BCE target이 가장 해석하기 쉽다.

## Prototype Modulation

각 grid cell에 learnable prototype을 둔다.

```text
prototype.shape = [S, S, C]
```

feature map 해상도에 맞춰 prototype map을 생성한다.

```text
P_grid: [1, C, S, S]
P_l = upsample(P_grid, size=(H_l, W_l))
```

miss gate도 같은 방식으로 upsample한다.

```text
A_grid = sigmoid(miss_logits_l)          # [B, 1, S, S]
A_l = upsample(A_grid, size=(H_l, W_l)) # [B, 1, H_l, W_l]
```

그 뒤 feature에 주입한다.

```text
F'_l = F_l + alpha * A_l * P_l
```

이 방식은 기존 hard injection의 soft한 일반화로 볼 수 있다. 기존 hard injection은 `A_grid`가 하나의 cell에서만 1이고 나머지는 0인 특수한 경우이다.

## Modulation 방식

초기 후보는 다음과 같다.

| 방식 | 수식 | 설명 |
|---|---|---|
| Additive | `F' = F + alpha * A * P` | 기존 MissInjection과 가장 가까운 방식 |
| Gated additive | `F' = F + alpha * sigmoid(A) * P` | gate confidence에 따라 주입 강도 조절 |
| FiLM-style | `F' = gamma(A) * F + beta(A)` | feature-wise affine modulation |
| Residual conv | `F' = F + Conv([F, A, P])` | prototype과 feature를 conv block으로 통합 |
| Dynamic conv | `F' = Conv_dynamic(F; A)` | miss map을 이용해 convolution kernel 또는 expert 선택 |

초기 구현은 `gated additive`가 가장 적절하다.

```text
F' = F + alpha * sigmoid(A) * P
```

이유는 기존 MissInjection과 해석이 이어지고, hard injection으로 쉽게 ablation할 수 있으며, false positive injection의 강도를 gate confidence로 줄일 수 있기 때문이다.

## Multi-Level FPN 처리

FPN은 level마다 해상도가 다르다.

```text
P3: [B, C, H3, W3]
P4: [B, C, H4, W4]
P5: [B, C, H5, W5]
```

ReMiss-Conv는 각 level에서 동일한 `S x S` grid target을 사용하되, feature map 해상도에 맞춰 gate와 prototype을 upsample한다.

```text
for each level l:
    A_grid_l = MissMapHead(F_l)              # [B, 1, S, S]
    A_l = upsample(A_grid_l, H_l, W_l)
    P_l = upsample(P_grid, H_l, W_l)
    F'_l = F_l + alpha_l * A_l * P_l
```

level별 prototype을 공유할지 분리할지는 ablation 대상이다.

| 설계 | 장점 | 단점 |
|---|---|---|
| level-shared prototype | 파라미터 적음, 구현 단순 | scale별 특성 반영 약함 |
| level-specific prototype | FPN scale별 보정 가능 | 파라미터 증가, 과적합 가능 |
| level-weighted prototype | scale attention과 결합 가능 | 구현 복잡도 증가 |

초기 실험은 level-shared prototype으로 시작하는 것이 적절하다.

## Absolute Region 문제

Convolution은 기본적으로 translation equivariant하다. 하지만 ReMiss의 `좌상단`, `우상단`, `좌하단`, `우하단`은 absolute image region이다. 따라서 순수 convolution만 사용하면 어떤 activation이 어느 절대 위치에 있는지를 충분히 구분하지 못할 수 있다.

이를 보완하기 위해 다음 중 하나를 사용한다.

1. fixed grid mask
2. normalized coordinate channel 추가
3. learnable positional embedding
4. grid-specific prototype

추천 초기 설계는 coordinate channel과 grid-specific prototype을 함께 사용하는 것이다.

```text
F_coord = concat(F, x_coord, y_coord)
miss_logits = Conv(F_coord)
```

`x_coord`, `y_coord`는 각 feature map 위치의 normalized coordinate이다.

```text
x_coord in [-1, 1]
y_coord in [-1, 1]
```

## Loss

ReMiss-Conv의 기본 loss는 grid-level binary cross entropy이다.

```text
L_miss_map = BCEWithLogits(miss_logits, target_miss_map)
```

전체 loss는 detector loss에 auxiliary loss를 더한다.

```text
L_total = L_detector + lambda_miss * L_miss_map
```

positive cell이 적고 none cell이 많기 때문에 class imbalance 처리가 필요하다.

추천 후보는 다음과 같다.

| 방식 | 설명 |
|---|---|
| positive weight BCE | positive cell에 더 큰 weight 부여 |
| focal BCE | 쉬운 negative cell의 영향 감소 |
| cell-balanced BCE | batch 내 positive/negative cell 비율로 weight 계산 |
| miss-count weighted BCE | 반복 미검출 count가 높은 cell에 더 큰 weight 부여 |

초기 실험은 `pos_weight: auto` 형태의 BCE가 적절하다.

## Inference

추론 시에는 MissBank target이 없다. ReMiss-Conv block은 feature만 보고 miss gate를 예측한다.

```text
F_l
  -> miss_logits_l
  -> A_l = sigmoid(miss_logits_l)
  -> F'_l = F_l + alpha * A_l * P_l
  -> detector head
```

hard mode에서는 threshold를 적용할 수 있다.

```text
A_l = 1 if sigmoid(miss_logits_l) >= threshold else 0
```

soft mode에서는 probability를 그대로 사용한다.

```text
A_l = sigmoid(miss_logits_l)
```

초기에는 hard injection보다 soft gated injection을 우선 권장한다. 현재 MissHead 실험에서 gate threshold가 성능에 민감했기 때문에, ReMiss-Conv에서는 threshold 없이 연속적인 gate를 사용하는 편이 더 안정적일 가능성이 높다.

## 기존 ReMiss 대비 차이

| 항목 | 기존 ReMiss | ReMiss-Conv |
|---|---|---|
| target | image-level region label | grid-level miss map |
| region 수 | 하나의 hard region 중심 | 여러 region 동시 positive 가능 |
| feature 처리 | GAP 후 MLP | spatial feature 유지 |
| injection | 별도 MissInjection module | convolutional modulation block |
| none 처리 | none class 또는 has-miss gate | cell-wise negative target |
| 확장성 | 2x2, 4x4 hard label 확장 | heatmap resolution 확장 |

## 관련 연구와 차별점

ReMiss-Conv는 기존 attention 또는 dynamic convolution 연구와 연결된다.

| 연구 | 유사점 | 차별점 |
|---|---|---|
| CBAM | channel/spatial attention으로 feature map을 재가중 | 반복 미검출 GT memory를 supervision으로 사용하지 않음 |
| Dynamic Head | detection head에서 scale, spatial, task attention을 통합 | 특정 image/GT의 반복 miss history를 직접 사용하지 않음 |
| FiLM | feature-wise affine modulation을 사용 | spatial miss region 보정과 GT miss memory는 없음 |
| CondConv / Dynamic Convolution | 입력별 동적 kernel 또는 expert 선택 | miss-prone region prototype injection과 목적이 다름 |
| Deformable ConvNets | feature sampling 위치를 학습해 object geometry에 적응 | detector의 반복 미검출 이력을 target으로 쓰지 않음 |
| DetectoRS | FPN/backbone feedback으로 detection feature를 재보정 | MissBank 기반 failure memory는 없음 |
| OHEM | 어려운 sample을 찾아 학습에 반영 | feature map 자체를 region-wise로 보정하지 않음 |

따라서 ReMiss-Conv의 차별점은 다음과 같이 정리할 수 있다.

```text
반복 미검출 GT memory
  -> spatial miss map supervision
  -> miss-prone region feature modulation
  -> detector recall 개선
```

일반 attention module은 현재 feature의 saliency를 학습하지만, ReMiss-Conv는 detector가 과거에 실제로 반복 실패한 GT를 기반으로 취약 영역을 학습한다는 점이 다르다.

## 권장 초기 실험 설계

초기 실험은 복잡한 dynamic convolution보다 단순하고 해석 가능한 `Grid MissMap + Gated Prototype Additive`로 시작한다.

```yaml
remiss_conv:
  enabled: true
  grid_size: 2
  in_channels: 256
  use_coord: true
  prototype:
    type: grid
    shared_across_levels: true
  gate:
    type: conv
    loss: bce
    pos_weight: auto
    soft_injection: true
  injection:
    mode: gated_additive
    alpha: 0.1
    levels: all
```

첫 ablation은 다음 순서가 적절하다.

1. `2x2` vs `4x4`
2. coordinate channel on/off
3. prototype shared vs level-specific
4. additive vs gated additive
5. soft gate vs hard threshold gate
6. MissHead 기반 기존 ReMiss와 ReMiss-Conv 비교

## 평가 지표

기존 MissHead 지표를 grid map 기준으로 확장한다.

| 지표 | 설명 |
|---|---|
| `miss_map_cell_acc` | 전체 grid cell에 대한 binary accuracy |
| `miss_map_pos_precision` | positive miss cell 예측 precision |
| `miss_map_pos_recall` | positive miss cell 예측 recall |
| `miss_map_pos_f1` | positive miss cell F1 |
| `miss_map_iou` | predicted positive cell과 target positive cell의 IoU |
| `missed_object_cell_acc` | 반복 미검출 GT가 속한 cell을 맞췄는지 |
| `missed_object_cell_acc_weighted` | miss_count로 가중한 object-level cell accuracy |
| `gate_positive_ratio` | gate가 positive로 연 region 비율 |
| `detector_recall_on_missed_objects` | ReMiss-Conv가 실제 반복 미검출 GT recall을 개선했는지 |

특히 `miss_map_pos_precision`은 hard injection에서 중요하다. false positive gate가 많으면 정상 feature까지 불필요하게 보정할 수 있다. soft injection에서는 precision과 recall뿐 아니라 gate calibration도 함께 봐야 한다.

## 예상 장점

ReMiss-Conv는 다음 장점을 기대할 수 있다.

1. spatial 위치 정보를 유지한 채 miss-prone region을 학습한다.
2. 여러 region에 동시에 miss가 있는 경우를 자연스럽게 처리한다.
3. MissHead의 none collapse 문제를 cell-wise BCE 문제로 바꿀 수 있다.
4. MissInjection을 detector feature path 안의 residual block으로 통합할 수 있다.
5. soft gated modulation을 통해 hard threshold 민감도를 줄일 수 있다.

## 예상 위험

다음 위험도 존재한다.

1. false positive gate가 많으면 정상 feature를 오염시킬 수 있다.
2. MissBank target이 noisy하면 feature modulation block이 잘못된 region을 강화할 수 있다.
3. grid resolution이 너무 작으면 위치 정보가 거칠고, 너무 크면 positive cell sparsity가 심해진다.
4. detector loss와 miss modulation loss가 충돌할 수 있다.
5. prototype이 특정 dataset의 spatial bias를 과하게 학습할 수 있다.

이를 줄이기 위해 초기에는 작은 `alpha`, soft gate, start epoch, positive weight BCE를 함께 사용한다.

## 결론

ReMiss-Conv는 기존 ReMiss의 memory-driven region prediction과 prototype injection을 convolutional spatial modulation으로 통합하는 방법이다. 핵심은 MissBank의 반복 미검출 정보를 image-level class label이 아니라 grid-level miss map supervision으로 바꾸고, 이 miss map을 이용해 neck feature를 residual하게 보정하는 것이다.

가장 현실적인 1차 구현은 다음 형태이다.

```text
MissBank
  -> grid miss target
  -> Conv MissMap Head
  -> gated prototype additive modulation
  -> detector head
```

이 구조는 기존 attention/dynamic convolution 연구와 연결되지만, detector가 실제로 반복해서 놓친 GT memory를 feature modulation target으로 사용한다는 점에서 ReMiss 고유의 차별점을 유지한다.
