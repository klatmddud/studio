# MissHead 설계 문서

## 개요

MissHead는 ReMiss에서 이미지 단위의 미검출 취약 영역을 예측하는 auxiliary head이다. MissBank가 누적한 반복 미검출 GT 정보를 학습 target으로 사용해, 현재 입력 이미지에서 어느 spatial region에 미검출 GT가 많을 가능성이 높은지 예측한다.

초기 버전은 `2x2` grid 기반 5-way classification으로 시작한다.

| Label | 의미 |
|---|---|
| `0` | none, 주입할 영역 없음 |
| `1` | 좌상단 |
| `2` | 우상단 |
| `3` | 좌하단 |
| `4` | 우하단 |

MissHead의 출력은 이후 MissInject 또는 prototype injection module이 어떤 region prototype을 주입할지 결정하는 데 사용된다.

## 역할

MissHead는 다음 책임을 가진다.

1. Backbone 또는 neck feature를 입력으로 받아 image-level region logits를 생성한다.
2. MissBank가 생성한 hard label을 supervision으로 사용한다.
3. Softmax cross entropy loss를 계산한다.
4. 추론 또는 hard injection 단계에서 `argmax(region_logits)`로 region id를 제공한다.
5. 향후 `4x4`, `8x8`, soft injection, gated injection으로 확장 가능한 interface를 제공한다.

MissHead는 memory를 직접 갱신하지 않는다. GT별 미검출 count와 target label 생성은 MissBank가 담당한다.

## 입력 Feature

MissHead는 detector의 feature를 입력으로 받는다. 초기 ablation 대상은 다음과 같다.

| 입력 위치 | 설명 |
|---|---|
| Backbone output | neck 이전의 backbone feature를 사용 |
| Neck output | FPN 또는 neck 이후의 detection feature를 사용 |
| 단일 FPN level | 특정 scale feature만 사용 |
| 다중 FPN level | 여러 scale feature를 pooling 후 통합 |

초기 구현에서는 FCOS/FPN 계열을 고려해 neck output의 다중 feature map을 입력으로 받는 구조가 가장 자연스럽다. 각 FPN level을 global average pooling한 뒤 concat 또는 mean aggregation하여 image-level representation을 만든다.

예시:

```text
F = {P3, P4, P5, P6, P7}
z_l = GAP(P_l)
z = aggregate(z_3, z_4, z_5, z_6, z_7)
region_logits = MLP(z)
```

## 출력

MissHead는 `num_regions + 1`개의 logit을 출력한다.

```text
num_labels = grid_size * grid_size + 1
region_logits.shape = [B, num_labels]
```

`grid_size=2`이면 `num_labels=5`이다. `0`은 none label이고, `1..4`는 spatial region을 의미한다.

## Target

MissHead의 학습 target은 MissBank에서 생성한다.

```python
target_labels = missbank.get_batch_labels(targets)
```

각 이미지의 label은 다음 규칙을 따른다.

1. 현재 missed 상태인 GT만 후보가 된다.
2. `miss_count >= miss_threshold`인 GT만 후보가 된다.
3. 후보 GT가 없으면 label은 `0`이다.
4. 후보 GT가 여러 개면 region별 `miss_count` 합산이 가장 큰 region을 label로 선택한다.
5. 동률이면 row-major 순서상 더 앞선 region을 선택한다.

MissHead는 GT box를 직접 보지 않고, MissBank가 만든 image-level label만 사용한다.

## Loss

초기 버전은 hard label 기반 softmax cross entropy를 사용한다.

```text
L_miss_head = CE(region_logits, target_region)
```

전체 loss는 detector 기본 loss와 가중합한다.

```text
L_total = L_detector + lambda_miss_head * L_miss_head
```

`lambda_miss_head`는 MissHead loss가 detector 학습을 과도하게 흔들지 않도록 작게 시작한다. 예를 들어 `0.05`, `0.1`, `0.2`를 ablation 후보로 둘 수 있다.

## 활성화 시점

MissBank update는 학습 초반부터 수행할 수 있지만, MissHead 학습은 `start_epoch` 이후에 활성화한다.

| Epoch 조건 | MissBank update | MissHead loss |
|---|---|---|
| `epoch < start_epoch` | 수행 | 비활성 |
| `epoch >= start_epoch` | 수행 | 활성 |

이 설계는 학습 초기에 불안정한 detector 예측으로 만들어진 noisy memory가 곧바로 MissHead를 오염시키는 문제를 줄인다.

## Hard Region Prediction

초기 injection은 hard prediction을 사용한다.

```text
pred_region = argmax(region_logits)
```

`pred_region == 0`이면 MissInject는 feature를 변경하지 않는다. `pred_region > 0`이면 해당 region prototype을 선택해 feature map의 대응 영역에 additive injection을 수행한다.

Training 중에는 두 가지 선택지가 있다.

| 방식 | 설명 |
|---|---|
| Predicted region injection | MissHead 예측값으로 injection 수행 |
| GT region injection | MissBank target label로 injection 수행 |

초기 실제 경로는 predicted region injection을 기준으로 하되, GT region injection은 upper-bound ablation으로 사용한다.

## 추천 구조

초기 MissHead는 단순한 pooling + MLP 구조로 충분하다.

```text
Feature maps
  -> per-level global average pooling
  -> level aggregation
  -> Linear + ReLU
  -> Dropout(optional)
  -> Linear(num_labels)
```

예시 hyperparameter:

| 설정 | 기본값 후보 |
|---|---:|
| `grid_size` | `2` |
| `in_channels` | `256` |
| `hidden_dim` | `256` |
| `num_layers` | `2` |
| `dropout` | `0.0` 또는 `0.1` |
| `loss_weight` | `0.1` |
| `pooling` | `gap` |
| `level_aggregation` | `mean` |

## Multi-level Aggregation

FPN의 여러 level을 사용할 경우 다음 aggregation을 비교한다.

| 방식 | 설명 |
|---|---|
| Mean | level별 pooled vector 평균 |
| Concat | level별 pooled vector concat 후 projection |
| Learned level weight | level별 가중치를 학습 |
| Scale-selected | GT 크기 또는 detector scale에 맞는 level만 사용 |

초기 구현은 단순성과 안정성을 위해 mean aggregation을 권장한다.

## None Label 처리

`0: none` label은 중요한 안정화 장치다. MissBank에서 target GT가 없는 이미지도 많기 때문에, none label이 없으면 MissHead가 항상 특정 region을 예측하도록 학습되어 잘못된 feature injection을 유발할 수 있다.

초기 정책은 다음과 같다.

```text
target_label == 0:
    loss에는 포함
    injection은 수행하지 않음
```

다만 none sample이 너무 많으면 MissHead가 none만 예측할 수 있다. 이 경우 다음 전략을 고려한다.

| 전략 | 설명 |
|---|---|
| class weight | none label loss weight를 낮춤 |
| balanced sampling | region target이 있는 batch 비율을 높임 |
| focal-style CE | 쉬운 none sample의 영향 감소 |
| loss warm-up | start_epoch 이후 loss weight를 점진적으로 증가 |

## Configuration 초안

MissHead 관련 설정은 `modules/cfg/remiss.yaml` 안에 포함하는 것이 자연스럽다.

```yaml
miss_head:
  enabled: false
  start_epoch: 1
  grid_size: 2
  in_channels: 256
  hidden_dim: 256
  num_layers: 2
  dropout: 0.0
  pooling: gap
  level_aggregation: mean
  loss_weight: 0.1
  ignore_none_loss: false
  none_loss_weight: 1.0
  class_balanced_loss: false
```

`class_balanced_loss: true`를 켜면 batch 안의 `0..num_regions` target label 빈도를 기준으로 inverse-frequency CE weight를 만든다. 이때 현재 batch에 존재하는 각 class가 loss에서 같은 총 기여도를 갖는다. `none_loss_weight`는 class-balanced weight 위에 추가로 곱해지므로, 0 class까지 완전히 동일하게 맞추려면 `none_loss_weight: 1.0`을 사용한다.

현재 구현은 FCOS에 대해 MissHead loss-only 학습을 지원한다. `miss_head.enabled: true`이고 `epoch >= miss_head.start_epoch`이면 FCOS training loss dict에 `miss_head_ce`가 추가된다. Prototype injection은 아직 연결하지 않는다.

`grid_size`는 MissBank와 MissInject의 grid 설정과 일치해야 한다. 구현에서는 ReMiss 상위 config의 `grid_size`를 공유하거나, 불일치 시 validation error를 내는 방식이 안전하다.

## Public API 초안

```python
class MissHead(nn.Module):
    def forward(self, features) -> torch.Tensor:
        ...

    def compute_loss(
        self,
        region_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...

    def predict_region(
        self,
        region_logits: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

반환 예시:

```python
{
    "miss_head_ce": ce_loss * loss_weight,
}
```

`predict_region`은 `[B]` shape의 integer tensor를 반환한다.

## 학습 흐름

초기 ReMiss training 흐름에서 MissHead는 다음 위치에 들어간다.

```text
1. detector backbone/neck feature 추출
2. MissHead가 region_logits 예측
3. detector head가 기본 detection loss 계산
4. detector 최종 prediction과 GT를 매칭해 MissBank update
5. MissBank에서 target_labels 생성
6. epoch >= start_epoch이면 MissHead CE loss 추가
7. region_logits의 argmax 또는 target label을 MissInject에 전달
```

주의할 점은 MissBank update에 필요한 최종 detection이 detector post-processing 이후 결과라는 점이다. 현재 runtime은 ReMiss가 활성화된 경우 training step 이후 eval-style no-grad prediction을 한 번 더 계산해서 MissBank를 갱신한다. 이 경로는 memory update용이며, MissHead loss와 injection은 별도 구현 단계에서 연결한다.

## 추론 흐름

추론 시에는 GT와 MissBank target이 없다. MissHead는 feature만 보고 region을 예측한다.

```text
1. feature 추출
2. MissHead region_logits 계산
3. pred_region = argmax(region_logits)
4. pred_region == 0이면 injection 없음
5. pred_region > 0이면 해당 region prototype injection
6. detector head와 post-processing 수행
```

## 평가 지표

MissHead 평가는 단순히 detector mAP만 보는 대신, MissHead가 MissBank target을 제대로 학습했는지와 그 예측이 실제 미검출 완화로 이어지는지를 분리해서 기록한다. 기본 label은 `0: none`, `1..num_regions: missed region`으로 둔다.

아래 notation을 사용한다.

```text
y_i = image i의 MissBank target label
p_i = image i의 MissHead predicted label
M_i = image i에서 현재 missed 상태이고 miss_count >= miss_threshold인 GT 집합
r_ij = image i의 missed GT j가 속한 region label
c_ij = image i의 missed GT j의 miss_count
```

| 지표 | 로그 키 | 정의 | 목적 |
|---|---|---|---|
| MissHead 5-way Accuracy | `miss_head_acc` | `mean(1[p_i == y_i])` | none을 포함한 전체 image-level target 예측 정확도 |
| MissHead Non-None Accuracy | `miss_head_non_none_acc` | `mean(1[p_i == y_i] | y_i > 0)` | 실제 미검출 region target이 있는 이미지에서 사분면 예측이 맞는지 확인 |
| Missed Object Region Accuracy | `missed_object_region_acc` | `sum_i sum_j 1[p_i == r_ij] / sum_i |M_i|` | image-level 예측이 각 missed GT의 위치를 얼마나 맞추는지 object-level로 평가 |
| Weighted Missed Object Region Accuracy | `missed_object_region_acc_weighted` | `sum_i sum_j c_ij * 1[p_i == r_ij] / sum_i sum_j c_ij` | 반복적으로 미검출되는 GT를 더 중요하게 반영 |
| None Precision / Recall | `miss_head_none_precision`, `miss_head_none_recall` | precision: `TP_none / predicted_none`, recall: `TP_none / target_none` | MissHead가 과도하게 none으로 collapse하거나 불필요한 injection을 유발하는지 확인 |
| Detector Recall on Missed Objects / FN Count | `missed_object_recall`, `missed_object_fn_count` | recall: `matched_missed_gt / total_missed_gt`, count: `total_missed_gt - matched_missed_gt` | MissHead와 injection이 실제 detector 미검출을 줄이는지 확인 |

`Missed Object Region Accuracy`와 `Weighted Missed Object Region Accuracy`는 MissHead 전용 핵심 지표로 본다. detector mAP가 상승하더라도 이 값이 낮으면 injection이 MissHead의 위치 예측이 아니라 다른 regularization 효과로 성능을 얻었을 가능성이 있다.

반대로 `Detector Recall on Missed Objects`는 MissHead 자체의 classification 정확도는 아니며, ReMiss 전체 경로의 downstream 효과를 보는 지표다. 따라서 MissHead ablation에서는 MissHead 지표와 detector 지표를 같이 보고 해석한다.

## Logging

MissHead의 평가 지표와 진단 통계는 detector의 기본 `results.csv`에 섞지 않고 ReMiss 전용 출력으로 저장한다.

| 출력 파일 | 설명 |
|---|---|
| `remiss/miss_head_epoch.json` | epoch별 MissHead loss와 train metric이 누적되는 JSON list |
| `remiss/miss_head_epoch.csv` | `miss_head_epoch.json`을 평탄화한 CSV |

기본 `history.json`과 `results.csv`에는 detector train/validation metric만 남긴다. MissHead 관련 key는 `miss_head_*`, `missed_object_*` prefix를 기준으로 분리한다.

MissHead는 평가 지표 외에도 다음 진단 통계를 기록하는 것이 좋다.

| 통계 | 설명 |
|---|---|
| `miss_head_loss` | weighted CE loss |
| `miss_head_none_ratio` | target label이 none인 비율 |
| `miss_head_pred_histogram` | 예측 region 분포 |
| `miss_head_target_histogram` | target region 분포 |
| `miss_head_entropy` | softmax entropy 평균 |

특히 none ratio와 pred histogram은 MissHead가 none collapse 또는 특정 region collapse를 보이는지 확인하는 데 중요하다.

## Ablation

MissHead 검증을 위해 다음 ablation을 권장한다.

| 실험 | 목적 |
|---|---|
| detector baseline | 기준 성능 |
| MissHead loss only | auxiliary supervision 자체의 영향 확인 |
| MissHead + predicted hard injection | 실제 ReMiss 경로 |
| target label hard injection | region 예측 오류를 제거한 upper-bound |
| none loss weight 변화 | none collapse 여부 확인 |
| `2x2` vs `4x4` | spatial granularity 영향 확인 |
| backbone feature vs neck feature | 입력 feature 위치 영향 확인 |
| mean vs concat aggregation | FPN aggregation 영향 확인 |

## 리스크

MissHead는 image-level coarse label을 예측하므로 supervision이 약하다. 반복 미검출 객체가 이미지의 작은 일부에만 존재하거나, 여러 region에 target이 동시에 존재하면 hard label 하나로는 정보 손실이 생긴다.

또한 none label이 많으면 MissHead가 보수적으로 none만 예측할 수 있고, 반대로 loss weight가 너무 크면 detector의 원래 representation 학습을 방해할 수 있다. 초기 실험에서는 loss weight를 작게 두고, target histogram과 prediction histogram을 함께 확인해야 한다.

## 확장 방향

초기 hard label MissHead 이후 다음 확장을 고려한다.

| 확장 | 설명 |
|---|---|
| Soft target | region별 miss score 분포를 soft label로 사용 |
| Soft injection | region probability로 prototype mixture 주입 |
| Multi-label head | 여러 region에 동시에 미검출이 있을 때 multi-hot target 사용 |
| Heatmap head | coarse grid classification 대신 low-resolution heatmap 예측 |
| Class-aware head | region과 class failure를 함께 예측 |
| Scale-aware head | FPN level 또는 객체 크기별 failure region 예측 |

초기 목표는 단순하고 해석 가능한 5-way MissHead를 구현한 뒤, MissBank target과 MissInject injection이 실제 recall 개선으로 이어지는지 확인하는 것이다.
