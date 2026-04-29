# MissInjection 설계 문서

## 개요

MissInjection은 ReMiss에서 MissHead가 예측한 취약 spatial region에 학습 가능한 prototype embedding을 주입하는 모듈이다. 목적은 detector가 반복적으로 놓치는 위치의 feature 표현을 보정하여, 최종 detection head가 해당 영역의 object evidence를 더 잘 활용하도록 만드는 것이다.

현재 repository 상태에서 MissBank와 MissHead loss-only 학습은 구현되어 있지만, MissInjection은 아직 detector forward에 연결되어 있지 않다. 이 문서는 이후 구현할 MissInjection의 동작, API, 설정, ablation 기준을 정리한다.

## 역할

MissInjection은 다음 책임을 가진다.

1. MissHead의 region 예측 결과를 입력으로 받는다.
2. region id에 대응하는 learnable prototype을 선택한다.
3. feature map에서 해당 region에 해당하는 공간 영역을 찾는다.
4. 선택된 prototype을 feature 영역에 additive 방식으로 주입한다.
5. `0: none` 예측이면 feature를 변경하지 않는다.

초기 구현은 단순성과 해석 가능성을 위해 hard additive injection만 지원한다. 이후 soft injection, gated additive, class-aware prototype으로 확장한다.

## Label 체계

기본 grid는 `2x2`이며 MissHead와 동일한 label 체계를 사용한다.

| Label | 의미 | 동작 |
|---|---|---|
| `0` | none | injection 없음 |
| `1` | 좌상단 | 좌상단 feature region에 prototype 주입 |
| `2` | 우상단 | 우상단 feature region에 prototype 주입 |
| `3` | 좌하단 | 좌하단 feature region에 prototype 주입 |
| `4` | 우하단 | 우하단 feature region에 prototype 주입 |

`NxN` grid로 확장할 경우 region id는 row-major 순서를 따른다.

```text
region_id = row * grid_size + col + 1
```

## Prototype

각 spatial region에는 하나의 learnable prototype embedding을 둔다.

```text
prototypes.shape = [num_regions, channels]
```

`2x2` grid, `channels=256`이면 다음 형태가 된다.

```text
prototypes.shape = [4, 256]
```

`0: none` label에는 prototype을 두지 않는다. 따라서 `region_id > 0`일 때만 `prototypes[region_id - 1]`을 선택한다.

## Hard Additive Injection

초기 구현은 hard prediction을 사용한다.

```text
pred_region = argmax(region_logits)
```

`pred_region == 0`이면 feature를 그대로 반환한다. `pred_region > 0`이면 해당 region의 prototype을 feature map의 같은 공간 region에 더한다.

```text
F'_b[:, y1:y2, x1:x2] =
    F_b[:, y1:y2, x1:x2] + alpha * P[pred_region_b - 1]
```

여기서 `alpha`는 injection strength이다. 초기값은 `1.0`을 사용하고, ablation에서 `0.25`, `0.5`, `1.0` 등을 비교할 수 있다.

## Region Mask 계산

feature map 크기가 `[B, C, H, W]`이고 `grid_size=N`이면 각 region은 feature map을 균등 분할한 cell이다.

```text
cell_h = H / N
cell_w = W / N
row = (region_id - 1) // N
col = (region_id - 1) % N
```

정수 index는 다음처럼 계산한다.

```text
y1 = floor(row * H / N)
y2 = floor((row + 1) * H / N)
x1 = floor(col * W / N)
x2 = floor((col + 1) * W / N)
```

마지막 row/column은 rounding 오차를 피하기 위해 각각 `H`, `W`까지 포함한다.

## 입력과 출력

MissInjection의 기본 입력은 다음과 같다.

```python
features: dict[str, Tensor] | list[Tensor] | Tensor
region_logits: Tensor  # [B, num_regions + 1]
```

출력은 입력과 같은 구조의 feature이다.

```python
injected_features: same structure as features
```

초기 FCOS 구현에서는 FPN output인 `P3..P7` feature list 또는 dict를 입력으로 받는다.

## Injection 위치

MissInjection은 detector의 feature 흐름 중 어디에 들어갈지 ablation 대상이다.

| 위치 | 설명 | 초기 우선순위 |
|---|---|---:|
| Backbone output | FPN 이전 backbone feature에 주입 | 낮음 |
| Neck/FPN output | detector head 입력 feature에 주입 | 높음 |
| 단일 FPN level | 특정 scale feature에만 주입 | 중간 |
| 전체 FPN level | 모든 FPN level에 같은 region 기준으로 주입 | 높음 |

초기 구현은 FCOS의 FPN output 전체 level에 동일한 predicted region을 적용하는 방식을 권장한다. MissHead도 FPN level을 aggregate해서 image-level region을 예측하므로, 같은 위치에 주입하는 것이 가장 단순하다.

## 학습 시 동작

MissInjection은 MissHead가 충분히 학습되기 전에는 켜지지 않아야 한다. 따라서 MissBank/MissHead와 동일하게 `start_epoch`를 둔다.

| Epoch 조건 | MissBank update | MissHead loss | MissInjection |
|---|---|---|---|
| `epoch < start_epoch` | 수행 | 비활성 | 비활성 |
| `epoch >= start_epoch` | 수행 | 활성 | 활성 |

초기 실험에서는 두 가지 학습 경로를 비교한다.

| 경로 | 설명 |
|---|---|
| Predicted region injection | MissHead의 `argmax(region_logits)`로 prototype 주입 |
| GT region injection | MissBank target label로 주입하는 upper-bound ablation |

실제 ReMiss 경로는 predicted region injection이다. GT region injection은 MissHead 예측 오류를 제거했을 때 injection 자체의 상한 성능을 보기 위한 실험이다.

## 추론 시 동작

추론 시에는 GT와 MissBank target이 없으므로 MissHead 예측만 사용한다.

```text
1. backbone/neck feature 추출
2. MissHead가 region_logits 계산
3. pred_region = argmax(region_logits)
4. pred_region == 0이면 feature 변경 없음
5. pred_region > 0이면 해당 region prototype 주입
6. detector head와 post-processing 수행
```

## 설정 초안

MissInjection 설정은 `modules/cfg/remiss.yaml` 안에 다음 형태로 추가하는 것을 권장한다.

```yaml
miss_injection:
  enabled: false
  start_epoch: 1
  grid_size: 2
  in_channels: 256
  mode: hard
  method: additive
  source: predicted
  target_feature: neck
  apply_to_levels: all
  strength: 1.0
  detach_region_logits: true
```

필드 의미는 다음과 같다.

| 설정 | 설명 |
|---|---|
| `enabled` | MissInjection 사용 여부 |
| `start_epoch` | injection 시작 epoch |
| `grid_size` | MissHead/MissBank와 공유하는 grid 크기 |
| `in_channels` | prototype channel 수 |
| `mode` | `hard` 또는 향후 `soft` |
| `method` | 초기값 `additive`, 향후 `gated_additive` |
| `source` | `predicted` 또는 `target` |
| `target_feature` | `backbone` 또는 `neck` |
| `apply_to_levels` | `all` 또는 특정 FPN level 목록 |
| `strength` | additive injection 강도 |
| `detach_region_logits` | injection 경로에서 MissHead gradient를 끊을지 여부 |

초기 구현에서는 `mode: hard`, `method: additive`, `source: predicted`, `target_feature: neck`, `apply_to_levels: all`만 지원해도 충분하다.

## Public API 초안

구현 파일은 `modules/nn/mi.py` 또는 `modules/nn/injection.py`로 두는 것을 권장한다. 현재 naming 흐름이 `mb.py`, `mh.py`이므로 `mi.py`가 가장 일관적이다.

```python
class MissInjection(nn.Module):
    def forward(
        self,
        features,
        region_logits: torch.Tensor,
        *,
        target_labels: torch.Tensor | None = None,
        epoch: int | None = None,
    ):
        ...

    def predict_region(
        self,
        region_logits: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

초기 구현의 핵심 함수는 다음과 같다.

```python
def inject_feature(
    feature: torch.Tensor,
    region_ids: torch.Tensor,
) -> torch.Tensor:
    ...
```

`feature`는 `[B, C, H, W]`, `region_ids`는 `[B]` 형태이다.

## Gradient 설계

초기 구현에서는 injection이 detector loss를 통해 prototype을 학습하도록 한다.

```text
detector loss -> injected feature -> prototype
```

MissHead의 region 선택은 hard `argmax`이므로 선택 연산 자체는 미분 가능하지 않다. 따라서 초기 hard injection에서 MissHead는 `miss_head_ce`로 학습되고, prototype은 detector loss로 학습된다.

권장 기본값은 다음과 같다.

```text
detach_region_logits = true
```

이렇게 하면 detector loss가 MissHead logits로 역전파되지 않는다. MissHead는 auxiliary CE로만 학습되고, injection prototype은 detector loss로 학습되어 두 역할이 분리된다.

## Metrics

MissInjection 자체는 독립적인 정답 label을 갖지 않으므로 detector 성능과 MissHead 지표를 함께 봐야 한다.

권장 지표는 다음과 같다.

| 지표 | 목적 |
|---|---|
| `bbox_mAP_50_95` | 전체 detector 성능 |
| `bbox_mAP_50` | 완화된 IoU 기준 성능 |
| `missed_object_recall` | 반복 미검출 GT 회복 여부 |
| `missed_object_fn_count` | 반복 미검출 GT 중 남은 FN 수 |
| `miss_head_acc` | MissHead target 예측 품질 |
| `miss_head_non_none_acc` | 실제 target region이 있을 때 예측 품질 |
| `missed_object_region_acc` | missed GT 위치와 predicted region 일치도 |

Injection을 켰는데 detector 성능은 오르지만 MissHead 지표가 낮으면 prototype이 의도한 region 보정보다는 regularization처럼 동작했을 가능성이 있다. 반대로 MissHead 지표는 높지만 detector recall이 오르지 않으면 injection 방식이나 위치가 적절하지 않을 수 있다.

## Ablation

초기 ablation은 다음 순서를 권장한다.

| 실험 | 목적 |
|---|---|
| Baseline detector | 기준 성능 |
| MissBank + MissHead loss-only | auxiliary supervision 효과 |
| Predicted hard additive injection | 실제 ReMiss injection 효과 |
| Target hard additive injection | MissHead 오류를 제거한 upper-bound |
| Neck vs backbone injection | injection 위치 영향 |
| All FPN levels vs single level | scale 적용 범위 영향 |
| strength sweep | injection 강도 영향 |
| `2x2` vs `4x4` | region granularity 영향 |

## Edge Cases

| 상황 | 처리 |
|---|---|
| `pred_region == 0` | feature 변경 없이 반환 |
| batch 내 일부 sample만 region 예측 | sample별로 독립 처리 |
| feature map 크기가 grid보다 작음 | 최소 1 pixel 이상 cell이 생기도록 index clamp |
| `grid_size` 불일치 | MissBank, MissHead, MissInjection 간 validation error |
| FPN level channel 불일치 | prototype projection layer 또는 level별 prototype 필요 |
| DDP 학습 | prototype은 일반 parameter이므로 DDP가 gradient 동기화 |

## 현재 구현 상태

현재 코드 기준 상태는 다음과 같다.

| 구성 요소 | 상태 |
|---|---|
| MissBank | 구현됨 |
| MissBank stability JSON/CSV | 구현됨 |
| MissHead loss-only 학습 | FCOS에 대해 구현됨 |
| MissHead metrics logging | `history.json`, `results.csv`에 기록 |
| MissInjection prototype | 미구현 |
| MissInjection runtime 연결 | 미구현 |

따라서 다음 구현 단계는 `modules/nn/mi.py`에 MissInjection을 추가하고, FCOS wrapper에서 MissHead logits 계산 이후 detector head 입력 feature에 injection을 적용하는 것이다.

## 구현 순서 제안

1. `modules/nn/mi.py`에 `MissInjectionConfig`, `MissInjection` 추가
2. `modules/cfg/remiss.yaml`에 `miss_injection` section 추가
3. `scripts/runtime/registry.py`에서 `miss_injection.enabled`일 때 FCOS wrapper에 attach
4. `models/detection/wrapper/fcos.py`에서 FPN feature 생성 후 MissHead logits 계산
5. `epoch >= miss_injection.start_epoch`이면 feature injection 수행
6. injected feature를 detector head에 전달
7. `history.json`, `results.csv`에 injection 활성 비율과 predicted region histogram 기록

이 단계까지 구현하면 ReMiss의 전체 경로는 다음처럼 완성된다.

```text
MissBank -> MissHead target
FPN feature -> MissHead logits
MissHead logits -> MissInjection region 선택
MissInjection -> feature 보정
보정된 feature -> detector head
```

