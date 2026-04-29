# MissBank 설계 문서

## 개요

MissBank는 ReMiss 방법론에서 반복 미검출 GT를 추적하는 train-time memory module이다. Detector가 어떤 GT 객체를 연속으로 놓치는지 기록하고, 해당 정보를 auxiliary region head의 학습 target 생성에 사용한다.

MissBank의 목적은 단순히 hard sample을 저장하는 것이 아니라, GT 단위의 반복 실패 상태를 안정적으로 관리하는 것이다. ReMiss는 이 상태를 기반으로 미검출이 집중되는 spatial region을 예측하고, 해당 region에 대응하는 prototype injection을 수행한다.

## 역할

MissBank는 다음 책임을 가진다.

1. 이미지별, GT별 미검출 상태를 저장한다.
2. GT가 연속으로 미검출된 횟수인 `miss_count`를 관리한다.
3. GT box가 속한 spatial region을 계산하고 저장한다.
4. 현재 상태가 미검출이고 `miss_count >= miss_threshold`인 GT를 target 후보로 반환한다.
5. 이미지 단위 5-way region label을 생성할 수 있는 정보를 제공한다.

MissBank는 detector의 forward feature를 직접 변경하지 않는다. Feature injection은 별도 모듈인 MissInject 또는 prototype injection module이 담당한다.

## 기본 용어

| 용어 | 설명 |
|---|---|
| GT instance | 학습 데이터의 annotation 하나 |
| Detection | 최종 post-processing 이후 남은 예측 box |
| Matched detection | class, score, IoU 조건을 만족해 GT와 매칭된 detection |
| Missed GT | matched detection이 없는 GT |
| `miss_count` | 해당 GT가 연속으로 missed 상태였던 횟수 |
| Target GT | 현재 missed 상태이고 `miss_count`가 threshold 이상인 GT |
| Region | 이미지 또는 feature map을 나눈 grid cell |

## Record 구조

MissBank는 GT 단위 record를 저장한다. 기본 record는 다음 필드를 가진다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `image_id` | str 또는 int | 이미지 식별자 |
| `gt_id` | str 또는 int | GT 식별자 |
| `gt_class` | int | GT class id |
| `bbox_xyxy` | tuple[float, float, float, float] | 원본 이미지 좌표계의 GT box |
| `region_id` | int | GT가 가장 많이 걸쳐 있는 region id |
| `miss_count` | int | 연속 미검출 횟수 |
| `is_missed` | bool | 현재 update에서의 미검출 여부 |
| `last_epoch` | int | 마지막으로 갱신된 epoch |
| `last_step` | int | 마지막으로 갱신된 step |
| `last_iou` | float 또는 None | matched detection의 IoU, 없으면 None |
| `last_score` | float 또는 None | matched detection의 score, 없으면 None |

구현에서는 다음 key를 기본으로 사용한다.

```text
record_key = (image_id, gt_id)
```

Dataset에 안정적인 annotation id가 있으면 이를 `gt_id`로 사용한다. 안정적인 id가 없으면 `image_id`와 이미지 내 annotation index를 조합해 만든다. 단, annotation 순서가 epoch마다 바뀌면 안 된다.

## Region ID 규칙

초기 baseline은 `2x2` grid를 사용한다.

| Label | 의미 |
|---|---|
| `0` | none |
| `1` | 좌상단 |
| `2` | 우상단 |
| `3` | 좌하단 |
| `4` | 우하단 |

MissBank record의 `region_id`는 `1..num_regions`만 저장한다. `0`은 이미지 단위 target 생성에서 target 후보가 없을 때 사용하는 none label이다.

`NxN` grid로 확장할 때는 row-major 순서를 사용한다.

```text
region_id = row * grid_size + col + 1
```

예를 들어 `4x4` grid에서는 region id가 `1..16`이고, `0`은 none이다.

## GT의 Region 계산

GT box가 여러 region에 걸쳐 있으면 가장 많이 겹치는 region을 선택한다.

```text
region_id(gt) = argmax_q area(intersection(gt_box, region_q))
```

동률이면 row-major 순서상 더 앞선 region을 선택한다. 이렇게 하면 동일 입력에 대해 deterministic한 target을 만들 수 있다.

Region 계산은 원본 이미지 좌표계를 기준으로 수행하는 것을 기본으로 한다. Augmentation 이후 좌표계를 기준으로 할 경우, detector 입력과 feature 위치의 정렬은 좋아지지만 같은 GT의 region이 augmentation에 따라 달라질 수 있다. 초기 버전은 구현 단순성과 기록 안정성을 위해 원본 이미지 좌표계를 권장한다.

## 미검출 판정

MissBank update는 GT 목록과 최종 detection 목록을 입력으로 받는다. Detection은 score threshold와 NMS가 적용된 최종 결과를 사용한다.

GT가 검출된 것으로 판정되려면 다음 조건을 만족하는 detection이 하나 이상 있어야 한다.

1. `pred_class == gt_class`
2. `pred_score >= score_threshold`
3. `IoU(pred_box, gt_box) >= match_iou_threshold`

조건을 만족하는 detection이 여러 개면 IoU가 가장 높은 detection을 matched detection으로 선택한다. matched detection이 없으면 해당 GT는 missed 상태다.

## Update 규칙

각 학습 step에서 MissBank는 batch의 모든 GT를 갱신한다.

```text
if gt is missed:
    miss_count = previous_miss_count + 1
    is_missed = true
    last_iou = None
    last_score = None
else:
    miss_count = 0
    is_missed = false
    last_iou = matched_iou
    last_score = matched_score
```

새로운 GT record가 처음 등장하면 `miss_count`는 `0`에서 시작한다. 해당 update에서 missed이면 곧바로 `1`이 된다.

## Target GT 선택

Auxiliary region head의 학습 target으로 사용할 GT는 다음 조건을 만족해야 한다.

```text
is_target_gt(record) =
    record.is_missed
    and record.miss_count >= miss_threshold
```

`miss_threshold`가 1이면 현재 한 번이라도 missed인 GT가 target 후보가 된다. 값이 커질수록 반복적으로 놓치는 GT만 target으로 선택한다.

## 이미지 단위 Label 생성

Region head는 이미지 단위 5-way label을 예측한다. 따라서 MissBank는 이미지 내 target GT들을 region label 하나로 집계한다.

기본 집계 방식은 `miss_count` 가중 합이다.

```text
region_score[q] = sum(
    record.miss_count
    for record in target_records_of_image
    if record.region_id == q
)
```

최종 label은 score가 가장 큰 region이다.

```text
if no target_records:
    target_label = 0
else:
    target_label = argmax_q region_score[q]
```

동률이면 `region_score`가 같은 후보 중 row-major 순서상 앞선 region을 선택한다.

## `start_epoch`과의 관계

MissBank update는 학습 시작부터 수행할 수 있다. 다만 ReMiss auxiliary module의 loss와 injection은 `start_epoch` 이후에만 활성화한다.

권장 동작은 다음과 같다.

| Epoch 조건 | MissBank update | Region loss | Prototype injection |
|---|---|---|---|
| `epoch < start_epoch` | 수행 | 비활성 | 비활성 |
| `epoch >= start_epoch` | 수행 | 활성 | 활성 |

이 설계는 초기 detector가 불안정할 때 생성되는 noisy target이 바로 auxiliary module을 학습시키는 문제를 줄인다.

## Public API 초안

구현 시 MissBank는 다음 API를 제공하는 구조가 적절하다.

```python
class MissBank:
    def update(
        self,
        *,
        targets,
        detections,
        epoch: int | None = None,
        step: int = 0,
        image_sizes=None,
    ) -> dict[str, int]:
        ...

    def get_image_targets(
        self,
        image_ids,
        *,
        miss_threshold: int | None = None,
    ) -> dict[str, int]:
        ...

    def get_batch_labels(
        self,
        targets=None,
        *,
        image_ids=None,
        miss_threshold: int | None = None,
        device=None,
    ):
        ...

    def get_records(
        self,
        image_id=None,
    ) -> list:
        ...

    def is_active(self, epoch: int | None = None) -> bool:
        ...

    def reset(self) -> None:
        ...
```

구현된 `update`는 detector의 최종 detection 결과와 target GT를 매칭해 record를 갱신한다. `targets`는 `boxes`, `labels`, `image_id`, 선택적 `gt_ids` 또는 `annotation_ids`를 포함하는 detection target 형식을 사용한다. `get_image_targets`는 각 이미지에 대해 `0..num_regions` 범위의 label을 반환하고, `get_batch_labels`는 학습 loss에 바로 넣을 수 있는 tensor label을 반환한다. `is_active`는 `start_epoch` 이후 auxiliary loss와 injection을 켤 때 사용할 수 있다.

반환 예시는 다음과 같다.

```python
{
    image_id_1: 0,
    image_id_2: 3,
    image_id_3: 1,
}
```

## Configuration

MissBank 관련 주요 설정은 다음과 같다.

| 설정 | 기본값 | 설명 |
|---|---:|---|
| `enabled` | `false` | MissBank 사용 여부 |
| `grid_size` | `2` | spatial grid 크기 |
| `target.miss_threshold` | `2` | target GT로 사용할 최소 연속 미검출 횟수 |
| `matching.score_threshold` | `0.05` | 최종 검출 score threshold |
| `matching.iou_threshold` | `0.5` | GT 검출 여부 판정 IoU |
| `start_epoch` | 실험 설정 | region loss와 injection 시작 epoch |
| `max_records` | `None` | memory record 상한, 대규모 dataset에서 사용 |

## 분산 학습 고려사항

Distributed Data Parallel 학습에서는 각 process가 서로 다른 mini-batch를 본다. 가장 단순한 구현은 process-local MissBank를 사용하는 것이다. 이 방식은 구현이 쉽지만, 같은 이미지가 여러 process에 분산되어 나타나는 경우 record 동기화가 되지 않는다.

초기 버전에서는 다음 조건을 권장한다.

1. 각 이미지가 한 epoch 안에서 하나의 process에만 배정되도록 sampler를 사용한다.
2. MissBank는 process-local로 유지한다.
3. Validation이나 logging에서는 rank 0에서만 요약 통계를 출력한다.

추후 더 정확한 전역 memory가 필요하면 epoch 종료 시 record를 all-gather해 병합하는 방식을 고려한다.

## Checkpoint 저장

MissBank는 train-time state이므로 checkpoint에 저장할지 여부를 선택할 수 있다.

초기 실험에서는 저장을 권장한다. 중간 checkpoint에서 재개할 때 `miss_count`가 초기화되면 `start_epoch` 이후 target 생성이 흔들릴 수 있기 때문이다.

저장 대상은 다음과 같다.

```text
records
grid_size
target.miss_threshold
matching.score_threshold
matching.iou_threshold
```

Dataset이 바뀌거나 annotation id 체계가 바뀌면 기존 MissBank state는 폐기해야 한다.

## Epoch 안정성 지표

MissHead의 `start_epoch`을 정하려면 MissBank target이 충분히 안정화됐는지 확인해야 한다. 이를 위해 MissBank는 매 epoch 종료 시 현재 epoch에서 miss된 GT snapshot을 만들고, 직전 epoch snapshot과 비교한 안정성 지표를 계산한다.

학습 결과는 `output_dir/remiss/` 아래에 JSON으로 저장한다.

| 파일 | 설명 |
|---|---|
| `miss_stability_epoch.json` | epoch별 안정성 지표가 누적되는 JSON list |
| `miss_stability_epoch.csv` | `miss_stability_epoch.json`을 평탄화한 CSV |
| `miss_stability_state.json` | 다음 epoch 비교를 위한 마지막 snapshot state |

기본 지표는 다음과 같다.

| 지표 | 설명 |
|---|---|
| `miss_gt_jaccard_stability` | 직전 epoch와 현재 epoch의 missed GT set Jaccard similarity |
| `miss_gt_churn_rate` | `1 - miss_gt_jaccard_stability` |
| `new_miss_rate` | 현재 missed GT 중 직전 epoch에는 missed가 아니었던 GT 비율 |
| `persistent_miss_ratio` | 현재 missed GT 중 `miss_count >= miss_threshold`인 target GT 비율 |
| `miss_region_js_divergence` | 직전 epoch와 현재 epoch의 missed region histogram JS divergence |
| `top1_miss_region_share` | 가장 많이 miss된 region이 전체 miss에서 차지하는 비율 |
| `miss_region_entropy` | missed region histogram의 normalized entropy |
| `miss_hotspot_overlap_at_k` | `(image_id, region_id)` hotspot Top-K가 직전 epoch와 겹치는 비율 |

`miss_gt_jaccard_stability`가 높고 `miss_gt_churn_rate`가 낮으면 같은 GT가 반복적으로 miss되고 있다는 뜻이다. `miss_region_js_divergence`가 낮아지고 `miss_hotspot_overlap_at_k`가 높아지면 특정 image-region 단위의 실패 패턴이 안정화됐다고 볼 수 있다. 이 시점을 MissHead 학습 시작 후보로 사용한다.

## Logging

MissBank는 다음 통계를 주기적으로 기록하는 것이 좋다.

| 통계 | 설명 |
|---|---|
| `num_records` | 저장 중인 GT record 수 |
| `num_missed` | 현재 missed 상태인 GT 수 |
| `num_target_gts` | `miss_threshold` 이상인 target GT 수 |
| `target_image_ratio` | batch 내 target label이 none이 아닌 이미지 비율 |
| `region_histogram` | region별 target 분포 |
| `mean_miss_count` | missed GT의 평균 연속 미검출 횟수 |
| `max_miss_count` | 최대 연속 미검출 횟수 |

이 통계는 MissBank가 너무 많은 이미지를 target으로 만들거나 특정 region에 과도하게 쏠리는 문제를 조기에 확인하는 데 필요하다.

## Edge Cases

| 상황 | 처리 |
|---|---|
| 이미지에 GT가 없음 | target label은 `0` |
| GT box가 이미지 밖으로 일부 나감 | image boundary로 clamp 후 region 계산 |
| GT box 면적이 0 | record 생성 제외 |
| detection이 없음 | 모든 GT를 missed로 처리 |
| class는 맞지만 IoU가 낮음 | missed로 처리 |
| IoU는 높지만 class가 다름 | missed로 처리 |
| score threshold 미만 detection만 있음 | missed로 처리 |

## 초기 구현 우선순위

1. Single-process MissBank 구현
2. `2x2` region 계산
3. 최종 detection과 GT의 class-aware IoU matching
4. `miss_count` update
5. `miss_threshold` 기반 image label 생성
6. checkpoint 저장과 로드
7. logging 통계 추가

`4x4`, `8x8`, soft label, distributed global merge는 초기 성능 확인 이후 확장한다.

## 검증 항목

MissBank는 다음 단위 테스트 또는 smoke test가 필요하다.

1. GT가 detection과 매칭되면 `miss_count`가 0으로 초기화되는지 확인한다.
2. GT가 연속 missed이면 `miss_count`가 step마다 증가하는지 확인한다.
3. `miss_count < miss_threshold`이면 target label에 반영되지 않는지 확인한다.
4. target GT가 없으면 image label이 `0`인지 확인한다.
5. GT가 여러 region에 걸치면 가장 큰 overlap region을 선택하는지 확인한다.
6. 여러 target GT가 있으면 region별 `miss_count` 합산으로 label을 선택하는지 확인한다.

## 설계 요약

MissBank는 ReMiss의 failure memory 역할을 담당한다. 초기 버전에서는 단순하고 해석 가능한 형태를 우선한다.

- GT 단위 record 저장
- 현재 missed 여부와 연속 미검출 횟수 관리
- `2x2` region hard label 생성
- `miss_threshold` 이상인 반복 실패만 target화
- `start_epoch` 전에도 memory는 갱신하되 auxiliary 학습은 지연

이 구조는 초기 hard injection 실험에 충분하며, 이후 soft label, soft injection, gated additive, 고해상도 grid로 자연스럽게 확장할 수 있다.
