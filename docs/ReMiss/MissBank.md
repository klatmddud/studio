# MissBank 설계 문서

## 개요

MissBank는 ReMiss 방법론에서 반복 미검출 GT를 추적하는 train-time memory module이다. Detector가 어떤 GT 객체를 연속으로 놓치는지 기록하고, 해당 정보를 image-level auxiliary target 생성에 사용한다.

MissBank의 목적은 단순히 hard sample을 저장하는 것이 아니라, GT 단위의 반복 실패 상태를 안정적으로 관리하는 것이다. 현재 구현은 spatial region을 나누지 않고, 이미지 전체에 반복 미검출 GT가 존재하는지 binary target으로 제공한다.

## 역할

MissBank는 다음 책임을 가진다.

1. 이미지별, GT별 미검출 상태를 저장한다.
2. GT가 연속으로 미검출된 횟수인 `miss_count`를 관리한다.
3. 현재 상태가 미검출이고 `miss_count >= miss_threshold`인 GT를 target 후보로 반환한다.
4. 이미지 단위 binary label을 생성할 수 있는 정보를 제공한다.

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
| Image target | 이미지 안에 Target GT가 하나 이상 있는지 나타내는 binary label |

## Record 구조

MissBank는 GT 단위 record를 저장한다. 기본 record는 다음 필드를 가진다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `image_id` | str 또는 int | 이미지 식별자 |
| `gt_id` | str 또는 int | GT 식별자 |
| `gt_class` | int | GT class id |
| `bbox_xyxy` | tuple[float, float, float, float] | 원본 이미지 좌표계의 GT box |
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

## Image-Level Target 규칙

MissBank는 spatial region을 나누지 않는다. 이미지 전체를 하나의 대상 영역으로 보고 binary label을 만든다.

| Label | 의미 |
|---|---|
| `0` | target missed GT 없음 |
| `1` | 이미지 안에 target missed GT가 하나 이상 있음 |

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

Image-level auxiliary target으로 사용할 GT는 다음 조건을 만족해야 한다.

```text
is_target_gt(record) =
    record.is_missed
    and record.miss_count >= miss_threshold
```

`miss_threshold`가 1이면 현재 한 번이라도 missed인 GT가 target 후보가 된다. 값이 커질수록 반복적으로 놓치는 GT만 target으로 선택한다.

## 이미지 단위 Label 생성

MissBank는 이미지 단위 binary label을 만든다. 현재 missed 상태이고 `miss_count >= miss_threshold`인 GT가 이미지 안에 하나라도 있으면 target label은 `1`이고, 없으면 `0`이다.

```text
if any(record.is_missed and record.miss_count >= miss_threshold for record in image_records):
    target_label = 1
else:
    target_label = 0
```

## `start_epoch`과의 관계

MissBank update는 학습 시작부터 수행할 수 있다. 다만 ReMiss auxiliary module의 loss와 injection은 `start_epoch` 이후에만 활성화한다.

권장 동작은 다음과 같다.

| Epoch 조건 | MissBank update | Image-level loss | Prototype injection |
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

구현된 `update`는 detector의 최종 detection 결과와 target GT를 매칭해 record를 갱신한다. `targets`는 `boxes`, `labels`, `image_id`, 선택적 `gt_ids` 또는 `annotation_ids`를 포함하는 detection target 형식을 사용한다. `get_image_targets`는 각 이미지에 대해 binary label `0` 또는 `1`을 반환하고, `get_batch_labels`는 학습 loss에 바로 넣을 수 있는 tensor label을 반환한다. `is_active`는 `start_epoch` 이후 auxiliary loss와 injection을 켤 때 사용할 수 있다.

반환 예시는 다음과 같다.

```python
{
    image_id_1: 0,
    image_id_2: 1,
    image_id_3: 1,
}
```

## Mining Type

MissBank는 `modules/cfg/remiss.yaml`의 `mining.type`으로 update 방식을 선택한다.

| 값 | 동작 |
|---|---|
| `online` | 각 optimization step 이후 같은 batch에 대해 no-grad detection을 한 번 더 수행하고 MissBank를 즉시 갱신한다. |
| `offline` | 학습 step 중에는 MissBank를 갱신하지 않고, `mining.start_epoch`와 `mining.interval_epoch` 조건을 만족하는 epoch 학습이 끝난 뒤 training loader를 no-grad로 한 번 더 순회하면서 MissBank를 갱신한다. |

`online`은 현재 batch의 detector 상태를 바로 반영하므로 구현이 단순하고 target이 빠르게 누적된다. 대신 학습 loop 안에서 매 step 추가 inference가 들어간다.

`offline`은 epoch 종료 시점의 모델로 전체 training set을 동일한 기준으로 다시 평가하므로 MissBank snapshot과 stability metric이 더 일관적이다. 대신 scheduled epoch마다 training loader를 한 번 더 돌기 때문에 시간이 더 걸리고, MissHead 학습 target은 직전 mining까지 누적된 MissBank 상태를 사용하게 된다. Runtime은 offline MissBank mining이 실행된 epoch의 wall-clock 시간을 `history.json`과 `results.csv`의 `remiss_mining_time_sec`에 기록한다.

기본값은 `online`이다.

## Configuration

MissBank 관련 주요 설정은 다음과 같다.

| 설정 | 기본값 | 설명 |
|---|---:|---|
| `enabled` | `false` | MissBank 사용 여부 |
| `mining.start_epoch` | `1` | offline mining을 처음 실행할 epoch |
| `mining.interval_epoch` | `1` | offline mining 실행 간격. 예: `5`이면 `mining.start_epoch`부터 5 epoch마다 실행 |
| `target.miss_threshold` | `2` | target GT로 사용할 최소 연속 미검출 횟수 |
| `loss_weight.enabled` | `false` | FCOS positive loss에 MissBank `miss_count` 기반 GT weight 적용 여부 |
| `loss_weight.alpha` | `0.5` | `1 + alpha * log1p(miss_count)`의 증가 계수 |
| `loss_weight.max_weight` | `2.0` | GT loss weight 상한 |
| `loss_weight.min_miss_count` | `1` | loss weight를 1보다 크게 만들 최소 `miss_count` |
| `loss_weight.min_observations` | `1` | loss weight를 적용할 최소 관측 횟수 |
| `loss_weight.normalize_batch_mean` | `true` | batch foreground weight 평균을 1로 정규화해 전체 loss scale 변동을 줄임 |
| `matching.score_threshold` | `auto` | detector의 최종 검출 score threshold를 사용한다. FCOS에서는 `head.score_thresh`로 resolve된다. |
| `matching.iou_threshold` | `auto` | detector의 최종 post-processing IoU threshold를 사용한다. FCOS에서는 `head.nms_thresh`로 resolve된다. |
| `start_epoch` | 실험 설정 | auxiliary loss와 injection 시작 epoch |
| `max_records` | `None` | memory record 상한, 대규모 dataset에서 사용 |

## Miss-Count Loss Weighting

`loss_weight.enabled: true`이면 FCOS training에서 각 positive location이 matched된 GT의 `miss_count`로 loss weight를 만든다.

```text
weight = min(max_weight, 1 + alpha * log1p(miss_count))
```

`min_miss_count`와 `min_observations`를 만족하지 못한 GT는 weight `1.0`을 사용한다. `normalize_batch_mean: true`이면 batch 안의 positive location weight 평균을 1로 맞춰서 hard GT에 상대적으로 더 큰 gradient를 주되 전체 loss scale 증가는 제한한다. 현재 이 연결은 FCOS classification, box regression, centerness loss에만 적용되며 Faster R-CNN은 MissBank logging과 replay source로만 사용한다.

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
target.miss_threshold
matching.score_threshold
matching.iou_threshold
```

Dataset이 바뀌거나 annotation id 체계가 바뀌면 기존 MissBank state는 폐기해야 한다.

## Failure-Type 기록

현재 runtime은 MissBank epoch summary를 `output_dir/missbank/` 아래에 저장한다. FTMB는 `modules/cfg/ftmb.yaml`에서 독립적으로 켜고 끄며, localization, classification, both, missed, duplicate, background failure type별 개수를 저장한다.

학습 결과는 `output_dir/ftmb/` 아래에 JSON으로 저장한다.

| 파일 | 설명 |
|---|---|
| `failure_type_epoch.json` | epoch별 failure type 개수가 누적되는 JSON list |
| `failure_type_epoch.csv` | `failure_type_epoch.json`을 평탄화한 CSV |
| `failure_type_state.json` | 마지막 count-only failure type snapshot state |

기본 failure type은 다음과 같다.

| Type | 설명 |
|---|---|
| `localization` | class는 맞지만 IoU가 matching threshold보다 낮은 GT failure |
| `classification` | IoU는 충분하지만 class가 틀린 GT failure |
| `both` | class도 틀리고 localization도 부족하지만 background threshold 이상 겹치는 GT failure |
| `missed` | 의미 있는 overlap prediction이 없는 GT failure |
| `duplicate` | 이미 matched된 GT에 대한 중복 prediction failure |
| `background` | 어떤 GT와도 background threshold 이상 겹치지 않는 prediction failure |

이 기록은 이후 Type-Aware Replay가 failure type별 replay slot을 구성하는 데 사용한다.

## Logging

MissBank는 다음 통계를 주기적으로 기록하는 것이 좋다.

| 통계 | 설명 |
|---|---|
| `num_seen_gts` | 해당 epoch에서 관측한 GT 수 |
| `num_missed_gts` | 해당 epoch에서 missed 상태인 GT 수 |
| `num_target_gts` | `miss_threshold` 이상인 target GT 수 |
| `num_images_seen` | 해당 epoch에서 관측한 이미지 수 |
| `num_images_with_miss` | missed GT가 하나 이상 있는 이미지 수 |
| `num_images_with_target` | target GT가 하나 이상 있는 이미지 수 |
| `missed_gt_ratio` | 관측 GT 중 missed GT 비율 |
| `target_gt_ratio` | 관측 GT 중 target GT 비율 |
| `image_miss_ratio` | 관측 이미지 중 missed GT가 있는 이미지 비율 |
| `image_target_ratio` | 관측 이미지 중 target GT가 있는 이미지 비율 |
| `mean_miss_count` | missed GT의 평균 연속 미검출 횟수 |
| `max_miss_count` | 최대 연속 미검출 횟수 |
| `mean_gt_miss_rate` | GT별 누적 미검출 비율의 평균 |
| `miss_gt_jaccard_stability` | 직전 MissBank snapshot 대비 missed GT set Jaccard |
| `miss_gt_churn_rate` | `1 - miss_gt_jaccard_stability` |
| `new_miss_rate` | 현재 missed GT 중 직전 snapshot에 없던 GT 비율 |
| `persistent_miss_ratio` | 현재 missed GT 중 target GT 비율 |

Runtime은 이 epoch summary를 `output_dir/missbank/missbank_epoch.json`과 `output_dir/missbank/missbank_epoch.csv`에 저장한다. Region histogram, region entropy, image-region hotspot 같은 region 기반 지표는 MissBank 출력에 저장하지 않는다.

## Edge Cases

| 상황 | 처리 |
|---|---|
| 이미지에 GT가 없음 | target label은 `0` |
| GT box가 이미지 밖으로 일부 나감 | image boundary로 clamp 후 record 갱신 |
| GT box 면적이 0 | record 생성 제외 |
| detection이 없음 | 모든 GT를 missed로 처리 |
| class는 맞지만 IoU가 낮음 | missed로 처리 |
| IoU는 높지만 class가 다름 | missed로 처리 |
| score threshold 미만 detection만 있음 | missed로 처리 |

## 초기 구현 우선순위

1. Single-process MissBank 구현
2. 최종 detection과 GT의 class-aware IoU matching
3. `miss_count` update
4. `miss_threshold` 기반 binary image label 생성
5. checkpoint 저장과 로드
6. logging 통계 추가

soft label, distributed global merge는 초기 성능 확인 이후 확장한다.

## 검증 항목

MissBank는 다음 단위 테스트 또는 smoke test가 필요하다.

1. GT가 detection과 매칭되면 `miss_count`가 0으로 초기화되는지 확인한다.
2. GT가 연속 missed이면 `miss_count`가 step마다 증가하는지 확인한다.
3. `miss_count < miss_threshold`이면 target label에 반영되지 않는지 확인한다.
4. target GT가 없으면 image label이 `0`인지 확인한다.
5. target GT가 하나 이상 있으면 image label이 `1`인지 확인한다.

## 설계 요약

MissBank는 ReMiss의 failure memory 역할을 담당한다. 초기 버전에서는 단순하고 해석 가능한 형태를 우선한다.

- GT 단위 record 저장
- 현재 missed 여부와 연속 미검출 횟수 관리
- binary image-level hard label 생성
- `miss_threshold` 이상인 반복 실패만 target화
- `start_epoch` 전에도 memory는 갱신하되 auxiliary 학습은 지연

이 구조는 초기 hard injection 실험에 충분하며, 이후 soft label, soft injection, gated additive, 고해상도 grid로 자연스럽게 확장할 수 있다.
