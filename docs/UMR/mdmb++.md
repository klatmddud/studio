# MDMB++ — Structured Failure Memory for UMR

## 1. Overview

MDMB++는 기존 MDMB를 `missed GT bank`에서 `structured failure memory`로 확장한 버전이다. 기존 MDMB가 "어떤 GT가 miss되었는가"를 기록했다면, MDMB++는 아래 질문에 답할 수 있어야 한다.

1. 왜 miss되었는가
2. 얼마나 오래 miss되고 있는가
3. 과거에 성공적으로 탐지된 적이 있는가
4. 이번 step에서 어떤 candidate가 GT 근처까지 왔는가
5. replay와 densification에서 우선순위를 얼마나 높게 줄 것인가

UMR 전체에서 MDMB++는 상태 저장소이자 의사결정 엔진의 입력이다. `Hard Replay`와 `Candidate Densification`은 직접 raw detector output을 읽지 않고, MDMB++가 정리한 canonical information을 사용한다.

## 2. Why MDMB is not enough

현재 MDMB는 다음 정도만 저장한다.

- `image_id`
- `class_id`
- normalized `bbox`
- `miss_type`
- `consecutive_miss_count`
- `max_consecutive_miss_count`
- `last_detected_epoch`

이 정보만으로는 아래를 알 수 없다.

- GT 근처에 proposal/query/point가 아예 없었는지
- 있었는데 점수 문제로 탈락했는지
- NMS나 top-k selection에서 사라졌는지
- 마지막 성공 시점의 representation이 무엇이었는지
- 다음 epoch replay priority를 어떻게 정할지

따라서 MDMB++는 miss memory를 `failure-aware control memory`로 승격해야 한다.

## 3. Core Design

MDMB++는 내부 상태를 네 층으로 나눈다.

1. `current miss bank`: 현재 epoch에서 unresolved인 miss GT 목록
2. `persistent GT record`: 모든 GT의 시간축 이력
3. `candidate snapshot`: 현재 step에서 GT 주변에 존재했던 candidate 요약
4. `support snapshot`: 마지막 성공 탐지 시점의 support 정보

이 구조를 통해 MDMB++는 현재 문제와 과거 맥락을 동시에 제공한다.

## 4. Data Structures

아래 dataclass 구조를 권장한다.

### 4.1 `FailureType`

```python
FailureType = Literal[
    "candidate_missing",
    "loc_near_miss",
    "cls_confusion",
    "score_suppression",
    "nms_suppression",
    "detected",
]
```

각 의미는 다음과 같다.

- `candidate_missing`: GT 근처에 candidate 자체가 거의 없음
- `loc_near_miss`: candidate는 있으나 IoU가 부족함
- `cls_confusion`: 위치는 맞지만 class가 틀림
- `score_suppression`: 정답 candidate가 있었으나 score/ranking 때문에 탈락
- `nms_suppression`: 정답 후보가 있었지만 suppression 단계에서 제거
- `detected`: 최종적으로 맞게 탐지됨

### 4.2 `CanonicalCandidate`

```python
@dataclass(slots=True)
class CanonicalCandidate:
    stage: str
    box: Tensor
    score: float
    label: int
    iou_to_gt: float
    survived_selection: bool
    survived_nms: bool | None
    rank: int | None
    level_or_stage_id: str | int | None
```

설명:

- `stage`: `fcos_pre_nms`, `rpn`, `roi_head`, `dino_decoder_l3` 같은 raw source 식별자
- `box`: normalized `xyxy`
- `score`: detector stage의 confidence 또는 objectness
- `label`: 해당 stage에서의 predicted class
- `iou_to_gt`: 대응 GT와의 IoU
- `survived_selection`: top-k / query selection 이후 살아남았는지
- `survived_nms`: NMS가 있는 detector에서 NMS 이후 살아남았는지
- `rank`: score 기준 순위

UMR의 상위 로직은 detector raw structure를 직접 보지 않고 이 객체만 읽는다.

### 4.3 `SupportSnapshot`

```python
@dataclass(slots=True)
class SupportSnapshot:
    epoch: int
    box: Tensor
    score: float
    feature: Tensor | None
    feature_level: str | int | None
```

설명:

- 마지막 성공 탐지 시점의 box / score / feature를 저장한다.
- `feature`는 replay 설명 자료 또는 auxiliary consistency loss에 사용 가능하다.
- 메모리 사용량이 부담되면 `feature=None`으로 두고 box/score만 저장할 수 있다.

### 4.4 `GTFailureRecord`

```python
@dataclass(slots=True)
class GTFailureRecord:
    gt_uid: str
    image_id: str
    class_id: int
    bbox: Tensor
    first_seen_epoch: int
    last_seen_epoch: int
    last_state: FailureType
    consecutive_miss_count: int
    max_consecutive_miss_count: int
    total_miss_count: int
    relapse_count: int
    last_detected_epoch: int | None
    last_failure_epoch: int | None
    last_failure_type: FailureType | None
    severity: float
    support: SupportSnapshot | None
```

설명:

- `gt_uid`는 epoch 간 동일 GT를 안정적으로 가리키는 키다.
- 가능하면 COCO annotation id를 쓰고, 없으면 `(image_id, class_id, bbox hash)`를 사용한다.
- 기존 `_GTRecord`보다 훨씬 많은 통계를 보존하므로 replay와 densification priority를 직접 계산할 수 있다.

### 4.5 `MDMBPlusEntry`

```python
@dataclass(slots=True)
class MDMBPlusEntry:
    gt_uid: str
    image_id: str
    class_id: int
    bbox: Tensor
    failure_type: FailureType
    consecutive_miss_count: int
    max_consecutive_miss_count: int
    total_miss_count: int
    relapse: bool
    severity: float
    best_candidate: CanonicalCandidate | None
    topk_candidates: list[CanonicalCandidate]
    support: SupportSnapshot | None
```

설명:

- 이 entry는 현재 unresolved miss에 한해 bank에 들어간다.
- `topk_candidates`는 detector-specific raw candidate를 정규화한 결과다.
- `best_candidate`는 replay / debugging / analysis에 자주 쓰이므로 별도 저장을 권장한다.

## 5. Failure Taxonomy Rule

Failure type은 GT $g$에 대한 candidate 집합 $C_t(g)$를 바탕으로 판정한다.

정의:

$$
u_t(g) = \max_{c \in C_t(g)} \mathrm{IoU}(c, b_g)
$$

$$
v_t(g) = \max_{c \in C_t(g),\; y_c = y_g} \mathrm{IoU}(c, b_g)
$$

여기서 $b_g$는 GT box, $y_g$는 GT class다.

판정 예시는 다음과 같다.

1. 최종 detection에 정답 class와 충분한 IoU가 있으면 `detected`
2. $u_t(g) < \theta_{\mathrm{near}}$ 이면 `candidate_missing`
3. $u_t(g) \ge \theta_{\mathrm{near}}$ 이고 $v_t(g) < \theta_{\mathrm{cls}}$ 이면 `cls_confusion`
4. $v_t(g) \ge \theta_{\mathrm{cls}}$ 인 candidate가 있으나 `survived_selection=False` 이면 `score_suppression`
5. NMS 이전에는 살아 있었으나 `survived_nms=False` 이면 `nms_suppression`
6. 그 외 위치만 약간 부족한 경우 `loc_near_miss`

보다 보수적인 구현을 원하면 `candidate_missing / cls_confusion / detected`의 3-way 분류부터 시작해도 된다.

## 6. Severity Score

replay와 densification은 동일한 severity를 공유하는 편이 좋다. 예시 정의는 다음과 같다.

$$
\phi_t(g)
=
\lambda_1 \cdot \frac{s_t(g)}{\max(s_{\mathrm{global}}, 1)}
\;+\;
\lambda_2 \cdot \mathbb{1}[\text{relapse}(g, t)]
\;+\;
\lambda_3 \cdot \rho(\text{failure\_type}(g, t))
\;+\;
\lambda_4 \cdot (1 - u_t(g))
$$

여기서 $\rho(\cdot)$는 failure type 별 difficulty prior다. 예를 들어:

- `candidate_missing`: 1.0
- `score_suppression`: 0.8
- `nms_suppression`: 0.7
- `cls_confusion`: 0.6
- `loc_near_miss`: 0.5

이 score는 persistent record와 current bank entry 모두에 저장하는 것이 좋다.

## 7. Update Algorithm

MDMB++ 업데이트는 `optimizer.step()` 이후에 수행하는 것이 가장 안전하다. 그 이유는 실제 학습 후의 current model state가 어떤 candidate를 만들었는지 반영해야 하기 때문이다.

### 7.1 Inputs

업데이트 입력은 다음과 같다.

- `targets`: GT box, label, image_id
- `final_detections`: post-selection 또는 final output
- `candidate_summary`: detector adapter가 제공한 canonical candidate 목록
- `epoch`: 현재 epoch

### 7.2 Steps

1. 현재 batch의 GT별 unique id를 생성한다.
2. GT별로 candidate summary를 모은다.
3. failure type을 판정한다.
4. persistent record를 갱신한다.
5. miss 상태라면 current bank에 `MDMBPlusEntry`를 저장한다.
6. detected 상태라면 support snapshot을 갱신한다.
7. global max miss streak와 class-level summary를 갱신한다.

### 7.3 Pseudocode

```python
for gt in gt_instances:
    gt_uid = make_gt_uid(gt)
    record = persistent_records.get(gt_uid)
    cands = candidate_summary.match(gt_uid)

    failure_type = classify_failure(gt, final_detections, cands)
    coverage = max_iou(cands, gt.box)
    severity = compute_severity(record, failure_type, coverage)

    if failure_type == "detected":
        record.consecutive_miss_count = 0
        record.last_detected_epoch = epoch
        record.support = build_support_snapshot(gt_uid, cands, epoch)
        bank.remove(gt_uid)
    else:
        record.consecutive_miss_count += 1
        record.total_miss_count += 1
        record.last_failure_type = failure_type
        record.last_failure_epoch = epoch
        bank[gt_uid] = build_bank_entry(record, cands, severity)
```

## 8. Candidate Collection by Detector

MDMB++는 detector-specific adapter를 반드시 필요로 한다.

### 8.1 FCOS

수집 대상:

- FPN level별 point logits
- centerness
- decode된 pre-NMS boxes
- top-k 후보
- final detections

권장 canonical mapping:

- `stage="fcos_pre_nms"` 또는 `stage=f"fcos_p{level}"`
- `level_or_stage_id`에 pyramid level 저장

### 8.2 Faster R-CNN

수집 대상:

- RPN pre-NMS proposals
- RPN post-NMS proposals
- ROI head class logits / scores
- final detections

권장 canonical mapping:

- `stage="rpn_pre_nms"`, `stage="rpn_post_nms"`, `stage="roi_head"`

### 8.3 DINO

수집 대상:

- decoder layer별 queries
- reference points
- box predictions
- class logits
- final selected predictions

권장 canonical mapping:

- `stage="dino_decoder_l{layer}"`
- `level_or_stage_id`에 decoder layer index 저장

## 9. Serialization

체크포인트에 아래 항목을 저장해야 한다.

- `version`
- `config`
- `current_epoch`
- `bank`
- `persistent_records`
- `global_max_consecutive_miss`
- `class_statistics`

대용량이 될 수 있는 항목은 옵션화하는 것이 좋다.

- `topk_candidates`
- `support.feature`

추천 정책:

- 기본: 최근 $K$개 candidate만 저장
- `feature`는 float16 CPU tensor로 압축 저장
- debug off 시 `topk_candidates` 길이를 3 또는 5로 제한

## 10. Public API

구현자가 쉽게 재사용하려면 아래 메서드를 제공하는 것이 좋다.

```python
class MDMBPlus(nn.Module):
    def update(...)
    def get_image_entries(image_id) -> list[MDMBPlusEntry]
    def get_entry(gt_uid) -> MDMBPlusEntry | None
    def get_record(gt_uid) -> GTFailureRecord | None
    def get_replay_priority(image_id) -> float
    def get_dense_targets(image_id) -> list[MDMBPlusEntry]
    def summary() -> dict[str, Any]
```

특히 `get_replay_priority()`와 `get_dense_targets()`는 하위 모듈이 internal field를 직접 읽지 않게 만드는 데 중요하다.

## 11. Summary Metrics

실험 로그에는 아래를 추천한다.

- `num_entries`
- `num_images`
- `num_relapse`
- `num_candidate_missing`
- `num_cls_confusion`
- `num_score_suppression`
- `global_max_consecutive_miss`
- `mean_severity`
- `recovery_rate_last_1_epoch`

이 지표는 단순 mAP 외에 UMR이 실제로 miss를 복원하고 있는지를 설명하는 데 중요하다.

## 12. Recommended Incremental Build

구현은 아래 순서를 권장한다.

1. 기존 MDMB에 `gt_uid`, `severity`, `failure_type`, `best_candidate` 추가
2. persistent record를 `GTFailureRecord`로 확장
3. detector별 canonical candidate collector 추가
4. support snapshot 추가
5. `topk_candidates`와 serialization 최적화 추가

즉 첫 버전은 작은 schema 확장에서 시작하고, 이후 memory richness를 점진적으로 늘리는 편이 안전하다.
