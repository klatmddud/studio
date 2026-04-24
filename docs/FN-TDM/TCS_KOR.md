# TCS - Transition Candidate Selection

## 목표

TCS는 현재 training GT instance 중 어떤 대상에 TAL supervision을 적용할지 선택한다.

HTM은 과거의 `FN -> TP` transition을 발견한다. TDB는 그 direction을 저장한다. TCS는 현재
mini-batch의 GT 중 어떤 것이 충분히 hard하고, 사용할 수 있는 memory direction이 있는지 판단한다.

```text
current GT g
  -> current difficulty 확인
  -> TDB direction availability 확인
  -> TAL 대상으로 선택 또는 skip
```

TCS는 FN-TDM이 모든 object에 transition alignment를 무차별적으로 적용하는 것을 막는다.

## FN-TDM 내 위치

```text
HTM: 과거 FN -> TP transition mining
TDB: transition direction 저장 및 집계
TCS: 현재 hard/FN-like GT candidate 선택
TAL: 선택된 candidate를 TDB direction으로 정렬
```

TCS는 lightweight training-only 모듈이어야 한다. Inference behavior를 바꾸면 안 된다.

## TCS가 필요한 이유

모든 GT에 TAL을 적용하면 해로울 수 있다.

잠재적 문제:

- Easy GT가 불필요한 feature-direction constraint를 받을 수 있다.
- Class-level transition direction이 class 내 모든 instance에 맞지는 않을 수 있다.
- TAL을 너무 넓게 적용하면 base detection loss와 충돌할 수 있다.
- 모든 GT가 TDB direction을 사용하면 memory noise가 증폭될 수 있다.

TCS는 TAL 적용 대상을 이득이 있을 가능성이 높은 GT로 좁힌다.

```text
hard current GT + available transition direction -> TAL candidate
```

## V0 Design

TCS-V0는 두 gate를 사용한다.

```text
1. Memory coverage gate:
   TDB가 class_id(g)에 대한 valid direction prototype을 가지고 있다.

2. Current hardness gate:
   GT g가 training-time detection signal 기준으로 현재 hard-like이다.
```

Selection rule:

```text
select g if:
    TDB.get_prototype(class_id(g)) is not None
    and hardness(g) >= tau_hard
```

권장 기본값:

```text
tau_hard: 0.5
```

V0에서는 classification confidence로 hardness를 정의한다.

```text
hardness(g) = 1 - confidence(g)
```

따라서:

```text
select if confidence(g) <= 1 - tau_hard
```

`tau_hard = 0.5`이면 confidence <= 0.5인 GT를 선택한다.

## Candidate Object

TCS는 TAL이 사용할 candidate object를 출력한다.

```python
TCSCandidate = {
    "gt_uid": str | None,
    "image_id": Any,
    "ann_id": Any,
    "batch_index": int,
    "target_index": int,
    "class_id": int,
    "bbox": Tensor[4],

    "hardness": float,
    "confidence": float,
    "selection_reason": str,

    "direction": Tensor[D],
    "direction_source": "class_prototype" | "failure_type_prototype" | "sampled_entry",
    "fn_type_hint": str | None,
}
```

`direction`은 detach되어야 하며 TAL이 사용할 device로 이동되어야 한다.

## Current Difficulty Signals

TCS는 detector에 따라 다양한 difficulty signal을 사용할 수 있다.

Detector-agnostic signals:

```text
classification confidence for class_id(g)
objectness or foreground confidence
localization quality / IoU estimate
assigned positive count
current per-GT classification loss
current per-GT box loss
historical GT state if available
```

V0에서는 normal training forward에서 이미 사용할 수 있는 signal을 우선한다. TCS 내부에서 추가
inference pass를 수행하지 않는다.

## FCOS V0 Hardness

FCOS에서는 각 GT가 하나 이상의 positive feature location에 assigned된다.

권장 confidence:

```text
confidence(g) =
    max over positive locations assigned to g:
        sigmoid(cls_logit[class_id(g)]) * sigmoid(centerness_logit)
```

Selection 시점에 centerness를 사용할 수 없다면:

```text
confidence(g) =
    max sigmoid(cls_logit[class_id(g)]) over positive locations assigned to g
```

Hardness:

```text
hardness(g) = 1 - confidence(g)
```

Assigned positive location이 없으면:

```text
confidence(g) = 0
hardness(g) = 1
selection_reason = "missing_assignment"
```

Missing assignment candidate는 risk가 높다. TAL이 reliable GT ROI feature를 추출할 수 있는지에 따라
V0에서 포함하거나 로그만 남길 수 있다.

권장 기본값:

```text
include_missing_assignment: true
```

TAL은 assigned FCOS positive가 없어도 GT ROIAlign feature를 사용할 수 있기 때문이다.

## Optional Historical Gate

TCS는 optional하게 HTM/TFM history를 사용할 수 있다.

```text
select if:
    current hard-like
    or gt_uid has previous FN history
    or gt_uid has relapse history
```

V0는 history를 요구하지 않는 것이 좋다. TDB 자체가 이미 historical하고 class-wise이기 때문이다.
History-aware selection은 ablation으로 도입할 수 있다.

Suggested variants:

```text
TCS-CurrentHard:
    current confidence만으로 선택

TCS-History:
    previous FN/relapse history가 있는 GT 선택

TCS-Hybrid:
    current hard 또는 previous FN/relapse history이면 선택
```

권장 V0 기본값:

```text
selection_mode: current_hard
```

## Memory Coverage Gate

TDB가 valid direction을 제공하지 못하면 TCS는 GT를 선택하면 안 된다.

```python
direction = tdb.get_prototype(class_id=class_id, fn_type=fn_type_hint, device=device)
if direction is None:
    skip
```

V0:

```text
use_failure_type_query: false
```

TCS는 class-wise prototype을 query한다.

```text
D_c = TDB.get_prototype(class_id=c)
```

Failure-type query는 HTM subtype quality를 검증한 뒤 켤 수 있다.

## Candidate Budget

TAL이 training을 지배하지 않도록 TCS는 batch당 candidate 수를 제한해야 한다.

권장 기본값:

```text
max_candidates_per_image: 8
max_candidates_per_batch: 32
min_candidates_per_batch: 0
```

사용 가능한 candidate가 더 많으면 다음 기준으로 ranking한다.

```text
hardness * direction_quality
```

V0에서 TDB가 quality 없이 prototype만 반환한다면 hardness로 ranking한다.

Tie-breaker:

```text
higher hardness first
then smaller area first
then stable gt_uid order
```

Small object는 자주 missed되므로 이 tie-breaker가 도움이 될 수 있다. 다만 V0에서는 size를 main
selection criterion으로 만들지 않는다.

## Selection Score

Internal selection score:

```text
selection_score(g) = hardness(g) * memory_score(class_id(g))
```

V0 memory score:

```text
memory_score(c) = 1 if TDB has valid prototype for c else 0
```

Optional V1:

```text
memory_score(c) = mean quality or max quality of TDB[c]
```

## Candidate Weight

TCS는 TAL에 per-candidate weight를 제공할 수 있다.

V0:

```text
candidate_weight = clamp(hardness, min_weight, max_weight)
```

권장 기본값:

```text
min_weight: 0.25
max_weight: 1.0
```

TAL은 각 candidate loss에 이 weight를 곱할 수 있다.

TAL V0가 복잡해진다면 모든 selected candidate weight를 1.0으로 두고 hardness는 selection/ranking에만
사용한다.

## Failure-Type Hint

TCS는 failure-type TDB query를 위한 `fn_type_hint`를 만들 수 있다.

Possible hints:

```text
low class confidence -> FN_BG or FN_CLS
good class confidence but poor localization -> FN_LOC
no assignment -> FN_MISS
```

V0에서는 metadata로만 유지한다.

```text
fn_type_hint: None
```

첫 버전은 class-wise TDB prototype을 query한다.

## TDB Retrieval Variant와의 상호작용

TCS는 TDB retrieval variant와 독립적이어야 한다.

```text
TDB-Last
TDB-TopK
TDB-TopK+Age
```

TCS는 direction만 요청한다.

```text
direction = TDB.get_prototype(...)
```

그 direction을 어떻게 만들지는 TDB config가 결정한다. 이렇게 해야 candidate selection ablation과 memory
retrieval ablation이 분리된다.

## Configuration Sketch

향후 `modules/cfg/fntdm.yaml`에 config를 추가한다.

```yaml
tcs:
  enabled: true
  selection_mode: current_hard

  memory_gate:
    require_tdb_direction: true
    use_failure_type_query: false

  hardness:
    source: assigned_cls_confidence
    tau_hard: 0.5
    include_centerness: true
    include_missing_assignment: true

  budget:
    max_candidates_per_image: 8
    max_candidates_per_batch: 32
    rank_by: hardness

  weighting:
    use_candidate_weight: false
    min_weight: 0.25
    max_weight: 1.0

  logging:
    save_summary: true
    save_candidate_stats: true
```

## Logging

Epoch 또는 iteration summary:

```text
num_gt_seen
num_memory_covered_gt
num_hard_gt
num_selected_candidates
num_skipped_no_memory
num_skipped_easy
mean_candidate_hardness
max_candidate_hardness
```

Optional per-class summary:

```text
class_id
num_gt_seen
num_memory_covered_gt
num_selected
mean_hardness
```

디버깅이 아니라면 매 iteration 모든 candidate를 기록하지 않는다. Candidate-level log는 매우 커질 수 있다.

## Edge Cases

- 어떤 class에 TDB entry가 없음: 해당 GT를 skip한다.
- Batch에 selected candidate가 없음: TAL은 zero loss를 반환한다.
- 모든 GT가 easy함: TAL은 zero loss를 반환한다.
- Candidate budget 초과: 가장 높은 ranking candidate만 유지한다.
- Missing assignment지만 valid GT ROI feature가 있음: config가 허용하면 candidate로 선택할 수 있다.
- Direction query가 invalid/None을 반환함: GT를 skip한다.

## V0 Implementation Checklist

1. `TransitionCandidateSelector` class를 추가한다.
2. `TCSCandidate` structure를 정의한다.
3. FCOS assigned positive location에서 per-GT confidence extraction을 구현한다.
4. `TDB.get_prototype`을 사용하는 memory coverage gate를 구현한다.
5. Hardness thresholding을 구현한다.
6. Per-image 및 per-batch budget을 구현한다.
7. Direction tensor가 포함된 selected candidate를 반환한다.
8. Summary counter를 추가한다.
9. TCS output을 TAL에 연결한다.

## Minimal Unit Tests

Selection:

```text
easy GT is skipped
hard GT with TDB direction is selected
hard GT without TDB direction is skipped
missing assignment follows include_missing_assignment config
```

Budget:

```text
max_candidates_per_image is enforced
max_candidates_per_batch is enforced
highest hardness candidates are kept
```

Device:

```text
direction is moved to requested device
direction is detached
```

No-op:

```text
empty batch returns empty candidates
no selected candidates makes TAL skip safely
```

## Research Claim

TCS는 다음처럼 설명할 수 있다.

```text
Transition Candidate Selection restricts direction alignment to current hard
ground-truth instances whose classes have reliable transition memories, preventing
the auxiliary objective from over-regularizing easy objects.
```

TCS는 FN-TDM을 targeted하게 유지한다. 이 방법은 historical recovery direction을 현재 detector가 여전히
어려움을 보이는 곳에만 사용한다.
