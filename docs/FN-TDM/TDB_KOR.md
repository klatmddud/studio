# TDB - Transition Direction Bank

## 목표

TDB는 FN-TDM의 memory 컴포넌트다.

HTM은 `FN -> TP` transition event를 생성한다. TDB는 이러한 transition direction을 저장, 필터링,
집계해서 TAL이 이후 training epoch에서 stable direction prior로 사용할 수 있게 한다.

```text
HTM event:  (class_id, fn_type, direction, quality, metadata)
TDB memory: class-wise top-K transition directions
TAL query:  current hard GT에 대한 representative direction
```

V0의 TDB는 의도적으로 단순하게 둔다. class-wise quality-gated top-K memory를 사용하고, class별
weighted prototype direction을 반환한다.

## FN-TDM 내 위치

```text
HTM: Hard Transition Mining
  - FN -> TP transition을 찾는다
  - transition event를 생성한다

TDB: Transition Direction Bank
  - transition direction을 저장한다
  - low-quality 또는 duplicate event를 필터링한다
  - class/failure-type direction prototype을 만든다

TAL: Transition Alignment Loss
  - TDB에서 direction을 조회한다
  - 현재 hard/FN-like GT embedding을 정렬한다
```

TDB는 inference를 수행하지 않고 loss도 계산하지 않는다. TDB는 transition memory만 관리한다.

## V0 Design

Class-wise top-K direction memory를 사용한다.

```text
TDB[c] = class c에 대한 top-K entries
```

각 entry는 HTM이 만든 normalized transition direction을 저장한다.

```text
d = normalize(z_tp - z_fn)
```

TAL은 현재 GT class로 TDB를 query한다.

```text
D_c = TDB.get_prototype(class_id=c)
```

반환되는 direction은 저장된 direction의 quality-weighted mean이다.

```text
D_c = normalize(sum_i w_i * d_i)
w_i = softmax(q_i / tau_proto)
```

## Responsibilities

TDB가 담당하는 역할:

- HTM의 `TransitionEvent`를 받는다.
- 저장 전에 transition entry를 검증한다.
- per-class memory 크기를 제한한다.
- quality 기준으로 entry를 ranking한다.
- 필요하면 failure subtype별로 entry를 group한다.
- TAL에 direction prototype 또는 sampled direction을 반환한다.
- 분석과 checkpointing을 위한 compact metadata를 저장한다.

TDB가 담당하지 않는 역할:

- TP/FN state 부여.
- feature 추출.
- training loss 적용.
- inference 시 detector prediction 변경.

## Input Event

TDB는 HTM이 생성한 `TransitionEvent`를 입력으로 받는다.

```python
TransitionEvent = {
    "gt_uid": str,
    "image_id": Any,
    "ann_id": Any,
    "class_id": int,
    "bbox": Tensor[4],

    "epoch_fn": int,
    "epoch_tp": int,
    "fn_type": str,

    "z_fn": Tensor[D],
    "z_tp": Tensor[D],
    "direction": Tensor[D],

    "score_fn": float,
    "score_tp": float,
    "iou_fn": float,
    "iou_tp": float,

    "quality": float,
}
```

V0에서는 기본적으로 `direction`과 compact metadata만 저장한다. `z_fn`과 `z_tp` 저장은 optional이며,
이후 TAL variant가 필요로 할 때만 켠다.

## Bank Entry

Internal entry schema:

```python
TDBEntry = {
    "entry_id": str,
    "gt_uid": str,
    "image_id": Any,
    "ann_id": Any,
    "class_id": int,
    "bbox": Tensor[4] | None,
    "fn_type": str,

    "epoch_fn": int,
    "epoch_tp": int,
    "age": int,

    "direction": Tensor[D],
    "quality": float,

    "score_fn": float,
    "score_tp": float,
    "iou_fn": float,
    "iou_tp": float,

    "z_fn": Tensor[D] | None,
    "z_tp": Tensor[D] | None,
}
```

`entry_id`는 deterministic하게 만들 수 있다.

```text
entry_id = hash(gt_uid, epoch_fn, epoch_tp, fn_type)
```

TDB에 저장되는 모든 tensor는 detach되어야 한다. Persistent tensor는 기본적으로 CPU에 둔다. TAL이
training step에서 사용할 때만 training device로 옮긴다.

## Memory Layout

V0 memory:

```python
bank = {
    class_id: List[TDBEntry]
}
```

Optional V1 layout:

```python
bank = {
    class_id: {
        fn_type: List[TDBEntry]
    }
}
```

V0에서는 main lookup이 class-wise이더라도 각 entry에 `fn_type`을 유지한다. 이렇게 하면 저장 형식을
바꾸지 않고 나중에 failure-type ablation을 할 수 있다.

## Entry Validation

저장 전에 TDB는 invalid event를 거부해야 한다.

필수 검사:

```text
class_id is valid
fn_type is allowed
direction is finite
direction norm is close to 1
quality is finite and positive
epoch_tp > epoch_fn
```

권장 기본값:

```text
allowed_fn_types: [FN_BG, FN_CLS, FN_MISS]
min_quality: 0.0
min_direction_norm: 1.0e-6
renormalize_direction: true
```

`renormalize_direction`이 true이면 TDB는 저장 전에 direction을 다시 normalize한다.

## Retention Policy

Bank는 bounded해야 한다.

권장 V0:

```text
max_entries_per_class: 128
replacement_policy: quality_topk
max_entries_per_gt: 1
```

새 entry가 들어오면:

1. validation에 실패하면 버린다.
2. 같은 `entry_id`가 이미 있으면 버린다.
3. 해당 GT가 `max_entries_per_gt`에 도달했다면 더 높은 quality entry만 유지한다.
4. `bank[class_id]`에 삽입한다.
5. `quality` 내림차순으로 정렬한다.
6. 앞의 `max_entries_per_class`개만 유지한다.

이 정책은 TDB를 full transition log가 아니라 high-quality direction memory로 만든다.

## Duplicate Handling

DDP 또는 반복 HTM pass에서 duplicate가 발생할 수 있다.

Duplicate keys:

```text
strict duplicate: entry_id
GT duplicate: (gt_uid, fn_type)
```

V0 behavior:

```text
same entry_id:
    keep higher-quality one

same gt_uid and max_entries_per_gt == 1:
    keep higher-quality one
```

V0에서는 duplicate entry를 평균내지 않는다. 평균은 불안정한 transition을 숨길 수 있다.

## Aging

Representation space가 변하기 때문에 transition direction은 stale해질 수 있다.

V0는 age metadata를 저장하되 처음부터 aggressive aging을 사용할 필요는 없다.

```text
age = current_epoch - epoch_tp
```

Prototype 계산 시 optional age decay:

```text
q_eff = quality * exp(-age / tau_age)
```

권장 기본값:

```text
use_age_decay: false
tau_age: 10
```

오래된 direction이 TAL 안정성을 떨어뜨릴 때만 age decay를 켠다.

## Prototype Direction

TAL은 보통 현재 GT에 대한 representative direction 하나가 필요하다.

Class-wise prototype:

```text
D_c = normalize(sum_i w_i * d_i)
w_i = softmax(q_i / tau_proto)
```

권장 기본값:

```text
tau_proto: 0.2
min_entries_for_query: 1
```

`tau_proto`가 작으면 high-quality direction이 지배적이다. 값이 크면 prototype이 uniform average에
가까워진다.

weighted sum norm이 너무 작으면 `None`을 반환한다.

```text
min_prototype_norm: 1.0e-6
```

## Retrieval Variants

TDB는 ablation을 위해 세 가지 retrieval variant를 지원하는 것이 좋다.

### TDB-Last

Query된 class에 대해 가장 최근 valid transition direction 하나만 사용한다.

```text
D_c = d_last
```

Selection:

```text
d_last = TDB[c] 안에서 epoch_tp가 가장 큰 entry
```

Rationale:

- Representation drift가 큰 경우 현재 feature space와 가장 잘 맞을 수 있다.
- 매우 단순하며 recency baseline으로 유용하다.

Expected risk:

- 단일 transition이 threshold crossing, NMS, 불안정한 GT 때문에 생긴 것일 수 있어 noisy하다.
- Seed variance가 커질 수 있다.
- TAL이 한 hard instance의 trajectory에 과적합될 수 있다.

### TDB-TopK

Query된 class에 대해 quality top-K entry를 사용하고, quality-weighted prototype을 반환한다.

```text
D_c = normalize(sum_i softmax(q_i / tau_proto) * d_i)
```

Selection:

```text
entries = TDB[c]에서 quality 기준 top-K
```

Rationale:

- 단일 transition noise를 줄인다.
- 여러 hard instance에서 공통적으로 나타나는 recovery direction을 포착한다.
- HTM quality를 사용해 confidence가 높고 gap이 짧은 `FN -> TP` transition을 선호한다.

Expected risk:

- 오래된 high-quality direction은 feature space drift 때문에 stale해질 수 있다.
- 한 class 안에 여러 hard mode가 있으면 서로 다른 direction이 부분적으로 상쇄될 수 있다.

V0의 권장 기본값이다.

### TDB-TopK+Age

Quality top-K entry를 사용하되, prototype 계산 전에 각 entry의 effective quality를 age로 decay한다.

```text
age_i = current_epoch - epoch_tp_i
q_eff_i = q_i * exp(-age_i / tau_age)
D_c = normalize(sum_i softmax(q_eff_i / tau_proto) * d_i)
```

Rationale:

- Top-K aggregation의 안정화 효과를 유지한다.
- 오래된 direction의 영향력을 줄인다.
- Quality와 recency의 균형을 잡는다.

Expected risk:

- `tau_age` tuning이 필요하다.
- `tau_age`가 너무 작으면 TDB-Last에 가까워진다.
- `tau_age`가 너무 크면 TDB-TopK와 거의 같아진다.

권장 ablation 순서:

```text
TDB-Last
TDB-TopK
TDB-TopK+Age
```

기본 선택:

```text
retrieval: topk
```

## Failure-Type Prototype

V1에서는 failure-type conditioned prototype을 반환할 수 있다.

```text
D_{c,t} = TDB.get_prototype(class_id=c, fn_type=t)
```

Fallback policy:

```text
if enough entries for (class_id, fn_type):
    return D_{c,t}
else:
    return D_c
```

권장 기본값:

```text
use_failure_type_query: false
min_type_entries_for_query: 4
```

이는 HTM subtype label이 안정적임을 확인한 뒤 사용하면 좋다.

## Sampling API

TDB는 prototype API와 sampling API를 모두 제공해야 한다.

Minimal API:

```python
class TransitionDirectionBank:
    def update(self, events: list[TransitionEvent], epoch: int) -> dict:
        ...

    def get_prototype(
        self,
        class_id: int,
        fn_type: str | None = None,
        device: torch.device | None = None,
    ) -> Tensor | None:
        ...

    def sample(
        self,
        class_id: int,
        k: int = 1,
        fn_type: str | None = None,
        device: torch.device | None = None,
    ) -> list[TDBEntry]:
        ...

    def summary(self) -> dict:
        ...

    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state: dict) -> None:
        ...
```

V0 TAL은 `get_prototype`을 사용한다. `sample`은 이후 contrastive 또는 multi-direction TAL variant를
위해 남겨둔다.

## Query Behavior

TAL이 memory가 없는 class를 query하면:

```text
return None
```

TAL은 해당 GT에 대한 auxiliary loss를 skip해야 한다.

Class memory는 있지만 prototype이 invalid이면:

```text
return None
```

V0에서는 다른 class로 fallback하지 않는다. 별도 semantic sharing module이 없다면 cross-class
direction은 noisy할 가능성이 높다.

## Device and Precision

Storage:

```text
CPU tensors by default
float32 by default
optional float16 for large datasets
```

Query:

```text
move returned direction to caller-provided device
detach returned direction
```

TDB direction은 prior이지 learnable tensor가 아니다. Stored entry로 gradient가 흘러가면 안 된다.

## Checkpointing

FN-TDM이 enabled이면 TDB state를 training checkpoint와 함께 저장해야 한다.

State에 포함할 내용:

```text
config
current_epoch
bank entries
prototype cache if used
summary counters
```

권장 format:

```python
{
    "version": 1,
    "config": ...,
    "bank": ...,
    "stats": ...
}
```

Checkpoint size가 커지면 다음만 저장한다.

```text
direction
quality
class_id
fn_type
gt_uid
epoch_fn
epoch_tp
score/iou metadata
```

명시적으로 설정하지 않는 한 `z_fn`과 `z_tp`는 저장하지 않는다.

## Prototype Cache

V0에서 prototype 계산은 저렴하지만, cache를 두면 반복 sorting과 weighted sum을 피할 수 있다.

Cache key:

```text
(class_id, fn_type or "ALL", bank_revision)
```

Entry가 삽입되거나 삭제되면 cache는 반드시 invalidate되어야 한다.

V0에서는 cache가 optional이다.

## Update Timing

TDB는 HTM의 epoch-end mining이 끝난 뒤 update된다.

```text
end of epoch e:
    events = HTM.mine(...)
    update_stats = TDB.update(events, epoch=e)
    save HTM/TDB summaries

epoch e + 1:
    TAL queries updated TDB
```

이 one-epoch delay는 의도된 것이다. Direction은 epoch `e` 이후의 모델에서 mining되고, 이후 training의
historical prior로 사용된다.

## DDP Behavior

권장 V0:

```text
rank 0 runs HTM
rank 0 updates TDB
rank 0 broadcasts TDB state to all ranks before next epoch
```

이 방식은 duplicate entry와 rank별 memory 불일치를 피한다.

모든 rank가 독립적으로 event를 mining한다면 `entry_id`로 merge하고 duplicate는 higher-quality entry를
유지한다.

## Configuration Sketch

향후 `modules/cfg/fntdm.yaml`에 config를 추가한다.

```yaml
tdb:
  enabled: true

  storage:
    max_entries_per_class: 128
    max_entries_per_gt: 1
    store_z_fn: false
    store_z_tp: false
    store_on_cpu: true
    dtype: float32

  filtering:
    allowed_fn_types: [FN_BG, FN_CLS, FN_MISS]
    min_quality: 0.0
    min_direction_norm: 1.0e-6
    renormalize_direction: true

  replacement:
    policy: quality_topk
    duplicate_policy: keep_best

  prototype:
    retrieval: topk
    tau_proto: 0.2
    min_entries_for_query: 1
    min_prototype_norm: 1.0e-6
    use_age_decay: false
    tau_age: 10
    use_failure_type_query: false
    min_type_entries_for_query: 4

  logging:
    save_summary: true
    save_bank_metadata: true
```

## Logging

Epoch summary:

```text
epoch
num_events_received
num_events_stored
num_events_rejected
num_duplicate_events
num_replaced_entries
num_classes_with_entries
total_entries
entries_per_class_mean
entries_per_class_max
```

Per-class summary:

```text
class_id
num_entries
mean_quality
max_quality
num_fn_bg
num_fn_cls
num_fn_miss
num_fn_loc
prototype_norm
```

CSV에는 full direction vector를 기록하지 않는다. 디버깅에는 norm과 metadata만 기록한다.

## Failure Modes

- Low-quality HTM event가 TDB를 오염시킬 수 있다.
- Dominant class에는 많은 entry가 쌓이지만 rare class에는 entry가 없을 수 있다.
- Representation space drift 때문에 오래된 direction이 stale해질 수 있다.
- 관련 없는 direction을 평균내면 prototype이 상쇄될 수 있다.
- 학습 초기에 FN subtype label이 noisy할 수 있다.

완화책:

- Quality gating.
- Per-class capacity.
- Optional age decay.
- Invalid low-norm prototype skip.
- HTM 시작 전 warmup.
- subtype quality를 검증한 뒤 failure-type query 사용.

## V0 Implementation Checklist

1. `TransitionDirectionBank` class를 추가한다.
2. `TDBEntry` dataclass 또는 typed dict를 추가한다.
3. event validation을 구현한다.
4. quality top-K insertion을 구현한다.
5. `entry_id`와 `gt_uid` 기반 duplicate handling을 구현한다.
6. class-wise prototype query를 구현한다.
7. optional entry sampling을 구현한다.
8. `summary`, `state_dict`, `load_state_dict`를 구현한다.
9. HTM event output을 TDB update에 연결한다.
10. DDP에서 TDB state broadcast 또는 synchronization을 추가한다.
11. logging summary를 추가한다.

## Minimal Unit Tests

Validation:

```text
invalid direction is rejected
disallowed fn_type is rejected
non-positive quality is rejected when min_quality > 0
```

Retention:

```text
bank keeps at most K entries per class
higher-quality duplicate replaces lower-quality duplicate
max_entries_per_gt keeps only one entry per GT
```

Prototype:

```text
prototype has unit norm
higher-quality entry receives larger softmax weight
empty class returns None
low-norm weighted sum returns None
```

State:

```text
state_dict/load_state_dict preserves entries
stored tensors are detached
query moves prototype to requested device
```

## Research Claim

TDB는 다음처럼 설명할 수 있다.

```text
Transition Direction Bank stores high-quality false-negative-to-true-positive
feature transition directions and consolidates them into class-wise direction priors
for later hard-instance alignment.
```

Novelty는 단순히 hard example을 저장하는 것이 아니다. TDB는 false-negative 상태에서 성공적으로
회복된 feature direction을 저장하고, 그 direction을 미래 hard GT에 재사용 가능하게 만든다.
