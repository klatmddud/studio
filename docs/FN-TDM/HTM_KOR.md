# HTM - Hard Transition Mining

## 목표

HTM은 FN-TDM의 transition mining 컴포넌트다.

HTM의 역할은 학습 중 false-negative 상태에서 true-positive 상태로 전환되는 신뢰 가능한
ground-truth instance를 찾고, Transition Direction Bank (TDB)에 필요한 feature transition을
생성하는 것이다.

```text
epoch k에서 FN-like GT  ->  epoch e에서 TP-like GT
d = normalize(z_e - z_k)
```

첫 구현은 속도보다 정확성과 해석 가능성을 우선한다. 따라서 baseline 설계는 매 epoch 이후
deterministic full-train inference를 사용한다.

## 범위

HTM은 모델을 직접 학습하지 않는다. HTM은 transition event만 생성한다.

담당하는 역할:

- epoch-end mining을 train set 전체에 대해 수행한다.
- 각 GT instance에 detection state를 부여한다.
- 각 GT에 대해 GT-aligned embedding을 추출한다.
- rolling per-GT history를 유지한다.
- `FN -> TP` transition event를 생성한다.
- 분석과 ablation에 필요한 metadata를 기록한다.

담당하지 않는 역할:

- 최종 direction memory 정책 저장. 이는 TDB가 담당한다.
- auxiliary loss 적용. 이는 TAL 또는 detector wrapper가 담당한다.
- inference 동작 변경.
- COCO evaluation 대체.

## Baseline Mining Mode

HTM baseline은 각 training epoch 이후 실행된다.

```text
for epoch e:
    train one epoch

    model.eval()
    for image, targets in deterministic_train_loader:
        predictions, feature_maps = model(image, return_features=True)
        update HTM with predictions, feature_maps, targets, epoch=e
```

deterministic train loader는 random training augmentation을 꺼야 한다. 매 epoch 같은 image scaling과
normalization을 사용해야 feature 차이가 view noise보다 model learning을 더 잘 반영한다.

권장 기본값:

```text
mine_interval: 1
warmup_epochs: 0 or 1
deterministic_view: resize + normalize only
random_flip: false
strong_augmentation: false
```

초기 detector prediction이 너무 noisy하면 `warmup_epochs`를 늘릴 수 있다.

## GT Identity

HTM은 GT 중심으로 동작한다. 모든 record는 stable GT identity로 keying된다.

권장 key:

```text
gt_uid = (image_id, ann_id)
```

annotation ID가 없을 때의 fallback:

```text
gt_uid = (image_id, class_id, normalized_bbox_hash, instance_index)
```

fallback은 static COCO-style dataset에서는 충분히 사용할 수 있다. 다만 HTM은 같은 instance를 epoch
간 비교하므로 annotation ID 사용을 강하게 권장한다.

## Detection State

최소 state는 다음과 같다.

```text
state_e(g) in {TP, FN}
```

분석과 transition filtering을 위해 `state_e(g) = FN`일 때 failure subtype도 저장하는 것이 좋다.

```text
FN_BG
    GT 주변의 target-class/object confidence가 낮다.
    FN-TDM의 주된 타깃이다.

FN_CLS
    candidate가 GT와 overlap되지만 predicted class가 틀렸거나 target-class score가 낮다.

FN_LOC
    target class candidate의 confidence는 있지만 localization quality가 부족하다.

FN_MISS
    GT 근처에 의미 있는 candidate가 없다.
```

초기 transition source:

```text
use for TDB: FN_BG, FN_CLS, FN_MISS
log only or downweight: FN_LOC
```

이유는 FN-TDM이 background confusion과 missed foreground recognition을 겨냥하기 때문이다.
`FN_LOC`는 background-to-object feature movement보다 box regression 또는 assignment 문제가
지배적일 수 있다.

## TP Assignment

각 GT `g`에 대해 post-processed prediction을 사용해 TP를 정의한다.

```text
TP_e(g) =
    exists prediction p such that:
        class(p) == class(g)
        IoU(p.box, g.box) >= tau_iou
        score(p) >= tau_tp
```

권장 초기값:

```text
tau_iou: 0.5
tau_tp: 0.3
```

여러 prediction이 같은 GT와 match되면 가장 높은 quality를 가진 prediction을 사용한다.

```text
quality(p, g) = score(p) * IoU(p.box, g.box)
```

선택된 prediction은 다음 정보를 제공한다.

```text
matched_score
matched_iou
matched_class
matched_box
```

## FN Subtype Assignment

valid TP가 없을 때 HTM은 nearby prediction에서 coarse evidence를 계산한다.

정의:

```text
best_any_iou:
    모든 prediction box와 GT box 사이의 최대 IoU.

best_target_score:
    IoU와 무관하게 class == class(g)인 prediction 중 최대 score.

best_near_target_score:
    IoU >= tau_near인 prediction 중 target-class score의 최대값.

best_near_wrong_score:
    IoU >= tau_near인 prediction 중 wrong-class score의 최대값.

best_target_iou:
    class == class(g)이고 score >= tau_cls_evidence인 prediction 중 최대 IoU.
```

권장 초기값:

```text
tau_near: 0.3
tau_cls_evidence: 0.3
tau_bg_score: 0.1
tau_loc_score: 0.3
```

Subtype heuristic:

```text
if best_any_iou < tau_near and best_target_score < tau_bg_score:
    fn_type = FN_MISS

elif best_near_target_score < tau_bg_score and best_near_wrong_score < tau_bg_score:
    fn_type = FN_BG

elif best_any_iou >= tau_near and best_near_target_score < tau_tp:
    fn_type = FN_CLS

elif best_target_score >= tau_loc_score and best_target_iou < tau_iou:
    fn_type = FN_LOC

else:
    fn_type = FN_BG
```

이 subtype은 diagnostic 목적이며 evaluation을 대체하지 않는다. V0에서는 단순하게 유지한다.

## Feature Extraction

HTM은 detector가 object를 놓친 경우에도 GT-aligned feature를 추출해야 한다.

Baseline method:

```text
FPN feature maps -> MultiScaleRoIAlign(gt_bbox) -> GAP -> projection head -> normalize
```

표기:

```text
z_e(g) = normalize(projector(pool(ROIAlign(F_e, bbox_g))))
```

권장 기본값:

```text
roi_output_size: 7
projector_dim: 256
normalize_embedding: true
detach_embedding: true
store_on_cpu: true
```

projection head는 FN-TDM training state의 일부다. 저장된 direction과 현재 training embedding이 같은
공간에 있도록 HTM과 TAL이 projection head를 공유해야 한다.

중요: HTM은 detached embedding만 저장해야 한다. Mining pass의 computation graph를 유지하면 안 된다.

## Rolling History

HTM은 기본적으로 모든 epoch의 GT embedding을 stack하지 않는다. 대신 GT별 rolling state를 유지하고,
transition이 발생하면 event를 생성한다.

Per-GT state:

```python
HTMHistory = {
    "gt_uid": str,
    "image_id": Any,
    "ann_id": Any,
    "class_id": int,
    "bbox": Tensor[4],

    "last_state": "UNSEEN" | "FN" | "TP",
    "last_epoch": int | None,

    "last_fn_epoch": int | None,
    "last_fn_z": Tensor[D] | None,
    "last_fn_score": float | None,
    "last_fn_iou": float | None,
    "last_fn_type": str | None,

    "last_tp_epoch": int | None,
    "last_tp_z": Tensor[D] | None,
    "last_tp_score": float | None,
    "last_tp_iou": float | None,

    "fn_count": int,
    "tp_count": int,
    "transition_count": int,
    "last_emitted_epoch": int | None,
}
```

GT가 FN이면 최신 FN snapshot을 갱신한다.

```text
last_fn_epoch = current_epoch
last_fn_z = z_current
last_fn_score = matched_or_evidence_score
last_fn_iou = matched_or_evidence_iou
last_fn_type = fn_type
```

GT가 TP이면 최신 TP snapshot을 갱신한다.

```text
last_tp_epoch = current_epoch
last_tp_z = z_current
last_tp_score = matched_score
last_tp_iou = matched_iou
```

## Transition Emission

Baseline transition rule:

```text
emit when previous state is FN and current state is TP
```

조금 더 일반적으로는 짧은 window 안의 최근 FN을 허용한다.

```text
k = last_fn_epoch
if current_state == TP
and last_state == FN
and k is not None
and current_epoch - k <= transition_window:
    emit transition
```

권장 기본값:

```text
transition_window: 3
max_transitions_per_gt: 1
allowed_fn_types: [FN_BG, FN_CLS, FN_MISS]
```

동작 예시:

```text
epoch 9  FN  -> update last_fn
epoch 10 TP  -> emit transition d = normalize(z10 - z9)
epoch 11 TP  -> no new transition
```

반복 transition을 허용하는 경우:

```text
epoch 9  FN
epoch 10 TP  -> transition A
epoch 11 TP  -> no transition
epoch 12 FN
epoch 13 TP  -> transition B
```

V0에서는 `max_transitions_per_gt = 1`을 사용해 불안정한 instance가 memory를 과도하게 차지하는
것을 줄인다.

## Transition Event

HTM은 TDB로 event를 보낸다.

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

Direction:

```text
direction = normalize(z_tp - z_fn)
```

norm이 너무 작으면 event를 skip한다.

```text
min_direction_norm: 1e-6
```

## Event Quality

Quality는 TDB의 ranking, replacement, sampling에 사용된다.

Baseline score:

```text
quality = score_tp * (1 - score_fn) * exp(-(epoch_tp - epoch_fn - 1) / lambda_gap)
```

권장 기본값:

```text
lambda_gap: 2.0
```

Optional subtype weighting:

```text
FN_BG:   1.0
FN_MISS: 1.0
FN_CLS:  0.8
FN_LOC:  0.3
```

그 다음:

```text
quality = quality * type_weight[fn_type]
```

V0에서는 `FN_LOC`를 제외하거나 분석용으로만 낮은 quality와 함께 유지한다.

## State Update Order

state를 덮어쓰기 전에 이전 state로 transition 여부를 검사한다.

```python
def update_gt(history, current, epoch):
    prev_state = history.last_state

    if current.state == "TP":
        if should_emit(history, current, prev_state, epoch):
            event = build_event(history.last_fn, current)
            emit(event)

        history.last_tp_epoch = epoch
        history.last_tp_z = current.z
        history.last_tp_score = current.score
        history.last_tp_iou = current.iou
        history.tp_count += 1

    else:
        history.last_fn_epoch = epoch
        history.last_fn_z = current.z
        history.last_fn_score = current.score
        history.last_fn_iou = current.iou
        history.last_fn_type = current.fn_type
        history.fn_count += 1

    history.last_state = current.state
    history.last_epoch = epoch
```

transition을 생성한 뒤에는 `TP -> TP`에서 다시 생성하지 않는다. 디버깅을 위해 `last_fn_*`를 그대로
둘 수도 있고 consumed로 표시할 수도 있다. 단 transition 조건은 반드시 `prev_state == FN`을 요구해야
한다.

## Configuration Sketch

향후 `modules/cfg/fntdm.yaml` 또는 `modules/cfg/htm.yaml`에 config를 추가한다.

```yaml
htm:
  enabled: false
  mode: epoch_end_full_train
  mine_interval: 1
  warmup_epochs: 0

  matching:
    tau_iou: 0.5
    tau_tp: 0.3
    tau_near: 0.3
    tau_bg_score: 0.1
    tau_cls_evidence: 0.3
    tau_loc_score: 0.3

  transition:
    transition_window: 3
    max_transitions_per_gt: 1
    allowed_fn_types: [FN_BG, FN_CLS, FN_MISS]
    min_direction_norm: 1.0e-6
    lambda_gap: 2.0

  features:
    source: fpn_roi_align_gt
    roi_output_size: 7
    projector_dim: 256
    normalize: true
    store_on_cpu: true

  logging:
    save_events_csv: true
    save_epoch_summary: true
```

## Logging

HTM은 compact summary와 transition event를 모두 기록해야 한다.

Epoch summary:

```text
epoch
num_gt
num_tp
num_fn
num_fn_bg
num_fn_cls
num_fn_loc
num_fn_miss
num_transitions
num_skipped_transition
```

Event CSV:

```text
epoch_fn,epoch_tp,gt_uid,image_id,ann_id,class_id,fn_type,
score_fn,score_tp,iou_fn,iou_tp,quality,direction_norm
```

CSV에는 큰 embedding tensor를 쓰지 않는다. event embedding을 persist해야 한다면 TDB가 소유하는 별도
tensor checkpoint를 사용한다.

## DDP Behavior

첫 baseline에서는 epoch가 끝난 뒤 rank 0에서만 epoch-end HTM을 실행하는 것을 선호한다. rank 0은
deterministic train loader를 만들고 transition을 mining할 수 있다.

구현 선택지:

1. Rank 0이 full train set을 mining하고 다음 epoch 전에 TDB state를 broadcast한다.
2. 모든 rank가 disjoint train subset을 mining한 뒤 state/event를 gather하고 merge한다.

V0에서는 option 1이 더 단순하므로 권장한다. 더 느릴 수는 있지만 duplicate event와 merge complexity를
피할 수 있다.

## Expected Overhead

이 baseline은 mining epoch마다 training set 전체에 대한 detector inference를 한 번 추가한다.
의도적으로 비싼 설계다.

저렴한 online HTM variant를 개발하기 전에 clean reference implementation으로 사용한다.

필요할 때의 완화책:

- `mine_interval`을 늘린다.
- `warmup_epochs` 이후 시작한다.
- 디버깅 시 fixed subset만 mining한다.
- 선택된 event 외 embedding persistence를 끈다.

## Edge Cases

- image에 prediction이 없는 경우: 모든 GT는 `FN_MISS`가 된다.
- 여러 prediction이 하나의 GT와 match되는 경우: 가장 높은 `score * IoU`를 선택한다.
- 하나의 prediction이 여러 GT와 match되는 경우: V0의 HTM state에서는 GT별 독립 matching을 허용한다.
  필요하면 나중에 evaluation-style one-to-one matching을 추가할 수 있다.
- direction norm이 거의 0인 경우: event를 skip한다.
- GT가 첫 관측부터 TP인 경우: TP state만 갱신하고 transition은 생성하지 않는다.
- GT가 자주 번갈아 바뀌는 경우: `max_transitions_per_gt`로 event spam을 제어한다.
- annotation ID가 없는 경우: bbox hash fallback을 사용하고 warning을 한 번 기록한다.

## V0 Implementation Checklist

1. 기본 disabled 상태의 HTM config를 추가한다.
2. stable GT UID helper를 만든다.
3. deterministic train mining loader를 추가한다.
4. post-processed prediction과 FPN feature를 반환하는 detector inference path를 추가한다.
5. prediction으로부터 GT-state assignment를 구현한다.
6. GT ROIAlign embedding extractor와 projection head를 추가한다.
7. rolling `HTMHistory` store를 추가한다.
8. transition emission과 quality scoring을 추가한다.
9. TDB로 event를 넘기는 API를 추가한다.
10. epoch summary와 event CSV logging을 추가한다.
11. state transition unit test를 추가한다.

## Minimal Unit Tests

State transition:

```text
FN -> TP emits one event
TP -> TP emits zero events
TP -> FN -> TP emits one new event if allowed
```

Limits:

```text
max_transitions_per_gt = 1 prevents repeated events
transition_window rejects stale FN snapshots
disallowed fn_type does not emit
```

Quality:

```text
higher score_tp increases quality
higher score_fn decreases quality
larger epoch gap decreases quality
```

Feature:

```text
direction has unit norm
near-zero direction is skipped
stored z tensors are detached and on CPU when configured
```

## Research Claim

HTM은 다음처럼 설명할 수 있다.

```text
Hard Transition Mining identifies reliable false-negative-to-true-positive turning
points of hard ground-truth instances and extracts the discriminative feature direction
that accompanied the recovery.
```

이는 FN-TDM을 일반적인 hard example mining과 구분한다. HTM은 어떤 instance가 hard한지만 묻지 않는다.
어떤 hard instance가 detectable해졌고, 그 회복을 표시한 feature direction이 무엇인지 묻는다.
