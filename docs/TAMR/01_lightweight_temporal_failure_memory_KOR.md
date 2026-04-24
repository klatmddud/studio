# Lightweight Temporal Failure Memory

## 목표

무거운 MDMB++ 저장 경로를 compact per-GT temporal state로 대체하고, 이 상태를 training-only
regularization의 prior로 사용한다.

현재 MDMB++는 `best_candidate`, `topk_candidates`, support snapshot 같은 rich candidate context를
저장한다. 분석에는 유용하지만, FCOS가 optimizer step 이후 dense candidate summary를 다시 만들어야 하므로
비싸다. TAMR은 이 경로를 피해야 한다.

## 저장 상태

Stable GT identity마다 하나의 record를 저장한다.

```text
gt_uid
image_id
class_id
bbox_norm
first_seen_epoch
last_seen_epoch
last_state
miss_streak
max_miss_streak
total_miss
relapse_count
last_detected_epoch
last_failure_epoch
last_failure_type
risk
support_proto optional
support_quality optional
support_epoch optional
```

이 state는 의도적으로 dense candidate와 full ROI feature를 제외한다.

## GT Identity

가능하면 dataset annotation ID를 사용한다. Annotation ID가 없다면 현재 MDMB++와 같은 fallback을 쓴다.

```text
image_id + class_id + normalized bbox hash
```

이 fallback은 static COCO-style dataset에서는 충분히 쓸 수 있다. 다만 augmentation이나 box transform이
복잡해지는 경우 annotation ID가 더 안정적이다.

## Update Signal

Memory는 normal training forward에서 이미 사용 가능한 signal로 갱신해야 한다.

- GT별 assigned positive location
- 해당 location의 classification target과 prediction
- assigned box의 regression quality 또는 IoU
- 사용 가능한 경우 centerness 또는 quality score
- 현재 training loss term

두 번째 no-grad inference pass는 피한다. 각 GT는 coarse state로만 분류해도 충분하다.

```text
detected_like
weak_positive
classification_confusion
localization_weak
missing_assignment
```

이 state가 final post-NMS detection outcome과 정확히 일치할 필요는 없다. TAMR에 필요한 것은 완벽한
evaluator가 아니라 유용한 training prior다.

## Risk Score

Assignment와 loss weight에 안전하게 사용할 수 있도록 bounded scalar risk를 사용한다.

```text
risk = sigmoid(
    a * normalized_miss_streak
  + b * log1p(total_miss)
  + c * relapse_count
  + d * failure_type_prior
  - e * recent_recovery
)
```

초기 권장값:

```text
a = 1.0
b = 0.5
c = 1.0
d = 0.5
e = 1.0
```

`recent_recovery`는 `last_detected_epoch`와 현재 epoch의 거리에 따라 decay할 수 있다.

## Support Prototype

Prototype distillation을 켜는 경우 full 7x7 ROI tensor 대신 compact vector를 저장한다.

```text
support_proto: float16 or float32 vector with shape [C] or [C_reduced]
support_quality: scalar
support_epoch: int
```

Prototype은 assigned positive-location feature를 pooling한 뒤 EMA로 갱신할 수 있다.

```text
proto <- normalize(momentum * proto + (1 - momentum) * current_proto)
```

현재 quality가 충분히 높을 때만 prototype을 갱신한다.

## 예상 Overhead

목표 overhead는 다음 수준이어야 한다.

- GT당 작은 dictionary update 1회
- 이미 선택된 positive location에 대한 optional feature pooling
- post-step inference 없음
- dense candidate summary 없음
- memory path 내부 NMS 없음

이 구조는 현재 MDMB++보다 훨씬 저렴해야 한다.

## Failure Mode

- Training-forward state가 final detection failure를 완벽히 예측하지 못할 수 있다.
- Risk를 clip하지 않으면 noisy label을 과도하게 강조할 수 있다.
- 강한 data transform에서는 bbox-hash identity가 불안정할 수 있다.
- 모든 GT에 prototype을 저장하면 큰 dataset에서는 여전히 memory가 커질 수 있다. 필요하면 top-risk retention을 사용한다.

