# RASD - Relapse-Aware Support Distillation

## 개요

RASD는 FCOS 학습 단계에서만 동작하는 보조 distillation 모듈이다. MDMB++가 저장한
`SupportSnapshot.feature`를 temporal teacher로 사용하고, 이전에는 검출되었지만 이후 다시
miss 상태로 relapse된 GT의 현재 RoI feature를 그 teacher feature에 가깝게 당긴다.

현재 구현된 1차 버전은 attraction-only 방식이다.

```text
loss_rasd = lambda_rasd * warmup * mean(weight_g * (1 - cosine(current_gt_feature, support_feature_g)))
```

RASD는 inference path를 바꾸지 않는다. NMS, score threshold, target assignment, detector
output은 그대로 두고, train forward에서만 `losses["rasd"]`를 추가한다.

## 문제의식

Hard Replay는 아직 해결되지 않은 GT를 더 자주 보게 만들어 학습 exposure를 늘린다. 하지만
같은 이미지를 다시 보여주는 것만으로는, 과거에 해당 GT가 검출 가능했던 object-specific
representation을 보존한다고 보장할 수 없다.

MDMB++는 GT가 정상 검출되었을 때 support snapshot을 저장한다. RASD는 이 support snapshot을
teacher로 재사용한다. 즉, 같은 GT가 이후 다시 miss 상태가 되었을 때 현재 feature가 과거의
성공적인 support feature에서 너무 멀어지지 않도록 regularization을 건다.

이 때문에 RASD는 모든 miss에 적용되지 않고, 다음과 같은 relapse case에만 선택적으로
작동한다.

- 현재 MDMB++ entry가 unresolved failure 상태여야 한다.
- `entry.relapse == true`여야 한다.
- 과거에 정상 검출되어 저장된 `SupportSnapshot.feature`가 있어야 한다.
- 현재 transformed target과 MDMB++ entry가 class와 normalized box IoU 기준으로 매칭되어야 한다.

한 번도 정상적으로 검출된 적이 없는 GT는 support teacher가 없으므로 RASD 대상이 될 수 없다.
이런 GT는 Hard Replay로 exposure를 늘릴 수는 있지만, RASD attraction loss는 적용되지 않는다.

## Target 선택

`RelapseAwareSupportDistillation.plan(...)`은 현재 batch target의 `image_id`를 기준으로
`mdmbpp.get_image_entries(image_id)`를 읽는다. 이후 현재 transformed GT와 MDMB++ entry를
class와 normalized box IoU로 매칭한다.

대상으로 선택되려면 다음 조건을 모두 만족해야 한다.

- `entry.relapse == true`
- `entry.support is not None`
- `entry.support.feature is not None`
- `entry.consecutive_miss_count >= min_relapse_streak`
- `entry.severity >= min_severity`
- `entry.failure_type`이 `failure_types`에 포함됨
- `support.score >= min_support_score`
- `current_epoch - support.feature_epoch <= max_support_age`
- 현재 GT와 MDMB++ entry의 normalized box IoU가 `record_match_threshold` 이상

target weight는 severity와 relapse count를 반영해 계산한다.

```text
weight_g = min(
  max_target_weight,
  1 + severity_weight_scale * severity + relapse_weight_scale * relapse_count
)
```

따라서 더 심각한 failure이거나 relapse가 반복된 GT일수록 RASD loss에서 더 큰 비중을 갖는다.

## Feature 경로

RASD teacher feature는 MDMB++ post-step update 과정에서 저장된다.

1. `FCOSWrapper.after_optimizer_step()`이 optimizer step 이후 호출된다.
2. `MDMBFCOS.flush_post_step_updates(...)`가 no-grad inference를 수행한다.
3. 이때 backbone/FPN feature도 함께 확보한다.
4. `mdmbpp.config.store_support_feature`가 true이면 transformed GT box 위치에서
   `MultiScaleRoIAlign`으로 feature를 pooling한다.
5. pooled feature는 GAP, L2 normalize, CPU detach 과정을 거쳐
   `MDMBPlus.update(..., support_feature_list=...)`로 전달된다.
6. 해당 GT가 `detected` 상태이면 MDMB++가 support snapshot을 생성하거나 갱신한다.

학습 forward에서는 RASD가 현재 GT box의 feature를 같은 방식으로 pooling하고, 저장된 support
feature와 cosine attraction loss를 계산한다.

## Relapse-Robust Support Memory

MDMB++는 support teacher를 항상 최신 검출 결과로 덮어쓰지 않는다. support snapshot에는
box, score, IoU, quality, feature, feature epoch가 저장된다.

기본 support quality는 다음과 같다.

```text
quality = score_weight * support_score + iou_weight * support_iou
```

새로운 support가 기존 teacher를 교체하는 조건은 다음 중 하나다.

- 기존 teacher가 없음
- 기존 teacher에는 feature가 없고 새 support에는 feature가 있음
- 기존 teacher feature가 `support_memory.refresh_age` 이상 오래됨
- 새 support quality가 기존 quality보다 `support_memory.replace_margin` 이상 높음

`support_memory.require_feature: true`이면 새 detected support가 feature를 만들지 못했을 때 기존
feature teacher를 제거하지 않는다. 기존 feature를 유지하는 경우 `feature_epoch`도 기존 값을
유지하므로, RASD의 `max_support_age`는 실제 teacher feature의 나이를 기준으로 동작한다.

## 설정

RASD 설정 파일은 `modules/cfg/rasd.yaml`이다.

주요 설정은 다음과 같다.

```yaml
enabled: false
warmup_epochs: 5
lambda_rasd: 0.03

min_relapse_streak: 1
min_severity: 1.0
min_support_score: 0.2
max_support_age: 15
record_match_threshold: 0.95

max_targets_per_image: 5
max_targets_per_batch: 16
severity_weight_scale: 0.25
relapse_weight_scale: 0.25
max_target_weight: 3.0

failure_types:
  - cls_confusion
  - score_suppression
  - nms_suppression
  - loc_near_miss

confuser:
  enabled: false
```

`warmup_epochs`는 RASD를 해당 epoch 이후에 켜는 hard start가 아니다. 현재 구현에서는
`lambda_rasd`에 곱해지는 linear warmup factor다. 예를 들어 `warmup_epochs: 5`이면 epoch 1부터
`0.2 * lambda_rasd`, epoch 2에는 `0.4 * lambda_rasd`가 적용되고, epoch 5부터 전체
`lambda_rasd`가 적용된다.

RASD를 활성화하려면 두 조건이 필요하다.

- MDMB++가 함께 enabled 상태여야 한다.
- `modules/cfg/mdmbpp.yaml`에서 `store_support_feature: true`여야 한다.

teacher replacement 정책은 `modules/cfg/mdmbpp.yaml`의 `support_memory`에서 조절한다.

```yaml
support_memory:
  enabled: true
  score_weight: 1.0
  iou_weight: 1.0
  replace_margin: 0.05
  refresh_age: 15
  require_feature: true
```

## 학습 통합

현재 FCOS train forward의 흐름은 다음과 같다.

```text
image/target transform
backbone + FPN feature 계산
FCOS head 출력
base FCOS target matching
base FCOS loss 계산
Hard Replay metadata가 있으면 replay-aware loss weighting 적용
RASD target plan 생성
현재 GT RoI feature pooling
support feature와 cosine attraction loss 계산
losses["rasd"] 추가
```

RASD는 training-only auxiliary loss이므로 평가와 추론에서는 사용되지 않는다.

## 로그 지표 해석

`rasd.summary()`는 `history.json`의 `record["rasd"]` 아래에 저장된다.

주요 지표는 다음과 같다.

- `targets`: 해당 epoch에서 RASD 대상으로 선택된 GT 수
- `losses`: 실제 RASD loss 계산에 사용된 target 수
- `relapse_targets`: relapse 조건을 만족한 target 수
- `mean_severity`: RASD target의 평균 MDMB++ severity
- `mean_support_age`: support teacher feature의 평균 나이
- `mean_target_weight`: severity/relapse 기반 target weight 평균
- `mean_support_loss`: `1 - cosine(current, support)`의 평균
- `skipped_no_support`: support snapshot이 없어 제외된 수
- `skipped_support_too_old`: support teacher가 너무 오래되어 제외된 수
- `skipped_low_support_score`: support score가 낮아 제외된 수
- `skipped_no_entry_match`: 현재 GT와 MDMB++ entry가 매칭되지 않아 제외된 수
- `skipped_no_feature`: support feature가 없어 제외된 수

해석상 중요한 패턴은 다음과 같다.

- `targets`와 `losses`가 0에 가까우면 RASD가 실제로 거의 작동하지 않은 것이다.
- `skipped_support_too_old`가 크면 teacher freshness 문제가 있을 수 있다.
- `mean_support_age`가 높고 성능이 낮으면 오래된 teacher가 현재 representation과 충돌할 수 있다.
- `mean_support_loss`가 초반에는 의미 있게 존재하고 후반에 안정적으로 낮아지는 것은 자연스럽다.
- `train.classification` 또는 `train.bbox_regression`이 baseline보다 올라가면 RASD loss가 detector objective와 간섭할 수 있다.

## Hard Replay와의 관계

Hard Replay와 RASD는 역할이 다르다.

- Hard Replay: 어려운 GT 또는 이미지의 학습 exposure를 늘린다.
- RASD: relapse된 GT의 현재 feature를 과거 성공 support feature에 가깝게 유지한다.

따라서 RASD의 정당성은 단순히 mAP 상승만이 아니라 relapse 안정성으로도 확인해야 한다.

Hard Replay only 대비 Hard Replay + RASD에서 기대하는 방향은 다음과 같다.

- `bbox_mAP_50_95` 유지 또는 상승
- `bbox_mAP_75`, `bbox_mAR_100` 상승
- `mdmbpp.num_relapse` 감소
- `mdmbpp.relapses_this_epoch` 감소
- `mdmbpp_mean_severity` 감소 또는 안정화
- `replay_num_active_gt` 감소
- `rasd.targets`와 `rasd.losses`가 충분히 발생
- detection loss component가 악화되지 않음

반대로 mAP가 낮아지고 relapse 지표도 개선되지 않으며 `skipped_support_too_old`가 크다면,
현재 RASD 설정은 teacher freshness나 loss weight를 재조정해야 한다.

## 권장 Ablation

기본 ablation 순서는 다음과 같다.

1. Baseline
2. Baseline + MDMB++
3. MDMB++ + RASD
4. MDMB++ + Hard Replay
5. MDMB++ + Hard Replay + RASD

이 순서는 MDMB++ memory tracking 자체의 영향, RASD의 temporal support distillation 효과,
Hard Replay의 exposure 효과, 그리고 Hard Replay 위에서 RASD가 추가적인 relapse 안정성을
제공하는지를 분리해서 확인할 수 있게 한다.

## 튜닝 방향

RASD가 성능을 낮춘다면 먼저 다음 설정을 확인한다.

- `lambda_rasd`를 낮춘다. 예: `0.03 -> 0.01 -> 0.003`
- `warmup_epochs`만 늘리는 것은 hard start가 아니므로, 초반 비활성화가 필요하면 별도
  `start_epoch` 설정을 추가해야 한다.
- `max_support_age`를 줄여 오래된 teacher를 더 강하게 배제한다. 예: `15 -> 5~8`
- `severity_weight_scale`, `relapse_weight_scale`, `max_target_weight`를 낮춰 hard target의
  gradient 지배를 줄인다.
- `failure_types`를 `cls_confusion`, `score_suppression` 등 feature distillation과 관련성이 큰
  failure 중심으로 제한한다.

현재 버전에서는 `confuser.enabled: false`이므로 negative contrastive constraint는 적용되지
않는다. class-confusion relapse를 더 직접적으로 다루려면 RASD v2에서 confuser-aware
contrastive loss를 활성화하는 방향을 검토할 수 있다.
