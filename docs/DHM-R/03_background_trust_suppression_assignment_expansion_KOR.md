# Background Trust Suppression + Assignment Expansion

## 개요

Background Trust Suppression + Assignment Expansion은 DHM-R의 `FN_BG` 전용 보정 모듈이다.
DHM mining에서 `FN_BG`로 판정된 GT는 GT 주변에 충분한 정답 class evidence도, 강한 wrong-class
evidence도 없어서 detector가 해당 영역을 background처럼 처리한 case다. 따라서 이 문제는 단순히
classification loss를 더 키우는 것보다, GT 주변의 background supervision을 의심하고 latent foreground
후보를 다시 학습 경로에 넣는 방식이 더 적합하다.

이 방법은 두 개의 complementary path로 구성된다.

- Background Trust Suppression: `FN_BG` hard GT 주변 negative point가 진짜 background인지 신뢰도를
  추정하고, 의심스러운 background supervision을 약화한다.
- Assignment Expansion: 반복 `FN_BG` GT에 대해 원래 negative였던 point 중 geometry 조건을 만족하는
  후보를 capped backup positive로 추가한다.

핵심은 hard GT를 단순히 더 크게 reweighting하는 것이 아니라, background로 학습되던 영역 자체를
temporal evidence 기반으로 재해석하는 것이다.

## 문제 정의

DHM 기준에서 `FN_BG`는 다음 상황을 의미한다.

- GT 주변에 TP 조건을 만족하는 same-class prediction이 없다.
- GT 근처의 same-class score와 wrong-class score가 모두 낮다.
- 즉 모델이 해당 GT 영역을 foreground object로 활성화하지 못한다.

`FN_MISS`가 물체 근처의 candidate 자체가 거의 없는 상태라면, `FN_BG`는 GT 주변이 training 과정에서
background로 강하게 눌렸거나, FCOS assignment가 충분한 positive point를 만들지 못했을 가능성이 크다.

따라서 `FN_BG`는 다음 두 원인이 섞여 있을 수 있다.

- harmful negative supervision: GT 주변 point가 background negative로 계속 학습됨
- insufficient positive assignment: GT에 배정된 positive point가 너무 적거나 부적절함

Background Trust Suppression + Assignment Expansion은 이 두 원인을 동시에 다룬다.

## 모듈 구성

### 1. Background Trust Memory

GT 단위로 background collapse 이력을 저장한다.

```text
BackgroundTrustRecord
  gt_uid
  image_id
  class_id
  last_state
  total_seen
  fn_bg_count
  consecutive_fn_bg
  last_seen_epoch
  ema_gt_score
  ema_near_score
  ema_near_iou
  ema_background_trust
  instability_score
  candidate_density
  support_positive_count
```

여기서 `ema_background_trust`는 GT 주변 negative point를 background로 믿어도 되는지를 나타낸다.
값이 낮을수록 "이 영역은 background label을 강하게 믿으면 안 된다"는 뜻이다.

### 2. Background Trust Estimator

FPN point 또는 FCOS head feature에서 background trust map을 예측하는 training-only branch를 둔다.

입력:

- FPN feature 또는 classification tower feature
- point-to-GT geometry embedding
- optional: DHM instability scalar
- optional: GT size/FPN level embedding

출력:

```text
background_trust: [N_points]
latent_foreground_score: [N_points]
```

`background_trust`는 해당 point의 background supervision을 얼마나 신뢰할지 예측한다.
`latent_foreground_score`는 현재 detector head가 놓쳤지만 foreground일 가능성이 있는 point를 찾기 위한
auxiliary signal이다.

이 branch는 inference에서 제거하는 것을 V0 기본값으로 둔다. 목적은 detector의 shared feature가 hard
foreground 영역을 background로 collapse하지 않도록 돕는 것이다.

### 3. Suspicious Background Region

`FN_BG` GT 주변에서 원래 FCOS assignment상 negative인 point 중 suspicious background를 찾는다.

후보 조건:

```text
point is currently negative
point is inside GT box
point is near GT center or within expanded center radius
point FPN level is compatible with GT size
DHM record satisfies FN_BG hard selector
```

이 후보들은 두 가지 방식으로 사용한다.

- background trust target으로 사용: "이 point는 background로 강하게 믿으면 안 됨"
- assignment expansion 후보로 사용: capped backup positive로 일부 전환

### 4. HLAE Assignment Expansion

현재 DHM의 `assignment_expansion`과 연결되는 path다. 반복 `FN_BG` GT에 대해 원래 negative였던 point를
제한적으로 positive로 추가한다.

선택 조건:

```text
last_state in target_failure_types
target_failure_types includes FN_BG
total_seen >= min_observations
instability_score >= min_instability
candidate is inside GT box
candidate satisfies scale range
candidate is center-nearest under expanded radius
```

추가 positive 수는 `backup_topk`와 `max_extra_positive_ratio`로 제한한다. 이 제한이 없으면 background
noise가 늘어 precision이 급격히 떨어질 수 있다.

## 학습 목표

### Background Trust Loss

`FN_BG` GT 주변 suspicious negative point에 대해 background trust를 낮추도록 학습한다.

```text
target_trust = 0 for suspicious background points
target_trust = 1 for reliable easy background points
L_trust = BCE(background_trust, target_trust)
```

easy background는 GT와 충분히 멀고, DHM hard record와 겹치지 않는 negative point에서 sample한다.
class imbalance를 피하기 위해 suspicious point와 easy background point를 balanced sampling한다.

### Latent Foregroundness Loss

`FN_BG` GT 주변 point가 foreground-like feature를 갖도록 auxiliary objectness target을 준다.

```text
target_fg = 1 for suspicious FN_BG region points
target_fg = 0 for reliable background points
L_fg = focal_loss(latent_foreground_score, target_fg)
```

이 branch는 최종 detection score로 직접 쓰지 않는다. shared feature가 objectness signal을 보존하도록
돕는 training-only objective다.

### Trust-Gated Negative Suppression

Background Trust Estimator의 출력으로 suspicious point의 negative classification gradient를 약화한다.
이는 정적 reweighting이 아니라 learned trust map에 기반한 background supervision gating이다.

```text
negative_cls_loss_i <- background_trust_i * negative_cls_loss_i
```

V0에서는 detach된 trust를 사용할 수 있다.

```text
negative_cls_loss_i <- stopgrad(background_trust_i) * negative_cls_loss_i
```

이렇게 하면 trust branch가 단순히 loss를 줄이기 위해 collapse하는 문제를 줄일 수 있다.

### Assignment Expansion Loss

backup positive로 추가된 point는 standard FCOS positive loss를 받는다. 단, 원래 positive보다 낮은
auxiliary quality target 또는 capped loss를 사용한다.

```text
L_backup = L_cls_backup + L_box_backup + L_ctr_backup
```

backup positive는 noisy할 수 있으므로 다음 제한을 둔다.

- box/centerness loss weight cap
- GT당 top-k 제한
- image당 extra positive ratio 제한
- warmup 이후 적용

### 전체 Loss

```text
L_total = L_fcos
        + lambda_trust * L_trust
        + lambda_fg * L_fg
        + lambda_backup * L_backup
```

Trust-Gated Negative Suppression은 `L_fcos`의 negative classification term에 적용되는 modifier로 본다.

## FCOS와의 통합

V0 통합 흐름은 다음과 같다.

1. DHM epoch-end mining이 GT별 `FN_BG` state와 instability를 기록한다.
2. 다음 epoch training forward에서 batch GT와 DHM record를 매칭한다.
3. `FN_BG` hard GT 주변 negative point를 suspicious background 후보로 찾는다.
4. Background Trust Estimator가 trust map과 latent foregroundness를 예측한다.
5. suspicious point의 negative classification supervision을 trust로 gate한다.
6. HLAE Assignment Expansion이 일부 suspicious point를 backup positive로 추가한다.
7. auxiliary trust/foregroundness loss와 backup positive loss를 base detection loss에 더한다.

기본 inference에서는 Background Trust Estimator와 latent foreground branch를 제거한다. 따라서 inference
latency, NMS, score threshold는 유지된다.

## DHM-R 내 역할

DHM-R은 failure type별로 서로 다른 repair path를 둔다.

- `FN_LOC`: Border-aware residual refinement
- `FN_CLS`: Confusion Prototype Memory Module
- `FN_BG`: Background Trust Suppression + Assignment Expansion

이 모듈은 `FN_BG` 전용 path다. `FN_LOC`처럼 box edge를 고치거나, `FN_CLS`처럼 class prototype을 분리하는
것이 아니라, GT 주변 영역이 background로 학습되는 과정을 완화하고 foreground assignment를 복구한다.

## Novelty 포인트

기존 foreground/background imbalance 방법은 대개 현재 mini-batch의 easy negative를 줄이거나 hard example을
강조한다. 이 방법은 다음 점에서 다르다.

- GT instance별 temporal `FN_BG` history를 사용한다.
- background supervision을 항상 신뢰하지 않고, DHM evidence로 unreliable background region을 정의한다.
- learned background trust map으로 negative gradient를 gate한다.
- suspicious background point 일부를 capped backup positive로 전환해 assignment 자체를 복구한다.
- inference graph를 바꾸지 않고 foreground representation collapse를 줄인다.

따라서 이 방법은 일반 hard-negative mining이 아니라, detection hysteresis를 이용한 background trust
estimation과 latent foreground assignment repair로 정의할 수 있다.

## 기대 효과

주요 개선 목표:

- `FN_BG` count 감소
- `num_current_failures` 감소
- recall 관련 지표 개선
- `bbox_mAR_10`, `bbox_mAR_100` 개선
- DHM `last_state_counts.FN_BG` 감소
- DHM `dominant_failure_counts.FN_BG` 감소

부수적으로 기대할 수 있는 변화:

- `FN_MISS`로의 악화 방지
- hard GT의 `ema_gt_score` 증가
- `recoveries` 증가
- 후반 epoch의 `relapses` 감소

## Ablation 계획

| 실험 | 구성 | 목적 |
|---|---|---|
| Baseline | FCOS + DHM mining only | 기준선 |
| Trust branch only | `L_trust`, `L_fg`만 추가 | background collapse 완화 효과 확인 |
| Trust-gated negative | learned trust로 negative cls loss gate | harmful background suppression 효과 확인 |
| Assignment expansion only | HLAE backup positive만 사용 | assignment 복구 효과 확인 |
| Trust + expansion | trust map과 HLAE 결합 | 두 path의 상호 보완성 확인 |
| no inference change | training-only branch 제거 후 평가 | representation transfer 확인 |

## 실패 가능성과 점검 항목

### False positive 증가

background suppression이나 backup positive가 과하면 background region이 foreground로 뜰 수 있다.
`backup_topk`, `max_extra_positive_ratio`, `lambda_fg`, `lambda_backup`을 보수적으로 둔다.

### Trust branch collapse

trust branch가 모든 point의 trust를 낮추면 training loss는 줄어도 detector가 망가질 수 있다. reliable
background negative를 명확히 sampling하고, trust target balance를 유지한다.

### Noisy FN_BG 판정

score threshold가 너무 높으면 사실상 weak `FN_CLS` 또는 near `FN_LOC`도 `FN_BG`로 들어올 수 있다.
`min_observations`, `consecutive_fn_bg`, `instability_score` 조건으로 안정적인 record만 사용한다.

### Assignment noise

backup positive가 잘못된 FPN level이나 box 외곽 point에 생기면 localization 성능이 떨어질 수 있다.
scale range, inside box, center radius, top-k 제한을 동시에 둔다.

### Precision-recall tradeoff

`FN_BG` 감소가 recall을 올리지만 precision을 낮출 수 있다. AP뿐 아니라 `num_predictions`,
confusion matrix의 background false positive, score distribution을 함께 본다.

## V0 구현 스케치

```text
for each DHM mining epoch:
  detections = model(train_images)
  for each GT:
    state = DHM.assign_detection_state(GT, detections)
    if state == FN_BG:
      memory.update_background_record(
        gt_uid,
        fn_bg_count += 1,
        near_score,
        near_iou,
        candidate_density,
      )

for each training batch:
  matched_idxs = FCOS.match_anchors_to_targets(...)
  hard_bg_records = memory.lookup_fn_bg_records(batch_gt)
  suspicious_points = find_negative_points_near_hard_gt(
    matched_idxs,
    hard_bg_records,
    anchors,
    gt_boxes,
  )

  trust, fg = background_trust_head(features)
  losses["dhmr_bg_trust"] = trust_loss(trust, suspicious_points, easy_bg_points)
  losses["dhmr_fg"] = foregroundness_loss(fg, suspicious_points, easy_bg_points)

  cls_loss = apply_trust_gated_negative_suppression(cls_loss, trust, suspicious_points)

  expanded_matched_idxs = apply_hlae_assignment_expansion(
    matched_idxs,
    hard_bg_records,
    suspicious_points,
    backup_topk,
    max_extra_positive_ratio,
  )
```

## 기본 설정 초안

```yaml
dhmr:
  enabled: true
  background_trust:
    enabled: true
    min_observations: 3
    min_fn_bg_count: 2
    min_instability: 0.25
    consecutive_fn_bg: 1
    trust_head_channels: 128
    detach_trust_for_cls_gate: true
    suspicious_region:
      center_radius_multiplier: 1.75
      max_radius: 3.0
      require_inside_box: true
      require_scale_match: true
    assignment_expansion:
      enabled: true
      backup_topk: 3
      max_extra_positive_ratio: 0.2
    loss:
      trust_weight: 0.1
      foregroundness_weight: 0.1
      backup_weight: 0.25
      warmup_epochs: 2
      max_gt_per_image: 16
      easy_bg_samples_per_image: 256
```

## 한 줄 요약

Background Trust Suppression + Assignment Expansion은 DHM이 찾아낸 `FN_BG` hard GT 주변의 background
supervision을 learned trust map으로 의심하고, 일부 latent foreground point를 capped backup positive로
복구해 foreground representation collapse를 줄이는 DHM-R의 background repair path다.
