# Confusion Prototype Memory Module

## 개요

Confusion Prototype Memory Module은 DHM-R의 `FN_CLS` 전용 보정 모듈이다. DHM mining에서
`FN_CLS`로 판정된 GT는 물체 근처에 detector 반응은 있지만, 정답 class confidence가 TP 기준을
넘지 못한 case다. 이 경우 문제의 핵심은 box regression보다 class decision boundary와 feature
discriminability에 있다.

이 모듈은 `FN_CLS` hard GT에 대해 다음 정보를 시간축으로 저장한다.

- GT가 TP였던 시점의 correct support prototype
- GT가 `FN_CLS`였던 시점의 current hard feature
- GT 주변에서 반복적으로 등장한 wrong-class confuser prototype
- 정답 class score와 confuser class score의 temporal margin

그 다음 training 중 같은 GT 또는 같은 class/confuser pair가 다시 등장하면, current GT feature를
correct prototype 쪽으로 당기고 recurring confuser prototype에서 밀어낸다. inference graph는 기본적으로
유지하며, prototype branch는 training-only auxiliary module로 둔다.

## 문제 정의

DHM 기준에서 `FN_CLS`는 다음 상황을 의미한다.

- GT 주변에 prediction이 존재한다.
- 하지만 같은 class near prediction의 score가 `tau_tp`보다 낮다.
- 따라서 TP로 인정되지 않는다.

즉 `FN_CLS`는 모델이 물체 위치 근처까지는 반응했지만, 정답 class evidence가 충분히 강하지 않거나
wrong class evidence가 상대적으로 강한 상태다. 예를 들어 `car` GT 주변에 `truck` score가 강하게 뜨거나,
`pedestrian` GT 주변 feature가 `cyclist`와 섞이는 경우가 여기에 해당한다.

일반적인 class reweighting은 class 전체의 loss 강도를 바꾸지만, 어떤 GT가 어떤 wrong class와 반복적으로
헷갈리는지는 보존하지 않는다. Confusion Prototype Memory Module은 DHM의 GT별 temporal state와
confuser 정보를 이용해 실제로 관측된 confusion pair만 선택적으로 보정한다.

## 모듈 구성

### 1. Confusion Prototype Memory

GT 단위 record와 class-pair 단위 prototype bank를 함께 둔다.

```text
ConfusionRecord
  gt_uid
  image_id
  class_id
  last_state
  total_seen
  fn_cls_count
  tp_count
  last_seen_epoch
  last_confuser_class
  confusion_counts: {wrong_class: count}
  ema_gt_score
  ema_confuser_score
  ema_margin
  instability_score
  support_quality
  support_feature
```

class-pair prototype bank는 반복 confusion을 class relation 수준에서도 공유하기 위한 구조다.

```text
ConfusionPrototypeBank
  key: (gt_class, confuser_class)
  positive_proto
  negative_proto
  count
  quality
  last_update_epoch
```

GT별 memory는 instance-specific correction을 담당하고, class-pair bank는 같은 confusion pair를 공유하는
다른 GT에도 일반화 신호를 제공한다.

### 2. Prototype Extractor

prototype은 FPN feature에서 추출한다.

후보 방식:

- GT box 기준 MultiScaleRoIAlign feature
- FCOS positive point 중 GT class score가 가장 높은 point feature
- detection candidate box 기준 RoIAlign feature
- head 직전 classification tower feature

V0에서는 GT box 기준 RoIAlign feature를 우선한다. post-processed detection의 box quality에 덜 민감하고,
GT identity와 직접 연결되기 때문이다.

```text
z_gt = Projector(RoIAlign(FPN, gt_box))
z_candidate = Projector(RoIAlign(FPN, candidate_box))
```

`Projector`는 작은 MLP 또는 `1x1 conv + norm + linear`로 둔다. memory에는 detached CPU tensor를 저장하고,
training loss 계산 시 현재 feature와 비교한다.

### 3. Correct Support Prototype

GT가 `TP`로 관측된 epoch에서는 correct support prototype을 갱신한다.

갱신 조건:

```text
state == TP
score >= support_score_threshold
iou >= support_iou_threshold
```

갱신은 quality-gated EMA로 한다.

```text
p_pos <- normalize(momentum * p_pos + (1 - momentum) * z_gt)
quality <- max_or_ema(score * iou)
```

기존 support보다 quality가 낮은 관측은 prototype을 덮어쓰지 않거나 낮은 비율로만 반영한다. 이렇게 해야
일시적인 noisy TP가 memory를 오염시키지 않는다.

### 4. Confuser Prototype

GT가 `FN_CLS`로 관측되면 GT 주변의 wrong-class candidate를 찾아 negative prototype으로 저장한다.

confuser 후보:

- `IoU >= tau_near`인 prediction 중 class가 GT class와 다른 candidate
- wrong-class score가 가장 높은 candidate
- 또는 GT 주변 positive point에서 가장 큰 competing class logit

```text
confuser_class = argmax_{c != gt_class} score_c
p_neg[confuser_class] <- EMA(Projector(candidate_feature))
```

명시적인 wrong-class detection이 없고 정답 class score만 낮은 경우에는 `unknown_confuser` bucket으로 두거나,
negative prototype 없이 positive attraction loss만 적용한다.

### 5. Hard GT Selector

모든 GT에 적용하지 않고, DHM이 충분히 안정적으로 `FN_CLS` 문제를 관측한 GT만 사용한다.

```text
last_state == FN_CLS
total_seen >= min_observations
fn_cls_count >= min_fn_cls_count
instability_score >= min_instability
support_feature exists
```

confuser negative를 쓰려면 다음 조건도 추가한다.

```text
confusion_counts[confuser_class] >= min_confuser_count
negative_proto exists
```

## 학습 목표

### Prototype Contrastive Loss

현재 GT feature `z`를 correct support prototype `p+`에 가깝게 하고, recurring confuser prototype
`p-`에서 멀어지게 한다.

```text
L_proto = -log exp(sim(z, p+) / T)
          / (exp(sim(z, p+) / T) + sum_c w_c exp(sim(z, p-_c) / T))
```

여기서 `w_c`는 해당 confuser class가 얼마나 자주 등장했는지를 반영한다.

```text
w_c = normalize(confusion_counts[c])
```

### Confusion Margin Loss

classification logit에도 직접적인 disambiguation margin을 건다. 단, class 전체가 아니라 DHM이 기록한
GT-confuser pair에만 적용한다.

```text
L_margin = max(0, m + logit_confuser - logit_gt)
```

margin은 temporal difficulty에 따라 조절한다.

```text
m = base_margin + beta * instability_score + gamma * normalized_confuser_count
```

이 loss는 reweighting이 아니라, 실제로 헷갈린 class pair 사이의 decision boundary를 벌리는 auxiliary
constraint다.

### Support Consistency Loss

현재 GT feature가 이전 TP support와 너무 멀어지지 않도록 한다.

```text
L_support = 1 - cosine(z_current, stopgrad(p_pos))
```

`FN_CLS`가 아닌 normal GT에도 약하게 적용할 수 있지만, V0에서는 hard `FN_CLS` GT에만 적용한다.

### 전체 Loss

```text
L_total = L_fcos
        + lambda_proto * L_proto
        + lambda_margin * L_margin
        + lambda_support * L_support
```

초기 epoch에는 support prototype이 충분하지 않으므로 warmup 이후 적용한다.

## FCOS와의 통합

V0 통합 흐름은 다음과 같다.

1. DHM epoch-end mining이 GT별 `TP`, `FN_CLS` state를 기록한다.
2. Confusion Prototype Memory가 mining 결과와 prediction candidate를 이용해 support/confuser prototype을 갱신한다.
3. 다음 epoch training forward에서 batch GT와 memory record를 매칭한다.
4. `FN_CLS` hard GT의 RoIAlign feature를 추출해 projection한다.
5. support prototype과 confuser prototype을 조회한다.
6. prototype contrastive loss와 confusion margin loss를 base detection loss에 더한다.

기본 inference에서는 prototype extractor와 projector를 제거한다. 따라서 inference latency와 NMS behavior는
바뀌지 않는다.

## DHM-R 내 역할

DHM-R은 failure type별로 서로 다른 repair path를 둔다.

- `FN_LOC`: HLRT
- `FN_CLS`: Confusion Prototype Memory Module
- `FN_BG`: Latent Foregroundness Branch

Confusion Prototype Memory Module은 classification confusion 전용 path다. `FN_LOC`처럼 box boundary를
고치는 것이 아니라, hard GT의 feature가 정답 class prototype과 confuser class prototype 사이에서 더
분리되도록 만든다.

## Novelty 포인트

기존 class imbalance 방법은 대개 class frequency나 current batch의 hard sample에 기반한다. Confusion
Prototype Memory Module은 다음 차이가 있다.

- GT instance별 temporal confusion memory를 사용한다.
- 실제로 반복 관측된 GT class와 wrong class pair만 보정한다.
- TP 시점의 correct support prototype과 FN_CLS 시점의 confuser prototype을 함께 저장한다.
- class-level prototype이 아니라 instance-level prototype과 class-pair prototype을 결합한다.
- inference 구조를 바꾸지 않고 classification-discriminative representation을 주입한다.

따라서 이 방법은 generic contrastive learning이나 long-tail rebalancing이 아니라, detection hysteresis를
이용한 failure-type-aware class disambiguation으로 정의할 수 있다.

## 기대 효과

주요 개선 목표:

- `FN_CLS` count 감소
- confusion matrix의 off-diagonal error 감소
- class별 AP 중 혼동이 큰 class pair 개선
- `bbox_mAP_50` 및 `bbox_mAP_50_95` 개선
- DHM `last_state_counts.FN_CLS` 감소
- DHM `dominant_failure_counts.FN_CLS` 감소

부수적으로 기대할 수 있는 변화:

- `relapses` 감소
- `type_switches` 감소
- `state_changes` 감소
- hard GT의 `ema_margin` 증가

## Ablation 계획

| 실험 | 구성 | 목적 |
|---|---|---|
| Baseline | FCOS + DHM mining only | 기준선 |
| Positive support only | correct support attraction만 사용 | TP support prototype 효과 확인 |
| + confuser negative | recurring wrong-class prototype 추가 | confusion-specific contrastive 효과 확인 |
| + class-pair bank | instance memory와 class-pair memory 결합 | 일반화 효과 확인 |
| + margin loss | GT/confuser logit margin 추가 | decision boundary 보정 효과 확인 |
| no inference change | training-only branch 제거 후 평가 | representation transfer 확인 |

## 실패 가능성과 점검 항목

### Support prototype 오염

TP로 판정됐더라도 box나 score quality가 낮으면 support prototype이 noisy할 수 있다. `support_score_threshold`,
`support_iou_threshold`, quality-gated update가 필요하다.

### Confuser가 명확하지 않은 FN_CLS

모든 `FN_CLS`가 wrong-class confusion은 아니다. 정답 class score가 전반적으로 낮은 weak-class case도 있다.
이 경우 negative prototype을 강제로 만들면 잘못된 repulsion이 생길 수 있다. 명확한 wrong-class evidence가
없으면 positive attraction만 적용한다.

### Rare class over-separation

confuser repulsion을 너무 강하게 주면 비슷한 class 사이 feature가 과도하게 벌어져 generalization이 나빠질 수
있다. class-pair별 loss cap과 temperature tuning이 필요하다.

### Memory staleness

초기 epoch의 prototype이 오래 남으면 현재 모델 feature와 distribution mismatch가 생긴다. `max_age`,
EMA momentum, quality refresh rule을 둔다.

### Capacity 부족

작은 backbone에서는 prototype auxiliary loss를 흡수할 표현력이 부족할 수 있다. ResNet50, head width,
projection dimension ablation을 함께 본다.

## V0 구현 스케치

```text
for each DHM mining epoch:
  detections = model(train_images)
  features = collect_fpn_features(train_images)
  for each GT:
    state = DHM.assign_detection_state(GT, detections)
    if state == TP:
      z_pos = projector(roi_align(features, GT.box))
      memory.update_support(gt_uid, z_pos, score, iou)
    if state == FN_CLS:
      confuser = find_near_wrong_class_candidate(GT, detections)
      if confuser exists:
        z_neg = projector(roi_align(features, confuser.box))
        memory.update_confuser(gt_uid, confuser.class, z_neg, confuser.score)
      memory.update_confusion_stats(gt_uid, state, gt_score, confuser_score)

for each training batch:
  hard_records = memory.lookup_fn_cls_records(batch_gt)
  z_current = projector(roi_align(current_fpn_features, hard_gt_boxes))
  p_pos = memory.get_support(gt_uid)
  p_negs = memory.get_confusers(gt_uid)
  losses["dhmr_confusion"] = prototype_contrastive_loss(z_current, p_pos, p_negs)
  losses["dhmr_margin"] = temporal_confusion_margin_loss(logits, hard_records)
```

## 기본 설정 초안

```yaml
dhmr:
  enabled: true
  confusion_prototype_memory:
    enabled: true
    min_observations: 3
    min_fn_cls_count: 2
    min_instability: 0.25
    min_confuser_count: 1
    support_score_threshold: 0.3
    support_iou_threshold: 0.5
    prototype_dim: 128
    prototype_momentum: 0.8
    max_age: 10
    loss:
      proto_weight: 0.1
      margin_weight: 0.05
      support_weight: 0.05
      temperature: 0.2
      base_margin: 0.2
      warmup_epochs: 2
      max_gt_per_image: 16
```

## 한 줄 요약

Confusion Prototype Memory Module은 DHM이 찾아낸 `FN_CLS` hard GT에 대해 TP 시점의 correct
support prototype과 반복 wrong-class confuser prototype을 기억하고, training-only prototype contrastive
task로 class decision boundary를 보정하는 DHM-R의 classification repair path다.
