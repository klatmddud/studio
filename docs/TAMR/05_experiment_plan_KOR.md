# TAMR Experiment Plan

## 핵심 질문

1. Temporal GT history가 Hard Replay 없이 detection을 개선할 수 있는가?
2. TAMR이 MDMB++보다 낮은 overhead로 relapse와 repeated miss를 줄일 수 있는가?
3. 가장 크게 기여하는 컴포넌트는 risk weighting, assignment bias, prototype distillation,
   failure-type margin 중 무엇인가?
4. TAMR은 Hard Replay와 additive하게 결합되는가?

## Baselines

아래 순서로 실행한다.

```text
Baseline
Baseline + current MDMB++ only
Baseline + Hard Replay only
Baseline + TAMR memory only
Baseline + TAMR assignment bias
Baseline + TAMR assignment bias + prototype distillation
Baseline + full TAMR
Baseline + Hard Replay + best TAMR variant
```

## Required Metrics

Detection:

```text
mAP
AP50
AP75
AP_small
AP_medium
AP_large
per-class AP
```

Temporal failure:

```text
num_high_risk_gt
mean_risk
mean_miss_streak
max_miss_streak
total_relapse
relapse_this_epoch
recovery_rate_last_epoch
failure_type_counts
failure_type_transition_matrix
```

Efficiency:

```text
train_seconds_per_epoch
images_per_second
GPU memory peak
checkpoint_size
memory_state_size
DDP sync time if distributed
```

## 최소 첫 구현

가장 저렴한 경로부터 시작한다.

1. Lightweight per-GT state를 추가한다.
2. Current training-forward statistics에서 GT risk를 계산한다.
3. Risk-gated loss reweighting만 적용한다.
4. Temporal metrics를 기록한다.

Risk signal이 실제로 유용하다는 것이 확인되기 전까지 prototype이나 conditional margin은 구현하지 않는다.

## Acceptance Criteria

TAMR variant는 아래 조건을 만족하면 유지할 가치가 있다.

```text
training_time <= baseline_time * 1.15
inference_time == baseline_time
checkpoint_size increase is acceptable
relapse count decreases
mean miss streak decreases
mAP or AP_small improves
```

Research value 관점에서는 mAP gain이 작더라도 temporal-metric 개선이 명확하면 우선순위를 둘 수 있다.
논문 스토리는 aggregate AP만이 아니라 recurrent false negative 감소를 중심으로 만들 수 있다.

## 주의해야 할 Negative Results

- mAP가 Hard Replay와 결합했을 때만 개선된다.
- Risk weighting이 recall은 올리지만 AP75를 낮춘다.
- Prototype loss가 small object에는 도움이 되지만 class confusion을 악화한다.
- Failure-type margin은 post-NMS diagnosis 없이 너무 noisy하다.

이런 결과도 유용하다. Temporal memory 중 어떤 부분이 실제로 가치 있는지 정의해주기 때문이다.

