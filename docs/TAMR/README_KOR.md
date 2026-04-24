# TAMR - Temporal Anti-Miss Regularization

TAMR은 객체 검출에서 반복적으로 발생하는 false negative를 줄이기 위한 후보 연구 방향이다.
핵심 목표는 inference-time 비용을 추가하지 않으면서, MDMB++의 장점인 GT별 temporal memory를
가볍게 유지하는 것이다. 현재 MDMB++처럼 post-step inference와 dense candidate summary를 저장하는
방식은 피한다.

## 동기

현재 UMR stack에서는 Hard Replay가 가장 강한 실용 축으로 보인다. 반면 MDMB++와 RASD는
diagnostic signal과 temporal support signal은 제공하지만 training-time overhead가 크다.
특히 MDMB++는 optimizer step 이후 추가 inference를 수행하고, FCOS dense candidate를 훑어
structured failure context를 만들기 때문에 비용이 커진다.

TAMR은 memory module을 lightweight training prior로 재정의한다.

- compact per-GT temporal state만 유지한다.
- normal training forward에서 이미 계산된 signal만 읽는다.
- auxiliary loss는 training 중에만 적용한다.
- inference에서는 모든 branch를 제거한다.

## 설계 제약

- Inference-time overhead가 없어야 한다.
- Training overhead가 낮아야 한다. 추가 detector forward, dense candidate summary pass를 사용하지 않는다.
- Per-GT state는 checkpoint와 DDP synchronization에 부담이 작아야 한다.
- Hard Replay와 결합 가능해야 하지만, Hard Replay 없이도 독립 방법론으로 성립해야 한다.
- Novelty는 단순 hard-sample reweighting이 아니라 temporal failure state에서 나와야 한다.

## 제안 컴포넌트

TAMR은 하나의 단일 기법보다 여러 컴포넌트를 독립적으로 ablation할 수 있는 방법론 군으로 보는 것이 좋다.

1. Lightweight Temporal Failure Memory
   - GT별 miss/recovery/relapse state와 optional support prototype을 compact하게 저장한다.
   - MDMB++의 dense `topk_candidates`를 scalar risk state로 대체한다.

2. Temporal Assignment Bias
   - 과거에 어렵게 학습된 GT의 temporal risk를 이용해 positive assignment나 loss weight를 조정한다.
   - 전체 image distribution을 replay하지 않고 repeated miss를 직접 겨냥한다.

3. Anti-Relapse Prototype Distillation
   - 과거 successful detection에서 compact feature prototype을 저장한다.
   - 현재 positive GT feature가 안정적인 historical prototype에서 멀어지지 않도록 유도한다.

4. Failure-Type Conditional Margin
   - 기억된 failure type에 따라 targeted correction loss를 선택한다.
   - 모든 failure에 동일한 hard weight를 주지 않고, 실패 원인별로 다른 supervision을 준다.

## Related Work 대비 포지셔닝

기존 hard-mining 및 assignment 방법들은 대부분 current mini-batch 또는 current forward pass 중심이다.

- OHEM은 hard example을 online으로 mining한다.
- Focal Loss는 easy example의 weight를 낮춘다.
- Libra R-CNN과 PISA는 sample/objective importance를 rebalance한다.
- ATSS, OTA, PAA는 현재 statistics 또는 cost를 기반으로 assignment를 조정한다.
- GFL은 dense detection의 quality 및 localization representation을 개선한다.

TAMR의 의도된 novelty는 longitudinal하다. 각 GT가 training history를 가지고, 그 history가 이후
epoch의 assignment, weighting, margin, feature regularization을 condition한다.

## 권장 Ablation 순서

1. Baseline detector.
2. Baseline + Lightweight Temporal Failure Memory only.
3. Temporal Assignment Bias 추가.
4. Anti-Relapse Prototype Distillation 추가.
5. Failure-Type Conditional Margin 추가.
6. TAMR과 Hard Replay 결합.

핵심 비교 대상은 final AP만이 아니다. False-negative recovery, relapse count, per-class miss streak,
AP_small/AP_medium/AP_large, training time, checkpoint state size를 함께 추적해야 한다.

## 파일

- [01_lightweight_temporal_failure_memory_KOR.md](01_lightweight_temporal_failure_memory_KOR.md)
- [02_temporal_assignment_bias_KOR.md](02_temporal_assignment_bias_KOR.md)
- [03_anti_relapse_prototype_distillation_KOR.md](03_anti_relapse_prototype_distillation_KOR.md)
- [04_failure_type_conditional_margin_KOR.md](04_failure_type_conditional_margin_KOR.md)
- [05_experiment_plan_KOR.md](05_experiment_plan_KOR.md)

