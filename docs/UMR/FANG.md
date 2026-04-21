# FANG — Failure-Aware Negative Gradient Shielding

FANG는 MDMB++가 기록한 unresolved hard GT 주변에서 harmful negative supervision을 줄이는 training-only 모듈이다. 목적은 hard GT 근처 point가 positive assignment를 받지 못한 상태에서 해당 객체의 true class logit을 계속 낮추는 자기강화 루프를 완화하는 것이다.

현재 구현 범위는 FCOS다. Faster R-CNN과 DINO는 proposal/query 단위 negative shielding으로 확장할 수 있지만 v1에서는 비활성화한다.

## 1. Difference from FAAR and MARC

| Method | Intervention | Adds loss | Changes assignment |
|---|---|:---:|:---:|
| FAAR | hard GT에 positive assignment 추가 | no | yes |
| MARC | hard GT candidate ranking 보정 | yes | no |
| FANG | hard GT 주변 true-class negative loss 완화 | no | no |

FANG는 positive를 만들지 않는다. 대신 `matched_idxs < 0`인 point 중 MDMB++ hard GT 주변에 있는 point의 class-wise focal loss weight만 낮춘다.

## 2. Runtime Flow

```text
base FCOS assignment
  -> optional FAAR assignment repair
  -> FANG shield plan from MDMB++ entries
  -> class-wise focal loss weighting
  -> optional MARC ranking loss
  -> optional Candidate Densification loss
```

FAAR 이후에 실행되므로 FAAR가 positive로 repair한 point는 FANG 대상에서 제외된다.

## 3. Shield Policy

FANG는 `mdmbpp.get_dense_targets(image_id)`에서 아래 조건을 만족하는 entry를 읽는다.

```text
failure_type in failure_types
severity >= min_severity
```

각 entry는 현재 transformed target과 class + normalized bbox IoU로 매칭한다. 매칭된 GT 주변 확장 region에서 `matched_idxs < 0`인 negative point만 선택하고, 가까운 point부터 `max_shield_points_per_gt`개까지 사용한다.

선택된 point `p`와 GT class `c_g`에 대해 FANG는 focal loss weight를 낮춘다.

```text
weight[p, c_g] = min(current_weight, shield_weight)
```

다른 class column, bbox regression, centerness loss는 변경하지 않는다.

## 4. Weighting

FANG strength는 warmup을 거친 `lambda_fang`이다.

```text
strength = lambda_fang * warmup_factor
target_weight = min(max_target_weight, 1 + severity_weight_scale * severity)
distance_factor = 1 - normalized_center_distance
shield_weight = clamp(1 - strength * target_weight * distance_factor,
                      min_negative_weight,
                      1)
```

동일 point/class에 여러 shield target이 겹치면 가장 낮은 weight를 사용한다.

## 5. Config

```yaml
enabled: false
warmup_epochs: 2
lambda_fang: 0.5
min_negative_weight: 0.25
min_severity: 1.0
record_match_threshold: 0.95
max_shield_targets_per_image: 5
max_shield_points_per_gt: 16
base_region_scale: 1.0
severity_region_scale: 0.1
max_region_scale: 1.5
severity_weight_scale: 0.25
max_target_weight: 3.0
failure_types:
  - candidate_missing
  - loc_near_miss
  - score_suppression
models:
  fcos: {}
  fasterrcnn:
    enabled: false
  dino:
    enabled: false
```

초기 실험은 `candidate_missing`, `loc_near_miss`, `score_suppression`만 켜고 `min_negative_weight`를 0.25 이상으로 유지하는 것을 권장한다. false positive가 늘면 `lambda_fang`, `max_shield_points_per_gt`, `max_region_scale`을 낮춘다.

## 6. Metrics

`engine.fit()`은 epoch record에 `fang` summary를 저장한다.

- `shield_targets`: shield plan에 포함된 GT 수
- `shield_points`: 실제 class-wise shielding이 적용된 point 수
- `shield_batches`: shield point가 있었던 batch 수
- `mean_shield_weight`: 적용된 shield weight 평균
- `lambda_fang`: warmup이 반영된 현재 strength
- `by_failure_*`: failure type별 target 수
- `skipped_no_entry_match`: 현재 target과 매칭되지 않은 MDMB++ entry 수
- `skipped_no_candidate_points`: shield 가능한 negative point가 없었던 target 수

## 7. Ablation

권장 순서:

1. Baseline
2. MDMB++ only
3. FANG only
4. FANG + Candidate Densification
5. FANG + FAAR
6. FANG + FAAR + MARC

핵심 비교는 `MDMB++ only`와 `FANG only`다. FANG 단독 개선이 있으면 hard GT 주변 negative classification pressure가 실제 bottleneck이었다는 근거가 된다.
