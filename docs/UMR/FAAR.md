# FAAR — Failure-Aware Assignment Repair

FAAR(Failure-Aware Assignment Repair)는 MDMB++가 기록한 hard GT를 이용해 detector의 positive assignment를 직접 보정하는 training-only recovery 모듈이다. 목적은 loss weight를 키우는 것이 아니라, 원래 background로 남았던 point/proposal/query가 hard GT의 positive supervision을 받도록 만드는 것이다.

현재 구현 범위는 FCOS다. Faster R-CNN과 DINO는 같은 개념을 proposal/query assignment repair로 확장할 수 있지만, v1에서는 config schema만 열어두고 비활성화한다.

## 1. Motivation

Missed detection은 항상 loss weight 부족으로만 발생하지 않는다. 특히 FCOS류 dense detector에서는 hard GT 주변에 충분한 positive point가 생기지 않으면, 이후 classification/regression loss를 아무리 키워도 해당 GT가 직접 학습되지 않는다.

기본 FCOS assignment는 아래 조건을 모두 만족해야 positive가 된다.

$$
p \in \mathcal{P}^{+}(g)
\iff
p \in \text{center\_region}(g)
\land
p \in \text{box}(g)
\land
s(p) \in \text{scale\_range}(g)
$$

FAAR는 MDMB++가 hard GT $g$를 보고하면, 제한된 수의 point를 추가 positive로 repair한다.

$$
\tilde{a}_p =
\begin{cases}
g & \text{if } p \in \mathcal{R}(g) \text{ and } p \text{ is selected by FAAR} \\
a_p & \text{otherwise}
\end{cases}
$$

여기서 $a_p$는 기본 FCOS assignment, $\tilde{a}_p$는 repair 이후 assignment다.

## 2. Difference from Candidate Densification

Candidate Densification은 hard GT 주변 point에 별도 auxiliary loss `candidate_dense`를 추가한다. 반면 FAAR는 main FCOS assignment 자체를 수정한다.

| Module | Intervention | Loss key | Main assignment 변경 |
|---|---|---:|---:|
| Candidate Densification | hard GT 주변 auxiliary positive point 추가 | `candidate_dense` | no |
| FAAR | `matched_idxs` 직접 보정 | 없음 | yes |

두 모듈을 같이 켜면 FAAR가 먼저 실행된다. 그 뒤 Candidate Densification은 repair된 assignment를 보고 남은 unassigned point를 대상으로 auxiliary supervision을 만든다.

## 3. Data Flow

FAAR의 runtime 흐름은 아래와 같다.

```text
MDMB++ unresolved entries
  -> FailureAwareAssignmentRepair.plan()
  -> RepairPlan / RepairTarget
  -> FCOS _repair_fcos_assignments()
  -> repaired matched_idxs
  -> standard FCOS loss
```

`RepairTarget`은 현재 batch target과 MDMB++ entry를 class + normalized bbox IoU로 매칭해서 만든다. 매칭 기준은 `record_match_threshold`다.

## 4. Target Selection

FAAR는 `mdmbpp.get_dense_targets(image_id)`를 읽고 아래 조건을 만족하는 entry만 repair 대상으로 사용한다.

$$
\text{severity}(g) \ge \tau_{\text{severity}}
\land
\text{failure\_type}(g) \in \mathcal{F}_{\text{FAAR}}
$$

기본 failure type은 아래 네 가지다.

- `candidate_missing`: GT 주변에 충분한 detector candidate가 없는 경우
- `loc_near_miss`: localization이 근접하지만 final detection으로 인정되지 않은 경우
- `score_suppression`: 후보는 있으나 score/ranking에서 밀린 경우
- `nms_suppression`: NMS/overlap으로 최종 detection에서 제거된 경우

`cls_confusion`은 기본값에서 제외한다. 이 경우는 assignment 부족보다 class discrimination 문제가 더 클 수 있어 ranking calibration이나 contrastive distillation 쪽이 더 직접적이다.

## 5. Repair Budget

GT별 repair point 수는 severity와 relapse 여부로 정한다.

$$
B(g) =
\min
\left(
B_{\max},
B_0 + \lfloor \alpha \cdot \text{severity}(g) \rfloor + \mathbb{1}_{relapse}(g) B_r
\right)
$$

현재 config 필드는 아래와 대응된다.

- $B_0$: `base_repair_points`
- $\alpha$: `severity_budget_scale`
- $B_{\max}$: `max_repair_points_per_gt`
- $B_r$: `relapse_budget_bonus`

## 6. FCOS Repair Policy

FCOS 구현은 기본 `_match_anchors_to_targets()` 실행 직후 동작한다.

1. hard GT 중심과 크기로 repair region을 만든다.
2. 기본값에서는 `matched_idxs == -1`인 background point만 후보로 둔다.
3. `respect_fcos_scale_range: true`이면 FCOS scale range를 먼저 존중한다.
4. 후보가 없고 `allow_adjacent_levels: true`이면 인접 FPN level까지 완화한다.
5. 그래도 후보가 없고 `allow_nearest_center_fallback: true`이면 assignment 조건을 만족하는 가장 가까운 point를 고른다.
6. 선택된 point의 `matched_idxs`를 hard GT index로 바꾼다.

기본값에서는 기존 positive assignment를 빼앗지 않는다. `allow_positive_reassignment: true`를 켜면 기존 positive point 재할당을 허용하지만, 기존 GT와 target hard GT의 IoU가 `protect_existing_positive_iou` 이상이면 보호한다.

## 7. Config

FAAR는 `modules/cfg/faar.yaml`에서 설정한다.

```yaml
enabled: false
warmup_epochs: 2
min_severity: 1.0
record_match_threshold: 0.95
max_repair_targets_per_image: 5
base_repair_points: 2
severity_budget_scale: 1.0
max_repair_points_per_gt: 8
base_region_scale: 1.0
severity_region_scale: 0.1
max_region_scale: 1.5
require_unassigned_points: true
allow_positive_reassignment: false
protect_existing_positive_iou: 0.3
respect_fcos_scale_range: true
allow_adjacent_levels: true
allow_nearest_center_fallback: true
include_relapse: true
relapse_budget_bonus: 1
failure_types:
  - candidate_missing
  - loc_near_miss
  - score_suppression
  - nms_suppression
models:
  fcos: {}
  fasterrcnn:
    enabled: false
  dino:
    enabled: false
```

초기 실험은 `enabled: true`, `require_unassigned_points: true`, `allow_positive_reassignment: false`를 권장한다. mAP가 떨어지거나 false positive가 늘면 `max_repair_points_per_gt`, `severity_budget_scale`, `allow_nearest_center_fallback`을 낮추는 쪽이 먼저다.

## 8. Summary Metrics

`engine.fit()`은 epoch record에 `faar` summary를 저장한다. 주요 지표는 아래와 같다.

- `repair_targets`: repair 대상으로 선택된 GT 수
- `repair_points`: 실제 positive로 바뀐 point 수
- `repair_images`: repair 대상이 있었던 이미지 수
- `mean_severity`: repair target의 평균 severity
- `by_failure_*`: failure type별 repair target 수
- `skipped_no_entry_match`: MDMB++ entry와 현재 target이 매칭되지 않은 수
- `skipped_no_candidate_points`: repair할 point가 없어 건너뛴 target 수
- `skipped_existing_positive`: 기존 positive 보호 때문에 제외된 point 수

동일 값은 `faar_*` prefix alias로도 저장된다.

## 9. Expected Ablations

FAAR의 효과를 해석하려면 아래 순서로 실험한다.

1. baseline
2. MDMB++ only
3. Candidate Densification only
4. FAAR only
5. FAAR + Candidate Densification
6. FAAR + Hard Replay
7. FAAR + Hard Replay + Candidate Densification

핵심 비교는 Candidate Densification only와 FAAR only다. Candidate Densification이 auxiliary supervision 효과라면, FAAR는 main assignment bottleneck을 직접 고치는 효과를 측정한다.
