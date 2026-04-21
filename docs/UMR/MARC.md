# MARC — Miss-Aware Ranking Calibration

MARC(Miss-Aware Ranking Calibration)는 MDMB++가 기록한 hard GT를 대상으로 현재 detector candidate의 score ordering을 직접 보정하는 training-only 모듈이다. 목적은 hard GT 주변의 올바른 후보가 confuser, suppressor, local distractor보다 높은 rank를 갖도록 만드는 것이다.

현재 구현 범위는 FCOS다. Faster R-CNN과 DINO는 같은 개념을 proposal/query ranking loss로 확장할 수 있지만, v1에서는 config schema만 열어두고 비활성화한다.

## 1. Motivation

Missed detection은 candidate 자체가 없어서 발생할 수도 있지만, candidate가 있어도 최종 score/rank에서 밀려 발생할 수도 있다. 이 경우 Hard Replay나 FAAR만으로는 충분하지 않을 수 있다.

MARC는 아래 failure type을 주 대상으로 둔다.

- `score_suppression`: GT와 겹치는 후보가 있지만 score가 낮아 threshold/top-k에서 탈락
- `nms_suppression`: 올바른 후보가 더 높은 score의 주변 후보에 눌림
- `cls_confusion`: 위치는 맞지만 wrong-class score가 correct-class score보다 높음
- `loc_near_miss`: near candidate가 있지만 final detection으로 인정되지 않음

`candidate_missing`은 기본 대상에서 제외한다. ranking할 후보 자체가 없을 가능성이 높기 때문이다.

## 2. Objective

GT $g$에 대해 positive candidate $c^+$와 negative candidate 집합 $\mathcal{N}(g)$를 만든다. MARC는 $c^+$의 score가 negative보다 높아지도록 softmax ranking loss를 적용한다.

$$
L_{\mathrm{MARC}}(g)
=
-w(g)
\log
\frac{\exp(s(c^+) / \tau)}
{\exp(s(c^+) / \tau) + \sum_{c^- \in \mathcal{N}(g)} \exp(s(c^-) / \tau)}
$$

여기서 $s(c)$는 FCOS score와 정렬되는 log-score다.

$$
s(c)
=
\frac{1}{2}
\left(
\log \sigma(z_{\mathrm{cls}}(c))
+
\log \sigma(z_{\mathrm{ctr}}(c))
\right)
$$

severity weight는 아래처럼 계산한다.

$$
w(g)
=
\min
\left(
w_{\max},
1 + \alpha \cdot \mathrm{severity}(g)
\right)
$$

최종 loss는 아래와 같다.

$$
L
=
\lambda_{\mathrm{rank}}
\cdot
\mathrm{warmup}(t)
\cdot
\frac{1}{|\mathcal{G}_{rank}|}
\sum_{g \in \mathcal{G}_{rank}}
L_{\mathrm{MARC}}(g)
$$

## 3. Data Flow

MARC runtime 흐름은 아래와 같다.

```text
MDMB++ unresolved entries
  -> MissAwareRankingCalibration.plan()
  -> RankingPlan / RankingTarget
  -> MissAwareRankingCalibration.compute_loss()
  -> FCOS loss dict["marc"]
```

FCOS forward 안에서는 FAAR 이후, Candidate Densification 이전에 MARC loss가 계산된다.

```text
base assignment
  -> FAAR assignment repair
  -> base FCOS loss
  -> MARC ranking loss
  -> Candidate Densification auxiliary loss
```

MARC는 assignment를 직접 바꾸지 않는다. inference path도 변경하지 않는다.

## 4. Candidate Mining

MARC는 MDMB++ entry와 현재 transformed target을 class + normalized bbox IoU로 매칭한다. 매칭 기준은 `record_match_threshold`다.

Positive candidate는 GT class score 기준으로 선택한다.

1. GT IoU가 `positive_iou_threshold` 이상인 후보 중 GT class rank score가 가장 높은 후보를 선택한다.
2. 없고 `allow_near_positive_fallback: true`이면 IoU가 `near_positive_iou_threshold` 이상인 후보 중 IoU가 가장 높은 후보를 선택한다.
3. 그래도 없으면 해당 GT는 skip하고 `skipped_no_positive`를 증가시킨다.

Negative candidate는 GT 주변 region 안에서 선택한다.

- Wrong-class confuser: GT class가 아닌 후보
- Same-class suppressor: GT class 후보지만 positive 후보보다 IoU가 `same_class_iou_gap` 이상 낮은 후보
- High-score local distractor: GT 주변 region 안에 있는 wrong-class 후보 중 score가 높은 후보

최종 negative는 score 기준 top-k로 자르며 개수는 `max_negatives_per_gt`가 제한한다.

## 5. Config

MARC는 `modules/cfg/marc.yaml`에서 설정한다.

```yaml
enabled: false
warmup_epochs: 2
lambda_rank: 0.05
temperature: 0.1
min_severity: 1.0
record_match_threshold: 0.95
max_rank_targets_per_image: 5
max_negatives_per_gt: 8
positive_iou_threshold: 0.3
near_positive_iou_threshold: 0.1
confuser_iou_threshold: 0.3
same_class_iou_gap: 0.1
region_scale: 1.5
allow_near_positive_fallback: true
severity_weight_scale: 0.25
max_target_weight: 3.0
failure_types:
  - score_suppression
  - nms_suppression
  - cls_confusion
  - loc_near_miss
models:
  fcos: {}
  fasterrcnn:
    enabled: false
  dino:
    enabled: false
```

초기 실험은 `lambda_rank: 0.05`, `temperature: 0.1`, `max_negatives_per_gt: 8`을 권장한다. false positive가 늘면 `lambda_rank`, `region_scale`, `max_negatives_per_gt`를 먼저 낮춘다.

## 6. Summary Metrics

`engine.fit()`은 epoch record에 `marc` summary를 저장한다.

- `rank_targets`: ranking 대상으로 선택된 GT 수
- `rank_losses`: 실제 ranking loss가 계산된 GT 수
- `rank_negatives`: 사용된 negative 후보 수
- `mean_rank_loss`: epoch 평균 MARC loss
- `by_failure_score_suppression`: score suppression target 수
- `by_failure_nms_suppression`: NMS suppression target 수
- `by_failure_cls_confusion`: class confusion target 수
- `by_failure_loc_near_miss`: localization near miss target 수
- `skipped_no_entry_match`: MDMB++ entry와 현재 target이 매칭되지 않은 수
- `skipped_no_positive`: positive candidate가 없어 skip한 수
- `skipped_no_negative`: negative candidate가 없어 skip한 수

## 7. Experiment Order

MARC 단독 효과와 조합 효과는 아래 순서로 보는 것이 해석하기 좋다.

1. baseline
2. MDMB++ only
3. MDMB++ + MARC
4. MDMB++ + FAAR
5. MDMB++ + FAAR + MARC
6. MDMB++ + Hard Replay + FAAR + MARC

핵심 비교는 `MDMB++ + MARC`와 `MDMB++ + FAAR + MARC`다. FAAR는 hard GT에 positive assignment를 만들고, MARC는 그 후보가 최종 ranking에서 살아남도록 보정한다.
