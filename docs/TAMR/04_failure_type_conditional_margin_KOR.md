# Failure-Type Conditional Margin

## 가설

모든 miss가 같은 원인으로 발생하지 않는다. Classification confusion, localization near miss,
score suppression, assignment miss는 같은 correction signal을 받아서는 안 된다.

Failure-Type Conditional Margin은 기억된 failure type에 따라 각 high-risk GT에 targeted auxiliary loss를
선택한다.

## Failure Types

MDMB++와 맞추되, training-forward signal로 계산 가능한 lightweight taxonomy를 사용한다.

```text
missing_assignment
classification_confusion
localization_weak
score_weak
quality_ranking_weak
recovered
```

정확한 diagnosis가 불가능하면 가능한 coarse type만 저장한다. 이 방법은 완벽한 post-NMS attribution에 의존하지 않는다.

## Conditional Corrections

### Classification Confusion

GT class와 가장 강한 competing class 사이 margin을 키운다.

```text
L_cls_margin = risk * max(0, m_cls - logit_gt + logit_confuser)
```

GT에 배정된 positive에만 사용한다.

### Localization Weak

해당 GT의 positive에 localization pressure를 높인다.

```text
L_loc_margin = risk * max(0, m_iou - IoU(pred_box, gt_box))
```

Cap을 두고, reasonable positive location에만 적용해야 한다.

### Score Weak

GT class confidence의 하한을 높인다.

```text
L_score_margin = risk * max(0, m_score - p_gt)
```

Localization은 괜찮지만 confidence가 낮은 경우에 유용하다.

### Quality Ranking Weak

가장 좋은 GT positive가 주변 low-quality positive보다 높은 score를 갖도록 유도한다.

```text
L_rank = risk * max(0, m_rank - score_best_gt + score_neighbor)
```

Dense detector는 이미 correlated positive가 많기 때문에 신중하게 사용해야 한다.

### Missing Assignment

처음부터 margin을 추가하지 않는다. Assignment expansion 또는 backup positive가 더 안전하다.

```text
center_radius <- center_radius * (1 + alpha * risk)
```

그 다음 새로 허용된 positive에 standard detection loss를 적용한다.

## Loss Composition

Step마다 GT당 하나의 conditional loss만 사용한다.

```text
L_ftcm = sum_g risk_g * L_condition(last_failure_type_g)
```

초기 weight는 작게 둔다.

```text
lambda_ftcm = 0.05 to 0.2
```

## Novelty

이 방법은 hard-example weighting을 넘어선다. Model은 failure mode를 기억하고, 이후 다른 corrective pressure를
적용한다. 즉 generic mining보다 temporal diagnosis-driven training에 가깝다.

## Metrics

Failure-type transition matrix를 추적한다.

```text
classification_confusion -> recovered
localization_weak -> recovered
score_weak -> recovered
missing_assignment -> recovered
recovered -> relapse
```

이 transition은 AP만 보는 것보다 훨씬 diagnostic하다.

## Risks

- Failure typing이 틀리면 잘못된 margin을 적용할 수 있다.
- Auxiliary loss가 많아지면 training이 불안정해질 수 있다.
- Margin loss를 과하게 주면 calibration이 나빠질 수 있다.

완화책:

- GT당 dominant failure type 하나만 사용한다.
- Base detector가 안정화된 뒤 warmup 이후 적용한다.
- Risk와 auxiliary loss를 clip한다.
- Failure type correction을 각각 독립적으로 ablation한다.

