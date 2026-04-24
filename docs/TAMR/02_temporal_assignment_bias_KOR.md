# Temporal Assignment Bias

## 가설

일부 GT는 충분히 유용한 positive training location을 배정받지 못하거나, 배정된 positive의 loss 영향력이
너무 약해서 반복적으로 실패할 수 있다. Static 또는 current-batch assignment는 어떤 GT가 과거에 반복적으로
relapse했는지 알지 못한다.

Temporal Assignment Bias는 GT의 historical risk를 이용해 training 중 assignment와 weighting을 약하게
bias한다.

## 방법

각 GT에 대해 Lightweight Temporal Failure Memory에서 `risk(gt)`를 읽는다. 이 risk로 하나 이상의 training
knob을 조정한다.

```text
assignment_radius(gt) = base_radius * (1 + alpha * risk(gt))
positive_topk(gt)    = base_topk + round(beta * risk(gt))
loss_weight(gt)      = 1 + gamma * risk(gt)
```

가장 덜 침습적인 loss reweighting부터 시작한다. Assignment expansion은 weight만으로 부족하다는 것이 확인된
뒤 추가하는 것이 안전하다.

현재 구현 상태:

- 구현됨: FCOS classification, box regression, centerness loss에 대한 risk-gated positive-point
  loss weighting.
- 아직 미구현: assignment radius expansion, backup positive, positive top-k 변경.
- Config: `modules/cfg/tfm.yaml`의 `assignment_bias` 섹션.

## FCOS Variant

FCOS에서는 기존 positive-location path에 risk를 적용한다.

- High-risk GT에 배정된 positive의 classification weight를 높인다.
- High-risk GT의 regression 및 centerness weight를 높인다.
- 선택적으로 high-risk GT의 center sampling radius를 확장한다.
- 선택적으로 near-center backup positive를 소량 허용한다.

Backup positive는 ambiguous sample이 batch를 지배하지 않도록 반드시 cap을 둔다.

## Loss Form

`G+`를 특정 GT에 배정된 positive location이라고 하자.

```text
L_tabi = sum_{g in GT} (1 + gamma * risk_g) * L_det(G+_g)
```

Classification과 localization에는 별도 cap을 둔다.

```text
w_cls = clamp(1 + gamma_cls * risk, 1, w_cls_max)
w_box = clamp(1 + gamma_box * risk, 1, w_box_max)
```

초기 권장 cap:

```text
w_cls_max = 2.0
w_box_max = 2.0
```

## Novelty

Focal Loss와 OHEM과 달리 weight가 현재 loss나 confidence만의 함수가 아니다. Repeated miss, recovery,
relapse를 포함한 per-instance temporal history에 의해 condition된다.

ATSS/OTA/PAA와도 다르다. Assignment가 current forward statistics만으로 다시 계산되는 것이 아니라,
historical failure risk가 prior로 들어간다.

## Metrics

추적할 metric:

- false-negative recovery rate
- relapse count
- mean miss streak
- high-risk GT 수
- high-risk GT당 positive location 수
- AP_small 및 per-class AP
- training time delta

## Risks

- Overweighting은 잘못된 annotation을 증폭할 수 있다.
- Radius를 너무 크게 확장하면 localization이 나빠질 수 있다.
- Risk decay가 느리면 이미 회복된 GT에 계속 과도하게 집중할 수 있다.

완화책:

- risk와 weight를 clip한다.
- recovery 이후 risk를 decay한다.
- 반복 실패가 확인된 뒤에만 assignment expansion을 적용한다.
