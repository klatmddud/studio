# HLRT: Hysteretic Localization Residual Transport

## 개요

HLRT는 DHM이 `FN_LOC`로 판정한 hard GT를 줄이기 위한 localization repair 방법론이다.

`FN_LOC`는 detector가 GT의 class identity는 어느 정도 알고 있지만, 같은 class 후보의 box alignment가 `tau_iou`를 넘지 못해 TP가 되지 못하는 경우다. HLRT는 이 GT가 반복적으로 어느 box edge에서 실패하는지 기억하고, 그 residual을 다음 학습의 hard localization supervision으로 운반한다.

핵심은 다음과 같다.

> 반복 localization 실패 residual을 GT별 memory에 저장하고, residual replay와 side-aware native box supervision으로 다시 학습시킨다.

## 현재 구현 범위

현재 코드베이스에서 DHM-R은 HLRT 전용 모듈이다. 기존 보조 브랜치는 제거되었고, 별도의 edge auxiliary loss도 더 이상 생성하지 않는다.

구현 위치:

- `modules/nn/dhmr.py`: HLRT config, residual memory, replay, loss hook helper
- `models/detection/wrapper/fcos.py`: FCOS assignment/loss hook 연결
- `modules/cfg/dhmr.yaml`: HLRT 요소별 on/off config

현재 FCOS에 연결된 요소는 다음과 같다.

- `residual_memory`: `FN_LOC` GT별 edge residual EMA와 squared EMA 저장
- `residual_replay`: 저장된 residual로 replay box를 만들고 FCOS positive point 추가
- `iou_loss_weighting`: 기존 GIoU bbox loss weight를 hard GT instability로 증폭
- `side_aware_loss`: l/t/r/b encoded distance에 side-aware Smooth-L1 loss 추가
- `quality_gate`: FCOS centerness target을 GT EMA IoU와 blend

## 문제 정의

일반적인 detector의 localization loss는 현재 mini-batch의 prediction과 GT box만 본다. 이 방식은 반복적으로 실패하는 GT의 과거 이력을 직접 활용하지 못한다.

HLRT가 다루는 문제는 다음과 같다.

- 특정 GT가 여러 epoch 동안 계속 `FN_LOC`로 남는다.
- class score는 충분하지만 box quality가 낮아 NMS나 ranking에서 밀린다.
- left, top, right, bottom 중 특정 edge residual이 반복된다.
- 작은 객체, 가려진 객체, 길쭉한 객체에서 boundary가 일관되게 어긋난다.

HLRT는 현재 상태만 보지 않고, GT별 hysteresis memory를 통해 과거 실패 방향까지 반영한다.

## GT별 Residual Memory

`FN_LOC` GT `g`에 대해 현재 positive point들의 prediction box와 GT box 차이를 side별 normalized residual로 계산한다.

$$
r_g^t =
\left[
\frac{x_1^g-x_1^p}{w_g},
\frac{y_1^g-y_1^p}{h_g},
\frac{x_2^g-x_2^p}{w_g},
\frac{y_2^g-y_2^p}{h_g}
\right]
$$

이 residual은 left, top, right, bottom edge가 각각 얼마나 어긋났는지를 나타낸다. DHM-R은 GT별로 다음 값을 저장한다.

- `edge_error_ema`: signed residual EMA
- `edge_abs_ema`: absolute residual EMA
- `edge_sq_ema`: residual squared EMA
- `observations`: residual 관측 횟수

EMA 업데이트는 다음 형태다.

$$
\bar{r}_g^t = \alpha \bar{r}_g^{t-1} + (1-\alpha) r_g^t
$$

## Residual Replay

Residual replay는 과거 실패 residual로 GT 주변의 hard localization box를 재현한 뒤, 그 box 주변 point를 FCOS positive assignment에 추가하는 방식이다.

Replay box는 GT box에서 residual 방향을 반대로 적용해 만든다.

$$
\tilde{b}_{g,k} = T^{-1}(b_g, -\tilde{r}_{g,k})
$$

사용되는 replay 후보는 기본적으로 다음 범위를 목표로 한다.

$$
\tau_{near} \le IoU(\tilde{b}_{g,k}, b_g) < \tau_{iou}
$$

즉 GT와 너무 멀지 않지만 TP 기준에는 아직 못 미치는 localization 영역을 다시 학습시킨다. 구현에서는 `max_points_per_gt`, `max_points_per_batch`로 추가 positive 수를 제한한다.

중요한 점은 residual replay가 dataset을 늘리지 않는다는 것이다.

- 이미지 수 증가 없음
- GT annotation 증가 없음
- training-time positive point 후보만 증가

## HLRT Loss

HLRT의 loss는 기존 FCOS loss를 대체하지 않고, native loss에 hook을 추가한다.

### IoU Loss Weighting

기존 GIoU 기반 bbox regression loss에 hard GT instability 기반 weight를 곱한다.

$$
\mathcal{L}_{iou}^{HLRT}
= (1+\gamma E_{loc}(g))\mathcal{L}_{GIoU}
$$

여기서 `E_loc(g)`는 DHM의 instability score다.

### Side-Aware Loss

반복적으로 틀리는 side에 더 큰 weight를 주는 l/t/r/b distance loss를 추가한다.

$$
\mathcal{L}_{side}
=
\sum_{j \in \{l,t,r,b\}}
(1+\eta d_{g,j})\mathcal{L}_{side,j}
$$

현재 FCOS 구현에서는 encoded l/t/r/b regression target과 prediction 사이의 Smooth-L1 loss로 계산하며, `dhmr_hlrt_side`라는 loss key로 기록된다.

### Quality Gate

`FN_LOC`에서는 class가 틀렸다기보다 localization quality가 낮은 경우가 많다. 따라서 classification을 과하게 벌주기보다 FCOS centerness target을 GT의 EMA IoU와 blend한다.

$$
q_g^t = EMA(IoU(\hat{b}_g^t, b_g))
$$

이 값은 `quality_gate.blend`, `min_quality`, `max_quality`로 제어한다.

## Config

각 요소는 `modules/cfg/dhmr.yaml`에서 독립적으로 켜고 끌 수 있다.

```yaml
enabled: true

hlrt:
  enabled: true
  residual_memory:
    enabled: true
  residual_replay:
    enabled: true
  iou_loss_weighting:
    enabled: true
  side_aware_loss:
    enabled: true
  quality_gate:
    enabled: true
```

권장 ablation 순서는 다음과 같다.

1. `residual_memory` only
2. `residual_memory + iou_loss_weighting`
3. `+ side_aware_loss`
4. `+ quality_gate`
5. `+ residual_replay`

## 기대 효과와 확인 지표

HLRT가 잘 작동하면 AP50보다 AP75에서 먼저 개선이 나타나는 것이 자연스럽다. AP50만 오르고 AP75가 떨어지면 coarse detection만 늘고 box precision은 악화된 것이므로 HLRT가 제대로 작동했다고 보기 어렵다.

주요 지표:

- `FN_LOC` count
- `FN_LOC -> TP` transition
- AP75
- mAP 50:95
- AR100
- `dhmr.hlrt.hlrt_memory_updates`
- `dhmr.hlrt.hlrt_replay_points`
- `dhmr_hlrt_side`

## 요약

HLRT는 반복적으로 localization에 실패하는 GT의 residual을 기억하고, 그 residual을 training-time assignment와 native bbox supervision에 다시 주입하는 방법론이다. 현재 DHM-R은 HLRT만 유지하며, 별도의 edge auxiliary branch를 사용하지 않는다.
