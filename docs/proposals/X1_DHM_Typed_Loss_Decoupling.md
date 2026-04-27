# X1. DHM-Typed Loss Decoupling

## 개요

DHM(Detection History Monitor)이 각 GT instance의 False Negative를 **L / C / B** 세 타입으로 분류한 결과를 활용하여, 해당 샘플에 흐르는 regression / classification / score loss의 **gradient 흐름을 선택적으로 차단하거나 증폭**하는 방법론.

새로운 loss term을 추가하지 않고, **기존 loss의 sample-wise routing만 변경**하므로 원래 모델의 목적 함수와 충돌하지 않는다.

---

## 핵심 아이디어

일반적인 object detection 학습에서, 하나의 positive sample은 regression loss와 classification loss를 **동시에** 역전파한다. 그러나 FN 타입에 따라 두 loss가 서로 다른 방향으로 gradient를 당기는 경우가 발생한다.

예를 들어:

- **Type L (Localization 실패)**: box는 틀렸지만 해당 영역의 class score는 이미 높을 수 있다. 이 샘플에 classification loss까지 역전파하면 score를 낮추는 방향으로 간섭이 발생한다.
- **Type C (Classification 실패)**: localization은 맞지만 class를 틀렸다. 이 샘플에 regression loss를 역전파하면 box가 이미 맞는데도 불필요한 gradient noise가 누적된다.
- **Type B (Score/IoU 임계값 미달)**: score 또는 IoU 자체가 낮아 background로 분류된 케이스. 두 loss 모두 관여하지만, 어느 loss를 더 강하게 줄 것인가에 따라 보정 방향이 달라진다.

DHM이 이 타입 정보를 이미 추적하고 있다면, 이를 **loss routing mask**로 변환하여 각 샘플이 받는 gradient를 조정할 수 있다.

---

## 방법론 설계

### 1. FN 타입 레이블 생성

DHM으로부터 매 iteration마다 각 GT에 대해 FN 타입 레이블을 받는다:

```
T(i) ∈ { L, C, B, TP, TN }
```

- **L**: IoU < threshold (localization 실패)
- **C**: IoU ≥ threshold & predicted class ≠ GT class (classification 실패)
- **B**: score 또는 IoU가 임계값 미달로 background 분류
- **TP**: 정상 검출
- **TN**: 정상 background

### 2. Loss Routing Mask 정의

각 GT sample *i* 에 대해, regression loss weight $w^{reg}_i$와 classification loss weight $w^{cls}_i$를 다음과 같이 설정한다:

| FN 타입 | $w^{reg}_i$ | $w^{cls}_i$ | 의도 |
|---------|------------|------------|------|
| **L** | $\alpha > 1$ | $\beta < 1$ | Regression에 집중, cls 간섭 억제 |
| **C** | $\beta < 1$ | $\alpha > 1$ | Classification에 집중, reg noise 억제 |
| **B** | $\gamma$ | $\gamma$ | 두 loss 모두 소폭 증폭 |
| **TP / TN** | $1.0$ | $1.0$ | 기본값 유지 |

여기서 $\alpha, \beta, \gamma$는 하이퍼파라미터로, 초기값으로 $\alpha=1.5, \beta=0.5, \gamma=1.2$를 권장한다.

### 3. Loss 계산

기존 detection loss를 다음과 같이 수정한다:

$$\mathcal{L}_{total} = \sum_{i} \left[ w^{reg}_i \cdot \mathcal{L}^{reg}_i + w^{cls}_i \cdot \mathcal{L}^{cls}_i \right]$$

- $\mathcal{L}^{reg}_i$: 기존 regression loss (L1, GIoU, CIoU 등 모델 원본 사용)
- $\mathcal{L}^{cls}_i$: 기존 classification loss (CE, Focal 등 모델 원본 사용)
- **loss 함수 자체는 일절 변경하지 않는다.**

### 4. Routing 적용 시점

DHM 타입은 이전 iteration 또는 이전 epoch의 기록에서 가져온다. 따라서:

- **Online 방식**: 이전 forward pass의 결과를 캐싱하여 현재 iteration에 적용 (배치 단위)
- **Offline 방식**: 매 epoch 종료 후 전체 데이터셋에 대해 DHM 타입 재계산 후 다음 epoch에 적용

구현 편의상 **Offline 방식**이 안정적이다.

---

## 모델별 적용 방안

### Faster R-CNN
- RPN head의 objectness loss와 RoI head의 cls/reg loss에 각각 독립적으로 mask 적용
- RPN 단계는 타입 L에 해당하는 샘플 위주로 reg weight 증폭

### FCOS
- Classification branch와 regression branch가 분리되어 있으므로 mask 적용이 직관적
- Center-ness loss는 타입 B 샘플에 대해 weight 증폭 권장

### DINO
- Bipartite matching 이후 할당된 query에 대해 타입 레이블 매핑
- Denoising group과 matching group에 각각 별도의 mask 스케줄 적용 가능

---

## 기대 효과

- **Type L 샘플**: regression gradient가 강화되어 box 위치가 더 빠르게 수렴
- **Type C 샘플**: classification head가 해당 샘플에 더 집중하고 regression noise 감소
- **Type B 샘플**: 전반적인 score calibration 효과
- **기존 TP/TN 샘플**: weight 변화 없으므로 정상 학습 유지

---

## 한계 및 고려사항

- DHM 타입 레이블의 **정확도에 의존**한다. 타입 분류가 불안정한 초기 학습 구간에서는 mask를 적용하지 않거나 약하게 적용하는 warm-up 전략이 필요할 수 있다.
- $\alpha, \beta, \gamma$ 하이퍼파라미터에 민감할 수 있으므로, grid search 또는 validation loss 기반 자동 조정을 권장한다.
- 타입 분류 비율이 epoch마다 변화하므로, 동적으로 weight를 조정하는 **adaptive scheduling** 도입을 고려할 수 있다.

---

## 구현 예시 (PyTorch 스타일)

```python
def apply_dhm_loss_routing(loss_reg, loss_cls, dhm_types, alpha=1.5, beta=0.5, gamma=1.2):
    """
    loss_reg: (N,) tensor, per-sample regression loss
    loss_cls: (N,) tensor, per-sample classification loss
    dhm_types: (N,) list of strings, 'L' / 'C' / 'B' / 'TP' / 'TN'
    """
    w_reg = torch.ones_like(loss_reg)
    w_cls = torch.ones_like(loss_cls)

    for i, t in enumerate(dhm_types):
        if t == 'L':
            w_reg[i] = alpha
            w_cls[i] = beta
        elif t == 'C':
            w_reg[i] = beta
            w_cls[i] = alpha
        elif t == 'B':
            w_reg[i] = gamma
            w_cls[i] = gamma

    loss = (w_reg * loss_reg + w_cls * loss_cls).mean()
    return loss
```
