# L2. Localization Confidence Disentanglement

## 개요

Classification score와 localization quality를 완전히 분리하여 예측하되, 두 값의 **불일치(discrepancy)가 클 때만** regression loss weight를 증폭하는 방법론.

기존의 IoU-aware head (IoU-Net, TOOD 등)가 localization score를 별도로 예측하는 것과는 달리, **두 score 간의 불일치 신호 자체를 hard sample mining의 기준**으로 사용한다는 점에서 차별화된다.

---

## 핵심 아이디어

일반적인 detection 모델에서 classification score $p_{cls}$와 localization quality $q_{loc}$(예: predicted IoU)은 **같은 feature**에서 파생되거나, 또는 $q_{loc}$를 별도로 예측하더라도 두 값의 관계를 학습에 직접 활용하지 않는다.

그러나 실제 FN 분석에서 다음과 같은 패턴이 관찰된다:

- **$p_{cls}$가 높은데 $q_{loc}$가 낮은 경우**: 모델이 "뭔가 있다"는 것은 알지만 어디 있는지 모르는 상태 → **Localization 실패의 전형**
- **$p_{cls}$가 낮은데 $q_{loc}$가 높은 경우**: box는 잘 잡았지만 class를 확신하지 못하는 상태 → **Classification 실패의 전형**

이 두 score의 **discrepancy**는 모델이 해당 sample에서 어떤 측면이 취약한지를 내재적으로 신호하고 있다. 이를 **regression loss reweighting의 동적 기준**으로 활용한다.

L2는 특히 **Localization 실패(Type L, 51%)** 타겟으로 설계되었으며, $p_{cls} \gg q_{loc}$인 샘플에 regression loss를 증폭하여 모델이 해당 instance의 box 정밀도를 더 집중적으로 학습하도록 유도한다.

---

## 방법론 설계

### 1. 두 Score의 분리 예측

기존 classification head에서 $p_{cls}$를 예측하는 것 외에, 동일한 RoI feature에서 **localization quality score** $q_{loc}$를 별도로 예측하는 경량 branch를 추가한다.

```
RoI Feature → [기존 cls head] → p_cls ∈ [0, 1]
           └→ [경량 loc head] → q_loc ∈ [0, 1]
```

$q_{loc}$의 학습 target은 **예측 box와 GT box의 IoU**를 사용한다 (기존 IoU-aware 방법과 동일).

> **Note**: $q_{loc}$ branch는 FC 1~2개의 경량 구조로 충분하며, 기존 regression head를 수정하지 않는다.

### 2. Discrepancy Score 계산

두 score 간의 불일치를 다음과 같이 정의한다:

$$D_i = p_{cls,i} - q_{loc,i}$$

- $D_i > 0$: cls score가 loc quality보다 높음 → **Localization 실패 가능성** (Type L)
- $D_i < 0$: loc quality가 cls score보다 높음 → **Classification 실패 가능성** (Type C)
- $D_i \approx 0$: 두 score가 일치 → 정상 sample

### 3. Regression Loss Reweighting

Discrepancy를 기반으로 regression loss weight를 다음과 같이 동적으로 계산한다:

$$w^{reg}_i = 1 + \lambda \cdot \max(D_i, 0)$$

- $\max(D_i, 0)$: Type L 샘플($D_i > 0$)에만 증폭을 적용
- $\lambda$: reweighting 강도 하이퍼파라미터 (권장 초기값: 1.0~2.0)

최종 loss는 다음과 같다:

$$\mathcal{L}_{total} = \sum_i \left[ w^{reg}_i \cdot \mathcal{L}^{reg}_i + \mathcal{L}^{cls}_i + \mathcal{L}^{loc}_i \right]$$

- $\mathcal{L}^{reg}_i$: 기존 regression loss (변경 없음)
- $\mathcal{L}^{cls}_i$: 기존 classification loss (변경 없음)
- $\mathcal{L}^{loc}_i$: $q_{loc}$ 예측을 위한 auxiliary IoU loss (BCE 사용)

### 4. Inference 시 활용

Inference 시 $q_{loc}$를 classification score와 결합하여 최종 detection score를 구성할 수 있다:

$$s_i = p_{cls,i}^{(1-\mu)} \cdot q_{loc,i}^{\mu}$$

- $\mu \in [0, 1]$: 두 score의 결합 비율 (기본값: 0.5)
- 이 방식은 NMS 전 score reranking에도 적용 가능

---

## 기존 방법론과의 차별점

| 방법론 | localization score | 활용 방식 |
|--------|-------------------|----------|
| **IoU-Net** | 별도 예측 | Inference 시 NMS score 보정 |
| **TOOD** | task-aligned head | cls/reg alignment 학습 |
| **GFL** | quality focal loss | label smoothing 방식으로 통합 |
| **L2 (제안)** | 별도 예측 | **Discrepancy를 학습 중 동적 reweighting 기준으로 사용** |

핵심 차별점: 기존 방법들은 localization score를 inference 시 활용하거나 loss를 재정의하는 방식이지만, L2는 **두 score의 불일치 자체를 hard sample의 신호로 해석**하여 학습 중 regression loss를 동적으로 조정한다.

---

## 모델별 적용 방안

### Faster R-CNN
- RoI head의 fc feature에서 $q_{loc}$ branch 추가 (FC 2개)
- RPN 단계는 objectness score가 이미 localization proxy 역할을 하므로 적용 제외 가능

### FCOS
- Classification branch feature에서 $q_{loc}$ branch 병렬 추가
- Center-ness prediction과 $q_{loc}$의 역할 분리 필요: center-ness는 그대로 유지하고, $q_{loc}$는 예측 box의 IoU를 target으로 별도 학습

### DINO
- Decoder의 각 layer output에서 $q_{loc}$ 예측 (iterative refinement와 호환)
- 마지막 decoder layer의 discrepancy만 reweighting에 사용하여 학습 안정성 확보 권장

---

## 기대 효과

- **Type L 샘플**: $p_{cls} \gg q_{loc}$인 샘플에서 regression loss가 증폭되어 localization 수렴 가속
- **Type C 샘플**: $D_i < 0$인 샘플은 reweighting 대상에서 제외되어 간섭 없음
- **정상 샘플**: $D_i \approx 0$이므로 $w^{reg}_i \approx 1$로 기존 학습과 동일

---

## 한계 및 고려사항

- 학습 **초기 단계**에서는 $q_{loc}$가 불안정하여 discrepancy 신호가 노이즈가 될 수 있다. Warm-up 구간(초기 N epoch)에서는 $\lambda=0$으로 설정하고 이후 선형 증가하는 스케줄을 권장한다.
- $q_{loc}$ branch의 target인 predicted IoU는 regression head가 안정화된 이후 의미 있는 신호를 제공한다. 이는 warm-up과 같은 이유로 해결 가능하다.
- FCOS의 경우 center-ness와 $q_{loc}$의 역할이 일부 겹칠 수 있으므로, **center-ness를 $q_{loc}$로 대체**하는 방식도 고려할 수 있다 (ablation 필요).

---

## 구현 예시 (PyTorch 스타일)

```python
class LocQualityBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, roi_feat):
        return self.fc(roi_feat).squeeze(-1)  # (N,)


def compute_disentangled_loss(loss_reg, loss_cls, p_cls, q_loc, iou_target, lam=1.5):
    """
    loss_reg  : (N,) per-sample regression loss
    loss_cls  : (N,) per-sample classification loss
    p_cls     : (N,) predicted classification score
    q_loc     : (N,) predicted localization quality score
    iou_target: (N,) GT IoU as supervision for q_loc
    lam       : reweighting strength
    """
    # Auxiliary IoU loss for q_loc branch
    loss_loc = F.binary_cross_entropy(q_loc, iou_target, reduction='none')

    # Discrepancy-based reweighting (only amplify Type L: p_cls > q_loc)
    discrepancy = (p_cls - q_loc).detach()
    w_reg = 1.0 + lam * torch.clamp(discrepancy, min=0.0)

    loss = (w_reg * loss_reg + loss_cls + loss_loc).mean()
    return loss
```
