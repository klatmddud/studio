# FAR — Forgetting-Aware feature Replay

## 1. Motivation

MDMB의 시간축 확장으로 각 GT에 대해 `last_detected_epoch` 과 `consecutive_miss_count` 를 얻게 되면서, 다음과 같은 "**관계절**(relapse)" 이벤트를 per-GT 단위로 식별할 수 있게 되었다.

$$
\text{relapse}(g, t) \;=\; \mathbb{1}\!\left[\text{last\_detected\_epoch}(g) \neq \text{None} \;\land\; s_t(g) > 0\right]
$$

여기서 $g$ 는 특정 GT, $t$ 는 현재 epoch, $s_t(g)$ 는 현재 연속 miss streak이다. Relapse는 모델이 **한 번 맞혔다가 다시 놓친 객체** 를 의미하며, 학습 도중 발생하는 **intra-task catastrophic forgetting** 의 직접적 신호다.

기존 detection 문헌은 이 현상을 명시적으로 다루지 않았다. Mean-Teacher 계열은 네트워크 수준의 teacher 를 유지하지만, **어떤 객체가 언제 잊혔는가** 에 대응하지 않는다. FAR 은 이 gap 을 채운다.

## 2. Core Idea

각 GT 에 대해 **과거 성공 시점의 feature 를 앵커로 저장** 하고, relapse 가 발생하는 동안에만 현재 feature 를 그 앵커로 당기는 object-level self-distillation 을 수행한다.

- 앵커는 **training-only**, gradient 없음, per-GT 작은 벡터.
- Relapse 가 해제되면 (다시 detected) 앵커는 자연스럽게 갱신된다.
- Loss 는 relapse 서브셋 위에서만 정의 → 초반엔 dormant, 원 loss 와 간섭 최소.

## 3. Data Structures

MDMB `_GTRecord` 에 다음을 추가한다.

```
anchor:        torch.Tensor | None   # shape [D], L2-normalized, cpu
anchor_epoch:  int | None            # 앵커가 저장된 epoch
anchor_frozen: bool                  # relapse 구간 중 동결 여부
```

- `anchor` 는 GT 영역에서 추출한 ROI feature 의 L2-normalized 벡터.
- `D` 는 architecture-dependent ($D_\text{FCOS} \!=\! C_\text{FPN}$, $D_\text{RCNN} \!=\! C_\text{RoI}$).
- 메모리: 전체 GT 수 $N_\text{GT}$, $D \!\approx\! 256$ 기준 $N_\text{GT} \!\times\! 256 \!\times\! 4\,\text{B}$ → 10k GT 기준 약 10 MB 수준.

## 4. Anchor Maintenance

### 4.1 Feature 추출

`update(...)` hook 에서 `detected` 로 분류된 GT 에 대해서만 앵커 후보 feature 를 뽑는다.

- **FCOS**: GT 중심 픽셀이 속한 FPN level 에서 해당 위치의 feature vector (centerness-weighted pooling 가능).
- **Faster R-CNN**: GT box 기반 RoIAlign → global-average → flatten.

$$
\tilde{f}_g \;=\; \text{ROIFeat}\!\left(\mathcal{F}, b_g\right), \qquad
f_g \;=\; \frac{\tilde{f}_g}{\lVert \tilde{f}_g \rVert_2 + \epsilon}
$$

### 4.2 갱신 규칙

현재 epoch 에서 GT $g$ 의 상태를 보고:

| 현재 상태 | 앵커 동작 |
|---|---|
| detected ∧ 앵커 없음 | 초기화: $a_g \leftarrow f_g$, `anchor_epoch = t`, `frozen = False` |
| detected ∧ 앵커 있음 ∧ `frozen = False` | EMA 갱신: $a_g \leftarrow \text{norm}(\mu \, a_g + (1{-}\mu)\, f_g)$ |
| relapse 시작 ( $s_t(g) = 1$, `last_detected` 존재) | `frozen = True` (앵커 보존) |
| relapse 지속 ( $s_t(g) > 1$ ) | 동결 유지 |
| relapse 해제 (다시 detected) | `frozen = False`, EMA 갱신 재개 |

$\mu$ 는 하이퍼파라미터 (권장 $\mu \!=\! 0.9$).

## 5. Loss Formulation

Relapse entry 집합을 $\mathcal{R}_t$ 로 정의한다. 각 entry 에 대해 현재 feature $f_g^{(t)}$ 와 동결 앵커 $a_g$ 사이의 cosine distance 를 사용한다.

$$
\mathcal{L}_\text{FAR}
\;=\;
\frac{\lambda_\text{far}}{|\mathcal{R}_t| + \epsilon}
\sum_{g \in \mathcal{R}_t}
w_g \cdot \bigl(1 - \cos\!\langle f_g^{(t)}, \, a_g \rangle\bigr)
$$

- $w_g$ 는 persistence 가중:

$$
w_g \;=\; 1 + \gamma \cdot \frac{s_t(g)}{\max\!\bigl(s_\text{global}, 1\bigr)}
$$

이는 relapse 가 길어질수록 복원 압력이 커지도록 한다.
- $\lambda_\text{far}$ 권장 범위: $[0.05, 0.3]$. Main loss 의 0.1 배 수준으로 시작.

## 6. Gradient Flow

- $a_g$ 는 `detach()` 된 상태로 저장 → gradient 없음.
- $f_g^{(t)}$ 만 gradient 를 받아 **backbone / FPN / RoI head** 까지 흐름.
- **Classification / regression head 파라미터는 공유하지 않는다.** $f_g^{(t)}$ 는 RoI feature 수준에서 추출되므로, detection head 내부의 cls/reg branch 와는 별개의 feature map 기반이다.
- 따라서 $\nabla \mathcal{L}_\text{FAR}$ 와 $\nabla \mathcal{L}_\text{det}$ 는 backbone 에서 합류하지만, **head 단계의 cls/reg gradient 를 오염시키지 않는다.**

## 7. Non-conflict with Original Loss

- $\mathcal{R}_t = \emptyset$ 이면 $\mathcal{L}_\text{FAR} = 0$ → warmup 기간과 학습 초반 대부분의 step 에서 자동 dormant.
- Relapse 는 본질적으로 **작은 서브셋** ( $\sim 1\text{-}5\%$ of GTs) 에 대한 현상이므로 batch loss 기여가 작다.
- Loss 형태가 cosine distance 로 cls/reg 와 **부호·스케일이 분리** → gradient 방향 충돌 가능성 낮음.
- $\lambda_\text{far}$ 가 작을 때 안정성을 이론적으로 보장:

$$
\left\lVert \nabla \mathcal{L}_\text{total} - \nabla \mathcal{L}_\text{det}\right\rVert
\;\le\;
\lambda_\text{far} \cdot L_\text{cos} \cdot \kappa
$$

여기서 $L_\text{cos} \!\le\! 1$, $\kappa$ 는 feature 의 Lipschitz 상수.

## 8. Integration Points

- **Hook**: `engine.fit()` 의 `after_optimizer_step` 직후, MDMB `update(...)` 다음 단계에 배치.
- **Training-only branch**: 앵커 저장·loss 계산 모두 `model.training == True` 경로 한정.
- **Checkpoint**: `get_extra_state / set_extra_state` 에 `anchor`, `anchor_epoch`, `frozen` 직렬화 추가. 버전 v4 → v5 승격.
- **Inference**: 앵커·FAR loss 사용 없음 → FLOPs / latency 영향 0.

## 9. Pseudocode (for clarity only)

```python
# after optimizer.step(), within training loop
with torch.no_grad():
    features = backbone_fpn(images)          # reuse cached feats
    roi_feats = extract_roi_feats(features, gt_boxes_per_image)
    roi_feats = F.normalize(roi_feats, dim=-1)

# relapse 대상만 현재 feature 에 gradient 활성 재계산 (mini-pass)
relapse_feats = forward_for_relapse(features, relapse_boxes)  # requires_grad=True
anchors = stack_anchors(relapse_ids)                           # detached

cos = (relapse_feats * anchors).sum(dim=-1)
loss_far = ((1.0 - cos) * relapse_weights).sum() / (relapse_feats.size(0) + 1e-6)
(lambda_far * loss_far).backward()

update_anchors(detected_ids, roi_feats, mu=0.9)
freeze_anchors(newly_relapsed_ids)
unfreeze_anchors(recovered_ids)
```

> 실제 구현에서는 main backward 와 **하나의 backward pass** 로 통합하는 게 효율적이다 (mini-pass 는 예시용).

## 10. Hyperparameters

| 이름 | 기본값 | 설명 |
|---|---|---|
| `lambda_far` | 0.1 | FAR loss 가중 |
| `anchor_ema_mu` | 0.9 | 앵커 EMA 계수 |
| `persistence_gamma` | 1.0 | 장기 relapse 가중 증폭 |
| `min_relapse_streak` | 1 | $s_t(g)$ 가 이 값 이상일 때만 active |
| `freeze_policy` | `on_relapse_start` | `on_relapse_start` \| `always_after_first_detect` |

## 11. Novelty Summary

1. **Object-level temporal memory**: network-level teacher 가 아닌 **per-GT anchor** 를 시간축으로 유지.
2. **Relapse-triggered activation**: 망각 이벤트가 있을 때만 loss 가 활성화되어 원 학습 궤도를 해치지 않음.
3. **Anchor freeze during relapse**: "잊히기 직전" 의 표현을 타겟으로 삼아, 현재 네트워크가 자기 자신의 과거 성공 상태로 회귀하도록 유도.
4. MDMB 의 시간축 확장 없이는 불가능한 구성 → 시간축 확장의 당위성을 동시에 입증.

## 12. Ablations (권장)

- FAR on/off at fixed $\lambda_\text{far}$
- $\lambda_\text{far} \in \{0.05, 0.1, 0.2, 0.5\}$
- Anchor freeze 정책 비교 (`on_relapse_start` vs `always`)
- Relapse 가중 $\gamma \in \{0, 0.5, 1.0, 2.0\}$
- FCOS / Faster R-CNN 양쪽에서 mAP, background FP rate, **recovery-rate of relapsed GTs** 측정
