# T-RECALL — Temporal RECALL

## 1. Motivation

기존 RECALL 은 MDMB 의 miss-type 정보만 활용해 per-sample loss 를 두 가지 상수 ( $\alpha_A$, $\alpha_B$ ) 로 증폭한다.

$$
w_\text{RECALL}(g) \;=\;
\begin{cases}
\alpha_A & \text{if } \text{type}(g) = \text{type\_a} \\
\alpha_B & \text{if } \text{type}(g) = \text{type\_b}
\end{cases}
$$

이는 다음 한계를 갖는다:

1. **시간축 무시**: "첫 번째로 놓친 GT" 와 "열 epoch 째 계속 놓치는 GT" 를 동일하게 다룸.
2. **Relapse 무시**: 한 번 맞혔다가 다시 놓친 GT (catastrophic forgetting) 에 대해 별도 가중이 없음.
3. **Overfit 위험 미관리**: 방금 풀린 GT 도 여전히 높은 가중으로 학습되어 불안정 가능.

MDMB 시간축 확장이 제공하는 신호들 ( `consecutive_miss_count`, `max_consecutive_miss_count`, `last_detected_epoch` ) 을 활용해 RECALL 을 시간 함수 기반으로 일반화한 것이 T-RECALL 이다.

## 2. Derived Signals

각 MDMB entry 또는 `_GTRecord` $g$ 에 대해 다음을 정의한다.

| 기호 | 정의 | 범위 |
|---|---|---|
| $s_t(g)$ | `consecutive_miss_count` (현재 연속 miss) | $\mathbb{N}_0$ |
| $m(g)$ | `max_consecutive_miss_count` (이력 최대) | $\mathbb{N}_0$ |
| $S_t$ | `_global_max_consecutive_miss` (전역 정규화 상수) | $\mathbb{N}$ |
| $p(g)$ | persistence score | $[0,1]$ |
| $r(g)$ | relapse flag | $\{0,1\}$ |
| $\rho(g)$ | resolved-recent flag | $\{0,1\}$ |

$$
p(g) \;=\; \frac{m(g)}{\max(S_t, 1)}, \qquad
r(g) \;=\; \mathbb{1}\!\left[\text{last\_detected}(g) \neq \text{None} \land s_t(g) > 0\right]
$$

$$
\rho(g) \;=\; \mathbb{1}\!\left[s_t(g) = 0 \land m(g) \ge m^\star\right]
$$

여기서 $m^\star$ 는 "만성이었던" 을 정의하는 임계치 (권장 $m^\star = 2$).

## 3. Temporal Weight Function

T-RECALL 의 최종 per-entry 가중은 기존 RECALL 상수에 **시간 변조항** 을 곱한 형태다.

$$
w_\text{T-RECALL}(g) \;=\;
\alpha_\text{type}(g) \,\cdot\, \phi\bigl(p(g),\, r(g),\, \rho(g)\bigr)
$$

$$
\phi(p, r, \rho) \;=\;
1 \;+\; \underbrace{\alpha \cdot \sigma(\beta \cdot p)}_{\text{chronic emphasis}}
\;+\; \underbrace{\gamma \cdot r}_{\text{relapse bonus}}
\;-\; \underbrace{\delta \cdot \rho}_{\text{resolved damping}}
$$

- $\sigma(\cdot)$: sigmoid. $\beta$ 는 chronicity 의 날카로움, $\alpha$ 는 상한.
- $\gamma$ 는 relapse 발생 시 일회성 보너스 (권장 $\gamma = 0.5$).
- $\delta$ 는 방금 풀린 GT 에 대한 감쇠 (권장 $\delta = 0.3$).
- 음수로 떨어지지 않도록 $\phi \ge \phi_\text{min}$ 으로 클립 (권장 $\phi_\text{min} = 0.25$).

### 3.1 직관

| 상황 | $p$ | $r$ | $\rho$ | $\phi$ | 해석 |
|---|---|---|---|---|---|
| 첫 miss | 낮음 | 0 | 0 | $\approx 1$ | 기존 RECALL 과 동일 |
| 만성 miss | 높음 | 0 | 0 | $\approx 1+\alpha$ | 강한 복원 압력 |
| Relapse | 중간 | 1 | 0 | $1 + \alpha\sigma + \gamma$ | 망각 이벤트에 추가 벌점 |
| 최근 해결 | — | 0 | 1 | $1 - \delta$ | 과적합 억제 |

## 4. Application over Losses

FCOS 기준 per-point loss (cls, reg, centerness) 를 예로 든다. Point $i$ 가 GT $g_i$ 에 할당되었을 때:

$$
\mathcal{L}_\text{cls} \;=\;
\frac{1}{N_\text{pos}} \sum_i
w_\text{T-RECALL}(g_i) \,\cdot\, v_i \,\cdot\, \ell_\text{cls}\!\left(\hat{y}_i, y_i\right)
$$

- $v_i \in \{0,1\}$: 기존 RECALL 의 `ignore_iou_threshold` 기반 valid mask.
- $\ell_\text{cls}$: focal loss (원본 FCOS 정의 그대로).
- 분모 $N_\text{pos}$ 는 기존과 동일하게 양성 샘플 수.

회귀·centerness 도 동일 계수 사용 (RECALL 의 원 구현과 일치).

## 5. Non-conflict with Original Loss

T-RECALL 은 **새 loss 항을 추가하지 않는다**. 오직 기존 per-sample 가중 상수를 **시간 함수로 대체** 한다.

- $\phi \equiv 1$ 로 설정하면 원 RECALL 와 수학적으로 동일.
- 따라서 ablation 에서 $\phi \to 1$ 이 baseline 이 되고, 추가 성능이 순수히 시간축에서 기인함을 분리 가능.
- Gradient 방향은 원 loss 와 일치하며, **스케일만 재분배** 된다:

$$
\nabla \mathcal{L}_\text{T-RECALL}
\;=\;
\sum_i w_i \nabla \ell_i
\quad\Rightarrow\quad
\text{direction} \in \operatorname{span}(\nabla \ell_i)
$$

즉, 원 loss 의 **해 공간을 벗어나지 않는다**.

## 6. Stability Considerations

Chronic weight 가 폭주하지 않도록 상한 제약:

$$
w_\text{T-RECALL}(g) \;\le\; w_\text{max}
$$

권장 $w_\text{max} = 5 \cdot \alpha_A$. 또한 $S_t$ 가 작은 학습 초반엔 $p(g)$ 가 1 근처로 튈 수 있으므로 warmup 이후 $S_t \ge S_\text{min}$ 일 때만 활성화한다.

$$
\phi(p, r, \rho) \;\leftarrow\;
\begin{cases}
\phi(p, r, \rho) & \text{if } S_t \ge S_\text{min} \\
1 & \text{otherwise}
\end{cases}
$$

권장 $S_\text{min} = 2$.

## 7. Integration with RECALL

기존 `MDMBSelectiveLoss.compute_weights` 의 반환 로직을 다음과 같이 확장한다 (설계만, 코드 수정 아님).

```
def compute_weights(..., entries_by_gt_index):
    # entries_by_gt_index: GT index -> MDMBEntry (with temporal fields)
    weights = ones(...)
    for point_idx, gt_idx in assignments:
        entry = entries_by_gt_index.get(gt_idx)
        if entry is None:
            continue
        alpha_type = amp_type_a if entry.miss_type == "type_a" else amp_type_b
        phi = compute_phi(entry, S_t)
        weights[point_idx] = clip(alpha_type * phi, lo=phi_min, hi=w_max)
    return weights, valid
```

Relapse / resolved 플래그는 `MDMBEntry` 의 `last_detected_epoch`, `consecutive_miss_count`, `max_consecutive_miss_count` 에서 즉시 계산된다.

## 8. Hyperparameters

| 이름 | 기본값 | 설명 |
|---|---|---|
| `amp_type_a` | 2.5 | 기존 RECALL 과 동일 |
| `amp_type_b` | 1.5 | 기존 RECALL 과 동일 |
| `alpha` (chronic amp) | 1.0 | sigmoid 상한 |
| `beta` (chronic sharpness) | 4.0 | sigmoid 기울기 |
| `gamma` (relapse bonus) | 0.5 | relapse 가산 |
| `delta` (resolved damp) | 0.3 | 최근 해결 감쇠 |
| `m_star` | 2 | resolved flag 임계 |
| `S_min` | 2 | 전역 활성화 임계 |
| `phi_min` | 0.25 | $\phi$ 하한 |
| `w_max` | $5 \cdot \alpha_A$ | weight 상한 |

## 9. Novelty Summary

1. **Per-GT temporal curriculum**: 난이도만 보는 기존 curriculum 과 달리 **학습 동역학** ( relapse / chronic / resolved ) 을 반영.
2. **Relapse as first-class signal**: 한 번 맞혔다가 놓친 GT 에 대한 명시적 가중. Detection 문헌에서 처음 형식화.
3. **Resolved damping**: 풀린 직후의 hard-sample 을 자동으로 de-emphasize → overfitting 억제.
4. RECALL 의 **진정한 generalization**: $\phi \equiv 1$ 로 원본 회귀 가능 → ablation 이 청결.

## 10. Failure Modes & Mitigations

| 실패 양상 | 원인 | 대응 |
|---|---|---|
| 초반 loss 폭발 | $S_t$ 가 작을 때 $p$ 가 1 근처 | $S_\text{min}$ 게이트 |
| Chronic GT 가 annotation noise | $p \to 1$ 이지만 실제로는 bad label | HSSA 와 조합 (별도 문서) |
| Relapse 진동 | batch 변동성으로 $r$ 이 켜졌다 꺼졌다 | $\gamma$ 를 낮추고 EMA 로 smoothing 고려 |

## 11. Ablations (권장)

- $\phi \equiv 1$ (= 원본 RECALL) vs T-RECALL 전체
- Chronic-only ($\gamma = \delta = 0$), Relapse-only ($\alpha = \delta = 0$), Resolved-only ($\alpha = \gamma = 0$)
- $\beta$ 와 $\alpha$ 의 sensitivity
- FCOS 에서 검증; RCNN 으로 확장 시 동일 weight 를 RPN classification 및 RoI head 에 공통 적용
- 지표: mAP, AR, **chronic-miss recovery rate** ( $r=1 \to r=0$ transition 빈도 )
