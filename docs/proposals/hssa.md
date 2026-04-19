# HSSA — Hazard-Scheduled Sample Annealing

## 1. Motivation

만성(chronic) miss 를 무조건 더 세게 학습시키는 것은 위험하다. 어떤 GT 는 **원래 맞히기 불가능** 할 수 있다:

- 주석 오류 (annotation noise)
- 극심한 occlusion / truncation
- 해상도 한계로 판별 불가능한 작은 객체
- 학습 데이터 분포 밖의 희귀 패턴

이들을 가중 증가로 계속 누르면 **나머지 정상 GT 의 학습을 희생** 하고 loss 가 노이즈에 과적합된다. 한편 단순히 가중을 상한으로 고정하면 "실제로 어려운 중요한 GT" 와 "불가능한 GT" 를 구분하지 못한다.

HSSA 는 survival analysis 의 **hazard function** 을 차용해 이 딜레마를 원리적으로 해결한다. "얼마나 오래 놓쳤는가" 가 아니라 **"지금 시점에서 풀릴 확률이 얼마인가"** 를 가중 근거로 삼는다.

## 2. Survival View of Miss Streaks

GT $g$ 의 "miss streak 지속 시간" 을 확률변수 $T$ 로 본다. Streak 길이 $t = s_t(g)$ 에서 다음 epoch 에 detected 가 될 조건부 확률을 **hazard** 로 정의한다.

$$
h(t) \;=\; \Pr\!\left[\,T = t+1 \,\mid\, T > t\,\right]
$$

- $h(t)$ 가 **높음** → streak 이 곧 끊길 가능성이 큼 → 모델이 학습 중.
- $h(t)$ 가 **낮고 감소** → streak 이 길어질수록 더 못 푼다 → 점점 어려워지거나 구제 불가.
- $h(t)$ 가 **plateau** → 길어져도 풀리지 않는 정상 상태 → **구제 불가능 후보** (label noise 포함).

그리고 survival function 은

$$
S(t) \;=\; \prod_{k=0}^{t-1} \bigl(1 - h(k)\bigr)
$$

$S(t)$ 가 아주 작아지는 $t$ 구간은 "이 정도로 길게 놓친 GT 는 거의 없음 = 예외적 영구 실패" 를 의미한다.

## 3. Empirical Hazard Estimation

MDMB 의 시간축 이력만으로 $h(t)$ 를 집계 가능하다. Epoch $e$ 종료 시점의 누적 통계:

- $N(t)$ = "과거에 streak 길이 $t$ 에 도달한 GT × epoch 관측 수"
- $D(t)$ = "streak 길이 $t$ 에 도달한 후 **다음 epoch 에 detected 된** GT × epoch 관측 수"

$$
\hat{h}_e(t) \;=\; \frac{D(t) + \eta}{N(t) + \eta \cdot K}
$$

- $\eta$, $K$ 는 Laplace smoothing (권장 $\eta = 1$, $K = 2$).
- EMA 로 시간에 걸쳐 안정화:

$$
h_e(t) \;=\; \tau \cdot h_{e-1}(t) + (1-\tau)\cdot \hat{h}_e(t),
\quad \tau = 0.8
$$

### 3.1 데이터 구조

HSSA 는 MDMB 와 독립적으로 **작은 전역 테이블** 만 유지한다.

```
hazard_table: dict[int t -> (float h_t, int N_t, int D_t)]
```

- 메모리: $t_\text{max} \!\sim\! 50$ 정도로 cap → 무시할 수준.
- `engine.fit()` 의 `end_epoch` hook 에서 업데이트.
- Checkpoint: `get_extra_state` 에 포함.

## 4. Hazard-based Weight Modulation

HSSA 는 **자체 loss 를 추가하지 않는다**. T-RECALL 의 시간 변조항 $\phi$ 에 **곱해지는 hazard 계수** $\eta(t)$ 를 제공한다.

$$
w_\text{final}(g) \;=\; \alpha_\text{type}(g) \,\cdot\, \phi\!\left(p(g), r(g), \rho(g)\right) \,\cdot\, \eta\!\left(s_t(g)\right)
$$

$\eta(t)$ 의 설계 원칙:

1. $h(t)$ 가 **감소 구간** → 아직 모델이 배우는 중 → **가열** 유지.
2. $h(t)$ 가 **plateau** → 학습 불가능 후보 → **냉각** ( $\eta \to \eta_\text{floor}$ ).
3. $h(t)$ 가 **상승** 또는 해당 streak 길이에서 검출이 관측됨 → 오버슛 방지로 **감쇠**.

구체 형태:

$$
\eta(t) \;=\;
\eta_\text{floor}
\;+\;
\bigl(1 - \eta_\text{floor}\bigr) \cdot
\underbrace{\frac{h(t)}{h(0)+\epsilon}}_{\text{hazard ratio}}
\cdot
\underbrace{\mathbb{1}\!\left[S(t) \ge S_\text{cut}\right]}_{\text{tail gate}}
$$

- $\eta_\text{floor}$: hazard 가 거의 0 인 경우의 잔여 가중 (권장 $\eta_\text{floor} = 0.2$).
- $h(0)$: streak 가 이제 막 시작된 "정상 난이도" 기준. 이로써 정규화된 ratio 가 된다.
- $S_\text{cut}$: 극단 tail (예: $S(t) < 0.01$) 에서는 $\eta(t) = \eta_\text{floor}$ 로 강제.

### 4.1 Plateau detection

"학습 불가능 plateau" 는 다음 조건으로 판정한다:

$$
\bigl|\,h(t) - h(t-1)\,\bigr| \;<\; \delta_h
\quad\text{for}\quad
t \in [t^\star, t^\star + k]
$$

와 동시에

$$
h(t) \;<\; h_\text{dead}
$$

이면 $t \ge t^\star$ 구간의 $\eta(t)$ 를 $\eta_\text{floor}$ 로 고정. 권장 $\delta_h = 0.01$, $h_\text{dead} = 0.05$, $k = 3$.

## 5. Non-conflict with Original Loss

- HSSA 는 **스칼라 가중만** 수정하므로 T-RECALL 과 동일하게 원 loss 의 해 공간을 벗어나지 않는다.
- Hazard 값은 MDMB 관측에서만 파생되며 gradient 를 갖지 않는다 ( `torch.no_grad` ).
- $\eta \equiv 1$ 로 두면 T-RECALL 과 수학적으로 동일 → 깔끔한 ablation.

## 6. Interpretation as Self-Regularization against Annotation Noise

라벨 노이즈 이론에서 반복적으로 증명된 경험칙:

> "영원히 못 맞히는 샘플에 loss 를 쏟을수록 clean sample 의 학습이 저하된다."

HSSA 는 데이터 자체에서 "영원히 못 맞히는" 을 **통계적으로 식별** 한다. 외부 clean label oracle 없이, MDMB 이력만으로 노이즈 추정이 이루어진다는 점이 장점이다.

이는 **Co-teaching** 이나 **Early-stopping based label cleaning** 계열과 달리,

- 별도 네트워크 없음,
- 사전 label review 없음,
- inference 영향 없음.

## 7. Algorithm

### 7.1 End-of-epoch hazard update

```
for each GT record g in MDMB:
    t_prev = s_{e-1}(g)
    detected_now = (s_e(g) == 0 and t_prev > 0)
    missed_now   = (s_e(g) == t_prev + 1)

    N[t_prev] += 1
    if detected_now:
        D[t_prev] += 1

h_hat = (D + eta) / (N + eta * K)
h = tau * h_prev + (1 - tau) * h_hat
```

### 7.2 Per-step weight application

```
for each entry e in batch:
    phi   = phi(p(e), r(e), rho(e))       # T-RECALL
    eta_t = hazard_weight(s_t(e), hazard_table)
    w[e]  = clip(alpha_type[e] * phi * eta_t, lo=phi_min, hi=w_max)
```

## 8. Integration Points

- **Hook**: `engine.fit()` 의 `end_epoch` 에서 hazard table 갱신. MDMB `update(...)` 에서 per-GT streak 전이만 노출하면 됨.
- **Module**: `modules/nn/hssa.py` 신규 (제안). T-RECALL 구현 안의 `compute_phi(...)` 와 짝을 이루는 `hazard_weight(...)` 제공.
- **Config**: `modules/cfg/hssa.yaml` 에 `enabled`, thresholds, smoothing 계수.
- **Checkpoint**: hazard table 직렬화 / 복원.
- **Inference**: hazard table 미사용 → 완전히 training-only.

## 9. Hyperparameters

| 이름 | 기본값 | 설명 |
|---|---|---|
| `eta_floor` | 0.2 | plateau 구간 최소 가중 |
| `S_cut` | 0.01 | tail cutoff survival prob |
| `delta_h` | 0.01 | plateau 검출 민감도 |
| `h_dead` | 0.05 | dead-hazard 임계 |
| `plateau_k` | 3 | plateau 판정 연속 epoch 수 |
| `smooth_eta` | 1.0 | Laplace smoothing $\eta$ |
| `smooth_K` | 2 | Laplace smoothing $K$ |
| `ema_tau` | 0.8 | hazard EMA 계수 |
| `t_max` | 50 | hazard table 최대 streak 길이 |

## 10. Novelty Summary

1. **Survival analysis → detection 도입**: hazard function 을 detection 학습 가중에 사용한 사례는 없음. 생체 통계학에서 온 이 포맷은 "얼마나 오래 놓쳤는가" 가 아니라 **"지금 풀릴 확률"** 이라는 관점 전환을 제공.
2. **Annotation-noise robustness without extra model**: Co-teaching / self-labeling 계열과 달리 추가 네트워크 없이 **MDMB 통계만으로** noise 후보를 통계적 식별.
3. **Composable with T-RECALL / FAR**: 독립 loss 가 아닌 weight multiplier 라 다른 모듈과 자연스럽게 결합. $\eta \equiv 1$ 로 off 가능 → 명확한 ablation.
4. **Chronic miss 처리의 원리적 상한**: "가중을 무한히 올리지 말라" 는 실무 경험칙을 **survival statistics** 위에 정당화.

## 11. Limitations

- Hazard 추정치가 수렴하려면 epoch 수가 충분해야 함 ($\sim 10$ epochs 이후에 안정). 초기엔 $\eta \equiv 1$ 로 bypass.
- Streak 길이 별 표본이 부족할 수 있음 → Laplace smoothing 필수.
- Class-agnostic hazard 를 쓰면 class imbalance 가 섞임 → 확장으로 class-conditional hazard $h(t \mid c)$ 가능 (메모리 증가 감수).

## 12. Ablations (권장)

- HSSA off ($\eta \equiv 1$) vs on
- $\eta_\text{floor} \in \{0.0, 0.1, 0.2, 0.3, 1.0\}$
- Plateau gate on/off (tail_cut 유무)
- Class-agnostic vs class-conditional hazard
- **핵심 지표**:
  - mAP / AR (상승 확인)
  - **noisy-like GT 에 대한 loss contribution** (HSSA 가 낮춰야 함)
  - **recoverable chronic GT 의 recovery rate** (HSSA 가 낮추지 않아야 함)
  - training stability (loss variance across epochs)
