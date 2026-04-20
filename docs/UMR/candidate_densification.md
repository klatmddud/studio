# Candidate Densification — Creating More Recovery Opportunities

## 1. Overview

Candidate Densification은 MDMB++가 식별한 hard GT 주변에 추가 training candidate를 공급하는 모듈이다. UMR에서 가장 중요한 공격적 성능 향상 장치이며, 단순 reweighting과 가장 분명하게 구분되는 부분이기도 하다.

핵심 아이디어는 다음과 같다.

- chronic miss GT는 현재 detector가 적절한 candidate를 충분히 만들지 못했을 가능성이 높다.
- 그렇다면 해당 GT에 대해서는 training 중에만 candidate budget을 늘려야 한다.
- 이 추가 candidate는 detector family마다 다른 형태를 갖지만, 상위 논리는 동일하다.

즉 Candidate Densification은 `hard GT 주변의 관찰 기회`를 인위적으로 늘리는 training-time intervention이다.

## 2. Unified View

GT $g$에 대해 현재 step의 candidate 집합을 $C_t(g)$라 하자. candidate coverage를 다음처럼 정의한다.

$$
u_t(g) = \max_{c \in C_t(g)} \mathrm{IoU}(c, b_g)
$$

정답 class coverage는 다음처럼 정의한다.

$$
v_t(g) = \max_{c \in C_t(g),\; y_c = y_g} \mathrm{IoU}(c, b_g)
$$

Candidate Densification의 기본 트리거는 아래와 같이 둘 수 있다.

$$
z_t(g) =
\mathbb{1}\left[
\phi_t(g) \ge \tau_{\phi}
\;\lor\;
u_t(g) < \tau_u
\;\lor\;
\text{failure\_type}(g, t) \in \mathcal{D}
\right]
$$

여기서:

- $\phi_t(g)$: severity
- $\tau_{\phi}$: hard GT threshold
- $\tau_u$: coverage threshold
- $\mathcal{D}$: densification에 특히 적합한 failure type 집합

권장 $\mathcal{D}$:

- `candidate_missing`
- `loc_near_miss`
- `score_suppression`

## 3. Densification Budget

dense target으로 선택된 GT는 추가 candidate 수 $k_t(g)$를 가진다.

$$
k_t(g)
=
k_0 + \left\lfloor \gamma \cdot \phi_t(g) \right\rfloor
$$

여기서:

- $k_0$: 기본 추가 candidate 수
- $\gamma$: severity에 따른 확장 계수

예시:

- mild hard GT: $k_t(g)=2$
- chronic relapse GT: $k_t(g)=4$ 또는 $6$

이 budget은 detector별 adapter가 실제 candidate primitive로 바꾼다.

## 4. Failure-aware Densification Policy

모든 failure type에 동일한 densification을 주는 것은 비효율적이다. 아래처럼 failure-aware policy를 권장한다.

| Failure type | Densification 방향 |
|---|---|
| `candidate_missing` | 공간적으로 더 많은 candidate 생성 |
| `loc_near_miss` | GT 주변 localization refinement candidate 추가 |
| `cls_confusion` | 동일 위치 candidate는 유지하되 class-separating supervision 강화 |
| `score_suppression` | ranking 이전 단계 candidate 수와 positive quota 증가 |
| `nms_suppression` | 중복 candidate 분석 후 soft selection 기반 auxiliary supervision |

즉 Candidate Densification은 단순히 candidate 개수를 늘리는 것이 아니라, `어떤 형태의 candidate를 늘릴지`를 failure type에 맞춰 조절한다.

## 5. FCOS Adapter

FCOS에서 candidate는 point-based assignment와 FPN level별 predictions로 해석할 수 있다.

### 5.1 Base idea

hard GT에 대해 다음 중 하나 또는 여러 개를 수행한다.

1. positive assignment region 확대
2. center sampling radius 완화
3. 추가 auxiliary positive points 생성
4. adjacent FPN level까지 positive 허용

### 5.2 Auxiliary positive point set

GT $g$의 center를 $c_g$라 할 때, dense positive set $\mathcal{P}_g^{\mathrm{dense}}$를 다음처럼 생성할 수 있다.

$$
\mathcal{P}_g^{\mathrm{dense}}
=
\left\{
p \in \mathcal{P}
\;\middle|\;
\|p - c_g\|_{\infty} \le r_g
\right\}
$$

여기서 $r_g$는 severity에 따라 결정한다.

$$
r_g = r_0 \cdot \left(1 + \alpha \phi_t(g)\right)
$$

구현 포인트:

- 기존 positive assignment는 유지
- dense positive는 auxiliary branch에서만 사용
- base branch와 dense branch를 분리하면 안정성이 좋다

### 5.3 Level relaxation

기존 FCOS는 box size에 따라 FPN level을 제한한다. hard GT에 대해서는 인접 level까지 허용 범위를 넓힐 수 있다.

예시:

- 원래 `P4`만 허용되던 GT가 있으면 `P3-P5`까지 허용
- 단, dense branch에서만 허용

### 5.4 FCOS auxiliary loss

base loss 외에 dense points에 대한 auxiliary loss를 둔다.

$$
\mathcal{L}_{\mathrm{dense}}^{\mathrm{FCOS}}
=
\frac{1}{|\mathcal{P}_{\mathrm{dense}}|}
\sum_{p \in \mathcal{P}_{\mathrm{dense}}}
\Bigl(
\mathcal{L}_{\mathrm{cls}}(p) + \mathcal{L}_{\mathrm{reg}}(p) + \mathcal{L}_{\mathrm{ctr}}(p)
\Bigr)
$$

권장사항:

- dense branch weight를 작게 시작
- hard GT에만 적용
- positive 수 폭증을 막기 위해 `max_dense_points_per_gt` 설정

## 6. Faster R-CNN Adapter

Faster R-CNN에서 candidate densification은 proposal과 ROI sampling 차원에서 해석하는 것이 자연스럽다.

### 6.1 Base idea

hard GT에 대해:

1. GT 주변 jittered proposal을 추가로 주입
2. ROI sampling에서 positive quota를 보장
3. proposal score가 낮더라도 hard GT 근처 후보는 훈련 시 보존

### 6.2 Proposal injection

GT box $b_g$에서 $k_t(g)$개의 proposal을 생성한다.

$$
\tilde{b}_{g,j}
=
\mathrm{jitter}(b_g; \sigma_x, \sigma_y, \sigma_w, \sigma_h)
$$

생성 규칙:

- center perturbation
- width/height perturbation
- image boundary clipping
- 너무 작은 box 제거

이 proposal을 RPN post-NMS proposal set 또는 ROI input set에 concat한다.

### 6.3 Positive quota guarantee

ROI sampling 단계에서 hard GT와 매칭되는 proposal이 일정 수 이상 유지되도록 강제한다.

예시 정책:

- hard GT당 최소 `m`개 positive ROI 보장
- 일반 positive와 별도 bucket 사용

이는 score suppression이나 proposal sparsity 문제에 특히 효과적이다.

### 6.4 Faster R-CNN dense loss

proposal injection 자체로도 효과를 볼 수 있지만, 보조 ROI loss를 함께 쓰면 더 안정적이다.

$$
\mathcal{L}_{\mathrm{dense}}^{\mathrm{RCNN}}
=
\frac{1}{|\mathcal{R}_{\mathrm{dense}}|}
\sum_{r \in \mathcal{R}_{\mathrm{dense}}}
\Bigl(
\mathcal{L}_{\mathrm{roi\_cls}}(r) + \mathcal{L}_{\mathrm{roi\_box}}(r)
\Bigr)
$$

여기서 $\mathcal{R}_{\mathrm{dense}}$는 주입된 proposal 중 hard GT와 연결된 subset이다.

## 7. DINO Adapter

DINO는 query-based detector이므로 candidate densification을 `recovery query injection`으로 해석한다.

### 7.1 Base idea

hard GT마다 추가 query를 몇 개 배정한다.

- GT 중심 reference point 근처에 query 초기화
- decoder supervision을 hard GT에 집중
- 기존 query set은 유지

### 7.2 Recovery query generation

GT box $b_g$의 center-normalized reference point를 $r_g$라 할 때, 추가 query reference는 다음처럼 생성할 수 있다.

$$
\tilde{r}_{g,j}
=
r_g + \epsilon_j,
\qquad
\epsilon_j \sim \mathcal{N}(0, \sigma^2 I)
$$

severity가 클수록 $k_t(g)$를 늘릴 수 있다.

### 7.3 Query budgeting

DINO는 전체 query 수가 고정되어 있는 경우가 많으므로 두 가지 방식이 있다.

1. extra recovery query를 별도 budget으로 추가
2. 기존 denoising query 예산 일부를 hard GT 쪽으로 재배치

1번이 구현은 단순하지만 메모리 비용이 늘고, 2번은 budget 보존이 쉽다.

### 7.4 DINO dense loss

recovery query에도 일반 decoder supervision을 동일하게 적용할 수 있다.

$$
\mathcal{L}_{\mathrm{dense}}^{\mathrm{DINO}}
=
\frac{1}{|\mathcal{Q}_{\mathrm{dense}}|}
\sum_{q \in \mathcal{Q}_{\mathrm{dense}}}
\Bigl(
\mathcal{L}_{\mathrm{cls}}(q) + \mathcal{L}_{\mathrm{box}}(q)
\Bigr)
$$

핵심은 hard GT 주변 query density를 높여 matching 기회를 더 많이 만드는 것이다.

## 8. Generic Planning API

Candidate Densification은 planning 단계와 injection 단계로 분리하는 것이 좋다.

```python
@dataclass(slots=True)
class DenseTarget:
    gt_uid: str
    image_id: str
    class_id: int
    bbox: Tensor
    failure_type: str
    severity: float
    budget: int

@dataclass(slots=True)
class DensePlan:
    targets: list[DenseTarget]
```

planner는 detector family와 무관한 로직을 가진다.

1. MDMB++에서 dense target 후보를 가져온다.
2. severity threshold, recency, per-image cap을 적용한다.
3. detector adapter에 넘길 `DensePlan`을 만든다.

adapter는 detector-specific primitive로 변환한다.

## 9. Training Loop Placement

Candidate Densification은 forward 이전에 dense plan이 준비되어 있어야 한다.

권장 순서:

1. batch target 로드
2. `dense_plan = densifier.plan(mdmbpp, targets)`
3. model forward 시 `dense_plan` 전달
4. detector adapter가 dense candidate 생성
5. base loss + dense loss 계산

즉 post-step에서 memory를 갱신하고, 다음 step의 forward에서 densification에 사용한다.

## 10. Stability Rules

Candidate Densification은 공격적인 방법이므로 아래 안전장치를 두는 것이 좋다.

1. dense target 수에 image-level cap 설정
2. GT당 extra candidate 수에 상한 설정
3. dense loss weight는 warmup 이후 천천히 증가
4. very easy GT에는 절대 적용하지 않음

예시:

- `max_dense_targets_per_image = 5`
- `max_extra_candidates_per_gt = 6`
- `lambda_dense` warmup: 0 -> target over 1-2 epochs

## 11. Why It Can Improve mAP Aggressively

Candidate Densification은 recall bottleneck을 직접 건드린다. hard GT에 대해:

- 더 많은 positive assignment가 생성되고
- proposal/query sparsity가 줄어들며
- ranking 전에 살아남을 기회가 늘어난다

즉 reweighting이 "있는 candidate를 더 세게 학습"하는 접근이라면, densification은 "학습 가능한 candidate를 더 많이 만든다"는 점에서 훨씬 공격적이다.

특히 아래 상황에서 효과가 클 가능성이 높다.

- small object
- crowded scene
- low recall baseline
- query/proposal budget이 빡빡한 구조

## 12. Minimal Implementation Path

현 저장소 기준으로는 아래 순서를 권장한다.

1. FCOS에 auxiliary positive point densification 추가
2. Faster R-CNN에 proposal injection 추가
3. DINO에 recovery query 추가

첫 단계에서 FCOS adapter만 잘 구현해도 UMR의 핵심 아이디어를 빠르게 검증할 수 있다.
