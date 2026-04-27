---
title: "DHM 기반 Object Detection False Negative 저감 방법론 제안"
subtitle: "FN_LOC, FN_CLS, FN_BG 각각에 대한 독립적이면서 통합 가능한 novelty 설계"
author: ""
date: "2026-04-27"
mainfont: "Noto Sans CJK KR"
CJKmainfont: "Noto Sans CJK KR"
monofont: "Noto Sans Mono CJK KR"
fontsize: 10pt
geometry: margin=19mm
colorlinks: true
linkcolor: blue
urlcolor: blue
toc: true
toc-depth: 2
---

# 1. 문제 정의와 결론 요약

연구 상황은 다음과 같이 정리할 수 있습니다.

- Baseline detector: Faster R-CNN, FCOS, DINO.
- 추가 모듈: Detection Hysteresis Memory, DHM.
- DHM 저장 단위: GT별 history.
- 주요 평가 상태: TP, FN_BG, FN_CLS, FN_LOC, FN_MISS.
- 최종 epoch 기준 전체 FN 중 비율: FN_LOC 52%, FN_CLS 28%, FN_BG 19% 내외.

이 비율이면 단순히 "missed object recall"을 올리는 접근보다, **GT별 실패 유형을 분리한 후 각 실패 유형에 다른 training signal을 되먹임하는 방식**이 더 설득력 있습니다. 특히 FN_LOC가 52%로 가장 크므로 localization repair가 1순위이고, FN_CLS는 class-confusion repair, FN_BG는 objectness/background assignment repair로 독립 설계하는 것이 논문 novelty를 만들기 쉽습니다.

이 문서에서 제안하는 세 방법론은 다음과 같습니다.

| Target error | 제안 방법명 | 핵심 아이디어 | 주 효과 |
|---|---|---|---|
| FN_LOC | **HLRT: Hysteretic Localization Residual Transport** | 반복 localization 실패 GT의 box residual 분포를 DHM에 저장하고, 다음 학습에서 residual 기반 hard localization replay와 side-aware loss를 주입 | FN_LOC 감소, AP75/AR 상승 |
| FN_CLS | **CEPR: Confusion-Edge Prototype Rectification** | GT 근처 후보는 있지만 correct-class score가 약한 GT에 대해 confusion edge 또는 true-class evidence 부족을 저장하고, instance-conditioned margin/contrastive/prototype loss로 class decision boundary를 국소 보정 | FN_CLS 감소, class confusion matrix 개선 |
| FN_BG | **LFR: Latent Foreground Revival** | GT가 background/no-object로 밀린 경우를 latent foreground로 재활성화하고, objectness resurrection head와 hysteretic assignment rescue로 foreground 후보를 되살림 | FN_BG 감소, recall/AR 상승 |

세 방법은 모두 DHM의 "hysteresis" 성질을 핵심 novelty로 사용합니다. 즉, 한 epoch의 우발적 오분류에 반응하지 않고, GT별 실패 점수가 on-threshold를 넘을 때만 개입하고 off-threshold 아래로 내려가면 개입을 중단합니다. 이 설계는 hard example mining과 다릅니다. Hard example mining은 보통 현재 batch 또는 현재 loss 중심이고, 여기서는 **GT identity, error type, temporal trajectory**가 intervention을 결정합니다.

# 2. 관련 연구 대비 novelty 포지셔닝

Faster R-CNN은 RPN이 object bounds와 objectness를 동시에 예측하여 proposal을 생성하는 two-stage detector입니다 [Ren2015]. FCOS는 anchor/proposal 없이 per-pixel prediction 방식으로 detection을 수행하는 anchor-free one-stage detector입니다 [Tian2019]. DINO는 DETR 계열 end-to-end detector로, denoising anchor boxes, mixed query selection, look-forward-twice box prediction 등을 사용합니다 [Zhang2023]. 이 세 baseline은 구조가 달라서 동일한 add-on 모듈을 설계하려면 proposal, dense point, transformer query를 모두 포괄하는 추상화가 필요합니다.

TIDE는 detection error를 classification, localization, background, missed 등으로 분해하여 mAP만으로는 보이지 않는 실패 원인을 분석할 수 있음을 보였습니다 [Bolya2020]. Focal Loss는 dense detector의 foreground-background imbalance를 다루기 위해 easy negative의 loss를 낮추는 방향으로 설계되었습니다 [Lin2017]. ATSS는 positive/negative sample assignment가 detector 성능에 중요함을 보이고 adaptive training sample selection을 제안했습니다 [Zhang2020]. GFL은 localization quality와 classification을 joint representation으로 다루고, box location의 분포적 표현을 도입했습니다 [Li2020]. TOOD는 classification과 localization branch 사이의 spatial misalignment를 task-aligned learning으로 줄이려 했습니다 [Feng2021].

본 제안의 차별점은 다음입니다.

1. **Error-type-aware memory intervention**: 단일 hard loss가 아니라 FN_LOC, FN_CLS, FN_BG별로 서로 다른 intervention을 수행합니다.
2. **GT-level temporal hysteresis**: 현재 batch의 high loss sample이 아니라, 동일 GT가 여러 epoch 동안 어떤 실패 유형으로 남는지 추적합니다.
3. **Cross-detector plug-in**: Faster R-CNN의 proposal/RoI, FCOS의 point/FPN level, DINO의 query/denoising token에 동일 원리를 다른 hook으로 삽입합니다.
4. **Oracle-like diagnosis를 training signal로 변환**: error analysis에서 끝나지 않고, DHM에 저장된 실패 원인을 box residual, confusion edge, latent foreground prior로 바꿔 loss에 직접 연결합니다.

논문화 관점의 핵심 주장은 다음과 같습니다.

> "Object detector의 FN은 단일 recall failure가 아니라 GT별 temporal failure mode로 분해될 수 있으며, DHM은 이 failure mode를 memory-conditioned training signal로 변환하여 localization, classification, foregroundness를 선택적으로 복구한다."

# 3. 공통 DHM 정의

## 3.1 GT별 memory state

각 GT를 $g=(i, y_g, b_g)$로 둡니다. 여기서 $i$는 image id, $y_g$는 class label, $b_g$는 GT box입니다. 현재 구현의 `DHMRecord`는 `last_state`, `last_score`, `last_iou`, count, streak, recovery/forgetting, instability를 저장합니다. 아래 schema는 HLRT/CEPR/LFR까지 확장할 때 필요한 logical memory입니다.

```text
DHM[g] = {
  y_g, b_g, image_id,
  status_hist: [TP, FN_BG, FN_CLS, FN_LOC, FN_MISS],
  e_loc, e_cls, e_bg, e_miss,
  best_iou, best_pred_cls, best_score,
  best_near_target_score, best_near_wrong_score,
  logit_gap, class_entropy,
  loc_residual_ema, loc_residual_cov,
  feat_ema, feat_queue,
  source: {fpn_level | anchor_id | point_xy | proposal_id | query_id}
}
```

`feat_ema`는 RoI feature, FCOS location feature, DINO decoder query embedding 등 detector별 feature를 공통 embedding으로 project한 값입니다. 메모리 비용을 줄이기 위해 full feature queue 대신 EMA vector와 top-K 실패 snapshot만 저장합니다.

## 3.2 Hysteresis score

각 error type $e \in \{loc, cls, bg, miss\}$에 대해 GT별 temporal score를 둡니다.

$$
E_e^t(g) = \beta E_e^{t-1}(g) + (1-\beta) \mathbf{1}[status_t(g)=e]
$$

권장값은 $\beta=0.85$에서 $0.95$입니다. 활성화는 Schmitt trigger처럼 on/off threshold를 다르게 둡니다.

$$
active_e(g)=
\begin{cases}
1, & E_e^t(g) > \tau_{on} \\
0, & E_e^t(g) < \tau_{off} \\
active_e^{t-1}(g), & \text{otherwise}
\end{cases}
$$

권장 초기값은 $\tau_{on}=0.40$, $\tau_{off}=0.20$입니다. 이 구조가 "hysteresis"의 기술적 정의이며, 한 번의 noisy error에 과잉 반응하는 것을 막습니다.

## 3.3 Error labeling 규칙

현재 코드베이스의 DHM mining은 `modules/nn/dhm.py`의 `_assign_detection_state()`를 기준으로 GT별 상태를 판정합니다. 이 판정은 raw objectness/no-object logit을 직접 보지 않고, 최종 detection의 `boxes`, `labels`, `scores`만 사용합니다. 따라서 문서의 FN_BG는 실제 구현에서는 "no-object logit 우세"가 아니라 **GT 근처에서 confidence 있는 후보가 살아나지 못한 상태의 proxy**로 해석해야 합니다.

현재 사용되는 threshold는 다음과 같습니다.

| 이름 | 의미 | 기본값 |
|---|---:|---:|
| `tau_iou` | TP로 인정할 IoU threshold | 0.5 |
| `tau_tp` | TP로 인정할 correct-class score threshold | 0.3 |
| `tau_near` | GT 주변 후보로 볼 최소 IoU | 0.3 |
| `tau_bg_score` | foreground evidence가 약하다고 볼 score threshold | 0.1 |
| `tau_cls_evidence` | correct-class localization evidence를 계산할 score threshold | 0.3 |
| `tau_loc_score` | FN_LOC로 볼 correct-class score threshold | 0.3 |

코드 기준 판정 순서는 다음과 같습니다.

1. Prediction box가 하나도 없으면 `FN_MISS`.
2. 같은 class prediction 중 `IoU >= tau_iou`이고 `score >= tau_tp`인 후보가 있으면 `TP`.
3. 모든 prediction이 GT와 멀고(`best_any_iou < tau_near`), 같은 class prediction score도 낮으면(`best_target_score < tau_bg_score`) `FN_MISS`.
4. GT 근처 후보(`IoU >= tau_near`) 중 같은 class score와 다른 class score가 모두 낮으면 `FN_BG`.
5. GT 근처 후보는 있지만 같은 class near score가 TP score threshold보다 낮으면 `FN_CLS`.
6. 같은 class score가 충분하지만(`best_target_score >= tau_loc_score`) 같은 class localization evidence의 IoU가 부족하면(`best_target_iou < tau_iou`) `FN_LOC`.
7. 위 조건에 걸리지 않는 애매한 false negative는 `FN_BG`.

상태별 의미를 코드 기준으로 다시 쓰면 다음과 같습니다.

- `TP`: class가 맞고, IoU와 score가 모두 충분한 detection이 존재하는 상태.
- `FN_LOC`: correct-class evidence는 있지만 box alignment가 `tau_iou`에 미치지 못하는 상태.
- `FN_CLS`: GT 근처에 detection evidence는 있으나 correct-class near score가 약한 상태. 문서상의 "IoU는 충분하지만 class만 틀림"보다 넓은 정의입니다.
- `FN_BG`: GT 근처에서 correct/wrong class 모두 confidence가 약하거나, 다른 조건으로 설명되지 않는 background-suppressed proxy 상태.
- `FN_MISS`: prediction이 없거나, GT 근처 후보도 없고 correct-class evidence도 약한 상태.

주의할 점은 본 문서의 FN_BG가 TIDE의 "background false positive"와 동일하지 않다는 것입니다. 여기서는 **GT가 background로 처리되어 검출되지 않은 false negative**를 의미하며, 현재 코드에서는 final detection score 기반 proxy로 판정합니다.

# 4. 방법론 1: HLRT - Hysteretic Localization Residual Transport

## 4.1 Target

HLRT는 FN_LOC를 줄이기 위한 방법입니다. 현재 코드 기준 FN_LOC는 detector가 해당 GT의 class identity는 어느 정도 알고 있지만, 같은 class 후보의 box alignment가 `tau_iou`를 넘지 못한 경우입니다. 현재 비율이 52%이면 다음과 같은 문제가 있을 가능성이 큽니다.

- assignment된 positive sample이 boundary를 충분히 학습하지 못함.
- class branch와 localization branch의 optimal point/proposal/query가 어긋남.
- 작은 객체, occlusion, elongated object에서 특정 side residual이 반복됨.
- detector가 class score는 높게 주지만 localization quality가 낮아 NMS/ranking에서 밀림.

HLRT의 핵심은 **반복적으로 실패한 localization residual을 다음 epoch의 hard localization probe로 운반**하는 것입니다.

## 4.2 Localization residual memory

FN_LOC 상태인 GT $g$에 대해 class가 맞는 prediction 중 IoU가 가장 큰 box $\hat{b}_g^t$를 찾습니다.

$$
\hat{b}_g^t = \arg\max_{p \in P_i^t,\; \hat{y}_p=y_g} IoU(b_p, b_g)
$$

normalized side residual을 다음과 같이 저장합니다.

$$
r_g^t = \left[
\frac{x_1^g-x_1^p}{w_g},
\frac{y_1^g-y_1^p}{h_g},
\frac{x_2^g-x_2^p}{w_g},
\frac{y_2^g-y_2^p}{h_g}
\right]
$$

DHM은 residual EMA와 covariance를 업데이트합니다.

$$
\bar{r}_g^t = \alpha \bar{r}_g^{t-1} + (1-\alpha) r_g^t
$$

$$
\Sigma_g^t = \alpha \Sigma_g^{t-1} + (1-\alpha)(r_g^t-\bar{r}_g^t)(r_g^t-\bar{r}_g^t)^\top
$$

side dominance는 다음처럼 계산합니다.

$$
d_g = softmax(|\bar{r}_g| / T_r)
$$

$d_g$가 큰 side는 해당 GT에서 반복적으로 흔들리는 boundary입니다.

## 4.3 Hysteretic residual replay

활성화 조건은 $active_{loc}(g)=1$입니다. 활성화된 GT에 대해 residual distribution에서 hard box를 생성합니다.

$$
\tilde{r}_{g,k} \sim \mathcal{N}(\bar{r}_g, \Sigma_g + \sigma^2 I)
$$

$\tilde{r}_{g,k}$를 GT box 주변의 perturbed box $\tilde{b}_{g,k}$로 변환합니다. 변환 방향은 "이전 실패처럼 어긋난 box"를 재현하도록 둡니다.

$$
\tilde{b}_{g,k} = T^{-1}(b_g, -\tilde{r}_{g,k})
$$

그리고 다음 조건을 만족하는 box만 hard localization replay sample로 사용합니다.

$$
\tau_{near} \le IoU(\tilde{b}_{g,k}, b_g) < \tau_{iou}
$$

이 조건은 이미 쉬운 positive인 box를 제외하고, 실제 FN_LOC에 가까운 hard localization 영역만 학습시킵니다. 현재 DHM 코드의 FN_LOC 판정은 correct-class score evidence도 함께 보므로, replay source GT는 `best_target_score >= tau_loc_score`를 만족한 DHM `FN_LOC` record로 제한하는 것이 구현과 가장 잘 맞습니다.

## 4.4 Side-aware Boundary Distribution Loss

기본 box regression loss를 다음처럼 바꿉니다.

$$
\mathcal{L}_{HLRT} =
\lambda_{iou}(1+\gamma E_{loc}(g))\mathcal{L}_{GIoU}(\hat{b}, b_g) +
\lambda_{side}\sum_{j \in \{l,t,r,b\}} (1+\eta d_{g,j})\mathcal{L}_{side,j}
$$

여기서 $\mathcal{L}_{side,j}$는 detector에 따라 Smooth-L1, Distribution Focal Loss, L1 distance 등으로 구현합니다. 핵심은 모든 side를 동일하게 보지 않고, DHM이 저장한 반복 residual side에 더 큰 gradient를 주는 것입니다.

## 4.5 Localization-quality aligned classification gate

FN_LOC에서는 class는 맞지만 box가 부정확합니다. 이때 classification branch를 과도하게 벌주면 FN_CLS나 FN_BG로 전이될 수 있습니다. 따라서 class target과 localization-quality target을 분리합니다.

$$
q_g^t = EMA(IoU(\hat{b}_g^t, b_g))
$$

classification score는 class correctness를 유지하고, ranking score 또는 quality score만 $q_g^t$에 맞춥니다. FCOS/GFL 계열에서는 quality-aware class target으로 연결하고, Faster R-CNN에서는 별도 IoU quality head를 추가할 수 있습니다. DINO에서는 matching cost와 decoder output score에 quality prior를 추가합니다.

## 4.6 Detector별 삽입 위치

### Faster R-CNN

- RPN 단계: 활성 GT 주변에 residual replay proposal을 top proposal candidate로 추가합니다.
- RoI sampler: replay proposal을 positive RoI로 강제 포함합니다.
- RoI box head: side-aware loss를 bbox regression loss에 더합니다.
- Optional: IoU quality head를 추가하여 class score와 box quality를 분리합니다.

### FCOS

- FPN level: 과거 실패가 발생한 level을 DHM에 저장하고 동일 level 또는 인접 level에서 replay point를 생성합니다.
- Center sampling: FN_LOC GT에 대해서만 center radius를 약간 확장합니다.
- Regression head: l/t/r/b distance loss에 side dominance weight를 적용합니다.
- Centerness/quality: localization quality target을 hysteresis score로 보정합니다.

### DINO

- Denoising input: residual replay box를 noisy anchor box로 삽입합니다.
- Hungarian matching: 활성 GT에 대해 memory query prior를 matching cost에 추가합니다.
- Decoder: 과거 FN_LOC query embedding을 momentum projection하여 초기 query seed로 사용할 수 있습니다.
- Box loss: repeated residual side에 가중치를 부여합니다.

## 4.7 Novelty claim

HLRT는 단순 IoU loss 강화가 아닙니다. novelty는 다음 3개입니다.

1. **GT별 temporal residual distribution**: 실패한 box residual을 GT identity에 붙여 EMA/covariance로 모델링합니다.
2. **Residual-to-replay transport**: 과거 실패 residual을 다음 학습의 hard localization proposal/point/query로 변환합니다.
3. **Side-specific hysteretic loss**: 반복적으로 흔들린 side만 선택적으로 강화하여 localization loss를 세분화합니다.

## 4.8 Ablation

필수 ablation은 다음입니다.

- Baseline + DHM logging only.
- HLRT without hysteresis, 즉 current epoch FN_LOC만 사용.
- HLRT without residual replay.
- HLRT without side-aware loss.
- Random residual replay, 즉 residual memory 대신 random jitter.
- Detector별: Faster R-CNN, FCOS, DINO 각각에서 동일 protocol.

주요 지표는 FN_LOC ratio, AP75, AP50:95, AR100, size별 AP, crowd/occlusion subset recall입니다.

# 5. 방법론 2: CEPR - Confusion-Edge Prototype Rectification

## 5.1 Target

CEPR은 FN_CLS를 줄이기 위한 방법입니다. 현재 코드 기준 FN_CLS는 GT 근처 후보는 있지만 correct-class near score가 충분하지 않은 상태입니다. 즉, feature는 object region 근처의 evidence를 보고 있으나 class decision이 true class로 충분히 올라오지 못한 경우입니다. 이 경우 box loss를 더 키우는 것은 비효율적이며, class confusion 또는 class evidence 부족을 직접 공격해야 합니다.

핵심은 **GT별 class confusion edge**입니다. 예를 들어 실제 class가 `dog`인데 반복적으로 `cat`으로 예측된다면, CEPR은 단순히 dog CE loss를 키우는 것이 아니라 `dog -> cat` confusion edge를 memory에 저장하고 그 edge에 국소적인 margin/contrastive/prototype 보정을 적용합니다.

## 5.2 Confusion-edge memory

FN_CLS 상태인 GT $g$에 대해 GT 근처(`IoU \ge \tau_{near}`)의 wrong-class prediction $p$를 찾습니다. 기존 문서처럼 `IoU >= tau_iou`만 요구하면 현재 코드의 FN_CLS보다 좁아지므로, 구현 기준에서는 `tau_near` 이상 후보를 confusion edge 후보로 둡니다.

$$
p_g^t = \arg\max_{p \in P_i^t,\; IoU(b_p,b_g) \ge \tau_{near},\; \hat{y}_p \ne y_g} s_p
$$

예측 class를 $k=\hat{y}_{p_g^t}$라고 하면 confusion edge는 $y_g \rightarrow k$입니다. 다만 현재 코드 기준 `FN_CLS`는 wrong-class 후보가 반드시 존재해야 하는 상태가 아니므로, `tau_near` 이상 wrong-class 후보가 없으면 CEPR은 `confused_class = null`로 두고 true-class evidence boosting/prototype attraction만 적용합니다. DHM은 다음을 저장합니다.

```text
CE[g] = {
  true_class: y_g,
  confused_class: k | null,
  logit_gap: z_k - z_y,
  feature: f_p,
  entropy: H(softmax(z)),
  iou: IoU(b_p, b_g),
  edge_count[y_g, k]  # k가 있을 때만 사용
}
```

class별 prototype과 edge별 confusion prototype을 유지합니다.

$$
\mu_y^+ = EMA( f_{TP, y} )
$$

$$
\mu_{y \rightarrow k}^{conf} = EMA( f_{FN\_CLS, y \rightarrow k} )
$$

$\mu_y^+$는 정상적으로 맞춘 class prototype이고, $\mu_{y \rightarrow k}^{conf}$는 `y인데 k로 오인된 feature`의 prototype입니다.

## 5.3 Instance-conditioned confusion margin

활성화 조건은 $active_{cls}(g)=1$입니다. CEPR은 near wrong-class 후보가 있을 때 wrong class $k$에 대해서만 targeted margin을 부여합니다. wrong-class 후보가 없고 correct-class score만 약한 `FN_CLS`에서는 margin term을 끄고 true-class CE/prototype attraction term만 사용합니다.

$$
\mathcal{L}_{margin} =
\max(0, m_g - (z_{y_g} - z_k))
$$

margin은 GT별 hysteresis와 edge frequency로 조절합니다.

$$
m_g = m_0 + \eta E_{cls}(g) + \rho \log(1 + n_{y_g \rightarrow k})
$$

이 방식은 모든 negative class를 무차별적으로 밀어내지 않고, 실제로 반복 confusion을 만든 class만 겨냥합니다. 따라서 class imbalance나 long-tail class에서 불필요한 over-suppression을 줄일 수 있습니다.

## 5.4 Edge-constrained supervised contrastive loss

CEPR은 feature space에서도 confusion edge를 보정합니다. anchor는 FN_CLS feature $f_g$입니다. positive set은 같은 true class의 TP prototype 및 같은 GT의 historical feature입니다. negative set은 confused class prototype과 edge confusion prototype입니다.

$$
\mathcal{P}(g)=\{\mu_{y_g}^+, f_{g}^{hist}\}
$$

$$
\mathcal{N}(g)=\{\mu_k^+, \mu_{y_g \rightarrow k}^{conf}\}
$$

loss는 다음과 같습니다.

$$
\mathcal{L}_{CEPR}^{con} = -
\log \frac{\sum_{p \in \mathcal{P}(g)} \exp(sim(f_g,p)/\tau)}
{\sum_{p \in \mathcal{P}(g)} \exp(sim(f_g,p)/\tau) +
\sum_{n \in \mathcal{N}(g)} \exp(sim(f_g,n)/\tau)}
$$

전체 CEPR loss는 다음입니다.

$$
\mathcal{L}_{CEPR} =
\lambda_{ce}\mathcal{L}_{CE} +
\lambda_{m}E_{cls}(g)\mathcal{L}_{margin} +
\lambda_{con}E_{cls}(g)\mathcal{L}_{CEPR}^{con}
$$

## 5.5 Counterfactual class replay

FN_CLS는 localization이 이미 맞으므로, 같은 box 또는 query를 유지한 상태에서 class branch만 다시 학습시키는 것이 좋습니다.

- Stored feature $f_g^{hist}$를 stop-gradient 또는 momentum feature로 replay합니다.
- box branch는 detach합니다.
- classifier/projection head만 업데이트합니다.
- augmentation은 feature-level jitter, MixStyle, small dropout 정도로 제한합니다.

이 구조는 "box는 이미 맞았으니 class decision만 고친다"는 error-specific learning을 명확히 보여줍니다.

## 5.6 Detector별 삽입 위치

### Faster R-CNN

- RoI head의 pooled feature를 CEPR feature로 사용합니다.
- FN_CLS RoI는 positive RoI로 유지하되, bbox regression gradient는 detach하고 classifier/projection head에만 CEPR loss를 적용합니다.
- class-specific bbox regression을 쓰는 경우, wrong class bbox branch 업데이트가 섞이지 않도록 true class branch만 업데이트합니다.

### FCOS

- GT와 IoU가 가장 높은 predicted location의 FPN feature를 사용합니다.
- 해당 point의 classification logits에 targeted margin을 적용합니다.
- centerness/regression branch는 유지하거나 detach합니다.

### DINO

- Hungarian matching에서 GT와 높은 IoU를 가진 query embedding을 사용합니다.
- decoder query embedding에 contrastive projection head를 추가합니다.
- no-object logit은 건드리지 않고, confused class $k$와 true class $y$ 사이의 margin만 직접 수정합니다.

## 5.7 Novelty claim

CEPR의 novelty는 다음입니다.

1. **Global class confusion matrix가 아니라 GT-level class-evidence failure**를 저장합니다.
2. **GT-near but weak true-class evidence sample만 사용**하므로 classification repair가 완전한 background miss와 분리됩니다.
3. **Wrong class targeted margin**을 적용하여 모든 negative class를 동일하게 밀지 않습니다.
4. **Historical feature replay**를 통해 mini-batch에 같은 confusion pair가 없어도 안정적으로 class boundary를 보정합니다.

## 5.8 Ablation

- CEPR without prototype.
- CEPR without targeted margin, 즉 standard CE만 강화.
- CEPR without historical replay.
- Global confusion matrix 기반 margin과 비교.
- All negative contrastive vs edge-constrained contrastive 비교.
- Rare class / visually similar class subset 분석.

주요 지표는 FN_CLS ratio, class confusion matrix, per-class AP, macro AP, long-tail AP, calibration ECE입니다.

# 6. 방법론 3: LFR - Latent Foreground Revival

## 6.1 Target

LFR은 FN_BG를 줄이기 위한 방법입니다. 현재 코드 기준 FN_BG는 raw no-object logit을 직접 본 결과가 아니라, GT 근처에서 correct/wrong class 모두 confidence가 낮거나 다른 FN 조건으로 설명되지 않는 상태입니다. 따라서 이 문서의 LFR은 **background/no-object 억제의 직접 관측값**이 아니라, DHM의 score 기반 FN_BG proxy를 foreground 후보 복구 신호로 사용하는 방법입니다.

FN_BG의 원인은 보통 다음 중 하나입니다.

- RPN/anchor/point assignment에서 positive가 충분히 생성되지 않음.
- small/low-contrast/occluded object의 foreground score가 낮음.
- dense detector에서 background imbalance가 너무 강함.
- DINO/DETR 계열에서 query가 해당 GT에 충분히 attention하지 않거나 no-object logit이 우세함.

LFR의 핵심은 배경으로 밀린 GT를 "hard negative"가 아니라 **latent foreground**로 취급하는 것입니다.

## 6.2 Background-rejected memory

FN_BG 상태인 GT $g$에 대해, 현재 코드베이스에서는 final detection 기준으로 GT와 공간적으로 가장 관련 있지만 confidence가 낮은 candidate를 저장합니다. detector가 raw objectness/no-object logit을 노출하는 경우에는 그 값을 추가 feature로 저장할 수 있지만, 기본 DHM 판정 자체는 final detection score 기반입니다.

```text
BR[g] = {
  best_bg_candidate,
  obj_score,
  no_object_logit,
  max_cls_logit,
  center_response,
  feature_patch,
  source_level_or_query,
  spatial_distance_to_gt_center
}
```

candidate 선택 기준은 detector별로 다릅니다.

- Faster R-CNN: GT와 가까운 anchor/proposal 중 objectness가 낮아 background로 간주된 후보.
- FCOS: GT center 또는 center radius 주변의 point 중 foreground score가 낮은 후보.
- DINO: GT에 attention하거나 spatial prior가 가까우나 no-object logit이 우세한 query.

## 6.3 Objectness Resurrection Head

LFR은 class와 분리된 binary foregroundness head를 추가합니다.

$$
p_{obj} = \sigma(h_{obj}(f))
$$

활성 FN_BG GT에 대해서는 latent foreground candidate를 soft positive로 둡니다.

$$
y_{obj}^{LFR}(g) = \min(1, y_{obj}^{base} + \delta E_{bg}(g))
$$

objectness resurrection loss는 다음입니다.

$$
\mathcal{L}_{obj}^{LFR} =
E_{bg}(g) \cdot BCE(p_{obj}, y_{obj}^{LFR})
$$

이 head는 class label을 바로 예측하지 않고 "배경이 아니라 객체다"를 먼저 학습합니다. 이후 class branch가 충분한 signal을 받을 수 있게 candidate를 foreground pool로 되돌립니다.

## 6.4 Hysteretic assignment rescue

FN_BG가 반복되는 GT에 한해 positive assignment를 완화합니다.

$$
\tau_{pos}(g) = \tau_{base} - \Delta_{\tau} E_{bg}(g)
$$

또는 center sampling radius를 확장합니다.

$$
r_{center}(g) = r_{base}(1 + \gamma E_{bg}(g))
$$

중요한 점은 이 완화가 전체 dataset에 적용되지 않는다는 것입니다. 오직 $active_{bg}(g)=1$인 GT에만 적용됩니다. 이 조건이 없으면 false positive가 크게 늘 수 있습니다.

## 6.5 Latent foreground contrast

LFR은 background feature를 두 종류로 분리합니다.

- True background: 어떤 GT와도 공간적으로 관련이 없는 background.
- Latent foreground background: GT 주변이지만 detector가 background로 억제한 feature.

두 prototype을 유지합니다.

$$
\mu_{bg}^{true}=EMA(f_{true\_bg})
$$

$$
\mu_{fg}^{latent}=EMA(f_{FN\_BG})
$$

활성 FN_BG feature $f_g$에 대해 다음 contrastive loss를 적용합니다.

$$
\mathcal{L}_{LFR}^{con} =
-\log \frac{\exp(sim(f_g, \mu_{fg}^{latent})/\tau)}
{\exp(sim(f_g, \mu_{fg}^{latent})/\tau) + \exp(sim(f_g, \mu_{bg}^{true})/\tau)}
$$

이 loss는 foreground를 바로 특정 class로 강제하지 않고, true background와 latent object background를 먼저 분리합니다.

## 6.6 Detector별 삽입 위치

### Faster R-CNN

- RPN objectness head에 LFR objectness loss를 추가합니다.
- 반복 FN_BG GT 주변 anchor를 soft positive로 rescue합니다.
- Proposal sampler에서 rescued proposal을 최소 1개 이상 보장합니다.
- RoI head에서는 class label을 약하게 부여하되, 초기에는 objectness loss weight를 더 크게 둡니다.

### FCOS

- GT 중심 반경 내 point 중 background로 분류된 point를 latent positive로 저장합니다.
- foreground/objectness auxiliary head를 추가하거나 centerness branch를 확장합니다.
- center sampling radius를 active FN_BG GT에 대해서만 확장합니다.
- class loss에는 초기에 낮은 weight를 주고, objectness가 안정화된 뒤 class weight를 올립니다.

### DINO

- no-object logit이 우세했던 query를 DHM에 저장합니다.
- active FN_BG GT에 대해 memory query seed를 decoder input에 추가합니다.
- Hungarian matching cost에 foreground prior를 추가합니다.

$$
C'(q,g)=C(q,g)-\lambda_{bg}E_{bg}(g)\cdot \mathbf{1}[q \in Q_{mem}(g)]
$$

- denoising query에 GT box 중심 기반 low-noise anchor를 추가하되, class target은 초기에 soft label로 둡니다.

## 6.7 False positive 억제 장치

FN_BG를 줄이는 방법은 FP 증가 위험이 큽니다. 따라서 LFR에는 다음 safety gate가 필요합니다.

1. active hysteresis gate: 한두 번의 FN_BG에는 반응하지 않습니다.
2. max rescue budget: image당 rescued candidate 수를 제한합니다.
3. precision-aware decay: validation FP가 늘면 $\lambda_{obj}$와 $\Delta_{\tau}$를 자동 감소시킵니다.
4. ignore/crowd filtering: crowd, truncated, ambiguous GT는 LFR에서 제외하거나 낮은 weight를 줍니다.
5. revival-to-TP decay: GT가 TP로 전환되면 LFR weight를 빠르게 줄입니다.

## 6.8 Novelty claim

LFR의 novelty는 다음입니다.

1. **Background-as-latent-foreground**: GT 주변 background candidate를 hard negative가 아니라 recoverable latent foreground로 정의합니다.
2. **Hysteretic assignment rescue**: positive assignment relaxation을 전체가 아닌 반복 FN_BG GT에만 적용합니다.
3. **Objectness-first repair**: class confusion을 다루기 전에 foregroundness를 독립 복구합니다.
4. **Detector-agnostic memory query/proposal/point revival**: proposal, point, query 기반 detector에 모두 대응합니다.

## 6.9 Ablation

- LFR without objectness head.
- LFR without assignment rescue.
- LFR without latent foreground contrast.
- Hysteresis 없이 모든 FN_BG 즉시 rescue.
- Rescue budget 변화에 따른 recall/precision curve.
- Small object, occlusion, low-contrast subset 분석.

주요 지표는 FN_BG ratio, AR100, AP_small, false positive per image, no-object recall, proposal recall입니다.

# 7. 세 방법론의 통합 controller

세 방법은 독립 실험이 가능하지만, 최종적으로는 하나의 DHM controller에서 error type에 따라 loss를 라우팅하는 것이 가장 좋은 논문 형태입니다.

## 7.1 Total loss

기본 detector loss를 $\mathcal{L}_{det}$라고 하면 전체 loss는 다음입니다.

$$
\mathcal{L}_{total} = \mathcal{L}_{det} +
\lambda_{loc}\mathcal{L}_{HLRT} +
\lambda_{cls}\mathcal{L}_{CEPR} +
\lambda_{bg}\mathcal{L}_{LFR}
$$

각 term은 해당 active GT에 대해서만 계산합니다.

$$
\mathcal{L}_{HLRT}=0 \quad \text{if } active_{loc}(g)=0
$$

$$
\mathcal{L}_{CEPR}=0 \quad \text{if } active_{cls}(g)=0
$$

$$
\mathcal{L}_{LFR}=0 \quad \text{if } active_{bg}(g)=0
$$

## 7.2 우선순위 policy

한 GT가 여러 error score를 동시에 가질 수 있으므로 priority를 둡니다.

권장 priority는 다음입니다.

1. FN_BG active: foreground 후보 자체가 없으므로 LFR 우선.
2. FN_LOC active: class는 맞지만 localization이 부족하므로 HLRT.
3. FN_CLS active: localization이 맞는 candidate가 있으므로 CEPR.
4. FN_MISS active: 초기에는 LFR로 보내고, 후보가 생기면 HLRT/CEPR로 분기.

다만 현재 최종 FN 구성에서 FN_LOC가 52%로 가장 크기 때문에, main experiment에서는 HLRT를 1차 contribution으로 강조하고 CEPR/LFR을 추가 모듈로 확장하는 narrative가 설득력 있습니다.

## 7.3 Training schedule

권장 schedule은 다음입니다.

- Warm-up: 1-3 epoch. DHM logging만 수행하고 intervention은 비활성화합니다.
- Phase 1: LFR 약하게 활성화. foreground 후보를 복구합니다.
- Phase 2: HLRT 활성화. 가장 큰 FN_LOC를 집중 감소시킵니다.
- Phase 3: CEPR 활성화. localized candidate의 class boundary를 정밀 보정합니다.
- Final fine-tuning: all modules on, but rescue budget과 loss weight를 낮춰 FP 증가를 억제합니다.

# 8. 학습 알고리즘 pseudocode

```text
Algorithm: DHM-guided FN Repair Training

Input:
  Detector D_theta
  Training set with stable GT ids
  DHM table M
  Hysteresis thresholds tau_on, tau_off

for epoch t in 1..T:
  for minibatch B:
    predictions = D_theta(B)

    active_loc = select_gt(M, error="FN_LOC")
    active_cls = select_gt(M, error="FN_CLS")
    active_bg  = select_gt(M, error="FN_BG")

    L_det = baseline_detection_loss(predictions, B)
    L_loc = HLRT_loss(predictions, B, M, active_loc)
    L_cls = CEPR_loss(predictions, B, M, active_cls)
    L_bg  = LFR_loss(predictions, B, M, active_bg)

    L_total = L_det + lambda_loc*L_loc
                    + lambda_cls*L_cls
                    + lambda_bg*L_bg

    update theta by gradient descent

  # epoch-end or EMA-model inference
  for each GT g in training set:
    p_star = find_best_related_prediction(D_theta, g)
    status = classify_status(p_star, g)
    update_hysteresis_scores(M[g], status)
    update_error_specific_memory(M[g], p_star, status)
```

실제로는 epoch-end full inference 비용이 크므로, 다음 중 하나를 선택합니다.

- mini-batch online update: 학습 중 현재 batch prediction으로 DHM update.
- EMA teacher inference: 매 N epoch마다 EMA model로 안정적 update.
- hybrid: online update + sparse epoch-end correction.

# 9. 실험 설계

## 9.1 Main comparison

각 baseline에 대해 다음 setting을 비교합니다.

```text
1. Baseline
2. Baseline + DHM logging only
3. Baseline + HLRT
4. Baseline + CEPR
5. Baseline + LFR
6. Baseline + HLRT + CEPR + LFR
```

DHM logging only가 필요한 이유는 메모리 계산 자체가 inference/training pipeline에 영향을 주는지 확인하기 위해서입니다.

## 9.2 Metrics

기본 detection metric:

- AP50:95, AP50, AP75.
- AP_small, AP_medium, AP_large.
- AR1, AR10, AR100.

DHM-specific metric:

- FN_LOC count and ratio.
- FN_CLS count and ratio.
- FN_BG count and ratio.
- Error-to-TP transition rate.
- TP-to-error regression rate.
- Per-class confusion edge reduction.
- Proposal/point/query foreground survival rate.

권장하는 핵심 표는 다음 구조입니다.

| Model | Method | AP | AP75 | AR100 | FN_LOC↓ | FN_CLS↓ | FN_BG↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | baseline | - | - | - | - | - | - |
| Faster R-CNN | +HLRT | - | - | - | - | - | - |
| FCOS | baseline | - | - | - | - | - | - |
| FCOS | +HLRT | - | - | - | - | - | - |
| DINO | baseline | - | - | - | - | - | - |
| DINO | +HLRT | - | - | - | - | - | - |

## 9.3 Error transition matrix

단순 최종 ratio보다 transition matrix가 더 강한 증거가 됩니다.

| Previous epoch | Next TP | Next FN_LOC | Next FN_CLS | Next FN_BG | Next FN_MISS |
|---|---:|---:|---:|---:|---:|
| FN_LOC | - | - | - | - | - |
| FN_CLS | - | - | - | - | - |
| FN_BG | - | - | - | - | - |

논문에서는 "HLRT는 FN_LOC -> TP transition을 증가시켰고, CEPR은 FN_CLS -> TP transition을 증가시켰으며, LFR은 FN_BG -> candidate survival -> TP transition을 증가시켰다"는 식으로 보여주는 것이 좋습니다.

## 9.4 Qualitative visualization

- FN_LOC: 이전 epoch 실패 box residual 화살표와 HLRT 후 box refinement.
- FN_CLS: confusion edge graph. 예: class A -> class B edge weight 감소.
- FN_BG: background로 억제된 GT의 foreground score heatmap이 LFR 후 상승.
- DINO: no-object query attention이 GT region으로 이동하는 attention map.

# 10. 구현 세부 권장값

## 10.1 Hyperparameter 초기값

| Parameter | 권장값 | 설명 |
|---|---:|---|
| $\beta$ | 0.90 | hysteresis EMA |
| $\tau_{on}$ | 0.40 | error intervention 활성화 |
| $\tau_{off}$ | 0.20 | intervention 비활성화 |
| $K_{loc}$ | 2-4 | GT당 residual replay sample 수 |
| $K_{cls}$ | 2-4 | GT당 class replay feature 수 |
| $K_{bg}$ | 1-3 | GT당 rescue candidate 수 |
| $\lambda_{loc}$ | 0.25-1.0 | HLRT loss weight |
| $\lambda_{cls}$ | 0.10-0.50 | CEPR loss weight |
| $\lambda_{bg}$ | 0.10-0.50 | LFR loss weight |
| $T_r$ | 0.5 | side dominance softmax temperature |
| contrastive $\tau$ | 0.1-0.2 | CEPR/LFR contrastive temperature |

## 10.2 Memory size

COCO-scale에서도 full history를 저장할 필요는 없습니다. 권장 memory는 GT당 다음만 유지합니다.

- 상태 EMA 4개: loc, cls, bg, miss.
- residual EMA 4차원 + diagonal variance 4차원.
- feature EMA 128 또는 256차원 projection.
- top-K failure snapshot: 2-4개.
- confusion edge id: top 1-3개.

이렇게 하면 GT당 수 KB 이하로 유지할 수 있습니다.

## 10.3 Stability rule

다음 GT는 memory intervention에서 제외하거나 weight를 낮춥니다.

- crowd/ignore annotation.
- box가 매우 작고 annotation noise가 큰 경우.
- severe truncation으로 visible box 기준이 불명확한 경우.
- 동일 위치에 중복 GT가 있는 경우.
- class label 자체가 ambiguous한 경우.

# 11. 논문 기여도 문장 예시

논문 introduction 또는 contribution에 쓸 수 있는 문장은 다음과 같습니다.

1. We introduce Detection Hysteresis Memory, a GT-level temporal memory that records not only whether an object is missed, but also how it is missed across training epochs.
2. We decompose false negatives into localization, classification, and background suppression failures, and propose error-specific memory-conditioned repair losses for each failure mode.
3. For localization failures, we propose Hysteretic Localization Residual Transport, which converts historical box residuals into hard localization replay samples and side-aware boundary supervision.
4. For classification failures, we propose Confusion-Edge Prototype Rectification, which stores instance-level class confusion edges and applies targeted margin and contrastive rectification.
5. For background false negatives, we propose Latent Foreground Revival, which treats background-suppressed GTs as latent foregrounds and rescues them through objectness-first assignment relaxation.
6. The proposed modules are detector-agnostic and can be integrated into two-stage, anchor-free, and transformer-based detectors.

# 12. 예상 reviewer 질문과 방어 논리

## Q1. 이것은 hard example mining과 무엇이 다른가?

Hard example mining은 일반적으로 현재 loss가 큰 sample을 더 학습합니다. 본 방법은 현재 loss 크기가 아니라 **GT별 temporal error type**을 저장하고, 실패 원인에 따라 다른 intervention을 적용합니다. 예를 들어 FN_LOC는 residual replay, FN_CLS는 confusion edge margin, FN_BG는 objectness revival로 처리됩니다. 따라서 sample mining이 아니라 error-causal repair에 가깝습니다.

## Q2. 기존 ATSS/TOOD/GFL과 겹치지 않는가?

ATSS는 positive/negative assignment의 중요성을 다루고, TOOD는 classification-localization alignment를 다루며, GFL은 localization quality/class representation을 다룹니다. 본 방법은 이들과 달리 **반복 실패한 GT identity를 기억**하고, 해당 GT의 과거 residual/confusion/background suppression pattern을 다음 학습에 사용합니다. 즉, 기존 방법이 sample assignment 또는 representation을 전역적으로 개선한다면, DHM 기반 방법은 instance-specific temporal failure를 국소적으로 복구합니다.

## Q3. GT별 memory가 overfitting을 만들지 않는가?

가능합니다. 그래서 hysteresis gate, replay budget, memory decay, ignore filtering, validation precision-aware decay가 필요합니다. 또한 main claim은 train GT memorization이 아니라 반복 failure mode를 model update에 반영하는 것입니다. 이를 입증하려면 validation set error breakdown과 cross-dataset generalization 실험이 필요합니다.

## Q4. 세 모듈을 다 넣으면 contribution이 산만하지 않은가?

논문 구조는 DHM을 하나의 framework로 두고, error-specific intervention을 세 개의 instantiation으로 제시하면 됩니다. 실험에서는 FN_LOC 비율이 가장 크므로 HLRT를 main module로 강조하고, CEPR/LFR은 complementary module로 둡니다. 최종 contribution은 "DHM framework + error-specific repair"입니다.

# 13. 최종 추천 연구 narrative

현재 FN_LOC 52%, FN_CLS 28%, FN_BG 19%라면 다음 순서로 논문을 구성하는 것이 가장 안전합니다.

1. **Observation**: mAP/AP만으로는 개선 방향이 불명확하지만, DHM error decomposition은 최종 FN의 대부분이 FN_LOC임을 보여준다.
2. **Hypothesis**: FN은 단일 missing event가 아니라 GT별 temporal failure mode이며, failure mode에 맞는 intervention이 필요하다.
3. **Method**: DHM controller가 GT별 hysteresis score를 계산하고, FN_LOC/FN_CLS/FN_BG에 대해 HLRT/CEPR/LFR을 선택 적용한다.
4. **Main result**: HLRT가 FN_LOC와 AP75를 개선한다.
5. **Complementary result**: CEPR은 class confusion, LFR은 foreground recall을 개선한다.
6. **Generalization**: Faster R-CNN, FCOS, DINO에 모두 적용되어 detector-agnostic성을 보인다.
7. **Analysis**: error transition matrix와 qualitative visualization으로 실제 실패 유형이 TP로 전환됨을 보인다.

# 14. 참고문헌

[Ren2015] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." arXiv:1506.01497. https://arxiv.org/abs/1506.01497

[Tian2019] Zhi Tian, Chunhua Shen, Hao Chen, Tong He. "FCOS: Fully Convolutional One-Stage Object Detection." ICCV 2019. https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html

[Zhang2023] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum. "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection." ICLR 2023. https://openreview.net/forum?id=3mRwyG5one

[Bolya2020] Daniel Bolya, Sean Foley, James Hays, Judy Hoffman. "TIDE: A General Toolbox for Identifying Object Detection Errors." ECCV 2020. https://dbolya.com/tide/paper.pdf

[Lin2017] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar. "Focal Loss for Dense Object Detection." arXiv:1708.02002. https://arxiv.org/abs/1708.02002

[Zhang2020] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, Stan Z. Li. "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection." CVPR 2020. https://arxiv.org/abs/1912.02424

[Li2020] Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang. "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection." arXiv:2006.04388. https://arxiv.org/abs/2006.04388

[Feng2021] Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R. Scott, Weilin Huang. "TOOD: Task-aligned One-stage Object Detection." ICCV 2021. https://arxiv.org/abs/2108.07755
