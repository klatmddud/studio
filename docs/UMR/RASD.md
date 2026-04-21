# RASD — Relapse-Aware Support Distillation

## 1. Overview

RASD (Relapse-Aware Support Distillation) 는 과거에 성공적으로 detect되었지만 현재 다시 miss된 GT를 대상으로, 과거 성공 시점의 support representation을 temporal teacher로 사용하는 training-only distillation 모듈이다.

핵심 아이디어는 다음과 같다.

1. MDMB++는 GT별로 `last_detected_epoch`, `relapse_count`, `support`, `failure_type`, `topk_candidates`를 저장한다.
2. RASD는 relapse 또는 high-severity unresolved miss GT를 선택한다.
3. 현재 GT feature를 같은 GT의 과거 support feature에 가깝게 만들고, confuser feature와는 멀어지게 한다.

RASD는 모델 외부 teacher를 요구하지 않는다. 같은 모델이 과거에 성공했던 object-level state를 teacher로 쓰기 때문에, lightweight detector나 single-run training에도 적용 가능하다.

## 2. Motivation

Missed detection은 두 종류로 나눌 수 있다.

1. never-detected hard GT: 학습 내내 거의 맞히지 못한 객체
2. relapse GT: 한 번 이상 맞혔지만 이후 다시 놓친 객체

Relapse GT는 중요하다. 모델이 해당 객체를 표현할 수 있었던 시점이 이미 존재하기 때문이다. 단순 loss reweighting은 현재 miss 상태만 크게 벌주지만, RASD는 "과거 성공 상태로 돌아가라"는 더 구체적인 supervision을 제공한다.

RASD의 목적은 다음과 같다.

- relapse GT의 feature drift를 줄인다.
- class confusion을 일으키는 confuser representation과 분리한다.
- support memory를 이용해 object-level temporal consistency를 만든다.

## 3. Difference from Feature-Anchor Distillation and Knowledge Distillation

RASD는 feature-anchor consistency 및 일반 object detection KD와 겹치는 부분이 있지만 목적과 정보원이 다르다.

| Method | Teacher / anchor | Target GT | Loss shape | RASD difference |
|---|---|---|---|---|
| Feature-anchor consistency | frozen feature anchor | relapse GT | cosine consistency | MDMB++ support + confuser-aware contrastive loss |
| Teacher-student KD | external teacher model | many predictions or features | mimic teacher outputs/features | 외부 teacher 없음, same-object past success 사용 |
| Self-distillation | same model or EMA model | high-confidence predictions | consistency / relation loss | relapse/miss GT에 selective 적용 |
| Hard sample memory bank | hard sample prototype | hard positives | contrastive / prototype loss | GT temporal support와 failure type을 함께 사용 |

Feature-anchor consistency가 "과거 feature anchor로 당긴다"에 가깝다면, RASD는 "과거 성공 support로 당기고 현재 confuser와는 분리한다"에 가깝다.

## 4. Relapse Definition

GT $g$가 epoch $t$에서 relapse 상태라는 것은 과거 detected 이력이 있고 현재는 unresolved miss라는 뜻이다.

$$
\mathrm{relapse}(g, t)
=
\mathbb{1}
\left[
\mathrm{last\_detected\_epoch}(g) \ne \mathrm{None}
\;\land\;
s_t(g) \ge s_{\min}
\right]
$$

여기서:

- $s_t(g)$: 현재 consecutive miss streak
- $s_{\min}$: RASD를 적용할 최소 miss streak

RASD는 relapse GT를 기본 대상으로 삼고, 선택적으로 high-severity unresolved miss도 포함할 수 있다.

$$
\mathcal{R}_t =
\{g \mid \mathrm{relapse}(g, t) = 1\}
\cup
\{g \mid \phi_t(g) \ge \tau_{\phi}\}
$$

## 5. Support, Current, and Confuser Features

RASD는 GT별로 세 종류의 feature를 사용한다.

| Feature | 의미 | Source |
|---|---|---|
| $a_g$ | support feature | 마지막 성공 detection 시점 |
| $f_g^t$ | current GT feature | 현재 training forward |
| $n_g^t$ | confuser feature | 현재 또는 최근 candidate 중 class가 틀린 hard candidate |

### 5.1 Support feature

Support feature는 MDMB++의 `SupportSnapshot`에서 읽는다.

```python
@dataclass(slots=True)
class SupportSnapshot:
    epoch: int
    box: Tensor
    score: float
    feature: Tensor | None
    feature_level: str | int | None
```

`feature`가 저장되어 있으면 그대로 사용한다. `store_support_feature=false`인 경우에는 support box를 이용해 현재 feature map에서 re-pooling한다. 논문 실험에서는 두 방식을 ablation으로 비교한다.

### 5.2 Current feature

현재 feature는 GT box 기반 RoIAlign으로 추출한다.

$$
\tilde{f}_g^t =
\mathrm{RoIAlign}(\mathcal{F}_t, b_g),
\qquad
f_g^t =
\frac{\mathrm{GAP}(\tilde{f}_g^t)}
{\lVert \mathrm{GAP}(\tilde{f}_g^t) \rVert_2 + \epsilon}
$$

FCOS에서는 FPN feature keys `("0", "1", "2", "p6", "p7")`를 사용한다. 현재 FAR가 이미 `MultiScaleRoIAlign` 기반 feature extraction을 사용하므로, RASD는 이 코드를 재사용할 수 있다.

### 5.3 Confuser feature

Confuser는 같은 GT 근처에서 높은 score를 받았지만 class가 틀린 candidate다.

$$
c_g^- =
\arg\max_{c \in C_t(g),\; y_c \ne y_g}
\mathrm{score}(c)
$$

Confuser feature $n_g^t$는 `c_g^-`의 box에서 RoIAlign으로 추출한다. 적절한 confuser가 없으면 contrastive term을 생략하고 support attraction만 적용한다.

## 6. Loss Formulation

RASD loss는 support attraction과 confuser-aware separation으로 구성된다.

### 6.1 Support attraction

현재 feature가 support feature에 가까워지도록 cosine distance를 최소화한다.

$$
\mathcal{L}_{\mathrm{sup}}(g)
=
1 -
\cos(f_g^t, a_g)
$$

### 6.2 Confuser-aware contrastive loss

Confuser가 있을 때는 support를 positive, confuser를 negative로 하는 InfoNCE-style loss를 사용한다.

$$
\mathcal{L}_{\mathrm{con}}(g)
=
-
\log
\frac{
\exp(\cos(f_g^t, a_g) / \tau)
}{
\exp(\cos(f_g^t, a_g) / \tau)
+
\exp(\cos(f_g^t, n_g^t) / \tau)
}
$$

여기서 $\tau$는 contrastive temperature다.

Triplet margin 형태를 선택할 수도 있다.

$$
\mathcal{L}_{\mathrm{tri}}(g)
=
\max
\left(
0,\;
m
+
d(f_g^t, a_g)
-
d(f_g^t, n_g^t)
\right)
$$

초기 구현은 InfoNCE-style loss를 권장한다. margin tuning보다 temperature tuning이 안정적인 경우가 많기 때문이다.

### 6.3 Severity and streak weighting

RASD는 모든 GT에 균등하게 적용하지 않는다. relapse가 길고 severity가 높을수록 더 강하게 적용한다.

$$
w_g =
1
+
\gamma_s
\frac{s_t(g)}{\max(s_{\mathrm{global}}, 1)}
+
\gamma_{\phi}
\phi_t(g)
+
\gamma_r
\mathrm{relapse\_count}(g)
$$

최종 loss는 다음과 같다.

$$
\mathcal{L}_{\mathrm{RASD}}
=
\frac{\lambda_{\mathrm{rasd}}}
{|\mathcal{R}_t| + \epsilon}
\sum_{g \in \mathcal{R}_t}
w_g
\left(
\mathcal{L}_{\mathrm{sup}}(g)
+
\alpha_{\mathrm{con}}
\mathcal{L}_{\mathrm{con}}(g)
\right)
$$

Confuser가 없는 GT는 $\mathcal{L}_{\mathrm{con}}(g)=0$으로 둔다.

## 7. Target Selection

RASD 대상은 MDMB++에서 선택한다.

필수 조건:

- `entry.support is not None`
- `record.last_detected_epoch is not None`
- `entry.consecutive_miss_count >= min_relapse_streak`
- `entry.severity >= min_severity`

권장 filter:

- `failure_type in {"cls_confusion", "score_suppression", "nms_suppression"}`
- `support.score >= min_support_score`
- `current_epoch - support.epoch <= max_support_age`

`candidate_missing`은 feature support가 불안정할 수 있으므로 초기 버전에서는 제외하거나 support attraction만 적용한다.

## 8. Current Codebase Integration

RASD는 `modules/nn/rasd.py` 신규 모듈로 두는 것이 가장 명확하다.

### 8.1 Config

`modules/cfg/rasd.yaml` 제안:

```yaml
enabled: false
lambda_rasd: 0.05
temperature: 0.2
alpha_contrastive: 1.0
min_relapse_streak: 1
min_severity: 1.0
min_support_score: 0.2
max_support_age: 20
max_targets_per_batch: 32
feature_keys: ["0", "1", "2", "p6", "p7"]
roi_output_size: 7
roi_sampling_ratio: 2
store_support_feature_required: false

models:
  fcos: {}
  fasterrcnn:
    enabled: false
  dino:
    enabled: false
```

### 8.2 Module API

```python
class RelapseAwareSupportDistillation(nn.Module):
    def start_epoch(self, epoch: int) -> None: ...
    def end_epoch(self, epoch: int | None = None) -> None: ...
    def should_apply(self, *, mdmbpp: MDMBPlus | None) -> bool: ...
    def compute_loss(
        self,
        *,
        image_ids: Sequence[Any],
        gt_boxes_list: Sequence[Tensor],
        gt_labels_list: Sequence[Tensor],
        features: Mapping[str, Tensor],
        image_shapes: Sequence[Sequence[int]],
        mdmbpp: MDMBPlus,
    ) -> Tensor: ...
    def summary(self) -> dict[str, Any]: ...
```

### 8.3 Registry integration

`scripts/runtime/registry.py`에 `_build_rasd(arch)`를 추가하고 FCOS wrapper에 전달한다.

```python
rasd = _build_rasd(normalized_arch)
return FCOSWrapper(..., rasd=rasd)
```

### 8.4 FCOS forward integration

FCOS training forward에서 backbone features가 계산된 뒤 RASD loss를 추가한다.

```python
rasd = self._get_rasd()
mdmbpp = self._get_mdmbpp()
if rasd is not None and rasd.should_apply(mdmbpp=mdmbpp):
    rasd_loss = rasd.compute_loss(
        image_ids=[target["image_id"] for target in targets],
        gt_boxes_list=[target["boxes"] for target in targets],
        gt_labels_list=[target["labels"] for target in targets],
        features=features,
        image_shapes=images.image_sizes,
        mdmbpp=mdmbpp,
    )
    if rasd_loss.requires_grad or float(rasd_loss.detach().item()) != 0.0:
        losses = dict(losses)
        losses["rasd"] = rasd_loss
```

주의: `features`와 `targets`는 transform 이후 coordinate space에 있다. MDMB++ support box가 pre-transform normalized coordinate일 경우, re-pooling 시 coordinate conversion을 명확히 해야 한다. 초기 구현은 current GT feature와 stored support feature를 사용하고, support box re-pooling은 별도 ablation으로 미룬다.

### 8.5 MDMB++ support feature

RASD가 가장 안정적으로 동작하려면 `mdmbpp.config.store_support_feature=true`가 필요하다. 현재 기본값은 false이므로 RASD config에서 다음 중 하나를 강제해야 한다.

1. RASD enabled 시 MDMB++ `store_support_feature=true`를 요구하고 아니면 fail-fast.
2. `store_support_feature=false`이면 support box를 현재 feature map에서 re-pooling한다.

논문용 명확성을 위해 1번을 권장한다. 2번은 coordinate-space mismatch와 feature drift 때문에 해석이 약해질 수 있다.

## 9. Confuser Mining

Confuser는 MDMB++의 `topk_candidates` 또는 `best_candidate`에서 선택한다.

선택 규칙:

1. GT class와 다른 label이어야 한다.
2. `iou_to_gt >= confuser_iou_threshold`를 만족해야 한다.
3. score가 가장 높은 candidate를 선택한다.

수식:

$$
n_g =
\arg\max_{c \in C_t(g)}
\mathrm{score}(c)
\quad
\mathrm{s.t.}
\quad
y_c \ne y_g,\;
\mathrm{IoU}(c, b_g) \ge \theta_{\mathrm{conf}}
$$

`cls_confusion` entry는 confuser를 찾을 가능성이 높다. `nms_suppression`과 `score_suppression`은 confuser가 없을 수 있으므로 support attraction만 적용해도 된다.

## 10. Summary Metrics

RASD는 epoch별로 아래 값을 기록한다.

- `rasd_enabled`
- `rasd_targets`
- `rasd_relapse_targets`
- `rasd_confuser_targets`
- `rasd_mean_severity`
- `rasd_mean_support_age`
- `rasd_loss`
- `rasd_support_attraction_loss`
- `rasd_contrastive_loss`
- `rasd_skipped_no_support`
- `rasd_skipped_no_confuser`

이 지표는 RASD가 실제로 relapse GT에 작동하고 있는지 확인하는 데 필요하다.

## 11. Stability Rules

RASD는 feature-level objective이므로 아래 안전장치를 둔다.

| Rule | 권장값 | 이유 |
|---|---:|---|
| `lambda_rasd` | 0.03-0.1 | base detection loss 간섭 방지 |
| `max_targets_per_batch` | 32 | RoIAlign overhead 제한 |
| `min_support_score` | 0.2 | 낮은 품질 support 제외 |
| `max_support_age` | 20 epochs | 너무 오래된 support 사용 방지 |
| `temperature` | 0.1-0.3 | contrastive gradient 안정화 |
| `min_relapse_streak` | 1 or 2 | transient miss 과반응 방지 |

초기 실험은 `lambda_rasd=0.05`, `temperature=0.2`, `max_targets_per_batch=16`으로 시작한다.

## 12. Inference Behavior

RASD는 training-only 모듈이다.

- inference graph 변경 없음
- prediction postprocess 변경 없음
- NMS/score threshold 변경 없음
- additional latency 없음

모든 support feature와 confuser mining은 training state 또는 checkpoint state에만 존재한다.

## 13. Evaluation and Ablations

권장 실험:

1. baseline
2. MDMB++ only
3. feature-anchor consistency baseline
4. RASD attraction-only
5. RASD confuser-aware
6. FCDR only
7. FCDR + RASD

RASD 내부 ablation:

- support attraction only vs attraction + contrastive
- stored support feature vs support box re-pooled feature
- severity weighting off/on
- relapse-only vs relapse + high-severity miss
- confuser threshold sweep

주요 지표:

- mAP 50:95
- AP75
- class confusion reduction
- relapse recovery rate
- relapse resolution time
- RASD target count

## 14. Minimal Implementation Version

최소 구현은 아래 범위로 제한한다.

1. `store_support_feature=true`를 요구한다.
2. RASD 대상은 `relapse=True`이고 `support.feature is not None`인 GT만 사용한다.
3. current GT feature는 RoIAlign + GAP + L2 normalize로 추출한다.
4. loss는 support attraction만 먼저 적용한다.
5. confuser-aware contrastive term은 `topk_candidates` coordinate handling이 안정화된 뒤 추가한다.

이 최소 버전은 FAR와 직접 비교하기 쉽고, 이후 confuser-aware loss를 추가해 RASD의 novelty를 강화할 수 있다.

## 15. Novelty Summary

RASD의 핵심 기여는 다음이다.

1. 외부 teacher 없이 같은 GT의 과거 성공 state를 temporal teacher로 사용한다.
2. 모든 object가 아니라 relapse 또는 high-severity miss GT에만 selective distillation을 적용한다.
3. support-positive attraction과 confuser-negative separation을 함께 사용한다.
4. MDMB++의 structured failure memory 없이는 구성하기 어려운 object-level recovery objective다.
