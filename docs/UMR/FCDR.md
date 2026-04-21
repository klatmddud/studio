# FCDR — Failure-Conditioned Counterfactual Replay

## 1. Overview

FCDR (Failure-Conditioned Counterfactual Replay) 는 MDMB++가 기록한 missed-detection failure를 이용해 replay sample을 원인별로 다르게 생성하는 data-layer recovery 모듈이다. 기존 Hard Replay가 hard image를 더 자주 샘플링하는 방식이라면, FCDR은 hard GT가 실패한 원인을 보고 그 실패를 반대로 뒤집는 counterfactual replay sample을 만든다.

핵심 목표는 다음과 같다.

1. 단순 image-level oversampling을 넘어 object-level replay를 실제 학습 배치에 투입한다.
2. `failure_type`, `severity`, `support` 정보를 사용해 replay 변환을 원인별로 선택한다.
3. FCOS, Faster R-CNN, DINO처럼 detector 내부 candidate 구조가 달라도 data layer에서 공통 적용 가능하게 만든다.

FCDR은 candidate를 detector 내부에 추가하지 않는다. 대신 입력 데이터 자체를 바꿔 hard GT가 더 복원 가능한 조건에서 다시 학습되게 한다. 따라서 Candidate Densification과 달리 detector head나 assignment logic을 직접 바꾸지 않는 것이 기본 원칙이다.

## 2. Motivation

현재 UMR의 Hard Replay는 MDMB++ severity를 이용해 hard image를 더 자주 샘플링한다. 이 방식은 안정적이지만, 이미지가 그대로 다시 들어오기 때문에 아래 문제를 해결하지 못한다.

- `candidate_missing`: object가 너무 작거나 주변 context 때문에 모델이 candidate를 만들지 못한다.
- `cls_confusion`: object는 보이지만 비슷한 class와 계속 헷갈린다.
- `score_suppression`: 맞는 후보가 있어도 score/ranking에서 밀린다.
- `nms_suppression`: crowded scene에서 올바른 후보가 suppression된다.
- `relapse`: 과거에는 맞혔지만 현재 학습 상태에서는 다시 놓친다.

이 문제들은 같은 replay를 반복한다고 항상 해결되지 않는다. FCDR은 failure type을 학습 데이터 변환 정책으로 직접 연결한다.

## 3. Difference from Existing Replay and Mining

FCDR은 기존 hard mining, resampling, object replay와 다음 점에서 다르다.

| Method family | Selection signal | Replay unit | Main limitation | FCDR difference |
|---|---|---|---|---|
| OHEM / Focal / GHM | current loss or gradient | sample / RoI / point | persistent miss memory 없음 | per-GT temporal failure memory 사용 |
| PISA | prime sample importance | sample / RoI | miss 원인 분석 없음 | failure type별 replay 변환 |
| RFS / IRFS | class or instance frequency | image | 모델이 실제로 실패한 GT를 보지 않음 | model-specific unresolved miss 기반 |
| ABR / OSR | old-class replay | object crop | incremental learning 목적 | current-task missed detection recovery 목적 |
| Copy-Paste | random or rare object | pasted object | failure-conditioned policy 없음 | MDMB++ failure type으로 paste/crop 정책 결정 |

즉 FCDR의 novelty는 "hard object를 다시 넣는다"가 아니라, "현재 detector가 실제로 실패한 GT를 원인별 counterfactual sample로 다시 넣는다"에 있다.

## 4. Core Data Flow

FCDR의 데이터 흐름은 다음과 같다.

```text
MDMBPlusEntry
  -> CounterfactualReplayPlan
  -> ReplaySample
  -> DataLoader batch
  -> base detector training
```

각 단계의 역할은 다음과 같다.

| Stage | 역할 |
|---|---|
| `MDMBPlusEntry` | unresolved miss GT와 failure context 저장 |
| `CounterfactualReplayPlan` | failure type과 severity를 replay policy로 변환 |
| `ReplaySample` | crop, paste, pair replay로 생성된 실제 학습 sample |
| DataLoader batch | 원본 sample과 replay sample을 mixed batch로 제공 |

FCDR은 MDMB++의 아래 필드를 읽는다.

- `entry.gt_uid`
- `entry.image_id`
- `entry.class_id`
- `entry.bbox`
- `entry.failure_type`
- `entry.severity`
- `entry.best_candidate`
- `entry.topk_candidates`
- `entry.support`
- `record.last_detected_epoch`
- `record.relapse_count`
- `record.last_failure_epoch`

## 5. Counterfactual Replay Principle

GT $g$의 현재 failure type을 $f_t(g)$, severity를 $\phi_t(g)$라 하자. FCDR은 replay 변환 함수 $T$를 다음처럼 선택한다.

$$
x'_g = T_{f_t(g)}(x, b_g, c_g, s_g; \phi_t(g))
$$

여기서:

- $x$: 원본 이미지
- $b_g$: GT box
- $c_g$: class id
- $s_g$: optional support snapshot
- $x'_g$: counterfactual replay sample

중요한 점은 $T$가 random augmentation이 아니라 failure-conditioned augmentation이라는 것이다. 같은 GT라도 `cls_confusion`일 때와 `nms_suppression`일 때 replay sample이 달라져야 한다.

## 6. Failure-Type Replay Policies

### 6.1 `candidate_missing`

의미: GT 주변에 유효 candidate가 거의 없었다.

권장 replay:

- zoom-in crop
- scale-up paste
- context-preserving crop

목표:

- object의 effective size를 키운다.
- detector가 object를 foreground로 인식할 기회를 늘린다.
- 작은 객체나 저해상도 객체의 feature response를 강화한다.

Replay crop은 GT box를 중심으로 생성하되 최소 context를 보장한다.

$$
b_{\mathrm{crop}} =
\mathrm{expand}(b_g, \eta_w w_g, \eta_h h_g)
$$

small object는 최소 context 크기 $m_{\min}$을 추가로 보장한다.

$$
\eta'_w = \max(\eta_w w_g, m_{\min}), \qquad
\eta'_h = \max(\eta_h h_g, m_{\min})
$$

### 6.2 `loc_near_miss`

의미: candidate는 있었지만 localization이 부족했다.

권장 replay:

- localization jitter crop
- boundary-emphasized crop
- partial-context crop

목표:

- object boundary와 box regression signal을 강화한다.
- GT 주변의 near-miss candidate가 정확한 box로 이동하도록 유도한다.

Jittered crop은 GT 중심을 유지하지 않고 box 경계 주변을 약하게 흔든다.

$$
\tilde{b}_g =
\mathrm{jitter}(b_g; \sigma_x, \sigma_y, \sigma_w, \sigma_h)
$$

단, FCDR은 proposal을 detector 내부에 주입하지 않는다. jitter는 입력 crop 또는 pasted object placement에만 사용한다.

### 6.3 `cls_confusion`

의미: 위치는 맞지만 class가 틀렸다.

권장 replay:

- confuser-aware paired crop
- class-disambiguating context crop
- same-class support pair replay

목표:

- 헷갈린 class와 GT class를 구분하는 시각 단서를 보존한다.
- object만 잘라내어 class context를 잃지 않도록 한다.

`entry.best_candidate.label`이 GT class와 다르면 해당 label을 confuser class로 기록한다.

$$
c_{\mathrm{conf}} =
\arg\max_{c \in C_t(g),\; y_c \ne y_g} \mathrm{score}(c)
$$

Replay sample은 GT crop 단독보다 주변 context를 포함하는 것이 우선이다.

### 6.4 `score_suppression`

의미: 정답에 가까운 candidate가 있었지만 score/ranking에서 밀렸다.

권장 replay:

- last-success/current-miss pair replay
- score-preserving context crop
- weak augmentation replay

목표:

- 과거에는 높은 score를 받았던 support 상태와 현재 낮은 score 상태를 함께 노출한다.
- 강한 augmentation보다 안정적인 weak replay를 우선한다.

지원 정보가 있을 때는 support crop $x_g^+$와 current miss crop $x_g^-$를 같은 epoch 안에서 함께 샘플링한다.

### 6.5 `nms_suppression`

의미: crowded scene 또는 overlap 때문에 후보가 suppression되었다.

권장 replay:

- crowd-preserving crop
- overlap-preserving crop
- occlusion-aware paste

목표:

- object 주변의 competing object를 제거하지 않는다.
- NMS 문제를 유발한 local crowd context를 그대로 유지한다.

이 경우 crop은 GT 단독이 아니라 주변 overlapping GT까지 포함해야 한다.

$$
\mathcal{N}(g) =
\{h \mid \mathrm{IoU}(b_g, b_h) > \theta_{\mathrm{overlap}}\}
$$

Replay crop은 $b_g$와 $\mathcal{N}(g)$를 모두 포함하는 union box를 기반으로 만든다.

### 6.6 `relapse`

의미: 과거에는 detected였지만 현재 unresolved miss 상태다.

권장 replay:

- support-miss pair replay
- support-guided crop scale
- support-guided weak/strong pair

목표:

- 같은 GT의 성공 상태와 실패 상태를 동시에 학습시킨다.
- 모델이 한 번 배웠던 object representation을 다시 회복하도록 한다.

Relapse replay priority는 일반 miss보다 높게 둔다.

$$
w_{\mathrm{relapse}}(g) =
1 + \lambda_r \cdot \mathbb{1}[\mathrm{last\_detected\_epoch}(g) \ne \mathrm{None}]
$$

## 7. Replay Plan Schema

구현 시 아래 dataclass를 권장한다.

```python
@dataclass(slots=True)
class CounterfactualReplayPlan:
    epoch: int
    samples: list[CounterfactualReplaySpec]
    summary: dict[str, float | int | bool]

@dataclass(slots=True)
class CounterfactualReplaySpec:
    gt_uid: str
    image_id: str
    class_id: int
    failure_type: str
    severity: float
    mode: str
    crop_box_abs: tuple[int, int, int, int]
    source_bbox_abs: tuple[int, int, int, int]
    support_box_abs: tuple[int, int, int, int] | None
    paste_scale: float | None
    pair_with_support: bool
```

`mode`는 아래 값 중 하나로 시작한다.

- `zoom_crop`
- `context_crop`
- `scale_paste`
- `confuser_pair`
- `crowd_crop`
- `support_pair`

## 8. Sampling Formulation

FCDR replay 확률은 image-level replay weight와 object-level counterfactual weight를 함께 사용한다.

$$
p_{\mathrm{FCDR}}(g)
=
\frac{
\left(1 + \beta \phi_t(g) + \lambda_f \rho(f_t(g))\right)^\tau
}{
\sum_{h \in \mathcal{M}_t}
\left(1 + \beta \phi_t(h) + \lambda_f \rho(f_t(h))\right)^\tau
}
$$

여기서:

- $\mathcal{M}_t$: 현재 unresolved miss GT 집합
- $\rho(f_t(g))$: failure type prior
- $\tau$: sampling temperature

Replay batch 구성은 기존 Hard Replay와 동일하게 mixed batch를 유지한다.

$$
B = B_{\mathrm{base}} \cup B_{\mathrm{fcdr}},
\qquad
|B_{\mathrm{fcdr}}| = \lfloor r_{\mathrm{fcdr}} |B| \rfloor
$$

## 9. Current Codebase Integration

현재 코드베이스에서 FCDR은 `scripts/runtime/hard_replay.py`를 확장하는 것이 가장 자연스럽다.

### 9.1 Extend `HardReplayPlanner`

구현된 Hard Replay 확장판은 `HardReplayPlanner.build_epoch_index(...)`에서 image-level replay index와 object-level replay specs를 함께 만든다. FCDR은 이 object replay path 위에서 `failure_type` 기반 policy를 선택하는 preset으로 동작한다.

1. `mdmbpp.get_dense_targets(image_key)`로 unresolved entries를 읽는다.
2. `mdmbpp.get_record(entry.gt_uid)`로 temporal state를 읽는다.
3. `entry.failure_type`과 `entry.severity`로 replay mode를 선택한다.
4. `ReplayIndex.replay_samples`에 `ReplaySampleSpec`을 채운다.

`ReplaySampleSpec`은 `kind`, `dataset_index`, `failure_type`, `mode`, `crop_box_abs`, `source_bbox_abs`, `support_box_abs`, `target_dataset_index`, `paste_box_abs`, `pair_id`, `role`, `sampling_weight`, `replay_cap`, `loss_weight`를 포함한다. Sampler는 base dataset 길이 이후의 virtual index를 object replay sample로 사용한다.

### 9.2 Add Replay Dataset Wrapper

`scripts/runtime/data.py`에서 `CocoDetectionDataset`을 감싸는 wrapper를 추가한다.

```python
class HardReplayDatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        ...

    def set_replay_index(self, replay_index):
        ...

    def __getitem__(self, index):
        if index >= len(self.base_dataset):
            return self._build_replay_sample(index)
        return self.base_dataset[index]
```

현재 구현은 crop, copy-paste, pair replay를 지원한다.

1. base image와 annotations를 로드한다.
2. `crop_box_abs`로 image crop을 만든다.
3. crop 내부에 남는 GT boxes를 crop coordinate로 변환한다.
4. 너무 작거나 crop 밖으로 완전히 나간 boxes는 제거한다.
5. target의 `image_id`는 negative synthetic id로 설정한다.
6. 원본 id와 replay metadata는 `source_image_id`, `replay_kind`, `replay_gt_uid`, `replay_pair_id`, `replay_role`, `replay_failure_type`, `replay_mode`로 보존한다.
7. `is_replay: true`를 설정해 post-step MDMB/MDMB++ memory update에서 제외한다.
8. `replay_box_weights`를 설정해 FCOS에서 replay-aware positive loss weighting을 적용할 수 있게 한다.

Replay index는 epoch마다 바뀐다. worker process가 stale replay index를 들고 있지 않도록 `HardReplayDatasetWrapper` 사용 시 persistent workers는 비활성화한다.

### 9.3 Copy-Paste Replay

copy-paste는 현재 구현되어 있으며 rectangular paste를 사용한다.

1. source object crop을 만든다.
2. target base image를 deterministic seed로 샘플링한다.
3. `max_paste_overlap` constraint를 만족하는 위치에 paste한다.
4. target annotations에 pasted box를 추가한다.

첫 버전에서는 segmentation mask가 없으므로 rectangular paste를 사용한다. mask가 없는 COCO detection setting에서는 rectangle paste artifact가 생길 수 있으므로 `copy_paste_ratio`를 낮게 시작한다.

### 9.4 Pair Replay

pair replay는 동일 epoch 안에서 support crop과 current miss crop을 모두 노출하는 방식으로 구현한다.

간단한 구현:

- batch sampler가 같은 `gt_uid`의 support spec과 miss spec을 같은 epoch replay pool에 넣는다.
- replay slot이 충분하면 같은 mini-batch에 `pair_miss`와 `pair_support`를 함께 넣는다.
- RASD와 결합할 때만 same mini-batch pair constraint를 고려한다.

## 10. Config Proposal

현재 구현은 `modules/cfg/hard_replay.yaml`의 nested `fcdr`, `object_replay`, `loss` 섹션을 사용한다. FCDR은 top-level Hard Replay `enabled: true`일 때만 활성화된다.

```yaml
enabled: false
beta: 1.0
temperature: 1.0
replay_ratio: 0.25
max_image_weight: 5.0
min_replay_weight: 1.0
replacement: true
max_replays_per_gt_per_epoch: 4
replay_recency_window: 3

fcdr:
  enabled: false
  counterfactual_ratio: 0.5
  min_severity: 0.0
  max_crops_per_gt_per_epoch: 1
  crop_context_scale: 1.0
  min_crop_context_px: 16
  overlap_threshold: 0.1
  copy_paste_prob: 0.0
  pair_replay_prob: 0.0

object_replay:
  enabled: false
  crop_ratio: 0.4
  copy_paste_ratio: 0.3
  pair_ratio: 0.3
  crop:
    enabled: false
    context_scale: 1.0
    min_context_px: 16
  copy_paste:
    enabled: false
    paste_scale: 1.0
    max_paste_overlap: 0.3
    max_attempts: 20
  pair:
    enabled: false
    require_same_batch: true
    min_replay_slots: 2

loss:
  enabled: false
  cls_weight: 1.0
  reg_weight: 1.0
  ctr_weight: 1.0
  crop_box_weight: 1.5
  pasted_box_weight: 2.0
  pair_box_weight: 1.5
  max_weight: 3.0
```

FCDR을 `hard_replay.yaml` 안에 넣어 기존 controller, sampler, epoch refresh hook을 재사용한다. 별도 `fcdr.yaml` 분리는 추후 ablation 관리가 필요할 때 고려한다.

## 11. Safety Rules

FCDR은 data distribution을 강하게 바꾸므로 아래 제약을 기본값으로 둔다.

| Rule | 권장값 | 이유 |
|---|---:|---|
| `max_crops_per_gt_per_epoch` | 1 | 특정 GT crop 과노출 방지 |
| `counterfactual_ratio` | 0.5 이하 | replay batch 전체가 synthetic이 되는 것 방지 |
| `copy_paste_ratio` | 0.3 이하 | rectangular paste artifact 완화 |
| `pair_ratio` | 0.3 이하 | support/miss pair가 replay batch를 지배하는 것 방지 |
| `overlap_threshold` | 0.1 | NMS/crowd crop 주변 객체 포함 |
| `min_crop_context_px` | 16 | context 없는 tiny crop 방지 |

또한 validation에는 절대 replay transform을 적용하지 않는다.

## 12. Monitoring

FCDR summary에는 아래 값을 기록한다.

- `fcdr_enabled`
- `fcdr_num_crops`
- `fcdr_counterfactual_ratio_requested`
- `fcdr_samples`
- `fcdr_ratio_effective`
- `fcdr_unique_crops`
- `fcdr_failure_*`
- `fcdr_policy_*`
- `replay_crop_samples`
- `replay_copy_paste_samples`
- `replay_pair_samples`
- `replay_loss_weight_mean`
- `replay_skipped_*`

MDMB++ summary와 함께 보면 FCDR이 어떤 failure type을 실제로 줄였는지 추적할 수 있다.

## 13. Evaluation and Ablations

권장 실험은 다음 순서다.

1. baseline
2. MDMB++ only
3. naive image-level Hard Replay
4. FCDR crop-only
5. FCDR crop + copy-paste
6. FCDR crop + support pair
7. FCDR full

Failure-type ablation:

- remove `candidate_missing` policy
- remove `cls_confusion` policy
- remove `nms_suppression` policy
- remove relapse support pair

주요 평가 지표:

- COCO mAP
- `AP_small`, `AP_medium`, `AP_large`
- recovery rate
- relapse resolution time
- failure type별 unresolved entry 감소량
- replay 대상 GT의 before/after detection rate

## 14. Minimal Implementation Version

최소 구현은 아래 범위로 제한한다.

1. `HardReplayPlanner`가 `ReplayIndex.replay_samples`를 채운다.
2. `HardReplayDatasetWrapper`가 crop, copy-paste, pair replay sample을 생성한다.
3. failure type별 crop policy는 `candidate_missing`, `loc_near_miss`, `cls_confusion`, `score_suppression`, `nms_suppression`, `relapse`를 지원한다.
4. FCOS는 `replay_box_weights`로 positive point loss를 재가중한다.

이 최소 버전만으로도 단순 hard image resampling과 FCDR의 차이를 검증할 수 있다.

## 15. Novelty Summary

FCDR의 핵심 기여는 다음이다.

1. Detector가 실제로 실패한 GT를 online memory에서 읽는다.
2. 실패 원인별로 counterfactual replay transformation을 다르게 적용한다.
3. 같은 framework 안에서 image-level replay, crop replay, copy-paste replay, support pair replay를 통합한다.
4. Candidate Densification 없이도 data layer에서 missed detection recovery를 직접 겨냥한다.
