# Hard Replay — Memory-Guided Data Redistribution

## 1. Overview

Hard Replay는 MDMB++가 식별한 hard GT를 기준으로 학습 데이터 분포를 재편하는 모듈이다. 목적은 단순 oversampling이 아니라, `복원이 필요한 객체가 더 자주, 더 다양한 형태로, 더 직접적으로` 모델 앞에 나타나게 만드는 것이다.

이 모듈은 detector 내부 head 구조를 거의 건드리지 않으므로 FCOS, Faster R-CNN, DINO 모두에 쉽게 적용할 수 있다. 따라서 UMR 전체에서 가장 먼저 붙이기 좋은 범용 성능 모듈이다.

## 2. Why Replay Matters

Loss reweighting은 현재 batch 안에 hard GT가 들어왔을 때만 효과를 낸다. 하지만 chronic miss는 애초에 다음 문제가 있을 수 있다.

- hard GT가 전체 데이터셋에서 너무 드물다.
- hard GT가 들어 있는 이미지 자체가 적게 샘플링된다.
- 이미지에 들어와도 해당 object scale이나 context가 충분히 다양하지 않다.

Hard Replay는 이 문제를 데이터 분포 단계에서 해결한다.

1. hard image를 더 자주 샘플링한다.
2. hard object crop을 별도 replay instance로 재노출한다.
3. 필요하면 support-guided paste를 통해 recoverable context를 생성한다.

## 3. Replay Unit

Hard Replay는 두 종류의 replay unit을 사용한다.

### 3.1 Image-level replay

이미지 단위 재샘플링이다.

- chronic miss GT가 많이 포함된 이미지의 샘플링 확률을 높인다.
- 원본 데이터 분포를 완전히 버리지 않고 mixing ratio를 통해 완화한다.

### 3.2 Object-level replay

object crop 단위 재노출이다.

- hard GT 주변 crop을 추출한다.
- 다른 이미지에 copy-paste 하거나 원본 이미지 안에서 zoom-in augmentation에 사용한다.
- 마지막 성공 support crop이 있다면 pair replay를 구성할 수 있다.

## 4. Replay Priority

hard replay의 핵심은 per-image priority 계산이다. 이미지 $i$에 대한 replay weight를 $w_i$로 정의하면 다음처럼 계산할 수 있다.

$$
w_i = 1 + \beta \sum_{g \in \mathcal{G}(i)} \phi_t(g)
$$

여기서:

- $\mathcal{G}(i)$: 이미지 $i$의 GT 집합
- $\phi_t(g)$: GT $g$의 severity
- $\beta$: replay intensity

실제 샘플링 분포는 다음처럼 둔다.

$$
p_t(i) = \frac{w_i^\tau}{\sum_j w_j^\tau}
$$

여기서 $\tau$는 sampling temperature다.

- $\tau = 1$: 선형 유지
- $\tau > 1$: hard image 집중 강화
- $\tau < 1$: 완만한 분포

초기 추천값은 $\beta = 0.5 \sim 2.0$, $\tau = 1.0 \sim 1.5$다.

## 5. Object-level Replay Score

object crop replay는 per-GT priority를 사용한다. GT $g$에 대한 crop replay 확률을 $q_t(g)$로 두면 다음처럼 정의할 수 있다.

$$
q_t(g) = \min\left(q_{\max}, \gamma \cdot \phi_t(g)\right)
$$

여기서:

- $\phi_t(g)$가 클수록 crop replay 확률이 커진다.
- $q_{\max}$는 hard object가 batch를 과도하게 지배하지 않도록 두는 상한이다.

권장값:

- $q_{\max} = 0.5$
- $\gamma = 0.3 \sim 0.8$

## 6. Replay Modes

구현은 아래 4단계로 점진적으로 확장하는 것을 권장한다.

### 6.1 Mode A: Weighted image sampling

가장 단순한 버전이다.

- epoch 시작 시 MDMB++에서 이미지별 `w_i` 계산
- `WeightedRandomSampler`로 DataLoader 구성
- 별도 이미지 수정 없이 노출 빈도만 증가

장점:

- 구현이 가장 쉽다.
- 모든 detector에 동일하게 적용된다.

### 6.2 Mode B: Hard crop replay

hard GT 주변 crop을 별도 replay pool에 넣고 배치 내에 섞는다.

- GT box를 중심으로 context margin을 포함한 crop 생성
- resize / color jitter / blur / random context padding 적용
- 원본 이미지 대신 replay sample로 배치에 투입하거나, 원본 이미지와 섞어서 사용

crop 저장 형식 예시:

```python
@dataclass(slots=True)
class ReplayCrop:
    gt_uid: str
    image_id: str
    class_id: int
    crop_box_abs: tuple[int, int, int, int]
    source_bbox_abs: tuple[int, int, int, int]
    severity: float
    support_box_abs: tuple[int, int, int, int] | None
```

### 6.3 Mode C: Copy-paste replay

hard crop을 다른 이미지에 삽입한다.

- 붙일 위치는 scale과 aspect ratio를 유지하면서 선택
- crowding이 심한 이미지에서는 overlap threshold를 두어 삽입
- GT annotation도 함께 업데이트

이 방식은 small object나 long-tail class 노출 빈도를 늘리는 데 특히 효과적일 수 있다.

### 6.4 Mode D: Pair replay

마지막 성공 support와 현재 miss crop을 pair로 사용한다.

- support crop: 과거 detected 시점의 GT context
- miss crop: 현재 unresolved miss context

이를 통해 다음 중 하나를 할 수 있다.

1. 단순히 둘 다 더 자주 보이게 한다.
2. 같은 class / 같은 GT의 recoverable pair를 augmentation unit으로 다룬다.
3. optional contrastive auxiliary branch와 연결한다.

Hard Replay 문서에서는 1번만 필수로 보고, 2번과 3번은 선택 사항으로 둔다.

## 7. Batch Composition

batch 전체를 hard replay로 채우면 overfitting 위험이 높다. 따라서 mixed batch를 권장한다.

예시:

- `base_ratio = 0.75`
- `replay_ratio = 0.25`

즉 배치의 75%는 일반 샘플, 25%는 hard replay 샘플로 구성한다.

보다 일반적으로는 다음처럼 쓸 수 있다.

$$
B = B_{\mathrm{base}} \cup B_{\mathrm{replay}},
\qquad
|B_{\mathrm{replay}}| = \lfloor r \cdot |B| \rfloor
$$

여기서 $r$은 replay ratio다.

## 8. Anti-overfitting Rules

Hard Replay는 공격적인 방법이므로 아래 제약을 두는 것이 좋다.

1. 동일 `gt_uid`가 한 epoch에 replay되는 횟수에 상한을 둔다.
2. replay 대상은 최근 $K$ epoch 내 unresolved miss만 사용한다.
3. severity가 매우 높더라도 image-level weight에 clipping을 둔다.
4. class imbalance가 심해질 경우 class-balanced cap을 추가한다.

추천 제약:

- `max_replays_per_gt_per_epoch = 4`
- `replay_recency_window = 3`
- `max_image_weight = 5.0`

## 9. Implementation Plan

구현은 data layer 중심으로 진행한다.

### 9.1 `ReplayIndex` 생성

epoch 시작 시 아래 정보를 계산한다.

```python
@dataclass(slots=True)
class ReplayIndex:
    image_weights: dict[str, float]
    replay_crops: list[ReplayCrop]
    replay_gt_ids: set[str]
```

생성 절차:

1. MDMB++에서 unresolved miss entry를 읽는다.
2. image별 severity 합으로 `image_weights` 계산
3. threshold 이상인 GT에 대해 crop 생성
4. replay 횟수 제한과 recency window 적용

### 9.2 DataLoader integration

선택지 두 가지가 있다.

1. `WeightedRandomSampler` 사용
2. 기존 dataset을 감싸는 `ReplayDatasetWrapper` 추가

권장 순서:

- 1단계: sampler만 추가
- 2단계: wrapper로 crop replay까지 확장

### 9.3 Dataset wrapper

```python
class ReplayDatasetWrapper(Dataset):
    def __init__(self, base_dataset, replay_index, replay_ratio):
        ...

    def __getitem__(self, idx):
        if should_return_replay():
            return build_replay_sample(...)
        return self.base_dataset[idx]
```

이 wrapper는 detector family와 무관하게 동작한다.

## 10. Crop Construction

crop을 만들 때는 GT box만 딱 자르면 안 된다. context가 너무 적어져서 원래 어려운 원인을 보존하지 못할 수 있다.

권장 규칙:

1. GT box 바깥으로 margin을 둔다.
2. margin은 GT 크기의 배수 또는 최소 픽셀 수 기준으로 둔다.
3. crop resize 시 extreme distortion을 피한다.

예시:

$$
b_{\mathrm{crop}} =
\mathrm{expand}(b_g, m_x, m_y),
\qquad
m_x = \eta_x \cdot w_g,\quad m_y = \eta_y \cdot h_g
$$

권장 시작값:

- $\eta_x = \eta_y = 0.5$
- small object에는 최소 context 16px 보장

## 11. Support-guided Replay

MDMB++가 support snapshot을 저장하고 있다면 replay crop에 support 정보를 활용할 수 있다.

가능한 방법:

1. support crop과 current miss crop을 번갈아 샘플링
2. support crop을 weak augmentation, miss crop을 strong augmentation으로 학습
3. support crop의 scale / aspect prior를 현재 crop augmentation에 반영

이 기능은 필수는 아니지만, relapse object 복원에 도움이 될 가능성이 높다.

## 12. Monitoring

Hard Replay는 실제로 노출 빈도를 바꾸는 모듈이므로 아래 지표를 기록하는 것이 좋다.

- `replay_num_images`
- `replay_num_crops`
- `replay_mean_image_weight`
- `replay_mean_gt_severity`
- `replay_exposure_per_gt`
- `replay_ratio_effective`

또한 validation에서 아래 분석을 함께 보는 것이 좋다.

- replay 대상 GT의 recovery rate
- replay 대상 class의 AP 변화
- replay 대상 object size bin별 AP 변화

## 13. Why It Can Improve mAP Aggressively

Hard Replay가 공격적인 이유는 gradient scale이 아니라 data exposure 자체를 바꾸기 때문이다.

- chronic miss GT가 훨씬 자주 배치에 들어온다.
- hard object 주변 context variation이 증가한다.
- long-tail class나 small object가 baseline보다 훨씬 자주 학습된다.

따라서 특히 `AP_small`, `AR100`, long-tail class AP에서 눈에 띄는 향상을 기대할 수 있다.

## 14. Minimal First Version

가장 먼저 구현할 버전은 아래다.

1. MDMB++ severity 기반 이미지 weighted sampler
2. epoch-level `ReplayIndex`
3. mixed batch composition

이 세 가지는 detector 내부 코드를 거의 바꾸지 않으면서도 바로 실험 가능한 Hard Replay의 최소 버전이다.
