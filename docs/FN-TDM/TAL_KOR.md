# TAL - Transition Alignment Loss

## 목표

TAL은 FN-TDM의 auxiliary loss 컴포넌트다.

HTM은 과거 `FN -> TP` recovery에 동반된 feature direction을 발견한다. TDB는 그 direction을 저장하고
집계한다. TCS는 현재 hard GT candidate를 선택한다. TAL은 조회된 direction을 사용해 현재 candidate
embedding을 recovery direction 쪽으로 부드럽게 끌어당긴다.

```text
selected hard GT g
current embedding z_g
TDB direction D_c
TAL encourages z_g to move toward D_c
```

TAL은 training-only 모듈이다. Detector inference를 바꾸면 안 된다.

## FN-TDM 내 위치

```text
HTM: historical FN -> TP direction mining
TDB: direction prior 저장/조회
TCS: 현재 hard candidate 선택
TAL: 선택된 candidate에 direction alignment loss 적용
```

전체 training objective:

```text
L_total = L_det + lambda_tal * L_TAL
```

TAL은 작은 auxiliary term이어야 하며 base detection loss를 대체하지 않는다.

## Inputs

TAL은 TCS의 candidate를 입력으로 받는다.

```python
TCSCandidate = {
    "batch_index": int,
    "target_index": int,
    "class_id": int,
    "bbox": Tensor[4],
    "hardness": float,
    "direction": Tensor[D],
    "direction_source": str,
}
```

TAL은 detector의 current feature map도 필요로 한다.

```text
FPN feature maps or detector feature maps
```

현재 candidate embedding은 GT box에서 추출한다.

```text
z_g = normalize(projector(pool(ROIAlign(F, bbox_g))))
```

## V0 Main Method: Anchor-Shift Alignment

V0의 main method는 **Anchor-Shift Alignment**다.

TAL은 current embedding `z_g` 하나와 historical transition direction `D_c` 하나를 받는다.
Single forward pass는 현재 movement vector를 직접 제공하지 않기 때문에, TAL은 current embedding을
historical recovery direction 쪽으로 shift한 detached target을 만든다.

```text
z_anchor = stopgrad(z_g)
z_target = stopgrad(normalize(z_anchor + alpha * D_c))
L_TAL(g) = 1 - cos(z_g, z_target)
```

동등한 compact form:

```text
L_TAL(g) = 1 - cos(z_g, stopgrad(normalize(stopgrad(z_g) + alpha D_c)))
```

여기서:

```text
z_g: current trainable GT embedding
D_c: TDB에서 온 detached transition direction
alpha: direction step size
```

권장 기본값:

```text
alpha: 0.2
lambda_tal: 0.05
```

탐색 범위:

```text
alpha: [0.1, 0.2, 0.5]
lambda_tal: [0.02, 0.05, 0.1, 0.2]
```

## Anchor-Shift를 사용하는 이유

이상적인 아이디어는 현재 feature movement를 historical recovery direction과 정렬하는 것이다.
하지만 현재 batch는 각 GT에 대해 previous feature를 따로 저장하지 않는 한 하나의 embedding만 제공한다.

Anchor-Shift는 가장 단순한 direction-based approximation이다.

```text
current feature z_g
historical recovery direction D_c
desired local target z_g + alpha * D_c
```

장점:

- TDB direction을 직접 사용한다.
- Per-GT previous-feature cache가 필요 없다.
- Detector assignment가 없어도 GT ROIAlign feature로 동작한다.
- Gradient는 current embedding에만 흐른다.
- TDB direction은 detached 상태로 안정적으로 유지된다.

위험:

- Target이 self-anchored이므로 loss가 local하고 비교적 약하다.
- `alpha`가 너무 크면 target이 local valid feature neighborhood를 벗어날 수 있다.
- Easy GT에 적용하면 over-regularization이 될 수 있다. TCS가 이를 막아야 한다.

## Feature Extraction

HTM과 같은 projection space를 사용한다.

```text
FPN feature maps -> MultiScaleRoIAlign(gt_bbox) -> GAP -> projection head -> normalize
```

Projection head는 HTM과 TAL이 공유해야 한다.

```text
HTM stores directions in projector space
TAL computes current embeddings in the same projector space
```

권장 기본값:

```text
roi_output_size: 7
projector_dim: 256
normalize_embedding: true
```

Detector가 여러 FPN level을 사용한다면 HTM과 같은 `MultiScaleRoIAlign` 설정을 사용한다.

## Loss Aggregation

선택된 candidate에 대해:

```text
L_TAL = mean_g w_g * L_TAL(g)
```

V0 candidate weight:

```text
w_g = 1.0
```

Optional hardness weighting:

```text
w_g = clamp(hardness(g), min_weight, max_weight)
```

권장 optional 값:

```text
min_weight: 0.25
max_weight: 1.0
```

선택된 candidate가 없으면:

```text
L_TAL = 0
```

Zero loss는 distributed training과 logging에서 안전해야 한다.

## Stop-Gradient Rules

TAL은 엄격한 detach rule을 따라야 한다.

Gradient가 흘러야 하는 곳:

```text
current detector features
projection head, if trainable
```

Gradient가 흐르면 안 되는 곳:

```text
TDB directions
z_anchor used to construct target
z_target
HTM stored embeddings
```

Implementation:

```python
z = normalize(projector(roi_feat))
D = candidate.direction.detach()
z_anchor = z.detach()
z_target = normalize(z_anchor + alpha * D).detach()
loss = 1.0 - cosine_similarity(z, z_target)
```

## Projection Head

TAL은 projection head를 필요로 한다.

Minimal projector:

```text
Linear(C -> D)
LayerNorm or BatchNorm optional
ReLU optional
Linear(D -> D) optional
L2 normalize output
```

권장 V0:

```text
projector: Linear(C -> 256)
normalize output: true
```

첫 실험에서는 projector를 단순하게 유지한다. MLP capacity가 커지면 성능 향상이 FN-TDM 때문인지 새로운
representation head 때문인지 알기 어려워진다.

## Scheduling

TAL은 TDB에 유용한 entry가 생기기 전에는 시작하면 안 된다.

권장 조건:

```text
tal_enabled_epoch >= htm_warmup_epochs + 1
TDB has at least one valid entry for the candidate class
```

Optional loss weight warmup:

```text
lambda_tal(e) = lambda_tal_max * min(1, (e - tal_start_epoch) / warmup_epochs)
```

권장 기본값:

```text
tal_start_epoch: 1 or 2
lambda_warmup_epochs: 2
```

HTM이 더 늦게 시작되면 TDB가 채워질 때까지 TAL은 사실상 inactive 상태로 남는다.

## TAL Variants for Ablation

### TAL-Target / Anchor-Shift

Main V0 method:

```text
z_target = stopgrad(normalize(stopgrad(z_g) + alpha D_c))
L = 1 - cos(z_g, z_target)
```

Pros:

- 단순하다.
- Previous-feature cache가 필요 없다.
- Transition direction을 직접 사용한다.

Cons:

- Local/self-anchored approximation이다.

### TAL-Delta

실제 current feature movement를 TDB direction과 정렬한다.

```text
delta_g = normalize(z_g_current - stopgrad(z_g_prev))
L = 1 - cos(delta_g, D_c)
```

Pros:

- 원래 feature-trajectory 아이디어에 가장 가깝다.
- Movement direction을 직접 정렬한다.

Cons:

- GT별 previous-feature cache가 필요하다.
- Augmentation/view difference에 민감하다.
- Stale-feature 처리를 신중하게 해야 한다.

V0가 동작한 뒤 강한 V1 후보로 볼 수 있다.

### TAL-SuccessProto

Direction 대신 저장된 successful embedding 쪽으로 current embedding을 끌어당긴다.

```text
z_success_proto = weighted_mean(z_tp_i)
L = 1 - cos(z_g, z_success_proto)
```

Pros:

- 안정적이고 최적화하기 쉽다.
- Prototype distillation과 비슷하다.

Cons:

- Transition direction이 아니라 successful feature prototype을 쓰므로 novelty가 약하다.
- TDB가 `z_tp`를 저장하거나 집계해야 한다.

Main FN-TDM claim이 아니라 ablation으로 사용한다.

## TCS와의 상호작용

TAL은 TCS가 선택한 candidate에만 적용해야 한다.

```text
all GTs -> TCS -> selected candidates -> TAL
```

TAL은 safety check 외에 TCS logic을 반복하지 않는다.

```text
candidate has direction
candidate bbox is valid
candidate class is valid
```

이렇게 해야 candidate-selection ablation과 loss-design ablation이 분리된다.

## TDB Retrieval과의 상호작용

TAL은 TDB가 direction을 어떻게 만드는지와 독립적이다.

Supported retrieval variants:

```text
TDB-Last
TDB-TopK
TDB-TopK+Age
```

TAL은 다음만 소비한다.

```text
D_c or D_{c,t}
```

Direction이 latest entry에서 왔는지, quality top-K prototype인지, age-decayed top-K prototype인지는
TDB config가 결정한다.

## FCOS Integration Notes

FCOS V0:

1. Normal training forward의 FPN feature를 사용한다.
2. Current target의 GT box를 사용한다.
3. `MultiScaleRoIAlign`으로 GT feature를 추출한다.
4. Shared FN-TDM projection head를 사용한다.
5. FCOS assignment/classification signal에서 TCS candidate를 선택한다.
6. FCOS training loss에 `lambda_tal * loss_tal`을 더한다.

TAL이 수정하면 안 되는 것:

```text
FCOS target assignment
classification logits
box regression targets
centerness targets
post-processing / NMS
```

## Configuration Sketch

향후 `modules/cfg/fntdm.yaml`에 config를 추가한다.

```yaml
tal:
  enabled: true
  variant: anchor_shift

  loss:
    lambda_tal: 0.05
    alpha: 0.2
    reduction: mean
    use_candidate_weight: false
    min_weight: 0.25
    max_weight: 1.0

  schedule:
    start_epoch: 1
    lambda_warmup_epochs: 2

  features:
    source: fpn_roi_align_gt
    roi_output_size: 7
    projector_dim: 256
    normalize: true

  safety:
    skip_invalid_direction: true
    min_direction_norm: 1.0e-6
    skip_invalid_bbox: true

  logging:
    save_summary: true
```

## Logging

Iteration 또는 epoch summary:

```text
num_candidates
num_valid_candidates
num_skipped_invalid_direction
loss_tal
mean_cosine
mean_candidate_hardness
lambda_tal_current
```

Optional per-class summary:

```text
class_id
num_candidates
loss_tal_mean
cosine_mean
```

기본적으로 embedding이나 full direction vector를 기록하지 않는다.

## Edge Cases

- Candidate가 없음: zero loss를 반환한다.
- Candidate direction이 `None`: skip한다.
- Direction norm이 invalid: skip한다.
- GT bbox가 invalid 또는 empty: skip한다.
- ROIAlign 결과가 NaN/Inf: candidate를 skip하고 log한다.
- Projection output norm이 너무 작음: candidate를 skip한다.
- Training 초기에 TDB가 비어 있음: TAL은 inactive 상태로 남는다.

## V0 Implementation Checklist

1. `TransitionAlignmentLoss` module을 추가한다.
2. HTM과 공유되는 GT ROIAlign feature extraction을 추가한다.
3. Simple projection head를 추가한다.
4. Anchor-Shift target construction을 구현한다.
5. Zero-candidate behavior가 안전한 cosine loss를 구현한다.
6. Lambda scheduling을 추가한다.
7. TCS candidate를 TAL에 연결한다.
8. Detector training loss dict에 TAL loss를 추가한다.
9. Summary logging을 추가한다.
10. Detach 및 no-op behavior unit test를 추가한다.

## Minimal Unit Tests

Loss behavior:

```text
aligned z and target gives smaller loss than opposite direction
empty candidates returns zero scalar loss
invalid direction is skipped
```

Gradient:

```text
gradient flows to current z/projector
gradient does not flow to candidate.direction
gradient does not flow to z_target
```

Scheduling:

```text
lambda is zero before start_epoch
lambda warms up to lambda_tal
```

Device:

```text
candidate directions and ROI features are on the same device
```

## Research Claim

TAL은 다음처럼 설명할 수 있다.

```text
Transition Alignment Loss converts historical false-negative recovery directions
into a training-time auxiliary objective by locally shifting selected hard-instance
embeddings toward class-wise transition direction priors.
```

TAL은 FN-TDM을 analysis memory에서 active regularizer로 바꾸는 컴포넌트다.
