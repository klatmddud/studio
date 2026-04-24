# Anti-Relapse Prototype Distillation

## 가설

한 번 detected되었다가 나중에 missed된 GT는 유용한 object representation을 잃은 상태일 수 있다.
Compact historical support prototype은 같은 GT의 temporal teacher로 동작하여 현재 representation을 안정화할 수 있다.

목표는 second model, teacher detector, inference-time branch 없이 relapse를 줄이는 것이다.

## 저장 Teacher

GT마다 compact prototype 하나를 저장한다.

```text
support_proto
support_quality
support_epoch
support_level optional
```

Prototype은 training에서 이미 사용되는 feature에서 가져와야 한다. FCOS에서는 GT에 배정된 positive location의
feature를 average 또는 pooling해서 만들 수 있다.

필요하기 전까지 full ROIAlign tensor는 피한다. 첫 버전은 `[C]` vector가 적절하다.

## Prototype Update

현재 GT state가 신뢰할 만할 때만 teacher를 갱신한다.

```text
if current_quality >= support_quality + margin:
    support_proto = current_proto
elif support_age >= refresh_age and current_quality >= min_quality:
    support_proto = ema(support_proto, current_proto)
```

Quality는 training-forward signal로 근사할 수 있다.

```text
quality = q_cls * cls_confidence + q_iou * regression_iou + q_ctr * centerness
```

이렇게 하면 post-NMS final detection이 필요 없다.

## Distillation Loss

Support prototype이 있고 temporal risk가 의미 있는 GT에만 적용한다.

```text
L_proto = risk(gt) * (1 - cosine(current_proto, stopgrad(support_proto)))
```

Temperature 또는 projection 기반 버전:

```text
L_proto = risk(gt) * || normalize(P(current_proto)) - stopgrad(normalize(support_proto)) ||_2^2
```

`P`는 training 중에만 사용하는 작은 projection head가 될 수 있다.

## 적용 대상

적용:

- relapsed GT
- miss streak이 높고 support prototype이 있는 GT
- 최근 recovered되어 짧은 stabilization window 안에 있는 GT

Skip:

- support prototype이 없는 GT
- 현재 positive feature quality가 낮은 경우
- assignment가 모호한 경우

## Novelty

이 방법은 generic self-distillation이 아니다. Teacher는 다른 model이나 큰 model이 아니라, 같은 detector가 과거에
성공적으로 검출했던 per-GT temporal support prototype이다. 또한 loss는 모든 object에 균일하게 적용되지 않고,
failure history에 의해 활성화된다.

현재 RASD 아이디어와 비교하면, 이 버전은 post-step feature extraction과 full support tensor를 피하므로 더 저렴해야 한다.

## 예상 Overhead

낮은 편이다.

- assigned positive에 대한 feature aggregation
- 선택된 GT당 cosine loss 1개
- optional small projection head, training only
- compact prototype storage

Inference overhead는 없다.

## Risks

- Backbone이 변하면서 old prototype이 stale해질 수 있다.
- 나쁜 early prototype이 약한 representation에 model을 고정시킬 수 있다.
- Prototype이 너무 generic하면 class-level collapse가 생길 수 있다.

완화책:

- quality-gated update
- max prototype age
- prototype 저장 전 warmup
- risk-gated loss activation
- optional class-specific normalization

