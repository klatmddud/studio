# UMR — Universal Miss Recovery

## 1. Overview

UMR (Universal Miss Recovery) 는 detection 학습 중 반복적으로 놓치는 GT를 중심으로 학습 분포, candidate 생성, 보조 supervision을 재구성하는 범용 학습 프레임워크다. 핵심 목표는 다음 세 가지다.

1. 한 번 miss된 객체를 단순히 더 큰 loss로만 처리하지 않고, 왜 miss되었는지 구조적으로 기록한다.
2. 그 기록을 이용해 다음 학습 step 또는 다음 epoch에서 해당 객체를 다시 "보게" 만든다.
3. FCOS, Faster R-CNN, DINO처럼 candidate 생성 방식이 다른 detector에도 동일한 상위 원리로 적용한다.

UMR은 세 개의 하위 구성 요소로 이뤄진다.

1. `MDMB++`: miss failure memory를 구조적으로 저장하는 공통 메모리 계층
2. `Hard Replay`: MDMB++의 정보를 이용해 학습 데이터 분포를 hard case 중심으로 재편하는 입력 단계 모듈
3. `Candidate Densification`: MDMB++의 정보를 이용해 hard GT 주변에 더 많은 training candidate를 공급하는 구조 단계 모듈

UMR의 기본 입장은 간단하다. 기존 reweighting 계열은 이미 존재하는 positive나 loss term에만 압력을 준다. 그러나 chronic miss는 애초에 모델이 해당 객체를 충분히 보지 못했거나, 적절한 candidate가 형성되지 않았거나, ranking 과정에서 반복적으로 탈락하는 경우가 많다. 따라서 UMR은 `loss amplification`이 아니라 `recovery opportunity creation`을 목표로 한다.

## 2. Why UMR

현재 MDMB 기반 모듈은 대체로 다음 패턴에 머문다.

- miss된 GT에 더 큰 loss를 부여한다.
- miss streak가 길수록 weight를 더 키운다.
- 일부 feature-level replay나 embedding modulation을 추가한다.

이 접근은 구현 난이도가 낮고 안정적이지만, 논문 기여 관점에서는 한계가 뚜렷하다.

1. 모델이 그 객체를 아예 candidate로 제안하지 못하는 상황에는 직접 대응하지 못한다.
2. miss 원인이 localization인지 classification인지 ranking인지 분해하지 못하면, 가중치만 높여도 gradient가 엉뚱한 방향으로 커질 수 있다.
3. detector family가 바뀌면 loss 구조가 달라져 재사용성이 낮아진다.

UMR은 이 문제를 다음 방식으로 해결한다.

- `MDMB++`가 failure taxonomy를 저장한다.
- `Hard Replay`가 데이터 분포를 hard object 중심으로 재조정한다.
- `Candidate Densification`이 detector-specific adapter를 통해 hard GT 주변에 recovery candidate를 추가한다.

즉 UMR은 "놓친 객체를 더 세게 혼내는" 프레임워크가 아니라, "놓친 객체를 다시 맞힐 수 있는 기회를 더 많이 만든다"는 프레임워크다.

## 3. Design Goals

UMR은 다음 요구사항을 만족하도록 설계한다.

### 3.1 Cross-architecture applicability

상위 개념은 동일해야 한다.

- 어떤 detector든 GT별 miss 상태를 기록할 수 있어야 한다.
- 어떤 detector든 현재 step에서 생성한 candidate를 canonical schema로 요약할 수 있어야 한다.
- 어떤 detector든 hard GT에 대해 추가 candidate를 공급하는 adapter를 둘 수 있어야 한다.

### 3.2 Training-only overhead

UMR은 학습 시에만 활성화되고, inference에는 영향을 주지 않는 것이 원칙이다.

- MDMB++ 업데이트: training loop 내부에서만 동작
- Hard Replay: data pipeline에만 영향
- Candidate Densification: training candidate 또는 auxiliary supervision에만 영향
- inference path: 변경 없음

### 3.3 Recoverability-oriented memory

기록해야 하는 것은 단순 miss 여부가 아니라 "어떻게 복원할 것인가"에 필요한 정보다.

- 동일 GT의 시간축 이력
- miss 원인 분류
- 마지막 성공 시점의 support 정보
- 현재 step의 candidate coverage 정보
- replay와 densification에 필요한 severity score

## 4. High-level Pipeline

UMR 학습은 아래 순서로 동작한다.

1. DataLoader가 `Hard Replay` 정책에 따라 hard image / hard object를 더 자주 샘플링한다.
2. 모델 forward에서 `Candidate Densification`이 현재 memory를 참고하여 hard GT 주변 candidate budget을 늘린다.
3. base detector loss와 densification auxiliary loss를 계산한다.
4. `optimizer.step()` 이후, post-step prediction과 intermediate candidate summary를 이용해 `MDMB++`를 갱신한다.
5. epoch 종료 시점에 MDMB++의 누적 통계를 바탕으로 replay index와 densification priority를 갱신한다.

이를 함수 수준으로 쓰면 다음과 같다.

```python
for epoch in range(num_epochs):
    replay_index = hard_replay.build_epoch_index(mdmbpp)
    sampler = hard_replay.build_sampler(dataset, replay_index)

    for images, targets in loader(sampler=sampler):
        dense_plan = candidate_densifier.plan(mdmbpp, targets)
        outputs = model(images, targets, dense_plan=dense_plan)
        loss = outputs["det_loss"] + lambda_dense * outputs.get("dense_loss", 0.0)
        loss.backward()
        optimizer.step()

        candidate_summary = adapter.collect_candidate_summary(model, images, targets)
        mdmbpp.update(targets, outputs["detections"], candidate_summary)
```

## 5. Core Representation

UMR에서 가장 중요한 개념은 per-GT failure severity다. GT $g$의 현재 epoch $t$에서의 severity를 $\phi_t(g)$라 두면, replay와 densification은 모두 이 값을 기반으로 작동한다.

예시 정의는 다음과 같다.

$$
\phi_t(g)
=
\alpha_1 \cdot \frac{s_t(g)}{\max(s_{\mathrm{global}}, 1)}
\;+\;
\alpha_2 \cdot \mathbb{1}[\text{relapse}(g, t)]
\;+\;
\alpha_3 \cdot \mathbb{1}[\text{failure\_type}(g, t) \in \mathcal{H}]
\;+\;
\alpha_4 \cdot (1 - u_t(g))
$$

여기서:

- $s_t(g)$: 현재 GT의 consecutive miss streak
- $s_{\mathrm{global}}$: 전체 GT 중 최대 miss streak
- $\text{relapse}(g, t)$: 과거에는 detected였으나 현재 miss인 경우
- $\mathcal{H}$: 특히 복원이 어려운 failure type 집합
- $u_t(g)$: 현재 GT에 대한 candidate coverage score

즉 severity는 단순 streak가 아니라 `시간축 어려움 + 현재 coverage 부족 + failure type`을 함께 반영한다.

## 6. UMR Components

### 6.1 MDMB++

MDMB++는 기존 MDMB를 확장한 failure memory 계층이다.

- 현재 epoch에서 miss된 GT만 저장하는 bank
- 모든 GT의 시간축 상태를 보존하는 persistent record
- detector family별 raw candidate를 공통 schema로 변환한 candidate snapshot
- 마지막 성공 시점의 support feature / support box / support score 기록

MDMB++는 UMR의 중심 상태 저장소이며, 나머지 두 모듈은 모두 여기서 파생된 정보를 사용한다.

자세한 내용은 [mdmb++.md](/Users/klatmddud/studio/docs/UMR/mdmb++.md)를 참고한다.

### 6.2 Hard Replay

Hard Replay는 MDMB++의 severity를 이용해 학습 분포를 다시 짠다.

- chronic miss가 포함된 이미지를 더 자주 샘플링
- hard object crop을 별도 replay pool에 저장
- 마지막 성공 support와 현재 miss crop을 함께 노출하는 pair replay 지원

이 모듈은 detector 내부 구조를 거의 건드리지 않아 가장 범용적인 성능 상승 장치다.

자세한 내용은 [hard_replay.md](/Users/klatmddud/studio/docs/UMR/hard_replay.md)를 참고한다.

### 6.3 Candidate Densification

Candidate Densification은 hard GT 주변에 더 많은 training candidate를 의도적으로 공급한다.

- FCOS: positive point / assignment region 확대
- Faster R-CNN: proposal injection, positive quota 보장
- DINO: recovery query 추가

이 모듈은 UMR의 가장 공격적인 성능 향상 축이다. hard GT가 학습 중 실제로 더 많이 관찰되게 만들기 때문이다.

자세한 내용은 [candidate_densification.md](/Users/klatmddud/studio/docs/UMR/candidate_densification.md)를 참고한다.

## 7. Shared Interfaces

구현 시 detector별 차이를 숨기기 위해 아래 추상 인터페이스를 권장한다.

### 7.1 Candidate collector adapter

```python
class CandidateCollectorAdapter(Protocol):
    def collect(
        self,
        model,
        images: list[Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[PerImageCandidateSummary]:
        ...
```

역할:

- FCOS/Faster R-CNN/DINO 내부 raw candidate를 추출
- 이를 `canonical candidate snapshot`으로 변환
- MDMB++ 업데이트에 필요한 coverage, ranking, suppression 정보를 제공

### 7.2 Hard replay planner

```python
class HardReplayPlanner:
    def build_epoch_index(self, mdmbpp) -> ReplayIndex:
        ...

    def build_sampler(self, dataset, replay_index) -> Sampler:
        ...
```

역할:

- 이미지별 replay weight 계산
- crop replay pool 정리
- epoch-level weighted sampler 또는 mixed batch composition 생성

### 7.3 Candidate densifier

```python
class CandidateDensifier(Protocol):
    def plan(self, mdmbpp, targets) -> DensePlan:
        ...

    def inject(self, model, dense_plan: DensePlan, state) -> dict[str, Any]:
        ...
```

역할:

- 현재 batch의 GT 중 densification 대상 선택
- detector-specific 추가 candidate 생성
- auxiliary supervision 또는 assignment override 제공

## 8. Architecture Adapters

| Detector | Candidate source | Densification primitive | UMR compatibility |
|---|---|---|---|
| FCOS | points, per-level boxes, logits, centerness | positive region expansion, auxiliary points | 매우 높음 |
| Faster R-CNN | RPN proposals, ROI logits, ROI scores | proposal injection, positive quota | 높음 |
| DINO | decoder queries, reference points, class logits, box deltas | recovery query injection | 높음 |

중요한 점은 detector마다 "candidate의 모양"은 다르지만, UMR이 요구하는 정보는 동일하다는 것이다.

- GT와 얼마나 가까웠는가
- 올바른 class를 제안했는가
- 높은 score였는가
- 최종 출력까지 살아남았는가

따라서 raw candidate를 canonical schema로 변환하는 adapter만 있으면 상위 알고리즘은 공유할 수 있다.

## 9. Optimization View

UMR은 loss-only 모듈이 아니므로 전체 학습 objective는 다음처럼 해석하는 것이 자연스럽다.

$$
\mathcal{L}_{\mathrm{UMR}}
=
\mathcal{L}_{\mathrm{det}}
\;+\;
\lambda_{\mathrm{dense}} \mathcal{L}_{\mathrm{dense}}
\;+\;
\lambda_{\mathrm{aux}} \mathcal{L}_{\mathrm{aux}}
$$

여기서:

- $\mathcal{L}_{\mathrm{det}}$: base detector loss
- $\mathcal{L}_{\mathrm{dense}}$: densified candidate에 대한 auxiliary loss
- $\mathcal{L}_{\mathrm{aux}}$: optional feature/support consistency loss

Hard Replay는 loss를 추가하지 않고 데이터 분포를 바꾸는 역할을 한다. 즉 objective보다 training distribution을 바꿔 empirical risk minimization이 집중하는 영역을 재배치한다.

학습 분포는 다음처럼 표현할 수 있다.

$$
p_{\mathrm{UMR}}(i)
=
\frac{w_i^\tau}{\sum_j w_j^\tau},
\qquad
w_i = 1 + \beta \cdot \sum_{g \in \mathcal{G}(i)} \phi_t(g)
$$

여기서 $i$는 이미지, $\mathcal{G}(i)$는 이미지 $i$의 GT 집합이다.

## 10. Training Loop Integration

구현 시점에는 아래 순서를 따르는 것이 가장 안전하다.

### Phase 1. MDMB++ only

- 기존 MDMB를 대체하거나 확장하는 형태로 구현
- post-step prediction과 candidate summary 수집
- failure taxonomy와 support snapshot 직렬화

### Phase 2. Hard Replay

- epoch 단위 replay index 생성
- `WeightedRandomSampler` 또는 batch composer 추가
- crop replay augmentation 추가

### Phase 3. Candidate Densification

- FCOS adapter 먼저 구현
- 이후 Faster R-CNN proposal adapter
- 마지막으로 DINO query adapter 추가

이 순서를 추천하는 이유는 replay는 data layer만 바꾸므로 가장 안정적이고, densification은 detector 내부를 건드려 가장 큰 이득과 가장 큰 구현 복잡도를 동시에 가져오기 때문이다.

## 11. Expected Gains

UMR이 공격적인 mAP 향상을 기대할 수 있는 이유는 서로 다른 failure mode를 서로 다른 단계에서 다루기 때문이다.

- `Hard Replay`: hard case 노출 빈도 증가
- `Candidate Densification`: positive candidate coverage 증가
- `MDMB++`: failure taxonomy와 support memory 제공

특히 아래 조건에서 효과를 기대할 수 있다.

- small / crowded / occluded object가 많은 데이터셋
- long-tail class가 있는 데이터셋
- baseline detector가 recall bottleneck을 보이는 경우
- "한 번 맞췄다가 다시 놓치는" relapse가 자주 나타나는 경우

## 12. Evaluation Protocol

UMR 논문/실험에서는 아래 지표를 함께 보는 것이 좋다.

1. COCO mAP: `AP`, `AP50`, `AP75`, `AP_small`, `AP_medium`, `AP_large`
2. recall 계열: `AR1`, `AR10`, `AR100`
3. UMR 전용 지표:
   - recovery rate: miss 후 $k$ epoch 이내 복원 비율
   - relapse resolution time: relapse가 detected로 복귀하기까지 걸린 epoch 수
   - candidate coverage gain: densification 전후 $u_t(g)$ 증가량
   - hard replay exposure: hard GT가 epoch당 몇 번 재노출되었는지

## 13. Novelty Summary

UMR의 논문 기여는 단일 loss 항이 아니다. 다음 세 층위의 통합이 핵심이다.

1. miss를 시간축 이력과 candidate 상태까지 포함한 structured memory로 확장했다.
2. 그 memory를 이용해 데이터 분포와 detector 내부 candidate budget을 함께 조절했다.
3. one-stage, two-stage, transformer detector를 하나의 failure-aware recovery framework 안에서 다뤘다.

즉 UMR은 "MDMB + 추가 loss"가 아니라, `memory-driven recovery training framework`로 제안하는 것이 맞다.

## 14. Recommended File-Level Implementation Plan

이 저장소 기준으로는 아래 순서를 권장한다.

1. `modules/nn/mdmb.py`를 확장하거나 `modules/nn/mdmbpp.py`를 신설한다.
2. `scripts/runtime/engine.py`에 candidate summary 수집 hook과 epoch-end replay refresh hook을 추가한다.
3. `models/detection/wrapper/fcos.py`에 FCOS candidate collector / densifier를 붙인다.
4. `models/detection/wrapper/fasterrcnn.py`, `models/detection/wrapper/dino.py`에 adapter를 순차적으로 추가한다.
5. `scripts/runtime/data.py`에 replay-aware sampler 또는 dataset wrapper를 추가한다.

구현 세부는 각 하위 문서를 참고한다.
