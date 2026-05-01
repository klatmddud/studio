# Query-guided Adaptive Feature Pyramid 방법론 정리

**문서 목적.** Object Detection에서 `Query-guided Adaptive Feature Pyramid`를 하나의 neck/head 재설계 방법론으로 보았을 때, 비교되는 선행 연구들이 무엇을 해결했고 무엇을 남겼는지, 그리고 이 방법론이 어떤 문제를 해결하도록 설계될 수 있는지 정리합니다.

**작성 기준.** 2026-04-28 기준 공개 논문/공식 논문 페이지를 근거로 정리했습니다. 공개 문헌에서 `QAFPN`이라는 약어는 FastQAFPN-YOLOv8s 논문처럼 `QARepNext-Rep-PAN` 기반 경량 neck을 뜻하는 경우가 있어, 본 문서의 `Query-guided Adaptive Feature Pyramid`와 구분합니다. 이하에서는 혼동을 피하기 위해 본 방법론을 **QG-AFP**라고 표기합니다.

---

## 1. 핵심 요약

QG-AFP의 핵심 아이디어는 **detection query가 feature pyramid의 읽기(read), 융합(fusion), 재기록(write-back), 고해상도 계산 예산을 제어하게 하는 것**입니다. 기존 FPN 계열은 대체로 backbone에서 나온 다중 해상도 feature를 정해진 경로 또는 feature 자체의 attention으로 합칩니다. DETR 계열은 object query를 decoder/head에서 강하게 사용하지만, query가 neck 단계의 feature pyramid를 직접 재구성하는 경우는 제한적입니다.

QG-AFP는 이 간극을 겨냥합니다.

1. **고정 경로 fusion의 한계**를 줄입니다. FPN/PANet/BiFPN/AFPN은 경로 또는 fusion topology를 개선하지만, 대부분의 fusion은 query-free입니다.
2. **feature-level semantic mismatch와 scale conflict**를 줄입니다. 각 object query가 “이 후보 객체는 어느 level의 어떤 위치 정보를 더 필요로 하는가”를 동적으로 결정합니다.
3. **dense background 계산 낭비**를 줄입니다. QueryDet처럼 coarse query로 sparse high-resolution computation을 유도할 수 있지만, QG-AFP는 이를 단순 head acceleration이 아니라 pyramid fusion 자체의 routing 신호로 사용합니다.
4. **neck과 head의 목적 불일치**를 줄입니다. Query가 head에서만 쓰이지 않고 neck의 multi-scale representation 형성에 관여하므로, 최종 classification/regression에 필요한 feature가 더 직접적으로 생성됩니다.

한 문장으로 정리하면, QG-AFP는 **FPN류의 multi-scale feature fusion**과 **DETR/Sparse R-CNN류의 object query reasoning**을 결합해, query-conditioned, object-aware, sparse-adaptive feature pyramid를 만드는 방법론입니다.

---

## 2. 문제 정의

### 2.1 Feature Pyramid가 해결하려는 기본 문제

Object detection은 작은 객체, 큰 객체, 부분 가림, 밀집 장면, 배경 clutter를 동시에 다룹니다. Backbone의 깊은 feature는 semantic 정보가 강하지만 해상도가 낮고, 얕은 feature는 위치 정보가 풍부하지만 semantic 정보가 약합니다. FPN은 이 둘을 top-down path와 lateral connection으로 결합해, 모든 scale에서 semantic feature map을 만들었습니다. FPN 논문은 backbone의 pyramidal hierarchy를 활용해 큰 추가 비용 없이 feature pyramid를 만들고, Faster R-CNN 기반 COCO detection에서 강한 성능을 보였다고 보고했습니다 [1].

### 2.2 기존 pyramid neck의 공통 한계

대부분의 FPN 계열 neck은 다음 네 가지 한계를 공유합니다.

- **Fusion rule이 객체별로 다르지 않습니다.** 작은 객체, 큰 객체, 길쭉한 객체, occluded 객체가 필요로 하는 level 조합은 다른데, 일반 FPN/PAN류는 동일한 fusion rule을 모든 위치와 객체 후보에 적용합니다.
- **Feature 자체만 보고 fusion합니다.** ASFF, AdaFPN, FaPN 등은 spatial weight, adaptive upsampling, alignment를 도입하지만, “head가 추적 중인 object hypothesis” 또는 “query token”이 fusion을 직접 지휘하지는 않습니다.
- **Head와 neck 사이의 semantic gap이 남습니다.** DETR류는 query가 prediction을 담당하지만, neck에서 만들어지는 multi-scale feature는 query와 분리되어 있습니다.
- **고해상도 계산이 비효율적입니다.** 작은 객체를 위해 P2/P3 같은 high-resolution level을 전부 계산하면 background 영역에도 큰 비용이 듭니다. QueryDet은 이 문제를 query mechanism으로 완화했지만, 목표는 주로 high-resolution head computation의 sparsification입니다 [12].

### 2.3 QG-AFP가 겨냥하는 명확한 문제

QG-AFP의 문제 설정은 다음과 같습니다.

> 입력 feature pyramid `P = {P3, P4, P5, ...}`와 object/proposal/query set `Q = {q1, ..., qN}`가 주어졌을 때, 각 query가 자신의 위치, scale, semantic 불확실성에 따라 pyramid level, sampling 위치, fusion weight, high-resolution refinement budget을 동적으로 선택하게 하여 detection에 필요한 multi-scale feature를 재구성한다.

즉, QG-AFP는 단순히 “더 좋은 FPN”이 아니라 **query-conditioned adaptive pyramid routing module**입니다.

---

## 3. 비교되는 선행 연구: 해결한 것과 남긴 것

### 3.1 FPN: top-down semantic pyramid

**해결한 문제.** FPN은 backbone의 multi-level feature를 top-down pathway와 lateral connection으로 결합하여, low-level feature에도 high-level semantic 정보를 부여했습니다. 이 구조는 handcrafted image pyramid보다 효율적이고, multi-scale object detection의 표준 neck이 되었습니다 [1].

**해결하지 못한 문제.** FPN의 fusion은 주로 upsampling 후 element-wise addition입니다. 이는 feature context, level 간 semantic gap, spatial misalignment를 명시적으로 다루지 않습니다. 또한 모든 위치와 객체 후보에 동일한 fusion rule을 적용하므로 object-specific scale selection이 없습니다.

**QG-AFP와의 차이.** QG-AFP는 FPN의 multi-scale feature base를 유지하되, object query가 level gate와 spatial mask를 생성해 객체별 feature routing을 수행합니다.

### 3.2 PANet / PAFPN: bottom-up localization path와 adaptive feature pooling

**해결한 문제.** PANet은 bottom-up path augmentation을 추가하여 lower layer의 localization signal이 top layer로 더 짧은 경로를 통해 전달되게 했고, adaptive feature pooling으로 RoI와 모든 feature level을 연결해 proposal subnetwork에 직접 전달되게 했습니다 [2]. YOLO 계열의 PAFPN류도 top-down + bottom-up 경로를 통해 semantic 정보와 localization 정보를 함께 흐르게 합니다.

**해결하지 못한 문제.** Fusion 경로는 여전히 구조적으로 고정되어 있습니다. Object query가 “이 proposal은 P3의 위치 정보가 더 중요하다” 또는 “이 후보는 P5 semantic이 충분하다”처럼 개별 routing을 수행하지 않습니다. 모든 level과 위치가 유사한 방식으로 계산되므로 background 계산 절감 효과도 제한적입니다.

**QG-AFP와의 차이.** QG-AFP는 PAN류의 양방향 경로를 query-conditioned gate로 보완합니다. Proposal 또는 top-k dense query가 feature flow의 우선순위를 결정합니다.

### 3.3 NAS-FPN: feature pyramid topology search

**해결한 문제.** NAS-FPN은 manually designed pyramid 구조 대신 Neural Architecture Search로 cross-scale connection space를 탐색하여, top-down과 bottom-up connection의 조합을 자동으로 찾았습니다. RetinaNet 기반에서 accuracy-latency trade-off를 개선했다고 보고했습니다 [3].

**해결하지 못한 문제.** 학습된 topology는 inference 시 고정 구조입니다. Image instance, object candidate, class/query 상태에 따라 연결 구조가 바뀌지는 않습니다. 또한 NAS 비용과 재현성이 부담입니다.

**QG-AFP와의 차이.** QG-AFP는 topology를 완전히 search하지 않아도, query-conditioned gating으로 입력 이미지와 객체 후보에 따라 effective topology를 동적으로 바꿉니다.

### 3.4 Libra R-CNN / Balanced Feature Pyramid

**해결한 문제.** Libra R-CNN은 training imbalance를 sample, feature, objective level로 나누어 분석하고, feature level imbalance에는 Balanced Feature Pyramid를 사용했습니다. 논문은 IoU-balanced sampling, Balanced Feature Pyramid, Balanced L1 loss를 결합해 FPN Faster R-CNN 및 RetinaNet 대비 AP 향상을 보고했습니다 [4].

**해결하지 못한 문제.** Balanced Feature Pyramid는 level 간 균형을 맞추는 데 초점이 있으며, 객체별 query-conditioned 선택은 수행하지 않습니다. Feature 균형은 전역적/구조적 보정에 가깝습니다.

**QG-AFP와의 차이.** QG-AFP는 “feature level 전체의 균형”보다 “각 query가 필요로 하는 level 조합”을 학습합니다.

### 3.5 BiFPN / EfficientDet: weighted bidirectional fusion

**해결한 문제.** EfficientDet은 BiFPN을 제안해 bidirectional multi-scale feature fusion을 효율적으로 수행하고, learnable fusion weight와 compound scaling을 결합했습니다. EfficientDet-D7은 COCO test-dev에서 55.1 AP를 77M parameters, 410B FLOPs로 달성했다고 보고했습니다 [5].

**해결하지 못한 문제.** BiFPN의 fusion weight는 level/node 단위의 learned scalar에 가깝고, object query 또는 instance별 상태에 따라 달라지는 routing은 아닙니다. 즉, 같은 이미지 안의 작은 객체와 큰 객체가 서로 다른 query-level policy를 가지지는 않습니다.

**QG-AFP와의 차이.** QG-AFP는 BiFPN의 weighted fusion을 query-conditioned weight로 확장합니다. 전역 level weight가 아니라 `g_{i,l}`처럼 query `i`와 level `l`에 의존하는 gate를 학습할 수 있습니다.

### 3.6 ASFF: adaptive spatial feature fusion

**해결한 문제.** ASFF는 single-shot detector에서 feature scale 간 inconsistency를 주요 한계로 보고, spatial position별로 서로 다른 scale feature를 weighting하여 conflictive information을 suppress합니다. YOLOv3 기반에서 speed-accuracy trade-off 개선을 보고했습니다 [6].

**해결하지 못한 문제.** ASFF의 weight는 feature map에서 직접 예측되는 spatial fusion weight입니다. 객체 후보 또는 query token이 fusion policy를 명시적으로 지휘하지 않습니다. Dense spatial fusion이므로 background 영역에도 fusion이 일어납니다.

**QG-AFP와의 차이.** QG-AFP는 ASFF의 spatial adaptivity에 object query를 추가합니다. 예를 들어 query의 reference box, class embedding, uncertainty가 spatial fusion mask를 조절할 수 있습니다.

### 3.7 CARAFE: content-aware upsampling

**해결한 문제.** CARAFE는 feature upsampling을 단순 bilinear/deconvolution이 아니라 content-aware reassembly로 바꾸었습니다. 큰 receptive field, sample-specific kernel generation, 낮은 overhead를 장점으로 제시했고 object detection, instance/semantic segmentation, inpainting에서 consistent gain을 보고했습니다 [7].

**해결하지 못한 문제.** CARAFE는 upsampling operator의 content adaptivity를 해결하지만, 어느 객체가 어느 level을 얼마나 필요로 하는지 결정하지 않습니다. Query semantics, proposal 상태, detection uncertainty와는 직접 연결되지 않습니다.

**QG-AFP와의 차이.** QG-AFP는 query-conditioned upsampling/alignment로 확장될 수 있습니다. 즉, dynamic kernel 또는 offset이 local content뿐 아니라 object query에 의해 조절됩니다.

### 3.8 AugFPN: FPN의 세 가지 설계 결함 보완

**해결한 문제.** AugFPN은 FPN의 design defects를 분석하고 Consistent Supervision, Residual Feature Augmentation, Soft RoI Selection을 제안했습니다. Consistent Supervision은 scale 간 semantic gap을 줄이고, Residual Feature Augmentation은 highest pyramid level의 information loss를 줄이며, Soft RoI Selection은 RoI feature를 adaptive하게 선택합니다. Faster R-CNN, RetinaNet, FCOS에서 AP 향상을 보고했습니다 [8].

**해결하지 못한 문제.** AugFPN의 RoI selection은 query-like 선택과 가깝지만, neck 전체의 feature pyramid fusion을 object query가 동적으로 제어하지는 않습니다. 또한 one-stage dense detector와 transformer detector에서 query-conditioned pyramid reconstruction으로 일반화되지는 않습니다.

**QG-AFP와의 차이.** QG-AFP는 RoI feature selection을 넘어서, query가 pyramid fusion 자체의 gate, offset, mask, write-back update를 생성하게 합니다.

### 3.9 AdaFPN: adaptive upsampling + adaptive fusion

**해결한 문제.** Adaptive Feature Pyramid Networks for Object Detection은 FPN의 단순 fusion이 context를 고려하지 못하고, 전통적 upsampling이 feature misalignment와 detail loss를 유발한다고 보았습니다. 이를 위해 adaptive feature upsampling(AdaUp)과 adaptive feature fusion(AFF)을 도입했고, Faster R-CNN과 FCOS에서 각각 AP 향상을 보고했습니다 [9].

**해결하지 못한 문제.** AdaFPN의 adaptive upsampling/fusion은 feature-driven입니다. Object query가 후보 객체의 semantic, 위치, scale, uncertainty를 바탕으로 fusion을 제어하지는 않습니다.

**QG-AFP와의 차이.** QG-AFP는 AdaFPN의 adaptive operator를 query-aware operator로 바꿉니다. 예를 들어 `AdaUp(P_l, P_{l-1})` 대신 `Q-AdaUp(P_l, P_{l-1}, q_i, r_i)`처럼 query/reference point가 sampling 위치와 kernel을 조건화합니다.

### 3.10 FaPN: feature alignment

**해결한 문제.** FaPN은 upsampled higher-level feature와 local feature의 direct pixel addition이 context misalignment를 만들고 boundary prediction에 악영향을 준다고 지적했습니다. Feature Alignment Module과 Feature Selection Module을 통해 contextual alignment와 lower-level detail 강조를 수행했고 Faster/Mask R-CNN과 결합 시 FPN 대비 AP/mIoU 향상을 보고했습니다 [10].

**해결하지 못한 문제.** Alignment는 feature map 사이의 geometric/contextual alignment에 집중합니다. Query가 객체별로 필요한 alignment 범위와 direction을 제어하지 않습니다.

**QG-AFP와의 차이.** QG-AFP는 query reference point와 predicted box geometry를 이용해 alignment offset을 조건화할 수 있습니다. 이는 작은 객체, 긴 객체, occluded 객체에서 더 세밀한 adaptive alignment를 가능하게 합니다.

### 3.11 AFPN: non-adjacent level interaction과 asymptotic fusion

**해결한 문제.** AFPN은 top-down/bottom-up FPN류가 non-adjacent level interaction에서 feature loss/degradation을 겪는다고 보고, adjacent low-level feature부터 시작해 high-level feature를 점진적으로 통합하는 asymptotic fusion을 제안했습니다. 또한 multi-object information conflict를 줄이기 위해 adaptive spatial fusion을 사용했습니다 [11].

**해결하지 못한 문제.** AFPN은 non-adjacent level fusion과 spatial conflict를 다루지만, fusion policy가 object query 또는 detection hypothesis에 의해 직접 바뀌지는 않습니다. 모든 공간 위치가 pyramid fusion 대상이므로 dense background 계산도 남습니다.

**QG-AFP와의 차이.** QG-AFP는 AFPN의 multi-level direct interaction에 query-conditioned routing을 추가합니다. Non-adjacent fusion을 하더라도 query가 어떤 pair를 활성화할지 결정합니다.

### 3.12 QueryDet: sparse query로 high-resolution small object detection 가속

**해결한 문제.** QueryDet은 작은 객체 탐지를 위해 high-resolution feature를 전부 쓰면 계산량이 커지는 문제를 지적하고, low-resolution feature에서 작은 객체의 coarse location을 먼저 예측한 뒤 그 위치가 guide하는 high-resolution feature 영역에서만 정확한 detection을 수행합니다. COCO와 VisDrone에서 mAP 및 mAP-small 개선과 high-resolution inference acceleration을 보고했습니다 [12].

**해결하지 못한 문제.** QueryDet의 query는 주로 high-resolution head computation을 sparse하게 하는 coarse location guide입니다. Multi-scale pyramid fusion 전체를 query-conditioned하게 재구성하거나, query semantic이 level fusion weight를 계속 업데이트하는 구조는 아닙니다.

**QG-AFP와의 차이.** QG-AFP는 QueryDet의 sparse high-res activation을 포함할 수 있지만, 목표가 더 넓습니다. Query가 high-resolution 영역 선택뿐 아니라 scale routing, feature alignment, cross-level write-back까지 관여합니다.

### 3.13 Dynamic Head: level/spatial/channel/task attention head

**해결한 문제.** Dynamic Head는 detection head의 localization/classification 복잡성을 scale-aware, spatial-aware, task-aware attention으로 통합했습니다. Feature level 간, spatial location 간, output channel/task 간 attention을 결합해 head representation을 강화했습니다 [13].

**해결하지 못한 문제.** Dynamic Head는 head 표현력 강화에 초점이 있으며, neck의 pyramid feature를 query-conditioned하게 재생성하는 방법은 아닙니다. Query-to-pyramid feedback loop가 명시적이지 않습니다.

**QG-AFP와의 차이.** QG-AFP는 head attention 이전 또는 내부에서 query가 pyramid 자체를 조건화합니다. Dynamic Head와 결합하면, query-guided neck + attention-guided head라는 계층적 설계가 가능합니다.

### 3.14 DETR / Deformable DETR / Sparse R-CNN: object query 기반 detection

**해결한 문제.** DETR은 object detection을 set prediction으로 보고, Hungarian matching과 transformer encoder-decoder, learned object queries를 사용해 anchor/NMS 등 hand-designed component를 줄였습니다 [14]. Deformable DETR은 DETR의 느린 수렴과 제한된 feature resolution 문제를 multi-scale deformable attention으로 완화했고, reference 주변의 소수 sampling point만 attend합니다 [15]. Sparse R-CNN은 dense anchor candidate 대신 고정된 수의 learnable proposals를 사용해 NMS 없이 최종 prediction을 출력합니다 [16].

**해결하지 못한 문제.** 이 계열은 query를 head/decoder에서 강하게 사용하지만, 일반적으로 query가 FPN류 neck의 feature fusion topology, alignment, high-resolution budget을 직접 제어하지 않습니다. Deformable DETR의 query는 multi-scale feature를 읽지만, 그 읽기 결과가 pyramid map을 다시 객체별로 재구성하는 것은 별도 설계가 필요합니다.

**QG-AFP와의 차이.** QG-AFP는 query를 단순 read pointer가 아니라 pyramid fusion controller로 사용합니다. 즉, query가 feature map에서 정보를 읽고 끝나는 것이 아니라, feature map을 다시 업데이트해 downstream dense/sparse head 모두에 영향을 주게 합니다.

### 3.15 FastQAFPN-YOLOv8s: 약어 충돌 사례

**해결한 문제.** FastQAFPN-YOLOv8s는 walnut unseparated material detection을 위해 Fasternet backbone, ECIoU loss, QAFPN neck을 결합했습니다. 여기서 QAFPN은 `QARepNext-Rep-PAN` feature fusion network로, YOLOv8s의 PAN-FPN을 QARepNext reparameterization 기반 Rep-PAN으로 대체해 속도와 경량화를 추구합니다 [17].

**해결하지 못한 문제.** 이 QAFPN은 query-guided pyramid가 아니라 quantization-aware/reparameterization 기반 경량 feature fusion입니다. Query token이 pyramid fusion을 제어하는 방법론과는 다른 축입니다.

**QG-AFP와의 차이.** QG-AFP는 QAFPN이라는 약어를 쓰더라도, `QARepNext-Rep-PAN`이 아니라 `Query-guided Adaptive Feature Pyramid`로 정의되어야 합니다. 논문 작성 시 약어 충돌을 피하기 위해 `QG-AFP`, `QAF-Pyramid`, `QAPN` 등 다른 표기를 권장합니다.

---

## 4. QG-AFP가 해결하는 문제

### 4.1 Fixed fusion에서 query-conditioned fusion으로

기존 FPN 계열은 대부분 level graph가 고정되어 있습니다. BiFPN처럼 learnable weight를 쓰더라도, weight는 object-specific하지 않습니다. QG-AFP는 각 query `q_i`에 대해 level gate `g_{i,l}`를 학습합니다.

```text
입력: query q_i, reference box r_i, pyramid features {P_l}
출력: query별 level gate g_i = softmax_l Router(q_i, r_i, pooled(P_l, r_i)))

예: 작은 객체 query  -> P2/P3 gate 증가
    큰 객체 query    -> P5/P6 gate 증가
    occluded query   -> local detail + high-level context 동시 증가
```

이렇게 하면 pyramid가 이미지 전체에 대해 하나의 fusion policy를 쓰지 않고, candidate object 단위로 fusion policy를 바꿀 수 있습니다.

### 4.2 Semantic gap과 scale conflict 완화

FPN류의 핵심 문제 중 하나는 low-level feature의 detail과 high-level feature의 semantic이 서로 다른 distribution을 갖는다는 점입니다. AugFPN, AdaFPN, FaPN, AFPN은 이 문제를 다양한 방식으로 다루었습니다. QG-AFP는 여기에 query semantic을 추가합니다.

- Class-aware query는 class-discriminative detail이 필요한 level을 선택합니다.
- Box-aware query는 reference box의 scale/aspect ratio로 적절한 level과 sampling radius를 조절합니다.
- Uncertainty-aware query는 localization uncertainty가 클 때 더 넓은 multi-level search를 허용합니다.

즉, scale fusion의 기준이 “feature map 자체의 통계”에서 “검출해야 하는 객체 hypothesis”로 이동합니다.

### 4.3 Dense background 계산 절감

작은 객체 성능을 위해 P2/P3 feature를 전부 강화하면 계산량이 급증합니다. QG-AFP는 query heatmap 또는 proposal reference를 이용해 high-resolution computation을 foreground candidate 주변에 집중시킬 수 있습니다.

```text
dense FPN: 모든 P2 위치를 동일하게 계산
QG-AFP: query가 있는 sparse region 또는 high uncertainty region만 고해상도 강화
```

QueryDet이 coarse low-resolution query로 high-resolution detection을 sparse하게 만든 것처럼, QG-AFP도 foreground-query mask를 사용해 background 영역의 expensive fusion을 줄일 수 있습니다. 차이는 QG-AFP가 high-resolution head만 줄이는 것이 아니라 pyramid fusion 자체를 sparse-adaptive하게 만든다는 점입니다.

### 4.4 Neck-head semantic alignment

DETR류에서는 object query가 decoder에서 feature를 읽고 prediction합니다. YOLO/FCOS류에서는 dense head가 각 pyramid level feature를 직접 분류/회귀합니다. 두 경우 모두 neck이 만든 feature와 head의 decision variable 사이에 mismatch가 생길 수 있습니다.

QG-AFP는 query가 neck에 개입하므로, 최종 head가 사용할 representation을 query 목적에 맞게 정렬합니다.

- DETR/RT-DETR류: encoder proposal 또는 decoder query가 pyramid feature를 condition합니다.
- Sparse R-CNN류: learnable proposal이 RoI feature pooling 전에 pyramid를 재가중합니다.
- YOLO/FCOS류: top-k objectness seed query가 dense head 직전의 P3/P4/P5를 query-aware로 보강합니다.

### 4.5 작은 객체, 밀집 객체, occlusion에 대한 효과 가능성

QG-AFP의 성능 향상 가능성이 큰 영역은 다음입니다.

- **Small object.** Query가 P2/P3의 고해상도 detail을 선택적으로 강화합니다.
- **Dense object.** Query별 local mask를 분리하면 인접 객체의 feature conflict를 줄일 수 있습니다.
- **Occlusion.** Query가 high-level context와 low-level boundary를 동시에 가져오도록 multi-level gate를 학습할 수 있습니다.
- **Remote sensing/UAV.** 작은 객체가 많고 background가 넓은 장면에서 sparse query-guided high-resolution update가 계산 효율에 유리합니다.
- **Industrial defect.** defect의 scale과 texture가 불규칙할 때 query-conditioned local refinement가 유효할 수 있습니다.

---

## 5. QG-AFP 방법론 상세

### 5.1 전체 구조

QG-AFP는 다음 네 블록으로 구성할 수 있습니다.

```text
Backbone C3-C5
   ↓
Base Pyramid Builder: FPN / PAFPN / BiFPN / AFPN 중 하나
   ↓
Query Generator: learned query, proposal query, top-k objectness query, class prototype query
   ↓
Query-guided Adaptive Pyramid Module
   ├─ Query-Scale Router
   ├─ Query-guided Alignment Sampler
   ├─ Query-to-Pyramid Sparse Fusion
   └─ Query-to-Feature Write-back
   ↓
Detection Head
   ├─ dense head: YOLO/FCOS/RetinaNet류
   ├─ RoI head: Faster/Cascade/Sparse R-CNN류
   └─ transformer decoder: DETR/RT-DETR/DINO류
```

### 5.2 Query Generator

Query는 feature pyramid를 제어하는 controller입니다. Detector 유형에 따라 query source를 다르게 둘 수 있습니다.

#### 5.2.1 1-stage detector용 query

Anchor-free detector에서는 P3/P4/P5의 objectness 또는 classification logit에서 top-k seed를 뽑을 수 있습니다.

```text
q_i = MLP([P_l(x_i, y_i), level_embedding_l, position_embedding(x_i, y_i), objectness_i])
r_i = predicted/reference point 또는 coarse box
```

장점은 YOLO/FCOS에 쉽게 삽입 가능하고, query가 foreground 후보에서 나오므로 background 계산을 줄이기 쉽다는 점입니다.

#### 5.2.2 2-stage detector용 query

Faster R-CNN/Cascade R-CNN에서는 RPN proposal 또는 RoI feature를 query로 변환합니다.

```text
q_i = MLP(RoIAlign(P, proposal_i))
r_i = proposal_i
```

이 방식은 RoIAlign 이전의 pyramid feature를 proposal-aware하게 재조정하거나, RoI feature pooling 이후 refinement head에 query-guided level selection을 적용할 수 있습니다.

#### 5.2.3 Transformer detector용 query

DETR/Deformable DETR/DINO/RT-DETR류에서는 learned object query, encoder proposal, reference point를 그대로 사용할 수 있습니다.

```text
q_i = decoder query 또는 encoder proposal embedding
r_i = reference point / anchor box
```

가장 자연스러운 구현은 Deformable DETR의 reference point 기반 multi-scale sampling을 neck fusion까지 확장하는 것입니다.

### 5.3 Query-Scale Router

Query-Scale Router는 query별로 어떤 pyramid level을 얼마나 사용할지 결정합니다.

```text
s_{i,l} = MLP([q_i, e_l, pooled(P_l, r_i), log(area(r_i)), uncertainty_i])
g_{i,l} = softmax_l(s_{i,l})
```

- `q_i`: object/proposal query embedding
- `e_l`: pyramid level embedding
- `pooled(P_l, r_i)`: query reference 주변 feature
- `area(r_i)`: reference box scale prior
- `uncertainty_i`: classification entropy 또는 box variance

해석은 간단합니다. `g_{i,l}`이 크면 query `i`가 level `l`의 정보를 많이 필요로 한다는 뜻입니다.

### 5.4 Query-guided Alignment Sampler

단순 upsampling은 feature misalignment를 만들 수 있습니다. QG-AFP는 query를 이용해 sampling offset과 aggregation kernel을 생성할 수 있습니다.

```text
Δp_{i,l,k}, a_{i,l,k} = OffsetKernelNet(q_i, r_i, P_l around r_i)
z_i = Σ_l Σ_k g_{i,l} · a_{i,l,k} · W_v P_l(project_l(r_i) + Δp_{i,l,k})
```

이 구조는 Deformable Attention의 sparse sampling과 유사하지만, 목적이 decoder output 생성에만 있지 않고 pyramid update에도 사용된다는 점이 다릅니다.

### 5.5 Query-to-Pyramid Sparse Fusion

Query가 읽은 정보를 다시 pyramid feature로 주입합니다. 가장 단순한 방식은 query별 update vector를 reference 위치 주변에 scatter하는 것입니다.

```text
u_i = MLP([q_i, z_i])
M_{i,l}(x, y) = g_{i,l} · Gaussian(project_l(r_i), σ_i)
P'_l(x, y) = P_l(x, y) + Σ_i M_{i,l}(x, y) · W_u u_i
```

여기서 `M_{i,l}`은 query가 level `l`에서 영향을 줄 spatial mask입니다. 작은 객체는 작은 radius, 불확실한 객체는 큰 radius, occluded 객체는 더 넓은 context radius를 가질 수 있습니다.

### 5.6 Cross-level Adaptive Fusion

Write-back 이후에는 cross-level fusion을 다시 수행합니다. 이때 fusion도 query mask에 의해 조절됩니다.

```text
P^out_l = Norm(
  P'_l + Σ_m A_{l←m}(x, y; Q) · Transform_{m→l}(P'_m)
)
```

`A_{l←m}`은 query aggregate mask 또는 query-level gate에서 만들어집니다. 예를 들어 P3 위치가 여러 small-object query와 강하게 연결되어 있으면 P4/P5 semantic을 더 많이 받아오고, background 위치는 update를 약하게 받습니다.

### 5.7 Detection Head 연결 방식

#### 5.7.1 Dense head 방식

YOLO/FCOS/RetinaNet류에서는 `P^out_l`을 기존 head에 그대로 넣습니다. 기존 head 구조를 크게 바꾸지 않고 neck만 교체할 수 있어 ablation이 쉽습니다.

```text
C3-C5 → PAFPN/BiFPN → QG-AFP → decoupled cls/reg head
```

#### 5.7.2 RoI head 방식

Faster/Cascade R-CNN류에서는 QG-AFP를 RoIAlign 전에 넣으면 proposal 주변의 pyramid feature를 더 좋게 만들 수 있고, RoIAlign 이후에 넣으면 proposal feature selection module처럼 동작합니다.

```text
RPN proposals → proposal queries → QG-AFP(P) → RoIAlign(P') → box/mask head
```

#### 5.7.3 Transformer decoder 방식

DETR류에서는 encoder output 또는 decoder query를 사용해 pyramid feature를 반복적으로 refine할 수 있습니다.

```text
P → query generator → QG-AFP(P, Q) → deformable decoder(Q, P')
```

반복형 구조를 쓰면 `Q`와 `P`가 서로 갱신됩니다.

```text
for t in 1..T:
    Q_t = query_update(Q_{t-1}, read(P_{t-1}))
    P_t = pyramid_update(P_{t-1}, Q_t)
```

이는 query-to-feature feedback loop를 명시화하는 설계입니다.

---

## 6. 손실 함수와 학습 전략

### 6.1 기본 detection loss

Detector 계열에 맞는 loss를 그대로 둡니다.

- YOLO/FCOS/RetinaNet류: classification focal/varifocal loss, box IoU/GIoU/DIoU/CIoU, DFL 등
- Faster/Cascade R-CNN류: RPN loss, RoI classification/regression loss
- DETR류: Hungarian matching 기반 classification, L1, GIoU loss

### 6.2 Query-scale supervision

GT box scale에 따라 권장 level을 정의하고 query gate를 보조 supervision할 수 있습니다.

```text
l*(b) = floor(l0 + log2(sqrt(area(b)) / s0))
L_scale = CE(g_i, l*(matched_gt_i))
```

다만 너무 강하게 주면 모델이 cross-scale fusion을 자유롭게 학습하지 못합니다. 초반에는 약하게 주고, 후반에는 entropy regularization과 병행하는 방식을 권장합니다.

### 6.3 Sparse budget regularization

Query-guided mask가 모든 위치를 활성화하면 계산 절감 효과가 사라집니다.

```text
L_sparse = Σ_l mean(M_l)
또는 top-k hard/sparse mask budget 제약
```

이 항은 AP와 latency를 함께 최적화해야 하는 edge/real-time detector에서 중요합니다.

### 6.4 Query-feature consistency loss

Query가 강조한 영역과 objectness heatmap이 일치하도록 consistency loss를 둘 수 있습니다.

```text
L_cons = BCE(QueryMask_l, foreground_heatmap_l)
```

작은 객체가 많은 데이터셋에서는 foreground recall을 우선시해 false negative query를 줄이는 것이 중요합니다.

### 6.5 Warm-up strategy

Query가 학습 초기에 불안정하면 pyramid update가 noise를 증폭할 수 있습니다. 다음 전략이 안전합니다.

1. **Stage 1:** baseline FPN/PAFPN detector를 안정적으로 학습합니다.
2. **Stage 2:** QG-AFP를 identity-biased residual module로 삽입합니다.
3. **Stage 3:** gate entropy와 sparse budget을 점진적으로 강화합니다.
4. **Stage 4:** end-to-end fine-tuning으로 query generator와 pyramid update를 함께 최적화합니다.

---

## 7. 선행 연구 대비 QG-AFP의 novelty 포인트

### 7.1 ASFF 대비

ASFF는 spatially adaptive fusion입니다. QG-AFP는 **spatially adaptive + query-conditioned** fusion입니다. 이 차이는 단순 attention 추가가 아니라, detection hypothesis가 fusion의 독립 변수로 들어간다는 점입니다.

### 7.2 AdaFPN/FaPN 대비

AdaFPN과 FaPN은 feature alignment 및 adaptive upsampling/fusion을 다룹니다. QG-AFP는 alignment의 기준을 feature map pair에서 object query로 확장합니다.

### 7.3 AFPN 대비

AFPN은 non-adjacent level interaction과 asymptotic fusion을 다룹니다. QG-AFP는 non-adjacent interaction을 query별로 선택합니다. 따라서 effective fusion path가 객체별로 달라집니다.

### 7.4 Deformable DETR 대비

Deformable DETR은 query가 multi-scale feature에서 sparse point를 읽습니다. QG-AFP는 query가 읽은 정보를 다시 pyramid feature에 write-back하고, 이후 dense/sparse head가 그 feature를 사용하게 만듭니다. 즉, **query-to-feature feedback**이 핵심 차이입니다.

### 7.5 QueryDet 대비

QueryDet은 sparse high-resolution detection을 통해 small object detection의 비용 문제를 해결합니다. QG-AFP는 sparse high-resolution activation을 포함하면서도, scale routing, feature alignment, feature fusion, head alignment까지 일반화합니다.

---

## 8. 구현 설계안

### 8.1 Minimal version: Query-guided scale gate만 추가

가장 작은 실험은 기존 neck 뒤에 query-scale gate만 붙이는 것입니다.

```text
P3, P4, P5 = baseline_neck(C3, C4, C5)
Q = top_k_query(P3, P4, P5)
G = scale_router(Q, P3, P4, P5)
P'_l = P_l · (1 + aggregate_query_gate(G_l, Q))
pred = head(P')
```

장점은 구현이 단순하고, latency overhead가 작으며, ablation이 명확하다는 점입니다.

### 8.2 Strong version: query-guided read-write fusion

더 강한 버전은 query가 feature를 읽고 다시 feature map에 write-back합니다.

```text
P = base_pyramid(C)
Q, R = query_generator(P)

for t in range(T):
    G = scale_router(Q, R, P)
    Z = deformable_read(Q, R, P, G)
    Q = query_update(Q, Z)
    M = query_spatial_mask(Q, R, G)
    P = sparse_write_back(P, Q, M)
    P = cross_level_fusion(P, M)

pred = detection_head(P, Q)
```

### 8.3 YOLO/FCOS 계열 삽입 위치

YOLOv8/YOLOv11/FCOS/RetinaNet에서는 다음 순서가 좋습니다.

1. 기존 PAFPN/BiFPN 출력 `P3, P4, P5`를 얻습니다.
2. 각 level에서 objectness 또는 cls prior로 top-k seed를 뽑습니다.
3. Seed query로 `P3/P4/P5`에 level gate와 sparse mask를 생성합니다.
4. Enhanced pyramid `P'_3/P'_4/P'_5`를 decoupled head에 넣습니다.

실험은 다음 순서로 진행합니다.

- Baseline PAFPN
- PAFPN + ASFF-style spatial fusion
- PAFPN + query-scale gate
- PAFPN + query-scale gate + sparse write-back
- PAFPN + query-scale gate + sparse write-back + query-guided alignment

### 8.4 Faster/Cascade R-CNN 계열 삽입 위치

2-stage detector에서는 RPN proposal을 query로 쓰는 것이 직관적입니다.

1. Backbone + FPN으로 기본 pyramid를 만듭니다.
2. RPN proposal을 query로 변환합니다.
3. Query가 RoIAlign 전에 pyramid를 재가중합니다.
4. RoIAlign은 `P'`에서 수행합니다.
5. Box head와 mask head는 기존처럼 둡니다.

이 버전은 RoI 단위의 object hypothesis가 명확하므로 query-scale supervision을 적용하기 쉽습니다.

### 8.5 DETR/RT-DETR/DINO 계열 삽입 위치

Transformer detector에서는 encoder proposal 또는 decoder reference point를 사용할 수 있습니다.

- Encoder proposal 기반: decoder 전에 QG-AFP를 한 번 수행합니다.
- Decoder iterative 기반: 각 decoder layer 사이에 query-to-pyramid update를 얕게 넣습니다.
- Lightweight 기반: RT-DETR류에서는 latency를 고려해 top-k query만 사용하고, write-back radius를 제한합니다.

---

## 9. 실험 설계와 ablation

### 9.1 비교 baseline

QG-AFP 논문/실험을 설계한다면 다음 baseline과 비교해야 합니다.

| 범주 | 비교 대상 | 비교 이유 |
|---|---|---|
| 기본 pyramid | FPN, PAFPN | 가장 강한 표준 baseline |
| weighted fusion | BiFPN | 효율적 learnable weight fusion 대비 |
| spatial adaptive fusion | ASFF | query-free spatial fusion 대비 |
| adaptive upsampling/alignment | CARAFE, AdaFPN, FaPN | query-free adaptive operator 대비 |
| advanced pyramid topology | NAS-FPN, AFPN | topology/fusion path 개선 대비 |
| query/sparse detector | QueryDet, Deformable DETR | query-guided sparse computation 대비 |
| dynamic head | Dynamic Head | head attention 강화 대비 |

### 9.2 필수 ablation

| Ablation | 목적 | 기대 관찰 |
|---|---|---|
| + Query-Scale Router | query별 level 선택 효과 | AP_s/AP_m/AP_l 변화, gate entropy |
| + Query Spatial Mask | foreground 중심 sparse update 효과 | latency 감소, background false positive 변화 |
| + Query-guided Alignment | misalignment 보정 효과 | AP75, boundary/localization 개선 |
| + Write-back | query-to-feature feedback 효과 | dense head AP 및 query head AP 동시 변화 |
| query 수 N | sparse budget 영향 | AP-latency Pareto curve |
| sampling point K | deformable read 비용/성능 | AP_s vs FLOPs |
| gate supervision 유무 | scale routing 안정성 | early convergence, gate collapse 여부 |
| residual identity init | 학습 안정성 | 초반 loss spike 감소 |

### 9.3 평가 지표

성능 지표는 단순 AP만으로는 부족합니다.

- COCO-style AP, AP50, AP75
- AP_s, AP_m, AP_l
- FPS, latency, FLOPs, GPU memory
- high-resolution activation ratio
- query foreground recall: top-k query가 GT를 얼마나 cover하는지
- gate entropy: 모든 level을 무차별적으로 쓰는지, 특정 level로 collapse하는지
- mask sparsity: foreground 집중 정도
- dense scene benchmark: CrowdHuman, VisDrone, DOTA/DIOR 등 task에 맞는 데이터셋

---

## 10. 예상 장점과 실패 가능성

### 10.1 예상 장점

- 작은 객체와 밀집 객체에서 high-resolution feature를 더 선택적으로 쓸 수 있습니다.
- Query가 객체별로 scale routing을 하므로 fixed level assignment보다 유연합니다.
- Deformable attention류의 sparse sampling과 FPN류의 dense feature map 장점을 결합할 수 있습니다.
- Neck과 head를 분리하지 않고, head query가 neck feature 형성에 관여하게 할 수 있습니다.
- Query mask를 통해 latency-aware detector로 확장하기 쉽습니다.

### 10.2 실패 가능성

- Query generator의 recall이 낮으면 high-resolution refinement가 필요한 객체를 놓칠 수 있습니다.
- Training 초기에 query가 불안정하면 pyramid update가 noise를 증폭합니다.
- Query mask가 지나치게 sparse하면 recall이 떨어지고, 지나치게 dense하면 계산 절감이 사라집니다.
- Multi-query가 같은 객체에 몰리면 feature write-back이 중복되어 feature conflict가 생길 수 있습니다.
- DETR류와 결합할 때 decoder query와 neck query의 역할이 중복될 수 있습니다.

### 10.3 완화 전략

- Early stage에서는 dense fallback path를 유지합니다.
- Query foreground recall을 보조 loss로 모니터링합니다.
- Query mask는 hard top-k보다 soft mask에서 시작하고 점진적으로 sparse화합니다.
- Write-back은 residual identity initialization으로 시작합니다.
- Query diversity loss 또는 NMS-free matching loss를 사용해 query collapse를 줄입니다.

---

## 11. 논문화 관점의 주장 구조

QG-AFP를 논문화할 때 주장의 중심은 다음처럼 잡는 것이 적절합니다.

### 11.1 Motivation claim

기존 FPN 변형은 feature-level, spatial-level, topology-level adaptivity를 도입했지만, detection head의 object query/hypothesis가 pyramid fusion을 직접 제어하지 못한다. 그 결과 neck과 head 사이에 semantic mismatch가 남고, 작은 객체를 위한 high-resolution computation이 background 영역에 낭비된다.

### 11.2 Method claim

QG-AFP는 object/proposal/dense queries를 pyramid fusion controller로 사용해, query-specific scale routing, query-guided sparse high-resolution update, query-conditioned alignment, query-to-feature write-back을 수행한다.

### 11.3 Empirical claim

검증해야 할 핵심은 다음입니다.

- Baseline neck 대비 AP_s 또는 AP75 개선
- BiFPN/ASFF/AFPN 대비 AP-latency Pareto 우위
- Query sparse mask로 high-resolution computation 절감
- Gate visualization에서 small/large/dense object별 level routing 차이 확인
- Query 없이 feature-only gate로 바꾸면 성능이 떨어지는 ablation

### 11.4 가장 강한 novelty sentence

> Unlike previous adaptive pyramid methods that infer fusion weights only from feature maps, QG-AFP conditions pyramid construction on object queries, enabling object-aware scale routing and query-to-feature feedback before final detection.

한국어로는 다음과 같이 쓸 수 있습니다.

> 기존 adaptive FPN은 feature map 자체에서 fusion weight를 추정하는 데 머물렀지만, QG-AFP는 detection query를 pyramid construction의 조건 변수로 사용하여 객체별 scale routing과 query-to-feature feedback을 수행한다.

---

## 12. 결론

QG-AFP는 FPN류 연구와 query-based detector 연구 사이의 미해결 접점을 공략하는 방법론입니다. FPN, PANet, BiFPN, ASFF, AdaFPN, FaPN, AFPN은 multi-scale feature fusion의 구조, weight, alignment, topology 문제를 단계적으로 해결했습니다. DETR, Deformable DETR, Sparse R-CNN, QueryDet은 query 또는 sparse hypothesis를 detection에 도입했습니다. 그러나 이 두 흐름은 대체로 분리되어 있었고, query가 feature pyramid neck 자체를 객체별로 재구성하는 방향은 상대적으로 덜 탐구되었습니다.

따라서 QG-AFP의 핵심 기여는 **object query를 feature pyramid의 adaptive controller로 승격시키는 것**입니다. 이는 작은 객체, 밀집 장면, occlusion, high-resolution computation budget이 중요한 task에서 성능 향상 가능성이 큽니다. 논문 기획상으로는 ASFF/AdaFPN/FaPN/AFPN과의 차이를 “feature-driven adaptivity vs query-conditioned adaptivity”로 명확히 제시하고, QueryDet/Deformable DETR과의 차이를 “query read-only vs query-guided pyramid write-back”으로 제시하는 것이 가장 설득력 있습니다.

---

## References

[1] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. **Feature Pyramid Networks for Object Detection.** CVPR 2017. https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html

[2] Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, Jiaya Jia. **Path Aggregation Network for Instance Segmentation.** CVPR 2018. https://arxiv.org/abs/1803.01534

[3] Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le. **NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.** CVPR 2019. https://arxiv.org/abs/1904.07392

[4] Jiangmiao Pang, Kai Chen, Jianping Shi, Huajun Feng, Wanli Ouyang, Dahua Lin. **Libra R-CNN: Towards Balanced Learning for Object Detection.** CVPR 2019. https://arxiv.org/abs/1904.02701

[5] Mingxing Tan, Ruoming Pang, Quoc V. Le. **EfficientDet: Scalable and Efficient Object Detection.** CVPR 2020. https://arxiv.org/abs/1911.09070

[6] Songtao Liu, Di Huang, Yunhong Wang. **Learning Spatial Fusion for Single-Shot Object Detection.** arXiv 2019. https://arxiv.org/abs/1911.09516

[7] Jiaqi Wang, Kai Chen, Rui Xu, Ziwei Liu, Chen Change Loy, Dahua Lin. **CARAFE: Content-Aware ReAssembly of FEatures.** ICCV 2019. https://arxiv.org/abs/1905.02188

[8] Chaoxu Guo, Bin Fan, Qian Zhang, Shiming Xiang, Chunhong Pan. **AugFPN: Improving Multi-scale Feature Learning for Object Detection.** CVPR 2020. https://arxiv.org/abs/1912.05384

[9] Chengyang Wang, Caiming Zhong. **Adaptive Feature Pyramid Networks for Object Detection.** IEEE Access 2021. https://www.researchgate.net/publication/353470279_Adaptive_Feature_Pyramid_Networks_for_Object_Detection

[10] Shihua Huang, Zhichao Lu, Ran Cheng, Cheng He. **FaPN: Feature-aligned Pyramid Network for Dense Image Prediction.** ICCV 2021. https://arxiv.org/abs/2108.07058

[11] Guoyu Yang, Jie Lei, Zhikuan Zhu, Siyu Cheng, Zunlei Feng, Ronghua Liang. **AFPN: Asymptotic Feature Pyramid Network for Object Detection.** arXiv 2023. https://arxiv.org/abs/2306.15988

[12] Chenhongyi Yang, Zehao Huang, Naiyan Wang. **QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection.** CVPR 2022. https://arxiv.org/abs/2103.09136

[13] Xiyang Dai, Yinpeng Chen, Bin Xiao, Dongdong Chen, Mengchen Liu, Lu Yuan, Lei Zhang. **Dynamic Head: Unifying Object Detection Heads with Attentions.** CVPR 2021. https://arxiv.org/abs/2106.08322

[14] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko. **End-to-End Object Detection with Transformers.** ECCV 2020. https://arxiv.org/abs/2005.12872

[15] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai. **Deformable DETR: Deformable Transformers for End-to-End Object Detection.** ICLR 2021. https://openreview.net/forum?id=gZ9hCDWe6ke

[16] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, Ping Luo. **Sparse R-CNN: End-to-End Object Detection with Learnable Proposals.** CVPR 2021. https://arxiv.org/abs/2011.12450

[17] J. Li et al. **FastQAFPN-YOLOv8s-Based Method for Rapid and Lightweight Detection of Walnut Unseparated Material.** Journal of Imaging 2024. https://www.mdpi.com/2313-433X/10/12/309
