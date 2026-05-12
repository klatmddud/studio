# Background Confuser Prototype Calibration (BCPC)

## 1. Goal

BCPC reduces high-confidence background false positives in object detection.

The core idea is simple:

> Treat recurring high-confidence background false positives as class-conditioned confuser patterns, store them in prototype memory, and use their similarity to calibrate detection scores.

This method intentionally removes the broader components from `main.md`:

- no unknown-object branch
- no new localization-quality estimator
- no negative-aware assignment
- no top-K ranking loss in the core method
- no contrastive loss in the first implementation

The primary novelty is the class-conditioned background confuser prototype memory.

## 2. Problem Setup

Given a detector output candidate `i`:

- `P[i, c]`: class probability or class score for class `c`
- `B[i]`: predicted box
- `F[i]`: candidate feature used for classification
- `G[j]`: ground-truth boxes

The baseline detector may be FCOS, GFL, VFNet, TOOD, RTMDet, YOLO-style, or DETR-style. BCPC should be implemented as a lightweight calibration module on top of the baseline detector.

If the baseline already has a quality-aware score, use it as `S_base[i, c]`. Otherwise, use `P[i, c]`.

## 3. Hard Background Mining

A candidate is a hard background false positive when:

```text
max_gt_iou(B[i], G) < tau_bg
and
max_class_score(P[i]) > tau_cls
```

Let:

```text
c_hat = argmax_c P[i, c]
```

Then candidate `i` is treated as a hard background sample for class `c_hat`.

Recommended starting thresholds:

```text
tau_bg  = 0.1 or 0.2
tau_cls = 0.3 to 0.5
```

Use conservative thresholds early. Memory contamination is the main failure mode.

## 4. Class-Conditioned Background Prototype Memory

Maintain one background prototype bank per class:

```text
M_bg[c] = {p[c, 1], p[c, 2], ..., p[c, K]}
```

Each prototype is an L2-normalized vector representing background regions that the detector tends to misclassify as class `c`.

Recommended starting config:

```text
K per class = 16
prototype dim = 128 or 256
EMA momentum eta = 0.95
```

For COCO-style 80 classes, `80 * 16 * 256` floats is small.

## 5. Feature Choice

Preferred feature for prototype memory:

```text
classification tower feature before the final class prediction layer
```

Fallback options:

- shared detection feature
- query feature for DETR-style detectors
- RoI/candidate pooled feature if the detector has region features

Always project the feature to a fixed dimension before memory update:

```text
z_i = normalize(proj(F[i]))
```

## 6. Memory Update

For each hard background sample `(z_i, c_hat)`, update only `M_bg[c_hat]`.

Find nearest prototype:

```text
k_star = argmax_k cosine(z_i, p[c_hat, k])
```

Update by EMA:

```text
p[c_hat, k_star] = eta * p[c_hat, k_star] + (1 - eta) * z_i
p[c_hat, k_star] = normalize(p[c_hat, k_star])
```

If all similarities are below a novelty threshold, replace the least recently updated prototype or use a FIFO slot.

Minimal first implementation can skip novelty replacement and always update nearest prototype.

## 7. Background Similarity

For candidate `i` and class `c`:

```text
A_bg[i, c] = max_k cosine(z_i, p[c, k])
```

This is the candidate's similarity to class `c` background confusers.

Efficient implementation:

```text
Z:     [N, D]
M_bg:  [C, K, D]
A_bg:  [N, C] = max over K of matmul(Z, M_bg[c].T)
```

## 8. Background Confuser Risk

Predict class-conditioned background risk:

```text
R[i, c] = sigmoid(h_r([z_i, A_bg[i, c], P[i, c]]))
```

Minimal risk head:

```text
input:  concat(projected feature, bg similarity scalar, class score scalar)
module: 2-layer MLP
output: scalar risk in [0, 1]
```

For dense detectors, a lighter implementation is:

```text
R[i, c] = sigmoid(a * A_bg[i, c] + b * P[i, c] + bias)
```

Start with the MLP if code structure allows it.

## 9. Risk Supervision

Train `R[i, c]` with positives and hard background samples.

Targets:

```text
R[i, y_i] = 0  for matched positive object candidate i
R[i, c_hat] = 1  for hard background candidate i predicted as c_hat
```

Loss:

```text
L_bg = BCE(R[i, c], y_bg[i, c])
```

Use high-confidence weighting for hard background samples:

```text
w_i = P[i, c_hat]^rho
```

Recommended:

```text
rho = 1.0 or 2.0
lambda_bg = 0.5
```

Weighted loss:

```text
L_bg = mean(w_i * BCE(R[i, c], y_bg[i, c]))
```

Positive samples can use weight `1.0`.

## 10. Score Calibration

BCPC calibrates the baseline score using background risk:

```text
S[i, c] = S_base[i, c] * (1 - R[i, c])^gamma
```

If the detector has no separate baseline score:

```text
S_base[i, c] = P[i, c]
```

If the detector already uses a quality-aware score:

```text
S_base[i, c] = detector_native_score[i, c]
```

Recommended starting value:

```text
gamma = 0.5
```

Tune:

- increase `gamma` when background FP reduction matters more
- decrease `gamma` when recall drops

Avoid hard rejection in the first implementation. Use score calibration only.

## 11. Training Schedule

### Stage 0: Baseline Warm-Up

Train the baseline detector normally, or load a pretrained detector.

During warm-up:

```text
memory update: off
L_bg: off
score calibration: off
```

### Stage 1: Memory Bootstrap

Run detector training or inference on training batches and collect hard background samples.

During bootstrap:

```text
memory update: on
L_bg: optional
score calibration: off
```

Keep thresholds conservative.

### Stage 2: Risk Head Training

Train background risk head with:

```text
L = L_det + lambda_bg * L_bg
```

During this stage:

```text
memory update: on, conservative
score calibration during training: optional
score calibration during validation: on
```

### Stage 3: Joint Fine-Tuning

Fine-tune detector and risk head together:

```text
L = L_det + lambda_bg * L_bg
```

Use calibrated score for validation and final inference.

If joint fine-tuning destabilizes the detector, freeze the baseline detector and train only projection + risk head.

## 12. Inference

Inference procedure:

```text
P, B, F = detector(x)
Z = normalize(proj(F))
A_bg = prototype_similarity(Z, M_bg)
R = risk_head(Z, A_bg, P)

for each candidate i and class c:
    S[i, c] = S_base[i, c] * (1 - R[i, c])^gamma

run thresholding and NMS using S
```

The detector boxes are unchanged. Only class scores are calibrated.

## 13. Minimal Pseudocode

```python
def mine_hard_background(boxes, scores, gt_boxes, tau_bg=0.2, tau_cls=0.4):
    max_iou = box_iou_max(boxes, gt_boxes)
    max_score, cls_hat = scores.max(dim=1)
    mask = (max_iou < tau_bg) & (max_score > tau_cls)
    return mask, cls_hat


@torch.no_grad()
def update_bg_memory(memory, z, cls_hat, mask, momentum=0.95):
    z = F.normalize(z, dim=-1)
    for feat, c in zip(z[mask], cls_hat[mask]):
        proto = memory[c]
        sim = proto @ feat
        k = sim.argmax()
        proto[k] = momentum * proto[k] + (1.0 - momentum) * feat
        proto[k] = F.normalize(proto[k], dim=0)
    return memory


def background_similarity(z, memory):
    # z: [N, D], memory: [C, K, D]
    z = F.normalize(z, dim=-1)
    memory = F.normalize(memory, dim=-1)
    sim = torch.einsum("nd,ckd->nck", z, memory)
    return sim.max(dim=-1).values


def calibrate_score(base_score, risk, gamma=0.5, eps=1e-6):
    risk = risk.clamp(eps, 1.0 - eps)
    return base_score * ((1.0 - risk) ** gamma)
```

## 14. Ablation Plan

Required ablations:

```text
1. Baseline detector
2. Baseline + hard-negative BCE only, no memory
3. Baseline + risk head without prototype memory
4. Baseline + global background memory, not class-conditioned
5. Baseline + class-conditioned prototype memory
6. Full BCPC score calibration
```

Optional ablations:

```text
K per class: 4, 8, 16, 32
gamma: 0.25, 0.5, 1.0
tau_bg: 0.1, 0.2, 0.3
tau_cls: 0.3, 0.5, 0.7
feature source: shared tower vs classification tower
frozen detector vs joint fine-tuning
```

## 15. Metrics

Report standard detection metrics:

```text
AP, AP50, AP75
APS, APM, APL
```

Report false-positive-focused metrics:

```text
FP per image
background-only false alarm rate
high-confidence FP count
pre-NMS top-K FP ratio
NMS survivor FP ratio
class-wise FP reduction
```

The main expected gain is not necessarily large AP improvement. The primary claim is reducing high-confidence background false positives while preserving recall.

## 16. Expected Failure Modes

### Memory Contamination

Unlabeled true objects can be mined as hard background.

Mitigation:

```text
use conservative tau_bg and tau_cls
exclude candidates near GT boxes
warm up detector before mining
update memory only with stable high-confidence unmatched predictions
```

### Recall Drop

If `R` is too high for true objects, calibrated scores suppress true positives.

Mitigation:

```text
lower gamma
increase positive samples in L_bg
train risk head with strong positive supervision
avoid hard rejection
```

### Class Imbalance

Some classes may have many more hard backgrounds.

Mitigation:

```text
limit memory updates per class per batch
balance L_bg by class
use fixed K per class
```

## 17. Paper Claim

Recommended core claim:

> High-confidence background false positives are not random negatives; they form recurring class-conditioned confuser patterns. BCPC stores these patterns as prototype memory and uses prototype similarity to calibrate detection scores.

Recommended contributions:

```text
1. We identify high-confidence background false positives as recurring class-conditioned confuser patterns.
2. We introduce a class-conditioned background confuser prototype memory for object detectors.
3. We calibrate detection scores using prototype-based background risk, reducing top-ranked background false positives with minimal changes to the detector.
```

Recommended title:

```text
Class-Conditioned Background Confuser Prototypes for Calibrated Object Detection
```

