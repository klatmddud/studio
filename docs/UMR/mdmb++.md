# MDMB++ - Structured Failure Memory

MDMB++ extends the original missed-detection memory from a simple missed-GT bank into structured
failure memory. It records why each GT failed, how severe the failure is, whether the GT relapsed,
and what support state was last available when the GT was detected.

## Core State

### `CanonicalCandidate`

Detector-specific candidates are normalized into a common schema:

```python
@dataclass(slots=True)
class CanonicalCandidate:
    stage: str
    box: Tensor
    score: float
    label: int
    iou_to_gt: float
    survived_selection: bool
    survived_nms: bool | None
    rank: int | None
    level_or_stage_id: str | int | None
```

### `SupportSnapshot`

The last successful detection state for a GT:

```python
@dataclass(slots=True)
class SupportSnapshot:
    epoch: int
    box: Tensor
    score: float
    feature: Tensor | None
    feature_level: str | int | None
    iou_to_gt: float
    quality: float
    feature_epoch: int | None
```

`feature` is optional and is stored only when `store_support_feature: true`.
`feature_epoch` records when that feature was created, which can differ from `epoch` if metadata is
updated while the previous feature teacher is retained.

## Relapse-Robust Support Memory

MDMB++ no longer treats the newest detection as automatically better. When support memory is
enabled, a new detected support replaces the previous teacher only if:

- no previous support exists
- the previous support lacks a feature and the new support has one
- the previous feature is older than `support_memory.refresh_age`
- the new support quality clears `support_memory.replace_margin`

Support quality is:

```text
quality = score_weight * score + iou_weight * iou_to_gt
```

Config:

```yaml
support_memory:
  enabled: true
  score_weight: 1.0
  iou_weight: 1.0
  replace_margin: 0.05
  refresh_age: 15
  require_feature: true
```

With `require_feature: true`, MDMB++ will not replace an existing feature teacher with a detected
support that failed to produce a new feature. This prevents a low-information support update from
silently resetting RASD's teacher age.

### `GTFailureRecord`

Persistent per-GT history:

```python
@dataclass(slots=True)
class GTFailureRecord:
    gt_uid: str
    image_id: str
    class_id: int
    bbox: Tensor
    first_seen_epoch: int
    last_seen_epoch: int
    last_state: FailureType
    consecutive_miss_count: int
    max_consecutive_miss_count: int
    total_miss_count: int
    relapse_count: int
    last_detected_epoch: int | None
    last_failure_epoch: int | None
    last_failure_type: FailureType | None
    severity: float
    support: SupportSnapshot | None
```

### `MDMBPlusEntry`

Current unresolved failure entry:

```python
@dataclass(slots=True)
class MDMBPlusEntry:
    gt_uid: str
    image_id: str
    class_id: int
    bbox: Tensor
    failure_type: FailureType
    consecutive_miss_count: int
    max_consecutive_miss_count: int
    total_miss_count: int
    relapse: bool
    severity: float
    best_candidate: CanonicalCandidate | None
    topk_candidates: list[CanonicalCandidate]
    support: SupportSnapshot | None
```

## Failure Types

```python
FailureType = Literal[
    "candidate_missing",
    "loc_near_miss",
    "cls_confusion",
    "score_suppression",
    "nms_suppression",
    "detected",
]
```

The taxonomy separates missing candidates, localization misses, class confusion, score suppression,
NMS suppression, and recovered detections. Downstream modules can use these labels without reading
raw detector internals.

## Public API

```python
class MDMBPlus(nn.Module):
    def update(...): ...
    def get_image_entries(image_id) -> list[MDMBPlusEntry]: ...
    def get_record(gt_uid) -> GTFailureRecord | None: ...
    def get_replay_priority(image_id) -> float: ...
    def get_dense_targets(image_id) -> list[MDMBPlusEntry]: ...
    def summary() -> dict[str, Any]: ...
```

`get_replay_priority()` is used by Hard Replay. `get_image_entries()` and `get_record()` are used by
RASD to select relapse targets and fetch support metadata.

## Summary Metrics

Important summary fields include:

- `num_entries`
- `num_images`
- `num_relapse`
- `num_candidate_missing`
- `num_cls_confusion`
- `num_score_suppression`
- `global_max_consecutive_miss`
- `mean_severity`
- `recovery_rate_last_1_epoch`

These metrics help separate raw mAP changes from the underlying failure-memory dynamics.
