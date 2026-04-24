# TCS - Transition Candidate Selection

## Goal

TCS selects which current training GT instances should receive TAL supervision.

HTM discovers historical `FN -> TP` transitions. TDB stores their directions. TCS decides which
GTs in the current mini-batch are hard enough and memory-covered enough to use those directions.

```text
current GT g
  -> check current difficulty
  -> check TDB direction availability
  -> select or skip for TAL
```

TCS prevents FN-TDM from applying transition alignment to every object indiscriminately.

## Position in FN-TDM

```text
HTM: mine past FN -> TP transitions
TDB: store and aggregate transition directions
TCS: select current hard/FN-like GT candidates
TAL: align selected candidates using TDB directions
```

TCS should be lightweight and training-only. It must not change inference behavior.

## Why TCS Is Needed

Applying TAL to every GT can be harmful.

Potential issues:

- Easy GTs may receive unnecessary feature-direction constraints.
- A class-level transition direction may not fit every instance in the class.
- TAL can conflict with the base detection loss if applied too broadly.
- Memory noise can be amplified if every GT consumes TDB directions.

TCS narrows TAL to GTs that are likely to benefit:

```text
hard current GT + available transition direction -> TAL candidate
```

## V0 Design

TCS-V0 uses two gates:

```text
1. Memory coverage gate:
   TDB has a valid direction prototype for class_id(g).

2. Current hardness gate:
   GT g is currently hard-like according to training-time detection signals.
```

Selection rule:

```text
select g if:
    TDB.get_prototype(class_id(g)) is not None
    and hardness(g) >= tau_hard
```

Recommended default:

```text
tau_hard: 0.5
```

For V0, define hardness from classification confidence:

```text
hardness(g) = 1 - confidence(g)
```

Then:

```text
select if confidence(g) <= 1 - tau_hard
```

With `tau_hard = 0.5`, this means selecting GTs with confidence <= 0.5.

## Candidate Object

TCS outputs candidate objects consumed by TAL.

```python
TCSCandidate = {
    "gt_uid": str | None,
    "image_id": Any,
    "ann_id": Any,
    "batch_index": int,
    "target_index": int,
    "class_id": int,
    "bbox": Tensor[4],

    "hardness": float,
    "confidence": float,
    "selection_reason": str,

    "direction": Tensor[D],
    "direction_source": "class_prototype" | "failure_type_prototype" | "sampled_entry",
    "fn_type_hint": str | None,
}
```

`direction` should be detached and moved to the same device TAL will use.

## Current Difficulty Signals

TCS can use different difficulty signals depending on the detector.

Detector-agnostic signals:

```text
classification confidence for class_id(g)
objectness or foreground confidence
localization quality / IoU estimate
assigned positive count
current per-GT classification loss
current per-GT box loss
historical GT state if available
```

For V0, prioritize signals already available in the normal training forward. Avoid an extra
inference pass inside TCS.

## FCOS V0 Hardness

For FCOS, each GT is assigned to one or more positive feature locations.

Recommended confidence:

```text
confidence(g) =
    max over positive locations assigned to g:
        sigmoid(cls_logit[class_id(g)]) * sigmoid(centerness_logit)
```

If centerness is not available at the selection point:

```text
confidence(g) =
    max sigmoid(cls_logit[class_id(g)]) over positive locations assigned to g
```

Hardness:

```text
hardness(g) = 1 - confidence(g)
```

If no positive location is assigned:

```text
confidence(g) = 0
hardness(g) = 1
selection_reason = "missing_assignment"
```

Missing assignment candidates are high risk. V0 may either include them or log them only depending on
whether TAL can extract a reliable GT ROI feature.

Recommended default:

```text
include_missing_assignment: true
```

because TAL can use GT ROIAlign features even without assigned FCOS positives.

## Optional Historical Gate

TCS can optionally use HTM/TFM history.

```text
select if:
    current hard-like
    or gt_uid has previous FN history
    or gt_uid has relapse history
```

V0 should not require history because TDB is already historical and class-wise. History-aware
selection can be introduced as an ablation.

Suggested variants:

```text
TCS-CurrentHard:
    select by current confidence only

TCS-History:
    select GTs with previous FN/relapse history

TCS-Hybrid:
    select if current hard OR previous FN/relapse history
```

Recommended V0 default:

```text
selection_mode: current_hard
```

## Memory Coverage Gate

TCS must not select a GT if TDB cannot provide a valid direction.

```python
direction = tdb.get_prototype(class_id=class_id, fn_type=fn_type_hint, device=device)
if direction is None:
    skip
```

For V0:

```text
use_failure_type_query: false
```

TCS queries class-wise prototype:

```text
D_c = TDB.get_prototype(class_id=c)
```

Failure-type query can be enabled after HTM subtype quality is validated.

## Candidate Budget

TAL should not dominate training. TCS should limit candidates per batch.

Recommended defaults:

```text
max_candidates_per_image: 8
max_candidates_per_batch: 32
min_candidates_per_batch: 0
```

If more candidates are available, rank by:

```text
hardness * direction_quality
```

For V0, if TDB returns only a prototype without quality, rank by hardness.

Tie-breaker:

```text
higher hardness first
then smaller area first
then stable gt_uid order
```

Small objects may benefit from this tie-breaker because they are often missed, but avoid making size
the main selection criterion in V0.

## Selection Score

Internal selection score:

```text
selection_score(g) = hardness(g) * memory_score(class_id(g))
```

V0 memory score:

```text
memory_score(c) = 1 if TDB has valid prototype for c else 0
```

Optional V1:

```text
memory_score(c) = mean quality or max quality of TDB[c]
```

## Candidate Weight

TCS can provide a per-candidate weight for TAL.

V0:

```text
candidate_weight = clamp(hardness, min_weight, max_weight)
```

Recommended defaults:

```text
min_weight: 0.25
max_weight: 1.0
```

TAL may multiply each candidate loss by this weight.

If this complicates TAL V0, set all selected candidate weights to 1.0 and use hardness only for
selection/ranking.

## Failure-Type Hint

TCS may produce a `fn_type_hint` for failure-type TDB query.

Possible hints:

```text
low class confidence -> FN_BG or FN_CLS
good class confidence but poor localization -> FN_LOC
no assignment -> FN_MISS
```

For V0, keep this as metadata only:

```text
fn_type_hint: None
```

The first version should query class-wise TDB prototypes.

## Interaction With TDB Retrieval Variants

TCS should be independent of the TDB retrieval variant.

```text
TDB-Last
TDB-TopK
TDB-TopK+Age
```

TCS only asks for a direction:

```text
direction = TDB.get_prototype(...)
```

The TDB config decides how that direction is built. This keeps candidate selection ablations separate
from memory retrieval ablations.

## Configuration Sketch

Future config under `modules/cfg/fntdm.yaml`:

```yaml
tcs:
  enabled: true
  selection_mode: current_hard

  memory_gate:
    require_tdb_direction: true
    use_failure_type_query: false

  hardness:
    source: assigned_cls_confidence
    tau_hard: 0.5
    include_centerness: true
    include_missing_assignment: true

  budget:
    max_candidates_per_image: 8
    max_candidates_per_batch: 32
    rank_by: hardness

  weighting:
    use_candidate_weight: false
    min_weight: 0.25
    max_weight: 1.0

  logging:
    save_summary: true
    save_candidate_stats: true
```

## Logging

Epoch or iteration summary:

```text
num_gt_seen
num_memory_covered_gt
num_hard_gt
num_selected_candidates
num_skipped_no_memory
num_skipped_easy
mean_candidate_hardness
max_candidate_hardness
```

Optional per-class summary:

```text
class_id
num_gt_seen
num_memory_covered_gt
num_selected
mean_hardness
```

Do not log every candidate every iteration unless debugging. Candidate-level logs can become large.

## Edge Cases

- No TDB entries for a class: skip GT.
- No selected candidates in a batch: TAL returns zero loss.
- All GTs are easy: TAL returns zero loss.
- Candidate budget exceeded: keep highest-ranked candidates.
- Missing assignment but valid GT ROI feature exists: candidate can be selected if configured.
- Direction query returns invalid/None: skip GT.

## V0 Implementation Checklist

1. Add `TransitionCandidateSelector` class.
2. Define `TCSCandidate` structure.
3. Implement FCOS per-GT confidence extraction from assigned positive locations.
4. Implement memory coverage gate using `TDB.get_prototype`.
5. Implement hardness thresholding.
6. Implement per-image and per-batch budgets.
7. Return selected candidates with direction tensors.
8. Add summary counters.
9. Wire TCS output into TAL.

## Minimal Unit Tests

Selection:

```text
easy GT is skipped
hard GT with TDB direction is selected
hard GT without TDB direction is skipped
missing assignment follows include_missing_assignment config
```

Budget:

```text
max_candidates_per_image is enforced
max_candidates_per_batch is enforced
highest hardness candidates are kept
```

Device:

```text
direction is moved to requested device
direction is detached
```

No-op:

```text
empty batch returns empty candidates
no selected candidates makes TAL skip safely
```

## Research Claim

TCS should be described as:

```text
Transition Candidate Selection restricts direction alignment to current hard
ground-truth instances whose classes have reliable transition memories, preventing
the auxiliary objective from over-regularizing easy objects.
```

TCS keeps FN-TDM targeted: the method uses historical recovery directions only where the current
detector still shows signs of difficulty.
