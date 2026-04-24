# TDB - Transition Direction Bank

## Goal

TDB is the memory component of FN-TDM.

HTM emits `FN -> TP` transition events. TDB stores, filters, and aggregates those transition
directions so TAL can use them as stable direction priors during later training epochs.

```text
HTM event:  (class_id, fn_type, direction, quality, metadata)
TDB memory: class-wise top-K transition directions
TAL query:  representative direction for the current hard GT
```

TDB is intentionally simple in V0: use a class-wise quality-gated top-K memory and return a
weighted prototype direction per class.

## Position in FN-TDM

```text
HTM: Hard Transition Mining
  - finds FN -> TP transitions
  - emits transition events

TDB: Transition Direction Bank
  - stores transition directions
  - filters low-quality or duplicate events
  - builds class/failure-type direction prototypes

TAL: Transition Alignment Loss
  - retrieves directions from TDB
  - aligns current hard/FN-like GT embeddings
```

TDB should not perform inference and should not compute losses. It only manages transition memory.

## V0 Design

Use class-wise top-K direction memory.

```text
TDB[c] = top-K entries for class c
```

Each entry stores a normalized transition direction produced by HTM:

```text
d = normalize(z_tp - z_fn)
```

TAL queries TDB using the current GT class:

```text
D_c = TDB.get_prototype(class_id=c)
```

The returned direction is a quality-weighted mean of stored directions:

```text
D_c = normalize(sum_i w_i * d_i)
w_i = softmax(q_i / tau_proto)
```

## Responsibilities

TDB is responsible for:

- Receiving `TransitionEvent` objects from HTM.
- Validating transition entries before storage.
- Keeping per-class memory bounded.
- Ranking entries by quality.
- Optionally grouping entries by failure subtype.
- Returning direction prototypes or sampled directions for TAL.
- Saving compact metadata for analysis and checkpointing.

TDB is not responsible for:

- Assigning TP/FN states.
- Extracting features.
- Applying training losses.
- Modifying detector predictions at inference time.

## Input Event

TDB consumes the `TransitionEvent` emitted by HTM.

```python
TransitionEvent = {
    "gt_uid": str,
    "image_id": Any,
    "ann_id": Any,
    "class_id": int,
    "bbox": Tensor[4],

    "epoch_fn": int,
    "epoch_tp": int,
    "fn_type": str,

    "z_fn": Tensor[D],
    "z_tp": Tensor[D],
    "direction": Tensor[D],

    "score_fn": float,
    "score_tp": float,
    "iou_fn": float,
    "iou_tp": float,

    "quality": float,
}
```

V0 should store `direction` and compact metadata by default. Storing `z_fn` and `z_tp` is optional
and should be disabled unless a later TAL variant needs them.

## Bank Entry

Internal entry schema:

```python
TDBEntry = {
    "entry_id": str,
    "gt_uid": str,
    "image_id": Any,
    "ann_id": Any,
    "class_id": int,
    "bbox": Tensor[4] | None,
    "fn_type": str,

    "epoch_fn": int,
    "epoch_tp": int,
    "age": int,

    "direction": Tensor[D],
    "quality": float,

    "score_fn": float,
    "score_tp": float,
    "iou_fn": float,
    "iou_tp": float,

    "z_fn": Tensor[D] | None,
    "z_tp": Tensor[D] | None,
}
```

`entry_id` can be deterministic:

```text
entry_id = hash(gt_uid, epoch_fn, epoch_tp, fn_type)
```

All tensors stored in TDB should be detached. Persistent tensors should live on CPU unless the
current training step explicitly moves them to the training device for TAL.

## Memory Layout

V0 memory:

```python
bank = {
    class_id: List[TDBEntry]
}
```

Optional V1 layout:

```python
bank = {
    class_id: {
        fn_type: List[TDBEntry]
    }
}
```

For V0, keep `fn_type` in each entry even if the main lookup is class-wise. This enables later
failure-type ablations without changing the stored event format.

## Entry Validation

Before storage, TDB should reject invalid events.

Required checks:

```text
class_id is valid
fn_type is allowed
direction is finite
direction norm is close to 1
quality is finite and positive
epoch_tp > epoch_fn
```

Recommended defaults:

```text
allowed_fn_types: [FN_BG, FN_CLS, FN_MISS]
min_quality: 0.0
min_direction_norm: 1.0e-6
renormalize_direction: true
```

If `renormalize_direction` is true, TDB normalizes the direction again before storage.

## Retention Policy

The bank must be bounded.

Recommended V0:

```text
max_entries_per_class: 128
replacement_policy: quality_topk
max_entries_per_gt: 1
```

When a new entry arrives:

1. Drop it if validation fails.
2. Drop it if the same `entry_id` already exists.
3. If `max_entries_per_gt` is reached for that GT, keep only the higher-quality entry.
4. Insert into `bank[class_id]`.
5. Sort by `quality` descending.
6. Keep the first `max_entries_per_class` entries.

This policy makes TDB a high-quality direction memory rather than a full transition log.

## Duplicate Handling

Potential duplicates can occur in DDP or repeated HTM passes.

Duplicate keys:

```text
strict duplicate: entry_id
GT duplicate: (gt_uid, fn_type)
```

V0 behavior:

```text
same entry_id:
    keep higher-quality one

same gt_uid and max_entries_per_gt == 1:
    keep higher-quality one
```

Do not average duplicate entries in V0. Averaging can hide unstable transitions.

## Aging

Transition directions can become stale as the representation space changes.

V0 should store age metadata but does not need aggressive aging at first.

```text
age = current_epoch - epoch_tp
```

Optional age decay for prototype computation:

```text
q_eff = quality * exp(-age / tau_age)
```

Recommended default:

```text
use_age_decay: false
tau_age: 10
```

Enable age decay only if old directions degrade TAL stability.

## Prototype Direction

TAL usually needs one representative direction for the current GT.

Class-wise prototype:

```text
D_c = normalize(sum_i w_i * d_i)
w_i = softmax(q_i / tau_proto)
```

Recommended default:

```text
tau_proto: 0.2
min_entries_for_query: 1
```

If `tau_proto` is small, high-quality directions dominate. If it is large, the prototype becomes
closer to a uniform average.

If the weighted sum norm is too small, return `None`.

```text
min_prototype_norm: 1.0e-6
```

## Retrieval Variants

TDB should support three retrieval variants for ablation.

### TDB-Last

Use only the most recent valid transition direction for the queried class.

```text
D_c = d_last
```

Selection:

```text
d_last = entry with the largest epoch_tp among entries in TDB[c]
```

Rationale:

- Best matches the current feature space if representation drift is large.
- Very simple and useful as a recency baseline.

Expected risk:

- Noisy because a single transition may be caused by threshold crossing, NMS, or an unstable GT.
- Higher seed variance.
- Can overfit TAL to one hard instance's trajectory.

### TDB-TopK

Use the quality top-K entries for the queried class and return a quality-weighted prototype.

```text
D_c = normalize(sum_i softmax(q_i / tau_proto) * d_i)
```

Selection:

```text
entries = top-K by quality in TDB[c]
```

Rationale:

- Reduces noise from any single transition.
- Captures common recovery directions across multiple hard instances.
- Uses HTM quality to prefer confident and short-gap `FN -> TP` transitions.

Expected risk:

- Old high-quality directions can become stale if the feature space drifts.
- Opposing directions can partially cancel if the class has multiple hard modes.

This is the recommended V0 default.

### TDB-TopK+Age

Use quality top-K entries, but decay each entry's effective quality by age before computing the
prototype.

```text
age_i = current_epoch - epoch_tp_i
q_eff_i = q_i * exp(-age_i / tau_age)
D_c = normalize(sum_i softmax(q_eff_i / tau_proto) * d_i)
```

Rationale:

- Keeps the stabilizing effect of top-K aggregation.
- Reduces the influence of stale directions.
- Balances quality and recency.

Expected risk:

- Requires tuning `tau_age`.
- If `tau_age` is too small, it degenerates toward TDB-Last.
- If `tau_age` is too large, it behaves like TDB-TopK.

Recommended ablation order:

```text
TDB-Last
TDB-TopK
TDB-TopK+Age
```

Default choice:

```text
retrieval: topk
```

## Failure-Type Prototype

V1 can return failure-type conditioned prototypes:

```text
D_{c,t} = TDB.get_prototype(class_id=c, fn_type=t)
```

Fallback policy:

```text
if enough entries for (class_id, fn_type):
    return D_{c,t}
else:
    return D_c
```

Recommended default:

```text
use_failure_type_query: false
min_type_entries_for_query: 4
```

This is useful after HTM subtype labels are verified to be stable.

## Sampling API

TDB should expose both prototype and sampling APIs.

Minimal API:

```python
class TransitionDirectionBank:
    def update(self, events: list[TransitionEvent], epoch: int) -> dict:
        ...

    def get_prototype(
        self,
        class_id: int,
        fn_type: str | None = None,
        device: torch.device | None = None,
    ) -> Tensor | None:
        ...

    def sample(
        self,
        class_id: int,
        k: int = 1,
        fn_type: str | None = None,
        device: torch.device | None = None,
    ) -> list[TDBEntry]:
        ...

    def summary(self) -> dict:
        ...

    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state: dict) -> None:
        ...
```

V0 TAL should use `get_prototype`. `sample` is kept for later contrastive or multi-direction TAL
variants.

## Query Behavior

When TAL queries a class with no memory:

```text
return None
```

TAL must skip auxiliary loss for that GT.

When class memory exists but prototype is invalid:

```text
return None
```

Do not fall back to another class in V0. Cross-class directions are likely noisy unless a separate
semantic sharing module is designed.

## Device and Precision

Storage:

```text
CPU tensors by default
float32 by default
optional float16 for large datasets
```

Query:

```text
move returned direction to caller-provided device
detach returned direction
```

TDB directions are priors, not learnable tensors. Gradients should not flow into stored entries.

## Checkpointing

TDB state should be saved with training checkpoints when FN-TDM is enabled.

State should include:

```text
config
current_epoch
bank entries
prototype cache if used
summary counters
```

Recommended format:

```python
{
    "version": 1,
    "config": ...,
    "bank": ...,
    "stats": ...
}
```

If checkpoint size becomes large, store only:

```text
direction
quality
class_id
fn_type
gt_uid
epoch_fn
epoch_tp
score/iou metadata
```

Do not store `z_fn` and `z_tp` unless explicitly configured.

## Prototype Cache

Prototype computation is cheap for V0, but a cache can avoid repeated sorting and weighted sums.

Cache key:

```text
(class_id, fn_type or "ALL", bank_revision)
```

The cache must be invalidated whenever entries are inserted or removed.

For V0, a cache is optional.

## Update Timing

TDB updates after HTM finishes epoch-end mining.

```text
end of epoch e:
    events = HTM.mine(...)
    update_stats = TDB.update(events, epoch=e)
    save HTM/TDB summaries

epoch e + 1:
    TAL queries updated TDB
```

This one-epoch delay is intentional. Directions are mined from the model after epoch `e` and used
as historical priors in later training.

## DDP Behavior

Recommended V0:

```text
rank 0 runs HTM
rank 0 updates TDB
rank 0 broadcasts TDB state to all ranks before next epoch
```

This avoids duplicate entries and inconsistent memory across ranks.

If every rank mines events independently, merge by `entry_id` and keep higher-quality duplicates.

## Configuration Sketch

Future config under `modules/cfg/fntdm.yaml`:

```yaml
tdb:
  enabled: true

  storage:
    max_entries_per_class: 128
    max_entries_per_gt: 1
    store_z_fn: false
    store_z_tp: false
    store_on_cpu: true
    dtype: float32

  filtering:
    allowed_fn_types: [FN_BG, FN_CLS, FN_MISS]
    min_quality: 0.0
    min_direction_norm: 1.0e-6
    renormalize_direction: true

  replacement:
    policy: quality_topk
    duplicate_policy: keep_best

  prototype:
    retrieval: topk
    tau_proto: 0.2
    min_entries_for_query: 1
    min_prototype_norm: 1.0e-6
    use_age_decay: false
    tau_age: 10
    use_failure_type_query: false
    min_type_entries_for_query: 4

  logging:
    save_summary: true
    save_bank_metadata: true
```

## Logging

Epoch summary:

```text
epoch
num_events_received
num_events_stored
num_events_rejected
num_duplicate_events
num_replaced_entries
num_classes_with_entries
total_entries
entries_per_class_mean
entries_per_class_max
```

Per-class summary:

```text
class_id
num_entries
mean_quality
max_quality
num_fn_bg
num_fn_cls
num_fn_miss
num_fn_loc
prototype_norm
```

Do not log full direction vectors in CSV. For debugging, log only norms and metadata.

## Failure Modes

- Low-quality HTM events can pollute TDB.
- Dominant classes can receive many entries while rare classes have none.
- Old directions can become stale if representation space drifts.
- Averaging unrelated directions can cancel the prototype.
- FN subtype labels may be noisy early in training.

Mitigations:

- Quality gating.
- Per-class capacity.
- Optional age decay.
- Skip invalid low-norm prototypes.
- Warmup before HTM starts.
- Use failure-type query only after subtype quality is verified.

## V0 Implementation Checklist

1. Add `TransitionDirectionBank` class.
2. Add `TDBEntry` dataclass or typed dict.
3. Implement event validation.
4. Implement quality top-K insertion.
5. Implement duplicate handling by `entry_id` and `gt_uid`.
6. Implement class-wise prototype query.
7. Implement optional entry sampling.
8. Implement `summary`, `state_dict`, and `load_state_dict`.
9. Wire HTM event output to TDB update.
10. Broadcast or synchronize TDB state under DDP.
11. Add logging summaries.

## Minimal Unit Tests

Validation:

```text
invalid direction is rejected
disallowed fn_type is rejected
non-positive quality is rejected when min_quality > 0
```

Retention:

```text
bank keeps at most K entries per class
higher-quality duplicate replaces lower-quality duplicate
max_entries_per_gt keeps only one entry per GT
```

Prototype:

```text
prototype has unit norm
higher-quality entry receives larger softmax weight
empty class returns None
low-norm weighted sum returns None
```

State:

```text
state_dict/load_state_dict preserves entries
stored tensors are detached
query moves prototype to requested device
```

## Research Claim

TDB should be described as:

```text
Transition Direction Bank stores high-quality false-negative-to-true-positive
feature transition directions and consolidates them into class-wise direction priors
for later hard-instance alignment.
```

The novelty is not simply storing hard examples. TDB stores the direction of successful recovery
from false-negative states and makes that direction reusable for future hard GTs.
