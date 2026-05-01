# Research Modules

The current runtime-connected research-module surface includes ReMiss MissBank, LMB, and QG-AFP. Older unrelated research-module code paths have been removed.

## Config Resolution

`scripts/train.py` accepts:

- `--remiss-config`
- `--lmb-config`
- `--qg-afp-config`

`scripts/runtime/module_configs.py` resolves default module config paths, and `scripts/runtime/module_metadata.py` persists enabled module snapshots to `metadata/modules.yaml` for reproducibility.

## ReMiss MissBank (`modules/nn/mb.py`)

MissBank stores per-GT recurrent missed-detection state for the ReMiss method.

Key concepts:

- Matching: a GT is detected only when a final prediction has the same class, score above `matching.score_threshold`, and IoU above `matching.iou_threshold`. Use `auto` to resolve these thresholds from the detector's final post-processing config. FCOS maps score to `head.score_thresh` and IoU to `head.nms_thresh`.
- Mining: `mining.type: online` updates MissBank after each optimization step, while `mining.type: offline` runs a separate no-grad training-set pass after each epoch.
- Records: `MissBankRecord` stores image ID, GT ID, class, box, region ID, consecutive `miss_count`, current missed state, last match score/IoU, best same-class score/IoU diagnostics, and aggregate seen/missed counts.
- Regions: baseline `grid_size: 2` yields labels `0..4`, where `0` is none and `1..4` are row-major spatial cells. Larger `NxN` grids use labels `0..N^2`.
- Targets: `get_image_targets()` and `get_batch_labels()` emit image-level hard labels. A GT contributes only when it is currently missed and `miss_count >= target.miss_threshold`.
- State: `get_extra_state()` and `set_extra_state()` make MissBank checkpointable.
- Summary: `missbank.summary()` reports record counts, current missed counts, target GT counts, region histograms, class histograms, and update stats.
- Stability: `epoch_snapshot()` and `stability_metrics()` support epoch-to-epoch miss stability metrics.
- Config: `modules/cfg/remiss.yaml`.

## Runtime Status

When ReMiss is enabled, MissBank is attached to FCOS and updated from final post-processed detections using the configured online or offline mining mode. MissBank does not alter detector forward computation, add auxiliary losses, or inject features.

## Support Matrix

| Module | FCOS | Faster R-CNN | DINO |
|---|---:|---:|---:|
| ReMiss MissBank | memory update + stability logging | no | no |
| LMB | offline mining + stability logging | no | no |
| QG-AFP v0 | post-neck query-scale gate | no | no |

## Localization Memory Bank (`modules/nn/lmb.py`)

LMB tracks GT-level localization quality instead of missed-detection state. It is independent from ReMiss and uses `modules/cfg/lmb.yaml`.

Key concepts:

- Matching: for each GT, LMB finds the best final prediction above `matching.score_threshold`. `score_threshold: auto` must be resolved from the detector's final score threshold before constructing the module.
- States: `missing` means `best_iou < low_iou_threshold` or no candidate exists; `low_iou` means `low_iou_threshold <= best_iou < good_iou_threshold`; `good` means `best_iou >= good_iou_threshold`.
- Stability: `stability.stable_epochs` defines how many consecutive epochs a GT must remain in `low_iou` to count as stable low-IoU.
- Regions: `grid_size` assigns each GT box to a row-major spatial region, using the region with the largest box overlap.
- Runtime: when enabled for FCOS, LMB runs one no-grad pass over the training loader after each epoch from `start_epoch`.
- Metrics: `epoch_snapshot()` records low-IoU counts, stable low-IoU counts, streak statistics, IoU deficit statistics, state transitions, region histograms, and image-region hotspots. Runtime writes accumulated metrics to `lmb/lmb_stability_epoch.json` and `lmb/lmb_stability_epoch.csv`.
- State: `get_extra_state()` and `set_extra_state()` make LMB checkpointable.

LMB does not alter detector forward computation, add losses, or change inference.

## QG-AFP v0 (`modules/nn/qg_afp.py`)

QG-AFP v0 is a FCOS post-neck feature modulation module. It keeps the TorchVision FPN feature-dict contract intact and returns the same keys and tensor shapes.

Key concepts:

- Query source: v0 does not reuse FCOS head logits, because that would create a head-to-neck-to-head loop. It predicts a lightweight proxy objectness map inside the post-neck module and mines top-k feature locations as query seeds.
- Query encoding: each seed combines the local feature vector, a level embedding, normalized spatial position, and proxy score.
- Scale routing: the query MLP predicts a soft level gate over active pyramid levels.
- Residual feature update: query gates are aggregated per batch and level, then applied as `P_l * (1 + residual_scale * alpha_l)`.
- Stability: `residual_scale_init: 0.0` makes the module identity-biased at startup.
- Metrics: training logs include gate-collapse diagnostics such as `qg_afp_gate_entropy`, `qg_afp_gate_max_mean`, `qg_afp_level_usage_entropy`, `qg_afp_level_top1_share`, `qg_afp_alpha_l0`, `qg_afp_residual_scale`, and `qg_afp_query_count` when the module has run. These are emitted through the normal train metric path and are persisted in `history.json` and `results.csv`.
- Config: `modules/cfg/qg_afp.yaml`.

QG-AFP v0 changes detector forward computation when enabled, but it does not add an auxiliary loss. It is currently wired only for FCOS.
