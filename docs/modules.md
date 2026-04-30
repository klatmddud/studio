# Research Modules

The current implemented research-module surface is ReMiss MissBank only. Older unrelated research-module code paths have been removed.

## Config Resolution

`scripts/train.py` accepts:

- `--remiss-config`

`scripts/runtime/module_configs.py` resolves the default ReMiss config path, and `scripts/runtime/module_metadata.py` persists enabled module snapshots to `metadata/modules.yaml` for reproducibility.

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
