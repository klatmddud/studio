# Research Modules

The runtime-connected research-module surface is now limited to ReMiss MissBank and Hard Replay.

## Config Resolution

`scripts/train.py` accepts:

- `--remiss-config`
- `--hard-replay-config`

`scripts/runtime/module_configs.py` resolves default module config paths, and `scripts/runtime/module_metadata.py` persists enabled module snapshots to `metadata/modules.yaml` for reproducibility.

## ReMiss MissBank (`modules/nn/mb.py`)

MissBank stores per-GT recurrent missed-detection state for the ReMiss method.

Key concepts:

- Matching: a GT is detected only when a final prediction has the same class, score above `matching.score_threshold`, and IoU above `matching.iou_threshold`. Use `auto` to resolve these thresholds from the detector's final post-processing config. FCOS maps score to `head.score_thresh` and IoU to `head.nms_thresh`; Faster R-CNN maps score to `roi_head.box_score_thresh` and IoU to `roi_head.box_nms_thresh`.
- Mining: `mining.type: online` updates MissBank after each optimization step, while `mining.type: offline` runs a separate no-grad training-set pass on epochs that satisfy `mining.start_epoch` and `mining.interval_epoch`.
- Records: `MissBankRecord` stores image ID, GT ID, class, box, consecutive `miss_count`, current missed state, last match score/IoU, best same-class score/IoU diagnostics, and aggregate seen/missed counts.
- Targets: `get_image_targets()` and `get_batch_labels()` emit binary image-level labels. `0` means no target missed GT in the image, and `1` means at least one currently missed GT satisfies `miss_count >= target.miss_threshold`.
- Loss weighting: when `loss_weight.enabled: true` for FCOS, GTs with larger MissBank `miss_count` increase the positive classification, box regression, and centerness loss weight. Faster R-CNN remains logging/replay-only.
- State: `get_extra_state()` and `set_extra_state()` make MissBank checkpointable.
- Summary: `missbank.summary()` reports record counts, current missed counts, target GT counts, class histograms, and update stats.
- Outputs: runtime writes epoch-level MissBank summaries to `missbank/missbank_epoch.json` and `missbank/missbank_epoch.csv`.
- Config: `modules/cfg/remiss.yaml`.

## Hard Replay (`scripts/runtime/hard_replay.py`)

Hard Replay is a data-layer policy driven by ReMiss MissBank. It replays images containing GTs that MissBank currently marks as missed.

Key concepts:

- Source: current MissBank records from the previous mining/update state.
- Eligibility: `is_missed`, `miss_count >= min_miss_count`, `total_seen >= min_observations`, and `last_epoch` inside `replay_recency_window`. When `latest_mined_epoch_only: true`, `last_epoch` must equal the latest `last_epoch` currently stored in MissBank records and the recency window is ignored. `replay_epochs_after_mining > 0` gates replay to the first N epochs after the latest MissBank mining epoch; `0` keeps unlimited replay.
- Image-level replay: images containing eligible missed GTs receive replay candidates. Weight is `1 + beta * priority`, clipped by `min_image_weight` and `max_image_weight`, then raised by `temperature`.
- Batch mixing: `MixedReplayBatchSampler` walks the base dataset once and adds replay slots according to `replay_ratio`, with optional `max_replays_per_batch`.
- Offline mining: MissBank mining uses base-only loader iteration so replay does not distort mining statistics.
- Config: `modules/cfg/hard_replay.yaml`.

## Support Matrix

| Module | FCOS | Faster R-CNN | DINO |
|---|---:|---:|---:|
| ReMiss MissBank | memory update for Hard Replay; optional miss-count loss weighting | memory update for Hard Replay | no |
| Hard Replay | MissBank-guided image sampling | MissBank-guided image sampling | no |
