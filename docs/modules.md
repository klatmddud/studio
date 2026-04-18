# Research Modules

All modules live in `modules/nn/` and are configured via `modules/cfg/*.yaml`.
**All modules are disabled by default.**

## Enabling a Module

Edit the corresponding `modules/cfg/<module>.yaml`:

```yaml
enabled: true   # set to true to activate
```

Per-architecture overrides can be set in the same file:

```yaml
fcos:
  some_param: value
fasterrcnn:
  some_param: other_value
```

## Module Overview

### MDMB — Missed Detection Memory Bank (`mdmb.py`)

Tracks false-negative detections (missed objects) across training batches.

- **Hooks**: `start_epoch()`, `end_epoch()` called by `engine.fit()`
- **Hook**: `after_optimizer_step(images, targets, epoch_index)` called after each optimizer step
- **Summary**: `mdmb.summary()` → logged as `mdmb_entries`, `mdmb_images` in `history.json`
- **Arch support**: FCOS, Faster R-CNN
- **Config**: `modules/cfg/mdmb.yaml`

### RECALL — Selective Loss Reweighting (`recall.py`)

Uses MDMB observations to upweight losses on missed detections.

- Depends on MDMB being enabled
- **Arch support**: FCOS
- **Config**: `modules/cfg/recall.yaml`

### CFP — Counterfactual Feature Perturbation (`cfp.py`)

Generates counterfactual augmentations at the feature level.

- Loss: `ops/cfp_loss.py`
- **Arch support**: Faster R-CNN
- **Config**: `modules/cfg/cfp.yaml`

### MODS — Missed Object Direct Supervision (`mods.py`)

Adds direct supervision signal for hard-negative/missed object regions.

- Loss: `ops/mods_loss.py`
- **Config**: `modules/cfg/mods.yaml`

### SCA — Soft Counterfactual Assignment (`sca.py`)

Soft label assignment strategy to handle ambiguous detections.

- Loss: `ops/sca_loss.py`
- **Arch support**: FCOS
- **Config**: `modules/cfg/sca.yaml`

## Module × Architecture Compatibility

| Module | FCOS | Faster R-CNN | DINO |
|---|:---:|:---:|:---:|
| MDMB | ✓ | ✓ | — |
| RECALL | ✓ | — | — |
| CFP | — | ✓ | — |
| MODS | ✓ | ✓ | — |
| SCA | ✓ | — | — |
