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

Tracks false-negative detections (missed objects) across training batches with **temporal tracking** per GT.

- **Hooks**: `start_epoch()`, `end_epoch()` called by `engine.fit()`
- **Hook**: `after_optimizer_step(images, targets, epoch_index)` called after each optimizer step
- **Summary**: `mdmb.summary()` → logged as `mdmb_entries`, `mdmb_images` in `history.json`
- **Arch support**: FCOS
- **Config**: `modules/cfg/mdmb.yaml`

#### 데이터 구조

| 구조체 | 설명 |
|---|---|
| `MDMBEntry` | bank에 저장되는 miss-detected GT 정보 |
| `_GTRecord` | epoch 간 GT별 이력을 추적하는 내부 persistent 레코드 |

**`MDMBEntry` 필드** (bank에서 조회 가능):

| 필드 | 타입 | 설명 |
|---|---|---|
| `image_id` | `str` | COCO image ID |
| `class_id` | `int` | COCO category ID |
| `bbox` | `Tensor[4]` | normalized xyxy bbox |
| `miss_type` | `str` | `"type_a"` (위치 miss) / `"type_b"` (클래스 miss) |
| `consecutive_miss_count` | `int` | 현재까지 연속으로 miss된 epoch 수 |
| `max_consecutive_miss_count` | `int` | 이 GT가 기록한 최대 연속 miss epoch 수 |
| `last_detected_epoch` | `int \| None` | 마지막으로 detected된 epoch (없으면 `None`) |

**Bank 수준 속성**:

| 속성 | 설명 |
|---|---|
| `bank._global_max_consecutive_miss` | 전체 GT 중 최대 연속 miss 횟수 (normalized 용도) |
| `bank.summary()["global_max_consecutive_miss"]` | 동일 값, `history.json`에 기록됨 |

#### 동작 방식 (epoch T)

1. 각 GT에 대해 `classify_miss()` 실행 → detected / type_a / type_b 판정
2. 이전 epoch의 `_GTRecord`와 IoU > 0.95 매칭으로 동일 GT 식별
3. **detected** → `consecutive_miss_count = 0`, `last_detected_epoch = T` 기록 (bank에서 제거)
4. **miss** → `consecutive_miss_count = prev + 1`, `max_consecutive_miss_count` 갱신 후 bank에 저장
5. `_global_max_consecutive_miss` = 전체 `_GTRecord`의 `max_consecutive_miss_count` 최댓값

#### 체크포인트 호환성

| version | 동작 |
|---|---|
| v4 (현재) | `_gt_records`, `_global_max_consecutive_miss` 완전 복원 |
| v3 | bank 복원, `_gt_records`는 빈 상태로 시작 (다음 update()에서 재구성) |
| v1/v2 | bank 초기화 (구조 비호환) |

### RECALL — Selective Loss Reweighting (`recall.py`)

Uses MDMB observations to upweight losses on missed detections.

- Depends on MDMB being enabled
- **Arch support**: FCOS
- **Config**: `modules/cfg/recall.yaml`

### FAR — Forgetting-Aware feature Replay (`far.py`)

Pulls a feature-level consistency loss toward a frozen per-GT anchor captured when the
object was last successfully detected, so that GTs which **relapse** (previously detected,
currently missed) receive a dedicated pressure signal.

- Depends on MDMB being enabled (reads `mdmb._gt_records` for relapse state)
- **Hooks**: `start_epoch()`, `end_epoch()` called by `engine.fit()`
- **Anchor update**: runs inside `after_optimizer_step` via `MDMBFCOS.flush_far_update(...)`, after MDMB has been refreshed
- **Training loss**: `far.compute_loss(...)` adds a `far` entry to the FCOS loss dict
- **Summary**: `far.summary()` → logged as `far` in `history.json`
- **Arch support**: FCOS
- **Config**: `modules/cfg/far.yaml`
- **Design**: see [docs/proposals/far.md](proposals/far.md)

Key config fields:

| 필드 | 기본값 | 설명 |
|---|---|---|
| `lambda_far` | `0.1` | FAR loss scale |
| `anchor_ema_mu` | `0.9` | anchor EMA 계수 (detected 상태 유지 시) |
| `persistence_gamma` | `1.0` | 연속 miss streak 기반 가중 증폭 계수 |
| `min_relapse_streak` | `1` | relapse로 판정하여 anchor를 freeze할 최소 연속 miss 수 |
| `match_threshold` | `0.95` | GT-to-anchor IoU 매칭 임계치 |
| `warmup_epochs` | `1` | FAR loss/anchor 갱신을 시작하기까지 무시할 초기 epoch 수 |
| `feature_keys` | `["0","1","2","p6","p7"]` | FPN 레벨 키 (FCOS 기본) |

## Module × Architecture Compatibility

| Module | FCOS | Faster R-CNN | DINO |
|---|:---:|:---:|:---:|
| MDMB | ✓ | — | — |
| RECALL | ✓ | — | — |
| FAR | ✓ | — | — |
