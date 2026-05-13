# ReMiss Hard Replay

Hard Replay는 ReMiss MissBank가 기록한 missed GT를 기반으로 학습 batch에 replay image slot을 섞는 data-layer 기능이다. 현재 구현은 full-image replay만 수행한다.

## 위치

| 항목 | 경로 |
|---|---|
| 구현 | `scripts/runtime/hard_replay.py` |
| 기본 설정 | `modules/cfg/hard_replay.yaml` |
| DataLoader 연결 | `scripts/runtime/data.py` |
| Epoch refresh | `scripts/runtime/engine.py` |

## 역할

Hard Replay는 detector forward를 바꾸지 않는다. `build_train_dataloaders()`에서 일반 sampler 대신 `MixedReplayBatchSampler`를 사용해 base sample과 replay sample을 batch 안에 함께 넣는다.

Replay 후보는 MissBank record에서 온다. 이미지 안에 eligible missed GT가 있으면 그 image가 replay 후보가 된다.

## 지원 범위

| 모델 | Hard Replay |
|---|---:|
| FCOS | yes |
| Faster R-CNN | yes |
| DINO | no |

Hard Replay는 MissBank가 attach된 모델에서만 의미가 있다. MissBank가 없으면 controller는 `missing_missbank` summary를 기록하고 replay를 수행하지 않는다.

## 설정

기본값은 비활성화 상태다.

```yaml
enabled: false
start_epoch: 2
warmup_epochs: 0
replay_ratio: 0.25
max_replays_per_batch: 0
beta: 1.0
temperature: 1.0
max_image_weight: 5.0
min_image_weight: 1.0
replacement: true
min_miss_count: 1
min_observations: 1
replay_recency_window: 1
latest_mined_epoch_only: true
replay_epochs_after_mining: 0
max_replays_per_gt_per_epoch: 4
```

| 필드 | 의미 |
|---|---|
| `enabled` | Hard Replay controller 생성 여부 |
| `start_epoch` | replay ratio가 활성화되기 시작하는 epoch |
| `warmup_epochs` | `replay_ratio`까지 선형 증가하는 warmup 길이 |
| `replay_ratio` | batch slot 중 replay로 채울 목표 비율. `0 <= ratio < 1` |
| `max_replays_per_batch` | batch당 replay slot 상한. `0`이면 ratio 기반 값만 사용 |
| `beta` | image weight 계산에서 priority 계수 |
| `temperature` | sampling weight에 적용하는 exponent |
| `max_image_weight` | image weight 상한 |
| `min_image_weight` | image weight 하한 |
| `replacement` | replay schedule 구성 시 후보 반복 허용 정책. 현재 sampler는 expanded candidate에서 비복원 추출 |
| `min_miss_count` | replay eligible GT의 최소 연속 miss count |
| `min_observations` | replay eligible GT의 최소 seen count |
| `replay_recency_window` | 최근 epoch 기준 eligible window. `latest_mined_epoch_only: true`이면 무시됨 |
| `latest_mined_epoch_only` | MissBank record의 `last_epoch`가 최신 mining epoch와 같은 GT만 사용 |
| `replay_epochs_after_mining` | 최신 mining epoch 이후 N epoch 동안만 replay. `0`이면 제한 없음 |
| `max_replays_per_gt_per_epoch` | GT 수에 따른 image별 replay cap 계산 계수 |

## Epoch 흐름

`engine.fit()`은 매 epoch 시작 시 다음 순서로 replay index를 갱신한다.

```text
set_data_loader_epoch(epoch)
set_missbank_epoch(epoch)
refresh_hard_replay(train_loader, model, epoch)
train_one_epoch(...)
offline MissBank mining if due
write Hard Replay summary
```

중요한 점은 Hard Replay refresh가 epoch 학습 전에 실행된다는 점이다. 따라서 replay 후보는 직전까지 MissBank에 저장된 상태를 사용한다. Offline mining을 쓰는 경우 일반적으로 이번 epoch 끝에서 갱신된 MissBank 상태는 다음 epoch replay에 반영된다.

Offline MissBank mining 중에는 replay가 mining 통계를 왜곡하지 않도록 sampler의 `base_only`가 임시로 켜진다.

## Replay 후보 조건

GT record는 다음 조건을 모두 만족해야 eligible이다.

1. `record.is_missed == true`
2. `record.miss_count >= min_miss_count`
3. `record.total_seen >= min_observations`
4. epoch recency 조건 통과

Recency 조건은 설정에 따라 달라진다.

| 설정 | 조건 |
|---|---|
| `latest_mined_epoch_only: true` | `record.last_epoch == latest_mined_epoch` |
| `latest_mined_epoch_only: false`, `replay_recency_window <= 0` | recency 제한 없음 |
| `latest_mined_epoch_only: false`, `replay_recency_window > 0` | `epoch - record.last_epoch <= replay_recency_window` |

`replay_epochs_after_mining > 0`이면 추가로 `1 <= epoch - latest_mined_epoch <= replay_epochs_after_mining`이어야 한다.

## Priority와 Sampling Weight

GT priority는 다음 식으로 계산된다.

```text
priority =
  miss_count
  + 0.25 * total_missed
  + 0.1 * max_miss_count
  + max(0, iou_threshold - best_iou)
  + 0.25 * max(0, score_threshold - best_score)
```

`best_iou`나 `best_score`가 없으면 해당 gap은 0이다.

Image priority는 image 안의 eligible GT priority 합이다.

```text
raw_image_weight = 1 + beta * image_priority
image_weight = clamp(raw_image_weight, min_image_weight, max_image_weight)
sampling_weight = image_weight ** temperature
cap = max_replays_per_gt_per_epoch * eligible_gt_count
```

Sampler는 image candidate를 cap만큼 확장한 뒤 `torch.multinomial(..., replacement=False)`로 epoch replay schedule을 만든다.

## Batch 구성

`MixedReplayBatchSampler`는 base dataset을 매 epoch 한 번 순회한다. Replay가 active이면 batch size 안에서 일부 slot을 replay로 채운다.

```text
replay_slots = floor(batch_size * replay_ratio)
if max_replays_per_batch > 0:
  replay_slots = min(replay_slots, max_replays_per_batch)
base_slots = batch_size - replay_slots
```

`replay_slots >= batch_size`이면 base sample이 없어지므로 에러를 낸다.

DDP에서는 각 rank가 base indices를 shard한다. Replay schedule도 rank와 epoch seed를 포함해 rank별로 생성된다.

## Summary와 출력

Runtime은 다음 파일을 쓴다.

| 파일 | 내용 |
|---|---|
| `output_dir/hard-replay/hard_replay_epoch.json` | epoch별 replay summary list |
| `output_dir/hard-replay/hard_replay_epoch.csv` | flatten된 CSV |
| `output_dir/hard-replay/hard_replay_state.json` | 마지막 replay summary |

주요 summary 필드:

| 필드 | 의미 |
|---|---|
| `enabled` | controller 활성 여부 |
| `active` | 해당 epoch에 replay sample이 실제로 들어갔는지 여부 |
| `reason` | inactive 또는 active 이유 |
| `replay_ratio_requested` | config schedule이 요청한 replay ratio |
| `replay_ratio_effective` | 실제 replay sample 비율 |
| `replay_num_images` | replay 후보 image 수 |
| `replay_num_active_gt` | replay 후보 GT 수 |
| `latest_mined_epoch` | MissBank record 기준 최신 mining epoch |
| `epochs_since_mining` | 현재 epoch와 최신 mining epoch 차이 |
| `replay_epochs_after_mining` | 설정된 replay window |
| `replay_mean_image_weight` | 후보 image weight 평균 |
| `replay_mean_gt_priority` | 후보 GT priority 평균 |
| `replay_samples` | 실제 replay sample 수 |
| `replay_unique_images` | 실제 replay된 unique image 수 |
| `replay_exposure_per_gt` | active GT당 replay 노출 수 |
| `replay_slots_per_batch` | batch당 replay slot 수 |
| `base_slots_per_batch` | batch당 base slot 수 |

대표적인 `reason` 값:

| reason | 의미 |
|---|---|
| `disabled` | config disabled |
| `warmup` | scheduled ratio가 0 |
| `missing_missbank` | 모델에 MissBank가 없음 |
| `outside_mining_window` | `replay_epochs_after_mining` 조건 밖 |
| `no_missed_gt` | eligible missed GT 없음 |
| `active` | replay 후보가 있음 |

## 구현 범위

현재 Hard Replay는 MissBank record를 source로 사용하는 full-image replay sampler다. Replay 후보를 실패 유형별로 나누거나 crop sample로 변환하지 않는다.
