# ReMiss MissBank

이 문서는 현재 코드베이스에 구현되어 있는 MissBank 기준 문서다. 현재 런타임에서 ReMiss는 MissBank, 선택적 FCOS loss weighting, Hard Replay 연동까지만 제공한다.

## 위치

| 항목 | 경로 |
|---|---|
| 구현 | `modules/nn/mb.py` |
| 기본 설정 | `modules/cfg/remiss.yaml` |
| 모델 연결 | `scripts/runtime/registry.py` |
| 학습 루프 연결 | `scripts/runtime/engine.py` |
| FCOS loss weighting | `models/detection/wrapper/fcos.py` |

## 역할

MissBank는 학습 중 detector가 반복해서 놓치는 GT instance를 GT 단위로 기록하는 train-time memory다.

기록 단위는 image-level sample이 아니라 GT record다. 각 record는 `image_id`, `gt_id`, class, box, 최근 missed 여부, 연속 miss count, 누적 seen/missed count, best same-class IoU/score 진단값을 가진다.

MissBank는 세 곳에서 사용된다.

1. Online 또는 offline mining으로 missed GT 상태를 갱신한다.
2. FCOS에서 `loss_weight.enabled: true`일 때 positive location loss weight를 만든다.
3. Hard Replay가 replay 후보 이미지를 고를 때 MissBank record를 읽는다.

## 지원 범위

| 모델 | MissBank attach | Online/offline mining | Loss weighting | Hard Replay source |
|---|---:|---:|---:|---:|
| FCOS | yes | yes | yes | yes |
| Faster R-CNN | yes | yes | no | yes |
| DINO | no | no | no | no |

`modules/cfg/remiss.yaml`에는 `models.dino.enabled: false`가 명시되어 있고, `scripts/runtime/registry.py`도 MissBank attach 대상을 `fcos`, `fasterrcnn`으로 제한한다.

## 설정

기본값은 모두 비활성화 상태다.

```yaml
enabled: false
start_epoch: 1
max_records: null

mining:
  type: offline
  start_epoch: 1
  interval_epoch: 1

matching:
  score_threshold: auto
  iou_threshold: auto

target:
  miss_threshold: 2
  aggregation: miss_count

loss_weight:
  enabled: false
  start_epoch: null
  alpha: 0.5
  max_weight: 2.0
  min_miss_count: 1
  min_observations: 1
  normalize_batch_mean: true
```

| 필드 | 의미 |
|---|---|
| `enabled` | MissBank 생성 및 attach 여부 |
| `start_epoch` | MissBank active epoch. `is_active()` 기준 |
| `max_records` | 최대 record 수. `null`이면 제한 없음 |
| `mining.type` | `online` 또는 `offline` |
| `mining.start_epoch` | offline mining 시작 epoch |
| `mining.interval_epoch` | offline mining 주기 |
| `matching.score_threshold` | final prediction score threshold. `auto`면 detector 설정에서 가져옴 |
| `matching.iou_threshold` | final prediction IoU threshold. `auto`면 detector 설정에서 가져옴 |
| `target.miss_threshold` | image label과 target GT 판정에 필요한 연속 miss count |
| `loss_weight.enabled` | FCOS positive loss weighting 사용 여부 |
| `loss_weight.start_epoch` | loss weighting 시작 epoch. `null`이면 `start_epoch` 사용 |
| `loss_weight.alpha` | `1 + alpha * log1p(miss_count)`의 alpha |
| `loss_weight.max_weight` | GT weight 상한 |
| `loss_weight.min_miss_count` | weight 증가에 필요한 최소 miss count |
| `loss_weight.min_observations` | weight 증가에 필요한 최소 seen count |
| `loss_weight.normalize_batch_mean` | foreground weight 평균을 1로 정규화할지 여부 |

`matching.*: auto`는 registry에서 detector post-processing 설정으로 해석한다.

| 모델 | score source | IoU source |
|---|---|---|
| FCOS | `head.score_thresh` | `head.nms_thresh` |
| Faster R-CNN | `roi_head.box_score_thresh` | `roi_head.box_nms_thresh` |

## Record 키

가능하면 COCO annotation id를 사용한다.

```text
{image_id}:ann:{gt_id}
```

`gt_ids`, `annotation_ids`, `ann_ids`가 없거나 유효하지 않으면 class와 normalized box를 SHA1로 해시한다.

```text
{image_id}:box:{gt_class}:{digest}
```

따라서 COCO annotation id가 안정적이면 resume과 replay 분석도 더 안정적이다.

## Matching 기준

GT 하나가 detected로 판정되려면 final prediction 중 다음 조건을 동시에 만족하는 후보가 있어야 한다.

1. prediction class가 GT class와 같다.
2. prediction score가 `matching.score_threshold` 이상이다.
3. GT box와 prediction box IoU가 `matching.iou_threshold` 이상이다.

조건을 만족하는 후보가 있으면 matched로 기록하고 `miss_count`를 0으로 리셋한다. 없으면 missed로 기록하고 `miss_count`, `total_missed`, `max_miss_count`를 갱신한다.

진단용으로 같은 class prediction 중 best IoU, best score도 별도로 저장한다. Hard Replay priority 계산은 이 진단값을 사용한다.

## Mining 흐름

### Online

`mining.type: online`이면 `train_one_epoch()`에서 optimizer step 이후 같은 batch를 eval-style no-grad forward로 다시 통과시켜 MissBank를 갱신한다.

```text
train batch
-> loss backward / optimizer step
-> no-grad model(images)
-> MissBank.update(targets, detections)
```

### Offline

`mining.type: offline`이면 epoch 학습이 끝난 뒤 조건을 만족하는 epoch에서 train loader 전체를 no-grad로 한 번 더 순회한다.

```text
epoch train
-> if mining.start_epoch / interval_epoch due:
   -> set replay sampler base_only=true
   -> no-grad pass over base train set
   -> MissBank.update(...)
   -> restore replay sampler state
```

Offline mining 시간은 해당 epoch의 `history.json` 및 `results.csv`에 `remiss_mining_time_sec`로 기록된다.

## Image Target API

현재 구현은 binary image-level target만 제공한다.

| label | 의미 |
|---:|---|
| `0` | image 안에 target missed GT가 없음 |
| `1` | image 안에 `is_missed == true`이고 `miss_count >= target.miss_threshold`인 GT가 하나 이상 있음 |

관련 API:

- `get_image_targets(image_ids)`
- `get_batch_labels(targets=...)`

이 label은 현재 detector forward에 auxiliary head로 연결되어 있지 않다.

## FCOS Loss Weighting

`loss_weight.enabled: true`이면 FCOS wrapper가 기본 TorchVision FCOS loss path 대신 MissBank-aware path를 사용한다.

GT별 weight는 다음 조건을 만족할 때만 1보다 커진다.

- record가 존재함
- `miss_count >= loss_weight.min_miss_count`
- `total_seen >= loss_weight.min_observations`

계산식:

```text
raw_weight = 1 + alpha * log1p(miss_count)
weight = clamp(raw_weight, min=1, max=max_weight)
```

`normalize_batch_mean: true`이면 foreground location weight 평균이 1이 되도록 나눈다. 적용 대상은 FCOS classification, bbox regression, centerness positive loss다. Inference 경로는 바뀌지 않는다.

학습 metric에는 다음 값이 추가될 수 있다.

- `train_remiss_loss_weight_active`
- `train_remiss_loss_weight_positive`
- `train_remiss_loss_weight_raw_mean`
- `train_remiss_loss_weight_raw_max`
- `train_remiss_loss_weight_used_mean`
- `train_remiss_loss_weight_used_max`

## Checkpoint

MissBank는 `nn.Module.get_extra_state()` / `set_extra_state()`를 구현한다. checkpoint state dict에는 `missbank._extra_state`가 포함된다.

저장되는 항목:

- `version`
- `current_epoch`
- resolved config
- records
- update stats

Baseline checkpoint에서 MissBank를 새로 켜는 경우 `missbank._extra_state` missing key는 optional로 허용된다.

## Epoch 출력

Runtime은 MissBank epoch metric을 다음 위치에 저장한다.

| 파일 | 내용 |
|---|---|
| `output_dir/missbank/missbank_epoch.json` | epoch별 MissBank metric list |
| `output_dir/missbank/missbank_epoch.csv` | flatten된 CSV |

주요 필드:

| 필드 | 의미 |
|---|---|
| `num_seen_gts` | 해당 epoch에 갱신된 GT 수 |
| `num_missed_gts` | 현재 missed로 기록된 GT 수 |
| `num_target_gts` | `miss_threshold`를 넘은 missed GT 수 |
| `num_images_seen` | 해당 epoch에 갱신된 image 수 |
| `num_images_with_miss` | missed GT가 있는 image 수 |
| `num_images_with_target` | target GT가 있는 image 수 |
| `missed_gt_ratio` | `num_missed_gts / num_seen_gts` |
| `target_gt_ratio` | `num_target_gts / num_seen_gts` |
| `image_miss_ratio` | `num_images_with_miss / num_images_seen` |
| `image_target_ratio` | `num_images_with_target / num_images_seen` |
| `mean_miss_count` | missed GT의 평균 연속 miss count |
| `max_miss_count` | missed GT의 최대 연속 miss count |
| `mean_gt_miss_rate` | GT별 누적 miss rate 평균 |
| `miss_gt_jaccard_stability` | 직전 snapshot 대비 missed GT set Jaccard |
| `miss_gt_churn_rate` | `1 - miss_gt_jaccard_stability` |
| `new_miss_rate` | 현재 missed GT 중 새로 들어온 비율 |
| `persistent_miss_ratio` | 현재 missed GT 중 target GT 비율 |

## 구현 범위

현재 MissBank는 GT 단위 missed-state memory, binary image target 생성, FCOS loss weighting, Hard Replay source 역할만 가진다. Detector feature를 직접 주입하거나 별도 auxiliary prediction head를 만들지는 않는다.
