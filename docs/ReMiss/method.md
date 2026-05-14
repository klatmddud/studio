# ReMiss Method

이 문서는 현재 코드베이스에 실제로 연결된 ReMiss 구현 기준으로 정리한다.

현재 ReMiss 경로는 다음 두 컴포넌트로 제한된다.

1. MissBank: 반복 missed GT를 기록하는 train-time memory
2. Hard Replay: MissBank record를 기반으로 replay image를 batch에 섞는 sampler

현재 ReMiss 런타임은 memory와 replay 중심 경로이며, detector feature를 직접 주입하지 않는다.

## 구현 표면

| 컴포넌트 | 경로 | 런타임 연결 |
|---|---|---|
| MissBank | `modules/nn/mb.py` | `scripts/runtime/registry.py`가 FCOS/Faster R-CNN 모델에 attach |
| Hard Replay | `scripts/runtime/hard_replay.py` | `scripts/runtime/data.py`가 train loader batch sampler로 attach |

기본 설정:

- `modules/cfg/remiss.yaml`
- `modules/cfg/hard_replay.yaml`

CLI override:

```bash
uv run scripts/train.py \
  --config scripts/cfg/train.yaml \
  --model models/detection/cfg/fcos.yaml \
  --data kitti \
  --remiss-config modules/cfg/remiss.yaml \
  --hard-replay-config modules/cfg/hard_replay.yaml
```

## 전체 흐름

```text
train.py
  -> resolve_module_config_paths(remiss, hard_replay)
  -> build_model_from_path()
     -> attach MissBank when remiss.enabled=true and arch in {fcos, fasterrcnn}
  -> build_train_dataloaders()
     -> attach HardReplayController when hard_replay.enabled=true
  -> engine.fit()
     -> per epoch:
        1. set MissBank current epoch
        2. refresh Hard Replay from current MissBank records
        3. train one epoch
        4. update MissBank online after each step or offline after epoch
        5. write MissBank and Hard Replay summaries
```

## MissBank

MissBank는 final post-processed detection과 GT를 비교해 GT별 missed 상태를 갱신한다.

GT가 detected가 되려면 같은 class prediction 중 score와 IoU가 설정 threshold를 통과해야 한다. 통과하지 못하면 missed로 기록하고 `miss_count`를 증가시킨다. 통과하면 `miss_count`를 0으로 리셋한다.

MissBank 문서:

- [MissBank.md](MissBank.md)

## Hard Replay

Hard Replay는 MissBank에서 현재 missed이고 충분히 반복 관측된 GT를 찾고, 그 GT를 포함한 image를 replay 후보로 만든다.

Replay는 crop이 아니라 full image 단위다. Sampler는 base dataset을 한 번 순회하면서 batch 안에 replay slot을 추가한다.

Hard Replay 문서:

- [HardReplay.md](HardReplay.md)

## Mining 모드

### Online mining

Online mining은 매 train step의 optimizer update 이후 같은 batch에 대해 no-grad inference를 한 번 더 수행하고 MissBank를 갱신한다.

장점:

- MissBank가 빠르게 최신 상태를 반영한다.

비용:

- 학습 step마다 추가 forward가 필요하다.

### Offline mining

Offline mining은 epoch 학습이 끝난 뒤 train loader를 no-grad로 한 번 더 순회한다. 실행 epoch는 `mining.start_epoch`, `mining.interval_epoch`로 결정된다.

장점:

- 한 epoch 단위의 일관된 MissBank snapshot을 만든다.
- Hard Replay와 함께 사용할 때 mining 중 replay를 `base_only`로 끄므로 replay sample이 mining 통계를 오염시키지 않는다.

비용:

- mining epoch마다 train set forward pass가 추가된다.

## 지원 행렬

| 기능 | FCOS | Faster R-CNN | DINO |
|---|---:|---:|---:|
| MissBank attach | yes | yes | no |
| Online MissBank update | yes | yes | no |
| Offline MissBank mining | yes | yes | no |
| MissBank checkpoint state | yes | yes | no |
| Hard Replay | yes | yes | no |

## 출력

| 경로 | 내용 |
|---|---|
| `history.json` | epoch별 train/val metric. offline mining epoch에는 `remiss_mining_time_sec` 포함 |
| `results.csv` | `history.json` flatten CSV |
| `metadata/modules.yaml` | enabled module config snapshot |
| `missbank/missbank_epoch.json` | MissBank epoch metric list |
| `missbank/missbank_epoch.csv` | MissBank metric CSV |
| `hard-replay/hard_replay_epoch.json` | Hard Replay summary list |
| `hard-replay/hard_replay_epoch.csv` | Hard Replay summary CSV |
| `hard-replay/hard_replay_state.json` | 마지막 Hard Replay summary |

## 구현 범위

이 문서의 범위는 현재 연결된 memory와 replay sampler에 한정한다. 새로운 auxiliary head, feature injection, 실패 유형별 replay 같은 기능을 도입하려면 별도 코드 경로, 설정, 출력 문서가 함께 추가되어야 한다.
