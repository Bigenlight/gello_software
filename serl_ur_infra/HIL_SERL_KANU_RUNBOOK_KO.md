# HIL-SERL learner Kanu 실행 runbook

> 상태: production-shaped CLI의 dry-run/초기 bounded-run 절차
>
> 기준일: 2026-07-27 KST
>
> 구현 상태와 차단점: [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](./HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)

## 먼저 읽을 요약

- Kanu learner server는 `127.0.0.1:50053`에만 bind하고 laptop은 SSH local forwarding으로 접속한다.
- Kanu에서는 JAX/JAXLIB 0.5.3 CUDA 환경을 사용하고 CLI에 `--require-jax-backend gpu`를 반드시 준다.
- fake canonical demo는 `--dry-run` 전용이다. CLI가 live serving에서 자동 거부한다.
- 첫 순서는 fake demo 생성 → actual classifier/ResNet SHA 확인 → GPU dry-run이다.
- live/bounded run에는 real canonical robot demo가 필요하다. fake marker를 제거하거나 검사를 우회하지 않는다.
- checkpoint는 5,000 learner step마다 약 305 MiB가 추가되며 삭제·덮어쓰기·pruning하지 않는다. filesystem reserve 기본값은 2 GiB다.
- replay/intervention buffer는 RAM-only다. process restart와 checkpoint resume가 replay를 복구하지 않는다.
- 현재 코드는 raw image + random crop이다. no-aug GAP512 feature mode는 결정/구현 전이므로 Kanu 장기 run 전에 memory gate를 다시 확인한다.

---

## 1. 사전 조건

이 PC의 실제 SSH alias는 `kanu`다. 2026-07-27 read-only preflight에서 접속, host RAM, GPU, 기존 `il` Python 환경을 확인했다. production learner dry-run 자체는 아직 실행하지 않았다.

```bash
ssh -G kanu | sed -n '1,40p'
ssh -T kanu true
```

확인 당시 Kanu는 RAM 251 GiB(available 152 GiB), RTX A4000 16 GB 8장을 제공했고 `/home/junhyeong/miniconda3/envs/il/bin/python`에서 JAX/JAXLIB 0.5.3, Flax 0.10.5, backend `gpu`, device 8개를 확인했다. 이 값은 실행 직전에 다시 확인한다.

두 번째 command가 성공하기 전에는 아래 Kanu command를 실행 가능하다고 간주하지 않는다. 조직 VPN, SSH config 또는 실제 hostname은 사용자가 관리하는 값이므로 문서에서 추측해 바꾸지 않는다.

### 1.1 repository와 commit

Kanu에는 검증할 production learner commit이 checkout돼 있어야 한다. 아래 경로는 현재 historical Kanu workspace layout을 따른 예시다. 실제 clone 위치가 다르면 `HIL_KANU_REPO`만 바꾼다.

```bash
export HIL_KANU_REPO=/home/junhyeong/workspace/youngwoong/gello_software

cd "$HIL_KANU_REPO"
git status --short --branch
git rev-parse HEAD
git submodule status third_party/hil-serl
```

dirty workspace나 예상하지 않은 commit에서 production run을 시작하지 않는다. `third_party/hil-serl` submodule도 초기화돼 있어야 한다.

```bash
cd "$HIL_KANU_REPO"
git submodule update --init --recursive third_party/hil-serl
```

### 1.2 Python/JAX GPU 환경

아래는 현재 검증 target이다. `runtime fail-closed` 항목은 production CLI가 시작 시 버전을 직접 비교한다. 나머지는 CPU lock/known-good environment에 고정된 값이며 아직 같은 runtime validator가 강제하지 않으므로 Kanu environment 준비 단계에서 수동 확인한다.

| package | version | 현재 enforcement |
| --- | --- | --- |
| JAX / JAXLIB | 0.5.3 / 0.5.3 | runtime fail-closed |
| Flax | 0.10.5 | runtime fail-closed |
| Distrax | 0.1.5 | runtime fail-closed |
| TensorFlow Probability | 0.25.0 | runtime fail-closed |
| W&B | 0.26.0 | W&B enabled일 때 runtime fail-closed |
| NumPy | 1.26.4 | lock/known environment, 수동 확인 |
| Optax | 0.2.4 | lock/known environment, 수동 확인 |
| protobuf | 7.34.1, pure-Python compatibility mode | lock + implementation compatibility 검사; version 비교는 미구현 |
| grpcio | 1.74.0 | lock/known environment, 수동 확인 |
| Orbax | 0.11.5 | lock/known environment, 수동 확인 |

`requirements-learner.lock`은 CPU-local 검증 환경용으로 `jaxlib==0.5.3`을 포함한다. 공유 Kanu conda environment에 그대로 설치하거나 upgrade하지 않는다. 별도의 CUDA-capable environment를 준비하고 그 interpreter 경로를 명시한다.

이전 receive-only overlay(`/tmp/gello-hil-rl-receive-overlay-v2`)는 protobuf 3.20.3과 당시 Kanu base JAX를 전제로 하므로 production learner 환경으로 재사용하지 않는다.

```bash
export HIL_KANU_PYTHON=/absolute/path/to/jax-0.5.3-cuda-env/bin/python
export HIL_GPU_INDEX=GPU_NUMBER_SELECTED_AFTER_NVIDIA_SMI

CUDA_VISIBLE_DEVICES="$HIL_GPU_INDEX" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$HIL_KANU_REPO/serl_ur_infra" \
"$HIL_KANU_PYTHON" - <<'PY'
import jax
import jaxlib
import flax
import distrax
import tensorflow_probability
import wandb
import grpc
import numpy
import optax
from google import protobuf
from google.protobuf.internal import api_implementation
from importlib.metadata import version

print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("flax", flax.__version__)
print("distrax", distrax.__version__)
print("tensorflow_probability", tensorflow_probability.__version__)
print("wandb", wandb.__version__)
print("numpy", numpy.__version__)
print("optax", optax.__version__)
print("protobuf", protobuf.__version__)
print("protobuf implementation", api_implementation.Type())
print("grpcio", grpc.__version__)
print("orbax-checkpoint", version("orbax-checkpoint"))
print("backend", jax.default_backend())
print("devices", jax.devices())
assert jax.default_backend() == "gpu"
assert api_implementation.Type() == "python"
PY
```

`GPU_NUMBER_SELECTED_AFTER_NVIDIA_SMI`를 그대로 실행하지 말고 직전에 `nvidia-smi`로 확인한 번호로 바꾼다. GPU availability는 이전 세션 결과를 재사용하지 않는다.

```bash
nvidia-smi
```

### 1.3 immutable assets

actual reward classifier:

```text
/home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150
```

expected SHA-256:

```text
e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997
```

ResNet repository asset expected SHA-256:

```text
175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b
```

검사 명령:

```bash
export HIL_CLASSIFIER=/home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150
export HIL_RESNET_SOURCE="$HIL_KANU_REPO/third_party/hil-serl/examples/experiments/resnet10_params.pkl"
export HIL_CLASSIFIER_SHA256=e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997
export HIL_RESNET_SHA256=175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b

printf '%s  %s\n' "$HIL_CLASSIFIER_SHA256" "$HIL_CLASSIFIER" | sha256sum --check --strict -
printf '%s  %s\n' "$HIL_RESNET_SHA256" "$HIL_RESNET_SOURCE" | sha256sum --check --strict -
```

upstream classifier는 `/home/junhyeong/.serl/resnet10_params.pkl`도 사용한다. 파일이 이미 있으면 같은 SHA인지 확인한다. 다르면 지우거나 덮어쓰지 말고 run을 중단한다.

```bash
if test -e /home/junhyeong/.serl/resnet10_params.pkl; then
  printf '%s  %s\n' "$HIL_RESNET_SHA256" /home/junhyeong/.serl/resnet10_params.pkl | sha256sum --check --strict -
fi
```

## 2. run directory와 disk preflight

run마다 새 영속 directory를 사용한다. `/tmp`는 acceptance scratch에는 쓸 수 있지만 실제 checkpoint lineage에는 쓰지 않는다.

```bash
export HIL_RUN_ID=UNIQUE_RUN_ID
export HIL_RUN_ROOT="/absolute/persistent/path/hil-serl-runs/$HIL_RUN_ID"
export HIL_CHECKPOINT_ROOT="$HIL_RUN_ROOT/checkpoints"
export HIL_WANDB_DIR="$HIL_RUN_ROOT/wandb"
export HIL_JSONL_PATH="$HIL_RUN_ROOT/logs/learner.jsonl"
export HIL_RESNET_CACHE="$HIL_RUN_ROOT/assets/resnet10_params.pkl"
export HIL_GRASP_PENALTY=-0.02

mkdir -p "$HIL_RUN_ROOT/logs" "$HIL_RUN_ROOT/wandb" "$HIL_RUN_ROOT/assets"
df -h "$HIL_RUN_ROOT"
df -B1 "$HIL_RUN_ROOT"
```

fresh run의 checkpoint root에는 기존 `checkpoint_*` entry가 없어야 한다. CLI가 root를 만들 수 있으므로 미리 만들 필요는 없다.

checkpoint 하나의 local 실측 payload는 약 305 MiB다. 기본 5,000-step 주기이고 pruning하지 않는다. 예상 checkpoint 수에 payload 총량과 `--checkpoint-reserve-gib`를 더해 disk를 잡는다.

동일 checkpoint root에는 learner process 하나만 허용된다. `.learner-writer.lock`은 advisory lock metadata file이며 process 종료 후 파일 자체가 남아도 lock은 해제된다. 파일 존재 여부만 보고 임의 삭제하지 않는다.

## 3. fake acceptance demo 생성

output path는 존재하면 안 된다. generator는 overwrite하지 않는다.

```bash
export HIL_FAKE_DEMO="$HIL_RUN_ROOT/fake-canonical-demo.pkl"

cd "$HIL_KANU_REPO"
PYTHONPATH="$HIL_KANU_REPO/serl_ur_infra" \
"$HIL_KANU_PYTHON" \
  serl_ur_infra/scripts/generate_fake_canonical_demo.py \
  --output "$HIL_FAKE_DEMO"

sha256sum "$HIL_FAKE_DEMO"
```

출력 JSON에 아래가 있어야 한다.

```text
"synthetic_acceptance_only": true
"transition_count": 2
"synthetic_transition_count": 2
```

이 파일은 다음 절의 `--dry-run`에만 사용한다.

현재 fake generator의 penalty 값은 `-0.02`로 고정돼 있다. 따라서 fake dry-run의 `HIL_GRASP_PENALTY`도 `-0.02`를 유지한다. 다른 task penalty의 synthetic artifact가 필요하면 generator를 명시적으로 확장하고 strict test를 추가해야 하며 marker나 pickle을 손으로 고치지 않는다.

## 4. Kanu GPU dry-run

dry-run은 실제 classifier, ResNet, hybrid SAC agent, production composition, fingerprint, JSONL/W&B offline을 준비하지만 gRPC port를 bind하거나 learner update를 실행하지 않는다.

```bash
cd "$HIL_KANU_REPO"

CUDA_VISIBLE_DEVICES="$HIL_GPU_INDEX" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_SILENT=true \
WANDB_DISABLE_CODE=true \
PYTHONPATH="$HIL_KANU_REPO/serl_ur_infra:$HIL_KANU_REPO/third_party/hil-serl/serl_launcher" \
"$HIL_KANU_PYTHON" \
  serl_ur_infra/scripts/run_rlpd_learner_server.py \
  --host 127.0.0.1 \
  --port 50053 \
  --classifier-checkpoint "$HIL_CLASSIFIER" \
  --expected-classifier-sha256 "$HIL_CLASSIFIER_SHA256" \
  --reward-threshold 0.85 \
  --reward-model-id cube-in-cup-checkpoint-150 \
  --demo-path "$HIL_FAKE_DEMO" \
  --checkpoint-root "$HIL_CHECKPOINT_ROOT" \
  --checkpoint-reserve-gib 2 \
  --jsonl-path "$HIL_JSONL_PATH" \
  --wandb-dir "$HIL_WANDB_DIR" \
  --wandb-mode offline \
  --wandb-project hil-serl \
  --run-name "$HIL_RUN_ID-kanu-dry-run" \
  --hil-serl-root "$HIL_KANU_REPO/third_party/hil-serl" \
  --resnet-source "$HIL_RESNET_SOURCE" \
  --resnet-cache "$HIL_RESNET_CACHE" \
  --replay-capacity 128 \
  --intervention-capacity 32 \
  --grasp-penalty "$HIL_GRASP_PENALTY" \
  --require-jax-backend gpu \
  --dry-run
```

성공 시 stdout에 `rlpd_learner_dry_run_passed`가 있어야 한다. JSONL과 W&B offline directory도 확인한다.

```bash
grep -n 'learner_process_ready' "$HIL_JSONL_PATH"
find "$HIL_WANDB_DIR" -maxdepth 2 -type d -name 'offline-run-*' -print
```

주의:

- dry-run은 port를 bind하지 않는다.
- fake demo로 CTA update/checkpoint를 검증하지 않는다.
- fake demo를 그대로 두고 `--dry-run`만 제거하면 CLI가 의도적으로 실패해야 한다.
- dry-run도 classifier와 agent를 GPU에 올리므로 RSS/GPU memory를 기록한다.
- dry-run은 replay나 update를 검증하지 않으므로 작은 128/32 capacity를 사용한다. 기본 50k/10k는 raw camera arrays의 virtual allocation이 약 10.99 GiB이므로 dry-run에서 쓸 이유가 없다. RSS와 함께 VMS도 기록한다.

```bash
nvidia-smi
```

## 5. real canonical demo가 준비된 뒤 bounded learner run

이 절은 fake demo로 실행하면 안 된다. strict loader를 통과하는 실제 EEF-space canonical robot demo path를 지정한다.

dry-run과 log/checkpoint lineage를 섞지 않도록 live run에는 새 run ID와 새 root를 잡는다.

```bash
export HIL_RUN_ID=UNIQUE_LIVE_RUN_ID
export HIL_RUN_ROOT="/absolute/persistent/path/hil-serl-runs/$HIL_RUN_ID"
export HIL_CHECKPOINT_ROOT="$HIL_RUN_ROOT/checkpoints"
export HIL_WANDB_DIR="$HIL_RUN_ROOT/wandb"
export HIL_JSONL_PATH="$HIL_RUN_ROOT/logs/learner.jsonl"
export HIL_RESNET_CACHE="$HIL_RUN_ROOT/assets/resnet10_params.pkl"
export HIL_GRASP_PENALTY=-0.02

mkdir -p "$HIL_RUN_ROOT/logs" "$HIL_RUN_ROOT/wandb" "$HIL_RUN_ROOT/assets"
df -h "$HIL_RUN_ROOT"
```

첫 production acceptance는 continuous mode보다 `--target-learner-step 5000` bounded run을 권장한다. target은 checkpoint period의 배수여야 한다.

```bash
export HIL_REAL_DEMO=/absolute/path/to/canonical-robot-demo.pkl

cd "$HIL_KANU_REPO"

CUDA_VISIBLE_DEVICES="$HIL_GPU_INDEX" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_SILENT=true \
WANDB_DISABLE_CODE=true \
PYTHONPATH="$HIL_KANU_REPO/serl_ur_infra:$HIL_KANU_REPO/third_party/hil-serl/serl_launcher" \
"$HIL_KANU_PYTHON" \
  serl_ur_infra/scripts/run_rlpd_learner_server.py \
  --host 127.0.0.1 \
  --port 50053 \
  --classifier-checkpoint "$HIL_CLASSIFIER" \
  --expected-classifier-sha256 "$HIL_CLASSIFIER_SHA256" \
  --reward-threshold 0.85 \
  --reward-model-id cube-in-cup-checkpoint-150 \
  --demo-path "$HIL_REAL_DEMO" \
  --checkpoint-root "$HIL_CHECKPOINT_ROOT" \
  --checkpoint-reserve-gib 2 \
  --jsonl-path "$HIL_JSONL_PATH" \
  --wandb-dir "$HIL_WANDB_DIR" \
  --wandb-mode offline \
  --wandb-project hil-serl \
  --run-name "$HIL_RUN_ID-kanu-5000" \
  --hil-serl-root "$HIL_KANU_REPO/third_party/hil-serl" \
  --resnet-source "$HIL_RESNET_SOURCE" \
  --resnet-cache "$HIL_RESNET_CACHE" \
  --replay-capacity 50000 \
  --intervention-capacity 10000 \
  --grasp-penalty "$HIL_GRASP_PENALTY" \
  --max-workers 4 \
  --max-message-bytes 16777216 \
  --require-jax-backend gpu \
  --target-learner-step 5000 \
  --poll-interval 0.1
```

server는 online replay가 100개에 도달할 때까지 policy version 0으로 inference/ingress를 제공하며 학습을 기다린다. stdout의 `rlpd_learner_server_ready`를 확인한 뒤 laptop tunnel과 actor를 시작한다.

현재 raw replay 기본 capacity의 camera arrays만 약 10.99 GiB가 될 수 있다. GAP512 feature mode가 구현되기 전에 이 run을 한다면 Kanu host RAM을 먼저 확인하거나 승인된 더 작은 capacity를 명시한다. capacity를 바꾸면 실험 기록에 남긴다.

## 6. SSH loopback tunnel

Kanu server는 loopback에만 bind된다. laptop terminal에서 다음 tunnel을 유지한다.

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 \
  kanu
```

다른 local process가 50053을 쓰고 있으면 양쪽에서 비어 있는 다른 port를 선택하고 server `--port`, tunnel 두 port, actor `--server-port`를 모두 같은 값으로 바꾼다.

## 7. laptop actor

실제 task config module은 `CONFIG_MAPPING`을 export하고 해당 config/environment에 `GRASP_PENALTY`가 있어야 한다. run 시작 전 `HIL_GRASP_PENALTY`를 task에서 승인된 값으로 설정하고 actor config와 server CLI가 같은지 확인한다. server는 offline/online data에서 `0` 또는 그 값만 허용한다. reward/termination은 Kanu classifier가 authoritative하다.

```bash
cd /home/laptop3/gello_software

PYTHONPATH=/home/laptop3/gello_software/serl_ur_infra \
python serl_ur_infra/scripts/run_remote_rlpd_actor.py \
  --exp-name EXPERIMENT_NAME \
  --ur-config-module PYTHON_MODULE_WITH_CONFIG_MAPPING \
  --network-type grpc \
  --server-host 127.0.0.1 \
  --server-port 50053 \
  --timeout-s 0.6 \
  --max-response-age-s 0.8 \
  --observation-schema-hash 625c6933a03fca4b9a306788c4846cc2cd10179d02dec86740c9570170d6d515 \
  --expected-model-id hil-serl-hybrid-sac-resnet10 \
  --expected-reward-authority server_classifier \
  --expected-reward-model-id cube-in-cup-checkpoint-150
```

실제 robot 전에 같은 command에 `--fake-env`를 붙여 wrapper/network contract를 확인한다. 단 task config의 fake environment 구현 여부는 별도로 확인한다.

위 pin은 server의 `GetServerInfo`를 episode 시작마다 새로 확인한다. policy model, reward authority, reward model 또는 observation schema가 다르면 첫 inference 전에 actor가 실패해야 정상이다. server의 `--reward-model-id`를 바꾸면 actor의 `--expected-reward-model-id`도 승인된 같은 값으로 바꾼다.

실제 robot 실행은 workspace, camera streams, GELLO intervention, reset/fault, action limits를 operator가 확인한 뒤 진행한다.

## 8. resume

### 8.1 같은 lineage의 최신 valid checkpoint

step 5,000에서 끝난 같은 root를 step 10,000까지 이어갈 때:

```bash
# 5절 command와 같은 immutable classifier/demo/config 옵션을 그대로 사용한다.
# 아래 세 옵션만 resume/target 관점에서 달라진다.
--checkpoint-root "$HIL_CHECKPOINT_ROOT" \
--resume-latest \
--target-learner-step 10000
```

fingerprint에는 classifier와 demo SHA도 포함되므로 artifact가 바뀌면 resume가 실패해야 정상이다.

`--resume-latest`와 explicit `--resume-path` 모두 `completion.json`이 있는 complete checkpoint만 허용한다. library에는 one-off legacy migration용 markerless load 옵션이 있지만 production CLI에는 노출되지 않는다.

### 8.2 explicit checkpoint에서 새 lineage root로 복구

더 높은 incomplete/damaged `checkpoint_*` entry가 있거나 다른 root의 checkpoint를 사용하려면 기존 entry를 삭제하지 않는다. 새 빈 output root를 만들고 explicit source를 지정한다.

```bash
export HIL_RESUME_SOURCE=/absolute/old/run/checkpoints/checkpoint_000000005000
export HIL_RECOVERY_ROOT=/absolute/persistent/path/hil-serl-runs/RECOVERY_RUN_ID/checkpoints

# 5절의 전체 immutable 옵션과 함께 사용한다.
--checkpoint-root "$HIL_RECOVERY_ROOT" \
--resume-path "$HIL_RESUME_SOURCE" \
--target-learner-step 10000
```

기존 incomplete directory도 조사 증거이므로 자동 삭제하지 않는다.

## 9. checkpoint와 log 확인

```bash
find "$HIL_CHECKPOINT_ROOT" -maxdepth 2 -type f \
  \( -name 'completion.json' -o -name 'metadata.json' \) -print

tail -n 50 "$HIL_JSONL_PATH"
df -h "$HIL_RUN_ROOT"
```

완료 checkpoint는 세 파일이 있어야 한다.

```text
agent_state.msgpack
metadata.json
completion.json
```

`completion.json`은 마지막 commit marker다. 파일을 손으로 수정하거나 marker를 복제하지 않는다.

## 10. fault와 shutdown

### 10.1 주요 stdout event

| event | 의미 | 조치 |
| --- | --- | --- |
| `rlpd_learner_server_ready` | server/worker 시작 | tunnel/actor 시작 가능 |
| `rlpd_learner_worker_fault` | learner가 fault, last-known-good policy serving 중 | actor를 안전 정지하고 JSONL/checkpoint/GPU 상태 수집 |
| `rlpd_learner_actor_service_fault` | inference/classifier/ingress service fault | process exit code 3 예상, actor fail-stop 확인 |
| `rlpd_learner_waiting_for_worker_shutdown` | current JAX update 종료 대기 | GPU/process 상태 확인, 무한 대기 시 escalation |
| `rlpd_learner_grpc_shutdown_timeout` | gRPC grace stop 실패 | exit code 5, port/process 확인 |

continuous run의 learner fault는 process를 즉시 종료하지 않고 last-known-good policy를 계속 제공한다. 현재 gRPC health에 degraded learner 상태가 표시되지 않으므로 stdout/JSONL event monitor가 필수다.

### 10.2 정상 종료

먼저 learner terminal에 한 번 `Ctrl-C`를 보내거나 process에 `SIGTERM`을 보낸다. gRPC를 닫고 non-daemon learner worker가 current update를 끝낼 때까지 기다린다.

JAX/native backend가 hang하면 join이 계속될 수 있다. 즉시 `SIGKILL`하지 말고 다음을 먼저 기록한다.

- PID와 command line
- `nvidia-smi` process/memory
- 마지막 stdout/JSONL event
- 마지막 complete checkpoint
- incomplete checkpoint directory 유무
- filesystem free space

강제 종료는 operator escalation 뒤 수행한다. replay RAM 내용은 복구되지 않으며 incomplete checkpoint는 삭제하지 않는다.

## 11. 현재 허용하지 않는 것

- fake demo를 live learner에 사용
- server를 `0.0.0.0`에 bind
- `--require-jax-backend cpu`로 Kanu production 실행
- checkpoint overwrite, rename 재사용, manual completion marker 생성
- markerless legacy checkpoint를 production CLI에서 resume
- automatic checkpoint pruning/delete
- fingerprint가 다른 classifier/demo/config로 resume
- raw checkpoint를 미래 GAP512 checkpoint로 자동 migration
- shared Kanu environment를 즉석 upgrade
- learner fault event를 무시한 채 robot actor 계속 운용

## 12. GAP512 전환 전 메모

현재 command는 raw-pixel `random_crop_pad4` contract다. 향후 no-augmentation GAP512가 승인되면 다음 항목이 바뀌므로 이 runbook도 함께 versioning해야 한다.

- demo schema와 offline conversion command
- replay observation keys/dtypes/shapes
- capacity별 host RAM 계산
- agent visual module과 optimizer target
- encoder fingerprint와 backend pin
- run contract revision
- raw classifier → feature ingress ordering
- checkpoint compatibility rule

그 전까지 fake dry-run은 가능하지만, 기본 raw capacity의 장시간 Kanu learner는 메모리 승인 없이 시작하지 않는다.
