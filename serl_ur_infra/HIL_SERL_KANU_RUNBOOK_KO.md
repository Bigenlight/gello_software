# HIL-SERL learner Kanu 실행 runbook

> 상태: production-shaped CLI의 dry-run/초기 bounded-run 절차
>
> 기준일: **2026-07-29 KST** (하드웨어 브랜치 통합 머지 `3f199d4` + threshold 커밋 `1b02857` 이후 재검증)
>
> 검증 브랜치: `feat/gello-ur7e-humble-22.04`
>
> 구현 상태와 차단점: [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](./HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)
>
> reward threshold 근거와 classifier 실측: [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)

## 먼저 읽을 요약

- ⚠️ **이 문서의 모든 예시가 쓰던 classifier `e329986b...`는 폐기됐다.** 0724 도메인 success recall이 `0.0%`다. 실기에 물리면 reward가 영원히 0이다. 새 정본 경로와 orbax 디렉터리 제약은 1.3절에 있다. `--expected-classifier-sha256`/`--expected-checkpoint-sha256`는 **절대 생략하지 마라** — 코드 기본값이 아직 폐기된 SHA다.
- ⚠️ **reward threshold 숫자를 이 문서에서 베끼지 마라.** 이틀 사이에 0.85 → 0.5(`53d5cf6`) → **0.2**(`1b02857`)로 두 번 움직였다. 권위 있는 값은 코드 상수 하나뿐이다:
  `serl_ur_infra/ur_env/rlpd_receive_server.py`의 `DEFAULT_REWARD_THRESHOLD`.
  근거와 조건은 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)에 있다.
  아래 명령들은 값을 적어 넣지 않고 2.0절에서 코드로부터 읽어온 `$HIL_REWARD_THRESHOLD`를 쓴다(2026-07-29 확인 시점 값은 `0.2`였다).
  threshold는 **fingerprint에 포함된다**(`run_rlpd_learner_server.py`의 `run_contract["reward_classifier"]["threshold"]`).
  다른 threshold로 만든 checkpoint는 resume이 fail-closed로 거부된다 — 의도된 동작이다.
- Kanu learner server는 `127.0.0.1:50053`에만 bind하고 laptop은 SSH local forwarding으로 접속한다.
- Kanu에서는 JAX/JAXLIB 0.5.3 CUDA 환경을 사용하고 CLI에 `--require-jax-backend gpu`를 반드시 준다.
- fake canonical demo는 construction `--dry-run` 또는 bounded `--synthetic-e2e` acceptance에만 허용된다. production robot-data serving은 계속 자동 거부한다.
- 첫 순서는 fake demo 생성 → actual classifier/ResNet SHA 확인 → GPU dry-run → bounded laptop→Kanu synthetic E2E다.
- `--synthetic-e2e`는 target 1..10, fresh/restored step에서 exact +1, replay capacity 100 이상, exact 100 transition, all-synthetic demo, actor/run allowlist, bounded timeout을 강제한다. batch 256/training-starts 100/CTA 2를 유지하고 publish/checkpoint period만 1로 줄인다.
- production bounded/continuous run에는 real canonical robot demo가 필요하다. fake marker를 제거하거나 검사를 우회하지 않는다.
- production checkpoint는 5,000 learner step마다 약 305 MiB가 추가되며 삭제·덮어쓰기·pruning하지 않는다. synthetic E2E에서만 period 1이며 target을 1..10으로 제한한다. filesystem reserve 기본값은 2 GiB다.
- replay/intervention buffer는 RAM-only다. process restart와 checkpoint resume가 replay를 복구하지 않는다.
- external policy/classifier는 raw `uint8 (1,128,128,3)`를 사용하고, learner replay/demo는 frozen ResNet-10 `stop_gradient` 직후 camera당 `float32 (1,4,4,512)` current/next map을 저장한다. GAP은 없고 augmentation은 `none`이다.
- trainable `SpatialLearnedEmbeddings/Dropout/Dense256/LayerNorm/tanh`는 sample time에 적용된다. frozen trunk의 online/target exact invariant와 target repin을 유지한다.
- 기본 50k/10k ring camera tensor는 `7,864,320,000 B = 7.32421875 GiB`다. `--feature-memory-reserve-gib`를 포함한 startup RAM preflight가 fail-closed한다.
- Kanu GPU actual classifier/agent production dry-run과 feature CTA smoke는 통과했다. unified schema v2에서 laptop→SSH tunnel→Kanu exact 100 transition, 실제 CTA step 1, publish/checkpoint full-load roundtrip, fresh-process resume와 version 1 inference까지 통과했다. production robot E2E와 continuous learner는 아직 미검증이다.
- ⚠️ **2026-07-29 기준 Kanu에는 HIL 프로세스가 하나도 떠 있지 않다.** port 50053 미바인딩, GPU 유휴. 이 문서에 "server가 떠 있다"고 읽히는 문장이 있으면 그건 과거 run의 기록이지 현재 상태가 아니다. 매번 1.0절의 확인 명령으로 직접 본다.
- ⚠️ **`/home/laptop3/gello_software`는 Kanu에 존재하지 않는다.** 그 경로는 laptop3 전용이다. Kanu 쪽 실제 경로는 1.1절 표에 있다.

---

## 1. 사전 조건

이 PC의 실제 SSH alias는 `kanu`다. 조직 VPN, SSH config 또는 실제 hostname은 사용자가 관리하는 값이므로 문서에서 추측해 바꾸지 않는다.

```bash
ssh -G kanu | sed -n '1,40p'
ssh -T kanu true
```

두 번째 command가 성공하기 전에는 아래 Kanu command를 실행 가능하다고 간주하지 않는다.

### 1.0 지금 Kanu에서 뭐가 돌고 있는지부터 본다

이 절의 수치는 **관측값이 아니라 관측 방법**이다. 아래를 실행해서 나온 값이 사실이고, 문서에 적힌 과거 값은 기록일 뿐이다.

```bash
ssh kanu 'ss -ltnp 2>/dev/null | grep -E ":(50053|5594)\b" || echo "50053/5594 unbound"'
ssh kanu 'pgrep -af "run_rlpd_(learner|receive)_server|remote_reward_classifier_server" || echo "no HIL process"'
ssh kanu nvidia-smi
ssh kanu 'df -h ~/workspace'
```

> 📌 **스냅샷 (2026-07-29, 재실행 필요):** 위 네 명령 모두에서 HIL 관련 활동이 없었다 — port 50053 미바인딩, HIL 프로세스 0개, GPU 유휴, 디스크 95% 사용(여유 약 90 GB). PID·GPU 번호·포트 점유는 이 문서에 고정값으로 적지 않는다.
>
> 📌 **스냅샷 (2026-07-27, 재실행 필요):** RAM 251 GiB(available 152 GiB), RTX A4000 16 GB 8장, `/home/junhyeong/miniconda3/envs/il/bin/python`에서 JAX/JAXLIB 0.5.3, Flax 0.10.5, backend `gpu`, device 8개. 그날의 GPU dry-run/CTA smoke는 기존 dirty detached repository를 건드리지 않으려고 `/tmp/hil-feature-dryrun-BUJNWu`에 rsync/symlink로 만든 일회성 tree에서 수행했다. **그 tree는 `/tmp`이므로 지금 남아 있다고 가정하지 않는다.**

### 1.1 Kanu 경로 — 확인된 것과 확인되지 않은 것

이 표는 2026-07-29 공유 팩트 시트 기준이다. `~`는 `/home/junhyeong`이다.

| 경로 | 정체 | 상태 |
| --- | --- | --- |
| `~/workspace/youngwoong/hil-serl` | classifier 학습처. YWhero/hil-serl fork, branch `agent/cube-in-cup-classifier` @ `d753571` | 확인됨 |
| `~/workspace/youngwoong/dataset/cube_in_cup_all3/` | 학습 데이터 + **Jul-27 정본 checkpoint** | 확인됨 |
| `~/workspace/youngwoong/gello_software_remote_classifier` | ZMQ 뷰어 checkout + **Jul-24 폐기 checkpoint** | 확인됨 |
| `/tmp/gello-hil-rl-receive-server-v2` | receive-server milestone용 Kanu 전용 worktree | 확인됨 (`/tmp`, 휘발 가능) |
| `/home/junhyeong/miniconda3/envs/il/bin/python` | 공유 conda base Python | 확인됨 |
| `/home/laptop3/gello_software` | — | **Kanu에 없음.** laptop3 전용 경로다 |
| `$HIL_KANU_REPO` (아래) | production learner를 돌릴 `gello_software` checkout | **미확인 — 직접 확인하거나 새로 clone해야 한다** |

learner를 돌리려면 통합 브랜치가 checkout된 `gello_software`가 Kanu에 있어야 하는데, 위 표의 확인된 경로 중에는 그런 checkout이 없다. 즉 **`HIL_KANU_REPO`는 예시가 아니라 사용자가 만들어야 하는 값이다.** 아무 값이나 넣고 진행하지 말고 존재를 먼저 확인한다.

```bash
export HIL_KANU_REPO=/absolute/path/to/gello_software/on/kanu

cd "$HIL_KANU_REPO" || { echo "이 경로부터 만들어야 한다"; exit 1; }
git status --short --branch
git rev-parse HEAD          # 통합 브랜치 feat/gello-ur7e-humble-22.04 의 commit 이어야 한다
git submodule status third_party/hil-serl
```

dirty workspace나 예상하지 않은 commit에서 production run을 시작하지 않는다. `third_party/hil-serl` submodule도 초기화돼 있어야 한다.

```bash
cd "$HIL_KANU_REPO"
git submodule update --init --recursive third_party/hil-serl
```

> 🪤 test suite는 **녹색인지가 아니라 passed 개수**로 판단한다. laptop3 canonical checkout에서 완전한 기준선은 **333 passed / 11 skipped**다(2026-07-29 실행 확인). `PYTHONPATH`에서 `third_party/hil-serl/serl_launcher`를 빼면 조용히 **300 passed / 13 skipped**로 줄고, submodule을 초기화하지 않은 새 worktree에서는 296 / 17이 된다. 두 경우 모두 실패는 하나도 안 나오므로 "green"만 보면 못 잡는다. skip 사유 문자열("submodule is not checked out")도 그대로 믿지 않는다.
>
> laptop3 기준선 재현 명령(테스트는 laptop3에서 돌린다. Kanu에서 돌리는 절차가 아니다):
>
> ```bash
> set +u; source /opt/ros/humble/setup.bash; source ros2_ur_ws/install/setup.bash; set -u
> OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")
> env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
>   PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher:$OVERLAY" \
>   /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q -p no:cacheprovider serl_ur_infra/tests
> ```
>
> 시스템 `python3 -m pytest`로 돌리지 않는다. 시스템 python3의 grpcio는 1.30.2라 이 suite에서 오류 없이 100% CPU로 멈춘다.

### 1.2 Python/JAX GPU 환경

아래는 현재 검증 target이다. `runtime fail-closed` 항목은 production CLI가 시작 시 버전을 직접 비교한다(`ur_env/learner/agent.py`의 `validate_learner_dependencies()`). 나머지는 `requirements-learner.lock`에만 있는 값이며 **어떤 runtime validator도 강제하지 않는다** — 어긋나도 프로세스는 그대로 뜬다. 그래서 Kanu environment 준비 단계에서 반드시 눈으로 확인한다.

| package | lock/expected | 현재 enforcement | 2026-07-29 Kanu `il` 실측 |
| --- | --- | --- | --- |
| JAX / JAXLIB | 0.5.3 / 0.5.3 | runtime fail-closed | 0.5.3 (07-27 확인) |
| Flax | 0.10.5 | runtime fail-closed | 0.10.5 (07-27 확인) |
| Distrax | 0.1.5 | runtime fail-closed | — |
| TensorFlow Probability | 0.25.0 | runtime fail-closed | — |
| W&B | 0.26.0 | W&B enabled일 때 runtime fail-closed | — |
| NumPy | 1.26.4 | **없음** (lock 전용) | 🔴 **2.2.5 — 드리프트** |
| Orbax | 0.11.5 | **없음** (lock 전용) | 🔴 **0.11.12 — 드리프트** |
| grpcio | 1.74.0 | **없음** (lock 전용) | 🔴 **1.80.0 — 드리프트** |
| Optax | 0.2.4 | **없음** (lock 전용) | — |
| protobuf | 7.34.1, pure-Python compatibility mode | lock + implementation compatibility 검사; version 비교는 미구현 | — |

> 🪤 **드리프트는 자동으로 안 걸린다.** runtime fail-closed 대상은 jax / jaxlib / flax / distrax / tensorflow_probability (+ W&B enabled 시 wandb) **뿐**이다. 위 표에서 🔴 표시된 numpy·orbax·grpcio는 lock에서 벗어나 있는데도 CLI가 아무 말 없이 시작한다. orbax 드리프트는 특히 1.3절의 checkpoint 포맷 문제와 같은 축에 있으므로, run 전에 실제 값을 기록해 둔다.
>
> 위 실측 열은 **2026-07-29 공유 팩트 시트에서 가져온 값**이고 laptop3에서 코드로 재확인할 수 없다(이 PC에는 JAX/Flax가 설치돼 있지 않다). 아래 preflight 블록을 실제로 돌려 나온 값이 권위 있다.

`requirements-learner.lock`은 CPU-local 검증 환경용으로 `jaxlib==0.5.3`을 포함한다. 공유 Kanu conda environment에 그대로 설치하거나 upgrade하지 않는다. 별도의 CUDA-capable environment를 준비하고 그 interpreter 경로를 명시한다.

이전 receive-only overlay(`/tmp/gello-hil-rl-receive-overlay-v2`)는 protobuf 3.20.3과 당시 Kanu base JAX를 전제로 하므로 production learner 환경으로 **재사용하지 않는다.** protobuf 3.20.3 핀이 `wandb` import를 깨뜨린다.

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

#### 1.3.0 classifier checkpoint 한눈에 — 어느 것을, 무슨 포맷으로, 코드는 뭘 pin 중인가

| | **Jul-24 (폐기)** | **Jul-27 `cube_in_cup_all3` (정본)** |
| --- | --- | --- |
| Kanu 경로 | `~/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150` | `~/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt/checkpoint_150` |
| 포맷 | **단일 파일** flax msgpack, 약 87 MB | **orbax 디렉터리**, 약 43 MB (`_CHECKPOINT_METADATA`, `manifest.ocdbt`, `ocdbt.process_0/` …) |
| SHA-256 | `e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997` | **없음 — 단일 파일이 아니라서 정의되지 않는다** |
| 성능 | 0724 도메인 success recall **0.0%** (성공 1,123 프레임 중 0건, mean 확률 `0.007`). held-out 0720 test(무크롭)에서는 93.4% @0.85 | 0720 test split(n=166, 무크롭) 100.0% @0.5, FPR 0.0%. 0720 held-out pool(test 166 + val 100 = 266프레임, 6 take) 86.8% @0.5 / 83.1% @0.85 |
| 코드 기본값이 pin 중? | ✅ **예** — `run_rlpd_learner_server.py`의 `DEFAULT_CLASSIFIER_CHECKPOINT_SHA256`, `run_rlpd_receive_server.py`의 `DEFAULT_CHECKPOINT_SHA256`가 둘 다 위 SHA다 (2026-07-29 코드 확인) | ❌ 아니오 |
| 지금 CLI로 로드 가능? | 예 (다만 폐기됐으므로 실기에 쓰지 않는다) | ❌ **아니오** — 1.3.1의 차단점 |

> 정본의 두 성능 수치가 다른 건 **분할이 달라서**이고 둘 다 맞다 — 100.0%는 test split(166프레임)만, 86.8%는 val을 포함한 266프레임이며 차이는 전부 취약 take인 `take_21`이 val에 있기 때문이다. 보수적으로 보려면 **266프레임 쪽(86.8% @0.5)**을 쓴다. (참고: 이 둘 중 어느 것도 leave-one-take-out CV가 아니다. 진짜 CV는 `fold_take_01/02/03` 별도 체크포인트이고 @0.5에서 89.5 / 89.9 / 86.5로 오히려 더 높다.)
> 두 수치 모두 **크롭 없는 입력**에서 측정됐다. 실제 actor 경로는 `ur7e_env.get_im()`이 `IMAGE_CROP`을 적용하므로 이 숫자가 그대로 옮겨가지 않는다. 미해결 항목이며 근거는 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)에 있다.
> `take_21`은 **양쪽 checkpoint 모두** 약하다.

#### 1.3.1 reward classifier — 현재 정본 (orbax 디렉터리, 아직 로드 불가)

```text
/home/junhyeong/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt/checkpoint_150
```

2026-07-27 생성, 약 43 MB. 이 정본은 **단일 파일이 아니라 orbax checkpoint 디렉터리**다. 따라서 아래가 성립한다.

- **단일 SHA-256이 정의되지 않는다.** 1.3.2의 `sha256sum --check --strict` 형태 명령은 디렉터리에 대해 작동하지 않는다. 출력은 `sha256sum: <path>: Is a directory` / `<path>: FAILED open or read` / `WARNING: 1 listed file could not be read`이고 exit code는 `1`이다(2026-07-29 확인).
- 🪤 **차단점(코드로 확인함):** `ur_env/rlpd_receive_server.py`의 `checkpoint_sha256()`이 `os.path.isfile()`을 강제한다. 그리고 `RewardClassifierRuntime.__init__`은 **loader보다 먼저** 이 함수를 호출한다. 그래서 이 경로를 `--classifier-checkpoint`/`--checkpoint`에 주면 classifier를 열어보기도 전에 `FileNotFoundError`로 죽는다.
- 🔍 **아직 확인 못 한 것:** 그 SHA 게이트를 넘겼을 때 upstream loader가 orbax 디렉터리를 실제로 읽을 수 있는지는 laptop3에서 확인할 수 없다(이 PC에 JAX/Flax가 없다). `load_classifier_func()`는 `flax.training.checkpoints.restore_checkpoint()`를 호출하고, 같은 리포의 ZMQ 뷰어(`remote_reward_classifier_server.py`)는 같은 함수에 **orbax 디렉터리를 넘겨 쓰고 있다** — 즉 로드 자체는 될 가능성이 높지만 **이 CLI 경로에서 검증된 적은 없다.**

**정본으로 전환하려면 바뀌어야 하는 것(순서대로):**

1. 디렉터리용 digest 계약을 정한다 — 예: 파일별 SHA-256 manifest. `--expected-*-sha256`는 `_validated_sha256()`이 정확히 64자 hex를 요구하므로 "검사 끄기" 값이 없다. 계약을 바꾸지 않으면 우회할 방법 자체가 없다.
2. `checkpoint_sha256()`의 `os.path.isfile()` 강제를 그 계약에 맞게 확장한다.
3. `DEFAULT_CLASSIFIER_CHECKPOINT_SHA256` / `DEFAULT_CHECKPOINT_SHA256`의 폐기 SHA 기본값을 걷어낸다.
4. 그 다음에야 orbax 디렉터리 로드를 실제로 한 번 통과시켜 본다.

이 문서는 코드가 그렇게 되어 있다는 사실만 기록하며 **우회 방법을 제시하지 않는다.** 차단점 추적은 [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](./HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)에 있다.

현재 단계에서 가능한 것은 존재/구조 확인뿐이다.

```bash
export HIL_CLASSIFIER=/home/junhyeong/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt/checkpoint_150

test -d "$HIL_CLASSIFIER" || echo "정본은 디렉터리여야 한다"
ls -la "$HIL_CLASSIFIER"
du -sh "$HIL_CLASSIFIER"
```

> ℹ️ 같은 artifact를 쓰는 **다른 경로**가 하나 더 있다. ZMQ 뷰어(`run_remote_reward_classifier_server.sh` → `tcp://127.0.0.1:5594`)는 `classifier_ckpt/cube_in_cup_all3` 디렉터리를 그대로 받아서 이미 정본을 서빙한다. **그건 사람이 눈으로 보는 뷰어이고, gRPC RL 경로(port 50053)와 아무 호출 관계가 없다.** 뷰어가 정본으로 돈다고 해서 learner가 정본을 쓸 수 있게 되는 것이 아니다. 두 경로를 섞지 않는다.

#### 1.3.2 reward classifier — 폐기 (recall 0%)

아래는 2026-07-27까지의 모든 dry-run/E2E가 사용한 **구 checkpoint**다. 역사적 기록으로 남기며 **신규 run에는 사용하지 않는다.**

```text
경로:
/home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150

SHA-256:
e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997

상태:
폐기 — 0724 도메인 success recall 0.0% (성공 1,123 프레임 중 0건, mean 확률 0.007)
```

이 checkpoint로 통과한 과거 dry-run/E2E 결과는 파이프라인 배선 검증으로서 그대로 유효하다. 다만 그 검증 범위는 SHA/load/warm-up과 ingress 배선이었고 **분류 성능은 검증되지 않았다.** 근거는 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)에 있다.

> ⚠️ **expected-SHA 옵션을 절대 생략하지 마라.**
> `scripts/run_rlpd_learner_server.py`의 `DEFAULT_CLASSIFIER_CHECKPOINT_SHA256`와 `scripts/run_rlpd_receive_server.py`의 `DEFAULT_CHECKPOINT_SHA256`는 **2026-07-29 현재도 위 폐기된 SHA를 기본값으로 갖고 있다.** orbax 지원이 선행돼야 하므로 코드는 아직 고치지 않았다.
> 옵션을 생략한 채 구 checkpoint 파일을 주면 검사가 조용히 통과하고, 폐기된 classifier로 run이 시작된다. 항상 `--expected-classifier-sha256`(learner) 또는 `--expected-checkpoint-sha256`(receive server)를 명시하고, 그 값이 실제로 쓰려는 artifact의 것인지 확인한다.

4절 dry-run과 5절 bounded synthetic E2E는 fake demo로 배선만 확인하는 절차이고 과거 실측이 구 checkpoint로 수행됐다. 그 결과를 **재현**할 때만 아래 두 변수로 1.3.1의 값을 덮어쓴다. 6절 실기 run에는 사용하지 않는다.

```bash
# 폐기된 구 checkpoint — 4/5절 과거 acceptance 재현 전용
export HIL_CLASSIFIER=/home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150
export HIL_CLASSIFIER_SHA256=e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997

printf '%s  %s\n' "$HIL_CLASSIFIER_SHA256" "$HIL_CLASSIFIER" | sha256sum --check --strict -
```

#### 1.3.3 ResNet repository asset

expected SHA-256:

```text
175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b
```

검사 명령:

```bash
export HIL_RESNET_SOURCE="$HIL_KANU_REPO/third_party/hil-serl/examples/experiments/resnet10_params.pkl"
export HIL_RESNET_SHA256=175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b

printf '%s  %s\n' "$HIL_RESNET_SHA256" "$HIL_RESNET_SOURCE" | sha256sum --check --strict -
```

ResNet asset은 단일 파일이므로 위 검사가 그대로 유효하다. classifier와 달리 SHA 계약이 바뀌지 않았다.

upstream classifier는 `/home/junhyeong/.serl/resnet10_params.pkl`도 사용한다. 파일이 이미 있으면 같은 SHA인지 확인한다. 다르면 지우거나 덮어쓰지 말고 run을 중단한다.

```bash
if test -e /home/junhyeong/.serl/resnet10_params.pkl; then
  printf '%s  %s\n' "$HIL_RESNET_SHA256" /home/junhyeong/.serl/resnet10_params.pkl | sha256sum --check --strict -
fi
```

### 1.4 final unified observation schema

learner `8f242d8`의 Kanu E2E는 pre-hardware schema v1 interim run이다. hardware `6a0b127`은 unified merge `248255f`에 통합됐으며, 최종 검증은 다음 v2 schema/hash와 5.4의 final acceptance 결과를 기준으로 한다.

```text
schema id: hil-serl-ur-canonical-observation-v2
schema hash: 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
state group order: gripper_pose, tcp_force, tcp_pose, tcp_torque, tcp_vel
gripper_position index: 0
```

19-D flat layout(알파벳 그룹 순서, 2026-07-29 코드에서 재계산해 확인):

```text
[0]     gripper_pose   : gripper_position
[1:4]   tcp_force      : x, y, z
[4:10]  tcp_pose       : position x,y,z + euler x,y,z
[10:13] tcp_torque     : x, y, z
[13:19] tcp_vel        : linear x,y,z + angular x,y,z
```

shape `(1,19)`만 같다고 v1과 v2를 혼합하지 않는다. actor/server는 ordered schema hash를 pin하고 gripper는 `GRIPPER_POSITION_INDEX`/`gripper_position_from_state()`로만 읽는다. `state[0,-1]`은 v2에서 gripper가 아니라 TCP angular velocity z다.

> 위 hash를 손으로 옮겨 적지 않는다. 값은 `ur_env/observation_schema.py`가 schema document로부터 계산하며, 언제든 코드에서 다시 뽑을 수 있다:
>
> ```bash
> PYTHONPATH=serl_ur_infra python3 -c \
>   "from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH as h; print(h)"
> ```

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

checkpoint 하나의 payload는 약 305 MiB다. schema v1과 최종 v2 Kanu synthetic E2E에서 `320,100,609 B`를 관측했다. production은 기본 5,000-step 주기이고 pruning하지 않는다. 예상 checkpoint 수에 payload 총량과 `--checkpoint-reserve-gib`를 더해 disk를 잡는다. synthetic E2E는 period 1이므로 target 1..10 제한을 우회하지 않는다.

동일 checkpoint root에는 learner process 하나만 허용된다. `.learner-writer.lock`은 advisory lock metadata file이며 process 종료 후 파일 자체가 남아도 lock은 해제된다. 파일 존재 여부만 보고 임의 삭제하지 않는다.

### 2.0 reward threshold를 코드에서 읽어온다

threshold는 이틀 사이 0.85 → 0.5 → 0.2로 두 번 움직였다. 문서에 적힌 숫자를 베끼지 말고 **run 직전에 코드에서 읽는다.** 아래 값이 4·5·6절 모든 command에서 `--reward-threshold`로 들어간다.

```bash
export HIL_REWARD_THRESHOLD=$(
  cd "$HIL_KANU_REPO" && PYTHONPATH="$HIL_KANU_REPO/serl_ur_infra" "$HIL_KANU_PYTHON" -c \
    "from ur_env.rlpd_receive_server import DEFAULT_REWARD_THRESHOLD as t; print(t)"
)
echo "reward threshold = $HIL_REWARD_THRESHOLD"
```

이 import는 JAX를 끌어오지 않으므로 numpy만 있는 interpreter에서도 된다(2026-07-29 laptop3에서 확인, 값 `0.2`).

바꿔야 할 이유가 생기면 코드 상수를 먼저 바꾸고 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)에 근거를 남긴다. run 중간에 CLI 인자만 다른 값으로 주면 fingerprint가 달라져 그 lineage를 다시는 resume할 수 없다.

### 2.1 host RAM preflight

`--feature-memory-reserve-gib`(기본 2)는 checkpoint disk reserve와 다른 **host RAM reserve**다. CLI는 raw demo를 feature로 변환하거나 ring을 할당하기 전에 다음을 계산한다.

```text
required = feature replay/intervention fixed tensors
         + converted offline demo fixed tensors
         + feature-memory reserve
```

기본 50k/10k의 replay/intervention fixed tensor는 `7,875,840,000 B`(camera `7,864,320,000 B` 포함)다. offline demo는 transition당 current/next, cam1/cam2 map을 추가한다. Linux `MemAvailable`이 required보다 작으면 할당 전에 실패한다.

`--demo-extraction-batch-size`는 startup one-time trunk conversion의 temporary device/host batch를 제어한다. fake demo는 2, production default는 64를 사용하되 GPU memory가 부족하면 실험 기록을 남기고 줄인다. 이 값은 persistent pool size나 feature 의미를 바꾸지 않는다.

```bash
grep '^MemAvailable:' /proc/meminfo
free -h
```

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

이 파일은 다음 절의 `--dry-run` 또는 5절의 bounded `--synthetic-e2e`에만 사용한다. 일반 production learner option없이 사용하지 않는다.

현재 fake generator의 penalty 값은 `-0.02`로 고정돼 있다. 따라서 fake dry-run/synthetic E2E의 `HIL_GRASP_PENALTY`도 `-0.02`를 유지한다. 다른 task penalty의 synthetic artifact가 필요하면 generator를 명시적으로 확장하고 strict test를 추가해야 하며 marker나 pickle을 손으로 고치지 않는다.

## 4. Kanu GPU dry-run

dry-run은 실제 classifier, ResNet, dual raw/cached hybrid SAC agent, raw demo의 one-time frozen-trunk conversion, feature RAM preflight, production composition, fingerprint, JSONL/W&B offline을 준비하지만 gRPC port를 bind하거나 learner update를 실행하지 않는다.

아래 command의 `$HIL_CLASSIFIER`/`$HIL_CLASSIFIER_SHA256`는 1.3.2의 **폐기된 구 checkpoint** 값을 전제한다. 이 절이 검증하는 것은 construction 배선이지 분류 성능이 아니므로 재현 목적에는 그대로 쓸 수 있다. 새 정본으로 dry-run하려면 1.3.1의 orbax 제약을 먼저 해소해야 한다.

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
  --reward-threshold "$HIL_REWARD_THRESHOLD" \
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
  --feature-memory-reserve-gib 2 \
  --demo-extraction-batch-size 2 \
  --grasp-penalty "$HIL_GRASP_PENALTY" \
  --require-jax-backend gpu \
  --dry-run
```

성공 시 stdout에 `rlpd_learner_dry_run_passed`가 있어야 한다. JSONL과 W&B offline directory도 확인한다.

> 📌 **기록 (2026-07-27) — 재현용이지 재입력용이 아니다.**
> Kanu 기존 repository가 dirty detached였기 때문에 그 tree를 수정하지 않고 `/tmp/hil-feature-dryrun-BUJNWu`에 rsync/symlink로 일회성 tree를 만들어 수행했다. `CUDA_VISIBLE_DEVICES=0`(그날 비어 있던 GPU일 뿐, 고정값 아님), `il` environment, 128/32 capacity에서 actual classifier + agent production dry-run이 통과했다.
> 아래 fingerprint는 `execution_scope` field가 추가되기 **전**, threshold가 `0.85`이던 시점, 폐기된 Jul-24 classifier로 계산된 값이다. **지금 어떤 CLI에도 다시 입력하지 마라** — 현재 코드로 같은 명령을 돌려도 이 값은 나오지 않는다(정상). checkpoint resume identity로도 쓰지 않는다.

```text
fingerprint (2026-07-27 기록, 현재 무효): 8465e464b3f4eb638513eaa4ab3daea85a9435a9ddf47bdbf841a2c2f2aacce9
```

별도 실제 GPU feature CTA smoke도 통과했다(같은 날의 기록).

```text
backend: gpu
visible devices: 1
feature shape: [1,4,4,512]
gradient_step: 2
augmentation_function: null  # Python None, no augmentation
raw/cached deterministic action max abs: 0.00012614415027201176
online trunk invariant: passed
target trunk invariant after repin: passed
```

GPU fusion/materialization 경계 때문에 raw/cached action은 bitwise equality가 아니라 위 수치 차이로 동치했다. 이 결과는 construction + one-step CTA smoke이며 server bind, 50-step publish, 5,000-step checkpoint, continuous/robot E2E를 대체하지 않는다.

```bash
grep -n 'learner_process_ready' "$HIL_JSONL_PATH"
find "$HIL_WANDB_DIR" -maxdepth 2 -type d -name 'offline-run-*' -print
```

주의:

- dry-run은 port를 bind하지 않는다.
- `--dry-run`은 fake demo를 로드하지만 CTA update/checkpoint를 검증하지 않는다. 그 범위는 5절 `--synthetic-e2e`가 담당한다.
- fake demo를 그대로 두고 `--dry-run`만 제거하면 CLI가 의도적으로 실패해야 한다. live acceptance를 의도했다면 bounded `--synthetic-e2e` 계약을 모두 만족해야 한다.
- dry-run도 classifier와 agent를 GPU에 올리므로 RSS/GPU memory를 기록한다.
- dry-run은 replay나 update를 검증하지 않으므로 작은 128/32 capacity를 사용한다. 기본 50k/10k feature ring camera tensor는 7.32421875 GiB이므로 construction dry-run에서 할당할 이유가 없다. RSS와 함께 VMS도 기록한다.

```bash
nvidia-smi
```

## 5. bounded fake laptop→Kanu learning E2E

이 mode는 사용자가 지정한 현재 milestone acceptance를 위한 것이다. robot actor를 대체하는 production mode가 아니며 다음을 fail-closed로 강제한다.

- `--dry-run`과 `--synthetic-e2e`는 상호 배타적이다.
- `--target-learner-step`은 1..10의 bounded 값이어야 한다.
- target은 fresh/restored learner step에서 정확히 +1이어야 한다.
- `--replay-capacity`는 production `training_starts=100` 이상이어야 한다.
- `--synthetic-transition-count`는 production threshold와 같은 정확히 100이어야 하고 pass 시 insert count도 정확히 100이어야 한다.
- `--synthetic-actor-id`/`--synthetic-run-id`와 exact match하는 actor/run만 service가 허용한다.
- `--synthetic-timeout-s`는 1..1,800초의 유한 wall-clock deadline이다.
- offline demo 전체가 `synthetic_acceptance_only=true`여야 하며 real/synthetic 혼합을 거부한다.
- batch 256, online/demo 50:50, training starts 100, CTA 2, optimizer/model/discount은 production과 동일하다.
- publish/checkpoint period만 acceptance에서 1 step으로 줄인다.
- fingerprint `execution_scope`은 `synthetic_laptop_server_e2e_v1`이며 production robot scope와 resume 호환되지 않는다.
- advertised policy model ID는 `hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1`으로 production robot ID와 다르다.
- pass event는 server stop, worker join, process-stopped JSONL, logger close, full checkpoint load roundtrip/counter 검사, frozen online/target trunk invariant 검사까지 끝난 뒤에만 출력된다.

### 5.1 fresh step 1 server

dry-run 산출물과 섞지 않도록 새 run root를 쓴다. 아래 generator는 output을 overwrite하지 않으므로 완전히 새 `HIL_SYNTH_RUN_ID`를 지정한다.

4절과 마찬가지로 이 절의 `$HIL_CLASSIFIER`/`$HIL_CLASSIFIER_SHA256`는 1.3.2의 폐기된 구 checkpoint 값을 전제한다. bounded synthetic acceptance가 검증하는 것은 ingress/CTA/publish/checkpoint/resume 배선이지 reward 품질이 아니므로 재현 목적에는 그대로 쓴다.

```bash
export HIL_SYNTH_RUN_ID=UNIQUE_SYNTH_E2E_RUN_ID
export HIL_SYNTH_RUN_ROOT="/absolute/persistent/path/hil-serl-runs/$HIL_SYNTH_RUN_ID"
export HIL_SYNTH_CHECKPOINT_ROOT="$HIL_SYNTH_RUN_ROOT/checkpoints"
export HIL_SYNTH_WANDB_DIR="$HIL_SYNTH_RUN_ROOT/wandb"
export HIL_SYNTH_JSONL_PATH="$HIL_SYNTH_RUN_ROOT/logs/learner.jsonl"
export HIL_SYNTH_RESNET_CACHE="$HIL_SYNTH_RUN_ROOT/assets/resnet10_params.pkl"
export HIL_SYNTH_FAKE_DEMO="$HIL_SYNTH_RUN_ROOT/fake-canonical-demo.pkl"

mkdir -p "$HIL_SYNTH_RUN_ROOT/logs" \
  "$HIL_SYNTH_RUN_ROOT/wandb" \
  "$HIL_SYNTH_RUN_ROOT/assets"

cd "$HIL_KANU_REPO"
PYTHONPATH="$HIL_KANU_REPO/serl_ur_infra" \
"$HIL_KANU_PYTHON" \
  serl_ur_infra/scripts/generate_fake_canonical_demo.py \
  --output "$HIL_SYNTH_FAKE_DEMO"
```

Kanu terminal에서 다음 server를 시작한다. 100 transition이 들어오기 전까지 policy version 0을 serving하며 learner는 기다린다.

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
  --reward-threshold "$HIL_REWARD_THRESHOLD" \
  --reward-model-id cube-in-cup-checkpoint-150 \
  --demo-path "$HIL_SYNTH_FAKE_DEMO" \
  --checkpoint-root "$HIL_SYNTH_CHECKPOINT_ROOT" \
  --checkpoint-reserve-gib 2 \
  --jsonl-path "$HIL_SYNTH_JSONL_PATH" \
  --wandb-dir "$HIL_SYNTH_WANDB_DIR" \
  --wandb-mode offline \
  --wandb-project hil-serl \
  --run-name "$HIL_SYNTH_RUN_ID-step-1" \
  --hil-serl-root "$HIL_KANU_REPO/third_party/hil-serl" \
  --resnet-source "$HIL_RESNET_SOURCE" \
  --resnet-cache "$HIL_SYNTH_RESNET_CACHE" \
  --replay-capacity 128 \
  --intervention-capacity 32 \
  --feature-memory-reserve-gib 2 \
  --demo-extraction-batch-size 2 \
  --grasp-penalty -0.02 \
  --max-workers 4 \
  --max-message-bytes 16777216 \
  --require-jax-backend gpu \
  --target-learner-step 1 \
  --poll-interval 0.1 \
  --synthetic-actor-id fake-e2e-actor \
  --synthetic-run-id "$HIL_SYNTH_RUN_ID-fresh" \
  --synthetic-transition-count 100 \
  --synthetic-timeout-s 300 \
  --synthetic-e2e
```

### 5.2 laptop tunnel과 fake actor

laptop의 별도 terminal에서 tunnel을 유지한다.

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 \
  kanu
```

server에 `rlpd_learner_server_ready`가 출력된 뒤 laptop의 검증할 commit checkout에서 fake actor를 실행한다. 이 script는 robot env를 열지 않고 canonical raw observation을 gRPC로 보내는 acceptance tool이다. synthetic-only model ID `hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1`, observation schema, reward authority/model을 inference 전에 pin한다.

이 script는 `rclpy`나 카메라를 열지 않으므로 ROS overlay 없이 gRPC 클라이언트 환경만 있으면 된다. 시스템 `python3`(grpcio 1.30.2)로 실행하지 않는다 — `/tmp/gello-hil-grpc-venv`(grpcio 1.74.0 / numpy 1.26.4 / protobuf 3.20.3, `requirements-grpc.lock`과 일치, 2026-07-29 확인)나 `/home/laptop3/venvs/gello-hil-actor/bin/python`을 쓴다.

```bash
export HIL_LAPTOP_REPO=/home/laptop3/gello_software   # laptop3 전용 경로. Kanu에는 없다
export HIL_SYNTH_RUN_ID=THE_EXACT_SAME_SYNTH_E2E_RUN_ID_USED_ON_KANU

cd "$HIL_LAPTOP_REPO"
PYTHONPATH="$HIL_LAPTOP_REPO/serl_ur_infra" \
/tmp/gello-hil-grpc-venv/bin/python serl_ur_infra/scripts/run_fake_e2e_actor.py \
  --target 127.0.0.1:50053 \
  --actor-id fake-e2e-actor \
  --run-id "$HIL_SYNTH_RUN_ID-fresh" \
  --transition-count 100 \
  --expected-start-policy-version 0 \
  --expected-reward-model-id cube-in-cup-checkpoint-150 \
  --grasp-penalty -0.02 \
  --timeout-s 30 \
  --max-response-age-s 120
```

actor는 exactly 100 transition ACK/replay insert를 확인한다. server는 실제 batch 256 CTA를 수행한 뒤 다음을 만족해야 exit 0으로 종료한다.

```text
event: rlpd_learner_synthetic_e2e_passed
learner_step: 1
gradient_step: 2
policy_version: 1
checkpoint: checkpoint_000000000001
replay_size: 100
checkpoint_roundtrip_verified: true
```

### 5.3 fresh process resume to step 2

같은 Kanu run root, classifier, demo, feature contract, capacity와 execution scope를 유지한다. 5.1의 server command를 fresh process에서 다시 실행하되 다음을 바꾼다.

```text
--run-name "$HIL_SYNTH_RUN_ID-step-2"
--synthetic-run-id "$HIL_SYNTH_RUN_ID-resume"
--resume-latest
--target-learner-step 2
```

`--synthetic-e2e`는 계속 필수다. replay는 checkpoint에 없는 RAM-only이므로 새 process에서 100 transition을 다시 보내야 한다. laptop actor command에서 다음을 바꾸어 실행한다.

```text
--run-id "$HIL_SYNTH_RUN_ID-resume"
--expected-start-policy-version 1
```

resume server의 첫 action은 policy version 1이어야 하고 다음 최종 상태로 exit 0해야 한다.

```text
event: rlpd_learner_synthetic_e2e_passed
learner_step: 2
gradient_step: 4
policy_version: 2
checkpoint: checkpoint_000000000002
replay_size: 100
checkpoint_roundtrip_verified: true
```

### 5.4 실측 기록 (재입력 금지)

> 📌 **이 절 전체가 과거 run의 기록이다.** 아래 fingerprint / checkpoint 이름 / RTT / 시각은 그때 나온 값이지 지금 나와야 하는 값이 아니다.
> 두 결과 모두 **threshold `0.85`**, **폐기된 Jul-24 classifier**, 그리고 (v1 쪽은) 구 observation schema에서 나왔다.
> 지금 코드로 같은 절차를 돌리면 fingerprint는 반드시 달라진다 — 그게 정상이다. **어떤 값도 CLI에 다시 입력하지 마라.**
> 새 acceptance를 돌렸으면 이 절 밑에 새 날짜로 append하고 옛 블록은 그대로 둔다.

#### 5.4.1 2026-07-27 schema v1 interim (superseded)

Kanu GPU actual classifier/agent server와 laptop3 SSH tunnel/actor를 사용한 fresh + resume 두 process가 통과했다.

```text
fingerprint (기록, 현재 무효):
defda67b4463526a6aca4fb397327ff01b93ffd07b9250cc538a46b105bf96cb

fresh:
  transitions accepted: 100
  begin RTT max/mean: 91.158068 / 63.2926349 ms
  learner/gradient/policy: 1 / 2 / 1
  checkpoint roundtrip/trunk invariant: passed

resume fresh process:
  initial served policy: 1
  new transitions accepted: 100
  begin RTT max/mean: 95.694507 / 64.40843136 ms
  learner/gradient/policy: 2 / 4 / 2
  checkpoint roundtrip/trunk invariant: passed
```

두 JSONL의 learner update loss는 모두 finite였다. replay 100/offline demo 2에서 batch 256의 online/demo 128:128 샘플링이 확인됐고 `policy_published`, `checkpoint_saved`, `learner_process_stopped(exit_code=0)` event가 모두 존재했다. 최종 pass stdout event는 gRPC/worker/logger cleanup 후 checkpoint full-load roundtrip과 trunk invariant을 통과한 경우에만 `checkpoint_roundtrip_verified=true`로 출력됐다.

이 결과는 learner staging commit `8f242d8`의 pre-hardware canonical schema v1 interim 근거다. hardware commit `6a0b127`이 unified merge `248255f`에 병합되면서 schema ID는 v2, gripper index는 0, hash는 `3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903`이 됐다(2026-07-29 코드에서 재계산해 일치 확인). 따라서 위 fingerprint/checkpoint는 final unified branch에서 authoritative하지 않다.

#### 5.4.2 unified schema v2 acceptance, merge `248255f` (superseded)

> 📌 이 블록도 기록이다. 이후 하드웨어 브랜치 머지(`3f199d4`)와 threshold 커밋(`1b02857`)이 올라갔으므로 아래 fingerprint는 현재 HEAD에서 재현되지 않는다.

```text
final unified schema v2 acceptance (merge 248255f) — 기록, 현재 무효:
  fingerprint: fa1985378ad2729f466783e4f112d54022e14090430374a6531e4fb715440fcd
  serl default suite: 253 passed, 4 skipped, 6 warnings in 3.10s
  ur_gello suite: 436 passed in 7.35s
  actual opt-in: feature 2 passed / checkpoint 1 passed / local E2E 1 passed
  fresh sender: exact 100 inserts, RTT max/mean 372.820324 / 84.88630425 ms
  fresh counters: learner/gradient/policy 0/0/0 -> 1/2/1
  checkpoint: checkpoint_000000000001, 320100609 B, full-load roundtrip passed
  update timing ms: learner 83997.304 / critic 36496.960 / full 36822.142 / sample 1754.703
  resume smoke: fresh process restored 1/2/1, served finite 7D policy v1 action,
                gripper -1, RTT 133.65432 ms, clean stop exit_code=0
```

사용자 요청에 따라 final v2 resume process의 두 번째 SAC update는 생략했다. continued update/checkpoint는 local actual integration test와 위 schema v1 full resume run에서 이미 검증됐다.

## 6. real canonical demo가 준비된 뒤 bounded learner run

> ⚠️ **이 절은 실기 production 진입점이다. 구 checkpoint(`e329986b...`)로 실행하지 마라.**
>
> 그 checkpoint는 0724 도메인 success recall이 `0.0%`다(성공 1,123 프레임 중 0건, mean 확률 `0.007`). 그대로 실행하면 **로봇이 실제로 성공해도 classifier가 한 번도 threshold를 넘지 않으므로 reward가 영원히 0이고, HIL-SERL 학습이 시작조차 하지 않는다.** threshold를 더 낮춰도 해결되지 않는다.
>
> 아래 command의 `$HIL_CLASSIFIER`/`$HIL_CLASSIFIER_SHA256`가 1.3.2의 폐기 값으로 남아 있지 않은지 실행 직전에 반드시 확인한다.
>
> ```bash
> echo "$HIL_CLASSIFIER"
> echo "$HIL_CLASSIFIER_SHA256"   # e329986b... 이면 중단
> ```
>
> **현재 상태로는 이 절을 실행할 수 없다.** 1.3.1의 새 정본은 orbax 디렉터리이고 `checkpoint_sha256()`가 `os.path.isfile()`을 강제하므로 CLI가 SHA 단계에서(로드 시도 전에) 죽는다. 디렉터리 digest 계약과 `checkpoint_sha256()` 확장이 먼저 필요하다 — 1.3.1의 전환 조건 목록을 본다. 차단점 추적은 [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](./HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)에 있다.
>
> `--reward-threshold`에는 2.0절에서 코드로부터 읽은 `$HIL_REWARD_THRESHOLD`를 넣는다. 문서에서 숫자를 베끼지 않는다. threshold는 fingerprint에 포함되므로 **production run이 시작된 뒤에는 바꾸지 않는다** — 바꾸면 기존 lineage를 resume할 수 없다. 근거는 [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md)에 있다.
>
> ⚠️ 그리고 이 절의 전제인 **real canonical robot demo가 아직 존재하지 않는다.** 실기에서 팔을 움직인 것은 전부 `tests/run_real_hil.py`였고 그건 다른 코드 경로다. `run_remote_rlpd_actor.py`는 실기에서 한 번도 실행된 적이 없다(8절 참조). 게다가 `CubeInCupConfig.buffer_period`가 `0`이라 actor가 demo pickle을 아예 쓰지 않는다.

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
  --reward-threshold "$HIL_REWARD_THRESHOLD" \
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
  --feature-memory-reserve-gib 2 \
  --demo-extraction-batch-size 64 \
  --grasp-penalty "$HIL_GRASP_PENALTY" \
  --max-workers 4 \
  --max-message-bytes 16777216 \
  --require-jax-backend gpu \
  --target-learner-step 5000 \
  --poll-interval 0.1
```

server는 online replay가 100개에 도달할 때까지 policy version 0으로 inference/ingress를 제공하며 학습을 기다린다. stdout의 `rlpd_learner_server_ready`를 확인한 뒤 laptop tunnel과 actor를 시작한다.

기본 capacity의 feature camera tensor는 정확히 `7,864,320,000 B = 7.32421875 GiB`다. CLI는 이 ring에 offline demo feature tensor와 `--feature-memory-reserve-gib` 값을 더해 할당 전 `MemAvailable`을 검사한다. Python/JAX/XLA/classifier 오버헤드는 reserve 정책으로 별도 여유를 잡는다. capacity나 reserve를 바꾸면 실험 기록에 남긴다.

## 7. SSH loopback tunnel

Kanu server는 loopback에만 bind된다. laptop terminal에서 tunnel을 유지한다.

**원격 port는 항상 `50053`이고, 로컬 진입 port는 laptop3에서 비어 있는 아무 port여도 된다.** 두 값이 같을 필요가 없다.

```bash
# 양쪽 같은 번호를 쓰는 형태 (로컬 50053이 비어 있을 때)
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 \
  kanu
```

> 🪤 **laptop3의 실제 운용 관례는 로컬 `50153` → 원격 `50053`이다.** 2026-07-27에 로컬 `50053`이 다른 프로세스에 잡혀 있어서 옮겼고, 그 뒤로 운영 launcher가 그 값을 기본값으로 갖고 있다: `ros2_ur_ws/run_hil_actor.sh`의 `SERVER_PORT` 기본값이 `50153`이다(코드 확인). 이 runbook의 예시와 그 launcher를 섞어 쓰면 port가 어긋난다.
>
> ```bash
> # laptop3 관례 형태
> ssh -N -T -o ExitOnForwardFailure=yes \
>   -L 127.0.0.1:50153:127.0.0.1:50053 \
>   kanu
>
> ss -ltnp | grep 50153   # ssh가 127.0.0.1:50153 에서 LISTEN 중이어야 한다
> ```

어느 쪽을 쓰든 **로컬 진입 port와 actor `--server-port`가 같아야 하고**, tunnel의 원격 쪽과 server `--port`가 같아야 한다. 8절 command는 `--server-port 50053`으로 적혀 있으니 `50153` tunnel을 쓰면 그 값을 함께 바꾼다.

## 8. laptop robot actor

> ⚠️ **`run_remote_rlpd_actor.py`는 실기에서 아직 한 번도 실행되지 않았다** (2026-07-29). 지금까지 UR7e를 움직인 것은 전부 `serl_ur_infra/tests/run_real_hil.py`이고 그건 다른 코드 경로다. 아래는 코드에 대해 검증한 command이지 실기 검증된 절차가 아니다.

이 절은 **laptop3**에서 실행한다. `/home/laptop3/gello_software`는 laptop3의 canonical checkout이고 Kanu에는 없다.

실제 task config module은 `CONFIG_MAPPING`을 export하고 해당 config/environment에 `GRASP_PENALTY`가 있어야 한다. run 시작 전 `HIL_GRASP_PENALTY`를 task에서 승인된 값으로 설정하고 actor config와 server CLI가 같은지 확인한다. server는 offline/online data에서 `0` 또는 그 값만 허용한다. `CubeInCupEnvConfig.GRASP_PENALTY = -0.02`다. reward/termination은 Kanu classifier가 authoritative하다.

**interpreter와 ROS overlay.** actor는 `rclpy` / `ur_gello_bringup` / `cv2` / `pyrealsense2`를 import하므로 ROS 2 overlay를 source한 뒤 실행해야 한다. 그리고 시스템 `python3`의 grpcio는 **1.30.2**로, gRPC 계약이 요구하는 1.74.0이 아니다. 검증된 조합은 다음과 같다(2026-07-29 laptop3에서 확인).

```bash
set +u
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash
set -u
# /home/laptop3/venvs/gello-hil-actor: grpcio 1.74.0, system-site-packages=true
#   -> rclpy / cv2 / pyrealsense2 / gymnasium 모두 보임
```

```bash
cd /home/laptop3/gello_software

PYTHONPATH=/home/laptop3/gello_software/serl_ur_infra \
/home/laptop3/venvs/gello-hil-actor/bin/python \
  serl_ur_infra/scripts/run_remote_rlpd_actor.py \
  --exp-name cube_in_cup \
  --ur-config-module ur_experiments.mappings \
  --network-type grpc \
  --server-host 127.0.0.1 \
  --server-port 50053 \
  --timeout-s 0.6 \
  --max-response-age-s 0.8 \
  --observation-schema-hash 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903 \
  --expected-model-id hil-serl-hybrid-sac-resnet10-trunk-cache-v1 \
  --expected-reward-authority server_classifier \
  --expected-reward-model-id cube-in-cup-checkpoint-150 \
  --deadman topic
```

`--observation-schema-hash`는 1.4절 one-liner로 코드에서 다시 뽑아 대조한다.
`--expected-model-id`는 production scope의 값이다(`ur_env/learner/config.py`의 `FROZEN_TRUNK_MODEL_REVISION`). bounded synthetic server에 붙을 때는 `...-synthetic-e2e-v1`이며 5절의 `run_fake_e2e_actor.py`가 그 값을 자체적으로 pin하므로 이 command로 대체하지 않는다.
`--expected-reward-model-id`는 코드 상수가 아니라 **server `--reward-model-id`에 넣은 문자열**이다. 양쪽이 정확히 같아야 하고, classifier artifact를 바꾸면 양쪽 모두 새 이름으로 바꾼다.

### 8.1 arm 하기 전에

| 플래그 | 기본값 | 의미 |
| --- | --- | --- |
| `--arm` | **off** | 없으면 task config의 `DRY_RUN`이 유지되어 **팔도 그리퍼도 움직이지 않는다.** 붙이면 UR7e가 물리적으로 움직인다. arm 시 command topic publisher 수를 세어 이중 publisher면 거부한다(teleop bridge가 떠 있으면 실패한다) |
| `--deadman` | `topic` | `topic` = `/hil/deadman` 20 Hz 하트비트 + staleness 워치독. `spacebar`는 전역 pynput 리스너에 워치독이 없어 ON으로 붙어버릴 수 있다 — **실기에서는 `topic`만 쓴다** |
| `--mock-policy-noise S` | `0.0` | zero-action server 상대로 로봇을 움직여 개입 경로를 실증하는 용도. 6개 pose 채널에만 N(0,S) 노이즈, gripper 채널은 손대지 않는다. 이 run의 모든 전이에 `meta.policy_actions_synthetic=true`가 박힌다 — **demo나 정책 근거로 쓰면 안 된다** |
| `--fake-env` | off | robot/카메라 없이 wrapper/network 계약만 확인 |

첫 시도에서는 초기 SAC 정책의 action 크기가 확인되지 않았다는 점을 감안해 `--mock-policy-noise`를 작게 주거나 낮은 scale로 시작한다.

⚠️ 열려 있는 두 가지 위험:
- workspace box. `DefaultUR7eEnvConfig.ABS_POSE_LIMIT_LOW/HIGH`는 `zeros((6,))`이고 env가 이를 강제하지 않는다(`tests/run_real_hil.py`의 주석이 같은 사실을 적어 둔다). `CubeInCupEnvConfig`는 실측 박스를 채워 두었지만, 그 값이 실제로 클리핑에 쓰이는지는 arm 전에 확인한다. DRY RUN 300스텝 중 241스텝이 박스 밖이었고 최대 73.9 cm 이탈한 실측이 있다.
- 전역 ESC 리스너. `ur_env/envs/ur7e_env.py:197-208`이 pynput 전역 리스너를 건다. deadman과 별개로 **아무 창에서나 ESC를 누르면 에피소드가 끝난다.**

실제 robot 전에 같은 command에 `--fake-env`를 붙여 wrapper/network contract를 확인한다. 단 task config의 fake environment 구현 여부는 별도로 확인한다.

위 pin은 server의 `GetServerInfo`를 episode 시작마다 새로 확인한다. policy model, reward authority, reward model 또는 observation schema가 다르면 첫 inference 전에 actor가 실패해야 정상이다. server의 `--reward-model-id`를 바꾸면 actor의 `--expected-reward-model-id`도 승인된 같은 값으로 바꾼다.

실제 robot 실행은 workspace, camera streams, GELLO intervention, reset/fault, action limits를 operator가 확인한 뒤 진행한다.

## 9. production robot lineage resume

### 9.1 같은 lineage의 최신 valid checkpoint

step 5,000에서 끝난 같은 root를 step 10,000까지 이어갈 때:

```bash
# 6절 command와 같은 immutable classifier/demo/config 옵션을 그대로 사용한다.
# 아래 세 옵션만 resume/target 관점에서 달라진다.
--checkpoint-root "$HIL_CHECKPOINT_ROOT" \
--resume-latest \
--target-learner-step 10000
```

fingerprint에는 classifier와 demo SHA도 포함되므로 artifact가 바뀌면 resume가 실패해야 정상이다.

이 절은 `execution_scope=production_robot_data_v1`인 production lineage용이다. 5절 synthetic checkpoint는 `synthetic_laptop_server_e2e_v1`이므로 production option으로 resume할 수 없다. synthetic step 1→2 resume는 5.3의 bounded 절차만 사용한다.

`--resume-latest`와 explicit `--resume-path` 모두 `completion.json`이 있는 complete checkpoint만 허용한다. library에는 one-off legacy migration용 markerless load 옵션이 있지만 production CLI에는 노출되지 않는다.

### 9.2 explicit checkpoint에서 새 lineage root로 복구

더 높은 incomplete/damaged `checkpoint_*` entry가 있거나 다른 root의 checkpoint를 사용하려면 기존 entry를 삭제하지 않는다. 새 빈 output root를 만들고 explicit source를 지정한다.

```bash
export HIL_RESUME_SOURCE=/absolute/old/run/checkpoints/checkpoint_000000005000
export HIL_RECOVERY_ROOT=/absolute/persistent/path/hil-serl-runs/RECOVERY_RUN_ID/checkpoints

# 6절의 전체 immutable 옵션과 함께 사용한다.
--checkpoint-root "$HIL_RECOVERY_ROOT" \
--resume-path "$HIL_RESUME_SOURCE" \
--target-learner-step 10000
```

기존 incomplete directory도 조사 증거이므로 자동 삭제하지 않는다.

## 10. checkpoint와 log 확인

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

## 11. fault와 shutdown

### 11.1 주요 stdout event

| event | 의미 | 조치 |
| --- | --- | --- |
| `rlpd_learner_server_ready` | server/worker 시작 | tunnel/actor 시작 가능 |
| `rlpd_learner_synthetic_e2e_passed` | bounded synthetic target/counters/checkpoint 검증 완료 | exit code 0, JSONL/checkpoint 수집 |
| `rlpd_learner_synthetic_e2e_failed` | synthetic target/counter/checkpoint 계약 불충족 | exit code 6 이상, robot에 연결하지 말고 산출물 수집 |
| `rlpd_learner_worker_fault` | learner가 fault, last-known-good policy serving 중 | actor를 안전 정지하고 JSONL/checkpoint/GPU 상태 수집 |
| `rlpd_learner_actor_service_fault` | inference/classifier/ingress service fault | process exit code 3 예상, actor fail-stop 확인 |
| `rlpd_learner_waiting_for_worker_shutdown` | current JAX update 종료 대기 | GPU/process 상태 확인, 무한 대기 시 escalation |
| `rlpd_learner_grpc_shutdown_timeout` | gRPC grace stop 실패 | exit code 5, port/process 확인 |

continuous run의 learner fault는 process를 즉시 종료하지 않고 last-known-good policy를 계속 제공한다. 현재 gRPC health에 degraded learner 상태가 표시되지 않으므로 stdout/JSONL event monitor가 필수다.

### 11.2 정상 종료

먼저 learner terminal에 한 번 `Ctrl-C`를 보내거나 process에 `SIGTERM`을 보낸다. gRPC를 닫고 non-daemon learner worker가 current update를 끝낼 때까지 기다린다.

JAX/native backend가 hang하면 join이 계속될 수 있다. 즉시 `SIGKILL`하지 말고 다음을 먼저 기록한다.

- PID와 command line
- `nvidia-smi` process/memory
- 마지막 stdout/JSONL event
- 마지막 complete checkpoint
- incomplete checkpoint directory 유무
- filesystem free space

강제 종료는 operator escalation 뒤 수행한다. replay RAM 내용은 복구되지 않으며 incomplete checkpoint는 삭제하지 않는다.

## 12. 현재 허용하지 않는 것

- 폐기된 classifier `e329986b...`로 6절 실기 run 실행
- `--expected-classifier-sha256`/`--expected-checkpoint-sha256` 생략(코드 기본값이 폐기된 SHA다)
- production run 시작 이후 `--reward-threshold` 변경
- 이 문서에 적힌 threshold 숫자를 command에 그대로 베끼기 — 2.0절에서 코드로부터 읽는다
- 5.4절·4절의 과거 fingerprint/checkpoint 값을 CLI에 재입력 (전부 옛 threshold·폐기 classifier 기준의 기록이다)
- orbax 정본 디렉터리를 `--classifier-checkpoint`/`--checkpoint`에 그대로 지정 (1.3.1의 전환 조건이 먼저다)
- fake demo를 bounded `--synthetic-e2e` 외의 live learner에 사용
- `--synthetic-e2e`에 real/synthetic 혼합 demo, restored step +1이 아닌 target, target 0/음수/11 이상, replay capacity 100 미만, transition count 100 이외, timeout 1..1,800초 범위 밖을 사용
- allowlist와 다른 actor/run ID로 synthetic server에 접속하거나 production model ID를 synthetic actor에 pin
- synthetic E2E checkpoint를 production robot scope로 resume하거나 그 반대로 사용
- server를 `0.0.0.0`에 bind
- `--require-jax-backend cpu`로 Kanu production 실행
- checkpoint overwrite, rename 재사용, manual completion marker 생성
- markerless legacy checkpoint를 production CLI에서 resume
- automatic checkpoint pruning/delete
- fingerprint가 다른 classifier/demo/config로 resume
- 기존 raw/random-crop checkpoint를 frozen-trunk/no-aug checkpoint로 자동 migration
- shared Kanu environment를 즉석 upgrade
- learner fault event를 무시한 채 robot actor 계속 운용

## 13. frozen-trunk feature 불변 계약

현재 command의 run contract는 `frozen_trunk_feature_hybrid_sac_v1`이다.

```text
external policy/classifier:
  state float32 (1,19)
  cam1/cam2 uint8 (1,128,128,3)

learner replay/demo current and next:
  state float32 (1,19)
  cam1/cam2 float32 (1,4,4,512)

feature cut:
  pretrained_resnet10.stop_gradient
pooling:
  none
augmentation:
  none
model id:
  production: hil-serl-hybrid-sac-resnet10-trunk-cache-v1
  synthetic:  hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1
```

`SpatialLearnedEmbeddings(8)`, `Dropout(0.1)`, `Dense(256)`, `LayerNorm`, `tanh`는 feature에 포함되지 않으며 sample time의 현재 trainable weight로 실행한다. replay/demo에 raw image, GAP512, 또는 trainable 256-D head 출력을 저장하지 않는다.

offline raw demo는 startup에 `--demo-extraction-batch-size`로 verified trunk를 한 번만 통과하고, online transition은 classifier finalize 후 current/next를 encode한다. long-lived pool/ring은 raw image array를 보유하지 않는다.

fingerprint는 feature revision, shape/dtype/cut point, augmentation, ResNet SHA, classifier/demo/config과 execution scope을 묶는다. production robot scope는 `production_robot_data_v1`, bounded synthetic scope는 `synthetic_laptop_server_e2e_v1`이다. 두 scope, 기존 raw/random-crop lineage, 다른 encoder weight의 checkpoint는 서로 resume 불가다. online trunk은 verified initial tree와 exact equal이어야 하고 target trunk은 CTA candidate마다 verified tree로 repin한 뒤 검사한다.
