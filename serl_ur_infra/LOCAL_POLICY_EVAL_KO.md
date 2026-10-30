# 로컬 GPU(laptop3) BC/FM 서빙 + latency 측정 (한국어)

> **이 문서는 [`POLICY_EVAL_KO.md`](POLICY_EVAL_KO.md)의 곁가지다.** 평가 스택의 전체 그림·핀
> 3종·평가 프로토콜은 그쪽이 정본이고, 여기서는 **정책 서버를 서버(junhyeong_ai) 대신 laptop3
> 자체 GPU에서 띄우는 경로**만 다룬다.
>
> **상태: 2026-08-06 벤치(G3)·E2E(G4) 실측 완료.** §5에 실측값이 채워져 있고, E2E 서빙 경로 실측은 BC infer ~29 ms(연속 스텝)/FM ~198 ms였다.
> 표를 채우는 것은 벤치·E2E 검증이다 — **추정치를 적어 넣지 말 것.**

---

## 1. 목적

같은 BC/FM **artifact를 그대로** laptop3의 RTX 3060 Laptop 6GB에서 서빙해, 서버
(RTX 5070 Ti) 대비 **추론 속도와 스텝 latency**가 얼마나 차이 나는지 잰다.

**왜 비교가 성립하나** — 서버 `il` 환경의 jax와 로컬 새 venv의 jax가 **둘 다 0.5.3**이고,
정책 로더·서버 스크립트·actor·gRPC 계약이 **완전히 같은 코드**다. 바뀌는 것은 **디바이스와
터널 유무**뿐이다.

| | 서버 경로 (기존) | 로컬 경로 (이 문서) |
|---|---|---|
| 정책 서버 | `junhyeong_ai:50054`(BC) / `:50055`(FM) | laptop3에서 **`127.0.0.1:50153` 직접 바인드** |
| 터널 | ssh(laptop3 50153 → 서버) | **없다** |
| GPU | RTX 5070 Ti (sm_120) | RTX 3060 Laptop 6GB (sm_86) |
| jax | 0.5.3 | 0.5.3 (`jax[cuda12]`) |
| T2 / T3 | production CLI 그대로 | **완전 동일 — 핀 3종도 같다** |

**아닌 것**: 학습이 아니고, reward classifier도 쓰지 않으며, production `:50053` learner와
무관하다(→ [`POLICY_EVAL_KO.md`](POLICY_EVAL_KO.md) §1). **production 파일은 한 줄도 고치지
않는다** — 이 경로는 전부 신규 파일이다.

---

## 2. 1회 세팅 (2단계)

모든 명령은 `cd /home/laptop3/gello_software/ros2_ur_ws` 기준이다. 둘 다 **멱등**이라
이미 끝난 상태면 검증만 하고 빠진다.

| # | 명령 | 무엇을 하나 | 성공 표식 |
|---|---|---|---|
| 1 | `./setup_local_policy_venv.sh` | 새 venv **`/home/laptop3/venvs/gello-local-policy`**(python3.10) 생성. `~/venvs/hilserl`의 패키지셋을 기반으로 하되 jax/jaxlib 줄만 **`jax[cuda12]==0.5.3`**으로 교체 | `jax.devices()`에 **CudaDevice**, matmul 스모크 통과, `import grpc, flax` OK |
| 2 | `./fetch_policy_artifacts.sh` | 서버에서 BC(**29M**) + FM(**27M**) artifact 디렉터리를 `scp`로 `~/hil-serl-data/diagnostics/`에 내려받고 sha 전수 검증 | manifest의 parameter sha256 전부 일치 + resnet sha 일치 |

**세팅에 관한 사실 몇 개**

- **기존 venv 2개(`gello-hil-actor`, `hilserl`)는 손대지 않는다.** pip 설치는 새 venv 안에서만
  일어난다.
- 새 venv는 디스크를 **~4.5G** 먹는다. 스크립트가 **여유 6G 미만이면 중단**한다
  (기준선: `/`에 28G free, 사용률 94 % — 실측).
- 서버 접근은 **읽기 전용**이다. ⚠️ **`/home/junhyeong/gello_software`(접미사 없음)는 다른
  사람의 작업 트리라 읽지도 쓰지도 않는다** — artifact는 `~/hil-serl-data/diagnostics/` 아래에만
  있다.
- resnet trunk 검증의 **소스 경로는 리포 안**이다:
  `third_party/hil-serl/examples/experiments/resnet10_params.pkl`
  (`~/.serl/`은 캐시일 뿐이다). 둘 다 로컬에 있고 sha `175745d4…`로 일치한다(실측).

**내려받는 artifact 두 개** (서버 런처의 기본값과 같은 디렉터리다 → model_id가 같고, 그래서
**핀 3종을 바꿀 필요가 없다**):

| 정책 | 디렉터리 이름 (`~/hil-serl-data/diagnostics/` 아래) |
|---|---|
| BC 20epoch | `bc_cube_in_cup_raw_0731_bce_group_holdout_20epoch_20260731_213638.bc-init` |
| FM 200epoch | `jax_fm_cube_in_cup_raw_0731_h16_euler8_200epoch_20260731_215835.fm-init` |

---

## 3. 순수 추론 벤치 (gRPC 없음, 로봇 없음)

정책 callable만 로드해 **모델 추론 시간만** 잰다. 서빙·네트워크·기록이 섞이지 않으므로
"모델이 이 GPU에서 몇 ms인가"에 대한 가장 깨끗한 답이다.

```bash
# GPU (새 venv)
cd /home/laptop3/gello_software
/home/laptop3/venvs/gello-local-policy/bin/python serl_ur_infra/scripts/bench_local_policy.py \
  --policy both --device gpu --out <출력 디렉터리>
```

```bash
# CPU 대조 (jax 0.5.3 CPU가 이미 있는 기존 venv — 여기서는 아무것도 설치하지 않는다)
cd /home/laptop3/gello_software
/home/laptop3/venvs/hilserl/bin/python serl_ur_infra/scripts/bench_local_policy.py \
  --policy both --device cpu --out <출력 디렉터리>
```

| 옵션 | 뜻 |
|---|---|
| `--policy bc\|fm\|both` | 무엇을 잴지. `both`면 BC/FM 둘 다 |
| `--iters` (기본 200) | 측정 반복 수 |
| `--warmup` (기본 10) | **jit 컴파일을 흡수하는 구간.** 여기 포함된 시간은 통계에서 빠진다 |
| `--device cpu\|gpu\|auto` | `import jax` **이전에** `JAX_PLATFORMS`로 강제한다 |
| `--obs-pkl <경로>` | 합성 관측 대신 **실관측**을 주입한다 (아래) |
| `--out <디렉터리>` | 결과 두 파일이 여기 떨어진다. **기본값은 `/home/laptop3/hil-serl-data/bench/bench_<UTC ts>`** — 즉 생략해도 어딘가에 쌓인다(§6 (c)) |

`--out`을 생략하면 매 실행이 **새 타임스탬프 디렉터리**를 만든다(두 벤치가 조용히 같은
디렉터리를 공유하지 못하게 하려는 것이고, 기존 `bench_results.jsonl`이 있으면 벤치는 아예
**거부**한다). 대신 정리 대상이 하나 더 생긴다 — §6 (c)의 `du` 줄이 `bench/*`를 포함하는
이유다.

**관측을 어디서 가져오나** — 기본은 계약 shape/dtype의 **합성 관측**이다(추론 latency는 픽셀
*내용*과 무관하다). 실관측으로 확인하고 싶으면 로컬 canonical demo를 준다:

```bash
--obs-pkl /home/laptop3/hil-serl-artifacts/demos/cube_in_cup_20260720_success_23takes.pkl
```

**무엇이 남나** — `--out` 디렉터리 아래 두 파일:

| 파일 | 내용 |
|---|---|
| `bench_results.jsonl` | 정책 1건당 1줄(정책 · `mode` · 디바이스 · iters · 통계 · **jax 버전과 디바이스 문자열** · `gpu_memory_used_mib` · `note`) |
| `bench_report.md` | 같은 내용의 사람이 읽는 표 |

⚠️ **정책당 줄은 하나다 — 모드를 두 개 재지 않는다.** BC는 **`served-argmax` 1회**, FM은
**`stochastic` 1회**다. 이유는 둘 다 같다: **두 정책 모두 wire의 `deterministic` 플래그를
의도적으로 무시한다.** BC의 서빙 샘플러는 `del deterministic` 뒤 `argmax=True`를 하드코딩하고
(`ur_env/learner/bc_init.py:451-456`, `deterministic_sample_action` — BC 평가는 언제나 mode
액션을 서빙해야 하므로 의도된 설계다), FM의 `FmServedPolicy.__call__`도 플래그를 버리고 매번
새 가우시안에서 ODE를 적분한다. 그래서 "deterministic / stochastic 두 모드"를 재면 **같은 trace를
두 번 재서 그 사이의 순수 측정 노이즈를 비교 가능한 차이인 양 표에 싣게 된다.** 벤치는 각 정책을
한 번만 재고 그 이유를 `note` 필드와 리포트 Notes에 남긴다.

⚠️ **이 수치는 서버 `infer_ms`의 하한이다(편향 방향이 하나다: 벤치 ≤ 서버).** 서버의 timed span
안에는 있고 벤치 span 밖에 있는 것이 셋이다 — `copy_observation`(관측 방어적 복사,
`ur_env/actor_network.py:1505`), `validate_action`/`validate_counter`(`1507-1511`), 그리고
**호출마다 json 1줄을 쓰고 `flush`하는** `InferenceLoggingPolicy`
(`ur_env/bc_inference_log.py:134-181`, 로컬 런처 양쪽에서 **기본 ON**이고 `--no-inference-log`로만
꺼진다). 즉 서버 `infer_ms`가 벤치보다 **조금 큰 것은 정상**이고 GPU가 느리다는 증거가 아니다 —
놀라야 하는 것은 그 반대 방향이다.

---

## 4. 로컬 서빙 E2E (T1만 바뀐다)

### 4.1 T1 — 로컬 정책 서버

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
HIL_STEP_TIMING=1 ./run_bc_server_local.sh     # FM은 ./run_fm_server_local.sh
```

**성공 표식:** `LOCAL_SERVER_RESULT=started` + `[bc-server] ready …`(FM은 `[fm-server] ready …`)
+ **프롬프트가 돌아오지 않는다.** `HIL_STEP_TIMING=1`을 줬다면 ready 줄에 `step_timing=1` 토큰이
붙는다.

- 서버 스크립트(`run_bc_policy_server.py` / `run_fm_policy_server.py`)를 **수정 없이 그대로**
  laptop3에서 실행한다. 다른 것은 `--host 127.0.0.1 --port 50153`으로 **actor가 원래 dialing하는
  주소에 직접 바인드**한다는 점뿐이다 — 터널이 필요 없어진 것이다.
- **foreground다.** `Ctrl-C`가 곧 종료이고 pid 파일도 ssh도 없다.
- 런처는 `--artifact-dir`를 **로컬 절대경로로 명시**한다(서버 스크립트의 기본값은 서버 경로다).
- 기록은 서버 레이아웃을 그대로 미러한다:
  `/home/laptop3/hil-serl-data/{bc_eval,fm_eval}/<이름>_local_<ts>/served/`.

### 4.2 T2 / T3 — **기존 CLI 그대로, 핀 3종 동일**

artifact가 같으니 서버가 광고하는 model_id도 같다. 즉 [`POLICY_EVAL_KO.md`](POLICY_EVAL_KO.md)
§3의 핀을 **그대로** 쓴다.

```bash
# T2
cd /home/laptop3/gello_software/ros2_ur_ws
./run_hil_hardware.sh
```

```bash
# T3 (BC)
cd /home/laptop3/gello_software/ros2_ur_ws
HIL_STEP_TIMING=1 EXPECTED_MODEL_ID=bc-cube-in-cup-raw0731-bcinit-v1 EXPECTED_REWARD_AUTHORITY=local EXPECTED_REWARD_MODEL_ID=operator-manual-success-v1 ./run_hil_session.sh --no-classifier-sidecar
```

FM이면 `EXPECTED_MODEL_ID=fm-cube-in-cup-raw0731-h16-euler8-v1`, 나머지 둘은 같다.

**로봇 없이 확인만 할 때(dry handshake)** — T1이 떠 있는 상태에서 다른 터미널에:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
EXPECTED_MODEL_ID=bc-cube-in-cup-raw0731-bcinit-v1 EXPECTED_REWARD_AUTHORITY=local EXPECTED_REWARD_MODEL_ID=operator-manual-success-v1 ./run_hil_actor.sh --fake-env --no-classifier-sidecar
```

핸드셰이크 오류 없이 `BeginEpisode`까지 가고 종료하면 통과다. 센서 토픽 WARN은 `--fake-env`에서
정상이다.

### 4.3 포트 소유자 검증 (**건너뛰지 말 것**)

```bash
ss -tlnp | grep 50153
```

리스너 PID가 **방금 띄운 `gello-local-policy` python**이어야 한다. `ssh`가 잡고 있으면 그것은
**서버를 재고 있는 것**이다 — §6 (a).

### 4.4 분석 — rsync가 없다

served 디렉터리가 **로컬**이므로 서버에서 받아오는 단계가 사라진다. 나머지는 서버 세션과 완전히
같다.

```bash
cd /home/laptop3/gello_software
/home/laptop3/venvs/gello-hil-actor/bin/python serl_ur_infra/scripts/analyze_bc_rollout.py \
  --served /home/laptop3/hil-serl-data/bc_eval/<이름>_local_<ts>/served \
  --actor-timing gello_logs/step_timing/actor_step_timing_<ts>.jsonl
```

서버 `timing.jsonl`은 `--served` 아래에서 자동으로 찾는다. 필드 스키마는
[`POLICY_EVAL_CODE_MAP_KO.md`](POLICY_EVAL_CODE_MAP_KO.md)가 정본이다.

🪤 **`wire_ms`의 의미가 바뀐다.** 서버 경로에서 그 값은 **ssh 터널 왕복**이었지만 로컬 경로에서는
**localhost 왕복**이다. 여전히 "서버측 분해와 액터측 분해의 차"로 계산되므로 T1과 T3를 **둘 다**
`HIL_STEP_TIMING=1`로 켜야 나온다.

---

## 5. 비교표 (실측이 들어갈 자리)

| 항목 | 서버 GPU (RTX 5070 Ti) | 로컬 GPU (RTX 3060 6GB) | 로컬 CPU |
|---|---|---|---|
| BC 추론 (jit 후) | **~6 ms** | **30.8 ms** (p50 31.3) | **12.6 ms** (p50 12.5) |
| FM 추론 (jit 후) | **~72 ms** (Euler 8-step) | **195.7 ms** (p50 194.2) | **144.6 ms** (p50 142.9) |
| jax | 0.5.3 | 0.5.3 | 0.5.3 |
| 측정 수단 | 세션 `inference.jsonl` | §3 벤치 + §4 E2E | §3 벤치 |
| `wire_ms` | ssh 터널 왕복 | localhost 왕복 (TBD) | — |

📌 로컬 두 열 provenance: **측정일 2026-08-06**, RTX 3060 Laptop 6GB, iters 200 / warmup 10, 무부하(GPU 상주 71 MiB), 합성 관측 — 원본 `~/hil-serl-data/bench/bench_20260806_083654`(GPU) · `bench_20260806_083914`(CPU).

서버 두 수치의 출처는 [`POLICY_EVAL_KO.md`](POLICY_EVAL_KO.md) §6이다. **표의 수치는 벤치·E2E가
실제로 돈 뒤에 채운 실측이다** — 이 문서에는 추정치를 적지 않는다.

🪤 **FM jit 교훈은 로컬에서도 그대로다.** eager면 Euler 8-step이 op 단위로 디스패치되어 한 액션에
수백 ms가 든다. 서버 스크립트는 ready를 광고하기 **전에** 컴파일을 흡수하고, 벤치는 `--warmup`이
같은 일을 한다 — **warmup 없는 수치를 인용하면 안 된다.**

---

## 6. 함정

| # | 함정 | 내용 |
|---|---|---|
| (a) | **원격 T1 터널이 살아 있는데 로컬을 잰 줄 안다** | 50153은 production/BC/FM/로컬이 **다 같이 쓰는 하나뿐인 로컬 포트**다. 원격 T1이 떠 있으면 로컬 런처는 바인드 실패를 "터널을 먼저 내려라"로 번역해 **거부**한다. 🪤 문제는 런처를 우회해 손으로 T2/T3만 띄웠을 때다 — artifact가 같아 **핀 3종이 전부 통과한 채 원격을 측정**하고, 아무 경고도 나오지 않는다. 런처의 포트 소유자 검증(§4.3)이 존재하는 이유가 이것이다 |
| (b) | **6GB GPU를 카메라 뷰어·GUI와 공유한다** | 세션 중에는 RealSense 뷰어와 GUI가 같은 GPU를 쓴다. 서버 스크립트가 `XLA_PYTHON_CLIENT_PREALLOCATE=false`를 setdefault하므로 공존은 되지만, **무부하 벤치 수치 ≠ 세션 중 수치**다. §5의 두 열(벤치 / E2E)을 섞어 인용하지 말 것. FM은 jit 컴파일 순간 메모리 스파이크가 있을 수 있어 기동 시 `nvidia-smi` 스냅샷을 남긴다 |
| (c) | **디스크가 빠듯하다** | 시작 시점 `/` 여유 **28G**(94 % 사용)에서 venv가 ~4.5G, artifact가 ~56M, 그 뒤로는 **에피소드 pickle이 계속 쌓인다.** 정리: `du -sh /home/laptop3/hil-serl-data/*_eval/* /home/laptop3/hil-serl-data/bench/* \| sort -h`로 보고 필요 없는 run 디렉터리를 통째로 `rm -rf`한다(서버 원본은 그대로다). ⚠️ **`bench/*`를 빠뜨리지 말 것** — §3 벤치는 `--out`을 생략하면 거기에 타임스탬프 디렉터리를 만드는데 `*_eval/*` glob은 그것을 **하나도 못 본다**. 개별 벤치는 작지만(두 파일) 반복 실행분이 조용히 남는다 |
| (d) | **기존 venv 2개는 건드리지 않는다** | gRPC·actor의 정본은 `gello-hil-actor`, CPU jax의 정본은 `hilserl`이다. 둘 중 어디에도 `jax[cuda12]`를 설치하지 말 것 — GPU용은 **`gello-local-policy` 하나뿐**이고, 그래서 되돌리기가 "디렉터리 하나 삭제"로 끝난다 |

---

## 7. 문서 지도

평가 스택 전체는 [`POLICY_EVAL_KO.md`](POLICY_EVAL_KO.md), 조작 절차는
[`BC_DEPLOY_KO.md`](BC_DEPLOY_KO.md) / [`FM_DEPLOY_KO.md`](FM_DEPLOY_KO.md), 코드·기록 스키마는
[`POLICY_EVAL_CODE_MAP_KO.md`](POLICY_EVAL_CODE_MAP_KO.md)가 정본이다 — 이 문서는 그중 **T1을
laptop3 GPU로 갈아 끼우는 델타**만 담는다.
