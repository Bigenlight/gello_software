# 09 — HIL actor 기동 런북 (Stage A: fake-env / Stage B: 실센서 DRY_RUN)

> ## 🛑 지금 이 문서를 읽는 사람이 먼저 알아야 할 것 (2026-07-29)
>
> 1. **2026-07-29 actor entrypoint의 첫 실제 production-model run을 완료했다.**
>    실제 replay 201개, intervention 153개, learner 102 step, policy version 2까지 갔다.
>    첫 publish 경계에서 `Step RPC DEADLINE_EXCEEDED`로 종료됐으므로 continuous run은
>    아직 PARTIAL이다.
> 2. **actor entrypoint에 CLI 플래그 3개가 생겼다** (§1.2에 정리):
>    `--arm`, `--deadman {topic,spacebar}`(기본 `topic`), `--mock-policy-noise SIGMA`.
>    §4의 "DRY_RUN 해제는 이 문서의 범위가 아니다"는 **더 이상 맞지 않는다.**
>    (2026-07-29에 분류기 sidecar 플래그 4개가 더 붙었다 — §1.2.1.)
> 3. **`clip_safety_box`는 구현돼 있다.** §4.1의 "아직 구현되어 있지 않다"는 낡았다 —
>    그 자리에 지금의 사실을 다시 적어 뒀다.
> 4. **🟢 분류기 크롭 불일치(`08` G15)는 2026-07-29에 닫혔다 — 코드에서만.**
>    ~~머지가 크롭 불일치를 들여왔고 Stage B 전에 처리해야 한다~~는 더 이상 맞지 않는다.
>    actor가 분류기에게 **자기 몫의 무크롭 128×128 JPEG(sidecar)**를 약 2 Hz로 따로 붙여
>    보내고, 정책은 실측 `IMAGE_CROP`을 그대로 쓴다 (`05` §3.2).
>    이 경로는 첫 실제 actor run에서 production server로 전송됐다. 다만 현재 GUI/JSONL에는
>    per-transition classifier probability가 보이지 않으므로 **online verdict 정합 검증**은
>    아직 남아 있다. `DRY_RUN`이 팔만 막고 보상/종단은 막지 않는다는 사실도 그대로다.
> 5. **actor에 sidecar 플래그 4개, 서버에 `--reward-model-id` 기본값이 생겼다.**
>    `EXPECTED_REWARD_MODEL_ID`가 **`cube-in-cup-all3-ckpt150+sidecar-v1`**로 바뀌었고,
>    옛 값 `cube-in-cup-checkpoint-150`은 **핸드셰이크에서 거부된다** (§1.1, §2.2).
> 6. 첫 E2E learner는 문서 작성 시점 PID `159159`, port 50053에서 아직 살아 있었다.
>    다음 세션에는 이 값을 믿지 말고 `pgrep`/`ss`로 확인한다. 살아 있으면 duplicate를 띄우지 않는다.
>
> 현재 상태는 [`HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](../../serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md)를 볼 것.

> 이 문서는 **로봇 랩톱(`laptop3`)에서 HIL-SERL actor 프로세스를 띄우는 절차**만 다룬다.
> 학습 알고리즘, 보상 분류기 학습, 정책 성능은 범위 밖이다.
> 정상 운용은 `run_hil_server.sh` / `run_hil_hardware.sh` / `run_hil_session.sh` 세 wrapper로
> 통일한다. actor만 진단할 때는 `run_hil_actor.sh`를 쓰되 **손으로 python 명령을 치지 않는다** —
> 2026-07-27 실기에서 그것 때문에만 4번 실패했다 (§0).

```bash
# 이 문서의 모든 명령이 쓰는 변수
export WT=/home/laptop3/gello_software
```

> `run_hil_actor.sh`는 **자기 파일 위치로부터 repo root를 계산한다.** 통합 checkout이든
> 워크트리든 스크립트가 들어 있는 트리를 그대로 쓴다. 경로를 하드코딩할 필요가 없다.

---

## 0. 왜 이 런북과 래퍼가 생겼나 — 📌 2026-07-27의 실패 4건

📌 2026-07-27 실기 세션에서 actor를 띄우는 데 **코드가 아니라 실행 명령 때문에만** 4번 실패했다.
(아래 표는 그 세션의 기록이다. 래퍼가 생긴 뒤로는 재발하지 않았다.)

| # | 증상 | 진짜 원인 | 래퍼가 막는 방법 |
|---|---|---|---|
| 1 | 에러도 로그도 없이 **CPU 100%로 영구 정지** | `python3`(시스템 인터프리터)로 실행 → 시스템 grpcio `1.30.2`가 손상돼 있음 | 항상 `/home/laptop3/venvs/gello-hil-actor/bin/python`을 절대경로로 `exec`. preflight [2]가 grpcio 버전을 보고 1.30.2면 거부, [3]이 `cygrpc.CompletionQueue()`를 **타임아웃 건 서브프로세스**로 실제 호출해 코어 생존을 확인 |
| 2 | `ModuleNotFoundError: ur_gello_bringup` | `PYTHONPATH=serl_ur_infra:...` 로 **덮어써서** ROS 오버레이 경로가 날아감 | 항상 `"...${PYTHONPATH:+:$PYTHONPATH}"` 로 **이어붙임**. preflight [4]가 오버레이 경로가 PYTHONPATH에 남아 있는지 직접 확인하고, [5]가 네 모듈의 **실제 해석 경로**를 출력 |
| 3 | 오버레이 없음 | `source install/setup.bash` 누락 | 래퍼가 `/opt/ros/humble/setup.bash` → `<repo>/ros2_ur_ws/install/setup.bash` 순서로 항상 소스 |
| 4 | 상대경로 실패 | `cd` 안 함 | 래퍼가 repo root로 `cd` 하고 모든 경로를 절대경로로 만듦 |

추가로 preflight [5]는 그때까지 드러나지 않았던 함정 하나를 더 막는다:
`serl-ur-infra`가 **다른 checkout에 editable 설치**돼 있으면 `ur_env`는 import되지만
지금 트리의 코드가 아니고, `ur_experiments`는 아예 없어서 `--ur-config-module`이 깨진다.
preflight는 네 모듈이 **이 트리 안에서** 해석됐는지 경로로 검증한다.

---

## 1. `run_hil_actor.sh` — 사용법

### 1.0 운영용 3-CLI quick start

개별 terminal T0~T6 명령은 장애 진단의 정본으로 아래에 보존한다. 평상시 운용은 다음 세
wrapper로 묶는다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws

# Terminal A: Kanu learner + SSH tunnel
./run_hil_server.sh

# Terminal B: UR7e driver + gripper + GELLO reader
./run_hil_hardware.sh

# Terminal C: cameras + HIL GUI + preposition/preflight + armed actor
./run_hil_session.sh
```

- A는 exact production learner를 재사용하거나 없을 때만 RAM/artifact gate 뒤 새로 띄우고,
  local `50153 -> Kanu 50053` tunnel을 유지한다. `Ctrl-C`는 tunnel만 닫는다.
- B는 UR7e/Robotiq/GELLO만 소유한다. 충돌·연결 해제 뒤 C를 내리고 B의 cleanup 완료 후 B만
  다시 띄울 수 있다.
- C는 카메라/GUI/preposition/armed preflight/actor를 순서대로 실행한다. 현재는 시작 전 GUI
  `ENGAGED` 확인이 필요하고, actor 기동 후 `DISENGAGE`해야 policy 제어가 시작된다.

읽기 전용/무접촉 진단은 다음과 같다.

```bash
./run_hil_server.sh --check
./run_hil_hardware.sh --dry-run
./run_hil_session.sh --no-arm --plan
```

매-step classifier 평가는 지연 진단이 필요할 때만
`./run_hil_session.sh --classifier-sidecar-interval 1`로 켠다. 기본 cadence는 5 step이다.

세 wrapper의 옵션과 현재 제약은
[`HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](../../serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md)
§9를 따른다. 아래 개별 명령은 wrapper 고장이나 하드웨어 재연결을 진단할 때 사용한다.

```bash
cd $WT/ros2_ur_ws

./run_hil_actor.sh --help              # 상단 주석(용도/안전/중단) 출력
./run_hil_actor.sh --dry-preflight     # 점검만. actor를 기동하지 않는다 (안전)
./run_hil_actor.sh --dry-preflight --arm  # marker/live pose까지 arm 준비를 읽기 전용 검증
./run_hil_actor.sh --fake-env          # Stage A
./run_hil_actor.sh                     # Stage B
```

* `--dry-preflight`와 `--help`만 래퍼가 **소비**한다. **나머지 인자는 전부 그대로 통과**한다
  (`--arm`, `--deadman`, `--mock-policy-noise`, `--save-video`, `--actor-id`,
  `--checkpoint-path`, …). `--fake-env`는 예외적으로 **관찰하고 통과시킨다** —
  센서 점검을 WARN으로 강등하는 데 쓰고, actor에게도 넘긴다 (`run_hil_actor.sh:113-114`).
* 래퍼가 먼저 넣는 기본 인자보다 **뒤에** 붙으므로, 같은 옵션을 다시 주면 사용자 값이 이긴다
  (argparse는 뒤가 이김).
* no-arm/fake는 `exec`로 바로 actor를 실행한다. 실기 `--arm`은 래퍼가 부모로 남아
  Ctrl-C를 actor에 정확히 한 번 전달하고, actor publisher 종료를 확인한 뒤
  `forward_position_controller`에서 `scaled_joint_trajectory_controller`로 자동 복귀한다.

### 1.1 환경변수 오버라이드

| 변수 | 기본값 | 비고 |
|---|---|---|
| `ACTOR_VENV` | `/home/laptop3/venvs/gello-hil-actor` | `--system-site-packages` venv (rclpy 상속). grpcio 1.74.0 |
| `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `50153` | 로컬 터널 입구. 원격은 Kanu `50053` |
| `EXP_NAME` | `cube_in_cup` | `ur_experiments/mappings.py`의 `CONFIG_MAPPING` 키 |
| `UR_CONFIG_MODULE` | `ur_experiments.mappings` | |
| `OBS_SCHEMA_HASH` | `3459098d…0352903` | 양쪽이 같아야 한다 (§2.1) |
| `EXPECTED_MODEL_ID` | `hil-serl-hybrid-sac-resnet10-trunk-cache-v1` | **서버 종류에 따라 반드시 바꾼다** (§2.3) |
| `EXPECTED_REWARD_AUTHORITY` | `server_classifier` | |
| `EXPECTED_REWARD_MODEL_ID` | `cube-in-cup-all3-ckpt150+sidecar-v1` | 서버 `--reward-model-id`와 같아야 함. **id가 체크포인트 + 입력 계약(sidecar)을 둘 다 담는다** — 어긋나면 핸드셰이크에서 거부된다. *(이전 값 `cube-in-cup-checkpoint-150`은 이제 거부된다)* |
| `TIMEOUT_S` / `MAX_RESPONSE_AGE_S` | `0.6` / `0.8` | 프로덕션 예산. 늘리지 말 것 (`05` §5) |
| `HZ_TIMEOUT_S` | `6` | 토픽당 정상 종료형 liveness/rate probe 최대 대기 시간. 5개 fresh·advancing 샘플과 최소 Hz를 검사 |
| `HIL_PREPOSITION_MARKER` | `$XDG_RUNTIME_DIR/hil-preposition.ready` | `run_hil_preposition.sh`와 actor가 공유하는 0600 proof. 보통 직접 지정하지 않는다 |
| `HIL_PREPOSITION_MARKER_MAX_AGE_S` | `900` | marker 최대 수명. 1~3600초만 허용 |
| `SKIP_ROS_CHECKS` | (미설정) | `1`이면 [7][8][9] 건너뜀. `--arm`과 같이 쓰면 **즉시 FAIL** |
| `ROS_SETUP` | `/opt/ros/humble/setup.bash` | |

📌 2026-07-29 확인: 위 기본값은 전부 `ros2_ur_ws/run_hil_actor.sh:65-93`과 일치한다.

### 1.2 actor CLI 플래그 (래퍼가 통과시킨다)

| 플래그 | 기본 | 무엇을 하나 |
|---|---|---|
| `--arm` | off | 태스크 config의 `DRY_RUN`을 끄기 전에 shell preflight가 controller 쌍과 퍼블리셔 0을 확인한다. STJC active/FPC inactive이면 유효한 preposition marker + 현재 RESET 자세(≤0.10 rad)를 재검증하고 strict switch한다. 이미 FPC active/STJC inactive이면 marker는 생략하지만 **live RESET 자세는 동일하게 요구**한다. 그 외는 actor 미기동. ⚠️ **UR7e가 물리적으로 움직인다.** |
| `--deadman {topic,spacebar}` | **`topic`** | 개입 데드맨 소스. 최신 `engaged=0`은 정책 복귀, 첫 heartbeat 뒤 0.5 s 단절은 액터 fail-stop. `spacebar`는 **전역 pynput + 워치독 없음**이라 실기 금지 |
| `--mock-policy-noise SIGMA` | `0.0` (비활성) | 서버 액션을 σ의 zero-mean 가우시안으로 **대체**한다 (`:72-84`). zero-action 서버 상대로 로봇을 움직여 개입 경로를 실증하는 용도. 교란된 액션이 **실행되는 값이자 저장되는 값**이라 버퍼는 자기일관적이다 |
| `--fake-env` | off | ROS 백엔드도 `GelloIntervention`도 붙이지 않는다 (Stage A) |
| `--checkpoint-path` | 없음 | ⚠️ **실질적으로 아무 pickle도 안 나온다.** 다만 이유는 아래 각주대로 이전 판의 설명과 다르다 → `08` G20 |

> ### 🔧 정정 (2026-07-29, 코드 확인) — `--checkpoint-path`가 안 되는 **진짜 이유**
> 이전 판은 *"`cube_in_cup`의 `buffer_period = 0`이라 덤프 자체가 게이트에서 걸린다"*고 적었다.
> **절반만 맞다.** `buffer_period`가 막는 것은 **주기적 중간 덤프**뿐이고
> (`remote_actor.py:519`), **루프가 끝난 뒤의 최종 덤프는 게이트가 없다**
> (`remote_actor.py:568`, `if checkpoint_path and replay_data:`).
>
> 진짜 이유는 **그 최종 덤프에 도달할 수 없다는 것**이다:
> `cube_in_cup`의 `max_steps = 1_000_000`(`cube_in_cup.py:269`)이라 10 Hz에서 **약 27.8시간**이고,
> actor CLI에는 이 값을 줄일 플래그가 **없다.** 그리고 `Ctrl-C`는 루프 **안에서** 예외를
> 던지므로 최종 덤프를 **건너뛴다**(`try/finally`가 아니다).
> 즉 `--checkpoint-path`는 배선돼 있지만 실기에서 파일을 남기지 못한다.
> **§4.4의 서버 확률 판독구가 없는 것도 같은 이유다.**

#### 1.2.1 분류기 sidecar 플래그 (2026-07-29 신규)

기본값은 전부 태스크 config의 `CubeInCupConfig.CLASSIFIER_SIDECAR`에서 온다.
플래그는 **리그에서 값을 다시 잡아 보되 config를 실수로 커밋하지 않게** 하려고 있다
(`run_remote_rlpd_actor.py::_sidecar_settings`).

| 플래그 | 기본 | 무엇을 하나 |
|---|---|---|
| `--no-classifier-sidecar` | off | **킬 스위치**(config보다 우선). 서버가 채점할 것이 없으므로 **모든 transition이 `classifier_evaluated=false` / reward 0**으로 돌아온다. sidecar의 지연 비용만 분리하거나 sidecar 이전 동작을 재현할 때만 쓴다 |
| `--classifier-sidecar-interval N` | **5** (10 Hz → 약 2 Hz) | N 스텝마다 최대 1회 부착. **성긴 것이 의도다** — 판정 깜빡임을 줄이고 큐브를 놓은 뒤 장면이 가라앉을 시간을 준다. `1`은 매 스텝 채점 = 지연 비용 최대 |
| `--classifier-stationary-speed-max M/S` | **0.05** ⚠️ | TCP 속도가 이 값 미만일 때만 부착 (움직이는 중의 블러 프레임 차단). **config 주석이 이 값을 PLACEHOLDER로 명시한다** — 녹화 take에서 "팔이 가라앉은 뒤 실제로 머무는 속도"를 재서 정해야 한다 |
| `--classifier-escalate-probability P` | **0.05** | **확률적 추첨이 아니다.** 분류기 확률이 P 이상이면 interval을 버리고 **매 스텝** 채점한다. 임계(0.2)를 실제로 넘는 스텝을 최대 interval−1 스텝 놓치지 않으려는 것이라 **`DEFAULT_REWARD_THRESHOLD`보다 낮게** 둔다 |

> 종료 시 actor가 **부착/미부착 왕복을 따로** 찍는다
> (`sidecar_round_trip_ms_mean/max` vs `plain_round_trip_ms_mean/max`).
> sidecar가 100 ms 예산에서 얼마를 먹는지는 **추정하지 말고 이 줄을 읽는다.**
> `sidecar_build_failures`가 0이 아니면 그 스텝들은 reward가 없다.

> ### 🛑 `--mock-policy-noise`로 만든 데이터를 demo로 쓰지 말 것
> 전이마다 `meta.policy_actions_synthetic = true`가 박히고, actor가 종료 시
> 그 사실을 경고한다 (`run_remote_rlpd_actor.py:341-346`).
> 이건 **배선 실증용**이지 수집용이 아니다.

> ### ⚠️ `--arm`을 이 문서로 처음 쓰는 사람에게
> actor는 **정책 액션**을 실행한다. 초기 SAC 정책의 액션 크기는 **아직 확인된 적이 없다.**
> 첫 시도는 zero-action 서버 + `--mock-policy-noise`(작은 σ)로 하거나,
> 태스크 config의 `ACTION_SCALE`을 낮춰서 한다.
> `run_real_hil.py`의 `--scale` 같은 배율 플래그는 actor에 **없다.**

### 1.3 preflight가 보는 것

| # | 점검 | 실패 시 |
|---|---|---|
| 1 | venv python 존재 + `sys.prefix`가 그 venv | FAIL |
| 2 | `grpcio` 버전 ≠ `1.30.2` (import 없이 메타데이터만 읽음) | FAIL |
| 3 | `cygrpc.CompletionQueue()` — **`timeout --signal=KILL 20` 서브프로세스** | 20초 초과 = FAIL. 인라인으로 부르면 이 스크립트도 같이 멈추므로 절대 그렇게 하지 않는다 |
| 4 | `/opt/ros/humble/setup.bash` + `<repo>/ros2_ur_ws/install/setup.bash` 존재·소스, PYTHONPATH 이어붙임 | 없으면 FAIL |
| 5 | `ur_env` / `ur_experiments` / `ur_gello_bringup` / `serl_launcher` + `rclpy`/`gymnasium`/`numpy`/`cv2` 해석 경로 | 못 찾거나 트리 밖이면 FAIL |
| 6 | `SERVER_HOST:SERVER_PORT` TCP connect (3s) | FAIL — 터널/서버 확인 |
| 7 | `/joint_states`, `/gello/joint_states`, cam1/cam2 compressed | 실기 모드 FAIL / fake-env WARN |
| 7b | `/robotiq_gripper/position_percent` | WARN (19-D state의 그리퍼 채널) |
| 8 | STJC/FPC의 정확한 상태 쌍 | `active/inactive`(handoff 전) 또는 `inactive/active`(이미 완료)만 허용. 둘 다 active/inactive면 FAIL |
| 9 | `/forward_position_controller/commands`의 **퍼블리셔 수** | 1개라도 있으면 **FAIL·거부** (이 리그 최대 하자) |
| 10 | `--dry-preflight --arm`이면 현재 관절 오차 ≤0.10 rad. switch 전 상태면 marker·수명·owner/mode·ROS domain·RESET 값도 확인 | 하나라도 다르면 FAIL. 이미 FPC active인 idempotent 경로도 임의 자세면 거부. 검증만 하고 switch하지 않음 |

`--dry-preflight`의 모든 점검은 **읽기 전용**이다. 실제 `--arm`은 [1]~[10]이
통과한 다음에만 publisher 수를 다시 세고 strict controller switch를 수행한다. switch 후
`STJC=inactive`, `FPC=active`, publisher 0을 다시 확인하며 하나라도 실패하면 actor를 exec하지 않는다.
이 handoff는 reset/preposition 이동을 자동 호출하지 않는다.

---

## 2. Kanu(서버) 쪽 — Stage A/B 공통 전제

### 2.1 📌 2026-07-27 세션의 **기록** — 재입력용 설정값이 아니다

> ## 🛑 이 표에서 값을 복사하지 마라
> 아래는 그날 그 세션이 무엇을 썼는지의 **기록**이다. 커밋·GPU·체크포인트 SHA는
> **전부 그 뒤에 바뀌었거나 바뀔 수 있다.** 각 항목 옆에 "지금은 어떻게 확인하나"를 적었다.

| 항목 | 📌 그날의 값 | 지금은 |
|---|---|---|
| Kanu worktree | `/tmp/gello-hil-rl-receive-server-v2` @ `5fb716b` | **커밋이 다르다.** 랩톱과 같은 커밋인지 양쪽에서 `git rev-parse --short HEAD`로 대조한다 |
| overlay venv | `/tmp/gello-hil-rl-receive-overlay-v2` | 경로는 유효하나 **learner에는 재사용 금지** (protobuf 3.20.3이 wandb를 깨뜨린다 → `05` §1.2) |
| GPU | `CUDA_VISIBLE_DEVICES=7` | **7번 고정이 아니다.** `nvidia-smi`로 비어 있는 카드를 매번 다시 고른다 (2026-07-28 기준 5/6/7 전부 유휴) |
| 서버 포트 | `50053` (loopback bind) | 그대로 (코드 기본값) |
| 로컬 터널 입구 | `50153` → 원격 `50053` | 그대로 (`run_hil_actor.sh:72`의 `SERVER_PORT` 기본값도 50153) |
| 분류기 checkpoint SHA-256 | **`512b6575…62846d`** | ✅ `cube_in_cup_all3/checkpoint_150`의 **디렉터리** digest. *(그날 기록된 `e329986b…d7a997`는 recall 0%짜리 폐기 체크포인트였다. 코드 기본값도 07-29에 교체됐다 → `08` G19)* |
| observation schema hash | `3459098d…0352903` | 📌 2026-07-29 랩톱에서 동일. **그래도 양쪽에서 출력해 대조한다** (`05` §4.2) |

> 로컬 포트가 50053이 아니라 **50153**인 이유: 그날 로컬 50053이 다른 프로세스에
> 잡혀 있었다. 터널 로컬 쪽만 바꾸고 원격 쪽은 50053 그대로 둔다. 이 배치가 코드
> 기본값이 됐으므로 그대로 쓴다.

📌 **Kanu 쪽 실제 경로** (`/home/laptop3/gello_software`는 kanu에 **없다**):

| 무엇 | 경로 |
|---|---|
| 분류기 학습처 (YWhero/hil-serl fork, `agent/cube-in-cup-classifier` @ `d753571`) | `~/workspace/youngwoong/hil-serl` |
| 학습 데이터 + 07-27 체크포인트 | `~/workspace/youngwoong/dataset/cube_in_cup_all3/` |
| ZMQ 뷰어 + 07-24 체크포인트 | `~/workspace/youngwoong/gello_software_remote_classifier` |
| 이 리포의 Kanu 전용 worktree | `/tmp/gello-hil-rl-receive-server-v2` |
| python | `/home/junhyeong/miniconda3/envs/il/bin/python` |

### 2.2 Kanu에서 서버 띄우기

```bash
# Kanu에서 — 먼저 확인 3가지
cd /tmp/gello-hil-rl-receive-server-v2
git rev-parse --short HEAD          # 랩톱의 HEAD와 같은가? (특정 SHA를 기대하지 말 것)
nvidia-smi                          # 쓰려는 GPU가 비어 있는가? (7번 고정 아님)
ss -ltn | grep 50053                # 이미 잡혀 있지 않은가?

export XLA_PYTHON_CLIENT_PREALLOCATE=false   # 필수. 안 하면 JAX가 카드를 통째로 선점한다

CUDA_VISIBLE_DEVICES=<위에서 고른 번호> \
PYTHONPATH=serl_ur_infra:third_party/hil-serl/serl_launcher \
/tmp/gello-hil-rl-receive-overlay-v2/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_server.py \
  --host 127.0.0.1 \
  --port 50053 \
  --checkpoint <체크포인트 경로> \
  --expected-checkpoint-sha256 <그 체크포인트의 sha256> \
  --reward-model-id cube-in-cup-all3-ckpt150+sidecar-v1 \
  --require-jax-backend gpu
```

> ### 🔧 `--threshold`를 빼 놓은 이유 (2026-07-29 정정)
> 이전 판들은 `--threshold 0.85`, 그다음 `0.5`를 박아 뒀다. **둘 다 이제 틀리다.**
> 코드 기본값은 **`0.2`**다 (`DEFAULT_REWARD_THRESHOLD`, `rlpd_receive_server.py:73`;
> `0.85` → `0.5`(`53d5cf6`) → `0.2`(`1b02857`)).
> 문서에 리터럴을 두면 또 어긋나므로 **생략해서 코드 기본값을 쓰게 한다.**
>
> 명시해야 하는 경우는 하나뿐이다: **다른 threshold로 학습된 checkpoint를 resume할 때.**
> threshold는 learner fingerprint의 `run_contract`에 들어가고
> (`run_rlpd_learner_server.py:548-550`), 불일치는 **fail-closed로 거부**된다.
> 그때는 learner 쪽에 `--reward-threshold <그 값>`을 준다.
>
> ⚠️ ~~그리고 threshold를 정한 근거 수치는 **전부 크롭 없는 입력에서 측정된 것**이다.
> 크롭이 활성인 지금 경로에서는 재측정이 필요하다.~~
> **🔧 정정 (2026-07-29): 재측정이 필요 없다.** 분류기는 sidecar 덕분에 **계속 무크롭을 먹는다**.
> 그래서 `REWARD_CLASSIFIER_THRESHOLD_KO.md`의 수치가 이 경로에 **그대로 적용된다** (`08` G15).

> ### 🔧 `--reward-model-id`는 이제 기본값이 있다
> `cube-in-cup-all3-ckpt150+sidecar-v1`. **이 id는 체크포인트와 입력 계약을 둘 다 이름에 담는다.**
> sidecar 이전 actor ↔ 이후 server(또는 반대)는 **서로 다른 픽셀에서 reward를 계산**하므로,
> 그 조합을 **핸드셰이크에서 거부**한다 — 한 세션을 통째로 잘못된 reward로 돌리는 것보다 낫다.
> actor 쪽 짝은 `run_hil_actor.sh`의 `EXPECTED_REWARD_MODEL_ID`이고 같은 값이다.

> ### 🔧 `--checkpoint` / `--expected-checkpoint-sha256` — 기본값이 이제 정본이다
> **🔧 정정 (2026-07-29):** 이전 판은 *"생략하면 폐기 대상인 07-24 체크포인트(`e329986b…`)를
> pin한다 / 07-27 체크포인트는 orbax 디렉터리라 `os.path.isfile` 요구를 통과하지 못한다"*고 적었다.
> **둘 다 해소됐다** — `checkpoint_sha256()`이 `classifier_sidecar.directory_sha256()`에 위임하고,
> 두 기본 SHA는 `512b6575…`로 교체됐다 (`08` G19).
> 그래도 **어느 체크포인트가 로드됐는지 기동 로그에서 확인하는 습관은 유지할 것** — 이 실패는 조용하다.
>
> `--replay-capacity` / `--intervention-capacity`도 뺐다. 코드 기본값이 정확히
> 50000 / 10000이다 (`rlpd_receive_server.py:49-50`).

근거: `serl_ur_infra/RL_RECEIVE_SERVER.md` §"Preferred Kanu runtime",
`serl_ur_infra/HIL_RLPD_RECEIVE_SERVER_KO.md` §"Kanu 실행 환경".

> **주의 — 서버 종류가 두 가지다.**
> * `run_rlpd_receive_server.py` = **receive-only** 마일스톤. 정책 없음, 항상 zero action,
>   `model_id = "fake-zero-action-v0"` (`ur_env/rlpd_receive_server.py:169`).
> * `run_rlpd_learner_server.py` = 실제 RLPD learner. `model_id =
>   "hil-serl-hybrid-sac-resnet10-trunk-cache-v1"` (`ur_env/learner/config.py:14`).
>
> 어느 쪽을 띄웠는지에 따라 actor의 `EXPECTED_MODEL_ID`가 달라진다 (§2.3).

### 2.3 `EXPECTED_MODEL_ID`를 맞추는 법

actor는 **에피소드 시작마다** `GetServerInfo`를 다시 읽고 pin과 대조한다. 틀리면
첫 inference **전에** 죽는다 — 그게 설계된 동작이다. 에러 메시지에 서버가 광고한
실제 값이 들어 있다 (`ur_env/grpc_actor_transport.py:551-557`):

```
ActorProtocolError: server model_id is 'fake-zero-action-v0', expected
  'hil-serl-hybrid-sac-resnet10-trunk-cache-v1'
```

→ 그 값을 그대로 환경변수로 넘긴다:

```bash
EXPECTED_MODEL_ID=fake-zero-action-v0 ./run_hil_actor.sh --fake-env
```

### 2.4 SSH 터널 (로컬 터미널 T0, 계속 띄워 둔다)

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50153:127.0.0.1:50053 kanu
```

확인 (다른 터미널):

```bash
ss -ltnp | grep 50153      # ssh가 127.0.0.1:50153에 LISTEN 중이어야 한다
```

`ExitOnForwardFailure=yes`가 핵심이다 — 포워딩에 실패하면 ssh가 조용히 붙어 있지 않고 죽는다.

---

## 3. Stage A — fake-env로 Kanu 왕복 (**로봇 사용 안 함**)

목적: **로봇을 전혀 건드리지 않고** wrapper 체인 · 관측 스키마 · gRPC 계약 ·
버퍼 삽입까지 왕복을 증명한다. `fake_env=True`면 `GelloIntervention` 래퍼가 아예
붙지 않고 ROS 백엔드도 열리지 않는다 (`ur_experiments/cube_in_cup.py`
`get_environment()`의 `if not fake_env:` 분기).

### 3.1 절차 (복붙)

```bash
# T0 — 터널 (§2.4). 그대로 띄워 둔다.
ssh -N -T -o ExitOnForwardFailure=yes -L 127.0.0.1:50153:127.0.0.1:50053 kanu
```

```bash
# T1 — preflight만 먼저. actor는 안 뜬다.
cd $WT/ros2_ur_ws
./run_hil_actor.sh --dry-preflight --fake-env
```

fake-env 모드에서는 센서 토픽이 없어도 **WARN으로만** 뜨고 통과한다.
반드시 통과해야 하는 것은 [1]~[6]이다.

```bash
# T2 — actor 기동 (Ctrl-C로 중단)
cd $WT/ros2_ur_ws
EXPECTED_MODEL_ID=<서버가 광고하는 값> ./run_hil_actor.sh --fake-env
```

### 3.2 순수 전송 계층만 보고 싶을 때 (receive server 인수 테스트)

actor/wrapper를 빼고 gRPC 전송과 버퍼 삽입만 보려면 100-step 인수 클라이언트를 쓴다.
**서버 버퍼에 synthetic transition 100개가 실제로 들어간다** — 프로덕션 수집 중에는 쓰지 않는다.

```bash
cd $WT
PYTHONPATH=$WT/serl_ur_infra${PYTHONPATH:+:$PYTHONPATH} \
  /home/laptop3/venvs/gello-hil-actor/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_smoke_client.py \
  --host 127.0.0.1 --port 50153 --steps 100
```

### 3.3 판정 — 📌 Stage A는 **통과했다** (2026-07-27 기록, PASS)

서버 로그에서 확인된 값:

```
replay_insert_count : 100
state_shape         : [8, 1, 19]
```

* `replay_insert_count: 100` = 보낸 100개 transition이 전부 replay buffer에 삽입되고 ACK됨.
* `state_shape: [8, 1, 19]` = 서버가 replay에서 **실제로 batch를 샘플링해 본** 결과
  (batch 8 × chunk 1 × 19-D state).
* 즉 **스키마 해시 일치 → 분류기 추론 → replay 삽입 → 샘플링 가능**까지 한 줄로 증명됐다.
* ⚠️ 상대는 **zero-action 서버**였고 관측은 synthetic이었다. 이것은 **전송 계약의 증명**이지
  정책·보상의 증명이 아니다.

> ### 🔧 정정 (2026-07-29) — 이 절이 적고 있던 19-D 순서가 **틀렸다**
> 이전 판은 `state_shape` 설명에
> *"TCP position XYZ / Euler XYZ / linear vel XYZ / angular vel XYZ / force XYZ / torque XYZ /
> gripper"* 라고 적고 `RL_RECEIVE_SERVER.md`를 근거로 달았다.
> **그 문서가 옛 v1 순서에서 갱신되지 않았고, 이 문서가 그걸 그대로 복사했다.**
>
> canonical v2의 실제 순서는 **알파벳순**이다:
>
> ```
> [0:1)  gripper_pose     [1:4)  tcp_force    [4:10) tcp_pose
> [10:13) tcp_torque      [13:19) tcp_vel
> ```
>
> 즉 **그리퍼는 index 0이고 `state[..., -1]`은 `tcp_angular_velocity_z`다.**
> 근거는 `ur_env/observation_schema.py`(라이브 출력은 `05_COMMS_GRPC.md` §4.2),
> 이유는 upstream `SERLObsWrapper`가 쓰는 `gym.spaces.Dict`가 매핑을 알파벳순으로
> 재정렬하기 때문이다. **인덱스는 `GRIPPER_POSITION_INDEX` /
> `gripper_position_from_state()`로만 읽는다.**

---

## 4. Stage B — 실센서 + GELLO 개입 (DRY_RUN)

### 4.1 안전 전제 — 읽고 시작할 것

* 🔧 **`clip_safety_box`는 구현돼 있다** (2026-07-29 정정. 이전 판의 "아직 구현되어 있지
  않다"는 낡았다). `UR7eEnv.clip_safety_box()` / `_clip_command_pose()`가 있고 후자가
  `PolicyDeltaController(clip_pose=...)`로 배선되어 **명령 포즈**에 적용된다
  (`ur7e_env.py:188`, `:334`, `:356`). 실측 박스는 `ur_experiments/cube_in_cup.py:162-167`.
  **그러나 실기에서 클립이 발동한 적은 한 번도 없다** — 그래서 Stage B는 여전히
  **DRY_RUN에서만** 한다. → `08_OPEN_GAPS.md` G1
* 🟢 **분류기 크롭 불일치는 코드에서 닫혔다** (2026-07-29, `08` G15).
  ~~`EXP_NAME=cube_in_cup`이면 `IMAGE_CROP`이 적용되는데 pin된 분류기는 크롭 없이 학습됐다~~ —
  이제 분류기는 이 크롭을 **아예 보지 않는다.** actor가 무크롭 sidecar를 따로 붙인다 (`05` §3.2).
  그러나 **`DRY_RUN`이 팔만 막고 보상/종단은 막지 않는다는 사실은 그대로다.**
  그리고 이 경로는 **실기에서 한 번도 안 돌았다.** Stage B로 데이터를 모으기 전에
  **§4.4를 먼저 한다** — 그게 크롭 불일치가 실제로 고쳐졌다는 가장 값싼 증거다.
* 🔴 **아직 안 고쳐진 것: 팔 가림(occlusion).** sidecar는 전처리 불일치를 고쳤지
  **시야 문제를 고치지 못했다.** 팔이 cam1 시야를 쓸고 지나가는 동안 확률이
  0.005 → 1.0으로 진동한다(`take_21`은 @0.85에서 recall 0.0 %). 정지 게이트가 **완화**할 뿐이다.
  → `08` G15 §잔여.
* `cube_in_cup` 태스크 config의 `DRY_RUN`은 **기본 `True`**
  (`ur_experiments/cube_in_cup.py:227`). 래퍼는 그 값을 건드리지 않는다.
  🔧 다만 **actor의 `--arm`이 그 값을 끈다** (`run_remote_rlpd_actor.py:288`) —
  "DRY_RUN 해제는 이 문서의 범위가 아니다"는 더 이상 맞지 않는다. §1.2 참조.
* **이중 퍼블리셔 금지.** `gello_ur_bridge`(텔레옵 브리지)를 띄운 채로 actor를 올리면
  두 퍼블리셔가 `/forward_position_controller/commands`를 서로 다른 목표로 때린다.
  preflight [9]가 이를 감지하고 **거부**한다(`--arm` 여부 무관). 거부당하면 브리지를 먼저 끈다.
  `--arm`을 주면 actor 자신도 한 번 더 센다 (`:156-198`).
* 펜던트 E-STOP을 손 닿는 곳에. 작업 공간을 비운다.
* ⚠️ **ESC 키를 조심한다.** env가 전역 pynput 리스너를 띄우므로 **아무 창에서 누른 ESC가
  에피소드를 끝내고, 다음 `reset()`이 팔을 `RESET_JOINTS`로 옮긴다** (`08` G17).

### 4.2 터미널 구성

```bash
# 모든 터미널 공통 (run_hil_actor.sh를 쓰는 T6은 예외 — 스크립트가 알아서 한다)
cd $WT && source /opt/ros/humble/setup.bash && source ros2_ur_ws/install/setup.bash
```

**T1 — UR 드라이버 (툴 통신 포함)**

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur7e \
  robot_ip:=192.168.10.11 \
  headless_mode:=true \
  launch_rviz:=false \
  initial_joint_controller:=scaled_joint_trajectory_controller \
  use_tool_communication:=true \
  tool_voltage:=24 \
  tool_device_name:=/tmp/ttyUR
```

* `headless_mode:=true` → 펜던트를 **REMOTE 모드**로 둬야 한다.
* `initial_joint_controller:=scaled_joint_trajectory_controller` → 기동 시
  `forward_position_controller`는 **inactive**다. preflight [8]이 이걸 확인한다.
  (실측 2026-07-27: `forward_position_controller  …  inactive`)
* `use_tool_communication:=true`면 **드라이버가 `:54321`을 점유**하고 `/tmp/ttyUR`을 만든다.
  `:54321`은 **단일 클라이언트**이므로, 이 상태에서 그리퍼 노드는 TCP가 아니라
  **serial `/tmp/ttyUR`** 로 붙어야 한다 (`robotiq_gripper_modbus_node.py:69`의
  `serial_port` 파라미터: 비어 있지 않으면 TCP 대신 serial 사용).

**T2 — 그리퍼 (2F-85)** — 로봇이 **POWER ON** 이어야 툴 전압이 나온다.

```bash
ros2 run ur_gello_bringup robotiq_gripper_modbus \
  --ros-args -p serial_port:=/tmp/ttyUR
```

(실행 파일 이름은 `robotiq_gripper_modbus` — `_node` 없음.
`ros2_ur_ws/src/ur_gello_bringup/setup.py:32`)

확인: `/robotiq_gripper/position_percent`가 ≈5 Hz (실측 2026-07-27: 5.001 Hz).

**T3 — 카메라 2대**

```bash
cd $WT/ros2_ur_ws && ./launch_cameras.sh        # 뷰어가 기본 ON이다
```

뷰어로 **cam1 = 장면(삼각대), cam2 = 손목**을 눈으로 확인한다. 뒤바뀌면 관측이 조용히 망가진다.
판정은 **팔을 한 번 흔들어** cam2 화면 전체가 흐르고 손가락이 제자리인지 보는 것이다.
확인: 두 토픽 다 ≈30 Hz.

> ⚠️ 시리얼은 이제 **live USB 버스에서 자동 해석된다** (`43ba314`/`fb48100`).
> 기동 로그에 `WARN auto-selected by model class` 또는 `WARN model classes are ambiguous`가
> 뜨면 자동 배정이 일어난 것이므로 **뷰어 판정을 반드시 한다.** 시리얼을 명령줄에
> 적을 필요는 없다 — 강제하려면 `CAM1_SERIAL=... CAM2_SERIAL=...`. → `06_SENSORS.md` §1.1

**T4 — GELLO 리더** (`/dev/ttyUSB0` 점유 — 두 개 띄우지 말 것)

```bash
GELLO_REPO_ROOT=$WT \
ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
  $WT/ros2_ur_ws/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml
```

확인: `/gello/joint_states` ≈30 Hz (실측 2026-07-27: 29.98 Hz).
`GELLO_REPO_ROOT`가 없으면 `No module named 'gello'`로 죽는다.

**T5 — HIL 데드맨 GUI**

```bash
cd $WT/ros2_ur_ws && ./run_hil_gui.sh
```

ENGAGE/DISENGAGE 버튼 + 감도 슬라이더. 20 Hz 하트비트를 `/hil/deadman`에 쏜다.
첫 메시지가 15 s 안에 없으면 actor 시작을 거부한다. 한 번 받은 뒤 0.5 s 끊기면
`DeadmanHeartbeatStaleError`로 actor가 종료되고 정책 fallback은 실행되지 않는다.
정상 DISENGAGE 메시지만 정책에 제어를 돌려준다. **스페이스바 데드맨은 쓰지 않는다**
(워치독이 없어 stuck-ON 위험 — `04_HIL_INTERVENTION.md` §1.1).

**T0 — 터널** (§2.4)

**T6 — preflight → actor**

```bash
cd $WT/ros2_ur_ws

# 1) actor를 내린 채 RESET 자세를 검증/사전 배치한다.
#    이미 0.10 rad 안이면 이동 없이 proof만 만들고 종료한다.
#    멀면 기존 GO gate를 통과한 뒤에만 JTC 궤적이 움직인다.
./run_hil_preposition.sh

# 2) HIL GUI를 먼저 ENGAGE하고 GELLO를 RESET anchor에 고정한다.
#    dry-preflight도 연속 ENGAGED heartbeat 3개를 요구한다.

# 3) 실제 --arm과 같은 handoff 조건을 읽기 전용으로 검사한다.
#    이 명령은 controller를 전환하지도 actor를 띄우지도 않는다.
EXPECTED_MODEL_ID=<서버가 광고하는 값> \
  ./run_hil_actor.sh --dry-preflight --arm --deadman topic

# 4) B3 승인 조건을 별도로 만족한 뒤에만 실제 기동.
#    marker + live pose를 다시 확인하고 STJC -> FPC strict switch 후 actor를 exec한다.
#    controller switch 직전 ENGAGED heartbeat를 한 번 더 검사한다.
EXPECTED_MODEL_ID=<서버가 광고하는 값> \
  ./run_hil_actor.sh --arm --deadman topic
```

`run_hil_actor.sh`는 `run_hil_preposition.sh`를 대신 실행하지 않는다. 즉 `--arm` 한 줄이
사전 배치 이동을 몰래 시작하는 일은 없다. proof가 없거나 15분이 지났거나, proof 뒤 팔이
RESET 자세에서 벗어났으면 전환 전에 실패한다. 첫 B3는 아래 §4.4의 정지·ENGAGE 조건을
포함한 operator-gated smoke이며, 일반 자율 policy run으로 바로 DISENGAGE하는 절차가 아니다.

### 4.3 실기 모드 preflight의 통과 기준 (2026-07-27 실측 예시)

```
[1] interpreter = /home/laptop3/venvs/gello-hil-actor/bin/python (python 3.10.12)
[2] grpcio 1.74.0 (손상판 아님)
[3] gRPC 코어 정상 (CompletionQueue 생성 성공)
[4] sourced /opt/ros/humble/setup.bash + <repo>/ros2_ur_ws/install/setup.bash
    ROS 오버레이가 PYTHONPATH에 살아 있다 (덮어쓰기 아님)
[5] ur_env / ur_experiments / ur_gello_bringup / serl_launcher 전부 이 트리에서 해석
[6] TCP 127.0.0.1:50153 — 연결 성공
[7] 정상 종료형 production-QoS probe:
    /joint_states ≈ 100.4 Hz · /gello/joint_states ≈ 30.0 Hz
    cam1 ≈ 30.0 Hz · cam2 ≈ 30.0 Hz · 그리퍼 ≈ 5.0 Hz
[8] scaled_joint_trajectory_controller=active, forward_position_controller=inactive
[9] 퍼블리셔 0개 — actor가 유일한 퍼블리셔가 된다
[10] preposition marker + 현재 RESET 자세 확인 (읽기 전용; switch하지 않음)
[11] deadman ENGAGED: 3 consecutive heartbeats
```

`[7]`은 `ros2 topic hz`를 timeout 뒤 강제 종료하지 않는다. 각 probe는 5개의 fresh하고
증가하는 샘플을 받은 즉시 `destroy_node()`/`rclpy.shutdown()`으로 reader를 정상 정리한다.
GUI에 cam1 영상이 보이더라도 `[7]`의 cam1만 실패하면 우회하지 말 것. 기존 GUI reader는
살아 있으나 **새 actor reader에는 큰 이미지가 전달되지 않는 Fast DDS writer 상태**일 수
있다. 카메라 터미널에서 `Ctrl-C`로 두 카메라를 정상 종료한 뒤
`./launch_cameras.sh`를 다시 띄우고 `[7]`을 재검증한다.

### 4.4 🔎 첫 `--arm`의 정지·ENGAGE 구간에서 classifier sidecar를 확인한다

**이것이 G15 수정의 가장 값싼 실기 증거다.** no-arm actor는 이제 의도적으로
`BeginEpisode` 뒤 종료하므로 transition/classifier sidecar를 만들지 않는다. 따라서 sidecar의
실제 gRPC 왕복은 첫 `--arm` smoke 안에서 확인해야 한다. 이때 GUI를 미리 ENGAGE하고 GELLO를
RESET anchor에서 움직이지 않아 human action이 zero에 가깝게 유지되도록 한다. policy action은
counterfactual로 기록되지만 로봇에 실행되지 않는다.

**원리.** 라이브 뷰어와 서버는 **같은 전처리 함수**를 돈다 —
뷰어는 `gello_recorder.reward_classifier_runtime.decode_classifier_image()`,
서버는 `ur_env.classifier_sidecar.decode_classifier_frames()`이고
둘 다 *imdecode → **무크롭** → `resize(128,128)` → RGB → batch axis*다.
그리고 **평활이 꺼져 있으므로**(`--success-confirmations 1`, 서버 기본값)
서버의 `classifier_probability`는 **시간 필터가 전혀 없는 순간 sigmoid**다 —
뷰어가 화면에 찍는 것과 **같은 종류의 수치**다. 그래서 이 비교가 근사가 아니라
**등가성 점검**이 된다.

> ⚠️ 남는 차이는 **딱 하나, 128×128에서의 JPEG 1세대**다. sidecar는 랩톱에서 resize 후
> 재인코딩되므로 뷰어와 **비트 단위로 같지는 않다.** 실측 기대치:
> 순수 JPEG 왕복 대조군은 `|Δp| ≤ 0.010`, 판정 경계 근처(합성 프레임)는 `0.022–0.090`으로
> **주변 센서 노이즈와 같은 수준**이다 (`05` §5.4).
> **판정은 "소수점까지 같은가"가 아니라 "자릿수가 같은가"다.**
> 뷰어가 0.9인데 서버가 0.02 같은 차이가 나면 **전처리가 아직 갈라져 있다는 뜻**이다.

**절차 (터널·카메라·UR driver 필요. 팔은 RESET에 두고 움직이지 않는다):**

1. 카메라 2대를 띄운다(§4.2 T3). **장면을 고정한다** — 손을 넣지 않는다.
   팔은 **정지**해 있어야 한다. sidecar의 정지 게이트가 안 열리면 서버가 채점할 것이 없다.
2. 라이브 뷰어를 띄우고 `p(success)`가 안정될 때까지 둔다.
   절차 정본은 [`serl_ur_infra/REWARD_CLASSIFIER_LIVE_KO.md`](../../serl_ur_infra/REWARD_CLASSIFIER_LIVE_KO.md)다
   (랩톱 CPU는 `run_classifier_viewer.sh`, kanu GPU + 터널은 `run_remote_classifier_viewer.sh`).
   두 카메라의 값을 적어 둔다.
3. HIL GUI를 **먼저 ENGAGE**하고 GELLO를 놓지 말고 RESET anchor에서 정지시킨다. §4.2의
   preposition과 `--dry-preflight --arm`을 통과한 뒤, 부착을 매 스텝으로 올린 짧은
   operator-gated `--arm` smoke를 시작한다. 이 명령은 실제 controller를 전환하므로 E-STOP을
   손에 두고, 확인 표본을 얻으면 Ctrl-C로 종료한다:

   ```bash
   cd $WT/ros2_ur_ws
   EXPECTED_MODEL_ID=<서버가 광고하는 값> \
     ./run_hil_actor.sh --arm --deadman topic --classifier-sidecar-interval 1
   ```

   ENGAGE를 풀지 않는다. heartbeat가 끊기면 actor는 policy fallback 없이 fail-stop해야 한다.

4. 서버가 광고한 계약을 기동 로그에서 확인한다 — `rlpd_receive_server_ready` 한 줄에
   전부 들어 있다: `reward_model_id`, `checkpoint_sha256`, `threshold`,
   **`classifier_input_contract`**, **`success_confirmations`**.
   `success_confirmations`가 1이 아니면 **이 비교는 성립하지 않는다**(평활이 켜진 것).
5. actor stdout에 `classifier sidecar not built`가 없어야 하고, server에는 연속 미분류
   경고나 classifier fault가 없어야 한다. 현재 actor는 Ctrl-C 때 summary를 출력하지 않으므로
   `sidecar n=`을 짧은 run의 강한 판독구로 쓸 수 없다(짧은 `max_steps` CLI와 함께 후속).

> ### 🪤 그런데 **서버의 확률을 스텝마다 찍어 주는 곳이 지금 없다** (2026-07-29 코드 확인)
> 정직하게 적는다. `classifier_probability`는
> (a) `TransitionOutcome`으로 actor에 돌아가 transition dict에 들어가고
> (`ur_env/remote_actor.py:494-504`), (b) 서버 replay buffer에 저장된다.
> **그러나 어느 쪽도 로그로 나오지 않는다.**
> 서버가 스스로 말하는 경우는 두 가지뿐이다: **연속 100건이 미분류**일 때, 그리고
> **한 세션이 단 한 건도 분류되지 않은 채 끝났을 때**
> (`rlpd_receive_server.py::RewardTransitionFinalizer`의 경고).
> `--checkpoint-path`의 로컬 pickle도 이 용도로는 못 쓴다 — 아래 §1.2 각주를 볼 것.
>
> **그래서 4번까지는 오늘 그대로 실행되지만, "서버 숫자 대 뷰어 숫자" 대조는 판독구가
> 하나 생겨야 완결된다.** 그 전까지의 대체 판정은 **같은 라이브 프레임 한 쌍에** 두 레시피를
> 직접 걸어 확률을 비교하는 것이다 —
> `decode_classifier_image()`(뷰어 경로) vs `build_sidecar()` → `decode_classifier_frames()`
> (서버가 실제로 도는 경로). 서버는 후자를 **그대로** 부르므로, 이 둘이 맞으면
> 전처리는 맞은 것이다. 오프라인 등가성은 `tests/test_classifier_sidecar.py`가
> 이미 강제한다(resize-only 경로는 **비트 단위**, 인코드 왕복은 **측정된 오차 범위**).

---

## 5. 실패 대응표

| preflight 메시지 | 조치 |
|---|---|
| `actor venv python이 없다` | `ACTOR_VENV` 확인. 없으면 `python3 -m venv --system-site-packages /home/laptop3/venvs/gello-hil-actor` 후 `serl_ur_infra/requirements-grpc.lock` 설치 |
| `grpcio 1.30.2 — 손상된 것으로 확인된 바로 그 버전` | venv 안에 최신 grpcio 재설치. **절대 이 상태로 띄우지 말 것** (조용히 영구 정지) |
| `cygrpc.CompletionQueue()가 20초 안에 돌아오지 않았다` | 위와 같은 증상의 직접 증거. grpcio 재설치 |
| `… 이 저장소 밖에서 해석됨` | 다른 checkout에 editable 설치된 `serl-ur-infra`가 이기고 있다. 지금 트리에서 스크립트를 실행하고 있는지 확인 |
| `ur_experiments: 찾을 수 없음` | 지금 checkout에 `serl_ur_infra/ur_experiments/`가 없다 = 브랜치가 틀렸다 |
| `TCP …:50153 연결 실패` | 터널이 죽었다. §2.4 재실행 → 그래도 안 되면 Kanu에서 서버가 살아 있는지 확인 |
| `cam1 … 에서 6s 동안 메시지가 없다` | 카메라 노드는 살아 있는데 스트림이 멈춘 상태일 수 있다. `ros2 topic info`의 Publisher count가 1인데 `hz`가 비면 **USB 재연결 후 `launch_cameras.sh` 재기동** |
| `예상 밖 controller 조합` | STJC/FPC가 둘 다 active 또는 둘 다 inactive다. 수동으로 우회하지 말고 driver/이전 actor 종료 상태를 확인 |
| `arm handoff proof 검증 실패` | actor를 내리고 `./run_hil_preposition.sh`를 실행. 이미 RESET 0.10 rad 안이면 움직이지 않고 새 marker만 만든다 |
| `퍼블리셔가 N개 있다` (FAIL) | 텔레옵 브리지/다른 러너가 살아 있다. `ros2 topic info -v /forward_position_controller/commands`로 범인을 찾아 끄고 재실행 |

### 5.1 `RESET_MAX_DIST_RAD` 게이트 (실기에서 자주 만난다)

리셋은 **현재 자세 → `RESET_JOINTS`** 사이를 250 Hz 업샘플러로 그대로 쓸고 지나간다.
그래서 그 거리가 `RESET_MAX_DIST_RAD`를 넘으면 **에러로 멈춘다 — 그게 의도된 안전 실패다.**

> 🔧 **정정 (2026-07-29): 값이 `0.5`가 아니라 `0.9`다.**
> `cube_in_cup`의 게이트는 **`RESET_MAX_DIST_RAD = 0.9`**이다
> (`serl_ur_infra/ur_experiments/cube_in_cup.py:103`; `DefaultUR7eEnvConfig` 기본값 `1.5`보다
> 좁다). `ee3240e`에서 `0.5` → `0.9`로 올라갔다 — `0.5`는 23테이크 중 **16개의 정상 종료
> 자세를 거부**했기 때문이다(= 대부분의 에피소드 뒤에 `reset()`이 예외를 던졌다).
> 이 문서가 `0.5`로 남아 있으면 "왜 안 걸리지?"로 시간을 버린다.

거리는 **branch-cut safe**하게 계산된다 (`wrapped_nearest`, `08` G13) — 물리적으로 같은
자세가 ~2π 떨어진 것으로 보고되지 않는다.

증상 (`ur_env/envs/ur7e_env.py:588-592`):

```
RuntimeError: reset distance 1.23 rad exceeds RESET_MAX_DIST_RAD=0.9 —
  pre-position the arm with move-to-start first
```

원인: 사람이 GELLO 개입으로 팔을 시작 자세에서 멀리 옮긴 뒤 다음 에피소드를 리셋했다.

조치: **팔을 먼저 시작 자세 근처로 옮긴 뒤** 다시 시작한다.

```bash
# 현재 관절값이 RESET_JOINTS에 가까운지 먼저 눈으로 확인
ros2 topic echo /joint_states --once
```

멀다면 **actor를 내린 상태에서** 팔을 `RESET_JOINTS` 근처(**0.9 rad 이내**, 관절별 최대 차이
기준)로 사전 배치한다. 참고로 23테이크의 정상 종료 자세는 중앙값 0.615 / 최대 0.774였다.
전용 도구가 같은 트리에 있다 (**다른 담당 영역** — 그 스크립트의 자체 안내를 따를 것):

```bash
cd $WT/ros2_ur_ws && ./run_hil_preposition.sh
```

`gello_move_to_start`의 `init_align` 모드를 재사용해 `scaled_joint_trajectory_controller`로
보간 이동하며, 조작자의 명시적 승인 뒤에만 움직인다. `gello_move_to_start`를 맨손으로
단독 실행하는 방법은 이 리그에서 검증되지 않았으므로 여기에 적지 않는다.

성공 시 스크립트는 현재 사용자만 읽고 쓸 수 있는 proof marker를 만든다. marker는 기본
15분만 유효하며 RESET 값, ROS domain, 허용오차를 담는다. `run_hil_actor.sh --arm`은 marker만
믿지 않고 `/joint_states`를 다시 읽어 관절별 최대 오차 0.10 rad 이하를 재확인한다.

> ⚠️ 사전 배치 도구가 도는 동안에는 그것이 팔의 명령 소유자다. **끝난 뒤 반드시 내리고**
> GUI를 ENGAGE한 뒤 `./run_hil_actor.sh --dry-preflight --arm --deadman topic`에서
> [8] controller 쌍, [9] 퍼블리셔 0, [10] proof + live pose, [11] 연속 ENGAGED
> heartbeat를 모두 확인한 다음 actor를 띄운다.

게이트 값을 **키워서 통과시키지 말 것** — 그러면 리셋이 작업 공간을 가로질러 쓸고 간다.

---

## 6. 중단 · 복구

* **actor 중단:** T6에서 `Ctrl-C`. armed 래퍼가 신호를 actor에게 전달하고 실제 child 종료까지 기다린다.
* **팔이 움직이는 중이라면 먼저 펜던트 E-STOP.**
* 정상 종료/예외 뒤에는 actor의 command publisher가 사라진 것을 확인한 다음 STJC로
  strict 복귀한다. `controller cleanup PASS`를 확인한다. publisher가 남거나 controller 쌍이
  예상 밖이면 자동 switch를 거부하고 rc 70으로 끝나므로 `ros2 control list_controllers`와
  `ros2 topic info -v /forward_position_controller/commands`를 직접 확인한다.
* **DISENGAGE는 정지가 아니다.** 살아 있는 actor에서 누르면 즉시 policy가 제어한다.
  actor terminal의 Ctrl-C 또는 필요 시 E-STOP으로 actor/robot을 먼저 멈추고,
  `controller cleanup PASS` 뒤 GUI 상태를 정리한다.
* 터널이 죽으면 actor는 타임아웃으로 실패한다. §2.4를 다시 띄우고 actor를 재시작한다.
* Kanu 세션을 끝낼 때는 서버와 터널을 정상 종료해 **쓰던 GPU와 포트 50053을 반납**한다.
  문서의 PID/GPU 스냅샷을 믿지 말고 `run_hil_server.sh --check`와 Terminal 1 READY 배너에서
  현재 process/run root를 확인한다.

---

## 7. 검증 상태표

📌 A1~B1은 **2026-07-27 세션의 기록**이다. 그 뒤 머지가 있었으므로, 다시 돌릴 때는
`--dry-preflight`부터 새로 돌려 값을 다시 만든다.

| # | 항목 | 상태 | 📌 근거 / 실측치 (2026-07-27) |
|---|---|---|---|
| A1 | `run_hil_actor.sh` preflight (실기 모드) | **PASS** | 1회차 [1]~[9] 전부 초록: `/joint_states` 99.9–100.4 Hz, `/gello/joint_states` 29.98–30.02 Hz, cam1 30.03 Hz, cam2 30.02 Hz, 그리퍼 5.001 Hz, fpc `inactive`, commands 퍼블리셔 0 |
| A1b | preflight가 **실제 장애를 잡아냈다** | **PASS** | 같은 세션 후반, cam1(이어서 cam2)의 노드는 살아 있고 `ros2 topic info`의 Publisher count도 1인데 데이터가 끊겼다. preflight [7]이 FAIL로 잡고 actor 기동을 막았다 → §5의 카메라 항목 |
| A2 | preflight 실패 경로 | **PASS** | 없는 venv → FAIL·중단(rc=1). 닫힌 포트 → `ConnectionRefusedError`로 FAIL. 격리 도메인 → 토픽 4건 FAIL + 컨트롤러/퍼블리셔 FAIL, actor 미기동 |
| A3 | `--fake-env`에서 센서 점검이 WARN으로 강등 | **PASS** | 격리 ROS 도메인에서 경고 4건 + 통과 |
| A4 | PYTHONPATH 이어붙임 | **PASS** | `PYTHONPATH=/pre/existing`를 미리 잡고 실행해도 `ur_gello_bringup`이 오버레이에서 해석됨 |
| A5 | 인자 통과 | **PASS** | `--fake-env --save-video --actor-id …`가 최종 argv 끝에 그대로 붙음 |
| A6 | `--arm` controller handoff + 자동 복귀 | **실기 PASS** | RESET proof 뒤 STJC→FPC strict switch, actor deadline 예외 뒤 publisher-first teardown과 FPC→STJC `controller cleanup PASS`를 실제 controller_manager에서 확인 |
| B1 | Stage A (fake-env, Kanu 왕복) | **PASS** | 서버 `replay_insert_count: 100`, `state_shape: [8, 1, 19]`. 상대는 zero-action 서버 |
| B2 | Stage B (실센서 + GELLO 개입, DRY_RUN) | **미검증(TODO)** | 절차는 §4에 있으나 아직 실행되지 않았다. PASS로 승격하지 말 것 |
| B2c | **분류기 sidecar 실기 왕복** (§4.4) | **배선 PASS / verdict 관측 미완료** | production sidecar 설정으로 실제 transition이 들어갔다. per-transition `p(success)`를 GUI/영구 로그에서 볼 수 없어 장면별 판정 정합은 아직 미확인 |
| B3 | actor `--arm` (실제 팔 구동) | **핵심 E2E PASS / continuous PARTIAL** | replay 201, intervention 153, learner 102/gradient 204, policy publish v1/v2. 첫 publish가 5.474 s 걸린 경계에서 actor `Step RPC` 0.6 s timeout. learner publish가 actor에 전달됐다는 증거는 별도 미확인 |
| B4 | 같은 개입 루프를 `run_real_hil.py`로 | **PASS (2026-07-28)** | **다른 코드 경로다.** 이 표의 어느 줄도 승격시키지 않는다 → `04` §4.5 |

---

## 8. 관련 문서

* `04_HIL_INTERVENTION.md` — 데드맨 / 앵커 / 좌표계 / 두 퍼블리셔 충돌
* `05_COMMS_GRPC.md` — gRPC 계약과 19-D state 레이아웃
* `06_SENSORS.md` — RealSense 2대
* `08_OPEN_GAPS.md` — `clip_safety_box` 등 실기 투입 전 미해결 갭 (G15/G19는 닫혔다)
* `serl_ur_infra/REWARD_CLASSIFIER_LIVE_KO.md` — 라이브 분류기 뷰어 정본. **§4.4가 이걸 쓴다**
* `serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md` — Kanu learner 서버 전체 런북
* `serl_ur_infra/RL_RECEIVE_SERVER.md` / `HIL_RLPD_RECEIVE_SERVER_KO.md` — receive server 마일스톤
