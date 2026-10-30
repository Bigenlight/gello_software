# 09 — HIL actor 기동 런북 (3-CLI 실물 HIL-SERL 운영)

> # 🔴 GPU 서버가 바뀌었다 — `kanu` → `junhyeong_ai` (2026-07-31)
>
> learner는 이제 **`junhyeong_ai`**(166.104.146.29, hostname `junhyeong`, 계정 `junhyeong`)에서
> 돈다. 터널은 laptop3 `127.0.0.1:50153` → junhyeong_ai `127.0.0.1:50053`이다.
>
> ## 🟢 조작자 절차는 **바뀌지 않았다** — 이 박스에서 제일 중요한 문장이다
>
> | 터미널 | 서버 이전이 미친 영향 |
> |---|---|
> | **1** `run_hil_server.sh` | **여기만 바뀌었다.** 기본 호스트·repo·python·데이터 뿌리가 junhyeong_ai로 이동 |
> | **2** `run_hil_hardware.sh` | **서버 참조가 한 줄도 없다** (2026-07-31 확인). UR7e/Robotiq/GELLO만 소유한다 |
> | **3** `run_hil_session.sh` / `run_hil_actor.sh` | 언제나 **`127.0.0.1:50153`**(터널의 로컬 입구)만 본다. 반대편이 어느 머신인지 **원래부터 몰랐다** |
>
> 그래서 §1의 preflight `[1]`~`[11]`, §4의 손 절차, controller handoff, GUI 버튼은
> **한 글자도 달라지지 않았다.** 서버가 옮겨간 것을 조작자가 알아차릴 곳은 Terminal 1의
> READY 배너뿐이다.
>
> 환경변수는 **이름만** 바뀌었고 **옛 이름이 alias로 살아 있다**:
> `HIL_KANU_REPO` → **`HIL_REMOTE_REPO`**(`/home/junhyeong/gello_software_runtime`),
> `HIL_KANU_PYTHON` → **`HIL_REMOTE_PYTHON`**(`/home/junhyeong/miniconda3/envs/il/bin/python`),
> 신설 **`HIL_REMOTE_DATA_ROOT`**(`/home/junhyeong/hil-serl-data`).
> **평상시에는 환경변수를 하나도 주지 않는다** — 기본값이 이 서버다.
>
> ⚠️ **`run_hil_server.sh`는 이제 kanu를 구동할 수 없고, 그게 의도된 fail-closed다.**
> kanu의 classifier는 `hil-serl-data` 밖(`workspace/youngwoong/…`)에 있어서 **어떤 단일
> `HIL_REMOTE_DATA_ROOT`도 kanu를 만족시키지 못한다.** kanu는 **읽기 전용으로만** 본다
> (`ssh kanu 'ps -p <pid> -o pid,etime'`). **kanu에서 아무 프로세스도 종료하지 말 것** —
> 옛 learner는 아직 살아 있고 사용자 소유다.
>
> 🗄️ **worktree는 이제 거부된다.** `run_hil_server.sh`가 원격 checkout의
> `--git-dir == --git-common-dir`를 검사해서 linked worktree면 죽는다. 그래서 이 문서에
> 남아 있던 `/tmp/gello-hil-rl-receive-server-v2`(옛 "Kanu 전용 worktree")는 **경로도
> 형태도 은퇴했다** → §2.1 / §2.2.
>
> 서버 쪽 경로·데이터·모델 정본은
> [`serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](../../serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md),
> 이전 검증 기록은
> [`serl_ur_infra/SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](../../serl_ur_infra/SERVER_MIGRATION_E2E_JUNHYEONG_AI.md).
>
> 📌 **이 문서에 남은 `Kanu` 표기는 대부분 2026-07-27~07-30의 기록이다.** 측정값과 PASS
> 근거는 **그날 그 호스트의 사실**이므로 지우지 않고 🗄️ 표시만 붙였다.
> **사람이 실행하는 명령과 경로는 전부 새 호스트로 고쳤다.**

> ## 🆕 2026-07-31 — 새 서버에서 실기 세션 PASS
>
> - **실물 UR7e 세션이 `junhyeong_ai` learner를 상대로 통과했다**(조작자 확인):
>   replay **316** / intervention **210** / last_env_step **68**, run root
>   `junhyeong_ai:~/hil-serl-data/runs/cube_in_cup_real_20260731_054929`.
>   이것이 합성 acceptance가 못 닫던 두 가지를 닫았다 — **실제 classifier sidecar**와
>   **`intervened=1` ingress**.
> - 합성 acceptance E2E(200 transition, gRPC → finalize → feature replay → CTA → publish →
>   checkpoint → resume) per-RPC: BeginEpisode **57.7 ms 평균 / 82.2 최대**,
>   Step **156.1 ms 평균 / 211.0 최대**. 🗄️ 비교 대상 kanu는 BeginEpisode
>   **84.9 평균 / 372.8 최대**였다 — tail이 약 4.5배 짧아졌다. ICMP RTT는 두 호스트가
>   같으므로 **이득은 네트워크가 아니라 호스트 연산**이다.
> - jax 0.5.3이 sm_120(Blackwell)에서 **네이티브로 돈다**(XLA가 `.target sm_120a` 생성).
>   핀 상향은 필요 없었다.
> - 🗄️ **kanu에서 학습된 policy 체크포인트는 애초에 하나도 없었다** — run root 8개 전부
>   `checkpoints/`가 비어 있었고(`checkpoint_period`=5000, 최고 도달 learner step 301),
>   이전으로 잃은 것은 없다.

> ## 🆕 현재 운영 스냅샷 (2026-07-30, 🗄️ **당시 서버는 `kanu`**)
>
> - 실물 UR7e에서 Kanu policy, GELLO intervention, online replay/learner update까지 구동됐다.
> - 정상 진입점은 `run_hil_server.sh` / `run_hil_hardware.sh` / `run_hil_session.sh` 세 개다.
> - actor wire는 protocol 2 / schema 3, reward threshold는 **0.5**다.
> - 성공 판정 기본값은 **MANUAL**이다. classifier는 계속 실행·표시·저장하지만 GUI
>   `MARK SUCCESS`만 terminal을 만든다. AUTO에서는 strict `p > 0.5`가 terminal 권한을 가진다.
> - terminal 뒤 순서는 `WAIT_HOME_APPROVAL`(hold) → `APPROVE HOME` → HOME →
>   `WAIT_SCENE_READY` → 사람이 장면 재배치 → `START / NEXT ITERATION`이다.
> - 예전 시작 `GO`/Enter 타이핑은 기본 경로에서 제거됐다. 이것은 타이핑 제거이며
>   pose/controller proof, fresh deadman heartbeat, episode별 HOME/NEXT 승인은 남아 있다.

> ## 🆕 2026-07-30 저녁 — 조작자 절차가 네 군데 바뀌었다 (**실기 미검증**)
>
> 1. **🛑 시작할 때 ENGAGE를 누르지 않는다.** 세 gate(session 폴링, preflight `[11]`,
>    `[ARM]` 직전 재검증)가 전부 **fresh DISENGAGED heartbeat 3개**를 요구하도록 바뀌었다.
>    ENGAGE는 이제 **개입할 때만** 누른다. → §1.4
> 2. **새 버튼 `END EPISODE (truncate & re-home)`.** episode를 지금 끝내고 re-home한다.
>    🛑 **데이터를 버리지 않는다** — 이미 보낸 transition은 learner replay에 남는다. → §4.5
> 3. **충돌 뒤 세션이 죽지 않는다.** 카메라·GUI는 살아 있고 3~5단계만 다시 돈다. → §6.1
> 4. cv2/Qt 폰트 경고 두 줄이 `launch_cameras.sh`/`run_hil_actor.sh`에서 걸러진다. → §4.2 T3
>
> 이 네 가지는 **실기에서 아직 한 번도 돌지 않았다.** §7 표의 C1~C4를 볼 것.

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
>    이 경로는 첫 실제 actor run에서 production server로 전송됐다. 2026-07-30부터 GUI에는
>    마지막 evaluated classifier probability/threshold/verdict가 보인다. 아직 남은 것은
>    **실물 episode에서 그 숫자의 정합을 관측하는 일과 영구 JSONL 기록**이다.
>    `DRY_RUN`이 팔만 막고 보상/종단은 막지 않는다는 사실도 그대로다.
> 5. **actor에 sidecar 플래그 4개, 서버에 `--reward-model-id` 기본값이 생겼다.**
>    `EXPECTED_REWARD_MODEL_ID`가 **`cube-in-cup-all3-ckpt150+sidecar-v1`**로 바뀌었고,
>    옛 값 `cube-in-cup-checkpoint-150`은 **핸드셰이크에서 거부된다** (§1.1, §2.2).
> 6. PID/run root는 고정값이 아니다. `./run_hil_server.sh --check`가 현재 exact learner의
>    process, schema, buffer와 health를 읽는다. 문서의 과거 PID를 재사용하지 않는다.
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

# Terminal A: 학습 서버(junhyeong_ai) learner + SSH tunnel
./run_hil_server.sh

# Terminal B: UR7e driver + gripper + GELLO reader   (서버와 무관 — 로컬 하드웨어 전용)
./run_hil_hardware.sh

# Terminal C: cameras + HIL GUI + preposition/preflight + armed actor
./run_hil_session.sh
```

- A는 exact production learner를 재사용하거나 없을 때만 RAM/artifact gate 뒤 새로 띄우고,
  local `50153 -> junhyeong_ai 50053` tunnel을 유지한다. **`Ctrl-C`는 tunnel만 닫는다** —
  learner는 서버에 detached로 살아남는다. 그 비대칭이 실제로 사람을 물었다 → **§5.4**.
- B는 UR7e/Robotiq/GELLO만 소유한다. 충돌·연결 해제 뒤 C를 내리고 B의 cleanup 완료 후 B만
  다시 띄울 수 있다.
- C는 카메라/GUI/preposition/armed preflight/actor를 순서대로 실행한다. **controller handoff
  전에 필요한 것은 GUI가 떠서 heartbeat를 내는 것뿐이다 — ENGAGE를 누르지 않는다.**
  gate는 fresh **DISENGAGED** heartbeat 3개다(§1.4). handoff 뒤 actor는 HOME에서
  `WAIT_SCENE_READY`로 멈추며, 장면을 배치하고 GUI의 `START / NEXT ITERATION`을 누르면
  fresh observation으로 policy가 첫 action을 시작한다. 그 뒤 개입하고 싶을 때만 ENGAGE한다.
- C는 actor가 **rc 75(RECOVERABLE)**로 죽으면 카메라·GUI를 그대로 둔 채 조작자가 B를
  재기동하기를 기다렸다가 3~5단계만 다시 돈다 (§6.1). 그 외 종료 코드는 재시도하지 않는다.
- 시작 자세가 RESET 0.10 rad 밖이면 C가 `run_hil_preposition.sh`를 호출한다. 현재 기본값은
  체크리스트 출력 뒤 별도 키 입력·카운트다운 없이 JTC 이동이다. 취소 창이 필요하면
  `PREPOSITION_DELAY_S=5 ./run_hil_session.sh`, 옛 GO 프롬프트가 필요하면
  `PREPOSITION_CONFIRM=1 ./run_hil_session.sh`로 opt-in한다.
- episode 중에는 MANUAL/AUTO와 무관하게 classifier 숫자가 계속 갱신된다. MANUAL에서는
  사람이 성공을 확인했을 때 `MARK SUCCESS`; AUTO에서는 `p(success) > 0.5`가 성공이다.

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
| `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `50153` | 로컬 터널 입구. 원격 끝은 학습 서버(`junhyeong_ai`) `50053`. 🟢 **이 두 값은 서버 이전으로 바뀌지 않았다** — actor는 터널의 로컬 입구만 알고 반대편 호스트를 모른다 |
| `EXP_NAME` | `cube_in_cup` | `ur_experiments/mappings.py`의 `CONFIG_MAPPING` 키 |
| `UR_CONFIG_MODULE` | `ur_experiments.mappings` | |
| `OBS_SCHEMA_HASH` | `3459098d…0352903` | 양쪽이 같아야 한다 (§2.1) |
| `EXPECTED_MODEL_ID` | `hil-serl-hybrid-sac-resnet10-trunk-cache-v1` | **서버 종류에 따라 반드시 바꾼다** (§2.3) |
| `EXPECTED_REWARD_AUTHORITY` | `server_classifier` | |
| `EXPECTED_REWARD_MODEL_ID` | `cube-in-cup-all3-ckpt150+sidecar-v1` | 서버 `--reward-model-id`와 같아야 함. **id가 체크포인트 + 입력 계약(sidecar)을 둘 다 담는다** — 어긋나면 핸드셰이크에서 거부된다. *(이전 값 `cube-in-cup-checkpoint-150`은 이제 거부된다)* |
| `TIMEOUT_S` / `MAX_RESPONSE_AGE_S` | `1.5` / `2.0` | 2026-07-30 실기 startup에서 정상 transition reply가 832.3 ms에 도착해 기존 0.6/0.8 s 경계를 넘은 뒤 완화. 여전히 bounded이며 무제한 대기는 아니다 |
| `HZ_TIMEOUT_S` | `12` | session 토픽당 정상 종료형 liveness probe 최대 대기 시간. 5개 fresh·advancing 샘플만 요구하며 시작 순간의 최소 Hz는 차단 조건이 아니다 |
| `HIL_PREPOSITION_MARKER` | `$XDG_RUNTIME_DIR/hil-preposition.ready` | `run_hil_preposition.sh`와 actor가 공유하는 0600 proof. 보통 직접 지정하지 않는다 |
| `HIL_PREPOSITION_MARKER_MAX_AGE_S` | `900` | marker 최대 수명. 1~3600초만 허용 |
| `SKIP_ROS_CHECKS` | (미설정) | `1`이면 [7][8][9] 건너뜀. `--arm`과 같이 쓰면 **즉시 FAIL** |
| `ROS_SETUP` | `/opt/ros/humble/setup.bash` | |
| `HIL_STARTUP_DEADMAN` | **`disengaged`** 🆕 | deadman gate가 요구하는 상태. `disengaged`\|`engaged`만 유효하고 **오타는 fail-closed로 거부**된다(gate 없음으로 조용히 넘어가지 않는다). `engaged`가 옛 commissioning 동작이다 → §1.4 |
| `HIL_ACTOR_EXIT_MAP` | `1` 🆕 | `0`이면 rc 75 승격을 끄고 actor의 raw rc를 그대로 낸다 → §6.1 |

📌 2026-07-29 확인: 위 기본값은 `ros2_ur_ws/run_hil_actor.sh`의 상단 기본값 블록과 일치한다.
🆕 두 줄은 2026-07-30 저녁에 추가됐다.

`run_hil_session.sh` 자신이 읽는 변수는 따로 있다 (actor에게 넘기지 않는다):

| 변수 | 기본값 | 비고 |
|---|---|---|
| `HIL_STARTUP_DEADMAN` | `disengaged` | 검사한 뒤 `run_hil_actor.sh`로 **export** 해서 세 gate가 같은 값을 쓰게 한다 |
| `DEADMAN_WAIT_S` (별칭 `ENGAGE_WAIT_S`) | `120` | 요구 상태를 폴링으로 기다리는 시간. 초과해도 진행하고 **preflight `[11]`이 같은 조건으로 FAIL 시킨다.** 옛 이름 `ENGAGE_WAIT_S`는 alias로 남아 있다 |
| `HIL_ACTOR_RETRY` | `1` | rc 75 뒤 3~5단계 자동 재시도 → §6.1. `0`이면 끈다 |
| `HIL_ACTOR_RETRY_MAX` | `3` (최대 10) | 실기 팔을 자동 재arming하므로 상한이 걸려 있다 |
| `HIL_HARDWARE_RECYCLE_WAIT_S` | `900` | 조작자가 Terminal 2를 껐다 켜는 것을 기다리는 최대 시간 |
| `HIL_RETRY_RESUME_DELAY_S` | `5` | 3단계(팔이 움직인다) 재시작 전 취소 가능한 카운트다운 |

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
| `--no-classifier-sidecar` | off | **킬 스위치**(config보다 우선). 모든 transition이 `classifier_evaluated=false`가 된다. AUTO에서는 classifier reward가 0이지만, MANUAL의 `MARK SUCCESS`는 sidecar 없이도 operator reward/done을 만들 수 있다. sidecar 지연 비용만 분리할 때 쓴다 |
| `--classifier-sidecar-interval N` | **5** (10 Hz → 약 2 Hz) | N 스텝마다 최대 1회 부착. **성긴 것이 의도다** — 판정 깜빡임을 줄이고 큐브를 놓은 뒤 장면이 가라앉을 시간을 준다. `1`은 매 스텝 채점 = 지연 비용 최대 |
| `--classifier-stationary-speed-max M/S` | **0.05** ⚠️ | TCP 속도가 이 값 미만일 때만 부착 (움직이는 중의 블러 프레임 차단). **config 주석이 이 값을 PLACEHOLDER로 명시한다** — 녹화 take에서 "팔이 가라앉은 뒤 실제로 머무는 속도"를 재서 정해야 한다 |
| `--classifier-escalate-probability P` | **0.05** | **확률적 추첨이 아니다.** 분류기 확률이 P 이상이면 interval을 버리고 **매 스텝** 채점한다. 성공 임계(현재 0.5)를 실제로 넘는 스텝을 최대 interval−1 스텝 놓치지 않으려는 것이라 **`DEFAULT_REWARD_THRESHOLD`보다 낮게** 둔다 |

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
| 11 🆕 | **deadman 채널 생존 증명** — `/hil/deadman`에서 fresh heartbeat 3개 연속, 상태는 `HIL_STARTUP_DEADMAN`이 정한다(기본 **DISENGAGED**) | FAIL. `--arm` 없으면 불필요(INFO), `--fake-env`/overlay 실패면 SKIP → §1.4 |

`--dry-preflight`의 모든 점검은 **읽기 전용**이다. 실제 `--arm`은 [1]~[11]이
통과한 다음에만 publisher 수를 다시 세고 strict controller switch를 수행한다. `[ARM]` 직전에
**[11]과 같은 검사를 한 번 더** 돌린다([11] 이후 GUI가 죽었거나 조작자가 상태를 바꿨을 수
있다). switch 후 `STJC=inactive`, `FPC=active`, publisher 0을 다시 확인하며 하나라도
실패하면 actor를 exec하지 않는다. 이 handoff는 reset/preposition 이동을 자동 호출하지 않는다.

### 1.4 🆕 deadman gate — **시작할 때 ENGAGE를 누르지 않는다** (2026-07-30)

> ### 🛑 절차가 바뀌었다
> 이 문서의 이전 판은 곳곳에서 *"HIL GUI를 먼저 ENGAGE하고 GELLO를 RESET anchor에
> 고정한다"*고 지시했다. **더 이상 그렇게 하지 않는다.** 세 gate가 전부 반대 상태를
> 요구한다. 아래 「이전 판(보존)」 문구를 남겨 두지만 **따르지 말 것.**
>
> **이전 판(보존):** *"2) HIL GUI를 먼저 ENGAGE하고 GELLO를 RESET anchor에 고정한다.
> dry-preflight도 연속 ENGAGED heartbeat 3개를 요구한다. … controller switch 직전
> ENGAGED heartbeat를 한 번 더 검사한다."*

**gate의 의미는 바뀌지 않았다 — 요구 상태만 뒤집혔다.** 이것은 여전히 **채널 생존 증명**이다:
"GUI가 살아서 `/hil/deadman`에 20 Hz로 유효 heartbeat를 내고 있다"를 팔을 프로그램에 넘기기
전에 확인한다. 연속 fresh heartbeat 3개 요구도 그대로다. 바뀐 것은 **어느 상태여야 하는가**다.

| | 이전 | 지금 (기본) |
|---|---|---|
| 요구 상태 | `engaged=1` | **`engaged=0`** |
| 조작자가 하는 일 | ENGAGE를 두 번 눌러 arm하고 GELLO를 잡은 채 대기 | **아무것도 안 한다.** GUI만 떠 있으면 된다 |
| handoff 직후 GELLO를 잡으면 | 팔이 따라온다 | **따라오지 않는다** |

세션은 policy 제어로 시작하므로 handoff 시점에 ENGAGED일 이유가 없고, DISENGAGED로 넘기면
handoff 직후 누군가 GELLO를 건드려도 팔이 움직이지 않는다 — **더 안전한 쪽**이다.

**세 곳이 같은 값을 쓴다** (`run_hil_session.sh`가 검사 후 export 한다):

| # | 어디 | 무엇 |
|---|---|---|
| 1 | `run_hil_session.sh`의 폴링 루프 | 요구 상태가 될 때까지 최대 `DEADMAN_WAIT_S`(120 s) 대기. 초과하면 경고만 하고 진행한다 — 강제는 아래 둘이다 |
| 2 | `run_hil_actor.sh` preflight `[11]` | FAIL |
| 3 | `run_hil_actor.sh` `[ARM]` 직전 재검증 | 실패하면 **controller를 전환하지 않고** rc 1로 종료 |

**되돌리기 / 도구:**

```bash
# 옛 commissioning 동작 (ENGAGED 요구) — 세 gate가 함께 바뀐다
HIL_STARTUP_DEADMAN=engaged ./run_hil_session.sh

# 단독 확인 (읽기 전용). rc 0=OK, 1=상태 틀림/malformed, 2=ROS/토픽 없음
cd $WT/ros2_ur_ws
python3 _hil_deadman_check.py --topic /hil/deadman --samples 3 --timeout 2.0 \
  --require disengaged
```

`--require`의 기본값은 스크립트 자체에서는 **`engaged`**(옛 동작)이지만, 두 wrapper는 항상
명시적으로 넘기므로 운영 경로의 기본은 `disengaged`다. `HIL_STARTUP_DEADMAN`에 오타를 내면
두 wrapper 모두 **fail-closed로 거부**한다 — "gate 없음"으로 조용히 해석되지 않는다.

> ⚠️ **받아들인 대가:** DISENGAGED로 시작하면 policy가 팔을 몰기 **전에** ENGAGE 전이가 한
> 번도 실행되지 않는다. GUI는 actor status가 `POLICY_RUNNING`/`HUMAN_INTERVENTION`/`HOLD`가
> 되기 전까지 ENGAGE 버튼을 **비활성화**하므로(`engage_button_enabled`), 그 세션의 첫 ENGAGE는
> **policy가 이미 움직이는 중**에 일어난다. 조작자에게 이 사실을 보이고 명시적으로 수정하지
> 않기로 했다 — `08_OPEN_GAPS.md` G22의 「받아들인 대가」 항목에 기록돼 있고, 탈출구는
> `HIL_STARTUP_DEADMAN=engaged`다.

---

## 2. 학습 서버(`junhyeong_ai`) 쪽 — Stage A/B 공통 전제

> ### 🟢 정상 운용에서는 이 절을 **읽을 필요가 없다**
> Terminal 1의 `./run_hil_server.sh`가 exact learner 재사용/기동과 터널을 전부 한다.
> **환경변수를 하나도 주지 않는다** — 기본값이 `junhyeong_ai`다.
> 읽기 전용 확인은 `./run_hil_server.sh --check`.
> 아래는 그 wrapper가 고장났을 때의 수동 진단 경로와, **2026-07-27 kanu 세션의 기록**이다.

### 2.1 📌 2026-07-27 세션의 **기록** — 재입력용 설정값이 아니다

> ## 🛑 이 표에서 값을 복사하지 마라
> 아래는 그날 그 세션이 무엇을 썼는지의 **기록**이고, 그날의 서버는 **`kanu`**였다.
> 커밋·GPU·체크포인트 SHA는 **전부 그 뒤에 바뀌었거나 바뀔 수 있다.**
> 각 항목 옆에 "지금은 어떻게 확인하나"를 적었다.

| 항목 | 📌 그날의 값 (🗄️ kanu) | 지금은 (`junhyeong_ai`) |
|---|---|---|
| 서버 checkout | `/tmp/gello-hil-rl-receive-server-v2` @ `5fb716b` (kanu의 **worktree**) | 🗄️ **그 경로도 그 형태도 은퇴했다.** 현재 HIL checkout은 **`/home/junhyeong/gello_software_runtime`**(독립 clone)이고, `run_hil_server.sh`가 linked worktree를 **거부**한다(`--git-dir == --git-common-dir` 검사). 커밋은 laptop3와 대조한다 — 그 대조를 `run_hil_server.sh`가 ssh로 자동 수행한다 |
| overlay venv | `/tmp/gello-hil-rl-receive-overlay-v2` | 🗄️ kanu 전용. 지금 learner python은 **`/home/junhyeong/miniconda3/envs/il/bin/python`**(py 3.10.20, jax 0.5.3 exact-pin, fail-closed). 옛 overlay는 **재사용 금지**였다(protobuf 3.20.3이 wandb를 깨뜨린다 → `05` §1.2) |
| GPU | `CUDA_VISIBLE_DEVICES=7` (kanu는 A4000 **×8**) | **`junhyeong_ai`는 GPU가 1장뿐이다** — RTX 5070 Ti 16 GB, sm_120, **index 0**. `HIL_GPU_INDEX` 기본값도 `0`이다. ⚠️ 그 1장을 **다른 사람과 공유**하므로 `nvidia-smi`로 점유를 먼저 본다 |
| 서버 포트 | `50053` (loopback bind) | 그대로 (코드 기본값) |
| 로컬 터널 입구 | `50153` → 원격 `50053` | 그대로 (`run_hil_actor.sh`의 `SERVER_PORT` 기본값도 50153). 🟢 **서버 이전으로 바뀌지 않았다** |
| 분류기 checkpoint SHA-256 | **`512b6575…62846d`** | ✅ `checkpoint_150`의 **디렉터리** digest. 경로만 옮겨졌다 → `junhyeong_ai:~/hil-serl-data/classifier_ckpt/checkpoint_150`. **SHA 핀은 하나도 안 바뀌었다**(경로 핀이 아니라 내용 핀이라 이전이 정확했는지를 오히려 검사해 줬다). *(그날 기록된 `e329986b…d7a997`는 recall 0%짜리 폐기 체크포인트였다 → `08` G19)* |
| observation schema hash | `3459098d…0352903` | 📌 불변. `run_hil_server.sh`가 이 값을 핀으로 들고 있고 핸드셰이크가 exact 비교다. **그래도 양쪽에서 출력해 대조한다** (`05` §4.2) |

> 로컬 포트가 50053이 아니라 **50153**인 이유: 그날 로컬 50053이 다른 프로세스에
> 잡혀 있었다. 터널 로컬 쪽만 바꾸고 원격 쪽은 50053 그대로 둔다. 이 배치가 코드
> 기본값이 됐으므로 그대로 쓴다.

**학습 서버 쪽 실제 경로** (`/home/laptop3/gello_software`는 서버에 **없다**):

| 무엇 | 경로 (`junhyeong_ai`) |
|---|---|
| HIL checkout (**유일**, 독립 clone) | `/home/junhyeong/gello_software_runtime` |
| 데이터·모델 뿌리 | `/home/junhyeong/hil-serl-data/` |
| ↳ canonical offline demo | `~/hil-serl-data/demos/cube_in_cup_20260720_success_23takes.pkl` |
| ↳ 운영 reward classifier | `~/hil-serl-data/classifier_ckpt/checkpoint_150` |
| ↳ learner run root | `~/hil-serl-data/runs/<run_id>/` |
| ↳ classifier 재학습 원재료 | `~/hil-serl-data/datasets/` |
| ↳ 🗄️ kanu 시절 이력 (읽기 전용) | `~/hil-serl-data/archive/` |
| python | `/home/junhyeong/miniconda3/envs/il/bin/python` |

> 🚫 **`/home/junhyeong/gello_software`(뒤에 `_runtime`이 없는 것)는 다른 사람의 작업
> 트리다.** 읽지도 쓰지도 말 것. HIL이 쓰는 것은 **`gello_software_runtime`**뿐이다.

> ### 🗄️ 2026-07-27 판(보존) — **kanu 쪽 경로. 지금 따라가지 말 것**
> | 무엇 | 경로 (kanu) |
> |---|---|
> | 분류기 학습처 (YWhero/hil-serl fork, `agent/cube-in-cup-classifier` @ `d753571`) | `~/workspace/youngwoong/hil-serl` |
> | 학습 데이터 + 07-27 체크포인트 | `~/workspace/youngwoong/dataset/cube_in_cup_all3/` |
> | ZMQ 뷰어 + 07-24 체크포인트 | `~/workspace/youngwoong/gello_software_remote_classifier` |
> | 이 리포의 Kanu 전용 worktree | `/tmp/gello-hil-rl-receive-server-v2` (**은퇴**) |
>
> kanu에서는 **아무것도 지우지 않았다** — 이전은 전부 복사였다. 그래서 위 경로들은
> 아직 kanu에 실재하지만, **운영 경로가 아니다.** 특히 classifier가 FM/diffusion 스택
> 디렉터리 안에 있던 것이 **`HIL_REMOTE_DATA_ROOT` 하나로 kanu를 기술할 수 없는 이유**이고,
> 그래서 `run_hil_server.sh`는 kanu를 아예 구동하지 못한다(의도된 fail-closed).

### 2.2 서버에서 수동으로 receive server 띄우기 (진단 전용)

> 🛑 **정상 운용 경로가 아니다.** 운영은 `./run_hil_server.sh`(learner 서버)다.
> 아래는 **receive-only 마일스톤 서버**를 손으로 띄우는 진단용 CLI다 — 정책이 없고
> 항상 zero action을 낸다(§2.2 아래의 "서버 종류가 두 가지다" 박스).

```bash
# junhyeong_ai에서 — 먼저 확인 3가지
cd /home/junhyeong/gello_software_runtime
git rev-parse --short HEAD          # 랩톱의 HEAD와 같은가? (특정 SHA를 기대하지 말 것)
nvidia-smi                          # GPU가 1장뿐이다(index 0). 다른 사람이 쓰고 있지 않은가?
ss -ltn | grep 50053                # 이미 잡혀 있지 않은가?

export XLA_PYTHON_CLIENT_PREALLOCATE=false   # 필수. 안 하면 JAX가 카드를 통째로 선점한다

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=serl_ur_infra:third_party/hil-serl/serl_launcher \
/home/junhyeong/miniconda3/envs/il/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_server.py \
  --host 127.0.0.1 \
  --port 50053 \
  --checkpoint /home/junhyeong/hil-serl-data/classifier_ckpt/checkpoint_150 \
  --expected-checkpoint-sha256 <그 체크포인트의 sha256> \
  --reward-model-id cube-in-cup-all3-ckpt150+sidecar-v1 \
  --require-jax-backend gpu
```

> ### 🗄️ 이전 판(보존) — kanu 기준. **경로가 전부 은퇴했다**
> ```
> cd /tmp/gello-hil-rl-receive-server-v2
> CUDA_VISIBLE_DEVICES=<비어 있는 카드>            # kanu는 A4000 ×8이었다
> /tmp/gello-hil-rl-receive-overlay-v2/bin/python
> ```
> `/tmp/gello-hil-rl-receive-server-v2`는 worktree였고 지금은 **형태 자체가 거부된다.**

> ### 🔧 threshold production 계약 (2026-07-30)
> 코드 기본값과 `run_hil_server.sh`의 production pin은 모두 **`0.5`**다.
> `0.2`는 2026-07-29의 과거 lineage 값이다. 위 수동 진단 CLI는 기본값을 사용하지만,
> 정상 운용에서는 긴 CLI를 복사하지 말고 wrapper가 exact `0.5` 계약을 검사하게 한다.
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
🗄️ **두 문서는 절 제목까지 kanu 기준이고 호스트·경로가 낡았다** — 인자의 *의미*만 근거로
쓰고, 호스트·경로는
[`serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](../../serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md)가
우선한다.

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

> 🟢 **정상 운용에서는 손으로 열지 않는다** — Terminal 1의 `run_hil_server.sh`가 같은 터널을
> 열고 소유한다. 아래는 wrapper 없이 진단할 때만 쓴다. **터널 소유권이 왜 중요한지는 §5.4.**

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50153:127.0.0.1:50053 junhyeong_ai
```

확인 (다른 터미널):

```bash
ss -ltnp | grep 50153      # ssh가 127.0.0.1:50153에 LISTEN 중이어야 한다
```

`ExitOnForwardFailure=yes`가 핵심이다 — 포워딩에 실패하면 ssh가 조용히 붙어 있지 않고 죽는다.
`run_hil_server.sh`는 여기에 `BatchMode=yes`까지 얹으므로 **키 인증이 없으면 비밀번호를
묻지 않고 즉시 실패**한다.

> 🗄️ **이전 판(보존):** 마지막 인자가 `kanu`였다. 2026-07-31 이전으로 **`junhyeong_ai`**다.
>
> 🛑 **옛 명령을 그대로 붙여넣지 마라 — 이 실수는 조용하다.** kanu의 07-30 learner는 아직
> 살아 있고, 그것도 `run_rlpd_learner_server.py`라 **model_id·reward model id·schema hash가
> 전부 같다.** 즉 actor 핸드셰이크는 **통과한다.** 두 learner를 갈라 보는 것은
> `run_hil_server.sh`의 `validate_process_contract`(run 이름 `-hil-5000` vs 옛 `-kanu-5000`
> 등을 exact 비교)이고, 터널을 손으로 열면 **그 검사를 통째로 건너뛴다.**
> 그러면 실기 전이가 은퇴한 lineage로 흘러 들어간다. 터널은 Terminal 1이 열게 할 것.

---

## 3. Stage A — fake-env로 학습 서버 왕복 (**로봇 사용 안 함**)

목적: **로봇을 전혀 건드리지 않고** wrapper 체인 · 관측 스키마 · gRPC 계약 ·
버퍼 삽입까지 왕복을 증명한다. `fake_env=True`면 `GelloIntervention` 래퍼가 아예
붙지 않고 ROS 백엔드도 열리지 않는다 (`ur_experiments/cube_in_cup.py`
`get_environment()`의 `if not fake_env:` 분기).

### 3.1 절차 (복붙)

```bash
# T0 — 터널 (§2.4). 그대로 띄워 둔다. 평상시에는 Terminal 1이 대신 연다.
ssh -N -T -o ExitOnForwardFailure=yes -L 127.0.0.1:50153:127.0.0.1:50053 junhyeong_ai
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

GUI는 별도로 `/hil/actor_status` JSON을 읽어 control owner, episode/step,
classifier의 마지막 `p(success)`/threshold, terminal reason을 표시한다. terminal 뒤
`WAIT_HOME_APPROVAL`에서는 같은 버튼이 **APPROVE HOME**으로 활성화되고, HOME 뒤
`WAIT_SCENE_READY`에서는 **START / NEXT ITERATION**으로 바뀐다. START/NEXT는 deadman을
DISENGAGE한 뒤 `/hil/scene_ready` Trigger를 호출한다. early/ENGAGED/stale 요청은 actor가
거절한다. WAIT status는 0.5 s마다 재발행하므로 WAIT 중 GUI를 재시작해도 버튼이 복구된다.
새 actor `run_id`가 보이면 GUI는 이전 비동기 요청을 폐기한다. `/hil/deadman`은 정확히
두 필드 `[0.0|1.0, gain]`와 gain `[0.10, 1.00]`만 유효 heartbeat로 인정한다. 빈 배열,
NaN, fractional engaged 같은 malformed 메시지는 freshness를 갱신하지 않는다. 이
status/service는 `/hil/deadman`의 frozen payload를 변경하지 않는다.

성공 모드는 GUI에서 바꾼다. 시작값은 `MANUAL`이고 actor status가 선택 상태의 권위다.
MANUAL에서도 서버 classifier sidecar를 끄지 않는다. 숫자와 verdict는 계속 보이고 replay에도
classifier 결과가 남지만 classifier-positive만으로 reward/done이 되지 않는다. 사람이
`MARK SUCCESS`를 누르면 현재 `(run_id, episode_id)`에 한 번만 operator success가 들어간다.
AUTO로 전환하면 수동 성공 버튼은 비활성화되고 서버의 strict `p(success) > threshold`만
성공을 만든다. 현재 compact GUI는 classifier headline 16 pt이고 status/value 열을 왼쪽에
정렬해 이전 900 px급 세로 레이아웃보다 짧다.

공식 `third_party/hil-serl` 원본(`c32939bcc`)도 확인했다. 원본에는 SpaceMouse action
replacement, task-local reward classifier wrapper, `env.reset()`과 일부 task-specific
terminal prompt는 있지만, 범용 actor GUI·원격 classifier probability overlay·
`HOME → WAIT_SCENE_READY → operator resume` 상태기계는 없다. 또한 원본 Agentlace/local
classifier 흐름은 이 저장소의 **서버 authoritative**(현재 `junhyeong_ai`) gRPC
reward/termination 계약과 다르므로
그 코드를 그대로 끼우지 않고, 원본의 terminal→reset 순서만 현재 환경에 맞춰 유지한다.

> 현재 기준선은 infra `768 passed, 11 skipped, 1 xfailed`(actor venv), ROS package
> `489 passed`(시스템 python3 + overlay)다. **두 수를 합치지 말 것 — 인터프리터도
> PYTHONPATH도 다르다.** 서버 이전으로 바뀌지 않았다(2026-07-31 재확인).
> 🗄️ 이전 판은 07-30 중간값 `614` / `461`을 적고 있었다.
> clean build, 격리 DDS late-join→WAIT 수신→Trigger 왕복까지 통과했다.
> 실제 UR7e에서는 episode limit→`WAIT_HOME_APPROVAL`과 classifier 표시, schema-3 online
> transition/learner update까지 관측했다. MANUAL `MARK SUCCESS`로 끝낸 episode의 one-shot
> provenance만 다음 실기에서 한 번 재확인한다.

**T0 — 터널** (§2.4)

**T6 — preflight → actor**

```bash
cd $WT/ros2_ur_ws

# 1) actor를 내린 채 RESET 자세를 검증/사전 배치한다.
#    이미 0.10 rad 안이면 이동 없이 proof만 만들고 종료한다.
#    멀면 기본값은 체크리스트 출력 뒤 즉시 JTC 궤적을 시작한다.
#    PREPOSITION_DELAY_S=5 또는 PREPOSITION_CONFIRM=1은 명시적 opt-in이다.
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
RESET 자세에서 벗어났으면 전환 전에 실패한다. handoff 뒤에는 GUI가
`WAIT_SCENE_READY`를 표시한다. 실제 장면을 배치한 다음 `START / NEXT ITERATION`을 눌러야
`BeginEpisode`와 첫 policy action이 발생한다. success 또는 episode terminal이면 actor는 먼저
`WAIT_HOME_APPROVAL`에서 terminal pose를 hold한다. `APPROVE HOME` 뒤에만 HOME으로 이동하고,
그 다음 `WAIT_SCENE_READY`에 들어간다. 두 WAIT 중에는 Step RPC/transition이 없다.

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
여기 표시되는 Hz는 정보값이며 session 시작 시 Qt·DDS·RealSense 부하 때문에 순간적으로
낮아져도 차단하지 않는다. 정상 steady-state 최소 rate는 Terminal 2 하드웨어 launcher와
`launch_cameras.sh`가 최초 기동 때 이미 확인한다.
GUI에 cam1 영상이 보이더라도 `[7]`의 cam1만 실패하면 우회하지 말 것. 기존 GUI reader는
살아 있으나 **새 actor reader에는 큰 이미지가 전달되지 않는 Fast DDS writer 상태**일 수
있다. 카메라 터미널에서 `Ctrl-C`로 두 카메라를 정상 종료한 뒤
`./launch_cameras.sh`를 다시 띄우고 `[7]`을 재검증한다.

### 4.4 🔎 실제 `--arm` episode에서 classifier sidecar와 GUI verdict를 확인한다

**이것이 G15 수정의 가장 값싼 실기 증거다.** no-arm actor는 이제 의도적으로
`BeginEpisode` 뒤 종료하므로 transition/classifier sidecar를 만들지 않는다. 따라서 sidecar의
실제 gRPC 왕복은 `--arm` episode 안에서 확인해야 한다. controller handoff를 위해 GUI를 먼저
ENGAGE하지만, actor가 HOME/WAIT에 들어간 뒤에는 장면을 배치하고 `START / NEXT ITERATION`을
눌러야 한다. 이 버튼이 fresh DISENGAGED를 확인한 뒤 policy-first episode를 연다.

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
   (랩톱 CPU는 `run_classifier_viewer.sh`, 서버 GPU + 터널은 `run_remote_classifier_viewer.sh` —
   그 스크립트의 `CLASSIFIER_SSH_HOST` 기본값도 **`junhyeong_ai`**로 옮겨졌다.
   kanu 사본을 보려면 `CLASSIFIER_SSH_HOST=kanu`로 override한다).
   두 카메라의 값을 적어 둔다.
3. HIL GUI를 **먼저 ENGAGE**하고 GELLO를 RESET anchor에서 정지시킨다. §4.2의
   preposition과 `--dry-preflight --arm`을 통과한 뒤, 부착을 매 스텝으로 올린 `--arm`
   smoke를 시작한다. GUI가 `WAIT_SCENE_READY`를 표시하면 장면을 확인한 뒤
   `START / NEXT ITERATION`을 누른다. 이 명령은 실제 controller를 전환하고 policy를
   움직이므로 E-STOP을 손에 두고, 확인 표본을 얻으면 Ctrl-C로 종료한다:

   ```bash
   cd $WT/ros2_ur_ws
   EXPECTED_MODEL_ID=<서버가 광고하는 값> \
     ./run_hil_actor.sh --arm --deadman topic --classifier-sidecar-interval 1
   ```

   policy 동작 중 필요하면 ENGAGE로 GELLO 개입하고, 다시 DISENGAGE하면 policy로 돌려준다.
   heartbeat가 끊기면 actor는 policy fallback 없이 fail-stop해야 한다.

4. 서버가 광고한 계약을 기동 로그에서 확인한다 — `rlpd_receive_server_ready` 한 줄에
   전부 들어 있다: `reward_model_id`, `checkpoint_sha256`, `threshold`,
   **`classifier_input_contract`**, **`success_confirmations`**.
   `success_confirmations`가 1이 아니면 **이 비교는 성립하지 않는다**(평활이 켜진 것).
5. actor stdout에 `classifier sidecar not built`가 없어야 하고, server에는 연속 미분류
   경고나 classifier fault가 없어야 한다. 현재 actor는 Ctrl-C 때 summary를 출력하지 않으므로
   `sidecar n=`을 짧은 run의 강한 판독구로 쓸 수 없다(짧은 `max_steps` CLI와 함께 후속).

> ### 2026-07-30 변경 — GUI에서 마지막 classifier 결과를 볼 수 있다
> `classifier_probability`는 `TransitionOutcome`으로 actor에 돌아오며, actor가
> `/hil/actor_status`로 GUI에 전달한다. GUI는 sparse/unscored 현재 step과 마지막 evaluated
> `p(success)`/threshold/env step을 구분해 표시한다. 다만 이 UI는 **영구 로그가 아니므로**
> 장시간 실험의 사후 감사를 대신하지 않는다.
> 서버가 스스로 말하는 경우는 두 가지뿐이다: **연속 100건이 미분류**일 때, 그리고
> **한 세션이 단 한 건도 분류되지 않은 채 끝났을 때**
> (`rlpd_receive_server.py::RewardTransitionFinalizer`의 경고).
> `--checkpoint-path`의 로컬 pickle도 이 용도로는 못 쓴다 — 아래 §1.2 각주를 볼 것.
>
> standalone viewer와 actor GUI를 같은 정지 장면에서 비교하면 server 숫자 대 viewer 숫자의
> 실기 판독이 가능하다. JPEG 재인코딩 때문에 비트 단위 일치를 요구하지 말고 §4.4의 측정
> 오차 범위를 적용한다. 오프라인 전처리 등가성은 `tests/test_classifier_sidecar.py`가 강제한다.
>
> MANUAL/AUTO는 classifier 실행 여부가 아니라 **누가 success terminal 권한을 갖는가**만
> 바꾼다. 두 모드 모두 probability/threshold를 계속 전송하고 표시한다. 현재 threshold는
> 0.5이며 strict 비교이므로 `p == 0.5`는 성공이 아니다.

---

## 5. 실패 대응표

| preflight 메시지 | 조치 |
|---|---|
| `actor venv python이 없다` | `ACTOR_VENV` 확인. 없으면 `python3 -m venv --system-site-packages /home/laptop3/venvs/gello-hil-actor` 후 `serl_ur_infra/requirements-grpc.lock` 설치 |
| `grpcio 1.30.2 — 손상된 것으로 확인된 바로 그 버전` | venv 안에 최신 grpcio 재설치. **절대 이 상태로 띄우지 말 것** (조용히 영구 정지) |
| `cygrpc.CompletionQueue()가 20초 안에 돌아오지 않았다` | 위와 같은 증상의 직접 증거. grpcio 재설치 |
| `… 이 저장소 밖에서 해석됨` | 다른 checkout에 editable 설치된 `serl-ur-infra`가 이기고 있다. 지금 트리에서 스크립트를 실행하고 있는지 확인 |
| `ur_experiments: 찾을 수 없음` | 지금 checkout에 `serl_ur_infra/ur_experiments/`가 없다 = 브랜치가 틀렸다 |
| `TCP 127.0.0.1:50153 연결 실패 — ConnectionRefusedError` (preflight `[6]`) | **터널이 죽었다. learner는 대개 멀쩡하다.** 가장 흔한 원인은 **Terminal 1을 Ctrl-C 했거나 그 창을 닫은 것**이다. 조치는 Terminal 1에서 `./run_hil_server.sh`를 **다시 실행**하는 것 하나뿐이다 — 살아 있는 learner를 재사용하고 새 터널만 연다. 🛑 **learner를 죽이거나 새로 띄우려 하지 말 것.** 전문 → **§5.4** |
| `cam1 … fresh advancing samples … 12.0s` | 순간 Hz 저하는 더 이상 실패가 아니다. 12초 동안 새 timestamp가 실제로 오지 않은 경우이므로 카메라 로그와 `ros2 topic info`를 확인하고 필요할 때만 카메라를 재기동 |
| `예상 밖 controller 조합` | STJC/FPC가 둘 다 active 또는 둘 다 inactive다. 수동으로 우회하지 말고 driver/이전 actor 종료 상태를 확인 |
| `arm handoff proof 검증 실패` | actor를 내리고 `./run_hil_preposition.sh`를 실행. 이미 RESET 0.10 rad 안이면 움직이지 않고 새 marker만 만든다 |
| `퍼블리셔가 N개 있다` (FAIL) | 텔레옵 브리지/다른 러너가 살아 있다. `ros2 topic info -v /forward_position_controller/commands`로 범인을 찾아 끄고 재실행 |

### 5.1 `RESET_MAX_DIST_RAD` 게이트 (실기에서 자주 만난다)

actor의 승인 없는 일반 리셋은 **현재 자세 → `RESET_JOINTS`** 사이를 250 Hz 업샘플러로
그대로 쓸고 지나간다. 그래서 그 거리가 `RESET_MAX_DIST_RAD`를 넘으면 에러로 멈춘다.
다만 terminal 뒤 GUI `APPROVE HOME`을 받은 경로는 `operator_approved_home=True`로 이 거리
검사를 우회한다. 즉 episode reset에서 0.9 rad가 넘었다고 actor를 죽이는 대신 먼저 사람에게
HOME 이동 승인을 받고, 승인 뒤 기존 HOME 스트림을 수행한다.

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
보간 이동한다. 현재 wrapper 기본값은 체크리스트 뒤 즉시 진행 서비스를 호출하며 별도 GO
입력이 없다. 이 preposition wrapper에는 actor의 0.9 rad 거리 제한과 별개인 최대 거리 gate가
없으므로, 경로가 크면 출력된 current/target 표를 보고 조작자가 중단해야 한다. 취소 가능한
5초 창은 `PREPOSITION_DELAY_S=5`, 옛 GO 입력은 `PREPOSITION_CONFIRM=1`로만 켠다.

성공 시 스크립트는 현재 사용자만 읽고 쓸 수 있는 proof marker를 만든다. marker는 기본
15분만 유효하며 RESET 값, ROS domain, 허용오차를 담는다. `run_hil_actor.sh --arm`은 marker만
믿지 않고 `/joint_states`를 다시 읽어 관절별 최대 오차 0.10 rad 이하를 재확인한다.

> ⚠️ 사전 배치 도구가 도는 동안에는 그것이 팔의 명령 소유자다. **끝난 뒤 반드시 내리고**
> GUI를 ENGAGE한 뒤 `./run_hil_actor.sh --dry-preflight --arm --deadman topic`에서
> [8] controller 쌍, [9] 퍼블리셔 0, [10] proof + live pose, [11] 연속 ENGAGED
> heartbeat를 모두 확인한 다음 actor를 띄운다.

게이트 값을 **키워서 통과시키지 말 것** — 그러면 리셋이 작업 공간을 가로질러 쓸고 간다.

---

## 5.2 🆕 `END EPISODE` — 망친 episode를 지금 끝낸다 (2026-07-30 저녁)

GUI 세 번째 버튼(빨강, **두 번 클릭**). 서비스는 `/hil/abort_episode`(`std_srvs/Trigger`).
episode 도중 아무 때나 눌러 **성공 판정 없이** 지금 끝내고, 평소와 같은
`WAIT_HOME_APPROVAL` → `APPROVE HOME` → HOME → `WAIT_SCENE_READY` 경로를 탄다.
`terminal_reason`은 `OPERATOR_ABORT`. MANUAL/AUTO **양쪽에서** 받는다 — 망친 episode를
끝내는 것은 성공 주장이 아니기 때문이다.

### 🛑 무엇이 버려지고 무엇이 안 버려지나 — 조작자는 이걸 반드시 알아야 한다

**아무것도 버려지지 않는다.** proto에 cancel/retract RPC가 **없고**(`Health` /
`GetServerInfo` / `GetBufferStatus` / `BeginEpisode` / `Step` 5개뿐), 서버는 `Ack`를 만들기
**전에** Step 핸들러 안에서 동기적으로 replay store에 insert한다. 즉 `network.step()`이
돌아온 시점에 그 행은 **이미 learner가 gradient batch를 뽑는 버퍼 안에 있다.**

> 클릭 시점까지의 모든 전이는 — 충돌도, 이상한 자세도 — **정상적으로 학습된다.**
> 게다가 그 순간 조작자는 대개 GELLO를 잡고 있으므로 그 행들은 `intervened=1`이라
> **두 버퍼 모두에** 들어가고, RLPD 50:50 분할이 **가중치를 올려 준다.**

버튼이 실제로 사주는 것은 둘뿐이다:

1. **step limit까지 안 기다리고 지금 끝내는 것**
2. **조작된 terminal 대신 정직한 bootstrap-safe truncation** — `done=False, truncated=True,
   masks=1.0, success=False`. 이게 없으면 critic이 "조작자가 포기한 지점에서 세상이 끝난다"고
   배운다(→ `08_OPEN_GAPS.md` **G35**가 바로 그 병이다).

⚠️ 사후 감사는 아직 불가능하다 → **G36**.

### ⏱️ 즉시가 아니다

actor는 토큰을 **iteration당 두 번** 읽는다 — `env.step` **직전**(대기 중이던 policy action이
폐기된다)과 **직후**(방금 실행된 전이가 truncated로 기록된다). 최악은 **루프 한 주기**
(실측 평균 512 ms, 최대 854 ms)다.

GUI는 Trigger **전에** 데드맨을 먼저 놓아 GELLO 추종을 **~33 ms**에 멈춘다.
🛑 **그러나 데드맨은 policy 경로를 전혀 게이팅하지 않는다** — 배경 follower와
`GelloIntervention.action()`만 본다. **policy가 몰고 있을 때 데드맨을 놓는 것은 아무것도
멈추지 않는다.** 버튼 문구가 이것을 말하도록 되어 있다.

### 거절되는 경우

* 활성 상태(`POLICY_RUNNING`/`HUMAN_INTERVENTION`/`HOLD`)가 아닐 때
* **`terminal_reason`이 이미 세워졌을 때** — episode가 방금 끝난 창이다. 여기서 수락하면
  다음 publish의 boundary 규칙에 토큰이 조용히 버려져 **거짓 SUCCESS가 그대로 남는다.**
  그래서 보이는 거절로 바꿨다:
  `episode 0 has already ended (SUCCESS); it is too late to abort it. Press APPROVE HOME, then abort the next episode.`
* 이미 abort가 큐에 있을 때

---

## 5.3 🆕 충돌 복구 — 세션이 하드웨어 재기동에서 살아남는다 (2026-07-30 저녁)

**예전:** 충돌 → actor 사망 → `run_hil_session.sh`의 EXIT 트랩이 GUI와 카메라까지 무조건
정리 → 세션 전체를 다시 만든다.
**지금:** 카메라·GUI·터널은 그대로 살아 있고 **3~5단계(preposition → preflight → actor)만**
재시도 루프를 돈다.

### 조작자 절차

```
① 로봇 상태 정리 — 펜던트에서 fault 해제. 팔이 어디 껴 있으면 local control로 뺀다.
② Terminal 2에서 Ctrl-C → [cleanup] 완료 대기 → ./run_hil_hardware.sh 다시 실행
③ Terminal 3은 그대로 둔다   ← 여기가 바뀐 부분
```

Terminal 3이 새 번들을 자동 감지하고 `HIL_RETRY_RESUME_DELAY_S`(기본 5 s) 뒤 preposition부터
다시 시작한다. **키 입력은 없다.**

### 종료 코드 계약 (`run_hil_actor.sh`)

| rc | 의미 | 재시도 |
| --- | --- | --- |
| `0` | 정상 종료 | ✗ |
| `1` | preflight FAIL 또는 handoff 거부 — **arming 자체가 없었음** | ✗ |
| `2` | 래퍼 usage/config 오류 | ✗ |
| `70` | actor 종료 후 **controller 복귀 실패** — 소유권 불명, 펜던트 확인 필요 | ✗ |
| `75` | **recoverable** | 후보 |
| `75` | ↳ **ros2_control 스택 소멸**(= 번들 사망). 진행 증거 없이도 승격 | 후보 |
| `>=128` | 신호(130 = Ctrl-C) | ✗ |

🔧 **2026-07-31 실기에서 이 표가 한 번 틀렸다.** Terminal 2를 내렸더니 actor가
`/joint_states stale`로 죽고, 이어진 controller 복귀가 **드라이버가 사라졌으니 당연히 실패**해
`70`이 나왔다 → 재시도 금지 → 세션 자동 종료. 정확히 복구 루프가 막으려던 상황을 복구 루프가
막았다. 이제 `hil_read_controller_states`가 **"스택이 이상하다"(1)와 "스택이 아예 없다"(2)를
구분**한다. controller_manager가 무응답이거나 우리 controller 쌍이 목록에 **하나도** 없으면
그것은 소유권 불명이 아니라 **번들 사망**이고, 그때는 ros2_control 컨트롤러가 하나도 없으므로
**팔을 몰 수 있는 것도 없다**(우리 publisher가 0인 것은 그 직전에 이미 확인한다). → `75`.

**`75`는 그냥 "죽었다"가 아니다.** 감시자가 `/hil/actor_status`에서 **`env_step >= 0`을 실제로
본 경우에만** 승격된다(`_OperatorReporter`는 `env_step`을 −1로 시작하고 `position()`은 전이
루프 안에서만 불린다). 그래서 핸드셰이크 거부, schema-hash 불일치, **첫 전이
`ActorProtocolError`**(2026-07-30 `d6965a9`가 고친 그 실패) 같은 **결정론적 실패는 승격되지
않고 재시도되지 않는다** — 재시도해 봐야 같은 실패를 반복하며 매번 자동 이동만 한 번씩 쓴다.
승격을 끄려면 `HIL_ACTOR_EXIT_MAP=0`.

### 재시도가 걸리는 세 조건 — 전부 만족해야 한다

1. **조작자가 번들을 정말 재기동했다는 증거.** 세 entrypoint(`ur_control.launch.py` /
   `robotiq_gripper_modbus` / `gello_publisher`)가 모두 있고 **PID 집합이 이전 세대와 하나도
   겹치지 않을 것.** "떠 있나"가 아니라 **세대 교체**를 본다.
2. 세 토픽이 `run_hil_hardware.sh` 자신의 READY 기준을 만족
   (`/joint_states` 50 Hz, `/robotiq_gripper/position_percent` 2 Hz, `/gello/joint_states` 15 Hz).
3. 🔑 **dashboard 서비스로 읽은 robot mode `RUNNING` + safety mode `NORMAL`.**

**세 번째가 핵심이다.** 위 토픽 셋은 전부 RTDE **읽기**라 `PROTECTIVE_STOP` 중에도 계속
흐른다. 그것만 보고 재arming하면, 팔이 우연히 RESET 0.10 rad 안에 있을 때
`run_hil_preposition.sh`가 무동작 분기를 타고 마커를 쓰고, controller 전환도 성공하고
(controller_manager는 안전 상태를 모른다), actor가 **움직이지 않는 로봇에 명령을 흘린다.**

### 한계와 스위치

`HIL_ACTOR_RETRY`(0/1, 기본 1) · `HIL_ACTOR_RETRY_MAX`(기본 3, 상한 10) ·
`HIL_HARDWARE_RECYCLE_WAIT_S`(기본 900) · `HIL_RETRY_RESUME_DELAY_S`(기본 5).

**모든 안전 proof가 매 시도마다 처음부터 다시 돈다 — 캐시되는 것은 없다.** preposition 마커
무효화 후 재생성, RESET 자세 재증명, publisher 0 확인, strict switch와 사후 검증, 그리고
`[11]`·`[ARM]` 두 곳의 fresh heartbeat 3개. **resume이 아니라 새 arming이다.**

🛑 **재시도 경로에서 RESET 복귀는 shell의 `run_hil_preposition.sh`가 한다 — GUI의
`APPROVE HOME`이 아니다.** `APPROVE HOME`과 `START / NEXT ITERATION`은 같은
`/hil/scene_ready` Trigger이고 그 **서버는 actor 프로세스 안에서 생성**된다. actor가 죽어 있는
동안 GUI는 client 핸들만 들고 있으므로 **그 버튼들은 아무 효과가 없다.** 재시도 경로에서
GUI에 요구되는 것은 버튼이 아니라 **DISENGAGED heartbeat라는 채널 생존**뿐이고, GUI는 기본이
DISENGAGED 20 Hz 발행이라 조작자가 아무것도 안 해도 통과한다.

⚠️ 재시도 뒤 첫 관측의 **그리퍼 값은 믿지 말 것** → **G37**(그리퍼만 staleness gate가 없어
죽은 채널이 `0.0` = OPEN으로 읽힌다).

---

## 5.4 🆕 Terminal 1을 닫으면 터널만 죽는다 — 📌 2026-07-31 실기에서 실제로 겪었다

**증상.** Terminal 3에서 preflight `[6]`이 이렇게 떨어진다:

```
TCP 127.0.0.1:50153 연결 실패 — ConnectionRefusedError
```

**조치 (이것 하나다).** Terminal 1에서 `./run_hil_server.sh`를 다시 실행한다.
스크립트가 서버의 **살아 있는 exact learner를 재사용**하고 **새 터널만** 연다.
그다음 Terminal 3을 다시 시작한다. Terminal 2(하드웨어)는 건드리지 않는다.

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./run_hil_server.sh --check   # (선택) 읽기 전용으로 learner 생사부터 확인
./run_hil_server.sh           # 재사용 + 새 터널
```

### 왜 헷갈리나 — **소유권이 셋 다 다르다**

| 무엇 | 누가 소유하나 | Terminal 1을 Ctrl-C 하면 |
|---|---|---|
| **learner** (서버 프로세스) | **`junhyeong_ai`가 소유. detached다** | 🟢 **살아남는다.** 명시적으로 서버에서 죽여야만 죽는다 |
| **SSH 터널** (`127.0.0.1:50153`) | **Terminal 1이 소유** | 🔴 **죽는다** |
| **actor / GUI / 카메라** | Terminal 3 | 영향 없음. 다만 다음 RPC/preflight에서 터널 부재로 실패한다 |

`run_hil_server.sh`는 마지막에 `wait "$TUNNEL_PID"`로 **터널을 붙들고 foreground에서
블록**한다. 그래서 그 터미널은 "서버 콘솔"처럼 보이지만 **실제로 들고 있는 것은 ssh child
하나뿐**이다. 스크립트가 READY 배너에서 이미 그렇게 말한다:

```
  learner : <run> PID <pid> (server-owned)
  tunnel  : PID <pid> (owned by this launcher)
Keep this terminal open. Ctrl-C closes only the tunnel.
The learner on junhyeong_ai keeps running until you stop it explicitly there.
```

### 🛑 하지 말 것

* **learner를 죽이고 새로 띄우기.** 지금 lineage의 replay·intervention 버퍼는 **그 프로세스의
  RAM에만** 있다. 재기동하면 그때까지 모은 온라인 전이가 전부 사라지고 offline demo부터
  다시 시작한다(디스크에 저장되는 것은 로그·wandb이지 replay가 아니다).
* **터널을 손으로 열어 때우기.** 열리기는 하지만 `run_hil_server.sh`의
  `validate_process_contract`(exact 문자열 비교)를 건너뛴다 → §2.4의 🛑 박스.
* **Terminal 2를 재기동하기.** 이건 하드웨어 문제가 아니다. 하드웨어 재기동이 필요한 경우는
  §5.3이고 증상이 다르다(actor rc 75, `/joint_states stale`).

---

## 6. 중단 · 복구

* **actor 중단:** T6에서 `Ctrl-C`. armed 래퍼가 신호를 actor에게 전달하고 실제 child 종료까지 기다린다.
  Qt 폰트 경고 2줄은 `launch_cameras.sh`와 `run_hil_actor.sh`에서 stderr 필터로 걸러진다.
  필터는 `trap '' INT TERM` 아래에서 도므로 **Ctrl-C가 필터를 먼저 죽여 actor의 종료 경로
  stderr를 통째로 날리는 일은 없다.** 진짜 Qt 오류는 그대로 통과한다.
* **팔이 움직이는 중이라면 먼저 펜던트 E-STOP.**
* 정상 종료/예외 뒤에는 actor의 command publisher가 사라진 것을 확인한 다음 STJC로
  strict 복귀한다. `controller cleanup PASS`를 확인한다. publisher가 남거나 controller 쌍이
  예상 밖이면 자동 switch를 거부하고 rc 70으로 끝나므로 `ros2 control list_controllers`와
  `ros2 topic info -v /forward_position_controller/commands`를 직접 확인한다.
* **DISENGAGE는 정지가 아니다.** 살아 있는 actor에서 누르면 즉시 policy가 제어한다.
  단 `HOMING/WAIT_SCENE_READY`에서는 독립 episode gate가 우선하므로 DISENGAGE만으로
  policy가 시작되지 않는다. `START / NEXT ITERATION` 승인이 필요하다.
  actor terminal의 Ctrl-C 또는 필요 시 E-STOP으로 actor/robot을 먼저 멈추고,
  `controller cleanup PASS` 뒤 GUI 상태를 정리한다.
* 터널이 죽으면 actor는 타임아웃으로 실패한다. **Terminal 1에서 `./run_hil_server.sh`를
  다시 실행**하면 learner를 재사용하고 새 터널을 연다 → **§5.4**(2026-07-31 실기에서 실제로
  겪은 경로다). wrapper 없이 진단할 때만 §2.4를 손으로 띄운다.
* 세션을 끝낼 때는 서버와 터널을 정상 종료해 **쓰던 GPU와 포트 50053을 반납**한다.
  ⚠️ `junhyeong_ai`는 **GPU가 1장(index 0)뿐이고 다른 사람과 공유**하므로 반납이 예전보다
  중요하다. 다만 Terminal 1의 `Ctrl-C`는 **터널만** 닫는다 — learner는 서버에서 명시적으로
  멈춰야 한다(§5.4).
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
| B1 | Stage A (fake-env, 학습 서버 왕복) | **PASS** 🗄️ *(kanu 기록)* | 서버 `replay_insert_count: 100`, `state_shape: [8, 1, 19]`. 상대는 zero-action 서버. **2026-07-27 kanu에서 측정** — 새 호스트에서 이 형태를 다시 돌리지는 않았고, 대신 M1의 200-transition 합성 acceptance가 같은 경로를 덮었다 |
| B2 | Stage B (실센서 + GELLO 개입, DRY_RUN) | **미검증(TODO)** | 절차는 §4에 있으나 아직 실행되지 않았다. PASS로 승격하지 말 것 |
| B2c | **분류기 sidecar 실기 왕복** (§4.4) | **실기 GUI 관측 PASS / AUTO 정확도 미승인** | production sidecar transition과 GUI의 실제 `p(success)`/threshold/verdict를 관측했다. MANUAL에서도 계속 표시된다. classifier 정확도가 부족해 AUTO는 기본이 아니며 장시간 영구 verdict audit은 남아 있다 |
| B3 | actor `--arm` (실제 팔 구동) | **핵심 E2E PASS** 🗄️ *(kanu 기록)* | Kanu policy 움직임, ENGAGE=GELLO/DISENGAGE=policy, replay/learner update를 관측했다. 옛 0.6/0.8 s stale 경계는 정상 832.3 ms reply를 거부해 1.5/2.0 s로 완화했다. **2026-07-30, 서버는 kanu.** 새 호스트 재현은 M2 |
| B4 | 같은 개입 루프를 `run_real_hil.py`로 | **PASS (2026-07-28)** | **다른 코드 경로다.** 이 표의 어느 줄도 승격시키지 않는다 → `04` §4.5 |
| B5 | terminal operator state machine | **schema-3 실기 진행 / MARK SUCCESS provenance 재확인** | episode limit 뒤 GUI `WAIT_HOME_APPROVAL`, classifier `p=0.009`, threshold `0.500`, HOME 승인 버튼이 실제 표시됐다. 🗄️ 그 lineage(replay 400/intervention 225, learner 301)는 **kanu**에서의 값이다. MANUAL `MARK SUCCESS`로 끝낸 episode의 one-shot provenance만 별도 확인한다 |

### 7.1 🆕 서버 이전 (`kanu` → `junhyeong_ai`) — 2026-07-31

| # | 항목 | 상태 | 근거 / 실측치 |
|---|---|---|---|
| M1 | 합성 acceptance E2E (200 transition) | **PASS** | gRPC → finalize → feature replay → CTA → publish → checkpoint → resume 전 구간. per-RPC BeginEpisode **57.7 ms 평균 / 82.2 최대**, Step **156.1 ms 평균 / 211.0 최대**. 🗄️ kanu BeginEpisode는 **84.9 / 372.8**이었다 — tail 약 4.5배 단축. ICMP RTT는 두 호스트가 같으므로 **호스트 연산 이득**이다. 전문 `serl_ur_infra/SERVER_MIGRATION_E2E_JUNHYEONG_AI.md` |
| M2 | **실물 UR7e 세션** (새 서버 상대) | 🟢 **PASS (조작자 확인)** | replay **316** / intervention **210** / last_env_step **68**, run root `~/hil-serl-data/runs/cube_in_cup_real_20260731_054929`. 합성 run이 못 닫던 둘을 닫았다 — **실제 classifier sidecar**와 **`intervened=1` ingress** |
| M3 | jax 핀 유지 (sm_86 → sm_120) | **PASS** | jax 0.5.3이 Blackwell에서 **네이티브** 실행(XLA `.target sm_120a`, PTX 폴백 아님). 핀 상향 불필요. warm-up 28.4 / 16.0 / 0.101 s 🗄️ vs kanu 46.83 / 37.22 / 0.466 s(**kanu learner도 A4000을 1장만 썼다**) |
| M4 | 3-CLI 조작자 절차 불변 | **PASS (코드 확인)** | `run_hil_hardware.sh`에 서버 참조 0건, `run_hil_session.sh`/`run_hil_actor.sh`는 `127.0.0.1:50153`만 참조. **Terminal 1의 스크립트만 호스트가 바뀌었다** |
| M5 | 터널 소유권 실패 모드 | **문서화됨 (실기 발생)** | Terminal 1을 Ctrl-C/종료하면 learner는 살고 터널만 죽는다 → preflight `[6]` `ConnectionRefusedError`. 조치는 `run_hil_server.sh` 재실행(재사용 + 새 터널) → §5.4 |
| M6 | 이전으로 잃은 학습 결과 | **없음** | kanu run root 8개 전부 `checkpoints/`가 비어 있었다(`checkpoint_period`=5000, 최고 learner step 301). 옮긴 것은 **복사**였고 kanu에서 지운 것은 없다 |

---

## 8. 관련 문서

* `04_HIL_INTERVENTION.md` — 데드맨 / 앵커 / 좌표계 / 두 퍼블리셔 충돌
* `05_COMMS_GRPC.md` — gRPC 계약과 19-D state 레이아웃
* `06_SENSORS.md` — RealSense 2대
* `08_OPEN_GAPS.md` — `clip_safety_box` 등 실기 투입 전 미해결 갭 (G15/G19는 닫혔다)
* `serl_ur_infra/REWARD_CLASSIFIER_LIVE_KO.md` — 라이브 분류기 뷰어 정본. **§4.4가 이걸 쓴다**
* `serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md` — **학습 서버 경로·데이터·모델 정본.**
  호스트/경로가 다른 문서와 어긋나면 이쪽이 이긴다
* `serl_ur_infra/SERVER_MIGRATION_E2E_JUNHYEONG_AI.md` — `kanu` → `junhyeong_ai` 이전 검증 기록
* `serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md` — learner 서버 전체 런북.
  🗄️ **파일명과 본문이 kanu 기준이다** — learner 옵션 설명은 유효하고, 호스트·경로는 위 두
  문서가 대체한다
* `serl_ur_infra/RL_RECEIVE_SERVER.md` / `HIL_RLPD_RECEIVE_SERVER_KO.md` — receive server 마일스톤
  (🗄️ 호스트 표기 kanu 기준)
