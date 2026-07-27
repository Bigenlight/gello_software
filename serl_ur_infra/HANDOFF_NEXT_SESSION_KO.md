# 다음 세션 인수인계 — HIL-SERL 실기 투입

> 작성: 2026-07-27 KST · 작성 시점 HEAD `db01e64` (branch `test/hil-hardware-comms`, origin 푸시 완료)
>
> **이 문서 하나만 읽고 바로 이어서 작업할 수 있게 쓴 것이다.** 배경이 더 필요하면 §9의 문서 지도를 따라간다.

---

## 0. 30초 요약

GELLO 리더 + UR7e + 개입(intervention) 시스템 + 원격 GPU 서버 양방향 통신을 실기에서 돌리는 것이 목표다. 학습 모델과 reward classifier는 **다른 분 담당**이라 지금은 비어 있어도 된다.

오늘(2026-07-27) 로봇 액터는 **"실행조차 불가능"에서 "기동·통신·안전 게이트 동작"까지** 올라왔다. 그리퍼·GELLO·서버 왕복은 실기 검증을 마쳤다.

**아직 팔을 한 번도 움직이지 않았다** (`DRY_RUN=True`). 그게 다음 관문이다.

**다음에 할 일은 §5에 순서대로, 복붙 가능한 CLI와 함께 있다.** §5-A를 먼저 읽어라.

---

## 1. 작업 좌표 — 여기서 헷갈리면 아무것도 안 된다

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
cd $WT
git log --oneline -1     # db01e64 여야 한다
```

| | 값 |
| --- | --- |
| 작업 위치 | `/home/laptop3/gello_worktrees/hil-hardware-comms` (**워크트리**) |
| branch | `test/hil-hardware-comms` (origin 푸시됨) |
| 분기점 | `5fb716b` — 여기서 8 commit 앞 |
| 액터 python | `/home/laptop3/venvs/gello-hil-actor/bin/python` |
| 원격 GPU | Kanu `166.104.35.33`, 8× RTX A4000 |

**⚠️ `/home/laptop3/gello_software`(canonical checkout, branch `feat/gello-ur7e-humble-22.04`)에는 오늘 만든 코드가 없다.** `serl_ur_infra/ur_experiments/` 디렉터리 자체가 존재하지 않는다. 같은 PC에서 다른 작업이 진행 중이라 일부러 워크트리로 분리했다 — **canonical checkout을 건드리지 말 것.**

텔레옵·그리퍼·GELLO·카메라(`ros2_ur_ws/**`) 절차는 양쪽이 같아서 어디서 돌려도 되지만, `serl_ur_infra` 절차는 **워크트리에서만** 돈다. 워크트리에도 자체 `ros2_ur_ws/install/`이 빌드돼 있다.

---

## 2. 지금 물리적으로 어떤 상태인가

| 장비 | 상태 |
| --- | --- |
| UR7e (192.168.10.11) | 마지막 세션에서 **전원 ON**. 다음 세션 시작 시 재확인 필요 |
| GELLO 리더 | 마지막 세션에서 **전원 ON**, 7개 모터 baud 57600 응답 확인 |
| 2F-85 그리퍼 | 개폐·방향 실기 확인 완료 |
| 카메라 cam1/cam2 | **🔴 케이블 분리 상태.** USB 허브 `4-4`에서 둘 다 `uvcvideo Non-zero status (-71)` (EPROTO)로 죽어서 운영자가 뽑음 |
| Kanu | `run_rlpd_receive_server.py` (PID 1096786, port 50053) 가동 중 — **zero action만 돌려준다** |

**cam2는 손목(wrist) 카메라다.** 이전에 고정 데스크 카메라로 오판한 적이 있으니 주의. 두 카메라를 구별하려면 팔을 한 번 움직여 보면 된다.

---

## 3. 무엇이 검증됐고 무엇이 안 됐나

### ✅ 실기에서 사람이 확인한 것

- **2F-85 그리퍼** — Modbus RTU over UR tool-comm `:54321`. 개폐 및 방향을 눈으로 확인. crush-hazard 게이트가 이걸로 닫혔다.
- **GELLO 리더** — 7개 모터 전부 응답.
- **laptop → SSH 터널 → Kanu 100-step gRPC 왕복** — `replay_insert_count:100`, `state_shape:[8,1,19]`, observation schema hash `3459098d…` 양쪽 일치. 상대는 receive server(zero action).
- **지연/대역폭** — RTT p50 58.6 / p95 75.8 / **p99 97.1 ms**, step당 96.1 KiB → 7.9 Mbit/s. 병목은 **WiFi 대역폭(약 13 Mbit/s)**. 유선 재측정 권장.

### 🟡 코드는 됐고 단위 테스트도 통과했지만 실기 미검증

- `clip_safety_box` (workspace box) — 부호 보존 abs-clip
- `go_to_reset` branch-cut 게이트 — `wrapped_nearest`
- 전체 wrapper chain + task config (actor가 기동은 한다)

### 🔴 아직 못 한 것

- **팔을 실제로 움직이는 것** (`DRY_RUN=True`로 잠겨 있음) ← 다음 관문
- 카메라 경로 (하드웨어 고장)
- 개입 루프 실기 검증
- Kanu 실제 정책 서빙 (지금은 zero action)
- frame-map 측정 — **한 번 했으나 무효**(§6.1)

### 테스트 — 이 명령 그대로 쓸 것 (2026-07-27 실행 검증)

```bash
cd $WT
set +u; source /opt/ros/humble/setup.bash; source $WT/ros2_ur_ws/install/setup.bash; set -u
OVERLAY=$(python3 -c "import ur_gello_bringup,os;print(os.path.dirname(os.path.dirname(ur_gello_bringup.__file__)))")

env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$WT/serl_ur_infra:$WT/third_party/hil-serl/serl_launcher:$OVERLAY" \
  /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
  -p no:cacheprovider serl_ur_infra/tests
# -> 332 passed, 11 skipped, 1 xfailed
```

**🪤 `third_party/hil-serl/serl_launcher`를 PYTHONPATH에서 빼면 조용히 299 passed, 13 skipped가 된다.** 사라지는 것이 하필 이번 작업의 핵심 테스트 2개다 — `test_cube_in_cup_config.py`(task config 가드 전체)와 `test_frame_wrappers.py`(이식한 wrapper 전체). 게다가 skip 사유가 **"third_party/hil-serl submodule is not checked out"이라고 거짓말을 한다.** 서브모듈은 체크아웃돼 있다(`c32939b`, heads/main). 진짜 원인은 `serl_launcher`가 PYTHONPATH에 없는 것뿐이다.

즉 **"녹색이니까 검증됐다"고 믿기 전에 passed 수를 확인할 것.** 299가 나오면 config와 wrapper는 아무것도 검증되지 않은 것이다.

UR·GELLO suite는 별도로 **436 passed**.

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`이 빠지면 시스템 pytest 플러그인과 충돌해 `ModuleNotFoundError: _pytest.scope`로 죽는다.
- venv를 쓰되 pytest에는 ROS PYTHONPATH 전체를 넣지 않는다(§7).

---

## 4. 오늘 무엇을 고쳤나 (8 commit)

| commit | 내용 |
| --- | --- |
| `cc13710` | `RecordEpisodeStatistics`를 gymnasium 패키지 루트에서 import |
| `ebc77f7` | Franka task registry 없이도 actor 기동 |
| `14c57d8` | canonical observation을 만드는 wrapper 이식 |
| `50f65de` | UR7e task config와 `CONFIG_MAPPING` 추가 |
| `a4a2b0a` | robot state stream을 기다린 뒤 `__init__` 반환 |
| `d49d0f6` | workspace box 활성화 + reset이 먼 길로 돌지 않게 |
| `ee3240e` | z 바닥과 reset gate 정정, frame test를 실제로 물게 만듦 |
| `db01e64` | 상태 문서 + `docs/testing/` 갱신 |

`5fb716b`에서 액터는 **4개 층이 동시에 막고 있었다**: wrapper 3종 부재 / task config 부재 / import 시점 실패 2건 / DDS discovery race(약 1.0 s). 어느 하나만 고쳐도 다음 층에서 다시 멈췄다. 전부 해소됐다.

### 검수로 뒤집힌 값 2개 — 이건 꼭 알고 있어야 한다

둘 다 처음 산출값이 틀렸고, 독립 검수에서 잡혔다. **녹색 테스트만 봐서는 절대 안 보이는 종류의 오류였다.**

1. **z 바닥 `0.1785` → `0.185`.** `0.1785`는 테이블 표면이 아니라 **충돌 깊이**였다. 그 최소값을 만든 40개 샘플이 전부 take_11이고, 그 구간의 그리퍼 개구가 0.047–0.176(빈 손), `fz`가 −133.2 ~ −24.2 N. 빈 그리퍼로 테이블을 눌러 박은 자세다. 접촉 없는 샘플만 필터링하면 표면은 `0.1808`.
2. **`RESET_MAX_DIST_RAD` `0.5` → `0.9`.** `0.5`는 23개 take 중 **16개**의 정상 에피소드 종료 자세를 거부한다. 종료 자세에서 `RESET_JOINTS`까지의 branch-safe 거리가 중앙값 0.615 / 최대 0.774 rad다.

---

## 5. 다음에 할 일 — 순서대로

### A. frame-map 재측정 ← **여기서 시작**

카메라가 필요 없고, 팔도 안 움직이므로 **가장 먼저 할 수 있고 가장 안전하다.**

**왜 다시 하나:** 이전 측정에서 잔차 0.864가 나왔는데, 스텝의 **73.7%가 속도 제한에 포화**돼 있었다. `--scale 0.25`에서 명령이 2.5 cm/s로 잘리는 동안 리더는 중앙값 24.5 cm를 움직였고 명령은 6.0 cm만 나갔다. **포화된 데이터로는 좌표계 매핑을 측정할 수 없다.** 변환 행렬 M의 대각이 전부 양수여서 **부호 뒤집힘은 없다**는 것까지만 말할 수 있다.

**어떻게 다시 하나:** `--scale 1.0`으로 올리고, **move-then-hold** 프로토콜을 쓴다 — 리더를 한 방향으로 움직인 뒤 **멈춰서 명령이 따라잡게 둔다.** 계속 움직이면 또 포화된다.

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
cd $WT

# ROS 오버레이 (set -u 아래에서 source하면 AMENT_TRACE_SETUP_FILES 오류 — §7 참조)
set +u; source /opt/ros/humble/setup.bash; source $WT/ros2_ur_ws/install/setup.bash; set -u

# ⚠️ --arm 없음 = 팔이 안 움직인다. 이대로 먼저 돌린다.
python3 serl_ur_infra/tests/run_real_hil.py \
  --scale 1.0 \
  --deadman topic \
  --episodes 1 --max-steps 300 \
  --csv ~/gello_hil_logs/framemap_scale1.csv
```

- 데드맨 GUI가 따로 필요하다: `./ros2_ur_ws/run_hil_gui.sh` (기본 `--deadman topic` = `/hil/deadman`, 20 Hz 하트비트 + 워치독)
- CSV를 사후 분석해서 (a) 좌표계 매핑 (b) 앵커 래치 (c) gain 래치 (d) "저장 액션 == 실행 액션" 불변식을 확인한다
- **포화 비율을 먼저 확인할 것.** 다시 70%대면 그 측정도 무효다

### B. 첫 실물 구동 — 팔이 움직인다

A가 말이 되는 값을 내면 진행한다.

**사전 조건:**
- 펜던트 E-STOP을 손 닿는 곳에
- 작업 공간 비우기, 사람이 궤적 안에 들어가지 않기
- **`/forward_position_controller/commands`에 다른 퍼블리셔가 없어야 한다.** gello_ur_bridge(텔레옵)가 살아 있으면 두 퍼블리셔가 같은 컨트롤러를 다른 목표로 때려서 팔이 떤다. 러너가 `count_publishers()`로 세어 보고 거부하지만, 미리 끄는 게 맞다
- `forward_position_controller` 활성화 확인

```bash
python3 serl_ur_infra/tests/run_real_hil.py \
  --arm --scale 0.25 \
  --deadman topic \
  --episodes 1 --max-steps 300
```

`--scale`은 ACTION_SCALE / GOVERNOR / UPSAMPLER **3층을 함께** 곱한다. 처음엔 반드시 낮게. `--scale > 소프트맥스`는 `--allow-fast`로 별도 잠금 해제해야 한다.

리셋은 **무동작**으로 설계돼 있다 — 기동 시점 관절을 그대로 `RESET_JOINTS`로 잡고 `--reset-max-dist 0.05`로 조인다. 사람이 개입으로 팔을 많이 옮긴 뒤 리셋하면 **에러로 멈추는데 그게 의도된 안전 실패다**(멀리서 쓸고 오지 않는다). 그 경우 `--reset-mode hold` 또는 `--episodes 1`.

러너가 죽거나 Ctrl-C로 끝나면 업샘플러가 발행을 멈추고 컨트롤러는 **마지막 명령을 유지**한다(팔은 그 자리에 선다).

### C. 액터 결함 2개 수정

1. **deadman GUI가 액터에서 무효다.** `run_remote_rlpd_actor.py`가 `deadman`을 전혀 넘기지 않아 `GelloIntervention`이 항상 `SpacebarDeadman()`으로 떨어진다. pynput 리스너는 **전역**이라 아무 창에서나 스페이스바가 개입을 켜고 **ESC가 에피소드를 종료**시킨다. `test_deadman_wiring.py`가 `--deadman` 플래그 부재를 strict xfail로 고정해 뒀으니, 플래그를 추가하면 그 xfail이 xpass로 바뀐다.
   *(참고: `run_real_hil.py`에는 이미 `--deadman {topic,spacebar}`가 있고 기본이 `topic`이다. 문제는 액터 entrypoint 쪽이다.)*
2. **`DRY_RUN`을 CLI로 뒤집을 방법이 없다.** 지금은 `cube_in_cup.py`를 편집해야 한다.
3. (작음) `_await_robot_state`에 카메라 첫 프레임 대기가 없다.

### D. 카메라 복구

USB 허브 `4-4` 고장. **카메라 없이는 액터 전 경로가 돌지 않는다** (canonical observation에 cam1/cam2가 필수). 팔 단독 검증은 `run_real_hil.py` 경로로만 가능하다.

### E. Kanu에 진짜 정책 서버 올리기 — §8 참조

---

## 6. 미해결 위험 / 열린 질문

1. **초기 정책이 어떤 크기의 action을 내는지 아직 모른다.** 학습 전 초기화된 SAC 정책이 full-scale action을 내면 팔이 튄다. learner server를 붙이는 첫 시도는 `DRY_RUN` 또는 낮은 `--scale`로 이걸 먼저 **관측**할 것. 이 질문에 답하려던 에이전트가 사용 한도로 중단됐다.
2. **frame-map이 아직 미측정이다** (§5-A). 부호 뒤집힘이 없다는 것까지만 안다.
3. **actor↔learner 계약 정밀 대조가 미완료다.** `policy_version` 단조성, 실제 추론이 붙었을 때의 지연 예산을 확인하려던 에이전트가 중단됐다. 실제 추론이 들어가면 현재 p99 97.1 ms에 추론 시간이 더해진다.
4. **`cygrpc.so`의 기계어 원인은 확인되지 않았다.** 이전에 `__wrap_memcpy`가 `endbr64; jmp $` 무한 루프라고 적었으나 **그 심볼이 바이너리에 없다.** 행동(시스템 python3에서 100% CPU 무한 정지)은 100% 재현되므로 결론(venv를 써라)은 그대로지만, 원인은 미확인이다.

---

## 7. 함정 모음 — 반복해서 당한 것들

| 함정 | 대응 |
| --- | --- |
| **시스템 grpcio 1.30.2가 고장** — gRPC를 쓰면 오류/로그 없이 100% CPU로 무한 정지(rc=124, 단일 스레드 `R`, `wchan` 비어 있음) | 반드시 `/home/laptop3/venvs/gello-hil-actor/bin/python` (grpcio 1.74.0, rclpy용 `--system-site-packages`). **시스템 `python3`로 gRPC 코드를 절대 실행하지 말 것** |
| `PYTHONPATH=…`가 ROS 오버레이를 날림 → `ModuleNotFoundError: ur_gello_bringup` | **덮어쓰지 말고 이어붙인다**: `:$PYTHONPATH` |
| **반대로** pytest는 ROS PYTHONPATH가 있으면 1개만 수집 | 원인은 ROS의 pytest 플러그인. `-p no:launch_testing`으로 해결. **`--ignore=`로는 안 막힌다** (다음 파일이 같은 자리를 물려받는다) |
| ROS `setup.bash`를 `set -u` 아래에서 source → `AMENT_TRACE_SETUP_FILES: unbound variable` | `set +u` / `set -u`로 감싼다 |
| `ros2 launch`에 전부 숫자인 `key:=value`를 주면 문자열 파라미터도 int로 강제됨 | 따옴표를 안에 넣는다: `"key:='val'"` |
| 그리퍼 `:54321`은 클라이언트를 **하나만** 받음 | 반드시 `Ctrl-C`로 종료. **`kill -9` 금지** — FIN-WAIT-2가 재접속을 30–45초 굶긴다 |
| `/joint_states`의 name 순서가 canonical이 아님 | 실제 순서: `[shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan]`. **인덱스로 가정하지 말고 name으로 매핑할 것** (이걸 잘못 읽어서 관절 편차를 10배로 잘못 계산한 적이 있다) |

---

## 8. Kanu 정책 서빙 — 현재 zero action이다

**떠 있는 것은 `run_rlpd_receive_server.py` 하나뿐이고, `FakeActionRuntime`이라 항상 zero action을 돌려준다.** `model_id`도 `fake-zero-action-v0`다. 실제 정책 추론은 일어나지 않는다.

실제 서빙은 `run_rlpd_learner_server.py`가 한다 — receive server의 **엄격한 상위집합**이다(같은 gRPC ingress + 같은 classifier에 더해 실제 `SACAgentHybridSingleArm` 추론 + CTA 학습 + checkpoint). 둘 다 기본 port 50053이라 **동시 실행 불가**. receive를 내리고 learner가 인수하면 랩탑 SSH 터널도 그대로 재사용된다.

### 🔴 유일한 하드 블로커: canonical robot demo가 없다

`--demo-path`가 required이고 3중으로 검증된다. **우회로가 없다:**

- `--dry-run`은 gRPC를 bind하지 않고 즉시 종료 → 액터가 붙을 수 없다
- `--synthetic-e2e`는 서버가 **run_id 화이트리스트**를 강제하는데, `run_remote_rlpd_actor.py`는 `--run-id`를 노출하지 않고 `remote_actor.py:225`에서 매번 `uuid4()`로 새로 만든다 → 실제 액터를 미리 등록할 방법이 없어 `FailedPreconditionError`

**하지만 생산 경로는 코드에 이미 있다.** `remote_actor.py`의 `_dump_data`가 `--checkpoint-path`를 받으면 `<ckpt>/actor_data/<run_id>/replay/data_<step>.pkl`을 남기고, `load_demo_pickles`(`demo.py:92-98`)가 정확히 그 `{"meta":…, "transition":…}` 형식을 받는다. **receive server를 띄운 채 텔레오퍼레이션으로 성공 에피소드를 녹화하면 그게 demo가 된다.**

strict loader 계약(아직 실제 dump로 확인 안 함): canonical v2 스키마, `actions` f32 `(7,)` `[-1,1]` 마지막 원소 `{-1,0,1}`, `rewards`/`masks` `{0,1}`, `grasp_penalty ∈ {0, −0.02}`, joint-space/LeRobot 키가 있으면 즉시 거부.

### Kanu 환경 — 추가 설치 불필요

| 항목 | 값 |
| --- | --- |
| python | **`/home/junhyeong/miniconda3/envs/il/bin/python`** (베이스 `il`) |
| jax·jaxlib·flax·distrax·tfp·wandb | `0.5.3 / 0.5.3 / 0.10.5 / 0.1.5 / 0.25.0 / 0.26.0` — lock과 정확히 일치 |
| GPU | 8× RTX A4000 (16 GiB). **2/4/5/6 권장**, 7번은 receive server가 12.3 GiB 점유 |
| RAM / disk | available 73 G / 여유 **127 G (93% 사용)** |
| learner 파일 최신성 | Kanu `5fb716b`와 로컬 `ee3240e`의 learner 파일 19개 전부 SHA256 동일 (차이는 전부 actor 쪽) |

**⚠️ overlay venv `/tmp/gello-hil-rl-receive-overlay-v2`를 learner에 재사용하지 말 것.** protobuf를 3.20.3으로 핀했는데 wandb 0.26.0은 `wandb/proto/`에 v4~v7만 배포한다 → `ImportError: cannot import name 'Imports'`. lock도 protobuf 7.34.1을 요구한다.

기타:
- **`XLA_PYTHON_CLIENT_PREALLOCATE=false` 필수.** receive server가 12.3 GiB를 잡은 건 JAX 기본 75% preallocation 때문이고, 16 GiB 카드에서 이걸 두면 learner + classifier가 한 GPU에 못 들어간다
- `preflight_checkpoint_run`은 `--resume-*` 없이 시작할 때 checkpoint root에 기존 항목이 있으면 **거부** → 새 run은 반드시 빈 root
- `--target-learner-step`을 생략하면 무한 continuous. 첫 실전은 `5000` 권장
- 학습 워커가 fault를 내도 서버는 last-known-good 정책을 계속 서빙해 gRPC health가 ready로 보인다 → `rlpd_learner_worker_fault` 이벤트를 로그로 감시

액터가 pin해야 하는 값:

```text
--expected-model-id       hil-serl-hybrid-sac-resnet10-trunk-cache-v1
--expected-reward-authority server_classifier
--expected-reward-model-id  cube-in-cup-checkpoint-150
--observation-schema-hash 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
```

전체 실행 명령은 `HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` §11.5와 `HIL_SERL_KANU_RUNBOOK_KO.md`에 있다.

---

## 9. 문서 지도

| 문서 | 용도 |
| --- | --- |
| **이 파일** | 다음 세션 시작점 |
| `HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md` | 전체 상태. learner는 §1–10, **actor/하드웨어는 §11** |
| `HIL_SERL_KANU_RUNBOOK_KO.md` | Kanu 실행 절차 |
| `docs/testing/README.md` | 하드웨어·통신 검증 런북 인덱스 (00~09) |
| `docs/testing/09_HIL_ACTOR_RUNBOOK.md` | 액터 실행 런북 |
| `docs/testing/08_OPEN_GAPS.md` | 미해결 갭 목록 |
| `serl_ur_infra/tests/run_real_hil.py` | 실기 개입 러너 — **파일 상단 주석이 안전 설계를 전부 설명한다** |

핵심 코드:

```
serl_ur_infra/ur_experiments/cube_in_cup.py       task config (측정값 전부 여기)
serl_ur_infra/ur_env/envs/frame_wrappers.py       RelativeFrame + Quat2EulerWrapper
serl_ur_infra/ur_env/envs/chunking.py             ChunkingWrapper (jax-free)
serl_ur_infra/ur_env/envs/ur7e_env.py             clip_safety_box, go_to_reset, _await_robot_state
serl_ur_infra/scripts/run_remote_rlpd_actor.py    액터 entrypoint (deadman 미배선)
ros2_ur_ws/run_hil_actor.sh                       액터 실행 래퍼
ros2_ur_ws/run_hil_gui.sh                         데드맨/개입 GUI
```

---

## 10. 절대 하지 말 것

- **GELLO Dynamixel에 토크를 걸지 말 것.** 수동 read-only 리더다.
- **RViz에서 좌우/전후가 뒤집혀 보인다는 이유로 `wrappers.py`의 X/Y 부호를 뒤집지 말 것.** 카메라 방위각 artifact이며 `R_align = I`다. 이미 한 번 잘못 판단한 적이 있다.
- **프레임 통일 명목으로 wrench(`tcp_force`/`tcp_torque`)를 회전시키지 말 것.** upstream도 건드리지 않는다. 우리 `/force_torque_sensor_broadcaster/wrench`는 `tool0`에, libfranka `K_F_ext_hat_K`는 stiffness(EE) frame에 publish하므로 양쪽 다 tool frame으로 끝난다.
- **`RelativeFrame.step`의 두 시점을 하나로 합치지 말 것.** action은 step **이전** 행렬로, observation은 **이후** 행렬로 변환한다. 의도적으로 한 제어 주기 떨어져 있다.
- **`TCP_POSE_SOURCE` / `TCP_OFFSET_XYZ_RPY` / `ABS_POSE_LIMIT`은 결합된 한 세트다.** box를 flange frame에서 측정했으므로 tool offset 0.174 m를 넣거나 pose source를 `fk`로 바꾸면 관측 pose가 **오류 없이** 17.4 cm 이동해 z 바닥이 테이블 아래로 내려간다.
- **`wrapped_nearest`가 elbow(index 2, ±π 한계)를 unwrap하지 않는 것은 의도다.** 이 게이트가 없으면 reset이 wrist_3를 한 바퀴 통째로 돌려 2F-85 tool-comm 케이블을 감는다 (실기에서 wrist_3 `+3.1795` vs 목표 `−3.1331` = 실제로는 0.029 rad인데 순진한 차분은 6.31 rad인 상황이 나왔다).
- headless 세션에서 `ur_play`/`ur_load`/`ur_stop`을 실행하지 말 것. `ur_resend`만 쓴다.
- canonical checkout `/home/laptop3/gello_software`를 건드리지 말 것.

---

## 11. 진행 방식 (사용자 선호)

- 사용자는 실제 로봇이 처음이다. **한 번에 한 단계씩 가르치듯 설명하고, 명령은 사용자가 직접 실행한다.**
- 문제 해결 시 **검수·피드백 과정을 반드시 넣을 것.** 오늘 그 과정에서 값 2개가 뒤집혔다(§4).
- upstream hil-serl 공식 코드가 하는 대로 따른다.
- 병렬 에이전트를 적극 사용해도 좋다 — **단 2026-07-27 시점에 월 사용 한도에 도달해 에이전트 3명이 중단됐다.** 다음 세션에서 먼저 한도 상태를 확인할 것.
