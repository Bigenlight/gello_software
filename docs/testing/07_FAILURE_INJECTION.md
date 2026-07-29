# 07 — 장애 주입 매트릭스

**상태: 전 항목 미검증.** 아래는 실행 절차와 **코드 근거가 있는 기대 동작**이다.
실행하기 전에 기대 동작을 먼저 읽고, 다르게 나오면 그것이 발견이다.

> ### 2026-07-27 실기 세션 반영
> - **E5(카메라)의 전제가 바뀌었다:** cam2는 손목 카메라다. 유발 방법과 안전 절차가 다르다.
> - **E13(인터프리터 hang)이 새로 추가됐다.** 이건 가설이 아니라 오늘 실제로 4번 물린 것이다.
> - **E14(리셋 branch-cut)가 새로 추가됐다.** 실기에서 반드시 `DRY_RUN`으로 먼저 본다.
> - E4의 포트가 **로컬 50153 / 원격 50053**으로 바뀌었다 (`05_COMMS_GRPC.md` §6).

```bash
export WT=/home/laptop3/gello_worktrees/hil-hardware-comms
```

---

## 0. 실행 규칙

1. **mock부터.** E1~E5(통신/프로세스 계열)는 `04_HIL_INTERVENTION.md` §3의 mock+RViz 구성에서
   먼저 돌린다. 실기에서 처음 하지 말 것.
2. **한 번에 하나.** 두 장애를 동시에 넣으면 어느 안전망이 잡았는지 알 수 없다.
3. **매번 손을 E-STOP 위에.** 특히 E6~E11.
4. **기록 서식:** 아래 표를 그대로 복사해서 결과를 채운다.

```
| ID | 일시 | 유발한 방법 | 관측된 동작 | 기대와 일치? | 복구에 걸린 시간 | 비고 |
```

---

## E1 — GELLO 리더 끊김

| | |
|---|---|
| **유발** | (a) GELLO USB를 뽑는다, 또는 (b) `pkill -f gello_publisher` |
| **기대 (텔레옵 경로)** | 브리지의 leader 워치독(`staleness_timeout_s` 기본 0.5 s, `gello_ur_bridge_node.py:136-137`)이 걸려 **발행 중단 + 재시드 강제**. 컨트롤러는 **마지막 명령 자세를 홀드**. GELLO가 돌아오면 **자동 재개** (`GELLO_UR7E_SETUP_CLI.md:667`) |
| **기대 (HIL/RL 경로)** | `LEADER_STALE_S = 0.3`(`wrappers.py:190`) 초과 → `_disengage()` → **정책이 이어받아 팔을 계속 움직인다** (`wrappers.py:287-290`). **정지가 아니다.** |
| **함정 (a)** | USB를 뽑아도 `gello_publisher` **프로세스는 살아 있다.** `get_joint_state()` 실패는 `except`로 잡혀 2초 throttle 경고 후 사이클만 스킵한다 (`gello_publisher_node.py:167-173`). 즉 **`pgrep`으로는 장애를 감지할 수 없다** |
| **확인** | `ros2 topic hz /gello/joint_states` (멈춤), 러너 로그의 `intervened` 값, `ros2 topic echo /forward_position_controller/commands` |
| **PASS** | ① 팔이 튀지 않는다(점프 0) ② 텔레옵: 홀드 후 복귀 시 자동 재개, 재개 순간 점프 없음 ③ HIL: 0.3 s 안에 `intervened`가 0으로 떨어진다 |
| **복구** | USB 재연결 / `gello_publisher` 재기동. 텔레옵은 자동 재개. HIL은 데드맨 재-ENGAGE |

---

## E2 — 로봇 `/joint_states` 끊김

| | |
|---|---|
| **유발** | 드라이버 launch 터미널에서 Ctrl-C, 또는 `ros2 lifecycle`/컨트롤러 언로드로 `joint_state_broadcaster` 정지 |
| **기대** | env가 다음 `_update_currpos()`에서 **예외를 던진다**: 0.2 s 초과 시 `RuntimeError: /joint_states stale` (`ur7e_env.py:365-367`, `config.py:121`), 아예 없으면 `RuntimeError: no /joint_states received yet` (`:362-363`) → **actor 프로세스가 죽는다** |
| **⚠️ 이것은 graceful safe-stop이 아니다** | 코드에도 그렇게 적혀 있다: `TODO(together): safe-stop policy (freeze + operator prompt) instead of raise` (`ur7e_env.py:365-367`). 현재는 그냥 예외 → 크래시다 |
| **확인** | 러너 스택트레이스, `ros2 topic hz /joint_states`, 팔이 마지막 자세를 홀드하는지 |
| **PASS** | 팔이 **마지막 명령 자세에서 정지**하고, 예외 메시지가 원인을 정확히 지목한다 |
| **복구** | 드라이버 재기동 → Method A면 `ur_play`, Method B면 `ur_resend`. env는 새로 띄운다 |

---

## E3 — 프로세스 kill

세 변종을 **따로** 본다. 결과가 다르다.

### E3a — 러너/actor를 `kill -9`

| | |
|---|---|
| **유발** | `kill -9 <runner pid>` |
| **기대** | 백엔드의 250 Hz 업샘플러 스레드는 **daemon**이므로 프로세스와 함께 죽는다 (`ros_backend.py:129`). 명령 발행 중단 → `forward_position_controller`가 **마지막 명령 자세를 홀드** |
| **PASS** | 팔이 즉시 그 자리에 서고, 감속 램프도 fault도 없다 |
| **복구** | 러너 재기동. `reset()`이 `go_to_reset()`으로 **팔을 RESET_JOINTS로 이동시킨다** — 재기동 = 움직임이다, 각오하고 누를 것 |

### E3b — 🛑 러너가 **살아 있는데 멈춘 경우** (가장 위험한 변종)

| | |
|---|---|
| **유발** | `kill -STOP <runner pid>` (SIGSTOP) |
| **기대** | **업샘플러에는 타깃 스테일 정책이 없다.** 코드 주석이 명시한다: `TODO(together): target-staleness policy (stop publishing after N s without a fresh target?)` (`ros_backend.py:246-250`). 따라서 스레드가 계속 돌면 **마지막 타깃을 무한히 재발행**한다 |
| **왜 위험한가** | 파이썬 스레드는 SIGSTOP으로 프로세스 전체가 멈추므로 이 경우엔 발행도 멎는다. 하지만 **GIL을 잡고 있는 다른 스레드가 블록된 경우**(예: 카메라 디코드 hang, gRPC 대기)에는 업샘플러만 계속 돌 수 있다. 그때 로봇은 "누구도 지시하지 않는 명령"을 계속 받는다 |
| **확인** | `ros2 topic hz /forward_position_controller/commands` — 러너가 멈췄는데도 250 Hz가 유지되는가 |
| **PASS 판정** | 이 시나리오는 **현재 코드로는 PASS 조건을 정의할 수 없다.** 관측 결과를 기록하고 `08_OPEN_GAPS.md` G4에 반영한다 |
| **복구** | `kill -CONT` 후 정상 종료, 또는 E-STOP |

### E3c — GUI / 카메라 노드 kill

| | |
|---|---|
| **유발** | `pkill -f gello_hil_gui` / `pkill -f realsense2_camera_node` |
| **기대 (GUI)** | 0.5 s 워치독 → `engaged=False` → **정책 복귀** (`wrappers.py:158-165`). 정지 아님 |
| **기대 (카메라)** | 0.5 s(`IMAGE_STALE_S`) 후 `get_im()`이 `RuntimeError: camera 'camX' has no fresh frame` (`ur7e_env.py:463-469`) → 러너 크래시 |
| **PASS** | 두 경우 모두 팔이 마지막 자세에서 정지하고, 원인 메시지가 정확하다 |

---

## E4 — 네트워크 / gRPC

| | |
|---|---|
| **유발** | (a) SSH 터널 프로세스 kill, (b) `sudo tc qdisc add dev lo root netem delay 800ms` 로 지연 주입 |
| **기대 (a)** | `UNAVAILABLE` = transient → **똑같은 직렬화 요청을 정확히 1회 재시도** (`grpc_actor_transport.py:733-746`) → 실패 시 transition은 **pending**으로 남고 **reset/새 액션이 차단**되며 **Local fails stopped**. **폴백/랜덤 액션은 절대 실행되지 않는다** (`REMOTE_ACTOR_GRPC.md` "Failure rules") |
| **기대 (b)** | 800 ms 지연 > `timeout_s` 0.6 s + `max_response_age_s` 0.8 s → (a)와 같은 경로 |
| **확인** | 러너 로그에 재시도 1회 후 fail-stop, `ss -ltnp \| grep 50153` (**로컬 터널 포트는 50153**, 원격이 50053) |
| **PASS** | ① 재시도가 **정확히 1회** ② 서버에 중복 삽입이 없다(응답 캐싱) ③ 팔이 정지 ④ 임의 액션이 실행되지 않았다 |
| **복구** | `sudo tc qdisc del dev lo root` / 터널 재수립. 에피소드는 버린다 |
| **주의** | 스모크 기본 타임아웃(2.0/3.0 s)으로 테스트하면 800 ms 지연을 **통과해버린다.** 반드시 프로덕션 값(0.6/0.8)으로 → `05_COMMS_GRPC.md` §5 |

---

## E5 — 카메라

> 🔧 **cam2는 손목 카메라다** (`06_SENSORS.md` §1.1). USB를 뽑으려면 **팔에 붙어 있는**
> 케이블을 만지게 된다 — 팔이 명령을 받고 있지 않은 상태에서만 할 것.
> 렌즈 가림 테스트(c)도 손목에 손을 넣는 동작이다.

| | |
|---|---|
| **유발** | (a) cam2 USB 뽑기(**손목에 손이 들어간다 — 팔 정지 확인 후**), (b) `pkill -f "realsense2_camera_node.*cam2"`, (c) 렌즈를 가려 프레임 내용만 죽이기 |
| **기대 (a)(b)** | 0.5 s 후 `RuntimeError: camera 'cam2' has no fresh frame (age=...s)` → 러너 크래시 (`ur7e_env.py:705-717`) |
| **기대 (c)** | **아무 일도 안 일어난다.** 신선도만 보고 내용은 안 본다. 검은 화면이 그대로 관측/버퍼에 들어간다 |
| **⚠️ TRANSIENT_LOCAL 함정** | 퍼블리셔가 TRANSIENT_LOCAL이라 **카메라가 이미 죽은 뒤에 붙은 구독자도 캐시된 마지막 프레임을 한 장 받는다.** `ros2 topic echo --once`가 성공하는 것을 "살아 있음"의 근거로 쓰지 말 것 → `06_SENSORS.md` §3.0 |
| **확인** | `ros2 topic hz /cam2/cam2/color/image_raw/compressed` (**`echo`가 아니라 `hz`**) |
| **PASS** | (a)(b)에서 0.5 s 안에 명확한 에러로 정지. (c)는 **탐지 불가임을 확인**하고 기록 |
| **복구** | USB 재연결 후 `./launch_cameras.sh` 재기동. DFU(`8086:0adb`)면 물리적 재연결 → `06_SENSORS.md` §2.2 |

> ### E5 손목 카메라 특유의 추가 관측 (기록만, 판정 아님)
> 팔을 크게 움직이면 cam2 화면 전체가 흐른다. **모션 블러/노출 변화가 정책 입력에 실제로
> 들어간다.** 고정 카메라 전제로 설계한 크롭·전처리는 여기서 통하지 않는다.
> `IMAGE_CROP["cam2"]`가 테이블이 아니라 **측정된 grasp 축(x=781)** 기준인 이유다.

---

## E6 — E-STOP

| | |
|---|---|
| **유발** | 펜던트/외부 E-STOP 버튼을 누른다 |
| **기대** | 즉시 관절 전원 차단 + 브레이크. ROS 쪽은 `/joint_states`가 멎고 드라이버가 연결을 잃는다 → E2와 같은 경로로 러너가 죽는다 |
| **⚠️ 부수 효과** | **로봇 전원이 내려가면 tool 전압도 내려가 그리퍼가 죽는다.** `robotiq_gripper`가 `no/invalid status response`로 재연결 루프에 들어간다 (`robotiq_gripper_modbus_node.py:155-162`) |
| **확인** | `echo -e 'robotmode\n' \| nc 192.168.10.11 29999`, `echo -e 'safetystatus\n' \| nc ... 29999` |
| **PASS** | 팔이 즉시 정지. 소프트웨어 상태와 무관하게 작동 |
| **복구** | E-STOP 해제 → `ur_poweron` / `ur_brake_release` (또는 `power on` + `brake release` dashboard) → Method A `ur_play` / Method B `ur_resend` → 그리퍼 노드가 자동 재연결되는지 확인 |

---

## E7 — 펜던트 조작

### E7a — Remote → Local 전환

| | |
|---|---|
| **유발** | 펜던트에서 Remote를 끈다 |
| **기대** | Method B(headless)는 URScript를 더 못 보낸다 → 팔 정지. `is in remote control`이 `false` |
| **⚠️** | **Remote → Local 복귀는 펜던트에서만 가능하다** (안전 설계, `remote_helpers.sh:12-14`). ROS에서 되돌릴 수 없다 |
| **복구** | 펜던트에서 다시 Remote → `ur_resend` |

### E7b — 프로그램 정지 (Method A)

| | |
|---|---|
| **유발** | 펜던트 Stop, 또는 `ur_stop` |
| **기대** | External Control 프로그램 종료 → 드라이버가 명령을 못 보냄 → 팔 정지 |
| **복구** | `ur_load ur_caps.urp` → `ur_play` |

### E7c — 속도 슬라이더

| | |
|---|---|
| **유발** | 슬라이더를 0%로 내린다 |
| **기대** | **정지가 아니다.** 명령 스트림은 계속 흐르고, 슬라이더를 올리면 밀린 것이 실행된다 |
| **PASS 기준** | "슬라이더는 비상 정지 수단이 아니다"를 팀 전원이 눈으로 확인 → `00_SETUP_AND_SAFETY.md` §6.1 |

---

## E8 — 관절 리밋

| | |
|---|---|
| **유발** | 개입/정책으로 어느 한 관절을 리밋 쪽으로 계속 민다 |
| **기대** | IK 해가 `within_joint_limits(q_sol, margin=0.0)`를 통과 못 하면 컨트롤러가 **HOLD**하고 `reject_reason = "JOINT_LIMIT"`을 낸다 (`policy_delta_controller.py:105-106`). 해 자체가 없으면 `"NO_IK"` (`:104`) |
| **확인** | `step()` info의 `held` / `reject_reason` (env가 `info`에 합쳐서 반환한다, `ur7e_env.py:263-266`) |
| **PASS** | 리밋 근처에서 팔이 **정지(HOLD)**하고, 계속 밀어도 넘어가지 않으며, 손을 되돌리면 즉시 추종이 재개된다 |
| **복구** | 리더를 반대 방향으로 되돌린다. 텔레옵 EEF면 disengage → 재배치 → engage |

---

## E9 — 특이점

| | |
|---|---|
| **유발** | 팔을 완전 신전(어깨-손목 정렬) 자세로 몰고 간다 |
| **기대 (텔레옵 EEF 경로)** | `sigma_min` 기반 **자동 감속(gamma)**. 감속은 정상 동작이지 결함이 아니다 (`GELLO_UR7E_EEF_MODE.md:198`). `~/eef/state`의 `sigma_min`/`gamma`로 관측 |
| **🛑 기대 (RL 경로)** | **`sigma_min` 감속이 없다.** `PolicyDeltaController`는 `eef_delta` 후반부의 **단순화판**이고, 특이점 감속·keepout·anti-windup·해석적 line search·branch-lock IK가 **전부 빠져 있다** (`serl_ur_infra/README.md`의 현황표). 대신 line search로 스텝을 줄이고, 안 되면 `"STEP_LIMIT"` HOLD (`policy_delta_controller.py:108-134`) |
| **확인** | 텔레옵: `ros2 topic echo /gello_ur_bridge/eef/state`. RL: `info["reject_reason"]` 빈도 |
| **PASS** | 텔레옵: 부드럽게 감속. RL: **HOLD가 폭풍처럼 뜨지 않고** 드물게 `STEP_LIMIT`만 (line search 도입 후 측정치 98% held → 0% held, `policy_delta_controller.py:117`) |
| **복구** | 특이점에서 빠져나오는 방향으로 리더를 움직인다 |

---

## E10 — protective stop (속도 제한 초과)

| | |
|---|---|
| **유발** | 안전하게 유발하려면: `v_max`를 크게 올린 상태에서 리더를 급격히 흔든다. **또는** 브리지의 move-to-start 핸드셰이크를 건너뛴다 (금지 사항이지만 이 시나리오의 정확한 원인이다) |
| **배경** | `forward_position_controller`는 **자체 속도 제한을 하지 않고**, 명령이 너무 빠르면 로봇이 protective stop을 낸다 (`GELLO_UR7E_EEF_MODE.md:178`). 그래서 rate limiting은 전부 우리 쪽(one-euro + 카테시안 거버너 + joint slew clamp)에서 한다 |
| **과거 실사례** | 브리지 첫 명령을 GELLO 포즈로 시드하던 시절 부팅 시 ~0.42 rad(≈210 rad/s)의 스냅이 발생해 실기에서 protective stop을 유발했다. 지금은 **로봇 실제 `/joint_states`에서 시드**한다 — 이건 튜닝 편의가 아니라 **반드시 지켜야 할 안전 설계**다 (`GELLO_UR7E_ROS2_BRINGUP.md:381`) |
| **확인** | `ur_mode`, `safetystatus`, 펜던트 팝업 |
| **PASS** | protective stop이 **정상적으로 걸린다**(=로봇 안전망 작동) + `ur_unlock`/`ur_resend`로 복구된다 |
| **복구** | ① 원인 제거(속도 낮추기) → ② `ur_unlock` → ③ `ur_resend`(B) / `ur_play`(A). **순서를 지킬 것** — 원인을 안 없애고 unlock하면 바로 다시 걸린다 |

---

## E11 — 그리퍼 버스 (`:54321`)

| | |
|---|---|
| **유발** | (a) 그리퍼 노드가 도는 중에 두 번째 `run_ur7e_gripper.sh`를 띄운다, (b) 로봇을 POWER_OFF한다, (c) 팔 드라이버(`use_tool_communication:=true`)와 TCP 모드 그리퍼 노드를 동시에 띄운다 |
| **기대** | 전부 `no/invalid status response` → `_drop_connection()` → 단일 재연결 루프 (`robotiq_gripper_modbus_node.py:203-215`, `:155-162`). **여러 재연결 루프가 동시에 뜨지는 않는다**(`:158-160`의 가드) |
| **확인** | 그리퍼 노드 로그, `ros2 topic hz /robotiq_gripper/position_percent`(멈춤) |
| **⚠️ 조용한 오정보** | env는 `position_percent`가 안 오면 `curr_gripper_pos = 0.0`으로 둔다 = **"완전 열림"으로 보인다** (`ur7e_env.py:394-395`). 그리퍼가 죽은 것과 열린 것이 관측상 구별되지 않는다 |
| **PASS** | ① 두 번째 노드가 조용히 성공하지 않는다 ② 원인 제거 후 자동 재연결 ③ 재연결 후 `set_closed`가 정상 동작 |
| **복구** | (a) 중복 노드 종료 (b) `power on` + `brake release` (c) 그리퍼를 `serial_port:=/tmp/ttyUR`로 전환 → `01_GRIPPER.md` §5 |

---

## E12 (부록) — RL 경로로 그리퍼 구동

`01_GRIPPER.md` §7의 미검증 항목. 장애 주입이 아니라 **미검증 정상 경로**지만 여기서 같이 돈다.

| | |
|---|---|
| **절차** | mock+RViz에서 러너를 띄우고 액션 `[0,0,0,0,0,0,-1]`(닫기) / `[0,...,+1]`(열기)을 쏜다. 실기 그리퍼로 하려면 그리퍼 노드만 실기에 붙인다 |
| **기대** | `-1` → `send_gripper_percent(1.0)`(닫기), `+1` → `0.0`(열기). 단 `GRIPPER_SLEEP = 0.6 s` 디바운스 때문에 **10 Hz 루프에서 6스텝에 1번만** 반영된다 (`ur7e_env.py:436-446`, `config.py:124`) |
| **PASS** | 방향이 맞고, 디바운스가 예상대로 동작하며, `position_percent`가 따라 움직인다 |

---

## E13 (신규) — 🛑 "아무 일도 안 일어난다": 잘못된 인터프리터

장애를 **주입할 필요가 없다.** 오늘 4번 저절로 일어났다. 이건 시나리오가 아니라
**증상 인식 훈련**이다.

| | |
|---|---|
| **증상** | 프로세스가 기동 로그를 몇 줄 찍고 **아무 출력 없이 멈춘다.** 에러도, 타임아웃도, 스택트레이스도 없다. Ctrl-C는 먹힌다 |
| **진짜 원인** | 시스템 `python3`(grpcio 1.30.2)로 gRPC 채널을 만들었다 → `08_OPEN_GAPS.md` G14 |
| **오진하기 쉬운 것** | "네트워크가 느리다", "서버가 안 뜬다", "터널이 안 붙었다", "GPU가 잡혔다" — 전부 아니다 |
| **1분 진단** | ① `ps -L -o pid,tid,pcpu,stat,wchan -p <pid>` → 단일 스레드 **100% CPU / `R` / `wchan` 비어 있음**<br>② `ls -l /proc/<pid>/exe` → `/usr/bin/python3`이면 확정 |
| **즉시 조치** | 죽이고 `/home/laptop3/venvs/gello-hil-actor/bin/python`으로 다시 띄운다. 실기면 `run_hil_actor.sh`를 쓴다 (preflight [2][3]이 이걸 막는다) |
| **오프라인 재현** | `00_SETUP_AND_SAFETY.md` §3.4의 `/tmp/g.py` 스니펫 |
| **PASS** | 위 진단 두 줄로 **60초 안에** 원인을 특정할 수 있다 |

> **왜 매트릭스에 넣는가:** 이 장애의 특징은 **어떤 안전망도 반응하지 않는다**는 것이다.
> 워치독도, 타임아웃도, staleness도 안 걸린다. 팔은 마지막 명령 자세를 홀드하지만
> (프로세스가 살아 있으므로 업샘플러가 계속 발행할 수 있다 → **E3b와 같은 상황**),
> 아무도 그것을 지시하고 있지 않다. **E3b의 "가장 위험한 변종"이 실제로 일어나는 경로가 이것이다.**

---

## E14 (신규) — 리셋이 먼 길로 가는지 (branch-cut)

| | |
|---|---|
| **유발** | 팔을 `RESET_JOINTS`에서 **wrist_3가 반대 분기**인 자세로 옮긴 뒤 `reset()`을 부른다. 실측 사례: 실제 `wrist_3 = +3.1795`, 목표 `-3.1331` (물리적으로 0.029 rad) |
| **기대 (수정 후)** | `go_to_reset()`이 `wrapped_nearest`로 목표를 현재 회전수로 옮긴다 → 명령되는 `wrist_3`는 **+3.1501 부근**. 거리 가드가 보는 값도 6.31이 아니라 branch-safe 값 |
| **🛑 수정 전이었다면** | 6.31 rad를 계산 → 가드가 배선 고장처럼 보이는 에러를 내거나, 통과 시 **손목이 한 바퀴 돌며 2F-85 케이블을 감음**(H3) |
| **확인 (DRY_RUN에서 전부 가능)** | 러너 로그/CSV의 **명령된 관절 목표값**. `wrist_3` 명령값이 현재 값 근처인가? |
| **PASS** | ① 명령 `wrist_3`가 현재 값에서 0.1 rad 이내 ② 거리 가드가 통과 ③ `DRY_RUN=False`로 갔을 때 손목이 한 방향으로 짧게만 움직인다 |
| **⚠️ 순서** | **반드시 `DRY_RUN=True`로 먼저** 로그를 본다. 이 항목은 눈으로 확인하기 전에 arm하면 안 된다 |
| **회귀** | `tests/test_reset_branch_cut.py` (10 passed) |

같이 볼 것: `RESET_MAX_DIST_RAD`가 `0.5` → **`0.9`**로 올라갔다. `0.5`는 23테이크 중
16개의 **정상 종료 자세**를 거부했다. 리셋이 자주 예외로 죽으면 이 값을 의심한다.

---

## 결과 기록표

| ID | 일시 | 유발 방법 | 관측된 동작 | 기대 일치? | 복구 시간 | 비고 |
|---|---|---|---|---|---|---|
| E1a | | | | | | |
| E1b | | | | | | |
| E2 | | | | | | |
| E3a | | | | | | |
| E3b | | | | | | |
| E3c | | | | | | |
| E4a | | | | | | |
| E4b | | | | | | |
| E5a | | | | | | |
| E5c | | | | | | |
| E6 | | | | | | |
| E7a | | | | | | |
| E7b | | | | | | |
| E7c | | | | | | |
| E8 | | | | | | |
| E9 | | | | | | |
| E10 | | | | | | |
| E11a | | | | | | |
| E11b | | | | | | |
| E11c | | | | | | |
| E12 | | | | | | |
| E13 | 2026-07-27 | (주입 불필요 — 실제 발생 4회) | 무출력·CPU 100% 정지 | 예 (사후 특정) | — | 원인 = 시스템 grpcio 1.30.2. venv로 전환해 해결 |
| E14 | | | | | | |
