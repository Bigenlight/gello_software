# HIL-SERL 실기 투입 — 통신·하드웨어 검증 런북 (인덱스)

> 🚩 **새 세션이라면 [`serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md`](../../serl_ur_infra/HANDOFF_NEXT_SESSION_KO.md)를 먼저 읽어라.** 현재 상태·다음 할 일·안전 규칙이 거기 모여 있다. 이 디렉터리는 개별 검증 절차다.

이 디렉터리는 **HIL-SERL을 실제 UR7e에 올리기 전에 통신·하드웨어 경로를 사람이 직접
확인하는 절차**를 담는다. 학습 모델(learner/policy)은 이 문서의 범위가 **아니다**.

```bash
export WT=/home/laptop3/gello_software
```

> ### 🔧 정정 (2026-07-29) — `WT`가 통합 checkout으로 되돌아왔다
> 2026-07-27 판은 `WT=/home/laptop3/gello_worktrees/hil-hardware-comms`를 가리켰다.
> **머지가 끝나서 더 이상 맞지 않는다.** 머지 커밋 `3f199d4`가 `test/hil-hardware-comms`를
> `feat/gello-ur7e-humble-22.04`로 가져왔고, 그 위에 `1b02857`(threshold 0.5→0.2),
> `43ba314`(카메라 시리얼 자동 해석)이 올라와 있다. HIL 신규 코드
> (`serl_ur_infra/ur_experiments/`, `clip_safety_box`, branch-cut `go_to_reset`,
> `ros2_ur_ws/run_hil_actor.sh`)는 이제 **통합 checkout에 전부 있다.** 확인:
>
> ```bash
> git -C /home/laptop3/gello_software log --oneline -3
> ls /home/laptop3/gello_software/serl_ur_infra/ur_experiments   # -> cube_in_cup.py mappings.py __init__.py
> ```
>
> 워크트리 `/home/laptop3/gello_worktrees/hil-hardware-comms`는 아직 디스크에 남아 있지만
> **머지 이전(`1a4f93d`)에 멈춰 있다.** 거기서 04~09를 돌리면 threshold·카메라 해석 등
> 머지 이후 변경이 빠진 코드를 돌리게 된다. **작업은 통합 checkout에서 한다.**

> ### 이 문서의 규칙
> - **검증됨(PASS)** = 실제 하드웨어에서 사람이 확인한 것. 날짜와 실측치를 남긴다.
> - **미검증(TODO)** = 절차만 적혀 있고 아직 실행되지 않은 것. 절대 PASS로 승격하지 말 것.
> - 📌 **기록(RECORD)** = 특정 날짜에 관측된 값. **재입력용 설정값이 아니다.**
>   시리얼·SHA·커밋·GPU 번호·레이턴시처럼 옮겨 적으면 위험한 값은 이 표시 안에만 둔다.
>   기록 블록의 값을 명령줄에 복사하지 말고, 항상 라이브로 다시 확인한다.
> - 중요한 주장에는 `파일:줄` 근거를 단다. 근거 없는 문장은 추측이며 그렇게 표시한다.
> - **줄 번호는 빠르게 낡는다.** 2026-07-27 하루에만 `ur7e_env.py`가 200줄 넘게 밀렸고,
>   머지 이후 다시 밀렸다. `파일:줄`이 안 맞으면 틀린 게 아니라 밀린 것이다 —
>   `rg`로 내용을 다시 찾는다.

---

## 0. 지금 당장 알아야 할 것 (2026-07-29 머지 반영)

1. **🛑 시스템 `python3`로 gRPC를 쓰면 조용히 영구 정지한다.**
   `python3-grpcio 1.30.2`(apt)가 이 머신에서 손상돼 있다. 채널을 하나만 만들어도
   **에러도 로그도 없이 단일 스레드가 CPU 100%로 무한 회전**한다. 반드시
   `/home/laptop3/venvs/gello-hil-actor/bin/python`(grpcio 1.74.0)을 쓴다.
   → `00_SETUP_AND_SAFETY.md` §3.4, `05_COMMS_GRPC.md` §1

2. **✅ `clip_safety_box`는 구현됐지만, 지금까지 팔을 움직인 경로에서는 꺼져 있었다.**
   `UR7eEnv.clip_safety_box()` / `_clip_command_pose()`가 있고 후자가
   `PolicyDeltaController(clip_pose=...)`로 배선되어 **명령 포즈**에 적용된다
   (`ur7e_env.py:188`, `:334`, `:356`; `policy_delta_controller.py:140-141`).
   실측 박스는 `ur_experiments/cube_in_cup.py`에 있다.
   **그러나 2026-07-28에 실제로 팔을 구동한 `run_real_hil.py`는
   `DefaultUR7eEnvConfig`(`ABS_POSE_LIMIT_*` = 0벡터)를 쓰므로 박스가 비활성이었다** —
   REFUSE-DON'T-CLAMP 설계라 경고만 찍고 꺼진다. → `08_OPEN_GAPS.md` G1

3. **19-D `state` 레이아웃은 알파벳순이 정본이다.**
   `[0] gripper_pose | [1:4) tcp_force | [4:10) tcp_pose | [10:13) tcp_torque | [13:19) tcp_vel`.
   `state[..., -1]`은 그리퍼가 **아니라 `tcp_angular_velocity_z`**다.
   `GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`만 쓴다.
   → `05_COMMS_GRPC.md` §4, `06_SENSORS.md` §5
   ⚠️ `serl_ur_infra/RL_RECEIVE_SERVER.md:62-70`은 아직 **옛 v1 순서**(pose6, vel6,
   force3, torque3, gripper 마지막)를 적고 있다. 그 문서를 근거로 쓰지 말 것.

4. **cam2는 손목(wrist) 카메라다.** cam2는 그리퍼에 강체로 물려 있어 손가락이 항상 같은
   픽셀에 있고 배경이 팔 자세를 따라 움직인다
   (`ur_experiments/cube_in_cup.py`의 `IMAGE_CROP` 주석). → `06_SENSORS.md` §1.1

5. **`PYTHONPATH`는 덮어쓰지 말고 이어붙인다.** 덮어쓰면 ROS 오버레이가 날아가
   `ModuleNotFoundError: ur_gello_bringup`이 난다. 반대로 **`serl_ur_infra`의 pytest에는
   ROS `PYTHONPATH`가 붙어 있으면 안 된다** — 수집이 통째로 죽고 `1 skipped`만 뜬다.
   → `00_SETUP_AND_SAFETY.md` §3.4, §4

6. **🟢 분류기 크롭 불일치(G15)는 2026-07-29에 닫혔다 — 재학습이 아니라 분리로.**
   ~~머지가 크롭 불일치를 들여왔고 `IMAGE_CROP`이 활성인데 분류기는 무크롭으로 학습됐다~~는
   더 이상 현재 상태가 아니다. actor가 분류기에게 **자기 몫의 무크롭 128×128 JPEG(sidecar)**를
   약 2 Hz로, **팔이 정지해 있을 때만** 따로 붙여 보낸다. **`IMAGE_CROP`은 그대로다** —
   실측값이고 정책이 1차 소비자다. proto 변경 0, **observation schema hash 불변**.
   → `08_OPEN_GAPS.md` G15, `05_COMMS_GRPC.md` §3.2
   첫 production-model actor run에서 sidecar 설정으로 실제 transition이 들어갔다. 다만
   per-transition verdict를 GUI/영구 로그로 보지 못했고, 팔 가림(occlusion)도 **안 고쳐졌다.**
   그리고 `reward_model_id`가 **`cube-in-cup-all3-ckpt150+sidecar-v1`**로 바뀌어
   옛 값 `cube-in-cup-checkpoint-150`은 **핸드셰이크에서 거부된다.**

7. **`DEFAULT_REWARD_THRESHOLD`는 `0.2`다** (`rlpd_receive_server.py:73`, commit `1b02857`).
   문서에 남아 있던 `--threshold 0.85` / `0.5` 예시는 전부 낡았다. threshold는 learner
   fingerprint에 들어가므로 다른 값으로 학습된 checkpoint resume은 fail-closed로 거부된다.

> ### 통합 상태 (2026-07-29)
> 통합 브랜치 `feat/gello-ur7e-humble-22.04`. 머지 커밋 `3f199d4`(부모 `3ff5f80` + `1a4f93d`)가
> `test/hil-hardware-comms`를 가져왔고, 그 위에 `1b02857`(threshold 0.2),
> `43ba314`·`fb48100`(RealSense 시리얼 live-bus 해석), `792092a`(문서 재정렬)이 있다. 하드웨어 계약 변경은 `6a0b127` /
> merge `248255f`, HIL 안전 수정은 `d49d0f6`~`ee3240e`.
> 작업 전 `git status`를 확인하고 기존 사용자 변경을 지우지 않는 원칙은 그대로다.

---

## 1. 현재 검증 상태표

| # | 항목 | 상태 | 근거 / 실측치 |
|---|---|---|---|
| 1 | UR7e 도달성·상태 | **PASS** | `Robotmode: RUNNING`, `Safetystatus: NORMAL`, remote control `true`, IP `192.168.10.11` |
| 2 | 그리퍼 경로 (Modbus over tool-comm :54321) | **PASS** | 열기 `position_percent`=0.0118, 빈손 완전닫힘=0.8980(=229/255), 액션 `position: 0.085`(=열림) → `reached_goal: true`, 피드백 5.000 Hz (std 1.3 ms) |
| 3 | 그리퍼 **방향** 육안 확인 | **PASS** | crush 게이트 해소. `_send_gripper_command`의 `VERIFY(hw)` 주석(`ur7e_env.py:723-725`) 조건 충족. 📌 2026-07-27 |
| 4 | GELLO 리더 (Dynamixel) | **PASS** | baud 57600, ID1~6 = model 1200, ID7 = model 1190 전부 응답 |
| 5 | GELLO 발행 안정성 | **PASS** | 30.004 Hz (std 0.15 ms), 30초 901샘플, 드롭 0, `comm failed` 0회, 트리거 0.000~1.000 전 구간 |
| 6 | EEF 텔레옵 (실기) | **PASS** (사용자 직접 검증) | `HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef` + `./run_eef_gui.sh` |
| 7 | 오프라인 단위 테스트 `ur_gello_bringup` | **PASS** | **436 passed in 7.09s** (2026-07-29 통합 checkout에서 재실행). §4에 재현 명령 |
| 7b | 오프라인 단위 테스트 `serl_ur_infra` | **PASS** | ⚠️ **기준선이 움직이는 중이다.** `333`(07-29 오전) → `337`(`40b99f8` recorder→demo 변환기) → **`429 passed, 11 skipped in 3.77s`** (sidecar 작업 트리에서 이 문서 작성 중 실측). **코드 에이전트가 아직 붙어 있어 더 오를 수 있다 — 이 숫자를 고정 기준으로 쓰지 말고 매번 다시 돌려라.** 불변인 것은 두 가지다: **skipped는 정확히 11**이어야 하고, **passed가 *내려가면*** PYTHONPATH에서 `serl_launcher`가 빠진 것이며 그때 **skip 사유가 거짓말을 한다** → `00` §4.2 |
| 8 | 타이밍 baseline | **매 실행 재측정** | `test_ur_kin.py`(k)가 매 실행마다 찍는다. 📌 2026-07-29 실측 `worst-case tick = 0.836 ms (generic pose)`, 2026-07-27은 `1.314 ms (near-singular)`. **값도 pose 종류도 실행마다 바뀐다 — 고정값으로 인용하지 말 것.** 판정은 "예산 4.0 ms @250 Hz 미만"이다 |
| 9 | HIL 개입 루프 (mock + RViz) | **미검증(이 브랜치에서)** | 절차는 `serl_ur_infra/RVIZ_HIL_TEST_CLI.md`에 존재. → `04_HIL_INTERVENTION.md` |
| 9b | HIL 개입 루프 (**실기, 팔 구동**) | **PASS (2026-07-28, `run_real_hil.py` 경로에 한함)** | `--arm --scale 0.25`, 100스텝 중 개입 64, `held=0`. 개입 불변식 4종(anchor-latch 0 / gain-latch 0 / 저장==실행 1.000 / held-rate 0%) 통과. **frame-map = 단위행렬**(포화 제외 잔차 0.093, 기준 0.15). → `04` §4.5 |
| 9b′ | 같은 루프를 **actor entrypoint**로 | **핵심 E2E PASS / continuous PARTIAL** | 실제 replay 201, GELLO intervention 153, policy 48. learner 102/gradient 204/policy publish v2. 첫 publish 경계 RPC timeout 뒤 controller cleanup PASS → 최신 상태 문서 |
| 9c | 리더 트리거 → 개입 그리퍼 배선 | **코드 통합·커밋됨, 하드웨어 미검증** | `ros_backend.py:81-155`, `wrappers.py:301-328`, `tests/test_gello_gripper_wiring.py`(📌 2026-07-29 **23 passed**); commit `6a0b127`. 07-28 실기도 이 채널은 껐다 |
| 10 | gRPC actor 루프백 스모크 | **PASS (오프라인)** | `test_actor_grpc_transport/identity_pinning/smoke/rlpd_receive_smoke` = **35 passed** (venv python). 같은 4개 파일이 **시스템 python3에서는 무한 hang** → §0-1 |
| 10b | **Kanu 왕복 (Stage A, fake-env)** | 📌 **PASS (2026-07-27 기록)** | 100스텝 acceptance 통과. 서버 `replay_insert_count: 100`, `state_shape: [8, 1, 19]`. 상대는 zero-action 서버였다. 절차·수치 정본은 [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md) |
| 10c | 레이턴시 실측 (Kanu 왕복) | 📌 **두 세션이 약 6배 다르다 — 세션마다 재측정** | 07-27: RTT p50 **58.6** / p95 75.8 / **p99 97.1 ms**, 링크 약 **13 Mbit/s**. 07-29(**유휴 리그**): ssh-실효 **약 83 Mbit/s**(min 75.5/max 98.5), ICMP p50 **1.75** / p99 24.4 ms, 손실 0%. 링크는 **2.4 GHz ch.3 `iptime_709`**(5 GHz SSID 없음), kanu는 **캠퍼스 4홉**이지 WAN이 아니다. 🛑 **"해결됐다"로 읽지 마라** — 07-29는 카메라·actor·조작자가 **전부 꺼진** 상태였다. 유선 NIC `enx00e04c3600bd`가 **있는데 안 꽂혀 있다** → `05` §5.3 |
| 10d | 관측·sidecar 대역폭 | 📌 **측정됨 (2026-07-29)** | 관측 **98,888 B = 96.57 KiB/step**(이미지는 **raw uint8**, 장당 48 KiB) → 10 Hz **7.911 Mbit/s**. sidecar q95 쌍 **13.32 KiB** @2 Hz → 합계 **8.129 Mbit/s (+2.8 %)**, 부착 스텝 **+8.4 ms** @13 Mbit/s. 기각된 720p passthrough는 쌍 **400 KiB**, +252 ms → `05` §5.4 |
| 11 | RealSense 2대 동시 스트림 | **actor 실기 PASS** | preflight와 실제 actor loop에서 cam1/cam2 약 30 Hz 확인 → `06_SENSORS.md` |
| 11b | RealSense QoS 호환성 | **PASS (해소됨)** | 퍼블리셔가 RELIABLE/TRANSIENT_LOCAL → 백엔드의 기본 reliable 구독과 호환. 이전의 "best-effort면 콜백이 안 뜬다" 우려는 **이 리그에서는 해소**. 단 TRANSIENT_LOCAL 부작용 있음 → `06` §3 |
| 11c | cam1/cam2 역할 | **정정됨 / 손목 배정은 미확정** | cam1 = 삼각대 SCENE, cam2 = 손목(wrist). 다만 **연결된 두 개체 중 어느 쪽이 손목인지는 모델 클래스 추론**이고 육안 미확인이다 → `06` §1.1 |
| 11d | 카메라 시리얼 | **하드코딩 폐기 — 자동 해석 (`43ba314`, `fb48100`)** | 이 리그에 D435 쌍이 **두 벌** 있고 어느 쪽이 열거되는지가 두 번 뒤집혔다. `ros2_ur_ws/_resolve_camera_serials.sh`가 live USB 버스와 대조해 모델 클래스로 배정한다(plain D435→cam1, D435IF→cam2). **문서·명령줄에 특정 시리얼을 적지 말 것** → `06` §1.1 |
| 12 | `clip_safety_box` (워크스페이스 박스) | **구현·단위검증, 실기 경로에서는 비활성** | `tests/test_clip_safety_box.py` **26 passed**. 실측 박스는 `cube_in_cup`에만 있고, 팔을 구동한 `run_real_hil.py`는 `DefaultUR7eEnvConfig`(0벡터)를 써서 박스가 꺼진 채 돌았다 → `08` G1 |
| 12b | `go_to_reset` branch-cut | **실기 경로 PASS** | preposition과 실제 actor의 100-step episode reset 경로를 통과. 기존 단위검증도 유지 → `08` G13 |
| 13 | 장애 주입 매트릭스 | **미검증 (E13 제외)** | → `07_FAILURE_INJECTION.md` |
| 14 | **RL 정책** 경로로 실기 팔 구동 | **원형 PASS / continuous PARTIAL** | 48 non-intervention transition에서 policy가 실제 action 소유. 첫 publish 5.474 s + 동시 contention으로 0.6 s RPC timeout → `08` G21 |
| 15 | reward classifier ↔ 크롭 정합 | 🟡 **sidecar 실기 유입 / verdict 관측 미완료** | 분류기는 무크롭 sidecar, 정책은 crop을 유지한다. production server로 실제 transition은 들어갔으나 per-step `p(success)` GUI/영구 로그와 가림 검증이 남음 → `08` G15/G23 |
| 15b | 분류기 checkpoint SHA pin (orbax 디렉터리) | 🟢 **해결 (2026-07-29)** | `checkpoint_sha256()`이 `classifier_sidecar.directory_sha256()`에 위임. 두 `DEFAULT_*_SHA256`가 폐기된 `e329986b…`(새 도메인 recall 0%)에서 **`512b6575…62846d`**(= `classifier_ckpt/cube_in_cup_all3/checkpoint_150`, 정규 파일 14개)로 교체. **learner fingerprint가 한 번 깨진다 — 의도된 것** → `08` G19 |
| 16 | canonical offline demo artifact | 🟢 **해결 (2026-07-29)** | 사용자가 `take_23` 제외 23개 take를 success로 승인했고 2,037-transition 영구 pickle을 생성했다. laptop3/Kanu strict-load와 SHA256 일치를 확인했다 → `08` G20 |

---

## 2. checkout 지도 (2026-07-29 갱신 — 머지 후)

| 역할 | 경로 | 브랜치 | 상태 |
|---|---|---|---|
| **정본 checkout ⭐ (여기서 작업한다)** | `/home/laptop3/gello_software` | `feat/gello-ur7e-humble-22.04` | `ur_experiments/`, `clip_safety_box`, branch-cut reset, `run_hil_actor.sh`, `run_hil_preposition.sh` **전부 여기 있다** |
| HIL 하드웨어·통신 작업본 (구) | `/home/laptop3/gello_worktrees/hil-hardware-comms` | `test/hil-hardware-comms` | **머지 완료 (`1a4f93d` → `3f199d4`). 머지 이전에 멈춰 있다 — 여기서 실행하지 말 것** |
| 보상 오버라이드 작업본 | `/home/laptop3/gello_worktrees/human-reward-override` | `feat/human-reward-override` | (범위 밖) |

```bash
git -C /home/laptop3/gello_software worktree list
git -C /home/laptop3/gello_software log --oneline -3   # 3f199d4 머지가 보여야 한다
```

세션 시작마다 `git status`와 `git submodule status`를 확인하고, checkout에 이미 있던
사용자 변경이나 submodule 내 산출물을 `reset`, `clean`, `stash`로 지우지 않는다.
`third_party/hil-serl`은 pinned submodule이며 직접 수정하지 않는다.

> ⚠️ **여러 에이전트/사람이 같은 checkout을 동시에 고치고 있다.** 파일이 몇 분 사이에
> 바뀔 수 있다. 숫자를 인용하기 전에 `git log --oneline -5`로 최신 커밋을 확인할 것 —
> 실제로 `ABS_POSE_LIMIT_LOW[2]`가 `0.1785` → `0.185`로, `RESET_MAX_DIST_RAD`가
> `0.5` → `0.9`로(`ee3240e`), `DEFAULT_REWARD_THRESHOLD`가 `0.85` → `0.5` → `0.2`로
> (`53d5cf6`, `1b02857`) 바뀌었다.

---

## 3. 문서 목록

| 파일 | 내용 |
|---|---|
| [`00_SETUP_AND_SAFETY.md`](00_SETUP_AND_SAFETY.md) | checkout 셋업·빌드·환경변수 함정(인터프리터/`PYTHONPATH`), 오프라인 테스트 2종, 세션 전 체크리스트, **비상 정지 우선순위와 "정지가 아닌 것들"** |
| [`01_GRIPPER.md`](01_GRIPPER.md) | Robotiq 2F-85 단독 검증(**PASS**) + RL/개입 배선과 `:54321` 단일 클라이언트 규칙 |
| [`02_GELLO_LEADER.md`](02_GELLO_LEADER.md) | GELLO 리더 검증(**PASS**) + Dynamixel 진단 스캔 + 트리거가 별도 토픽인 이유 |
| [`03_EEF_MODE.md`](03_EEF_MODE.md) | EEF 텔레옵 단계 상승 P6 → P7 → P8 → P9a → P9b (사용자 검증 완료, 재현 절차) |
| [`04_HIL_INTERVENTION.md`](04_HIL_INTERVENTION.md) | 데드맨 2종, 앵커/gain 래치, mock RViz 루프, **실기 러너 `run_real_hil.py`**, 좌표계 3×3, 그리퍼 개입, 개입 메타데이터 계약 |
| [`05_COMMS_GRPC.md`](05_COMMS_GRPC.md) | venv 격리, 루프백 스모크, 포트 기본값, schema fail-fast(v2), **분류기 sidecar 전송 계약(§3.2)**, **레이턴시·대역폭 실측(§5.3–5.4)**, Kanu 터널 |
| [`06_SENSORS.md`](06_SENSORS.md) | RealSense 2대(시리얼·크롭·역할), QoS/TRANSIENT_LOCAL 함정, 토픽 유량 점검, 19-D state 계약, F/T 프레임 |
| [`07_FAILURE_INJECTION.md`](07_FAILURE_INJECTION.md) | 장애 주입 매트릭스 E1~E14 (유발·기대·확인·PASS·복구) + 결과 기록표 |
| [`08_OPEN_GAPS.md`](08_OPEN_GAPS.md) | 안전 갭 **G1~G20**과 임시 완화책, 그리고 다른 문서에서 발견된 낡은 서술 목록. G15/G19/G20은 2026-07-29에 닫혔다 |
| [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md) | **HIL actor 기동 런북** — 정상 운용용 3-CLI(`run_hil_server.sh` / `run_hil_hardware.sh` / `run_hil_session.sh`), `run_hil_actor.sh` preflight, actor·sidecar 옵션, Stage A fake-env / Stage B 실센서, Kanu 서버 기동 |

관련 기존 문서(이 디렉터리 밖, 읽기 전용 참조):

- `docs/ros2/GELLO_UR7E_EEF_MODE.md` — EEF 모드 정본. `03_EEF_MODE.md`는 여기서 파생.
- `docs/ros2/GELLO_UR7E_GRIPPER.md` — 그리퍼 경로 정본.
- `docs/ros2/GELLO_UR7E_SETUP_CLI.md` — 실기 텔레옵 세션 체크리스트 정본.
- `serl_ur_infra/RVIZ_HIL_TEST_CLI.md` — mock HIL 4터미널 절차 정본.
- `serl_ur_infra/REMOTE_ACTOR_GRPC.md`, `serl_ur_infra/RL_RECEIVE_SERVER.md` — gRPC/수신서버 정본.
- `serl_ur_infra/README.md` — env 설계 요점 + `PolicyDeltaController` 현황표.

---

## 4. 전체 실행 순서 (권장)

각 단계는 **앞 단계가 PASS일 때만** 진행한다. 굵은 글씨는 로봇이 실제로 움직이는 단계다.

```
[A] 오프라인 (로봇 불필요, 위험 0)
 A1  ROS2 워크스페이스 빌드             -> 00 §2
 A2  ur_gello_bringup 단위테스트 436개  -> 00 §4.1  [PASS 2026-07-29]
 A3  serl_ur_infra 단위테스트 (개수 변동) -> 00 §4.2  [PASS 2026-07-29, 429p/11s]
 A4  gRPC 루프백 스모크 (mock 서버)     -> 05 §2    [PASS 오프라인]
        ↓
[B] 하드웨어 단독 (팔 미동작)
 B1  로봇 도달성 / dashboard 상태       -> 00 §5   [PASS]
 B2  그리퍼 단독                        -> 01      [PASS]
 B3  GELLO 리더 단독                    -> 02      [PASS]
 B4  RealSense 2대                      -> 06 §1   [actor 실기 PASS]
        ↓
[C] mock 하드웨어 + RViz (실기 위험 0)
 C1  mock RViz HIL 개입 루프            -> 04 §3   [미검증]
 C2  좌표계 3x3 검증                    -> 04 §5   [오프라인 실측으로 대체 통과]
        ↓
[D] 실기 EEF 텔레옵 (팔 움직임)  ** 사람이 E-STOP 위에 손 **
 D1  P6  pos_scale:=0.0                 -> 03 §4   [사용자 검증 완료]
 D2  P7  병진 위주                      -> 03 §5   [사용자 검증 완료]
 D3  P8  회전 위주                      -> 03 §6   [사용자 검증 완료]
 D4  P9a / P9b 6-DoF + 재클러치         -> 03 §7   [사용자 검증 완료]
        ↓
[D'] 실기 HIL 개입 (팔 움직임, zero-policy)  ** run_real_hil.py **
 D'1 DRY_RUN 300스텝 + CSV 검토          -> 04 §4.5 [PASS 2026-07-28]
 D'2 --arm --scale 0.25, 개입 불변식 4종 -> 04 §4.5 [PASS 2026-07-28]
        ↓
[E] 장애 주입 (팔 움직임 포함)
 E1  통신/프로세스 계열 (E1~E5)         -> 07      [미검증]
 E2  로봇 안전 계열 (E6~E11)            -> 07      [미검증]
        ↓
[F] 원격 통신 (Kanu)
 F1  SSH 터널 + 스키마 핸드셰이크       -> 05 §6, 09 §2   [PASS 2026-07-27]
 F2  레이턴시 예산 실측                  -> 05 §5.3       [세션마다 재측정 — 값이 6배 흔들린다]
 F3  Stage A actor (fake-env) 왕복       -> 09 §3         [PASS 2026-07-27]
 F3b 분류기 sidecar production 전송       -> 09 §4.4       [배선 PASS / verdict 미확인]
 F4  no-arm live-sensor policy probe      -> 09 §4         [PASS, transition 0]
        ↓
[G] RL 정책 경로 실기 first E2E          [핵심 PASS / continuous PARTIAL]
 G1  policy/GELLO 실제 action 전환        [PASS]
 G2  replay201 -> learner102 -> publish v2 [PASS, actor의 v1/v2 수신은 미확인]
 G3  첫 publish 경계 RPC deadline          [FAIL — 08 G21]
```

**현재 위치: 실물 HIL-SERL 원형 [G1~G2]까지 도달했다. 다음 blocker는 [G3] 동시
학습/추론 latency와 policy-first/episode GUI 운영 흐름이다.**

> - [F]가 [C]/[E]보다 먼저 끝난 것은 순서를 어긴 게 아니라, [F]가 **로봇을 전혀 움직이지 않는
>   fake-env 경로**이기 때문이다.
> - [D′]가 [C]보다 먼저 끝난 것은 순서를 어긴 것이 맞다. mock RViz 루프를 건너뛰고 실기에서
>   개입 경로를 검증했다. 그래서 [C]는 **여전히 미검증이고**, mock 전용 완화값
>   (`run_rviz_hil.py`의 `DRY_RUN=False` / `ACTION_SCALE 0.3 m/s`)을 실기 config로
>   가져오지 않도록 특히 조심해야 한다 → `04` §3.
> - [D′]에서 움직인 것은 zero-policy + 사람 개입이었지만, [G]에서는 실제 policy 48 transition과
>   GELLO intervention 153 transition이 실행됐다. 두 실험을 섞어 인용하지 않는다.
> - 이전 판의 “[G] 금지” gate는 사용자의 명시적 승인 아래 first smoke에서 넘어갔다. 이는
>   원형 검증 승인이지 `08_OPEN_GAPS.md`의 모든 항목이 닫혔다는 뜻이 아니다.
