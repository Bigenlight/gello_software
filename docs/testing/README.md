# HIL-SERL 실기 투입 — 통신·하드웨어 검증 런북 (인덱스)

> 🚩 **새 세션이라면 루트 [`CLAUDE.md`](../../CLAUDE.md) → [`serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`](../../serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md) 순서로 읽어라.** 현재 상태·다음 할 일·안전 규칙이 거기 모여 있다. 이 디렉터리는 개별 검증 절차다.
> *(이전 판(보존)은 `HANDOFF_NEXT_SESSION_KO.md`를 진입점으로 지목했다. 그 문서는 **첫 실물 E2E 이전**의 리그 조사 기록이라 최신 상태 지침이 아니다 — 다만 §"테스트 — 이 명령 그대로"(`:254-264`)는 여전히 pytest 재현 명령의 정본이다.)*

이 디렉터리는 **HIL-SERL을 실제 UR7e에 올리기 전에 통신·하드웨어 경로를 사람이 직접
확인하는 절차**를 담는다. 학습 모델(learner/policy)은 이 문서의 범위가 **아니다**.

```bash
export WT=/home/laptop3/gello_software
```

> ## 🔴 GPU 서버가 바뀌었다 — `kanu` → `junhyeong_ai` (2026-07-31)
>
> learner는 **`junhyeong_ai`**(166.104.146.29, hostname `junhyeong`)에서 돈다.
> 터널은 laptop3 `127.0.0.1:50153` → junhyeong_ai `127.0.0.1:50053`으로 **변함없다.**
>
> 🟢 **이 디렉터리의 절차 중 바뀐 것은 사실상 없다.** `run_hil_hardware.sh`에는 서버 참조가
> 한 줄도 없고, `run_hil_session.sh`/`run_hil_actor.sh`는 언제나 터널의 **로컬 입구
> `127.0.0.1:50153`**만 본다 — 반대편이 어느 머신인지 원래부터 몰랐다.
> 호스트가 바뀐 것은 **Terminal 1의 `run_hil_server.sh` 하나뿐**이고, 그 기본값이 이미
> 새 서버라 **환경변수를 하나도 줄 필요가 없다**. 옛 이름 `HIL_KANU_REPO` /
> `HIL_KANU_PYTHON`은 `HIL_REMOTE_REPO` / `HIL_REMOTE_PYTHON`의 alias로 살아 있고,
> `HIL_REMOTE_DATA_ROOT`(`/home/junhyeong/hil-serl-data`)가 신설됐다.
>
> ⚠️ `run_hil_server.sh`는 **kanu를 더 이상 구동하지 못한다(의도된 fail-closed)** — kanu의
> classifier가 데이터 뿌리 밖에 있어 단일 `HIL_REMOTE_DATA_ROOT`로 기술되지 않는다.
> kanu는 **읽기 전용**으로만 본다(`ssh kanu 'ps -p <pid> -o pid,etime'`).
> **거기서 아무것도 종료하지 말 것** — 옛 learner는 아직 살아 있고 사용자 소유다.
>
> 📌 **이 문서의 `Kanu` 표기 대부분은 07-27~07-30 기록이다.** 측정값과 PASS 근거는 그날 그
> 호스트의 사실이므로 지우지 않고 🗄️ 표시만 붙였다 — 새 호스트를 비교할 유일한 기준선이다.
> 정본: [`serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md`](../../serl_ur_infra/DATA_AND_MODELS_JUNHYEONG_AI_KO.md) ·
> [`serl_ur_infra/SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](../../serl_ur_infra/SERVER_MIGRATION_E2E_JUNHYEONG_AI.md) ·
> 조작자 절차와 터널 소유권은 [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md) §5.4.

> ### 🆕 현재 운영 요약 (2026-07-30 · 🗄️ **당시 서버는 `kanu`**)
> 실물 HIL-SERL 원형은 구동됐다. 정상 운용은 `run_hil_server.sh` / `run_hil_hardware.sh` /
> `run_hil_session.sh` 3-CLI다. actor transport는 protocol 2 / schema 3, reward threshold는
> 0.5다. 성공 기본값은 MANUAL이지만 classifier는 계속 실행·표시·replay 기록된다.
> terminal 뒤에는 `WAIT_HOME_APPROVAL` → GUI HOME 승인 → `WAIT_SCENE_READY` → 사람이 장면
> 재배치 → START/NEXT 순서다. startup의 예전 GO/Enter 타이핑은 기본 경로에서 제거됐지만
> pose/controller proof와 fresh deadman heartbeat 3개는 남아 있다
> (🔧 요구 상태는 07-30 저녁부터 **DISENGAGED**다 — 표 17행, `09` §1.4).
>
> **개입 제어 경로가 2026-07-30 저녁에 바뀌었다 (`edbb3f5`).** `UR7eEnv`의 **배경 데몬 추종
> 스레드**가 ENGAGED 동안 30 Hz로 리더를 따라가고 `env.step`은 관찰자다
> (`INTERVENTION["follow_mode"]="background"`). `InterventionBudget`은 개입 **제어** 경로에서
> 제거됐다 — 팔의 상한은 governor + 워크스페이스 박스 + 250 Hz 업샘플러뿐이다. 이 표의
> 9d/9d′ 행은 옛 `in_window` 경로의 실측이며 **새 기본 경로의 증거가 아니다.**
> 🛑 두 대기 상태(`WAIT_HOME_APPROVAL` / `WAIT_SCENE_READY`)에서는 `suspend_follower()`가
> 아직 배선되지 않았고 후자 경로는 데드맨을 보지 않는다 — **그 화면에서는 GELLO에서 손을
> 떼라.** → 표 9d″

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
   ⚠️ `serl_ur_infra/RL_RECEIVE_SERVER.md`(§ "Exact observation contract")은 **옛 v1 순서**(pose6, vel6,
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
   production-model actor에서 sidecar transition과 실제 GUI probability/threshold/verdict를
   관측했다. 별도 장시간 영구 verdict 로그와 팔 가림(occlusion)은 **안 고쳐졌다.**
   그리고 `reward_model_id`가 **`cube-in-cup-all3-ckpt150+sidecar-v1`**로 바뀌어
   옛 값 `cube-in-cup-checkpoint-150`은 **핸드셰이크에서 거부된다.**

7. **`DEFAULT_REWARD_THRESHOLD`는 현재 `0.5`다** (`rlpd_receive_server.py`).
   0.85/0.2는 과거 lineage 기록이다. threshold는 learner
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
| 7 | 오프라인 단위 테스트 `ur_gello_bringup` | **PASS** | 📌 **489 passed** (시스템 `python3` + ROS overlay, 7.48 s). **서버 이전으로 바뀌지 않았다 — 2026-07-31 재확인.** 이전 값 `436 passed in 7.09s`(2026-07-29)는 END EPISODE 버튼·operator 경로 추가 이전이다. §4에 재현 명령. 🛑 **7b의 수와 절대 합치지 말 것 — 인터프리터도 PYTHONPATH도 다르다** |
| 7b | 오프라인 단위 테스트 `serl_ur_infra` | **PASS** | 📌 **`768 passed, 11 skipped, 1 xfailed`** (actor venv `/home/laptop3/venvs/gello-hil-actor/bin/python`, numpy 2.2.6, 13.86 s). **서버 이전으로 바뀌지 않았다 — 2026-07-31 재확인.** 계보: `333`(07-29 오전) → `337`(`40b99f8`) → `429`(classifier sidecar) → `497`(07-30 오전) → `579`(`4197f5b`) → `595`(`ee8af5e`) → `701`(`d6965a9`; 배경 추종 스레드 · 컨트롤러 스레드 안전화 · norm 축소 회귀 = `test_intervention_follower`) → **`768`**(07-30 저녁; END EPISODE + DISENGAGED gate + 재기동 복구 = `test_operator_abort` · `test_actor_abort_lifecycle`). 🛑 **`1 xfailed`를 빼고 인용하지 말 것** — 알려진 결함의 못이다(저장 액션이 IK line-search 경로에서 실행 액션을 과대 진술할 수 있다; strict xfail이라 고치면 XPASS로 터진다). ⚠️ **인터프리터를 안 적은 passed 수는 무의미하다** — `venvs/hilserl`(jax 0.5.3, numpy 1.26.4)은 jax skip들이 실제로 돌아 **`741 / 4 / 1`**이 된다. ✅ 한때 여기서 났던 `1 failed`(`test_governor_dt.py::test_env_step_surfaces_governed_in_info`)는 **해소됐다** — numpy 승격 차이가 아니라 허용범위가 float32 산술보다 타이트했던 것(`rel=1e-9` → `1e-6`), 근거는 `00` §4.2. 불변인 것: **actor venv에서 skipped는 정확히 11**, **passed가 *내려가면*** PYTHONPATH에서 `serl_launcher`가 빠진 것이고 그때 **skip 사유가 거짓말을 한다** → `00` §4.2. 🪤 `tests/test_env_fake_backend.py`는 **0개 수집**되어 이 총계에 흔적이 없다 → `08` G25 |
| 8 | 타이밍 baseline | **매 실행 재측정** | `test_ur_kin.py`(k)가 매 실행마다 찍는다. 📌 2026-07-29 실측 `worst-case tick = 0.836 ms (generic pose)`, 2026-07-27은 `1.314 ms (near-singular)`. **값도 pose 종류도 실행마다 바뀐다 — 고정값으로 인용하지 말 것.** 판정은 "예산 4.0 ms @250 Hz 미만"이다 |
| 9 | HIL 개입 루프 (mock + RViz) | **미검증(이 브랜치에서)** | 절차는 `serl_ur_infra/RVIZ_HIL_TEST_CLI.md`에 존재. → `04_HIL_INTERVENTION.md` |
| 9b | HIL 개입 루프 (**실기, 팔 구동**) | **PASS (2026-07-28, `run_real_hil.py` 경로에 한함)** | `--arm --scale 0.25`, 100스텝 중 개입 64, `held=0`. 개입 불변식 4종(anchor-latch 0 / gain-latch 0 / 저장==실행 1.000 / held-rate 0%) 통과. **frame-map = 단위행렬**(포화 제외 잔차 0.093, 기준 0.15). → `04` §4.5 |
| 9b′ | 같은 루프를 **actor entrypoint**로 | **핵심 E2E PASS** | 실제 policy/GELLO 전환, replay 201, intervention 153, learner update를 관측. 정상 832.3 ms reply를 거부하던 옛 0.6/0.8 s 경계는 bounded 1.5/2.0 s로 완화. 장시간 tail 계측은 남음 |
| 9c | 리더 트리거 → 개입 그리퍼 배선 | **코드 통합·커밋됨, 하드웨어 미검증** | `ros_backend.py`의 `GELLO_TRIGGER_TOPIC`·`GELLO_TRIGGER_STALE_S`·`_on_gello_trigger`, `wrappers.py::GelloIntervention._expert_gripper`, `tests/test_gello_gripper_wiring.py`(📌 2026-07-30 재측정 **24 passed**); commit `6a0b127`. 07-28 실기도 이 채널은 껐다. *(줄 번호 제거 2026-07-30: 옛 판의 `ros_backend.py:81-155` / `wrappers.py:301-328`은 `4197f5b`로 크게 밀렸고 `wrappers.py`는 아직 동시 편집 중이다 — **심볼로 찾을 것.**)* |
| 9d | 개입 **손맛** 1차 — `follow_mode="in_window"` (창 안 30 Hz 재샘플링 + One-Euro + 변위 예산) | **PASS (2026-07-30 오전, 실제 UR7e, `4197f5b`)** · 🗄️ **이 경로는 더 이상 기본값이 아니다** | `run_real_hil.py` **3 run**(DRY `--scale 0.5` / DRY `--scale 1.0` / ARMED `--scale 1.0`) 중 뒤 두 run PASS, 첫 run은 고친 판정으로 **SKIP**(포화 제외 후 축별 여기 2 cm 미달). **DRY RUN `--scale 1.0`** 300스텝(개입 272): frame-map 잔차 0.016 / alpha 1.005 / 표본 141(포화 131 제외) · action-exec dp_ratio 중앙값 1.000 · held 0 %. **ARMED `--scale 1.0 --max-steps 150`**(개입 120): 잔차 0.130 / alpha 0.983 / 표본 51(포화 69 제외) · dp_ratio 1.000 · held 0 % · **조작자 손맛 확인 양호**. 모든 개입 스텝 `substeps=2`(창당 타깃 3회 갱신), `governed=0`(예산이 governor보다 타이트해 먼저 묶는 설계대로) → `04` §9 |
| 9d′ | 같은 손맛 수정의 **연속 운용**(창이 늘어질 때) | 🗄️ **이전 판(보존) — `background`가 이 결론을 뒤집었다** | 옛 판정: *"미검증 — 코드로는 못 고친다. 창 0.700 s에서 HOLD 51.5 %, 리더속도 추종 66 %. 예산 소진 후 남는 HOLD는 필터·rate·외삽 어느 것도 못 없앤다."* 그 판정은 `in_window` + `InterventionBudget` 전제였다. `edbb3f5`에서 추종이 RL 창 밖으로 나가고 예산이 제거되어 **창 길이가 개입 속도를 더 이상 묶지 않는다.** G21은 여전히 열려 있지만 이제 **transition 밀도**·policy 반응성 문제이지 손맛 문제가 아니다 → `08` G21, `04` §9.5 |
| 9d″ | 개입 손맛 2차 — `follow_mode="background"` (배경 데몬 추종 스레드, **shipped 기본값**) | 🟡 **조작자 확인만 — 체계적 실기 검증 미완** | 조작자 실기 코멘트 "개입 속도는 좀 고쳐졌어"(📌 2026-07-30, `edbb3f5`). 9d처럼 **CSV 불변식 전수(frame-map 잔차 / alpha / dp_ratio / held-rate)를 돌리지 않았다** → PASS로 승격하지 말 것. 설계 실측(오프라인·커밋 근거): 개입 최고속 2.4 → **15 cm/s**(⚠️ 커밋 요약의 "12.5 cm/s"는 `ACTION_SCALE[0]×HZ` 유래라 틀렸다. `follow_xi`에는 예산도 `_paced_request`도 없으므로 실제 상한은 governor `v_max` 0.15 m/s이고, 코드 4곳이 인용하는 검증 리그 EEF teleop 실측은 12.4 cm/s다 → `04` §9 말미 📌), 창 밖 HOLD 27.7 % → 없음, 업샘플러 slew 상한 46 % → 100 %, **데드맨 release 512 ms → 33 ms**(추종 틱마다 재읽기 — 손맛이 아니라 안전 이득). 함께 고쳐진 것: `PolicyDeltaController` 스레드 안전화(찢어진 `(T_cmd,q_cmd)` 11.1 % / HOLD 틱이 움직임 발행 25건 최대 0.2047 rad / `dq_step_max` 누출 6.08 %), One-Euro 시정수 창 개수 → 벽시계(`6ca35ac`, 실효 tau 2.945 → 0.498 s). 🛑 **실기에서 한 번 터졌다**(`d6965a9`): 기록 액션을 축별 `np.clip`으로 자르면 `RelativeFrame` 회전 뒤 `[-1,1]`을 벗어나 `ActorProtocolError`로 actor가 즉사한다 — **norm 비례 축소만** → 루트 `CLAUDE.md` "반드시 지킬 것", `04` §9 |
| 9d‴ (G32) | 배경 추종 중 **operator 대기 상태**(`WAIT_HOME_APPROVAL` / `WAIT_SCENE_READY`) | 🛑 **알려진 갭 — 미배선** | `UR7eEnv.suspend_follower()`는 구현돼 있으나 **production 호출부가 0개**다(📌 2026-07-30 `rg suspend_follower` → `tests/test_intervention_follower.py`만). 두 대기 상태는 RL 스레드를 무한 블록하고 `WAIT_HOME_APPROVAL` 경로는 데드맨을 보지 않는다 → **그 화면에서 GELLO를 잡으면 팔이 따라온다.** 현재 완화책은 조작자 안내뿐 |
| 9f | 그리퍼 열기 — 세션 시작 + 매 에피소드 경계 | **코드 통합, 실기 미검증** | `run_hil_preposition.sh` `[5b/6]`(`OPEN_GRIPPER=0`으로 끔, `GRIPPER_OPEN_WAIT_S` 대기)와 `UR7eEnv.open_gripper_for_reset`. 근거: 모든 offline demo가 열린 그리퍼에서 시작하고 `gripper_position`은 `state[0]`이라, 닫힌 채 시작하면 첫 스텝부터 OOD다 |
| 9e | 개입 서브스텝의 **mock RViz** 확인 | **미검증** | `04` §3 루프는 이번에도 건너뛰었다. 실기 PASS가 mock PASS를 대체하지 않는다(G24 검증 순서 ②) |
| 10 | gRPC actor 루프백 스모크 | **PASS (오프라인)** | `test_actor_grpc_transport/identity_pinning/smoke/rlpd_receive_smoke` = **35 passed** (venv python). 같은 4개 파일이 **시스템 python3에서는 무한 hang** → §0-1 |
| 10b | **학습 서버 왕복 (Stage A, fake-env)** | 📌 **PASS (2026-07-27 기록)** 🗄️ *(서버 = kanu)* | 100스텝 acceptance 통과. 서버 `replay_insert_count: 100`, `state_shape: [8, 1, 19]`. 상대는 zero-action 서버였다. 새 호스트에서 이 형태를 다시 돌리지는 않았고 21행의 200-transition 합성 acceptance가 같은 경로를 덮었다. 절차 정본은 [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md) |
| 10c | 레이턴시 실측 (학습 서버 왕복) | 📌 **세션마다 재측정 — 호스트가 바뀌었고 세션 간 편차도 크다** | 🗄️ **kanu 기록:** 07-27 RTT p50 **58.6** / p95 75.8 / **p99 97.1 ms**, 링크 약 **13 Mbit/s**. 07-29(**유휴 리그**) ssh-실효 **약 83 Mbit/s**(min 75.5/max 98.5), ICMP p50 **1.75** / p99 24.4 ms, 손실 0%. 링크는 **2.4 GHz ch.3 `iptime_709`**(5 GHz SSID 없음), kanu는 **캠퍼스 4홉**이지 WAN이 아니다. 🛑 **"해결됐다"로 읽지 마라** — 07-29는 카메라·actor·조작자가 **전부 꺼진** 상태였다. 유선 NIC `enx00e04c3600bd`가 **있는데 안 꽂혀 있다** → `05` §5.3. 🆕 **junhyeong_ai (2026-07-31, 합성 200-transition E2E):** per-RPC BeginEpisode **57.7 ms 평균 / 82.2 최대**, Step **156.1 ms 평균 / 211.0 최대** 🗄️ vs kanu BeginEpisode **84.9 / 372.8** — **tail 약 4.5배 단축**. **ICMP RTT는 두 호스트가 같으므로 이득은 네트워크가 아니라 호스트 연산이다** → 21행 |
| 10d | 관측·sidecar 대역폭 | 📌 **측정됨 (2026-07-29)** | 관측 **98,888 B = 96.57 KiB/step**(이미지는 **raw uint8**, 장당 48 KiB) → 10 Hz **7.911 Mbit/s**. sidecar q95 쌍 **13.32 KiB** @2 Hz → 합계 **8.129 Mbit/s (+2.8 %)**, 부착 스텝 **+8.4 ms** @13 Mbit/s. 기각된 720p passthrough는 쌍 **400 KiB**, +252 ms → `05` §5.4 |
| 11 | RealSense 2대 동시 스트림 | **actor 실기 PASS** | preflight와 실제 actor loop에서 cam1/cam2 약 30 Hz 확인 → `06_SENSORS.md` |
| 11b | RealSense QoS 호환성 | **PASS (해소됨)** | 퍼블리셔가 RELIABLE/TRANSIENT_LOCAL → 백엔드의 기본 reliable 구독과 호환. 이전의 "best-effort면 콜백이 안 뜬다" 우려는 **이 리그에서는 해소**. 단 TRANSIENT_LOCAL 부작용 있음 → `06` §3 |
| 11c | cam1/cam2 역할 | **정정됨 / 손목 배정은 미확정** | cam1 = 삼각대 SCENE, cam2 = 손목(wrist). 다만 **연결된 두 개체 중 어느 쪽이 손목인지는 모델 클래스 추론**이고 육안 미확인이다 → `06` §1.1 |
| 11d | 카메라 시리얼 | **하드코딩 폐기 — 자동 해석 (`43ba314`, `fb48100`)** | 이 리그에 D435 쌍이 **두 벌** 있고 어느 쪽이 열거되는지가 두 번 뒤집혔다. `ros2_ur_ws/_resolve_camera_serials.sh`가 live USB 버스와 대조해 모델 클래스로 배정한다(plain D435→cam1, D435IF→cam2). **문서·명령줄에 특정 시리얼을 적지 말 것** → `06` §1.1 |
| 12 | `clip_safety_box` (워크스페이스 박스) | **구현·단위검증, 실기 경로에서는 비활성** | `tests/test_clip_safety_box.py` **26 passed**. 실측 박스는 `cube_in_cup`에만 있고, 팔을 구동한 `run_real_hil.py`는 `DefaultUR7eEnvConfig`(0벡터)를 써서 박스가 꺼진 채 돌았다 → `08` G1. 🛑 **2026-07-30부터 이게 더 위험해졌다** — 개입 변위 예산이 제거되면서 박스가 **유일한 위치 상한**이 됐다. 즉 `run_real_hil.py --arm`은 production 3-CLI 경로보다 **덜 안전하다** |
| 12b | `go_to_reset` branch-cut | **실기 경로 PASS** | preposition과 실제 actor의 100-step episode reset 경로를 통과. 기존 단위검증도 유지 → `08` G13 |
| 13 | 장애 주입 매트릭스 | **미검증 (E13 제외)** | → `07_FAILURE_INJECTION.md` |
| 14 | **RL 정책** 경로로 실기 팔 구동 | **원형 PASS** (🗄️ 07-30 kanu) · 🆕 **junhyeong_ai에서 재현 (07-31)** | policy가 실제 action을 소유하고 ENGAGE로 GELLO 개입, 해제 뒤 policy 복귀를 관측. 07-31 세션은 새 서버 상대로 replay 316 / intervention 210까지 진행했다(21행). 장시간 latency는 `08` G21 |
| 15 | reward classifier ↔ 크롭 정합 | 🟢 **sidecar + GUI 실기 관측 PASS** · 🆕 **새 서버에서 실 sidecar ingress 확인 (07-31)** | 분류기는 무크롭 sidecar, 정책은 crop 유지. 실제 `p(success)`/threshold/verdict 표시 확인. MANUAL에서도 계속 돈다. checkpoint는 `junhyeong_ai:~/hil-serl-data/classifier_ckpt/checkpoint_150`으로 옮겨졌고 **디렉터리 SHA `512b6575…62846d`는 그대로다**. 영구 audit/가림은 남음 → `08` G15 |
| 15c | operator episode 상태기계 | **구현·schema-3 실기 진행** | MANUAL/AUTO, MARK SUCCESS, WAIT_HOME_APPROVAL, APPROVE HOME, WAIT_SCENE_READY, START/NEXT 구현. episode-limit GUI와 schema-3 online 학습 관측 완료; MARK SUCCESS episode의 one-shot provenance 재확인만 남음 → `08` G23 |
| 15b | 분류기 checkpoint SHA pin (orbax 디렉터리) | 🟢 **해결 (2026-07-29)** | `checkpoint_sha256()`이 `classifier_sidecar.directory_sha256()`에 위임. 두 `DEFAULT_*_SHA256`가 폐기된 `e329986b…`(새 도메인 recall 0%)에서 **`512b6575…62846d`**(= `classifier_ckpt/cube_in_cup_all3/checkpoint_150`, 정규 파일 14개)로 교체. **learner fingerprint가 한 번 깨진다 — 의도된 것** → `08` G19 |
| 16 | canonical offline demo artifact | 🟢 **해결 (2026-07-29)** · 🆕 **서버 이전 후에도 동일 (07-31)** | 사용자가 `take_23` 제외 23개 take를 success로 승인했고 2,037-transition 영구 pickle을 생성했다. 🗄️ 07-29에 laptop3/kanu strict-load와 SHA256 일치를 확인했다. 이제 정본은 `junhyeong_ai:~/hil-serl-data/demos/` (SHA256 `f9718558…032fa`, 195 MB)이고 laptop3·kanu 사본까지 **3벌 전부 같은 해시**다. **SHA 핀은 경로 핀이 아니라 내용 핀이라 이전을 그대로 통과했고, 오히려 전송 정확성을 검사해 줬다** → `08` G20 |
| 17 | 세션이 **DISENGAGED**로 시작 (시작 ENGAGE 불필요) | **코드 통합, 실기 미검증** | 세 gate 전부(`run_hil_session.sh` 폴링 · preflight `[11]` · `[ARM]` 재검증)가 fresh **DISENGAGED** heartbeat 3개를 요구한다. `_hil_deadman_check.py --require engaged\|disengaged`, `HIL_STARTUP_DEADMAN`(오타는 fail-closed). gate의 의미는 처음부터 intent가 아니라 **데드맨 채널 생존 증명**이었다. 🟡 **알면서 받아들인 trade-off**(사용자 명시 거절): 이제 policy가 팔을 몰기 전에 ENGAGE 전이가 한 번도 실행되지 않는다 — GUI가 `POLICY_RUNNING` 전까지 ENGAGE를 막기 때문. 탈출구 `HIL_STARTUP_DEADMAN=engaged` → `09` §1.4 |
| 18 | `END EPISODE` 버튼 (`/hil/abort_episode`) | **코드 통합, 실기 미검증** | one-shot 토큰((run_id, episode_id) scope), `ACTIVE_CONTROL_STATES`에서만 합법, `terminal_reason`이 서 있으면 **보이는 거절**. `done=False, truncated=True, masks=1.0, success=False` → 평소 HOME 경로. 🛑 **아무것도 버려지지 않는다** — proto에 cancel RPC가 없고 서버가 Ack 전에 replay에 insert한다 → `09` §5.2, `08` G36. ⏱️ 즉시가 아니다(iteration당 2회 읽기, 최악 한 주기 854 ms) |
| 19 | 충돌 복구 — 하드웨어 재기동에서 세션 생존 | **코드 통합, 실기 미검증** | 카메라·GUI 유지, 3~5단계만 재시도. rc 계약 75/1/70/`>=128`, **75는 `env_step >= 0` 관측이 있어야만** 승격(결정론적 startup 실패는 재시도 안 함). 재시도 3조건: PID 세대 교체 + 토픽 READY + **dashboard `RUNNING`/`NORMAL`**(토픽 probe는 RTDE 읽기라 PROTECTIVE_STOP을 못 본다). 모든 proof 매번 재실행 → `09` §5.3 |
| 20 | Qt 폰트 경고 억제 | **코드 통합** | `launch_cameras.sh` · `run_hil_actor.sh`에서 정확히 두 줄만 stderr 필터. `QT_QPA_FONTDIR`(cv2가 import 시 덮어씀)과 scoped `QT_LOGGING_RULES`(카테고리 없는 qWarning) **둘 다 실측 무효** — 다시 시도하지 말 것. 필터는 `trap '' INT TERM` 아래라 Ctrl-C가 actor 종료 stderr를 못 지운다 |
| 21 | 🆕 **학습 서버 이전 (`kanu` → `junhyeong_ai`)** | 🟢 **실기 PASS (2026-07-31, 조작자 확인)** | 실물 UR7e 세션이 새 서버 learner를 상대로 통과: replay **316** / intervention **210** / last_env_step **68**, run root `~/hil-serl-data/runs/cube_in_cup_real_20260731_054929`. 합성 acceptance가 못 닫던 둘을 닫았다 — **실제 classifier sidecar**와 **`intervened=1` ingress**. 선행 합성 E2E는 200 transition으로 gRPC → finalize → feature replay → CTA → publish → checkpoint → resume 전 구간 통과(수치는 10c행). 🗄️ **kanu에서 학습된 policy 체크포인트는 애초에 하나도 없었다** — run root 8개 전부 `checkpoints/`가 비었고(`checkpoint_period`=5000, 최고 learner step 301) **이전으로 잃은 것이 없다**. 전문 `serl_ur_infra/SERVER_MIGRATION_E2E_JUNHYEONG_AI.md` |
| 21b | jax 핀 유지 (sm_86 → **sm_120 Blackwell**) | **PASS (2026-07-31 실측)** | 이전의 최대 위험이었다. jax **0.5.3이 네이티브로 돈다** — XLA가 `.target sm_120a`를 생성하고 구형 아치 PTX 폴백이 **아니다**. **핀 상향 불필요.** warm-up 28.4 / 16.0 / 0.101 s 🗄️ vs kanu 46.83 / 37.22 / 0.466 s(총 1.9배 빠름; **kanu learner도 A4000을 1장만 썼다** — 8장이 아니다) |
| 21c | 3-CLI 조작자 절차 불변 | **PASS (코드 확인 2026-07-31)** | `run_hil_hardware.sh`에 서버 참조 **0건**. `run_hil_session.sh` / `run_hil_actor.sh`는 `127.0.0.1:50153`(터널 로컬 입구)만 참조하며 반대편 호스트를 모른다. **Terminal 1의 스크립트만 호스트가 바뀌었다** → `09` 상단 박스 |
| 21d | 터널 소유권 실패 모드 | **문서화됨 — 실기에서 발생 (2026-07-31)** | Terminal 1을 Ctrl-C 하거나 닫으면 **learner는 서버에서 살아남고 터널만 죽는다** → Terminal 3 preflight `[6]` `TCP 127.0.0.1:50153 연결 실패 — ConnectionRefusedError`. 조치는 `./run_hil_server.sh` 재실행 하나(살아 있는 learner 재사용 + 새 터널). 🛑 learner를 재기동하면 그 프로세스 RAM에만 있던 online replay가 사라진다 → `09` **§5.4** |

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
> `0.5` → `0.9`로(`ee3240e`), `DEFAULT_REWARD_THRESHOLD`가
> `0.85` → `0.5` → `0.2` → **현재 production `0.5`**로 바뀌었다.

---

## 3. 문서 목록

| 파일 | 내용 |
|---|---|
| [`00_SETUP_AND_SAFETY.md`](00_SETUP_AND_SAFETY.md) | checkout 셋업·빌드·환경변수 함정(인터프리터/`PYTHONPATH`), 오프라인 테스트 2종, 세션 전 체크리스트, **비상 정지 우선순위와 "정지가 아닌 것들"** |
| [`01_GRIPPER.md`](01_GRIPPER.md) | Robotiq 2F-85 단독 검증(**PASS**) + RL/개입 배선과 `:54321` 단일 클라이언트 규칙 |
| [`02_GELLO_LEADER.md`](02_GELLO_LEADER.md) | GELLO 리더 검증(**PASS**) + Dynamixel 진단 스캔 + 트리거가 별도 토픽인 이유 |
| [`03_EEF_MODE.md`](03_EEF_MODE.md) | EEF 텔레옵 단계 상승 P6 → P7 → P8 → P9a → P9b (사용자 검증 완료, 재현 절차) |
| [`04_HIL_INTERVENTION.md`](04_HIL_INTERVENTION.md) | 데드맨 2종, 앵커/gain 래치, mock RViz 루프, **실기 러너 `run_real_hil.py`**, 좌표계 3×3, 그리퍼 개입, 개입 메타데이터 계약, **§9 손맛 실측 5개 표 + 🛑 되돌리면 안 되는 것 3개(2026-07-30)** |
| [`05_COMMS_GRPC.md`](05_COMMS_GRPC.md) | venv 격리, 루프백 스모크, 포트 기본값, schema fail-fast(v2), **분류기 sidecar 전송 계약(§3.2)**, **레이턴시·대역폭 실측(§5.3–5.4)**, 학습 서버 터널 |
| [`06_SENSORS.md`](06_SENSORS.md) | RealSense 2대(시리얼·크롭·역할), QoS/TRANSIENT_LOCAL 함정, 토픽 유량 점검, 19-D state 계약, F/T 프레임 |
| [`07_FAILURE_INJECTION.md`](07_FAILURE_INJECTION.md) | 장애 주입 매트릭스 E1~E14 (유발·기대·확인·PASS·복구) + 결과 기록표 |
| [`08_OPEN_GAPS.md`](08_OPEN_GAPS.md) | 안전·데이터·운영 갭 **G1~G31**과 완화책. G22/G23은 operator 상태기계로 닫혔고 G30은 현재 preposition 기본값, G31은 global max-step 종료 edge를 기록 |
| [`09_HIL_ACTOR_RUNBOOK.md`](09_HIL_ACTOR_RUNBOOK.md) | **HIL actor 기동 런북** — 정상 운용용 3-CLI(`run_hil_server.sh` / `run_hil_hardware.sh` / `run_hil_session.sh`), `run_hil_actor.sh` preflight, actor·sidecar 옵션, Stage A fake-env / Stage B 실센서, 학습 서버(`junhyeong_ai`) 기동. **§5.4 = Terminal 1 터널 소유권 실패 모드** |
| [`10_LATENCY_PROFILING.md`](10_LATENCY_PROFILING.md) | **opt-in per-step 레이턴시 계측**(`HIL_LATENCY_PROFILE=1`) — §8 P1이 요구하는 phase 귀속. 양쪽 호스트가 각자 JSONL을 쓰고 `transition_id`로 오프라인 join, 분석기 `scripts/analyze_hil_latency.py`가 p50/p90/p99/max·loop budget·learner 경합·포화 비율을 낸다. 🪤 **재사용된 learner(`HIL_SERVER_RESULT=reused`)는 서버 쪽을 안 남긴다.** 🛑 **시계 규칙: 호스트 간 타임스탬프는 절대 빼지 않는다** — network+queue는 `step_rpc − server total`로 **유도**한다. 🛑 실기 미검증(**재는 방법**이지 측정 결과가 아니다) |

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
 A2  ur_gello_bringup 단위테스트 489개  -> 00 §4.1  [PASS, 시스템 python3 + overlay]
 A3  serl_ur_infra 단위테스트 (개수 변동) -> 00 §4.2  [PASS, 768p/11s/1xf, actor venv]
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
 D'3 손맛 1차 in_window(30 Hz 재샘플링)   -> 04 §9   [PASS 2026-07-30 오전]
       DRY_RUN --scale 1.0 300스텝 -> --arm --scale 1.0 150스텝, 둘 다 전체 PASS
       (🗄️ 이 경로는 더 이상 shipped 기본값이 아니다)
 D'4 손맛 2차 background(추종 스레드)     -> 04 §9   [🟡 조작자 확인만 — CSV 전수 미실시]
       ** 같은 CSV 불변식(frame-map 잔차/alpha/dp_ratio/held)을 이 경로로 다시 돌릴 것 **
        ↓
[E] 장애 주입 (팔 움직임 포함)
 E1  통신/프로세스 계열 (E1~E5)         -> 07      [미검증]
 E2  로봇 안전 계열 (E6~E11)            -> 07      [미검증]
        ↓
[F] 원격 통신 (학습 서버 — 2026-07-31부터 junhyeong_ai)
 F1  SSH 터널 + 스키마 핸드셰이크       -> 05 §6, 09 §2   [PASS 2026-07-27, 서버=kanu]
 F2  레이턴시 예산 실측                  -> 05 §5.3       [세션마다 재측정 — 호스트가 바뀌었다]
 F3  Stage A actor (fake-env) 왕복       -> 09 §3         [PASS 2026-07-27, 서버=kanu]
 F3b 분류기 sidecar + GUI verdict          -> 09 §4.4       [실기 관측 PASS]
 F4  no-arm live-sensor policy probe      -> 09 §4         [PASS, transition 0]
 F5  서버 이전 합성 acceptance 200 transition -> 09 §7.1  [PASS 2026-07-31, junhyeong_ai]
        ↓
[G] RL 정책 경로 실기 first E2E          [핵심 PASS]
 G1  policy/GELLO 실제 action 전환        [PASS]
 G2  replay201 -> learner102 -> publish v2 [PASS, actor의 v1/v2 수신은 미확인]
 G3  첫 publish 경계 RPC deadline          [1.5/2.0 s bounded 완화 — 장시간 계측 남음]
 G4  HOME/scene/operator episode GUI       [구현, episode-limit WAIT 실기 관측]
 G5  schema-3 MANUAL MARK SUCCESS provenance [다음 실기에서 재확인]
 G6  같은 루프를 junhyeong_ai learner로     [PASS 2026-07-31: replay316/interv210/step68]
```

**현재 위치: 실물 HIL-SERL 원형과 operator episode GUI [G1~G4]까지 도달했고, [G6]으로 그
루프를 새 서버(`junhyeong_ai`)에서 재현했다. 다음 확인은
[G5], 장시간 latency, classifier 재학습, G27/G28 데이터 정합성이다.**
2026-07-30 오전에 [D′3](개입 손맛 1차, `in_window`)이 PASS했다 — **[G3]와 독립**이다
(learner·gRPC 서버를 쓰지 않는 zero-policy 경로). 같은 날 저녁 [D′4](`background` 추종
스레드)가 그 경로를 대체했고 **조작자 실기 코멘트만 있고 CSV 전수 검증은 없다** → 표 9d″.
**다음 실기의 개입 항목은 [D′4]의 CSV 전수 재실행이다.**
> **이전 판(보존):** *"손맛의 나머지 절반은 [G3]/G21이 열려 있는 동안 회수되지 않는다."*
> 그 문장은 예산이 창당 변위를 묶던 `in_window` 전제였다 → 표 9d′.

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
