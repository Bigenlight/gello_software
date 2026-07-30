"""실기(REAL UR7e) HIL 개입 경로 검증 러너 — 계측 CSV 포함.

================================================================================
이 파일은 무엇인가 / 왜 필요한가
================================================================================
`tests/run_rviz_hil.py`는 **mock 전용**이다(파일 자체에 "Never point this at a
real robot"이라고 쓰여 있다). 거기에는 mock 전용 완화값이 하드코딩돼 있다:

    RESET_MAX_DIST_RAD = 7.0     (실기 기본 1.5)     <- 리셋이 온 사방을 쓸고 감
    ACTION_SCALE       = [0.03, 0.10, 1.0]           <- 실기 기본의 2.4배(0.3 m/s)
    GOVERNOR v_max     = 0.36                        <- 실기 기본 0.15의 2.4배

    (2026-07-30 정정: 이 세 줄은 오래 [0.0375, 0.1875, 1.0] / v_max 0.45 /
     "3배"로 적혀 있었다. run_rviz_hil.py의 실제 값은 위와 같고 배율은 2.4배다.
     mock 완화값을 실기 config로 옮기는 사고를 막으려는 경고문인데 그 숫자가
     틀려 있었다 — 값은 run_rviz_hil.py에서 직접 확인할 것.)

즉 **실기에서 HIL 개입 경로(GelloIntervention)를 검증할 러너가 없었다.**
이 파일이 그 자리를 채운다. 설계 원칙은 세 가지다.

  1. 기본이 안전하다 — `DRY_RUN=True`가 기본이라 로봇 명령을 **한 줄도**
     발행하지 않는다. 팔을 실제로 움직이려면 `--arm`을 명시해야 한다.
  2. 속도는 **3층을 함께** 올린다 — `--scale`은 ACTION_SCALE / GOVERNOR /
     UPSAMPLER를 동시에 곱한다 (아래 "3층 스케일링" 참고).
  3. 모든 스텝을 CSV로 남긴다 — 나중에 앵커 래치 / gain 래치 / 좌표계 매핑 /
     "저장 액션 == 실행 액션" 불변식을 **사후에 수치로** 검증할 수 있게.

정책은 **항상 zero**다. 학습 정책은 붙이지 않는다. 이 러너는 "사람이 개입할 때
무슨 일이 벌어지는가"만 격리해서 본다.

================================================================================
⚠️  안전 경고 (실기)
================================================================================
* `--arm` 없이는 로봇이 절대 움직이지 않는다. 처음에는 **반드시 `--arm` 없이**
  돌려서 CSV/좌표계/앵커가 말이 되는지 먼저 확인한다. DRY_RUN에서도 아래
  (a)(b)(c)(d)(e) 검증은 **전부** 가능하다(뒤의 "DRY_RUN으로 검증되는 것" 참고).
* `--arm`을 붙이면 UR7e가 **물리적으로 움직인다.** 펜던트 E-STOP을 손에 닿는
  곳에 두고, 작업 공간을 비우고, 사람이 팔의 궤적 안에 들어가지 않게 한다.
* `/forward_position_controller/commands`에 **다른 퍼블리셔가 있으면 안 된다.**
  gello_ur_bridge(teleop)가 살아 있는 채로 이 러너를 arm하면 두 퍼블리셔가
  같은 컨트롤러를 서로 다른 목표로 때려서 팔이 떨거나 튄다. 러너가 기동 시
  `count_publishers()`로 세어 보고, 우리 것 외에 하나라도 더 있으면 **arm을
  거부한다.**
* 리셋은 **무동작**으로 설계했다: 기동 시점의 실제 관절값을 그대로
  `RESET_JOINTS`로 잡고 `RESET_MAX_DIST_RAD=0.05`로 조인다. 즉 "리셋 = 지금
  자세 유지"다. 사람이 개입으로 팔을 많이 옮긴 뒤 다음 에피소드를 리셋하면
  가드에 걸려 **에러로 멈춘다 — 그게 의도된 안전 실패다**(멀리서 쓸고 오지
  않는다). 그 경우 `--reset-mode hold`를 쓰거나 `--episodes 1`로 돌린다.
* 러너가 죽거나 Ctrl-C로 끝나면 업샘플러가 발행을 멈추고
  forward_position_controller는 **마지막 명령을 유지**한다(팔은 그 자리에서
  홀드). 튀지 않는다.
* 그리퍼는 기본 **비활성**이다(`ACTION_SCALE[2]=0.0`). 리더 트리거를 당겨도
  2F-85가 닫히지 않는다. 팔 경로만 격리 검증하기 위해서다. 그리퍼까지 보려면
  `--gripper`. (이 모드에서 CSV의 `ia6`는 여전히 wrapper가 만든 값을 그대로
  기록하지만 실행은 되지 않는다 — 그리퍼 채널에 한해 "저장==실행"이 깨지는
  유일한 지점이며, 의도된 것이고 여기 명시해 둔다.)

================================================================================
터미널 구성 (실기)
================================================================================
공통 선행 (모든 터미널):

    cd ~/gello_software && source /opt/ros/humble/setup.bash \
      && source ros2_ur_ws/install/setup.bash

🛑 인터프리터와 PYTHONPATH — 2026-07-30에 실제로 이걸로 한 번 실패했다
    * 인터프리터는 **시스템 `python3`가 맞다.** 이 러너는 gRPC를 import하지 않으므로
      CLAUDE.md의 "gRPC 코드는 `/home/laptop3/venvs/gello-hil-actor/bin/python`으로만"
      규칙의 대상이 아니다. 여기 필요한 것은 ROS 오버레이(`rclpy`,
      `ur_gello_bringup.ur_kin`)이고, 위 두 `source`가 그것을 준다.
    * **`PYTHONPATH`를 덮어쓰지 마라.** 덮어쓰면 오버레이가 사라져
      `RuntimeError: rclpy not available`로 죽는다. 더할 것이 있으면 반드시
      `PYTHONPATH="...:$PYTHONPATH"`로 **이어붙인다.**
    * 그 실패의 출처: 오프라인 테스트용 명령
      (`env -u PYTHONPATH PYTHONPATH="<serl_launcher ...>" .../gello-hil-actor/bin/python
      -m pytest tests`)을 이 러너에 복사해 오는 것. 그 명령은 **ROS 없이 도는 pytest 전용**
      이다. 두 명령을 섞지 말 것.

--------------------------------------------------------------------------------
T1 — 실기 드라이버 + forward_position_controller (브리지 없이)
--------------------------------------------------------------------------------
🛑 3-CLI 운영 워크플로우(`run_hil_server.sh` / `run_hil_hardware.sh` /
   `run_hil_session.sh`)를 여기에 쓰지 마라. `run_hil_hardware.sh`는 UR7e를
   **STJC**(scaled_joint_trajectory_controller)로 띄운다. 이 러너는 **FPC가 active**여야
   하고, **컨트롤러 전환 로직이 아예 없다.** STJC 리그에 이 러너를 붙이면:

     - DRY: 아래 preflight가 "구독자가 없다"를 **경고로만** 찍고 그대로 진행한다
       (발행 자체를 안 하므로 CSV는 정상적으로 채워진다 — 그래서 눈치채기 어렵다).
     - `--arm`: preflight가 subscribers < 1 을 보고 **거부**한다.
     - 최악: 구독자 수가 0이 아니어도 **FPC가 active라는 보장은 없다.** 그러면 명령이
       아무데도 가지 않고 러너는 조용히 성공한 것처럼 보인다.

   그래서 아래 "T1 확인"의 `ros2 control list_controllers`를 **눈으로** 봐야 한다.
   운영 3-CLI와 이 러너의 차이는
   serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md §9.6에도 적어 두었다.

권장 (A) 표준 드라이버를 직접, teleop 브리지 없이 띄운다:

    ros2 launch ur_robot_driver ur_control.launch.py \
        ur_type:=ur7e robot_ip:=192.168.10.11 \
        initial_joint_controller:=forward_position_controller \
        headless_mode:=true launch_rviz:=false

    - headless_mode:=true 는 펜던트 REMOTE 모드가 필요하다(Method B).
      펜던트에서 External Control을 PLAY하는 Method A를 쓰려면
      headless_mode:=false 로 바꾸고 프로그램을 PLAY한다.
    - forward_position_controller는 ForwardCommandController라서 **첫 commands
      메시지가 올 때까지 아무것도 쓰지 않는다.** 그래서 활성 상태로 띄워 놔도
      팔은 가만히 있다(ur7e_gello_real.launch.py 주석에서 mock으로 검증된 사실).

대안 (B) 기존 스크립트 재사용 (브리지가 함께 뜨므로 주의):

    START_MODE=switch_only ./run_ur7e_gello_real.sh control_mode:=eef \
        launch_rviz:=false

    - switch_only는 팔을 움직이지 않고 제자리에서 컨트롤러만 바꾼다.
    - 다만 gello_ur_bridge가 같은 commands 토픽의 퍼블리셔로 살아 있을 수 있다.
      이 러너는 그걸 감지해서 `--arm`을 거부한다. 거부당하면 (A)로 가라.

T1 확인:

    ros2 control list_controllers | grep forward_position_controller   # active
    ros2 topic info -v /forward_position_controller/commands
      -> Publisher count: 0   (러너 켜기 전)
         Subscription count: 1 (컨트롤러가 듣고 있다)

--------------------------------------------------------------------------------
T2 — GELLO 리더 발행 (`/gello/joint_states`) — 로봇 브리지 없이 리더 관절만
--------------------------------------------------------------------------------
    GELLO_REPO_ROOT=$HOME/gello_software \
    ros2 run ur_gello_bringup gello_publisher --ros-args --params-file \
      ~/gello_software/ros2_ur_ws/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml

    `GELLO_REPO_ROOT`가 없으면 `No module named 'gello'`로 죽는다.
    이 노드가 /dev/ttyUSB0(GELLO)을 점유한다 — 동시에 두 개 띄우지 말 것.

--------------------------------------------------------------------------------
T3 — HIL GUI (데드맨 `/hil/deadman` 발행)
--------------------------------------------------------------------------------
    cd ~/gello_software/ros2_ur_ws && ./run_hil_gui.sh

    ENGAGE/DISENGAGE 버튼 + 감도 슬라이더(0.10–1.00). 20 Hz 하트비트를 쏘고,
    첫 수신 뒤 0.5 s 끊기면 DeadmanHeartbeatStaleError로 러너가 종료된다.
    단절은 DISENGAGE가 아니며 정책 fallback을 실행하지 않는다.
    **스페이스바(--deadman spacebar)를 기본으로 쓰지 않는 이유**: 워치독이 없어
    프로세스가 멈추거나 키 이벤트를 놓치면 stuck-ON이 될 수 있다. 실기에서는
    하트비트가 있는 topic 데드맨을 쓴다.

--------------------------------------------------------------------------------
T4 — 이 러너
--------------------------------------------------------------------------------
    cd ~/gello_software/serl_ur_infra

    # 1단계: DRY RUN (로봇 안 움직임). 여기서 CSV가 맞는지부터 본다.
    python3 tests/run_real_hil.py --scale 0.5

    # 2단계: 아주 느리게 실제 구동
    python3 tests/run_real_hil.py --arm --scale 0.25

    # 3단계: 기본 속도(= config.py 기본값과 동일)
    python3 tests/run_real_hil.py --arm --scale 1.0

    # 4단계: 기본보다 빠르게 (별도 잠금 해제 필요)
    python3 tests/run_real_hil.py --arm --scale 2.0 --allow-fast

    ⚠️ `--scale`의 기본값 0.5는 **안전을 위한 기본이고 판정에 유리한 값이 아니다.**
       0.5에서는 창 예산이 6.25 cm/s뿐이라 사람 손 속도로 쉽게 포화되고, 포화 표본은
       frame-map 판정에서 제외되므로 SKIP이 되기 쉽다(2026-07-30 실측: 개입 144스텝
       중 59.7 %가 포화되어 판정 표본이 58개로 줄고, 축 여기가 2 cm 게이트에 미달했다).
       frame-map을 **판정**받으려면 `--scale 1.0`으로 올리거나 리더를 더 천천히 움직인다.
       0.5로 볼 것은 (a)(b)(d)(e)와 명령이 말이 되는지다.

    2026-07-30 실기에서 실제로 돌린 순서:
       python3 tests/run_real_hil.py --scale 1.0                       # DRY, 개입 272
       python3 tests/run_real_hil.py --arm --scale 1.0 --max-steps 150  # ARMED, 개입 120

--------------------------------------------------------------------------------
T5 (선택) — 카메라. `--cameras`를 쓸 때만 필요.
--------------------------------------------------------------------------------
    cd ~/gello_software/ros2_ur_ws && ./launch_cameras.sh

    기본은 카메라 없이 돈다(개입 경로만 격리 검증). 카메라가 붙으면 프레임
    staleness가 또 하나의 실패 원인이 되므로 1단계에서는 켜지 않는 게 낫다.

================================================================================
3층 스케일링 — 왜 한 층만 올리면 안 되는가
================================================================================
속도 상한은 서로 물린 세 층으로 걸린다:

    ACTION_SCALE * HZ   ->   GOVERNOR(v_max/w_max/dq_step_max)   ->   UPSAMPLER slew
    (액션 1.0이 주장하는     (task-space 레이트 캡 +               (250 Hz 조인트
     한 스텝 변위)            per-tick 조인트 스텝 게이트)          슬루 캡)

한 층만 올리면 다음 층이 **조용히 잘라먹는다.** 그러면 버퍼에 저장된 액션은
"내가 3 cm 갔다"고 주장하는데 실제로는 1 cm만 간 상태가 되어, SERL이 학습하는
(obs, action, next_obs) 전이가 전부 과장된다. `config.py:58-63`의 INVARIANT과
`ur7e_env.py:94-110`의 경고가 정확히 이 얘기다. (줄번호는 2026-07-30 `4197f5b`
기준 — 그 커밋이 두 파일의 줄을 크게 밀었다.)

그래서 `--scale s`는 **한 곳에서** 세 층을 함께 곱한다:

    ACTION_SCALE = [base_pos*s, base_rot*s, gripper]
    GOVERNOR     = {v_max: base_v*s, w_max: base_w*s, dq_step_max: base_dq*s}
    UPSAMPLER    = {hz: 250, max_step_rad: base_step*s}

base 값은 하드코딩하지 않고 `DefaultUR7eEnvConfig`에서 읽는다 — 기본값이 나중에
바뀌어도 "기본 대비 s배"라는 의미가 유지되고, 층 사이의 비율(헤드룸 20%,
dq_step_max*HZ == max_step_rad*250)이 자동으로 보존된다. 기동 시 그 비율을
다시 검산해서 표로 출력하고, 불변식이 깨지면 **거부**한다.

================================================================================
CSV 스키마 — 이걸로 무엇을 검증하는가
================================================================================
한 스텝당 한 줄. 기본 저장 위치는 `~/gello_hil_logs/real_hil_<타임스탬프>.csv`
(`--csv`로 변경). 같은 이름의 `.meta.json`에 설정값 전체가 함께 남는다.

(a) 앵커 래치 검증
    `g_anchor_*`(리더 앵커 flange xyz), `r_anchor_*`(로봇 명령 앵커 flange xyz).
    ENGAGE 구간 내내 **변하지 않아야 한다.** 변한다면 매 틱 재앵커링되는
    버그(= 개입이 절대 위치 추종으로 변질)다.

(b) gain 래치 검증
    `gain_latched`(engage 순간에 잠긴 값) vs `gain_live`(GUI 슬라이더 실시간
    값). 구간 중 슬라이더를 움직여도 `gain_latched`는 고정이어야 한다.

(c) 좌표계 3x3 매핑 검증
    leader_dp = (leader_tcp_* - g_anchor_*),  robot_dp = (cmd_tcp_* - r_anchor_*).
    robot_dp ≈ M @ (gain_latched * leader_dp) 를 최소자승으로 풀면 M이 나온다.
    **M은 단위행렬이어야 한다.** 부호가 뒤집혀 보인다고 wrappers.py에서 X/Y를
    반전시키는 짓은 절대 금지(RVIZ_HIL_TEST_CLI.md "방향/좌표 주의") — 그건
    카메라 시점 문제이고, 코드를 뒤집으면 SERL 버퍼가 오염된다. 이 CSV가 바로
    "눈"이 아니라 "숫자"로 판정하기 위한 물건이다.

(d) 저장 액션 == 실행 액션 불변식
    `dp_*` = 이 스텝에서 실제로 명령된 TCP 변위(= cmd_tcp_post - cmd_tcp_pre),
    `req_dp_norm` = |ia[:3]| * ACTION_SCALE[0] = 저장된 액션이 주장하는 변위.
    `dp_ratio = dp_norm / req_dp_norm` 이 **1.0**이어야 한다. 0.5 같은 값이
    나오면 governor/업샘플러가 잘라먹고 있다는 뜻이고, 3층 스케일링이 어긋난
    것이다. `held`/`reject_reason`이 그 원인을 알려준다.

(e) 개입 서브스텝 · governor 절삭 관측 — 컬럼 3개 (2026-07-30 신규)
    `substeps` = 이 창에서 **서브스텝이 타깃을 갱신한 횟수.** 30 Hz 서브스텝 /
    100 ms 창의 설계값은 **2**이고, `_apply_action`이 세팅한 첫 타깃을 합치면
    **창당 타깃 3회 갱신**이다. **0이면 서브스텝 경로가 아예 돌지 않았다** —
    정책 스텝이거나, held 창이거나, config의 `INTERVENTION.substep_hz <= HZ`로
    (또는 `INTERVENTION` 블록 부재로) 기능이 꺼진 것이다. 즉 이 컬럼이 "개입이
    빳빳한가"의 1차 관측 수단이다.

    `governed` / `governed_scale` = governor의 task-space rate cap이 요청을
    깎았는지와 그 배율. **창 전체 기준**이다: `governed`는 첫 타깃과 모든
    서브스텝에 대해 OR, `governed_scale`은 그중 **최솟값**(= 창 안에서 가장 센
    절삭)이다. 최솟값인 이유는 0.7로 깎인 서브스텝이 뒤이은 1.0에 가려지면
    안 되기 때문이다.
    ⚠️ 2026-07-30 실기 세 run은 **첫 타깃만 집계하던 코드**로 측정됐다. 그때의
    `governed` 전부 0은 "창 전체에서 절삭 없음"이 아니라 "첫 타깃에서 절삭
    없음"만 증명한다 — 서브스텝 2회의 절삭은 그 run에서 관측되지 않았다.
    개입 창에서 보통 0인 이유: 창 변위 예산(= ACTION_SCALE 1스텝, scale 1.0에서
    0.0125 m)이 governor 캡(v_max/HZ = 0.0150 m)보다 **더 타이트해서 예산이 먼저
    묶기 때문**이다. 참고로 같은 커밋이 처음 드러낸 사실 — ACTION_SCALE 헤드룸이
    축별로만 성립해서 **대각 이동은 상시 절삭된다**(2축 0.849배, 3축 0.693배).
    정책 경로에서는 이 컬럼이 그 절삭을 잡아낸다.

    `reject_reason == BUDGET_EXHAUSTED`는 그 예산이 소진된 창이고 **held가 아니다.**
    예산만큼은 정확히 실행됐으므로 `dp_ratio`는 여전히 1.0이다. 포화는 "리더가
    예산보다 빨랐다"는 뜻이며, 아래 frame-map 판정에서 **제외**된다.

컬럼:
    t_wall, t_mono, episode, step
    intervened, anchored, held, reject_reason
    substeps, governed, governed_scale
    deadman_engaged, deadman_age_s, gain_latched, gain_live
    leader_age_s, lq0..lq5, leader_grip, leader_tcp_x/y/z
    g_anchor_x/y/z, r_anchor_x/y/z
    cmd_tcp_pre_x/y/z, cmd_tcp_x/y/z, dp_x/y/z, dw_x/y/z
    ia0..ia6 (intervene_action), pa0..pa6 (policy_action)
    meas_tcp_x/y/z, q0..q5, joint_age_s
    dp_norm, req_dp_norm, dp_ratio

샘플링 시점 주의: leader/deadman 값은 `env.step()` **직전**에 읽은 표본이고,
anchor/gain/anchored는 `step()` **직후** 값이다. engage 엣지에서 앵커는 step
안에서 잠기므로 직후 값이 "이 스텝에 실제로 쓰인" 앵커다.

================================================================================
무엇이 PASS인가
================================================================================
러너가 종료할 때 위 (a)~(d)를 자동 채점해서 PASS/FAIL을 찍는다:

  PASS anchor-latch : ENGAGE 구간 내 앵커 변동 < 1e-9 m
  PASS gain-latch   : ENGAGE 구간 내 gain_latched 변동 없음
  PASS frame-map    : "매핑은 단위행렬(래그 이득 alpha 배)"이라는 가설의 상대잔차
                      < 0.15. 최소자승 M은 **참고 진단으로만** 찍는다 — 판정은
                      잔차다(3x3을 식별하려 들면 손으로 만든 여기가 약할 때
                      멀쩡한 시스템을 FAIL로 오판한다. summarize() 주석 참고).
  PASS action-exec  : 개입/비-held 스텝의 dp_ratio 중앙값이 0.85~1.15
  PASS held-rate    : held 비율 < 10%

⚠️ frame-map은 **포화 표본(reject_reason == BUDGET_EXHAUSTED)을 판정에서 제외한다.**
   왜: alpha는 래그의 **크기**는 흡수하지만 **방향 발산**은 흡수하지 못한다. 예산이
   소진된 창에서는 명령이 리더의 순간 델타 방향이 아니라 **누적 오차 방향**으로 가고,
   사람이 나갔다 되돌아오면 L(리더 누적)-R(로봇 누적) 관계가 직선이 아니라
   **히스테리시스 루프**가 되어 멀쩡한 시스템이 FAIL로 오판된다.
   2026-07-30 실측(같은 CSV, 표본만 다르게 — 리더가 앵커에서 최대 74.68 cm 나갔고
   그 run의 예산은 6.25 cm/s였다):

       전체 144표본    alpha 0.227   잔차 0.776   -> FAIL
       비포화 58표본   alpha 0.992   잔차 0.058   -> 기준 안쪽

   제외가 게이트를 헐렁하게 만드는 것은 아니다: 표본이 줄어 **SKIP으로 떨어질 수 있다.**
   위 CSV는 지금 코드에서 축 여기가 1.4/3.6/1.4 cm뿐이어서 PASS가 아니라 SKIP이다.
   **PASS는 다시 천천히 움직여서 받는 것이다.**

📌 조작 지침 — 이걸 안 지키면 판정 표본이 사라진다
   `X -> Y -> Z` **한 축씩** 5~10 cm를, **초당 3~5 cm로 천천히**, 축을 섞지 말고 움직인다.
   창 예산은 `ACTION_SCALE[0] * HZ * scale`(scale 1.0에서 12.5 cm/s, 0.5에서 6.25 cm/s)이고
   그보다 빠르면 그 창은 포화된다. 판정 게이트는 비포화 표본 20개 이상 **그리고**
   축당 여기 2 cm 초과다. 한 축씩 5~10 cm면 넉넉하다.

📌 2026-07-30 실기 실측 (실제 UR7e). 상세·리그·CSV 경로는
   serl_ur_infra/HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md §3A:

     DRY   --scale 1.0 (개입 272)  전체 PASS  잔차 0.016  alpha 1.005  표본 141(포화 131 제외)
     ARMED --scale 1.0 (개입 120)  전체 PASS  잔차 0.130  alpha 0.983  표본  51(포화  69 제외)

   세 run 모두 `substeps`가 개입 창 **전부 2**(창당 타깃 3회 갱신), `governed` 전부 0
   (단 그 시점 코드는 첫 타깃만 집계했다 — 위 (e) 경고 참고),
   `dp_ratio` 중앙값 1.000, held 0 %, 스텝 주기 중앙값 101 ms. 조작자 주관 확인
   "손맛 양호" — 이것은 **보고이고 계측이 아니다.**
   ⚠️ 이 검증에는 learner도 gRPC도 없었다(정책 zero). 그래서 창 **사이**(RPC 구간)의
   부드러움은 여기서 **판정되지 않는다** — 08_OPEN_GAPS.md G21 / G24를 볼 것.

추가로 사람 눈으로 확인할 것:
  - DISENGAGE 하면 즉시 정책(zero)으로 돌아가 팔이 멈추는가
  - GUI를 끄면 첫 수신 후 0.5 s 안에 stale 예외로 러너가 종료되고,
    해당 틱에 정책 fallback 명령이 나가지 않는가
  - `--arm`에서 팔이 리더를 따라 "느리지만 매끄럽게" 따라오는가
    (이 감각의 수치 대응물이 `substeps` 컬럼이다. 손맛이 나빠졌는데 원인을 모르면
     먼저 `substeps`가 0으로 떨어진 창이 있는지 보라.)

================================================================================
이 러너로 검증할 수 없는 것
================================================================================
  - 학습 정책의 행동(정책은 zero 고정). actor/learner 루프, 리워드, 버퍼 전송.
  - 카메라 관측 파이프라인의 실사용 성능(--cameras는 staleness 확인 수준).
  - 그리퍼 물리 동작(기본 비활성; --gripper로 켜야 하고 2F-85 Modbus 스택 필요).
  - workspace safety box(ABS_POSE_LIMIT_*) — 현재 env는 이걸 강제하지 않는다
    (ur7e_env.py `_apply_action`의 TODO). 이 러너도 강제하지 않는다.
  - UR 폴트/보호정지 복구, E-STOP 후 재개 시나리오.
  - DRY_RUN에서는: 실제 팔이 명령을 따라오는지(업샘플러->FPC->하드웨어 추종),
    실제 관성/지연, 그리퍼 구동. DRY_RUN은 "명령이 옳게 계산되는가"까지만
    본다 — 다만 (a)(b)(c)(d)(e)는 전부 DRY_RUN에서 검증된다(컨트롤러의 T_cmd가
    가상으로 진행하므로 명령 궤적이 그대로 나온다).
  - (e)의 서브스텝 확인도 DRY_RUN에서 그대로 된다: DRY_RUN이 막는 것은 **발행뿐**
    이고(ros_backend.py:702, 710), 30 Hz 페이싱 루프·리더 재읽기·One-Euro·창 예산·
    governor는 ARMED와 동일하게 돈다. 그래서 `substeps`가 개입 창에서 2로 나오는지,
    `dp_ratio`가 1.0인지, 포화가 몇 %인지는 **팔을 움직이지 않고** 먼저 확인할 수 있다.
    반대로 DRY_RUN이 판정하지 **못하는** 것은 그 매끄러운 명령열을 실제 하드웨어가
    따라오는지, 그리고 조작자의 손맛이다.
"""

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

# 경로 독립: 이 파일 위치에서 serl_ur_infra 루트를 계산해 sys.path에 넣는다
# (run_rviz_hil.py의 `sys.path.insert(0, ".")`는 cwd에 의존한다).
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# config는 numpy만 의존하므로 --help 경로에서도 안전하게 import 가능하다.
# ROS/gymnasium/ur_gello_bringup을 끌어오는 것들은 전부 main() 안에서 늦게 부른다.
from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402

# --arm + --scale 상한. 이보다 빠르게 가려면 --allow-fast 를 명시해야 한다.
ARM_SCALE_SOFT_MAX = 1.0
SCALE_HARD_MAX = 3.0

COLUMNS = (
    ["t_wall", "t_mono", "episode", "step",
     "intervened", "anchored", "held", "reject_reason",
     # 30 Hz 개입 서브스텝 계측. substeps=0 이면 서브스텝 경로가 아예 돌지
     # 않았다는 뜻이다(정책 스텝이거나 held). governed/governed_scale은
     # governor rate cap이 요청을 깎았는지 — 대각 이동은 ACTION_SCALE 헤드룸이
     # 축별로만 성립하기 때문에 상시 걸린다(policy_delta_controller.py 참고).
     "substeps", "governed", "governed_scale",
     # 백그라운드 추종(follow_mode="background") 계측.
     #   follow_ticks — 이 transition 창 동안 추종 스레드가 실제로 발행한 타깃
     #     수. env.step 창 밖(gRPC 왕복)까지 포함한 obs_k -> obs_{k+1} 전 구간이다.
     #     0이면 추종 경로가 안 돈 것(정책 스텝이거나 in_window 모드).
     #   saturation — 창 순변위 ÷ ACTION_SCALE, **클립 전** 비율. 1.0이면 딱 한
     #     스텝, 4.94면 저장된 액션이 실제 이동을 4.94배 과소보고한다는 뜻이다.
     #     추종 스레드에서 InterventionBudget을 뺀 대가이고(조작자 결정
     #     2026-07-30), 이 컬럼이 그 대가를 재는 유일한 관측치다.
     #   saturated — saturation > 1.0.
     # 이 숫자로 다음 결정을 한다: (a) 포화 transition 서버측 제외(proto 필요)
     # 인지 (b) 창 주기 단축(08_OPEN_GAPS.md G21)인지.
     "follow_ticks", "saturated", "saturation",
     "deadman_engaged", "deadman_age_s", "gain_latched", "gain_live",
     "leader_age_s"]
    + [f"lq{i}" for i in range(6)]
    + ["leader_grip", "leader_tcp_x", "leader_tcp_y", "leader_tcp_z",
       "g_anchor_x", "g_anchor_y", "g_anchor_z",
       "r_anchor_x", "r_anchor_y", "r_anchor_z",
       "cmd_tcp_pre_x", "cmd_tcp_pre_y", "cmd_tcp_pre_z",
       "cmd_tcp_x", "cmd_tcp_y", "cmd_tcp_z",
       "dp_x", "dp_y", "dp_z", "dw_x", "dw_y", "dw_z"]
    + [f"ia{i}" for i in range(7)]
    + [f"pa{i}" for i in range(7)]
    + ["meas_tcp_x", "meas_tcp_y", "meas_tcp_z"]
    + [f"q{i}" for i in range(6)]
    + ["joint_age_s", "dp_norm", "req_dp_norm", "dp_ratio"]
)


# ---------------------------------------------------------------------------- #
# 설정 만들기 — 3층 스케일링이 사는 곳                                          #
# ---------------------------------------------------------------------------- #
def build_config(args):
    """DefaultUR7eEnvConfig를 base로, 3층(ACTION_SCALE/GOVERNOR/UPSAMPLER)을
    같은 계수로 스케일한 실기용 config 인스턴스를 만든다.

    base를 하드코딩하지 않고 기본 config에서 읽는 이유: 층 사이의 비율
    (헤드룸 20%, dq_step_max*HZ == max_step_rad*upsampler_hz)이 기본값에 이미
    들어 있으므로, 전부 같은 s를 곱하면 그 비율이 **자동으로 보존**된다.
    """
    base = DefaultUR7eEnvConfig
    s = float(args.scale)

    base_a = np.asarray(base.ACTION_SCALE, dtype=float)
    base_gov = dict(base.GOVERNOR)
    base_ups = dict(base.UPSAMPLER)

    action_scale = np.array(
        [base_a[0] * s, base_a[1] * s, 0.0 if not args.gripper else base_a[2]],
        dtype=float,
    )
    governor = {
        "v_max": base_gov["v_max"] * s,
        "w_max": base_gov["w_max"] * s,
        "dq_step_max": base_gov["dq_step_max"] * s,
    }
    upsampler = {
        "hz": float(base_ups["hz"]),
        "max_step_rad": float(base_ups["max_step_rad"]) * s,
    }

    class RealHilConfig(DefaultUR7eEnvConfig):
        # 로봇에 명령을 발행할지. --arm 없으면 True(발행 안 함)가 기본.
        DRY_RUN = not args.arm
        # fk: 관측 tcp_pose와 컨트롤러/개입 앵커가 **같은 프레임**(flange)에
        # 놓이므로 CSV의 meas_tcp_*와 cmd_tcp_*를 그대로 비교할 수 있다.
        # driver 경로는 tcp_pose_broadcaster + 펜던트 TCP 오프셋에 의존한다.
        TCP_POSE_SOURCE = args.tcp_source
        TCP_OFFSET_XYZ_RPY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        DISPLAY_IMAGE = False
        CAMERAS = dict(DefaultUR7eEnvConfig.CAMERAS) if args.cameras else {}
        IMAGE_STALE_S = 1.0
        MAX_EPISODE_LENGTH = args.max_steps
        # 리셋 = 무동작. RESET_JOINTS는 기동 직후 측정값으로 덮어쓴다(main 참고).
        RESET_JOINTS = np.zeros(6)
        RESET_MAX_DIST_RAD = args.reset_max_dist
        RESET_TOLERANCE_RAD = 0.02
        RESET_TIMEOUT_S = 10.0

    cfg = RealHilConfig()
    cfg.ACTION_SCALE = action_scale
    cfg.GOVERNOR = governor
    cfg.UPSAMPLER = upsampler
    return cfg, base_a, base_gov, base_ups


def check_layers(cfg, verbose=True):
    """3층이 서로 어긋나지 않았는지 검산. 어긋나면 (False, 사유들)."""
    a = np.asarray(cfg.ACTION_SCALE, dtype=float)
    gov = cfg.GOVERNOR
    ups = cfg.UPSAMPLER
    hz = float(cfg.HZ)
    problems = []

    v_demand = a[0] * hz
    w_demand = a[1] * hz
    if v_demand > gov["v_max"] + 1e-12:
        problems.append(
            f"ACTION_SCALE pos*HZ={v_demand:.4f} m/s > governor v_max={gov['v_max']:.4f}"
        )
    if w_demand > gov["w_max"] + 1e-12:
        problems.append(
            f"ACTION_SCALE rot*HZ={w_demand:.4f} rad/s > governor w_max={gov['w_max']:.4f}"
        )
    # governor의 per-tick 조인트 게이트와 업샘플러 슬루는 같은 rad/s여야 한다.
    gov_rate = gov["dq_step_max"] * hz
    ups_rate = ups["max_step_rad"] * ups["hz"]
    if not np.isclose(gov_rate, ups_rate, rtol=0.05):
        problems.append(
            f"governor dq_step_max*HZ={gov_rate:.4f} rad/s != "
            f"upsampler max_step_rad*hz={ups_rate:.4f} rad/s "
            "(한 층만 스케일된 상태 — 느린 쪽이 조용히 잘라먹는다)"
        )

    if verbose:
        print("\n---- 3층 속도 상한 (모두 --scale로 함께 스케일됨) ----")
        print(f"  L1 ACTION_SCALE : pos={a[0]:.4f} m/step  rot={a[1]:.4f} rad/step "
              f"grip={a[2]:.2f}   -> {v_demand:.3f} m/s, {w_demand:.3f} rad/s @ {hz:g} Hz")
        print(f"  L2 GOVERNOR     : v_max={gov['v_max']:.4f} m/s  "
              f"w_max={gov['w_max']:.4f} rad/s  dq_step_max={gov['dq_step_max']:.4f} rad "
              f"(={gov_rate:.3f} rad/s)")
        print(f"  L3 UPSAMPLER    : {ups['hz']:g} Hz  max_step_rad={ups['max_step_rad']:.5f} "
              f"(={ups_rate:.3f} rad/s)")
        print(f"  헤드룸          : v {gov['v_max'] / max(v_demand, 1e-12):.2f}x, "
              f"w {gov['w_max'] / max(w_demand, 1e-12):.2f}x  (기본 설계값 1.20x)")
        print("------------------------------------------------------\n")
    return (len(problems) == 0), problems


# ---------------------------------------------------------------------------- #
# 작은 유틸                                                                     #
# ---------------------------------------------------------------------------- #
def _f(x):
    """CSV 셀용: None/NaN은 빈 칸, +-inf는 그대로, 나머지는 유효숫자 9자리.

    9자리인 이유: t_mono(monotonic 초, 6자리 정수부)까지 마이크로초 해상도를
    유지해야 스텝 간격/지연을 사후에 볼 수 있다.
    """
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(v):
        return ""
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.9g}"


def _deadman_age(dm):
    """RosTopicDeadman의 마지막 수신 이후 경과(초). 없으면 NaN, 미수신은 inf."""
    if not hasattr(dm, "_last_rx"):
        return float("nan")
    last = getattr(dm, "_last_rx")
    if last is None:
        return float("inf")
    return time.monotonic() - float(last)


def _xyz(T):
    return (None, None, None) if T is None else (T[0, 3], T[1, 3], T[2, 3])


# ---------------------------------------------------------------------------- #
# preflight                                                                     #
# ---------------------------------------------------------------------------- #
def wait_for(fn, timeout_s, period=0.1):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(period)
    return False


def preflight_command_topic(node, topic, arming):
    """commands 토픽에 우리 말고 다른 퍼블리셔가 있으면 arm을 거부한다.

    teleop 브리지(gello_ur_bridge)가 살아 있는 채로 arm하면 두 퍼블리셔가 같은
    ForwardCommandController를 서로 다른 목표로 때린다 = 팔 떨림/점프.
    """
    n_pub = node.count_publishers(topic)
    n_sub = node.count_subscribers(topic)
    print(f"  {topic}: publishers={n_pub} (우리 것 1 포함), subscribers={n_sub}")
    ok = True
    if n_pub > 1:
        print("  !! 다른 퍼블리셔가 이 토픽에 붙어 있다 (teleop 브리지?). "
              "이 상태로 arm하면 명령이 충돌한다.")
        ok = False
    if n_sub < 1:
        print("  !! 구독자가 없다 — forward_position_controller가 active가 아닐 수 "
              "있다. `ros2 control list_controllers`로 확인하라.")
        if arming:
            ok = False
    return ok


def preflight_command_topics(node, ros_config, *, arming, gripper_enabled):
    """Check every actuator topic this process can publish before arming."""

    topics = [ros_config["command_topic"]]
    if gripper_enabled:
        topics.append(ros_config["gripper_command_topic"])
    results = [
        preflight_command_topic(node, topic, arming) for topic in topics
    ]
    return all(results)


# ---------------------------------------------------------------------------- #
# 종료 시 자동 채점                                                             #
# ---------------------------------------------------------------------------- #
def _segments(rows):
    """연속된 개입(intervened==1) 구간들을 인덱스 리스트로 쪼갠다."""
    segs, cur = [], []
    for i, r in enumerate(rows):
        if r["intervened"] == 1:
            cur.append(i)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def summarize(rows):
    """(a)(b)(c)(d) 자동 채점. 콘솔에 PASS/FAIL을 찍고 결과 dict를 돌려준다."""
    out = {}
    n = len(rows)
    n_iv = sum(r["intervened"] for r in rows)
    n_held = sum(1 for r in rows if r["held"])
    print("\n================= 결과 요약 =================")
    print(f"  총 스텝 {n},  개입 스텝 {n_iv},  held {n_held}")
    # 포화 계측. 추종 스레드에 예산이 없으므로 창이 길면 저장 액션이 실제
    # 이동을 과소보고한다 — 그 크기를 여기서 처음 숫자로 본다.
    # 이 요약이 (a) 서버측 포화 제외 / (b) 창 주기 단축 중 무엇을 할지 정한다.
    follow_rows = [r for r in rows if int(r.get("follow_ticks", 0) or 0) > 0]
    if follow_rows:
        ratios = [
            float(r["saturation"]) for r in follow_rows
            if str(r.get("saturation", "")) != ""
        ]
        # int(), not truthiness: these rows are also read back from the CSV in
        # post-hoc analysis, where "0" is a non-empty (truthy) string.
        n_sat = sum(1 for r in follow_rows if int(r.get("saturated", 0) or 0))
        pct = 100.0 * n_sat / len(follow_rows)
        worst = max(ratios) if ratios else float("nan")
        median = sorted(ratios)[len(ratios) // 2] if ratios else float("nan")
        print(
            f"  포화 창 {n_sat}/{len(follow_rows)} ({pct:.1f}%) — "
            f"saturation 중앙값 {median:.2f}, 최대 {worst:.2f} "
            "(1.0 초과 = 저장 액션이 그 배수만큼 이동을 과소보고)"
        )
        out["saturated_windows"] = n_sat
        out["saturation_max"] = worst
    if n == 0:
        print("  FAIL — 기록된 스텝이 없다")
        return {"pass": False}

    segs = _segments(rows)
    print(f"  ENGAGE 구간 수: {len(segs)}  (길이: "
          f"{[len(s) for s in segs] if len(segs) <= 12 else '...'} )")

    # ---- (a) 앵커 래치 ---- #
    worst_anchor = 0.0
    for seg in segs:
        for key in ("g_anchor", "r_anchor"):
            vals = np.array(
                [[rows[i][f"{key}_{c}"] for c in "xyz"] for i in seg], dtype=float
            )
            if len(vals) > 1 and np.all(np.isfinite(vals)):
                worst_anchor = max(worst_anchor, float(np.max(np.ptp(vals, axis=0))))
    if segs:
        ok_a = worst_anchor < 1e-9
        out["anchor_latch"] = ok_a
        print(f"  {'PASS' if ok_a else 'FAIL'} anchor-latch : 구간 내 앵커 최대 변동 "
              f"{worst_anchor:.3e} m (기준 <1e-9)")
    else:
        out["anchor_latch"] = None
        print("  SKIP anchor-latch : 개입 구간 없음")

    # ---- (b) gain 래치 ---- #
    worst_gain = 0.0
    for seg in segs:
        g = np.array([rows[i]["gain_latched"] for i in seg], dtype=float)
        if len(g) > 1 and np.all(np.isfinite(g)):
            worst_gain = max(worst_gain, float(np.ptp(g)))
    if segs:
        ok_b = worst_gain < 1e-12
        out["gain_latch"] = ok_b
        print(f"  {'PASS' if ok_b else 'FAIL'} gain-latch    : 구간 내 "
              f"gain_latched 최대 변동 {worst_gain:.3e}")
    else:
        out["gain_latch"] = None
        print("  SKIP gain-latch    : 개입 구간 없음")

    # ---- (c) 좌표계 3x3 매핑 ---- #
    # 포화(rate-limited) 표본은 이 검정에서 제외한다. 아래 alpha는 래그의
    # **크기**를 흡수하지만 **방향 발산**은 흡수하지 못한다: 예산이 소진된
    # 창에서는 명령이 리더의 순간 델타 방향이 아니라 누적 오차 방향으로 가고,
    # 사람이 나갔다 되돌아오면 L(리더 누적)-R(로봇 누적) 관계가 직선이 아니라
    # 히스테리시스 루프가 된다. 그러면 멀쩡한 시스템이 FAIL로 오판된다 —
    # 2026-07-30 실측: 전체 144표본 alpha=0.227/잔차 0.776 FAIL 인데 비포화
    # 58표본만 보면 alpha=0.992/잔차 0.058 PASS 였다(리더가 앵커에서 74.7 cm
    # 나갔고 예산은 6.25 cm/s, 표본 59.7%가 포화).
    #
    # 07-28 기록의 "잔차 0.093 PASS"도 포화 표본을 제외한 값이다. 당시엔 포화가
    # reject_reason에 남지 않아 러너가 걸러낼 수 없었다 — 30 Hz 서브스텝이
    # 도입되며 BUDGET_EXHAUSTED가 기록되기 시작해 처음으로 가능해졌다.
    SATURATED_REASONS = {"BUDGET_EXHAUSTED"}
    L, R = [], []
    n_sat_skipped = 0
    for seg in segs:
        for i in seg:
            r = rows[i]
            if (r["reject_reason"] or "") in SATURATED_REASONS:
                n_sat_skipped += 1
                continue
            try:
                gain = float(r["gain_latched"])
                ld = np.array([float(r[f"leader_tcp_{c}"]) - float(r[f"g_anchor_{c}"])
                               for c in "xyz"])
                rd = np.array([float(r[f"cmd_tcp_{c}"]) - float(r[f"r_anchor_{c}"])
                               for c in "xyz"])
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(ld)) and np.all(np.isfinite(rd)):
                L.append(gain * ld)
                R.append(rd)
    # 판정 방식: 3x3을 "식별"하려 들면 리더가 세 축을 서로 독립적으로 크게
    # 흔들어야만(설계행렬 full-rank) 답이 유일해진다 — 손으로 GELLO를 잡고
    # 하기엔 까다롭고, 여기가 약할 때 lstsq는 min-norm 해를 뱉어 **멀쩡한
    # 시스템을 FAIL로 오판**할 수 있다. 오판은 최악이다("그럼 코드에서 X를
    # 뒤집자"는 반사행동을 유발한다).
    #
    # 그래서 식별이 아니라 **가설 검정**을 한다: "매핑은 단위행렬(래그 이득
    # alpha 배)이다"라는 가설의 잔차를 본다. robot_dp ≈ alpha * leader_dp_gained
    # 가 맞으면 여기가 약해도 잔차는 작고, X/Y가 뒤집혔거나 축이 섞였으면
    # 잔차가 즉시 커진다. M(최소자승 3x3)은 참고용 진단으로 함께 찍는다.
    #
    # alpha를 분리하는 이유: 개입은 rate-limited chase라 리더가 ACTION_SCALE*HZ
    # 보다 빨리 움직이면 명령이 뒤처져 M ≈ alpha*I (alpha<1)이 된다. 그건 매핑
    # 오류가 아니라 래그다.
    excitation = np.ptp(np.array(L), axis=0) if L else np.zeros(3)
    if len(L) >= 20 and np.all(excitation > 0.02):
        Lm, Rm = np.array(L), np.array(R)
        alpha = float(np.sum(Lm * Rm) / max(np.sum(Lm * Lm), 1e-18))
        resid = float(np.linalg.norm(Rm - alpha * Lm) / max(np.linalg.norm(Rm), 1e-12))
        M, *_ = np.linalg.lstsq(Lm, Rm, rcond=None)   # robot_dp ≈ leader_dp @ M
        M = M.T                                        # 열벡터 규약으로 되돌림
        ok_c = resid < 0.15 and alpha > 0.0
        out["frame_map"] = ok_c
        print(f"  {'PASS' if ok_c else 'FAIL'} frame-map     : 단위행렬 가설 상대잔차 "
              f"{resid:.3f} (기준 <0.15), 추종이득 alpha={alpha:.3f} "
              f"(1.0=래그 없음), 표본 {len(L)}"
              + (f" (포화 {n_sat_skipped}개 제외)" if n_sat_skipped else ""))
        print("      최소자승 M (참고) =\n" + "\n".join(
            "        [" + "  ".join(f"{v:+.3f}" for v in row) + "]" for row in M))
        if not ok_c:
            print("      !! 부호가 뒤집혔다고 wrappers.py에서 축을 반전시키지 말 것 —"
                  " SERL 버퍼가 오염된다. 원인을 먼저 찾아라.")
    else:
        out["frame_map"] = None
        print(f"  SKIP frame-map     : 여기 부족 (표본 {len(L)}/20 필요, 리더 변위 "
              f"x/y/z = {excitation[0] * 100:.1f}/{excitation[1] * 100:.1f}/"
              f"{excitation[2] * 100:.1f} cm, 각 축 2 cm 초과 필요)")
        print("      -> ENGAGE 상태에서 GELLO를 X, Y, Z 각각 5 cm 이상 움직여라.")
        # 게이트 미달이어도 진단값은 찍는다. SKIP은 "판정 불가"이지 "정보 없음"이
        # 아니다 — 잔차/alpha가 이미 좋다면 다음 run에서 무엇을 늘려야 하는지가
        # 축 여기(excitation)뿐임을 바로 알 수 있다. 이 값으로 PASS를 주지는
        # 않는다(표본이 편향됐을 수 있다).
        if len(L) >= 4:
            Lm, Rm = np.array(L), np.array(R)
            denom = float(np.sum(Lm * Lm))
            if denom > 1e-18:
                a_d = float(np.sum(Lm * Rm) / denom)
                r_d = float(
                    np.linalg.norm(Rm - a_d * Lm) / max(np.linalg.norm(Rm), 1e-12)
                )
                print(f"      (참고, 판정 아님) alpha={a_d:.3f} 잔차={r_d:.3f}")
        if n_sat_skipped:
            print(f"      -> 포화(BUDGET_EXHAUSTED) {n_sat_skipped}개를 제외했다. "
                  f"리더를 ACTION_SCALE*HZ 안쪽(현재 상한 참고)에서 **천천히** "
                  f"움직이면 비포화 표본이 늘어난다.")

    # ---- (d) 저장 액션 == 실행 액션 ---- #
    ratios = [
        float(r["dp_ratio"]) for r in rows
        if r["intervened"] == 1 and not r["held"] and r["dp_ratio"] not in (None, "")
        and np.isfinite(float(r["dp_ratio"]))
    ]
    if ratios:
        med = float(np.median(ratios))
        ok_d = 0.85 <= med <= 1.15
        out["action_exec"] = ok_d
        print(f"  {'PASS' if ok_d else 'FAIL'} action-exec   : dp_ratio 중앙값 "
              f"{med:.3f} (기준 0.85~1.15), 표본 {len(ratios)}")
        if not ok_d:
            print("      -> 3층 중 하나가 잘라먹고 있다. 위의 '3층 속도 상한' 표와 "
                  "reject_reason 컬럼을 보라.")
    else:
        out["action_exec"] = None
        print("  SKIP action-exec   : 개입 중 유효 표본 없음")

    # ---- held 비율 ---- #
    held_rate = n_held / n
    ok_h = held_rate < 0.10
    out["held_rate"] = ok_h
    print(f"  {'PASS' if ok_h else 'FAIL'} held-rate     : {held_rate * 100:.1f}% "
          "(기준 <10%)")

    required_checks = (
        "anchor_latch",
        "gain_latch",
        "frame_map",
        "action_exec",
        "held_rate",
    )
    verdict = all(out.get(name) is True for name in required_checks)
    out["pass"] = verdict
    print(f"\n  전체: {'PASS' if verdict else 'FAIL / 미검증 항목 있음'}")
    print("=============================================\n")
    return out


# ---------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="run_real_hil.py",
        description="실기 UR7e HIL 개입 경로 검증 러너 (기본 DRY_RUN, 정책 zero 고정)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예)  python3 tests/run_real_hil.py                # DRY RUN(안 움직임)\n"
               "     python3 tests/run_real_hil.py --arm --scale 0.25\n"
               "     python3 tests/run_real_hil.py --arm --scale 2.0 --allow-fast\n",
    )
    ap.add_argument(
        "--arm", action="store_true",
        help="로봇에 실제로 명령을 발행한다(DRY_RUN 해제). 기본은 발행 안 함. "
             "⚠️ 팔이 물리적으로 움직인다.",
    )
    ap.add_argument(
        "--scale", type=float, default=0.5,
        help="속도 배율. ACTION_SCALE/GOVERNOR/UPSAMPLER 3층을 함께 곱한다. "
             "1.0 = config.py 기본값. (기본: 0.5)",
    )
    ap.add_argument(
        "--allow-fast", action="store_true",
        help=f"--arm 상태에서 --scale > {ARM_SCALE_SOFT_MAX} 를 허용한다(별도 잠금 해제).",
    )
    ap.add_argument(
        "--deadman", choices=["topic", "spacebar"], default="topic",
        help="데드맨 소스. 기본 topic(/hil/deadman, GUI, 20 Hz 하트비트+워치독). "
             "spacebar는 워치독이 없어 stuck-ON 위험이 있으므로 실기 비권장.",
    )
    ap.add_argument("--episodes", type=int, default=1, help="에피소드 수 (기본 1)")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="에피소드당 스텝 수 (10 Hz 기준 300 = 30초, 기본 300)")
    ap.add_argument(
        "--reset-mode", choices=["startup", "hold"], default="startup",
        help="startup(기본): 기동 시 측정한 관절을 RESET_JOINTS로 고정 — 리셋이 "
             "무동작이고, 사람이 팔을 멀리 옮겼으면 다음 리셋이 안전하게 거부된다. "
             "hold: 매 에피소드마다 현재 관절을 다시 잡아 리셋이 항상 무동작.",
    )
    ap.add_argument("--reset-max-dist", type=float, default=0.05,
                    help="RESET_MAX_DIST_RAD (기본 0.05 — 사실상 무동작만 허용)")
    ap.add_argument("--tcp-source", choices=["fk", "driver"], default="fk",
                    help="관측 tcp_pose 출처. 기본 fk (컨트롤러/앵커와 같은 프레임).")
    ap.add_argument("--cameras", action="store_true",
                    help="카메라 관측을 켠다(launch_cameras.sh 필요). 기본은 끔.")
    ap.add_argument("--gripper", action="store_true",
                    help="그리퍼 채널을 활성화한다. 기본은 ACTION_SCALE[2]=0으로 비활성.")
    ap.add_argument("--csv", default=None,
                    help="CSV 경로 (기본 ~/gello_hil_logs/real_hil_<타임스탬프>.csv)")
    ap.add_argument("--leader-timeout", type=float, default=15.0,
                    help="리더/데드맨/카메라 대기 타임아웃 (초, 기본 15)")
    ap.add_argument("--yes", action="store_true",
                    help="--arm 확인 프롬프트를 건너뛴다(스크립트용).")
    args = ap.parse_args(argv)

    if not (0.0 < args.scale <= SCALE_HARD_MAX):
        ap.error(f"--scale 은 (0, {SCALE_HARD_MAX}] 범위여야 한다 (받은 값 {args.scale})")
    if args.arm and args.scale > ARM_SCALE_SOFT_MAX and not args.allow_fast:
        ap.error(
            f"--arm 과 --scale {args.scale} (> {ARM_SCALE_SOFT_MAX}) 조합은 "
            "--allow-fast 없이는 거부한다. 느린 단계부터 올려라: 0.25 -> 0.5 -> 1.0."
        )
    if args.max_steps < 1:
        ap.error("--max-steps 는 1 이상이어야 한다")
    if args.episodes < 1:
        ap.error("--episodes 는 1 이상이어야 한다")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg, base_a, base_gov, base_ups = build_config(args)

    print("\n########################################################")
    print("#  실기 HIL 개입 검증 러너")
    print(f"#  모드      : {'ARMED — 팔이 실제로 움직인다 ⚠️' if args.arm else 'DRY RUN — 명령 미발행(안전)'}")
    print(f"#  scale     : {args.scale}  (1.0 = config.py 기본값)")
    print(f"#  deadman   : {args.deadman}")
    print("#  정책      : zero 고정 (학습 정책 없음)")
    print(f"#  카메라    : {'ON' if args.cameras else 'OFF'}   "
          f"그리퍼: {'ON' if args.gripper else 'OFF(비활성)'}")
    print("########################################################")

    ok_layers, problems = check_layers(cfg)
    if not ok_layers:
        for p in problems:
            print(f"  !! 3층 불일치: {p}")
        sys.exit("3층 스케일링이 어긋났다 — 저장 액션과 실행 액션이 달라진다. 중단.")

    # ---- 늦은 import: --help / 인자 검증은 ROS 없이도 되어야 한다 ---- #
    from ur_env.envs.ur7e_env import UR7eEnv
    from ur_env.envs.wrappers import GelloIntervention, RosTopicDeadman
    from ur_gello_bringup.ur_kin import fk, so3_log

    env = UR7eEnv(fake_env=False, config=cfg)
    base_env = env
    deadman = None
    csv_path = Path(
        args.csv
        or Path.home() / "gello_hil_logs"
        / f"real_hil_{_dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    ).expanduser()
    rows = []
    fh = None

    try:
        # ---- 로봇 피드백 ---- #
        print("\n[preflight] /joint_states 대기...")
        if not wait_for(lambda: base_env.backend.get_joint_state()[0] is not None, 10.0):
            sys.exit("no /joint_states — T1(드라이버)이 떠 있는지 확인하라")
        q0, _, age0 = base_env.backend.get_joint_state()
        print(f"  joint_states ok: q={np.round(q0, 3)} age={age0:.2f}s")

        # ---- 리셋 무동작화: 지금 자세를 RESET_JOINTS로 ---- #
        cfg.RESET_JOINTS = np.asarray(q0, dtype=float).reshape(6).copy()
        print(f"  RESET_JOINTS <- 현재 관절 (리셋은 무동작), "
              f"RESET_MAX_DIST_RAD={cfg.RESET_MAX_DIST_RAD}")

        # ---- commands 토픽 충돌 검사 ---- #
        print("[preflight] 명령 토픽 점검...")
        topic_ok = preflight_command_topics(
            base_env.backend._node,
            cfg.ROS,
            arming=args.arm,
            gripper_enabled=args.gripper,
        )
        if args.arm and not topic_ok:
            sys.exit("명령 토픽 상태가 arm하기에 안전하지 않다 — 중단.")

        # ---- 데드맨 ---- #
        if args.deadman == "topic":
            deadman = RosTopicDeadman(base_env.backend._node)
            print("[preflight] /hil/deadman 대기 (T3: ./run_hil_gui.sh)...")
            if not wait_for(lambda: np.isfinite(_deadman_age(deadman)), args.leader_timeout):
                sys.exit("no /hil/deadman — HIL GUI(T3)를 먼저 띄워라")
            if deadman.is_engaged():
                sys.exit("데드맨이 이미 ENGAGE 상태다 — GUI에서 DISENGAGE 후 다시 시작하라")
            print(f"  deadman ok (DISENGAGED, gain={deadman.gain():.2f})")
        else:
            print("  !! spacebar 데드맨: 하트비트/워치독이 없어 stuck-ON 위험이 있다. "
                  "실기에서는 --deadman topic 권장.")

        env = GelloIntervention(env, deadman=deadman)
        iv = env  # 앵커/gain을 읽을 wrapper 인스턴스
        for attr in ("T_g_anchor", "T_r_anchor", "_anchored", "_gain"):
            if not hasattr(iv, attr):
                print(f"  !! GelloIntervention에 '{attr}'가 없다 — 해당 CSV 컬럼은 빈다")
        if deadman is None:
            deadman = iv.expert.deadman

        # ---- 리더 ---- #
        print("[preflight] /gello/joint_states (리더) 대기 (T2: gello_publisher)...")
        if not wait_for(lambda: base_env.backend.get_gello_state()[0] is not None,
                        args.leader_timeout):
            sys.exit("no /gello/joint_states — T2(gello_publisher)를 띄워라. "
                     "리더가 없으면 개입 자체가 불가능하다.")
        larr, lage = base_env.backend.get_gello_state()
        print(f"  leader ok: q={np.round(np.asarray(larr[:6]), 3)} age={lage:.2f}s")

        # ---- 카메라(선택) ---- #
        if cfg.CAMERAS:
            print(f"[preflight] 카메라 {list(cfg.CAMERAS)} 신선한 프레임 대기...")

            def _cams_ok():
                return all(
                    (lambda j, a: j is not None and a <= cfg.IMAGE_STALE_S)(
                        *base_env.backend.get_image(k)
                    )
                    for k in cfg.CAMERAS
                )

            if not wait_for(_cams_ok, args.leader_timeout):
                sys.exit("카메라 프레임이 안 온다 — launch_cameras.sh 확인 (또는 "
                         "--cameras 빼고 개입 경로만 먼저 보라)")
            print("  cameras ok")

        # ---- arm 확인 ---- #
        if args.arm:
            print("\n" + "!" * 66)
            print("!!  ARMED: 이제부터 /forward_position_controller/commands 로 실제")
            print("!!  명령이 나간다. UR7e가 물리적으로 움직인다.")
            print("!!  - 작업 공간을 비웠는가?  펜던트 E-STOP이 손에 닿는가?")
            print("!!  - teleop 브리지 등 다른 명령 퍼블리셔를 모두 껐는가?")
            print(f"!!  - 최고 속도 약 {cfg.ACTION_SCALE[0] * cfg.HZ:.2f} m/s "
                  f"({cfg.ACTION_SCALE[1] * cfg.HZ:.2f} rad/s)")
            print("!" * 66)
            if not args.yes:
                try:
                    if input("계속하려면 정확히 ARM 을 입력: ").strip() != "ARM":
                        sys.exit("취소됨")
                except EOFError:
                    sys.exit("확인 입력을 받을 수 없다 (--yes 로 건너뛸 수 있음)")
            for k in range(3, 0, -1):
                print(f"  시작 {k}...")
                time.sleep(1.0)

        # ---- CSV ---- #
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(csv_path, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        meta = {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv[1:] if argv is None else list(argv),
            "dry_run": bool(cfg.DRY_RUN),
            "armed": bool(args.arm),
            "scale": args.scale,
            "hz": cfg.HZ,
            "action_scale": [float(v) for v in cfg.ACTION_SCALE],
            "governor": {k: float(v) for k, v in cfg.GOVERNOR.items()},
            "upsampler": {k: float(v) for k, v in cfg.UPSAMPLER.items()},
            "base_action_scale": [float(v) for v in base_a],
            "base_governor": {k: float(v) for k, v in base_gov.items()},
            "base_upsampler": {k: float(v) for k, v in base_ups.items()},
            "deadman": args.deadman,
            "tcp_pose_source": cfg.TCP_POSE_SOURCE,
            "cameras": list(cfg.CAMERAS),
            "gripper_enabled": bool(args.gripper),
            "reset_joints": [float(v) for v in cfg.RESET_JOINTS],
            "reset_max_dist_rad": float(cfg.RESET_MAX_DIST_RAD),
            "policy": "zero",
        }
        with open(str(csv_path) + ".meta.json", "w") as mf:
            json.dump(meta, mf, indent=2)
        print(f"\nCSV: {csv_path}")
        print(f"META: {csv_path}.meta.json")

        engage_help = (
            "  GUI에서 ENGAGE -> GELLO를 천천히 움직인다 -> DISENGAGE 하면 정책(zero)"
            " 복귀.\n  GUI를 닫거나 하트비트가 0.5 s 끊기면 stale 예외로 종료되며"
            " 정책 fallback은 없다."
            if args.deadman == "topic"
            else "  이 터미널에 포커스를 준 채 SPACEBAR를 홀드하고 GELLO를 움직인다."
        )
        print("\n================= 조작 안내 =================")
        print(engage_help)
        print("  ESC = 에피소드 종료, Ctrl-C = 즉시 종료(팔은 마지막 명령에서 홀드)")
        print(f"  에피소드 {args.episodes} x {cfg.MAX_EPISODE_LENGTH} 스텝 @ {cfg.HZ:g} Hz")
        print("=============================================\n")

        zero = np.zeros(7, dtype=np.float32)

        for ep in range(args.episodes):
            if args.reset_mode == "hold":
                qn, _, _ = base_env.backend.get_joint_state()
                if qn is not None:
                    cfg.RESET_JOINTS = np.asarray(qn, dtype=float).reshape(6).copy()
            print(f"=== episode {ep}: reset (무동작 설계) ===")
            try:
                obs, _ = env.reset()
            except RuntimeError as e:
                print(f"  리셋 거부됨: {e}")
                print("  -> 팔이 기동 시 자세에서 많이 벗어났다. 의도된 안전 실패다. "
                      "이전 자세로 되돌리거나 --reset-mode hold 로 다시 시작하라.")
                break

            prev_iv = None
            while True:
                # ---- step 직전 표본 ---- #
                t_mono = time.monotonic()
                T_pre = base_env.controller.tcp_cmd()
                q_lead, grip, l_age = iv.expert.get_leader()
                dm_engaged = int(bool(deadman.is_engaged())) if deadman is not None else ""
                dm_age = _deadman_age(deadman) if deadman is not None else float("nan")
                gain_live = float(deadman.gain()) if deadman is not None else float("nan")
                lead_T = fk(np.asarray(q_lead, dtype=float)) if q_lead is not None else None

                obs, rew, done, trunc, info = env.step(zero)

                # ---- step 직후 (이 스텝에 실제로 쓰인 앵커/gain) ---- #
                T_post = base_env.controller.tcp_cmd()
                dp = T_post[:3, 3] - T_pre[:3, 3]
                dw = so3_log(T_post[:3, :3] @ T_pre[:3, :3].T)
                ia = info.get("intervene_action")
                pa = info.get("policy_action", zero)
                intervened = int(info.get("intervened", 0))
                dp_norm = float(np.linalg.norm(dp))
                req = (
                    float(np.linalg.norm(np.asarray(ia, dtype=float)[:3]))
                    * float(cfg.ACTION_SCALE[0])
                    if ia is not None else float("nan")
                )
                ratio = dp_norm / req if (np.isfinite(req) and req > 1e-9) else float("nan")

                q_now, _, q_age = base_env.backend.get_joint_state()
                gx, gy, gz = _xyz(getattr(iv, "T_g_anchor", None))
                rx, ry, rz = _xyz(getattr(iv, "T_r_anchor", None))
                lx, ly, lz = _xyz(lead_T)

                row = {
                    "t_wall": _dt.datetime.now().isoformat(timespec="milliseconds"),
                    "t_mono": t_mono,
                    "episode": ep,
                    "step": base_env.curr_path_length,
                    "intervened": intervened,
                    "anchored": int(bool(getattr(iv, "_anchored", False))),
                    "held": int(bool(info.get("held", False))),
                    "reject_reason": info.get("reject_reason") or "",
                    "substeps": int(info.get("intervention_substeps", 0) or 0),
                    "governed": int(bool(info.get("governed", False))),
                    "governed_scale": info.get("governed_scale", ""),
                    "follow_ticks": int(
                        info.get("intervention_follow_ticks", 0) or 0
                    ),
                    "saturated": int(bool(info.get("intervention_saturated", False))),
                    "saturation": info.get("intervention_saturation", ""),
                    "deadman_engaged": dm_engaged,
                    "deadman_age_s": dm_age,
                    "gain_latched": getattr(iv, "_gain", None),
                    "gain_live": gain_live,
                    "leader_age_s": l_age,
                    "leader_grip": grip,
                    "leader_tcp_x": lx, "leader_tcp_y": ly, "leader_tcp_z": lz,
                    "g_anchor_x": gx, "g_anchor_y": gy, "g_anchor_z": gz,
                    "r_anchor_x": rx, "r_anchor_y": ry, "r_anchor_z": rz,
                    "cmd_tcp_pre_x": T_pre[0, 3], "cmd_tcp_pre_y": T_pre[1, 3],
                    "cmd_tcp_pre_z": T_pre[2, 3],
                    "cmd_tcp_x": T_post[0, 3], "cmd_tcp_y": T_post[1, 3],
                    "cmd_tcp_z": T_post[2, 3],
                    "dp_x": dp[0], "dp_y": dp[1], "dp_z": dp[2],
                    "dw_x": dw[0], "dw_y": dw[1], "dw_z": dw[2],
                    "meas_tcp_x": obs["state"]["tcp_pose"][0],
                    "meas_tcp_y": obs["state"]["tcp_pose"][1],
                    "meas_tcp_z": obs["state"]["tcp_pose"][2],
                    "joint_age_s": q_age,
                    "dp_norm": dp_norm,
                    "req_dp_norm": req,
                    "dp_ratio": ratio,
                }
                for i in range(6):
                    row[f"lq{i}"] = None if q_lead is None else float(q_lead[i])
                    row[f"q{i}"] = None if q_now is None else float(q_now[i])
                for i in range(7):
                    row[f"ia{i}"] = None if ia is None else float(np.asarray(ia)[i])
                    row[f"pa{i}"] = float(np.asarray(pa)[i])

                writer.writerow([_f(row[c]) for c in COLUMNS])
                fh.flush()
                # 채점기는 숫자 dict를 본다 (빈 셀은 NaN으로).
                rows.append({
                    k: (float("nan") if v is None else v) for k, v in row.items()
                })

                # ---- HUD ---- #
                if intervened != prev_iv or row["step"] % 10 == 0:
                    tag = "INTERVENE" if intervened else "POLICY   "
                    line = (f"  step {row['step']:4d} {tag} "
                            f"cmd=[{T_post[0, 3]:+.3f} {T_post[1, 3]:+.3f} {T_post[2, 3]:+.3f}] "
                            f"held={row['held']} {row['reject_reason']}")
                    if intervened:
                        line += (f"\n              leader_age={l_age:.2f}s "
                                 f"gain={row['gain_latched']} "
                                 f"dp={dp_norm * 100:.2f}cm ratio={ratio:.2f}")
                    print(line)
                    prev_iv = intervened
                if done:
                    break
            print(f"  episode {ep} 종료 ({base_env.curr_path_length} 스텝)")

    except KeyboardInterrupt:
        print("\nCtrl-C — 중단. 업샘플러가 멈추면 팔은 마지막 명령에서 홀드한다.")
    finally:
        if fh is not None:
            fh.close()
        try:
            env.close()
        except Exception as e:  # noqa: BLE001
            print(f"  close() 예외 무시: {e}")
        if rows:
            summarize(rows)
            print(f"CSV: {csv_path}")
        else:
            print("기록된 스텝이 없다 — CSV는 헤더만 있거나 비어 있다.")


if __name__ == "__main__":
    main()
