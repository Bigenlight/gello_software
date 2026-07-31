#!/usr/bin/env bash
# =============================================================================
# run_hil_actor.sh — HIL-SERL 로봇 랩톱 actor 실행 래퍼 (+ 사전 점검)
# =============================================================================
#
# ── 이 스크립트가 존재하는 이유 ───────────────────────────────────────────────
# 실기에서 actor를 띄우다 "환경 문제로만" 4번 실패했다. 코드가 아니라 매번
# 손으로 치는 실행 명령이 취약해서다. 그 4가지를 여기서 구조적으로 없앤다.
#
#   1) `python3`로 실행  → 시스템 grpcio 1.30.2가 손상돼 있어 에러도 로그도
#      없이 CPU 100%로 영구 정지한다. 이 스크립트는 항상
#      $ACTOR_VENV/bin/python (기본 /home/laptop3/venvs/gello-hil-actor)를
#      절대경로로 exec 한다. `python3`가 쓰일 경로 자체가 없다.
#   2) PYTHONPATH=... 로 "덮어쓰기" → ROS 오버레이(ur_gello_bringup)가 날아가
#      ModuleNotFoundError. 여기서는 항상 `:$PYTHONPATH`로 "이어붙인다".
#   3) `source install/setup.bash` 누락 → 오버레이 없음. 여기서 항상 소스한다.
#   4) `cd` 누락 → 상대경로 실패. 여기서는 스크립트 위치로부터 REPO_ROOT를
#      계산해 cd 하고, 모든 경로를 절대경로로 만든다. (워크트리 안전:
#      /home/laptop3/gello_software 를 하드코딩하지 않는다.)
#
# ── 안전 전제 (실기 모드) ─────────────────────────────────────────────────────
#   * actor는 /forward_position_controller/commands 의 퍼블리셔가 된다.
#     텔레옵 브리지(gello_ur_bridge)가 같은 토픽을 물고 있으면 두 퍼블리셔가
#     서로 다른 목표로 컨트롤러를 때려 팔이 떨거나 튄다. 이 리그 최대 하자다.
#     → 사전 점검에서 다른 퍼블리셔가 하나라도 있으면 **거부하고 중단**한다.
#   * --arm은 controller handoff까지 이 래퍼가 소유한다. 정상 입력은 둘뿐이다:
#       (a) scaled_joint_trajectory_controller active + FPC inactive
#           -> run_hil_preposition.sh의 짧은 수명 proof marker + 현재 RESET 자세를
#              다시 확인한 뒤에만 strict switch
#       (b) FPC active + STJC inactive -> live RESET 자세 확인 후 idempotent, switch 없음
#     둘 다 아니거나 switch 후 상태가 정확하지 않으면 actor를 exec하지 않는다.
#   * deadman gate는 **채널 생존 증명**이다 — GUI가 살아서 /hil/deadman을 실제로
#     퍼블리시하고 있다는 것을 팔을 프로그램에 넘기기 전에 확인한다. 기본 요구 상태는
#     **DISENGAGED**다: 세션은 policy 제어로 시작하므로 handoff 시점에 ENGAGED일
#     이유가 없고, DISENGAGED로 시작하면 handoff 직후 GELLO를 잡아도 팔이 따라오지
#     않는다(더 안전하다). ENGAGE는 개입할 때만 누른다.
#     옛 동작(ENGAGED 요구)은 HIL_STARTUP_DEADMAN=engaged로 되돌린다.
#     연속 fresh heartbeat 3개 요구는 두 모드 모두 동일하다.
#   * --arm 없는 actor와 --dry-preflight는 controller를 절대 전환하지 않는다.
#   * 태스크 config의 DRY_RUN=True가 기본이다(cube_in_cup). DRY_RUN 해제는
#     --arm을 actor에게 넘길 때만 일어난다. 이 래퍼는 관절 command를 발행하거나
#     reset 이동을 호출하지 않는다. 다만 실제 --arm에서는 검증 뒤 controller만 전환한다.
#   * [1]~[11] 사전 점검은 전부 읽기 전용이다(topic liveness / topic info /
#     control list_controllers / TCP connect). 실제 switch는 그 뒤 [ARM] 한 곳뿐이다.
#
# ── 종료 코드 계약 (run_hil_session.sh의 재시도 루프가 이것만 보고 판단한다) ──
#   0   정상 종료. 재시도하지 않는다.
#   1   preflight FAIL 또는 handoff 거부/실패 — **arming 자체가 일어나지 않았다**.
#       사람이 봐야 한다. 재시도하지 않는다.
#   2   래퍼 사용법/설정 오류. 재시도하지 않는다.
#   70  actor 종료 뒤 **controller 자동 복귀 실패** — controller 소유권이 불명이다.
#       펜던트를 들고 직접 확인해야 하므로 기본적으로 재시도하지 않는다.
#       ⚠️ **"스택이 아예 없다"는 여기 해당하지 않는다.** controller_manager가
#       응답하지 않거나 우리 controller 쌍이 목록에 하나도 없으면 그것은
#       "소유권 불명"이 아니라 **하드웨어 번들이 죽었다**는 뜻이고, 그때는
#       ros2_control 컨트롤러가 하나도 없으므로 팔을 몰 수 있는 것도 없다.
#       그 경우는 아래 75로 간다(2026-07-31 실기에서 Terminal 2를 내렸더니
#       70이 나와 세션이 자동 종료된 것이 이 구분이 없었기 때문이다).
#   75  **RECOVERABLE**: actor가 실제로 떴다가 죽었고(0<rc<128), 그 뒤
#       controller 복귀는 PASS했으며, **actor가 transition 루프까지 실제로
#       도달했다는 양성 증거**가 있다. 팔은 trajectory controller가 잡고 있고
#       command publisher는 0이다. 충돌/protective stop 뒤의 정상적인 모습이다.
#       run_hil_session.sh는 **이 코드에서만** 재시도를 고려한다(그것도 하드웨어
#       번들이 실제로 재기동된 것을 확인한 뒤에만).
#       ⚠️ 양성 증거가 없으면 **75로 승격하지 않고 actor의 raw rc를 그대로 낸다.**
#       근거: 0<rc<128 전체를 승격하면 run_remote_rlpd_actor.py의 모든
#       `raise SystemExit(...)`(인자 검증), 핸드셰이크/schema-hash 거부,
#       첫 transition의 ActorProtocolError(`d6965a9`가 고친 그 실패), exec 실패
#       126/127까지 "재시도 가능"이 된다. 그것들은 **결정론적**이라 재시도해도
#       같은 지점에서 같이 죽고, 시도마다 8초짜리 충돌회피 없는 RESET 이동만
#       한 번씩 더 실행된다. 증거는 아래 "진행 증거" 항목 참조.
#       ✅ **예외 하나 — ros2_control 스택 소멸.** controller_manager가 아예
#       없거나 우리 controller 쌍이 목록에 없으면 진행 증거를 **요구하지 않고**
#       75를 낸다. 그 게이트의 목적은 "결정론적 startup 실패를 반복하지 마라"인데
#       결정론적 startup 실패는 controller_manager를 사라지게 만들지 않는다.
#       스택 소멸은 그것과 독립적이고 더 직접적인 환경 실패의 증거다.
#   >=128  신호로 죽음(130=Ctrl-C 등). 그대로 전파하고 재시도하지 않는다.
#   재mapping 끄기: HIL_ACTOR_EXIT_MAP=0 (그러면 actor의 raw rc를 그대로 낸다).
#
# ── 진행 증거 (75 승격의 전제) ───────────────────────────────────────────────
#   실기 --arm 동안 이 래퍼는 `/hil/actor_status`(std_msgs/String, JSON)를
#   구독하는 작은 읽기 전용 감시자를 함께 띄운다. `env_step >= 0`인 status를
#   한 번이라도 보면 마커 파일을 남기고 즉시 종료한다.
#   왜 이 신호인가 (`ur_env/remote_actor.py` 확인):
#     * `_OperatorReporter`는 env_step을 **-1**로 시작하고(`self.env_step = -1`),
#       `position()`은 **transition 루프 안에서만** 호출된다.
#     * 따라서 HOME / WAIT_SCENE_READY / 핸드셰이크 구간의 status는 전부
#       env_step=-1이고, env_step>=0은 "actor가 transition 루프 안으로
#       들어갔다"는 뜻이다. 그 값을 세우는 곳은 두 군데뿐인데 하나는 Step RPC
#       ack 직후(정상 경로), 다른 하나는 조작자가 step 직전에 누른 abort다 —
#       후자도 actor가 살아서 GUI 서비스에 응답하고 있었다는 증거다.
#     * 첫 transition의 ActorProtocolError는 `build_data`->`validate_action`,
#       즉 그 publish **이전**에 터지므로 마커가 생기지 않는다 → 재시도 안 함.
#   감시자를 띄울 수 없거나(rclpy 없음 등) 마커가 없으면 **fail-closed**:
#   승격하지 않는다. 증거가 없으면 자동 재arming도 없다.
#   75는 "안전하다"는 뜻이 **아니다**. "이 wrapper가 아는 한 팔은 controller가
#   붙잡은 정지 상태이고, 다시 arming하려면 모든 proof를 처음부터 다시 통과해야
#   한다"는 뜻이다. 재시도는 resume이 아니라 **새로운 arming**이다.
#
# ── 중단 방법 ────────────────────────────────────────────────────────────────
#   * 이 터미널에서 Ctrl-C. 실기 --arm에서는 래퍼가 신호를 actor에게 전달하고,
#     actor가 publisher를 닫은 뒤 trajectory controller로 자동 복귀한다.
#   * 팔이 이미 움직이는 중이라면 먼저 펜던트 E-STOP. actor가 죽으면
#     forward_position_controller는 마지막 명령을 유지하므로 팔은 제자리 홀드
#     (튀지 않는다).
#   * GELLO 개입 중이면 HIL GUI에서 DISENGAGE.
#
# ── 사용법 ───────────────────────────────────────────────────────────────────
#   ./run_hil_actor.sh --dry-preflight          # 점검만, actor 미기동 (읽기 전용)
#   ./run_hil_actor.sh --dry-preflight --arm    # arm handoff 준비까지 읽기 전용 검증
#   ./run_hil_actor.sh --fake-env               # Stage A: fake env로 learner 왕복
#   ./run_hil_actor.sh                          # Stage B: 실센서 actor
#   ./run_hil_actor.sh --arm --deadman topic    # proof 확인 + FPC 전환 + 실제 actor
#   ./run_hil_actor.sh --save-video --actor-id foo   # 인자는 그대로 통과
#
#   환경변수로 기본값 덮어쓰기:
#     ACTOR_VENV, SERVER_HOST, SERVER_PORT, EXP_NAME, UR_CONFIG_MODULE,
#     TIMEOUT_S, MAX_RESPONSE_AGE_S, OBS_SCHEMA_HASH, EXPECTED_MODEL_ID,
#     EXPECTED_REWARD_AUTHORITY, EXPECTED_REWARD_MODEL_ID, ROS_SETUP,
#     HZ_TIMEOUT_S, HIL_PREPOSITION_MARKER, HIL_PREPOSITION_MARKER_MAX_AGE_S,
#     HIL_STARTUP_DEADMAN (disengaged|engaged — 아래 deadman gate 항목),
#     HIL_ACTOR_EXIT_MAP (1|0 — 위 종료 코드 계약의 75 재mapping)
#   SKIP_ROS_CHECKS=1은 fake/no-arm 진단 전용이며 --arm과 함께 쓰면 거부한다.
#
# NOTE: `set -e`만 쓴다. `set -u`는 쓰지 않는다 — ROS의 setup.bash가 -u에서
#       죽는다(이 저장소에서 이미 두 스크립트가 같은 버그로 깨졌다가 고쳐졌다).
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HANDOFF_LIB="$SCRIPT_DIR/_hil_controller_handoff.sh"
POSE_CHECKER="$SCRIPT_DIR/_hil_joint_pose_check.py"
DEADMAN_CHECKER="$SCRIPT_DIR/_hil_deadman_check.py"
TOPIC_CHECKER="$SCRIPT_DIR/_hil_topic_rate_check.py"
if [ ! -f "$HANDOFF_LIB" ] || [ ! -f "$POSE_CHECKER" ] || \
   [ ! -f "$DEADMAN_CHECKER" ] || [ ! -f "$TOPIC_CHECKER" ]; then
    echo "FATAL: HIL handoff/deadman/topic helper missing under $SCRIPT_DIR" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$HANDOFF_LIB"

# ---------------------------------------------------------------------------
# 설정 (환경변수로 덮어쓸 수 있음)
# ---------------------------------------------------------------------------
ACTOR_VENV="${ACTOR_VENV:-/home/laptop3/venvs/gello-hil-actor}"
ACTOR_PY="$ACTOR_VENV/bin/python"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WS_SETUP="$REPO_ROOT/ros2_ur_ws/install/setup.bash"
ACTOR_SCRIPT="$REPO_ROOT/serl_ur_infra/scripts/run_remote_rlpd_actor.py"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-50153}"          # 로컬 터널 입구 → learner 50053
EXP_NAME="${EXP_NAME:-cube_in_cup}"
UR_CONFIG_MODULE="${UR_CONFIG_MODULE:-ur_experiments.mappings}"
# Real startup measurements include observation serialization, SSH transport,
# policy inference, and (periodically) reward-classifier inference.  The old
# 0.6/0.8 s pair rejected an otherwise healthy fifth transition at 832.3 ms.
# Keep both waits bounded, but leave enough margin for the measured cold path.
TIMEOUT_S="${TIMEOUT_S:-1.5}"
MAX_RESPONSE_AGE_S="${MAX_RESPONSE_AGE_S:-2.0}"
OBS_SCHEMA_HASH="${OBS_SCHEMA_HASH:-3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903}"
EXPECTED_MODEL_ID="${EXPECTED_MODEL_ID:-hil-serl-hybrid-sac-resnet10-trunk-cache-v1}"
EXPECTED_REWARD_AUTHORITY="${EXPECTED_REWARD_AUTHORITY:-server_classifier}"
# 서버 `--reward-model-id` 와 **정확히 같은 문자열**이어야 한다. 이 값은 자유
# 문자열이고 GetServerInfo 에서 동일성만 검사하므로, 체크포인트뿐 아니라 **분류기가
# 무슨 픽셀을 봤는지**(input contract)까지 이름에 넣는다:
#
#   cube-in-cup-all3-ckpt150 : classifier_ckpt/cube_in_cup_all3/checkpoint_150
#   +sidecar-v1              : ur_env/classifier_sidecar.py CLASSIFIER_INPUT_ID
#                              = "fullframe-jpeg-passthrough-v1"
#
# 이렇게 해야 sidecar 이전 actor ↔ 이후 server (또는 그 반대)가 **핸드셰이크에서
# 거부**된다. 그러지 않으면 양쪽이 서로 다른 이미지로 계산된 reward 를 아무 경고 없이
# 한 세션 내내 주고받는다 — 이 리그에서 가장 알아채기 어려운 고장이다.
# 기본값은 serl_ur_infra/scripts/run_rlpd_{receive,learner}_server.py 의
# DEFAULT_REWARD_MODEL_ID 와 한 커밋에서 같이 바꾼다.
EXPECTED_REWARD_MODEL_ID="${EXPECTED_REWARD_MODEL_ID:-cube-in-cup-all3-ckpt150+sidecar-v1}"

# 손상된 시스템 grpcio 버전 (조용한 무한정지의 원인)
BROKEN_GRPCIO_VERSION="1.30.2"

# 읽기 전용 ROS 점검 대상
JOINT_STATES_TOPIC="/joint_states"
GELLO_TOPIC="/gello/joint_states"
CAM1_TOPIC="/cam1/cam1/color/image_raw/compressed"
CAM2_TOPIC="/cam2/cam2/color/image_raw/compressed"
GRIPPER_STATE_TOPIC="/robotiq_gripper/position_percent"
COMMAND_TOPIC="/forward_position_controller/commands"
ARM_CONTROLLER="forward_position_controller"
SOURCE_ARM_CONTROLLER="scaled_joint_trajectory_controller"
# Session startup briefly loads DDS, two RealSense processes, Qt, and the actor
# environment at the same time.  This preflight therefore checks liveness, not
# a five-frame instantaneous rate.  Give a newly-created reader enough time to
# be discovered; the hardware launcher has already applied its stricter
# steady-state readiness checks before this script runs.
HZ_TIMEOUT_S="${HZ_TIMEOUT_S:-12}"
# Must match CubeInCupEnvConfig.RESET_JOINTS exactly.  This is checked against
# the marker and against a fresh /joint_states sample before an automatic switch.
RESET_JOINTS_CSV="3.1382,-1.5276,1.7168,-1.7592,-1.5216,-3.1331"
ARM_POSE_TOL_RAD="${HIL_ARM_POSE_TOL_RAD:-0.10}"
PREPOSITION_MARKER_MAX_AGE_S="${HIL_PREPOSITION_MARKER_MAX_AGE_S:-900}"
PREPOSITION_MARKER="${HIL_PREPOSITION_MARKER:-$(hil_default_preposition_marker)}"
if ! awk -v value="$ARM_POSE_TOL_RAD" \
    'BEGIN { exit !(value > 0 && value <= 0.10) }'; then
    echo "FATAL: HIL_ARM_POSE_TOL_RAD=$ARM_POSE_TOL_RAD must be in (0, 0.10]" >&2
    exit 1
fi
if [[ ! "$PREPOSITION_MARKER_MAX_AGE_S" =~ ^[1-9][0-9]*$ ]] || \
   (( PREPOSITION_MARKER_MAX_AGE_S > 3600 )); then
    echo "FATAL: HIL_PREPOSITION_MARKER_MAX_AGE_S must be an integer in [1, 3600]" >&2
    exit 1
fi

# deadman gate가 요구하는 상태.  기본은 DISENGAGED (policy-first 시작).
# 오타를 조용히 "gate 없음"으로 해석하지 않도록 값 검사는 fail-closed다.
STARTUP_DEADMAN="${HIL_STARTUP_DEADMAN:-disengaged}"
case "$STARTUP_DEADMAN" in
    disengaged|engaged) ;;
    *)
        echo "FATAL: HIL_STARTUP_DEADMAN must be 'disengaged' or 'engaged' (got '$STARTUP_DEADMAN')" >&2
        exit 1
        ;;
esac
STARTUP_DEADMAN_LABEL="$(printf '%s' "$STARTUP_DEADMAN" | tr '[:lower:]' '[:upper:]')"

# 종료 코드 재mapping (파일 상단 "종료 코드 계약" 참조).  0으로 두면 actor의 raw rc.
ACTOR_EXIT_MAP="${HIL_ACTOR_EXIT_MAP:-1}"
case "$ACTOR_EXIT_MAP" in
    0|1) ;;
    *)
        echo "FATAL: HIL_ACTOR_EXIT_MAP must be 0 or 1 (got '$ACTOR_EXIT_MAP')" >&2
        exit 2
        ;;
esac
RECOVERABLE_EXIT_CODE=75

# ---------------------------------------------------------------------------
# 인자 파싱: --dry-preflight / --help 만 소비하고 나머지는 전부 통과시킨다
# ---------------------------------------------------------------------------
DRY_PREFLIGHT=0
FAKE_ENV=0
ARM_REQUESTED=0
PASSTHRU=()
for arg in "$@"; do
    case "$arg" in
        --dry-preflight)
            DRY_PREFLIGHT=1
            ;;
        -h|--help)
            # 파일 상단 주석 블록만 출력 (첫 비주석 줄에서 멈춘다)
            awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' \
                "${BASH_SOURCE[0]}"
            exit 0
            ;;
        --fake-env)
            FAKE_ENV=1
            PASSTHRU+=("$arg")
            ;;
        --arm)
            ARM_REQUESTED=1
            PASSTHRU+=("$arg")
            ;;
        *)
            PASSTHRU+=("$arg")
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 점검 결과 리포팅
# ---------------------------------------------------------------------------
FAIL_COUNT=0
WARN_COUNT=0
FAILURES=()
WARNINGS=()

_say()  { printf '%s\n' "$*"; }
p_ok()   { printf '  [ OK ] %s\n' "$*"; }
p_info() { printf '  [INFO] %s\n' "$*"; }
p_skip() { printf '  [SKIP] %s\n' "$*"; }
p_warn() {
    printf '  [WARN] %s\n' "$1"
    [ -n "$2" ] && printf '         → %s\n' "$2"
    WARNINGS+=("$1")
    WARN_COUNT=$((WARN_COUNT + 1))
}
p_fail() {
    printf '  [FAIL] %s\n' "$1"
    [ -n "$2" ] && printf '         → %s\n' "$2"
    FAILURES+=("$1")
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

# ---------------------------------------------------------------------------
# Qt 폰트 경고 억제 (actor는 DISPLAY_IMAGE=True라 cv2 창을 연다)
# ---------------------------------------------------------------------------
# 무엇을 지우나: pip cv2(opencv-python 4.13.0)가 번들한 Qt 플러그인이 창을 열 때마다
# 내는 두 줄, 매 프로세스 5회:
#     QFontDatabase: Cannot find font directory .../cv2/qt/fonts.
#     Note that Qt no longer ships fonts. Deploy some (...) or switch to fontconfig.
#
# 왜 환경변수가 아닌가 (측정, 2026-07-30): cv2/config-3.py 가 **import 시점에**
#     os.environ["QT_QPA_FONTDIR"] = <cv2>/qt/fonts
# 로 **덮어쓴다**. 그래서 쉘에서 QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu 를
# 줘도 경고는 그대로다(직접 확인함). 그리고 이 메시지는 카테고리 없는 qWarning이라
# QT_LOGGING_RULES='qt.qpa.fonts.warning=false' / 'qt.qpa.*.warning=false' 로는
# 안 지워지고, 지워지는 유일한 규칙은 'default.warning=false' — 그건 **모든** Qt
# 경고를 죽이므로 쓰지 않는다(진짜 Qt 오류를 숨긴다). site-packages 안에 폰트
# 디렉터리를 만드는 것도 하지 않는다(pip가 덮어쓰고 다른 체크아웃과 공유된다).
# 남는 정확한 방법이 이것이다: 알려진 두 줄만 stderr에서 지운다. 다른 Qt 메시지
# (qt.qpa.plugin 오류 등)는 그대로 통과한다.
# 되돌리려면: 아래 두 launch 지점의 `2> >(...)` 리다이렉션을 지운다.
#
# 패턴은 **양 끝을 고정한다**. 두 번째 줄을 `Deploy some ` 까지만 맞추면 그 줄에
# 무엇이 덧붙어도 통째로 삼킨다. 문구가 바뀌면 걸러지지 않고 그대로 보이는 쪽이
# 맞는 실패 방향이다(안 보이는 것보다 낫다).
hil_filter_qt_font_noise() {
    grep --line-buffered -v -E \
        -e '^QFontDatabase: Cannot find font directory .*/cv2/qt/fonts\.$' \
        -e '^Note that Qt no longer ships fonts\. Deploy some \(.*\) or switch to fontconfig\.$' || true
}

# ---------------------------------------------------------------------------
# actor 진행 증거 감시자 (파일 상단 "진행 증거" 항목이 근거 전부)
# ---------------------------------------------------------------------------
# 읽기 전용이다: /hil/actor_status 를 구독만 하고 아무것도 발행/호출하지 않는다.
# env_step >= 0 인 status를 한 번 보면 마커를 쓰고 **즉시 종료**한다 (DDS reader
# 수명을 최소로 유지). 부모(이 래퍼)가 사라지면 스스로 빠져나온다.
ACTOR_STATUS_TOPIC="/hil/actor_status"
PROGRESS_DIR=""
PROGRESS_MARKER=""
PROGRESS_WATCH_PID=""

hil_stop_progress_watch() {
    if [ -n "$PROGRESS_WATCH_PID" ]; then
        kill "$PROGRESS_WATCH_PID" 2>/dev/null || true
        wait "$PROGRESS_WATCH_PID" 2>/dev/null || true
        PROGRESS_WATCH_PID=""
    fi
}

hil_cleanup_progress_watch() {
    hil_stop_progress_watch
    if [ -n "$PROGRESS_DIR" ]; then
        rm -rf "$PROGRESS_DIR" 2>/dev/null || true
        PROGRESS_DIR=""
    fi
}

hil_start_progress_watch() {
    PROGRESS_DIR="$(mktemp -d /tmp/hil_actor_progress_XXXXXX)" || {
        PROGRESS_DIR=""
        return 1
    }
    PROGRESS_MARKER="$PROGRESS_DIR/first_transition"
    trap hil_cleanup_progress_watch EXIT
    python3 - "$PROGRESS_MARKER" "$$" "$ACTOR_STATUS_TOPIC" \
        >"$PROGRESS_DIR/watch.log" 2>&1 <<'PYEOF' &
import json
import os
import sys

marker, parent_pid, topic = sys.argv[1], int(sys.argv[2]), sys.argv[3]

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception as exc:  # noqa: BLE001 - any import failure is equivalent
    print(f"actor progress watch unavailable: {exc}", file=sys.stderr)
    raise SystemExit(2)


class Watch(Node):
    def __init__(self) -> None:
        super().__init__("hil_actor_progress_watch")
        self.seen = False
        self.create_subscription(String, topic, self._callback, 10)

    def _callback(self, message) -> None:
        if self.seen:
            return
        try:
            env_step = int(json.loads(message.data)["env_step"])
        except Exception:  # noqa: BLE001 - a malformed status proves nothing
            return
        # env_step is -1 for HOME / WAIT_SCENE_READY / handshake statuses and is
        # first set >= 0 inside the transition loop.  See the "진행 증거" block
        # at the top of run_hil_actor.sh.
        if env_step >= 0:
            self.seen = True


rclpy.init()
node = Watch()
try:
    while rclpy.ok() and not node.seen and os.getppid() == parent_pid:
        rclpy.spin_once(node, timeout_sec=0.2)
finally:
    node.destroy_node()
    rclpy.shutdown()

if not node.seen:
    raise SystemExit(1)
with open(marker, "w", encoding="utf-8") as handle:
    handle.write(f"env_step>=0 observed on {topic}\n")
PYEOF
    PROGRESS_WATCH_PID=$!
}

_say "============================================================"
_say " HIL actor preflight"
_say "   repo root : $REPO_ROOT"
_say "   venv      : $ACTOR_VENV"
_say "   server    : $SERVER_HOST:$SERVER_PORT"
_say "   exp/config: $EXP_NAME / $UR_CONFIG_MODULE"
if [ "$FAKE_ENV" -eq 1 ]; then
    _say "   mode      : FAKE ENV (로봇/센서 미사용)"
else
    _say "   mode      : REAL (실센서 + 실기 배선)"
fi
if [ "$ARM_REQUESTED" -eq 1 ]; then
    if [ "$DRY_PREFLIGHT" -eq 1 ]; then
        _say "   arm       : READ-ONLY READINESS CHECK (--dry-preflight; switch 금지)"
    else
        _say "   arm       : REQUESTED (preflight 후 proof 기반 handoff)"
    fi
else
    _say "   arm       : off (controller switch 금지)"
fi
_say "============================================================"

if [ "$ARM_REQUESTED" -eq 1 ] && [ "$FAKE_ENV" -eq 1 ]; then
    p_fail "--fake-env와 --arm을 함께 쓸 수 없다" \
           "fake 계약 점검에는 --arm을 빼고, 실기는 --fake-env를 빼라"
fi
if [ "$ARM_REQUESTED" -eq 1 ] && [ "${SKIP_ROS_CHECKS:-0}" = "1" ]; then
    p_fail "--arm에서 SKIP_ROS_CHECKS=1은 금지된다" \
           "controller/topic proof를 건너뛴 채 물리 명령 경로를 열 수 없다"
fi

# ---------------------------------------------------------------------------
# 1. venv 존재 / 올바른 인터프리터
# ---------------------------------------------------------------------------
_say ""
_say "[1] Python 환경 (python3 금지 — 손상된 시스템 grpcio 회피)"
VENV_OK=0
if [ ! -x "$ACTOR_PY" ]; then
    p_fail "actor venv python이 없다: $ACTOR_PY" \
           "ACTOR_VENV를 확인하거나 venv를 만든다: python3 -m venv --system-site-packages $ACTOR_VENV"
else
    PY_PREFIX="$(timeout 30 "$ACTOR_PY" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
    if [ "$PY_PREFIX" != "$ACTOR_VENV" ]; then
        p_fail "$ACTOR_PY 의 sys.prefix가 venv가 아니다 (prefix='$PY_PREFIX')" \
               "venv가 깨졌다. 다시 만든다."
    else
        PY_VER="$(timeout 30 "$ACTOR_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true)"
        p_ok "interpreter = $ACTOR_PY (python $PY_VER)"
        VENV_OK=1
    fi
fi

# ---------------------------------------------------------------------------
# 2. grpcio 버전 — 1.30.2(손상판)면 거부
# ---------------------------------------------------------------------------
_say ""
_say "[2] grpcio 버전 (손상판 $BROKEN_GRPCIO_VERSION 감지)"
if [ "$VENV_OK" -ne 1 ]; then
    p_skip "venv가 없어 건너뜀"
else
    # grpc를 import하지 않고 메타데이터만 읽는다 (import 자체가 멈출 수 있으므로).
    GRPCIO_VER="$(timeout 30 "$ACTOR_PY" -c \
        'import importlib.metadata as m; print(m.version("grpcio"))' 2>/dev/null || true)"
    if [ -z "$GRPCIO_VER" ]; then
        p_fail "venv 안에서 grpcio 버전을 읽지 못했다" \
               "$ACTOR_PY -m pip install -r $REPO_ROOT/serl_ur_infra/requirements-grpc.lock"
    elif [ "$GRPCIO_VER" = "$BROKEN_GRPCIO_VERSION" ]; then
        p_fail "grpcio $GRPCIO_VER — 이 리그에서 손상된 것으로 확인된 바로 그 버전" \
               "이 버전은 에러도 로그도 없이 CPU 100%로 영구 정지한다. venv에 최신 grpcio를 설치할 것."
    else
        p_ok "grpcio $GRPCIO_VER (손상판 아님)"
    fi
fi

# ---------------------------------------------------------------------------
# 3. gRPC 코어 생존 확인 — 반드시 타임아웃 + 서브프로세스로
#    (이 스크립트가 같이 멈추면 안 되므로 절대 인라인으로 부르지 않는다)
# ---------------------------------------------------------------------------
_say ""
_say "[3] gRPC 코어 생존 (cygrpc.CompletionQueue, 타임아웃 20s 서브프로세스)"
if [ "$VENV_OK" -ne 1 ]; then
    p_skip "venv가 없어 건너뜀"
else
    set +e
    CYGRPC_OUT="$(timeout --signal=KILL 20 "$ACTOR_PY" -c '
from grpc._cython import cygrpc
cq = cygrpc.CompletionQueue()
print("cygrpc_ok")
' 2>&1)"
    CYGRPC_RC=$?
    set -e
    if [ "$CYGRPC_RC" -eq 137 ] || [ "$CYGRPC_RC" -eq 124 ]; then
        p_fail "cygrpc.CompletionQueue()가 20초 안에 돌아오지 않았다 (KILL)" \
               "이게 바로 '에러도 로그도 없이 CPU 100%'의 증상이다. 이 상태로 actor를 띄우면 영구 정지한다. venv의 grpcio를 재설치할 것."
    elif [ "$CYGRPC_RC" -ne 0 ]; then
        p_fail "cygrpc import/생성 실패 (rc=$CYGRPC_RC)" \
               "$(printf '%s' "$CYGRPC_OUT" | tail -n 3 | tr '\n' ' ')"
    else
        p_ok "gRPC 코어 정상 (CompletionQueue 생성 성공)"
    fi
fi

# ---------------------------------------------------------------------------
# 4. ROS 오버레이 소스 (누락 함정 #3) + PYTHONPATH 이어붙이기 (함정 #2)
# ---------------------------------------------------------------------------
_say ""
_say "[4] ROS 오버레이 / PYTHONPATH"
OVERLAY_OK=0
if [ ! -f "$ROS_SETUP" ]; then
    p_fail "ROS setup을 찾을 수 없다: $ROS_SETUP" "ROS_SETUP 환경변수로 지정할 것"
elif [ ! -f "$WS_SETUP" ]; then
    p_fail "워크스페이스 오버레이가 없다: $WS_SETUP" \
           "cd $REPO_ROOT/ros2_ur_ws && ./build_ur7e.sh 로 먼저 빌드할 것"
else
    # shellcheck disable=SC1090
    source "$ROS_SETUP"
    # shellcheck disable=SC1090
    source "$WS_SETUP"
    p_ok "sourced $ROS_SETUP"
    p_ok "sourced $WS_SETUP"
    OVERLAY_OK=1
fi

# ★ 덮어쓰기(=)가 아니라 앞에 붙이고 기존 값을 반드시 뒤에 잇는다.
#   ${PYTHONPATH:+:$PYTHONPATH} 는 PYTHONPATH가 비어 있으면 콜론도 안 붙인다.
export PYTHONPATH="$REPO_ROOT/serl_ur_infra:$REPO_ROOT/third_party/hil-serl/serl_launcher:$REPO_ROOT/third_party/hil-serl/examples${PYTHONPATH:+:$PYTHONPATH}"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$REPO_ROOT}"
p_info "PYTHONPATH(앞 3개) = $REPO_ROOT/serl_ur_infra : .../serl_launcher : .../examples"
# 함정 #2의 직접 확인: 오버레이가 PYTHONPATH 뒤쪽에 살아남았는가.
if [ "$OVERLAY_OK" -eq 1 ]; then
    case ":$PYTHONPATH:" in
        *":$REPO_ROOT/ros2_ur_ws/install/ur_gello_bringup/lib/python3.10/site-packages:"*)
            p_ok "ROS 오버레이가 PYTHONPATH에 살아 있다 (덮어쓰기 아님)"
            ;;
        *)
            p_warn "PYTHONPATH에서 ur_gello_bringup 오버레이 경로를 못 찾았다" \
                   "오버레이 레이아웃이 다를 수 있다 — 아래 [5] 모듈 해석 결과로 판단할 것"
            ;;
    esac
fi

# 함정 #4: 상대경로가 어디서 호출되든 같은 뜻이 되게 한다.
cd "$REPO_ROOT"
p_ok "cwd = $(pwd)"

# ---------------------------------------------------------------------------
# 5. 모듈 import 가능성 + "이 저장소에서 나왔는가" 확인
#    (serl-ur-infra는 다른 체크아웃에 editable 설치돼 있을 수 있다. 그러면
#     ur_env는 import되지만 이 워크트리의 코드가 아니고, ur_experiments는
#     아예 없어서 --ur-config-module이 깨진다.)
# ---------------------------------------------------------------------------
_say ""
_say "[5] 모듈 해석 (ur_env / ur_gello_bringup / serl_launcher / ur_experiments)"
if [ "$VENV_OK" -ne 1 ]; then
    p_skip "venv가 없어 건너뜀"
else
    set +e
    MOD_OUT="$(HIL_REPO_ROOT="$REPO_ROOT" timeout --signal=KILL 60 "$ACTOR_PY" - <<'PYEOF' 2>&1
import importlib.util as iu
import os
import sys

root = os.environ["HIL_REPO_ROOT"]
# 이 저장소에서 나와야 하는 모듈들
LOCAL = ("ur_env", "ur_experiments", "ur_gello_bringup", "serl_launcher")
rc = 0
for name in LOCAL:
    try:
        spec = iu.find_spec(name)
    except Exception as exc:  # noqa: BLE001
        print("FAIL|%s|import 시도가 예외로 죽음: %s: %s" % (name, type(exc).__name__, exc))
        rc = 1
        continue
    if spec is None:
        print("FAIL|%s|찾을 수 없음 (PYTHONPATH/오버레이 확인)" % name)
        rc = 1
        continue
    origin = spec.origin
    if origin in (None, "namespace"):
        locs = list(spec.submodule_search_locations or [])
        origin = locs[0] if locs else "<unknown>"
    if not os.path.realpath(origin).startswith(os.path.realpath(root) + os.sep):
        print("FAIL|%s|이 저장소 밖에서 해석됨: %s" % (name, origin))
        rc = 1
    else:
        print("OK|%s|%s" % (name, origin))

# 런타임에 실제로 필요한 서드파티
for name in ("rclpy", "gymnasium", "numpy", "cv2"):
    spec = None
    try:
        spec = iu.find_spec(name)
    except Exception as exc:  # noqa: BLE001
        print("FAIL|%s|%s: %s" % (name, type(exc).__name__, exc))
        rc = 1
        continue
    if spec is None:
        print("FAIL|%s|찾을 수 없음" % name)
        rc = 1
    else:
        print("OK|%s|%s" % (name, spec.origin))
sys.exit(rc)
PYEOF
)"
    MOD_RC=$?
    set -e
    while IFS='|' read -r status name detail; do
        [ -z "$status" ] && continue
        case "$status" in
            OK)   p_ok   "$name → $detail" ;;
            FAIL) p_fail "$name: $detail" \
                         "PYTHONPATH 앞부분이 $REPO_ROOT 를 가리키는지, 오버레이를 소스했는지 확인" ;;
            *)    p_info "$status|$name|$detail" ;;
        esac
    done <<< "$MOD_OUT"
    if [ "$MOD_RC" -eq 137 ] || [ "$MOD_RC" -eq 124 ]; then
        p_fail "모듈 해석이 60초 안에 끝나지 않았다" "import가 멈췄다 — grpcio/numpy 설치 상태 확인"
    fi
fi

# actor 진입점 존재
if [ -f "$ACTOR_SCRIPT" ]; then
    p_ok "actor 진입점 = $ACTOR_SCRIPT"
else
    p_fail "actor 진입점이 없다: $ACTOR_SCRIPT" "브랜치/워크트리를 확인할 것"
fi

# ---------------------------------------------------------------------------
# 6. 서버 포트 (SSH 터널 입구)에 실제로 리스닝 중인가
# ---------------------------------------------------------------------------
_say ""
_say "[6] 학습 서버 접속점 $SERVER_HOST:$SERVER_PORT"
if [ "$VENV_OK" -ne 1 ]; then
    p_skip "venv가 없어 건너뜀"
else
    set +e
    PORT_OUT="$(HIL_HOST="$SERVER_HOST" HIL_PORT="$SERVER_PORT" \
        timeout --signal=KILL 15 "$ACTOR_PY" - <<'PYEOF' 2>&1
import os
import socket
import sys

host = os.environ["HIL_HOST"]
port = int(os.environ["HIL_PORT"])
try:
    with socket.create_connection((host, port), timeout=3.0):
        print("연결 성공")
except Exception as exc:  # noqa: BLE001
    print("%s: %s" % (type(exc).__name__, exc))
    sys.exit(1)
PYEOF
)"
    PORT_RC=$?
    set -e
    if [ "$PORT_RC" -eq 0 ]; then
        p_ok "TCP $SERVER_HOST:$SERVER_PORT — $PORT_OUT"
    else
        p_fail "TCP $SERVER_HOST:$SERVER_PORT 연결 실패 — $PORT_OUT" \
               "터널은 run_hil_server.sh(Terminal 1)가 연다. 직접 열려면: ssh -N -T -o ExitOnForwardFailure=yes -L 127.0.0.1:$SERVER_PORT:127.0.0.1:50053 junhyeong_ai (그리고 그 서버에서 learner가 떠 있어야 한다)"
    fi
fi

# ---------------------------------------------------------------------------
# 7~9. ROS 실데이터 / 컨트롤러 / 이중 퍼블리셔 (읽기 전용)
# ---------------------------------------------------------------------------
# fake-env는 ROS를 전혀 쓰지 않으므로 실데이터 점검은 정보성으로만 낸다.
ros_rate_check() {
    # $1=topic $2=label $3=message type $4=min Hz $5=1이면 FAIL/0이면 WARN
    local topic="$1" label="$2" message_type="$3" min_rate="$4" fatal="$5"
    local out rc rate
    set +e
    # The helper owns its timeout and performs rclpy destroy_node()/shutdown().
    # TERM+kill-after is only a backstop for a broken middleware teardown; the
    # old unconditional SIGKILL left short-lived DDS readers behind and could
    # aggravate large RealSense writer delivery to later subscribers.
    out="$(timeout --signal=TERM --kill-after=2 "$((HZ_TIMEOUT_S + 4))" \
        python3 "$TOPIC_CHECKER" --topic "$topic" --type "$message_type" \
        --samples 5 --timeout "$HZ_TIMEOUT_S" --min-rate "$min_rate" 2>&1)"
    rc=$?
    set -e
    rate="$(printf '%s\n' "$out" | sed -n 's/.* rate=\([0-9.][0-9.]*\) Hz.*/\1/p' | tail -1)"
    if [ "$rc" -eq 0 ] && [ -n "$rate" ]; then
        p_ok "$label ($topic) ≈ ${rate} Hz"
        return 0
    fi
    if [ "$fatal" -eq 1 ]; then
        p_fail "$label ($topic) 수신 검증 실패 — ${out:-no diagnostic} (rc=$rc)" \
               "해당 publisher와 신규 subscriber 전달 경로를 확인한다"
    else
        p_warn "$label ($topic) 수신 검증 실패 — ${out:-no diagnostic} (rc=$rc)" \
               "fake-env에서는 쓰이지 않으므로 진행 가능"
    fi
    return 1
}

_say ""
_say "[7] ROS 실데이터 (fresh-message liveness only; 최소 Hz 제한 없음, 각 최대 ${HZ_TIMEOUT_S}s)"
if [ "$OVERLAY_OK" -ne 1 ]; then
    p_skip "오버레이를 소스하지 못해 건너뜀"
elif [ "$SKIP_ROS_CHECKS" = "1" ]; then
    p_skip "SKIP_ROS_CHECKS=1"
elif [ "$FAKE_ENV" -eq 1 ]; then
    p_info "fake-env 모드 — 센서 스트림은 사용되지 않는다 (경고로만 표시)"
    ros_rate_check "$JOINT_STATES_TOPIC" "로봇 관절"  joint_state      0 0 || true
    ros_rate_check "$GELLO_TOPIC"        "GELLO 리더" joint_state      0 0 || true
    ros_rate_check "$CAM1_TOPIC"         "cam1 장면"  compressed_image 0 0 || true
    ros_rate_check "$CAM2_TOPIC"         "cam2 손목"  compressed_image 0 0 || true
else
    # These publishers were already rate-gated by run_hil_hardware.sh and
    # launch_cameras.sh.  Repeating a strict five-frame Hz gate here produced
    # false 9--12 Hz failures during startup even though sustained streams were
    # 30 Hz.  Keep only the fresh/advancing-message requirement.
    ros_rate_check "$JOINT_STATES_TOPIC" "로봇 관절"  joint_state      0 1 || true
    ros_rate_check "$GELLO_TOPIC"        "GELLO 리더" joint_state      0 1 || true
    ros_rate_check "$CAM1_TOPIC"         "cam1 장면"  compressed_image 0 1 || true
    ros_rate_check "$CAM2_TOPIC"         "cam2 손목"  compressed_image 0 1 || true
    # 그리퍼 위치는 19-D state의 마지막 채널이다. 없으면 관측이 불완전하지만
    # DRY_RUN 배선 확인은 가능하므로 경고로만 낸다.
    ros_rate_check "$GRIPPER_STATE_TOPIC" "그리퍼 상태" float32 0 0 || true
fi

_say ""
_say "[8] arm controller 쌍 상태 (정확한 조합만 허용)"
CONTROLLER_STATE_OK=0
if [ "$OVERLAY_OK" -ne 1 ]; then
    p_skip "오버레이를 소스하지 못해 건너뜀"
elif [ "$SKIP_ROS_CHECKS" = "1" ]; then
    p_skip "SKIP_ROS_CHECKS=1"
else
    # settle = 재시도 포함 읽기.  자동 재활성화(both-inactive -> source hold)는
    # 실기 --arm에서만 허용한다 -- no-arm preflight는 read-only 계약이다.
    ALLOW_SETTLE_ACTIVATE=0
    if [ "$ARM_REQUESTED" -eq 1 ] && [ "$FAKE_ENV" -eq 0 ]; then
        ALLOW_SETTLE_ACTIVATE=1
    fi
    if ! hil_settle_controller_states "$SOURCE_ARM_CONTROLLER" "$ARM_CONTROLLER" "$ALLOW_SETTLE_ACTIVATE"; then
        if [ "$FAKE_ENV" -eq 1 ]; then
            p_info "controller_manager 응답 없음 (fake-env에서는 무관)"
        else
            p_fail "controller_manager에서 두 controller의 정확한 상태를 못 읽었다" \
                   "UR 드라이버와 controller 이름/상태를 확인"
        fi
    else
        case "$HIL_SOURCE_STATE:$HIL_TARGET_STATE" in
            active:inactive)
                CONTROLLER_STATE_OK=1
                if [ "$ARM_REQUESTED" -eq 1 ]; then
                    p_ok "$SOURCE_ARM_CONTROLLER=active, $ARM_CONTROLLER=inactive — proof 확인 후 전환 가능한 상태"
                else
                    p_ok "$SOURCE_ARM_CONTROLLER=active, $ARM_CONTROLLER=inactive — no-arm이므로 전환하지 않음"
                fi
                ;;
            inactive:active)
                CONTROLLER_STATE_OK=1
                if [ "$ARM_REQUESTED" -eq 1 ]; then
                    p_ok "$SOURCE_ARM_CONTROLLER=inactive, $ARM_CONTROLLER=active — idempotent 후보([10] live RESET pose 확인 필요)"
                else
                    p_warn "$ARM_CONTROLLER=ACTIVE (no-arm은 상태를 바꾸지 않음)" \
                           "명령 퍼블리셔가 없어야 하며, 의도하지 않은 상태면 controller를 수동 복구"
                fi
                ;;
            *)
                if [ "$FAKE_ENV" -eq 1 ]; then
                    p_info "controller 조합이 실기형이 아님: $SOURCE_ARM_CONTROLLER=$HIL_SOURCE_STATE, $ARM_CONTROLLER=$HIL_TARGET_STATE"
                else
                    p_fail "예상 밖 controller 조합" \
                           "$SOURCE_ARM_CONTROLLER=$HIL_SOURCE_STATE, $ARM_CONTROLLER=$HIL_TARGET_STATE — 둘 다 active/inactive인 상태로 actor 금지"
                fi
                ;;
        esac
    fi
fi

_say ""
_say "[9] $COMMAND_TOPIC 이중 퍼블리셔 (있으면 거부)"
if [ "$OVERLAY_OK" -ne 1 ]; then
    p_skip "오버레이를 소스하지 못해 건너뜀"
elif [ "$SKIP_ROS_CHECKS" = "1" ]; then
    p_skip "SKIP_ROS_CHECKS=1"
else
    set +e
    TOPIC_OUT="$(timeout --signal=KILL 15 ros2 topic info "$COMMAND_TOPIC" 2>&1)"
    TOPIC_RC=$?
    set -e
    PUB_COUNT="$(printf '%s\n' "$TOPIC_OUT" | grep -m1 'Publisher count:' | awk '{print $NF}')"
    if [ "$TOPIC_RC" -ne 0 ] || [ -z "$PUB_COUNT" ]; then
        if [ "$FAKE_ENV" -eq 1 ]; then
            p_info "토픽 정보를 못 읽었다 (fake-env에서는 무관)"
        else
            p_fail "$COMMAND_TOPIC 의 퍼블리셔 수를 읽지 못했다" \
                   "ROS 그래프가 안 보인다 — 드라이버/ROS_DOMAIN_ID 확인"
        fi
    elif [ "$PUB_COUNT" -eq 0 ]; then
        p_ok "퍼블리셔 0개 — actor가 유일한 퍼블리셔가 된다"
    else
        p_fail "$COMMAND_TOPIC 에 이미 퍼블리셔가 $PUB_COUNT 개 있다" \
               "텔레옵 브리지(gello_ur_bridge)나 다른 러너가 살아 있다. 두 퍼블리셔가 같은 컨트롤러를 서로 다른 목표로 때리면 팔이 떨거나 튄다 — 그쪽을 먼저 끄고 다시 실행할 것. (확인: ros2 topic info -v $COMMAND_TOPIC)"
    fi
fi

# --dry-preflight --arm is a read-only rehearsal of the exact mutation gate.
# It validates marker + current reset pose but deliberately exits before the
# handoff function below.  Plain --dry-preflight/no-arm never needs a marker.
_say ""
_say "[10] controller handoff proof"
if [ "$ARM_REQUESTED" -ne 1 ]; then
    p_info "--arm 없음 — proof 불필요, controller switch 금지"
elif [ "$FAKE_ENV" -eq 1 ]; then
    p_skip "fake-env에서는 controller handoff 없음"
elif [ "$OVERLAY_OK" -ne 1 ] || [ "$CONTROLLER_STATE_OK" -ne 1 ]; then
    p_skip "선행 controller/overlay 점검 실패"
elif [ "$HIL_SOURCE_STATE" = inactive ] && [ "$HIL_TARGET_STATE" = active ]; then
    if hil_verify_live_reset_pose \
            "$POSE_CHECKER" "$RESET_JOINTS_CSV" "$ARM_POSE_TOL_RAD" "/joint_states"; then
        p_ok "$ARM_CONTROLLER가 이미 active — marker 없이 live RESET pose 검증(idempotent)"
    else
        p_fail "$ARM_CONTROLLER가 active지만 현재 팔이 RESET 자세가 아니다" \
               "leftover FPC 상태에서 actor 자동 reset을 허용하지 않는다; driver/preposition 절차로 복구"
    fi
elif [ "$DRY_PREFLIGHT" -eq 1 ]; then
    if hil_validate_preposition_marker \
            "$PREPOSITION_MARKER" "$RESET_JOINTS_CSV" \
            "$ARM_POSE_TOL_RAD" "$PREPOSITION_MARKER_MAX_AGE_S" && \
       hil_verify_live_reset_pose \
            "$POSE_CHECKER" "$RESET_JOINTS_CSV" "$ARM_POSE_TOL_RAD" "/joint_states"; then
        p_ok "preposition marker + 현재 RESET 자세 확인 (읽기 전용; switch하지 않음)"
    else
        p_fail "arm handoff proof 검증 실패" \
               "actor를 내린 상태에서 $SCRIPT_DIR/run_hil_preposition.sh를 실행"
    fi
else
    p_info "preflight가 전부 통과한 뒤 marker + live pose를 재검증하고 strict switch 예정"
fi

_say ""
_say "[11] deadman $STARTUP_DEADMAN_LABEL heartbeat gate (채널 생존 증명)"
if [ "$ARM_REQUESTED" -ne 1 ]; then
    p_info "--arm 없음 — deadman gate 불필요"
elif [ "$FAKE_ENV" -eq 1 ]; then
    p_skip "fake-env에서는 deadman gate 없음"
elif [ "$OVERLAY_OK" -ne 1 ]; then
    p_skip "ROS overlay 점검 실패"
elif python3 "$DEADMAN_CHECKER" --topic /hil/deadman --samples 3 --timeout 2.0 \
        --require "$STARTUP_DEADMAN"; then
    p_ok "연속 $STARTUP_DEADMAN_LABEL heartbeat 3개 확인 (GUI가 살아 있다)"
elif [ "$STARTUP_DEADMAN" = "disengaged" ]; then
    p_fail "deadman이 fresh DISENGAGED 상태가 아니다" \
           "HIL GUI가 떠서 20 Hz heartbeat를 내고 있어야 하고, 시작 시점에는 DISENGAGED여야 한다 (세션은 policy 제어로 시작한다 — ENGAGE는 개입할 때만). 옛 동작은 HIL_STARTUP_DEADMAN=engaged"
else
    p_fail "deadman이 fresh ENGAGED 상태가 아니다" \
           "HIL GUI에서 ENGAGE한 뒤 GELLO를 RESET anchor에 고정하고 다시 실행"
fi

# ---------------------------------------------------------------------------
# 결과 요약
# ---------------------------------------------------------------------------
_say ""
_say "============================================================"
if [ "$WARN_COUNT" -gt 0 ]; then
    _say " 경고 $WARN_COUNT 건:"
    for w in "${WARNINGS[@]}"; do _say "   - $w"; done
fi
if [ "$FAIL_COUNT" -gt 0 ]; then
    _say " ✗ preflight 실패 $FAIL_COUNT 건 — actor를 띄우지 않는다:"
    for f in "${FAILURES[@]}"; do _say "   - $f"; done
    _say "============================================================"
    exit 1
fi
_say " ✓ preflight 통과 (경고 $WARN_COUNT 건)"
_say "============================================================"

if [ "$DRY_PREFLIGHT" -eq 1 ]; then
    _say ""
    _say "--dry-preflight: actor를 기동하지 않고 종료한다."
    _say "실제 기동은 --dry-preflight 를 빼고 같은 명령을 다시 실행."
    exit 0
fi

# ---------------------------------------------------------------------------
# 실제 --arm controller handoff.  이 지점은 모든 preflight가 통과한 뒤이고,
# --dry-preflight는 이미 종료했다.  no-arm/fake는 절대 이 함수를 호출하지 않는다.
# 함수 자체도 marker + live reset pose + publisher 0을 다시 확인하며 reset 이동은
# 수행하지 않는다. 실패하면 actor를 exec하지 않는다.
# ---------------------------------------------------------------------------
if [ "$ARM_REQUESTED" -eq 1 ] && [ "$FAKE_ENV" -eq 0 ]; then
    _say ""
    _say "[ARM] controller handoff (여기서만 상태 변경 가능)"
    # 채널 생존 재검증. [11] 이후 GUI가 죽었거나 조작자가 상태를 바꿨을 수 있다.
    if ! python3 "$DEADMAN_CHECKER" \
            --topic /hil/deadman --samples 3 --timeout 2.0 \
            --require "$STARTUP_DEADMAN"; then
        _say " ✗ deadman $STARTUP_DEADMAN_LABEL 재검증 실패 — controller를 전환하지 않는다."
        exit 1
    fi
    if ! hil_arm_controller_handoff \
            "$SOURCE_ARM_CONTROLLER" "$ARM_CONTROLLER" "$COMMAND_TOPIC" \
            "$PREPOSITION_MARKER" "$POSE_CHECKER" "$RESET_JOINTS_CSV" \
            "$ARM_POSE_TOL_RAD" "$PREPOSITION_MARKER_MAX_AGE_S"; then
        _say " ✗ controller handoff 실패 — actor를 기동하지 않는다."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# actor 기동 — 항상 venv python 절대경로, 항상 절대경로 스크립트.
# 우리 기본 인자를 먼저 놓고 사용자 인자를 뒤에 붙인다(argparse는 뒤가 이김).
# no-arm/fake는 종전처럼 exec한다. 실기 --arm만 래퍼가 부모로 남아 actor의
# 모든 종료 경로(정상/Ctrl-C/RPC 예외) 뒤 controller ownership을 복구한다.
# ---------------------------------------------------------------------------
_say ""
_say "actor 기동 (Ctrl-C로 중단):"
_say "  $ACTOR_PY $ACTOR_SCRIPT --exp-name $EXP_NAME ... ${PASSTHRU[*]}"
_say ""

ACTOR_CMD=("$ACTOR_PY" "$ACTOR_SCRIPT" \
    --exp-name "$EXP_NAME" \
    --ur-config-module "$UR_CONFIG_MODULE" \
    --network-type grpc \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT" \
    --timeout-s "$TIMEOUT_S" \
    --max-response-age-s "$MAX_RESPONSE_AGE_S" \
    --observation-schema-hash "$OBS_SCHEMA_HASH" \
    --expected-model-id "$EXPECTED_MODEL_ID" \
    --expected-reward-authority "$EXPECTED_REWARD_AUTHORITY" \
    --expected-reward-model-id "$EXPECTED_REWARD_MODEL_ID" \
    "${PASSTHRU[@]}")

if [ "$ARM_REQUESTED" -ne 1 ] || [ "$FAKE_ENV" -eq 1 ]; then
    # stderr는 알려진 cv2/Qt 폰트 경고 두 줄만 걸러 내보낸다(위 함수 주석 참조).
    # 프로세스 치환은 exec 전에 fork되므로 exec 뒤에도 살아 있다.
    # `trap '' INT TERM`은 필터에만 적용된다 — 아래 실기 경로와 같은 이유다.
    exec "${ACTOR_CMD[@]}" 2> >(trap '' INT TERM; hil_filter_qt_font_noise >&2)
fi

# 이 아래는 실기 --arm 경로뿐이다.  75 승격의 전제가 되는 진행 증거 감시자를
# actor보다 **먼저** 띄운다(첫 transition status를 놓치지 않기 위해).
if [ "$ACTOR_EXIT_MAP" = "1" ]; then
    if ! hil_start_progress_watch; then
        _say "[ARM] 경고: 진행 증거 감시자를 띄우지 못했다 — rc는 75로 승격되지 않는다."
    fi
fi

ACTOR_PID=""
forward_actor_signal() {
    local signal_name="$1"
    if [[ -n "$ACTOR_PID" ]] && kill -0 "$ACTOR_PID" 2>/dev/null; then
        kill -s "$signal_name" "$ACTOR_PID" 2>/dev/null || true
    fi
}
trap 'forward_actor_signal INT' INT
trap 'forward_actor_signal TERM' TERM
trap 'forward_actor_signal HUP' HUP

# Put the actor in its own session so terminal Ctrl-C reaches the parent only;
# the trap below then forwards exactly one signal.  Without this separation the
# terminal and the parent both signal Python, and the second KeyboardInterrupt
# can interrupt env.close().  Non-interactive bash also starts asynchronous
# commands with SIGINT ignored, so restore the defaults before exec.
#
# stderr 필터는 subshell 바깥에 붙는다. $! 는 여전히 이 subshell의 PID이므로
# 신호 전달/wait/controller 복귀 로직은 영향이 없다(bash 5.1.16에서 확인).
#
# 진짜 함정은 $! 가 아니라 **필터에 신호가 전달되는 것**이었다: 프로세스 치환의
# grep은 이 래퍼의 **foreground process group**에 남는데 actor는 일부러
# setsid로 그 group 밖으로 내보냈다. 그래서 터미널 Ctrl-C가 group을 때리면
# grep이 **먼저** 죽고, 그 뒤 actor가 `finally: env.close()` / `network.close()`
# 로 풀리며 쓰는 stderr는 닫힌 파이프로 들어간다 → KeyboardInterrupt traceback과
# publisher/gRPC teardown 진단이 **통째로 사라지고** actor는 BrokenPipeError(32)를
# 본다. 즉 하필 Ctrl-C 종료 경로의 로그만 잃는다.
# `trap '' INT TERM`은 필터 자신에게만 적용되므로(actor에는 영향 없음) 필터가
# actor보다 오래 살아 마지막 stderr까지 흘려보낸다. 필터는 stdin EOF로 끝난다.
(
    trap - INT TERM HUP
    exec setsid "${ACTOR_CMD[@]}"
) 2> >(trap '' INT TERM; hil_filter_qt_font_noise >&2) &
ACTOR_PID=$!
set +e
# A trapped signal interrupts bash's wait before the child necessarily exits.
# Re-wait until it is really gone; controller cleanup must never race a still
# live command publisher or orphan the armed actor.
while true; do
    wait "$ACTOR_PID"
    ACTOR_RC=$?
    if ! kill -0 "$ACTOR_PID" 2>/dev/null; then
        break
    fi
done
set -e
ACTOR_PID=""
trap - INT TERM HUP

# 감시자는 증거를 잡는 즉시 스스로 끝난다. 아직 살아 있다면 증거가 없다는 뜻이다.
ACTOR_PROGRESS_SEEN=0
ACTOR_PROGRESS_DETAIL=""
hil_stop_progress_watch
if [ -n "$PROGRESS_MARKER" ] && [ -f "$PROGRESS_MARKER" ]; then
    ACTOR_PROGRESS_SEEN=1
elif [ -n "$PROGRESS_DIR" ] && [ -s "$PROGRESS_DIR/watch.log" ]; then
    ACTOR_PROGRESS_DETAIL="$(tail -n 2 "$PROGRESS_DIR/watch.log" | tr '\n' ' ')"
fi

_say ""
_say "[ARM] actor 종료(rc=$ACTOR_RC) — controller 자동 복귀"
RESTORE_RC=0
hil_restore_controller_after_actor \
        "$SOURCE_ARM_CONTROLLER" "$ARM_CONTROLLER" "$COMMAND_TOPIC" || RESTORE_RC=$?

if [ "$RESTORE_RC" -eq 2 ]; then
    # ros2_control 스택 자체가 없다 = 하드웨어 번들이 죽었다.
    # 이것이 70이 아닌 이유: 70은 "소유권을 모르겠다"이고 그건 controller_manager가
    # 살아 있는데 전환이 실패했을 때의 진단이다. 여기서는 controller_manager가
    # 아예 없으므로 **어떤 ros2_control 컨트롤러도 팔을 몰 수 없다** — 팔은 UR
    # 자체 제어로 자세를 유지한다. 그리고 우리 command publisher가 사라진 것은
    # hil_restore_controller_after_actor가 이 지점 **전에** 이미 확인했다.
    # 번들을 다시 켜면 STJC가 active로 올라온다(run_hil_hardware.sh) — 즉 이것은
    # 재기동으로 고쳐지는 유일한 실패이고, 정확히 복구 루프가 존재하는 이유다.
    _say " !! ros2_control 스택이 사라졌다 = 하드웨어 번들이 죽었다."
    _say "    복귀할 대상이 없고, 팔을 몰 수 있는 것도 없다(우리 publisher는 이미 0)."
    _say "    -> 종료 코드 $RECOVERABLE_EXIT_CODE (RECOVERABLE 후보)."
    _say "    Terminal 2에서 ./run_hil_hardware.sh 를 다시 실행하면 세션이 이어진다."
    _say "    재시도는 PID 세대 교체 + 토픽 READY + robot RUNNING/safety NORMAL을"
    _say "    모두 확인한 뒤에만, 모든 proof를 처음부터 다시 통과해야 일어난다."
    # 진행 증거(env_step>=0)를 요구하지 않는다. 그 게이트는 "결정론적 startup
    # 실패를 반복 재시도하지 마라"는 뜻인데, 결정론적 startup 실패는
    # controller_manager를 사라지게 만들지 않는다. 스택 소멸은 그것과 독립적인,
    # 그리고 더 직접적인 환경 실패의 증거다.
    exit "$RECOVERABLE_EXIT_CODE"
fi
if [ "$RESTORE_RC" -ne 0 ]; then
    _say " ✗ controller 자동 복귀 실패. 펜던트를 들고 아래 상태를 직접 확인하라:" >&2
    _say "   ros2 control list_controllers" >&2
    _say "   ros2 topic info -v $COMMAND_TOPIC" >&2
    _say "   (종료 코드 70 = 재시도 금지. 파일 상단 '종료 코드 계약' 참조)" >&2
    exit 70
fi

# ---------------------------------------------------------------------------
# 종료 코드 계약 (파일 상단 참조).  여기까지 왔다는 것은:
#   * actor가 실제로 기동됐고(모든 preflight + handoff PASS),
#   * 어떤 이유로든 종료했으며,
#   * controller 복귀가 PASS했다 = trajectory controller가 팔을 잡고 있고
#     $COMMAND_TOPIC publisher는 0이다.
#   * 그리고 actor가 **transition 루프까지 실제로 도달했다**(진행 증거).
# 이 조합만 "재시도를 고려해도 되는 상태"로 승격한다(75).  신호로 죽은 경우
# (>=128, Ctrl-C 포함)는 사람의 의도이므로 그대로 전파한다.
# 진행 증거가 없으면 결정론적 startup 실패로 보고 raw rc를 그대로 낸다 — 파일
# 상단 "진행 증거" 항목이 근거다. (그 rc가 1이면 preflight FAIL과 값이 겹치지만
# 두 경우의 처방은 같다: 자동 재시도 금지, 사람이 로그를 본다.)
# ---------------------------------------------------------------------------
if [ "$ACTOR_EXIT_MAP" = "1" ] && [ "$ACTOR_RC" -gt 0 ] && [ "$ACTOR_RC" -lt 128 ]; then
    if [ "$ACTOR_PROGRESS_SEEN" -eq 1 ]; then
        _say "[ARM] actor rc=$ACTOR_RC + controller 복귀 PASS + transition 진행 증거 확인"
        _say "      ($ACTOR_STATUS_TOPIC 에서 env_step>=0 status를 관측했다)"
        _say "      -> 종료 코드 $RECOVERABLE_EXIT_CODE (RECOVERABLE 후보)."
        _say "      재시도는 하드웨어 번들 재기동이 확인된 뒤에만, 모든 proof를 처음부터"
        _say "      다시 통과해야 일어난다. 재시도 판단은 run_hil_session.sh가 한다."
        exit "$RECOVERABLE_EXIT_CODE"
    fi
    _say "[ARM] actor rc=$ACTOR_RC + controller 복귀 PASS, 그러나 **transition 진행 증거가 없다**."
    _say "      $ACTOR_STATUS_TOPIC 에서 env_step>=0 status를 한 번도 보지 못했다"
    _say "      = actor가 첫 transition을 ack받기 전에 죽었다(인자/핸드셰이크/schema"
    _say "      /첫 transition 검증 등). 이런 실패는 결정론적이라 재시도하면 같은"
    _say "      지점에서 다시 죽고 RESET 이동만 한 번 더 실행된다."
    if [ -n "$ACTOR_PROGRESS_DETAIL" ]; then
        _say "      (감시자 진단: $ACTOR_PROGRESS_DETAIL)"
    fi
    _say "      -> 75로 승격하지 않고 rc=$ACTOR_RC 를 그대로 낸다 (자동 재시도 없음)."
fi

exit "$ACTOR_RC"
