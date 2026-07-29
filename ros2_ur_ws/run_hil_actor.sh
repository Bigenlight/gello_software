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
#   * --arm 없는 actor와 --dry-preflight는 controller를 절대 전환하지 않는다.
#   * 태스크 config의 DRY_RUN=True가 기본이다(cube_in_cup). DRY_RUN 해제는
#     --arm을 actor에게 넘길 때만 일어난다. 이 래퍼는 관절 command를 발행하거나
#     reset 이동을 호출하지 않는다. 다만 실제 --arm에서는 검증 뒤 controller만 전환한다.
#   * [1]~[10] 사전 점검은 전부 읽기 전용이다(ros2 topic hz / topic info /
#     control list_controllers / TCP connect). 실제 switch는 그 뒤 [ARM] 한 곳뿐이다.
#
# ── 중단 방법 ────────────────────────────────────────────────────────────────
#   * 이 터미널에서 Ctrl-C. actor는 exec로 이 셸을 대체하므로 Ctrl-C가 곧바로
#     actor에게 간다(중간 래퍼 프로세스 없음).
#   * 팔이 이미 움직이는 중이라면 먼저 펜던트 E-STOP. actor가 죽으면
#     forward_position_controller는 마지막 명령을 유지하므로 팔은 제자리 홀드
#     (튀지 않는다).
#   * GELLO 개입 중이면 HIL GUI에서 DISENGAGE.
#
# ── 사용법 ───────────────────────────────────────────────────────────────────
#   ./run_hil_actor.sh --dry-preflight          # 점검만, actor 미기동 (읽기 전용)
#   ./run_hil_actor.sh --dry-preflight --arm    # arm handoff 준비까지 읽기 전용 검증
#   ./run_hil_actor.sh --fake-env               # Stage A: fake env로 Kanu 왕복
#   ./run_hil_actor.sh                          # Stage B: 실센서 actor
#   ./run_hil_actor.sh --arm --deadman topic    # proof 확인 + FPC 전환 + 실제 actor
#   ./run_hil_actor.sh --save-video --actor-id foo   # 인자는 그대로 통과
#
#   환경변수로 기본값 덮어쓰기:
#     ACTOR_VENV, SERVER_HOST, SERVER_PORT, EXP_NAME, UR_CONFIG_MODULE,
#     TIMEOUT_S, MAX_RESPONSE_AGE_S, OBS_SCHEMA_HASH, EXPECTED_MODEL_ID,
#     EXPECTED_REWARD_AUTHORITY, EXPECTED_REWARD_MODEL_ID, ROS_SETUP,
#     HZ_TIMEOUT_S, HIL_PREPOSITION_MARKER, HIL_PREPOSITION_MARKER_MAX_AGE_S
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
if [ ! -f "$HANDOFF_LIB" ]; then
    echo "FATAL: controller handoff library missing: $HANDOFF_LIB" >&2
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
SERVER_PORT="${SERVER_PORT:-50153}"          # 로컬 터널 입구 → Kanu 50053
EXP_NAME="${EXP_NAME:-cube_in_cup}"
UR_CONFIG_MODULE="${UR_CONFIG_MODULE:-ur_experiments.mappings}"
TIMEOUT_S="${TIMEOUT_S:-0.6}"
MAX_RESPONSE_AGE_S="${MAX_RESPONSE_AGE_S:-0.8}"
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
HZ_TIMEOUT_S="${HZ_TIMEOUT_S:-6}"
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
               "SSH 터널을 먼저 띄운다: ssh -N -T -o ExitOnForwardFailure=yes -L 127.0.0.1:$SERVER_PORT:127.0.0.1:50053 kanu (그리고 Kanu에서 receive server가 떠 있어야 한다)"
    fi
fi

# ---------------------------------------------------------------------------
# 7~9. ROS 실데이터 / 컨트롤러 / 이중 퍼블리셔 (읽기 전용)
# ---------------------------------------------------------------------------
# fake-env는 ROS를 전혀 쓰지 않으므로 실데이터 점검은 정보성으로만 낸다.
ros_hz_check() {
    # $1=topic  $2=사람이 읽을 이름  $3=1이면 치명(FAIL) / 0이면 경고(WARN)
    local topic="$1" label="$2" fatal="$3"
    local out rate
    set +e
    out="$(timeout --signal=KILL "$HZ_TIMEOUT_S" ros2 topic hz --window 5 "$topic" 2>&1)"
    set -e
    rate="$(printf '%s\n' "$out" | grep -m1 'average rate' | awk '{print $3}')"
    if [ -n "$rate" ]; then
        p_ok "$label ($topic) ≈ ${rate} Hz"
        return 0
    fi
    if [ "$fatal" -eq 1 ]; then
        p_fail "$label ($topic) 에서 ${HZ_TIMEOUT_S}s 동안 메시지가 없다" \
               "해당 노드를 먼저 띄운다 (드라이버 / gello_publisher / launch_cameras.sh)"
    else
        p_warn "$label ($topic) 에서 ${HZ_TIMEOUT_S}s 동안 메시지가 없다" \
               "fake-env에서는 쓰이지 않으므로 진행 가능"
    fi
    return 1
}

_say ""
_say "[7] ROS 실데이터 (읽기 전용 ros2 topic hz, 각 ${HZ_TIMEOUT_S}s)"
if [ "$OVERLAY_OK" -ne 1 ]; then
    p_skip "오버레이를 소스하지 못해 건너뜀"
elif [ "$SKIP_ROS_CHECKS" = "1" ]; then
    p_skip "SKIP_ROS_CHECKS=1"
elif [ "$FAKE_ENV" -eq 1 ]; then
    p_info "fake-env 모드 — 센서 스트림은 사용되지 않는다 (경고로만 표시)"
    ros_hz_check "$JOINT_STATES_TOPIC" "로봇 관절"   0 || true
    ros_hz_check "$GELLO_TOPIC"        "GELLO 리더" 0 || true
    ros_hz_check "$CAM1_TOPIC"         "cam1 장면"  0 || true
    ros_hz_check "$CAM2_TOPIC"         "cam2 손목"  0 || true
else
    ros_hz_check "$JOINT_STATES_TOPIC" "로봇 관절"   1 || true
    ros_hz_check "$GELLO_TOPIC"        "GELLO 리더" 1 || true
    ros_hz_check "$CAM1_TOPIC"         "cam1 장면"  1 || true
    ros_hz_check "$CAM2_TOPIC"         "cam2 손목"  1 || true
    # 그리퍼 위치는 19-D state의 마지막 채널이다. 없으면 관측이 불완전하지만
    # DRY_RUN 배선 확인은 가능하므로 경고로만 낸다.
    ros_hz_check "$GRIPPER_STATE_TOPIC" "그리퍼 상태" 0 || true
fi

_say ""
_say "[8] arm controller 쌍 상태 (정확한 조합만 허용)"
CONTROLLER_STATE_OK=0
if [ "$OVERLAY_OK" -ne 1 ]; then
    p_skip "오버레이를 소스하지 못해 건너뜀"
elif [ "$SKIP_ROS_CHECKS" = "1" ]; then
    p_skip "SKIP_ROS_CHECKS=1"
else
    if ! hil_read_controller_states "$SOURCE_ARM_CONTROLLER" "$ARM_CONTROLLER"; then
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
# exec 이므로 Ctrl-C가 곧바로 actor에게 간다.
# ---------------------------------------------------------------------------
_say ""
_say "actor 기동 (Ctrl-C로 중단):"
_say "  $ACTOR_PY $ACTOR_SCRIPT --exp-name $EXP_NAME ... ${PASSTHRU[*]}"
_say ""

exec "$ACTOR_PY" "$ACTOR_SCRIPT" \
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
    "${PASSTHRU[@]}"
