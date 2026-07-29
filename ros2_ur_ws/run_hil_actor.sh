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
#   * forward_position_controller가 active면 경고한다. active 상태에서 실수로
#     한 번이라도 발행되면 팔이 즉시 그 자세로 간다.
#   * 태스크 config의 DRY_RUN=True가 기본이다(cube_in_cup). DRY_RUN 해제는
#     이 스크립트가 아니라 config/러너 인자로만 이루어져야 한다. 이 래퍼는
#     로봇에 아무것도 발행하지 않으며, --dry-preflight면 actor조차 안 띄운다.
#   * 사전 점검은 전부 읽기 전용이다(ros2 topic hz / topic info /
#     control list_controllers / TCP connect). 명령 발행은 하지 않는다.
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
#   ./run_hil_actor.sh --dry-preflight          # 점검만, actor 미기동 (안전)
#   ./run_hil_actor.sh --fake-env               # Stage A: fake env로 Kanu 왕복
#   ./run_hil_actor.sh                          # Stage B: 실센서 actor
#   ./run_hil_actor.sh --save-video --actor-id foo   # 인자는 그대로 통과
#
#   환경변수로 기본값 덮어쓰기:
#     ACTOR_VENV, SERVER_HOST, SERVER_PORT, EXP_NAME, UR_CONFIG_MODULE,
#     TIMEOUT_S, MAX_RESPONSE_AGE_S, OBS_SCHEMA_HASH, EXPECTED_MODEL_ID,
#     EXPECTED_REWARD_AUTHORITY, EXPECTED_REWARD_MODEL_ID, ROS_SETUP,
#     HZ_TIMEOUT_S, SKIP_ROS_CHECKS=1
#
# NOTE: `set -e`만 쓴다. `set -u`는 쓰지 않는다 — ROS의 setup.bash가 -u에서
#       죽는다(이 저장소에서 이미 두 스크립트가 같은 버그로 깨졌다가 고쳐졌다).
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
HZ_TIMEOUT_S="${HZ_TIMEOUT_S:-6}"

# ---------------------------------------------------------------------------
# 인자 파싱: --dry-preflight / --help 만 소비하고 나머지는 전부 통과시킨다
# ---------------------------------------------------------------------------
DRY_PREFLIGHT=0
FAKE_ENV=0
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
_say "============================================================"

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
_say "[8] $ARM_CONTROLLER 상태 (active면 경고)"
if [ "$OVERLAY_OK" -ne 1 ]; then
    p_skip "오버레이를 소스하지 못해 건너뜀"
elif [ "$SKIP_ROS_CHECKS" = "1" ]; then
    p_skip "SKIP_ROS_CHECKS=1"
else
    set +e
    CTRL_OUT="$(timeout --signal=KILL 15 ros2 control list_controllers 2>&1)"
    CTRL_RC=$?
    set -e
    if [ "$CTRL_RC" -ne 0 ] || [ -z "$CTRL_OUT" ]; then
        if [ "$FAKE_ENV" -eq 1 ]; then
            p_info "controller_manager 응답 없음 (fake-env에서는 무관)"
        else
            p_fail "controller_manager에서 컨트롤러 목록을 못 읽었다" \
                   "UR 드라이버(ur_control.launch.py)가 떠 있는지 확인"
        fi
    else
        CTRL_LINE="$(printf '%s\n' "$CTRL_OUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -m1 "^$ARM_CONTROLLER " || true)"
        CTRL_STATE="$(printf '%s' "$CTRL_LINE" | awk '{print $NF}')"
        case "$CTRL_STATE" in
            inactive)
                p_ok "$ARM_CONTROLLER = inactive (안전한 기본 상태)"
                ;;
            active)
                p_warn "$ARM_CONTROLLER = ACTIVE" \
                       "이 상태에서 명령이 한 줄이라도 나가면 팔이 즉시 움직인다. 의도한 것이 아니면: ros2 control switch_controllers --deactivate $ARM_CONTROLLER"
                ;;
            "")
                p_warn "$ARM_CONTROLLER 를 컨트롤러 목록에서 못 찾았다" \
                       "드라이버 launch 인자를 확인 (initial_joint_controller 등)"
                ;;
            *)
                p_warn "$ARM_CONTROLLER = $CTRL_STATE (예상 밖 상태)" ""
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
