#!/usr/bin/env bash
# =============================================================================
# run_hil_session.sh — laptop3 HIL-SERL 운용을 한 터미널에서 소유하는 래퍼
# =============================================================================
#
# 전제: UR7e/GELLO/gripper ROS topic과 localhost:50153 Kanu 터널은 이미 준비돼 있다.
# 이 스크립트는 검증된 기존 도구를 순서대로 실행할 뿐, 그 내부 안전 계약을 우회하지
# 않는다.
#
# 기본 실행 순서:
#   1. launch_cameras.sh (cam1 scene + cam2 wrist, 기본 viewer 포함)
#   2. run_hil_gui.sh (/hil/deadman GUI)
#   3. run_hil_preposition.sh
#      - RESET 0.10 rad 밖이면 기본값은 체크리스트 뒤 즉시 JTC 이동이다.
#      - PREPOSITION_DELAY_S 또는 PREPOSITION_CONFIRM로 대기/GO 입력을 opt-in한다.
#   4. run_hil_actor.sh --dry-preflight --arm --deadman topic (읽기 전용)
#   5. run_hil_actor.sh --arm --deadman topic (실제 controller handoff + actor)
#
# 중요:
#   * actor handoff 전 GUI를 ENGAGED로 둬야 한다. 스크립트가 이를 자동으로 누르지 않는다.
#   * controller handoff까지는 ENGAGED gate를 유지한다. actor가 HOME에서
#     WAIT_SCENE_READY를 표시하면 GUI의 START/NEXT ITERATION을 누른다. 버튼이
#     deadman을 DISENGAGE하고 fresh observation으로 policy episode를 시작한다.
#   * actor가 정상 종료하거나 RPC 오류로 죽거나 이 스크립트를 Ctrl-C 하면, 이 스크립트가
#     띄운 HIL GUI와 camera launcher도 정리한다.
#
# 사용법:
#   ./run_hil_session.sh
#   ./run_hil_session.sh --classifier-sidecar-interval 1
#   VIEW=false ./run_hil_session.sh              # camera viewer만 생략
#   ./run_hil_session.sh --no-arm                 # 카메라/GUI + 읽기 전용 점검 후 종료
#   ./run_hil_session.sh --no-arm --plan          # 아무 process도 띄우지 않고 계획만 출력
#
# `--arm`, `--dry-preflight`, `--fake-env`, `--deadman`은 이 래퍼가 소유한다.
# 그 밖의 인자는 run_hil_actor.sh에 그대로 전달한다.
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_SCRIPT="$SCRIPT_DIR/launch_cameras.sh"
GUI_SCRIPT="$SCRIPT_DIR/run_hil_gui.sh"
PREPOSITION_SCRIPT="$SCRIPT_DIR/run_hil_preposition.sh"
ACTOR_SCRIPT="$SCRIPT_DIR/run_hil_actor.sh"
CAMERA_READY_TIMEOUT_S="${HIL_CAMERA_READY_TIMEOUT_S:-90}"

NO_ARM=0
PLAN_ONLY=0
ACTOR_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_hil_session.sh [ACTOR_ARGS...]
  ./run_hil_session.sh --no-arm [ACTOR_ARGS...]
  ./run_hil_session.sh --no-arm --plan [ACTOR_ARGS...]

Modes:
  default    cameras + HIL GUI + preposition + read-only armed
             preflight + actual --arm actor
  --no-arm   cameras + HIL GUI + read-only no-arm preflight, then cleanup/exit
  --plan     print the selected commands without starting any process

Examples:
  ./run_hil_session.sh
  ./run_hil_session.sh --classifier-sidecar-interval 1
  VIEW=false ./run_hil_session.sh
  ./run_hil_session.sh --no-arm

The localhost:50153 Kanu tunnel and UR7e/GELLO/gripper topics must already be
ready. This wrapper owns --arm, --dry-preflight, --fake-env, and --deadman;
all other arguments are passed to run_hil_actor.sh.
EOF
}

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --no-arm)
            NO_ARM=1
            shift
            ;;
        --plan)
            PLAN_ONLY=1
            shift
            ;;
        --)
            shift
            ;;
        --arm|--arm=*|--dry-preflight|--dry-preflight=*|--fake-env|--fake-env=*|--deadman|--deadman=*)
            echo "FATAL: $1 은 run_hil_session.sh가 소유하는 옵션이다." >&2
            echo "       기본 실기는 자동으로 --arm --deadman topic을 사용한다." >&2
            exit 2
            ;;
        *)
            ACTOR_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! "$CAMERA_READY_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: HIL_CAMERA_READY_TIMEOUT_S는 양의 정수여야 한다." >&2
    exit 2
fi

for required in \
    "$CAMERA_SCRIPT" "$GUI_SCRIPT" "$PREPOSITION_SCRIPT" "$ACTOR_SCRIPT"; do
    if [[ ! -x "$required" ]]; then
        echo "FATAL: 실행할 수 없는 필수 스크립트: $required" >&2
        exit 1
    fi
done

PREFLIGHT_CMD=("$ACTOR_SCRIPT" --dry-preflight)
ACTOR_CMD=("$ACTOR_SCRIPT")
PREPOSITION_CMD=(env SWITCH_TO_FPC=0 DRY_RUN=0 "$PREPOSITION_SCRIPT")
if ((NO_ARM == 0)); then
    PREFLIGHT_CMD+=(--arm)
    ACTOR_CMD+=(--arm)
fi
PREFLIGHT_CMD+=("${ACTOR_ARGS[@]}" --deadman topic)
ACTOR_CMD+=("${ACTOR_ARGS[@]}" --deadman topic)

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

echo "============================================================"
echo " HIL-SERL laptop3 session"
if ((NO_ARM)); then
    echo "   mode : NO-ARM VALIDATION (robot 이동/switch/actor 기동 없음)"
else
    echo "   mode : REAL ROBOT (--arm)"
fi
echo "   camera viewer : ${VIEW:-true}"
echo "============================================================"

if ((NO_ARM)); then
    STAGE_TOTAL=3
else
    STAGE_TOTAL=5
fi

if ((PLAN_ONLY)); then
    echo ""
    echo "[plan] cameras"
    print_command "$CAMERA_SCRIPT"
    echo "[plan] HIL deadman GUI"
    print_command "$GUI_SCRIPT"
    if ((NO_ARM)); then
        echo "[plan] preposition 생략 (no-arm은 robot pose를 바꾸지 않음)"
        echo "[plan] read-only no-arm preflight 후 종료"
        print_command "${PREFLIGHT_CMD[@]}"
    else
        echo "[plan] preposition (기본 즉시 진행; optional delay/GO는 환경변수)"
        print_command "${PREPOSITION_CMD[@]}"
        echo "[plan] read-only armed-readiness preflight"
        print_command "${PREFLIGHT_CMD[@]}"
        echo "[plan] actual actor"
        print_command "${ACTOR_CMD[@]}"
    fi
    exit 0
fi

CAMERA_PID=""
GUI_PID=""
CLEANED=0
SESSION_TMP="$(mktemp -d /tmp/hil_session_XXXXXX)"
CAMERA_LOG="$SESSION_TMP/cameras.log"
GUI_LOG="$SESSION_TMP/hil_gui.log"
: >"$CAMERA_LOG"
: >"$GUI_LOG"

group_has_live_processes() {
    local pgid="$1"
    ps -eo pgid=,stat= | awk -v wanted="$pgid" '
        $1 == wanted && $2 !~ /^Z/ { found=1 }
        END { exit !found }
    '
}

stop_child_group() {
    local pid="$1" name="$2" grace_s="$3"
    [[ -n "$pid" ]] || return 0
    # camera/GUI는 각각 setsid로 띄워 pid==pgid인 독립 process group이다.
    # launcher shell이 먼저 죽어도 같은 group의 ROS grandchild까지 소유해서 정리한다.
    if group_has_live_processes "$pid"; then
        echo "[cleanup] $name 종료 중 (pid $pid) ..."
        kill -TERM -- "-$pid" 2>/dev/null || true
        local waited=0
        while group_has_live_processes "$pid" && ((waited < grace_s)); do
            sleep 1
            waited=$((waited + 1))
        done
        if group_has_live_processes "$pid"; then
            echo "[cleanup] $name가 종료되지 않아 process group SIGKILL (pgid $pid)"
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
    fi
    wait "$pid" 2>/dev/null || true
}

verify_session_leader() {
    local pid="$1" name="$2" actual_pgid=""
    for _ in $(seq 1 20); do
        actual_pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
        [[ "$actual_pgid" == "$pid" ]] && return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.05
    done
    echo "FATAL: $name 독립 process group 생성 실패 (pid=$pid, pgid=${actual_pgid:-none})." >&2
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    return 1
}

cleanup() {
    local rc=$?
    ((CLEANED == 0)) || return "$rc"
    CLEANED=1
    trap - EXIT INT TERM HUP
    # 출력 대상이 먼저 닫힌 경우에도 kill/wait는 끝까지 수행한다.
    set +e
    echo ""
    echo "[session] child process 정리"
    stop_child_group "$GUI_PID" "HIL GUI" 5
    # launch_cameras.sh가 viewer + 두 camera node를 자체적으로 순서대로 정리한다.
    # 독립 process group kill은 그 자체 cleanup이 실패해도 grandchild를 남기지 않는다.
    stop_child_group "$CAMERA_PID" "camera launcher" 20
    echo "[session] logs: $CAMERA_LOG | $GUI_LOG"
    return "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

echo ""
echo "[1/$STAGE_TOTAL] RealSense pair 기동"
if [[ "${VIEW:-true}" == "false" || "${VIEW:-true}" == "0" ]]; then
    echo "      VIEW=false: camera viewer를 띄우지 않는다."
else
    echo "      cam1=SCENE, cam2=WRIST를 viewer에서 직접 확인하고 창을 닫지 마라."
fi
echo "      READY 대기 중 (log: $CAMERA_LOG)"
setsid "$CAMERA_SCRIPT" >"$CAMERA_LOG" 2>&1 &
CAMERA_PID=$!
verify_session_leader "$CAMERA_PID" "camera launcher"

deadline=$((SECONDS + CAMERA_READY_TIMEOUT_S))
while ! grep -q '^### READY' "$CAMERA_LOG"; do
    if ! kill -0 "$CAMERA_PID" 2>/dev/null; then
        set +e
        wait "$CAMERA_PID"
        camera_rc=$?
        set -e
        CAMERA_PID=""
        ((camera_rc != 0)) || camera_rc=1
        echo "FATAL: launch_cameras.sh가 READY 전에 종료했다 (rc=$camera_rc)." >&2
        echo "       log: $CAMERA_LOG" >&2
        exit "$camera_rc"
    fi
    if ((SECONDS >= deadline)); then
        echo "FATAL: camera READY를 ${CAMERA_READY_TIMEOUT_S}s 안에 확인하지 못했다." >&2
        echo "       log: $CAMERA_LOG" >&2
        exit 1
    fi
    sleep 0.5
done
echo "      camera launcher READY"

echo ""
echo "[2/$STAGE_TOTAL] HIL deadman GUI 기동"
setsid "$GUI_SCRIPT" >"$GUI_LOG" 2>&1 &
GUI_PID=$!
verify_session_leader "$GUI_PID" "HIL GUI"
sleep 1
if ! kill -0 "$GUI_PID" 2>/dev/null; then
    set +e
    wait "$GUI_PID"
    gui_rc=$?
    set -e
    GUI_PID=""
    ((gui_rc != 0)) || gui_rc=1
    echo "FATAL: run_hil_gui.sh가 시작 직후 종료했다 (rc=$gui_rc)." >&2
    tail -n 30 "$GUI_LOG" >&2 || true
    exit "$gui_rc"
fi

if ((NO_ARM)); then
    echo ""
    echo "[3/3] NO-ARM read-only preflight"
    echo "      preposition/controller switch/실제 actor는 실행하지 않는다."
    "${PREFLIGHT_CMD[@]}"
    echo ""
    echo "PASS: no-arm session validation 완료. camera와 GUI를 정리하고 종료한다."
    exit 0
fi

echo ""
echo "[3/5] UR7e preposition"
cat <<'EOF'
현재 자세가 RESET 허용범위 밖이면 run_hil_preposition.sh가 안전 확인 뒤
기본값에서 별도 입력 없이 JTC 이동을 시작한다. preposition 자체에는 0.9 rad 최대
거리 gate가 없으므로 위 current/target 표를 확인한다. 이동은 충돌 회피 없는 관절
궤적이며 시작 뒤 실제 정지는 E-STOP이다. 대기/GO 입력은 각각
PREPOSITION_DELAY_S=N / PREPOSITION_CONFIRM=1로만 켠다.
EOF
"${PREPOSITION_CMD[@]}"

echo ""
echo "[operator] actor startup ENGAGED gate"
cat <<'EOF'
HIL GUI에서 ENGAGE를 두 번 눌러 ENGAGED로 만들고 GELLO를 움직이지 말고 고정하라.
read-only preflight와 실제 handoff가 각각 fresh ENGAGED heartbeat 3개를 검증한다.
controller handoff 뒤 actor는 HOME에서 WAIT_SCENE_READY로 멈춘다. 장면을 배치한 뒤
GUI의 START / NEXT ITERATION 버튼을 누르면 deadman이 자동 DISENGAGE되고 policy가
첫 action부터 제어한다. 실행 중 다시 ENGAGE하면 GELLO 개입으로 전환된다.
EOF
echo ""
# 예전에는 여기서 Enter를 받았다. 그 프롬프트는 안전장치가 아니라 알림이었다 —
# 실제 강제는 preflight [11]과 [ARM] 직전 재검증(각각 fresh ENGAGED heartbeat
# 3개, run_hil_actor.sh)이고 그 둘은 그대로다. 매 세션 키를 치는 대신 ENGAGED가
# 될 때까지 폴링한다: 이미 눌러 뒀으면 즉시 지나가고, 안 눌렀으면 여기서 기다린다
# (예전에는 preflight [11]에서 FAIL 나고 세션을 다시 시작해야 했다).
ENGAGE_WAIT_S="${ENGAGE_WAIT_S:-120}"
if [[ -x "$(command -v python3)" && -f "$SCRIPT_DIR/_hil_deadman_check.py" ]]; then
    _deadline=$(( SECONDS + ENGAGE_WAIT_S ))
    _notified=0
    until python3 "$SCRIPT_DIR/_hil_deadman_check.py" \
              --topic /hil/deadman --samples 3 --timeout 2.0 >/dev/null 2>&1; do
        if (( SECONDS >= _deadline )); then
            echo "  !! ${ENGAGE_WAIT_S}s 안에 ENGAGED가 되지 않았다 — 그대로 진행한다."
            echo "     preflight [11]이 같은 조건을 다시 검사하고 실패시킨다."
            break
        fi
        if (( _notified == 0 )); then
            echo "  .. GUI에서 ENGAGE를 누르면 자동으로 진행한다 (대기 최대 ${ENGAGE_WAIT_S}s, Ctrl-C로 중단)"
            _notified=1
        fi
        sleep 1
    done
    (( _notified == 1 )) && echo "  ENGAGED 확인 — 계속한다."
else
    echo "  (deadman checker 없음 — 확인 없이 진행한다)"
fi

echo ""
echo "[4/5] actor armed-readiness preflight (읽기 전용)"
"${PREFLIGHT_CMD[@]}"

echo ""
echo "[5/5] 실제 actor 기동"
echo "      이제 같은 preflight를 재검증한 뒤에만 controller를 handoff한다."
echo "      handoff 뒤 GUI의 WAIT_SCENE_READY에서 장면을 배치하고 START/NEXT를 누른다."
echo "      Ctrl-C 또는 actor/RPC 종료 시 controller 복귀 후 cameras/GUI도 정리한다."
set +e
"${ACTOR_CMD[@]}"
actor_rc=$?
set -e
echo ""
echo "[session] actor 종료 (rc=$actor_rc)"
exit "$actor_rc"
