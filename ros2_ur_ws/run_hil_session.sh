#!/usr/bin/env bash
# =============================================================================
# run_hil_session.sh — laptop3 HIL-SERL 운용을 한 터미널에서 소유하는 래퍼
# =============================================================================
#
# 전제: UR7e/GELLO/gripper ROS topic과 localhost:50153 학습 서버 터널은 이미 준비돼 있다.
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
#   * 세션은 **policy 제어로 시작한다.** 시작할 때 ENGAGE를 누를 필요가 없다 —
#     ENGAGE는 나중에 개입할 때만 누른다. GUI를 띄우기만 하면 된다.
#   * controller handoff 전 deadman gate는 그대로 있다. 다만 요구 상태가
#     **DISENGAGED**다: 연속 fresh heartbeat 3개로 GUI가 살아서 /hil/deadman을
#     퍼블리시하고 있다는 **채널 생존**을 증명하되, 팔은 GELLO를 잡아도 따라오지 않는
#     상태에서 넘긴다. 옛 ENGAGED 요구는 HIL_STARTUP_DEADMAN=engaged로 되돌린다.
#   * actor가 HOME에서 WAIT_SCENE_READY를 표시하면 장면을 배치하고 GUI의
#     START/NEXT ITERATION을 누른다. 그 버튼이 policy episode를 시작한다
#     (버튼은 deadman이 DISENGAGED일 때만 받아들여진다).
#   * 이 스크립트를 Ctrl-C 하거나 세션이 끝나면 이 스크립트가 띄운 HIL GUI와 camera
#     launcher도 정리한다.
#
# 하드웨어 재기동 자동 복구 (3~5단계만 재시도):
#   충돌 → PROTECTIVE_STOP → actor 사망은 이 리그의 일상이고, 펜던트에서 오류를
#   지우는 것만으로는 부족해서 조작자가 Terminal 2의 run_hil_hardware.sh를 껐다 켠다.
#   예전에는 그때 camera·GUI·actor·gRPC 세션을 한꺼번에 잃었다. 이제는:
#     1(cameras)·2(GUI)단계는 그대로 살아 있고 3~5단계만 다시 돈다.
#   재시도는 다음이 **전부** 참일 때만 일어난다:
#     * actor wrapper가 정확히 **rc 75**로 끝났다 (run_hil_actor.sh의 "종료 코드
#       계약" = actor가 떴다가 죽었고 controller 복귀는 PASS. 그 외 코드는 전부
#       재시도 금지 — 특히 preflight 실패(1), controller 복귀 실패(70), 신호(>=128)),
#     * 하드웨어 번들 세 entrypoint(ur_control.launch.py / robotiq_gripper_modbus /
#       gello_publisher)의 **PID 세대가 완전히 교체**됐다 = 진짜 재기동했다,
#     * 그 새 번들의 topic이 run_hil_hardware.sh의 READY 기준으로 흐르고,
#     * **로봇 자신이 robotmode=RUNNING / safetystatus=NORMAL 이다.** topic 세 개는
#       전부 읽기라 protective stop 중에도 흐른다 — 그것만 보고 재arming하면
#       움직이지 않는 로봇에 명령을 흘리게 된다 (hil_robot_state_ready 주석 참조).
#   재시도는 resume이 **아니다**: preposition(RESET 이동 + 새 marker) → read-only
#   preflight → 실제 handoff 순으로 marker 신선도·live RESET pose·publisher 0·
#   strict switch·사후검증·deadman 채널 gate를 **처음부터 전부 다시** 통과한다.
#   조작자가 칠 키는 없다 — 감지되면 취소 가능한 카운트다운 뒤 자동으로 진행한다.
#   끄기/조이기: HIL_ACTOR_RETRY=0, HIL_ACTOR_RETRY_MAX(기본 3, 최대 10),
#   HIL_HARDWARE_RECYCLE_WAIT_S(기본 900), HIL_RETRY_RESUME_DELAY_S(기본 5).
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
TOPIC_CHECKER="$SCRIPT_DIR/_hil_topic_rate_check.py"
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

The localhost:50153 learner-host tunnel and UR7e/GELLO/gripper topics must already be
ready. This wrapper owns --arm, --dry-preflight, --fake-env, and --deadman;
all other arguments are passed to run_hil_actor.sh.

Environment:
  HIL_STARTUP_DEADMAN=disengaged|engaged
        Deadman state required before the controller handoff. Default
        'disengaged': the session starts under POLICY control and ENGAGE is
        only for intervention. The proof itself (3 consecutive fresh
        heartbeats = the deadman channel is alive) is identical either way.
        Set 'engaged' to restore the old commissioning behaviour.
  DEADMAN_WAIT_S (alias: ENGAGE_WAIT_S)
        How long to wait for that state before continuing anyway (default 120;
        run_hil_actor.sh preflight [11] re-checks and fails).
  HIL_ACTOR_RETRY=1|0            (default 1)
        After a collision/PROTECTIVE_STOP the actor dies but the cameras, the
        GUI and the learner-host tunnel are still fine. With 1, stages 3-5 (preposition,
        preflight, actor) re-run once the operator has restarted
        run_hil_hardware.sh; stages 1-2 are never torn down. Only run_hil_actor.sh
        exit code 75 (RECOVERABLE) is eligible -- see its --help. Every arming
        proof runs again from scratch; nothing is resumed or cached.
  HIL_ACTOR_RETRY_MAX            (default 3, max 10)
  HIL_HARDWARE_RECYCLE_WAIT_S    (default 900) wait for the operator's restart
  HIL_RETRY_RESUME_DELAY_S       (default 5)   cancellable countdown before
                                               stage 3 moves the arm again
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

# actor handoff 전에 deadman이 어느 상태여야 하는가.  기본은 DISENGAGED —
# 세션은 policy 제어로 시작하고 ENGAGE는 개입 전용이다.  gate 자체(연속 fresh
# heartbeat 3개 = 채널 생존 증명)는 두 값에서 동일하다.  오타를 "gate 없음"으로
# 조용히 해석하지 않도록 fail-closed로 검사하고, run_hil_actor.sh가 같은 값을
# 쓰도록 export 한다.
STARTUP_DEADMAN="${HIL_STARTUP_DEADMAN:-disengaged}"
case "$STARTUP_DEADMAN" in
    disengaged|engaged) ;;
    *)
        echo "FATAL: HIL_STARTUP_DEADMAN은 'disengaged' 또는 'engaged'여야 한다 (받은 값 '$STARTUP_DEADMAN')." >&2
        exit 2
        ;;
esac
export HIL_STARTUP_DEADMAN="$STARTUP_DEADMAN"
STARTUP_DEADMAN_LABEL="${STARTUP_DEADMAN^^}"

# --- 하드웨어 재기동 자동 복구 루프 설정 -------------------------------------
# 충돌/protective stop 뒤 조작자는 Terminal 2의 run_hil_hardware.sh를 껐다 켠다.
# 그때 camera(1단계)와 GUI(2단계)까지 같이 잃을 이유는 없다 — 3~5단계만 다시 돈다.
# 상세 계약은 아래 hil_wait_for_hardware_recycle() 위 주석에 있다.
RETRY_ENABLED="${HIL_ACTOR_RETRY:-1}"
RETRY_MAX="${HIL_ACTOR_RETRY_MAX:-3}"
RECYCLE_WAIT_S="${HIL_HARDWARE_RECYCLE_WAIT_S:-900}"
RESUME_DELAY_S="${HIL_RETRY_RESUME_DELAY_S:-5}"
# run_hil_actor.sh의 "종료 코드 계약"에서 재시도를 고려할 수 있는 **유일한** 코드.
RECOVERABLE_ACTOR_RC=75
case "$RETRY_ENABLED" in
    0|1) ;;
    *)
        echo "FATAL: HIL_ACTOR_RETRY는 0 또는 1이어야 한다 (받은 값 '$RETRY_ENABLED')." >&2
        exit 2
        ;;
esac
for _name in RETRY_MAX RECYCLE_WAIT_S RESUME_DELAY_S; do
    if [[ ! "${!_name}" =~ ^[0-9]+$ ]]; then
        echo "FATAL: $_name은 음이 아닌 정수여야 한다 (받은 값 '${!_name}')." >&2
        exit 2
    fi
done
if (( RETRY_MAX > 10 )); then
    echo "FATAL: HIL_ACTOR_RETRY_MAX는 10 이하여야 한다 (실기 팔을 자동 재arming한다)." >&2
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
echo "   startup deadman : $STARTUP_DEADMAN_LABEL (기본 DISENGAGED = policy로 시작)"
if ((NO_ARM == 0)); then
    if ((RETRY_ENABLED)); then
        echo "   hw recovery : ON (rc=75 + 번들 PID 세대교체 확인 시 3~5단계만 최대 ${RETRY_MAX}회 재시도)"
    else
        echo "   hw recovery : OFF (HIL_ACTOR_RETRY=0)"
    fi
fi
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

# ---------------------------------------------------------------------------
# 하드웨어 번들 재기동 감지 (읽기 전용)
# ---------------------------------------------------------------------------
# run_hil_hardware.sh가 소유하는 세 entrypoint를 /proc에서 그대로 읽는다. 토큰
# 목록은 그 스크립트의 existing_owners 검사와 같은 것이다 — 여기서 다른 정의를
# 만들면 두 스크립트가 서로 다른 "번들"을 뜻하게 된다.
# 출력: "<token> <pid>" 한 줄씩. 읽기 전용이고 어떤 프로세스도 건드리지 않는다.
hil_hardware_owner_lines() {
    python3 - <<'PY' 2>/dev/null || true
from pathlib import Path

TOKENS = {"ur_control.launch.py", "robotiq_gripper_modbus", "gello_publisher"}
found = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = [
            item.decode("utf-8", "replace")
            for item in (proc / "cmdline").read_bytes().split(b"\0")
            if item
        ]
    except OSError:
        continue
    for item in argv:
        if Path(item).name in TOKENS:
            found.append((Path(item).name, int(proc.name)))
            break
for token, pid in sorted(found):
    print(token, pid)
PY
}

hil_owner_token_count() {
    printf '%s\n' "$1" | awk 'NF { print $1 }' | sort -u | grep -c . || true
}

hil_owner_pid_list() {
    printf '%s\n' "$1" | awk 'NF { print $2 }' | sort -u
}

# 새 번들이 실제로 topic을 흘리고 있는지 — run_hil_hardware.sh의 READY 조건과
# 같은 세 probe를 같은 helper로 돌린다(같은 type/min-rate). ROS overlay는 이
# 서브셸 안에서만 source해서 세션 셸의 환경을 오염시키지 않는다.
hil_hardware_topics_ready() {
    local ros_setup="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
    local ws_setup="$SCRIPT_DIR/install/setup.bash"
    [[ -f "$ros_setup" && -f "$ws_setup" && -f "$TOPIC_CHECKER" ]] || return 1
    (
        set +eu
        # shellcheck disable=SC1090
        source "$ros_setup" >/dev/null 2>&1
        # shellcheck disable=SC1090
        source "$ws_setup" >/dev/null 2>&1
        rc=0
        while read -r topic type min_rate; do
            [ -n "$topic" ] || continue
            timeout --signal=TERM --kill-after=2 16 \
                python3 "$TOPIC_CHECKER" --topic "$topic" --type "$type" \
                --samples 5 --timeout 12 --min-rate "$min_rate" >/dev/null 2>&1
            if [ $? -ne 0 ]; then
                rc=1
                break
            fi
        done <<EOF
/joint_states joint_state 50
/robotiq_gripper/position_percent float32 2
/gello/joint_states joint_state 15
EOF
        exit "$rc"
    )
}

# 로봇이 **명령을 받을 수 있는 상태**인지 (읽기 전용).
# ---------------------------------------------------------------------------
# 왜 topic READY만으로는 부족한가: 위 세 probe(/joint_states,
# /robotiq_gripper/position_percent, /gello/joint_states)는 전부 RTDE/시리얼
# **읽기**다. PROTECTIVE_STOP, 브레이크 체결, non-REMOTE 상태에서도 그대로
# 흐른다. 그리고 팔이 마침 RESET 0.10 rad 안에 있으면 run_hil_preposition.sh는
# 이동 없이 `verified_existing` 분기로 marker를 쓰고, preflight가 통과하고,
# STJC->FPC strict switch도 성공한다(controller_manager는 안전 상태를 모른다).
# 그러면 actor가 arming되어 **움직이지 않는 로봇에 명령을 흘리고**, 조작자가
# protective stop을 푸는 순간에야 그 사실이 드러난다.
#
# 왜 dashboard **서비스**인가 (00_SETUP_AND_SAFETY.md §5.2의 nc 한 줄짜리 대신):
#   * §5.2가 말하는 바로 그 두 값(robotmode / safetystatus)을 그대로 준다.
#     `/io_and_status_controller/robot_program_running`(Bool) 하나로는 왜 false
#     인지(전원? 브레이크? protective stop? 프로그램 재전송 중?) 구분이 안 되고
#     조작자에게 보여 줄 문구가 없다.
#   * 드라이버가 이미 열어 둔 **하나뿐인** dashboard(29999) 연결을 재사용한다.
#     nc로 두 번째 클라이언트를 붙이는 것은 그 포트의 단일 클라이언트 성격과
#     드라이버 자신의 재접속/resend 경로를 건드릴 수 있다. ROBOT_IP를 이
#     스크립트가 새로 알 필요도 없다.
#   * 덤으로 **새 번들의 dashboard_client 노드가 살아서 답한다**는 것까지
#     증명된다 — 이 게이트의 목적(진짜 재기동 확인)과 정확히 같은 방향이다.
#     (ur_control.launch.py는 launch_dashboard_client 기본 true로 이 노드를 띄운다.)
# 실패는 전부 fail-closed다: 서비스가 없거나 timeout이거나 파싱이 안 되면
# "준비 안 됨"으로 보고 계속 기다린다 → 증거 없이 재arming하지 않는다.
# 성공하면 한 줄 요약을 stdout으로 낸다(실패해도 읽은 만큼은 낸다).
hil_robot_state_ready() {
    local ros_setup="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
    local ws_setup="$SCRIPT_DIR/install/setup.bash"
    [[ -f "$ros_setup" && -f "$ws_setup" ]] || {
        echo "dashboard probe 불가 (ROS overlay 없음)"
        return 1
    }
    (
        set +eu
        # shellcheck disable=SC1090
        source "$ros_setup" >/dev/null 2>&1
        # shellcheck disable=SC1090
        source "$ws_setup" >/dev/null 2>&1
        command -v ros2 >/dev/null 2>&1 || {
            echo "dashboard probe 불가 (ros2 CLI 없음)"
            exit 1
        }
        # RobotMode.mode=7 == RUNNING, SafetyMode.mode=1 == NORMAL
        # (/opt/ros/humble/share/ur_dashboard_msgs/msg/{RobotMode,SafetyMode}.msg).
        # 숫자 enum으로 판정한다 — dashboard의 answer 문자열은 펌웨어 판에 따라
        # 문구가 바뀔 수 있지만 enum은 메시지 정의가 보장한다.
        probe() {  # $1=service $2=type $3=expected repr fragment
            local out
            out="$(timeout --signal=TERM --kill-after=2 20 \
                ros2 service call "$1" "$2" '{}' 2>&1)" || {
                printf '%s=응답없음' "$1"
                return 1
            }
            printf '%s\n' "$out" | grep -q 'success=True' || {
                printf '%s=%s' "$1" \
                    "$(printf '%s\n' "$out" | sed -n "s/.*answer='\([^']*\)'.*/\1/p" | tail -1)"
                return 1
            }
            printf '%s\n' "$out" | grep -qF "$3" || {
                printf '%s=%s' "$1" \
                    "$(printf '%s\n' "$out" | sed -n "s/.*answer='\([^']*\)'.*/\1/p" | tail -1)"
                return 1
            }
            return 0
        }
        probe /dashboard_client/get_robot_mode \
              ur_dashboard_msgs/srv/GetRobotMode \
              'RobotMode(mode=7)' || exit 1
        probe /dashboard_client/get_safety_mode \
              ur_dashboard_msgs/srv/GetSafetyMode \
              'SafetyMode(mode=1)' || exit 1
        echo "robotmode=RUNNING safetystatus=NORMAL"
        exit 0
    )
}

# 조작자가 out-of-band로 번들을 껐다 켠 것을 **증거로** 확인한다:
#   (a) 세 entrypoint가 모두 다시 살아 있고,
#   (b) 그 PID들이 이전 세대와 **하나도 겹치지 않으며**(= 진짜 재기동),
#   (c) 세 topic이 run_hil_hardware.sh의 READY 기준으로 흐르고,
#   (d) 로봇 자신이 robotmode=RUNNING / safetystatus=NORMAL 이다
#       (= protective stop / 브레이크 / non-REMOTE 가 아니다).
# (b)가 핵심이다: 폴링이 down 구간을 놓쳐도 PID 교체는 놓칠 수 없으므로,
# "번들이 안 죽었는데 protective stop만 걸린" 상태로 다시 arming하는 일이 없다.
# 성공하면 새 세대 목록을 HIL_NEW_OWNER_LINES에 남긴다.
HIL_NEW_OWNER_LINES=""
hil_wait_for_hardware_recycle() {
    local baseline="$1"
    local deadline=$(( SECONDS + RECYCLE_WAIT_S ))
    local next_note=0 current tokens overlap remaining robot_state last_robot_note=""
    HIL_NEW_OWNER_LINES=""
    while true; do
        current="$(hil_hardware_owner_lines)"
        tokens="$(hil_owner_token_count "$current")"
        overlap="$(comm -12 \
            <(hil_owner_pid_list "$baseline") \
            <(hil_owner_pid_list "$current") | grep -c . || true)"
        if (( tokens == 3 && overlap == 0 )); then
            echo "  .. 새 하드웨어 번들 감지 (PID 세대 교체 확인) — topic READY 확인 중 ..."
            if hil_hardware_topics_ready; then
                # topic이 흘러도 로봇은 protective stop일 수 있다 (위 함수 주석).
                if robot_state="$(hil_robot_state_ready)"; then
                    echo "  .. 로봇 상태 확인: $robot_state"
                    HIL_NEW_OWNER_LINES="$current"
                    return 0
                fi
                if [[ "$robot_state" != "$last_robot_note" ]]; then
                    echo "  !! 로봇이 아직 명령을 받을 수 있는 상태가 아니다 — ${robot_state:-dashboard 조회 실패}"
                    echo "     펜던트에서 protective stop/오류를 클리어하고 REMOTE·전원·브레이크를 확인하라."
                    echo "     (기대값: robotmode RUNNING / safetystatus NORMAL — 00_SETUP_AND_SAFETY.md §5.2)"
                    last_robot_note="$robot_state"
                fi
            else
                echo "  .. 프로세스는 떴지만 topic이 아직 READY 기준에 못 미친다 — 계속 기다린다."
            fi
        fi
        if (( SECONDS >= deadline )); then
            echo "  !! ${RECYCLE_WAIT_S}s 안에 하드웨어 번들 재기동을 확인하지 못했다."
            return 1
        fi
        if (( SECONDS >= next_note )); then
            remaining=$(( deadline - SECONDS ))
            echo "  .. 대기 중 (entrypoint ${tokens}/3, 이전 세대와 겹치는 PID ${overlap}개, 남은 시간 ${remaining}s, Ctrl-C로 종료)"
            next_note=$(( SECONDS + 30 ))
        fi
        sleep 2
    done
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

# ---------------------------------------------------------------------------
# 3~5단계 재시도 루프.  1(camera)·2(GUI)단계는 이 아래에서 무슨 일이 있어도
# 살아남는다 — EXIT trap과 CAMERA_PID/GUI_PID는 actor가 죽어도 건드리지 않는다.
#
# 재시도 조건은 **전부** 만족해야 한다:
#   * actor wrapper 종료 코드가 정확히 75 (RECOVERABLE; run_hil_actor.sh의
#     "종료 코드 계약" — actor가 떴다가 죽었고, controller 복귀 PASS이며,
#     transition 루프까지 실제로 도달했다는 진행 증거가 있다),
#   * 재시도가 켜져 있고 남은 횟수가 있고,
#   * 하드웨어 번들이 **PID 세대 교체로 증명된** 재기동을 했고 topic이 READY이며
#     로봇이 robotmode=RUNNING / safetystatus=NORMAL 이다.
# 그 뒤 3단계부터 다시 도는데, 이것은 resume이 아니라 **새로운 arming**이다:
# preposition(RESET 이동 + 새 marker), read-only preflight, 그리고 실제 handoff가
# marker 신선도 / live RESET pose / publisher 0 / strict switch + 사후검증 /
# deadman 채널 gate를 전부 처음부터 다시 통과한다. 이 루프는 그 어떤 proof도
# 건너뛰거나 캐시하지 않는다.
BASELINE_OWNER_LINES="$(hil_hardware_owner_lines)"
BASELINE_TOKENS="$(hil_owner_token_count "$BASELINE_OWNER_LINES")"
if (( RETRY_ENABLED == 1 && BASELINE_TOKENS != 3 )); then
    RETRY_ENABLED=0
    echo ""
    echo "[session] 자동 재시도 비활성: 지금 run_hil_hardware.sh 번들의 세 entrypoint를"
    echo "          모두 관측하지 못했다 (${BASELINE_TOKENS}/3). 기준 세대를 모르면 '재기동됨'을"
    echo "          증명할 수 없고, 증명 없는 재arming은 하지 않는다."
fi

ATTEMPT=1
while true; do

ATTEMPT_TAG=""
if (( RETRY_ENABLED == 1 )); then
    ATTEMPT_TAG=" (arming 시도 $ATTEMPT/$((RETRY_MAX + 1)))"
fi

echo ""
echo "[3/5] UR7e preposition$ATTEMPT_TAG"
cat <<'EOF'
현재 자세가 RESET 허용범위 밖이면 run_hil_preposition.sh가 안전 확인 뒤
기본값에서 별도 입력 없이 JTC 이동을 시작한다. preposition 자체에는 0.9 rad 최대
거리 gate가 없으므로 위 current/target 표를 확인한다. 이동은 충돌 회피 없는 관절
궤적이며 시작 뒤 실제 정지는 E-STOP이다. 대기/GO 입력은 각각
PREPOSITION_DELAY_S=N / PREPOSITION_CONFIRM=1로만 켠다.
EOF
"${PREPOSITION_CMD[@]}"

echo ""
echo "[operator] actor startup deadman gate — 요구 상태: $STARTUP_DEADMAN_LABEL"
if [[ "$STARTUP_DEADMAN" == "disengaged" ]]; then
    cat <<'EOF'
시작할 때 ENGAGE를 누르지 마라. 이 세션은 policy 제어로 시작한다.
ENGAGE는 나중에 **개입할 때만** 누른다 (GELLO로 직접 몰 때).
지금 필요한 것은 HIL GUI가 떠서 heartbeat를 내고 있는 것뿐이다 — read-only
preflight와 실제 handoff가 각각 fresh DISENGAGED heartbeat 3개로 deadman
채널이 살아 있다는 것을 증명한다 (증명은 그대로, 요구 상태만 바뀌었다).
controller handoff 뒤 actor는 HOME에서 WAIT_SCENE_READY로 멈춘다. 장면을
배치한 뒤 GUI의 START / NEXT ITERATION 버튼을 누르면 policy가 첫 action부터
제어한다. 실행 중 ENGAGE하면 GELLO 개입으로 전환된다.
EOF
else
    cat <<'EOF'
HIL_STARTUP_DEADMAN=engaged (옛 동작). HIL GUI에서 ENGAGE를 두 번 눌러
ENGAGED로 만들고 GELLO를 움직이지 말고 고정하라. read-only preflight와 실제
handoff가 각각 fresh ENGAGED heartbeat 3개를 검증한다. controller handoff 뒤
actor는 HOME에서 WAIT_SCENE_READY로 멈춘다. 장면을 배치한 뒤 GUI의
START / NEXT ITERATION 버튼을 누르면 deadman이 자동 DISENGAGE되고 policy가
첫 action부터 제어한다.
EOF
fi
echo ""
# 예전에는 여기서 Enter를 받았다. 그 프롬프트는 안전장치가 아니라 알림이었다 —
# 실제 강제는 preflight [11]과 [ARM] 직전 재검증(각각 fresh heartbeat 3개,
# run_hil_actor.sh)이고 그 둘은 그대로다. 여기서는 요구 상태가 만족될 때까지
# 폴링한다: 기본 DISENGAGED에서는 GUI가 뜨는 즉시 지나가고(GUI는 DISENGAGED로
# 시작해 20 Hz heartbeat를 계속 낸다), 조작자가 GELLO를 잡으려고 ENGAGE를 눌러
# 둔 상태라면 여기서 알려 준다 (예전에는 preflight [11]에서 FAIL 나고 세션을
# 다시 시작해야 했다).
DEADMAN_WAIT_S="${DEADMAN_WAIT_S:-${ENGAGE_WAIT_S:-120}}"
if [[ ! "$DEADMAN_WAIT_S" =~ ^[0-9]+$ ]]; then
    echo "FATAL: DEADMAN_WAIT_S/ENGAGE_WAIT_S는 음이 아닌 정수여야 한다." >&2
    exit 2
fi
if [[ -x "$(command -v python3)" && -f "$SCRIPT_DIR/_hil_deadman_check.py" ]]; then
    _deadline=$(( SECONDS + DEADMAN_WAIT_S ))
    _notified=0
    until python3 "$SCRIPT_DIR/_hil_deadman_check.py" \
              --topic /hil/deadman --samples 3 --timeout 2.0 \
              --require "$STARTUP_DEADMAN" >/dev/null 2>&1; do
        if (( SECONDS >= _deadline )); then
            echo "  !! ${DEADMAN_WAIT_S}s 안에 $STARTUP_DEADMAN_LABEL heartbeat를 못 봤다 — 그대로 진행한다."
            echo "     preflight [11]이 같은 조건을 다시 검사하고 실패시킨다."
            break
        fi
        if (( _notified == 0 )); then
            if [[ "$STARTUP_DEADMAN" == "disengaged" ]]; then
                echo "  .. GUI heartbeat(DISENGAGED) 대기 중 — GUI가 ENGAGED면 DISENGAGE하라 (최대 ${DEADMAN_WAIT_S}s, Ctrl-C로 중단)"
            else
                echo "  .. GUI에서 ENGAGE를 누르면 자동으로 진행한다 (대기 최대 ${DEADMAN_WAIT_S}s, Ctrl-C로 중단)"
            fi
            _notified=1
        fi
        sleep 1
    done
    (( _notified == 1 )) && echo "  $STARTUP_DEADMAN_LABEL 확인 — 계속한다."
else
    echo "  (deadman checker 없음 — 확인 없이 진행한다)"
fi

echo ""
echo "[4/5] actor armed-readiness preflight (읽기 전용)$ATTEMPT_TAG"
"${PREFLIGHT_CMD[@]}"

echo ""
echo "[5/5] 실제 actor 기동$ATTEMPT_TAG"
echo "      이제 같은 preflight를 재검증한 뒤에만 controller를 handoff한다."
echo "      handoff 뒤 GUI의 WAIT_SCENE_READY에서 장면을 배치하고 START/NEXT를 누른다."
echo "      Ctrl-C 시 controller 복귀 후 cameras/GUI까지 정리하고 종료한다."
if (( RETRY_ENABLED == 1 )); then
    echo "      충돌/protective stop으로 actor가 죽으면 cameras/GUI는 그대로 두고"
    echo "      하드웨어 번들 재기동을 기다렸다가 3단계부터 다시 arming한다."
fi
set +e
"${ACTOR_CMD[@]}"
actor_rc=$?
set -e
echo ""
echo "[session] actor 종료 (rc=$actor_rc)"

# --- 재시도 판단 ------------------------------------------------------------
# 기본은 "재시도하지 않는다". 아래 세 조건을 전부 통과할 때만 루프가 한 번 더 돈다.
if (( actor_rc != RECOVERABLE_ACTOR_RC )); then
    if (( actor_rc != 0 )); then
        echo "[session] rc=$actor_rc 는 RECOVERABLE(75)이 아니다 — 자동 재시도하지 않는다."
        echo "          (rc 의미: run_hil_actor.sh --help 의 '종료 코드 계약')"
    fi
    exit "$actor_rc"
fi
if (( RETRY_ENABLED == 0 )); then
    echo "[session] 자동 재시도가 꺼져 있다 (HIL_ACTOR_RETRY=0) — 종료한다."
    exit "$actor_rc"
fi
if (( ATTEMPT > RETRY_MAX )); then
    echo "[session] 재시도 한도 소진 ($RETRY_MAX회) — 종료한다."
    echo "          같은 실패가 반복되고 있다. 로그를 보고 원인을 먼저 해결하라."
    exit "$actor_rc"
fi

cat <<EOF

============================================================
 actor가 죽었지만 controller 복귀는 PASS했다 (rc=75).
 팔은 trajectory controller가 잡고 있고 command publisher는 0이다.
 cameras / HIL GUI / 학습 서버 터널은 **그대로 살아 있다** — 다시 만들지 마라.

 지금 할 일 (Terminal 2):
   1) 펜던트에서 protective stop / 오류를 클리어한다.
   2) Ctrl-C 로 run_hil_hardware.sh를 멈추고 "[cleanup] complete"를 기다린다.
   3) ./run_hil_hardware.sh 를 다시 실행한다.

 이 터미널은 새 번들을 **자동으로 감지**한다 (세 entrypoint의 PID 세대가
 완전히 교체되고, topic이 READY가 되고, 로봇이 robotmode=RUNNING /
 safetystatus=NORMAL 로 답할 때까지 — protective stop을 안 풀면 여기서 멈춰
 서서 무엇이 막고 있는지 출력한다). 감지되면 3단계부터 다시 arming하고,
 그때 marker/RESET pose/publisher 0/strict switch/deadman gate를 전부 다시
 통과한다. 재시도하지 않으려면 지금 Ctrl-C 하면 된다.
 최대 대기 ${RECYCLE_WAIT_S}s.
============================================================
EOF
if ! hil_wait_for_hardware_recycle "$BASELINE_OWNER_LINES"; then
    echo "[session] 하드웨어 재기동을 확인하지 못했다 — 재arming 없이 종료한다."
    exit "$actor_rc"
fi
BASELINE_OWNER_LINES="$HIL_NEW_OWNER_LINES"
echo "[session] 새 하드웨어 번들 READY."
if (( RESUME_DELAY_S > 0 )); then
    echo "          ${RESUME_DELAY_S}s 뒤 3단계(preposition, 팔이 RESET으로 이동)부터 다시 시작한다."
    echo "          지금 Ctrl-C 하면 취소된다."
    _left="$RESUME_DELAY_S"
    while (( _left > 0 )); do
        printf '          .. %ds\r' "$_left"
        sleep 1
        _left=$(( _left - 1 ))
    done
    printf '                              \r'
fi
ATTEMPT=$(( ATTEMPT + 1 ))
done
