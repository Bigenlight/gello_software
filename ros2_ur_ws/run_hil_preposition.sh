#!/usr/bin/env bash
# =============================================================================
# run_hil_preposition.sh — HIL-SERL 세션 전 UR7e를 RESET_JOINTS로 사전 배치
# =============================================================================
#
# 무엇을 하는가
# -------------
# HIL-SERL 액터의 ur7e_env.go_to_reset() 은 RESET_MAX_DIST_RAD(=0.9 rad) 게이트로
# 먼 거리 리셋을 거부한다(의도된 안전 동작 — 리셋은 250 Hz 업샘플러로 경로를 모른 채
# 스트리밍으로 쓸고 지나가기 때문). 그래서 매 세션 시작 전에 팔을 RESET_JOINTS
# 근처(0.9 rad 이내)로 가져다 놔야 한다. 이 도구의 PASS 허용치는 더 좁은 0.10 rad다.
#
# 지금까지는 펜던트 Freedrive로 손으로 옮겼지만, 이 스크립트는 이미 검증된
# `gello_move_to_start` 노드의 init_align 모드를 재사용해서 그 일을 재현 가능하게
# 자동화한다:
#
#   1) 사전 점검(읽기 전용): 컨트롤러 상태 / GELLO 스트림 / 충돌 노드 확인
#   2) RESET 밖이면 체크리스트 출력 뒤 노드를 띄운다. 기본은 즉시 진행이며,
#      PREPOSITION_DELAY_S / PREPOSITION_CONFIRM로 대기 또는 GO 입력을 opt-in한다
#   3) gello_move_to_start(start_mode:=init_align, init_pose:=RESET_JOINTS) 기동
#   4) GATE 1(~/proceed) 승인 → 팔이 scaled_joint_trajectory_controller 로
#      RESET_JOINTS 까지 **시간 파라미터화된 부드러운 궤적**으로 이동
#   5) 도착 후 ~/abort 를 호출해 **컨트롤러 전환 없이** 노드를 종료
#   6) 최종 자세를 branch-cut 안전 거리로 재측정하여 PASS/FAIL 판정
#   7) PASS한 경우에만 짧은 수명의 proof marker를 남긴다. run_hil_actor.sh
#      --arm은 marker와 현재 자세를 다시 확인한 뒤 controller를 전환한다.
#
# 왜 gello_move_to_start 인가 (직접 궤적을 쏘지 않는 이유)
# --------------------------------------------------------
#   * scaled_joint_trajectory_controller 를 쓰므로 **보간**된다(속도 제한 준수,
#     protective stop 위험이 낮다). forward_position_controller 는 보간하지 않는다.
#   * init_pose 를 보낼 때 `angle_utils.wrapped_nearest(init_pose, actual)` 로
#     **branch-cut(±pi 경계)을 처리**한다. RESET_JOINTS 는 shoulder_pan 이 +pi를
#     ~0.003 rad, wrist_3 가 -pi를 ~0.008 rad 넘어간 값이라, 이 처리를 안 하면
#     wrist_3 가 한 바퀴(≈2π) 도는 사고가 난다. 이 노드는 그 처리를 한다.
#     go_to_reset()도 현재 같은 branch-cut 처리를 하지만, 부드러운 사전 배치는
#     충돌 회피 없는 250 Hz reset stream 대신 JTC의 시간 파라미터 궤적을 쓴다.
#   * 노드의 서비스 gate를 사용하지만 wrapper가 기본값에서 자동 승인한다.
#
# 안전 확인 (실행 전에 반드시)
# ----------------------------
#   [ ] 팔 주변, RESET_JOINTS 까지의 **경로 위**에 사람/물체가 없다.
#       ★ 이 이동은 관절 공간 직선 보간이다. 충돌 회피를 하지 않는다. ★
#   [ ] 펜던트를 손에 들고 있고 비상정지 버튼에 손이 닿는다.
#   [ ] 로봇 전원 ON + REMOTE 모드(headless) 이고 scaled_joint_trajectory_controller
#       가 active 다(스크립트가 자동 점검한다).
#   [ ] gello_ur_bridge 가 떠 있지 않다(이중 퍼블리셔 방지 — 자동 점검한다).
#   [ ] HIL 액터(run_real_hil.py 등)가 ARM 상태로 스트리밍 중이 아니다.
#   [ ] 그리퍼에 물체가 물려 있으면 미리 정리한다.
#
# 중단 방법 (위험도 순서)
# -----------------------
#   1) 펜던트 **비상정지(E-STOP)** — 궤적 실행 중 유일하게 확실한 즉시 정지.
#   2) 펜던트 protective-stop / 프로그램 정지.
#   3) 이 스크립트에서 Ctrl-C — ★주의★ ROS 노드는 죽지만 **이미 수락된 궤적은
#      컨트롤러 쪽에서 계속 실행된다.** 즉 Ctrl-C 는 이동 중 정지 수단이 아니다.
#      이동 시작 전(승인 프롬프트 단계)에는 안전하게 취소된다.
#   4) 다른 터미널에서:
#        ros2 service call /gello_move_to_start/abort std_srvs/srv/Trigger "{}"
#      → 이동이 끝난 뒤 게이트에서 대기 중일 때 안전하게 빠져나온다
#        (컨트롤러 전환 없음).
#
# PASS 판정
# ---------
#   PASS : 모든 관절의 branch-cut-safe 거리 ≤ PASS_TOL_RAD(기본 0.10 rad).
#          이 상태면 go_to_reset() 의 0.9 rad 게이트를 여유 있게 통과한다.
#   FAIL : 위를 만족하지 못함. 스크립트가 관절별 오차를 출력한다.
#          → 대개 원인은 (a) 궤적이 중간에 abort 됨(속도/protective stop),
#            (b) stjc 가 inactive 로 떨어짐, (c) 조작자가 중간에 중단.
#          FAIL 이어도 컨트롤러 전환은 하지 않으므로 팔은 그 자리에 멈춰 있다.
#
# 이 스크립트가 하지 않는 것
# --------------------------
#   * 컨트롤러를 forward_position_controller 로 바꾸지 않는다(기본값).
#     proof를 받은 run_hil_actor.sh --arm이 publisher/controller 상태를 다시 확인한
#     뒤 strict switch한다. 수동 ros2 control switch는 proof 계약을 우회하므로 쓰지 않는다.
#   * GELLO 리더를 건드리지 않는다(GELLO 는 끝까지 수동/passive, 읽기만 한다).
#
# 그리퍼 (2026-07-30 변경 — 이전 판은 "그리퍼를 건드리지 않는다"였다)
# ------------------------------------------------------------------
# 이동이 끝난 뒤 그리퍼를 OPEN 한다. 모든 offline demo가 열린 그리퍼에서 시작하므로
# 닫힌 채 세션을 시작하면 정책이 한 번도 행동하기 전에 이미 out-of-distribution이고,
# gripper_position 은 state[0] 이다. 이동 "뒤"에 여는 이유는 물체를 물고 있었을 때
# 낙하 지점을 임의의 중간 자세가 아니라 항상 같은 RESET 자세로 고정하기 위해서다.
# 그리퍼 노드가 없으면 조용히 건너뛴다. OPEN_GRIPPER=0 으로 끌 수 있다.
# (같은 이유로 UR7eEnv.reset() 도 매 에피소드 경계에서 연다 — 이건 세션 시작용이다.)
#
# 사용법
# ------
#   ./run_hil_preposition.sh                 # 기본(궤적 8초, proof만 생성)
#   PREPOSITION_DELAY_S=5 ./run_hil_preposition.sh  # 이동 전 Ctrl-C 가능한 5초 창
#   PREPOSITION_CONFIRM=1 ./run_hil_preposition.sh  # 예전 GO 입력을 다시 요구
#   TRAJ_DUR=15 ./run_hil_preposition.sh     # 더 천천히(멀리 있을 때 권장)
#   SWITCH_TO_FPC=1 ./run_hil_preposition.sh # 명시적 opt-in: proof 후 즉시 전환
#   DRY_RUN=1 ./run_hil_preposition.sh       # 자세 검증/proof만; 절대 전환하지 않음
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HANDOFF_LIB="$SCRIPT_DIR/_hil_controller_handoff.sh"
POSE_CHECKER="$SCRIPT_DIR/_hil_joint_pose_check.py"
if [[ ! -f "$HANDOFF_LIB" || ! -f "$POSE_CHECKER" ]]; then
    echo "FATAL: HIL handoff helper missing under $SCRIPT_DIR" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$HANDOFF_LIB"

# ROS setup 스크립트는 unset 변수를 참조하므로 -u 를 잠시 끈다.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$SCRIPT_DIR/install/setup.bash"
set -u

# --- 튜너블 -----------------------------------------------------------------
# serl_ur_infra/ur_experiments/cube_in_cup.py 의 CubeInCupConfig.RESET_JOINTS 와
# 반드시 동일해야 한다 (pan, lift, elbow, w1, w2, w3).
RESET_JOINTS_CSV="3.1382,-1.5276,1.7168,-1.7592,-1.5216,-3.1331"
TRAJ_DUR="${TRAJ_DUR:-8.0}"          # 궤적 소요 시간(s). 멀면 늘려라(속도가 낮아진다).
PASS_TOL_RAD="${PASS_TOL_RAD:-0.10}" # PASS 판정 임계값(rad). go_to_reset 게이트는 0.9.
HANDOFF_POSE_TOL_RAD="${HIL_ARM_POSE_TOL_RAD:-0.10}"
ARRIVAL_TOL="${ARRIVAL_TOL:-0.05}"   # JTC goal tolerance(실기 검증된 기본값).
SWITCH_TO_FPC="${SWITCH_TO_FPC:-0}"
DRY_RUN="${DRY_RUN:-0}"
NODE_LOG="${NODE_LOG:-/tmp/hil_preposition_$(date +%Y%m%d_%H%M%S).log}"
PREPOSITION_MARKER="${HIL_PREPOSITION_MARKER:-$(hil_default_preposition_marker)}"
PREPOSITION_MARKER_MAX_AGE_S="${HIL_PREPOSITION_MARKER_MAX_AGE_S:-900}"

if ! awk -v value="$HANDOFF_POSE_TOL_RAD" \
    'BEGIN { exit !(value > 0 && value <= 0.10) }'; then
    echo "FATAL: HIL_ARM_POSE_TOL_RAD=$HANDOFF_POSE_TOL_RAD must be in (0, 0.10]" >&2
    exit 1
fi
if ! awk -v value="$PASS_TOL_RAD" -v maximum="$HANDOFF_POSE_TOL_RAD" \
    'BEGIN { exit !(value > 0 && value <= maximum) }'; then
    echo "FATAL: PASS_TOL_RAD=$PASS_TOL_RAD must be in (0, $HANDOFF_POSE_TOL_RAD] for actor handoff proof" >&2
    exit 1
fi
if [[ ! "$PREPOSITION_MARKER_MAX_AGE_S" =~ ^[1-9][0-9]*$ ]] || \
   (( PREPOSITION_MARKER_MAX_AGE_S > 3600 )); then
    echo "FATAL: HIL_PREPOSITION_MARKER_MAX_AGE_S must be an integer in [1, 3600]" >&2
    exit 1
fi
case "$SWITCH_TO_FPC:$DRY_RUN" in
    0:0|0:1|1:0|1:1) ;;
    *) echo "FATAL: SWITCH_TO_FPC and DRY_RUN must each be 0 or 1" >&2; exit 1 ;;
esac

SRC_CTRL="scaled_joint_trajectory_controller"
TGT_CTRL="forward_position_controller"

NODE_PID=""
SENT_ABORT=0

cleanup() {
    if [[ -n "$NODE_PID" ]] && kill -0 "$NODE_PID" 2>/dev/null; then
        echo ""
        echo "[cleanup] gello_move_to_start 종료 중 (PID $NODE_PID)..."
        if [[ "$SENT_ABORT" -eq 0 ]]; then
            timeout 5 ros2 service call /gello_move_to_start/abort \
                std_srvs/srv/Trigger "{}" >/dev/null 2>&1
        fi
        kill -INT "$NODE_PID" 2>/dev/null
        sleep 1
        kill -0 "$NODE_PID" 2>/dev/null && kill -TERM "$NODE_PID" 2>/dev/null
    fi
    if [[ -f "$NODE_LOG" ]]; then
        echo "[cleanup] 노드 로그: $NODE_LOG"
        echo "[cleanup] ★ Ctrl-C 는 이미 수락된 궤적을 멈추지 않는다. 실제 정지는 E-STOP."
    fi
}
trap cleanup EXIT

pose_check() {
    local timeout_s="$1"
    shift
    python3 "$POSE_CHECKER" \
        --topic /joint_states \
        --target "$RESET_JOINTS_CSV" \
        --tolerance "$PASS_TOL_RAD" \
        --timeout "$timeout_s" \
        "$@"
}

record_proof_and_optional_switch() {
    local proof="$1"
    hil_assert_preposition_ready "$SRC_CTRL" "$TGT_CTRL" \
        "/forward_position_controller/commands" || return 1
    hil_write_preposition_marker \
        "$PREPOSITION_MARKER" "$RESET_JOINTS_CSV" "$PASS_TOL_RAD" "$proof" || return 1
    if [[ "$SWITCH_TO_FPC" == "1" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "DRY_RUN=1 — proof만 생성했고 controller는 전환하지 않았다."
            return 0
        fi
        hil_arm_controller_handoff \
            "$SRC_CTRL" "$TGT_CTRL" "/forward_position_controller/commands" \
            "$PREPOSITION_MARKER" "$POSE_CHECKER" "$RESET_JOINTS_CSV" \
            "$HANDOFF_POSE_TOL_RAD" "$PREPOSITION_MARKER_MAX_AGE_S" || {
                # Do not leave a newly-created approval usable after an
                # optional immediate handoff failed before consuming it.
                hil_invalidate_preposition_marker "$PREPOSITION_MARKER" || true
                return 1
            }
    fi
    return 0
}

# A failed/cancelled new preposition attempt must not leave an older approval
# marker usable.  Only a fresh PASS below recreates it.
if ! hil_invalidate_preposition_marker "$PREPOSITION_MARKER"; then
    echo "FATAL: 이전 preposition proof marker를 무효화하지 못했다: $PREPOSITION_MARKER" >&2
    exit 1
fi

banner() { echo ""; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

# =============================================================================
# 1) 사전 점검 (전부 읽기 전용)
# =============================================================================
banner "[1/6] 사전 점검 (읽기 전용)"

CTRL_LIST="$(timeout 20 ros2 control list_controllers 2>/dev/null)"
if [[ -z "$CTRL_LIST" ]]; then
    echo "FAIL: controller_manager 응답 없음. UR 드라이버가 떠 있는지 확인하라."
    exit 1
fi

# NOTE: "[^in]active" 로 써야 한다. ".*active" 는 "inactive" 에도 매치한다.
if ! grep -qE "^${SRC_CTRL}\s.*[^in]active" <<<"$(sed 's/\x1b\[[0-9;]*m//g' <<<"$CTRL_LIST")"; then
    echo "FAIL: ${SRC_CTRL} 가 active 가 아니다."
    echo "      headless 모드에서는 펜던트를 REMOTE + Real Robot 으로 두면 자동 연결된다."
    echo "      (또는 펜던트에서 External Control 프로그램을 Play)"
    sed 's/\x1b\[[0-9;]*m//g' <<<"$CTRL_LIST" | grep -E "joint_trajectory|forward_position"
    exit 1
fi
echo "OK: ${SRC_CTRL} = active (부드러운 보간 궤적 사용 가능)"

if grep -qE "^${TGT_CTRL}\s.*[^in]active" <<<"$(sed 's/\x1b\[[0-9;]*m//g' <<<"$CTRL_LIST")"; then
    echo "FAIL: ${TGT_CTRL} 가 active 다. HIL 액터/브리지가 팔을 스트리밍 중일 수 있다."
    echo "      먼저 액터를 DISARM/종료한 뒤 다시 실행하라."
    exit 1
fi
echo "OK: ${TGT_CTRL} = inactive (스트리밍 퍼블리셔 없음)"

if ! hil_assert_no_command_publishers "/forward_position_controller/commands"; then
    echo "FAIL: command topic 소유자가 이미 있다. actor/bridge를 먼저 종료하라."
    exit 1
fi
echo "OK: /forward_position_controller/commands 퍼블리셔 0개"

NODES="$(timeout 15 ros2 node list 2>/dev/null)"
if grep -q "gello_ur_bridge" <<<"$NODES"; then
    echo "FAIL: gello_ur_bridge 가 떠 있다. 이중 퍼블리셔 위험 — 먼저 종료하라."
    exit 1
fi
echo "OK: gello_ur_bridge 없음"
if grep -q "gello_move_to_start" <<<"$NODES"; then
    echo "FAIL: gello_move_to_start 가 이미 떠 있다. 먼저 종료하라."
    exit 1
fi

# GELLO 스트림은 이 노드의 **필수 전제조건**이다: gello_move_to_start 는
# start_mode 와 무관하게 첫 /gello/joint_states 메시지를 받을 때까지 블록한다.
# (GELLO 는 읽기만 한다 — 토크는 계속 꺼져 있고 리더는 움직이지 않는다.)
if ! timeout 8 ros2 topic echo /gello/joint_states --once >/dev/null 2>&1; then
    echo "FAIL: /gello/joint_states 가 안 나온다."
    echo "      gello_move_to_start 는 GELLO 메시지를 하나 받기 전까지 진행하지 않는다."
    echo "      gello_publisher 를 먼저 띄우거나, 아래 '대안(Freedrive)' 절차를 쓰라."
    exit 1
fi
echo "OK: /gello/joint_states 수신 중 (읽기 전용으로만 사용)"

banner "[2/6] 현재 자세 vs RESET_JOINTS"
pose_check 5.0
PRE_RC=$?
if [[ $PRE_RC -eq 2 ]]; then exit 1; fi
if [[ $PRE_RC -eq 0 ]]; then
    echo ""
    echo "이미 PASS 범위(${PASS_TOL_RAD} rad) 안에 있다. 사전 배치가 필요 없다."
    echo "그래도 다시 정확히 맞추고 싶으면 FORCE=1 로 실행하라."
    if [[ "${FORCE:-0}" != "1" ]]; then
        record_proof_and_optional_switch verified_existing
        exit $?
    fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "DRY_RUN=1 — 여기까지. 노드를 띄우거나 controller를 전환하지 않았다."
    if [[ $PRE_RC -eq 0 ]]; then
        record_proof_and_optional_switch verified_existing
    else
        echo "현재 자세가 PASS 범위 밖이므로 proof marker를 만들지 않았다."
    fi
    exit 0
fi

# =============================================================================
# 3) 조작자 승인
# =============================================================================
banner "[3/6] 조작자 안전 확인"
cat <<'EOF'
지금부터 팔이 자율적으로 움직인다. 아래를 눈으로 확인하라:

  [ ] 팔 주변과 이동 경로에 사람/케이블/물체가 없다
      ★ 관절 공간 직선 보간이다. 충돌 회피를 하지 않는다 ★
  [ ] 펜던트를 손에 들고 있고 E-STOP 에 손이 닿는다
  [ ] 그리퍼에 물린 물체가 없다
  [ ] HIL 액터가 ARM 상태가 아니다

★ 이동이 시작된 뒤에는 Ctrl-C 로 멈출 수 없다. 오직 E-STOP 이다. ★
EOF
echo ""
# 예전에는 여기서 대문자 GO 타이핑을 요구했다. 매 세션 반복이 불편하다는 조작자
# 요청으로 카운트다운으로 바꿨다. 확인 자체를 없애지는 않았다 — 이 게이트가 막는
# 것은 "알림"이 아니라 **충돌 회피 없는 자율 관절 이동**이고, 이동이 시작되면
# Ctrl-C가 듣지 않아 E-STOP밖에 없기 때문이다. 카운트다운 동안에는 아직 로봇에
# 아무 명령도 나가지 않았으므로 Ctrl-C가 정상 동작한다 — 즉 "타이핑 없는 중단 창"이다.
#
#   PREPOSITION_CONFIRM=1  -> 예전처럼 GO 타이핑을 요구한다
#   PREPOSITION_DELAY_S=N  -> 카운트다운 길이 (기본 0 = 즉시 진행)
#
# 이 게이트는 애초에 매번 뜨지 않는다. 현재 자세가 PASS 범위(위 [2/6]) 안이면
# 이동 자체가 생략되므로 여기까지 오지 않는다.
if [[ "${PREPOSITION_CONFIRM:-0}" == "1" ]]; then
    read -r -p "위를 모두 확인했으면 GO 를 입력하고 Enter (그 외는 취소): " CONFIRM
    if [[ "$CONFIRM" != "GO" ]]; then
        echo "취소됨. 로봇에 아무 명령도 보내지 않았다."
        exit 0
    fi
elif (( $(printf '%.0f' "${PREPOSITION_DELAY_S:-0}") > 0 )); then
    echo "위 항목을 확인하라. ${PREPOSITION_DELAY_S}초 뒤 이동을 시작한다 — 지금은 Ctrl-C로 중단된다."
    for (( _i = PREPOSITION_DELAY_S; _i > 0; _i-- )); do
        printf '\r  이동까지 %2ds  (Ctrl-C = 취소, PREPOSITION_CONFIRM=1 = 예전 GO 프롬프트)  ' "$_i"
        sleep 1
    done
    printf '\r  이동 시작.%-60s\n' ""
else
    # 기본값: 확인 없이 즉시 이동(조작자 요청, 2026-07-30). 이 이동은 개입
    # 경로가 아니라 TRAJ_DUR=8s JTC 궤적이다. 주의: 위 [2/6]은 0.10 rad PASS
    # 여부만 판정하며, 이 wrapper 자체에는 RESET_MAX_DIST_RAD=0.9 같은 최대
    # 거리 거부가 없다. current/target 표에 나온 실제 경로를 조작자가 판단한다.
    #
    # 그래도 남는 것: 관절 공간 직선 보간이라 충돌 회피가 없고, 일단 시작되면
    # Ctrl-C가 듣지 않아 정지 수단은 E-STOP뿐이다. 위 체크리스트는 그대로 출력된다.
    #   PREPOSITION_DELAY_S=5  -> 취소 가능한 카운트다운
    #   PREPOSITION_CONFIRM=1  -> 예전 GO 타이핑 프롬프트
    echo "이동을 시작한다 (확인 생략). 정지는 E-STOP."
    echo "  되돌리려면: PREPOSITION_DELAY_S=5 (카운트다운) 또는 PREPOSITION_CONFIRM=1 (GO 프롬프트)"
fi

# =============================================================================
# 4) gello_move_to_start (init_align) 기동
# =============================================================================
banner "[4/6] gello_move_to_start 기동 (start_mode=init_align)"
echo "로그: $NODE_LOG"
echo "궤적 시간: ${TRAJ_DUR}s / 도착 허용오차: ${ARRIVAL_TOL} rad"

# NOTE: 파라미터 파일(ur7e_gello.yaml)을 일부러 로드하지 않는다.
#       그 파일의 start_mode 는 "gello"(리더 추종)이고 init_pose 는 텔레옵용
#       홈 자세라, 여기서 필요한 값과 다르다. 아래 -p 만으로 충분하다.
#       resume_bridge=false 이므로 브리지 서비스를 건드리지 않는다.
#       alignment_timeout=0 => GATE 2 에서 무한 대기(우리는 곧 abort 한다).
ros2 run ur_gello_bringup gello_move_to_start --ros-args \
    -p start_mode:=init_align \
    -p "init_pose:=[${RESET_JOINTS_CSV}]" \
    -p source_controller:="${SRC_CTRL}" \
    -p target_controller:="${TGT_CTRL}" \
    -p trajectory_duration:="${TRAJ_DUR}" \
    -p arrival_tolerance:="${ARRIVAL_TOL}" \
    -p resume_bridge:=false \
    -p alignment_timeout:=0.0 \
    >"$NODE_LOG" 2>&1 &
NODE_PID=$!

echo "PID $NODE_PID — GATE 1 준비 대기 중..."
for _ in $(seq 1 30); do
    if timeout 2 ros2 service type /gello_move_to_start/proceed >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done
if ! timeout 2 ros2 service type /gello_move_to_start/proceed >/dev/null 2>&1; then
    echo "FAIL: /gello_move_to_start/proceed 서비스가 나타나지 않았다. 로그 확인: $NODE_LOG"
    exit 1
fi
# 노드가 GELLO 수신 + 컨트롤러 active 확인을 마치고 GATE 1 에 도달할 시간을 준다.
sleep 3

# =============================================================================
# 5) GATE 1 승인 → 이동
# =============================================================================
banner "[5/6] GATE 1 승인 — 팔이 지금 움직인다"
PROCEED_OUT="$(timeout 15 ros2 service call /gello_move_to_start/proceed \
    std_srvs/srv/Trigger "{}" 2>&1)"
echo "$PROCEED_OUT" | tail -3
if ! grep -q "success=True" <<<"$PROCEED_OUT"; then
    echo "FAIL: GATE 1 승인이 거부됐다. 로그 확인: $NODE_LOG"
    exit 1
fi

# 궤적 시간 + 여유만큼 폴링하며 도착을 기다린다.
WAIT_S=$(python3 -c "print(int(float('$TRAJ_DUR')) + 12)")
echo "이동 중... (최대 ${WAIT_S}s 대기)"
ARRIVED=1
for _ in $(seq 1 "$WAIT_S"); do
    sleep 1
    if pose_check 3.0 --quiet; then
        ARRIVED=0
        break
    fi
    if ! kill -0 "$NODE_PID" 2>/dev/null; then
        echo "경고: 노드가 예기치 않게 종료됐다. 로그 확인: $NODE_LOG"
        break
    fi
done

# GATE 2(= GELLO 리더를 init_pose 에 맞추라는 단계)는 HIL 에서는 의미가 없다.
# 여기서 abort 하면 노드가 **컨트롤러 전환 없이** fail-safe 로 종료한다:
# scaled_joint_trajectory_controller 가 그대로 active 로 남고 팔은 제자리를 유지한다.
echo ""
echo "GATE 2 는 건너뛴다(GELLO 정렬은 HIL 에 불필요) — abort 로 안전 종료."
timeout 10 ros2 service call /gello_move_to_start/abort std_srvs/srv/Trigger "{}" \
    2>&1 | tail -2
SENT_ABORT=1
sleep 2
if kill -0 "$NODE_PID" 2>/dev/null; then
    kill -INT "$NODE_PID" 2>/dev/null
    sleep 1
fi
NODE_PID=""

# =============================================================================
# 5b) 그리퍼 OPEN
# =============================================================================
# 왜: 모든 offline demo가 열린 그리퍼에서 시작한다. 닫힌 채로 세션을 시작하면
# 정책이 한 번도 행동하기 전에 이미 out-of-distribution이고, gripper_position은
# state[0] — 인코더가 보는 첫 원소다.
#
# 이동이 끝난 뒤에 여는 이유: 물체를 물고 있었다면 낙하 지점이 임의의 중간 자세가
# 아니라 항상 같은 RESET 자세가 된다.
#
# 스케일 계약(robotiq_gripper_modbus_node): command_percent 0.0 = OPEN .. 1.0 = CLOSED.
# 그리퍼 노드가 없으면(팔만 검증하는 경우) 조용히 건너뛴다 — 이 스크립트는 팔
# 사전 배치가 본업이고, 여기서 실패해 proof 생성을 막을 이유가 없다.
GRIPPER_CMD_TOPIC="${GRIPPER_CMD_TOPIC:-/robotiq_gripper/command_percent}"
GRIPPER_STATE_TOPIC="${GRIPPER_STATE_TOPIC:-/robotiq_gripper/position_percent}"
GRIPPER_OPEN_WAIT_S="${GRIPPER_OPEN_WAIT_S:-3}"
if [[ "${OPEN_GRIPPER:-1}" == "1" ]]; then
    banner "[5b/6] 그리퍼 OPEN"
    if timeout 3 ros2 topic info "$GRIPPER_CMD_TOPIC" >/dev/null 2>&1; then
        timeout 5 ros2 topic pub --once "$GRIPPER_CMD_TOPIC" \
            std_msgs/msg/Float32 "{data: 0.0}" >/dev/null 2>&1 \
            && echo "OPEN 명령 발행 ($GRIPPER_CMD_TOPIC = 0.0)" \
            || echo "!! OPEN 명령 발행 실패 — 그리퍼 상태를 눈으로 확인하라"
        # Robotiq은 물리적으로 ~0.5s 걸린다. 확인만 하고 실패해도 진행한다.
        _g_deadline=$(( SECONDS + GRIPPER_OPEN_WAIT_S ))
        _g_ok=0
        while (( SECONDS < _g_deadline )); do
            _g_val="$(timeout 2 ros2 topic echo --once --field data \
                        "$GRIPPER_STATE_TOPIC" 2>/dev/null | head -1)"
            if [[ -n "$_g_val" ]] && awk -v v="$_g_val" \
                   'BEGIN { exit !(v <= 0.15) }' 2>/dev/null; then
                echo "OPEN 확인 (position_percent=$_g_val)"
                _g_ok=1
                break
            fi
            sleep 0.3
        done
        (( _g_ok == 1 )) || echo "!! ${GRIPPER_OPEN_WAIT_S}s 안에 OPEN 확인 실패 — 눈으로 확인하라"
    else
        echo "그리퍼 노드 없음 ($GRIPPER_CMD_TOPIC) — 건너뛴다."
    fi
else
    echo "[5b/6] 그리퍼 OPEN 생략 (OPEN_GRIPPER=0)"
fi

# =============================================================================
# 6) 최종 판정
# =============================================================================
banner "[6/6] 최종 판정"
pose_check 5.0
FINAL_RC=$?

echo ""
if [[ $FINAL_RC -eq 0 ]]; then
    echo "==== PASS ===="
    echo "모든 관절이 RESET_JOINTS 로부터 ${PASS_TOL_RAD} rad 이내다."
    echo "go_to_reset() 의 RESET_MAX_DIST_RAD=0.9 게이트를 여유 있게 통과한다."
    if ! record_proof_and_optional_switch operator_preposition; then
        echo "FAIL: 자세는 도착했지만 proof/controller handoff 검증에 실패했다."
        FINAL_RC=1
    fi
else
    echo "==== FAIL ===="
    echo "위 표에서 오차가 큰 관절을 확인하라. 컨트롤러 전환은 하지 않았으므로"
    echo "팔은 그 자리에 정지해 있고 ${SRC_CTRL} 가 여전히 active 다."
    echo "재시도: TRAJ_DUR=15 FORCE=1 $0"
    echo "노드 로그: $NODE_LOG"
fi

echo ""
if [[ $FINAL_RC -ne 0 ]]; then
    echo "proof marker/controller handoff 없음. 실패 원인을 해결한 뒤 다시 실행하라."
elif [[ "$SWITCH_TO_FPC" == "1" ]]; then
    echo "SWITCH_TO_FPC=1 요청까지 완료했다. 위 handoff PASS를 확인하라."
else
    echo "proof marker를 생성했다: $PREPOSITION_MARKER"
    echo "run_hil_actor.sh --arm이 marker와 현재 자세를 다시 검증한 뒤 FPC로 전환한다."
fi

exit $FINAL_RC
