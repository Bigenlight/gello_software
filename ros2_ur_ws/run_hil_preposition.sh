#!/usr/bin/env bash
# =============================================================================
# run_hil_preposition.sh — HIL-SERL 세션 전 UR7e를 RESET_JOINTS로 사전 배치
# =============================================================================
#
# 무엇을 하는가
# -------------
# HIL-SERL 액터의 ur7e_env.go_to_reset() 은 RESET_MAX_DIST_RAD(=0.5 rad) 게이트로
# 먼 거리 리셋을 거부한다(의도된 안전 동작 — 리셋은 250 Hz 업샘플러로 경로를 모른 채
# 스트리밍으로 쓸고 지나가기 때문). 그래서 매 세션 시작 전에 팔을 RESET_JOINTS
# 근처(0.5 rad 이내)로 가져다 놔야 한다.
#
# 지금까지는 펜던트 Freedrive로 손으로 옮겼지만, 이 스크립트는 이미 검증된
# `gello_move_to_start` 노드의 init_align 모드를 재사용해서 그 일을 재현 가능하게
# 자동화한다:
#
#   1) 사전 점검(읽기 전용): 컨트롤러 상태 / GELLO 스트림 / 충돌 노드 확인
#   2) 조작자가 "GO"를 타이핑해야만 노드를 띄운다
#   3) gello_move_to_start(start_mode:=init_align, init_pose:=RESET_JOINTS) 기동
#   4) GATE 1(~/proceed) 승인 → 팔이 scaled_joint_trajectory_controller 로
#      RESET_JOINTS 까지 **시간 파라미터화된 부드러운 궤적**으로 이동
#   5) 도착 후 ~/abort 를 호출해 **컨트롤러 전환 없이** 노드를 종료
#   6) 최종 자세를 branch-cut 안전(circular) 거리로 재측정하여 PASS/FAIL 판정
#
# 왜 gello_move_to_start 인가 (직접 궤적을 쏘지 않는 이유)
# --------------------------------------------------------
#   * scaled_joint_trajectory_controller 를 쓰므로 **보간**된다(속도 제한 준수,
#     protective stop 위험이 낮다). forward_position_controller 는 보간하지 않는다.
#   * init_pose 를 보낼 때 `angle_utils.wrapped_nearest(init_pose, actual)` 로
#     **branch-cut(±pi 경계)을 처리**한다. RESET_JOINTS 는 shoulder_pan 이 +pi를
#     ~0.003 rad, wrist_3 가 -pi를 ~0.008 rad 넘어간 값이라, 이 처리를 안 하면
#     wrist_3 가 한 바퀴(≈2π) 도는 사고가 난다. 이 노드는 그 처리를 한다.
#     (go_to_reset() 에는 이 처리가 없다 — 별개 버그로 보고됨.)
#   * 모든 물리적 동작이 조작자의 명시적 서비스 승인 뒤에만 일어난다.
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
#   PASS : 모든 관절의 **circular(±pi 안전) 거리** ≤ PASS_TOL_RAD(기본 0.10 rad).
#          이 상태면 go_to_reset() 의 0.5 rad 게이트를 여유 있게 통과한다.
#   FAIL : 위를 만족하지 못함. 스크립트가 관절별 오차를 출력한다.
#          → 대개 원인은 (a) 궤적이 중간에 abort 됨(속도/protective stop),
#            (b) stjc 가 inactive 로 떨어짐, (c) 조작자가 중간에 중단.
#          FAIL 이어도 컨트롤러 전환은 하지 않으므로 팔은 그 자리에 멈춰 있다.
#
# 이 스크립트가 하지 않는 것
# --------------------------
#   * 컨트롤러를 forward_position_controller 로 바꾸지 않는다(기본값).
#     HIL 액터는 /forward_position_controller/commands 로 명령하므로 결국 fpc 가
#     active 여야 한다. 필요하면 SWITCH_TO_FPC=1 로 실행하거나, 끝난 뒤 수동으로:
#       ros2 control switch_controllers \
#         --activate forward_position_controller \
#         --deactivate scaled_joint_trajectory_controller
#   * GELLO 리더를 건드리지 않는다(GELLO 는 끝까지 수동/passive, 읽기만 한다).
#   * 그리퍼를 건드리지 않는다.
#
# 사용법
# ------
#   ./run_hil_preposition.sh                 # 기본(궤적 8초, fpc 전환 안 함)
#   TRAJ_DUR=15 ./run_hil_preposition.sh     # 더 천천히(멀리 있을 때 권장)
#   SWITCH_TO_FPC=1 ./run_hil_preposition.sh # 끝나고 fpc 로 전환까지
#   DRY_RUN=1 ./run_hil_preposition.sh       # 사전 점검 + 거리 측정만, 노드 안 띄움
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

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
PASS_TOL_RAD="${PASS_TOL_RAD:-0.10}" # PASS 판정 임계값(rad). go_to_reset 게이트는 0.5.
ARRIVAL_TOL="${ARRIVAL_TOL:-0.05}"   # JTC goal tolerance(실기 검증된 기본값).
SWITCH_TO_FPC="${SWITCH_TO_FPC:-0}"
DRY_RUN="${DRY_RUN:-0}"
NODE_LOG="${NODE_LOG:-/tmp/hil_preposition_$(date +%Y%m%d_%H%M%S).log}"

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

# --- 관절 거리 측정기 (읽기 전용, rclpy) -------------------------------------
POSE_CHECK="$(mktemp /tmp/hil_prepos_check_XXXXXX.py)"
cat >"$POSE_CHECK" <<'PYEOF'
"""Read ONE /joint_states and report branch-cut-safe distance to a target pose.

읽기 전용. 로봇에 아무것도 보내지 않는다.
argv: <target_csv> <pass_tol_rad> <wait_timeout_s> [--quiet]
exit 0 = 모든 관절 circular 오차 <= pass_tol, 1 = 초과, 2 = /joint_states 없음
"""
import math
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
SHORT = ["pan", "lift", "elbow", "w1", "w2", "w3"]

target = [float(x) for x in sys.argv[1].split(",")]
tol = float(sys.argv[2])
timeout = float(sys.argv[3])
quiet = "--quiet" in sys.argv[4:]


def wrap_to_pi(x):
    return math.remainder(x, 2.0 * math.pi)


class Once(Node):
    def __init__(self):
        super().__init__("hil_prepos_pose_check")
        self.pose = None
        self.create_subscription(JointState, "/joint_states", self._cb, 10)

    def _cb(self, msg):
        m = dict(zip(msg.name, msg.position))
        if all(j in m for j in ORDER):
            self.pose = [float(m[j]) for j in ORDER]


rclpy.init()
node = Once()
deadline = node.get_clock().now().nanoseconds + int(timeout * 1e9)
while rclpy.ok() and node.pose is None:
    if node.get_clock().now().nanoseconds > deadline:
        break
    rclpy.spin_once(node, timeout_sec=0.1)
pose = node.pose
node.destroy_node()
rclpy.shutdown()

if pose is None:
    print("ERROR: /joint_states 를 받지 못했다 (드라이버가 떠 있는가?)")
    sys.exit(2)

circ = [abs(wrap_to_pi(pose[i] - target[i])) for i in range(6)]
raw = [abs(pose[i] - target[i]) for i in range(6)]
if not quiet:
    print("  관절      현재        목표        circular    raw(순진한 차)")
    for i in range(6):
        flag = "  <-- WRAP!" if raw[i] - circ[i] > 1.0 else ""
        print(f"  {SHORT[i]:<6} {pose[i]:>9.4f}  {target[i]:>9.4f}  "
              f"{circ[i]:>9.4f}  {raw[i]:>9.4f}{flag}")
    worst = max(range(6), key=lambda i: circ[i])
    print(f"  => 최대 circular 오차 {circ[worst]:.4f} rad ({SHORT[worst]}), "
          f"최대 raw 오차 {max(raw):.4f} rad")
    print(f"  => go_to_reset 게이트(RESET_MAX_DIST_RAD=0.5) "
          f"{'통과 가능' if max(circ) <= 0.5 else '거부됨'} (circular 기준)")
sys.exit(0 if max(circ) <= tol else 1)
PYEOF
trap 'rm -f "$POSE_CHECK"; cleanup' EXIT

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
python3 "$POSE_CHECK" "$RESET_JOINTS_CSV" "$PASS_TOL_RAD" 5.0
PRE_RC=$?
if [[ $PRE_RC -eq 2 ]]; then exit 1; fi
if [[ $PRE_RC -eq 0 ]]; then
    echo ""
    echo "이미 PASS 범위(${PASS_TOL_RAD} rad) 안에 있다. 사전 배치가 필요 없다."
    echo "그래도 다시 정확히 맞추고 싶으면 FORCE=1 로 실행하라."
    if [[ "${FORCE:-0}" != "1" ]]; then exit 0; fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "DRY_RUN=1 — 여기까지. 노드를 띄우지 않았다."
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
read -r -p "위를 모두 확인했으면 GO 를 입력하고 Enter (그 외는 취소): " CONFIRM
if [[ "$CONFIRM" != "GO" ]]; then
    echo "취소됨. 로봇에 아무 명령도 보내지 않았다."
    exit 0
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
    if python3 "$POSE_CHECK" "$RESET_JOINTS_CSV" "$PASS_TOL_RAD" 3.0 --quiet; then
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
# 6) 최종 판정
# =============================================================================
banner "[6/6] 최종 판정"
python3 "$POSE_CHECK" "$RESET_JOINTS_CSV" "$PASS_TOL_RAD" 5.0
FINAL_RC=$?

echo ""
if [[ $FINAL_RC -eq 0 ]]; then
    echo "==== PASS ===="
    echo "모든 관절이 RESET_JOINTS 로부터 ${PASS_TOL_RAD} rad 이내다."
    echo "go_to_reset() 의 RESET_MAX_DIST_RAD=0.5 게이트를 여유 있게 통과한다."
else
    echo "==== FAIL ===="
    echo "위 표에서 오차가 큰 관절을 확인하라. 컨트롤러 전환은 하지 않았으므로"
    echo "팔은 그 자리에 정지해 있고 ${SRC_CTRL} 가 여전히 active 다."
    echo "재시도: TRAJ_DUR=15 FORCE=1 $0"
    echo "노드 로그: $NODE_LOG"
fi

echo ""
echo "현재 컨트롤러 상태: ${SRC_CTRL}=active, ${TGT_CTRL}=inactive"
if [[ "$SWITCH_TO_FPC" == "1" && $FINAL_RC -eq 0 ]]; then
    echo ""
    echo "SWITCH_TO_FPC=1 — ${TGT_CTRL} 로 전환한다."
    echo "  (fpc 는 ForwardCommandController 라서 첫 /commands 가 오기 전까지"
    echo "   아무것도 쓰지 않는다. 직전 JTC 가 붙잡고 있던 현재 자세를 유지한다.)"
    timeout 15 ros2 control switch_controllers \
        --activate "${TGT_CTRL}" --deactivate "${SRC_CTRL}" 2>&1 | tail -3
    timeout 15 ros2 control list_controllers 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -E "forward_position|scaled_joint"
else
    echo "HIL 액터는 /${TGT_CTRL}/commands 로 명령하므로 fpc 가 active 여야 한다."
    echo "필요하면:"
    echo "  ros2 control switch_controllers --activate ${TGT_CTRL} \\"
    echo "      --deactivate ${SRC_CTRL}"
fi

exit $FINAL_RC
