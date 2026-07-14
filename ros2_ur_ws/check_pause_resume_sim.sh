#!/usr/bin/env bash
# Rehearse the gello_ur_bridge PAUSE / RESUME-CHASE feature against the MOCK stack.
#
# This drives the ENTIRE pause/resume-chase safety story with NO real robot: it
# talks to the mock UR7e + fake_gello leader brought up by
#
#     ./run_ur7e_gello_sim.sh                                    # (fake source)
#   or
#     ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake
#
# in ANOTHER terminal, then exercises pause, the fail-closed refusals (moving
# leader, gap too large) and the accepted glide, printing PASS / FAIL per step.
#
# PRECONDITION: the mock stack is ALREADY running in another terminal. This
# script never launches or kills it.
#
# Usage:
#   ./run_ur7e_gello_sim.sh          # terminal 1 (leave running)
#   ./check_pause_resume_sim.sh      # terminal 2
#
# SIM PASS is necessary but NOT sufficient: the mock hardware enforces no velocity
# limits and no protective stops. Velocity-limit and smoothness safety MUST be
# re-verified by a human on the real UR7e before any data collection.
#
# NOTE: unlike run_ur7e_gello_sim.sh this does NOT 'set -e' — a test harness must
# run every step and tally results rather than abort on the first failure. It also
# avoids 'set -u' because the ROS setup.bash it sources is not unbound-var clean.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

# --- Contract constants (must match SERVICE/TOPIC CONTRACTS) ------------------
STATE_TOPIC="/gello_ur_bridge/state"
PAUSE_SRV="/gello_ur_bridge/pause"
RESUME_CHASE_SRV="/gello_ur_bridge/resume_chase"
FAKE_HOLD_SRV="/fake_gello/hold"
FAKE_SWEEP_SRV="/fake_gello/sweep"
FAKE_COLLAPSE_SRV="/fake_gello/collapse"
FAKE_SET_POSE_TOPIC="/fake_gello/set_pose"
JS_TOPIC="/joint_states"

FAILS=0
SKIPS=0

pass_() { echo "  [PASS] $*"; }
fail_() { echo "  [FAIL] $*"; FAILS=$((FAILS + 1)); }
skip_() { echo "  [SKIP] $*"; SKIPS=$((SKIPS + 1)); }
step_() { echo; echo "=== $* ==="; }

# --- Low-level helpers -------------------------------------------------------
get_state() {
    # Print the latest bridge state string (empty on timeout).
    timeout 5 ros2 topic echo --once "$STATE_TOPIC" std_msgs/msg/String 2>/dev/null \
        | sed -n 's/^data:[[:space:]]*//p' | tr -d "'\" " | head -n1
}

# Wait until the state matches an ERE ($1) or timeout $2 s. Sets LAST_STATE.
poll_state() {
    local want="$1" secs="$2" deadline
    deadline=$(( $(date +%s) + secs ))
    LAST_STATE=""
    while [ "$(date +%s)" -lt "$deadline" ]; do
        LAST_STATE="$(get_state)"
        if printf '%s' "$LAST_STATE" | grep -Eq "$want"; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

# Read /joint_states -> 6 positions in canonical UR order, matched BY NAME.
read_joints() {
    timeout 8 ros2 topic echo --once "$JS_TOPIC" sensor_msgs/msg/JointState 2>/dev/null \
        | python3 -c '
import sys, yaml
order = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# ros2 topic echo interleaves noise into stdout: "A message was lost!!!", tab-
# prefixed "total count ..." stats lines (sometimes with a "---" glued on), and a
# trailing "---" document separator. Strip all of it, then parse the one message.
raw = sys.stdin.read()
keep = []
for l in raw.splitlines():
    if l.startswith("\t"):
        continue
    if "count change" in l or "total count" in l:
        continue
    if "message was lost" in l.lower():
        continue
    if l.strip() == "---":
        continue
    keep.append(l)
doc = None
for cand in yaml.safe_load_all("\n".join(keep)):
    if isinstance(cand, dict) and "name" in cand and "position" in cand:
        doc = cand
        break
if doc is None:
    sys.exit(1)
m = dict(zip(doc["name"], doc["position"]))
try:
    print(" ".join(f"{float(m[n]):.6f}" for n in order))
except KeyError:
    sys.exit(1)
'
}

# Publish a 6-value set_pose to the fake leader (jumps it there + holds).
pub_set_pose() {  # $1..$6
    timeout 8 ros2 topic pub --once "$FAKE_SET_POSE_TOPIC" \
        std_msgs/msg/Float64MultiArray "{data: [$1, $2, $3, $4, $5, $6]}" \
        >/dev/null 2>&1
}

# Call a std_srvs/Trigger service; echo raw response (for message inspection).
call_trigger() {  # $1 = service name
    timeout 12 ros2 service call "$1" std_srvs/srv/Trigger "{}" 2>&1
}
trigger_ok() {  # $1 = raw response text -> 0 if success=True
    printf '%s' "$1" | grep -q "success=True"
}

# max abs per-joint diff of two 6-vectors (space separated). Prints a float.
max_diff() {  # $1 poseA  $2 poseB
    python3 -c '
import sys
a = list(map(float, sys.argv[1].split()))
b = list(map(float, sys.argv[2].split()))
print(f"{max(abs(x - y) for x, y in zip(a, b)):.6f}")
' "$1" "$2"
}
le() {  # $1 <= $2 ?  (floats)
    python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' "$1" "$2"
}

echo "############################################################"
echo "#  check_pause_resume_sim.sh  --  MOCK pause/resume-chase   #"
echo "############################################################"

# --- Preconditions -----------------------------------------------------------
step_ "Preconditions"
if ! ros2 topic list 2>/dev/null | grep -q "^${STATE_TOPIC}$"; then
    echo "  [FATAL] ${STATE_TOPIC} not found. Is the mock stack running?" >&2
    echo "          Start it in another terminal: ./run_ur7e_gello_sim.sh" >&2
    exit 1
fi
pass_ "bridge state topic present: ${STATE_TOPIC}"

HAVE_RESUME_CHASE=1
if ros2 service list 2>/dev/null | grep -q "^${RESUME_CHASE_SRV}$"; then
    pass_ "resume_chase service present: ${RESUME_CHASE_SRV}"
else
    HAVE_RESUME_CHASE=0
    echo "  [WARN] ${RESUME_CHASE_SRV} not present -- resume_chase asserts will be SKIPPED."
fi

# --- Step 1: baseline following ---------------------------------------------
step_ "Step 1: bridge is FOLLOWING/CHASING at baseline"
if poll_state '^(FOLLOWING|CHASING)$' 8; then
    pass_ "state=${LAST_STATE} (following the fake sweep)"
else
    fail_ "expected FOLLOWING/CHASING within 8s, saw '${LAST_STATE}'"
fi

# --- Step 2: hold the leader -> FOLLOWING ------------------------------------
step_ "Step 2: /fake_gello/hold -> expect FOLLOWING"
OUT="$(call_trigger "$FAKE_HOLD_SRV")"
if trigger_ok "$OUT"; then
    if poll_state '^FOLLOWING$' 6; then
        pass_ "leader held; state=FOLLOWING"
    else
        fail_ "after hold expected FOLLOWING within 6s, saw '${LAST_STATE}'"
    fi
else
    fail_ "hold service call failed: ${OUT}"
fi

# --- Step 3: pause -> PAUSED, capture pose -----------------------------------
step_ "Step 3: /gello_ur_bridge/pause -> expect PAUSED, capture /joint_states"
OUT="$(call_trigger "$PAUSE_SRV")"
if trigger_ok "$OUT"; then
    if poll_state '^PAUSED$' 5; then
        pass_ "state=PAUSED"
    else
        fail_ "after pause expected PAUSED within 5s, saw '${LAST_STATE}'"
    fi
else
    fail_ "pause service call failed (pause must ALWAYS succeed): ${OUT}"
fi
BASE_POSE="$(read_joints)"
if [ -n "$BASE_POSE" ]; then
    pass_ "captured paused pose: [${BASE_POSE}]"
else
    fail_ "could not read /joint_states"
fi

# --- Step 4: collapse leader; robot must NOT move while paused ----------------
step_ "Step 4: /fake_gello/collapse, sleep 3 -> robot pose unchanged, still PAUSED"
OUT="$(call_trigger "$FAKE_COLLAPSE_SRV")"
if ! trigger_ok "$OUT"; then
    fail_ "collapse service call failed: ${OUT}"
fi
sleep 3
POSE_AFTER="$(read_joints)"
if [ -z "$BASE_POSE" ] || [ -z "$POSE_AFTER" ]; then
    fail_ "missing joint captures (before='${BASE_POSE}' after='${POSE_AFTER}')"
else
    MD="$(max_diff "$BASE_POSE" "$POSE_AFTER")"
    if le "$MD" "0.001"; then
        pass_ "robot pose unchanged while paused (max_diff=${MD} rad <= 1e-3)"
    else
        fail_ "robot MOVED while paused! max_diff=${MD} rad > 1e-3 (SAFETY)"
    fi
fi
CUR_STATE="$(get_state)"
if [ "$CUR_STATE" = "PAUSED" ]; then
    pass_ "state still PAUSED after collapse"
else
    fail_ "expected PAUSED after collapse, saw '${CUR_STATE}'"
fi

# --- Step 5: NEGATIVE -- moving leader must be refused -----------------------
step_ "Step 5: NEGATIVE -- sweep (moving leader) then resume_chase -> refuse"
if [ "$HAVE_RESUME_CHASE" -eq 1 ]; then
    call_trigger "$FAKE_SWEEP_SRV" >/dev/null
    # Let the sweep motion fill the ~0.3 s quasi-still window so the refusal is
    # unambiguously "leader moving" (not "stillness not yet established").
    sleep 0.6
    OUT="$(call_trigger "$RESUME_CHASE_SRV")"
    if trigger_ok "$OUT"; then
        fail_ "resume_chase ACCEPTED with a MOVING leader (must refuse!): ${OUT}"
    else
        pass_ "resume_chase refused a moving leader (success=false)"
    fi
    if poll_state '^PAUSED$' 3; then
        pass_ "bridge stayed PAUSED after refusal (fail-closed)"
    else
        fail_ "bridge left PAUSED after a refused resume_chase (state='${LAST_STATE}')"
    fi
else
    skip_ "moving-leader refusal (resume_chase absent)"
    skip_ "fail-closed-stays-paused (resume_chase absent)"
fi

# Re-hold the leader so subsequent steps have a still baseline.
call_trigger "$FAKE_HOLD_SRV" >/dev/null

# --- Step 6: NEGATIVE -- gap too large must be refused -----------------------
step_ "Step 6: NEGATIVE -- set_pose >1.5 rad away -> resume_chase refuse (gap)"
if [ "$HAVE_RESUME_CHASE" -eq 1 ]; then
    BASE_POSE="$(read_joints)"          # still frozen (paused); use as reference
    read -r b0 b1 b2 b3 b4 b5 <<<"$BASE_POSE"
    FAR="$(python3 -c 'import sys; print(f"{float(sys.argv[1]) + 1.6:.6f}")' "$b0")"  # +1.6 rad pan
    pub_set_pose "$FAR" "$b1" "$b2" "$b3" "$b4" "$b5"
    sleep 1
    OUT="$(call_trigger "$RESUME_CHASE_SRV")"
    if trigger_ok "$OUT"; then
        fail_ "resume_chase ACCEPTED with a 1.6 rad gap (cap is 1.5, must refuse!): ${OUT}"
    else
        if printf '%s' "$OUT" | grep -qi "gap"; then
            pass_ "resume_chase refused on gap>1.5 rad (message cites gap)"
        else
            pass_ "resume_chase refused the oversized gap (success=false)"
        fi
    fi
    if poll_state '^PAUSED$' 3; then
        pass_ "bridge stayed PAUSED after gap refusal (fail-closed)"
    else
        fail_ "bridge left PAUSED after gap refusal (state='${LAST_STATE}')"
    fi
else
    skip_ "gap-too-large refusal (resume_chase absent)"
    skip_ "fail-closed-stays-paused (resume_chase absent)"
fi

# --- Step 7: POSITIVE -- accepted glide converges ----------------------------
step_ "Step 7: POSITIVE -- set_pose ~0.8 rad away -> resume_chase accept + glide"
if [ "$HAVE_RESUME_CHASE" -eq 1 ]; then
    BASE_POSE="$(read_joints)"          # still frozen (paused)
    read -r b0 b1 b2 b3 b4 b5 <<<"$BASE_POSE"
    NEAR="$(python3 -c 'import sys; print(f"{float(sys.argv[1]) + 0.8:.6f}")' "$b0")"  # +0.8 rad pan
    TARGET="${NEAR} ${b1} ${b2} ${b3} ${b4} ${b5}"
    pub_set_pose "$NEAR" "$b1" "$b2" "$b3" "$b4" "$b5"
    sleep 1
    OUT="$(call_trigger "$RESUME_CHASE_SRV")"
    if trigger_ok "$OUT"; then
        pass_ "resume_chase ACCEPTED for a 0.8 rad gap (still leader)"
    else
        fail_ "resume_chase REFUSED an in-spec 0.8 rad still-leader glide: ${OUT}"
    fi

    # Poll state through CHASING -> FOLLOWING within 10 s.
    saw_chasing=0
    reached_following=0
    deadline=$(( $(date +%s) + 10 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        s="$(get_state)"
        [ "$s" = "CHASING" ] && saw_chasing=1
        if [ "$s" = "FOLLOWING" ]; then reached_following=1; break; fi
        sleep 0.2
    done
    if [ "$saw_chasing" -eq 1 ]; then
        pass_ "observed CHASING during the glide"
    else
        skip_ "did not catch a CHASING sample (glide may have been fast)"
    fi
    if [ "$reached_following" -eq 1 ]; then
        pass_ "reached FOLLOWING within 10s"
    else
        fail_ "did not reach FOLLOWING within 10s (last='${s}')"
    fi

    # /joint_states must converge to the commanded target within 0.05 rad.
    FINAL="$(read_joints)"
    if [ -n "$FINAL" ]; then
        CD="$(max_diff "$FINAL" "$TARGET")"
        if le "$CD" "0.05"; then
            pass_ "robot converged to leader target (max_diff=${CD} rad <= 0.05)"
        else
            fail_ "robot did NOT converge (max_diff=${CD} rad > 0.05)"
        fi
    else
        fail_ "could not read /joint_states after glide"
    fi
else
    skip_ "accepted-glide acceptance (resume_chase absent)"
    skip_ "CHASING->FOLLOWING progression (resume_chase absent)"
    skip_ "convergence within 0.05 rad (resume_chase absent)"
fi

# --- Summary -----------------------------------------------------------------
step_ "Summary"
echo "  FAIL=${FAILS}  SKIP=${SKIPS}"
echo
echo "############################################################################"
echo "#  SIM PASS is necessary but NOT sufficient --                             #"
echo "#  real-robot verification is mandatory.                                   #"
echo "#  Mock hardware enforces NO velocity limits and NO protective stops;      #"
echo "#  a human MUST re-verify velocity-limit + smoothness safety on the real   #"
echo "#  UR7e before using this in data collection.                              #"
echo "############################################################################"

if [ "$FAILS" -gt 0 ]; then
    exit 1
fi
exit 0
