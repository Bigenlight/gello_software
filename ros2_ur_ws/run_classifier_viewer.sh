#!/usr/bin/env bash
# Read-only live cube-in-cup classifier viewer.
#
# Prerequisite: cam1/cam2 RealSense ROS topics are already running.  The robot
# state topics are optional (the right-hand panel shows missing values if the
# robot is disconnected).  No policy server or policy_leader_node is required.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export HIL_SERL_ROOT="${HIL_SERL_ROOT:-$REPO_ROOT/third_party/hil-serl}"
export REWARD_CLASSIFIER_CHECKPOINT="${REWARD_CLASSIFIER_CHECKPOINT:-$REPO_ROOT/classifier_ckpt/cube_in_cup}"
export PYTHONPATH="$HIL_SERL_ROOT/serl_launcher${PYTHONPATH:+:$PYTHONPATH}"
CLASSIFIER_THRESHOLD="${CLASSIFIER_THRESHOLD:-0.5}"
CLASSIFIER_PYTHON="${REWARD_CLASSIFIER_PYTHON:-python3}"

source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

# ROS Humble uses Python 3.10.  The classifier venv must therefore also use
# Python 3.10 and be created with --system-site-packages so it can import rclpy:
#   REWARD_CLASSIFIER_PYTHON=~/.venvs/hilserl/bin/python ./run_classifier_viewer.sh
if ! "$CLASSIFIER_PYTHON" - "$CLASSIFIER_THRESHOLD" <<'PY'
import math
import os
import sys

threshold = float(sys.argv[1])
if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
    raise SystemExit("CLASSIFIER_THRESHOLD must be finite and in [0, 1]")

# Import the actual executable module, ROS message packages, and all direct
# dependencies used while upstream HIL-SERL builds/restores the classifier.
import cv2
import flax
import jax
import optax
import rclpy
import requests
import sensor_msgs
import std_msgs
import tqdm
from gello_recorder import reward_classifier_node
from serl_launcher.networks.reward_classifier import load_classifier_func

weights = os.path.expanduser("~/.serl/resnet10_params.pkl")
if not os.path.isfile(weights):
    raise SystemExit(
        "missing %s; stage the official HIL-SERL ResNet-10 weights before "
        "running on the robot (create_classifier otherwise downloads them): "
        "https://github.com/rail-berkeley/serl/releases/download/resnet10/"
        "resnet10_params.pkl" % weights
    )
PY
then
    echo "Reward-classifier preflight failed." >&2
    echo "Use a Python 3.10 HIL-SERL venv with --system-site-packages:" >&2
    echo "  REWARD_CLASSIFIER_PYTHON=/path/to/venv/bin/python $0" >&2
    exit 2
fi

stop_child() {
    local pid="${1:-}"
    [[ -z "$pid" ]] && return
    kill -0 "$pid" 2>/dev/null || return
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    stop_child "${GUI_PID:-}"
    stop_child "${CLASSIFIER_PID:-}"
}
trap cleanup EXIT INT TERM

"$CLASSIFIER_PYTHON" -m gello_recorder.reward_classifier_node --ros-args \
    -p "hil_serl_root:=$HIL_SERL_ROOT" \
    -p "checkpoint_path:=$REWARD_CLASSIFIER_CHECKPOINT" \
    -p "threshold:=$CLASSIFIER_THRESHOLD" &
CLASSIFIER_PID=$!

ros2 run gello_recorder classifier_view_gui &
GUI_PID=$!

set +e
wait -n "$CLASSIFIER_PID" "$GUI_PID"
FIRST_EXIT=$?
set -e
if ! kill -0 "$CLASSIFIER_PID" 2>/dev/null; then
    echo "reward_classifier exited; closing classifier viewer (exit=$FIRST_EXIT)." >&2
    [[ "$FIRST_EXIT" -eq 0 ]] && exit 1
fi
exit "$FIRST_EXIT"
