#!/usr/bin/env bash
# Kanu: start the GPU reward-classifier service on loopback only.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${REWARD_CLASSIFIER_PYTHON:-python3}"
PORT="${CLASSIFIER_REMOTE_PORT:-5594}"
REQUIRE_JAX_GPU="${REQUIRE_JAX_GPU:-true}"
REQUIRE_JAX_GPU="$REQUIRE_JAX_GPU" "$PYTHON" -c '
import os
import cv2, flax, jax, numpy, zmq
backend = jax.default_backend()
print("JAX backend: %s" % backend)
if os.environ["REQUIRE_JAX_GPU"].lower() in ("1", "true", "yes") and backend != "gpu":
    raise SystemExit("GPU required but JAX backend is %s" % backend)
' || {
    echo "Classifier environment is incomplete (requires pyzmq and HIL-SERL dependencies)." >&2
    echo "GPU is required by default; use REQUIRE_JAX_GPU=false only for a CPU smoke test." >&2
    exit 2
}
exec "$PYTHON" "$SCRIPT_DIR/remote_reward_classifier_server.py" \
    --bind "tcp://127.0.0.1:${PORT}" \
    --hil-serl-root "${HIL_SERL_ROOT:-$REPO_ROOT/third_party/hil-serl}" \
    --checkpoint "${REWARD_CLASSIFIER_CHECKPOINT:-$REPO_ROOT/classifier_ckpt/cube_in_cup_all3}"
