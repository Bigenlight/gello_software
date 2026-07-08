#!/usr/bin/env python3
"""py3.12 ACT inference server for the UR7e "put right banana in pot" deploy.

Owns the trained ACT policy and its internal receding-horizon action queue. A
py3.10 rclpy `policy_leader_node` (different venv/distro) drives it over localhost
ZMQ REQ/REP multipart -- see BUILD_SPEC.md §3/§4 and zmq_protocol.py for the wire
format. This process does ALL the torch/lerobot/cv2 work; the ROS side stays pure
rclpy and forwards raw JPEG bytes.

Inference mirrors eval_offline.py / deploy_ur_act.py VERBATIM in spirit (the
ground-truth for correct ACT inference in lerobot 0.6.1):
    obs dict -> preprocessor(rename/batch/device/normalize)
             -> policy.select_action  (pops from ACT's chunk queue; re-runs the net
                only when the queue empties -> receding horizon k = n_action_steps)
             -> postprocessor(unnormalize -> cpu) -> (7,) numpy.

Key correctness points (from the adversarial review, see BUILD_SPEC §4):
  * The saved preprocessor has NO resize / NO BGR->RGB. image_preprocess.py does
    both (resize to 360x640, BGR->RGB) BEFORE the saved preprocessor runs.
  * `policy.config.n_action_steps` is set to 30 BEFORE the first `policy.reset()`,
    because reset() builds the action deque with maxlen=n_action_steps.
  * obs keys are EXACTLY observation.state / observation.images.cam1 /
    observation.images.cam2 / task.

Run:
    python -m gello_policy.policy_server.act_server \
        --checkpoint <ckpt_dir> --host 127.0.0.1 --port 5591 --device cuda \
        --n-action-steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import zmq

# lerobot 0.6.1 API (annotated with source file:line, like deploy_ur_act.py):
# ACTPolicy: lerobot/src/lerobot/policies/act/modeling_act.py  (from_pretrained ->
#   pretrained.py:162 loads config.json + model.safetensors, .eval(), .to()).
from lerobot.policies.act import ACTPolicy

# make_pre_post_processors: lerobot/src/lerobot/policies/factory.py:273 -- with
# pretrained_path it loads the saved processor pipelines (Normalizer stats baked in):
#   preprocessor : Rename -> AddBatchDim -> Device -> Normalize
#   postprocessor: Unnormalize -> Device(cpu)
from lerobot.policies import make_pre_post_processors

# Allow running both as a module (`-m gello_policy.policy_server.act_server`) and as
# a bare script (scripts/run_act_server.sh). Prefer the package-relative import.
try:
    from .image_preprocess import decode_jpeg_to_rgb_float_chw
    from . import zmq_protocol as proto
except ImportError:  # pragma: no cover - fallback for direct-path execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from image_preprocess import decode_jpeg_to_rgb_float_chw  # type: ignore
    import zmq_protocol as proto  # type: ignore


# =============================================================================
# Policy wrapper
# =============================================================================
class ACTInferenceEngine:
    """Loads the ACT policy + processors and turns (state, cam1, cam2) into a 7-dim
    action. Holds the receding-horizon queue across calls (via policy.reset())."""

    def __init__(self, checkpoint: str, device: str, n_action_steps: int) -> None:
        self.device = device
        self.n_action_steps = n_action_steps

        print(f"[act_server] loading checkpoint: {checkpoint} on {device}", flush=True)
        # from_pretrained loads config.json + model.safetensors (pretrained.py:162).
        self.policy = ACTPolicy.from_pretrained(checkpoint)
        self.policy.to(device)
        self.policy.eval()

        # Saved processor pipelines with normalization stats baked in (factory.py:273).
        # device_processor override guarantees inputs land on the right device, like
        # eval_offline.load_policy_and_processors.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        # CRITICAL: set n_action_steps BEFORE reset(). reset() builds the action deque
        # with maxlen=n_action_steps (modeling_act.py ACTPolicy.reset), so the net
        # re-runs every n_action_steps calls -> receding horizon k=30 on fresh obs.
        self.policy.config.n_action_steps = n_action_steps
        print(
            f"[act_server] n_action_steps={self.policy.config.n_action_steps} "
            f"chunk_size={self.policy.config.chunk_size} "
            f"temporal_ensemble_coeff={self.policy.config.temporal_ensemble_coeff}",
            flush=True,
        )

        # Review M1: the receding-horizon queue only exists when temporal ensembling is
        # OFF (modeling_act.py bypasses _action_queue entirely if the coeff is set), and
        # a chunk shorter than the horizon would silently refill more often than intended.
        # Fail loudly at load rather than mis-behave on the arm.
        assert self.policy.config.temporal_ensemble_coeff is None, (
            "temporal_ensemble_coeff must be None for the n_action_steps receding-horizon "
            f"queue to work; got {self.policy.config.temporal_ensemble_coeff}"
        )
        assert self.policy.config.chunk_size >= n_action_steps, (
            f"chunk_size ({self.policy.config.chunk_size}) must be >= n_action_steps "
            f"({n_action_steps})"
        )

        self.reset()
        self._act_calls = 0
        self._refills = 0
        # Review H1: force CUDA kernel compilation / lazy init NOW, so the operator's
        # first real EXECUTE tick doesn't blow past the ROS node's act_timeout_s (0.5s)
        # and trip a spurious startup FAULT-then-retry. Runs a couple of dummy inferences
        # on zero obs, then clears the queue so the first real episode starts fresh.
        self._warmup()

    @torch.no_grad()
    def _warmup(self) -> None:
        print("[act_server] warming up (compiling inference kernels)...", flush=True)
        # decode_jpeg_to_rgb_float_chw returns (3, 360, 640); mirror that shape here.
        zero_obs = {
            proto.OBS_STATE_KEY: torch.zeros(proto.STATE_DIM, dtype=torch.float32),
            proto.OBS_CAM1_KEY: torch.zeros(3, 360, 640, dtype=torch.float32),
            proto.OBS_CAM2_KEY: torch.zeros(3, 360, 640, dtype=torch.float32),
            proto.OBS_TASK_KEY: "",
        }
        t0 = time.time()
        for _ in range(2):
            proc = self.preprocessor(zero_obs)
            action = self.policy.select_action(proc)
            self.postprocessor(action)
        if self.device == "cuda":
            torch.cuda.synchronize()
        print(f"[act_server] warmup done in {time.time() - t0:.1f}s", flush=True)
        self.reset()  # discard warmup queue/state; real episode starts on the next RESET

    def reset(self) -> None:
        """Clear the ACT action-chunk queue (episode start). modeling_act.py reset()."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @torch.no_grad()
    def act(self, state: np.ndarray, cam1_jpeg: bytes, cam2_jpeg: bytes) -> np.ndarray:
        """Run one inference tick. Returns a (7,) float64 numpy action:
        [q1..q6 radians, grip_cmd 0..1]."""
        # Build obs dict with EXACT dataset feature keys (BUILD_SPEC §4). Images are
        # RGB, CHW, float32 [0,1], (3,360,640) -- resize+BGR->RGB done here, NOT by
        # the saved preprocessor. State is float32 (7,).
        obs = {
            proto.OBS_STATE_KEY: torch.from_numpy(state.astype(np.float32)),  # (7,)
            proto.OBS_CAM1_KEY: decode_jpeg_to_rgb_float_chw(cam1_jpeg),      # (3,360,640)
            proto.OBS_CAM2_KEY: decode_jpeg_to_rgb_float_chw(cam2_jpeg),      # (3,360,640)
            proto.OBS_TASK_KEY: "",  # ACT is single-task; empty string (deploy_ur_act.py)
        }

        # Log net-refills: ACT only re-runs the transformer when the queue is empty.
        # temporal_ensemble_coeff is null for this checkpoint, so the deque is used.
        refill = len(getattr(self.policy, "_action_queue", [])) == 0

        proc = self.preprocessor(obs)               # rename -> batch -> device -> normalize
        action = self.policy.select_action(proc)    # (1,7) normalized (modeling_act.py)
        action = self.postprocessor(action)         # (1,7) unnormalized, on cpu (processor_act.py)
        action = action.squeeze(0).cpu().numpy().astype(np.float64)  # (7,)

        self._act_calls += 1
        if refill:
            self._refills += 1
            print(
                f"[act_server] net refill #{self._refills} "
                f"(act call {self._act_calls}, horizon={self.n_action_steps})",
                flush=True,
            )
        return action


# =============================================================================
# ZMQ REP loop (BUILD_SPEC §3)
# =============================================================================
def _reply(sock: "zmq.Socket", payload: dict) -> None:
    """Send a single-frame JSON reply."""
    sock.send_multipart([json.dumps(payload).encode("utf-8")])


def serve(engine: ACTInferenceEngine, host: str, port: int) -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    endpoint = proto.default_endpoint(host, port)
    sock.bind(endpoint)
    print(f"[act_server] REP bound at {endpoint}; waiting for requests...", flush=True)

    try:
        while True:
            frames = sock.recv_multipart()
            try:
                payload = _handle(engine, frames)
            except Exception as exc:  # never die on a bad request; report it
                payload = {proto.KEY_OK: False, proto.KEY_ERR: f"{type(exc).__name__}: {exc}"}
                print(f"[act_server] ERROR handling request: {payload[proto.KEY_ERR]}", flush=True)
            _reply(sock, payload)
    except KeyboardInterrupt:
        print("\n[act_server] interrupted; shutting down.", flush=True)
    finally:
        sock.close(linger=0)


def _handle(engine: ACTInferenceEngine, frames: list) -> dict:
    """Parse one multipart request and produce the reply dict. Raises on malformed
    input (caller turns the exception into an {ok:false,err} reply)."""
    if not frames:
        raise ValueError("empty request (no frames)")

    ctrl = json.loads(frames[0].decode("utf-8"))
    cmd = ctrl.get(proto.KEY_CMD)

    if cmd == proto.CMD_RESET:
        engine.reset()
        return {proto.KEY_OK: True}

    if cmd == proto.CMD_ACT:
        if len(frames) != proto.ACT_REQUEST_NFRAMES:
            raise ValueError(
                f"act request needs {proto.ACT_REQUEST_NFRAMES} frames "
                f"[ctrl, cam1, cam2], got {len(frames)}"
            )
        state = np.asarray(ctrl.get(proto.KEY_STATE), dtype=np.float64)
        if state.shape != (proto.STATE_DIM,):
            raise ValueError(f"state must be length {proto.STATE_DIM}, got shape {state.shape}")
        cam1_jpeg, cam2_jpeg = frames[1], frames[2]

        action = engine.act(state, cam1_jpeg, cam2_jpeg)
        if action.shape != (proto.ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(f"policy returned invalid action: shape={action.shape}, "
                             f"finite={np.all(np.isfinite(action))}")
        return {proto.KEY_OK: True, proto.KEY_ACTION: [float(x) for x in action]}

    raise ValueError(f"unknown cmd: {cmd!r}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=os.environ.get("ACT_CHECKPOINT"),
                   help="Path to the pretrained_model dir (config.json + "
                        "model.safetensors + policy_pre/postprocessor.json).")
    p.add_argument("--host", default=os.environ.get("ACT_HOST", proto.DEFAULT_HOST))
    p.add_argument("--port", type=int, default=int(os.environ.get("ACT_PORT", proto.DEFAULT_PORT)))
    p.add_argument("--device", default=os.environ.get("ACT_DEVICE", "cuda"),
                   choices=["cuda", "cpu"])
    p.add_argument("--n-action-steps", type=int,
                   default=int(os.environ.get("ACT_N_ACTION_STEPS", "30")),
                   help="Receding horizon: net re-runs every N select_action calls.")
    return p.parse_args(argv)


def resolve_device(device: str) -> str:
    # Review: do NOT silently fall back to CPU. A CPU ACT forward can exceed the ROS
    # leader's act_timeout_s (0.5 s) and trip a spurious FAULT at every net refill.
    # Force an explicit choice: the operator must pass --device cpu (and raise the
    # timeout) knowingly.
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "[act_server] ERROR: --device cuda but CUDA is unavailable. Refusing to "
            "silently run on CPU (a CPU forward can exceed the leader's 0.5 s timeout "
            "and FAULT-loop). Fix CUDA, or pass --device cpu AND raise the leader's "
            "act_timeout_s deliberately.",
            file=sys.stderr, flush=True,
        )
        raise SystemExit(3)
    return device


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.checkpoint:
        print("[act_server] --checkpoint (or $ACT_CHECKPOINT) is required.", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    t0 = time.time()
    engine = ACTInferenceEngine(args.checkpoint, device, args.n_action_steps)
    print(f"[act_server] ready in {time.time() - t0:.1f}s", flush=True)
    serve(engine, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
