#!/usr/bin/env python3
"""py3.12 Flow-Matching (multi_task_dit) inference server for the UR7e "put right
banana in pot" deploy.

FM sibling of diffusion_server.py. It owns the trained MultiTaskDiTPolicy
(objective=flow_matching) and its internal receding-horizon action queue. A py3.10
rclpy `policy_leader_node` (different venv/distro) drives it over localhost ZMQ
REQ/REP multipart -- SAME wire format as the diffusion/ACT servers (see
zmq_protocol.py). This process does ALL the torch/lerobot/cv2 work; the ROS side
stays pure rclpy and forwards raw JPEG bytes.

The 7-D JOINT action contract ([q1..q6 rad, grip_cmd]) and the two-camera obs are
IDENTICAL to the diffusion JOINT model, so the whole ROS2 safety stack (leader,
bridge, clamps, gripper) is reused unchanged -- only this inference server differs.

Inference mirrors the select_action receding-horizon loop:
    obs dict -> preprocessor(rename/batch/TOKENIZE/device/normalize)
             -> policy.select_action  (pops from the FM action queue; re-runs the
                Euler ODE integrator only when the queue empties -> receding horizon
                k = n_action_steps)
             -> postprocessor(unnormalize -> cpu) -> (7,) numpy.

THREE things differ from diffusion_server.py (everything else -- ZMQ protocol,
gripper handling, queue logic, health checks, logging -- is identical):
  1. GENERIC POLICY LOAD (not a hardcoded class). We resolve the policy class from
     the checkpoint's config via lerobot's factory, so `multi_task_dit` ->
     MultiTaskDiTPolicy without importing it by name:
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        PolicyCls = get_policy_class(cfg.type)
        policy = PolicyCls.from_pretrained(ckpt, config=cfg)
  2. EULER ODE, NOT DDIM. Flow matching integrates an ODE over
     `cfg.num_integration_steps` Euler steps (integration_method=euler); there is
     NO noise-scheduler / num_inference_steps / DDIM override. We expose
     --num-integration-steps (hasattr-guarded) instead of --scheduler; default None
     keeps the checkpoint's trained value (100). A smaller value (e.g. 10) trades a
     little accuracy for a faster refill on the arm.
  3. NO EXTERNAL RESIZE. The FM policy resizes every frame to
     config.image_resize_shape (= [224, 224]) INTERNALLY (see image_preprocess_fm),
     so we feed the native-resolution RGB frame and drop the diffusion path's
     360x640 resize + resize_shape assert.
  Plus: the saved preprocessor TOKENIZES the `task` string (CLIP text conditioning),
  so -- unlike the single-task diffusion server which sent task="" -- we send the
  real task string ("put the right banana in the pot") on every tick.

Action spaces (CARROT_EEF_DESIGN §4). The wire format is v2: the RESET reply is a
handshake and the ACT reply carries `chunk_start`. `--action-space` (env
FM_ACTION_SPACE) selects what the 7-D action MEANS -- "joint" (q1..q6 + grip, the
banana contract above) or "eef_delta_v1" (base-frame Cartesian delta + grip, state
16). The server does NOT trust that flag: it reads the checkpoint's own
observation.state / action feature shapes and REFUSES TO START (exit 4) when they
contradict it. --task also defaults per action space, because the CLIP sentence
must be the one the dataset was trained with.

Run:
    python -m gello_policy.policy_server.fm_server \
        --checkpoint <ckpt_dir> --host 127.0.0.1 --port 5593 --device cuda \
        --n-action-steps 24 --num-integration-steps 100 \
        --action-space joint --task "put the right banana in the pot"
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

# lerobot 0.6.1 API. GENERIC load (change #1): resolve the policy class from the
# checkpoint's config instead of importing a concrete class. get_policy_class maps
# cfg.type "multi_task_dit" -> MultiTaskDiTPolicy (lerobot/policies/factory.py).
from lerobot.policies.factory import get_policy_class

# PreTrainedConfig.from_pretrained loads the checkpoint's config.json so we can
# mutate fields (num_integration_steps) BEFORE the model is built.
from lerobot.configs import PreTrainedConfig

# make_pre_post_processors: with pretrained_path it loads the saved processor
# pipelines (normalization stats + tokenizer baked in):
#   preprocessor : Rename -> Batch -> Tokenize(task) -> Device -> Normalize
#   postprocessor: Unnormalize -> Device(cpu)
from lerobot.policies import make_pre_post_processors

# ACTION = "action": key of the policy's internal action deque in policy._queues
# (modeling_multi_task_dit.py reset/select_action). Used for refill logging.
from lerobot.utils.constants import ACTION

# Default task strings for CLIP text conditioning. Single-task deploy, but the FM
# model IS text-conditioned, so this MUST be the phrasing the checkpoint was
# trained with -- which differs per dataset, hence per action space:
#   joint         -> the banana-in-pot joint checkpoint (unchanged)
#   eef_delta_v1  -> the carrot dataset's task string (CARROT_EEF_DESIGN §4)
# A wrong string is NOT an error anywhere in the stack; it just conditions the
# policy on the wrong sentence, so it has to be right by default.
DEFAULT_TASK = "put the right banana in the pot"
DEFAULT_TASK_EEF = "Put carrot in pot"

# Allow running both as a module (`-m gello_policy.policy_server.fm_server`) and as
# a bare script (scripts/run_fm_server.sh). Prefer the package-relative import.
try:
    from .image_preprocess_fm import decode_jpeg_to_rgb_float_chw
    from . import zmq_protocol as proto
except ImportError:  # pragma: no cover - fallback for direct-path execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from image_preprocess_fm import decode_jpeg_to_rgb_float_chw  # type: ignore
    import zmq_protocol as proto  # type: ignore


def default_task_for_space(action_space: str) -> str:
    """Trained task phrasing for an action space (see DEFAULT_TASK above)."""
    if action_space == proto.ACTION_SPACE_EEF:
        return DEFAULT_TASK_EEF
    return DEFAULT_TASK


# =============================================================================
# Policy wrapper
# =============================================================================
class FlowMatchingInferenceEngine:
    """Loads the MultiTaskDiTPolicy (flow_matching) + processors and turns
    (state, cam1, cam2) into a 7-dim action. Holds the receding-horizon queue across
    calls (via policy.reset())."""

    def __init__(self, checkpoint: str, device: str, n_action_steps: int,
                 num_integration_steps: int | None, task: str,
                 action_space: str = proto.ACTION_SPACE_JOINT) -> None:
        self.device = device
        self.n_action_steps = n_action_steps
        self.task = task
        self.checkpoint = checkpoint
        self.action_space = action_space
        # Set on every act() so _handle can report chunk_start without re-deriving it.
        self.last_chunk_start = False

        print(f"[fm_server] loading checkpoint: {checkpoint} on {device}", flush=True)
        # Load recipe: mutate the config BEFORE from_pretrained so the model is built
        # from it. GENERIC class resolution (change #1) -- no hardcoded policy import.
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        cfg.device = device

        # v2 handshake dims come from the CHECKPOINT's own features, never from the
        # CLI, so this is a real cross-check. Done on the CONFIG, before the model
        # (DiT + CLIP encoders) is built: a mismatch costs a second, not a GPU load.
        self.state_dim = int(cfg.input_features[proto.OBS_STATE_KEY].shape[0])
        self.action_dim = int(cfg.output_features[proto.POLICY_ACTION_FEATURE].shape[0])
        self.policy_type = str(getattr(cfg, "type", "?"))
        proto.enforce_action_space(action_space, self.state_dim, self.action_dim,
                                   checkpoint, "[fm_server]")
        print(f"[fm_server] action_space={action_space} state_dim={self.state_dim} "
              f"action_dim={self.action_dim} policy_type={self.policy_type}", flush=True)
        # Euler-ODE step count override (change #2). Flow matching integrates the
        # velocity field over cfg.num_integration_steps Euler steps -- there is NO
        # noise scheduler / DDIM here. Default None = keep the checkpoint's trained
        # value (100). hasattr-guarded so a non-FM checkpoint never crashes on it.
        if num_integration_steps is not None:
            if hasattr(cfg, "num_integration_steps"):
                cfg.num_integration_steps = num_integration_steps
            else:
                print(
                    f"[fm_server] WARNING: --num-integration-steps={num_integration_steps} "
                    f"ignored: config has no 'num_integration_steps' (not a flow-matching "
                    f"checkpoint?).",
                    flush=True,
                )
        PolicyCls = get_policy_class(cfg.type)  # "multi_task_dit" -> MultiTaskDiTPolicy
        print(f"[fm_server] policy class: {PolicyCls.__name__} (type={cfg.type}, "
              f"objective={getattr(cfg, 'objective', '?')})", flush=True)
        self.policy = PolicyCls.from_pretrained(checkpoint, config=cfg)
        self.policy.to(device)
        self.policy.eval()

        # Saved processor pipelines with normalization stats + tokenizer baked in.
        # device_processor override guarantees inputs land on the right device.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        # CRITICAL: set n_action_steps BEFORE reset(). reset() builds the action deque
        # with maxlen=n_action_steps (modeling_multi_task_dit.py reset), so the net
        # re-runs every n_action_steps calls -> receding horizon on fresh obs.
        self.policy.config.n_action_steps = n_action_steps
        print(
            f"[fm_server] n_obs_steps={self.policy.config.n_obs_steps} "
            f"horizon={self.policy.config.horizon} "
            f"n_action_steps={self.policy.config.n_action_steps} "
            f"num_integration_steps={getattr(self.policy.config, 'num_integration_steps', '?')} "
            f"integration_method={getattr(self.policy.config, 'integration_method', '?')} "
            f"image_resize_shape={getattr(self.policy.config, 'image_resize_shape', '?')}",
            flush=True,
        )

        # Load-time guard lerobot does NOT enforce itself: each net run predicts
        # `horizon` steps but the first n_obs_steps-1 overlap the observation window,
        # so at most horizon - n_obs_steps + 1 fresh actions exist per refill. A larger
        # n_action_steps would starve the queue. Fail loudly at load.
        horizon = self.policy.config.horizon
        n_obs_steps = self.policy.config.n_obs_steps
        assert n_action_steps <= horizon - n_obs_steps + 1, (
            f"n_action_steps ({n_action_steps}) must be <= horizon - n_obs_steps + 1 "
            f"({horizon} - {n_obs_steps} + 1 = {horizon - n_obs_steps + 1})"
        )

        # NOTE (change #3): NO resize_shape assert here. The diffusion server pre-resized
        # to 360x640 and asserted config.resize_shape matched; the FM policy resizes
        # INTERNALLY to config.image_resize_shape (= [224, 224]) and we feed native-res
        # frames, so there is no external-resize contract to enforce.

        print(f"[fm_server] task string: {self.task!r}", flush=True)

        self.reset()
        self._act_calls = 0
        self._refills = 0
        # Force CUDA kernel compilation / lazy init (incl. CLIP vision+text encoders)
        # NOW, so the operator's first real EXECUTE tick doesn't blow past the ROS
        # node's act_timeout_s and trip a spurious startup FAULT-then-retry.
        self._warmup()

    @torch.no_grad()
    def _warmup(self) -> None:
        print("[fm_server] warming up (compiling inference kernels)...", flush=True)
        # decode_jpeg_to_rgb_float_chw returns native-res frames; mirror the declared
        # camera resolution (3, 720, 1280). The policy resizes to 224 internally.
        zero_obs = {
            # self.state_dim (NOT proto.STATE_DIM): an eef_delta_v1 checkpoint wants
            # 16 and the normalizer would throw on a 7-vector.
            proto.OBS_STATE_KEY: torch.zeros(self.state_dim, dtype=torch.float32),
            proto.OBS_CAM1_KEY: torch.zeros(3, 720, 1280, dtype=torch.float32),
            proto.OBS_CAM2_KEY: torch.zeros(3, 720, 1280, dtype=torch.float32),
            proto.OBS_TASK_KEY: self.task,
        }
        t0 = time.time()
        for _ in range(2):
            proc = self.preprocessor(zero_obs)
            action = self.policy.select_action(proc)
            self.postprocessor(action)
        if self.device == "cuda":
            torch.cuda.synchronize()
        print(f"[fm_server] warmup done in {time.time() - t0:.1f}s", flush=True)
        self.reset()  # discard warmup queue/state; real episode starts on the next RESET

    def reset(self) -> None:
        """Clear the policy obs/action queues (episode start). modeling reset()."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def handshake(self) -> dict:
        """v2 RESET reply body: what this server is serving (zmq_protocol §wire)."""
        return proto.handshake(self.action_space, self.state_dim, self.action_dim,
                               self.policy_type, self.checkpoint)

    @torch.no_grad()
    def act(self, state: np.ndarray, cam1_jpeg: bytes, cam2_jpeg: bytes) -> np.ndarray:
        """Run one inference tick. Returns a (ACTION_DIM,) float64 numpy action --
        [q1..q6 radians, grip_cmd 0..1] in joint space, [dx,dy,dz,drx,dry,drz,grip]
        in eef_delta_v1. Also sets self.last_chunk_start."""
        # Build obs dict with EXACT dataset feature keys. Images are RGB, CHW, float32
        # [0,1] at NATIVE resolution (BGR->RGB here, resize to 224 done INSIDE the
        # policy). State is float32 (state_dim,). task = trained phrasing (CLIP text).
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"state must be length {self.state_dim} for action_space="
                f"{self.action_space}, got shape {state.shape}")
        obs = {
            proto.OBS_STATE_KEY: torch.from_numpy(state.astype(np.float32)),  # (S,)
            proto.OBS_CAM1_KEY: decode_jpeg_to_rgb_float_chw(cam1_jpeg),      # (3,H,W)
            proto.OBS_CAM2_KEY: decode_jpeg_to_rgb_float_chw(cam2_jpeg),      # (3,H,W)
            proto.OBS_TASK_KEY: self.task,
        }

        # Log net-refills: the policy only re-runs the full Euler ODE integration when
        # its action deque (policy._queues[ACTION]) is empty. The SAME boolean is
        # reported to the client as chunk_start (v2): it marks the tick whose
        # observation the whole next chunk was predicted from, which is where the EEF
        # executor re-anchors its integration target (DESIGN §3.3).
        refill = len(self.policy._queues[ACTION]) == 0
        self.last_chunk_start = refill

        proc = self.preprocessor(obs)               # rename -> batch -> tokenize -> device -> normalize
        action = self.policy.select_action(proc)    # (1,7) normalized
        action = self.postprocessor(action)         # (1,7) unnormalized, on cpu
        action = action.squeeze(0).cpu().numpy().astype(np.float64)  # (7,)

        self._act_calls += 1
        if refill:
            self._refills += 1
            print(
                f"[fm_server] net refill #{self._refills} "
                f"(act call {self._act_calls}, horizon={self.n_action_steps})",
                flush=True,
            )
        return action


# =============================================================================
# ZMQ REP loop
# =============================================================================
def _reply(sock: "zmq.Socket", payload: dict) -> None:
    """Send a single-frame JSON reply."""
    sock.send_multipart([json.dumps(payload).encode("utf-8")])


def serve(engine: FlowMatchingInferenceEngine, host: str, port: int) -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    endpoint = proto.default_endpoint(host, port)
    sock.bind(endpoint)
    print(f"[fm_server] REP bound at {endpoint}; waiting for requests...", flush=True)

    try:
        while True:
            frames = sock.recv_multipart()
            _reply(sock, handle_request(engine, frames))
    except KeyboardInterrupt:
        print("\n[fm_server] interrupted; shutting down.", flush=True)
    finally:
        sock.close(linger=0)


def handle_request(engine: FlowMatchingInferenceEngine, frames: list) -> dict:
    """_handle + the never-die error envelope. The REP loop must answer EVERY
    request (a REQ socket blocks forever otherwise), so a bad frame becomes an
    {ok:false, err} reply rather than an exception."""
    try:
        return _handle(engine, frames)
    except Exception as exc:  # never die on a bad request; report it
        payload = {proto.KEY_OK: False, proto.KEY_ERR: f"{type(exc).__name__}: {exc}"}
        print(f"[fm_server] ERROR handling request: {payload[proto.KEY_ERR]}", flush=True)
        return payload


def _handle(engine: FlowMatchingInferenceEngine, frames: list) -> dict:
    """Parse one multipart request and produce the reply dict. Raises on malformed
    input (caller turns the exception into an {ok:false,err} reply)."""
    if not frames:
        raise ValueError("empty request (no frames)")

    ctrl = json.loads(frames[0].decode("utf-8"))
    cmd = ctrl.get(proto.KEY_CMD)

    if cmd == proto.CMD_RESET:
        engine.reset()
        # v2: RESET is the handshake. The client refuses to arm unless these match
        # what it is configured to pack/unpack (DESIGN §3.7).
        return engine.handshake()

    if cmd == proto.CMD_ACT:
        if len(frames) != proto.ACT_REQUEST_NFRAMES:
            raise ValueError(
                f"act request needs {proto.ACT_REQUEST_NFRAMES} frames "
                f"[ctrl, cam1, cam2], got {len(frames)}"
            )
        state = np.asarray(ctrl.get(proto.KEY_STATE), dtype=np.float64)
        # Validate against the CHECKPOINT's state_dim, not the joint-space literal.
        if state.shape != (engine.state_dim,):
            raise ValueError(f"state must be length {engine.state_dim}, got shape {state.shape}")
        cam1_jpeg, cam2_jpeg = frames[1], frames[2]

        action = engine.act(state, cam1_jpeg, cam2_jpeg)
        if action.shape != (proto.ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(f"policy returned invalid action: shape={action.shape}, "
                             f"finite={np.all(np.isfinite(action))}")
        return {proto.KEY_OK: True,
                proto.KEY_ACTION: [float(x) for x in action],
                proto.KEY_CHUNK_START: bool(engine.last_chunk_start)}

    raise ValueError(f"unknown cmd: {cmd!r}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=os.environ.get("FM_CHECKPOINT"),
                   help="Path to the pretrained_model dir (config.json + "
                        "model.safetensors + policy_pre/postprocessor.json).")
    p.add_argument("--host", default=os.environ.get("FM_HOST", proto.DEFAULT_HOST))
    p.add_argument("--port", type=int, default=int(os.environ.get("FM_PORT", "5593")))
    p.add_argument("--device", default=os.environ.get("FM_DEVICE", "cuda"),
                   choices=["cuda", "cpu"])
    p.add_argument("--n-action-steps", type=int,
                   default=int(os.environ.get("FM_N_ACTION_STEPS", "24")),
                   help="Receding horizon: net re-runs every N select_action calls.")
    p.add_argument("--num-integration-steps", type=int,
                   default=(int(os.environ["FM_NUM_INTEGRATION_STEPS"])
                            if os.environ.get("FM_NUM_INTEGRATION_STEPS") else None),
                   help="Euler ODE steps per refill. Default (unset) keeps the "
                        "checkpoint's trained value (100); lower (e.g. 10) = faster refill.")
    p.add_argument("--action-space",
                   default=os.environ.get("FM_ACTION_SPACE", proto.ACTION_SPACE_JOINT),
                   choices=list(proto.ACTION_SPACES),
                   help="What the 7-D action MEANS. 'joint' (default, the original "
                        "contract) = absolute q1..q6 + grip; 'eef_delta_v1' = base-frame "
                        "Cartesian delta + grip with a 16-D state. Cross-checked against "
                        "the checkpoint's feature shapes; a mismatch exits 4.")
    # NOTE: default is None, not DEFAULT_TASK -- the task default DEPENDS on
    # --action-space, which argparse has not resolved yet. main() fills it in via
    # default_task_for_space(). An explicit flag or $FM_TASK still wins.
    p.add_argument("--task", default=os.environ.get("FM_TASK") or None,
                   help="Task string for CLIP text conditioning (the FM model is "
                        "text-conditioned). Default: the trained phrasing for the "
                        f"chosen action space ({DEFAULT_TASK!r} for joint, "
                        f"{DEFAULT_TASK_EEF!r} for eef_delta_v1).")
    return p.parse_args(argv)


def resolve_device(device: str) -> str:
    # Do NOT silently fall back to CPU. A CPU FM refill (full Euler ODE integration
    # through a DiT + CLIP encoders) can exceed the ROS leader's act_timeout_s and trip
    # a spurious FAULT at every net refill. Force an explicit choice.
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "[fm_server] ERROR: --device cuda but CUDA is unavailable. Refusing to "
            "silently run on CPU (a CPU refill can exceed the leader's timeout "
            "and FAULT-loop). Fix CUDA, or pass --device cpu AND raise the leader's "
            "act_timeout_s deliberately.",
            file=sys.stderr, flush=True,
        )
        raise SystemExit(3)
    return device


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.checkpoint:
        print("[fm_server] --checkpoint (or $FM_CHECKPOINT) is required.", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    # Resolve the per-action-space task default here (argparse could not: it depends
    # on another flag). Print it -- a silently wrong CLIP sentence is invisible
    # everywhere else in the stack.
    task = args.task if args.task else default_task_for_space(args.action_space)
    print(f"[fm_server] action_space={args.action_space} task={task!r}"
          f"{'' if args.task else ' (default for this action space)'}", flush=True)
    t0 = time.time()
    engine = FlowMatchingInferenceEngine(args.checkpoint, device, args.n_action_steps,
                                         args.num_integration_steps, task,
                                         action_space=args.action_space)
    print(f"[fm_server] ready in {time.time() - t0:.1f}s", flush=True)
    serve(engine, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
