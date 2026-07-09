#!/usr/bin/env python3
"""py3.12 Diffusion Policy inference server for the UR7e "put right banana in pot" deploy.

Owns the trained DiffusionPolicy and its internal receding-horizon action queue. A
py3.10 rclpy `policy_leader_node` (different venv/distro) drives it over localhost
ZMQ REQ/REP multipart -- see BUILD_SPEC.md §3/§4 and zmq_protocol.py for the wire
format. This process does ALL the torch/lerobot/cv2 work; the ROS side stays pure
rclpy and forwards raw JPEG bytes.

Inference mirrors eval_offline.py (apply_scheduler_overrides +
load_policy_and_processors) VERBATIM in spirit (the ground-truth for correct
diffusion inference in lerobot 0.6.1):
    obs dict -> preprocessor(rename/batch/device/normalize)
             -> policy.select_action  (pops from the diffusion action queue; re-runs
                the denoiser only when the queue empties -> receding horizon
                k = n_action_steps)
             -> postprocessor(unnormalize -> cpu) -> (7,) numpy.

Key correctness points (diffusion-specific, see BUILD_SPEC [IMPL-1]):
  * DDIM OVERRIDE AT LOAD: the checkpoint was TRAINED with DDPM/100 denoising
    steps. A plain `DiffusionPolicy.from_pretrained(checkpoint)` would silently
    keep that scheduler and make every queue refill ~10x slower -- a latency bug
    on the arm. We therefore mutate the loaded PreTrainedConfig
    (noise_scheduler_type=DDIM, num_inference_steps=10 by default) BEFORE
    `from_pretrained(checkpoint, config=cfg)`, so the noise scheduler is rebuilt
    from the overridden config at model init. DDIM is a valid sampler for a
    DDPM-trained epsilon model (same beta schedule). `--scheduler asis` keeps the
    trained scheduler deliberately.
  * REFILL COST: a refill tick runs the FULL DDIM-N sampling loop (N reverse
    diffusion passes through the U-Net), which is slower than ACT's single
    transformer forward. The leader's act_timeout_s must tolerate it -- run
    scripts/benchmark_diffusion_latency.py on the robot PC before the first arm
    run, and reduce --num-inference-steps if the p99 refill exceeds the budget.
  * The saved preprocessor has NO resize / NO BGR->RGB. image_preprocess.py does
    both (resize to 360x640, BGR->RGB) BEFORE the saved preprocessor runs.
  * `policy.config.n_action_steps` is set BEFORE the first `policy.reset()`,
    because reset() builds the action deque with maxlen=n_action_steps.
  * obs keys are EXACTLY observation.state / observation.images.cam1 /
    observation.images.cam2 / task. First real tick works immediately: lerobot
    pads the n_obs_steps=2 obs queue by repeating the first obs.

Run:
    python -m gello_policy.policy_server.diffusion_server \
        --checkpoint <ckpt_dir> --host 127.0.0.1 --port 5592 --device cuda \
        --n-action-steps 32 --num-inference-steps 10 --scheduler DDIM
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
# DiffusionPolicy: lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py
#   (from_pretrained -> pretrained.py:162 loads config.json + model.safetensors;
#   passing config= rebuilds the noise scheduler from the OVERRIDDEN config).
from lerobot.policies.diffusion import DiffusionPolicy

# PreTrainedConfig.from_pretrained loads the checkpoint's config.json so we can
# mutate scheduler fields BEFORE the model is built (eval_offline.py recipe).
from lerobot.configs import PreTrainedConfig

# make_pre_post_processors: lerobot/src/lerobot/policies/factory.py:273 -- with
# pretrained_path it loads the saved processor pipelines (Normalizer stats baked in):
#   preprocessor : Rename -> AddBatchDim -> Device -> Normalize
#   postprocessor: Unnormalize -> Device(cpu)
from lerobot.policies import make_pre_post_processors

# ACTION = "action": key of the diffusion policy's internal action deque in
# policy._queues (modeling_diffusion.py reset/select_action). Used for refill logging.
from lerobot.utils.constants import ACTION

# Allow running both as a module (`-m gello_policy.policy_server.diffusion_server`) and as
# a bare script (scripts/run_diffusion_server.sh). Prefer the package-relative import.
try:
    from .image_preprocess import decode_jpeg_to_rgb_float_chw, RESIZE_HW
    from . import zmq_protocol as proto
except ImportError:  # pragma: no cover - fallback for direct-path execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from image_preprocess import decode_jpeg_to_rgb_float_chw, RESIZE_HW  # type: ignore
    import zmq_protocol as proto  # type: ignore


# =============================================================================
# Policy wrapper
# =============================================================================
class DiffusionInferenceEngine:
    """Loads the DiffusionPolicy + processors and turns (state, cam1, cam2) into a 7-dim
    action. Holds the receding-horizon queue across calls (via policy.reset())."""

    def __init__(self, checkpoint: str, device: str, n_action_steps: int,
                 num_inference_steps: int, scheduler: str) -> None:
        self.device = device
        self.n_action_steps = n_action_steps

        print(f"[diffusion_server] loading checkpoint: {checkpoint} on {device}", flush=True)
        # THE critical load recipe (eval_offline.load_policy_and_processors +
        # apply_scheduler_overrides): mutate the config BEFORE from_pretrained so the
        # noise scheduler is (re)built from it at model init. A plain
        # from_pretrained(checkpoint) WITHOUT config= silently keeps the trained
        # DDPM/100-step scheduler -> ~10x slower refills on the arm.
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        cfg.device = device
        # NOTE: apply BOTH scheduler + step-count overrides ONLY for a non-"asis" choice,
        # mirroring eval_offline.apply_scheduler_overrides (which no-ops num_inference_steps
        # when it is None). Otherwise `--scheduler asis` + the default --num-inference-steps 10
        # would silently give DDPM-with-only-10-reverse-steps (severely under-denoised, OOD
        # actions on the arm) instead of the trained DDPM/100. "asis" now truly = as trained.
        if scheduler != "asis":
            cfg.noise_scheduler_type = scheduler       # "DDIM" (or "DDPM" explicitly)
            cfg.num_inference_steps = num_inference_steps  # e.g. 10 for DDIM
        # Skip the torchvision ImageNet backbone download: from_pretrained overwrites the
        # whole state_dict (backbone included) with the checkpoint's trained weights, so the
        # ImageNet init is redundant. Setting this to None removes a startup NETWORK
        # dependency (download.pytorch.org) + speeds load on the robot PC.
        cfg.pretrained_backbone_weights = None
        self.policy = DiffusionPolicy.from_pretrained(checkpoint, config=cfg)
        self.policy.to(device)
        self.policy.eval()

        # Saved processor pipelines with normalization stats baked in (factory.py:273).
        # device_processor override guarantees inputs land on the right device, like
        # eval_offline.load_policy_and_processors. (No dataset_stats needed -- stats
        # are baked into the saved processor jsons.)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        # CRITICAL: set n_action_steps BEFORE reset(). reset() builds the action deque
        # with maxlen=n_action_steps (modeling_diffusion.py DiffusionPolicy.reset), so
        # the net re-runs every n_action_steps calls -> receding horizon on fresh obs.
        self.policy.config.n_action_steps = n_action_steps
        print(
            f"[diffusion_server] n_obs_steps={self.policy.config.n_obs_steps} "
            f"horizon={self.policy.config.horizon} "
            f"n_action_steps={self.policy.config.n_action_steps} "
            f"num_inference_steps={self.policy.config.num_inference_steps} "
            f"noise_scheduler_type={self.policy.config.noise_scheduler_type}",
            flush=True,
        )

        # Load-time guard lerobot does NOT enforce itself: each denoiser run predicts
        # `horizon` steps but the first n_obs_steps-1 overlap the observation window,
        # so at most horizon - n_obs_steps + 1 fresh actions exist per refill. A larger
        # n_action_steps would starve the queue. Fail loudly at load rather than
        # mis-behave on the arm.
        horizon = self.policy.config.horizon
        n_obs_steps = self.policy.config.n_obs_steps
        assert n_action_steps <= horizon - n_obs_steps + 1, (
            f"n_action_steps ({n_action_steps}) must be <= horizon - n_obs_steps + 1 "
            f"({horizon} - {n_obs_steps} + 1 = {horizon - n_obs_steps + 1})"
        )

        # Load-bearing, unguarded-by-lerobot: image_preprocess.py pre-resizes every frame to
        # 360x640, and the RGB encoder re-applies config.resize_shape each forward. If this
        # checkpoint were trained with a different resize_shape, the SpatialSoftmax grid
        # (sized from resize_shape at init) would mismatch the pre-resized input -> crash /
        # silent degradation on the first tick. Confirm the contract at load.
        rs = getattr(self.policy.config, "resize_shape", None)
        assert rs is not None and list(rs) == list(RESIZE_HW), (
            f"checkpoint resize_shape={rs} but the server pre-resizes to {RESIZE_HW}; "
            f"they must match (retrain or fix image_preprocess.RESIZE_HW)."
        )

        self.reset()
        self._act_calls = 0
        self._refills = 0
        # Review H1: force CUDA kernel compilation / lazy init NOW, so the operator's
        # first real EXECUTE tick doesn't blow past the ROS node's act_timeout_s
        # and trip a spurious startup FAULT-then-retry. Runs a couple of dummy inferences
        # on zero obs, then clears the queue so the first real episode starts fresh.
        self._warmup()

    @torch.no_grad()
    def _warmup(self) -> None:
        print("[diffusion_server] warming up (compiling inference kernels)...", flush=True)
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
        print(f"[diffusion_server] warmup done in {time.time() - t0:.1f}s", flush=True)
        self.reset()  # discard warmup queue/state; real episode starts on the next RESET

    def reset(self) -> None:
        """Clear the diffusion obs/action queues (episode start). modeling_diffusion.py reset()."""
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
            proto.OBS_TASK_KEY: "",  # single-task; empty string (deploy_ur_act.py)
        }

        # Log net-refills: the diffusion policy only re-runs full DDIM-N sampling when
        # its action deque (policy._queues[ACTION], modeling_diffusion.py) is empty.
        refill = len(self.policy._queues[ACTION]) == 0

        proc = self.preprocessor(obs)               # rename -> batch -> device -> normalize
        action = self.policy.select_action(proc)    # (1,7) normalized (modeling_diffusion.py)
        action = self.postprocessor(action)         # (1,7) unnormalized, on cpu
        action = action.squeeze(0).cpu().numpy().astype(np.float64)  # (7,)

        self._act_calls += 1
        if refill:
            self._refills += 1
            print(
                f"[diffusion_server] net refill #{self._refills} "
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


def serve(engine: DiffusionInferenceEngine, host: str, port: int) -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    endpoint = proto.default_endpoint(host, port)
    sock.bind(endpoint)
    print(f"[diffusion_server] REP bound at {endpoint}; waiting for requests...", flush=True)

    try:
        while True:
            frames = sock.recv_multipart()
            try:
                payload = _handle(engine, frames)
            except Exception as exc:  # never die on a bad request; report it
                payload = {proto.KEY_OK: False, proto.KEY_ERR: f"{type(exc).__name__}: {exc}"}
                print(f"[diffusion_server] ERROR handling request: {payload[proto.KEY_ERR]}", flush=True)
            _reply(sock, payload)
    except KeyboardInterrupt:
        print("\n[diffusion_server] interrupted; shutting down.", flush=True)
    finally:
        sock.close(linger=0)


def _handle(engine: DiffusionInferenceEngine, frames: list) -> dict:
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
    p.add_argument("--checkpoint", default=os.environ.get("DIFFUSION_CHECKPOINT"),
                   help="Path to the pretrained_model dir (config.json + "
                        "model.safetensors + policy_pre/postprocessor.json).")
    p.add_argument("--host", default=os.environ.get("DIFFUSION_HOST", proto.DEFAULT_HOST))
    p.add_argument("--port", type=int, default=int(os.environ.get("DIFFUSION_PORT", "5592")))
    p.add_argument("--device", default=os.environ.get("DIFFUSION_DEVICE", "cuda"),
                   choices=["cuda", "cpu"])
    p.add_argument("--n-action-steps", type=int,
                   default=int(os.environ.get("DIFFUSION_N_ACTION_STEPS", "32")),
                   help="Receding horizon: net re-runs every N select_action calls.")
    p.add_argument("--num-inference-steps", type=int,
                   default=int(os.environ.get("DIFFUSION_NUM_INFERENCE_STEPS", "10")),
                   help="Denoising steps per refill (DDIM: ~10 is enough; lower = faster).")
    p.add_argument("--scheduler", default=os.environ.get("DIFFUSION_SCHEDULER", "DDIM"),
                   choices=["DDIM", "DDPM", "asis"],
                   help="Noise scheduler override; 'asis' keeps the trained scheduler "
                        "(DDPM/100 for this checkpoint -- much slower refills).")
    return p.parse_args(argv)


def resolve_device(device: str) -> str:
    # Review: do NOT silently fall back to CPU. A CPU diffusion refill (full DDIM-N
    # sampling) can exceed the ROS leader's act_timeout_s and trip a spurious FAULT
    # at every net refill. Force an explicit choice: the operator must pass
    # --device cpu (and raise the timeout) knowingly.
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "[diffusion_server] ERROR: --device cuda but CUDA is unavailable. Refusing to "
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
        print("[diffusion_server] --checkpoint (or $DIFFUSION_CHECKPOINT) is required.", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    t0 = time.time()
    engine = DiffusionInferenceEngine(args.checkpoint, device, args.n_action_steps,
                                      args.num_inference_steps, args.scheduler)
    print(f"[diffusion_server] ready in {time.time() - t0:.1f}s", flush=True)
    serve(engine, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
