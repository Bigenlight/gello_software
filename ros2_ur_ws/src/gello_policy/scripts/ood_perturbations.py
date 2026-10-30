#!/usr/bin/env python3
"""Controlled out-of-distribution (OOD) perturbations for the offline variance study.

WHY THIS FILE EXISTS
--------------------
The offline study wants to test one half of the ensemble-uncertainty hypothesis:

    "does across-sample variance actually RISE on out-of-distribution observations?"

All 51 demos in `Bigenlight/banana_in_pot_lerobot_v3` are *successes*; there are no
failure labels, so we cannot correlate variance against failure offline. What we CAN
do is manufacture a controlled distribution shift: take a real in-distribution (ID)
frame, apply a perturbation of known type and known severity, and measure how the
K-sample ensemble variance moves. If variance is a usable OOD signal it must be
monotone (or at least clearly increasing) in severity. If it is flat, the signal is
dead and no amount of on-robot tuning will resuscitate it.

WHERE THESE PERTURBATIONS ARE APPLIED (the injection point)
----------------------------------------------------------
They are injected at *exactly* the point where the live server hands an obs dict to
the saved lerobot preprocessor, i.e. AFTER the dataset-side preprocessing (resize /
BGR->RGB / [0,1] scaling) and BEFORE the checkpoint's own normalization.

Chain in the real pipeline, with line numbers:

  policy_server/image_preprocess.py
    L56-60  cv2.imdecode(jpeg)                  -> BGR uint8 HWC
    L63     cv2.cvtColor(BGR2RGB)               -> RGB uint8 HWC
    L67-68  permute(2,0,1).float().div_(255.)   -> RGB float32 CHW in [0,1]
    L39,L71 _RESIZE = v2.Resize(size=[360,640]) -> (3, 360, 640) float32 [0,1]

  policy_server/diffusion_server.py
    L352-357  obs = {observation.state: float32 (7,) RAW RADIANS + grip 0..1,
                     observation.images.cam1: (3,360,640) float32 [0,1] RGB,
                     observation.images.cam2: (3,360,640) float32 [0,1] RGB,
                     task: ""}
    <<<<<<<<  PERTURBATIONS GO HERE  >>>>>>>>
    L363      proc = self.preprocessor(obs)   # rename -> batch -> device -> NORMALIZE
    L364      action = self.policy.select_action(proc)
    L370      action = self.postprocessor(action)

So every image callable in this module consumes and returns a float32 CHW tensor of
shape (3, 360, 640) with values in [0, 1] (clamped), and the state callable consumes
and returns a float32 (7,) tensor in RAW UNITS (radians for [0:6], 0..1 grip for [6]).
That is the ONLY contract. The checkpoint's MEAN_STD (visual) / MIN_MAX (state,
action) normalization happens downstream inside the saved preprocessor
(policy_preprocessor_step_3_normalizer_processor.safetensors) and is untouched here.

This file is a pure library + a self-check `__main__`. It imports nothing from the
real-time control path and modifies nothing. `policy_leader_node.py`, `zmq_protocol.py`,
`act_server.py`, `fm_server.py`, `camera_viewer.py`, `diffusion_server.py` and all
launch files are left alone; the runner is expected to import this module and apply
the perturbation to its own copy of the obs dict.

DETERMINISM
-----------
Every callable takes `seed: int`. Stochastic ops (gaussian noise, noise-filled
occlusion, random-direction state offset) draw from a `torch.Generator` seeded with a
stable mix of (seed, op name), so the same (name, severity, seed) always produces the
same output, and two different ops with the same seed do not share a noise stream.
`severity == 0.0` is an exact identity for every registered perturbation (asserted in
`_selftest`).

USAGE
-----
    from ood_perturbations import PERTURBATIONS, severity_sweep, apply_perturbation

    for name, sev, fn in severity_sweep(seed=0):
        obs_ood = fn(obs)              # obs dict in, perturbed obs dict out
        ...run the K-sample ensemble, record variance...
"""

from __future__ import annotations

import math
import os
import sys
import zlib
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Iterable, Iterator, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TVF

# --------------------------------------------------------------------------------------
# Contract constants -- these mirror the live pipeline, do not drift from them.
# --------------------------------------------------------------------------------------

# image_preprocess.py L33 RESIZE_HW / checkpoint config.json resize_shape.
RESIZE_HW: Tuple[int, int] = (360, 640)

# zmq_protocol.py L48-51.
OBS_STATE_KEY = "observation.state"
OBS_CAM1_KEY = "observation.images.cam1"
OBS_CAM2_KEY = "observation.images.cam2"
OBS_TASK_KEY = "task"
CAMERA_KEYS: Tuple[str, ...] = (OBS_CAM1_KEY, OBS_CAM2_KEY)

# Per-joint std of observation.state over all 21524 dataset frames (from
# policy_preprocessor_step_3_normalizer_processor.safetensors, "observation.state.std").
# Used so a state offset can be expressed in dataset-sigma units rather than raw radians
# -- ur_q6 ranges over 3.08 rad while ur_q5 covers 0.69 rad, so a flat radian offset
# would be a wildly different amount of "surprise" per joint.
STATE_STD = torch.tensor(
    [0.17647733, 0.23182477, 0.25253725, 0.38063741, 0.14290138, 0.57726365, 0.24033694],
    dtype=torch.float32,
)

# observation.state MIN_MAX bounds the checkpoint was normalized with. Offsets that push
# the state outside these are, by construction, outside anything the policy ever saw.
STATE_MIN = torch.tensor(
    [2.5063789, -2.3910921, 1.2002540, -3.1413319, -2.1029339, -4.9144740, 0.0118],
    dtype=torch.float32,
)
STATE_MAX = torch.tensor(
    [3.5618761, -1.0215520, 2.4876161, -1.2666230, -1.4157391, -1.8315190, 0.8980],
    dtype=torch.float32,
)

# Default severity grid. 0.0 is included on purpose: it is the ID control arm, and it
# must reproduce the unperturbed variance exactly.
DEFAULT_SEVERITIES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------

def _generator(seed: int, tag: str) -> torch.Generator:
    """Stable per-(seed, op) CPU generator.

    zlib.crc32 is used rather than the builtin `hash()` because `hash()` on str is
    salted per process (PYTHONHASHSEED) and would silently break reproducibility
    across runs.
    """
    mix = (int(seed) * 0x9E3779B1 + zlib.crc32(tag.encode("utf-8"))) & 0x7FFFFFFF
    g = torch.Generator(device="cpu")
    g.manual_seed(mix)
    return g


def _check_image(img: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"{name}: expected torch.Tensor, got {type(img)!r}")
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"{name}: expected CHW with C=3, got {tuple(img.shape)}")
    return img


def _clamp01(img: torch.Tensor) -> torch.Tensor:
    """Keep the tensor inside the [0,1] range a real JPEG decode can produce.

    Without this a bright/noise perturbation would hand the checkpoint's MEAN_STD
    normalizer values no camera could ever emit, which conflates "OOD scene" with
    "impossible tensor" and would inflate variance for the wrong reason.
    """
    return img.clamp_(0.0, 1.0) if img.is_contiguous() else img.clamp(0.0, 1.0)


# --------------------------------------------------------------------------------------
# Image-space perturbations. Signature: (img (3,H,W) float32 [0,1], severity, seed) -> same
# --------------------------------------------------------------------------------------

def brightness(img: torch.Tensor, severity: float, seed: int = 0, *,
               sign: float = 1.0, max_delta: float = 0.35) -> torch.Tensor:
    """Additive luminance shift. severity 1.0 -> +/-0.35 on a [0,1] scale.

    Simulates the lab lights being changed / a window blind opening. Additive rather
    than multiplicative so it also moves the black point, which a multiplicative gain
    leaves pinned and which the ImageNet MEAN_STD normalizer is sensitive to.
    """
    _check_image(img, "brightness")
    if severity == 0.0:
        return img.clone()
    return _clamp01(img.clone() + sign * float(severity) * max_delta)


def contrast(img: torch.Tensor, severity: float, seed: int = 0, *,
             sign: float = 1.0, max_gain: float = 0.7) -> torch.Tensor:
    """Contrast scaling about the per-image mean.

    sign=+1 -> up to 1.7x contrast; sign=-1 -> down to 0.3x (washed out).
    """
    _check_image(img, "contrast")
    if severity == 0.0:
        return img.clone()
    gain = 1.0 + sign * float(severity) * max_gain
    mean = img.mean()
    return _clamp01((img - mean) * gain + mean)


def gaussian_noise(img: torch.Tensor, severity: float, seed: int = 0, *,
                   max_std: float = 0.20) -> torch.Tensor:
    """i.i.d. per-pixel per-channel gaussian noise. severity 1.0 -> std 0.20.

    Simulates sensor gain noise / a dirty or low-light camera.
    """
    _check_image(img, "gaussian_noise")
    if severity == 0.0:
        return img.clone()
    g = _generator(seed, f"gaussian_noise:{severity}")
    noise = torch.randn(img.shape, generator=g, dtype=torch.float32) * (float(severity) * max_std)
    return _clamp01(img + noise)


def occlusion(img: torch.Tensor, severity: float, seed: int = 0, *,
              center: Tuple[float, float] = (0.5, 0.5),
              fill: str = "black",
              max_area_frac: float = 0.30) -> torch.Tensor:
    """Rectangular occluder -- the gripper, an operator's hand, a dropped object.

    Args:
        center: (cy, cx) as fractions of H, W. (0.5, 0.75) sits over the pot/gripper
            region of cam1; (0.5, 0.5) is dead centre.
        fill: "black" (hard occluder) or "noise" (uniform noise patch, closer to a
            blurred hand sweeping past than a matte black card).
        max_area_frac: fraction of the frame covered at severity 1.0.

    The patch is a square in *area* terms: side = sqrt(sev * max_area_frac) * sqrt(H*W),
    so severity is linear in occluded area, which is the quantity that matters.
    """
    _check_image(img, "occlusion")
    if severity == 0.0:
        return img.clone()
    _, H, W = img.shape
    area = float(severity) * max_area_frac * H * W
    side = int(round(math.sqrt(max(area, 1.0))))
    cy = int(round(center[0] * H))
    cx = int(round(center[1] * W))
    y0 = max(0, cy - side // 2)
    x0 = max(0, cx - side // 2)
    y1 = min(H, y0 + side)
    x1 = min(W, x0 + side)
    out = img.clone()
    if fill == "black":
        out[:, y0:y1, x0:x1] = 0.0
    elif fill == "noise":
        g = _generator(seed, f"occlusion_noise:{severity}:{center}")
        patch = torch.rand((3, y1 - y0, x1 - x0), generator=g, dtype=torch.float32)
        out[:, y0:y1, x0:x1] = patch
    else:
        raise ValueError(f"occlusion: fill must be 'black' or 'noise', got {fill!r}")
    return out


def shift(img: torch.Tensor, severity: float, seed: int = 0, *,
          axis: str = "x", sign: float = 1.0, max_frac: float = 0.12) -> torch.Tensor:
    """Camera translation, approximated by an image-plane translation.

    FAIL-Detect used a physical ~10 cm camera move as its OOD condition. Offline we
    only have the recorded frames, so the honest analogue is a translation + edge
    replication: it reproduces the dominant first-order effect (every object lands on
    different pixels, the policy's spatial-softmax keypoints move) but NOT parallax,
    NOT occlusion changes, and NOT the lighting change a real camera move causes.
    Treat this as a lower bound on a real camera move, not a substitute for it.

    Edge-replicate padding is used instead of zero padding so the perturbation is a
    translation and not a translation-plus-black-bar; a black bar is a much stronger
    and much less realistic distribution shift.

    Args:
        axis: "x" (horizontal) or "y" (vertical).
        max_frac: shift at severity 1.0, as a fraction of that axis' size
            (0.12 * 640 = 77 px horizontally at 360x640).
    """
    _check_image(img, "shift")
    if severity == 0.0:
        return img.clone()
    _, H, W = img.shape
    span = W if axis == "x" else H
    px = int(round(sign * float(severity) * max_frac * span))
    if px == 0:
        return img.clone()
    a = abs(px)
    if axis == "x":
        pad = (a, a, 0, 0)
    elif axis == "y":
        pad = (0, 0, a, a)
    else:
        raise ValueError(f"shift: axis must be 'x' or 'y', got {axis!r}")
    padded = F.pad(img.unsqueeze(0), pad, mode="replicate").squeeze(0)
    if axis == "x":
        x0 = a - px
        out = padded[:, :, x0:x0 + W]
    else:
        y0 = a - px
        out = padded[:, y0:y0 + H, :]
    return out.contiguous()


def blur(img: torch.Tensor, severity: float, seed: int = 0, *,
         max_sigma: float = 6.0) -> torch.Tensor:
    """Gaussian blur -- defocus, a smudged lens, or motion during a fast reach.

    Kernel size is tied to sigma (~4 sigma each side, forced odd) so the kernel never
    truncates the gaussian and the effect stays monotone in severity.
    """
    _check_image(img, "blur")
    if severity == 0.0:
        return img.clone()
    sigma = max(float(severity) * max_sigma, 1e-3)
    k = int(2 * round(2.0 * sigma) + 1)
    k = max(k, 3)
    out = TVF.gaussian_blur(img.unsqueeze(0), kernel_size=[k, k], sigma=[sigma, sigma])
    return _clamp01(out.squeeze(0))


# YIQ hue-rotation matrices. Dependency-free, standard, and linear -- so it composes
# predictably with the downstream MEAN_STD normalization.
_RGB2YIQ = torch.tensor(
    [[0.299, 0.587, 0.114],
     [0.5959, -0.2746, -0.3213],
     [0.2115, -0.5227, 0.3112]], dtype=torch.float32
)
_YIQ2RGB = torch.tensor(
    [[1.0, 0.956, 0.619],
     [1.0, -0.272, -0.647],
     [1.0, -1.106, 1.703]], dtype=torch.float32
)

# Default region for `color_shift_region`, as (y0, x0, y1, x1) fractions of (H, W).
# Hand-picked on a real cam1 frame (episode 0, t~8s) to sit over the free banana lying
# on the table -- the object the task text singles out ("put the right banana in the
# pot"). Recolouring exactly that object is the closest offline stand-in for swapping in
# a visually different distractor.
DEFAULT_COLOR_REGION: Tuple[float, float, float, float] = (0.25, 0.35, 0.36, 0.51)


def color_shift_region(img: torch.Tensor, severity: float, seed: int = 0, *,
                       region: Tuple[float, float, float, float] = DEFAULT_COLOR_REGION,
                       max_deg: float = 150.0) -> torch.Tensor:
    """Rotate hue inside one bounding box: the "wrong banana" / distractor probe.

    Only the target region changes, so global statistics barely move -- this isolates
    *semantic* OOD (the object of interest no longer looks like the object of interest)
    from *photometric* OOD (the whole frame is brighter/noisier). If ensemble variance
    only reacts to the global perturbations and not to this one, that is a meaningful
    negative result about what the variance signal is actually tracking.

    Args:
        region: (y0, x0, y1, x1) as fractions of (H, W).
        max_deg: hue rotation at severity 1.0. 150 deg takes a yellow banana to a
            blue/violet one.
    """
    _check_image(img, "color_shift_region")
    if severity == 0.0:
        return img.clone()
    _, H, W = img.shape
    y0, x0, y1, x1 = region
    iy0, ix0 = int(round(y0 * H)), int(round(x0 * W))
    iy1, ix1 = int(round(y1 * H)), int(round(x1 * W))
    iy0, ix0 = max(0, iy0), max(0, ix0)
    iy1, ix1 = min(H, iy1), min(W, ix1)
    if iy1 <= iy0 or ix1 <= ix0:
        raise ValueError(f"color_shift_region: empty region {region} at {(H, W)}")

    theta = math.radians(float(severity) * max_deg)
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32)
    M = _YIQ2RGB @ rot @ _RGB2YIQ  # (3,3) RGB -> RGB

    out = img.clone()
    patch = out[:, iy0:iy1, ix0:ix1]                     # (3, h, w)
    ph, pw = patch.shape[1], patch.shape[2]
    flat = patch.reshape(3, -1)                          # (3, h*w)
    out[:, iy0:iy1, ix0:ix1] = (M @ flat).reshape(3, ph, pw)
    return _clamp01(out)


# --------------------------------------------------------------------------------------
# State-space perturbation
# --------------------------------------------------------------------------------------

def state_offset(state: torch.Tensor, severity: float, seed: int = 0, *,
                 direction: str = "random",
                 sigma_scale: float = 2.0,
                 include_gripper: bool = False) -> torch.Tensor:
    """Offset the observed joint angles -- "the arm is somewhere the demos never went".

    The offset is expressed in *dataset sigma* per joint:

        offset_j = severity * sigma_scale * STATE_STD[j] * u_j,   ||u|| = 1

    With the default sigma_scale=2.0 and a unit direction over 6 joints, severity 1.0
    displaces each joint by ~0.8 sigma on average. Using sigma units rather than a flat
    radian offset matters here: ur_q6 spans 3.08 rad and ur_q5 spans 0.69 rad, so a
    flat 0.1 rad offset would be a shrug for one joint and a large excursion for another.

    Args:
        direction: "random" -> seeded unit vector (a different arbitrary direction per
            seed, which is what you want when averaging over many frames);
            "fixed" -> the constant all-positive unit vector, for a reproducible
            single-direction sweep where you need severity to be the only variable.
        include_gripper: default False. Index 6 is deliberately excluded -- the
            commanded gripper is bimodal (95.4% of frames within 0.05 of a rail) while
            the *measured* state[6] is not, they have different distributions, and
            nudging it produces a physically meaningless half-open finger reading that
            would confound the joint-space result. Turn on only for a dedicated probe.

    NOTE: no clamping to STATE_MIN/STATE_MAX is applied -- leaving the training range is
    the entire point. The checkpoint's MIN_MAX normalizer will map such values outside
    [-1, 1], exactly as it would on a real robot in a novel pose.
    """
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"state_offset: expected torch.Tensor, got {type(state)!r}")
    if state.shape != (7,):
        raise ValueError(f"state_offset: expected shape (7,), got {tuple(state.shape)}")
    if severity == 0.0:
        return state.clone()

    n = 7 if include_gripper else 6
    if direction == "random":
        g = _generator(seed, f"state_offset:{direction}")
        u = torch.randn(n, generator=g, dtype=torch.float32)
    elif direction == "fixed":
        u = torch.ones(n, dtype=torch.float32)
    else:
        raise ValueError(f"state_offset: direction must be 'random' or 'fixed', got {direction!r}")
    u = u / u.norm()

    delta = torch.zeros(7, dtype=torch.float32)
    delta[:n] = float(severity) * sigma_scale * STATE_STD[:n] * u
    return state.to(torch.float32) + delta


# --------------------------------------------------------------------------------------
# obs-dict level wrappers + registry
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Perturbation:
    """One named, severity-parameterised perturbation of a full obs dict."""

    name: str
    kind: str                     # "image" | "state"
    fn: Callable                  # tensor-level callable
    description: str
    cameras: Tuple[str, ...] = CAMERA_KEYS

    def __call__(self, obs: Dict, severity: float, seed: int = 0) -> Dict:
        """Return a NEW obs dict with this perturbation applied at the given severity.

        The input dict is never mutated, and unperturbed entries (e.g. "task") are
        carried across by reference. Tensors that ARE perturbed are always fresh
        copies, so the caller can safely keep the ID obs around for comparison.
        """
        out = dict(obs)
        if self.kind == "image":
            for cam in self.cameras:
                if cam not in out:
                    raise KeyError(f"{self.name}: obs is missing camera key {cam!r}")
                out[cam] = self.fn(out[cam], severity, seed)
        elif self.kind == "state":
            if OBS_STATE_KEY not in out:
                raise KeyError(f"{self.name}: obs is missing {OBS_STATE_KEY!r}")
            out[OBS_STATE_KEY] = self.fn(out[OBS_STATE_KEY], severity, seed)
        else:  # pragma: no cover - guarded by construction
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        return out


def _P(name, kind, fn, description, cameras=CAMERA_KEYS) -> Perturbation:
    return Perturbation(name=name, kind=kind, fn=fn, description=description, cameras=cameras)


#: name -> Perturbation. The runner iterates this.
PERTURBATIONS: Dict[str, Perturbation] = {
    p.name: p
    for p in (
        _P("brightness_up", "image", partial(brightness, sign=+1.0),
           "Additive luminance +0..0.35 (lights turned up)."),
        _P("brightness_down", "image", partial(brightness, sign=-1.0),
           "Additive luminance -0..0.35 (lights turned down)."),
        _P("contrast_up", "image", partial(contrast, sign=+1.0),
           "Contrast gain up to 1.7x about the image mean."),
        _P("contrast_down", "image", partial(contrast, sign=-1.0),
           "Contrast gain down to 0.3x (washed out)."),
        _P("gaussian_noise", "image", gaussian_noise,
           "Per-pixel gaussian sensor noise, std up to 0.20."),
        _P("occlusion_center_black", "image", partial(occlusion, center=(0.5, 0.5), fill="black"),
           "Black square over the frame centre, up to 30% of frame area."),
        _P("occlusion_gripper_black", "image", partial(occlusion, center=(0.45, 0.72), fill="black"),
           "Black square over the gripper/pot region of cam1, up to 30% of area."),
        _P("occlusion_center_noise", "image", partial(occlusion, center=(0.5, 0.5), fill="noise"),
           "Uniform-noise square over the frame centre (soft occluder)."),
        _P("shift_x", "image", partial(shift, axis="x", sign=+1.0),
           "Horizontal camera translation up to 12% of width (~77 px), edge-replicated."),
        _P("shift_y", "image", partial(shift, axis="y", sign=+1.0),
           "Vertical camera translation up to 12% of height (~43 px), edge-replicated."),
        _P("blur", "image", blur,
           "Gaussian defocus blur, sigma up to 6.0 px."),
        _P("color_shift_region", "image", color_shift_region,
           "Hue rotation up to 150 deg inside the free-banana bbox (distractor probe)."),
        _P("state_offset_random", "state", partial(state_offset, direction="random"),
           "Joint offset along a seeded unit direction, up to ~0.8 sigma/joint."),
        _P("state_offset_fixed", "state", partial(state_offset, direction="fixed"),
           "Joint offset along the constant all-positive unit direction."),
    )
}

#: Perturbations that only touch one camera meaningfully (cam1 is the scene view whose
#: geometry the hand-picked regions were tuned on). Kept as metadata for the runner;
#: by default every image perturbation hits BOTH cameras, which is the stronger and
#: more honest OOD condition.
CAM1_TUNED_REGIONS = ("occlusion_gripper_black", "color_shift_region")


def apply_perturbation(obs: Dict, name: str, severity: float, seed: int = 0) -> Dict:
    """Apply a registered perturbation by name. Returns a new obs dict."""
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; known: {sorted(PERTURBATIONS)}")
    return PERTURBATIONS[name](obs, severity, seed)


def severity_sweep(
    names: Iterable[str] | None = None,
    severities: Sequence[float] = DEFAULT_SEVERITIES,
    seed: int = 0,
    kinds: Iterable[str] | None = None,
) -> Iterator[Tuple[str, float, Callable[[Dict], Dict]]]:
    """Yield (name, severity, bound_callable) over {perturbation} x {severity}.

    The bound callable takes just the obs dict, so the runner's inner loop is::

        for name, sev, fn in severity_sweep(seed=run_seed):
            obs_ood = fn(obs)
            var = ensemble_variance(obs_ood)
            record(name, sev, var)

    Args:
        names: subset of PERTURBATIONS to sweep. None -> all, in registry order.
        severities: severity grid. Keep 0.0 in it -- it is the ID control arm.
        seed: forwarded to every perturbation. Note each op mixes the seed with its own
            name, so ops do not share a noise stream.
        kinds: optional filter, e.g. ("image",) or ("state",).
    """
    if names is None:
        names = list(PERTURBATIONS)
    for name in names:
        if name not in PERTURBATIONS:
            raise KeyError(f"unknown perturbation {name!r}")
        p = PERTURBATIONS[name]
        if kinds is not None and p.kind not in kinds:
            continue
        for sev in severities:
            yield name, float(sev), partial(p.__call__, severity=float(sev), seed=seed)


# --------------------------------------------------------------------------------------
# Self-test + contact sheet (__main__ only)
# --------------------------------------------------------------------------------------

SCRATCH = os.environ.get(
    "OOD_SCRATCH",
    "/tmp/claude-1002/-home-theo-lab-gello-software/daf3ea1a-0972-4af7-ae82-6a2a58cf286b/scratchpad",
)
DEFAULT_DATASET = os.path.join(SCRATCH, "hf_data", "banana_in_pot_lerobot_v3")
OUT_PNG = os.path.join(SCRATCH, "ood_examples.png")


def _load_real_frame(dataset_dir: str = DEFAULT_DATASET,
                     camera: str = "cam1",
                     timestamp_s: float = 8.0) -> torch.Tensor:
    """Pull one real dataset frame through the EXACT live preprocessing chain.

    ffmpeg is used to decode (the dataset videos are AV1; cv2's AV1 path fails on this
    box with "Your platform doesn't support hardware accelerated AV1 decoding" and pyav
    is not installed in act_venv). ffmpeg writes a JPEG, and we feed those raw JPEG
    bytes to policy_server/image_preprocess.decode_jpeg_to_rgb_float_chw -- the very
    function diffusion_server.py L354-355 calls -- so the contact sheet shows exactly
    what the policy would see.
    """
    import glob
    import subprocess

    pattern = os.path.join(dataset_dir, "videos", f"observation.images.{camera}",
                           "chunk-*", "*.mp4")
    vids = sorted(glob.glob(pattern))
    if not vids:
        raise FileNotFoundError(f"no dataset videos matching {pattern}")

    jpg = os.path.join(SCRATCH, f"_ood_frame_{camera}.jpg")
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(timestamp_s), "-i", vids[0],
           "-frames:v", "1", "-q:v", "2", jpg]
    subprocess.run(cmd, check=True)
    with open(jpg, "rb") as fh:
        jpeg_bytes = fh.read()

    # Import the real preprocessing module (scripts/ -> package root -> policy_server/).
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from policy_server.image_preprocess import decode_jpeg_to_rgb_float_chw  # noqa: E402

    img = decode_jpeg_to_rgb_float_chw(jpeg_bytes)
    assert tuple(img.shape) == (3, RESIZE_HW[0], RESIZE_HW[1]), img.shape
    return img


def _selftest(img: torch.Tensor, state: torch.Tensor) -> None:
    """Assert the invariants the runner depends on. Raises on failure."""
    obs = {OBS_STATE_KEY: state, OBS_CAM1_KEY: img, OBS_CAM2_KEY: img.clone(),
           OBS_TASK_KEY: ""}

    for name, p in PERTURBATIONS.items():
        # 1. severity 0 is an exact identity.
        z = p(obs, 0.0, seed=0)
        if p.kind == "image":
            for cam in p.cameras:
                assert torch.equal(z[cam], obs[cam]), f"{name}: severity 0 not identity"
        else:
            assert torch.equal(z[OBS_STATE_KEY], obs[OBS_STATE_KEY]), \
                f"{name}: severity 0 not identity"

        # 2. determinism given a seed.
        a = p(obs, 0.5, seed=7)
        b = p(obs, 0.5, seed=7)
        key = OBS_CAM1_KEY if p.kind == "image" else OBS_STATE_KEY
        assert torch.equal(a[key], b[key]), f"{name}: not deterministic under a fixed seed"

        # 3. severity actually changes something.
        assert not torch.equal(a[key], obs[key]), f"{name}: severity 0.5 was a no-op"

        # 4. shape / range / dtype contract preserved.
        if p.kind == "image":
            assert a[OBS_CAM1_KEY].shape == img.shape, f"{name}: shape changed"
            assert a[OBS_CAM1_KEY].dtype == torch.float32, f"{name}: dtype changed"
            lo, hi = float(a[OBS_CAM1_KEY].min()), float(a[OBS_CAM1_KEY].max())
            assert -1e-6 <= lo and hi <= 1.0 + 1e-6, f"{name}: escaped [0,1] -> [{lo},{hi}]"
            # 5. the input dict was not mutated.
            assert torch.equal(obs[OBS_CAM1_KEY], img), f"{name}: mutated the input obs"
        else:
            assert a[OBS_STATE_KEY].shape == (7,), f"{name}: state shape changed"
            assert float(a[OBS_STATE_KEY][6]) == float(state[6]), \
                f"{name}: touched the gripper dim despite include_gripper=False"

        # 6. seeds actually differ for the stochastic ops.
        if name in ("gaussian_noise", "occlusion_center_noise", "state_offset_random"):
            c = p(obs, 0.5, seed=8)
            assert not torch.equal(a[key], c[key]), f"{name}: seed had no effect"

    # 7. monotonicity of image distortion in severity (sanity, not a hard requirement
    #    for shift/occlusion which are geometric, so check the photometric ones).
    for name in ("brightness_up", "gaussian_noise", "blur", "contrast_up"):
        p = PERTURBATIONS[name]
        d = [float((p(obs, s, seed=0)[OBS_CAM1_KEY] - img).abs().mean())
             for s in (0.25, 0.5, 1.0)]
        assert d[0] < d[1] < d[2], f"{name}: L1 distortion not monotone in severity: {d}"

    print("[selftest] all invariants hold "
          f"({len(PERTURBATIONS)} perturbations checked)", flush=True)


def _contact_sheet(img: torch.Tensor, state: torch.Tensor, out_png: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_names = [n for n, p in PERTURBATIONS.items() if p.kind == "image"]
    cols = [0.0, 0.25, 0.5, 1.0]
    nrows, ncols = len(img_names), len(cols)

    header_in = 1.6                      # inches reserved for the text header
    fig_h = 1.85 * nrows + header_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, fig_h))
    obs = {OBS_STATE_KEY: state, OBS_CAM1_KEY: img, OBS_CAM2_KEY: img.clone(),
           OBS_TASK_KEY: ""}

    for r, name in enumerate(img_names):
        p = PERTURBATIONS[name]
        for c, sev in enumerate(cols):
            ax = axes[r][c]
            out = p(obs, sev, seed=0)[OBS_CAM1_KEY]
            ax.imshow(out.permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            l1 = float((out - img).abs().mean())
            if c == 0:
                ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
            ax.set_title(f"sev={sev:.2f}  L1={l1:.4f}", fontsize=7)

    # State perturbation gets a text panel, not an image -- render the numbers so a
    # human can see the offsets are small and the gripper dim is untouched.
    lines = ["state_offset_random (radians, seed=0)",
             "        " + "".join(f"{f'q{j+1}':>9}" for j in range(6)) + f"{'grip':>9}",
             "ID      " + "".join(f"{float(state[j]):9.4f}" for j in range(7))]
    for sev in (0.25, 0.5, 1.0):
        s2 = PERTURBATIONS["state_offset_random"](obs, sev, seed=0)[OBS_STATE_KEY]
        d = s2 - state
        lines.append(f"sev={sev:<4.2f}" + "".join(f"{float(d[j]):+9.4f}" for j in range(7)))
    lines.append("(row values are DELTAS in radians; grip dim 6 is intentionally 0.0)")

    top = 1.0 - header_in / fig_h
    fig.suptitle(
        "OOD perturbations on a real banana_in_pot frame (cam1, post-resize 3x360x640, [0,1] RGB)\n"
        "injected at diffusion_server.py L352-357, before the checkpoint preprocessor L363\n\n"
        + "\n".join(lines),
        fontsize=8, family="monospace", y=0.995, va="top",
    )
    fig.tight_layout(rect=(0.02, 0.0, 1.0, top))
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> int:
    torch.manual_seed(0)
    img = _load_real_frame()
    # A real-ish ID state: the dataset mean pose (observation.state.mean).
    state = torch.tensor(
        [3.0546412, -1.5018350, 1.9273195, -2.1486695, -1.7061603, -3.3053951, 0.23604383],
        dtype=torch.float32,
    )
    print(f"[main] loaded real frame {tuple(img.shape)} "
          f"range [{float(img.min()):.3f}, {float(img.max()):.3f}]", flush=True)
    _selftest(img, state)
    path = _contact_sheet(img, state, OUT_PNG)
    print(f"[main] wrote {path} ({os.path.getsize(path)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
