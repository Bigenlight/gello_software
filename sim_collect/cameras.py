"""Camera rig for the capture process ([B], DESIGN.md §2.2 / §3.5).

A :class:`CameraRig` owns its OWN ``MjModel``/``MjData`` built from the scene MJCF the
sim process hands over (``get_scene_meta`` -> ``scene_xml`` + ``scene_assets``), mirrors
the physics state it receives (``qpos_full`` / ``qvel_full``) and renders the two
RealSense stand-ins:

* colour  -- camera ``camN``       -> RGB uint8 ``(720, 1280, 3)``
* depth   -- camera ``camN_depth`` -> uint16 millimetres ``(480, 848)``, ``0`` where
  the ray hits nothing within ``depth_max_m`` (10 m, like a D435 "no return")

The rendering facts that shaped this file (measured 2026-09-14 on this machine, see
DESIGN §1): only ``MUJOCO_GL=glfw`` with a ``DISPLAY`` works; a ``mujoco.Renderer`` must
never share a process with ``mujoco.viewer``; the model must declare
``<visual><global offwidth>=1280 offheight>=720/></visual>`` (this class raises the
framebuffer limits on its own copy of the model when the scene forgot); 2x1280x720 colour
= ~11 ms, 2x848x480 depth = ~8 ms.

``camera_info`` / ``extrinsics_depth_to_color`` reproduce the REAL D435 numbers from
``take_18_20260914_165926`` byte-for-byte (DESIGN §1 "Real data format") so a sim take is
indistinguishable in structure from a real one and ``convert_carrot_to_lerobot`` sees the
same metadata on every take. The intrinsics implied by the MuJoCo camera's ``fovy`` are
exposed separately (:meth:`render_intrinsics`) and recorded in ``sim_meta`` so nobody has
to guess whether the two agree.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402  (after MUJOCO_GL default)

COLOR_SIZE = (1280, 720)   # (width, height) of camN.mp4
DEPTH_SIZE = (848, 480)    # (width, height) of the depth PNGs
DEPTH_MAX_M = 10.0
DEPTH_SCALE_M = 0.001
CAMS = ("cam1", "cam2")

# Real D435 depth CameraInfo, verbatim from take_18_20260914_165926/depth.h5 (DESIGN §1).
_REAL_K = {
    "cam1": [426.7417907714844, 0.0, 423.93927001953125,
             0.0, 426.7417907714844, 233.14932250976562,
             0.0, 0.0, 1.0],
    "cam2": [425.26434326171875, 0.0, 425.0440979003906,
             0.0, 425.26434326171875, 232.82229614257812,
             0.0, 0.0, 1.0],
}
# Real depth->colour extrinsics, VERBATIM per camera from take_18_20260914_165926/depth.h5
# (column-major R, t in metres). Deliberately the measured values rather than an ideal
# I / [0.015, 0, 0]: convert_carrot_to_lerobot._cam_meta_equal requires camera_info AND
# extrinsics (and source_topic) to match across takes, so sim and real takes can only
# share one LeRobot dataset if the sim sidecar reproduces the real numbers exactly. The
# scene's depth cameras are still placed at +0.015 m along x (DESIGN §3.5).
_REAL_EXTRINSICS = {
    "cam1": ([0.9999342560768127, -0.01032120082527399, -0.004996659699827433,
              0.010314139537513256, 0.9999457597732544, -0.001437016180716455,
              0.005011220462620258, 0.001385385519824922, 0.999986469745636],
             [0.015069474466145039, -0.00010991686576744542, -5.541024438571185e-05]),
    "cam2": ([0.999984085559845, 9.567383676767349e-05, -0.005645133089274168,
              -0.00011083301797043532, 0.9999963641166687, -0.002685102168470621,
              0.005644855555146933, 0.0026856849435716867, 0.9999804496765137],
             [0.014892518520355225, 0.0002898888778872788, 0.0001435153535567224]),
}
# Real ROS topics, kept verbatim for the same reason (source_topic is part of the
# equality check). The sim provenance goes into vectors.h5 attrs["sim_meta"] instead.
_REAL_SOURCE_TOPIC = {cam: f"/{cam}/{cam}/depth/image_rect_raw/compressedDepth" for cam in CAMS}


def camera_info_dict(cam: str) -> Dict[str, Any]:
    """``sensor_msgs/CameraInfo`` fields for ``cam`` exactly as the real recorder stored
    them (848x480 plumb_bob, D = 0, R = I, P = [K | 0]). Identical on every take."""
    if cam not in _REAL_K:
        raise KeyError(f"unknown camera {cam!r}; expected one of {CAMS}")
    k = list(_REAL_K[cam])
    return {
        "width": DEPTH_SIZE[0],
        "height": DEPTH_SIZE[1],
        "distortion_model": "plumb_bob",
        "D": [0.0] * 5,
        "K": k,
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0],
        "frame_id": f"{cam}_depth_optical_frame",
    }


def extrinsics_dict(cam: str) -> Dict[str, Any]:
    """Depth->colour extrinsics as ``DepthH5Writer.set_extrinsics_depth_to_color`` wants
    them: ``rotation`` (9, column-major, identity) and ``translation`` (3, metres)."""
    if cam not in _REAL_EXTRINSICS:
        raise KeyError(f"unknown camera {cam!r}; expected one of {CAMS}")
    rot, tr = _REAL_EXTRINSICS[cam]
    return {"rotation": list(rot), "translation": list(tr)}


def depth_source_topic(cam: str) -> str:
    """``depth.h5/<cam>.attrs['source_topic']``: the REAL ROS topic string, verbatim, so
    ``_cam_meta_equal`` treats sim and real takes as one camera set (see
    ``_REAL_SOURCE_TOPIC``). ``vectors.h5`` ``sim_meta`` marks the take as simulated."""
    if cam not in _REAL_SOURCE_TOPIC:
        raise KeyError(f"unknown camera {cam!r}; expected one of {CAMS}")
    return _REAL_SOURCE_TOPIC[cam]


def depth_m_to_mm(depth_m: np.ndarray, depth_max_m: float = DEPTH_MAX_M) -> np.ndarray:
    """float32 metres -> uint16 millimetres; ``0`` for non-finite, ``<= 0`` or beyond
    ``depth_max_m`` (a D435 reports 0 for "no return")."""
    z = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0.0) & (z < float(depth_max_m))
    mm = np.where(valid, np.rint(z * 1000.0), 0.0)
    return np.clip(mm, 0, 65535).astype(np.uint16)


def build_model(scene_xml: str, assets: Optional[Mapping[str, bytes]] = None,
                xml_path: Optional[str] = None,
                min_offscreen: Tuple[int, int] = COLOR_SIZE) -> mujoco.MjModel:
    """Compile the scene into an ``MjModel`` and make sure the offscreen framebuffer is at
    least ``min_offscreen`` (width, height) so a 1280x720 ``Renderer`` can be created.

    ``xml_path`` wins over ``scene_xml`` when given (relative ``meshdir`` then resolves
    against the file, which an XML string cannot do)."""
    if xml_path:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    else:
        model = mujoco.MjModel.from_xml_string(scene_xml, dict(assets or {}))
    w, h = min_offscreen
    if model.vis.global_.offwidth < w:
        model.vis.global_.offwidth = int(w)
    if model.vis.global_.offheight < h:
        model.vis.global_.offheight = int(h)
    return model


class CameraRig:
    """Own copy of the scene + two lazily created renderers (colour / depth).

    Construct in the process that will render (never next to a passive viewer). Call
    :meth:`mirror` with the sim's ``qpos_full``/``qvel_full`` and then
    :meth:`render_color` / :meth:`render_depth`.
    """

    def __init__(self, scene_xml: str = "", assets: Optional[Mapping[str, bytes]] = None,
                 *, xml_path: Optional[str] = None, cams: Sequence[str] = CAMS,
                 color_size: Tuple[int, int] = COLOR_SIZE,
                 depth_size: Tuple[int, int] = DEPTH_SIZE,
                 depth_max_m: float = DEPTH_MAX_M,
                 render: Optional[Mapping[str, Any]] = None):
        if not scene_xml and not xml_path:
            raise ValueError("CameraRig needs scene_xml or xml_path")
        self.cams = tuple(cams)
        # Offscreen render quality (software GL on this machine). Measured on the real
        # scene at 1280x720: MSAA offsamples 4 -> 0 saves 4 ms; shadows+reflection+
        # skybox off saves another 12 ms (20 -> 3.6 ms). Defaults keep shadows (they
        # matter for realism) and drop MSAA/reflection/skybox. Keys (yaml `render:`):
        # capture_offsamples, capture_shadows, capture_reflections, capture_skybox.
        r = dict(render or {})
        self.render_opts = {
            "offsamples": int(r.get("capture_offsamples", 0)),
            "shadows": bool(r.get("capture_shadows", True)),
            "reflections": bool(r.get("capture_reflections", False)),
            "skybox": bool(r.get("capture_skybox", False)),
        }
        self.color_size = (int(color_size[0]), int(color_size[1]))
        self.depth_size = (int(depth_size[0]), int(depth_size[1]))
        self.depth_max_m = float(depth_max_m)
        self.model = build_model(scene_xml, assets, xml_path=xml_path,
                                 min_offscreen=(max(self.color_size[0], self.depth_size[0]),
                                                max(self.color_size[1], self.depth_size[1])))
        self.model.vis.quality.offsamples = self.render_opts["offsamples"]
        self.data = mujoco.MjData(self.model)
        # Start at the "home" keyframe when the scene has one, so previews before the
        # first state message show a sane arm rather than all-zeros.
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._color_renderer: Optional[mujoco.Renderer] = None
        self._depth_renderer: Optional[mujoco.Renderer] = None
        self._depth_cam_warned: set = set()
        self.n_mirrored = 0
        self.n_rejected = 0
        for cam in self.cams:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                raise ValueError(f"scene has no camera named {cam!r}")

    # ---- state ---------------------------------------------------------------
    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    def mirror(self, qpos_full: Sequence[float], qvel_full: Optional[Sequence[float]] = None) -> None:
        """Copy the sim's generalized state into this rig's ``MjData`` and run
        ``mj_forward``. Raises ``ValueError`` on a size mismatch (a different scene) so
        the caller can count it and keep going instead of rendering garbage."""
        q = np.asarray(qpos_full, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.nq:
            self.n_rejected += 1
            raise ValueError(f"qpos_full has {q.shape[0]} entries, model nq={self.model.nq}")
        if not np.all(np.isfinite(q)):
            self.n_rejected += 1
            raise ValueError("qpos_full contains non-finite values")
        self.data.qpos[:] = q
        if qvel_full is not None:
            v = np.asarray(qvel_full, dtype=np.float64).reshape(-1)
            if v.shape[0] == self.model.nv and np.all(np.isfinite(v)):
                self.data.qvel[:] = v
        mujoco.mj_forward(self.model, self.data)
        self.n_mirrored += 1

    # ---- rendering -----------------------------------------------------------
    def _apply_flags(self, r: mujoco.Renderer) -> mujoco.Renderer:
        F = mujoco.mjtRndFlag
        r.scene.flags[F.mjRND_SHADOW] = int(self.render_opts["shadows"])
        r.scene.flags[F.mjRND_REFLECTION] = int(self.render_opts["reflections"])
        r.scene.flags[F.mjRND_SKYBOX] = int(self.render_opts["skybox"])
        return r

    def _color(self) -> mujoco.Renderer:
        if self._color_renderer is None:
            w, h = self.color_size
            self._color_renderer = self._apply_flags(mujoco.Renderer(self.model, height=h, width=w))
        return self._color_renderer

    def _depth(self) -> mujoco.Renderer:
        if self._depth_renderer is None:
            w, h = self.depth_size
            r = mujoco.Renderer(self.model, height=h, width=w)
            r.enable_depth_rendering()
            self._depth_renderer = self._apply_flags(r)
        return self._depth_renderer

    def depth_camera_name(self, cam: str) -> str:
        """``camN_depth`` when the scene defines it, else ``camN`` (with one warning)."""
        name = f"{cam}_depth"
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0:
            return name
        if cam not in self._depth_cam_warned:
            self._depth_cam_warned.add(cam)
            print(f"[CameraRig] WARNING: scene has no camera {name!r}; rendering depth "
                  f"from {cam!r} (no 15 mm depth->colour offset)")
        return cam

    def render_color(self, cam: str) -> np.ndarray:
        """RGB uint8 ``(H, W, 3)`` from camera ``cam`` at the mirrored state."""
        r = self._color()
        r.update_scene(self.data, camera=cam)
        return r.render()

    def render_depth_m(self, cam: str) -> np.ndarray:
        """float32 metres ``(H, W)`` from ``cam``'s co-located depth camera."""
        r = self._depth()
        r.update_scene(self.data, camera=self.depth_camera_name(cam))
        return r.render()

    def render_depth(self, cam: str) -> np.ndarray:
        """uint16 millimetres ``(480, 848)``; ``0`` beyond ``depth_max_m`` / invalid."""
        return depth_m_to_mm(self.render_depth_m(cam), self.depth_max_m)

    # ---- metadata ------------------------------------------------------------
    def camera_info(self, cam: str) -> Dict[str, Any]:
        return camera_info_dict(cam)

    def extrinsics(self, cam: str) -> Dict[str, Any]:
        return extrinsics_dict(cam)

    def render_intrinsics(self, cam: str, depth: bool = False) -> Dict[str, Any]:
        """Pinhole intrinsics IMPLIED by the MuJoCo camera's ``fovy`` at the render size
        (what the pixels actually are, as opposed to the real-D435 numbers written to
        ``camera_info``). Recorded in ``sim_meta`` for honesty."""
        name = self.depth_camera_name(cam) if depth else cam
        w, h = self.depth_size if depth else self.color_size
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        fovy = float(self.model.cam_fovy[cid])
        f = 0.5 * h / math.tan(math.radians(fovy) / 2.0)
        return {"camera": name, "width": w, "height": h, "fovy_deg": fovy,
                "fx": f, "fy": f, "cx": w / 2.0, "cy": h / 2.0}

    def camera_pose(self, cam: str) -> Dict[str, Any]:
        """World pose of camera ``cam`` at the current mirrored state (pos + rotation
        matrix, row-major 9) plus ``fovy``. Wrist cameras move; this is a snapshot."""
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cid < 0:
            raise KeyError(cam)
        return {
            "pos": [float(x) for x in self.data.cam_xpos[cid]],
            "xmat": [float(x) for x in self.data.cam_xmat[cid]],
            "fovy_deg": float(self.model.cam_fovy[cid]),
            "fixed": int(self.model.cam_bodyid[cid]) == 0,
        }

    def close(self) -> None:
        for attr in ("_color_renderer", "_depth_renderer"):
            r = getattr(self, attr, None)
            if r is not None:
                try:
                    r.close()
                except Exception:  # noqa: BLE001 - best-effort on shutdown
                    pass
                setattr(self, attr, None)
