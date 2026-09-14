"""CameraRig: model build, mirroring, colour/depth renders, real-D435 metadata."""
import math
import time

import numpy as np
import pytest

from sim_collect import cameras
from sim_collect.tests.conftest import needs_display
from sim_collect.tests.f2_testlib import TINY_NQ, TINY_NV, TINY_SCENE_XML, make_state


def test_camera_info_matches_real_take_numbers():
    c1 = cameras.camera_info_dict("cam1")
    c2 = cameras.camera_info_dict("cam2")
    assert (c1["width"], c1["height"]) == (848, 480)
    assert c1["distortion_model"] == "plumb_bob" and c1["frame_id"] == "cam1_depth_optical_frame"
    assert c1["K"][0] == 426.7417907714844 and c1["K"][2] == 423.93927001953125 and c1["K"][5] == 233.14932250976562
    assert c2["K"][0] == 425.26434326171875 and c2["K"][2] == 425.0440979003906 and c2["K"][5] == 232.82229614257812
    assert c1["D"] == [0.0] * 5 and c1["R"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert c1["P"] == [c1["K"][0], 0, c1["K"][2], 0, 0, c1["K"][4], c1["K"][5], 0, 0, 0, 1, 0]
    # extrinsics + source_topic are the REAL take_18 values verbatim (R2 item 7): the
    # LeRobot converter's _cam_meta_equal must accept sim and real takes as one set.
    ext = cameras.extrinsics_dict("cam2")
    assert ext["rotation"][0] == 0.999984085559845 and ext["translation"][0] == 0.014892518520355225
    assert cameras.extrinsics_dict("cam1")["translation"][0] == 0.015069474466145039
    assert np.allclose(np.asarray(ext["rotation"]).reshape(3, 3) @ np.asarray(ext["rotation"]).reshape(3, 3).T,
                       np.eye(3), atol=1e-6)
    assert cameras.depth_source_topic("cam1") == "/cam1/cam1/depth/image_rect_raw/compressedDepth"
    with pytest.raises(KeyError):
        cameras.camera_info_dict("cam3")
    with pytest.raises(KeyError):
        cameras.extrinsics_dict("cam3")


def test_depth_m_to_mm_rules():
    z = np.array([[0.4571, 9.9999, 10.0], [np.nan, np.inf, -1.0]], dtype=np.float32)
    mm = cameras.depth_m_to_mm(z, 10.0)
    assert mm.dtype == np.uint16
    assert mm[0, 0] == 457 and mm[0, 1] == 10000 and mm[0, 2] == 0
    assert (mm[1] == 0).all()


def test_build_model_raises_offscreen_buffer_without_display():
    m = cameras.build_model(TINY_SCENE_XML)
    assert m.vis.global_.offwidth >= 1280 and m.vis.global_.offheight >= 720
    assert m.nq == TINY_NQ and m.nv == TINY_NV


def test_rig_mirror_rejects_wrong_sizes_without_rendering():
    rig = cameras.CameraRig(TINY_SCENE_XML)
    with pytest.raises(ValueError):
        rig.mirror([0.0] * (TINY_NQ + 1))
    with pytest.raises(ValueError):
        rig.mirror([float("nan")] * TINY_NQ)
    msg = make_state(0, time.time(), 0.5)
    rig.mirror(msg["qpos_full"], msg["qvel_full"])
    assert rig.n_mirrored == 1 and rig.n_rejected == 2
    assert rig.depth_camera_name("cam1") == "cam1_depth"
    pose = rig.camera_pose("cam1")
    assert pose["fixed"] and abs(pose["pos"][2] - 0.571) < 1e-9 and pose["fovy_deg"] == 42.0
    ki = rig.render_intrinsics("cam1", depth=True)
    assert (ki["width"], ki["height"]) == (848, 480)
    # fovy 58.7 deg at 480 px -> fx ~ 426.7 (the real D435 depth fx), by design.
    assert abs(ki["fx"] - 240.0 / math.tan(math.radians(58.7) / 2)) < 1e-9
    assert abs(ki["fx"] - 426.74) < 1.0
    rig.close()


@needs_display
def test_rig_renders_color_and_depth_with_expected_geometry():
    rig = cameras.CameraRig(TINY_SCENE_XML)
    msg = make_state(0, time.time(), 0.0)
    rig.mirror(msg["qpos_full"], msg["qvel_full"])
    rgb = rig.render_color("cam1")
    assert rgb.shape == (720, 1280, 3) and rgb.dtype == np.uint8
    assert rgb.std() > 5.0, "colour render is flat"
    z = rig.render_depth_m("cam1")
    mm = rig.render_depth("cam1")
    assert mm.shape == (480, 848) and mm.dtype == np.uint16
    # cam1_depth at (-0.7,-0.015,0.571) looks at the box top (-0.45, 0, ~0.1): the
    # centre ray hits the box or the floor at less than 1 m; the top rows see the sky (0).
    centre = mm[240, 424]
    assert 300 < centre < 1500, centre
    assert abs(centre - round(float(z[240, 424]) * 1000)) <= 1
    assert (mm[0, :] == 0).all(), "rays above the horizon must be 0 (no return)"
    assert (mm[mm > 0] < 10000).all()
    # cam2 too, and a second mirror moves the box (different depth image).
    mm2a = rig.render_depth("cam2")
    msg2 = make_state(0, time.time(), 3.0)
    rig.mirror(msg2["qpos_full"], msg2["qvel_full"])
    mm2b = rig.render_depth("cam2")
    assert mm2a.shape == (480, 848) and not np.array_equal(mm2a, mm2b)
    # timing (informational, printed with -s)
    t0 = time.perf_counter()
    for _ in range(10):
        rig.render_color("cam1"); rig.render_color("cam2")
    t1 = time.perf_counter()
    for _ in range(10):
        rig.render_depth("cam1"); rig.render_depth("cam2")
    t2 = time.perf_counter()
    print(f"\n[cameras] 2x colour 1280x720: {1e3 * (t1 - t0) / 10:.1f} ms  "
          f"2x depth 848x480 + mm: {1e3 * (t2 - t1) / 10:.1f} ms")
    assert (t1 - t0) / 10 < 0.2 and (t2 - t1) / 10 < 0.2
    rig.close()
