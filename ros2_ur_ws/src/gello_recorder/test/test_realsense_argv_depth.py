"""Headless tests for the RealSense launch argv the recorder GUIs build.

What is pinned here is the DEFAULT: depth OFF unless ``ENABLE_DEPTH`` opts IN,
and depth->color alignment OFF unless ``ALIGN_DEPTH`` opts in.

Depth was ON by default for exactly one day (2026-09-14) and that day cost a
54-take corpus its timestamps: two extra 30 Hz subscriptions per camera plus a
~6 MB/s HDF5 write on the recorder's single rclpy spin thread dropped the
executor's round rate to 60-69 Hz, and every topic publishing faster than that
was then read out of a permanently full queue -- ur_joint_states rows landed
0.900 s late. The defect is fixed (gello_recorder/spin_health.py) but the cost
is not, so RGB-only is the default and depth is asked for per session.

BOTH directions are pinned deliberately. An accidental default flip to ON costs
disk, CPU and spin-thread margin on every recording; an ``ENABLE_DEPTH=1`` that
silently fails to emit ``enable_depth:=true`` costs the DATA, and nothing in the
GUI shows it (the colour panes look identical either way). Alignment is opt-in
because it was measured at ~49 % CPU per camera node and ~25 Hz on BOTH streams
versus ~9 % / 30 Hz unaligned; a default that silently flips it would degrade
every recording, so that is pinned too.

Also pinned: the *embedded* single quotes around ``serial_no`` and
``rgb_camera.color_profile``. ``ros2 launch`` type-infers bare ``key:=value``
args, so dropping those quotes coerces an all-digit serial to an int and the
node dies on startup.

Only :func:`_realsense_argv` (and the pure topic-name helpers) are exercised --
deliberately never ``_launch_realsense``, which spawns real ``ros2 launch``
processes that would grab the cameras. The two share one argv builder, so this
covers both.
"""

import pytest

gui = pytest.importorskip(
    "gello_recorder.gello_recorder_gui",
    reason="gello_recorder_gui needs rclpy + PyQt5 from the ROS overlay",
)


CAM = "cam1"
SERIAL = "147122072740"
PROFILE = "1280x720x30"


def _argv(monkeypatch, env=None, align_env=None, **kwargs):
    """Build an argv with ENABLE_DEPTH / ALIGN_DEPTH forced (None == unset)."""

    if env is None:
        monkeypatch.delenv("ENABLE_DEPTH", raising=False)
    else:
        monkeypatch.setenv("ENABLE_DEPTH", env)
    if align_env is None:
        monkeypatch.delenv("ALIGN_DEPTH", raising=False)
    else:
        monkeypatch.setenv("ALIGN_DEPTH", align_env)
    return gui._realsense_argv(CAM, SERIAL, PROFILE, **kwargs)


# --------------------------------------------------------------------------- #
# ENABLE_DEPTH: OFF by default, opt-in
# --------------------------------------------------------------------------- #

def test_depth_is_off_when_the_env_var_is_unset(monkeypatch):
    """The whole point: an ordinary recording session is RGB-only."""

    assert "enable_depth:=false" in _argv(monkeypatch)


def test_the_gui_env_helper_agrees_with_the_argv(monkeypatch):
    """The helper both GUIs' main() call is the same decision as the argv."""

    monkeypatch.delenv("ENABLE_DEPTH", raising=False)
    assert gui._depth_enabled_from_env() is False
    monkeypatch.setenv("ENABLE_DEPTH", "1")
    assert gui._depth_enabled_from_env() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_explicit_on_values_opt_in_to_depth(monkeypatch, value):
    assert "enable_depth:=true" in _argv(monkeypatch, value)


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_explicit_off_values_keep_depth_off(monkeypatch, value):
    assert "enable_depth:=false" in _argv(monkeypatch, value)


@pytest.mark.parametrize("value", ["", "banana"])
def test_a_typo_in_the_value_fails_closed_to_off(monkeypatch, value):
    """Only the documented ON values turn depth on; garbage turns it off.

    The cost of a typo is a session without depth, never a surprise stream
    that starves the spin thread. Pinned so the rule cannot drift into
    "anything non-empty is on".
    """

    assert "enable_depth:=false" in _argv(monkeypatch, value)


@pytest.mark.parametrize(
    "env, kwarg, expected",
    [
        ("0", True, "enable_depth:=true"),    # kwarg turns it on
        ("1", False, "enable_depth:=false"),  # kwarg turns it off
        (None, True, "enable_depth:=true"),   # kwarg beats the OFF default
    ],
)
def test_an_explicit_kwarg_beats_the_env_var_both_ways(
    monkeypatch, env, kwarg, expected
):
    assert expected in _argv(monkeypatch, env, enable_depth=kwarg)


def test_exactly_one_enable_depth_element_is_emitted(monkeypatch):
    """Duplicates are the failure mode of "just append it": last wins, quietly."""

    argv = _argv(monkeypatch)
    assert len([a for a in argv if a.startswith("enable_depth:=")]) == 1


# --------------------------------------------------------------------------- #
# ALIGN_DEPTH: off by default, opt-in, and only ever emitted with depth on
# --------------------------------------------------------------------------- #

# Every test below passes ENABLE_DEPTH="1" explicitly: with depth off the argv
# carries no align element AT ALL (see test_no_align_element_at_all_when_depth
# _is_off), so alignment behaviour is only observable on the opt-in path.

def test_alignment_is_off_by_default_when_depth_is_on(monkeypatch):
    assert "align_depth.enable:=false" in _argv(monkeypatch, "1")


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_align_depth_env_opts_in(monkeypatch, value):
    assert "align_depth.enable:=true" in _argv(monkeypatch, "1", align_env=value)


@pytest.mark.parametrize("value", ["0", "false", "", "banana"])
def test_align_depth_anything_else_stays_off(monkeypatch, value):
    assert "align_depth.enable:=false" in _argv(monkeypatch, "1", align_env=value)


@pytest.mark.parametrize(
    "align_env, kwarg, expected",
    [
        (None, True, "align_depth.enable:=true"),
        ("1", False, "align_depth.enable:=false"),
    ],
)
def test_an_explicit_align_kwarg_beats_the_env_var(
    monkeypatch, align_env, kwarg, expected
):
    assert expected in _argv(
        monkeypatch, "1", align_env=align_env, align_depth=kwarg)


@pytest.mark.parametrize("align_env", [None, "1"])
def test_no_align_element_at_all_when_depth_is_off(monkeypatch, align_env):
    """With depth off the argv must be byte-identical to the pre-depth argv.

    Aligning a stream that is not running is meaningless, and emitting the
    element anyway would make ``ENABLE_DEPTH=0`` sessions differ from the
    argv every earlier recording was made with. Pinned for both ALIGN_DEPTH
    values so an opt-in cannot leak through the off path.
    """

    argv = _argv(monkeypatch, "0", align_env=align_env)
    # ...and the DEFAULT argv (env unset) must be exactly this one too.
    assert _argv(monkeypatch, None, align_env=align_env) == argv
    assert not [a for a in argv if a.startswith("align_depth.enable:=")]
    assert argv == [
        "ros2", "launch", "realsense2_camera", "rs_launch.py",
        "camera_name:=cam1",
        "camera_namespace:=cam1",
        "serial_no:='147122072740'",
        "rgb_camera.color_profile:='1280x720x30'",
        "enable_depth:=false",
    ]


def test_exactly_one_align_element_is_emitted_with_depth_on(monkeypatch):
    argv = _argv(monkeypatch, "1", align_env="1")
    assert len([a for a in argv if a.startswith("align_depth.enable:=")]) == 1


def test_align_element_comes_after_enable_depth(monkeypatch):
    """Order is not load-bearing for ros2 launch, but it IS what the shell
    launchers emit; keeping them in step makes the two argvs diffable."""

    argv = _argv(monkeypatch, "1")
    assert argv.index("enable_depth:=true") < argv.index("align_depth.enable:=false")


# --------------------------------------------------------------------------- #
# The type-inference workaround
# --------------------------------------------------------------------------- #

def test_the_type_inference_workaround_survives_the_refactor(monkeypatch):
    """Serial and profile keep their embedded quotes; the rest is unchanged."""

    argv = _argv(monkeypatch)
    assert "serial_no:='147122072740'" in argv
    assert "rgb_camera.color_profile:='1280x720x30'" in argv
    assert argv[:4] == ["ros2", "launch", "realsense2_camera", "rs_launch.py"]
    assert "camera_name:=cam1" in argv
    assert "camera_namespace:=cam1" in argv


# --------------------------------------------------------------------------- #
# depth_topics_for -- the one definition of the realsense-ros depth topic layout
# --------------------------------------------------------------------------- #

def test_depth_topics_unaligned_match_what_realsense_ros_publishes():
    """Measured 2026-09-14 on realsense-ros 4.58.2 with enable_depth:=true."""

    image, info, ext = gui.depth_topics_for("cam1", aligned=False)
    assert image == "/cam1/cam1/depth/image_rect_raw/compressedDepth"
    assert info == "/cam1/cam1/depth/camera_info"
    assert ext == "/cam1/cam1/extrinsics/depth_to_color"


def test_depth_topics_aligned_match_what_realsense_ros_publishes():
    """Measured 2026-09-14 with align_depth.enable:=true."""

    image, info, ext = gui.depth_topics_for("cam2", aligned=True)
    assert image == "/cam2/cam2/aligned_depth_to_color/image_raw/compressedDepth"
    assert info == "/cam2/cam2/aligned_depth_to_color/camera_info"
    # The depth->color transform does not depend on alignment.
    assert ext == "/cam2/cam2/extrinsics/depth_to_color"


def test_depth_topics_follow_the_camera_name_twice():
    """rs_launch.py nests camera_name under camera_namespace (both = name)."""

    image, info, ext = gui.depth_topics_for("wrist", aligned=False)
    for topic in (image, info, ext):
        assert topic.startswith("/wrist/wrist/"), topic


def test_depth_topics_are_the_same_object_the_node_module_defines():
    """The GUI re-exports gello_gui_node.depth_topics_for, never a copy."""

    node_mod = pytest.importorskip("gello_recorder.gello_gui_node")
    assert gui.depth_topics_for is node_mod.depth_topics_for


# --------------------------------------------------------------------------- #
# _depth_node_kwargs -- what main() hands the node
# --------------------------------------------------------------------------- #

def test_node_kwargs_are_empty_when_depth_is_off():
    assert gui._depth_node_kwargs("cam1", "cam2", False, False) == {}
    assert gui._depth_node_kwargs("cam1", "cam2", False, True) == {}


def test_node_kwargs_carry_all_six_topics_and_the_aligned_flag():
    kwargs = gui._depth_node_kwargs("cam1", "cam2", True, True)
    assert kwargs == {
        "cam1_depth_topic": "/cam1/cam1/aligned_depth_to_color/image_raw/compressedDepth",
        "cam2_depth_topic": "/cam2/cam2/aligned_depth_to_color/image_raw/compressedDepth",
        "cam1_depth_info_topic": "/cam1/cam1/aligned_depth_to_color/camera_info",
        "cam2_depth_info_topic": "/cam2/cam2/aligned_depth_to_color/camera_info",
        "cam1_extrinsics_topic": "/cam1/cam1/extrinsics/depth_to_color",
        "cam2_extrinsics_topic": "/cam2/cam2/extrinsics/depth_to_color",
        "depth_aligned_to_color": True,
    }
    unaligned = gui._depth_node_kwargs("cam1", "cam2", True, False)
    assert unaligned["depth_aligned_to_color"] is False
    assert unaligned["cam1_depth_topic"] == "/cam1/cam1/depth/image_rect_raw/compressedDepth"
