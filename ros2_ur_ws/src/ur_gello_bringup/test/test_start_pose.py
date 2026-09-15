"""Unit tests for ur_gello_bringup.start_pose (pure: stdlib + PyYAML).

This module feeds the EEF GUI's GO TO START POSE button, which MOVES THE ARM.
Its contract is fail-closed: every malformed / missing / implausible input must
be REJECTED with a message naming the file, and ``resolve_start_pose`` must
never raise and never fall back to a hard-coded pose. These tests pin every
error branch, the boolean trap (YAML ``true`` is an ``int`` subclass), the env
resolution, and that the REAL deploy yamls in this workspace parse.
"""

import importlib.util
import math
import os
import shutil

import pytest
import yaml

try:
    # Normal path. With the canonical invocation (`cd src/ur_gello_bringup &&
    # python3 -m pytest test`) `-m` puts the package dir first on sys.path, so
    # this resolves to src/ (not the overlay) and is always current. After a
    # `colcon build` it also resolves through the installed overlay from any cwd.
    from ur_gello_bringup import start_pose as sp
except ImportError:  # pragma: no cover - other cwd + copy-install overlay
    # This workspace is built with symlink_install=False (install/ is a COPY of
    # src/), so when pytest is run from a different cwd the overlay wins and a
    # module that is new in src/ is invisible until the next
    # `colcon build --packages-select ur_gello_bringup`. Load it straight from
    # src/ by path in that window (same trick test_launch_derivation uses for the
    # launch file). start_pose.py imports nothing from the package, so loading it
    # standalone is exact.
    _SRC = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ur_gello_bringup",
        "start_pose.py",
    )
    _spec = importlib.util.spec_from_file_location("ur_gello_bringup_start_pose_src", _SRC)
    sp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sp)

StartPose = sp.StartPose
StartPoseError = sp.StartPoseError
load_start_pose = sp.load_start_pose
resolve_start_pose = sp.resolve_start_pose
START_POSE_CONFIG_ENV = sp.START_POSE_CONFIG_ENV

BANANA = [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]
CARROT = [-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]

# ros2_ur_ws/src/gello_policy/config -- resolved relative to this test file so it
# works from any cwd and from the installed-overlay or src-path import alike.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIR = os.path.join(os.path.dirname(_PKG_DIR), "gello_policy", "config")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write(tmp_path, name, doc):
    """Dump ``doc`` (a python object) as YAML to tmp_path/name; return the path."""
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(p)


def _write_text(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


_OMIT = object()  # "leave this key out of the file" (None means the YAML literal null)


def _doc(start_pose=_OMIT, start_gripper=0.0, **extra_params):
    """The act_deploy.yaml SHAPE: policy_leader_node -> ros__parameters -> ...
    plus sibling nodes, so the loader is proven to ignore everything else."""
    params = {"publish_rate_hz": 30.0}
    if start_pose is not _OMIT:
        params["start_pose"] = start_pose
    if start_gripper is not _OMIT:
        params["start_gripper"] = start_gripper
    params.update(extra_params)
    return {
        "policy_leader_node": {"ros__parameters": params},
        "gello_ur_bridge": {"ros__parameters": {"rate_hz": 250.0}},
        "gello_move_to_start": {"ros__parameters": {"velocity_scale": 0.15}},
    }


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------
def test_happy_path_act_shape(tmp_path):
    path = _write(tmp_path, "act_like.yaml", _doc(BANANA, 0.0))
    pose = load_start_pose(path)
    assert isinstance(pose, StartPose)
    assert pose.joints == pytest.approx(tuple(BANANA))
    assert len(pose.joints) == 6
    assert isinstance(pose.joints, tuple)
    assert all(isinstance(v, float) for v in pose.joints)
    assert pose.gripper == 0.0
    assert pose.source == os.path.abspath(path)
    assert os.path.isabs(pose.source)
    assert pose.key == "policy_leader_node.ros__parameters.start_pose"


def test_happy_path_relative_path_becomes_absolute(tmp_path, monkeypatch):
    _write(tmp_path, "rel.yaml", _doc(CARROT, 0.0))
    monkeypatch.chdir(tmp_path)
    pose = load_start_pose("rel.yaml")
    assert pose.source == str(tmp_path / "rel.yaml")


def test_happy_path_ints_are_accepted_as_floats(tmp_path):
    # YAML `1` parses to int; a start pose of whole radians is legal.
    path = _write(tmp_path, "ints.yaml", _doc([3, -1, 1, -1, -1, -3], 0))
    pose = load_start_pose(path)
    assert pose.joints == (3.0, -1.0, 1.0, -1.0, -1.0, -3.0)
    assert all(isinstance(v, float) for v in pose.joints)
    assert pose.gripper == 0.0 and isinstance(pose.gripper, float)


def test_gripper_optional_defaults_to_open(tmp_path):
    path = _write(tmp_path, "nogrip.yaml", _doc(CARROT, start_gripper=_OMIT))
    pose = load_start_pose(path)
    assert pose.gripper == 0.0


def test_gripper_boundaries_accepted(tmp_path):
    for g in (0.0, 0.5, 1.0, 1):
        path = _write(tmp_path, f"g{g}.yaml", _doc(CARROT, g))
        assert load_start_pose(path).gripper == float(g)


def test_frozen_dataclass():
    pose = StartPose(joints=tuple(CARROT), gripper=0.0, source="/x/y.yaml")
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        pose.gripper = 1.0  # type: ignore[misc]


def test_describe_format():
    pose = StartPose(joints=tuple(CARROT), gripper=0.0, source="/some/dir/ifql_deploy.yaml")
    s = pose.describe()
    assert s == (
        "start_pose from ifql_deploy.yaml: "
        "[-3.164, -1.490, 1.726, -1.845, -1.579, -3.269] gripper 0.00"
    )
    assert "\n" not in s


# --------------------------------------------------------------------------
# error branches: file level
# --------------------------------------------------------------------------
def test_missing_file(tmp_path):
    path = str(tmp_path / "nope.yaml")
    with pytest.raises(StartPoseError, match="does not exist") as ei:
        load_start_pose(path)
    assert path in str(ei.value)


def test_not_a_file(tmp_path):
    with pytest.raises(StartPoseError, match="not a regular file") as ei:
        load_start_pose(str(tmp_path))
    assert str(tmp_path) in str(ei.value)


def test_invalid_yaml(tmp_path):
    path = _write_text(tmp_path, "bad.yaml", "policy_leader_node: [unclosed\n  - : : :\n")
    with pytest.raises(StartPoseError, match="invalid YAML") as ei:
        load_start_pose(path)
    assert path in str(ei.value)
    assert "\n" not in str(ei.value)  # one line for the GUI


def test_unreadable_file(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    path = _write(tmp_path, "locked.yaml", _doc(CARROT))
    os.chmod(path, 0)
    try:
        with pytest.raises(StartPoseError, match="cannot read file") as ei:
            load_start_pose(path)
        assert path in str(ei.value)
    finally:
        os.chmod(path, 0o644)


def test_non_utf8_file(tmp_path):
    p = tmp_path / "latin1.yaml"
    p.write_bytes(b"policy_leader_node:\n  ros__parameters:\n    start_pose: [1,1,1,1,1,1] # caf\xe9\n")
    with pytest.raises(StartPoseError) as ei:
        load_start_pose(str(p))
    assert str(p) in str(ei.value)


def test_error_is_a_value_error():
    assert issubclass(StartPoseError, ValueError)


# --------------------------------------------------------------------------
# error branches: document shape
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["", "# only a comment\n", "- a\n- b\n", "just a string\n", "42\n"])
def test_top_level_not_mapping(tmp_path, text):
    path = _write_text(tmp_path, "top.yaml", text)
    with pytest.raises(StartPoseError, match="top level is not a mapping") as ei:
        load_start_pose(path)
    assert path in str(ei.value)


def test_key_path_missing_node(tmp_path):
    path = _write(tmp_path, "nonode.yaml", {"gello_ur_bridge": {"ros__parameters": {}}})
    with pytest.raises(StartPoseError, match="key 'policy_leader_node' is missing"):
        load_start_pose(path)


def test_key_path_missing_ros_parameters(tmp_path):
    path = _write(tmp_path, "noparams.yaml", {"policy_leader_node": {"other": 1}})
    with pytest.raises(
        StartPoseError, match="key 'policy_leader_node.ros__parameters' is missing"
    ):
        load_start_pose(path)


def test_key_path_missing_start_pose(tmp_path):
    path = _write(tmp_path, "nopose.yaml", _doc(start_pose=_OMIT))
    with pytest.raises(
        StartPoseError,
        match="key 'policy_leader_node.ros__parameters.start_pose' is missing",
    ) as ei:
        load_start_pose(path)
    assert path in str(ei.value)


def test_key_path_intermediate_not_mapping(tmp_path):
    path = _write(tmp_path, "scalar.yaml", {"policy_leader_node": "oops"})
    with pytest.raises(StartPoseError, match="'policy_leader_node' is not a mapping"):
        load_start_pose(path)


# --------------------------------------------------------------------------
# error branches: start_pose value
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        "3.106, -1.817, 1.653, -1.618, -1.628, -3.195",  # a string, not a list
        3.106,
        {"pan": 3.106},
        None,
    ],
)
def test_start_pose_not_a_list(tmp_path, bad):
    path = _write(tmp_path, "notlist.yaml", _doc(bad))
    with pytest.raises(StartPoseError, match="must be a list of 6 numbers"):
        load_start_pose(path)


@pytest.mark.parametrize("n", [0, 5, 7, 12])
def test_start_pose_wrong_length(tmp_path, n):
    path = _write(tmp_path, "len.yaml", _doc([0.1] * n))
    with pytest.raises(StartPoseError, match=f"exactly 6 entries \\(got {n}\\)"):
        load_start_pose(path)


@pytest.mark.parametrize("i", range(6))
def test_start_pose_non_number_entry_names_index(tmp_path, i):
    vals = list(CARROT)
    vals[i] = "1.0"  # a string that LOOKS numeric must still be rejected
    path = _write(tmp_path, "str.yaml", _doc(vals))
    with pytest.raises(StartPoseError, match=f"start_pose\\[{i}\\]") as ei:
        load_start_pose(path)
    assert "is not a number" in str(ei.value)


def test_start_pose_null_entry(tmp_path):
    vals = list(CARROT)
    vals[2] = None
    path = _write(tmp_path, "null.yaml", _doc(vals))
    with pytest.raises(StartPoseError, match="start_pose\\[2\\].*is not a number"):
        load_start_pose(path)


@pytest.mark.parametrize("token", [".nan", ".inf", "-.inf"])
def test_start_pose_non_finite(tmp_path, token):
    # Written as raw YAML so the special float tokens survive the dump.
    text = (
        "policy_leader_node:\n  ros__parameters:\n"
        f"    start_pose: [3.106, -1.817, {token}, -1.618, -1.628, -3.195]\n"
    )
    path = _write_text(tmp_path, "nan.yaml", text)
    with pytest.raises(StartPoseError, match="start_pose\\[2\\].*not finite"):
        load_start_pose(path)


@pytest.mark.parametrize("value", [180.0, -180.0, 6.79, -6.79, 1e6])
def test_start_pose_implausible_radians(tmp_path, value):
    vals = list(CARROT)
    vals[0] = value
    path = _write(tmp_path, "deg.yaml", _doc(vals))
    with pytest.raises(StartPoseError, match="start_pose\\[0\\].*not plausible as radians"):
        load_start_pose(path)


def test_start_pose_plausibility_boundary_inclusive(tmp_path):
    limit = 2.0 * math.pi + 0.5
    vals = list(CARROT)
    vals[0] = limit - 1e-9
    assert load_start_pose(_write(tmp_path, "ok.yaml", _doc(vals))).joints[0] == pytest.approx(limit)
    vals[0] = -(limit - 1e-9)
    assert load_start_pose(_write(tmp_path, "ok2.yaml", _doc(vals))).joints[0] == pytest.approx(-limit)
    vals[0] = limit + 1e-6
    with pytest.raises(StartPoseError, match="not plausible"):
        load_start_pose(_write(tmp_path, "over.yaml", _doc(vals)))


# --------------------------------------------------------------------------
# error branches: booleans (the int-subclass trap)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("i", range(6))
def test_boolean_joint_rejected(tmp_path, i):
    # `True` is an int subclass: isinstance(True, int) is True. A naive
    # isinstance(x, (int, float)) check would accept it as 1.0 rad.
    vals = list(CARROT)
    vals[i] = True
    path = _write(tmp_path, "bool.yaml", _doc(vals))
    with pytest.raises(StartPoseError, match=f"start_pose\\[{i}\\].*is not a number") as ei:
        load_start_pose(path)
    assert "bool" in str(ei.value)


def test_boolean_joint_rejected_from_raw_yaml_tokens(tmp_path):
    # PyYAML 1.1 booleans: true/false/yes/no/on/off all parse to bool.
    for tok in ("true", "false", "yes", "no", "on", "off"):
        text = (
            "policy_leader_node:\n  ros__parameters:\n"
            f"    start_pose: [{tok}, -1.817, 1.653, -1.618, -1.628, -3.195]\n"
        )
        path = _write_text(tmp_path, f"b_{tok}.yaml", text)
        with pytest.raises(StartPoseError, match="start_pose\\[0\\].*is not a number"):
            load_start_pose(path)


@pytest.mark.parametrize("g", [True, False])
def test_boolean_gripper_rejected(tmp_path, g):
    path = _write(tmp_path, "bgrip.yaml", _doc(CARROT, g))
    with pytest.raises(StartPoseError, match="start_gripper.*is not a number"):
        load_start_pose(path)


# --------------------------------------------------------------------------
# error branches: start_gripper value
# --------------------------------------------------------------------------
@pytest.mark.parametrize("g", ["0.0", None, [0.0], {"v": 0.0}])
def test_gripper_not_a_number(tmp_path, g):
    path = _write(tmp_path, "gnn.yaml", _doc(CARROT, g))
    with pytest.raises(StartPoseError, match="start_gripper.*is not a number") as ei:
        load_start_pose(path)
    assert path in str(ei.value)


@pytest.mark.parametrize("g", [-0.01, 1.01, 2, -1, 100.0])
def test_gripper_out_of_range(tmp_path, g):
    path = _write(tmp_path, "grange.yaml", _doc(CARROT, g))
    with pytest.raises(StartPoseError, match="start_gripper.*outside \\[0, 1\\]"):
        load_start_pose(path)


@pytest.mark.parametrize("token", [".nan", ".inf"])
def test_gripper_non_finite(tmp_path, token):
    text = (
        "policy_leader_node:\n  ros__parameters:\n"
        f"    start_pose: [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]\n"
        f"    start_gripper: {token}\n"
    )
    path = _write_text(tmp_path, "gnan.yaml", text)
    with pytest.raises(StartPoseError, match="start_gripper"):
        load_start_pose(path)


# --------------------------------------------------------------------------
# resolve_start_pose (env resolution) -- must NEVER raise
# --------------------------------------------------------------------------
def test_resolve_unset():
    pose, reason = resolve_start_pose(environ={})
    assert pose is None
    assert reason == (
        "START_POSE_CONFIG is not set -- export it to a deploy yaml "
        "(e.g. src/gello_policy/config/ifql_deploy.yaml)"
    )


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_resolve_empty_counts_as_unset(value):
    pose, reason = resolve_start_pose(environ={START_POSE_CONFIG_ENV: value})
    assert pose is None
    assert reason.startswith("START_POSE_CONFIG is not set")


def test_resolve_missing_file(tmp_path):
    path = str(tmp_path / "gone.yaml")
    pose, reason = resolve_start_pose(environ={START_POSE_CONFIG_ENV: path})
    assert pose is None
    assert path in reason and "does not exist" in reason


def test_resolve_malformed_file_reports_error_message(tmp_path):
    path = _write(tmp_path, "short.yaml", _doc([1.0, 2.0]))
    pose, reason = resolve_start_pose(environ={START_POSE_CONFIG_ENV: path})
    assert pose is None
    assert path in reason and "exactly 6 entries" in reason


def test_resolve_happy(tmp_path):
    path = _write(tmp_path, "ifql_deploy.yaml", _doc(CARROT, 0.0))
    pose, reason = resolve_start_pose(environ={START_POSE_CONFIG_ENV: path})
    assert pose is not None
    assert pose.joints == pytest.approx(tuple(CARROT))
    assert reason == pose.describe()
    assert reason.startswith("start_pose from ifql_deploy.yaml:")


def test_resolve_strips_whitespace_and_expands_user(tmp_path, monkeypatch):
    path = _write(tmp_path, "p.yaml", _doc(CARROT, 0.0))
    monkeypatch.setenv("HOME", str(tmp_path))
    pose, _ = resolve_start_pose(environ={START_POSE_CONFIG_ENV: "  ~/p.yaml \n"})
    assert pose is not None and pose.source == path


def test_resolve_uses_os_environ_by_default(tmp_path, monkeypatch):
    path = _write(tmp_path, "env.yaml", _doc(BANANA, 0.0))
    monkeypatch.setenv(START_POSE_CONFIG_ENV, path)
    pose, _ = resolve_start_pose()
    assert pose is not None and pose.joints == pytest.approx(tuple(BANANA))
    monkeypatch.delenv(START_POSE_CONFIG_ENV)
    pose, reason = resolve_start_pose()
    assert pose is None and reason.startswith("START_POSE_CONFIG is not set")


def test_resolve_never_raises_on_unexpected_error(tmp_path, monkeypatch):
    # Even a bug-class exception inside the loader must surface as a reason,
    # never as a crash of the GUI's status refresh.
    def boom(_path):
        raise RuntimeError("simulated")

    monkeypatch.setattr(sp, "load_start_pose", boom)
    pose, reason = resolve_start_pose(environ={START_POSE_CONFIG_ENV: "/x.yaml"})
    assert pose is None
    assert "unexpected error" in reason and "simulated" in reason


# --------------------------------------------------------------------------
# the REAL deploy yamls in this workspace
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["act_deploy.yaml", "fm_deploy.yaml", "diffusion_deploy.yaml"])
def test_real_banana_deploy_yamls_parse(name):
    path = os.path.join(_CONFIG_DIR, name)
    assert os.path.isfile(path), f"expected tracked config at {path}"
    pose = load_start_pose(path)
    assert pose.joints == pytest.approx(tuple(BANANA))
    assert pose.gripper == 0.0
    assert pose.source == os.path.abspath(path)
    assert pose.describe().startswith(f"start_pose from {name}:")


def test_real_ifql_deploy_yaml_parses_when_present():
    # Untracked / in-progress in another session: test it when it is there,
    # skip (do not fail) when it is not.
    path = os.path.join(_CONFIG_DIR, "ifql_deploy.yaml")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not present (untracked carrot deploy config)")
    pose = load_start_pose(path)
    assert pose.joints == pytest.approx(tuple(CARROT))
    assert pose.gripper == 0.0
    # The banana/carrot trap this module exists for: pan differs by ~2*pi in
    # VALUE (nearly the same angle) while lift differs by ~0.33 rad.
    banana = load_start_pose(os.path.join(_CONFIG_DIR, "act_deploy.yaml"))
    assert abs(abs(pose.joints[0] - banana.joints[0]) - 2 * math.pi) < 0.02
    assert abs(pose.joints[1] - banana.joints[1]) > 0.3


def test_real_yaml_copied_to_tmp_still_parses(tmp_path):
    # The shape test above uses a synthetic doc; this proves the real file's
    # full contents (all sibling nodes, comments, anchors if any) go through.
    src = os.path.join(_CONFIG_DIR, "act_deploy.yaml")
    dst = tmp_path / "copy_of_act.yaml"
    shutil.copyfile(src, dst)
    assert load_start_pose(str(dst)).joints == pytest.approx(tuple(BANANA))
