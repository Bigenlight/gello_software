"""Evaluate nested launch arguments without starting processes or ROS nodes."""

from pathlib import Path

import pytest

pytest.importorskip("launch")
pytest.importorskip("launch_ros")

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetLaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.utilities import evaluate_parameters


def expand_include(include, context):
    descriptions = []
    for action in include.execute(context):
        if isinstance(action, SetLaunchConfiguration):
            action.execute(context)
        elif isinstance(action, LaunchDescription):
            descriptions.append(action)
    assert len(descriptions) == 1
    description = descriptions[0]
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    return description


def find_include(actions, context, filename):
    for action in actions:
        if isinstance(action, IncludeLaunchDescription):
            action.launch_description_source.get_launch_description(context)
            if action.launch_description_source.location.endswith(filename):
                return action
    raise AssertionError(f"Missing include: {filename}")


@pytest.mark.parametrize("package,filename", [
    ("gello_policy", "ur7e_diffusion_real.launch.py"),
    ("ur_gello_bringup", "ur7e_gello_real.launch.py"),
])
def test_calibration_reaches_nested_robot_description(package, filename, tmp_path):
    driver_share = Path(get_package_share_directory("ur_robot_driver"))
    rsp_path = driver_share / "launch" / "ur_rsp.launch.py"
    if not rsp_path.exists():
        pytest.skip("driver predates the split robot description launch")

    import yaml

    description_share = Path(get_package_share_directory("ur_description"))
    calibration = yaml.safe_load(
        (description_share / "config" / "ur7e" / "default_kinematics.yaml").read_text()
    )
    calibration["kinematics"]["hash"] = "jazzy_port_calibration_probe"
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(yaml.safe_dump(calibration))
    context = LaunchContext()
    context.launch_configurations.update({
        "robot_ip": "127.0.0.1",
        "kinematics_params_file": str(calibration_path),
        "launch_rviz": "false",
    })
    root_path = Path(get_package_share_directory(package)) / "launch" / filename
    root = expand_include(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(root_path)),
            launch_arguments={"robot_ip": "127.0.0.1"}.items(),
        ), context
    )
    driver_include = find_include(root.entities, context, "ur_control.launch.py")
    driver = expand_include(driver_include, context)
    from launch.actions import OpaqueFunction

    setup = next(action for action in driver.entities if isinstance(action, OpaqueFunction))
    driver_actions = setup.execute(context)
    rsp_include = find_include(driver_actions, context, "ur_rsp.launch.py")
    rsp = expand_include(rsp_include, context)
    assert context.launch_configurations["kinematics_params_file"] == str(calibration_path)
    from launch_ros.actions import Node

    publisher = next(
        action for action in rsp.entities
        if isinstance(action, Node)
        and action.node_package == "robot_state_publisher"
    )
    parameters = evaluate_parameters(context, publisher._Node__parameters)
    robot_description = next(
        entry["robot_description"] for entry in parameters if "robot_description" in entry
    )
    assert "jazzy_port_calibration_probe" in robot_description
