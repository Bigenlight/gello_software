"""Run the installed UR control launch without real-robot-only URScript I/O.

The Humble ur_robot_driver version installed on this laptop starts
``urscript_interface`` unconditionally, even when ``use_fake_hardware`` is
true.  This wrapper is used only by our fake-hardware validation path and
filters that one action from the official launch result.  Real hardware keeps
using the official launch file directly.
"""

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, OpaqueFunction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node


def _load_official_launch():
    path = os.path.join(
        get_package_share_directory("ur_robot_driver"),
        "launch",
        "ur_control.launch.py",
    )
    spec = importlib.util.spec_from_file_location("_official_ur_control_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OFFICIAL = _load_official_launch()


def _fake_launch_setup(context, *args, **kwargs):
    actions = _OFFICIAL.launch_setup(context, *args, **kwargs)
    filtered = []
    for action in actions:
        if isinstance(action, Node):
            executable = action.node_executable
            if not isinstance(executable, str):
                executable = perform_substitutions(context, executable)
            if executable == "urscript_interface":
                continue
        filtered.append(action)
    return [
        LogInfo(
            msg="Fake hardware: real-robot urscript_interface disabled "
            "(no connection to robot_ip:30002)."
        ),
        *filtered,
    ]


def generate_launch_description():
    official = _OFFICIAL.generate_launch_description()
    entities = list(official.entities)
    # The official launch ends in one OpaqueFunction invoking launch_setup.
    if not entities or not isinstance(entities[-1], OpaqueFunction):
        raise RuntimeError("Unsupported ur_control.launch.py structure")
    return LaunchDescription(entities[:-1] + [OpaqueFunction(function=_fake_launch_setup)])
