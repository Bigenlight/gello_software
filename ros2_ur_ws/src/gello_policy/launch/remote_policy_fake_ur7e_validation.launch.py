#!/usr/bin/env python3
"""Headless robotless validation against an already-running generic gRPC server."""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _validate_params_file(context):
    path = LaunchConfiguration("params_file").perform(context)
    if not os.path.isabs(path):
        raise RuntimeError("params_file must be an absolute path")
    if not os.path.isfile(path):
        raise RuntimeError(f"params_file does not exist: {path}")
    return []


def _validator_finished(event, context):
    del context
    if event.returncode != 0:
        # Raising from a launch event handler makes LaunchService return 1 on
        # ROS 2 Humble.  A plain Shutdown event would incorrectly return zero.
        raise RuntimeError(f"fake UR7e validator exited with code {event.returncode}")
    return [EmitEvent(event=Shutdown(reason="fake UR7e validator passed"))]


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("grpc_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("grpc_port", default_value="50051"),
        DeclareLaunchArgument(
            "params_file",
            description="Required absolute policy/bridge validation parameter YAML.",
        ),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument("validator_timeout_s", default_value="45.0"),
        DeclareLaunchArgument("validator_tolerance_rad", default_value="0.08"),
    ]
    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("gello_policy"), "launch", "ur7e_diffusion_real.launch.py",
            ])
        ),
        launch_arguments={
            "robot_ip": "127.0.0.1",
            "use_fake_hardware": "true",
            "fake_observations": "true",
            "inference_transport": "grpc",
            "act_host": LaunchConfiguration("grpc_host"),
            "grpc_port": LaunchConfiguration("grpc_port"),
            "params_file": LaunchConfiguration("params_file"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
        }.items(),
    )
    validator = Node(
        package="gello_policy",
        executable="remote_policy_fake_ur7e_validator",
        output="screen",
        parameters=[{
            "timeout_s": ParameterValue(
                LaunchConfiguration("validator_timeout_s"), value_type=float
            ),
            "tolerance_rad": ParameterValue(
                LaunchConfiguration("validator_tolerance_rad"), value_type=float
            ),
        }],
    )
    stop_when_validator_finishes = RegisterEventHandler(
        OnProcessExit(
            target_action=validator,
            on_exit=_validator_finished,
        )
    )
    return LaunchDescription(arguments + [
        OpaqueFunction(function=_validate_params_file),
        base,
        validator,
        stop_when_validator_finishes,
    ])
