"""Shared pytest setup for sim_collect. Puts the repo root and the two ROS-free source
trees on sys.path so tests run with the plain .venv interpreter:

    cd /home/laptop3/gello_software && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      .venv/bin/python -m pytest -q sim_collect/tests
"""
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (_ROOT, os.path.join(_ROOT, "ros2_ur_ws", "src", "ur_gello_bringup"),
          os.path.join(_ROOT, "ros2_ur_ws", "src", "gello_recorder")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("SIM_COLLECT_IPC", "tcp")


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


needs_display = pytest.mark.skipif(not has_display(), reason="needs DISPLAY for MUJOCO_GL=glfw rendering")
