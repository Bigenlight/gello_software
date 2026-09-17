#!/usr/bin/env bash
# Install ROS 2 Jazzy + the UR7e/GELLO/RealSense stack on this PC (Ubuntu 24.04 noble).
#
# Run this ONCE, by hand, with sudo available:   bash install_jazzy.sh
# It only installs system packages -- it does NOT build the workspace and does NOT
# touch the robot.  Build is a separate step (../build_ur7e.sh).
#
# WHY conda is deactivated below: this machine auto-activates miniconda `base`, whose
# python3 is 3.13.  Jazzy's rclpy/colcon C extensions are built against the system
# 3.12 -- a `python3` that resolves into miniconda silently breaks `rosdep` and
# `colcon build`.  We force the system interpreter for the whole script.
set -euo pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "" ]; then
    echo "### conda env '${CONDA_DEFAULT_ENV}' is active -- removing miniconda from PATH for this script"
fi
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)"
export PATH
echo "### python3 -> $(command -v python3)  ($(python3 --version 2>&1))"
case "$(command -v python3)" in
    /usr/bin/python3) ;;
    *) echo "ERROR: python3 is not /usr/bin/python3.  Refusing to continue." >&2; exit 1 ;;
esac

. /etc/os-release
[ "${VERSION_CODENAME}" = "noble" ] || { echo "ERROR: expected Ubuntu noble, got ${VERSION_CODENAME}" >&2; exit 1; }

# ---------------------------------------------------------------- 1. ROS 2 apt repo
sudo apt update
sudo apt install -y curl gnupg lsb-release software-properties-common
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${VERSION_CODENAME} main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

# ------------------------------------------------- 2. ROS 2 Jazzy + build tooling
# desktop (not ros-base): the bringup launches RViz on several paths and rqt is
# useful for the first hardware smoke.  ~2.5 GB; 403 GB free at install time.
sudo apt install -y \
    ros-jazzy-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-pip \
    python3-grpcio \
    python3-protobuf \
    python3-h5py \
    python3-zmq

# --------------------------------------------------------------- 3. UR7e driver
# Jazzy ships ur_robot_driver 3.8.0 (the laptop3 Humble box ran 2.13.2 -- a major
# version jump; the switch/handshake behaviour is what the first smoke test checks).
sudo apt install -y \
    ros-jazzy-ur \
    ros-jazzy-ur-robot-driver \
    ros-jazzy-ur-controllers \
    ros-jazzy-ur-description \
    ros-jazzy-ur-calibration \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers

# -------------------------------------------------------------- 4. RealSense D435
sudo apt install -y \
    ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-camera-msgs \
    ros-jazzy-librealsense2

# udev rules for non-root USB access to the D435.  Without these the camera node
# starts and then fails with LIBUSB_ERROR_ACCESS -- the usbfs node is root:root.
# ros-jazzy-librealsense2 does NOT ship them, so take them from upstream.
if [ ! -f /etc/udev/rules.d/99-realsense-libusb.rules ]; then
    sudo curl -sSL \
        https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
        -o /etc/udev/rules.d/99-realsense-libusb.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "### installed 99-realsense-libusb.rules -- REPLUG both cameras afterwards"
fi

# --------------------------------------------------------------- 5. GELLO / sim diagnostics / misc
# ffmpeg provides libx264 for sim_collect/eval/render_eval_video.py. The Python
# requirements cover frame composition, but the final H.264/yuv420p encode is an
# explicit system-process contract and fails closed when ffmpeg is unavailable.
sudo apt install -y \
    ffmpeg \
    ros-jazzy-dynamixel-sdk
# the gello package imports the pip module, not the ROS one
/usr/bin/python3 -m pip install --user --break-system-packages dynamixel-sdk

# ------------------------------------------------------------------- 6. rosdep
sudo rosdep init 2>/dev/null || true
rosdep update

echo
echo "### DONE.  Next (no sudo needed):"
echo "###   bash $(dirname "$(readlink -f "$0")")/../build_ur7e.sh"
