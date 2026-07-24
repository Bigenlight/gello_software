from setuptools import find_packages, setup

setup(
    name="serl_ur_infra",
    version="0.0.1",
    description=(
        "UR7e + GELLO robot infra for HIL-SERL "
        "(franka_env-compatible env, GELLO intervention)"
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium",
        "numpy",
        "scipy",
        "opencv-python",
        "pynput",
    ],
)
