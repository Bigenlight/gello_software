import os
from glob import glob

from setuptools import setup, find_packages

package_name = 'gello_policy'

setup(
    name=package_name,
    version='0.1.0',
    # Only the rclpy package belongs in the (py3.10 Humble) ament install space.
    # The sibling `policy_server/` is the py3.12 torch/lerobot side and must NOT be
    # installed here (review L3) — it runs from its source dir in the separate venv.
    packages=find_packages(include=['gello_policy', 'gello_policy.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Theo',
    maintainer_email='tpingouin@gmail.com',
    description='Real-robot ACT deploy for the UR7e via a synthetic GELLO leader + ZMQ inference server',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_leader_node = gello_policy.policy_leader_node:main',
        ],
    },
)
