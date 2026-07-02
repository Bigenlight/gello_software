import os
from glob import glob

from setuptools import setup, find_packages

package_name = 'ur_gello_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
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
    description='GELLO -> UR5e ROS2 teleop bringup (RViz visualization)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gello_publisher = ur_gello_bringup.gello_publisher_node:main',
            'gello_ur_bridge = ur_gello_bringup.gello_ur_bridge_node:main',
            'robotiq_urcap = ur_gello_bringup.robotiq_urcap_node:main',
            'fake_gello = ur_gello_bringup.fake_gello_node:main',
        ],
    },
)
