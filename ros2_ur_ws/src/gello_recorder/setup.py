from setuptools import setup, find_packages

package_name = 'gello_recorder'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Theo',
    maintainer_email='tpingouin@gmail.com',
    description='Diagnostic + camera recorder for the GELLO teleop pipeline',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gello_ur_recorder = gello_recorder.gello_ur_recorder_node:main',
            'gello_recorder_gui = gello_recorder.gello_recorder_gui:main',
            'policy_run_gui = gello_recorder.policy_run_gui:main',
            'reward_classifier = gello_recorder.reward_classifier_node:main',
            'remote_reward_classifier = gello_recorder.remote_reward_classifier_node:main',
            'classifier_view_gui = gello_recorder.classifier_view_gui:main',
        ],
    },
)
