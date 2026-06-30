import pickle
import threading
import time
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from dm_control import mjcf

from gello.robots.robot import Robot

assert mujoco.viewer is mujoco.viewer


def attach_hand_to_arm(
    arm_mjcf: mjcf.RootElement,
    hand_mjcf: mjcf.RootElement,
) -> None:
    """Attaches a hand to an arm.

    The arm must have a site named "attachment_site".

    Taken from https://github.com/deepmind/mujoco_menagerie/blob/main/FAQ.md#how-do-i-attach-a-hand-to-an-arm

    Args:
      arm_mjcf: The mjcf.RootElement of the arm.
      hand_mjcf: The mjcf.RootElement of the hand.

    Raises:
      ValueError: If the arm does not have a site named "attachment_site".
    """
    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)

    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("No attachment site found in the arm model.")

    # Expand the ctrl and qpos keyframes to account for the new hand DoFs.
    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)


def build_scene(
    robot_xml_path: str,
    gripper_xml_path: Optional[str] = None,
    add_scene: bool = False,
    add_cube: bool = False,
    stable_grasp: bool = False,
):
    # assert robot_xml_path.endswith(".xml")

    arena = mjcf.RootElement()

    if stable_grasp:
        # Contact/solver settings that keep a grasped object from slipping out of
        # the gripper. elliptic cone + high impratio make friction firm relative to
        # the normal force; noslip_iterations runs MuJoCo's dedicated anti-slip
        # pass; a smaller timestep + more solver iterations improve contact accuracy
        # (more compute). integrator stays implicitfast (panda.xml default).
        arena.option.cone = "elliptic"
        arena.option.impratio = 10
        arena.option.noslip_iterations = 10
        arena.option.integrator = "implicitfast"
        arena.option.timestep = 0.001
        arena.option.iterations = 150
        arena.option.ls_iterations = 50

    arm_simulate = mjcf.from_path(robot_xml_path)
    # arm_copy = mjcf.from_path(xml_path)

    if gripper_xml_path is not None:
        # attach gripper to the robot at "attachment_site"
        gripper_simulate = mjcf.from_path(gripper_xml_path)
        attach_hand_to_arm(arm_simulate, gripper_simulate)

    if add_scene:
        # Replicate the mujoco_menagerie scene.xml aesthetic (skybox + checker
        # floor + lighting) via the mjcf API so the scene isn't a black void.
        arena.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        arena.visual.headlight.ambient = [0.3, 0.3, 0.3]
        arena.visual.headlight.specular = [0.0, 0.0, 0.0]
        arena.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
        arena.visual.__getattr__("global").azimuth = 120
        arena.visual.__getattr__("global").elevation = -20

        arena.asset.add(
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.0, 0.0, 0.0],
            width=512,
            height=3072,
        )
        arena.asset.add(
            "texture",
            type="2d",
            name="groundplane",
            builtin="checker",
            mark="edge",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            markrgb=[0.8, 0.8, 0.8],
            width=300,
            height=300,
        )
        arena.asset.add(
            "material",
            name="groundplane",
            texture="groundplane",
            texuniform=True,
            texrepeat=[5, 5],
            reflectance=0.2,
        )
        arena.worldbody.add(
            "light",
            pos=[0, 0, 1.5],
            dir=[0, 0, -1],
            directional=True,
        )
        arena.worldbody.add(
            "geom",
            name="floor",
            type="plane",
            material="groundplane",
            size=[5, 5, 0.1],
        )

    arena.worldbody.attach(arm_simulate)
    # arena.worldbody.attach(arm_copy)

    if add_cube:
        # IMPORTANT: add the cube AFTER attaching the arm so the cube's freejoint
        # qpos is appended AFTER the arm joints in the compiled model. This keeps
        # qpos[:num_joints] mapping to the arm. The cube has no actuator, so nu is
        # unchanged. Placed on the floor in front of the panda base (+x direction).
        cube = arena.worldbody.add("body", name="cube", pos=[0.5, 0.0, 0.025])
        cube.add("freejoint", name="cube_free")
        cube.add(
            "geom",
            type="box",
            size=[0.02, 0.02, 0.02],
            rgba=[0.8, 0.2, 0.2, 1.0],
            mass=0.05,
            # condim=6 enables tangential + torsional + rolling friction so the cube
            # does not twist/slip out of the gripper. MuJoCo contact condim/friction
            # are the elementwise max of the two geoms, so raising them on the cube
            # also strengthens the cube<->fingertip-pad contacts (pads default to
            # condim=3, friction=[1,0.005,0.0001]).
            condim=6,
            friction=[2.0, 0.05, 0.001],
        )

    return arena


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                print("Timeout in ZMQLeaderServer serve")
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()


class MujocoRobotServer:
    def __init__(
        self,
        xml_path: str,
        gripper_xml_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
        gripper_builtin: bool = False,
        gripper_invert: bool = False,
        add_scene: bool = False,
        add_cube: bool = False,
        stable_grasp: bool = False,
    ):
        self._has_gripper = gripper_xml_path is not None
        # Some models (e.g. franka panda.xml) bundle the gripper actuator in the
        # main xml instead of attaching a separate gripper_xml. In that case the
        # GELLO's normalized [0,1] gripper command must be rescaled to the last
        # actuator's ctrlrange (panda actuator8 is [0, 255]).
        self._gripper_builtin = gripper_builtin
        # Invert the gripper polarity when the model's "open" end is the high end
        # of the ctrlrange (panda actuator8: 255 = open) but GELLO sends 1 = closed.
        self._gripper_invert = gripper_invert
        arena = build_scene(
            xml_path,
            gripper_xml_path,
            add_scene=add_scene,
            add_cube=add_cube,
            stable_grasp=stable_grasp,
        )

        assets: Dict[str, str] = {}
        for asset in arena.asset.all_children():
            if asset.tag == "mesh":
                f = asset.file
                assets[f.get_vfs_filename()] = asset.file.contents

        xml_string = arena.to_xml_string()
        # save xml_string to file
        with open("arena.xml", "w") as f:
            f.write(xml_string)

        self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        self._data = mujoco.MjData(self._model)

        self._num_joints = self._model.nu

        if self._gripper_builtin:
            self._gripper_ctrlrange = self._model.actuator_ctrlrange[-1].copy()

        self._joint_state = np.zeros(self._num_joints)
        self._joint_cmd = self._joint_state

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

        self._print_joints = print_joints

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints, (
            f"Expected joint state of length {self._num_joints}, "
            f"got {len(joint_state)}."
        )
        if self._has_gripper:
            _joint_state = joint_state.copy()
            _joint_state[-1] = _joint_state[-1] * 255
            self._joint_cmd = _joint_state
        elif self._gripper_builtin:
            # map normalized [0,1] gripper command to the actuator's ctrlrange
            _joint_state = joint_state.copy()
            g = _joint_state[-1]
            if self._gripper_invert:
                g = 1.0 - g
            lo, hi = self._gripper_ctrlrange
            _joint_state[-1] = lo + g * (hi - lo)
            self._joint_cmd = _joint_state
        else:
            self._joint_cmd = joint_state.copy()

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions = self._data.qpos.copy()[: self._num_joints]
        joint_velocities = self._data.qvel.copy()[: self._num_joints]
        ee_site = "attachment_site"
        try:
            ee_pos = self._data.site_xpos.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_mat = self._data.site_xmat.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat)
        except Exception:
            ee_pos = np.zeros(3)
            ee_quat = np.zeros(4)
            ee_quat[0] = 1
        gripper_pos = self._data.qpos.copy()[self._num_joints - 1]
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_pos,
        }

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                self._data.ctrl[:] = self._joint_cmd
                # self._data.qpos[:] = self._joint_cmd
                mujoco.mj_step(self._model, self._data)
                self._joint_state = self._data.qpos.copy()[: self._num_joints]

                if self._print_joints:
                    print(self._joint_state)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    # TODO remove?
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        self._data.time % 2
                    )

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()
