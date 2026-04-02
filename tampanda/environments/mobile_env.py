"""Differential-drive mobile robot environment."""

import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np
from typing import Optional, List, Tuple

from tampanda.core.mobile_robot_env import MobileRobotEnvironment
from tampanda.utils.rate_limiter import RateLimiter


# Body names that are part of the robot (used for collision detection).
# Contacts between these bodies and any non-robot, non-floor body are flagged.
_DEFAULT_ROBOT_BODIES = [
    "base_link", "left_wheel", "right_wheel",
    "caster_front", "caster_rear",
]

# Freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
_FREEJOINT_QPOS_LEN = 7


class MobileEnvironment(MobileRobotEnvironment):
    """Environment for a differential-drive mobile robot.

    The robot MJCF must contain:
    - A body named ``base_link`` with a ``<freejoint name="base_freejoint"/>``.
    - Two velocity actuators: ``left_wheel_vel`` and ``right_wheel_vel``.
    - Sensors named ``imu_accel`` and ``imu_gyro`` (optional but expected).
    - A site named ``lidar_site`` for lidar attachment.

    Args:
        path:             Path to the scene XML (as produced by SceneBuilder).
        rate:             Simulation rate in Hz.
        collision_bodies: Robot body names checked for obstacle collisions.
                          Defaults to the standard diffbot bodies.
    """

    def __init__(
        self,
        path: str,
        rate: float = 100.0,
        collision_bodies: Optional[List[str]] = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)

        body_names = collision_bodies if collision_bodies is not None else _DEFAULT_ROBOT_BODIES
        self._robot_body_ids: set[int] = set()
        for name in body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._robot_body_ids.add(bid)

        # Locate the base freejoint to read/write pose efficiently.
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
        if jid < 0:
            raise ValueError("Model must contain a joint named 'base_freejoint'.")
        self._base_qposadr = int(self.model.jnt_qposadr[jid])
        self._base_dofadr  = int(self.model.jnt_dofadr[jid])

        # Resting height of base_link (z when robot sits on flat floor).
        # Read from the initial qpos after a forward pass.
        mujoco.mj_forward(self.model, self.data)
        self._ground_z = float(self.data.qpos[self._base_qposadr + 2])

        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.initial_ctrl = self.data.ctrl.copy()

        self.rate = RateLimiter(frequency=rate, warn=False)
        self.sim_time = 0.0
        self.viewer = None

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def get_model(self):
        return self.model

    def get_data(self):
        return self.data

    def launch_viewer(self):
        self.sim_time = 0.0
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        mujoco.mj_forward(self.model, self.data)
        return self.viewer

    def reset(self):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.initial_ctrl
        mujoco.mj_forward(self.model, self.data)
        self.sim_time = 0.0

    def step(self):
        mujoco.mj_step(self.model, self.data)
        dt = self.rate.dt
        self.sim_time += dt
        if self.viewer is not None:
            self.viewer.sync()
        self.rate.sleep()
        return dt

    def rest(self, duration: float):
        steps = int(duration / self.rate.dt)
        for _ in range(steps):
            self.step()

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def check_collisions(self) -> bool:
        """Return False if any robot body is in contact with a non-floor obstacle."""
        if self.data.ncon == 0:
            return True
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            # Worldbody (id 0) hosts the floor — wheel/caster contacts are expected.
            if b1 == 0 or b2 == 0:
                continue
            b1_robot = b1 in self._robot_body_ids
            b2_robot = b2 in self._robot_body_ids
            # Skip pure self-collisions and pure environment-environment contacts.
            if b1_robot == b2_robot:
                continue
            if c.dist < 0.001:
                return False
        return True

    def get_robot_body_ids(self) -> set:
        """Return the set of MuJoCo body IDs that belong to this robot."""
        return self._robot_body_ids

    def is_collision_free(self, configuration: np.ndarray) -> bool:
        """Check whether pose [x, y, theta] is obstacle-free.

        Temporarily moves the robot to the given SE(2) pose, runs a forward
        pass, and checks contacts.  State is fully restored afterwards.

        Args:
            configuration: ``[x, y, theta]`` array (theta in radians).

        Returns:
            True if no obstacle contact is detected.
        """
        x, y, theta = float(configuration[0]), float(configuration[1]), float(configuration[2])

        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()

        half = theta / 2.0
        adr = self._base_qposadr
        self.data.qpos[adr + 0] = x
        self.data.qpos[adr + 1] = y
        self.data.qpos[adr + 2] = self._ground_z
        self.data.qpos[adr + 3] = np.cos(half)   # qw
        self.data.qpos[adr + 4] = 0.0             # qx
        self.data.qpos[adr + 5] = 0.0             # qy
        self.data.qpos[adr + 6] = np.sin(half)    # qz
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        result = self.check_collisions()

        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)
        return result

    # ------------------------------------------------------------------
    # Pose access
    # ------------------------------------------------------------------

    def get_pose(self) -> Tuple[float, float, float]:
        """Return current robot pose as ``(x, y, theta)`` in world frame."""
        adr = self._base_qposadr
        x = float(self.data.qpos[adr])
        y = float(self.data.qpos[adr + 1])
        qw = self.data.qpos[adr + 3]
        qx = self.data.qpos[adr + 4]
        qy = self.data.qpos[adr + 5]
        qz = self.data.qpos[adr + 6]
        theta = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                                 1.0 - 2.0 * (qy * qy + qz * qz)))
        return x, y, theta

    def set_pose(self, x: float, y: float, theta: float):
        """Teleport robot to SE(2) pose (zero velocities)."""
        half = theta / 2.0
        adr = self._base_qposadr
        self.data.qpos[adr + 0] = x
        self.data.qpos[adr + 1] = y
        self.data.qpos[adr + 2] = self._ground_z
        self.data.qpos[adr + 3] = np.cos(half)
        self.data.qpos[adr + 4] = 0.0
        self.data.qpos[adr + 5] = 0.0
        self.data.qpos[adr + 6] = np.sin(half)
        dadr = self._base_dofadr
        self.data.qvel[dadr: dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

