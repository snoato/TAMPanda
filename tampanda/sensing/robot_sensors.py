"""Generic sensor readout for MuJoCo robot models.

Wraps named sensors declared in the model's ``<sensor>`` block and exposes
clean accessors.  Each method returns ``None`` if the corresponding sensor is
not present in the loaded model, so the module works with any MJCF regardless
of which sensors are defined.

Joint torques are always available — they are read directly from
``data.qfrc_actuator`` rather than requiring a sensor definition.

Typical usage::

    from tampanda.sensing import RobotSensors

    sensors = RobotSensors(env)
    print(sensors.available())        # see what's present
    f = sensors.wrist_force()         # np.ndarray (3,) or None
    t = sensors.wrist_torque()        # np.ndarray (3,) or None
    lc, rc = sensors.fingertip_touch() or (None, None)
    q_tau = sensors.joint_torques()   # np.ndarray (n_actuators,), always works
    acc = sensors.imu_acceleration()  # np.ndarray (3,) or None
    gyr = sensors.imu_angular_vel()   # np.ndarray (3,) or None

Sensor name defaults match the convention used if you add sensors to
``base_panda.xml``, but every name is overridable at construction time for
custom robot models.
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from tampanda.core.base_env import BaseEnvironment


class RobotSensors:
    """Reads named sensors from a MuJoCo model.

    Args:
        env: Environment with ``get_model()`` and ``get_data()`` methods.
        wrist_force_name:  Name of the ``force`` sensor on the wrist site.
        wrist_torque_name: Name of the ``torque`` sensor on the wrist site.
        left_touch_name:   Name of the ``touch`` sensor on the left finger.
        right_touch_name:  Name of the ``touch`` sensor on the right finger.
        imu_accel_name:    Name of the ``accelerometer`` sensor.
        imu_gyro_name:     Name of the ``gyro`` sensor.
    """

    def __init__(
        self,
        env: "BaseEnvironment",
        *,
        wrist_force_name: str = "wrist_force",
        wrist_torque_name: str = "wrist_torque",
        left_touch_name: str = "left_touch",
        right_touch_name: str = "right_touch",
        imu_accel_name: str = "imu_accel",
        imu_gyro_name: str = "imu_gyro",
    ):
        self._env = env
        self._sensor_map: dict[str, tuple[int, int]] = {}  # key → (adr, dim)

        lookup = {
            "wrist_force":  wrist_force_name,
            "wrist_torque": wrist_torque_name,
            "left_touch":   left_touch_name,
            "right_touch":  right_touch_name,
            "imu_accel":    imu_accel_name,
            "imu_gyro":     imu_gyro_name,
        }
        model = env.get_model()
        for key, sensor_name in lookup.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sid != -1:
                self._sensor_map[key] = (int(model.sensor_adr[sid]),
                                         int(model.sensor_dim[sid]))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read(self, key: str) -> Optional[np.ndarray]:
        """Return sensor data for *key*, or None if sensor is absent."""
        if key not in self._sensor_map:
            return None
        adr, dim = self._sensor_map[key]
        return self._env.get_data().sensordata[adr: adr + dim].copy()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def available(self) -> list[str]:
        """Return list of sensor keys that are present in the model."""
        return list(self._sensor_map.keys())

    def wrist_force(self) -> Optional[np.ndarray]:
        """3D force at the wrist site in world frame (N). None if not defined."""
        return self._read("wrist_force")

    def wrist_torque(self) -> Optional[np.ndarray]:
        """3D torque at the wrist site in world frame (N·m). None if not defined."""
        return self._read("wrist_torque")

    def fingertip_touch(self) -> Optional[Tuple[float, float]]:
        """Normal contact force on each finger (N) as (left, right).

        Returns None if neither touch sensor is defined.  Individual values
        are None when only one finger sensor is missing.
        """
        left = self._read("left_touch")
        right = self._read("right_touch")
        if left is None and right is None:
            return None
        l_val = float(left[0]) if left is not None else None
        r_val = float(right[0]) if right is not None else None
        return l_val, r_val

    def joint_torques(self) -> np.ndarray:
        """Actuator forces for all actuators (N·m).

        Always available — read directly from ``data.actuator_force``.
        Returns an array of shape ``(n_actuators,)``, i.e. 9 for the Panda
        (7 arm joints + 2 finger actuators).
        """
        return self._env.get_data().actuator_force.copy()

    def imu_acceleration(self) -> Optional[np.ndarray]:
        """3D linear acceleration at the IMU site (m/s²). None if not defined."""
        return self._read("imu_accel")

    def imu_angular_vel(self) -> Optional[np.ndarray]:
        """3D angular velocity at the IMU site (rad/s). None if not defined."""
        return self._read("imu_gyro")
