"""Base environment class for robot environments."""

from abc import ABC, abstractmethod
import numpy as np


class BaseEnvironment(ABC):
    """Abstract base class for all robot environments.

    Defines the minimal interface shared across arm and mobile robots:
    model/data access, simulation stepping, collision checking, and
    basic object queries.
    """

    @abstractmethod
    def get_model(self):
        """Return the MuJoCo model."""
        pass

    @abstractmethod
    def get_data(self):
        """Return the MuJoCo data."""
        pass

    @abstractmethod
    def launch_viewer(self):
        """Launch the MuJoCo viewer."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the environment to initial state."""
        pass

    @abstractmethod
    def step(self):
        """Step the simulation forward."""
        pass

    @abstractmethod
    def is_collision_free(self, configuration: np.ndarray) -> bool:
        """Check if a configuration is collision-free.

        The configuration space depends on the robot type:
        arm environments take a joint configuration vector,
        mobile environments take an [x, y, theta] SE(2) pose.
        """
        pass

    def get_object_id(self, object_name: str) -> int:
        """Return the MuJoCo body ID for a named object."""
        import mujoco
        oid = mujoco.mj_name2id(
            self.get_model(), mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        if oid == -1:
            raise ValueError(f"Object '{object_name}' not found in model.")
        return oid

    def get_object_position(self, object_name: str) -> np.ndarray:
        """Get the world-frame position of a named body."""
        return self.get_data().xpos[self.get_object_id(object_name)].copy()

    def get_object_orientation(self, object_name: str) -> np.ndarray:
        """Get the world-frame quaternion [w,x,y,z] of a named body."""
        return self.get_data().xquat[self.get_object_id(object_name)].copy()

    def forward(self):
        """Run mj_forward to update all derived quantities."""
        import mujoco
        mujoco.mj_forward(self.get_model(), self.get_data())
