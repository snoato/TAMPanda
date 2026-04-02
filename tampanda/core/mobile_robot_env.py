"""Middle-tier base class for mobile robot environments."""

from abc import abstractmethod
from typing import Tuple

from tampanda.core.base_env import BaseEnvironment


class MobileRobotEnvironment(BaseEnvironment):
    """Base class for mobile robot environments.

    Extends BaseEnvironment with SE(2) pose access and collision primitives
    specific to ground-based mobile robots.
    """

    @abstractmethod
    def get_pose(self) -> Tuple[float, float, float]:
        """Return current robot pose as (x, y, theta) in world frame."""
        pass

    @abstractmethod
    def set_pose(self, x: float, y: float, theta: float):
        """Teleport robot to SE(2) pose."""
        pass

    @abstractmethod
    def check_collisions(self) -> bool:
        """Return False if any robot body is in contact with a non-floor obstacle."""
        pass

    @abstractmethod
    def get_robot_body_ids(self) -> set:
        """Return the set of MuJoCo body IDs that belong to this robot."""
        pass
