"""Middle-tier base class for mobile robot controllers."""

from abc import abstractmethod
from typing import List, TYPE_CHECKING

from tampanda.core.base_controller import BaseController

if TYPE_CHECKING:
    from tampanda.core.mobile_robot_env import MobileRobotEnvironment


class MobileRobotController(BaseController):
    """Base class for mobile robot controllers.

    Extends BaseController with unicycle velocity commands and waypoint
    navigation, which are specific to ground-based mobile robots.
    """

    @abstractmethod
    def set_velocity(self, v_lin: float, v_ang: float):
        """Command linear (m/s) and angular (rad/s) velocities."""
        pass

    @abstractmethod
    def drive_to_pose(
        self,
        target_x: float,
        target_y: float,
        target_theta: float,
        env: "MobileRobotEnvironment",
        **kwargs,
    ) -> bool:
        """Navigate to an SE(2) target pose. Returns True on success."""
        pass

    @abstractmethod
    def follow_waypoints(
        self,
        waypoints: List,
        env: "MobileRobotEnvironment",
        **kwargs,
    ) -> bool:
        """Drive through a sequence of (x, y) waypoints. Returns True on success."""
        pass
