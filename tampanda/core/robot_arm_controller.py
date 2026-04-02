"""Middle-tier base class for robot arm controllers."""

from abc import abstractmethod
from typing import List, Optional
import numpy as np

from tampanda.core.base_controller import BaseController


class RobotArmController(BaseController):
    """Base class for robot arm controllers.

    Extends BaseController with joint-space motion and gripper control,
    which are specific to articulated arm robots.
    """

    @abstractmethod
    def move_to(self, configuration: np.ndarray):
        """Move to a target joint configuration."""
        pass

    @abstractmethod
    def follow_trajectory(self, trajectory: List[np.ndarray], **kwargs):
        """Follow a trajectory of joint configurations."""
        pass

    @abstractmethod
    def open_gripper(self):
        """Open the end-effector gripper."""
        pass

    @abstractmethod
    def close_gripper(self):
        """Close the end-effector gripper."""
        pass
