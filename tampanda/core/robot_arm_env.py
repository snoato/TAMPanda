"""Middle-tier base class for robot arm environments."""

from abc import abstractmethod
import numpy as np

from tampanda.core.base_env import BaseEnvironment


class RobotArmEnvironment(BaseEnvironment):
    """Base class for robot arm environments.

    Extends BaseEnvironment with IK access and gravity compensation,
    which are specific to articulated arm robots.
    """

    @abstractmethod
    def get_ik(self):
        """Return the inverse kinematics solver."""
        pass

    def gravity_compensated_target(self, goal_q: np.ndarray) -> np.ndarray:
        """Return a ctrl target adjusted for gravity-induced steady-state error.

        The default implementation is a no-op (returns goal_q unchanged).
        Subclasses for specific robots should override this with an empirically
        calibrated implementation.
        """
        return goal_q
