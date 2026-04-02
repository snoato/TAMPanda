"""Base controller class."""

from abc import ABC, abstractmethod
import enum


class ControllerStatus(enum.Enum):
    """Controller status enumeration."""
    IDLE = "idle"
    MOVING = "moving"
    ERROR = "error"
    GRASPING = "grasping"


class BaseController(ABC):
    """Abstract base class for all robot controllers.

    Defines the minimal interface shared across arm and mobile controllers.
    """

    @abstractmethod
    def step(self):
        """Execute one control step."""
        pass

    @abstractmethod
    def get_status(self) -> ControllerStatus:
        """Get the current controller status."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the controller."""
        pass
