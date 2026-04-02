"""Differential-drive controller for the diffbot mobile robot."""

import numpy as np
from typing import List, TYPE_CHECKING

from tampanda.core.mobile_robot_controller import MobileRobotController
from tampanda.core.base_controller import ControllerStatus

if TYPE_CHECKING:
    from tampanda.core.mobile_robot_env import MobileRobotEnvironment


class DifferentialDriveController(MobileRobotController):
    """Velocity controller for a two-wheeled differential-drive robot.

    Converts unicycle (linear + angular) velocity commands into individual
    left/right wheel angular velocity commands using the standard kinematic
    equations::

        ω_l = (v − ω · L/2) / R
        ω_r = (v + ω · L/2) / R

    where *v* is the linear velocity, *ω* is the angular velocity, *L* is
    the wheelbase (centre-to-centre), and *R* is the wheel radius.

    The controller also provides ``drive_to_pose`` for closed-loop point
    navigation (rotate-to-face, drive, rotate-to-final heading) and
    ``follow_waypoints`` for sequencing a list of (x, y) targets.

    Args:
        model:        MuJoCo model object.
        data:         MuJoCo data object.
        wheelbase:    Distance between left and right wheel centres (m).
        wheel_radius: Radius of each drive wheel (m).
    """

    # Default diffbot parameters matching diffbot.xml
    DEFAULT_WHEELBASE    = 0.310   # 2 × 0.155 m
    DEFAULT_WHEEL_RADIUS = 0.060   # m

    def __init__(
        self,
        model,
        data,
        wheelbase: float = DEFAULT_WHEELBASE,
        wheel_radius: float = DEFAULT_WHEEL_RADIUS,
    ):
        self.model = model
        self.data = data
        self.wheelbase = wheelbase
        self.wheel_radius = wheel_radius
        self.status = ControllerStatus.IDLE

    # ------------------------------------------------------------------
    # Low-level velocity command
    # ------------------------------------------------------------------

    def set_velocity(self, v_lin: float, v_ang: float):
        """Command linear (m/s) and angular (rad/s) velocities.

        Converts to wheel angular velocities and writes to ``data.ctrl``.
        """
        half_L = self.wheelbase / 2.0
        R = self.wheel_radius
        omega_l = (v_lin - v_ang * half_L) / R
        omega_r = (v_lin + v_ang * half_L) / R
        self.data.ctrl[0] = omega_l   # left_wheel_vel
        self.data.ctrl[1] = omega_r   # right_wheel_vel

    # ------------------------------------------------------------------
    # Point navigation
    # ------------------------------------------------------------------

    def drive_to_pose(
        self,
        target_x: float,
        target_y: float,
        target_theta: float,
        env: "MobileEnvironment",
        *,
        linear_speed: float = 0.5,
        angular_speed: float = 1.5,
        heading_gain: float = 3.0,
        position_tol: float = 0.05,
        heading_tol: float = 0.05,
    ) -> bool:
        """Navigate to SE(2) target using a 3-phase strategy.

        Phases:
          1. Rotate in place to face the goal.
          2. Drive straight to the goal (with heading correction).
          3. Rotate in place to the final heading.

        Phases 1 and 3 use proportional angular control clamped to
        ``angular_speed``, which avoids the overshoot and residual spin
        that bang-bang control causes at phase transitions.

        This is a blocking call: it steps the simulation internally until
        the goal is reached or the robot is already there.

        Args:
            target_x, target_y: Goal position in world frame.
            target_theta:       Goal heading in radians.
            env:                MobileEnvironment used for stepping.
            linear_speed:       Maximum forward speed (m/s).
            angular_speed:      Maximum rotation speed (rad/s).
            heading_gain:       Proportional gain (rad/s per rad) for the
                                in-place rotation phases.  Full speed is
                                commanded for errors above
                                ``angular_speed / heading_gain``.
            position_tol:       Distance threshold to declare goal reached (m).
            heading_tol:        Angle threshold (rad) for heading phases.

        Returns:
            True on success, False if the robot was blocked for too long.
        """
        MAX_PHASE_STEPS = 10_000

        def _angle_diff(a, b):
            """Signed shortest angle from b to a, in (−π, π]."""
            d = a - b
            return (d + np.pi) % (2 * np.pi) - np.pi

        self.status = ControllerStatus.MOVING

        # --- Phase 1: rotate to face goal ---
        for _ in range(MAX_PHASE_STEPS):
            x, y, theta = env.get_pose()
            dx, dy = target_x - x, target_y - y
            dist = np.hypot(dx, dy)
            if dist < position_tol:
                break  # already at goal; skip drive phase
            bearing = np.arctan2(dy, dx)
            err = _angle_diff(bearing, theta)
            if abs(err) < heading_tol:
                break
            self.set_velocity(0.0, np.clip(heading_gain * err, -angular_speed, angular_speed))
            env.step()
        self.set_velocity(0.0, 0.0)

        # --- Phase 2: drive to goal position ---
        for _ in range(MAX_PHASE_STEPS):
            x, y, theta = env.get_pose()
            dx, dy = target_x - x, target_y - y
            dist = np.hypot(dx, dy)
            if dist < position_tol:
                break
            bearing = np.arctan2(dy, dx)
            heading_err = _angle_diff(bearing, theta)
            # Proportional heading correction while driving
            v = linear_speed * np.cos(heading_err)
            w = np.clip(2.0 * heading_err, -angular_speed, angular_speed)
            self.set_velocity(max(0.0, v), w)
            env.step()
        self.set_velocity(0.0, 0.0)

        # --- Phase 3: rotate to final heading ---
        for _ in range(MAX_PHASE_STEPS):
            _, _, theta = env.get_pose()
            err = _angle_diff(target_theta, theta)
            if abs(err) < heading_tol:
                break
            self.set_velocity(0.0, np.clip(heading_gain * err, -angular_speed, angular_speed))
            env.step()
        self.set_velocity(0.0, 0.0)

        self.status = ControllerStatus.IDLE
        _, _, theta = env.get_pose()
        x, y, _ = env.get_pose()
        reached = (np.hypot(target_x - x, target_y - y) < position_tol
                   and abs(_angle_diff(target_theta, theta)) < heading_tol)
        return reached

    def follow_waypoints(
        self,
        waypoints: List,
        env: "MobileEnvironment",
        final_theta: float = 0.0,
        **drive_kwargs,
    ) -> bool:
        """Drive through a sequence of (x, y) waypoints.

        The robot turns to face each successive waypoint at the transition
        between segments.  ``final_theta`` sets the heading at the last
        waypoint.

        Args:
            waypoints:    List of ``(x, y)`` tuples.
            env:          MobileEnvironment for simulation stepping.
            final_theta:  Desired heading at the last waypoint (radians).
            **drive_kwargs: Forwarded to :meth:`drive_to_pose`.

        Returns:
            True if all waypoints were reached successfully.
        """
        if not waypoints:
            return True

        self.status = ControllerStatus.MOVING
        for i, (wx, wy) in enumerate(waypoints):
            is_last = (i == len(waypoints) - 1)
            if is_last:
                theta = final_theta
            else:
                nx, ny = waypoints[i + 1]
                theta = np.arctan2(ny - wy, nx - wx)
            ok = self.drive_to_pose(wx, wy, theta, env, **drive_kwargs)
            if not ok:
                self.status = ControllerStatus.ERROR
                return False

        self.status = ControllerStatus.IDLE
        return True

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def step(self):
        """No-op: wheel commands are written directly via set_velocity()."""
        pass

    def get_status(self) -> ControllerStatus:
        return self.status

    def stop(self):
        self.set_velocity(0.0, 0.0)
        self.status = ControllerStatus.IDLE
