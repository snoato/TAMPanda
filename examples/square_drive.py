"""Drive the diffbot in a square, measuring pose error after each lap.

No planner — just the controller directly against the four corners.
Useful for measuring drift accumulation over multiple iterations.

Usage:
    mjpython examples/square_drive.py
"""

import numpy as np
from tampanda import MobileSceneBuilder, MobileEnvironment, DifferentialDriveController


SIDE   = 1.5   # metres per side
LAPS   = 3
SPEED  = 0.4   # m/s linear
ATOL   = 0.10  # position tolerance (m)


def main():
    env = MobileSceneBuilder().build_env(rate=100.0)
    env.set_pose(0.0, 0.0, 0.0)
    viewer = env.launch_viewer()

    ctrl = DifferentialDriveController(env.get_model(), env.get_data())

    # Square corners: (0,0) → (S,0) → (S,S) → (0,S) → (0,0)
    corners = [
        (SIDE, 0.0),
        (SIDE, SIDE),
        (0.0,  SIDE),
        (0.0,  0.0),
    ]

    print(f"Driving {LAPS} laps of a {SIDE} m square\n")
    print(f"{'Lap':>4}  {'x_err':>7}  {'y_err':>7}  {'dist_err':>9}")
    print("-" * 38)

    for lap in range(1, LAPS + 1):
        ctrl.follow_waypoints(
            corners, env,
            final_theta=0.0,
            linear_speed=SPEED,
            angular_speed=1.2,
            position_tol=ATOL,
        )
        x, y, _ = env.get_pose()
        xe, ye = x - 0.0, y - 0.0
        print(f"{lap:>4}  {xe:>+7.4f}  {ye:>+7.4f}  {np.hypot(xe, ye):>9.4f}")

    env.rest(2.0)
    viewer.close()


if __name__ == "__main__":
    main()
