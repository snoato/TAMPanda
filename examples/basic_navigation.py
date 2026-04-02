"""A* navigation demo for the differential-drive mobile robot (diffbot).

The scene has a slalom of three half-walls alternating top/bottom, plus a
pillar cluster that forces the robot to thread a gap.  The robot plans with
A*, executes the path, and reports Lidar + IMU readings at the goal.

Run with::

    python examples/basic_navigation.py
"""

import numpy as np

from tampanda import (
    MobileSceneBuilder,
    MobileEnvironment,
    DifferentialDriveController,
    AStarNav,
    Lidar,
    RobotSensors,
)
from tampanda.scenes import WALL_TEMPLATE, PILLAR_TEMPLATE


# ---------------------------------------------------------------------------
# Scene layout  (top view, robot starts at S, navigates to G)
# ---------------------------------------------------------------------------
#
#   y=+1.5 ·····················[wall_1_top]·········[wall_3_top]···
#                                     gap                   gap
#   y=-1.5 ···[wall_0_bot]·············    ·[wall_2_bot]···    ·····
#          +--0--0.5--0.9--1.4--1.8--2.3--2.7--3.2--> x
#          S=(0,0)          [P] [P]            G=(3.2,0)
#
#  wall_0_bot: left side, blocks bottom   → gap is at top  (y > ~0.35)
#  wall_1_top: middle-left, blocks top    → gap is at bot  (y < ~-0.35)
#  pillar_0/1: centre-left cluster, force the path to weave
#  wall_2_bot: middle-right, blocks bot   → gap is at top  (y > ~0.35)
#  wall_3_top: right side, blocks top     → gap is at bot  (y < ~-0.35)

def make_nav_builder() -> MobileSceneBuilder:
    b = MobileSceneBuilder()
    b.add_resource("wall",   WALL_TEMPLATE)
    b.add_resource("pillar", PILLAR_TEMPLATE)
    b._options = {"timestep": "0.004", "integrator": "implicitfast"}

    z_w = 0.30   # wall half-height
    z_p = 0.35   # pillar half-height

    # --- Two-segment stacked walls create a genuine slalom ---
    #
    # Each chicane uses TWO overlapping wall segments to span the full y-range
    # boundary, leaving a gap only on ONE side of y=0:
    #
    #   Chicane A (x≈1.0): gap at y > +0.5  (robot must go UP)
    #   Chicane B (x≈2.3): gap at y < −0.5  (robot must go DOWN → S-curve)
    #
    #  y=+1.4 ══════[A_top]══ gap ─────────── ══════════════════
    #                                ↑                          ↑
    #  y= 0   S→───────────────────── ──────────────────────────→G
    #                                ↓                          ↓
    #  y=−1.4 ════════════════════ ──── ════[B_bot]══ gap ─────
    #         0        1.0            1.65   2.3              3.4
    #
    # Each segment is 1.2 m (template half=0.6).  Two segments centred at
    # y=−1.0 and y=−0.1 together cover y=−1.6..+0.5 ⟹ gap at y > 0.5.
    # Similarly, centres at +0.1 and +1.0 cover y=−0.5..+1.6 ⟹ gap at y < −0.5.

    # Chicane A — gap at top (y > ~0.8 after inflation):
    b.add_object("wall", pos=[1.0, -1.0, z_w], euler=[0, 0, 90], name="wall_a0")  # y −1.6..−0.4
    b.add_object("wall", pos=[1.0, -0.1, z_w], euler=[0, 0, 90], name="wall_a1")  # y −0.7..+0.5

    # Chicane B — gap at bottom (y < ~−0.5 after inflation):
    b.add_object("wall", pos=[2.3,  0.1, z_w], euler=[0, 0, 90], name="wall_b0")  # y −0.5..+0.7
    b.add_object("wall", pos=[2.3,  1.0, z_w], euler=[0, 0, 90], name="wall_b1")  # y +0.4..+1.6

    # Decorative pillar pairs flanking each chicane entrance (well inside the
    # wall-blocked zones so they never appear on any valid path).
    b.add_object("pillar", pos=[1.0,  1.25, z_p], name="pillar_a0")
    b.add_object("pillar", pos=[1.0, -1.25, z_p], name="pillar_a1")
    b.add_object("pillar", pos=[2.3,  1.25, z_p], name="pillar_b0")
    b.add_object("pillar", pos=[2.3, -1.25, z_p], name="pillar_b1")

    return b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    START = (0.0, 0.0)
    GOAL  = (3.4, 0.0)

    print("Building scene …")
    env = make_nav_builder().build_env(env_class=MobileEnvironment, rate=100.0)
    env.set_pose(START[0], START[1], 0.0)

    viewer = env.launch_viewer()

    # --- Sensors ---
    lidar = Lidar(
        env,
        site="lidar_site",
        num_rays=180,
        fov_h=360.0,
        range_max=6.0,
        body_exclude="base_link",
    )
    sensors = RobotSensors(env, imu_accel_name="imu_accel", imu_gyro_name="imu_gyro")

    # --- A* planning ---
    planner = AStarNav(
        env,
        x_range=(-0.3, 4.0),
        y_range=(-1.4, 1.4),
        resolution=0.06,
        robot_radius_buffer=0.08,
    )

    print(f"\nPlanning {START} → {GOAL} …")
    path = planner.plan(START, GOAL)
    if path is None:
        print("No path found — adjust obstacle layout.")
        viewer.close()
        return

    path = planner.smooth_path(path)
    print(f"Path: {len(path)} waypoints after smoothing")
    for i, (px, py) in enumerate(path):
        print(f"  [{i}]  ({px:+.2f}, {py:+.2f})")

    # --- Execution ---
    controller = DifferentialDriveController(env.get_model(), env.get_data())

    print("\nExecuting …")
    ok = controller.follow_waypoints(
        path,
        env,
        final_theta=0.0,
        linear_speed=0.4,
        angular_speed=1.2,
        position_tol=0.12,
    )

    x, y, theta = env.get_pose()
    print(f"\nNavigation {'succeeded' if ok else 'FAILED'}")
    print(f"Final pose: x={x:.3f}  y={y:.3f}  θ={np.degrees(theta):.1f}°")

    # --- Sensor readout ---
    distances = lidar.scan()
    accel = sensors.imu_acceleration()
    gyro  = sensors.imu_angular_vel()
    print(f"\nLidar ({lidar.num_rays} rays, {lidar.range_max} m max):")
    print(f"  min={distances.min():.3f} m  mean={distances.mean():.3f} m")
    if accel is not None:
        print(f"IMU accel [m/s²]:  {np.round(accel, 3)}")
    if gyro is not None:
        print(f"IMU gyro  [rad/s]: {np.round(gyro, 4)}")

    env.rest(3.0)
    viewer.close()


if __name__ == "__main__":
    main()
