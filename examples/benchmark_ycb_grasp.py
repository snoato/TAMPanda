"""Grasp benchmark for YCB objects using PointCloudGraspPlanner.

Downloads YCB objects on first run (cached in ~/.cache/manipulation/assets/).
For each object the benchmark:
  1. Teleports it to a random pose on the table.
  2. Captures segmented point clouds from three cameras.
  3. Runs PointCloudGraspPlanner (PCA-based OBB, no ground-truth geometry).
  4. If no candidates are found, rotates the object in 45° Z steps to search for a
     graspable orientation (handles elongated objects that only fit the gripper at
     specific yaws).
  5. Executes the best candidate via PickPlaceExecutor + RRT*.
  6. Checks whether the object was lifted ≥ LIFT_THRESHOLD.

Reports overall grasp success rate and per-object breakdown.

Usage::

    python benchmark_ycb_grasp.py
    python benchmark_ycb_grasp.py --ycb 005_tomato_soup_can 010_potted_meat_can
    python benchmark_ycb_grasp.py --trials 3 --visualize
    python benchmark_ycb_grasp.py --list-objects

Set GITHUB_TOKEN in the environment to raise the GitHub API rate limit from
60 to 5000 requests/hour when downloading many objects.
"""

import argparse
import time
from typing import List, Optional

import mujoco
import numpy as np

from tampanda import FrankaEnvironment, RRTStar
from tampanda.planners import PickPlaceExecutor
from tampanda.planners.pointcloud_grasp_planner import PointCloudGraspPlanner
from tampanda.perception import MujocoCamera
from tampanda.scenes import ArmSceneBuilder, TABLE_SYMBOLIC_TEMPLATE
from tampanda.scenes.assets import YCBDownloader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLE_Z: float = 0.27
LIFT_THRESHOLD: float = 0.08         # object must rise ≥ 8 cm to count as success
HIDE_POS: List[float] = [100.0, 0.0, 0.10]

# Placement region — within reach and clear of the robot base
PLACE_X = (0.25, 0.60)
PLACE_Y = (0.25, 0.58)

CAMERAS = ["top_camera", "side_camera", "front_camera"]

# Curated default: diverse shapes spanning cylinders, boxes, bottles
# All have at least one graspable axis < Franka's 8 cm max opening.
DEFAULT_YCB_OBJECTS = [
    "tomato_soup_can",   # cylinder    ~7.1 cm diameter
    "mustard_bottle",    # bottle      ~5.7 cm wide
    "potted_meat_can",   # flat can    ~4.8 cm tall (needs side grasp)
    "sugar_box",         # box         ~4.5 cm thick
    "bleach_cleanser",   # tall bottle ~6.3 cm wide
    "mug",               # mug with handle — challenging geometry
]


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def _build_scene(ycb_names: List[str]) -> FrankaEnvironment:
    """Download & build a scene with all selected YCB objects parked off-screen."""
    dl = YCBDownloader()
    b  = ArmSceneBuilder()
    b.add_resource("table", TABLE_SYMBOLIC_TEMPLATE)
    b.add_object("table", name="simple_table", pos=[0.0, 0.4, 0.0],
                 quat=[0.0, 0.0, 0.0, 1.0])

    for name in ycb_names:
        print(f"  Fetching {name} …")
        dl.get(name)  # ensures cached; builder resolves via AssetRegistry
        b.add_resource(f"ycb_{name}", {"type": "ycb", "name": name})
        b.add_object(f"ycb_{name}", name=name, pos=HIDE_POS)

    return b.build_env(rate=200.0)


# ---------------------------------------------------------------------------
# Fast headless step — bypasses rate.sleep() so the sim runs at max speed
# ---------------------------------------------------------------------------

def _patch_fast_step(env: FrankaEnvironment) -> None:
    _dt = env.model.opt.timestep

    def _fast_step():
        if env._attached is not None:
            env._apply_attachment()
        mujoco.mj_step(env.model, env.data)
        env.sim_time += _dt
        if env.viewer is not None:
            env.viewer.sync()

    env.step = _fast_step


# ---------------------------------------------------------------------------
# Simulation helpers  (mirrors benchmark_grasping.py)
# ---------------------------------------------------------------------------

def _sim_steps(env: FrankaEnvironment, n: int) -> None:
    for _ in range(n):
        env.step()


def _reset_robot(env: FrankaEnvironment) -> None:
    """Reset arm to home, open gripper, stop controller."""
    env.reset_arm_to_home()
    env.data.ctrl[7] = 255          # gripper open (single actuator, ctrl has 8 entries)
    env.data.qvel[:9] = 0.0
    env.controller.stop()
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    _sim_steps(env, 30)


def _place_object(
    env:      FrankaEnvironment,
    name:     str,
    rng:      np.random.Generator,
    yaw:      Optional[float] = None,
) -> np.ndarray:
    """Teleport *name* to a random on-table pose; return settled position."""
    half_z = float(env.get_object_half_size(name)[2])
    x   = rng.uniform(*PLACE_X)
    y   = rng.uniform(*PLACE_Y)
    z   = TABLE_Z + half_z
    if yaw is None:
        yaw = rng.uniform(0.0, 2.0 * np.pi)
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    env.set_object_pose(name, np.array([x, y, z]), quat)
    mujoco.mj_forward(env.model, env.data)
    _sim_steps(env, 60)
    return env.get_object_position(name).copy()


def _hide_object(env: FrankaEnvironment, name: str) -> None:
    env.set_object_pose(name, np.array(HIDE_POS, dtype=float))
    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------
# Graspability check with Z-rotation search
# ---------------------------------------------------------------------------

def _find_candidates(
    env:       FrankaEnvironment,
    camera:    MujocoCamera,
    pc_planner: PointCloudGraspPlanner,
    name:      str,
    pos:       np.ndarray,
) -> tuple:
    """Return (candidates, z_rotations_tried).

    Tries the current pose first, then rotates in 45° Z steps until candidates
    are found or all 8 orientations are exhausted.  The object is left in the
    pose at which candidates were found (or the last rotation if none found).
    """
    half_z = float(env.get_object_half_size(name)[2])

    for step in range(8):
        # Refresh point cloud at current orientation
        segmented = camera.get_multi_camera_segmented_pointcloud(
            camera_names=CAMERAS,
            width=640, height=480,
            total_samples_per_object=2000,
            min_depth=0.10, max_depth=2.0,
        )
        pts_entry = segmented.get(name)
        if pts_entry is None or len(pts_entry[0]) < 20:
            break

        pts = pts_entry[0]
        candidates = pc_planner.generate_candidates(pts)

        if candidates:
            return candidates, step

        # No candidates — rotate 45° around Z and retry
        yaw = (step + 1) * np.pi / 4
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        env.set_object_pose(name, pos, quat)
        mujoco.mj_forward(env.model, env.data)
        _sim_steps(env, 30)

    return [], 8   # not graspable in any orientation


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _run_trial(
    env:        FrankaEnvironment,
    planner:    RRTStar,
    executor:   PickPlaceExecutor,
    pc_planner: PointCloudGraspPlanner,
    camera:     MujocoCamera,
    name:       str,
    verbose:    bool = True,
) -> dict:
    res = {
        "object":          name,
        "candidates_found": False,
        "z_rotations":     0,
        "grasp_success":   False,
        "grasp_type":      None,
    }

    pos = _place_object(env, name, np.random.default_rng())

    if verbose:
        print(f"  placed at {np.round(pos, 3)}", end="  ", flush=True)

    # Abort if object fell off the table
    if pos[2] < TABLE_Z - 0.02:
        if verbose:
            print("(off-table, skip)")
        _hide_object(env, name)
        return res

    # Find graspable pose (with Z search if needed)
    candidates, z_steps = _find_candidates(env, camera, pc_planner, name, pos)
    res["z_rotations"] = z_steps

    if not candidates:
        if verbose:
            print(f"no grasp candidates in 8 orientations")
        _hide_object(env, name)
        _reset_robot(env)
        return res

    res["candidates_found"] = True
    if verbose and z_steps > 0:
        print(f"(graspable after {z_steps}×45° rotation)", end="  ", flush=True)

    # Re-read position after possible rotation
    pos = env.get_object_position(name).copy()
    rest_z = pos[2]

    if verbose:
        print(f"{len(candidates)} candidates  ", end="", flush=True)

    # Execute pick — passes pre-generated candidates, no geometry oracle used
    success = executor.pick(
        name,
        block_pos=pos,                   # only used if candidates=None
        half_size=np.zeros(3),           # unused
        block_quat=np.array([1, 0, 0, 0]),  # unused
        candidates=candidates,
    )

    if success:
        lifted_z = env.get_object_position(name)[2]
        success = lifted_z > rest_z + LIFT_THRESHOLD

    res["grasp_success"] = success

    # Record grasp type from the winning candidate (executor logs it)
    if success and candidates:
        for cand in candidates:
            res["grasp_type"] = cand.grasp_type.value
            break   # executor tries best-first; first is usually winner

    if verbose:
        tag = "SUCCESS" if success else "FAIL"
        print(tag)

    # Cleanup
    env.detach_object()
    env.controller.open_gripper()
    _sim_steps(env, 100)
    env.clear_collision_exceptions()
    _hide_object(env, name)
    _reset_robot(env)

    return res


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark(
    ycb_names: List[str],
    trials_per_object: int = 3,
    visualize: bool = False,
    seed: int = 42,
) -> List[dict]:
    np.random.seed(seed)

    print(f"\nBuilding scene with {len(ycb_names)} YCB objects …")
    env = _build_scene(ycb_names)
    _patch_fast_step(env)

    if visualize:
        env.launch_viewer()

    planner = RRTStar(env)
    planner.max_iterations   = 2000
    planner.step_size        = 0.15
    planner.goal_sample_rate = 0.15

    pc_planner = PointCloudGraspPlanner(
        table_z=TABLE_Z,
        approach_dist=0.12,
        lift_height=0.22,
    )
    executor = PickPlaceExecutor(
        env, planner,
        use_attachment=True,
        max_plan_iters=2000,
    )
    camera = MujocoCamera(env, width=640, height=480)

    all_results = []
    t_start = time.time()

    for name in ycb_names:
        print(f"\n[{name}]")
        _reset_robot(env)

        for trial in range(trials_per_object):
            print(f"  trial {trial + 1}/{trials_per_object}: ", end="", flush=True)
            res = _run_trial(env, planner, executor, pc_planner, camera, name)
            res["trial"] = trial
            all_results.append(res)

    elapsed = time.time() - t_start
    _print_report(all_results, elapsed, trials_per_object)

    camera.close()
    if visualize and env.viewer is not None:
        env.viewer.close()

    return all_results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(results: List[dict], elapsed: float, tpo: int) -> None:
    print("\n" + "=" * 65)
    print("  YCB GRASP BENCHMARK  —  PointCloudGraspPlanner")
    print("=" * 65)

    n   = len(results)
    ok  = sum(r["grasp_success"]   for r in results)
    cf  = sum(r["candidates_found"] for r in results)
    rot = [r["z_rotations"] for r in results if r["candidates_found"]]

    print(f"\n  Objects tested   : {len(set(r['object'] for r in results))}")
    print(f"  Trials total     : {n}  ({tpo}/object)")
    print(f"  Wall time        : {elapsed:.1f} s")
    print(f"\n  Candidates found : {cf}/{n}  ({100*cf/max(n,1):.1f}%)")
    if rot:
        needs_rot = sum(1 for r in rot if r > 0)
        print(f"  Needed Z-rotate  : {needs_rot}/{len(rot)}  "
              f"(avg {np.mean(rot):.1f} steps when needed)")
    print(f"  Grasp success    : {ok}/{n}  ({100*ok/max(n,1):.1f}%)")

    print(f"\n  Per-object breakdown:")
    for name in dict.fromkeys(r["object"] for r in results):
        obj_r = [r for r in results if r["object"] == name]
        n_ok  = sum(r["grasp_success"]   for r in obj_r)
        n_cf  = sum(r["candidates_found"] for r in obj_r)
        short = name
        print(f"    {short:<30} {n_ok}/{len(obj_r)} grasp  "
              f"({n_cf}/{len(obj_r)} with candidates)")

    print(f"\n  By grasp type (successful grasps):")
    type_counts: dict = {}
    for r in results:
        if r["grasp_success"] and r["grasp_type"]:
            type_counts[r["grasp_type"]] = type_counts.get(r["grasp_type"], 0) + 1
    for gt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {gt:<20} {cnt}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Grasp benchmark for YCB objects using PointCloudGraspPlanner."
    )
    ap.add_argument(
        "--ycb", nargs="+", metavar="NAME",
        help="YCB object names to test (default: curated set of 6)",
    )
    ap.add_argument("--trials",    type=int,  default=3,
                    help="Trials per object (default: 3)")
    ap.add_argument("--visualize", action="store_true",
                    help="Open MuJoCo viewer during benchmark")
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--list-objects", action="store_true",
                    help="Print all available YCB objects and exit")
    args = ap.parse_args()

    if args.list_objects:
        print("Fetching YCB object list …")
        try:
            names = YCBDownloader().list_available()
            print(f"\n{len(names)} objects available:\n")
            for n in names:
                print(f"  {n}")
        except Exception as exc:
            print(f"Error: {exc}\nTip: set GITHUB_TOKEN to avoid rate limits.")
        return

    ycb_names = args.ycb or DEFAULT_YCB_OBJECTS
    benchmark(
        ycb_names=ycb_names,
        trials_per_object=args.trials,
        visualize=args.visualize,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
