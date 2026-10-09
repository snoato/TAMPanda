![TAMPanda](assets/TAMPanda_banner.svg)

[![License](https://img.shields.io/badge/license-MIT-bf5fcf?style=flat-square&labelColor=0d0d12)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-bf5fcf?style=flat-square&labelColor=0d0d12)](pyproject.toml)
[![MuJoCo](https://img.shields.io/badge/mujoco-3.x-7eecdf?style=flat-square&labelColor=0d0d12)](https://mujoco.org)

**TAMPanda** is a MuJoCo-based task and motion planning library developed at the Chair of Machine Learning and Reasoning (i6) at RWTH Aachen University. It combines IK, RRT\* motion planning, and PDDL symbolic planning — primarily for the Franka Emika Panda — along with A\* navigation for mobile robots, a programmatic scene builder, and remote asset support. In other words, just another MuJoCo wrapper.

## Features

**Simulation & Control** — MuJoCo environments for the Franka Panda and a differential-drive mobile robot; collision detection, gravity compensation, position-based trajectory controller

**Planning** — IK via [MINK](https://github.com/kevinzakka/mink); RRT\* with path smoothing; geometry-aware grasp candidate ranking; A\* navigation with occupancy-grid obstacle inflation

**Manipulation** — `PickPlaceExecutor` for end-to-end pick-and-place with multi-candidate retry and kinematic object attachment; `PointCloudGraspPlanner` for rudimentary grasp pose computation on unseen objects from segmented point clouds (WIP)

**Symbolic Planning** — `DomainBridge` connects any PDDL domain to the continuous stack: register Python callables as predicate evaluators or action-tracked fluents, map PDDL actions to executors, and call `ground_state`, `plan`, and `execute_action` without domain-specific glue code; built-in tabletop (grid-based) and blocks-world domains; `ActionFeasibilityChecker` validates symbolic actions against the continuous planner before committing; parallel dataset generation with BFS and optional W&B logging

**Scene & Assets** — `SceneBuilder` assembles scenes from reusable MJCF templates at runtime with hot-reload; `YCBDownloader` / `GSODownloader` fetch ~80 YCB objects and ~1 030 Google Scanned Objects on demand; `MujocoCamera` for RGB, depth, segmentation, and pointcloud rendering

**Gymnasium Integration** — `tampanda.gym` wraps any TAMPanda scene as a standard `gymnasium.Env`; configurable observation spaces (joints, EE pose, object poses, RGB, depth, pointcloud, segmented pointcloud, multi-camera pointcloud); three action spaces (joint delta, joint target, Cartesian EE delta via IK); goal-conditioned `TampandaGoalEnv` with HER-compatible `compute_reward`; `DomainBridge` wired as `bridge_factory` for symbolic state in `info` and predicate-vector goals; `PseudoGraspWrapper` for kinematic grasp attachment; `ExpertActionWrapper` for imitation learning; spawn-safe `make_vec_env` for parallel rollouts

## Setup

```bash
pip install -e .
```

Dependencies: `mujoco>=3.0.0`, `mink>=0.0.1`, `numpy`, `loop-rate-limiters`, `matplotlib`, `opencv-python`

> **macOS:** The MuJoCo passive viewer requires `mjpython` instead of `python`. Headless scripts run fine with standard `python`.

## Quick Start

### Build a scene

```python
from tampanda import ArmSceneBuilder
from tampanda.scenes import TABLE_SYMBOLIC_TEMPLATE, CYLINDER_THIN_TEMPLATE

builder = ArmSceneBuilder()
builder.add_resource("table",    TABLE_SYMBOLIC_TEMPLATE)
builder.add_resource("cylinder", CYLINDER_THIN_TEMPLATE)
builder.add_resource("can",      {"type": "ycb", "name": "master_chef_can"})
builder.add_object("table",    pos=[0.45,  0.00, 0.00])
builder.add_object("cylinder", pos=[0.40,  0.10, 0.36], rgba=[0.8, 0.3, 0.2, 1.0])
builder.add_object("can",      pos=[0.50, -0.10, 0.33])

env = builder.build_env(rate=200.0)
with env.launch_viewer() as viewer:
    while viewer.is_running():
        env.step()
```

### Pick and place

```python
from tampanda import ArmSceneBuilder, RRTStar, GraspPlanner, PickPlaceExecutor
from tampanda.scenes import TABLE_SYMBOLIC_TEMPLATE, BLOCK_SMALL_TEMPLATE
import numpy as np

builder = ArmSceneBuilder()
builder.add_resource("table", TABLE_SYMBOLIC_TEMPLATE)
builder.add_resource("block", BLOCK_SMALL_TEMPLATE)
builder.add_object("table", pos=[0.75, 0.80, 0.00])
builder.add_object("block", pos=[0.45, 0.40, 0.31], rgba=[0.2, 0.5, 0.9, 1.0], name="block_0")

env      = builder.build_env(rate=200.0)
planner  = RRTStar(env)
executor = PickPlaceExecutor(env, planner, GraspPlanner(table_z=0.27))

with env.launch_viewer() as viewer:
    env.rest(2.0)
    ok = executor.pick("block_0",
                       env.get_object_position("block_0"),
                       env.get_object_half_size("block_0"),
                       env.get_object_orientation("block_0"))
    if ok:
        executor.place("block_0", np.array([0.50, 0.25, 0.31]))
    while viewer.is_running():
        env.step()
```

For interactive walkthroughs see [`notebooks/franka_getting_started.ipynb`](notebooks/franka_getting_started.ipynb), [`notebooks/mobile_getting_started.ipynb`](notebooks/mobile_getting_started.ipynb), [`notebooks/domain_bridge_getting_started.ipynb`](notebooks/domain_bridge_getting_started.ipynb), and [`notebooks/gym_grasp_cube.ipynb`](notebooks/gym_grasp_cube.ipynb).

### Gymnasium RL environment

```python
from tampanda.gym import TampandaGymEnv
from tampanda.scenes import ArmSceneBuilder, TABLE_TEMPLATE, BLOCK_MEDIUM_TEMPLATE

builder = ArmSceneBuilder()
builder.add_resource("table", TABLE_TEMPLATE)
builder.add_resource("cube", BLOCK_MEDIUM_TEMPLATE)
builder.add_object("table", pos=[0.45, 0.0, 0.0])
builder.add_object("cube", name="cube_0", pos=[0.45, 0.0, 0.315])
builder.add_camera_orbit("workspace", target=[0.45, 0.0, 0.35], distance=0.75, elevation=45)

env = TampandaGymEnv(
    scene=builder,
    obs=["joints", "ee_pose", "object_poses", "rgb"],
    action_space_type="cartesian_delta",
    cameras=["workspace"],
    image_size=(64, 64),
    reward_fn="dense_grasp",
)
obs, info = env.reset()
```

## Examples

All examples are in `examples/`. On macOS, use `mjpython` for anything that opens a viewer.

**Arm — control and grasping**
- `basic_ik.py` — IK to a target pose, held in viewer
- `basic_rrt.py` — RRT\* between two joint configurations
- `grasping_ik_planner.py`, `grasping_rrt_planner.py` — geometry-aware grasping with ranked candidates
- `blocks_world_rrt.py` — pick two cubes onto a platform with `PickPlaceExecutor`

**Arm — symbolic planning (TAMP)**

The tabletop domain connects PDDL task planning to the continuous planner: symbolic actions (pick, put) are validated with IK + RRT\* before being committed to the plan. `demo_pick_put.py` runs the full loop end-to-end. `DomainBridge` provides a domain-agnostic version of the same pipeline — see the [notebook](notebooks/domain_bridge_getting_started.ipynb) for a walkthrough.

- `symbolic.py` — grid-based PDDL planning in viewer
- `tabletop_interactive.py` — real-time state grounding and interactive tabletop
- `demo_pick_put.py` — full TAMP execution pipeline
- `scene_builder.py` — programmatic scene construction with hot-reload

**Gymnasium / RL**
- `learn_stack_action.py` — SAC + HER to satisfy the `stack` action postconditions via `TampandaGoalEnv` and `DomainBridge`; geometric predicate grounding, shaped reward, `PseudoGraspWrapper`

**Mobile robot**
- `basic_navigation.py` — A\* through a slalom, Lidar/IMU readout at goal
- `square_drive.py` — drift measurement over multiple square laps

**Perception and assets**
- `camera_headless.py`, `pointcloud_demo.py` — RGB and pointcloud capture
- `object_browser.py` — browse, download, and preview YCB / Google Scanned Objects

**Benchmarks** (all headless) — `benchmark_grasping.py`, `benchmark_cylinder_grasping.py`, `benchmark_feasibility.py`, `benchmark_feasibility_params.py`, `benchmark_parallel_rrt.py`, `benchmark_ycb_grasp.py`

## Collision model and feasibility libraries

The physically honest checkers collide the whole arm, not only tampanda's hand and link7 stand-ins:
- the three shelf domains: Access-19 v2 (`symbolic/domains/access19_v2/`), the confined shelf
  (`confined_shelf/honest.py`) and the multi-level shelf v2 (`shelf_access_v2/`);
- the 3D-stacking checks (`multilevel_blocks/honest*.py`).

The default scenes and the open tabletop domain are unchanged.

### What collides

- **Robot model:** `environments/assets/franka_emika_panda/base_panda_v2.xml`. It equals `base_panda.xml` except
  that the collision geoms have `conaffinity="1"`, so MuJoCo compiles a convex hull for every link collision mesh.
  With `0`, no hull is built, and contacts enabled at runtime are silently lost.
- **Arm vs scene and objects:** `access19_v2.shelf._enable_arm_collisions` enables the 20 collision geoms of
  link1–link7, the hand and both fingers against the furniture and every object (meshes, plus boxes for the finger
  pads). It disables tampanda's `hand_capsule` and `link7_capsule` stand-ins, and raises if any arm collision mesh
  has no compiled hull.
- **Held object:** it collides with the furniture and with every other object along the whole motion.
- **Arm self-collision** (the three shelf libraries): arm geoms also collide with each other (contype 2 /
  conaffinity 3, set on the bodies too, because the broad phase filters on body flags). The 3D-stacking checks
  collide link1–link7, hand and fingers with every non-robot geom, and the held block with the scene, but not the
  arm with itself.

**Excluded pairs** (shelf scenes; listed from the compiled scenes):
- each link with its parent link (MuJoCo's default parent–child filter): link1–link0, …, hand–link7,
  fingers–hand;
- arm pairs already within the detection range (3 cm) at the home posture, which are structural neighbours: link5–
  link7 in all three scenes, plus left–right finger in Access-19 v2 (whose home posture has the fingers closed);
- the arm and the object it holds (fingers on the held object);
- `link0`, the bolted base: its geoms are disabled. In Access-19 v2 it is instead checked geometrically against the
  shelf (vertex-in-box).
- confined shelf only: link1 and the table the robot is mounted on, which already overlap at home (a scene defect,
  not avoidable by any motion).

### Margins

| | shelf libraries (Access-19 v2, confined, M3) | 3D stacking (`honest_grip`) |
|---|---|---|
| configured | 3 mm at the IK layers of the approach paths, 2 mm along every motion, dense check every 0.005 rad (max-norm) | 2 mm, exact distances (mid-phase off, exact box–box distances); held block ≥ 2 mm in transit, ≥ −0.1 mm on its own vertical line |
| enforced | **no penetration** (see [Known limitations](#known-limitations)) | 2 mm |

The shelf margins are raised at runtime, and MuJoCo 3.4's mid-phase ignores such margins against multi-geom bodies.
The shelf checkers therefore guarantee no penetration, and the clearance is reported as a measurement: the
minimum over every motion of the measured evaluation plans, checked exactly.

| | measured minimum clearance |
|---|---|
| Access-19 v2 | 3.034 mm in every measured set and variant (two-phase evaluations, including the fresh 18-blocker set); 0 samples below 2 mm |
| confined shelf | 2.053 mm; 0 samples below 2 mm |
| M3 | 1.819 mm (held box vs top deck); 48–168 samples below 2 mm per set |

### Feasibility libraries

The three shelf domains do not plan at query time. They look feasibility up in a precomputed grasp library: per
cell, a few complete pick/put motions (home → approach → grasp → retreat → home), each with its **conflict set**,
the cells whose object it would touch. Objects are identical and contacts are pairwise, so

    pick/put on cell c is feasible in occupancy S  <=>  some candidate of c has  conflicts ∩ (S − {c}) = ∅

A query is a bitmask test in pure Python and takes microseconds, against about 220 ms for v1's FAST check. It needs
no MuJoCo, IK or planner. Pick and put of a cell are time-reverses, so they always get the same verdict.

| domain | library (in `tampanda/symbolic/domains/`) | cells | candidates | shipped build time |
|---|---|---|---|---|
| Access-19 v2, geometry T17c | `access19_v2/data/library_T17C.{json,npz}` | 70 | 293 | 93.7 s (64-core desk, 60 workers) |
| confined shelf | `confined_shelf/honest_data/library_v1.{json,npz}` | 34 | 141 | 878.6 s |
| multi-level shelf v2, geometry M3 | `shelf_access_v2/data/library_M3.{json,npz}` | 67 | 276 | 2,538.3 s + 420.3 s |

The libraries ship in the repository as package data, each under 1 MB. `scripts/libraries/MANIFEST.json` lists
their sizes and sha256 hashes; the `.json` file is all a checker reads.

**Rebuilding a library.** Building needs MuJoCo 3.4.0, frantik 0.1 (analytic Franka IK) and vamp-planner 0.6.4,
installed with `pip install --target` and located through `A19V2_PYLIB` (see `scripts/libraries/README.md`). The
3D-stacking checks plan online and need these packages at query time as well: `honest_aik` needs frantik;
`honest_lib` and `honest_grip` need frantik and VAMP.

```bash
export MUJOCO_GL=egl A19V2_PYLIB=/path/to/frantik_dir:/path/to/vamp_dir
scripts/libraries/access19_v2/rebuild.sh    out/t17c        # build + validate + execute
scripts/libraries/confined_shelf/rebuild.sh out/confined    # build + validate
scripts/libraries/shelf_access_v2/rebuild.sh out/m3         # build + validate
python scripts/libraries/conflict_sets.py                   # per-cell conflict sets of the shipped libraries
```

Each `scripts/libraries/<env>/README.md` gives the inputs, the exact commands, the runtime and the outputs. A
from-scratch rebuild of T17c (2026-10-09) reproduced every candidate and joint path of the shipped file exactly.

**Validation against online planning.** For random (cell, occupancy) states, the library verdict was compared with
planning that query online in that scene: DP against the objects present, VAMP transits, and dense exact checks.

| library | states | agree | feasible | unsound | overly conservative |
|---|---|---|---|---|---|
| T17c | 700 | 700 | 164 | 0 | 0 |
| confined `library_v1` | 240 | 240 | 68 | 0 | 0 |
| M3 `library_M3` | 240 | 240 | 123 | 0 | 0 |

The online search is the same motion search, run per query. The comparison therefore validates the precomputation
and the pairwise conflict-set decomposition, not the library against an independent planner.

For T17c, the full-shelf unload plans also execute with library motions: 42/42 and 28/28 actions, 0 contacts in
108,239 exact-checked samples. The stored validation states are a test fixture: `tests/test_feasibility_libraries.py`
replays them against the shipped libraries and checks the conflict-set structure, without MuJoCo.

## Known limitations

**Clearance margins are not enforced against multi-geom bodies (MuJoCo mid-phase).** The honest feasibility checkers
(Access-19 v2 `access19_v2/`, confined shelf `confined_shelf/honest*`, multi-level shelf `shelf_access_v2/`) and the
library build tooling raise contact margins at runtime (`model.geom_margin[...] = ...`) to detect near-contacts. In
MuJoCo 3.4, the collision mid-phase ignores a margin raised after compile whenever either body of a pair has more than
one geom, as the world body holding the furniture always does. Such near-contacts are then not reported at all.
- **What still holds:** penetrations are always detected, because their bounding volumes overlap regardless of the margin. The checkers therefore guarantee **no penetration**, but **not** the configured clearance margin.
- **Measured:** the minimum clearance of the paper's evaluated motions is reported as a measurement. One certified M3 motion passes 1.82 mm from a deck; the configured margin is 2 mm.
- **Reproduction:** `python scripts/repro_mujoco_runtime_margin.py`, observed with MuJoCo 3.4.0.
- **Workarounds:**
  - compile the margins into the model;
  - or disable the mid-phase: `model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE`.
- **Pitfalls when re-measuring:**
  - With the mid-phase disabled, MuJoCo's primitive box-box collider can report large spurious penetrations for boxes that are separated but inside the margin; measure box-box pairs with an exact convex distance.
  - `mj_geomDistance` can return a spurious 0.0 for separated mesh-box and box-box pairs.
- **Released checkers:** they are the ones used for the paper and are left unchanged.

**Open tabletop domain: no collision guarantee for the arm.** Its online planner checks only the hand, the fingers and
the link7 capsule against the scene. The rest of the arm is not checked, and rendered executions of evaluated plans
penetrate neighbouring objects by up to 8.92 mm. The no-penetration statement above does not cover this domain.

## Citation

If you use TAMPanda in your research, please cite:

```bibtex
@software{tampanda,
  title  = {{TAMPanda}: Task and Motion Planning for the Franka Emika Panda},
  author = {Swoboda, Daniel},
  year   = {2025},
  url    = {https://github.com/snoato/TAMPanda},
}
```

## Acknowledgements

- [MuJoCo](https://mujoco.org) (Google DeepMind) — physics engine
- [MINK](https://github.com/kevinzakka/mink) (Kevin Zakka) — differential IK library
- [VAMP](https://github.com/KavrakiLab/vamp) (Thomason, Kingston, Kavraki) — vectorized sampling-based motion planning (library builds and the 3D-stacking checks)
- [frantik](https://github.com/CoMMALab/frantik) — analytic Franka IK after He & Liu (library builds and the 3D-stacking checks)
- [elpis-lab/ycb_dataset](https://github.com/elpis-lab/ycb_dataset) — YCB object assets
- [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) — Google Scanned Objects assets
