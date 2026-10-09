# Changelog

## 1.1.0 (unreleased)

The physically honest feasibility stack behind the camera-ready results: the whole arm collides, and the shelf
domains decide feasibility with precomputed grasp libraries. See the README, "Collision model and feasibility
libraries".

### Added

- **Robot model:** `environments/assets/franka_emika_panda/base_panda_v2.xml`, the Panda with convex hulls compiled
  for every link collision mesh (`conaffinity="1"`). `base_panda.xml` is unchanged.
- **Access-19 v2** (`symbolic/domains/access19_v2/`):
  - the T17c bookcase geometry and the grasp library `library_T17C`;
  - the pure-Python `Access19V2Checker` and the online checker;
  - the oracle planner and the `tabletop-access-v2` PDDL domain and problem writer.
- **Confined shelf:** the grasp library `library_v1`, with a pure-Python checker and an oracle in
  `confined_shelf/honest.py`.
- **Multi-level shelf v2** (`symbolic/domains/shelf_access_v2/`): geometry M3 and the grasp library `library_M3`.
- **3D stacking:** honest checks on the real link meshes (`multilevel_blocks/honest.py`, `honest_aik.py`,
  `honest_lib.py`, `honest_grip.py`, and the prototype `honest_lib_cont.py`). `make_multilevel_blocks_builder` takes
  `robot_xml=`.
- **3D stacking curriculum:** the mini-L4 templates `mini_bridge_2pillars` and `mini_tower_on_bridge`, at level 3.
- **Kinematics:** `tampanda/kinematics/`, an analytic Panda model (DH forward kinematics, Jacobian, limits) and its
  check against the MuJoCo model (`verify_panda.py`).
- **Library tooling:** `scripts/libraries/`, with:
  - rebuild and validation tooling for the three grasp libraries;
  - `MANIFEST.json`, giving sizes and sha256 of every precomputed file;
  - `conflict_sets.py`, a conflict-set inspector.
- **Reproduction script:** `scripts/repro_mujoco_runtime_margin.py` reproduces the runtime-margin limitation of
  MuJoCo's mid-phase. It is documented under "Known limitations".
- **Tests:** `tests/test_access19_v2.py` and `tests/test_feasibility_libraries.py`, both pure Python. The second
  checks library integrity, conflict-set structure, and `feasible(c, S)` on the stored validation states.

### Changed

- **Confined-shelf executor:** after a successful pick, it pulls the held cylinder 20 cm straight out of the shelf
  before any lateral move (`post_pick_retreat`).
- **3D-stacking curricula:** the level-3 quota is now 160 instead of 120, so `train_600` generates 640 problems and
  `train_l0_l4` 670.
- **3D-stacking IK seeds:** `multilevel_blocks/ik_seed_lut.npz` was regenerated.
- **Package data:** it now includes the PDDL domain files, the grasp libraries, the M3 geometry and the IK seed
  table, so non-editable installs can find them.
- **Version:** `tampanda.__version__` now matches the package version; it said 1.0.0 before.
