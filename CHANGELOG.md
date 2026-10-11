# Changelog

## 1.1.0 (2026-10-11)

The physically honest feasibility stack behind the camera-ready results: the whole arm collides, and the shelf
domains decide feasibility with precomputed grasp libraries. See the README, "Collision model and feasibility
libraries".

### Added

- **Licence:** `LICENSE`, MIT, copyright RWTH Aachen University (`pyproject.toml` already declared MIT). The
  Panda model in `environments/assets/franka_emika_panda/` keeps its own Apache-2.0 licence; the README section
  "License" names the copyleft runtime dependencies (`quadprog`, GPL-2.0-or-later; `qpsolvers`, LGPL-3.0).
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
- **Native build dependencies:** frantik 0.1 (analytic Franka IK) and vamp-planner 0.6.4 are imported from the
  `pip install --target` directories listed in `A19V2_PYLIB` (`access19_v2/_deps.py`). The variable is required;
  there is no default location. A missing package raises an `ImportError` that names it and gives the install
  command. Library builds, validation, execution and rendering, and the 3D-stacking checks `honest_aik`,
  `honest_lib` and `honest_grip` need them; the pure-Python checkers do not.
- **Research prototypes:** the Access-19 v2 geometry search behind T17c (`crowd`, `verify*`, `search*`, `stencil*`,
  `transit`, `render_*` and others) is kept for provenance in `research/access19_v2/`, outside the installed
  package and unmaintained.
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

### Fixed

- **Dependencies:** `quadprog` is now declared. `ik/mink_ik.py` hard-codes it as mink's QP solver, so a clean
  install without it raised `SolverNotFound` on every IK call (3D stacking, the tabletop checker).
- **Optional dependency:** `unified-planning`, which `DomainBridge` (`tamp/domain_bridge.py`) imports, is now
  declared as the extra `tamp` (`pip install "tampanda[tamp]"`). Without it the 14 `DomainBridge` tests in
  `tests/test_multilevel_blocks.py` are skipped instead of erroring. `DomainBridge.plan` also needs a
  unified-planning engine, such as `up-pyperplan`; none is pinned.
