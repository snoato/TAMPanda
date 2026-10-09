# Access-19 v2 library (`library_T17C`)

**Inputs.**
- The T17c geometry, `T17C` in `tampanda/symbolic/domains/access19_v2/layout.py`. Its geometry digest is
  `c08ead837b1a`; the library records it, and the tools refuse a library built for another geometry.
- The robot, `tampanda/environments/assets/franka_emika_panda/base_panda_v2.xml`.
- frantik and vamp-planner 0.6.4 (see [../README.md](../README.md)).

**Builder.** It is part of the package: `tampanda/symbolic/domains/access19_v2/library.py`, documented in
`access19_v2/README.md`, "The grasp library". Every cell gets DP-optimal open and held paths in six occupancy contexts
(empty shelf, other level occupied, everything behind and beside the target occupied, and corridors of ±2, ±1 and ±0
columns). One reference posture is fitted for the whole geometry, so the build must cover all cells to reproduce the
shipped file. Each cell keeps its subset-minimal and best-posture (open, held) pairs, each with validated home
transits.

**Rebuild.**

    scripts/libraries/access19_v2/rebuild.sh OUT_DIR          # WORKERS=60 as in the shipped build

which runs

    python -m tampanda.symbolic.domains.access19_v2.library  OUT_DIR/library_T17C --workers 60
    python -m tampanda.symbolic.domains.access19_v2.validate OUT_DIR/library_T17C OUT_DIR/validate_T17C.json --n 700
    python -m tampanda.symbolic.domains.access19_v2.execute  OUT_DIR/library_T17C unload:0,2,4 --report OUT_DIR/exec_T17C_024.json
    python -m tampanda.symbolic.domains.access19_v2.execute  OUT_DIR/library_T17C unload:1,3   --report OUT_DIR/exec_T17C_13.json

**Runtime.**
- The shipped build (2026-10-02): 93.7 s on a 64-core desk with `--workers 60`.
- Release check (2026-10-09): rebuilt from scratch on a 96-core desk with `--workers 60` in 161 s. Every candidate row
  of all 70 cells (masks, costs, routes, transit lengths), the reference posture and every joint path in the `.npz`
  were identical to the shipped files.

**Outputs.** `library_T17C.json` (geometry, digest, reference posture, bit order, candidate masks) and
`library_T17C.npz` (joint paths). The library has 70 cells and 293 candidates (4–6 per cell). To deploy a rebuilt
library, copy both files over `tampanda/symbolic/domains/access19_v2/data/` and update `../MANIFEST.json`.

**Validation (2026-10-02).**
- **Library vs online planning** (`validate.py`, 700 random states biased towards the cells that matter): 700/700
  agree, 164 feasible and 536 infeasible; 0 unsound, 0 overly conservative. Online planning means DP against the
  objects present, dense checks, and VAMP transits planned in that very scene.
- **Online checker** (`OnlineChecker`, 140 random states): 140/140 agree with the library.
- **Execution** (`execute.py unload:0,2,4` and `unload:1,3`, all 35 lower-deck cells moved to the top deck): 42/42
  and 28/28 actions, 0 contacts in 108,239 exact-checked samples, minimum clearance 3.0 mm.
- **Minimum clearance over the measured evaluation plans** (every motion, exact): 3.034 mm in every measured set
  and variant (two-phase evaluations, including the fresh 18-blocker set), 0 samples below 2 mm.

Pick and put are time-reverses, so the state graph is undirected. `access19_v2/oracle.py` plans on this library;
it is used for data generation.
