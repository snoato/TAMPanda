# Confined-shelf library (`library_v1`)

**Inputs.**
- The confined-shelf scene, unchanged from v1: `tampanda/symbolic/domains/confined_shelf/env_builder.py` with the
  default `ConfinedShelfConfig`. It is a 9 × 4 grid at 70 mm pitch with the two back-right deep cells excluded,
  leaving 34 cells. It holds identical cylinders, r = 30 mm, h = 20 cm.
- The robot is compiled from `base_panda_v2.xml`.
- frantik and vamp-planner 0.6.4 (see [../README.md](../README.md)).

`confined_scene.py` builds exactly the scene that the research build of `library_v1` used. Compared field by field
(geoms, bodies, meshes, collision flags, margins, cell centres, structural exclusions), the two scenes are
identical.

**Builder.** `../common/honest_library.py confined-shelf` is the Access-19 v2 library construction on this scene.
- **Grasp families:** the environment's FRONT grasp, and the same grasp with the hand rolled by 180°.
- **Approach:** horizontal along +y, from 6 cm in front of the open face.
- **Retreat with the bottle:** either lift 1.5 cm and slide out (`held`), or lift 4 mm (`held_low`).
- **DP contexts:** empty, behind, and corridors of ±2/±1/±0 columns.
- **Checks:** dense exact checks at 2 mm.
- **Transits:** VAMP, re-validated on the exact meshes.

Cells are independent, so `CELLS=...` rebuilds a subset. Release check (2026-10-09): with `CELLS=shelf_interior__0_2,
shelf_interior__4_3,shelf_interior__1_0,shelf_interior__8_1` and 4 workers, the rebuild took 288 s and reproduced
those cells' candidate rows (every field) and joint paths exactly. A full rebuild with `rebuild.sh` (34 workers, on a
96-core desk shared with the M3 rebuild) took 1,155 s to build and 202 s to validate. Its `.npz` has the shipped
sha256. Its `.json` matches the shipped file in every field, and byte for byte once the build metadata (`built`,
`build_s`) is restored. Validation again gave 240/240 agreement, 68 feasible.

**Rebuild.**

    scripts/libraries/confined_shelf/rebuild.sh OUT_DIR       # WORKERS=34 (one per cell) by default

which runs

    python common/honest_library.py confined-shelf library_v1 --workers 34          (output in OUT_DIR)
    python common/validate_library.py OUT_DIR/library_v1.json 240 libval.json --workers 24 --domain confined-shelf

**Runtime.** The shipped build (2026-10-03) took 878.6 s, about 15 min. Its `--workers` value was not logged; the
script's default is 34, one process per cell.

**Outputs.** `library_v1.json` and `library_v1.npz`: 34 cells, 141 candidates (4–5 per cell). Of those, 137 use the
plain front grasp and 4 the rolled one; 99 retreat `held` and 42 `held_low`. To deploy, copy both files over
`tampanda/symbolic/domains/confined_shelf/honest_data/` and update `../MANIFEST.json`.

**What the library encodes.** In every cell, the smallest conflict set equals the intersection over the cell's
candidates. The library is therefore equivalent to a per-cell rule: a pick or put at (i, j) is feasible iff these
are empty:
- both same-row neighbours (i ± 1, j);
- every cell in front of the target in columns i − 1, i, i + 1;
- for the back-row cells 3_3–7_3, also the front cell two columns to the left, (i − 2, 0).

Nothing behind the target ever blocks. `tests/test_feasibility_libraries.py` checks this rule against the shipped
file.

**Validation (2026-10-03).**
- **Library vs online planning:** 240 random states (target + 2–15 occupied cells, seed 0). 240/240 agree, 68
  feasible; 0 unsound, 0 overly conservative.
- **Minimum clearance over the measured evaluation plans** (every motion, exact): 2.053 mm (2.140 mm on the
  non-monotone 14-object set), 0 samples below 2 mm.
