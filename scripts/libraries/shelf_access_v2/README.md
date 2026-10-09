# Multi-level shelf v2 library (geometry M3, `library_M3`)

**Inputs.**
- The geometry `tampanda/symbolic/domains/shelf_access_v2/data/M3.kv`:
  - an open two-deck shelf, decks 80 × 45 cm centred at (0.40, 0.58) m with surfaces at z = 0.34 / 0.58 m, and a
    floor separator;
  - a 7 cm pitch both ways;
  - YCB-proxy boxes, 6 × 6 × 9 cm;
  - a front grasp 2.5 cm above the box centre.

  The grid has 67 cells: `floor_left` 5, `floor_right` 4, `middle_deck` 30, `top_deck` 28.
- The robot is compiled from `base_panda_v2.xml`.
- frantik and vamp-planner 0.6.4 (see [../README.md](../README.md)).

`msv2_scene.py` is the parametric scene builder the library was built with. It holds the robot, the static shelf as
boxes, and one free box per cell plus a spare. Its `MSGeometry` defaults are the v1 layout; `from_kv` applies a
geometry file on top of them.

**Builder.** `../common/honest_library.py msv2:<geometry.kv>`, the Access-19 v2 library construction on this scene:
- **Grasp families:** front, top-down x and top-down y, each also with the hand rolled by 180°. No top-down
  candidate survives the build.
- **DP contexts:** empty, other regions occupied, behind, and corridors of ±2/±1/±0 columns.
- **Checks:** dense exact checks at 2 mm.
- **Transits:** VAMP, re-validated on the exact meshes.
- **Fallback:** if a cell ends with fewer than two candidates, more diverse (open, held) pairs are tried for transits.

Cells are independent, so `CELLS=...` rebuilds a subset.

**Rebuild.**

    scripts/libraries/shelf_access_v2/rebuild.sh OUT_DIR      # WORKERS=34 by default

which runs

    python common/honest_library.py msv2:<M3.kv> library_M3 --workers 34            (output in OUT_DIR)
    python common/validate_library.py OUT_DIR/library_M3.json 240 libval.json --workers 24 --domain msv2:<M3.kv>

**How the shipped file was built (2026-10-03).** It took two passes.
1. A full build of the 67 cells took 2,538.3 s. `floor_left__0_1` came out without a candidate, because the
   diverse-pair fallback did not exist yet.
2. That cell was rebuilt alone with the fallback, in 420.3 s, giving 2 candidates, and merged into the library. The
   JSON records this in its `note` field.

The packaged builder includes the fallback. The fallback acts only on cells with fewer than two candidates, and every
other cell had at least two in the first pass. One full build with `rebuild.sh` therefore covers both passes.
Release check (2026-10-09): `CELLS=middle_deck__3_3,top_deck__0_4,floor_left__0_1,floor_right__0_2` with 4
workers took 1,294 s, in a single pass. It reproduced those cells' candidate rows (every field) and joint paths
exactly, including the two `floor_left__0_1` candidates from the fallback. A full rebuild with `rebuild.sh` (34
workers, on a 96-core desk shared with the confined rebuild) took 2,712 s to build, in one pass, and 277 s to
validate. Its `.json` matches the shipped file in every field, and byte for byte once the build metadata (`built`,
`build_s`, the geometry path in `domain`, and the two-pass `note`) is restored. Its `.npz` holds the same 1,380
arrays with identical bytes. Only the member order differs, because the shipped file appends `floor_left__0_1`
from the second pass; saved in the shipped order, the arrays give the shipped sha256. Validation again gave 240/240
agreement, 123 feasible.

**Outputs.** `library_M3.json` and `library_M3.npz`: 67 cells and 276 candidates (2–5 per cell). All candidates are
front grasps: 228 plain and 48 rolled; 155 retreat `held` and 121 `held_low`. To deploy, copy both files over
`tampanda/symbolic/domains/shelf_access_v2/data/` and update `../MANIFEST.json`.

**What the library encodes.**
- **Floor cells:** only the boxes in front, in the same column, block.
- **Deck cells:** both same-row neighbours, every cell in front in columns i − 1, i, i + 1, and, for cells two or
  more rows deep, some cells two columns away.
- **Never blocking:** nothing behind the target, and nothing in another region.
- **Alternative conflict sets:** `middle_deck__3_3` and `top_deck__3_3` each have two, and the action is feasible if
  either is clear.

As a consequence, two v1 OOD problems are unsolvable under this library and were excluded from the M3 evaluation
set:
- `eval_ood_0046`: the same-row blockers on `middle_deck__0_0` and `__1_0` pin each other;
- `eval_ood_0049`: the blockers on `middle_deck__0_1` and `__1_1` pin each other.

`tests/test_feasibility_libraries.py` checks these statements against the shipped file.

**Validation (2026-10-03).**
- **Library vs online planning:** 240 random states (target + 2–15 occupied cells, seed 0). 240/240 agree, 123
  feasible; 0 unsound, 0 overly conservative.
- **Minimum clearance over the measured evaluation plans** (every motion, exact): 1.819 mm, held box vs top deck.
  48–168 samples per set fall below 2 mm. This is the runtime-margin limitation described in the top-level README;
  no penetration occurs.
