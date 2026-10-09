# Access-19 v2

A physically honest Access-19: a bookcase centred in front of the Franka, 5 × 7 identical boxes on each of two
levels, ONE frontal grasp for every pick and put, and a collision model in which every arm link, the hand, the
fingers and the held object collide. Feasibility is answered by a precomputed **grasp library**. A query is a
pure-Python bitmask lookup, exact under the collision model, so training workers need no MuJoCo, no IK and no
motion planner.

Status: released with tampanda 1.1.0, validated on the T17c geometry (2026-10-02). The shipped library is
`data/library_T17C.{json,npz}`. To rebuild and validate it, see `scripts/libraries/access19_v2/`. The "Integration"
section below records the plan as of 2026-10-02.

## Why v2

v1's scene compiled the Panda with collisions disabled for every arm mesh. Only tampanda's hand/link7 capsule
stand-ins and the finger pads collided, so accepted v1 actions can have the arm passing through the shelf. v2
compiles `base_panda_v2.xml` (all collision meshes compiled), enables arm-vs-scene contacts for all moving links,
counts arm self-collision, checks the bolted base `link0` against the shelf geometrically (MuJoCo's parent–child
filter hides that pair), and plans with IK and motion planning that actually find the narrow postures the shelf
demands.

## Geometry: `layout.T17C`

| | |
|---|---|
| grid | 5 columns (x) × 7 rows (depth) on both levels; cells `shelf_interior__i_j`, `shelf_top__i_j` (i = column, j = row, row 0 = front) |
| boxes | 3 × 3 × 10 cm, pitch 4.0 cm (x) × 3.6 cm (depth) |
| shelf | front face 37.5 cm in front of the robot base, centred on its axis; compartment floor 34 cm, cavity 24 cm, top deck 59.2 cm; full-height side panels; inner width 39 cm (11.5 cm from the outer column centres to the panels) |
| grasp | site 0.7 cm below the box top, hand horizontal, approach +y; the same grasp on both levels (no regrasp) |

Every cell can be picked and placed on an otherwise empty shelf, and same-row neighbours block each other.
Research history: C3 → T1…T17 → T17c (`a19v2/checkpoints/`, research prototypes in this directory).

### What blocks what (measured by the library)

The Franka hand housing is about 21 cm wide along the finger axis. With a horizontal frontal grasp just below the
box tops, the housing passes through the top few centimetres of every box it travels past. A cell therefore needs
its same-row neighbours empty (lateral blocking), plus, for rows behind the front, the cells in front of it within
roughly ±10 cm of its column. At 4 cm pitch that is ±2 columns: the full width for the centre column. Nothing
behind the target or on the other level ever blocks.

| column pitch | inner width | same-row neighbours block | band in front that must be empty |
|---|---|---|---|
| 4.0 cm (T17c) | 39 cm | yes | ±2 columns |
| 5.5 cm (C3) | 50 cm | yes | ±2 columns |
| 6.0 cm | 47 cm | yes | ±1, partly ±2 |
| 6.5 cm | 49 cm | yes | ±1 (small ±2 parts in the two back interior rows) |
| 7.0 cm | 51 cm | **no** | ±1 |

Correction to the T17c row, measured on the shipped library in the camera-ready check (2026-10-09):
- interior cells in columns 3–4 at rows ≥ 2 also need cells three columns to the left, in front of them, empty in
  every candidate (for example, `shelf_interior__4_6` needs `1_0`–`1_4` empty);
- on the top deck, the band is ±1 column for every cell and ±2 for most.

v1's "only the own column in front" is not achievable with this hand while keeping lateral blocking. Rolling the
hand by 180° changes nothing, because the housing is symmetric. `python scripts/libraries/conflict_sets.py LIB.json`
prints the per-cell conflict sets of a library.

## The stack (local override of tampanda's generic IK / planners)

| module | role |
|---|---|
| `layout.py` | `ShelfGeometry` (+ `T17C`), cell ids and centres; no MuJoCo |
| `shelf.py` | `Scene`: compiled MuJoCo scene with per-cell mocap boxes; `contacts()` reports the static clearance (shelf, base, self) and the clearance to each present object separately |
| `kinematics.py` | grasp frame; closed-form IK via frantik (He & Liu): joint-7 grid (3°) × 4 analytic branches → **all** solutions, not one seeded solution |
| `grasp.py` | per cell the `open` path (open hand onto the box in its cell) and the `held` path (same grasp holding it); bottleneck DP over the IK layers (1.5 cm apart, joint jumps ≤ 0.25 rad); `profile()`: dense exact check at 0.005 rad (at 0.02 rad a fingertip moves ~16 mm between samples); posture reference |
| `motion.py` | transits: VAMP 0.6.4 (sphere model, RRT-Connect) proposes, MuJoCo decides (dense, ≥ 2 mm); straight back-off out of the shelf before handing over to VAMP; exact shortcutting (≥ 1 cm) |
| `library.py` | builds the grasp library (below) |
| `checker.py` | `GraspLibrary`, `Access19V2Checker`: pure Python |
| `online.py` | `OnlineChecker`: the same contract, but plans in the loop: per query, DP against the objects present, dense checks, VAMP transits validated in that scene (IK layers cached per cell) |
| `execute.py` | executes a symbolic plan with library motions, exact-checks every sample, optional video |
| `validate.py` | library vs online planning on random states |
| `problem.py`, `pddl/` | `tabletop-access-v2` domain + problem writer |
| `oracle.py` | data-generation planner: search over pick+put moves on the library; `bfws` (best-first width search) is the mode that solves the hard instances |
| `_deps.py` | locates frantik / vamp |

### The grasp library

Boxes are identical and sit at fixed positions, and arm-to-object distances are pairwise. A motion that is clear
of the shelf and of itself is clear in occupancy S iff no single object in S comes within 2 mm of it. So each
candidate motion carries its **conflict set**: the cells whose box it would touch, measured once with every cell
occupied. Then:

    pick/put on cell c is feasible in S  <=>  some candidate of c has  conflicts ∩ (S − {c}) = ∅

A candidate is a complete motion through home: home → approach → grasp → retreat → home. Pick and put of a cell
are time-reverses of each other and have the same feasibility.

Build (`python -m tampanda.symbolic.domains.access19_v2.library OUT_STEM --workers 60`, about 95 s on a 64-core
desk):
1. Per cell, DP-optimal open/held paths in six occupancy contexts:
   - the empty shelf;
   - only the other level occupied;
   - everything behind and beside the target occupied;
   - corridors of ±2, ±1 and ±0 columns.

   The DP avoids only what is present, so the crowded contexts find paths that keep clear of every box they can.
   Each path is dense-checked and its conflict set measured.
2. One reference posture for the geometry (median iteration). Without it, the first IK solution found is arbitrary
   and consecutive grasps end up 3–7 rad apart.
3. Per cell, the matched (open, held) pairs with subset-minimal conflict sets, plus the 4 best-posture pairs. Each
   gets validated home transits, planned with every cell occupied so that they never conflict; only if that fails
   are they planned on the empty shelf, with their conflict set measured.

Files: `OUT_STEM.json` (geometry, digest, reference posture, bit order, candidate masks; all the checker reads,
~60 KB) and `OUT_STEM.npz` (joint paths, for execution/rendering). A library is tied to its geometry by
`ShelfGeometry.digest()`; tools refuse a mismatched pair.

### Validation (T17c, 2026-10-02)

* **Library vs online planning** (`validate.py`, 700 random states biased towards the relevant cells): 700/700
  agree (164 feasible, 536 infeasible). Online planning = DP against the objects present, dense checks, VAMP
  transits planned in that very scene.
* **Execution** (`execute.py unload:0,2,4` and `unload:1,3`, lower deck → top deck, interior row r → top row 6 − r,
  i.e. all 35 cells): 42/42 and 28/28 actions, 0 contacts in 108,239 exact-checked samples, minimum clearance
  3.0 mm.
* Checker query: microseconds (vs ~220 ms for v1's FAST check).
* **Online checker** (`OnlineChecker`, single process, 140 random states): 140/140 agree with the library.
  Per query 2.1 s mean / 0.6 s median once a cell's IK layers are cached (first visit of a cell 2.6 s mean); an
  infeasible verdict costs 0.6 s, a feasible one 6.9 s, dominated by planning and validating the two transits;
  worst case 20 s.

## Dependencies

frantik (analytic IK) and vamp-planner 0.6.4 are needed only to BUILD a library, validate, execute or render, not
to check. They live in `pip install --target` directories, because frantik's wheel vendors numpy and would replace
the env's. Point `A19V2_PYLIB` at them (os.pathsep-separated); without it, `_deps.py` falls back to the authors' cluster
directories if they exist. MuJoCo 3.4.0. Gotchas this stack depends on:
- MuJoCo builds mesh hulls only for meshes that can collide at compile time (hence `base_panda_v2.xml`).
- Body-level contype/conaffinity gate the broad phase and are fixed at compile time (self-collision needs both).
- `mj_geomDistance` returns spurious zeros for mesh–box pairs (use contact distances).

## Oracle and data (2026-10-02)

v2 moves are exactly reversible (pick and put of a cell have the same feasibility), so the state graph is
undirected. `oracle.py` searches pick+put moves with a blocking-aware heuristic: a misplaced object costs 1, an
object sitting in a needed motion's conflict set costs 1 more, or 2 if it is at its own goal. Greedy search solves
extraction up to 17 objects instantly. The 18–19-object and return-all instances need detours, such as parking
the target in a lower-deck back corner, moving the boxes out of the goal's band and then placing the target.
Best-first width search (`--mode bfws`, novelty over (object, cell) atoms) finds them:
- extraction 18 objects 32/32 and 19 objects 36/36 (4–60 s, 86–134 steps);
- return-all up to 15 objects nearly all, 16–18 partially.

Its memory grows to ~10 GB per worker: use few workers, `--max-states`, and the `a19v2/memguard.sh` watchdog on
shared desks. The datasets live in rgnet `data/pddl_domains/`, built with the rgnet `slurm/data/*_v2.py` scripts.
- **Translation:** `access19v2_{a2ramp,v5flip,selection,heldout}` are v1's sets translated 1:1 (columns
  1/3/5 → 0/2/4, rows and v1's exposed rows kept, goal `shelf_top__2_3`).
- **A′ data:** `access19v2_a2ramp_prefix` has 609 training problems (315 prefixes of oracle plans) and 74
  validation problems.
- **B data:** `access19v2_return_src` holds generated return-all problems, with test layouts excluded.
  `access19v2_bladder` is B's ladder: full problems, hand-offs at the target's first goal arrival, and prefixes.

## Integration (prepared, not done)

1. rgnet: `plangolin/domains/access19_v2.py` registers PDDL domain `tabletop-access-v2`. Its checker wraps
   `GraspLibrary` by default, or `OnlineChecker` with `A19V2_CHECKER=online` (file prepared alongside this work; add it to `sim._DOMAIN_MODULES`). It needs only the JSON
   library, so it is spawn-safe and needs no timeout.
2. Datasets: v1's generators (rowscaled, prefix curriculum, reload ladder) target the 7 × 7 v1 grid. They need the
   5 × 7 grid via `problem.problem_text` and the v2 checker for their oracle planners (the cheap checker makes
   oracle planning fast).
3. Retrain A′ and B on v2 data. All v1 Access-19 numbers stay v1.
4. Choose the geometry first (pitch trade-off above). Changing it means rebuilding the library (~2 min) and
   regenerating the data.
