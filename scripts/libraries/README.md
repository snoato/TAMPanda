# Feasibility libraries: rebuild and validation tooling

Three shelf domains answer pick/put feasibility from a precomputed **grasp library** instead of planning online:

| domain | geometry | shipped library | cells | candidates | rebuild |
|---|---|---|---|---|---|
| Access-19 v2 | T17c | `tampanda/symbolic/domains/access19_v2/data/library_T17C.{json,npz}` | 70 | 293 | [`access19_v2/`](access19_v2/README.md) |
| Confined shelf | unchanged v1 scene | `tampanda/symbolic/domains/confined_shelf/honest_data/library_v1.{json,npz}` | 34 | 141 | [`confined_shelf/`](confined_shelf/README.md) |
| Multi-level shelf v2 | M3 (`shelf_access_v2/data/M3.kv`) | `tampanda/symbolic/domains/shelf_access_v2/data/library_M3.{json,npz}` | 67 | 276 | [`shelf_access_v2/`](shelf_access_v2/README.md) |

A library stores, for every cell, a few complete pick/put motions (home → approach → grasp → retreat → home) and,
per motion, its **conflict set**: the cells whose object the motion would come within 2 mm of, measured once with
every cell occupied. 2 mm is the configured margin; what is actually enforced is no penetration (see the top-level
README, "Known limitations"). Objects are identical and sit at fixed cell positions, and arm-to-object distances are pairwise,
so

    pick/put on cell c is feasible in occupancy S  <=>  some candidate of c has  conflicts ∩ (S − {c}) = ∅

The checkers (`access19_v2/checker.py`, `confined_shelf/honest.py`, `shelf_access_v2/__init__.py`) evaluate this rule
on bitmasks. They are pure Python and need neither MuJoCo nor a planner. Pick and put of a cell are time-reverses of
each other, so they have the same feasibility.

## How the libraries ship

All three are small enough to live in the repository, as package data (`pyproject.toml`). There are no release
assets. [`MANIFEST.json`](MANIFEST.json) lists every precomputed file with its size and sha256;
`tests/test_feasibility_libraries.py` checks the shipped files against it.

| file | bytes | sha256 |
|---|---|---|
| `access19_v2/data/library_T17C.json` | 53,628 | `c660c4338ec5ad0627ea6f8c1dcc520ebc7e484885a6134e9c9cb2088adfe157` |
| `access19_v2/data/library_T17C.npz` | 558,105 | `0fa605385b181df30830bf71e3ab106702cbc0424197a05f9c264b35610edb7c` |
| `confined_shelf/honest_data/library_v1.json` | 29,957 | `1e4ca287d2059b5fa3fbb77069b09937c131d3e55dbf78ef1fab283f65618c43` |
| `confined_shelf/honest_data/library_v1.npz` | 402,651 | `100a9b1f9fda671509378eaaacb5b8026858a1556e65122bbc538f3936dabcb3` |
| `shelf_access_v2/data/library_M3.json` | 60,343 | `1a7a2a16aedbfe4d3da0621d479324781a489af24e6a26520cc12d5347211107` |
| `shelf_access_v2/data/library_M3.npz` | 652,404 | `90aac1fa6924fc1c8a49db563e0bc4768819a2396155cbd7caeb63d880d46b7e` |
| `shelf_access_v2/data/M3.kv` (geometry) | 225 | `7df92a072378d94e3f59b55463296d19ab2d65ad94e3ca543e1bd6d3e00c2506` |

Paths are relative to `tampanda/symbolic/domains/`. The `.json` file is all a checker reads: bit order, per-cell
candidate masks, build metadata. The `.npz` file holds the candidates' joint paths, which only execution and
rendering need.

## Build dependencies

Checking needs only the JSON. Building, validating and executing need:
- MuJoCo 3.4.0 (the shipped libraries were built with it);
- frantik 0.1, closed-form Franka IK after He & Liu (github.com/CoMMALab/frantik);
- vamp-planner 0.6.4, for the RRT-Connect transits.

frantik's wheel vendors its own numpy, so install both with `pip install --target <dir>` (for example
`pip install --target <dir> vamp-planner==0.6.4`). Point `A19V2_PYLIB` at the directories, separated by
`os.pathsep`. The variable is required; there is no default location. The directories are appended to `sys.path`,
so the environment's own numpy and MuJoCo take precedence. See `tampanda/symbolic/domains/access19_v2/_deps.py`. Set `MUJOCO_GL=egl` on headless machines.

## Layout of this directory

| path | what |
|---|---|
| `access19_v2/rebuild.sh` | build + validate + execute T17c with the in-package builder (`access19_v2/library.py`, `validate.py`, `execute.py`) |
| `common/honest_check.py` | collision scene adapter (`EnvScene`) and per-query motion search for the confined and M3 scenes |
| `common/honest_library.py` | library builder for the confined and M3 scenes (the Access-19 v2 construction on another scene) |
| `common/validate_library.py` | library vs online planning on random states |
| `confined_shelf/confined_scene.py`, `rebuild.sh` | confined-shelf scene (one cylinder per cell) and the rebuild command |
| `shelf_access_v2/msv2_scene.py`, `rebuild.sh` | parametric multi-level shelf scene (geometry from `M3.kv`) and the rebuild command |
| `conflict_sets.py` | prints per-cell conflict sets of the shipped libraries (pure Python) |

The `common/` and scene modules are the research scripts that built the shipped confined and M3 libraries. Each
file's docstring records what changed when it was packaged.

## Validation against online planning

For random (cell, occupancy) states, the library verdict is compared with planning that same query online.
"Online" runs the same motion search as the build, per query: DP against the objects actually present, VAMP
transits, and dense exact checks. A disagreement is either unsound (library feasible, online not) or overly
conservative (online feasible, library not).

| library | states | agree | feasible | unsound | conservative |
|---|---|---|---|---|---|
| T17c | 700 | 700 | 164 | 0 | 0 |
| confined `library_v1` | 240 | 240 | 68 | 0 | 0 |
| M3 `library_M3` | 240 | 240 | 123 | 0 | 0 |

**Caveat.** Because online planning is the same search run per query, this validates the precomputation and the
pairwise conflict-set decomposition. It is not a comparison against an independent planner.

The stored rows of these three runs are the fixture `tests/data/library_validation.json`. The tests replay every row
against the shipped libraries.
