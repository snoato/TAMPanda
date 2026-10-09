# Access-19 v2 research prototypes (unmaintained)

These scripts are the geometry search that led to the shipped T17c bookcase (C3 → T1…T17 → T17c, 2026-10-01/02).
They are kept for provenance only. They are **not** part of the installed `tampanda` package (setuptools packages
`tampanda*` and `manipulation*` only), are not tested, and are not maintained. Most of them work on the earlier
parametric scenes (`geometry.py`, `crowd.py`: a 7 × 7 grid per level), not on the shipped `layout.T17C` /
`shelf.py` stack, and their results are not used by any checker or library.

The supported Access-19 v2 code lives in `tampanda/symbolic/domains/access19_v2/` (geometry, scene, IK, grasp
paths, transits, library build, checker, online checker, execution and video, oracle, PDDL writer); see its
README.

| file | what |
|---|---|
| `geometry.py` | parametric v2 scene (`V2Params`) with the honest collision model |
| `verify.py`, `verify_s4.py`, `verify_s5.py` | per-cell graspability and lateral-blocking verification on that scene (stage variants) |
| `analytic.py` | exhaustive closed-form-IK clearance analysis per cell |
| `reach.py` | reach-envelope maps for frontal grasps past the shelf front edge |
| `rrt_check.py` | transit planning with the full collision model (RRT*) |
| `stencil.py` | blocking stencil on the `geometry.py` scene |
| `render_plan.py` | executes and renders a symbolic plan on the `geometry.py` scene |
| `crowd.py` | crowded-shelf verifier on the parametric crowd scene |
| `crowd_show.py`, `contact_proof.py` | renders of crowded-shelf grasps |
| `transit.py` | VAMP transits between verified grasps, re-checked in MuJoCo |
| `render_v2plan.py` | executes and renders a plan on the crowd scene |
| `stencil_c.py`, `status_map.py`, `rowdrop.py`, `frontal_sweep.py` | blocking stencil, per-cell status map, row height drop, geometry screening on the crowd scene |
| `search.py`, `search3.py` … `search7.py`, `search_top.py` | random + local geometry searches with the crowded verifier |

## Running them

Run a script by its path from the repository root, with `tampanda` importable (installed, or `PYTHONPATH=.`). The
prototypes import each other as sibling modules, which works because Python puts the script's directory on
`sys.path`. `analytic.py`, `crowd.py` and `reach.py` import `frantik` directly, and `transit.py` imports `vamp`, so
put the `A19V2_PYLIB` directories on `PYTHONPATH` as well (see `scripts/libraries/README.md`, "Build
dependencies"). Usage lines are in each file's docstring, for example:

    export MUJOCO_GL=egl
    PYTHONPATH=.:$A19V2_PYLIB python research/access19_v2/verify.py --json out.json
    PYTHONPATH=.:$A19V2_PYLIB python research/access19_v2/crowd.py out.json cells="interior:3_6;top:0_0"

The search scripts' docstrings all show `search.py`; run the numbered variant you want the same way.
