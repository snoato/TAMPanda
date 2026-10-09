"""Confined-shelf collision scene for the honest grasp-library build (tampanda only, no rgnet).

The scene is the one the paper's confined-shelf checker compiles (rgnet ``ConfinedShelfFeasibilityChecker
._resources_for``): ``ConfinedShelfConfig(n_cylinders=n)`` with the default colour grouping,
``make_confined_shelf_builder``, ``build_env(rate=10000.0)`` and ``apply_runtime_tweaks``. Two things differ, both as
in the research build of ``library_v1``:
* the robot is compiled from ``base_panda_v2.xml``, so the real link collision meshes get convex hulls;
* there is one cylinder per cell plus a spare, so every cell can be occupied at once.

The research build reached this scene through rgnet (``a19v2/audit_clipping.make_checker``, which swaps
``ArmSceneBuilder`` for one with ``base=base_panda_v2.xml`` around ``_resources_for``). The executor, symbolic bridge
and IK planner that the checker also creates are not used by the library build and are not created here.

    sc, cells = build()      # honest_check.EnvScene, {cell_id: cylinder centre}
"""
from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import numpy as np

HALF_SIZES = (0.03, 0.03, 0.10)   # cylinder radius, radius, half height (as in the research build)
FRONT_Y = 0.208                   # y of the shelf's open face, the start of the horizontal approach lines

_SCRATCH = []                     # keeps the builders' scratch directories alive for the life of the process


def _builder(n_cylinders: int, robot_xml):
    from tampanda.scenes import ArmSceneBuilder
    from tampanda.symbolic.domains.confined_shelf import ConfinedShelfConfig, make_confined_shelf_builder
    from tampanda.symbolic.domains.confined_shelf import env_builder as EB

    cfg = ConfinedShelfConfig(n_cylinders=n_cylinders)
    groups = [i % cfg.n_color_groups for i in range(cfg.n_cylinders)]
    scratch = tempfile.TemporaryDirectory(prefix="honest_cshelf_")
    _SCRATCH.append(scratch)
    orig = EB.ArmSceneBuilder
    EB.ArmSceneBuilder = functools.partial(ArmSceneBuilder, base=robot_xml)
    try:
        return make_confined_shelf_builder(Path(scratch.name), cfg, cylinder_color_groups=groups)
    finally:
        EB.ArmSceneBuilder = orig


def build():
    """(EnvScene, cells): the honest confined-shelf scene with one cylinder per cell (+1) and the cell centres."""
    import mujoco

    import honest_check as H
    from tampanda.symbolic.domains.access19_v2.shelf import ROBOT_XML
    from tampanda.symbolic.domains.confined_shelf import apply_runtime_tweaks

    _, ws, _ = _builder(1, ROBOT_XML)
    n_cells = len(list(ws.cells()))
    builder, ws, cfg = _builder(n_cells + 1, ROBOT_XML)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    m = env.model
    objs = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("cyl_")]
    sc = H.EnvScene(env, objs, HALF_SIZES, FRONT_Y, "cylinder")
    cells = {c.id: np.asarray(ws.pose_for(c), float) for c in ws.cells()}
    return sc, cells
