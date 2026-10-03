"""Access-19 v2 scene: parametric geometry + an honest collision model (2026-10-01).

v1 problem (verified with tampanda.kinematics.verify_panda): the scene robot ``base_panda.xml`` disables
contacts for every arm mesh and FrankaEnvironment re-enables only link5-7, so link0-link4 (base, shoulder,
upper arm, elbow) pass through the shelf unnoticed. v2 enables the collision mesh of EVERY arm body against
the scene (contype 0 / conaffinity 1: arm-vs-scene contacts, no arm-vs-arm contacts) and removes the
oversized ``hand_capsule`` stand-in, since the real hand mesh now collides. The shelf, cube size, cell
pitch and shelf placement are parameters so the verification loop can search them.
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np

ARM_BODIES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand",
              "left_finger", "right_finger")


@dataclass
class V2Params:
    cube_half: Tuple[float, float, float] = (0.020, 0.020, 0.040)   # v1: 4 x 4 x 8 cm
    inter_row_gap: float = 0.020        # cell pitch = cube side + gap (v1 6 cm)
    interior_height: float = 0.24       # cubicle cavity height (v1)
    wall: float = 0.012
    pedestal: float = 0.40
    shelf_xy: Tuple[float, float] = (0.35, 0.40)   # shelf centre in the robot base frame (v1)
    side_margin: float = 0.10
    top_cells: int = 7
    arm_margin: float = 0.003           # contact margin on arm geoms (v1 used 3 mm on link5-7)
    finger_open: float = 0.04           # per-finger opening during approach (Panda max 0.04)
    keep_hand_capsule: bool = False     # v1 stand-in; v2 uses the real hand mesh
    keep_link7_capsule: bool = False    # tampanda stand-in (r 5.5 cm disc round the flange); v2 uses the link7 mesh
    pitches_interior: Tuple[float, ...] = (0.0,)   # approach pitch (deg, approach tilted downward) tried in order
    pitches_deck: Tuple[float, ...] = (0.0,)       # 90 = top-down grasp
    grasp_depths: Tuple[float, ...] = (0.010,)    # grasp point below the cube top (grid of grasp heights)
    deck_travel_z: float = 0.10                    # hand-site height above the deck surface for travel/descent start
    ik_seeds: int = 0                   # extra random IK seeds per cell (collision-unaware IK => multistart)

    def as_dict(self):
        return asdict(self)


V2_ROBOT_XML = (Path(__file__).resolve().parents[3] / "environments" / "assets" / "franka_emika_panda"
                / "base_panda_v2.xml")


def build(p: V2Params, scratch: Path | None = None):
    """Build env + workspace with the v2 collision model. Returns (env, ws, cfg).

    The robot is compiled from base_panda_v2.xml, whose collision class has conaffinity=1: MuJoCo only
    builds convex hulls for meshes some geom can collide with, so flipping contype/conaffinity AFTER
    compiling base_panda.xml (as v1 does for link5-7) leaves mesh_graphadr = -1 and produces no contacts.
    """
    import functools
    from tampanda.scenes import ArmSceneBuilder
    from tampanda.symbolic.domains.access19 import env_builder as EB
    scratch = Path(scratch or tempfile.mkdtemp(prefix="a19v2_"))
    _orig = EB.ArmSceneBuilder
    EB.ArmSceneBuilder = functools.partial(ArmSceneBuilder, base=V2_ROBOT_XML)
    try:
        builder, ws, cfg = EB.make_access19_builder(
        scratch_dir=scratch, cube_half_extents=tuple(p.cube_half), inter_row_gap=p.inter_row_gap,
        interior_height_z=p.interior_height, wall_thickness=p.wall, pedestal_height=p.pedestal,
        table_pos=(p.shelf_xy[0], p.shelf_xy[1], 0.0), side_margin=p.side_margin, top_grid_cells=p.top_cells)
        env = builder.build_env(rate=200.0)
    finally:
        EB.ArmSceneBuilder = _orig
    patch_collisions(env.model, p)
    return env, ws, cfg


def patch_collisions(m, p: V2Params):
    """Every arm collision geom (group 3, or any primitive pad) collides with the scene, never with the arm."""
    # link0 is bolted to the floor and never moves: leaving it contact-free keeps the base-vs-floor contact
    # out of every collision query. Its clearance from the shelf is checked once, statically (verify.py).
    arm_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_BODIES if b != "link0"}
    link0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == link0:
            m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
    n_on = 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in arm_ids:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if (name == "hand_capsule" and not p.keep_hand_capsule) or (name == "link7_capsule" and not p.keep_link7_capsule):
            m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
            continue
        if m.geom_group[g] == 3 or m.geom_conaffinity[g] or m.geom_contype[g]:
            m.geom_contype[g] = 0; m.geom_conaffinity[g] = 1; m.geom_margin[g] = p.arm_margin
            n_on += 1
    missing = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) for g in range(m.ngeom)
               if m.geom_bodyid[g] in arm_ids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
               and m.geom_conaffinity[g] and m.mesh_graphadr[m.geom_dataid[g]] < 0]
    if missing:
        raise RuntimeError("collision meshes without compiled hulls (contacts would be silently lost): %s" % missing)
    return n_on
