"""
Blocks-world domain wired up via DomainBridge.

This module is a reimplementation of BlocksStateManager using the generic
DomainBridge framework, demonstrating how an existing domain maps onto it.
Predicate evaluators replicate the geometry checks in BlocksStateManager;
action executors call PickPlaceExecutor for real robot motion.

Typical usage::

    from tampanda.symbolic.domains.blocks.env_builder import make_blocks_builder
    from tampanda.symbolic.domains.blocks.blocks_bridge import make_blocks_bridge
    from tampanda.planners.rrt_star import RRTStar

    env    = make_blocks_builder().build_env(rate=200.0)
    bridge, objects = make_blocks_bridge(env, block_indices=list(range(4)) + [12])

    state = bridge.ground_state(objects)
    plan  = bridge.plan(objects, goals=[("on", "block_0", "block_12")])

    for action, params in plan:
        bridge.execute_action(action, *params)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tampanda.tamp import DomainBridge
from tampanda.planners.grasp_planner import GraspPlanner, GRASP_CONTACT_OFFSET
from tampanda.planners.pick_place import PickPlaceExecutor
from tampanda.planners.rrt_star import RRTStar

# ── Block geometry (mirrors BlocksStateManager.BLOCK_SPECS) ───────────────
BLOCK_SPECS: Dict[int, Tuple[float, float, float]] = {
    **{i: (0.04, 0.04, 0.04) for i in range(6)},      # 0-5:  small cubes
    **{i: (0.06, 0.06, 0.06) for i in range(6, 12)},  # 6-11: medium cubes
    12: (0.10, 0.10, 0.05),   # platform
    13: (0.10, 0.10, 0.05),   # platform
    14: (0.15, 0.10, 0.05),   # large platform
    15: (0.15, 0.10, 0.05),   # large platform
}

GRASPABLE_BLOCKS = list(range(12))
PLATFORM_BLOCKS  = [12, 13, 14, 15]
GRIPPER_NAME     = "gripper1"

_PDDL_PATH = Path(__file__).parent / "pddl" / "blocks_domain.pddl"

# Z-threshold for hidden blocks (same sentinel as BlocksStateManager)
_HIDDEN_X = 50.0


# ── Geometry helpers ──────────────────────────────────────────────────────

def _idx(block_name: str) -> int:
    return int(block_name.split("_")[1])


def _half_size(block_name: str) -> np.ndarray:
    w, d, h = BLOCK_SPECS[_idx(block_name)]
    return np.array([w / 2, d / 2, h / 2])


def _xy_overlap(pos1, pos2, spec1, spec2, threshold: float = 0.8) -> bool:
    """True when the two blocks overlap enough in XY to count as stacked."""
    w1, d1, _ = spec1
    w2, d2, _ = spec2
    xi = max(0.0, min(pos1[0] + w1/2, pos2[0] + w2/2) - max(pos1[0] - w1/2, pos2[0] - w2/2))
    yi = max(0.0, min(pos1[1] + d1/2, pos2[1] + d2/2) - max(pos1[1] - d1/2, pos2[1] - d2/2))
    if xi == 0.0 or yi == 0.0:
        return False
    return (xi * yi) / min(w1 * d1, w2 * d2) >= threshold


# ── Factory ───────────────────────────────────────────────────────────────

def make_blocks_bridge(
    env,
    block_indices: Optional[List[int]] = None,
    table_height: Optional[float] = None,
    executor: Optional[PickPlaceExecutor] = None,
    planner: Optional[RRTStar] = None,
    grasp_planner: Optional[GraspPlanner] = None,
) -> Tuple[DomainBridge, Dict[str, List[str]]]:
    """Wire up a DomainBridge for the blocks world domain.

    This replicates BlocksStateManager's predicate grounding, state sampling,
    and action execution using the generic DomainBridge framework.

    Args:
        env: FrankaEnvironment built with ``make_blocks_builder()``.
        block_indices: Which block indices to make active.  Defaults to
            blocks 0–3 (small cubes) plus platforms 12–13.
        table_height: Table surface Z in metres.  Inferred from the environment
            if not provided.
        executor: Pre-built ``PickPlaceExecutor``.  Created from *planner* and
            *grasp_planner* if omitted.  Pass ``None`` and leave *planner* /
            *grasp_planner* as ``None`` to skip action-executor registration
            (useful for grounding/planning without robot motion).
        planner: Pre-built ``RRTStar`` instance.  Used only when *executor* is
            not provided.
        grasp_planner: Pre-built ``GraspPlanner``.  Used only when *executor*
            is not provided.

    Returns:
        ``(bridge, objects)`` where *objects* is the ``{type: [names]}`` dict
        ready to pass to ``ground_state``, ``plan``, and ``execute_action``.
    """
    if block_indices is None:
        block_indices = list(range(4)) + [12, 13]

    # Infer table height from the environment if not supplied
    if table_height is None:
        import mujoco
        try:
            geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "simple_table_surface")
        except TypeError:
            geom_id = -1
        if geom_id >= 0:
            pos = env.model.geom_pos[geom_id]
            size = env.model.geom_size[geom_id]
            table_height = float(pos[2] + size[2])
        else:
            table_height = 0.27  # fallback

    # Build executor only when robot motion is needed
    if executor is None and (planner is not None or grasp_planner is not None):
        if planner is None:
            planner = RRTStar(env)
        if grasp_planner is None:
            grasp_planner = GraspPlanner(table_z=table_height)
        executor = PickPlaceExecutor(env, planner, grasp_planner, use_attachment=True)

    block_names = [f"block_{i}" for i in block_indices]
    objects: Dict[str, List[str]] = {
        "block":   block_names,
        "gripper": [GRIPPER_NAME],
    }

    bridge = DomainBridge(_PDDL_PATH, env)

    # ── Code-evaluated predicates ─────────────────────────────────────────

    @bridge.predicate("on")
    def eval_on(env, fluents, block_top, block_bot):
        if block_top == block_bot:
            return False
        pos_top, _ = env.get_object_pose(block_top)
        pos_bot, _ = env.get_object_pose(block_bot)
        if pos_top[0] > _HIDDEN_X or pos_bot[0] > _HIDDEN_X:
            return False
        _, _, h_top = BLOCK_SPECS[_idx(block_top)]
        _, _, h_bot = BLOCK_SPECS[_idx(block_bot)]
        expected_z = pos_bot[2] + h_bot / 2 + h_top / 2
        if abs(pos_top[2] - expected_z) > 0.01:
            return False
        return _xy_overlap(pos_top, pos_bot, BLOCK_SPECS[_idx(block_top)], BLOCK_SPECS[_idx(block_bot)])

    @bridge.predicate("on-table")
    def eval_on_table(env, fluents, block):
        pos, _ = env.get_object_pose(block)
        if pos[0] > _HIDDEN_X:
            return False
        _, _, h = BLOCK_SPECS[_idx(block)]
        return abs(pos[2] - (table_height + h / 2)) < 0.015

    @bridge.predicate("clear")
    def eval_clear(env, fluents, block):
        pos_b, _ = env.get_object_pose(block)
        if pos_b[0] > _HIDDEN_X:
            return False
        for other in block_names:
            if other == block:
                continue
            pos_o, _ = env.get_object_pose(other)
            if pos_o[0] > _HIDDEN_X:
                continue
            _, _, h_b = BLOCK_SPECS[_idx(block)]
            _, _, h_o = BLOCK_SPECS[_idx(other)]
            expected_z = pos_b[2] + h_b / 2 + h_o / 2
            if abs(pos_o[2] - expected_z) < 0.01 and _xy_overlap(
                pos_o, pos_b, BLOCK_SPECS[_idx(other)], BLOCK_SPECS[_idx(block)]
            ):
                return False
        return True

    # ── Fluent predicates ─────────────────────────────────────────────────

    bridge.fluent("holding",       initial=None)
    bridge.fluent("gripper-empty", initial=[(GRIPPER_NAME,)])

    # ── Action executors ──────────────────────────────────────────────────
    # Only registered when an executor is available; omit for grounding-only use.

    if executor is not None:

        @bridge.action("pick-from-table")
        def exec_pick(env, fluents, gripper, block):
            pos  = env.get_object_position(block)
            half = _half_size(block)
            quat = env.get_object_orientation(block)
            ok = executor.pick(block, pos, half, quat)
            if not ok:
                return False, {}
            return True, {
                ("holding",      gripper, block): True,
                ("gripper-empty", gripper):       False,
            }

        @bridge.action("place-on-table")
        def exec_place_table(env, fluents, gripper, block):
            pos  = env.get_object_position(block)
            _, _, h = BLOCK_SPECS[_idx(block)]
            target_z = table_height + h / 2
            place_pos = np.array([pos[0], pos[1], target_z + GRASP_CONTACT_OFFSET])
            place_quat = np.array([0.0, 1.0, 0.0, 0.0])
            ok = executor.place(block, place_pos, place_quat)
            if not ok:
                return False, {}
            return True, {
                ("holding",      gripper, block): False,
                ("gripper-empty", gripper):       True,
            }

        @bridge.action("stack")
        def exec_stack(env, fluents, gripper, block_top, block_bot):
            pos_bot, _ = env.get_object_pose(block_bot)
            _, _, h_bot = BLOCK_SPECS[_idx(block_bot)]
            _, _, h_top = BLOCK_SPECS[_idx(block_top)]
            stack_z = pos_bot[2] + h_bot / 2 + h_top / 2
            place_pos  = np.array([pos_bot[0], pos_bot[1], stack_z + GRASP_CONTACT_OFFSET])
            place_quat = np.array([0.0, 1.0, 0.0, 0.0])
            ok = executor.place(block_top, place_pos, place_quat)
            if not ok:
                return False, {}
            return True, {
                ("holding",      gripper, block_top): False,
                ("gripper-empty", gripper):           True,
            }

        @bridge.action("unstack")
        def exec_unstack(env, fluents, gripper, block_top, block_bot):
            pos  = env.get_object_position(block_top)
            half = _half_size(block_top)
            quat = env.get_object_orientation(block_top)
            ok = executor.pick(block_top, pos, half, quat)
            if not ok:
                return False, {}
            return True, {
                ("holding",      gripper, block_top): True,
                ("gripper-empty", gripper):           False,
            }

    # ── State sampler ─────────────────────────────────────────────────────

    bounds = _get_working_bounds(env, table_height)

    @bridge.sampler("block")
    def sample_block(env, placed_so_far, rng):
        # placed_so_far entries are (x, y, w, d) footprints
        # We don't know which block we're placing, so use the max footprint
        # The caller should use sample_random_state in order of block_indices
        return None  # placeholder; see sample_blocks() below

    return bridge, objects


def sample_blocks(
    bridge: DomainBridge,
    env,
    block_indices: List[int],
    table_height: float,
    seed: Optional[int] = None,
) -> None:
    """Randomly place blocks on the table and apply positions to the environment.

    Replicates ``BlocksStateManager.sample_random_state``.  Hides all blocks
    first, then places the selected ones at collision-free positions.

    Args:
        bridge: The DomainBridge returned by ``make_blocks_bridge``.
        env: FrankaEnvironment.
        block_indices: Block indices to place (others are hidden).
        table_height: Table surface Z in metres.
        seed: Optional RNG seed.
    """
    rng = np.random.default_rng(seed)

    # Hide all blocks
    for idx in BLOCK_SPECS:
        w, d, h = BLOCK_SPECS[idx]
        env.set_object_pose(f"block_{idx}", np.array([100.0, 0.0, h / 2]))

    bounds = _get_working_bounds(env, table_height)
    margin = 0.05
    x_min, x_max = bounds["min_x"] + margin, bounds["max_x"] - margin
    y_min, y_max = bounds["min_y"] + margin, bounds["max_y"] - margin

    placed: List[Tuple[float, float, float, float]] = []  # (x, y, w, d)

    for idx in block_indices:
        w, d, h = BLOCK_SPECS[idx]
        for _ in range(100):
            x = rng.uniform(x_min + w / 2, x_max - w / 2)
            y = rng.uniform(y_min + d / 2, y_max - d / 2)
            clearance = 0.01
            if all(
                abs(x - px) >= (w + pw) / 2 + clearance or
                abs(y - py) >= (d + pd) / 2 + clearance
                for px, py, pw, pd in placed
            ):
                z = table_height + h / 2 + 0.003
                env.set_object_pose(f"block_{idx}", np.array([x, y, z]))
                placed.append((x, y, w, d))
                break
        else:
            print(f"Warning: could not place block_{idx} without collision.")

    env.reset_velocities()
    env.forward()


def _get_working_bounds(env, table_height: float) -> Dict[str, float]:
    """Extract table working bounds from the environment."""
    import mujoco
    try:
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "simple_table_surface")
    except TypeError:
        geom_id = -1
    if geom_id >= 0:
        pos  = env.model.geom_pos[geom_id]
        size = env.model.geom_size[geom_id]
        # Transform to world frame using body xpos
        body_id = env.model.geom_bodyid[geom_id]
        body_pos = env.data.xpos[body_id]
        cx = body_pos[0] + pos[0]
        cy = body_pos[1] + pos[1]
        return {
            "min_x": cx - size[0], "max_x": cx + size[0],
            "min_y": cy - size[1], "max_y": cy + size[1],
        }
    # Fallback
    return {"min_x": 0.15, "max_x": 0.65, "min_y": 0.15, "max_y": 0.65}
