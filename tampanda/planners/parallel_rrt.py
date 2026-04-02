"""Parallel RRT planners.

Two drop-in replacements for the standard planners that offload collision
checking to a ``CollisionWorkerPool``:

``ParallelEdgeRRTStar``
    RRTStar where every ``is_path_collision_free`` call (plan, choose_parent,
    rewire, smooth_path) dispatches its intermediate configs to the worker
    pool simultaneously.  Benchmarked at **~3.25x** over sequential RRTStar.

``SpeculativeFeasibilityRRT``
    FeasibilityRRT that draws ``batch_size`` random extensions per round and
    checks all of them in parallel (one full edge per worker), while greedy
    connect steps also use parallel intermediate-config checking.
    Benchmarked at **~2x** over sequential FeasibilityRRT.

Typical usage::

    from tampanda.planners import (
        ParallelEdgeRRTStar,
        SpeculativeFeasibilityRRT,
        CollisionWorkerPool,
    )

    pool = CollisionWorkerPool(xml_path, n_workers=4)

    # drop-in for RRTStar
    planner = ParallelEdgeRRTStar(env, pool, max_iterations=5000)
    path = planner.plan(start_q, goal_q)

    # drop-in for FeasibilityRRT
    feas = SpeculativeFeasibilityRRT(env, pool, batch_size=4)
    path = feas.plan(start_q, goal_q)

    pool.shutdown()
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import mujoco

from tampanda.planners.feasibility_rrt import FeasibilityRRT
from tampanda.planners.rrt_star import RRTStar
from tampanda.planners.parallel_collision import CollisionWorkerPool


# ---------------------------------------------------------------------------
# ParallelEdgeRRTStar
# ---------------------------------------------------------------------------

class ParallelEdgeRRTStar(RRTStar):
    """RRTStar with parallel intermediate-config collision checking.

    Every call to ``is_path_collision_free`` — including those inside
    ``choose_parent``, ``rewire``, and ``smooth_path`` — dispatches all
    intermediate configs to the worker pool simultaneously instead of
    checking them sequentially.

    This is a drop-in replacement for ``RRTStar``; all constructor
    parameters are identical, with ``pool`` added as the second argument.

    Args:
        environment: Robot environment (``FrankaEnvironment``).
        pool: Pre-warmed ``CollisionWorkerPool``.
        **kwargs: Forwarded verbatim to ``RRTStar``.
    """

    def __init__(self, environment, pool: CollisionWorkerPool, **kwargs) -> None:
        super().__init__(environment, **kwargs)
        self.pool = pool

    def plan(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        max_iterations: Optional[int] = None,
    ):
        self.pool.set_scene(self.env)
        return super().plan(start_config, goal_config, max_iterations)

    def is_path_collision_free(
        self,
        config1: np.ndarray,
        config2: np.ndarray,
        steps: int = None,
    ) -> bool:
        if steps is None:
            dist = float(np.linalg.norm(config2 - config1))
            steps = max(
                self.collision_check_steps,
                int(np.ceil(dist / self.step_size)) * self.collision_check_steps,
            )
        configs = [
            (1 - i / steps) * config1 + (i / steps) * config2
            for i in range(steps + 1)
        ]
        return all(self.pool.check_configs_parallel(configs))


# ---------------------------------------------------------------------------
# SpeculativeFeasibilityRRT
# ---------------------------------------------------------------------------

class SpeculativeFeasibilityRRT(FeasibilityRRT):
    """FeasibilityRRT with speculative batch sampling and parallel edge checking.

    Per round:

    1. Draw ``batch_size`` random samples.
    2. Find the nearest node in tree_a for each (vectorised).
    3. Steer → ``batch_size`` candidate new configs.
    4. Check all edges in parallel — one full edge per worker.
    5. Add valid extensions to tree_a.
    6. For each valid extension run the greedy connect toward tree_b;
       each connect-step edge check also uses the pool.
    7. Swap trees.

    This is a drop-in replacement for ``FeasibilityRRT``; all constructor
    parameters are identical, with ``pool`` and ``batch_size`` added.

    Args:
        environment: Robot environment (``FrankaEnvironment``).
        pool: Pre-warmed ``CollisionWorkerPool``.
        batch_size: Number of speculative extensions per round.
        **kwargs: Forwarded verbatim to ``FeasibilityRRT``.
    """

    def __init__(
        self,
        environment,
        pool: CollisionWorkerPool,
        batch_size: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(environment, **kwargs)
        self.pool = pool
        self.batch_size = batch_size

    def plan(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        max_iterations: Optional[int] = None,
    ) -> Optional[list[np.ndarray]]:
        if max_iterations is None:
            max_iterations = self.max_iterations

        if not self.env.is_collision_free(start_config):
            return None
        if not self.env.is_collision_free(goal_config):
            return None

        self.pool.set_scene(self.env)

        data = self.env.data
        model = self.env.model
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()

        K      = self.batch_size
        step   = self.step_size
        thr    = self.goal_threshold
        lo_lim = self.joint_limits_low
        hi_lim = self.joint_limits_high

        capacity = max_iterations * 3 + 10
        tree_s = self._Tree(start_config, capacity)
        tree_g = self._Tree(goal_config,  capacity)

        tree_a, tree_b = tree_s, tree_g
        a_is_start     = True
        path: Optional[list[np.ndarray]] = None

        n_rounds = (max_iterations + K - 1) // K

        for _ in range(n_rounds):
            # ── Sample K configs ────────────────────────────────────────
            q_rands = np.random.uniform(lo_lim, hi_lim, (K, len(lo_lim)))

            # ── Nearest nodes in tree_a (vectorised) ────────────────────
            configs_a = tree_a.configs[: tree_a.n]
            diff      = configs_a[None, :, :] - q_rands[:, None, :]
            sq_dists  = np.einsum("kni,kni->kn", diff, diff)
            near_idxs = np.argmin(sq_dists, axis=1)

            # ── Steer ───────────────────────────────────────────────────
            near_cfgs = configs_a[near_idxs]
            deltas    = q_rands - near_cfgs
            dists     = np.linalg.norm(deltas, axis=1, keepdims=True)
            scales    = np.where(dists > step, step / np.maximum(dists, 1e-12), 1.0)
            q_news    = near_cfgs + deltas * scales

            # ── Parallel edge checks for all K extensions ────────────────
            edges   = [(near_cfgs[i], q_news[i]) for i in range(K)]
            edge_ok = self.pool.check_edges_parallel(edges)

            # ── Add valid extensions; greedy connect for each ────────────
            for i in range(K):
                if not edge_ok[i]:
                    continue

                new_a_idx = tree_a.add(q_news[i], int(near_idxs[i]))
                q_new     = q_news[i]

                near_b_idx = tree_b.nearest_idx(q_new)
                while True:
                    near_b = tree_b.configs[near_b_idx]
                    diff_b = q_new - near_b
                    dist_b = np.linalg.norm(diff_b)

                    if dist_b < thr:
                        path = self._build_path(
                            tree_a, new_a_idx,
                            tree_b, near_b_idx,
                            a_is_start,
                        )
                        break

                    alpha = min(step / dist_b, 1.0)
                    q_ext = near_b + diff_b * alpha
                    if not self._is_edge_free(near_b, q_ext):
                        break
                    near_b_idx = tree_b.add(q_ext, near_b_idx)

                if path is not None:
                    break

            if path is not None:
                break

            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start

        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_forward(model, data)

        return path

    def _is_edge_free(self, c1: np.ndarray, c2: np.ndarray) -> bool:
        """Greedy-connect edge check using the worker pool."""
        steps = self.collision_check_steps
        order = self._get_check_order(steps)
        inv   = 1.0 / steps
        delta = c2 - c1
        configs = [c1 + (idx * inv) * delta for idx in order]
        return all(self.pool.check_configs_parallel(configs))
