"""Worker pool for parallel MuJoCo collision checking.

Each worker owns a private FrankaEnvironment instance so MuJoCo state is
never shared between processes.

Scene state (object positions, held body, collision exceptions) is sent once
per planning query via ``set_scene()``, which caches it in each worker's
process-global state.  Per-check payloads then only contain the small arm
configs or edge endpoints — no repeated snapshot serialisation.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Module-level worker state (one per worker process)
# ---------------------------------------------------------------------------

_env            = None
_steps: int     = 5
_order: np.ndarray = None
_scene_qpos: np.ndarray = None   # cached object positions (qpos[7:])
_scene_held     = None            # cached _collision_held_body dict
_scene_exc      = None            # cached _collision_exception_ids set


def _make_binary_order(steps: int) -> np.ndarray:
    """Indices 1..steps in binary-subdivision order (index 0 excluded)."""
    order: list[int] = []
    queue: list[tuple[int, int]] = [(1, steps)]
    while queue:
        lo, hi = queue.pop(0)
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        order.append(mid)
        queue.append((lo, mid - 1))
        queue.append((mid + 1, hi))
    return np.array(order, dtype=np.int32)


def _worker_init(xml_path: str, collision_check_steps: int) -> None:
    global _env, _steps, _order
    from tampanda.environments.franka_env import FrankaEnvironment
    _env = FrankaEnvironment(xml_path)
    _steps = collision_check_steps
    _order = _make_binary_order(collision_check_steps)


# ---------------------------------------------------------------------------
# Scene-sync task (sent once per planning query, not per edge)
# ---------------------------------------------------------------------------

def _set_scene_task(snapshot: dict) -> None:
    """Cache snapshot in this worker process and restore MuJoCo state."""
    global _scene_qpos, _scene_held, _scene_exc
    _scene_qpos = snapshot["qpos"][7:].copy()   # object positions only
    _scene_held = snapshot["held_body"]
    _scene_exc  = snapshot["exception_ids"]
    _env.data.qpos[:] = snapshot["qpos"]
    _env.data.qvel[:] = 0.0
    _env._collision_held_body    = _scene_held
    _env._collision_exception_ids = _scene_exc
    mujoco.mj_forward(_env.model, _env.data)


# ---------------------------------------------------------------------------
# Per-check tasks — no snapshot in args
# ---------------------------------------------------------------------------

def _check_single_config_task(config: np.ndarray) -> bool:
    """Check one arm configuration using cached scene state."""
    _env.data.qpos[7:] = _scene_qpos   # reset objects (held body may drift)
    _env._collision_held_body    = _scene_held
    _env._collision_exception_ids = _scene_exc
    return _env.is_collision_free_no_restore(config)


def _check_edge_task(args: tuple) -> bool:
    """Check edge c1→c2 with binary-subdivision using cached scene state."""
    c1, c2 = args
    env = _env
    inv   = 1.0 / _steps
    delta = c2 - c1
    env._collision_held_body    = _scene_held
    env._collision_exception_ids = _scene_exc
    for idx in _order:
        env.data.qpos[7:] = _scene_qpos   # reset objects before each check
        config = c1 + (idx * inv) * delta
        if not env.is_collision_free_no_restore(config):
            return False
    return True


# ---------------------------------------------------------------------------
# Public pool class
# ---------------------------------------------------------------------------

class CollisionWorkerPool:
    """Persistent pool of MuJoCo worker processes for parallel collision checks.

    Typical usage per planning query::

        pool.set_scene(env)               # sync scene state once
        pool.check_edges_parallel(edges)  # lightweight per-call payloads

    Args:
        xml_path: Path to the scene XML (must remain on disk while pool is alive).
        n_workers: Number of worker processes to spawn.
        collision_check_steps: Intermediate configs per edge (binary subdivision).
    """

    def __init__(
        self,
        xml_path: str,
        n_workers: int = 4,
        collision_check_steps: int = 5,
    ) -> None:
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            n_workers,
            initializer=_worker_init,
            initargs=(xml_path, collision_check_steps),
        )
        self.n_workers = n_workers
        self.collision_check_steps = collision_check_steps

    # ------------------------------------------------------------------
    # Scene sync — call once per planning query
    # ------------------------------------------------------------------

    def set_scene(self, env) -> None:
        """Broadcast current scene state to all workers (once per planning query).

        Workers cache the snapshot locally; subsequent check calls carry only
        the small arm configs or edge endpoints.
        """
        held = env._collision_held_body
        snapshot = {
            "qpos": env.data.qpos.copy(),
            "held_body": {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in held.items()
            } if held else None,
            "exception_ids": set(env._collision_exception_ids),
        }
        # One task per worker so every process caches the snapshot
        self._pool.map(_set_scene_task, [snapshot] * self.n_workers)

    # ------------------------------------------------------------------
    # Parallel checks — lightweight payloads after set_scene()
    # ------------------------------------------------------------------

    def check_configs_parallel(self, configs: list[np.ndarray]) -> list[bool]:
        """Check N arm configurations in parallel; returns N booleans."""
        if not configs:
            return []
        return self._pool.map(_check_single_config_task, configs)

    def check_edges_parallel(
        self, edges: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[bool]:
        """Check N edges in parallel (full binary-subdivision per edge); returns N booleans."""
        if not edges:
            return []
        return self._pool.map(_check_edge_task, edges)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
