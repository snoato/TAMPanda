"""Honest feasibility check for multilevel-blocks / 3D stacking (2026-10-04).

Why: the original check (feasibility.check_action, fast=True) runs the executor's IK / lerp / RRT chain on
base_panda.xml, whose arm link collision meshes are compiled WITHOUT convex hulls, so link1..7 never generate contacts
(only the finger pads collide; the hand capsule is disabled by apply_runtime_tweaks). A dense replay of the paper
model's plans against the real link meshes found the forearm (link4/5) up to 83 mm inside the stack table during
put-upright (rgnet renders_v2 / slurm/eval/mlb_dense_audit.py). Fast mode also skips motions: the post-pick/put return
and put-upright's lift-back/retract are teleported, and the gripper opening at a release is never checked.

What changes (only when this module is used; feasibility.check_action and the executor are untouched):
1. Scene: the same builder with the robot compiled from base_panda_v2.xml (every link collision mesh with a hull) and
   access19_v2.shelf._enable_arm_collisions: every arm collision geom collides with the scene (never with the arm),
   3 mm contact window; the oversized hand / link7 capsule stand-ins are off. This is the model the dense audit and
   the website-clip clearance check measure on.
2. Margin: an arm geom (link1..7, hand, fingers) must keep >= MARGIN = 2 mm from every other collidable geom (the
   Access-19 v2 library's margin, access19_v2.grasp.MARGIN). The carried block keeps the original rule (contact
   distance < 1 mm is a collision), so placement semantics do not change. Gripper vs carried block is self-contact
   and the executor's collision exceptions (the block being picked/placed) apply, both as before.
3. Every motion is dense-checked at <= DQ rad joint steps (the carried block following the hand), already while
   the executor PLANS: a candidate path from _try_plan_to_pose / _seeded_lerp_check, or a hand-off / home lerp
   segment, that violates the margin counts as a planning failure, so the executor falls back exactly as it does for
   its own collision rejections (next yaw, plan_to_pose, the Cartesian fallback). _execute re-checks every committed
   motion as a final guard; a violation there makes the action infeasible.
4. No unchecked shortcuts: the post-pick/put return (hand-off, then neutral home) and put-upright's lift-back /
   retract / hand-off run as in the real executor and are dense-checked; every action must end with the arm back at
   the neutral home (hand-off -> home lerp or its fallback), else it is infeasible.
   Release: the gripper opens to the pre-grasp width (_PREGRASP_CTRL = 20 mm per finger, 5 mm per side around a 30 mm
   block -- the width the executor already uses to descend between neighbours at a pick) instead of fully (40 mm),
   and that pose is checked against the neighbours (the released block excepted). Fully opening next to an adjacent
   block is a collision; the narrower release is a physically valid motion that the executor did not try.
5. Every check starts from the neutral home (restore_state(home_qpos=_HOME_NEUTRAL_Q)), so a verdict does not depend
   on where the previous check left the arm (the old check carries the arm pose over between checks).

Not covered (same as the audit): arm self-collision; the carried block keeps the original 1 mm rule.
"""
from __future__ import annotations

import math
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from tampanda.symbolic.domains.access19_v2.shelf import ROBOT_XML as V2_ROBOT_XML, _enable_arm_collisions
from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig, make_multilevel_blocks_builder,
)
from tampanda.symbolic.domains.multilevel_blocks.executor import (
    MultilevelBlocksExecutor, _HOME_NEUTRAL_Q, _PREGRASP_CTRL,
)
from tampanda.symbolic.domains.multilevel_blocks.feasibility import FastFeasibilityExecutor, dispatch_action
from tampanda.symbolic.domains.multilevel_blocks.prefilter import INFEASIBLE, filter_action
from tampanda.symbolic.domains.multilevel_blocks.state import restore_state

VERSION = "2026-10-04b"   # a: first version; b: dense check inside planning (fallbacks), release at pre-grasp width
# (b also got a speed-up with identical verdicts: vectorised contact rule + kinematics-only collision queries,
#  build_honest_env(fast=True); checked by mlb_honest/tests/t_equiv_fast.py: 25 checks, 0 verdict / sample differences)
MARGIN = 0.002         # arm geoms: required clearance (m)
HELD_MARGIN = 0.001    # carried block: FrankaEnvironment.check_collisions' original rule
# carried block vs the arm's own links 1..7: MARGIN by default (as Access-19 v2: "shelf, self, objects");
# MLB_HONEST_HELD_LINK_MARGIN (metres) overrides it -- a lever study only, never set in a run
import os as _os
HELD_LINK_MARGIN = float(_os.environ.get("MLB_HONEST_HELD_LINK_MARGIN", MARGIN))
DQ = 0.005             # max joint step (rad) between dense samples of a committed motion
HOME_TOL = 1e-4


class HonestViolation(RuntimeError):
    """A committed motion violates the honest clearance (raised inside the executor; the action is infeasible)."""


ARM_LINKS = ("link1", "link2", "link3", "link4", "link5", "link6", "link7")


def _check_collisions_honest(env) -> bool:
    """FrankaEnvironment.check_collisions with the honest rules (bound to one env instance).

    As the original: robot (arm + registered carried block) vs non-robot contacts count, collision exceptions (the
    block being picked / placed) and the carried block keep their old 1 mm rule. Honest additions: (a) arm geoms need
    MARGIN; (b) a collision exception only excuses the GRIPPER (hand, fingers) -- arm links link1..7 must keep MARGIN
    from the excepted block too; (c) arm links vs the carried block are checked (gripper vs carried block stays
    self-contact). (b) and (c) make the check cover exactly the pairs the dense audit measures."""
    if env.collision_diagnostics:
        env.last_collision = None
    m, d = env.model, env.data
    links = getattr(env, "_honest_link_ids", None)
    if links is None:
        links = env._honest_link_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_LINKS}
    held = env._collision_held_body["body_id"] if env._collision_held_body is not None else -1
    exc = env._collision_exception_ids
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        r1 = b1 in env._collision_body_ids
        r2 = b2 in env._collision_body_ids
        if r1 == r2:
            # robot-robot: only an arm link vs the carried block counts
            if not (r1 and ((b1 == held and b2 in links) or (b2 == held and b1 in links))):
                continue
            thr = HELD_LINK_MARGIN
        elif b1 in exc or b2 in exc:
            # excepted block: excuses the gripper (and the carried block itself), not the arm links
            if not (b1 in links or b2 in links):
                continue
            thr = MARGIN
        else:
            thr = HELD_MARGIN if held in (b1, b2) else MARGIN
        if c.dist < thr:
            if env.collision_diagnostics:
                env.last_collision = env._contact_diagnostic(c, b1, b2)
            return False
    return True


def _check_collisions_honest_np(env) -> bool:
    """_check_collisions_honest, vectorised over the contact list (same rules, same verdict; ~5x faster)."""
    if env.collision_diagnostics:
        env.last_collision = None
    m, d = env.model, env.data
    n = d.ncon
    if n == 0:
        return True
    cache = getattr(env, "_honest_np", None)
    if cache is None:
        links = np.zeros(m.nbody, bool)
        for b in ARM_LINKS:
            links[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)] = True
        cache = env._honest_np = (np.asarray(m.geom_bodyid), links)
    gb, links = cache
    rob = np.zeros(m.nbody, bool); rob[list(env._collision_body_ids)] = True
    exc = np.zeros(m.nbody, bool)
    if env._collision_exception_ids:
        exc[list(env._collision_exception_ids)] = True
    held = env._collision_held_body["body_id"] if env._collision_held_body is not None else -1
    con = d.contact
    b1 = gb[np.asarray(con.geom1[:n])]; b2 = gb[np.asarray(con.geom2[:n])]; dist = np.asarray(con.dist[:n])
    r1, r2 = rob[b1], rob[b2]
    h1, h2 = b1 == held, b2 == held
    l1, l2 = links[b1], links[b2]
    mixed = r1 != r2
    excepted = exc[b1] | exc[b2]
    thr = np.full(n, -np.inf)
    thr = np.where(mixed & ~excepted, np.where(h1 | h2, HELD_MARGIN, MARGIN), thr)
    thr = np.where(mixed & excepted & (l1 | l2), MARGIN, thr)
    thr = np.where(r1 & r2 & ((h1 & l2) | (h2 & l1)), HELD_LINK_MARGIN, thr)
    bad = dist < thr
    if not bad.any():
        return True
    if env.collision_diagnostics:
        i = int(np.argmax(bad))
        env.last_collision = env._contact_diagnostic(d.contact[i], int(b1[i]), int(b2[i]))
    return False


def _icf_no_restore_fast(env, configuration) -> bool:
    """FrankaEnvironment.is_collision_free_no_restore with mj_kinematics + mj_collision instead of mj_forward: the
    contacts come from the collision pipeline either way (mj_forward's constraint solve does not change them)."""
    m, d = env.model, env.data
    d.qpos[:7] = configuration
    d.qvel[:7] = 0.0
    mujoco.mj_kinematics(m, d)
    if env._collision_held_body is not None:
        env._apply_collision_held_body()
        mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    return env.check_collisions()


def build_honest_env(scratch_dir: Path, config: Optional[MultilevelBlocksConfig] = None, rate: float = 10000.0,
                     fast: bool = True):
    """(env, workspace, config): the multilevel-blocks scene on base_panda_v2.xml with real arm collision meshes."""
    builder, ws, cfg = make_multilevel_blocks_builder(scratch_dir=Path(scratch_dir), config=config,
                                                      robot_xml=V2_ROBOT_XML)
    env = builder.build_env(rate=rate)
    _enable_arm_collisions(env.model)
    if fast:     # identical verdicts (tests/t_equiv_fast.py), ~2x faster checks
        env.check_collisions = types.MethodType(_check_collisions_honest_np, env)
        env.is_collision_free_no_restore = types.MethodType(_icf_no_restore_fast, env)
    else:
        env.check_collisions = types.MethodType(_check_collisions_honest, env)
    mujoco.mj_forward(env.model, env.data)
    return env, ws, cfg


class HonestFastExecutor(FastFeasibilityExecutor):
    """FastFeasibilityExecutor whose committed motions are dense-checked and whose shortcuts are checked motions."""

    record = None    # optional list: every accepted dense sample's full qpos is appended (validation / audit)

    def _dense_ok(self, path) -> bool:
        env = self.env
        save_q, save_v = env.data.qpos.copy(), env.data.qvel.copy()
        ok = True
        try:
            prev = save_q[:7].copy()
            for wp in path:
                wp = np.asarray(wp, dtype=float)[:7]
                n = max(1, int(math.ceil(float(np.max(np.abs(wp - prev))) / DQ)))
                for k in range(1, n + 1):
                    if not env.is_collision_free_no_restore(prev + (wp - prev) * (k / n)):
                        ok = False
                        break
                    if self.record is not None:
                        self.record.append(env.data.qpos.copy())
                if not ok:
                    break
                prev = wp
        finally:
            env.data.qpos[:] = save_q
            env.data.qvel[:] = save_v
            mujoco.mj_forward(env.model, env.data)
        return ok

    # -- dense check inside planning: a violating candidate is a planning failure (the executor falls back)
    def _try_plan_to_pose(self, target, quats, n_substeps=12):
        for q in quats:
            path = self.lik.plan_joint_lerp(target, q, n_substeps=n_substeps)
            if path is not None and self._dense_ok(path):
                return path, q
        for q in quats:
            path = self.lik.plan_to_pose(target, q, n_substeps=n_substeps, slerp_orientation=False)
            if path is not None and self._dense_ok(path):
                return path, q
        return None

    def _seeded_lerp_check(self, *args, **kwargs):
        res = super()._seeded_lerp_check(*args, **kwargs)
        if res is not None and not self._dense_ok(res[0]):
            return None
        return res

    def _segment_collision_free(self, q_a, q_b, n):
        n = max(int(n), int(math.ceil(float(np.max(np.abs(np.asarray(q_b, float)[:7] - np.asarray(q_a, float)[:7]))) / DQ)))
        return self.env.is_path_collision_free(q_a, q_b, steps=n)

    def _execute(self, path, step_size=None, precision=False):
        if not path:
            return
        if not self._dense_ok(path):
            raise HonestViolation("committed motion within %.0f mm of the scene" % (1000 * MARGIN))
        super()._execute(path, step_size, precision)

    # -- shortcuts of the fast executor replaced by the real (now dense-checked) motions
    def _return_after_pickput(self, region, used_quat) -> None:
        MultilevelBlocksExecutor._return_after_pickput(self, region, used_quat)

    def _validate_put_upright_return(self) -> bool:
        return True

    def _detach_open(self) -> None:
        released = self._held_block
        super()._detach_open()                       # kinematic detach (+ fingers to fully open) ...
        env = self.env
        env.data.qpos[7] = _PREGRASP_CTRL            # ... honest: open only to the pre-grasp width
        env.data.qpos[8] = _PREGRASP_CTRL
        mujoco.mj_forward(env.model, env.data)
        added = released is not None and released not in env.collision_exceptions
        if added:
            env.add_collision_exception(released)
        try:
            ok = env.is_collision_free_no_restore(env.data.qpos[:7].copy())
        finally:
            if added:
                env.remove_collision_exception(released)
            mujoco.mj_forward(env.model, env.data)
        if not ok:
            raise HonestViolation("opened gripper within %.0f mm of a neighbour" % (1000 * MARGIN))
        if self.record is not None:
            self.record.append(env.data.qpos.copy())

    def back_home(self) -> bool:
        """The arm must be able to get back to the neutral home (dense-checked) at the end of every action."""
        if float(np.max(np.abs(self.env.data.qpos[:7] - _HOME_NEUTRAL_Q))) < HOME_TOL:
            return True
        return bool(self._to_neutral_home())


def make_honest_executor(env, workspace, config, max_iters: int = 3000) -> HonestFastExecutor:
    """Same construction as feasibility._make_executor(fast=True), with the honest executor class."""
    from tampanda.planners.rrt_star import RRTStar
    ex = HonestFastExecutor(env, workspace, config, motion_planner=RRTStar(env, max_iterations=max_iters))
    env.ik.max_iters = 100
    ex._yaw_pool = None
    return ex


def check_action_honest(env, workspace, config, state: Dict, action, *, executor: HonestFastExecutor) -> Dict[str, Any]:
    """feasibility.check_action(fast=True) with the honest executor, a neutral-home start and a required return home."""
    t0 = time.perf_counter()
    decision, reason = filter_action(state, action, config)     # sound geometric pre-filter (rejects only)
    if decision == INFEASIBLE:
        return {"success": False, "elapsed_s": time.perf_counter() - t0, "error": "prefilter:%s" % reason,
                "fast": True, "honest": True}
    restore_state(env, workspace, config, state, home_qpos=_HOME_NEUTRAL_Q, on_held="attach", executor=executor)
    name, *args = action
    try:
        ok = bool(dispatch_action(executor, name, tuple(args)))
        err = None if ok else "executor"
        if ok and not executor.back_home():
            ok, err = False, "honest: no collision-free return to neutral home"
    except HonestViolation as exc:
        ok, err = False, "honest: %s" % exc
    except Exception as exc:
        ok, err = False, "%s: %s" % (type(exc).__name__, exc)
    return {"success": ok, "elapsed_s": time.perf_counter() - t0, "error": err, "fast": True, "honest": True}
