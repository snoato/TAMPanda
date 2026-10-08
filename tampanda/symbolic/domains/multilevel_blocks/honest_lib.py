"""Honest 3D-stacking check with a library-style motion search (2026-10-05, Daniel's option B).

Same scene and rule as ``honest.py`` (VERSION 2026-10-04b: base_panda_v2 real link meshes, 2 mm for every arm geom,
the carried block's 1 mm rule, dense checks at <= 5 mrad of every committed motion, release at the pre-grasp width,
start and end at the neutral home), which stays unchanged, as does ``honest_aik.py``. What is new is HOW a motion is
looked for, for the actions whose rejection the 2026-10-04 diagnostic traced to the executor's fixed motion chain
(mlb_l4_diag: a clean full motion exists for all 506 rejected upright actions):

  pick-upright, pick-long-upright, put-upright, put-long-upright, put-cube

For these, before falling back to the executor's own chain (the honest check, unchanged), the check searches like the
shelf libraries (access19_v2 / confined / M3 builders):
  * grasp set: the executor's grasps for the action (upright: the 8 face-centred front grasps, _upright_grasp_quats;
    cube: the 4 top-down yaws, _cube_grasp_quats); for a put the block keeps its in-hand pose (the restored attachment),
    so the block's world pose follows from the hand orientation;
  * approach line: the vertical EE line from the executor's staging altitude (upright: the traverse altitude, stack
    top + max stack height + 8 cm; cube: the high-above pose, cell + 15 cm) down to the release / grasp pose (upright put:
    EE 40 mm above the block centroid's cell anchor as the executor, i.e. the block 25 mm (oblong) / 10 mm (long) above;
    cube put: 2 mm; pick: the grasp anchor), IK layers every 1 cm with the libraries' IK (frantik, all branches, q7 on
    the 3 deg grid), branch choice over the WHOLE line: every collision-free bottom solution is chained upward through the
    nearest solution of each layer (<= 0.15 rad joint step, i.e. the same branch), and the chain is dense-checked;
  * planned transits: neutral home / the restored start -> the top of the line and back, a straight joint motion if it
    passes the dense check, else VAMP RRT-Connect (vamp-planner 0.6.4, the libraries' planner: panda sphere model vs
    the tables and placed blocks as cuboids, the held block as spheres; padding 1/2/3 cm, 3 seeds) whose path is then
    dense-checked against the exact meshes with the honest rule;
  * release (put): fingers open to the pre-grasp width at the release pose (the released block excepted for the
    gripper), then the empty gripper retreats up the same line with the block resting in place; grasp (pick): fingers
    open on the way down (the block to pick excepted for the gripper, as the honest pick), close, lift up the line
    carrying the block, transit home.
  A motion is accepted if any grasp / branch passes every stage; collision exceptions are exactly the honest
  executor's (pick: the picked block, whole action; put-cube: the placed block, whole action; put-upright: none).
Every accepted motion is recorded (HonestFastExecutor.record) for the dense audit. Selected with MLB_CHECKER=honest_lib;
logged as mlblocks-honest-lib-2026-10-05.
"""
from __future__ import annotations

import copy
import math
from typing import List, Optional

import mujoco
import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import _deps
from tampanda.symbolic.domains.access19_v2 import kinematics as K
from tampanda.symbolic.domains.multilevel_blocks import honest as H
from tampanda.symbolic.domains.multilevel_blocks.executor import (
    _HOME_NEUTRAL_Q, _PREGRASP_CTRL, _TABLE_TOP_LOCAL_Z, _EE_TO_BLOCK_CENTRE_Z, _PLACE_CLEARANCE, _HIGH_ABOVE_HEIGHT,
    _upright_grasp_quats, _cube_grasp_quats,
)

VERSION = "2026-10-05b"   # a: first version; b: upright grasp point = top cell for every hand orientation (as the
                          # executor and its 180 deg flip), release opening 5 / 4 / 3 mm per finger (motion choice)
Q7_STEP_DEG = 3.0
LINE_STEP = 0.01           # IK layers every 1 cm along the approach line
MAX_STEP = 0.15            # rad, max joint change between neighbouring layers (same branch)
MAX_CHAINS = 16            # chains dense-checked per grasp
PADS = (0.01, 0.02, 0.03)
# Release: the honest executor opens to the pre-grasp width (_PREGRASP_CTRL = 20 mm per finger = 5 mm beyond a 30 mm
# block's faces). Next to a block one cell (30 mm) away the finger's outer face reaches it at >= 4 mm opening (measured:
# 0.61 mm at 4 mm, -1.0 mm at 5 mm); 3 mm per side clears it by >= 3 mm. How far to open is a motion choice, so the
# search tries 5, 4, 3 mm (every pose still checked with the unchanged honest rule); the retreat keeps that opening.
RELEASE_WIDTHS = (_PREGRASP_CTRL, 0.019, 0.018)
VAMP_EE_FROM_SITE = 0.105 - 0.09   # panda_grasptarget = hand + 0.105; attachment_site = hand + 0.09


def _site_to_tcp(env):
    m = env.model
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return P.TCP_OFFSET - float(m.site_pos[sid][2])


def ik_all(env, pos, R, q7s):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(pos, float) + env._lib_site_to_tcp * R[:, 2]
    frantik = _deps.load("frantik")
    out = []
    for q7 in q7s:
        for s in frantik.ik(T, float(q7), P.Q_HOME):          # robot base = world origin (checked at build)
            if np.isfinite(s).all() and (s >= P.Q_MIN).all() and (s <= P.Q_MAX).all():
                out.append(s)
    return np.array(out) if out else np.zeros((0, 7))


class HonestLibExecutor(H.HonestFastExecutor):
    lib_stats = None
    lib_debug = None      # optional list: (grasp index, chains, stage reached) per tried grasp (diagnostics only)

    def _dbg(self, *x):
        if self.lib_debug is not None:
            self.lib_debug.append(x)

    # ------------------------------------------------------------------ state snapshot (fallback to the honest chain)
    def _snap(self):
        e = self.env
        return (e.data.qpos.copy(), copy.deepcopy(e._attached), copy.deepcopy(e._collision_held_body),
                set(e._collision_body_ids), list(e.collision_exceptions), self._held_block, self._held_offset.copy())

    def _restore(self, s):
        e = self.env
        e.data.qpos[:] = s[0]; e._attached = copy.deepcopy(s[1]); e._collision_held_body = copy.deepcopy(s[2])
        e._collision_body_ids = set(s[3]); e.set_collision_exceptions(s[4])
        self._held_block, self._held_offset = s[5], s[6].copy()
        e.data.qvel[:] = 0.0; mujoco.mj_forward(e.model, e.data)

    def _with_fallback(self, lib_fn, chain_fn, *args):
        s = self._snap()
        rec0 = len(self.record) if self.record is not None else 0
        try:
            ok = lib_fn(*args)
        except H.HonestViolation:
            ok = False
        if ok:
            self._count("lib")
            return True
        if self.record is not None:
            del self.record[rec0:]
        self._restore(s)
        ok = chain_fn(*args)
        self._count("chain" if ok else "none")
        return ok

    def _count(self, k):
        if self.lib_stats is not None:
            self.lib_stats[k] = self.lib_stats.get(k, 0) + 1

    # ------------------------------------------------------------------ helpers
    def _q7s(self):
        if not hasattr(self, "_lib_q7s"):
            self._lib_q7s = K.q7_grid(Q7_STEP_DEG)
            self.env._lib_site_to_tcp = _site_to_tcp(self.env)
        return self._lib_q7s

    def _dense(self, Q):
        """Dense honest check of the joint path Q (list of 7-vectors) starting at Q[0]; samples recorded."""
        e = self.env
        e.data.qpos[:7] = Q[0]; mujoco.mj_forward(e.model, e.data)
        if not e.is_collision_free_no_restore(np.asarray(Q[0])):
            return False
        e.data.qpos[:7] = Q[0]
        return self._dense_ok([np.asarray(q) for q in Q[1:]]) if len(Q) > 1 else True

    def _try(self, Q):
        """_dense with a private record buffer: (ok, samples). The executor's record only ever receives the samples of
        the ACCEPTED candidate, stage by stage in execution order (None marks a discontinuity, e.g. the release)."""
        keep = self.record; self.record = []
        try:
            ok = self._dense(Q)
            return ok, self.record
        finally:
            self.record = keep

    def _line_chains(self, bottom, top, R, finger):
        """Branch-consistent IK chains along the vertical EE line bottom -> top: list of (n_layers x 7) arrays,
        bottom configuration first, ordered by the top configuration's distance to the neutral home."""
        e = self.env
        q7s = self._q7s()
        n = max(1, int(math.ceil((top[2] - bottom[2]) / LINE_STEP)))
        pts = [bottom + (top - bottom) * (k / n) for k in range(n + 1)]
        layers = [ik_all(e, p, R, q7s) for p in pts]
        if any(len(L) == 0 for L in layers):
            return []
        e.data.qpos[7] = e.data.qpos[8] = finger
        free0 = [s for s in layers[0] if e.is_collision_free(s)]
        chains, seen = [], set()
        for s0 in free0:
            ch = [s0]
            for L in layers[1:]:
                d = np.abs(L - ch[-1]).max(1); i = int(np.argmin(d))
                if d[i] > MAX_STEP:
                    ch = None; break
                ch.append(L[i])
            if ch is None:
                continue
            key = tuple(np.round(ch[-1], 2))
            if key in seen:
                continue
            seen.add(key); chains.append(np.array(ch))
        chains.sort(key=lambda c: float(np.abs(c[-1] - _HOME_NEUTRAL_Q).max()))
        return chains[:MAX_CHAINS]

    def _vamp_env(self, pad, held):
        vamp = _deps.load("vamp")
        e = self.env; m, d = e.model, e.data
        venv = vamp.Environment()
        arm = set(e._collision_body_ids)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b in arm or not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            if d.geom_xpos[g][0] > 5 or abs(d.geom_xpos[g][1]) > 5:           # parked blocks
                continue
            he = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g] + pad
            venv.add_cuboid(vamp.Cuboid(list(map(float, d.geom_xpos[g])), [0.0, 0.0, 0.0], list(map(float, he))))
        if held is not None:
            rel_pos, rel_mat, size = held
            T = np.eye(4, dtype=np.float32); T[:3, 3] = np.asarray(rel_pos) - np.array([0, 0, VAMP_EE_FROM_SITE])
            a = vamp.Attachment(np.asfortranarray(T))
            ax = int(np.argmax(size)); half = np.sort(size)[:2]
            rad = float(np.linalg.norm(half)) + pad
            axis = np.asarray(rel_mat)[:, ax]
            L = float(size[ax])
            k = max(1, int(math.ceil(2 * L / rad)))
            a.add_spheres([vamp.Sphere(list(map(float, axis * s)), rad) for s in np.linspace(-L + min(L, rad) * 0.5, L - min(L, rad) * 0.5, k)])
            venv.attach(a)
        return venv

    def _transit(self, qa, qb, held):
        """Joint motion qa -> qb: straight if it passes the dense check, else VAMP RRT-Connect + dense check.
        Returns (ok, samples of the accepted motion)."""
        ok, smp = self._try([qa, qb])
        if ok:
            return True, smp
        vamp = _deps.load("vamp"); VP = vamp.panda
        for pad in PADS:
            venv = self._vamp_env(pad, held)
            if not (VP.validate(np.asarray(qa, np.float32), venv) and VP.validate(np.asarray(qb, np.float32), venv)):
                continue
            for seed in range(3):
                rng = VP.halton() if seed == 0 else VP.xorshift()
                for _ in range(seed):
                    if hasattr(rng, "next"):
                        rng.next()
                r = VP.rrtc(np.asarray(qa, np.float32), np.asarray(qb, np.float32), venv, vamp.RRTCSettings(), rng)
                if not r.solved:
                    continue
                p = VP.simplify(r.path, venv, vamp.SimplifySettings(), rng).path
                p.interpolate_to_resolution(VP.resolution())
                path = [np.asarray(x, float) for x in np.array(p.numpy(), float)]
                path[0], path[-1] = np.asarray(qa, float), np.asarray(qb, float)
                ok, smp = self._try(path)
                if ok:
                    return True, smp
        return False, []

    def _held_geom(self, block):
        e = self.env; m = e.model
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, block)
        g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b][0]
        return np.asarray(m.geom_size[g], float)

    def _attach_at(self, q, block):
        e = self.env
        e.data.qpos[:7] = q; mujoco.mj_forward(e.model, e.data)
        e.attach_object_to_ee(block)

    # ------------------------------------------------------------------ puts
    def _lib_put(self, block, goal_centroid, quats, line_top_z, ee_release_z=None, block_lift=0.0, except_block=False,
                 upright_top_grasp=None):
        e = self.env
        att = e._attached
        if att is None or att["body_name"] != block:
            return False
        rel_pos, rel_mat = np.asarray(att["rel_pos"], float), np.asarray(att["rel_mat"], float)
        rel_pos0, rel_mat0 = rel_pos.copy(), rel_mat.copy()
        size = self._held_geom(block)
        start = e.data.qpos[:7].copy(); base = e.data.qpos.copy()
        fing = float(e.data.qpos[7])
        for q4 in quats:
            R = K.quat_to_R(q4)
            if upright_top_grasp is not None:
                # upright: the grasp point stays the block's TOP cell for every hand orientation (the executor's front
                # grasp and its 180 deg-flipped twin): the block hangs below the EE, upright; set the in-hand pose so
                h_top, R_bw = upright_top_grasp
                rel_pos = R.T @ np.array([0.0, 0.0, -h_top]); rel_mat = R.T @ R_bw
                e._attached["rel_pos"], e._attached["rel_mat"] = rel_pos, rel_mat
                e.set_collision_held_body(block, rel_pos, rel_mat)
            off = R @ rel_pos                                  # block centre relative to the EE site, world frame
            bpos = goal_centroid + np.array([0, 0, block_lift])
            ee = bpos - off
            if ee_release_z is not None:
                ee = np.array([goal_centroid[0] - off[0], goal_centroid[1] - off[1], ee_release_z])
                bpos = ee + off
            top = np.array([ee[0], ee[1], max(line_top_z, ee[2] + 0.02)])
            e.data.qpos[:] = base
            if except_block:
                e.add_collision_exception(block)
            e.data.qpos[:] = base
            if upright_top_grasp is not None:
                e.data.qpos[:7] = start; mujoco.mj_forward(e.model, e.data); e._apply_attachment()
                base = e.data.qpos.copy()
            chains = self._line_chains(ee, top, R, fing)
            self._dbg(int(np.argmax([np.allclose(q4, x) for x in quats])), "chains", len(chains))
            for ch in chains:
                e.data.qpos[:] = base
                ok, s_in = self._transit(start, ch[-1], (rel_pos, rel_mat, size))   # carrying, start -> top of line
                if not ok:
                    self._dbg("transit-in fail"); continue
                e.data.qpos[:] = base
                ok, s_line = self._try(list(ch[::-1]))         # carrying, top -> release
                if not ok:
                    lc = e.last_collision or {}
                    self._dbg("line fail", lc.get("body1"), lc.get("body2")); continue
                # release: fingers to the pre-grasp width at the release pose, the released block excepted
                q_rel = ch[0]
                e.data.qpos[:7] = q_rel; mujoco.mj_forward(e.model, e.data); e._apply_attachment(); mujoco.mj_forward(e.model, e.data)
                placed_q = e.data.qpos.copy()
                e.detach_object()
                had = block in e.collision_exceptions
                e.add_collision_exception(block)
                ok = False
                for width in RELEASE_WIDTHS:
                    e.data.qpos[:7] = q_rel
                    e.data.qpos[7] = e.data.qpos[8] = width; mujoco.mj_forward(e.model, e.data)
                    ok = e.is_collision_free_no_restore(q_rel)
                    if ok:
                        break
                s_rel = [e.data.qpos.copy()]
                if not ok:
                    lc = e.last_collision or {}
                    self._dbg("release fail", lc.get("body1"), lc.get("body2"))
                if not had and not except_block:
                    e.remove_collision_exception(block)
                if ok:
                    # block drops into place; empty gripper retreats up the line, then home
                    qa = e.model.jnt_qposadr[mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_JOINT, block + "_freejoint")]
                    e.data.qpos[qa:qa + 3] = goal_centroid
                    e.data.qpos[7] = e.data.qpos[8] = width; mujoco.mj_forward(e.model, e.data)
                    ok1, s_up = self._try(list(ch))
                    ok2, s_out = self._transit(ch[-1], _HOME_NEUTRAL_Q, None) if ok1 else (False, [])
                    if not ok1:
                        lc = e.last_collision or {}
                        self._dbg("retreat fail", lc.get("body1"), lc.get("body2"))
                    elif not ok2:
                        self._dbg("transit-out fail")
                    if ok1 and ok2:
                        if self.record is not None:
                            self.record.extend(s_in + s_line + s_rel + [None] + s_up + s_out)
                        e.data.qpos[:7] = _HOME_NEUTRAL_Q; mujoco.mj_forward(e.model, e.data)
                        self._held_block = None; self._held_offset = np.zeros(3)
                        e.clear_collision_exceptions()
                        return True
                # undo the release for the next candidate
                e.data.qpos[:] = placed_q
                e.data.qpos[7] = e.data.qpos[8] = fing
                e.data.qpos[:7] = q_rel; mujoco.mj_forward(e.model, e.data)
                e.attach_object_to_ee(block)
                e.set_collision_exceptions([block] if except_block else [])
            e.data.qpos[:] = base
            if e._attached is None or e._attached["body_name"] != block:
                self._attach_at(start, block)
            e._attached["rel_pos"], e._attached["rel_mat"] = rel_pos0, rel_mat0
            e.set_collision_held_body(block, rel_pos0, rel_mat0)
        return False

    def _traverse_z(self):
        cfg = self.config
        return cfg.stack_table_pos[2] + _TABLE_TOP_LOCAL_Z + cfg.stack_grid_cells[2] * cfg.cube_size + 0.08

    def _lib_put_upright(self, block, c_low, c_high):
        anchor = (self._cell_pose(c_low) + self._cell_pose(c_high)) / 2
        from tampanda.symbolic.domains.multilevel_blocks.state import _QUAT_UPRIGHT
        n_cells = 3 if block.startswith("long") else 2
        h_top = (n_cells - 1) * self.config.cube_size / 2
        return self._lib_put(block, anchor, _upright_grasp_quats(), self._traverse_z(), ee_release_z=anchor[2] + 0.04,
                             upright_top_grasp=(h_top, K.quat_to_R(_QUAT_UPRIGHT)))

    def _lib_put_cube(self, block, cell):
        centre = self._cell_pose(cell)
        return self._lib_put(block, centre, _cube_grasp_quats(), centre[2] + _EE_TO_BLOCK_CENTRE_Z + _HIGH_ABOVE_HEIGHT,
                             block_lift=_PLACE_CLEARANCE, except_block=True)

    # ------------------------------------------------------------------ picks
    def _lib_pick_upright(self, block, c_low, c_high):
        e = self.env
        anchor = self._cell_pose(c_high)                   # EE at the top cell (the executor's front grasp)
        start = e.data.qpos[:7].copy(); base = e.data.qpos.copy()
        e.add_collision_exception(block)
        try:
            for q4 in _upright_grasp_quats():
                R = K.quat_to_R(q4)
                top = np.array([anchor[0], anchor[1], max(self._traverse_z(), anchor[2] + 0.02)])
                e.data.qpos[:] = base
                for ch in self._line_chains(anchor, top, R, _PREGRASP_CTRL):
                    e.data.qpos[:] = base
                    e.data.qpos[7] = e.data.qpos[8] = _PREGRASP_CTRL
                    ok, s_in = self._transit(start, ch[-1], None)
                    if not ok:
                        continue
                    e.data.qpos[7] = e.data.qpos[8] = _PREGRASP_CTRL
                    ok, s_down = self._try(list(ch[::-1]))     # empty (pre-grasp width) down to the grasp
                    if not ok:
                        continue
                    # close on the block and attach
                    e.data.qpos[:7] = ch[0]
                    e.data.qpos[7] = e.data.qpos[8] = self.config.cube_half_extent
                    mujoco.mj_forward(e.model, e.data)
                    if not e.is_collision_free_no_restore(ch[0]):
                        continue
                    s_close = [e.data.qpos.copy()]
                    e.attach_object_to_ee(block)
                    att = e._attached
                    held = (np.asarray(att["rel_pos"], float), np.asarray(att["rel_mat"], float), self._held_geom(block))
                    ok1, s_up = self._try(list(ch))
                    ok2, s_out = self._transit(ch[-1], _HOME_NEUTRAL_Q, held) if ok1 else (False, [])
                    if ok1 and ok2:
                        if self.record is not None:
                            self.record.extend(s_in + s_down + [None] + s_close + s_up + s_out)
                        e.data.qpos[:7] = _HOME_NEUTRAL_Q; mujoco.mj_forward(e.model, e.data); e._apply_attachment()
                        self._held_block = block
                        self._held_offset = self._block_pos(block) - self._ee_pos()
                        return True
                    e.detach_object(); e.data.qpos[:] = base; mujoco.mj_forward(e.model, e.data)
            return False
        finally:
            e.clear_collision_exceptions()

    # ------------------------------------------------------------------ dispatch overrides
    def put_upright(self, block_name, c_low_id, c_high_id):
        return self._with_fallback(self._lib_put_upright, super().put_upright, block_name, c_low_id, c_high_id)

    def put_cube(self, block_name, cell_id):
        return self._with_fallback(self._lib_put_cube, super().put_cube, block_name, cell_id)

    def pick_upright(self, block_name, c_low_id, c_high_id):
        return self._with_fallback(self._lib_pick_upright, super().pick_upright, block_name, c_low_id, c_high_id)

    # put_long_upright / pick_long_upright call self.put_upright / self.pick_upright (executor) -> covered


def make_honest_lib_executor(env, workspace, config):
    from tampanda.planners.rrt_star import RRTStar
    m = env.model
    l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
    mujoco.mj_kinematics(m, env.data)
    if np.abs(env.data.xpos[l0]).max() > 1e-9 or np.abs(env.data.xmat[l0].reshape(3, 3) - np.eye(3)).max() > 1e-9:
        raise RuntimeError("honest_lib assumes the robot base at the world origin (frantik / VAMP base frame)")
    ex = HonestLibExecutor(env, workspace, config, motion_planner=RRTStar(env, max_iterations=3000))
    env.ik.max_iters = 100
    ex._yaw_pool = None
    return ex


check_action_honest_lib = H.check_action_honest
