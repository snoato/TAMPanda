"""Grip-aware honest 3D-stacking check (2026-10-06): MLB_CHECKER=honest_grip, logged mlblocks-honest-grip-<VERSION>.

Fixes the four defects the continuous verifier found in honest_lib (renders_v2/stacking_l4/verify_continuous):
1. Placement: the symbolic state could not tell how the held block is gripped, and the put assumed the canonical grasp,
   so after pick-upright + make-flat-* the block was released 15 / 30 mm beside its cells. Now the PDDL domain records
   the grip (grip-centre / grip-end, see the grip-aware domain file), and the check restores the held block at the REAL
   in-hand pose of that grip (a re-orientation is a rigid wrist rotation: the block does not move in the hand, so the
   in-hand pose depends on the grip only) and computes every hand pose BLOCK-CENTRICALLY: for a target block pose T_t
   and the in-hand pose T_rel, the candidate hand poses are T_t * S * T_rel^-1 over the box symmetries S, restricted to
   the action's grasp family (top-down: palm axis pointing down; front: palm axis horizontal). The block lands exactly
   on its cells by construction.
2. Held block: never a collision exception. In transit it keeps MARGIN (2 mm) from everything (and from the arm links);
   on the vertical line inside its own column (the last approach of a put, the first lift of a pick) it may touch but
   not penetrate (>= -0.1 mm): the support contact and the flush faces of blocks in the adjacent cells (the grid pitch
   equals the block size, so neighbours are flush at 0 mm by construction).
3. Release: the block is lowered to 2 mm above its support before the fingers open. Top-down puts always release at
   2 mm (the block is the lowest part of hand + block). Front (upright) puts try 2 mm first and then 5 / 10 / 16 / 20 /
   25 / 30 / 35 / 40 mm: with the palm axis horizontal the hand housing (32 mm) and the wrist (link6) hang below the
   palm axis, so a centre-gripped 60 mm block over the table interior cannot be released lower than 16 mm (measured:
   link6 1.46 mm from the stack table at 15 mm, 2.0 mm at 16 mm; at the table's front edge the wrist clears it and 2-5 mm
   work); the release height used is reported per action (stats).
4. Clearance measured exactly: MuJoCo mid-phase OFF (runtime margins are otherwise ignored for multi-geom bodies), every
   collidable geom has a 3 mm contact margin, and every box-box pair involving the robot or the held block gets the
   EXACT oriented-box distance (separating-axis value when overlapping or >= 2 mm; else closed form for aligned boxes,
   convex QP otherwise; same measure as renders_v2/tools/robust_contacts.py). Mesh pairs come from the pipeline.

Rule (every committed configuration, dense at <= 5 mrad per joint incl. VAMP-planned transits, finger motions swept in
0.5 mm steps): arm geoms (link1-7, hand, fingers) >= 2 mm from every non-robot geom; the gripper may come to 0 mm of
the action's block while the fingers are open on its line and is not checked against it while closing; the held block
as in 2.; arm links >= 2 mm from the held block. Each action: neutral home -> (transit: straight joint motion, else
VAMP RRT-Connect) -> top of the vertical line -> line -> grasp / release -> line -> home (an empty gripper ends at the
pre-grasp width). The per-action check restores the symbolic state canonically (arm at the neutral home, the held
block at the grip's in-hand pose), which is exactly the world the previous action leaves, so per-action verdicts chain
(the same executor runs plans continuously for the verifier: run_action without restore).
Grasp sets: pick-upright / pick-long-upright grip the end cell with the palm axis horizontal and the block hanging
below (or the bottom end with the block above, the same in-hand pose); top-down picks grip the centre. The in-hand pose
is unique per grip up to the box symmetries (the four front yaws differ by a block symmetry; the 180 deg roll would
grip the other way round and is not used). Both grips use the top-down grasp depth (site 14 mm short of the block's
centre line, finger tips inside the block): the executor's front grasp (site ON the centre line) let the finger tips
protrude ~7 mm beyond the block, so every flat put after make-flat-*-from-upright drove a finger into the support.
"""
from __future__ import annotations

import itertools
import math
import time
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.multilevel_blocks import honest as H
from tampanda.symbolic.domains.multilevel_blocks import honest_lib as L
from tampanda.symbolic.domains.multilevel_blocks.executor import (
    _HOME_NEUTRAL_Q, _PREGRASP_CTRL, _EE_TO_BLOCK_CENTRE_Z, _QUAT_TOP_DOWN_Y, _QUAT_FRONT_Y, _TABLE_TOP_LOCAL_Z,
)
from tampanda.symbolic.domains.multilevel_blocks.state import (
    _QUAT_FLAT_X, _QUAT_FLAT_Y, _QUAT_UPRIGHT, _HELD_FLUENTS, restore_state,
)

VERSION = "2026-10-06b"   # a: first version; b: release ladder for front puts 2/5/10/16/20/25/30/35/40 mm (16 mm = the
                          # wrist (link6) limit over the table interior, measured)
MARGIN = 0.002            # arm geoms vs scene; held block vs scene in transit; arm links vs held block
PEN_TOL = 1e-4            # held block on its own vertical line: no penetration (verifier tolerance)
GATE = MARGIN + 1e-9      # box-box pairs whose separating-axis gap is >= GATE keep that lower bound (no rule can fire)
CONTACT_MARGIN = 0.003    # contact window of every collidable geom (the arm's is already 3 mm)
FINGER_OPEN = _PREGRASP_CTRL          # 20 mm per finger: descent of a pick, empty gripper at rest
RELEASE_MIN = 0.018                   # release opening >= 3 mm per side beyond a 30 mm block
FING_STEP = 0.0005
TOP_LIFTS = (0.002,)
FRONT_LIFTS = (0.002, 0.005, 0.010, 0.016, 0.020, 0.025, 0.030, 0.035, 0.040)
LINE_STEP = 0.01
MAX_STEP = 0.15
MAX_CHAINS = 16
LINE_MIN_RISE = 0.10      # the vertical line rises >= 10 cm above the grasp / release pose ...
LINE_CLEAR = 0.03         # ... and until hand + block clear every block / table top on that table by >= 3 cm
GRIP_FLUENTS = ("grip-centre", "grip-end")
ARM_LINKS = ("link1", "link2", "link3", "link4", "link5", "link6", "link7")
GRIPPER = ("hand", "left_finger", "right_finger")

PICKS = {"pick-cube": ("top", "grip-centre"), "pick-flat-x": ("top", "grip-centre"), "pick-flat-y": ("top", "grip-centre"),
         "pick-long-x": ("top", "grip-centre"), "pick-long-y": ("top", "grip-centre"),
         "pick-upright": ("front", "grip-end"), "pick-long-upright": ("front", "grip-end")}
PUTS = {"put-cube": ("top", _QUAT_FLAT_X), "put-flat-x": ("top", _QUAT_FLAT_X), "put-flat-y": ("top", _QUAT_FLAT_Y),
        "put-long-x": ("top", _QUAT_FLAT_X), "put-long-y": ("top", _QUAT_FLAT_Y),
        "put-upright": ("front", _QUAT_UPRIGHT), "put-long-upright": ("front", _QUAT_UPRIGHT)}
TRANSFORMS = {"make-upright-from-x": "front", "make-upright-from-y": "front", "make-flat-x-from-upright": "top_down",
              "make-flat-y-from-upright": "top_down", "turn-x-to-y": "top_down", "turn-y-to-x": "top_down",
              "make-long-upright-from-x": "front", "make-long-upright-from-y": "front",
              "make-long-flat-x-from-upright": "top_down", "make-long-flat-y-from-upright": "top_down",
              "turn-long-x-to-y": "top_down", "turn-long-y-to-x": "top_down"}


class GripStateError(RuntimeError):
    """The symbolic state does not say how the held block is gripped (data of the old domain under honest_grip, or
    grip atoms under another checker): never silently treated as infeasible."""


# --------------------------------------------------------------------------------------------------- geometry helpers
def q2R(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def box_symmetries(h):
    """Proper rotations (signed permutation matrices, det +1) mapping a box with half sizes h onto itself."""
    out = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, p[i]] = s[i]
            if np.linalg.det(M) > 0 and np.allclose(np.abs(M) @ h, h, atol=1e-9):
                out.append(M)
    return out


def canonical_grip(grip: str, shape: str, cube_size: float):
    """In-hand pose (block centre and orientation in the EE-site frame) of a grip. Both grips use the executor's
    top-down grasp DEPTH (the site _EE_TO_BLOCK_CENTRE_Z = 14 mm short of the block's centre line along the palm axis,
    finger tips ~7 mm inside the far face) and close across the block's 30 mm width; they differ in WHERE along the
    block's long axis: grip-centre at the centre (= pick-cube / pick-flat-* / pick-long-x/y, _QUAT_TOP_DOWN_Y on a flat
    block); grip-end at the end cell (= pick-upright / pick-long-upright from the front, _QUAT_FRONT_Y, block upright
    hanging below, its centre h_top = (cells - 1) * cube / 2 below the site). NOTE: the executor's front grasp put the
    site ON the centre line, so the finger tips protruded ~7 mm beyond the block; after make-flat-*-from-upright every
    flat put then drove a finger into the support (measured: left finger -5.3 mm into the stack table)."""
    if grip == "grip-centre":
        Rs = q2R(_QUAT_TOP_DOWN_Y)
        return Rs.T @ np.array([0.0, 0.0, -_EE_TO_BLOCK_CENTRE_Z]), Rs.T @ q2R(_QUAT_FLAT_X)
    if grip == "grip-end":
        if shape == "cube":
            raise GripStateError("a cube cannot be gripped at its end")
        cells = 3 if shape == "long" else 2
        h_top = (cells - 1) * cube_size / 2
        Rs = q2R(_QUAT_FRONT_Y)                        # palm axis = +world y at this reference pose
        return Rs.T @ np.array([0.0, _EE_TO_BLOCK_CENTRE_Z, -h_top]), Rs.T @ q2R(_QUAT_UPRIGHT)
    raise GripStateError("unknown grip %r" % (grip,))


def shape_of(block: str) -> str:
    return "cube" if block.startswith("cube") else ("long" if block.startswith("long") else "oblong")


def held_and_grip(state: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(held fluent, held block, grip) of a symbolic state dict; raises GripStateError if a held block has no grip."""
    held = [(k[0], k[1]) for k, v in state.items() if v and isinstance(k, tuple) and len(k) == 2 and k[0] in _HELD_FLUENTS]
    grips = {k[1]: k[0] for k, v in state.items() if v and isinstance(k, tuple) and len(k) == 2 and k[0] in GRIP_FLUENTS}
    if len(held) > 1:
        raise GripStateError("more than one held block: %s" % held)
    if not held:
        if grips:
            raise GripStateError("grip atoms without a held block: %s" % grips)
        return None, None, None
    fl, b = held[0]
    if b not in grips or len(grips) != 1:
        raise GripStateError("held block %s has no (unique) grip atom (%s): use the grip-aware domain with "
                             "MLB_CHECKER=honest_grip" % (b, grips))
    return fl, b, grips[b]


# --------------------------------------------------------------------------------------------------- exact distances
def _sat(C1, R1, H1, C2, R2, H2):
    """Separating-axis value of N box pairs (> 0 separated: lower bound of the distance; <= 0 minus the overlap)."""
    A1 = np.transpose(R1, (0, 2, 1)); A2 = np.transpose(R2, (0, 2, 1))
    cr = np.cross(A1[:, :, None, :], A2[:, None, :, :]).reshape(len(C1), 9, 3)
    nrm = np.linalg.norm(cr, axis=2)
    ok = nrm > 1e-6
    cr = cr / np.where(ok, nrm, 1.0)[..., None]
    ax = np.concatenate([A1, A2, cr], axis=1)
    t = C2 - C1
    proj = np.abs(np.einsum("nka,na->nk", ax, t))
    r1 = np.einsum("nkb,nb->nk", np.abs(np.einsum("nka,nab->nkb", ax, R1)), H1)
    r2 = np.einsum("nkb,nb->nk", np.abs(np.einsum("nka,nab->nkb", ax, R2)), H2)
    gap = proj - r1 - r2
    valid = np.concatenate([np.ones((len(C1), 6), bool), ok], axis=1)
    return np.where(valid, gap, -np.inf).max(1)


def obb_exact(c1, R1, h1, c2, R2, h2):
    """Exact Euclidean distance of two separated oriented boxes: closed form when their axes are aligned (signed
    permutation), else min |c1 + R1 u - c2 - R2 v| over the box coordinates (convex QP, L-BFGS-B; robust_contacts)."""
    D = R1.T @ R2
    A = np.abs(D)
    if np.all((A > 1 - 1e-9) | (A < 1e-7)) and np.all(np.sum(A > 0.5, axis=1) == 1):
        t = R1.T @ (c2 - c1)
        g = np.maximum(np.abs(t) - h1 - A @ h2, 0.0)
        return float(np.linalg.norm(g))
    from scipy.optimize import minimize

    def f(x):
        r = c1 + R1 @ x[:3] - c2 - R2 @ x[3:]
        return float(r @ r), np.concatenate([2 * R1.T @ r, -2 * R2.T @ r])
    x0 = np.concatenate([np.clip(R1.T @ (c2 - c1), -h1, h1), np.clip(R2.T @ (c1 - c2), -h2, h2)])
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(-a, a) for a in h1] + [(-a, a) for a in h2],
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500})
    return float(np.sqrt(max(res.fun, 0.0)))


def fk_tcp(Q):
    """Panda TCP poses (N x 4 x 4, base = world origin) of N configurations: tampanda.kinematics.panda.fk, batched."""
    Q = np.atleast_2d(np.asarray(Q, float))
    N = len(Q)
    T = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
    for i, (a, dd, al) in enumerate(P.DH):
        c, s = np.cos(Q[:, i]), np.sin(Q[:, i]); ca, sa = math.cos(al), math.sin(al)
        A = np.zeros((N, 4, 4))
        A[:, 0, 0] = c; A[:, 0, 1] = -s; A[:, 0, 3] = a
        A[:, 1, 0] = s * ca; A[:, 1, 1] = c * ca; A[:, 1, 2] = -sa; A[:, 1, 3] = -sa * dd
        A[:, 2, 0] = s * sa; A[:, 2, 1] = c * sa; A[:, 2, 2] = ca; A[:, 2, 3] = ca * dd
        A[:, 3, 3] = 1.0
        T = T @ A
    F = np.eye(4); F[2, 3] = P.FLANGE_D
    Z = np.eye(4); Z[:2, :2] = [[math.cos(P.HAND_YAW), -math.sin(P.HAND_YAW)], [math.sin(P.HAND_YAW), math.cos(P.HAND_YAW)]]
    C = np.eye(4); C[2, 3] = P.TCP_OFFSET
    return T @ (F @ Z @ C)


IK_TOL = 1e-6


def exact_boxbox(m, d, ga, gb):
    C1, R1, H1 = d.geom_xpos[ga], d.geom_xmat[ga].reshape(-1, 3, 3), m.geom_size[ga][:, :3]
    C2, R2, H2 = d.geom_xpos[gb], d.geom_xmat[gb].reshape(-1, 3, 3), m.geom_size[gb][:, :3]
    s = _sat(C1, R1, H1, C2, R2, H2)
    out = s.copy()
    for k in np.where((s > 0) & (s < GATE))[0]:
        out[k] = obb_exact(C1[k], R1[k], H1[k], C2[k], R2[k], H2[k])
    return out


# --------------------------------------------------------------------------------------------------- the exact rule
def _rule_tables(env):
    m = env.model
    gb = np.asarray(m.geom_bodyid)
    link = np.zeros(m.nbody, bool); grip = np.zeros(m.nbody, bool)
    for b in ARM_LINKS:
        link[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)] = True
    for b in GRIPPER:
        grip[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)] = True
    isbox = np.asarray(m.geom_type) == int(mujoco.mjtGeom.mjGEOM_BOX)
    return {"gb": gb, "link": link, "grip": grip, "arm": link | grip, "isbox": isbox}


def _check_exact(env) -> bool:
    """The honest_grip rule on the current contact list (see module doc). env._grip_ctx: line (bool), act (body id of
    the action's block or -1), grip_thr (gripper vs the action's block: float, or None = not checked)."""
    m, d = env.model, env.data
    if env.collision_diagnostics:
        env.last_collision = None
    n = d.ncon
    if n == 0:
        return True
    T = env._grip_rule
    ctx = env._grip_ctx
    con = d.contact
    g1 = np.asarray(con.geom1[:n]); g2 = np.asarray(con.geom2[:n])
    b1 = T["gb"][g1]; b2 = T["gb"][g2]
    held = env._collision_held_body["body_id"] if env._collision_held_body is not None else -1
    a1 = T["arm"][b1]; a2 = T["arm"][b2]
    h1 = b1 == held; h2 = b2 == held
    r1 = a1 | h1; r2 = a2 | h2
    thr = np.full(n, -np.inf)
    as1 = a1 & ~r2; as2 = a2 & ~r1
    thr[as1 | as2] = MARGIN
    act = ctx["act"]
    if act >= 0:
        gvb = (as1 & T["grip"][b1] & (b2 == act)) | (as2 & T["grip"][b2] & (b1 == act))
        thr[gvb] = ctx["grip_thr"] if ctx["grip_thr"] is not None else -np.inf
    hs = (h1 & ~r2) | (h2 & ~r1)
    thr[hs] = -PEN_TOL if ctx["line"] else MARGIN
    hl = (h1 & T["link"][b2]) | (h2 & T["link"][b1])
    thr[hl] = MARGIN
    rel = np.isfinite(thr)
    if not rel.any():
        return True
    dist = np.array(con.dist[:n], float)
    bb = np.where(rel & T["isbox"][g1] & T["isbox"][g2])[0]
    if len(bb):
        dist[bb] = exact_boxbox(m, d, g1[bb], g2[bb])
    bad = rel & (dist < thr)
    if not bad.any():
        return True
    if env.collision_diagnostics:
        # prefer a violation fixed by the EE pose (gripper or held block): it holds for every IK solution
        eefix = bad & (T["grip"][b1] | T["grip"][b2] | h1 | h2)
        i = int(np.argmax(eefix)) if eefix.any() else int(np.argmax(bad))
        nm = lambda t, i_: mujoco.mj_id2name(m, t, int(i_)) or ""
        env.last_collision = {"body1": nm(mujoco.mjtObj.mjOBJ_BODY, b1[i]), "body2": nm(mujoco.mjtObj.mjOBJ_BODY, b2[i]),
                              "geom1": nm(mujoco.mjtObj.mjOBJ_GEOM, g1[i]), "geom2": nm(mujoco.mjtObj.mjOBJ_GEOM, g2[i]),
                              "distance": float(dist[i]), "threshold": float(thr[i]), "ee_fixed": bool(eefix.any())}
    return False


def _icf_exact(env, configuration) -> bool:
    m, d = env.model, env.data
    d.qpos[:7] = configuration
    d.qvel[:7] = 0.0
    mujoco.mj_kinematics(m, d)
    if env._collision_held_body is not None:
        env._apply_collision_held_body()
        mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    return _check_exact(env)


def build_grip_env(scratch_dir: Path, config=None, rate: float = 10000.0):
    """(env, workspace, config): the honest scene (base_panda_v2 real link meshes, H.build_honest_env) with the exact
    rule: mid-phase off, 3 mm contact window on every collidable geom, static scene geoms only collide with the robot
    (contype 1 / conaffinity 0; the held block gets conaffinity 1), check = _check_exact."""
    env, ws, cfg = H.build_honest_env(Path(scratch_dir), config=config, rate=rate, fast=True)
    m = env.model
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    T = _rule_tables(env)
    for g in range(m.ngeom):
        if T["arm"][m.geom_bodyid[g]]:
            continue
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            m.geom_margin[g] = CONTACT_MARGIN
            m.geom_gap[g] = 0.0
            m.geom_contype[g] = 1
            m.geom_conaffinity[g] = 0
    for g in range(m.ngeom):
        if T["arm"][m.geom_bodyid[g]] and m.geom_conaffinity[g]:
            assert m.geom_contype[g] == 0 and abs(m.geom_margin[g] - CONTACT_MARGIN) < 1e-12, g
    env._grip_rule = T
    env._grip_ctx = {"line": False, "act": -1, "grip_thr": MARGIN}
    env.check_collisions = types.MethodType(_check_exact, env)
    env.is_collision_free_no_restore = types.MethodType(_icf_exact, env)
    env.collision_diagnostics = True
    mujoco.mj_forward(m, d := env.data)
    return env, ws, cfg


# --------------------------------------------------------------------------------------------------- executor
class _Rec(list):
    """record buffer: every dense sample is stored as (full qpos, carried body id)."""

    def __init__(self, env):
        super().__init__()
        self.env = env

    def append(self, q):
        h = self.env._collision_held_body
        super().append((np.asarray(q, float).copy(), int(h["body_id"]) if h is not None else -1))


class GripExecutor(L.HonestLibExecutor):
    """Library-style motion search for EVERY action (picks, puts, re-orientations) under the exact rule."""

    record = None        # list: committed motion as (qpos, carried body id), in execution order (continuous runs)
    last_stats = None    # dict: the last action's search statistics (release height, candidate, chains, ...)

    # ---------------------------------------------------------------- bookkeeping
    def _init_grip(self):
        e = self.env
        m = e.model
        self._blocks = {}
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            b = int(m.jnt_bodyid[j]); nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm.startswith(("cube_", "oblong_", "long_")):
                gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
                self._blocks[nm] = {"bid": b, "jid": j, "qa": int(m.jnt_qposadr[j]), "va": int(m.jnt_dofadr[j]),
                                    "geoms": gs, "h": np.asarray(m.geom_size[gs[0]][:3], float).copy()}
        self._syms = {}
        cfg = self.config
        self._table_top = {"stack": cfg.stack_table_pos[2] + _TABLE_TOP_LOCAL_Z,
                           "parts": cfg.parts_table_pos[2] + _TABLE_TOP_LOCAL_Z}
        self._stack_y = float(cfg.stack_table_pos[1]); self._parts_y = float(cfg.parts_table_pos[1])
        self._q7s()

    def _syms_of(self, block):
        h = self._blocks[block]["h"]
        k = tuple(np.round(h, 6))
        if k not in self._syms:
            self._syms[k] = box_symmetries(h)
        return self._syms[k]

    def _ctx(self, line, act=-1, grip_thr=MARGIN):
        self.env._grip_ctx = {"line": bool(line), "act": int(act), "grip_thr": grip_thr}

    def _set_flags(self, block, held):
        m = self.env.model
        for g in self._blocks[block]["geoms"]:
            m.geom_contype[g] = 1
            m.geom_conaffinity[g] = 1 if held else 0

    def _scene_flags(self):
        """After a restore: parked blocks never collide, placed blocks collide with the robot only."""
        m, d = self.env.model, self.env.data
        for nm, x in self._blocks.items():
            parked = d.qpos[x["qa"]] > 5.0
            for g in x["geoms"]:
                m.geom_contype[g] = 0 if parked else 1
                m.geom_conaffinity[g] = 0
                m.geom_margin[g] = CONTACT_MARGIN
                m.geom_gap[g] = 0.0

    def _block_T(self, block):
        d = self.env.data; x = self._blocks[block]
        return d.qpos[x["qa"]:x["qa"] + 3].copy(), q2R(d.qpos[x["qa"] + 3:x["qa"] + 7])

    def _set_block(self, block, pos, R):
        d = self.env.data; x = self._blocks[block]
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
        d.qpos[x["qa"]:x["qa"] + 3] = pos; d.qpos[x["qa"] + 3:x["qa"] + 7] = q
        d.qvel[x["va"]:x["va"] + 6] = 0.0

    def _hold(self, block, rel_p, rel_R):
        e = self.env; x = self._blocks[block]
        e._attached = {"body_name": block, "joint_id": x["jid"], "qadr": x["qa"], "vadr": x["va"],
                       "rel_pos": np.asarray(rel_p, float).copy(), "rel_mat": np.asarray(rel_R, float).copy()}
        e.set_collision_held_body(block, np.asarray(rel_p, float), np.asarray(rel_R, float))
        self._set_flags(block, True)
        self._held_block = block

    def _unhold(self, block):
        e = self.env
        e.detach_object()
        self._set_flags(block, False)
        self._held_block = None
        self._held_offset = np.zeros(3)

    def _arm(self, q, fingers=None):
        e = self.env; m, d = e.model, e.data
        d.qpos[:7] = q
        if fingers is not None:
            d.qpos[7] = d.qpos[8] = fingers
        d.qvel[:9] = 0.0
        mujoco.mj_kinematics(m, d)
        if e._attached is not None:
            e._apply_attachment()
            mujoco.mj_kinematics(m, d)

    def _site_T(self):
        d = self.env.data; sid = self.ee_site_id
        return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()

    def _try(self, Q):
        """honest_lib._try with samples tagged by the carried body: (ok, [(qpos, carried id)])."""
        keep = self.record
        self.record = _Rec(self.env)
        try:
            ok = self._dense(Q)
            return ok, list(self.record)
        finally:
            self.record = keep

    def _sweep(self, q, f0, f1):
        """Finger motion f0 -> f1 at arm configuration q in <= 0.5 mm steps: (ok, samples, last passing width)."""
        e = self.env
        n = max(1, int(math.ceil(abs(f1 - f0) / FING_STEP)))
        smp, last = [], f0
        held = e._collision_held_body
        hid = int(held["body_id"]) if held is not None else -1
        for k in range(1, n + 1):
            f = f0 + (f1 - f0) * k / n
            e.data.qpos[7] = e.data.qpos[8] = f
            if not e.is_collision_free_no_restore(q):
                return False, smp, last
            smp.append((e.data.qpos.copy(), hid))
            last = f
        return True, smp, last

    # ---------------------------------------------------------------- restore (per-action check)
    def restore(self, state: Dict):
        """Canonical world of a symbolic state: blocks on their cells, arm at the neutral home, an empty gripper at
        the pre-grasp width, a held block at its grip's in-hand pose with the fingers closed on it."""
        e = self.env
        if not hasattr(self, "_blocks"):
            self._init_grip()
        fl, block, grip = held_and_grip(state)
        st = {k: v for k, v in state.items()
              if not (isinstance(k, tuple) and len(k) == 2 and (k[0] in _HELD_FLUENTS or k[0] in GRIP_FLUENTS))}
        restore_state(e, self.workspace, self.config, st, home_qpos=_HOME_NEUTRAL_Q, on_held="park", executor=self)
        e.clear_collision_exceptions()
        self._scene_flags()
        self._arm(_HOME_NEUTRAL_Q, FINGER_OPEN)
        if block is not None:
            rel_p, rel_R = canonical_grip(grip, shape_of(block), self.config.cube_size)
            sp, sR = self._site_T()
            self._set_block(block, sp + sR @ rel_p, sR @ rel_R)
            self._hold(block, rel_p, rel_R)
            self._arm(_HOME_NEUTRAL_Q, self.config.cube_half_extent)
        mujoco.mj_forward(e.model, e.data)
        self._ctx(False)

    # ---------------------------------------------------------------- geometry of an action
    def _cells_target(self, cells, quat):
        P = np.array([self._cell_pose(c) for c in cells])
        return P.mean(axis=0), q2R(quat)

    def _table_of(self, y):
        return "stack" if abs(y - self._stack_y) < abs(y - self._parts_y) else "parts"

    def _hand_candidates(self, R_block, rel_R, family, block):
        out, seen = [], set()
        for S in self._syms_of(block):
            Rh = R_block @ S @ rel_R.T
            z = Rh[:, 2]
            if family == "top" and z[2] > -0.999:
                continue
            if family == "front" and abs(z[2]) > 1e-3:
                continue
            k = tuple(np.round(Rh, 6).ravel())
            if k in seen:
                continue
            seen.add(k); out.append(Rh)
        return out

    def _line_top(self, ee_bottom, Rh, rel_p, rel_R, block, family):
        """EE height of the top of the vertical line: >= LINE_MIN_RISE above the bottom and hand + block >= LINE_CLEAR
        above every block / table top on that table (the transit starts and ends there)."""
        d = self.env.data
        tbl = self._table_of(ee_bottom[1])
        top = self._table_top[tbl]
        for nm, x in self._blocks.items():
            if nm == block or d.qpos[x["qa"]] > 5.0:
                continue
            p, R = self._block_T(nm)
            if self._table_of(p[1]) == tbl:
                top = max(top, p[2] + np.abs(R[2, :]) @ x["h"])
        h = self._blocks[block]["h"]
        bc = (Rh @ rel_p)[2]
        bh = np.abs((Rh @ rel_R)[2, :]) @ h
        low = min(bc - bh, -0.02 if family == "top" else -0.05)
        return max(ee_bottom[2] + LINE_MIN_RISE, top + LINE_CLEAR - low)

    def _ik(self, pos, R, q7s):
        """honest_lib.ik_all (frantik, all branches, q7 grid) with every solution verified by forward kinematics:
        frantik occasionally returns configurations ~1 cm off the target (seen on 2/38 solutions), which must not be
        mistaken for the target pose."""
        S = L.ik_all(self.env, pos, R, q7s)
        if not len(S):
            return S
        T = fk_tcp(S)
        tcp = np.asarray(pos, float) + self.env._lib_site_to_tcp * R[:, 2]
        ok = (np.linalg.norm(T[:, :3, 3] - tcp, axis=1) < IK_TOL) & (np.abs(T[:, :3, :3] - R).max(axis=(1, 2)) < IK_TOL)
        return S[ok]

    def _layers(self, bottom, top_z, R):
        q7s = self._q7s()
        n = max(1, int(math.ceil((top_z - bottom[2]) / LINE_STEP)))
        pts = [np.array([bottom[0], bottom[1], bottom[2] + (top_z - bottom[2]) * k / n]) for k in range(n + 1)]
        L0 = self._ik(pts[0], R, q7s)
        return pts, L0, q7s

    def _chains_from(self, free0, pts, R, q7s):
        """Branch-consistent chains bottom -> top from the collision-free bottom configurations (nearest solution of
        each 1 cm layer, <= MAX_STEP per joint), deduplicated, ordered by the top's distance to the neutral home."""
        if not len(free0):
            return []
        ends = np.array(free0)
        chains = [ends[:, None, :]]
        alive = np.ones(len(ends), bool)
        for p in pts[1:]:
            Lk = self._ik(p, R, q7s)
            if not len(Lk):
                return []
            D = np.abs(ends[:, None, :] - Lk[None, :, :]).max(2)
            i = np.argmin(D, axis=1)
            alive &= D[np.arange(len(ends)), i] <= MAX_STEP
            ends = Lk[i]
            chains.append(ends[:, None, :])
            if not alive.any():
                return []
        C = np.concatenate(chains, axis=1)[alive]
        out, seen = [], set()
        for c in C:
            key = tuple(np.round(c[-1], 2))
            if key in seen:
                continue
            seen.add(key); out.append(c)
        out.sort(key=lambda c: float(np.abs(c[-1] - _HOME_NEUTRAL_Q).max()))
        return out[:MAX_CHAINS]

    def _free_bottom(self, L0, stats):
        """Collision-free bottom configurations under the current context; stops early when the violation is fixed by
        the EE pose (gripper / held block), which then holds for every IK solution."""
        e = self.env
        free = []
        for s in L0:
            if e.is_collision_free_no_restore(s):
                free.append(s)
            else:
                lc = e.last_collision or {}
                if lc.get("ee_fixed"):
                    stats["ee_fixed"] = "%s/%s %.2fmm" % (lc.get("body1"), lc.get("body2"), 1000 * lc.get("distance", 0))
                    stats.setdefault("fail", []).append("%s:ee-fixed:%s" % (stats.get("_tag", ""), stats["ee_fixed"]))
                    return []
        if not free:
            stats.setdefault("fail", []).append("%s:no-free-bottom(%d)" % (stats.get("_tag", ""), len(L0)))
        return free

    # ---------------------------------------------------------------- transit
    def _transit_held(self, qa, qb, block):
        held = None
        if block is not None:
            att = self.env._attached
            held = (np.asarray(att["rel_pos"]), np.asarray(att["rel_mat"]), self._blocks[block]["h"])
        return self._transit(qa, qb, held)

    # ---------------------------------------------------------------- pick
    def _pick(self, block, cells, family, grip, stats):
        e = self.env
        x = self._blocks[block]
        p_b, R_b = self._block_T(block)
        c_t, _ = self._cells_target(cells, _QUAT_FLAT_X)
        if np.linalg.norm(p_b - c_t) > 1e-3:
            raise RuntimeError("world/symbolic mismatch: %s at %s, cells %s at %s" % (block, p_b, cells, c_t))
        rel_p, rel_R0 = canonical_grip(grip, shape_of(block), self.config.cube_size)
        base = e.data.qpos.copy()
        cands = self._hand_candidates(R_b, rel_R0, family, block)
        if family == "front":
            cands.sort(key=lambda Rh: (bool((Rh @ rel_p)[2] > 1e-6), -float(np.dot(Rh[:2, 2], p_b[:2]) / (np.linalg.norm(p_b[:2]) + 1e-9))))
        stats["candidates"] = len(cands)
        for ci, Rh in enumerate(cands):
            ee = p_b - Rh @ rel_p
            rel_R = Rh.T @ R_b                     # exact in-hand orientation (canonical up to a box symmetry)
            top_z = self._line_top(ee, Rh, rel_p, rel_R, block, family)
            pts, L0, q7s = self._layers(ee, top_z, Rh)
            if not len(L0):
                stats.setdefault("fail", []).append("c%d:no-ik" % ci); continue
            e.data.qpos[:] = base; self._arm(base[:7], FINGER_OPEN)
            self._ctx(True, x["bid"], 0.0)
            stats["_tag"] = "c%d" % ci
            free = self._free_bottom(L0, stats)
            chains = self._chains_from(free, pts, Rh, q7s)
            stats["chains"] = stats.get("chains", 0) + len(chains)
            for ch in chains:
                # 1. close on the block at the grasp configuration (gripper vs block not checked while closing)
                e.data.qpos[:] = base; self._arm(ch[0], FINGER_OPEN)
                self._ctx(True, x["bid"], None)
                ok, s_close, _ = self._sweep(ch[0], FINGER_OPEN, self.config.cube_half_extent)
                if not ok:
                    stats.setdefault("fail", []).append("c%d:close" % ci); break
                # 2. lift the block up the line, 3. transit home carrying it
                self._hold(block, rel_p, rel_R)
                self._ctx(True)
                ok, s_up = self._try(list(ch))
                if not ok:
                    self._unhold(block); e.data.qpos[:] = base
                    stats.setdefault("fail", []).append("c%d:lift:%s" % (ci, _lc(e))); continue
                self._arm(ch[-1]); self._ctx(False)
                ok, s_out = self._transit_held(ch[-1], _HOME_NEUTRAL_Q, block)
                self._unhold(block); e.data.qpos[:] = base
                if not ok:
                    stats.setdefault("fail", []).append("c%d:transit-out" % ci); continue
                # 4. empty: descent down the line (fingers open), 5. transit home -> top of the line
                self._arm(ch[-1], FINGER_OPEN); self._ctx(True, x["bid"], 0.0)
                ok, s_down = self._try(list(ch[::-1]))
                if not ok:
                    stats.setdefault("fail", []).append("c%d:descent:%s" % (ci, _lc(e))); continue
                self._arm(_HOME_NEUTRAL_Q, FINGER_OPEN); self._ctx(False)
                ok, s_in = self._transit_held(_HOME_NEUTRAL_Q, ch[-1], None)
                if not ok:
                    stats.setdefault("fail", []).append("c%d:transit-in" % ci); continue
                # commit
                e.data.qpos[:] = base
                self._arm(ch[0], self.config.cube_half_extent)
                self._hold(block, rel_p, rel_R)
                self._arm(_HOME_NEUTRAL_Q, self.config.cube_half_extent)
                if self.record is not None:
                    self.record.extend(s_in + s_down + s_close + s_up + s_out)
                stats.update({"candidate": ci, "line_top_mm": round(1000 * (top_z - ee[2]), 1)})
                return True
        e.data.qpos[:] = base; self._arm(base[:7])
        return False

    # ---------------------------------------------------------------- put
    def _put(self, block, cells, family, quat, stats):
        e = self.env
        x = self._blocks[block]
        att = e._attached
        if att is None or att["body_name"] != block:
            raise RuntimeError("put %s: block not held (held: %s)" % (block, att and att["body_name"]))
        rel_p, rel_R = np.asarray(att["rel_pos"], float).copy(), np.asarray(att["rel_mat"], float).copy()
        c_t, R_t = self._cells_target(cells, quat)
        base = e.data.qpos.copy()
        cands = self._hand_candidates(R_t, rel_R, family, block)
        if family == "front":
            cands.sort(key=lambda Rh: (bool((Rh @ rel_p)[2] > 1e-6), -float(np.dot(Rh[:2, 2], c_t[:2]) / (np.linalg.norm(c_t[:2]) + 1e-9))))
        stats["candidates"] = len(cands)
        lifts = FRONT_LIFTS if family == "front" else TOP_LIFTS
        for lift in lifts:
            for ci, Rh in enumerate(cands):
                ee = c_t + np.array([0.0, 0.0, lift]) - Rh @ rel_p
                top_z = self._line_top(ee, Rh, rel_p, rel_R, block, family)
                pts, L0, q7s = self._layers(ee, top_z, Rh)
                if not len(L0):
                    stats.setdefault("fail", []).append("L%.0f c%d:no-ik" % (1000 * lift, ci)); continue
                e.data.qpos[:] = base; self._arm(base[:7]); self._ctx(True)
                stats["_tag"] = "L%.0f c%d" % (1000 * lift, ci)
                free = self._free_bottom(L0, stats)
                chains = self._chains_from(free, pts, Rh, q7s)
                stats["chains"] = stats.get("chains", 0) + len(chains)
                for ch in chains:
                    # 1. release: the block (in the air, `lift` above its cells) is let go; fingers open >= 3 mm/side
                    e.data.qpos[:] = base; self._arm(ch[0])
                    p_air, R_air = self._block_T(block)
                    self._unhold(block)
                    self._ctx(True, x["bid"], 0.0)
                    ok, s_rel, w = self._sweep(ch[0], self.config.cube_half_extent, FINGER_OPEN)
                    if w < RELEASE_MIN - 1e-9:
                        self._hold(block, rel_p, rel_R); e.data.qpos[:] = base
                        stats.setdefault("fail", []).append("L%.0f c%d:release:%s" % (1000 * lift, ci, _lc(e))); break
                    if not ok:                      # keep the widest passing opening (>= 3 mm per side)
                        w = max(v for v in (0.018, 0.019, 0.020) if v <= w + 1e-12)
                        s_rel = [s for s in s_rel if s[0][7] <= w + 1e-12]
                    # 2. the block settles on its cells (drop = lift, straight down); 3. empty retreat up the line
                    self._set_block(block, c_t, R_air)
                    self._arm(ch[0], w)
                    s_rel = s_rel + [(e.data.qpos.copy(), -1)]
                    ok, s_up = self._try(list(ch))
                    if not ok:
                        self._set_block(block, p_air, R_air); self._hold(block, rel_p, rel_R); e.data.qpos[:] = base
                        stats.setdefault("fail", []).append("L%.0f c%d:retreat:%s" % (1000 * lift, ci, _lc(e))); continue
                    # 4. transit home (empty, the placed block an ordinary obstacle)
                    self._arm(ch[-1], w); self._ctx(False)
                    ok, s_out = self._transit_held(ch[-1], _HOME_NEUTRAL_Q, None)
                    self._set_block(block, p_air, R_air); self._hold(block, rel_p, rel_R); e.data.qpos[:] = base
                    if not ok:
                        stats.setdefault("fail", []).append("L%.0f c%d:transit-out" % (1000 * lift, ci)); continue
                    # 5. carrying: down the line (no penetration), 6. transit home -> top of the line
                    self._arm(ch[-1]); self._ctx(True)
                    ok, s_down = self._try(list(ch[::-1]))
                    if not ok:
                        stats.setdefault("fail", []).append("L%.0f c%d:descent:%s" % (1000 * lift, ci, _lc(e))); continue
                    self._arm(_HOME_NEUTRAL_Q); self._ctx(False)
                    ok, s_in = self._transit_held(_HOME_NEUTRAL_Q, ch[-1], block)
                    if not ok:
                        stats.setdefault("fail", []).append("L%.0f c%d:transit-in" % (1000 * lift, ci)); continue
                    # commit: block on its cells, arm home, fingers back to the pre-grasp width
                    e.data.qpos[:] = base
                    self._unhold(block)
                    self._set_block(block, c_t, R_air)
                    self._arm(_HOME_NEUTRAL_Q, w)
                    self._ctx(False)
                    ok, s_home, _ = self._sweep(_HOME_NEUTRAL_Q, w, FINGER_OPEN)
                    if not ok:
                        raise H.HonestViolation("finger opening at the neutral home")
                    self._arm(_HOME_NEUTRAL_Q, FINGER_OPEN)
                    if self.record is not None:
                        self.record.extend(s_in + s_down + s_rel + s_up + s_out + s_home)
                    stats.update({"candidate": ci, "release_mm": round(1000 * lift, 1), "release_width_mm": round(1000 * w, 1),
                                  "line_top_mm": round(1000 * (top_z - ee[2]), 1)})
                    return True
        e.data.qpos[:] = base; self._arm(base[:7])
        return False

    # ---------------------------------------------------------------- re-orientation
    def _transform(self, block, fam, stats):
        e = self.env
        att = e._attached
        if att is None or att["body_name"] != block:
            raise RuntimeError("transform %s: block not held" % block)
        base = e.data.qpos.copy()
        qh = self._handoff_qs[("stack", fam)]
        self._ctx(False)
        self._arm(_HOME_NEUTRAL_Q)
        ok1, s1 = self._transit_held(_HOME_NEUTRAL_Q, qh, block)
        ok2, s2 = (False, [])
        if ok1:
            self._arm(qh)
            ok2, s2 = self._transit_held(qh, _HOME_NEUTRAL_Q, block)
        e.data.qpos[:] = base; self._arm(_HOME_NEUTRAL_Q)
        if ok1 and ok2:
            if self.record is not None:
                self.record.extend(s1 + s2)
            return True
        stats.setdefault("fail", []).append("transit-%s" % ("out" if ok1 else "in"))
        return False

    # ---------------------------------------------------------------- dispatch
    def run_action(self, name: str, args: Tuple[str, ...]) -> bool:
        """Execute one PDDL action from the CURRENT world (arm at the neutral home). Used after restore() by the
        per-action check and back to back by the continuous certifier."""
        if not hasattr(self, "_blocks"):
            self._init_grip()
        e = self.env
        if float(np.abs(e.data.qpos[:7] - _HOME_NEUTRAL_Q).max()) > 1e-9:
            raise RuntimeError("run_action must start at the neutral home")
        name = name.lower()
        block, cells = args[0], tuple(args[1:])
        stats = {"action": name}
        self.last_stats = stats
        t0 = time.perf_counter()
        try:
            if name in PICKS:
                if e._attached is not None:
                    raise RuntimeError("pick with a held block")
                fam, grip = PICKS[name]
                ok = self._pick(block, cells, fam, grip, stats)
            elif name in PUTS:
                fam, quat = PUTS[name]
                ok = self._put(block, cells, fam, quat, stats)
            elif name in TRANSFORMS:
                ok = self._transform(block, TRANSFORMS[name], stats)
            else:
                raise ValueError("unknown action %r" % name)
        finally:
            self._ctx(False)
            stats["s"] = round(time.perf_counter() - t0, 4)
        if ok:
            assert float(np.abs(e.data.qpos[:7] - _HOME_NEUTRAL_Q).max()) < 1e-12
        return bool(ok)


def _lc(env):
    lc = env.last_collision or {}
    return "%s/%s:%.2f" % (lc.get("body1"), lc.get("body2"), 1000 * lc.get("distance", float("nan")))


def make_grip_executor(env, workspace, config):
    from tampanda.planners.rrt_star import RRTStar
    m = env.model
    l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
    mujoco.mj_kinematics(m, env.data)
    if np.abs(env.data.xpos[l0]).max() > 1e-9 or np.abs(env.data.xmat[l0].reshape(3, 3) - np.eye(3)).max() > 1e-9:
        raise RuntimeError("honest_grip assumes the robot base at the world origin (frantik / VAMP base frame)")
    if not hasattr(env, "_grip_rule"):
        raise RuntimeError("honest_grip executor needs build_grip_env")
    ex = GripExecutor(env, workspace, config, motion_planner=RRTStar(env, max_iterations=3000))
    env.ik.max_iters = 100
    ex._yaw_pool = None
    ex._init_grip()
    return ex


def check_action_grip(env, workspace, config, state: Dict, action, *, executor: GripExecutor) -> Dict:
    """Per-action check: canonical restore of `state` (incl. the grip), then the action's motion search."""
    t0 = time.perf_counter()
    held_and_grip(state)                       # GripStateError propagates: never a silent 'infeasible'
    name, *args = action
    try:
        executor.restore(state)
        ok = executor.run_action(name, tuple(args))
        err = None if ok else "no motion: %s" % ";".join((executor.last_stats or {}).get("fail", [])[:4])
    except H.HonestViolation as exc:
        ok, err = False, "honest: %s" % exc
    except GripStateError:
        raise
    except Exception as exc:
        ok, err = False, "%s: %s" % (type(exc).__name__, exc)
    return {"success": bool(ok), "elapsed_s": time.perf_counter() - t0, "error": err, "grip": True,
            "stats": dict(executor.last_stats or {})}
