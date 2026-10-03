"""Access-19 v2: crowded-shelf verifier on a fully parametric scene (2026-10-02).

Constraints (Daniel): Access-19 layout (two levels: cavity + top, a 7x7 cell grid on each; interior cells (0,0) and
(6,6) excluded as in v1), the Franka and its hand unchanged, ONE frontal grasp (hand horizontal, approach +y, several
heights on the object, no regrasp), every cell reachable even when the shelf is crowded, and objects in two
neighbouring cells of a row block each other.

Executable form, per cell:
  * paths: (a) open hand approach to the grasp with the target present, (b) the same grasp holding the object
    (= retreat after a pick / approach for a put). Interior: straight horizontal entry from 6 cm in front of the shelf.
    Top: vertical descent from above every object on the top (held object included).
    A cell is OK if (a) and (b) both have a continuous collision-free path (clearance >= margin) whose grasp
    configurations coincide (|dq| <= 0.1).
  * scenes: EMPTY (only the target), CROWDED (every cell of both levels occupied except the target and its designed
    blockers: interior = same-row neighbours + every cell in front within +-corridor columns; top = same-row
    neighbours + the row directly in front within +-1 column), LATERAL (target + one same-row neighbour): (a) and (b)
    must both FAIL.
IK: closed form over the redundancy (He & Liu via frantik: q7 grid x 4 branches), bottleneck DP over path layers.

    python -m tampanda.symbolic.domains.access19_v2.crowd out.json [k=v ...] [cells=interior:3_6;top:0_0] [render=dir]
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

import frantik
from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import verify as V
from tampanda.symbolic.domains.access19_v2.geometry import V2_ROBOT_XML, V2Params, patch_collisions

TCP_FROM_SITE = P.TCP_OFFSET - V.ATTACH_OFFSET
GCO = V.GRASP_CONTACT_OFFSET
DETECT = 0.03
FAR = np.array([30.0, 30.0, -30.0])


@dataclass
class Geo:
    c: float = 0.20            # shelf front edge (y, robot base frame)
    x0: float = 0.20           # lateral position of column 0 centre
    ncol: int = 7
    nrow: int = 7
    px: float = 0.065          # lateral pitch (interior; also the top unless pxt > 0)
    x0t: float = -9.0          # top grid: column 0 centre (-9 = same as the interior)
    pxt: float = 0.0           # top grid lateral pitch (0 = same as the interior)
    selfcol: int = 0           # 1: arm self-collision counts (non-adjacent links; pairs touching at home are excluded)
    no_exclude: int = 0        # 1: include v1's excluded interior corners (0,0) and (ncol-1, nrow-1)
    mirror_int: int = 0        # 1: interior column index runs the other way (moves v1's excluded corners)
    pyi: float = 0.065         # interior depth pitch
    pyt: float = 0.092         # top depth pitch
    gap0: float = 0.012        # front edge -> front face of the first-row object
    gap0t: float = -1.0        # top grid only (-1 = gap0): keeps the top's first row out of the zone the hand
                               # cannot reach right in front of the robot base
    f: float = 0.18            # cavity floor (top face of the base block)
    H: float = 0.30            # cavity height
    roof: float = 0.012
    wall: float = 0.012
    side: float = 0.12         # outer column centre -> inner side-wall face
    back: float = 0.02         # last-row object back face -> back wall
    r: float = 0.02            # object (cylinder) radius, or half side length for boxes
    box: int = 0               # 1: square boxes (2r x 2r x h) instead of cylinders
    h: float = 0.08            # object height (used when hts_* are empty)
    hts_int: tuple = ()        # per interior row object height (front row 0 .. back row 6); empty = all h
    hts_top: tuple = ()        # per top row object height
    fo: float = 0.04           # per-finger opening on approach
    depths: tuple = (0.01, 0.03)   # grasp height below the object top
    travel: float = 0.01       # clearance above the crowded top for the top-level descent start
    corridor: int = 1          # interior designed blockers: front cells within +-corridor columns
    top_corridor: int = 0      # 1: top blockers = same rule as the interior (everything in front within +-corridor cols)
    panels: int = 0            # 1: bookcase look -- full-height side panels from the ground to the top surface, open below the floor board
    legs: int = 0              # 1: shelf stands on four legs (open space under a 2 cm floor board) instead of a solid base
    q7_step_deg: float = 5.0
    step: float = 0.015        # path layer spacing (m)
    margin: float = 0.003
    max_dq: float = 0.25

    @property
    def s(self):
        return self.f + self.H + self.roof


def height(g, level, j):
    hs = g.hts_int if level == "interior" else g.hts_top
    return float(hs[j]) if hs else g.h


def max_height(g):
    return max([g.h] + list(g.hts_int) + list(g.hts_top))


def cells(g, level):
    out = []
    for i in range(g.ncol):
        for j in range(g.nrow):
            if level == "interior" and not g.no_exclude and (i, j) in ((0, 0), (g.ncol - 1, g.nrow - 1)):
                continue
            out.append((i, j))
    return out


def top_x(g):
    return (g.x0 if g.x0t < -5 else g.x0t), (g.pxt if g.pxt > 0 else g.px)


def centre(g, level, i, j):
    if level == "interior":
        ii = (g.ncol - 1 - i) if g.mirror_int else i
        return np.array([g.x0 + ii * g.px, g.c + g.gap0 + g.r + j * g.pyi, g.f])
    x0t, pxt = top_x(g)
    gt = g.gap0 if g.gap0t < 0 else g.gap0t
    return np.array([x0t + i * pxt, g.c + gt + g.r + j * g.pyt, g.s])


class Scene:
    def __init__(self, g: Geo):
        self.g = g
        spec = mujoco.MjSpec.from_file(str(V2_ROBOT_XML))
        wb = spec.worldbody
        fl = wb.add_geom(); fl.type = mujoco.mjtGeom.mjGEOM_PLANE; fl.size = [3, 3, 0.1]; fl.contype = 0
        fl.conaffinity = 0; fl.rgba = [0.55, 0.45, 0.35, 1]
        x0t, pxt = top_x(g)
        xl = g.x0 - g.side - g.wall; xr = g.x0 + (g.ncol - 1) * g.px + g.side + g.wall
        xl_t = x0t - g.side; xr_t = x0t + (g.ncol - 1) * pxt + g.side
        di = g.gap0 + 2 * g.r + (g.nrow - 1) * g.pyi + g.back        # interior depth to the back wall face
        dt = (g.gap0 if g.gap0t < 0 else g.gap0t) + 2 * g.r + (g.nrow - 1) * g.pyt + g.back        # top depth
        yb = g.c + max(di + g.wall, dt)
        self.boxes = []

        def box(name, lo, hi, rgba=(0.92, 0.92, 0.92, 1)):
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            b = wb.add_geom(); b.name = name; b.type = mujoco.mjtGeom.mjGEOM_BOX
            b.size = (hi - lo) / 2; b.pos = (hi + lo) / 2; b.contype = 1; b.conaffinity = 0; b.rgba = rgba
            self.boxes.append((name, lo, hi))
        XL, XR = min(xl, xl_t), max(xr, xr_t)
        if g.panels and g.f > 0.06:
            box("base", [XL, g.c, g.f - 0.02], [XR, yb, g.f])                 # floor board of the compartment
            box("panel_l", [XL, g.c, 0.0], [XL + g.wall, yb, g.f])            # side panels below the compartment
            box("panel_r", [XR - g.wall, g.c, 0.0], [XR, yb, g.f])
        elif g.legs and g.f > 0.06:
            box("base", [XL, g.c, g.f - 0.02], [XR, yb, g.f])                 # floor board
            for lx in (XL, XR - 0.04):
                for ly in (g.c, yb - 0.04):
                    box("leg_%d_%d" % (int(lx * 1000), int(ly * 1000)), [lx, ly, 0.0], [lx + 0.04, ly + 0.04, g.f - 0.02])
        else:
            box("base", [XL, g.c, min(0.0, g.f) - 0.6], [XR, yb, g.f])   # f < 0 = robot mounted above the shelf floor
        # where the top grid extends past the cavity the shelf is solid underneath (no overhang)
        if XL < xl:
            box("fill_l", [XL, g.c, g.f], [xl, yb, g.f + g.H])
        if XR > xr:
            box("fill_r", [xr, g.c, g.f], [XR, yb, g.f + g.H])
        box("roof", [min(xl, xl_t), g.c, g.f + g.H], [max(xr, xr_t), yb, g.s])
        box("wall_l", [xl, g.c, g.f], [xl + g.wall, yb, g.f + g.H])
        box("wall_r", [xr - g.wall, g.c, g.f], [xr, yb, g.f + g.H])
        box("back", [xl, g.c + di, g.f], [xr, yb, g.f + g.H])
        self.slots = {}
        for level in ("interior", "top"):
            for (i, j) in cells(g, level):
                b = wb.add_body(); b.name = "o_%s_%d_%d" % (level[0], i, j); b.mocap = True; b.pos = FAR
                c = b.add_geom()
                if g.box:
                    c.type = mujoco.mjtGeom.mjGEOM_BOX; c.size = [g.r, g.r, height(g, level, j) / 2]
                else:
                    c.type = mujoco.mjtGeom.mjGEOM_CYLINDER; c.size = [g.r, height(g, level, j) / 2, 0]
                c.contype = 1; c.conaffinity = 0; c.rgba = [0.75, 0.15, 0.15, 1]
                self.slots[(level, i, j)] = b.name
        self.heights = sorted({round(height(g, lv, j), 4) for lv in ("interior", "top") for j in range(g.nrow)})
        for k, hh in enumerate(self.heights):
            hb = wb.add_body(); hb.name = "held%d" % k; hb.mocap = True; hb.pos = FAR
            hc = hb.add_geom()
            if g.box:
                hc.type = mujoco.mjtGeom.mjGEOM_BOX; hc.size = [g.r, g.r, hh / 2]
            else:
                hc.type = mujoco.mjtGeom.mjGEOM_CYLINDER; hc.size = [g.r, hh / 2, 0]
            hc.contype = 0; hc.conaffinity = 1; hc.rgba = [0.15, 0.3, 0.85, 1]; hc.name = "held%d" % k
        # link0 (bolted base) is contact-free in queries; require static clearance of the shelf footprint instead
        dx = max(xl - 0.0, 0.0, 0.0 - xr); dy = max(g.c, 0.0)
        self.base_clear = float(np.hypot(dx, dy)) if not (xl <= 0 <= xr) else float(g.c)
        self.m = spec.compile(); self.d = mujoco.MjData(self.m)
        m = self.m
        patch_collisions(m, V2Params())
        arm_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in V.ARM_BODIES if b != "link0"}
        self.arm = [k for k in range(m.ngeom) if m.geom_bodyid[k] in arm_bodies and m.geom_conaffinity[k]]
        self.held_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "held%d" % k) for k in range(len(self.heights))]
        self.armset = set(self.arm) | set(self.held_geoms)
        m.geom_margin[self.arm] = DETECT; m.geom_margin[self.held_geoms] = DETECT
        self.self_exclude = set()
        if g.selfcol:
            m.geom_contype[self.arm] = 2; m.geom_conaffinity[self.arm] = 3      # arm-arm (bit 2) + arm-scene (bit 1)
            # the broad phase filters on BODY flags fixed at compile time: update them too
            for b in {int(m.geom_bodyid[k]) for k in self.arm}:
                m.body_contype[b] |= 2; m.body_conaffinity[b] |= 3
        # static robot base (link0) vs the shelf: link0 is otherwise contact-free in queries
        l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        l0g = [k for k in range(m.ngeom) if m.geom_bodyid[k] == l0 and m.geom_type[k] == mujoco.mjtGeom.mjGEOM_MESH
               and m.geom_group[k] == 3]
        saved = (m.geom_contype[l0g].copy(), m.geom_conaffinity[l0g].copy(), m.geom_margin[l0g].copy(),
                 int(m.body_contype[l0]), int(m.body_conaffinity[l0]))
        m.geom_contype[l0g] = 0; m.geom_conaffinity[l0g] = 1; m.geom_margin[l0g] = 0.01
        m.body_conaffinity[l0] |= 1
        self.mid = {k: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)] for k, n in self.slots.items()}
        self.held_mids = [m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "held%d" % k)]
                          for k in range(len(self.heights))]
        self.held_mid = self.held_mids[0]
        self.jq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)] for i in range(1, 8)]
        self.fq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in ("finger_joint1", "finger_joint2")]
        mujoco.mj_kinematics(m, self.d)
        self.base = np.eye(4); self.base[:3, :3] = self.d.xmat[l0].reshape(3, 3); self.base[:3, 3] = self.d.xpos[l0]
        self.d.qpos[:] = 0; self.d.qpos[self.jq] = P.Q_HOME; mujoco.mj_kinematics(m, self.d); mujoco.mj_collision(m, self.d)
        # link0 hangs off the world body, so MuJoCo's parent-child filter hides its contacts with the (world) shelf:
        # test its collision-mesh vertices against the shelf boxes geometrically (signed distance, < 0 = inside)
        best = 1.0
        for k in l0g:
            mid = m.geom_dataid[k]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = v @ self.d.geom_xmat[k].reshape(3, 3).T + self.d.geom_xpos[k]
            for _, lo, hi in self.boxes:
                out = np.maximum(np.maximum(lo - w, 0), w - hi); od = np.linalg.norm(out, axis=1)
                ins = np.minimum(w - lo, hi - w).min(1); sd = np.where(od > 0, od, -ins)
                best = min(best, float(sd.min()))
        self.base_clear_static = best
        m.geom_contype[l0g], m.geom_conaffinity[l0g], m.geom_margin[l0g] = saved[:3]
        m.body_contype[l0], m.body_conaffinity[l0] = saved[3], saved[4]
        if g.selfcol:     # arm-arm pairs already within the margin at home are structural neighbours: exclude them
            armset = set(self.arm)
            for cc in self.d.contact[:self.d.ncon]:
                if cc.geom1 in armset and cc.geom2 in armset:
                    self.self_exclude.add((min(cc.geom1, cc.geom2), max(cc.geom1, cc.geom2)))
        self.held_local = None

    # ---- scene state
    def occupy(self, occupied):
        for k, mid in self.mid.items():
            self.d.mocap_pos[mid] = FAR
        for k in occupied:
            c = centre(self.g, *k); self.d.mocap_pos[self.mid[k]] = c + np.array([0, 0, height(self.g, k[0], k[2]) / 2])

    def clearance(self, q, finger, held_pose=None):
        m, d = self.m, self.d
        d.qpos[self.jq] = q; d.qpos[self.fq] = finger
        for mid in self.held_mids:
            d.mocap_pos[mid] = FAR
        mujoco.mj_kinematics(m, d)
        if held_pose is not None:
            local, hk = held_pose
            site = self.site_T(q)
            d.mocap_pos[self.held_mids[hk]] = site[:3, 3] + site[:3, :3] @ local
            mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        best, pair = DETECT, None
        for k in range(d.ncon):
            cc = d.contact[k]
            a1, a2 = cc.geom1 in self.armset, cc.geom2 in self.armset
            if a1 != a2 and cc.dist < best:
                best, pair = cc.dist, (cc.geom1, cc.geom2) if a1 else (cc.geom2, cc.geom1)
            elif a1 and a2 and self.g.selfcol and cc.dist < best:
                kk = (min(cc.geom1, cc.geom2), max(cc.geom1, cc.geom2))
                if kk not in self.self_exclude and cc.geom1 not in self.held_geoms and cc.geom2 not in self.held_geoms:
                    best, pair = cc.dist, (cc.geom1, cc.geom2)
        return best, pair

    def site_T(self, q):
        F = P.fk(q, self.base)
        T = F.hand.copy(); T[:3, 3] = F.hand[:3, 3] + V.ATTACH_OFFSET * F.hand[:3, 2]
        return T

    def name(self, gid):
        b = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[gid])
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gid) or b


def ik_all(sc, T_site, q7_grid):
    T = T_site.copy(); T[:3, 3] = T[:3, 3] + TCP_FROM_SITE * T[:3, 2]
    Tb = np.linalg.inv(sc.base) @ T
    out = []
    for q7 in q7_grid:
        for s in frantik.ik(Tb, float(q7), P.Q_HOME):
            if np.isfinite(s).all() and (s >= P.Q_MIN).all() and (s <= P.Q_MAX).all():
                out.append(s)
    return np.array(out) if out else np.zeros((0, 7))


class CellPaths:
    """IK layers for every (grasp height, hand flip, path kind) of one cell -- obstacle independent, cached."""

    def __init__(self, sc, level, i, j, q7_grid):
        g = sc.g; self.level = level; self.layers = {}
        c = centre(g, level, i, j); ht = height(g, level, j)
        hk = sc.heights.index(round(ht, 4))
        for depth in g.depths:
            gz = c[2] + ht - depth
            site_g = np.array([c[0], c[1] - GCO, gz])
            # held object pose in the site frame at the grasp (cylinder centre)
            for qi, quat in enumerate(V.FRONT_QUATS):
                R = V.quat_to_R(quat)
                for kind in ("open", "held"):
                    lift = 0.004 if kind == "held" else 0.0
                    g_pt = site_g + np.array([0, 0, lift])
                    if level == "interior":
                        y0 = g.c - 0.06
                        ys = np.arange(y0, g_pt[1], g.step).tolist() + [g_pt[1]]
                        pts = [np.array([g_pt[0], y, g_pt[2]]) for y in ys]
                    else:
                        # the hand arrives over the rows in front (all but the designed row j-1 may be occupied)
                        front = [height(g, "top", jj) for jj in range(max(j - 1, 0))]
                        top_of_all = g.s + (max(front) if front else 0.0) + g.travel
                        z0 = top_of_all + (ht - depth if kind == "held" else 0.06)   # wrist hangs ~5 cm below the site
                        z0 = max(z0, g_pt[2])
                        variants = {}
                        zs = np.arange(z0, g_pt[2], -g.step).tolist() + [g_pt[2]]
                        variants["v"] = [np.array([g_pt[0], g_pt[1], z]) for z in zs]          # vertical descent
                        if j >= 2:   # diagonal: start above row j-2 (the nearest row that may be occupied)
                            s0 = np.array([g_pt[0], g_pt[1] - 2 * g.pyt, z0])
                            n = max(2, int(np.ceil(np.linalg.norm(g_pt - s0) / g.step)) + 1)
                            variants["d"] = [s0 + (g_pt - s0) * t for t in np.linspace(0, 1, n)]
                    if level == "interior":
                        variants = {"h": pts}
                    for var, pts in variants.items():
                      Ls = []
                      for pt in pts:
                        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pt
                        Ls.append(ik_all(sc, T, q7_grid))
                      if level == "interior":
                        # the entry may start at any reachable point in front of the shelf with the fingertips
                        # (2.2 cm ahead of the site) still outside: drop leading unreachable layers up to y = c - 0.03
                        k = 0
                        while k < len(Ls) - 1 and len(Ls[k]) == 0 and pts[k + 1][1] <= g.c - 0.03:
                            k += 1
                        Ls = Ls[k:]
                      held_local = (R.T @ (np.array([c[0], c[1], c[2] + ht / 2 + lift]) - g_pt), hk)
                      self.layers[(depth, qi, kind, var)] = (Ls, held_local)


def best_path(sc, Ls, finger, held_local, margin, max_dq):
    """Bottleneck DP. Returns (score, final_nodes_mask, final_Q, binding_pair)."""
    val = None; prev = None; bind = None
    for k, Q in enumerate(Ls):
        if len(Q) == 0:
            return -1.0, None, None, ("no_ik", k)
        res = [sc.clearance(q, finger, held_local) for q in Q]
        cl = np.array([r[0] for r in res])
        if val is None:
            val = cl
        else:
            jump = np.abs(Q[:, None, :] - prev[None, :, :]).max(2)
            cand = np.where(jump <= max_dq, val[None, :], -np.inf).max(1)
            val = np.minimum(cand, cl)
        prev = Q
        if val.max() < margin:
            i = int(np.argmax(cl)); pr = res[i][1]
            return float(val.max()), None, None, (sc.name(pr[0]), sc.name(pr[1]), k) if pr else ("cont", k)
    return float(val.max()), val >= margin, prev, None


def cell_ok(sc, cp, occupied_open, occupied_held):
    """(ok, info): some grasp has a feasible open path AND a feasible held path (any route variant each) whose grasp
    configurations coincide (|dq| <= 0.1)."""
    g = sc.g; info = []
    for depth in g.depths:
        for qi in range(len(V.FRONT_QUATS)):
            finals = {"open": [], "held": []}; best = {"open": None, "held": None}
            for kind, occ, finger in (("open", occupied_open, g.fo), ("held", occupied_held, g.r)):
                for key, (Ls, hl) in cp.layers.items():
                    if key[:3] != (depth, qi, kind):
                        continue
                    sc.occupy(occ)
                    sv, mv, Qv, bv = best_path(sc, Ls, finger, hl if kind == "held" else None, g.margin, g.max_dq)
                    if mv is not None:
                        finals[kind].append((Qv[mv], sv, key[3]))
                    elif best[kind] is None or sv > best[kind][0]:
                        best[kind] = (round(sv, 4), bv, key[3])
                if not finals[kind]:
                    info.append((kind, depth, qi) + (best[kind] or ()))
                    break
            if not finals["open"] or not finals["held"]:
                continue
            for A, sa, va in finals["open"]:
                for B, sb, vb in finals["held"]:
                    if (np.abs(A[:, None, :] - B[None, :, :]).max(2) <= 0.1).any():
                        return True, {"depth": depth, "quat": qi, "open_clear": round(sa, 4), "held_clear": round(sb, 4),
                                      "routes": va + vb}
            info.append(("mismatch", depth, qi))
    return False, info


def designed_blockers(g, level, i, j):
    out = {(level, i - 1, j), (level, i + 1, j)}
    if level == "interior" or g.top_corridor:
        for jj in range(j):
            for dx in range(-g.corridor, g.corridor + 1):
                out.add((level, i + dx, jj))
    else:
        for dx in (-1, 0, 1):
            out.add((level, i + dx, j - 1))
    return out


def main():
    out = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    only = kv.pop("cells", None); render_dir = kv.pop("render", None)
    g = Geo()
    for k, v in kv.items():
        cur = getattr(g, k)
        setattr(g, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    sc = Scene(g); t0 = time.time()
    q7_grid = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(g.q7_step_deg))
    allslots = set(sc.slots)
    todo = [(lv, i, j) for lv in ("interior", "top") for (i, j) in cells(g, lv)]
    if only:
        want = set()
        for s in only.split(";"):
            lv, ij = s.split(":"); i, j = map(int, ij.split("_")); want.add(({"i": "interior", "t": "top"}.get(lv[0]), i, j))
        todo = [k for k in todo if k in want]
    res = {"geo": asdict(g), "cells": {}}
    for key in todo:
        lv, i, j = key
        cp = CellPaths(sc, lv, i, j, q7_grid)
        row = {}
        row["empty"] = cell_ok(sc, cp, [key], [])
        crowd = allslots - designed_blockers(g, lv, i, j) - {key}
        row["crowded"] = cell_ok(sc, cp, crowd | {key}, crowd)
        lat = {}
        for dx in (-1, 1):
            nb = (lv, i + dx, j)
            if nb in allslots:
                ok, _ = cell_ok(sc, cp, [key, nb], [nb])
                lat[dx] = not ok
        row["lateral_blocked"] = lat
        res["cells"]["%s_%d_%d" % key] = row
        print("%-14s empty=%-5s crowded=%-5s lateral_blocked=%s   %s | %s   (%.0fs)" % (
            "%s_%d_%d" % key, row["empty"][0], row["crowded"][0], lat,
            row["empty"][1] if row["empty"][0] else str(row["empty"][1])[:160],
            row["crowded"][1] if row["crowded"][0] else str(row["crowded"][1])[:200], time.time() - t0), flush=True)
    C = res["cells"]
    for lv in ("interior", "top"):
        cs = {k: v for k, v in C.items() if k.startswith(lv)}
        if not cs:
            continue
        lat = [b for v in cs.values() for b in v["lateral_blocked"].values()]
        print("SUMMARY %-8s n=%d empty %d crowded %d lateral_blocked %d/%d" % (
            lv, len(cs), sum(v["empty"][0] for v in cs.values()), sum(v["crowded"][0] for v in cs.values()),
            sum(lat), len(lat)))
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
