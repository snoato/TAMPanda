"""Access-19 v2 grasp paths: the straight approach/retreat of the one grasp, solved over ALL IK branches.

Per cell there are two paths, both ending in the same grasp configuration:

* ``open`` -- open hand moving onto the object while it stands in its cell (pick approach, put retreat);
* ``held`` -- the same grasp holding the object, lifted 4 mm (pick retreat, put approach).

Interior cells: a horizontal line from 6 cm in front of the shelf to the grasp (route ``h``; leading points that
no IK reaches are dropped while the fingertips are still outside the shelf). Top cells: a vertical descent from
above every box on the deck (``v``) and, from row 2 on, a diagonal from above row ``j - 2`` (``d``).
Cartesian layers every ``g.step``; at each layer every IK solution (joint-7 grid x 4 analytic branches). A
bottleneck dynamic programme links layers with joint jumps <= ``g.max_dq`` and keeps, for every reachable final
(grasp) configuration, the path whose smallest layer clearance is largest (>= ``g.margin``).

Layers are 1.5 cm apart, so a path is only accepted after a DENSE check (``profile``): every joint-space sample at
``FINE`` rad (max-norm) is tested against the exact meshes. ``profile`` reports the clearance to the static scene
(shelf, base, arm self-collision) and, separately, to each present object -- the per-object part is what makes
exact conflict sets possible (see ``library.py``).
"""
from __future__ import annotations

import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import kinematics as K
from tampanda.symbolic.domains.access19_v2.shelf import DETECT, centre

FINE = 0.005       # joint-space sampling of dense checks (rad): at 0.02 a fingertip moves up to ~16 mm per sample
MARGIN = 0.002     # dense clearance required along every accepted path (shelf, self, objects)
MATCH = 0.1        # open and held paths belong to the same grasp if their final configurations agree this well


def _T(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


class CellPaths:
    """IK layers of the open and held paths of one cell (obstacle independent).
    ``layers[(kind, route)] = (list of (n_k x 7) solution arrays, held_local)``; ``held_local`` is the held
    object's centre in the site frame."""

    def __init__(self, sc, key, q7s):
        g = sc.g; lv, i, j = key; c = centre(g, key); R = K.grasp_R(g)
        site_g = np.array([c[0], c[1] - K.GRASP_CONTACT_OFFSET, c[2] + g.h - g.grasp_depth])
        self.key = key; self.layers = {}
        for kind in ("open", "held"):
            lift = K.HELD_LIFT if kind == "held" else 0.0
            gp = site_g + np.array([0, 0, lift])
            if lv == "interior":
                ys = list(np.arange(g.c - 0.06, gp[1], g.step)) + [gp[1]]
                routes = {"h": [np.array([gp[0], y, gp[2]]) for y in ys]}
            else:
                above = g.s + g.h + g.travel      # start over the top of every box, so transits never meet one
                z0 = max(above + ((g.h - g.grasp_depth) if kind == "held" else 0.06), gp[2])
                zs = list(np.arange(z0, gp[2], -g.step)) + [gp[2]]
                routes = {"v": [np.array([gp[0], gp[1], z]) for z in zs]}
                if j >= 2:
                    s0 = np.array([gp[0], gp[1] - 2 * g.pyt, z0])
                    n = max(2, int(np.ceil(np.linalg.norm(gp - s0) / g.step)) + 1)
                    routes["d"] = [s0 + (gp - s0) * t for t in np.linspace(0, 1, n)]
            held_local = R.T @ (np.array([c[0], c[1], c[2] + g.h / 2 + lift]) - gp)
            for route, pts in routes.items():
                Ls = [K.ik_all(sc.base, _T(R, p), q7s) for p in pts]
                if lv == "interior":
                    k = 0
                    while k < len(Ls) - 1 and len(Ls[k]) == 0 and pts[k + 1][1] <= g.c - 0.03:
                        k += 1
                    Ls = Ls[k:]
                self.layers[(kind, route)] = (Ls, held_local)


def dp_paths(sc, Ls, finger, held_local, g):
    """Bottleneck DP over the layers against the CURRENT scene occupancy.
    Returns [(path as list of configurations, bottleneck clearance)] for every final configuration >= margin."""
    val = None; prev = None; back = []
    for Q in Ls:
        if len(Q) == 0:
            return []
        cl = np.array([sc.clearance(q, finger, held_local) for q in Q])
        if val is None:
            val = cl
        else:
            jump = np.abs(Q[:, None, :] - prev[None, :, :]).max(2)
            cand = np.where(jump <= g.max_dq, val[None, :], -np.inf)
            jj = cand.argmax(1); val = np.minimum(cand[np.arange(len(jj)), jj], cl); back.append(jj)
        prev = Q
    out = []
    for i in np.where(val >= g.margin)[0]:
        idx = [int(i)]
        for jj in reversed(back):
            idx.append(int(jj[idx[-1]]))
        idx = idx[::-1]
        out.append((np.array([Ls[k][idx[k]] for k in range(len(Ls))]), float(val[i])))
    return out


def dense(Q, step=FINE):
    """Joint-space samples of a piecewise-linear path, at most ``step`` apart (max-norm), endpoints included."""
    Q = np.asarray(Q, float)
    if len(Q) == 1:
        return Q
    out = [Q[0]]
    for a, b in zip(Q[:-1], Q[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        out.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.array(out)


def profile(sc, Q, finger, held_local, step=FINE, margin=MARGIN):
    """Dense exact check of a joint path against the current occupancy.
    Returns (static_ok, static_min, {cell: min distance}) -- static = shelf, base, self-collision; stops early
    (static_ok False) as soon as the static clearance drops below ``margin``."""
    static = DETECT; per = {}
    for q in dense(Q, step):
        s, po = sc.contacts(q, finger, held_local)
        if s < static:
            static = s
            if static < margin:
                return False, static, per
        for k, v in po.items():
            if v < per.get(k, DETECT):
                per[k] = v
    return True, static, per


def match_pairs(opens, helds):
    """(open, held) index pairs whose final (grasp) configurations coincide."""
    if not opens or not helds:
        return []
    A = np.array([o["Q"][-1] for o in opens]); B = np.array([h["Q"][-1] for h in helds])
    return list(zip(*np.nonzero(np.abs(A[:, None, :] - B[None, :, :]).max(2) <= MATCH)))


def posture_cost(q_open_start, q_held_start, ref) -> float:
    """Distance of a grasp's two outer configurations to the reference posture (smaller = preferred)."""
    return float(np.linalg.norm(np.asarray(q_open_start) - ref) + np.linalg.norm(np.asarray(q_held_start) - ref))


def fit_reference(per_cell, iters=8):
    """One reference posture for a geometry, so that consecutive grasps use the same arm configuration family.

    ``per_cell``: {cell: (n x 14) array of [open start | held start] of its statically valid grasps}. Iterate
    REF <- coordinate-wise median of the per-cell choices (start: home), choice <- argmin posture_cost; keep the
    REF with the smallest mean interior<->top distance between the chosen held starts. (The first-found IK
    solution is arbitrary: branch flips between neighbouring cells put consecutive grasps 3-7 rad apart.)"""
    ref = P.Q_HOME.copy(); best = None
    for _ in range(iters):
        ch = {k: v[np.argmin(np.linalg.norm(v[:, :7] - ref, axis=1) + np.linalg.norm(v[:, 7:] - ref, axis=1))]
              for k, v in per_cell.items() if len(v)}
        I = np.array([v[7:] for k, v in ch.items() if k[0] == "interior"])
        T = np.array([v[7:] for k, v in ch.items() if k[0] == "top"])
        score = float(np.linalg.norm(I[:, None, :] - T[None, :, :], axis=2).mean()) if len(I) and len(T) else 0.0
        if best is None or score < best[0]:
            best = (score, ref.copy())
        ref = np.median(np.array([v[7:] for v in ch.values()]), axis=0)
    return best[1], best[0]
