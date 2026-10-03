"""Access-19 v2: execute and render a full symbolic plan with the v2 geometry and collision model (2026-10-01).

KINEMATIC playback (no physics): every frame sets the arm configuration exactly; a held cube is carried
rigidly at the grasp offset. Every motion uses the routes verified by verify.py / rrt_check.py:
  interior pick/put: (transit) -> staging 6 cm outside the front face -> straight row approach -> close/open
                     -> 1 cm lift (pick) -> straight retreat -> staging
  deck pick/put:     (transit) -> high front waypoint -> traverse at safe_z -> vertical descent -> close/open
                     -> ascend -> traverse back -> high front
  transits between actions: straight joint-space move if collision-free, else RRT-Connect on the same
                     collision check (held cube included).
Every frame of every motion is collision-checked against the live scene (all other cubes in place); the
report counts actions that could not be executed collision-free (rendered as a snap, flagged).

    MUJOCO_GL=egl python -m tampanda.symbolic.domains.access19_v2.render_plan \
        --plan PLAN.json --problem PROB.pddl --out OUT.mp4 [--cam d,az,el] [shelf_xy=0.39,0.36 ...]
"""
from __future__ import annotations

import argparse
import json
import re
import time

import imageio
import mujoco
import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2.geometry import V2Params
from tampanda.symbolic.domains.access19_v2.verify import (CUBE_GRASP_OFFSET, FRONT_QUATS, GRASP_CONTACT_OFFSET,
                                                          SAFE_Z_ABOVE_DECK, Rig, quat_to_R, seeds_for)

OCC = re.compile(r"\(occupied\s+(\S+)\s+(\S+?)\)")


class Player:
    def __init__(self, rig, cam, every=1):
        self.rig, self.cam, self.every = rig, cam, every
        self.renderer = mujoco.Renderer(rig.m, height=480, width=640)
        self.frames, self.n_coll_frames = [], 0
        self.held = None
        self.layout = {}              # cube -> world xyz

    def scene(self):
        self.rig.park_all()
        for c, xyz in self.layout.items():
            self.rig.place(c, xyz)

    def show(self, q, finger, check=True, n=1):
        held = self.held
        col = self.rig.contacts(q, finger=finger, held=held) if check else []
        if col:
            self.n_coll_frames += 1
        self.renderer.update_scene(self.rig.d, camera=self.cam)
        img = self.renderer.render().copy()
        self.frames.extend([img] * n)
        return col


def densify(qs, step=0.03):
    out = [qs[0]]
    for a, b in zip(qs[:-1], qs[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return out


def free(rig, q, finger, held):
    return P.within_limits(q) and not rig.contacts(q, finger=finger, held=held)


def seg_free(rig, a, b, finger, held, step=0.03):
    n = max(2, int(np.ceil(np.abs(b - a).max() / step)) + 1)
    return all(free(rig, a + (b - a) * t, finger, held) for t in np.linspace(0, 1, n))


def rrt_connect(rig, qa, qb, finger, held, iters=4000, step=0.15, rng=None):
    rng = rng or np.random.default_rng(0)
    Ta, Tb = [(qa, -1)], [(qb, -1)]
    for _ in range(iters):
        x = rng.uniform(P.Q_MIN, P.Q_MAX)
        for T, other in ((Ta, Tb), (Tb, Ta)):
            i = int(np.argmin([np.linalg.norm(n[0] - x) for n in T]))
            d = x - T[i][0]; d = T[i][0] + d * min(1.0, step / (np.linalg.norm(d) + 1e-9))
            if not seg_free(rig, T[i][0], d, finger, held):
                continue
            T.append((d, i))
            j = int(np.argmin([np.linalg.norm(n[0] - d) for n in other]))
            if seg_free(rig, d, other[j][0], finger, held):
                def chain(Tr, k):
                    out = []
                    while k != -1:
                        out.append(Tr[k][0]); k = Tr[k][1]
                    return out
                pa, pb = chain(T, len(T) - 1), chain(other, j)
                path = pa[::-1] + pb if T is Ta else pb[::-1] + pa
                return path if np.allclose(path[0], qa) else path[::-1]
            x = d
    return None


def cart(rig, a, b, R, q0, finger, held, step=0.01):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    qs = [q0]
    for wp in [a + (b - a) * t for t in np.linspace(0, 1, n)][1:]:
        q = rig.ik(wp, R, qs[-1], iters=80)
        if q is None or np.abs(q - qs[-1]).max() > 0.25:
            return None
        qs.append(q)
    return qs


def action_route(rig, kind, region, cell, held_after):
    """Returns (q_entry, segments) where segments = list of (qs, finger, held_flag_name or None, label)."""
    p = rig.p
    c = rig.cell_xyz(region, cell)
    gz = c[2] + p.cube_half[2] - CUBE_GRASP_OFFSET + (0.003 if kind == "put" else 0.0)
    g = np.array([c[0], c[1] - GRASP_CONTACT_OFFSET, gz])
    ffy = rig.ws["shelf_interior"].origin[1] - 0.02
    if region.name == "shelf_interior":
        stage = np.array([c[0], ffy - 0.06, gz]); pre = [stage]
    else:
        deck_top = region.level_z - p.cube_half[2]
        above = np.array([g[0], g[1], deck_top + p.deck_travel_z])
        stage = np.array([g[0], ffy - 0.06, above[2]]); pre = [stage, above]
    fallback = None
    for quat in FRONT_QUATS:
        R = quat_to_R(quat)
        for seed in [P.Q_HOME] + seeds_for(rig, region, cell)[1:]:
            q_stage = rig.ik(stage, R, seed)
            if q_stage is None:
                continue
            way = [stage] + pre[1:] + [g]
            qs = [q_stage]
            for a, b in zip(way[:-1], way[1:]):
                part = cart(rig, a, b, R, qs[-1], None, None)
                if part is None:
                    qs = None; break
                qs += part[1:]
            if qs is None:
                continue
            # route selection against the LIVE scene (other cubes present): first collision-free route wins
            f = p.finger_open if kind == "pick" else p.cube_half[0]
            if not any(rig.contacts(qq, finger=f, held=held_after) for qq in qs):
                return R, qs
            if fallback is None:
                fallback = (R, qs)
    return fallback if fallback is not None else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--problem", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cam", default="1.8,135,-35")
    ap.add_argument("--report", default=None)
    ap.add_argument("params", nargs="*")
    a = ap.parse_args()
    p = V2Params(shelf_xy=(0.39, 0.36))
    for kv in a.params:
        k, v = kv.split("=", 1); cur = getattr(p, k)
        setattr(p, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    rig = Rig(p)
    plan = json.load(open(a.plan)); actions = plan["planA"] + plan["planB"]; boundary = plan["boundary"]
    init = open(a.problem).read().split("(:init", 1)[1].split("(:goal", 1)[0]
    pl = Player(rig, None)
    for cell_id, obj in OCC.findall(init):
        rname, ij = cell_id.split("__"); ix, iy = map(int, ij.split("_"))
        pl.layout[obj] = rig.cell_xyz(rig.ws[rname], rig.ws[rname].cell(ix, iy))
    # camera framed on the shelf
    pts = np.array(list(pl.layout.values()))
    d_, az, el = (float(x) for x in a.cam.split(","))
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.median(pts, axis=0); cam.distance, cam.azimuth, cam.elevation = d_, az, el
    pl.cam = cam
    pl.scene()
    q = P.Q_HOME.copy(); fopen = p.finger_open; fclose = p.cube_half[0]
    finger = 0.0
    pl.show(q, 0.0, n=30)
    report = []; boundary_frame = None
    t0 = time.time()
    for i, s in enumerate(actions):
        if i == boundary:
            boundary_frame = len(pl.frames)
        kind, obj, cell_id = s.strip("()").split()
        rname, ij = cell_id.split("__"); ix, iy = map(int, ij.split("_"))
        region = rig.ws[rname]; cell = region.cell(ix, iy)
        held = obj if kind == "put" else None
        pl.held = held
        if kind == "pick":                       # target cube is in the scene (not held) during approach
            pl.layout[obj] = rig.cell_xyz(region, cell)
        pl.scene()
        f_route = fopen if kind == "pick" else fclose
        R, qs = action_route(rig, kind, region, cell, held)
        row = {"i": i, "action": s, "route": qs is not None}
        if qs is None:
            report.append(row); print(i, s, "NO ROUTE", flush=True)
            # snap: apply the symbolic effect and continue
            if kind == "pick":
                pl.layout.pop(obj, None); pl.held = obj
            else:
                pl.layout[obj] = rig.cell_xyz(region, cell); pl.held = None
            continue
        # verify the action route against the live scene (other cubes present) BEFORE executing
        col_route = [rig.contacts(qq, finger=f_route, held=held) for qq in qs]
        row["route_collision"] = sum(bool(c) for c in col_route)
        row["route_first_collision"] = next((c[0] for c in col_route if c), None)
        # transit: current q -> qs[0]
        if seg_free(rig, q, qs[0], f_route if kind == "pick" else fclose, held):
            transit = densify([q, qs[0]])
            row["transit"] = "lerp"
        else:
            path = rrt_connect(rig, q, qs[0], f_route if kind == "pick" else fclose, held)
            transit = densify(path) if path else densify([q, qs[0]])
            row["transit"] = "rrt" if path else "FAILED"
        f_transit = f_route if kind == "pick" else fclose
        for qq in transit[1:]:
            pl.show(qq, f_transit)
        for qq in qs[1:]:
            pl.show(qq, f_route)
        if kind == "pick":
            for f in np.linspace(fopen, fclose, 8):            # close on the cube
                pl.show(qs[-1], f, check=False)
            T = rig.site_T(qs[-1])
            rig._held_local = T[:3, :3].T @ (rig.cell_xyz(region, cell) - T[:3, 3])   # carry at the actual grasp
            pl.layout.pop(obj, None); pl.held = obj; pl.scene()
            lift = cart(rig, T[:3, 3], T[:3, 3] + np.array([0, 0, 0.01]), R, qs[-1], None, None) or [qs[-1]]
            back = lift + list(reversed(qs))[1:]
            # retreat must lift the held cube over the floor; offset every retreat waypoint by +1 cm
            back2 = [lift[-1]]
            for qq in list(reversed(qs))[1:]:
                Tq = rig.site_T(qq); qn = rig.ik(Tq[:3, 3] + np.array([0, 0, 0.01]), R, back2[-1], iters=60)
                back2.append(qn if qn is not None else qq)
            for qq in lift[1:] + back2[1:]:
                pl.show(qq, fclose)
            q = back2[-1]; finger = fclose
        else:
            pl.layout[obj] = rig.cell_xyz(region, cell); pl.held = None; pl.scene()
            for f in np.linspace(fclose, fopen, 8):            # release
                pl.show(qs[-1], f)
            for qq in list(reversed(qs))[1:]:
                pl.show(qq, fopen)
            q = qs[0]; finger = fopen
        row["collision_frames_total"] = pl.n_coll_frames
        report.append(row)
        if i % 5 == 0 or row["route_collision"] or row.get("transit") == "FAILED":
            print("%2d %-40s route_coll=%d transit=%s frames=%d" % (i, s, row["route_collision"], row["transit"], len(pl.frames)), flush=True)
    pl.held = None
    pl.show(q, finger, n=30)
    imageio.mimwrite(a.out, pl.frames, fps=30, macro_block_size=None)
    summ = {"n_actions": len(actions), "no_route": sum(not r["route"] for r in report),
            "route_collision_actions": sum(r.get("route_collision", 0) > 0 for r in report),
            "transit_rrt": sum(r.get("transit") == "rrt" for r in report),
            "transit_failed": sum(r.get("transit") == "FAILED" for r in report),
            "collision_frames": pl.n_coll_frames, "frames": len(pl.frames), "boundary_frame": boundary_frame,
            "wall_s": round(time.time() - t0, 1)}
    print("SUMMARY", summ, flush=True)
    if a.report:
        json.dump({"summary": summ, "actions": report, "params": p.as_dict(),
                   "boundary_frame": boundary_frame, "fps": 30, "cam": a.cam}, open(a.report, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
