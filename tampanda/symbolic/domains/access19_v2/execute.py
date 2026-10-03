"""Execute a symbolic Access-19 v2 plan with the grasp library's motions, checking every configuration.

Each action uses the checker's choice (``GraspLibrary.candidate``: the best-posture motion that is clear in the
current occupancy) and runs it through home:

    pick c : home -> open start -> grasp (open hand) -> close -> retreat holding -> home
    put  c : home -> held start (holding) -> grasp -> open -> retreat (open hand) -> home

Every joint-space sample (``grasp.FINE``) is checked against the exact scene with the objects actually present;
only the gripper's close/open instants ignore the object being grasped or released. A plan the checker accepts must
therefore execute with zero contacts -- this is the end-to-end test of the library semantics. With ``--video`` the
execution is rendered (front-oblique view + cutaway).

    python -m tampanda.symbolic.domains.access19_v2.execute LIB_STEM PLAN.json [--video OUT.mp4] [--geo FILE]
    PLAN.json: {"init": {obj: cell_id}, "plan": [["pick", obj, cell_id], ["put", obj, cell_id], ...]}
    or the shorthand  unload:0,2,4  (interior columns -> top, interior row r -> top row nrow-1-r, front rows first)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tampanda.symbolic.domains.access19_v2 import grasp as G
from tampanda.symbolic.domains.access19_v2.checker import GraspLibrary
from tampanda.symbolic.domains.access19_v2.library import load_paths
from tampanda.symbolic.domains.access19_v2.shelf import FAR, T17C, Scene, ShelfGeometry, cell_id, parse_cell_id


def unload_plan(g, cols):
    init, plan = {}, []
    for c in cols:
        for r in range(g.nrow):
            init["box_%d_%d" % (c, r)] = cell_id(("interior", c, r))
    for r in range(g.nrow):
        for c in cols:
            o = "box_%d_%d" % (c, r)
            plan += [("pick", o, cell_id(("interior", c, r))), ("put", o, cell_id(("top", c, g.nrow - 1 - r)))]
    return init, plan


def execute(sc, lib, paths, init, plan, on_config=None):
    """Run ``plan``; returns a report. ``on_config(q, finger, held_local, label)`` sees every executed sample."""
    g = sc.g; layout = dict(init); held = None; report = {"actions": [], "samples": 0, "contacts": 0,
                                                           "min_clearance": 1.0, "completed": False}

    def run(Q, finger, hl, label, occupied):
        sc.occupy(occupied)
        for q in G.dense(Q):
            cl = sc.clearance(q, finger, hl)
            report["samples"] += 1; report["min_clearance"] = min(report["min_clearance"], cl)
            if cl < 0:
                report["contacts"] += 1
            if on_config:
                on_config(q, finger, hl, label, occupied, layout=layout, held=held)

    for n, (name, obj, cid) in enumerate(plan):
        key = parse_cell_id(cid); label = "%2d/%d  %s %s %s" % (n + 1, len(plan), name, obj, cid)
        others = [c for o, c in layout.items() if o != obj]
        k = lib.candidate(cid, others)
        row = {"action": [name, obj, cid], "candidate": k}
        report["actions"].append(row)
        if k is None:
            row["infeasible_blockers"] = sorted(min(lib.blockers(cid, others), key=len))
            return report
        c = paths[cid][k]; hl = c["held_local"]
        occ = {parse_cell_id(x) for x in others}
        if name == "pick":
            if layout.get(obj) != cid or held is not None:
                raise ValueError("pick precondition violated at step %d" % n)
            run(c["t_open"], g.fo, None, label, occ | {key})
            run(c["open"], g.fo, None, label, occ | {key})
            if on_config:                                   # gripper closes on the object (contact by design)
                for f in np.linspace(g.fo, g.r, 6):
                    on_config(c["open"][-1], f, None, label, occ | {key}, layout=layout, held=None)
            del layout[obj]; held = obj
            run(c["held"][::-1], g.r, hl, label, occ)
            run(c["t_held"], g.r, hl, label, occ)
        else:
            if held != obj or cid in layout.values():
                raise ValueError("put precondition violated at step %d" % n)
            run(c["t_held"][::-1], g.r, hl, label, occ)
            run(c["held"], g.r, hl, label, occ)
            if on_config:
                for f in np.linspace(g.r, g.fo, 6):
                    on_config(c["open"][-1], f, None, label, occ | {key}, layout=layout, held=None)
            layout[obj] = cid; held = None
            run(c["open"][::-1], g.fo, None, label, occ | {key})
            run(c["t_open"][::-1], g.fo, None, label, occ | {key})
    report["completed"] = True
    return report


class Recorder:
    """Renders executed samples (every ``stride``-th) into a two-view video."""

    def __init__(self, sc, title, W=720, H=464, stride=8, dark_header=None):
        import mujoco
        from PIL import ImageFont
        self.mujoco = mujoco; self.sc = sc; self.W, self.H, self.stride = W, H, stride; self.title = title
        m = sc.m; m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
        self.ren = mujoco.Renderer(m, H, W); self.frames = []; self.n = 0; g = sc.g
        look = [g.x0 + (g.ncol - 1) * g.px / 2, g.c + 0.12, 0.5 * (g.f + g.s) + 0.05]
        self.cams = []
        style = getattr(sc, "look", None)
        # teaser look: front-quarter main view (azimuth 125, elevation -22) -- the robot stands IN FRONT of the v2
        # shelf, so the teaser strip's azimuth 100 would put the moving arm between camera and shelf
        views = ((125, -22, 1.8), (0, -10, 1.65)) if style == "teaser" else ((55, -20, 1.75), (0, -10, 1.65))
        for az, el, dist in views:
            cm = mujoco.MjvCamera(); cm.lookat[:] = [look[0], look[1] - 0.1, look[2]]; cm.distance = dist
            cm.azimuth = az; cm.elevation = el; self.cams.append(cm)
        light_bg = style == "teaser" if dark_header is None else not dark_header
        self.header_bg, self.header_fg, self.header_fg2 = (((242, 243, 246), (31, 36, 48), (110, 119, 138)) if light_bg
                                                           else ((18, 18, 26), (255, 255, 255), (200, 200, 200)))
        self.walls = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("wall_l", "wall_r", "panel_l", "panel_r")]
        self.rgba0 = m.geom_rgba.copy()
        try:
            self.font = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", s)
            self.font(12)
        except OSError:
            self.font = lambda s: ImageFont.load_default()

    TARGET = "ooi"

    def __call__(self, q, finger, hl, label, occupied, layout=None, held=None):
        self.n += 1
        if self.n % self.stride:
            return
        from PIL import Image, ImageDraw
        mujoco, sc = self.mujoco, self.sc; m, d = sc.m, sc.d
        sc.occupy(occupied); sc._pose(q, finger, hl); mujoco.mj_forward(m, d)
        base = self.rgba0.copy()
        if layout is not None:
            from tampanda.symbolic.domains.access19_v2.layout import parse_cell_id
            tc = layout.get(self.TARGET)
            if tc is not None:
                base[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, sc.slots[parse_cell_id(tc)])] = [0.15, 0.35, 0.95, 1]
            base[sc.held_geom] = [0.15, 0.35, 0.95, 1] if held == self.TARGET else [0.75, 0.15, 0.15, 1]
        tiles = []
        for ci, cm in enumerate(self.cams):
            m.geom_rgba[:] = base
            if ci == 1:
                for gid in self.walls:
                    if gid >= 0:
                        m.geom_rgba[gid, 3] = 0.15
            self.ren.update_scene(d, cm); tiles.append(Image.fromarray(self.ren.render()))
        m.geom_rgba[:] = self.rgba0
        im = Image.new("RGB", (2 * self.W, self.H + 56), self.header_bg); im.paste(tiles[0], (0, 56)); im.paste(tiles[1], (self.W, 56))
        dr = ImageDraw.Draw(im); dr.text((12, 6), label, fill=self.header_fg, font=self.font(22))
        dr.text((12, 34), self.title, fill=self.header_fg2, font=self.font(15))
        self.frames.append(np.asarray(im))

    def save(self, path, fps=30):
        import imageio
        imageio.mimwrite(path, self.frames, fps=fps, quality=8, macro_block_size=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib_stem"); ap.add_argument("plan")
    ap.add_argument("--video"); ap.add_argument("--geo"); ap.add_argument("--report")
    a = ap.parse_args()
    g = ShelfGeometry.from_kv(open(a.geo).read()) if a.geo else T17C
    lib = GraspLibrary.load(a.lib_stem); assert lib.meta["digest"] == g.digest(), "library built for another geometry"
    paths = load_paths(a.lib_stem); sc = Scene(g)
    if a.plan.startswith("unload:"):
        init, plan = unload_plan(g, [int(x) for x in a.plan.split(":")[1].split(",")])
    else:
        spec = json.load(open(a.plan)); init, plan = spec["init"], [tuple(x) for x in spec["plan"]]
    rec = Recorder(sc, "Access-19 v2 | inner width %.0f cm, cavity %.0f cm | library motions, every sample exact-checked"
                   % (100 * g.inner_width(), 100 * g.H)) if a.video else None
    rep = execute(sc, lib, paths, init, plan, rec)
    if rec:
        rec.save(a.video)
    out = {k: v for k, v in rep.items() if k != "actions"}
    out["actions_done"] = sum(1 for r in rep["actions"] if r["candidate"] is not None)
    out["first_infeasible"] = next((r for r in rep["actions"] if r["candidate"] is None), None)
    print(json.dumps(out))
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
