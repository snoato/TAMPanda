"""Per-cell status map for an Access-19 v2 geometry (2026-10-02).

Reads the crowd verification JSONs of a geometry and renders every cell's cylinder coloured by status:
  green  = pick + put feasible with the shelf CROWDED (only the cell's designed blockers removed) and the same-row
           neighbours block it,
  yellow = feasible with the shelf EMPTY but not crowded,
  red    = not feasible even on an empty shelf,
  purple = feasible but a same-row neighbour does NOT block it.
Writes <out>/status_3d.png (front-oblique + top views, robot at home) and <out>/status_grid.png (flat grids).

    MUJOCO_GL=egl python -m tampanda.symbolic.domains.access19_v2.status_map OUT GEOFILE crowd_*.json
"""
import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import crowd as C

COL = {"green": (0.15, 0.75, 0.25, 1), "yellow": (0.95, 0.8, 0.1, 1), "red": (0.85, 0.1, 0.1, 1),
       "purple": (0.6, 0.2, 0.8, 1)}


def font(sz):
    for f in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def status(row):
    if not row["empty"][0]:
        return "red"
    if not row["crowded"][0]:
        return "yellow"
    if not all(row["lateral_blocked"].values()):
        return "purple"
    return "green"


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    kv = dict(a.split("=", 1) for a in open(sys.argv[2]).read().split())
    g = C.Geo()
    for k, v in kv.items():
        cur = getattr(g, k); setattr(g, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    cells = {}
    for pat in sys.argv[3:]:
        for f in glob.glob(pat):
            cells.update(json.load(open(f))["cells"])
    sc = C.Scene(g); m, d = sc.m, sc.d
    st = {}
    for key, name in sc.slots.items():
        k = "%s_%d_%d" % key
        s = status(cells[k]) if k in cells else None
        st[key] = s
        gid = [gg for gg in range(m.ngeom) if m.geom_bodyid[gg] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)][0]
        m.geom_rgba[gid] = COL[s] if s else (0.5, 0.5, 0.5, 1)
    for nm in ("wall_l", "wall_r", "fill_l", "fill_r"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if gid >= 0:
            m.geom_rgba[gid, 3] = 0.25
    sc.occupy(set(sc.slots))
    d.qpos[sc.jq] = P.Q_HOME; d.qpos[sc.fq] = g.fo; mujoco.mj_forward(m, d)
    W, H = 800, 600
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    r = mujoco.Renderer(m, H, W)
    cx = float(np.mean([C.centre(g, *k)[0] for k in sc.slots])); cy = float(np.mean([C.centre(g, *k)[1] for k in sc.slots]))
    tiles = []
    for name, (az, el, dist) in (("front-oblique", (60, -22, 1.9)), ("from the robot", (90, -10, 1.7)), ("top", (90, -80, 1.6))):
        cam = mujoco.MjvCamera(); cam.lookat[:] = [cx, cy, g.f + 0.2]; cam.distance = dist; cam.azimuth = az; cam.elevation = el
        r.update_scene(d, cam); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 30], fill=(20, 20, 28)); dr.text((8, 5), name, fill="white", font=font(18)); tiles.append(im)
    n = {lv: {c: sum(1 for k, s in st.items() if k[0] == lv and s == c) for c in COL} for lv in ("interior", "top")}
    sheet = Image.new("RGB", (3 * W, H + 90), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * W, 60))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 8), "per-cell status (every cell occupied, robot at home)  green = reachable CROWDED + neighbours block   "
                     "yellow = only when empty   red = unreachable   purple = neighbour does not block", fill="black", font=font(20))
    dr.text((10, 34), "interior %s | top %s" % (n["interior"], n["top"]), fill=(60, 60, 60), font=font(18))
    dr.text((10, H + 64), " ".join(open(sys.argv[2]).read().split()), fill=(80, 80, 80), font=font(15))
    sheet.save(out / "status_3d.png")
    # flat grids: rows = depth (front at the bottom), columns = lateral index
    cw = 46; grid = Image.new("RGB", (2 * (7 * cw + 60) + 40, 7 * cw + 110), "white"); dg = ImageDraw.Draw(grid)
    for li, lv in enumerate(("interior", "top")):
        ox = 30 + li * (7 * cw + 80); oy = 50
        dg.text((ox, 10), lv + "  (front row at the bottom, robot side)", fill="black", font=font(18))
        for i in range(g.ncol):
            for j in range(g.nrow):
                s = st.get((lv, i, j))
                x0, y0 = ox + i * cw, oy + (g.nrow - 1 - j) * cw
                fill = tuple(int(255 * c) for c in COL[s][:3]) if s else (230, 230, 230)
                dg.rectangle([x0, y0, x0 + cw - 4, y0 + cw - 4], fill=fill, outline=(60, 60, 60))
                dg.text((x0 + 8, y0 + 12), "%d,%d" % (i, j), fill="black", font=font(14))
    grid.save(out / "status_grid.png")
    print("wrote", out / "status_3d.png", out / "status_grid.png", n)


if __name__ == "__main__":
    main()
