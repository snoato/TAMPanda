"""Render crowded-shelf grasps for visual verification (2026-10-02).

For each requested cell: build the CROWDED scene, find the best open-hand path and held path with back-pointers,
and render (i) the descent/entry start, (ii) the grasp pose, (iii) the held pose at the path start, from three
cameras, with per-frame clearance printed. Also an overview of the fully crowded shelf.

    MUJOCO_GL=egl python research/access19_v2/crowd_show.py out_dir cells=top:6_6;interior:3_6 [k=v]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tampanda.kinematics import panda as P
import crowd as C
import verify as V

W, H = 720, 540


def font(sz):
    for f in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def dp_path(sc, Ls, finger, held_local, g):
    val = None; prev = None; back = []; cls = []
    for Q in Ls:
        if len(Q) == 0:
            return None
        cl = np.array([sc.clearance(q, finger, held_local)[0] for q in Q]); cls.append(cl)
        if val is None:
            val = cl
        else:
            jump = np.abs(Q[:, None, :] - prev[None, :, :]).max(2)
            cand = np.where(jump <= g.max_dq, val[None, :], -np.inf)
            j = cand.argmax(1); val = np.minimum(cand[np.arange(len(j)), j], cl); back.append(j)
        prev = Q
    i = int(val.argmax())
    if val[i] < g.margin:
        return None
    idx = [i]
    for j in reversed(back):
        idx.append(int(j[idx[-1]]))
    idx = idx[::-1]
    return [Ls[k][idx[k]] for k in range(len(Ls))], [float(cls[k][idx[k]]) for k in range(len(Ls))]


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    cells = kv.pop("cells")
    g = C.Geo()
    for k, v in kv.items():
        cur = getattr(g, k)
        setattr(g, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    sc = C.Scene(g); m, d = sc.m, sc.d
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    r = mujoco.Renderer(m, H, W)
    q7 = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(g.q7_step_deg))
    allslots = set(sc.slots)
    gx = g.x0 + 3 * g.px; gy = g.c + 0.3
    roof_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "roof")
    wall_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("wall_l", "wall_r")]
    rgba0 = m.geom_rgba.copy()
    CAMS = [("front-left", 1.9, -125, -25), ("side", 1.7, 180, -12), ("above", 1.6, -90, -65)]

    def shot(title, lines, q, finger, held_local, see_inside=False):
        d.qpos[sc.jq] = q; d.qpos[sc.fq] = finger
        mujoco.mj_kinematics(m, d)
        for mid in sc.held_mids:
            d.mocap_pos[mid] = C.FAR
        if held_local is not None:
            local, hk = held_local
            T = sc.site_T(q); d.mocap_pos[sc.held_mids[hk]] = T[:3, 3] + T[:3, :3] @ local
        mujoco.mj_forward(m, d)
        m.geom_rgba[:] = rgba0
        if see_inside:
            for gid in wall_gids:
                m.geom_rgba[gid, 3] = 0.25
        tiles = []
        for name, dist, az, el in CAMS:
            cam = mujoco.MjvCamera(); cam.lookat[:] = [gx, gy, g.f + 0.25]; cam.distance = dist
            cam.azimuth = az; cam.elevation = el
            r.update_scene(d, cam); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, W, 30], fill=(20, 20, 28)); dr.text((8, 5), name, fill="white", font=font(18))
            tiles.append(im)
        sheet = Image.new("RGB", (W * 3, H + 40 + 24 * len(lines)), "white")
        for k, t in enumerate(tiles):
            sheet.paste(t, (k * W, 40))
        dr = ImageDraw.Draw(sheet); dr.text((10, 6), title, fill="black", font=font(24))
        for k, ln in enumerate(lines):
            dr.text((10, H + 46 + 24 * k), ln, fill=(40, 40, 40), font=font(18))
        fn = out / (title.split(":")[0].replace(" ", "_") + ".png"); sheet.save(fn); print("wrote", fn, flush=True)

    geo = "front edge c=%.2f, columns x %.2f..%.2f, floor %.2f, cavity %.2f, top %.2f; pitch %.1f x %.1f/%.1f cm; cylinders r %.1f cm, heights top rows %s cm" % (
        g.c, g.x0, g.x0 + 6 * g.px, g.f, g.H, g.s, 100 * g.px, 100 * g.pyi, 100 * g.pyt, 100 * g.r, [round(100 * x, 1) for x in g.hts_top])
    sc.occupy(allslots)
    shot("0 overview: every cell of both levels occupied", [geo], P.Q_HOME, g.fo, None, see_inside=True)
    for n, spec in enumerate(cells.split(";")):
        lv, ij = spec.split(":"); i, j = map(int, ij.split("_")); lv = {"i": "interior", "t": "top"}[lv[0]]
        key = (lv, i, j)
        cp = C.CellPaths(sc, lv, i, j, q7)
        crowd = allslots - C.designed_blockers(g, lv, i, j) - {key}
        found = None
        for kk, (Ls, hl) in cp.layers.items():
            depth, qi, kind, var = kk
            if kind != "open":
                continue
            sc.occupy(crowd | {key})
            po = dp_path(sc, Ls, g.fo, None, g)
            if po is None:
                continue
            for kh, (Lh, hlh) in cp.layers.items():
                if kh[:3] != (depth, qi, "held"):
                    continue
                sc.occupy(crowd)
                ph = dp_path(sc, Lh, g.r, hlh, g)
                if ph is not None and np.abs(po[0][-1] - ph[0][-1]).max() <= 0.1:
                    found = (depth, qi, po, ph, hlh); break
            if found:
                break
        if found is None:
            print("no path for", spec); continue
        depth, qi, (qo, co), (qh, ch), hlh = found
        lines = [geo, "grasp %.0f cm below the top, hand flip %d; CROWDED: all cells occupied except the target's designed blockers" % (100 * depth, qi),
                 "open-hand path: %d steps, min clearance %.1f mm | holding the object: %d steps, min clearance %.1f mm" % (
                     len(qo), 1000 * min(co), len(qh), 1000 * min(ch))]
        sc.occupy(crowd | {key})
        shot("%d.1 %s (%d,%d) path start (open hand)" % (n + 1, lv, i, j), lines, qo[0], g.fo, None, see_inside=lv == "interior")
        shot("%d.2 %s (%d,%d) at the grasp" % (n + 1, lv, i, j), lines, qo[-1], g.fo, None, see_inside=lv == "interior")
        sc.occupy(crowd)
        shot("%d.3 %s (%d,%d) holding the object, path start" % (n + 1, lv, i, j), lines, qh[0], g.r, hlh, see_inside=lv == "interior")


if __name__ == "__main__":
    main()
