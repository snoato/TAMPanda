"""Visual proof: for a top cell on a crowded flat top with equal-height objects, render the BEST posture over every
analytic IK solution at the grasp pose and show what it hits. (2026-10-02)

    MUJOCO_GL=egl python -m tampanda.symbolic.domains.access19_v2.contact_proof OUT GEOFILE top 3 4
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import crowd as C
from tampanda.symbolic.domains.access19_v2 import verify as V


def font(sz):
    for f in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    kv = dict(a.split("=", 1) for a in open(sys.argv[2]).read().split())
    lv, i, j = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    g = C.Geo()
    for k, v in kv.items():
        cur = getattr(g, k); setattr(g, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    g.q7_step_deg = 2.0
    sc = C.Scene(g); m, d = sc.m, sc.d
    key = (lv, i, j); allslots = set(sc.slots)
    crowd = allslots - C.designed_blockers(g, lv, i, j) - {key}
    sc.occupy(crowd | {key})
    q7 = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(g.q7_step_deg))
    c = C.centre(g, lv, i, j); ht = C.height(g, lv, j)
    best = (-9, None, None)
    n = 0
    for quat in V.FRONT_QUATS:
        R = V.quat_to_R(quat)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [c[0], c[1] - V.GRASP_CONTACT_OFFSET, c[2] + ht - 0.007]
        for q in C.ik_all(sc, T, q7):
            n += 1
            cl, pair = sc.clearance(q, g.fo)
            if cl > best[0]:
                best = (cl, q, pair)
    cl, q, pair = best
    hit_body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[pair[1]]) if pair else None
    arm_body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[pair[0]]) if pair else None
    for k, name in sc.slots.items():
        gid = [gg for gg in range(m.ngeom) if m.geom_bodyid[gg] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)][0]
        m.geom_rgba[gid] = (0.15, 0.3, 0.9, 1) if k == key else ((0.9, 0.1, 0.1, 1) if name == hit_body else (0.6, 0.6, 0.6, 1))
    d.qpos[sc.jq] = q; d.qpos[sc.fq] = g.fo; mujoco.mj_forward(m, d)
    opt = mujoco.MjvOption(); opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    m.vis.scale.contactwidth = 0.04; m.vis.scale.contactheight = 0.02
    m.geom_margin[:] = 0.0; mujoco.mj_forward(m, d)
    W, H = 760, 560
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    r = mujoco.Renderer(m, H, W)
    tiles = []
    for name, (az, el, dist) in (("side (along the row)", (0, -8, 0.9)), ("oblique", (40, -25, 1.0))):
        cam = mujoco.MjvCamera(); cam.lookat[:] = [c[0], c[1] - 0.12, c[2] + ht]; cam.distance = dist; cam.azimuth = az; cam.elevation = el
        r.update_scene(d, cam, scene_option=opt); im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 30], fill=(20, 20, 28)); dr.text((8, 5), name, fill="white", font=font(18)); tiles.append(im)
    sheet = Image.new("RGB", (2 * W, H + 80), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, (k * W, 70))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 8), "%s cell (%d,%d), crowded, equal-height cylinders: BEST of %d IK postures at the grasp" % (lv, i, j, n),
            fill="black", font=font(22))
    dr.text((10, 38), "best clearance %+.1f mm: %s hits %s (red); target = blue" % (1000 * cl, arm_body, hit_body),
            fill=(150, 0, 0) if cl < 0 else (0, 110, 0), font=font(20))
    sheet.save(out / ("proof_%s_%d_%d.png" % (lv, i, j)))
    print("best clearance %.4f %s vs %s over %d postures" % (cl, arm_body, hit_body, n))


if __name__ == "__main__":
    main()
