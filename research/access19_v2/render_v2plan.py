"""Execute + render an Access-19 v2 plan on the parametric crowd scene (2026-10-02).

Plan = list of (kind, level, i, j) with kind in {pick, put}; the scene starts from `start` occupancy. For every action:
verified grasp paths (crowd.CellPaths + bottleneck DP in the CURRENT scene), transit from the current configuration to
the action's path start with VAMP (exact MuJoCo re-check of every interpolated configuration), approach, close/open,
retreat. Every rendered frame is collision-checked (clearance < 0 is reported). Writes an mp4 and a JSON report.
"""
import json
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tampanda.kinematics import panda as P
import crowd as C
import transit as Tr
import verify as V

W, H = 720, 464
FINE = 0.005                 # joint-space check resolution (rad) for shortcut segments and final path checks


def font(sz):
    for f in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def densify(Q, step=0.04):
    out = [np.asarray(Q[0], float)]
    for a, b in zip(Q[:-1], Q[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return out


def main():
    out = Path(sys.argv[1]); geo_file = sys.argv[2]; plan_spec = sys.argv[3] if len(sys.argv) > 3 else "canonical"
    kv = dict(a.split("=", 1) for a in open(geo_file).read().split())
    g = C.Geo()
    for k, v in kv.items():
        cur = getattr(g, k); setattr(g, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    # ONE grasp for every pick and put (no regrasp): grasp height 0.7 cm below the object top, hand flip 0
    V.FRONT_QUATS = (V.FRONT_QUATS[0],); g.depths = (0.007,)
    sc = C.Scene(g); m, d = sc.m, sc.d
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    ren = mujoco.Renderer(m, H, W)
    q7 = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(g.q7_step_deg))
    allslots = set(sc.slots)
    import os
    ref = None
    if os.environ.get("A19V2_POSTURE", "1") != "0":
        ref, rscore = Tr.fit_reference(sc, q7)
        print("posture reference", np.round(ref, 3).tolist(), "mean interior<->top |dq| %.2f" % rscore, flush=True)
    if plan_spec.startswith("unload:"):
        # unload whole interior columns onto the top: interior row r -> top row (nrow-1-r), front rows first
        cols = [int(x) for x in plan_spec.split(":")[1].split(",")]
        occ = {("interior", c, r) for c in cols for r in range(g.nrow)} & allslots
        plan = []
        for r in range(g.nrow):
            for c in cols:
                if ("interior", c, r) in occ:
                    plan += [("pick", "interior", c, r), ("put", "top", c, g.nrow - 1 - r)]
        boundary = len(plan)
    # canonical Access-19 start: blockers in interior columns 1, 3, 5 rows 0..5, target at (3, 6)
    if not plan_spec.startswith("unload:"):
        occ = {("interior", c, r) for c in (1, 3, 5) for r in range(6)} | {("interior", 3, 6)}
    plan = plan if plan_spec.startswith("unload:") else []
    park = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1}           # interior row -> top row (same height class)
    if not plan_spec.startswith("unload:"):
        for r in range(6):
            plan += [("pick", "interior", 3, r), ("put", "top", 3, park[r])]
        plan += [("pick", "interior", 3, 6), ("put", "top", 6, 0)]        # target to its goal cell (off column 3)
        for r in range(5, -1, -1):
            plan += [("pick", "top", 3, park[r]), ("put", "interior", 3, r)]
        boundary = 14
    frames = []; report = []; held = None; q = P.Q_HOME.copy(); finger = g.fo
    cams = []
    pts = np.array([C.centre(g, *k) for k in sc.slots]); cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    cz = 0.5 * (g.f + g.s) + 0.05
    for (az, el, dist, look) in ((55, -20, 1.75, [cx, cy - 0.1, cz]), (0, -10, 1.65, [cx, cy - 0.15, cz])):
        cm = mujoco.MjvCamera(); cm.lookat[:] = look; cm.distance = dist; cm.azimuth = az; cm.elevation = el
        cams.append(cm)
    wall_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("wall_l", "fill_l", "wall_r", "fill_r", "panel_l", "panel_r")]
    rgba0 = m.geom_rgba.copy()
    n_bad = [0]

    def frame(qq, f, hl, label, ignore=None):
        # ignore: the object being grasped/released at the gripper close/open instant (contact by design)
        if ignore is not None:
            saved = d.mocap_pos[sc.mid[ignore]].copy(); d.mocap_pos[sc.mid[ignore]] = C.FAR
        cl, pr = sc.clearance(qq, f, hl)
        if cl < 0:
            print("BAD FRAME", label, len(frames), "%.4f" % cl, "finger %.3f" % f, "held" if hl is not None else "open", "ignore" if ignore is not None else "", pr and (sc.name(pr[0]), sc.name(pr[1])), flush=True)
        if ignore is not None:
            d.mocap_pos[sc.mid[ignore]] = saved
        if cl < 0:
            n_bad[0] += 1
        for mid in sc.held_mids:
            d.mocap_pos[mid] = C.FAR
        d.qpos[sc.jq] = qq; d.qpos[sc.fq] = f
        mujoco.mj_kinematics(m, d)
        if hl is not None:
            loc, hk = hl; T = sc.site_T(qq); d.mocap_pos[sc.held_mids[hk]] = T[:3, 3] + T[:3, :3] @ loc
        mujoco.mj_forward(m, d)
        tiles = []
        for ci, cm in enumerate(cams):
            m.geom_rgba[:] = rgba0
            if ci == 1:                    # cutaway: near side wall transparent
                for gid in wall_ids:
                    if gid >= 0:
                        m.geom_rgba[gid, 3] = 0.15
            ren.update_scene(d, cm); tiles.append(Image.fromarray(ren.render()))
        m.geom_rgba[:] = rgba0
        im = Image.new("RGB", (2 * W, H + 60), (18, 18, 26))
        im.paste(tiles[0], (0, 60)); im.paste(tiles[1], (W, 60)); dr = ImageDraw.Draw(im)
        dr.text((12, 6), label, fill="white", font=font(24))
        dr.text((12, 36), f"Access-19 v2 ({Path(geo_file).stem}) | one frontal grasp, no regrasp | all arm links collide (real meshes) | "
                "min clearance this frame %+.1f mm" % (1000 * cl), fill=(200, 200, 200), font=font(17))
        dr.text((W + 12, 66), "cutaway (side wall transparent)", fill=(230, 230, 230), font=font(16))
        frames.append(np.asarray(im))

    prev_dir = [0, 0, 1]; lens = []

    def shortcut(path, occupied, held_l, finger, tries=80, margin=0.01):
        # exact-geometry shortcutting of a validated path: replace Q[i..j] by the straight joint-space segment
        # when that segment keeps >= 1 cm clearance everywhere (VAMP's own simplify works on its sphere model)
        Q = [np.asarray(x, float) for x in path]; rng = np.random.default_rng(0); sc.occupy(occupied)
        for _ in range(tries):
            if len(Q) < 3:
                break
            i, j = sorted(rng.choice(len(Q), 2, replace=False))
            if j - i < 2:
                continue
            if Tr.mj_check(sc, np.array([Q[i], Q[j]]), finger, held_l, margin, step=FINE)[0]:
                Q = Q[:i + 1] + Q[j:]
        return np.array(Q)

    def jlen(Q):
        return float(np.linalg.norm(np.diff(np.asarray(Q), axis=0), axis=1).sum())

    def transit(q_from, q_to, occupied, held_l, R, finger, to_dir):
        path, ok, *_ = Tr.plan_validated(sc, q_to, q_from, occupied, held_l, R, finger, to_dir, q7, 0.002, step=FINE)
        how = "direct"
        if not ok[0]:
            pa, oka, *_ = Tr.plan_validated(sc, q_from, P.Q_HOME, occupied, held_l, R, finger, prev_dir, q7, 0.002,
                                            reverse=True, step=FINE)
            pb, okb, *_ = Tr.plan_validated(sc, q_to, P.Q_HOME, occupied, held_l, R, finger, to_dir, q7, 0.002, step=FINE)
            if not (oka[0] and okb[0]):
                return None, "failed"
            path, how = np.vstack([pa, pb]), "via_home"
        l0 = jlen(path); raw = path; path = shortcut(path, occupied, held_l, finger)
        # final check at FINE joint resolution (0.02 rad moves a fingertip up to ~16 mm between samples)
        sc.occupy(occupied)
        if not Tr.mj_check(sc, path, finger, held_l, 0.002, step=FINE)[0]:
            path = raw
            if not Tr.mj_check(sc, raw, finger, held_l, 0.0, step=FINE)[0]:
                how += "+dense_fail"
        lens.append((l0, jlen(path), float(np.linalg.norm(np.asarray(q_to) - np.asarray(q_from)))))
        return path, how

    for n, (kind, lv, i, j) in enumerate(plan):
        key = (lv, i, j); phase = ("unload lower deck -> upper deck" if plan_spec.startswith("unload:") else
                                   ("phase 1: extraction" if n < boundary else "phase 2: reload"))
        label = "%s | %2d/%d  %s %s (%d,%d)" % (phase, n + 1, len(plan), kind, lv, i, j)
        cp = C.CellPaths(sc, lv, i, j, q7)
        others = occ - {key}
        cands = (Tr.grasp_candidates(sc, cp, others | {key}, others, ref) if ref is not None else
                 [x for x in [Tr.grasp_paths(sc, cp, others | {key}, others)] if x is not None])
        gp = cands[0] if cands else None
        row = {"action": [kind, lv, i, j], "grasp": gp is not None}
        if gp is None:
            report.append(row); print(label, "NO GRASP -- plan aborted", flush=True); break
        R = V.quat_to_R(V.FRONT_QUATS[gp["quat"]])
        out_dir = [0, -1, 0] if lv == "interior" else [0, 0, 1]
        if kind == "pick":
            # transit (empty hand) current -> open-path start, scene with the target present
            for rank, gp in enumerate(cands):          # first grasp candidate with a valid transit
                path, how = transit(q, gp["open"][0], others | {key}, None, R, g.fo, out_dir)
                if path is not None:
                    break
            row["transit_ok"] = path is not None; row["transit"] = how; row["grasp_rank"] = rank
            sc.occupy(others | {key})
            for qq in densify(path if path is not None else [q, gp["open"][0]]):
                frame(qq, g.fo, None, label)
            for qq in densify(gp["open"]):
                frame(qq, g.fo, None, label)
            for f in np.linspace(g.fo, g.r, 6):
                frame(gp["open"][-1], f, None, label, ignore=key)
            sc.occupy(others); occ = others
            for qq in densify(gp["held"][::-1]):
                frame(qq, g.r, gp["held_local"], label)
            q = gp["held"][0]; held = gp["held_local"]
        else:
            # transit (holding) current -> held-path start, scene without the target cell
            for rank, gp in enumerate(cands):
                path, how = transit(q, gp["held"][0], others, held, R, g.r, out_dir)
                if path is not None:
                    break
            row["transit_ok"] = path is not None; row["transit"] = how; row["grasp_rank"] = rank
            sc.occupy(others)
            for qq in densify(path if path is not None else [q, gp["held"][0]]):
                frame(qq, g.r, held, label)
            for qq in densify(gp["held"]):
                frame(qq, g.r, held, label)
            occ = others | {key}; sc.occupy(occ)
            for f in np.linspace(g.r, g.fo, 6):
                frame(gp["open"][-1], f, None, label, ignore=key)
            for qq in densify(gp["open"][::-1]):
                frame(qq, g.fo, None, label)
            q = gp["open"][0]; held = None
        prev_dir = out_dir
        row["frames_total"] = len(frames); row["bad_frames_total"] = n_bad[0]
        report.append(row); print(label, row, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out / "plan.mp4", frames, fps=30, quality=8)
    imageio.imwrite(out / "poster.png", frames[len(frames) // 3])
    json.dump({"plan": plan, "report": report, "frames": len(frames), "bad_frames": n_bad[0],
               "posture_ref": None if ref is None else [float(x) for x in ref], "transit_lens": lens}, open(out / "report.json", "w"),
              default=str)
    print("frames", len(frames), "bad frames", n_bad[0])
    if lens:
        a = np.array(lens)
        print("transit joint-space length: raw %.1f rad -> shortcut %.1f rad (mean %.2f -> %.2f; straight-line lower bound mean %.2f)"
              % (a[:, 0].sum(), a[:, 1].sum(), a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()))


if __name__ == "__main__":
    main()
