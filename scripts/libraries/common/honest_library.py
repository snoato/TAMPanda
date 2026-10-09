"""Honest grasp library for a tampanda scene (confined shelf first): Access-19 v2's library construction
(tampanda access19_v2/library.py) on the environment's own scene with the real arm meshes (honest_check.EnvScene).

Objects of the domain are identical and sit at fixed cell positions; arm-vs-object distances are pairwise. So a
complete motion (transit -> open path -> grasp -> held path -> transit) that is clear of the static scene is clear in
occupancy S iff no object in S comes within MARGIN of it, and we record once, with EVERY other cell occupied, which
cells' objects it touches -- its conflict set:

    pick / put on cell c is feasible in S  <=>  some candidate of c has  conflicts & (S - {c}) == {}

Phase 1 (per cell): DP-optimal open / held paths for every grasp family under several crowding contexts (empty,
behind, corridor 2/1/0: the DP maximises clearance against what is PRESENT, so each context makes it avoid a larger
region), dense-checked against the static scene, conflict sets measured with all other cells occupied.
Phase 3 (per cell): matched (open, held) pairs with subset-minimal conflict sets + the best-posture pairs; transits to
home planned with every cell occupied (conflict-free) or, failing that, on the empty shelf with their conflict set
measured. Output: <stem>.json (cells, bit order, candidate masks -- all the checker needs) + <stem>.npz (paths).

    python honest_library.py DOMAIN OUT_STEM [--workers 34] [--cells id,id]

DOMAIN is ``confined-shelf`` (scene: ``confined_scene.py``) or ``msv2:<geometry.kv>`` (multi-level shelf v2, scene:
``msv2_scene.py``); both modules must be importable (``scripts/libraries/<env>/rebuild.sh`` sets PYTHONPATH).

Packaged for tampanda 1.1.0 from the research tooling that built the shipped libraries (a19v2/honest_library.py,
which built confined ``library_v1``, and its extension msv2/honest_library.py, which built ``library_M3``). The only
change is how ``scene`` reaches the confined-shelf scene: the research build went through rgnet's
``ConfinedShelfFeasibilityChecker`` (a19v2/audit_clipping.make_checker); ``confined_scene.py`` builds the same scene
from tampanda alone. The M3 extensions (other-region contexts, the diverse-pair transit fallback for cells with fewer
than two candidates) do not trigger on the confined shelf (one region; every cell has >= 4 candidates).
"""
import argparse, datetime, json, os, sys, time
import multiprocessing as mp
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

TOP_POSTURE = 4
MAX_CANDIDATES = 12
_S = {}


def scene(domain):
    """Honest scene with one object per cell (so every cell can be occupied at once) + cell centres."""
    if "sc" in _S:
        return _S["sc"], _S["cells"]
    import honest_check as H
    if domain.startswith("msv2:"):
        import msv2_scene as MS
        g = MS.from_kv(open(domain.split(":", 1)[1]).read()); H.GRASP_DZ = g.gz
        sc, cells = MS.build(g)
        _S.update(sc=sc, cells=cells)
        return sc, cells
    if domain != "confined-shelf":
        raise NotImplementedError(domain)
    import confined_scene as CS
    sc, cells = CS.build()
    _S.update(sc=sc, cells=cells)
    return sc, cells


def ij(cid):
    i, j = cid.split("__")[1].split("_")
    return int(i), int(j)


def region(cid):
    return cid.split("__")[0]


def contexts(cells, key):
    """Crowding contexts for the DP (v2 _contexts): the others the DP must keep clear of. Other regions count as
    other levels; within the target's region: behind it, and everything but a +-corridor in front."""
    i, j = ij(key); others = set(cells) - {key}; reg = region(key)
    other_level = {c for c in others if region(c) != reg}
    same = others - other_level

    def unavoidable(corr):
        out = {c for c in same if ij(c)[1] == j and abs(ij(c)[0] - i) == 1}            # same-row neighbours
        out |= {c for c in same if ij(c)[1] < j and abs(ij(c)[0] - i) <= corr}         # in front, +-corr columns
        return out
    behind = other_level | {c for c in same if ij(c)[1] > j or (ij(c)[1] == j and abs(ij(c)[0] - i) > 1)}
    ctx = [("empty", set())] + ([("other_level", other_level)] if other_level else []) + [("behind", behind)]
    for corr in (2, 1, 0):
        ctx.append(("corridor%d" % corr, others - unavoidable(corr)))
    return ctx


def layout_for(sc, cells, occ_cells, target=None, target_cell=None):
    """{object: centre}: target (objects[0]) at its cell if given, other objects on occ_cells; plus object->cell."""
    lay, who = {}, {}
    if target_cell is not None:
        lay[sc.objects[0]] = cells[target_cell]; who[sc.objects[0]] = target_cell
    for o, c in zip(sc.objects[1:], sorted(occ_cells)):
        lay[o] = cells[c]; who[o] = c
    return lay, who


def mask_of(per, who, bits, exclude=None):
    m = 0
    for o, d in per.items():
        c = who.get(o)
        if c is not None and c != exclude and d < MARGIN():
            m |= 1 << bits[c]
    return m


def MARGIN():
    import honest_check as H
    return H.MARGIN


def phase1(args):
    domain, key = args
    import honest_check as H
    from tampanda.symbolic.domains.access19_v2 import grasp as G, kinematics as K
    sc, cells = scene(domain); t0 = time.time(); q7s = K.q7_grid(3.0)
    bits = {c: n for n, c in enumerate(sorted(cells))}; others = set(cells) - {key}
    out = {"open": [], "held": []}; seen = set(); n_dp = {}
    for fam, R in H.grasp_families("shelf-access" if domain.startswith("msv2:") else domain).items():
        sc._grasp_R = R
        lines = H.grasp_lines(sc, cells[key], fam, R)
        layers = {}
        for kind, (pts, hl) in lines.items():
            Ls = [K.ik_all(sc.base, H._T(R, p), q7s) for p in pts]
            if fam.startswith("front"):
                k = 0
                while k < len(Ls) - 1 and len(Ls[k]) == 0 and pts[k + 1][1] <= sc.front_y - 0.03:
                    k += 1
                Ls = Ls[k:]
            layers[kind] = (Ls, hl)
        for ctx, occ in contexts(cells, key):
            for kind, (Ls, hl) in layers.items():
                if any(len(L) == 0 for L in Ls):
                    continue
                is_open = kind == "open"; finger = H.FINGER_OPEN if is_open else float(sc.half[0]); h = None if is_open else hl
                lay, _ = layout_for(sc, cells, occ, target_cell=key if is_open else None)
                sc.occupy(lay, held=None if is_open else sc.objects[0])
                finals = G.dp_paths(sc, Ls, finger, h, H.DP); n_dp[ctx] = n_dp.get(ctx, 0) + len(finals)
                lay, who = layout_for(sc, cells, others, target_cell=key if is_open else None)   # everything present
                sc.occupy(lay, held=None if is_open else sc.objects[0])
                for Q, _ in finals:
                    sig = (fam, kind, np.round(Q, 3).tobytes())
                    if sig in seen:
                        continue
                    seen.add(sig)
                    ok, static, per = G.profile(sc, Q, finger, h, margin=H.MARGIN)
                    if not ok or (is_open and per.get(sc.objects[0], 1.0) < H.MARGIN):
                        continue
                    out["open" if is_open else "held"].append(
                        {"Q": Q.astype(np.float32), "fam": fam, "kind": kind, "ctx": ctx,
                         "mask": mask_of(per, who, bits, exclude=key), "static": float(static), "hl": np.asarray(hl, float)})
    return key, out, {"dp_finals": n_dp, "open": len(out["open"]), "held": len(out["held"]), "s": round(time.time() - t0, 1)}


def shortcut(sc, path, lay, held_obj, held_local, finger, margin, tries=80, keep=0.01, seed=0):
    from tampanda.symbolic.domains.access19_v2 import motion as M
    Q = [np.asarray(x, float) for x in path]; rng = np.random.default_rng(seed); sc.occupy(lay, held=held_obj)
    for _ in range(tries):
        if len(Q) < 3:
            break
        i, j = sorted(rng.choice(len(Q), 2, replace=False))
        if j - i >= 2 and M.mj_check(sc, [Q[i], Q[j]], finger, held_local, keep)[0]:
            Q = Q[:i + 1] + Q[j:]
    Q = np.array(Q)
    return Q if M.mj_check(sc, Q, finger, held_local, margin)[0] else np.asarray(path, float)


def transit(sc, cells, key, q_end, held_local, kind, R, fam, q7s, bits):
    import honest_check as H
    from tampanda.symbolic.domains.access19_v2 import grasp as G
    is_open = kind == "open"; finger = H.FINGER_OPEN if is_open else float(sc.half[0])
    held_obj = None if is_open else sc.objects[0]
    direction = [0, -1, 0] if fam.startswith("front") else [0, 0, 1]
    others = set(cells) - {key}
    worst, who = layout_for(sc, cells, others, target_cell=key if is_open else None)
    alone, _ = layout_for(sc, cells, set(), target_cell=key if is_open else None)
    sc._grasp_R = R
    for lay in (worst, alone):
        sc.occupy(lay, held=held_obj)
        path = H.plan_validated(sc, q_end, held_local, finger, direction, R, q7s, reverse=not is_open)
        if path is None:
            continue
        path = shortcut(sc, path, lay, held_obj, held_local, finger, H.MARGIN)
        if lay is worst:
            return path, 0
        sc.occupy(worst, held=held_obj)
        ok, _, per = G.profile(sc, path, finger, held_local, margin=H.MARGIN)
        if ok:
            return path, mask_of(per, who, bits, exclude=key)
    return None, None


def phase3(args):
    domain, key, paths = args
    import honest_check as H
    from tampanda.symbolic.domains.access19_v2 import grasp as G, kinematics as K, motion as M
    sc, cells = scene(domain); t0 = time.time(); q7s = K.q7_grid(3.0)
    bits = {c: n for n, c in enumerate(sorted(cells))}; fams = H.grasp_families("shelf-access" if domain.startswith("msv2:") else domain)
    opens, helds = paths["open"], paths["held"]
    pairs = []
    for a, b in G.match_pairs(opens, helds):
        o, h = opens[a], helds[b]
        if o["fam"] != h["fam"]:
            continue
        pairs.append((G.posture_cost(o["Q"][0], h["Q"][0], H.HOME), o["mask"] | h["mask"], int(a), int(b)))
    pairs.sort(key=lambda p: (p[0], p[2], p[3]))
    best_of = {}
    for p in pairs:
        best_of.setdefault(p[1], p)
    masks = list(best_of)
    minimal = [m for m in masks if not any(o != m and (o & m) == o for o in masks)]
    chosen, seen = [], set()
    for p in pairs[:TOP_POSTURE] + sorted((best_of[m] for m in minimal), key=lambda p: p[0]):
        if (p[2], p[3]) not in seen:
            seen.add((p[2], p[3])); chosen.append(p)
    chosen = sorted(chosen, key=lambda p: (bin(p[1]).count("1"), p[0]))[:MAX_CANDIDATES]
    cands = []
    for cost, mask, a, b in chosen:
        o, h = opens[a], helds[b]; R = fams[o["fam"]]
        t_open, m1 = transit(sc, cells, key, o["Q"][0], None, "open", R, o["fam"], q7s, bits)
        if t_open is None:
            continue
        t_held, m2 = transit(sc, cells, key, h["Q"][0], h["hl"], "held", R, h["fam"], q7s, bits)
        if t_held is None:
            continue
        cands.append({"cost": cost, "mask": mask | m1 | m2, "grasp_mask": mask, "transit_mask": m1 | m2,
                      "fam": o["fam"], "held_kind": h["kind"], "open": o["Q"], "held": h["Q"],
                      "t_open": t_open.astype(np.float32), "t_held": t_held.astype(np.float32),
                      "held_local": np.asarray(h["hl"], np.float32),
                      "transit_len": M.joint_length(t_open) + M.joint_length(t_held)})
    if len(cands) < 2:      # the chosen pairs share a posture whose transits all fail: try more DIVERSE pairs
        tried = [opens[a]["Q"][0] for _, _, a, _ in chosen]; extra = 0
        for cost, mask, a, b in pairs:
            if extra >= 10 or len(cands) >= 2:
                break
            o, h = opens[a], helds[b]
            if any(np.abs(o["Q"][0] - t).max() < 0.3 for t in tried):
                continue
            tried.append(o["Q"][0]); extra += 1; R = fams[o["fam"]]
            t_open, m1 = transit(sc, cells, key, o["Q"][0], None, "open", R, o["fam"], q7s, bits)
            if t_open is None:
                continue
            t_held, m2 = transit(sc, cells, key, h["Q"][0], h["hl"], "held", R, h["fam"], q7s, bits)
            if t_held is None:
                continue
            cands.append({"cost": cost, "mask": mask | m1 | m2, "grasp_mask": mask, "transit_mask": m1 | m2,
                          "fam": o["fam"], "held_kind": h["kind"], "open": o["Q"], "held": h["Q"],
                          "t_open": t_open.astype(np.float32), "t_held": t_held.astype(np.float32),
                          "held_local": np.asarray(h["hl"], np.float32),
                          "transit_len": M.joint_length(t_open) + M.joint_length(t_held)})
    return key, cands, {"pairs": len(pairs), "distinct_masks": len(masks), "minimal": len(minimal),
                        "chosen": len(chosen), "candidates": len(cands), "s": round(time.time() - t0, 1)}


def save(domain, cells, lib, stem, build_s):
    import honest_check as H
    order = sorted(cells)
    meta = {"format": 1, "domain": domain, "kind": "honest grasp library (access19_v2 construction, real arm meshes)",
            "margin": H.MARGIN, "held_lift": H.HELD_LIFT, "bits": order,
            "centres": {c: [round(float(x), 4) for x in cells[c]] for c in order},
            "built": datetime.datetime.now().isoformat(timespec="seconds"), "build_s": round(build_s, 1), "cells": {}}
    arrays = {}
    for c in order:
        rows = []
        for n, cd in enumerate(lib.get(c, [])):
            rows.append({"mask": hex(cd["mask"]), "grasp_mask": hex(cd["grasp_mask"]), "transit_mask": hex(cd["transit_mask"]),
                         "cost": round(cd["cost"], 4), "fam": cd["fam"], "held_kind": cd["held_kind"],
                         "transit_len": round(cd["transit_len"], 3)})
            for f in ("open", "held", "t_open", "t_held", "held_local"):
                arrays["%s/%d/%s" % (c, n, f)] = cd[f]
        meta["cells"][c] = rows
    with open(stem + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)
    np.savez_compressed(stem + ".npz", **arrays)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("domain"); ap.add_argument("out_stem")
    ap.add_argument("--workers", type=int, default=34); ap.add_argument("--cells")
    a = ap.parse_args()
    sc, cells = scene(a.domain); _S.clear()
    todo = sorted(c for c in cells if not a.cells or c in a.cells.split(","))
    t0 = time.time(); ctx = mp.get_context("spawn")
    with ctx.Pool(min(a.workers, len(todo))) as pool:
        p1 = {}
        for key, out, st in pool.imap_unordered(phase1, [(a.domain, k) for k in todo]):
            p1[key] = out; print("phase1 %-22s %s" % (key, st), flush=True)
        lib = {}
        for key, cands, st in pool.imap_unordered(phase3, [(a.domain, k, p1[k]) for k in todo]):
            lib[key] = cands; print("phase3 %-22s %s" % (key, st), flush=True)
    save(a.domain, cells, lib, os.path.join(os.environ.get("AUDIT_CWD", "."), a.out_stem), time.time() - t0)
    n = {c: len(v) for c, v in lib.items()}
    print("built %d cells in %.0f s; candidates per cell min %d median %d; cells without a candidate: %s"
          % (len(n), time.time() - t0, min(n.values()), int(np.median(list(n.values()))), [c for c, v in n.items() if v == 0]), flush=True)


if __name__ == "__main__":
    main()
