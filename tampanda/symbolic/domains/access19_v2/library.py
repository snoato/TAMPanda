"""Access-19 v2 grasp library: every cell's verified pick/put motions with EXACT conflict sets.

Why a library. All boxes are identical and sit at fixed cell positions, and arm-vs-object distances are pairwise:
a motion that is clear of the shelf and of itself is clear in occupancy S iff no single object in S comes within
``MARGIN`` of it. So for each candidate motion we record once, with every cell occupied, WHICH cells' objects it
would touch -- its conflict set. Feasibility in any state is then a set test:

    action on cell c is feasible in S  <=>  some candidate of c has  conflicts & (S - {c}) == {}

which is exact under the same collision model, deterministic, and microseconds per query (``checker.py``). It is
also conservative: the candidates are the empty-shelf optima of the path search, so a motion that exists only by
weaving around particular objects is not represented (``validate.py`` measures that gap against online planning).

A candidate is one complete motion through home (``motion.HOME``):

    pick c : transit_open (home -> open start)  ->  open path  ->  close  ->  held path reversed  ->  transit_held
    put  c : transit_held reversed  ->  held path  ->  open  ->  open path reversed  ->  transit_open reversed

Pick and put of a cell therefore have the same feasibility in the same surrounding occupancy.

Build (``build``): (1) per cell, all DP-optimal open / held paths on the empty shelf, dense-checked against the
static scene, with their conflict sets measured with every other cell occupied; (2) one reference posture for the
geometry (``grasp.fit_reference``); (3) per cell, the matched (open, held) pairs whose conflict sets are
subset-minimal plus the ``TOP_POSTURE`` best-posture pairs; for each, a validated transit to home -- planned with
EVERY cell occupied (so it never conflicts) and, only if that fails, on the empty shelf with its conflict set
measured. Cells are independent, so all phases run in a process pool.

Files: ``<stem>.json`` (geometry, reference posture, bit order, per-cell candidate masks; all the checker needs)
and ``<stem>.npz`` (the joint paths, for execution and rendering).

    python -m tampanda.symbolic.domains.access19_v2.library OUT_STEM [--workers N] [--geo FILE] [--cells ID,ID]
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tampanda.symbolic.domains.access19_v2 import grasp as G
from tampanda.symbolic.domains.access19_v2 import kinematics as K
from tampanda.symbolic.domains.access19_v2 import motion as M
from tampanda.symbolic.domains.access19_v2.shelf import DETECT, T17C, Scene, ShelfGeometry, cell_id, cell_keys

FORMAT_VERSION = 1
MARGIN = G.MARGIN
TOP_POSTURE = 4          # always keep this many best-posture pairs per cell (execution quality)
MAX_CANDIDATES = 12      # transits are planned for at most this many pairs per cell

_SC = None               # per-process scene


def _scene(gd):
    global _SC
    if _SC is None or asdict(_SC.g) != gd:
        _SC = Scene(ShelfGeometry(**gd))
    return _SC


def _mask(per, exclude, idx):
    m = 0
    for k, v in per.items():
        if k != exclude and v < MARGIN:
            m |= 1 << idx[k]
    return m


# ---------------------------------------------------------------- phase 1: grasp paths + conflict sets
def unavoidable(g, key, corridor):
    """Cells whose objects any frontal grasp of ``key`` must pass: the same-row neighbours and the cells in
    front of it within +-corridor columns. Used only to build crowded DP contexts."""
    lv, i, j = key
    out = {(lv, i - 1, j), (lv, i + 1, j)}
    for jj in range(j):
        for dx in range(-corridor, corridor + 1):
            out.add((lv, i + dx, jj))
    return out


def _contexts(g, key):
    """DP contexts, from permissive to strict. The DP maximises clearance against what is PRESENT, so on the empty
    shelf it is free to sweep the arm through other cells; each crowded context makes it avoid a larger region,
    and the subset-minimal conflict sets over all contexts survive (phase 3)."""
    lv, i, j = key
    others = set(cell_keys(g)) - {key}
    other_level = {k for k in others if k[0] != lv}
    behind = other_level | {k for k in others if k[0] == lv and (k[2] > j or (k[2] == j and abs(k[1] - i) > 1))}
    ctx = [("empty", set()), ("other_level", other_level), ("behind", behind)]
    for corridor in (2, 1, 0):
        ctx.append(("corridor%d" % corridor, others - unavoidable(g, key, corridor)))
    return ctx


def _phase1(args):
    gd, key = args
    sc = _scene(gd); g = sc.g; t0 = time.time()
    keys = cell_keys(g); idx = {k: n for n, k in enumerate(keys)}; others = set(keys) - {key}
    q7s = K.q7_grid(g.q7_step_deg); cp = G.CellPaths(sc, key, q7s)
    out = {"open": [], "held": []}; n_dp = {}; seen = set()
    for ctx, occ in _contexts(g, key):
        for (kind, route), (Ls, held_local) in cp.layers.items():
            finger, hl = (g.fo, None) if kind == "open" else (g.r, held_local)
            sc.occupy(occ | ({key} if kind == "open" else set()))
            finals = G.dp_paths(sc, Ls, finger, hl, g); n_dp[ctx] = n_dp.get(ctx, 0) + len(finals)
            sc.occupy(others | ({key} if kind == "open" else set()))   # everything else present: measure conflicts
            for Q, _ in finals:
                sig = (kind, route, np.round(Q, 3).tobytes())
                if sig in seen:
                    continue
                seen.add(sig)
                ok, static, per = G.profile(sc, Q, finger, hl)
                if not ok or (kind == "open" and per.get(key, DETECT) < MARGIN):
                    continue
                out[kind].append({"Q": Q.astype(np.float32), "route": route, "ctx": ctx, "mask": _mask(per, key, idx),
                                  "static": float(static), "held_local": held_local})
    return key, out, {"dp_finals": n_dp, "open": len(out["open"]), "held": len(out["held"]), "s": time.time() - t0}


# ---------------------------------------------------------------- phase 3: candidates + transits
def _transit(sc, key, q_end, held_local, kind, q7s, idx):
    """Validated transit home <-> a grasp path's outer end, and its conflict mask."""
    g = sc.g; keys = set(cell_keys(g))
    finger = g.fo if kind == "open" else g.r
    direction = [0, -1, 0] if key[0] == "interior" else [0, 0, 1]
    worst = keys if kind == "open" else keys - {key}
    alone = {key} if kind == "open" else set()
    for occ in (worst, alone):
        path = M.plan_validated(sc, q_end, M.HOME, occ, held_local, finger, direction, q7s, MARGIN,
                                reverse=(kind == "held"))
        if path is None:
            continue
        path = M.shortcut(sc, path, occ, held_local, finger, MARGIN)
        if occ is worst:
            return path, 0
        sc.occupy(worst)
        ok, _, per = G.profile(sc, path, finger, held_local)
        if ok:
            return path, _mask(per, key, idx)
    return None, None


def _phase3(args):
    gd, key, paths, ref = args
    sc = _scene(gd); g = sc.g; t0 = time.time()
    keys = cell_keys(g); idx = {k: n for n, k in enumerate(keys)}; q7s = K.q7_grid(g.q7_step_deg)
    opens, helds = paths["open"], paths["held"]
    pairs = []
    for a, b in G.match_pairs(opens, helds):
        o, h = opens[a], helds[b]
        pairs.append((G.posture_cost(o["Q"][0], h["Q"][0], ref), o["mask"] | h["mask"], a, b))
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
    chosen = sorted(chosen, key=lambda p: p[0])[:MAX_CANDIDATES]
    cands = []
    for cost, mask, a, b in chosen:
        o, h = opens[a], helds[b]
        t_open, m1 = _transit(sc, key, o["Q"][0], None, "open", q7s, idx)
        if t_open is None:
            continue
        t_held, m2 = _transit(sc, key, h["Q"][0], h["held_local"], "held", q7s, idx)
        if t_held is None:
            continue
        cands.append({"cost": cost, "mask": mask | m1 | m2, "grasp_mask": mask, "transit_mask": m1 | m2,
                      "routes": o["route"] + h["route"], "open": o["Q"], "held": h["Q"],
                      "t_open": t_open.astype(np.float32), "t_held": t_held.astype(np.float32),
                      "held_local": np.asarray(h["held_local"], np.float32),
                      "transit_len": M.joint_length(t_open) + M.joint_length(t_held)})
    return key, cands, {"pairs": len(pairs), "distinct_masks": len(masks), "minimal": len(minimal),
                        "chosen": len(chosen), "candidates": len(cands), "s": time.time() - t0}


# ---------------------------------------------------------------- build / save / load
def build(g: ShelfGeometry = T17C, workers: int = 0, cells=None, log=print):
    gd = asdict(g); keys = cell_keys(g)
    todo = [k for k in keys if cells is None or cell_id(k) in cells]
    workers = workers or min(len(todo), max(1, (os.cpu_count() or 2) - 2))
    ctx = mp.get_context("spawn"); t0 = time.time()
    with ctx.Pool(workers) as pool:
        p1 = {}
        for key, out, st in pool.imap_unordered(_phase1, [(gd, k) for k in todo]):
            p1[key] = out; log("phase1 %-22s %s" % (cell_id(key), st))
        per_cell = {}
        for key, out in p1.items():
            rows = [np.concatenate([out["open"][a]["Q"][0], out["held"][b]["Q"][0]])
                    for a, b in G.match_pairs(out["open"], out["held"])]
            per_cell[key] = np.array(rows) if rows else np.zeros((0, 14))
        ref, score = G.fit_reference(per_cell)
        log("reference posture %s (mean interior<->top |dq| %.2f)" % (np.round(ref, 3).tolist(), score))
        lib = {}
        for key, cands, st in pool.imap_unordered(_phase3, [(gd, k, p1[k], ref) for k in todo]):
            lib[key] = cands; log("phase3 %-22s %s" % (cell_id(key), st))
    return {"geometry": g, "ref": ref, "ref_score": score, "cells": keys, "candidates": lib,
            "build_s": time.time() - t0}


def save(lib, stem) -> None:
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    g = lib["geometry"]; meta = {
        "format": FORMAT_VERSION, "geometry": asdict(g), "digest": g.digest(), "margin": MARGIN, "fine": G.FINE,
        "ref": [float(x) for x in lib["ref"]], "bits": [cell_id(k) for k in lib["cells"]],
        "built": datetime.datetime.now().isoformat(timespec="seconds"), "build_s": round(lib["build_s"], 1),
        "cells": {}}
    arrays = {}
    for key in lib["cells"]:
        cid = cell_id(key); rows = []
        for n, c in enumerate(lib["candidates"].get(key, [])):
            rows.append({"mask": hex(c["mask"]), "grasp_mask": hex(c["grasp_mask"]),
                         "transit_mask": hex(c["transit_mask"]), "cost": round(c["cost"], 4),
                         "routes": c["routes"], "transit_len": round(c["transit_len"], 3)})
            for f in ("open", "held", "t_open", "t_held", "held_local"):
                arrays["%s/%d/%s" % (cid, n, f)] = c[f]
        meta["cells"][cid] = rows
    with open(str(stem) + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)
    np.savez_compressed(str(stem) + ".npz", **arrays)


def load_paths(stem):
    """{cell_id: [{open, held, t_open, t_held, held_local}, ...]} in candidate order (for execution / rendering)."""
    stem = str(stem)
    if stem.endswith(".json") or stem.endswith(".npz"):
        stem = stem.rsplit(".", 1)[0]
    meta = json.load(open(stem + ".json")); z = np.load(stem + ".npz")
    return {cid: [{f: z["%s/%d/%s" % (cid, n, f)] for f in ("open", "held", "t_open", "t_held", "held_local")}
                  for n in range(len(rows))] for cid, rows in meta["cells"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_stem")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--geo", help="k=v geometry file (default: T17C)")
    ap.add_argument("--cells", help="comma-separated cell ids (default: all)")
    a = ap.parse_args()
    g = ShelfGeometry.from_kv(open(a.geo).read()) if a.geo else T17C
    lib = build(g, a.workers, set(a.cells.split(",")) if a.cells else None, log=lambda s: print(s, flush=True))
    save(lib, a.out_stem)
    n = {cell_id(k): len(v) for k, v in lib["candidates"].items()}
    print("built %d cells in %.0f s; candidates per cell min %d median %d; cells without a candidate: %s"
          % (len(n), lib["build_s"], min(n.values()), int(np.median(list(n.values()))),
             [c for c, v in n.items() if v == 0]))


if __name__ == "__main__":
    main()
