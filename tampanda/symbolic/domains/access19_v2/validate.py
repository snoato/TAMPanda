"""Validate the grasp library against online planning in the actual scene (2026-10-02).

For random (cell, occupancy) states, compare
  * library : ``GraspLibrary.feasible`` (pure lookup), and
  * online  : ``online.OnlineChecker`` -- the full stack planned in that very scene.
Both are sound (every motion they accept is collision-free in the exact model), so a disagreement means one of them
missed a motion: library-infeasible / online-feasible measures the library's incompleteness, the converse the online
search's. States are drawn per cell at several densities, biased towards the cells that matter (rows in front
of the target and its row), so that both verdicts occur.

    python -m tampanda.symbolic.domains.access19_v2.validate LIB_STEM OUT.json [--n 600] [--workers N] [--geo FILE]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict

import numpy as np

from tampanda.symbolic.domains.access19_v2.checker import GraspLibrary
from tampanda.symbolic.domains.access19_v2.online import OnlineChecker
from tampanda.symbolic.domains.access19_v2.layout import T17C, ShelfGeometry, parse_cell_id


_ONLINE = None


def _work(args):
    global _ONLINE
    gd, cid, others, ref = args
    if _ONLINE is None or asdict(_ONLINE.g) != gd:
        _ONLINE = OnlineChecker(ShelfGeometry(**gd), ref=ref, cache=False)
    t0 = time.time()
    ok = _ONLINE.feasible(cid, others)
    return cid, others, ok, _ONLINE.last["pairs_tried"], time.time() - t0


def sample_states(lib, n, seed=0):
    rng = random.Random(seed); cells = lib.cells; out = []
    for t in range(n):
        cid = cells[t % len(cells)]; lv, i, j = parse_cell_id(cid)
        relevant = [c for c in cells if c != cid and parse_cell_id(c)[0] == lv and parse_cell_id(c)[2] <= j]
        rest = [c for c in cells if c != cid and c not in relevant]
        p = rng.choice([0.05, 0.15, 0.3, 0.5])
        others = [c for c in relevant if rng.random() < p] + [c for c in rest if rng.random() < 0.3]
        out.append((cid, sorted(others)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib_stem"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--geo")
    a = ap.parse_args()
    g = ShelfGeometry.from_kv(open(a.geo).read()) if a.geo else T17C
    lib = GraspLibrary.load(a.lib_stem)
    assert lib.meta["digest"] == g.digest(), "library was built for another geometry"
    states = sample_states(lib, a.n)
    workers = a.workers or max(1, (os.cpu_count() or 2) - 2)
    res = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for cid, others, ok, n, s in pool.imap_unordered(_work, [(asdict(g), c, o, lib.meta["ref"]) for c, o in states]):
            res.append({"cell": cid, "others": others, "online": ok, "library": lib.feasible(cid, others),
                        "pairs_tried": n, "s": round(s, 2)})
    agree = sum(r["online"] == r["library"] for r in res)
    lib_only = [r for r in res if r["library"] and not r["online"]]
    online_only = [r for r in res if r["online"] and not r["library"]]
    summary = {"n": len(res), "agree": agree, "both_feasible": sum(r["online"] and r["library"] for r in res),
               "both_infeasible": sum(not r["online"] and not r["library"] for r in res),
               "library_only": len(lib_only), "online_only": len(online_only),
               "online_s_mean": round(float(np.mean([r["s"] for r in res])), 2)}
    print(json.dumps(summary))
    json.dump({"summary": summary, "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
