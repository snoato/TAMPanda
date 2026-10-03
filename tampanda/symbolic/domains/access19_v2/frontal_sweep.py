"""Screen geometries on hard cells: base clearance (static), and per cell empty-shelf pick+put + lateral blocking.
    python -m tampanda.symbolic.domains.access19_v2.frontal_sweep out.jsonl "geo1" "geo2" ...   (geo = k=v list)"""
import json, sys
import numpy as np
from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import crowd as C

HARD = [("interior", i, j) for i, j in ((0, 0), (4, 0), (2, 0), (0, 6), (4, 6), (2, 6), (0, 3), (4, 3))] + \
       [("top", i, j) for i, j in ((0, 0), (4, 0), (2, 0), (0, 6), (4, 6), (2, 6))]


def main():
    out = open(sys.argv[1], "a")
    for spec in sys.argv[2:]:
        g = C.Geo()
        for kv in spec.split():
            k, v = kv.split("="); cur = getattr(g, k)
            setattr(g, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
        sc = C.Scene(g); allslots = set(sc.slots)
        q7 = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(g.q7_step_deg))
        row = {"geo": spec, "base_clear": round(sc.base_clear_static, 4), "cells": {}}
        if sc.base_clear_static >= 0.005:
            for key in HARD:
                if key not in allslots:
                    continue
                cp = C.CellPaths(sc, *key, q7)
                ok, info = C.cell_ok(sc, cp, [key], [])
                lat = None
                if ok:
                    lat = all(not C.cell_ok(sc, cp, [key, (key[0], key[1] + dx, key[2])], [(key[0], key[1] + dx, key[2])])[0]
                              for dx in (-1, 1) if (key[0], key[1] + dx, key[2]) in allslots)
                row["cells"]["%s_%d_%d" % key] = {"ok": ok, "lat": lat,
                                                   "why": None if ok else str(info[:1])[:120]}
        n_ok = sum(v["ok"] and v["lat"] for v in row["cells"].values())
        row["n_ok"] = n_ok; row["n"] = len(row["cells"])
        out.write(json.dumps(row) + "\n"); out.flush()
        print("%2d/%2d  base %+.3f  %s  fails: %s" % (n_ok, row["n"], sc.base_clear_static, spec[:60],
              [k for k, v in row["cells"].items() if not (v["ok"] and v["lat"])]), flush=True)


if __name__ == "__main__":
    main()
