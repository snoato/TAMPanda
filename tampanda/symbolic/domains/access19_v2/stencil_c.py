"""Blocking stencil on the parametric crowd scene (2026-10-02): for each listed cell and each offset (dx, dy) with a
single object there, is the cell still pickable AND placeable (crowd.cell_ok)? Prints, per level, the fraction of
tested cells each offset blocks.

    python -m tampanda.symbolic.domains.access19_v2.stencil_c out.json cells=i:3_6;t:3_3 [geo k=v ...]
"""
import json
import sys

import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import crowd as C

OFFS = [(dx, dy) for dy in range(-3, 2) for dx in range(-2, 3) if (dx, dy) != (0, 0)]


def main():
    out = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    cells = kv.pop("cells")
    g = C.Geo()
    for k, v in kv.items():
        cur = getattr(g, k)
        setattr(g, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    sc = C.Scene(g); allslots = set(sc.slots)
    q7 = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(g.q7_step_deg))
    res = {}
    for spec in cells.split(";"):
        lv, ij = spec.split(":"); i, j = map(int, ij.split("_")); lv = {"i": "interior", "t": "top"}[lv[0]]
        key = (lv, i, j); cp = C.CellPaths(sc, lv, i, j, q7)
        base_ok = C.cell_ok(sc, cp, [key], [])[0]
        row = {"base_ok": base_ok, "blocks": []}
        if base_ok:
            for dx, dy in OFFS:
                nb = (lv, i + dx, j + dy)
                if nb not in allslots:
                    continue
                if not C.cell_ok(sc, cp, [key, nb], [nb])[0]:
                    row["blocks"].append([dx, dy])
        res[spec] = row
        print(spec, row, flush=True)
    for lv in ("i", "t"):
        cs = [(s, r) for s, r in res.items() if s.startswith(lv) and r["base_ok"]]
        if not cs:
            continue
        print("\n%s stencil (blocked fraction), rows dy=+1 (behind) .. -3 (front), cols dx=-2..+2" % lv)
        for dy in range(1, -4, -1):
            line = []
            for dx in range(-2, 3):
                if (dx, dy) == (0, 0):
                    line.append("  [X]"); continue
                tested = [r for s, r in cs if (lambda i, j: ({"i": "interior", "t": "top"}[lv], i + dx, j + dy) in allslots)(*map(int, s.split(":")[1].split("_")))]
                n_b = sum([dx, dy] in r["blocks"] for r in tested)
                line.append(" %4.2f" % (n_b / len(tested)) if tested else "    -")
            print("dy=%+d " % dy + "".join(line))
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
