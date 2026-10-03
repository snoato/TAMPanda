"""Minimal height drop d_k(pyt): how much shorter must a full row of objects k rows in front of a top target be so
that the target is still picked AND placed (open + held paths) with clearance >= margin? (2026-10-02)
Only that row is occupied (plus the target); the row directly in front is the designed blocker and stays empty.

    python -m tampanda.symbolic.domains.access19_v2.rowdrop "k=v ..." pyts=0.09,0.095 ks=2,3,4 cells=3_6,6_6
"""
import sys
import numpy as np
from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import crowd as C


def ok_for(base, pyt, k, cell, drop, ht=0.14):
    i, j = cell
    hts = [ht] * 7
    hts[j - k] = ht - drop
    g = C.Geo(**base, pyt=pyt, hts_top=tuple(hts), hts_int=tuple([0.08] * 7), h=0.08)
    sc = C.Scene(g)
    q7 = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(g.q7_step_deg))
    cp = C.CellPaths(sc, "top", i, j, q7)
    row = {("top", ii, j - k) for ii in range(7)}
    key = ("top", i, j)
    return C.cell_ok(sc, cp, row | {key}, row)[0]


def main():
    base = {}
    for kv in sys.argv[1].split():
        a, v = kv.split("="); cur = getattr(C.Geo, a)
        base[a] = tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v)
    opts = dict(a.split("=", 1) for a in sys.argv[2:])
    pyts = [float(x) for x in opts.get("pyts", "0.095,0.100,0.105").split(",")]
    ks = [int(x) for x in opts.get("ks", "2,3,4").split(",")]
    cells = [tuple(map(int, c.split("_"))) for c in opts.get("cells", "3_6").split(",")]
    for pyt in pyts:
        for cell in cells:
            for k in ks:
                if cell[1] - k < 0:
                    continue
                if ok_for(base, pyt, k, cell, 0.0):
                    print("pyt %.3f cell %s k=%d: drop 0.0 cm suffices" % (pyt, cell, k), flush=True); continue
                lo, hi = 0.0, 0.09
                if not ok_for(base, pyt, k, cell, hi):
                    print("pyt %.3f cell %s k=%d: even 9 cm drop fails" % (pyt, cell, k), flush=True); continue
                for _ in range(6):
                    mid = (lo + hi) / 2
                    if ok_for(base, pyt, k, cell, mid):
                        hi = mid
                    else:
                        lo = mid
                print("pyt %.3f cell %s k=%d: minimal drop %.1f cm (bracket %.1f-%.1f)" % (
                    pyt, cell, k, 100 * hi, 100 * lo, 100 * hi), flush=True)


if __name__ == "__main__":
    main()
