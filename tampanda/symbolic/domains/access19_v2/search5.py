"""Random + local search over the parametric Access-19 v2 geometry with the crowded verifier (2026-10-02).
Each candidate is scored on a fixed set of hard cells (corners, deepest rows, cells next to the robot); evaluation
stops at the first failing cell, so bad candidates cost seconds. Writes one JSON line per candidate.

    python -m tampanda.symbolic.domains.access19_v2.search out.jsonl seed=0 minutes=110
"""
import json, sys, time
import numpy as np
from tampanda.symbolic.domains.access19_v2 import crowd as C
from tampanda.kinematics import panda as P

HARD = [("top", 6, 6), ("top", 0, 6), ("top", 3, 6), ("top", 3, 4), ("top", 0, 2), ("interior", 0, 6), ("interior", 6, 5),
        ("interior", 3, 6), ("interior", 3, 0), ("top", 6, 0), ("top", 0, 0), ("interior", 6, 0), ("interior", 0, 1),
        ("interior", 0, 3), ("interior", 6, 3), ("top", 3, 3)]
SPACE = {"c": (0.12, 0.30), "x0": (-0.40, 0.30), "x0t": (-0.45, 0.25), "f": (-0.20, 0.20), "H": (0.26, 0.45), "pyt": (0.090, 0.105), "pyi": (0.045, 0.060), "px": (0.058, 0.085), "pxt": (0.058, 0.085), "r": (0.015, 0.020), "h0": (0.06, 0.10), "hstep": (0.054, 0.062), "side": (0.08, 0.16)}


def to_geo(p):
    """Height classes by row pairs: top rows (0,1),(2,3),(4,5),(6) rise by hstep; the interior mirrors it (front tall)."""
    q = dict(p); h0, st = q.pop("h0"), q.pop("hstep")
    top = tuple(round(h0 + st * (j // 2), 4) for j in range(7))
    q["hts_top"] = top; q["hts_int"] = top[::-1]; q["h"] = h0
    return q


def path_progress(sc, Ls, finger, held_local, g):
    """Fraction of layers a continuous path keeps the margin (1.0 = feasible), and the final bottleneck."""
    val = None; prev = None
    for k, Q in enumerate(Ls):
        if len(Q) == 0:
            return k / len(Ls), -1.0
        cl = np.array([sc.clearance(q, finger, held_local)[0] for q in Q])
        if val is None:
            val = cl
        else:
            jump = np.abs(Q[:, None, :] - prev[None, :, :]).max(2)
            val = np.minimum(np.where(jump <= g.max_dq, val[None, :], -np.inf).max(1), cl)
        prev = Q
        if val.max() < g.margin:
            return k / len(Ls), float(val.max())
    return 1.0, float(val.max())


def evaluate(params):
    g = C.Geo(**to_geo(params), depths=(0.007, 0.02))
    try:
        sc = C.Scene(g)
    except Exception as e:
        return {"params": params, "score": -1, "err": str(e)[:100]}
    if sc.base_clear < 0.13:
        return {"params": params, "score": -1, "err": "base_clear %.3f" % sc.base_clear}
    q7 = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(g.q7_step_deg))
    allslots = set(sc.slots); per = {}; total = 0.0; n_ok = 0
    for key in HARD:
        lv, i, j = key
        cp = C.CellPaths(sc, lv, i, j, q7)
        crowd = allslots - C.designed_blockers(g, lv, i, j) - {key}
        ok, info = C.cell_ok(sc, cp, crowd | {key}, crowd)
        if ok:
            n_ok += 1; cs = 1.0 + min(5 * min(info["open_clear"], info["held_clear"]), 0.1)
        else:   # partial credit: best path progress over grasps (open path first, then held path)
            best = 0.0
            for key3, (Ls, hl) in cp.layers.items():
                kind = key3[2]
                sc.occupy(crowd | {key} if kind == "open" else crowd)
                pr, _ = path_progress(sc, Ls, g.fo if kind == "open" else g.r, hl if kind == "held" else None, g)
                best = max(best, 0.45 * pr)
            cs = best
        per["%s_%d_%d" % key] = round(cs, 3); total += cs
    return {"params": params, "score": round(total, 4), "n": n_ok, "per": per}


def main():
    out = sys.argv[1]; kv = dict(a.split("=", 1) for a in sys.argv[2:])
    rng = np.random.default_rng(int(kv.get("seed", 0))); tmax = time.time() + 60 * float(kv.get("minutes", 100))
    best = None; f = open(out, "a")
    while time.time() < tmax:
        if best is None or rng.random() < 0.4:
            p = {k: float(rng.uniform(*b)) for k, b in SPACE.items()}
        else:   # local move around the incumbent
            p = {k: float(np.clip(best["params"][k] + rng.normal(0, 0.12 * (b[1] - b[0])), *b)) for k, b in SPACE.items()}
        p = {k: round(v, 4) for k, v in p.items()}
        r = evaluate(p); f.write(json.dumps(r) + "\n"); f.flush()
        if best is None or r["score"] > best["score"]:
            best = r; print("BEST %.3f n=%s %s per=%s" % (r["score"], r.get("n"), p, r.get("per")), flush=True)


if __name__ == "__main__":
    main()
