"""Top-grid-only search around candidate G1 (interior fixed): x0t, pxt, gap0t, pyt. Score = hard top cells passing
crowded + lateral (partial credit for path progress). (2026-10-02)"""
import json, sys, time
import numpy as np
import crowd as C
import search6 as S6

G1 = dict(c=0.14, x0=0.26, mirror_int=1, px=0.060, pyi=0.032, r=0.010, side=0.18, f=0.0, H=0.31,
          hts_top=(0.08, 0.08, 0.135, 0.135, 0.19, 0.19, 0.245), hts_int=(0.245, 0.19, 0.19, 0.135, 0.135, 0.08, 0.08),
          h=0.08)
SPACE = {"x0t": (-0.56, 0.20), "pxt": (0.056, 0.062), "gap0t": (0.0, 0.16), "pyt": (0.088, 0.100)}
S6.HARD[:] = [("top", 0, 0), ("top", 6, 0), ("top", 3, 0), ("top", 0, 6), ("top", 6, 6), ("top", 3, 6), ("top", 0, 3),
              ("top", 6, 3), ("top", 2, 5), ("top", 5, 1), ("top", 1, 1), ("top", 5, 5)]


def evaluate(p):
    S6.to_geo = lambda q: {**G1, **q}          # reuse search6.evaluate's scoring with the G1 base
    r = S6.evaluate(p)
    return r


def main():
    out = sys.argv[1]; kv = dict(a.split("=", 1) for a in sys.argv[2:])
    rng = np.random.default_rng(int(kv.get("seed", 0))); tmax = time.time() + 60 * float(kv.get("minutes", 55))
    best = None; f = open(out, "a")
    while time.time() < tmax:
        if best is None or rng.random() < 0.4:
            p = {k: float(rng.uniform(*b)) for k, b in SPACE.items()}
        else:
            p = {k: float(np.clip(best["params"][k] + rng.normal(0, 0.12 * (b[1] - b[0])), *b)) for k, b in SPACE.items()}
        p = {k: round(v, 4) for k, v in p.items()}
        r = evaluate(p); f.write(json.dumps(r) + "\n"); f.flush()
        if best is None or r["score"] > best["score"]:
            best = r; print("BEST %.3f n=%s %s per=%s" % (r["score"], r.get("n"), p, r.get("per")), flush=True)


if __name__ == "__main__":
    main()
