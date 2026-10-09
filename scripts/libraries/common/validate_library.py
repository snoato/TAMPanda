"""Library vs online planning (honest_check.check_motion) on random states.

    python validate_library.py LIB.json N OUT.json [--workers 24] [--domain confined-shelf | msv2:<geometry.kv>]

States: a random target cell plus 2-15 other random occupied cells, ``random.Random(0)``. Online planning runs the same
motion search as the library build (DP against the objects actually present, VAMP transits, dense exact checks) per
query, so this validates the precomputation and the pairwise conflict-set decomposition, not an independent planner.

Packaged for tampanda 1.1.0 from the research script msv2/validate_library.py (which validated library_M3; its a19v2/
twin validated confined library_v1 and differs only in the --domain option and a state-size guard that is a
no-op for these libraries). Change: ``HonestLibrary`` is imported
from tampanda (``confined_shelf/honest.py``) instead of the research copy ``confined_honest.py``; the two classes give
the same verdict for every cell of a library.
"""
import argparse, json, os, random, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_W = {}


def init(domain):
    import honest_library as HL
    from tampanda.symbolic.domains.access19_v2 import kinematics as K
    import honest_check as H
    sc, cells = HL.scene(domain)
    _W.update(sc=sc, cells=cells, q7s=K.q7_grid(3.0), fams=H.grasp_families("shelf-access" if domain.startswith("msv2:") else domain))


def work(task):
    import honest_check as H
    target, occ = task; sc, cells = _W["sc"], _W["cells"]; t0 = time.time()
    others = {o: cells[c] for o, c in zip(sc.objects[1:], occ)}
    ok = False
    for fam, R in _W["fams"].items():
        if H.check_motion(sc, sc.objects[0], cells[target], others, R, fam, _W["q7s"]) is not None:
            ok = True; break
    return {"target": target, "occ": occ, "online": ok, "s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("lib"); ap.add_argument("n", type=int); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--domain", default="confined-shelf"); a = ap.parse_args()
    import multiprocessing as mp
    from tampanda.symbolic.domains.confined_shelf.honest import HonestLibrary
    lib = HonestLibrary(a.lib); rng = random.Random(0); tasks = []
    while len(tasks) < a.n:
        k = rng.randint(2, min(15, len(lib.cells) - 1)); cells = rng.sample(lib.cells, k + 1); target, occ = cells[0], sorted(cells[1:])
        tasks.append((target, occ))
    rows = []
    with mp.get_context("spawn").Pool(a.workers, initializer=init, initargs=(a.domain,)) as pool:
        for r in pool.imap_unordered(work, tasks):
            r["library"] = lib.feasible(r["target"], r["occ"]); rows.append(r)
    json.dump(rows, open(os.path.join(os.environ.get("AUDIT_CWD", "."), a.out), "w"))
    agree = sum(r["library"] == r["online"] for r in rows)
    print("states %d: agree %d | library-only feasible (UNSOUND) %d | online-only feasible (conservative) %d | feasible online %d"
          % (len(rows), agree, sum(r["library"] and not r["online"] for r in rows), sum(r["online"] and not r["library"] for r in rows),
             sum(r["online"] for r in rows)), flush=True)


if __name__ == "__main__":
    main()
