"""Access-19 v2: blocking stencil (2026-10-01).

For every cell of a region and every relative offset (dx, dy) within +-R cells, one cube is placed at
cell + (dx, dy) and the pick (empty hand, target present) and the put (held cube) at the cell are re-tested over all
frontal grasps (grasp heights x both hand flips x IK seeds; IK paths are cached, so only the collision check reruns).
An offset "blocks" a cell if NO grasp survives. Output: per region, per offset, the fraction of cells it blocks,
plus the per-cell detail.

    python -m tampanda.symbolic.domains.access19_v2.stencil out.json region=shelf_top [R=3] [k=v ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from tampanda.symbolic.domains.access19_v2 import verify as V
from tampanda.symbolic.domains.access19_v2.geometry import V2Params


def main():
    out = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    rname = kv.pop("region"); Rr = int(kv.pop("R", 3)); cells_only = kv.pop("cells", None)
    p = V2Params()
    for k, v in kv.items():
        cur = getattr(p, k)
        setattr(p, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    rig = V.Rig(p); region = rig.ws[rname]
    pitch = p.cube_half[0] * 2 + p.inter_row_gap
    t0 = time.time()
    res = {"params": p.as_dict(), "region": rname, "R": Rr, "cells": {}}
    want = None if not cells_only else {tuple(map(int, c.split("_"))) for c in cells_only.split(";")}
    for cell in region.cells():
        if want and (cell.ix, cell.iy) not in want:
            continue
        seeds = V.seeds_for(rig, region, cell)
        rig.park_all(); rig.place("ooi", rig.cell_xyz(region, cell))
        ok_pick, why_pick, _ = rig.try_grasp(region, cell, seeds)
        rig.park_all()
        ok_put, why_put, _ = rig.try_grasp(region, cell, seeds, held="ooi")
        row = {"pick": ok_pick, "put": ok_put, "blocks_pick": [], "blocks_put": []}
        for dx in range(-Rr, Rr + 1):
            for dy in range(-Rr, Rr + 1):
                nx, ny = cell.ix + dx, cell.iy + dy
                if (dx, dy) == (0, 0) or not region.in_bounds(nx, ny) or region.is_excluded(nx, ny):
                    continue
                off = rig.cell_xyz(region, region.cell(nx, ny))
                if ok_pick:
                    rig.park_all(); rig.place("ooi", rig.cell_xyz(region, cell)); rig.place("blocker_0", off)
                    if not rig.try_grasp(region, cell, seeds)[0]:
                        row["blocks_pick"].append([dx, dy])
                if ok_put:
                    rig.park_all(); rig.place("blocker_0", off)
                    if not rig.try_grasp(region, cell, seeds, held="ooi")[0]:
                        row["blocks_put"].append([dx, dy])
        rig.park_all()
        res["cells"]["%d_%d" % (cell.ix, cell.iy)] = row
        print("%s %d_%d pick=%s put=%s  blocks_pick=%s  blocks_put=%s  (%.0fs)" % (
            rname, cell.ix, cell.iy, ok_pick, ok_put, row["blocks_pick"], row["blocks_put"], time.time() - t0),
            flush=True)
    # aggregate stencil: fraction of tested cells (where the offset is in bounds) blocked by each offset
    agg = {}
    for key, row in res["cells"].items():
        ix, iy = map(int, key.split("_"))
        for dx in range(-Rr, Rr + 1):
            for dy in range(-Rr, Rr + 1):
                nx, ny = ix + dx, iy + dy
                if (dx, dy) == (0, 0) or not region.in_bounds(nx, ny) or region.is_excluded(nx, ny):
                    continue
                a = agg.setdefault("%d,%d" % (dx, dy), {"n": 0, "pick": 0, "put": 0})
                a["n"] += 1
                a["pick"] += [dx, dy] in row["blocks_pick"]
                a["put"] += [dx, dy] in row["blocks_put"]
    res["stencil"] = agg
    for mode in ("pick", "put"):
        print("\n%s stencil (%s): fraction of cells blocked; rows dy=+%d (back) .. -%d (front), cols dx=-%d..+%d" % (
            rname, mode, Rr, Rr, Rr, Rr))
        for dy in range(Rr, -Rr - 1, -1):
            line = []
            for dx in range(-Rr, Rr + 1):
                a = agg.get("%d,%d" % (dx, dy))
                line.append("  [X] " if (dx, dy) == (0, 0) else (" %4.2f" % (a[mode] / a["n"]) if a else "   - "))
            print("dy=%+d " % dy + "".join(line))
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
