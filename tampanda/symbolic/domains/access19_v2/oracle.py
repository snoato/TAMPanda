"""Access-19 v2 oracle planner: greedy best-first search over pick+put moves, checked by the grasp library.

Data generation needs SOME verified plan per problem (solvability check, prefix curricula, reload hand-offs). v1 had
a hand-tuned phase solver for its 7-column deck; v2's feasibility is a microsecond lookup, so a general search is
affordable instead.

* A move = ``pick o`` + ``put o t`` (the hand is never left holding between moves); it is applicable iff the pick
  of o's cell and the put into t are both feasible with the other objects where they are (``GraspLibrary``).
* Heuristic: every object not at its goal costs one move; every object that sits where it blocks a NEEDED motion
  (the pick of a misplaced object, or the put into a misplaced object's goal, judged by the cells' smallest
  conflict sets) costs one more -- two if it is at its own goal, since it has to leave and come back. On the
  canonical 18-blocker return task this is 37 at the start, close to the true move count.
* Greedy best-first on (h, deeper destination first, g), duplicate detection, "do not move the same object twice in a row" (that pair is a
  single move), node and time limits. Plans are re-verified move by move.

    from tampanda.symbolic.domains.access19_v2.oracle import solve
    plan = solve(lib, init={obj: cell_id}, goal={obj: cell_id})     # [("pick", o, c), ("put", o, t), ...] or None
"""
from __future__ import annotations

import heapq
import time

from tampanda.symbolic.domains.access19_v2.checker import GraspLibrary


class Planner:
    def __init__(self, lib: GraspLibrary):
        self.lib = lib; self.cells = lib.cells; self.idx = {c: n for n, c in enumerate(self.cells)}
        self.row = [int(c.rsplit("_", 1)[1]) for c in self.cells]
        self.masks = [lib.masks[c] for c in self.cells]
        self.mmin = [min(ms, key=lambda m: bin(m).count("1")) for ms in self.masks]

    def feasible(self, c: int, others: int) -> bool:
        for m in self.masks[c]:
            if not m & others:
                return True
        return False

    def h(self, S, goal, occ):
        need = 0; mis = 0
        for o, gc in goal.items():
            c = S[o]
            if c != gc:
                mis += 1
                need |= self.mmin[c] | self.mmin[gc] | (1 << gc)
        if not mis:
            return 0
        extra = 0
        for o, c in enumerate(S):
            if need >> c & 1:
                gc = goal.get(o)
                extra += 2 if gc == c else 1
        return mis + extra

    def solve(self, init: dict, goal: dict, max_nodes=300_000, time_limit=120.0, mode="gbfs_depth", w=3.0,
              max_states=4_000_000, allowed=None):
        # cells the problem declares (v1's reduced-depth problems expose only some rows): the only destinations
        self.allowed = [n for n, c in enumerate(self.cells) if allowed is None or c in set(allowed)]
        self.max_states = max_states   # generated-state cap: bounds memory (~0.5 KB per stored state incl. bookkeeping)
        if mode == "ehc":
            return self.solve_ehc(init, goal, max_nodes, time_limit)
        if mode == "bfws":
            return self.solve_bfws(init, goal, max_nodes, time_limit)
        objs = sorted(init); oi = {o: n for n, o in enumerate(objs)}
        S0 = bytes(self.idx[init[o]] for o in objs)
        G = {oi[o]: self.idx[c] for o, c in goal.items()}
        ncell = len(self.cells); t0 = time.time()
        occ0 = 0
        for c in S0:
            occ0 |= 1 << c
        start = (S0, -1)
        h0 = self.h(S0, G, occ0)
        heap = [(h0, 0, 0, 0, start)]; parent = {S0: None}; g_of = {S0: 0}; tick = 0; nodes = 0
        while heap:
            _, _, g, _, (S, last) = heapq.heappop(heap)
            if self.h(S, G, sum(1 << c for c in S)) == 0:
                return self._extract(S, parent, objs), {"nodes": nodes, "s": time.time() - t0, "h0": h0}
            nodes += 1
            if nodes > max_nodes or time.time() - t0 > time_limit or len(parent) > self.max_states:
                return None, {"nodes": nodes, "s": time.time() - t0, "h0": h0, "why": "limit"}
            occ = 0
            for c in S:
                occ |= 1 << c
            for o, c in enumerate(S):
                if o == last:
                    continue
                others = occ & ~(1 << c)
                if not self.feasible(c, others):
                    continue
                for t in self.allowed:
                    if (occ >> t & 1) or not self.feasible(t, others):
                        continue
                    S2 = S[:o] + bytes((t,)) + S[o + 1:]
                    if S2 in g_of and g_of[S2] <= g + 1:
                        continue
                    g_of[S2] = g + 1; parent[S2] = (S, o, c, t); tick += 1
                    # ties: prefer the deeper destination -- rows fill back-to-front on both levels (a box in
                    # front blocks every later put behind it), for parking and for reloading alike
                    h2 = self.h(S2, G, others | (1 << t))
                    key = (h2, -self.row[t]) if mode == "gbfs_depth" else ((h2, 0) if mode == "gbfs" else (g + 1 + w * h2, -self.row[t]))
                    heapq.heappush(heap, (key[0], key[1], g + 1, tick, (S2, o)))
        return None, {"nodes": nodes, "s": time.time() - t0, "h0": h0, "why": "exhausted"}

    def _succ(self, S, last):
        occ = 0
        for c in S:
            occ |= 1 << c
        for o, c in enumerate(S):
            if o == last:
                continue
            others = occ & ~(1 << c)
            if not self.feasible(c, others):
                continue
            for t in self.allowed:
                if not (occ >> t & 1) and self.feasible(t, others):
                    yield o, c, t, S[:o] + bytes((t,)) + S[o + 1:], others | (1 << t)

    def solve_bfws(self, init, goal, max_nodes, time_limit):
        """Best-first width search: priority (novelty, h, g). Novelty 1 = the state has an (object, cell) atom never
        seen among generated states with the same h, 2 = a new atom pair, 3 = neither. Escapes plateaus that need
        'unhelpful' detours (parking the target, re-parking boxes) by preferring states that differ in new ways."""
        objs = sorted(init); oi = {o: n for n, o in enumerate(objs)}
        S0 = bytes(self.idx[init[o]] for o in objs); G = {oi[o]: self.idx[c] for o, c in goal.items()}
        t0 = time.time(); nodes = 0; seen1 = {}; seen2 = {}

        def novelty(S, hv):
            a1 = seen1.setdefault(hv, set()); a2 = seen2.setdefault(hv, set())
            atoms = [o * 128 + c for o, c in enumerate(S)]
            new1 = [a for a in atoms if a not in a1]
            w = 3
            if new1:
                w = 1
            pairs = [(atoms[i], atoms[j]) for i in range(len(atoms)) for j in range(i + 1, len(atoms))]
            new2 = [pq for pq in pairs if pq not in a2]
            if w == 3 and new2:
                w = 2
            a1.update(new1); a2.update(new2)
            return w

        h0 = self.h(S0, G, sum(1 << c for c in S0))
        heap = [(novelty(S0, h0), h0, 0, 0, S0, -1)]; parent = {S0: None}; tick = 0
        while heap:
            w, hv, g, _, S, last = heapq.heappop(heap)
            if hv == 0:
                return self._extract(S, parent, objs), {"nodes": nodes, "s": time.time() - t0, "h0": h0}
            nodes += 1
            if nodes > max_nodes or time.time() - t0 > time_limit or len(parent) > self.max_states:
                return None, {"nodes": nodes, "s": time.time() - t0, "h0": h0, "why": "limit"}
            for o, c, t, S2, occ2 in self._succ(S, last):
                if S2 in parent:
                    continue
                parent[S2] = (S, o, c, t); tick += 1
                h2 = self.h(S2, G, occ2)
                heapq.heappush(heap, (novelty(S2, h2), h2, g + 1, tick, S2, o))
        return None, {"nodes": nodes, "s": time.time() - t0, "h0": h0, "why": "exhausted"}

    def solve_ehc(self, init, goal, max_nodes, time_limit):
        """Enforced hill climbing: from the current state, breadth-first search until a strictly better h."""
        from collections import deque
        objs = sorted(init); oi = {o: n for n, o in enumerate(objs)}
        S = bytes(self.idx[init[o]] for o in objs); G = {oi[o]: self.idx[c] for o, c in goal.items()}
        t0 = time.time(); nodes = 0; moves = []; h_cur = self.h(S, G, sum(1 << c for c in S)); h0 = h_cur
        seen_global = {S}
        while h_cur > 0:
            q = deque([(S, -1, [])]); seen = {S}; found = None
            while q and found is None:
                X, last, path = q.popleft(); nodes += 1
                if nodes > max_nodes or time.time() - t0 > time_limit:
                    return None, {"nodes": nodes, "s": time.time() - t0, "h0": h0, "why": "limit", "h_reached": h_cur}
                for o, c, t, X2, occ2 in self._succ(X, last):
                    if X2 in seen:
                        continue
                    seen.add(X2); p2 = path + [(o, c, t)]
                    h2 = self.h(X2, G, occ2)
                    if h2 < h_cur and X2 not in seen_global:
                        found = (X2, p2, h2); break
                    q.append((X2, o, p2))
            if found is None:
                return None, {"nodes": nodes, "s": time.time() - t0, "h0": h0, "why": "plateau", "h_reached": h_cur}
            S, p2, h_cur = found; seen_global.add(S); moves += p2
        plan = []
        for o, c, t in moves:
            plan += [("pick", objs[o], self.cells[c]), ("put", objs[o], self.cells[t])]
        return plan, {"nodes": nodes, "s": time.time() - t0, "h0": h0}

    def _extract(self, S, parent, objs):
        moves = []
        while parent[S] is not None:
            S, o, c, t = parent[S]
            moves.append((objs[o], self.cells[c], self.cells[t]))
        plan = []
        for o, c, t in reversed(moves):
            plan += [("pick", o, c), ("put", o, t)]
        return plan

    def verify(self, init: dict, goal: dict, plan) -> bool:
        layout = dict(init); held = None
        for name, o, c in plan:
            others = [x for p, x in layout.items() if p != o]
            if name == "pick":
                if held is not None or layout.get(o) != c or not self.lib.feasible(c, others):
                    return False
                del layout[o]; held = o
            else:
                if held != o or c in layout.values() or not self.lib.feasible(c, list(layout.values())):
                    return False
                layout[o] = c; held = None
        return held is None and all(layout.get(o) == c for o, c in goal.items())


def solve(lib: GraspLibrary, init: dict, goal: dict, **kw):
    p = Planner(lib); plan, info = p.solve(init, goal, **kw)
    if plan is not None and kw.get("allowed") is not None:
        assert all(c in set(kw["allowed"]) for _, _, c in plan), "plan uses an undeclared cell"
    if plan is not None and not p.verify(init, goal, plan):
        raise AssertionError("oracle produced a plan that does not verify")
    return plan, info
