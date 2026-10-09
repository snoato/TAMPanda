"""Shipped feasibility libraries: Access-19 v2 (T17c), confined shelf (library_v1) and multi-level shelf v2 (M3).

Covers file integrity against scripts/libraries/MANIFEST.json, conflict-set queries (per-cell candidate masks, the
cells every candidate needs empty, subset-minimal conflict sets) and ``feasible(c, S)`` on the stored
library-vs-online-planning validation states (tests/data/library_validation.json). Pure Python: no MuJoCo
simulation, no planner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tampanda.symbolic.domains.access19_v2 import Access19V2Checker, GraspLibrary
from tampanda.symbolic.domains.access19_v2.checker import DEFAULT_LIBRARY as T17C_LIBRARY
from tampanda.symbolic.domains.confined_shelf.honest import DEFAULT_LIBRARY as CONFINED_LIBRARY
from tampanda.symbolic.domains.confined_shelf.honest import HonestLibrary as ConfinedLibrary
from tampanda.symbolic.domains.confined_shelf.honest import solve, verify
from tampanda.symbolic.domains.shelf_access_v2 import DEFAULT_LIBRARY as M3_LIBRARY
from tampanda.symbolic.domains.shelf_access_v2 import HonestLibrary as M3Library

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "scripts" / "libraries" / "MANIFEST.json"
VALIDATION = Path(__file__).resolve().parent / "data" / "library_validation.json"

PATHS = {"access19_v2": T17C_LIBRARY, "confined_shelf": CONFINED_LIBRARY, "shelf_access_v2": M3_LIBRARY}
SIZES = {"access19_v2": (70, 293), "confined_shelf": (34, 141), "shelf_access_v2": (67, 276)}   # cells, candidates
ENVS = sorted(PATHS)

pytestmark = pytest.mark.skipif(not all(p.exists() for p in PATHS.values()), reason="shipped libraries missing")


def _load(env):
    if env == "access19_v2":
        return GraspLibrary.load(PATHS[env])
    return {"confined_shelf": ConfinedLibrary, "shelf_access_v2": M3Library}[env](PATHS[env])


LIBS = {env: _load(env) for env in ENVS} if all(p.exists() for p in PATHS.values()) else {}


# ---------------------------------------------------------------- conflict-set helpers (pure functions of the JSON)
def parse(cell):
    region, ij = cell.split("__")
    i, j = map(int, ij.split("_"))
    return region, i, j


def conflict_sets(env):
    """{cell: [frozenset of conflicting cells, one per candidate]}."""
    meta = LIBS[env].meta
    bits = meta["bits"]
    return {c: [frozenset(bits[k] for k in range(len(bits)) if (int(r["mask"], 16) >> k) & 1) for r in rows]
            for c, rows in meta["cells"].items()}


def must_be_empty(sets):
    """Cells that every candidate of a cell conflicts with."""
    return frozenset.intersection(*sets)


def minimal_sets(sets):
    uniq = set(sets)
    return {s for s in uniq if not any(o < s for o in uniq)}


def cells_of(env):
    return set(LIBS[env].meta["bits"])


def needs_empty(env, cell, other):
    """``other`` blocks ``cell`` on its own: the action is infeasible when only ``other`` is occupied."""
    return not LIBS[env].feasible(cell, [other])


# ---------------------------------------------------------------- integrity
@pytest.mark.skipif(not MANIFEST.exists(), reason="scripts/libraries/MANIFEST.json not present")
def test_manifest_matches_shipped_files():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["release_assets"] == []
    for entry in manifest["files"]:
        path = REPO / entry["path"]
        data = path.read_bytes()
        assert len(data) == entry["bytes"], entry["path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    listed = {REPO / e["path"] for e in manifest["files"]}
    for p in PATHS.values():
        assert p in listed and p.with_suffix(".npz") in listed


@pytest.mark.parametrize("env", ENVS)
def test_library_format_and_size(env):
    meta = LIBS[env].meta
    bits = meta["bits"]
    assert meta["format"] == 1 and meta["margin"] == 0.002
    assert len(bits) == len(set(bits)) and set(meta["cells"]) == set(bits)
    n_cells, n_candidates = SIZES[env]
    assert len(bits) == n_cells and sum(len(v) for v in meta["cells"].values()) == n_candidates
    for cell, rows in meta["cells"].items():
        assert rows, cell
        for r in rows:
            m, g, t = (int(r[k], 16) for k in ("mask", "grasp_mask", "transit_mask"))
            assert m == g | t and m >> len(bits) == 0
            assert not (m >> bits.index(cell)) & 1, "a candidate conflicts with its own cell"


def test_access19_v2_library_matches_its_geometry():
    from tampanda.symbolic.domains.access19_v2 import T17C
    assert LIBS["access19_v2"].meta["digest"] == T17C.digest() == "c08ead837b1a"


# ---------------------------------------------------------------- conflict-set queries
@pytest.mark.parametrize("env", ENVS)
def test_every_cell_feasible_on_empty_shelf(env):
    lib = LIBS[env]
    assert all(lib.feasible(c, []) for c in lib.meta["bits"])


@pytest.mark.parametrize("env", ENVS)
def test_nothing_behind_and_no_other_region_blocks(env):
    for cell, sets in conflict_sets(env).items():
        r, _, j = parse(cell)
        for s in sets:
            for o in s:
                r2, _, j2 = parse(o)
                assert r2 == r and j2 <= j, (cell, o)


@pytest.mark.parametrize("env", ENVS)
def test_same_row_neighbours_block_each_other(env):
    cells = cells_of(env)
    for cell, sets in conflict_sets(env).items():
        r, i, j = parse(cell)
        for di in (-1, 1):
            n = "%s__%d_%d" % (r, i + di, j)
            if n in cells:
                assert n in must_be_empty(sets), (cell, n)
                assert needs_empty(env, cell, n)


@pytest.mark.parametrize("env", ENVS)
def test_conflict_masks_equal_set_queries(env):
    """Every candidate is clear exactly when its own conflict set is empty, and feasible() is the OR over them."""
    lib = LIBS[env]
    for cell, sets in conflict_sets(env).items():
        for s in sets:
            assert lib.feasible(cell, sorted(s)) == any(not (o & s) for o in sets)
            assert lib.feasible(cell, sorted(cells_of(env) - s - {cell}))


def test_confined_library_is_a_per_cell_rule():
    """Confined: one subset-minimal conflict set per cell (= what every candidate needs empty), equal to the rule
    'both same-row neighbours, every cell in front in columns i-1..i+1, and for the back-row cells 3_3-7_3 also the
    front cell two columns to the left'."""
    cells = cells_of("confined_shelf")
    for cell, sets in conflict_sets("confined_shelf").items():
        r, i, j = parse(cell)
        rule = {"%s__%d_%d" % (r, i + di, j) for di in (-1, 1)}
        rule |= {"%s__%d_%d" % (r, i + di, jj) for di in (-1, 0, 1) for jj in range(j)}
        if j == 3 and 3 <= i <= 7:
            rule.add("%s__%d_0" % (r, i - 2))
        rule &= cells
        assert minimal_sets(sets) == {must_be_empty(sets)}, cell
        assert must_be_empty(sets) == rule, cell


def test_m3_floor_cells_blocked_only_by_their_own_column_in_front():
    cells = cells_of("shelf_access_v2")
    for cell, sets in conflict_sets("shelf_access_v2").items():
        r, i, j = parse(cell)
        if not r.startswith("floor"):
            continue
        front = {"%s__%d_%d" % (r, i, jj) for jj in range(j)} & cells
        assert must_be_empty(sets) == front and frozenset().union(*sets) <= front, cell


def test_m3_deck_cells_need_the_band_in_front():
    cells = cells_of("shelf_access_v2")
    for cell, sets in conflict_sets("shelf_access_v2").items():
        r, i, j = parse(cell)
        if r.startswith("floor"):
            continue
        band = {"%s__%d_%d" % (r, i + di, jj) for di in (-1, 0, 1) for jj in range(j)} & cells
        assert band <= must_be_empty(sets), cell


def test_m3_candidate_choice_cells():
    """Only middle_deck__3_3 and top_deck__3_3 have two subset-minimal conflict sets: the action is feasible if
    either is clear, and infeasible once both are hit."""
    lib = LIBS["shelf_access_v2"]
    choice = {c: minimal_sets(s) for c, s in conflict_sets("shelf_access_v2").items() if len(minimal_sets(s)) > 1}
    assert set(choice) == {"middle_deck__3_3", "top_deck__3_3"}
    for cell, mins in choice.items():
        a, b = sorted(mins, key=sorted)
        assert lib.feasible(cell, sorted(b - a)) and lib.feasible(cell, sorted(a - b))
        assert not lib.feasible(cell, sorted((a - b) | (b - a)))


def test_m3_excluded_ood_problems_deadlock():
    """The mechanism that makes eval_ood_0046 / 0049 unsolvable under the library (camera-ready specs, sec. 1.2)."""
    def md(ij):
        return "middle_deck__" + ij

    # 0046: 0_0 and 1_0 pin each other; 2_1 needs 1_0 empty; the OOI on 4_3 needs 2_1 empty
    assert needs_empty("shelf_access_v2", md("0_0"), md("1_0")) and needs_empty("shelf_access_v2", md("1_0"), md("0_0"))
    assert needs_empty("shelf_access_v2", md("2_1"), md("1_0")) and needs_empty("shelf_access_v2", md("4_3"), md("2_1"))
    # 0049: 0_1 and 1_1 pin each other; the OOI on 3_4 needs 1_1 empty
    assert needs_empty("shelf_access_v2", md("0_1"), md("1_1")) and needs_empty("shelf_access_v2", md("1_1"), md("0_1"))
    assert needs_empty("shelf_access_v2", md("3_4"), md("1_1"))


def test_t17c_band_in_front():
    """Interior: every cell in front within +-2 columns blocks, and for columns 3-4 at rows >= 2 also the front cell
    three columns to the left (e.g. interior 4_6 needs 1_0-1_4). Top deck: within +-1 column everywhere."""
    cells = cells_of("access19_v2")
    sets = conflict_sets("access19_v2")
    for cell, s in sets.items():
        r, i, j = parse(cell)
        width = 2 if r == "shelf_interior" else 1
        band = {"%s__%d_%d" % (r, i + di, jj) for di in range(-width, width + 1) for jj in range(j)} & cells
        assert band <= must_be_empty(s), cell
        if r == "shelf_interior" and i in (3, 4) and j >= 2:
            assert "%s__%d_0" % (r, i - 3) in must_be_empty(s), cell
    col1 = {"shelf_interior__1_%d" % jj for jj in range(7)}
    assert must_be_empty(sets["shelf_interior__4_6"]) & col1 == {"shelf_interior__1_%d" % jj for jj in range(5)}


# ---------------------------------------------------------------- feasible(c, S) on the stored validation states
VAL = json.loads(VALIDATION.read_text()) if VALIDATION.exists() else {}


def _rows(env):
    bits = LIBS[env].meta["bits"]
    for cell, occ_hex, online in VAL[env]["rows"]:
        m = int(occ_hex, 16)
        yield cell, [bits[k] for k in range(len(bits)) if (m >> k) & 1], online


@pytest.mark.skipif(not VAL, reason="tests/data/library_validation.json missing")
@pytest.mark.parametrize("env", ENVS)
def test_feasible_matches_online_planning_on_validation_states(env):
    entry = VAL[env]
    assert entry["library_sha256"] == hashlib.sha256(PATHS[env].read_bytes()).hexdigest()
    lib = LIBS[env]
    rows = list(_rows(env))
    assert len(rows) == entry["states"]
    disagree = [(c, occ, online) for c, occ, online in rows if lib.feasible(c, occ) != online]
    assert not disagree, disagree[:3]
    assert sum(online for _, _, online in rows) == entry["online_feasible"]


@pytest.mark.skipif(not VAL, reason="tests/data/library_validation.json missing")
def test_validation_counts_as_reported():
    """700/700 (164 feasible), 240/240 (68), 240/240 (123): the numbers quoted in the README."""
    assert {e: (VAL[e]["states"], VAL[e]["online_feasible"]) for e in ENVS} == {
        "access19_v2": (700, 164), "confined_shelf": (240, 68), "shelf_access_v2": (240, 123)}


@pytest.mark.skipif(not VAL, reason="tests/data/library_validation.json missing")
@pytest.mark.parametrize("env", ENVS)
def test_check_action_pick_put_agree_on_validation_states(env):
    """The checker entry points agree with feasible(), and pick and put of a cell have the same verdict."""
    lib = LIBS[env]
    checker = Access19V2Checker(lib) if env == "access19_v2" else lib
    for n, (cell, occ, online) in enumerate(_rows(env)):
        layout = {"o%d" % k: c for k, c in enumerate(occ)}
        pick = checker.check_action({**layout, "target": cell}, None, ("pick", "target", cell))
        put = checker.check_action(layout, "target", ("put", "target", cell))
        assert pick == put == online, (env, n)
        if env != "access19_v2":
            assert lib.feasible_mask(cell, lib.occ(occ)) == online


def test_unknown_cell_is_infeasible():
    for env in ("confined_shelf", "shelf_access_v2"):
        assert not LIBS[env].feasible("no_such_region__0_0", [])


# ---------------------------------------------------------------- confined oracle on the library (pure Python)
def test_confined_oracle_plans_verify():
    lib = LIBS["confined_shelf"]
    init = {"a": "shelf_interior__0_0", "b": "shelf_interior__0_1"}
    goal = {"b": "shelf_interior__2_0"}
    blocked = [("pick", "b", "shelf_interior__0_1"), ("put", "b", "shelf_interior__2_0")]
    assert not verify(lib, init, goal, blocked)          # a, in front of b in its column, blocks the pick
    plan = solve(lib, init, goal, time_limit=30.0)
    assert plan is not None and verify(lib, init, goal, plan)
