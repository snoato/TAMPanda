"""Access-19 v2: checker semantics on the shipped T17c library, geometry and PDDL writer (no MuJoCo needed)."""
import pytest

from tampanda.symbolic.domains.access19_v2 import (DEFAULT_LIBRARY, DOMAIN_NAME, T17C, Access19V2Checker,
                                                   GraspLibrary, ShelfGeometry, cell_id, parse_cell_id, problem_text)

lib = GraspLibrary.load(DEFAULT_LIBRARY) if DEFAULT_LIBRARY.exists() else None
needs_lib = pytest.mark.skipif(lib is None, reason="shipped library missing")


def test_cell_ids_roundtrip():
    for key in [("interior", 0, 0), ("top", 4, 6)]:
        assert parse_cell_id(cell_id(key)) == key
    with pytest.raises(ValueError):
        parse_cell_id("table__0_0")


def test_geometry_kv_and_digest():
    g = ShelfGeometry.from_kv("px=0.065 x0=-0.13 box=1 depths=0.007,0.02")
    assert g.px == 0.065 and g.grasp_depth == 0.007 and g.digest() != T17C.digest()
    assert abs(T17C.inner_width() - 0.39) < 1e-9
    with pytest.raises(KeyError):
        ShelfGeometry.from_kv("nonsense=1")


@needs_lib
def test_library_matches_t17c():
    assert lib.meta["digest"] == T17C.digest()
    assert len(lib.cells) == 70 and all(len(lib.masks[c]) > 0 for c in lib.cells)


@needs_lib
def test_every_cell_feasible_on_empty_shelf():
    assert all(lib.feasible(c, []) for c in lib.cells)


@needs_lib
def test_same_row_neighbours_block_each_other():
    for lv in ("interior", "top"):
        for j in range(T17C.nrow):
            for i in range(T17C.ncol - 1):
                a, b = cell_id((lv, i, j)), cell_id((lv, i + 1, j))
                assert not lib.feasible(a, [b]) and not lib.feasible(b, [a])


@needs_lib
def test_front_blocks_back_never_the_reverse():
    for lv in ("interior", "top"):
        for i in range(T17C.ncol):
            front, back = cell_id((lv, i, 0)), cell_id((lv, i, 6))
            assert not lib.feasible(back, [front])
            assert lib.feasible(front, [back])


@needs_lib
def test_other_level_never_blocks():
    top_all = [cell_id(("top", i, j)) for i in range(5) for j in range(7)]
    assert all(lib.feasible(cell_id(("interior", i, j)), top_all) for i in range(5) for j in range(7))


@needs_lib
def test_checker_pick_put_symmetry():
    ck = Access19V2Checker(lib)
    layout = {"a": "shelf_interior__2_0", "b": "shelf_interior__0_6"}
    pick = ck.check_action(layout, None, ("pick", "b", "shelf_interior__0_6"))
    put = ck.check_action({"a": "shelf_interior__2_0"}, "b", ("put", "b", "shelf_interior__0_6"))
    assert pick == put


def test_problem_text():
    txt = problem_text("t", {"ooi": "shelf_interior__2_6"}, {"ooi": "shelf_top__0_0"})
    assert "(:domain %s)" % DOMAIN_NAME in txt and "(occupied shelf_interior__2_6 ooi)" in txt
    assert txt.count(" - cell") == 1 and "shelf_top__4_6" in txt
    with pytest.raises(ValueError):
        problem_text("t", {"a": "shelf_interior__0_0", "b": "shelf_interior__0_0"}, {})
