from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def _plan_actions(plan_path: Path) -> list[str]:
    return [ln.strip() for ln in plan_path.read_text().splitlines() if ln.strip().startswith("(")]


def _extract_object_count(plan_path: Path) -> int | None:
    for line in plan_path.read_text().splitlines():
        s = line.strip()
        if s.lower().startswith("; objects:"):
            payload = s.split(":", 1)[1].strip()
            if not payload:
                return 0
            return len([x.strip() for x in payload.split(",") if x.strip()])
    return None


def _extract_num_removals(plan_path: Path) -> int | None:
    """Return the number of cylinders removed before the target.

    Falls back to None if the comment is absent (older datasets).
    In that case callers should use plan_length - 1 as an approximation.
    """
    for line in plan_path.read_text().splitlines():
        s = line.strip().lower()
        if s.startswith("; num removals before target:"):
            try:
                return int(s.split(":", 1)[1].strip())
            except Exception:
                return None
    return None


def _copy_pairs(domain_path: Path, pairs: list[tuple[Path, Path]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(domain_path, out_dir / "domain.pddl")
    for problem_path, plan_path in pairs:
        shutil.copy2(problem_path, out_dir / problem_path.name)
        shutil.copy2(plan_path, out_dir / plan_path.name)


def _infer_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for plan_path in sorted(source_dir.glob("*.pddl.plan")):
        problem_path = source_dir / plan_path.name[:-5]
        if problem_path.exists():
            pairs.append((problem_path, plan_path))
    return pairs


def _resolve_domain(source_dir: Path) -> Path:
    local_domain = source_dir / "domain.pddl"
    parent_domain = source_dir.parent / "domain.pddl"
    if local_domain.exists():
        return local_domain
    if parent_domain.exists():
        return parent_domain
    raise FileNotFoundError(
        f"Missing domain.pddl in {source_dir} and {source_dir.parent}"
    )


def _select(pairs: list[tuple[Path, Path]], predicate, n: int, rng: random.Random) -> list[tuple[Path, Path]]:
    filtered = [pair for pair in pairs if predicate(pair)]
    if len(filtered) <= n:
        return filtered
    rng.shuffle(filtered)
    return filtered[:n]


def _is_occluded(pair: tuple[Path, Path]) -> bool:
    """True if the target required at least one prior removal.

    Uses the explicit '; num removals before target:' comment when present.
    Falls back to plan_length > 1 for older datasets that lack the comment.
    """
    _, plan_path = pair
    removals = _extract_num_removals(plan_path)
    if removals is not None:
        return removals > 0
    # Fallback: treat any multi-step plan as occluded
    return len(_plan_actions(plan_path)) > 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validation suites from a base dataset split.")
    parser.add_argument("--source", required=True, type=Path, help="Source directory with .pddl and .pddl.plan files")
    parser.add_argument("--output_root", required=True, type=Path, help="Output root for eval suite directories")
    parser.add_argument("--per_suite", default=20, type=int, help="Max samples per suite")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--small_grid_source",
        default=None,
        type=Path,
        help="Optional source dir for val_small_grid suite (expects same file layout)",
    )
    args = parser.parse_args()

    source = args.source
    domain_path = _resolve_domain(source)

    rng = random.Random(args.seed)
    pairs = _infer_pairs(source)
    if not pairs:
        raise RuntimeError(f"No .pddl/.pddl.plan pairs found in {source}")

    def obj_count_eq(k: int):
        return lambda pair: (_extract_object_count(pair[1]) == k)

    def plan_len_eq(k: int):
        return lambda pair: (len(_plan_actions(pair[1])) == k)

    suites: dict[str, list[tuple[Path, Path]]] = {
        "val_obj_3":          _select(pairs, obj_count_eq(3), args.per_suite, rng),
        "val_obj_5":          _select(pairs, obj_count_eq(5), args.per_suite, rng),
        "val_obj_7":          _select(pairs, obj_count_eq(7), args.per_suite, rng),
        "val_plan_1":         _select(pairs, plan_len_eq(1), args.per_suite, rng),
        "val_plan_3":         _select(pairs, plan_len_eq(3), args.per_suite, rng),
        "val_plan_5plus":     _select(pairs, lambda pair: len(_plan_actions(pair[1])) >= 5, args.per_suite, rng),
        "val_target_occluded": _select(pairs, _is_occluded, args.per_suite, rng),
    }

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    for name, selected in suites.items():
        suite_dir = out_root / name
        _copy_pairs(domain_path, selected, suite_dir)
        print(f"{name}: {len(selected)} examples -> {suite_dir}")

    if args.small_grid_source is not None:
        small = args.small_grid_source
        small_domain = _resolve_domain(small)
        small_pairs = _infer_pairs(small)
        selected_small = small_pairs if len(small_pairs) <= args.per_suite else _select(small_pairs, lambda _: True, args.per_suite, rng)
        suite_dir = out_root / "val_small_grid"
        _copy_pairs(small_domain, selected_small, suite_dir)
        print(f"val_small_grid: {len(selected_small)} examples -> {suite_dir}")


if __name__ == "__main__":
    main()
