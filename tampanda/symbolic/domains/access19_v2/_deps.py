"""Native dependencies of the Access-19 v2 motion stack, imported on demand.

* ``frantik`` -- closed-form 7-DoF Franka IK (He & Liu 2021), parametrised by joint 7.
* ``vamp``    -- SIMD sphere-model collision checking + RRT-Connect (vamp-planner 0.6.4).

Neither is installed into the shared conda env: frantik's wheel vendors its own numpy, and a pip install would
replace the env's. Both live in ``pip install --target`` directories instead. Set ``A19V2_PYLIB`` to an
``os.pathsep``-separated list of such directories; without it the cluster defaults below are used if present.
The directories are APPENDED to ``sys.path`` so the env's own packages (numpy, mujoco) always win.

Only the library BUILD and the renderer need these; the feasibility checker (``checker.py``) is pure Python.
"""
from __future__ import annotations

import importlib
import os
import sys

_DEFAULT_DIRS = ("/work/rleap1/daniel.swoboda/a19v2/pylib", "/work/rleap1/daniel.swoboda/a19v2/pylib_vamp")


def _extend_path() -> None:
    env = [p for p in os.environ.get("A19V2_PYLIB", "").split(os.pathsep) if p]
    for p in env or [d for d in _DEFAULT_DIRS if os.path.isdir(d)]:
        if p not in sys.path:
            sys.path.append(p)


def load(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        _extend_path()
        try:
            return importlib.import_module(name)
        except ImportError as e:
            raise ImportError(
                f"access19_v2 needs '{name}' (frantik = analytic Franka IK, vamp = vamp-planner 0.6.4). Install it "
                f"with `pip install --target <dir> ...` and point A19V2_PYLIB at <dir>; see access19_v2/README.md."
            ) from e
