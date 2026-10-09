"""Native dependencies of the Access-19 v2 motion stack, imported on demand.

* ``frantik`` -- closed-form 7-DoF Franka IK (He & Liu 2021), parametrised by joint 7.
* ``vamp``    -- SIMD sphere-model collision checking + RRT-Connect (vamp-planner 0.6.4).

Neither is installed into the shared environment: frantik's wheel vendors its own numpy, and a plain pip install
would replace the environment's. Install both with ``pip install --target <dir>`` and set ``A19V2_PYLIB`` to an
``os.pathsep``-separated list of those directories. ``A19V2_PYLIB`` is the only place searched besides ``sys.path``;
there is no default location. The directories are APPENDED to ``sys.path``, so the environment's own packages
(numpy, mujoco) always win.

Users: the library build (``library.py``, ``scripts/libraries/``), ``validate.py``, ``online.py`` and ``execute.py``
(through ``kinematics.py`` and ``motion.py``), and the 3D-stacking checks ``multilevel_blocks/honest_aik.py``,
``honest_lib.py`` and ``honest_grip.py``. The feasibility checker (``checker.py``) is pure Python and needs neither.
"""
from __future__ import annotations

import importlib
import os
import sys

_WHAT = {
    "frantik": ("analytic Franka IK (frantik 0.1, He & Liu 2021)",
                "pip install --target <dir> <frantik wheel or source checkout>   # github.com/CoMMALab/frantik"),
    "vamp": ("vamp-planner 0.6.4 (sphere-model collision checking + RRT-Connect)",
             "pip install --target <dir> vamp-planner==0.6.4"),
}


def _pylib_dirs() -> list:
    return [p for p in os.environ.get("A19V2_PYLIB", "").split(os.pathsep) if p]


def _extend_path() -> None:
    for p in _pylib_dirs():
        if p not in sys.path:
            sys.path.append(p)


def load(name: str):
    """Import ``name`` (``frantik`` or ``vamp``), searching the ``A19V2_PYLIB`` directories after ``sys.path``."""
    try:
        return importlib.import_module(name)
    except ImportError:
        _extend_path()
        try:
            return importlib.import_module(name)
        except ImportError as e:
            what, how = _WHAT.get(name, (name, f"pip install --target <dir> {name}"))
            where = (f"A19V2_PYLIB={os.environ['A19V2_PYLIB']!r} does not contain it" if _pylib_dirs()
                     else "A19V2_PYLIB is not set")
            raise ImportError(
                f"access19_v2 needs the module '{name}', {what}, but it is not importable and {where}. "
                f"Install it with\n    {how}\nthen set A19V2_PYLIB=<dir> (several directories separated by "
                f"{os.pathsep!r}). See tampanda/symbolic/domains/access19_v2/README.md, \"Dependencies\"."
            ) from e
