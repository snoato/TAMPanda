"""Access-19 v2: physically honest Access-19 (centred bookcase, 5 x 7 boxes per level, one frontal grasp).

Public API (see README.md):

* ``ShelfGeometry``, ``T17C``, ``cell_id``, ``parse_cell_id``   -- geometry and cell naming (layout.py)
* ``GraspLibrary``, ``Access19V2Checker``                        -- feasibility, pure Python (checker.py)
* ``DOMAIN_NAME``, ``problem_text``, ``static_facts``            -- PDDL (problem.py, pddl/)

Building / executing / validating libraries needs MuJoCo + frantik + vamp (library.py, execute.py, validate.py);
importing this package does not. The other modules in this directory are the research prototypes the geometry was
found with (crowd.py, transit.py, render_v2plan.py, ...); they are not API.
"""
from tampanda.symbolic.domains.access19_v2.checker import DEFAULT_LIBRARY, Access19V2Checker, GraspLibrary
from tampanda.symbolic.domains.access19_v2.layout import T17C, ShelfGeometry, cell_id, cell_keys, parse_cell_id
from tampanda.symbolic.domains.access19_v2.problem import DOMAIN_NAME, PDDL_DIR, problem_text, static_facts
