"""Render looks for the Access-19 v2 scene. Visual only: collision geometry, the checker and the library are untouched.

* ``plain``    -- the scene as compiled from ``base_panda_v2.xml`` (MuJoCo's default black background); the default.
* ``tampanda`` -- the look of every other tampanda environment (``tampanda/scenes/builder.py``): headlight
                  diffuse 0.6 / ambient 0.3, blue gradient skybox with haze, a directional light from 1.5 m above.
                  The floor keeps v2's colour.
* ``teaser``   -- the paper teaser look (``examples/render_final_state.py``): soft grey-to-white studio gradient
                  (seamless here), matching haze, solid light-grey floor; for stills the arm can be ghosted and folded
                  aside (``ghost_arm``). v2's robot stands IN FRONT of the shelf, so the front-quarter view (azimuth
                  125) frames it like the v1 teaser; the teaser strip's azimuth 100 puts the arm in front of the shelf.

    sc = Scene(T17C, look="teaser")       # shelf.Scene applies apply_spec before and apply_model after compiling
"""
from __future__ import annotations

import mujoco
import numpy as np

STYLES = ("plain", "tampanda", "teaser")

# tampanda/scenes/builder.py
TAMPANDA = dict(headlight_diffuse=(0.6, 0.6, 0.6), headlight_ambient=(0.3, 0.3, 0.3), haze=(0.15, 0.25, 0.35, 1.0),
                sky_rgb1=(0.3, 0.5, 0.7), sky_rgb2=(0.0, 0.0, 0.0))
# examples/render_final_state.py (paper Fig. 1 teaser strip)
TEASER = dict(floor=(0.66, 0.66, 0.68, 1.0), sky_top=(0.60, 0.64, 0.70), sky_bottom=(0.93, 0.93, 0.95),
              haze=(0.93, 0.93, 0.95, 1.0), arm_alpha=0.22,
              azimuth=100.0, elevation=-18.0)              # the shared teaser camera angle (lookat/distance per scene)
TUCKED_QPOS = np.array([1.4, -1.7, 0.0, -2.2, 0.0, 0.9, 0.785])       # render_final_state (v1 shelf beside the robot)
TUCKED_SIDE = np.array([-0.2, -1.7, 0.0, -2.2, 0.0, 0.9, 0.785])      # v2: the shelf is IN FRONT (+y); fold the arm sideways


def apply_spec(spec, style, floor_geom=None):
    """Before compiling: skybox texture, light, headlight, haze (and the teaser floor colour)."""
    if style in (None, "plain"):
        return
    if style not in STYLES:
        raise ValueError("unknown look %r (choose from %s)" % (style, STYLES))
    v = spec.visual
    v.headlight.diffuse = list(TAMPANDA["headlight_diffuse"]); v.headlight.ambient = list(TAMPANDA["headlight_ambient"])
    v.headlight.specular = [0.0, 0.0, 0.0]
    v.rgba.haze = list(TAMPANDA["haze"] if style == "tampanda" else TEASER["haze"])
    t = spec.add_texture(); t.name = "skybox"; t.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    t.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT; t.width = 512; t.height = 3072
    if style == "tampanda":
        t.rgb1 = list(TAMPANDA["sky_rgb1"]); t.rgb2 = list(TAMPANDA["sky_rgb2"])
    else:
        # studio dome: MuJoCo's direction-based gradient is seamless from any angle (render_final_state's row-wise
        # rewrite of the 6-face texture only looks right from the teaser's fixed camera); horizon ~ SKY_BOTTOM
        t.rgb1 = list(TEASER["sky_top"]); t.rgb2 = [1.0, 1.0, 1.0]
    light = spec.worldbody.add_light(); light.name = "sun"; light.pos = [0.0, 0.0, 1.5]; light.dir = [0.0, 0.0, -1.0]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL; light.castshadow = True
    if style == "teaser" and floor_geom is not None:
        floor_geom.rgba = list(TEASER["floor"])


def apply_model(m, style):
    """After compiling. Shadow map: MuJoCo sizes a directional light's shadow frustum by the model extent, which here
    includes the boxes parked 30 m away (absent cells) -- so shadows come out blocky; size it to the shelf instead
    (render-only statistics) and raise the map resolution."""
    if style in (None, "plain"):
        return
    m.stat.center[:] = [0.0, 0.4, 0.4]; m.stat.extent = 1.2
    m.vis.quality.shadowsize = 8192
    if style == "teaser":
        m.vis.rgba.haze[:] = TEASER["haze"]


def ghost_arm(sc, alpha=TEASER["arm_alpha"], tuck=True, pose=None):
    """Teaser stills: semi-transparent (and optionally tucked) arm. Visual only."""
    from tampanda.symbolic.domains.access19_v2.shelf import ARM_BODIES
    m, d = sc.m, sc.d
    ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_BODIES} - {-1}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in ids:
            m.geom_rgba[g][3] = alpha
    if tuck:
        d.qpos[sc.jq] = TUCKED_SIDE if pose is None else pose
    mujoco.mj_forward(m, d)


def camera(style, lookat, distance, azimuth=None, elevation=None):
    cam = mujoco.MjvCamera(); cam.lookat[:] = lookat; cam.distance = distance
    cam.azimuth = TEASER["azimuth"] if azimuth is None else azimuth
    cam.elevation = TEASER["elevation"] if elevation is None else elevation
    return cam
