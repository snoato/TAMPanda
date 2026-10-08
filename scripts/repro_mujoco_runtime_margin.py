"""Reproduce the MuJoCo mid-phase / runtime-margin limitation (see README, "Known limitations").

A box hovers 1 mm or 5 mm above a deck. The contact margin (10 mm) is either compiled into the model or raised
afterwards at runtime (on the held geom only, or on every geom). With the mid-phase on (MuJoCo's default), a margin
raised at runtime is ignored whenever either body of the pair has more than one geom (here: the world body with deck
and legs, or the box's body with a second geom), so the near-contact is not reported at all. With the margin
compiled in, or with the mid-phase disabled, it is reported at the correct distance. Penetrating pairs are always
reported (their bounding volumes overlap regardless of the margin).

    python scripts/repro_mujoco_runtime_margin.py      (observed with MuJoCo 3.4.0)
"""
import mujoco

MID = int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)


def main():
    print("mujoco", mujoco.__version__)
    missed = 0
    for gap in (0.001, 0.005):
        for legs in (0, 4):
            for multigeom_body in (False, True):
                for how in ("compile", "runtime_one", "runtime_all"):
                    mg = 'margin="0.01"' if how == "compile" else ""
                    leg_xml = "".join('<geom type="box" size="0.01 0.01 0.15" pos="%f %f 0.15"/>' % (sx * 0.38, sy * 0.2)
                                      for sx, sy in [(1, 1), (-1, 1), (1, -1), (-1, -1)][:legs])
                    extra = '<geom name="g2" type="sphere" size="0.01" pos="0 0 0.06" %s/>' % mg if multigeom_body else ""
                    z = 0.306 + 0.045 + gap                                       # deck top at z = 0.306
                    xml = ('<mujoco><worldbody><geom name="deck" type="box" size="0.4 0.225 0.006" pos="0 0 0.3" %s/>%s'
                           '<body name="b" pos="0.1 0.1 %f"><freejoint/><geom name="held" type="box" '
                           'size="0.03 0.03 0.045" %s/>%s</body></worldbody></mujoco>') % (mg, leg_xml, z, mg, extra)
                    m = mujoco.MjModel.from_xml_string(xml)
                    d = mujoco.MjData(m)
                    if how == "runtime_one":
                        m.geom_margin[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "held")] = 0.01
                    elif how == "runtime_all":
                        m.geom_margin[:] = 0.01
                    out = []
                    for flags in (0, MID):                                        # mid-phase on, then off
                        m.opt.disableflags = flags
                        mujoco.mj_forward(m, d)
                        out.append([round(1000 * c.dist, 2) for c in d.contact[:d.ncon]
                                    if {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)} == {"deck", "held"}])
                    miss = bool(out[1]) and not out[0]
                    missed += miss
                    print("gap %.0f mm, world geoms %d, body geoms %d, margin %-11s: mid-phase on %-24s off %-24s%s"
                          % (1000 * gap, 1 + legs, 2 if multigeom_body else 1, how, out[0], out[1],
                             "  <-- missed with mid-phase on" if miss else ""))
    print("%d configurations miss the near-contact with the mid-phase on" % missed)


if __name__ == "__main__":
    main()
