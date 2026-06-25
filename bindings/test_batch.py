"""test_batch.py. Smoke test and example for the gpu-box2d Python binding.

Drives a batch of independent worlds through the binding, steps them, and checks the
read-back state. It exercises the generic obs/state API: seed bodies and the static
boundary, step, then read positions, angles, velocities, awake flags, and body counts
as numpy arrays.

Run after building the extension (see bindings/setup.py):
    python bindings/test_batch.py
Exit code 0 means every check passed.
"""
import sys
import numpy as np
import gpu_box2d as gb


def approx(a, b, tol=1e-3):
    return abs(a - b) <= tol


def main():
    fails = 0

    # --- a batch of circles dropping onto a flat floor ----------------------
    NW = 8
    batch = gb.Batch(NW)
    assert batch.n_worlds == NW
    assert batch.max_bodies >= 2

    radius = 0.5
    for w in range(NW):
        batch.set_ground_edge(w, 0, -10.0, 0.0, 10.0, 0.0)
        # one dynamic circle per world, dropped from y=3 with a small x offset
        slot = batch.add_circle(w, 0.05 * w, 3.0, radius, 1.0, 2.0, gb.DYNAMIC_BODY)
        assert slot == 1, "first added body should be slot 1 (slot 0 is the ground)"

    body_count = batch.body_count()
    assert body_count.shape == (NW,)
    assert np.all(body_count == 2), "each world has the ground plus one circle"

    for _ in range(300):
        batch.step(1)

    pos = batch.positions()
    assert pos.shape == (NW, batch.max_bodies, 2)
    rest_y = pos[:, 1, 1]   # body slot 1, y coordinate, per world
    if not np.all(np.abs(rest_y - radius) < 0.05):
        print("FAIL circles did not rest near the floor:", rest_y)
        fails += 1
    else:
        print("ok   %d circles rest near y=%.3f (mean %.4f)" % (NW, radius, float(rest_y.mean())))

    awake = batch.awake()
    assert awake.shape == (NW, batch.max_bodies)
    # after settling, the circles should be asleep
    if np.all(awake[:, 1] == 0):
        print("ok   settled circles are asleep")
    else:
        print("note settled circles still awake (count %d)" % int(awake[:, 1].sum()))

    vel = batch.velocities()
    assert vel.shape == (NW, batch.max_bodies, 3)
    ang = batch.angles()
    assert ang.shape == (NW, batch.max_bodies)
    print("ok   numpy shapes: positions %s angles %s velocities %s awake %s body_count %s"
          % (pos.shape, ang.shape, vel.shape, awake.shape, body_count.shape))

    # --- determinism: two identical batches step to identical state ----------
    def build_and_step(seed_offset):
        b = gb.Batch(4)
        for w in range(4):
            b.set_ground_edge(w, 0, -10.0, 0.0, 10.0, 0.0)
            b.add_circle(w, 0.1 * w + seed_offset, 2.5, 0.5, 1.0, 2.0, gb.DYNAMIC_BODY)
        for _ in range(120):
            b.step(1)
        return b.positions()

    p1 = build_and_step(0.0)
    p2 = build_and_step(0.0)
    if np.array_equal(p1, p2):
        print("ok   identical seeds step to byte-identical state (deterministic)")
    else:
        print("FAIL determinism: identical seeds diverged")
        fails += 1

    # --- a box resting on the floor (polygon path) --------------------------
    if hasattr(batch, "add_box"):
        bb = gb.Batch(1)
        bb.set_ground_edge(0, 0, -10.0, 0.0, 10.0, 0.0)
        bb.add_box(0, 0.0, 3.0, 0.5, 0.5, 1.0, 12.0, gb.DYNAMIC_BODY)
        for _ in range(300):
            bb.step(1)
        box_y = float(bb.positions()[0, 1, 1])
        if approx(box_y, 0.5, tol=0.05):
            print("ok   box rests on the floor at y=%.4f" % box_y)
        else:
            print("FAIL box did not rest near y=0.5 (got %.4f)" % box_y)
            fails += 1

    if fails == 0:
        print("PASS gpu_box2d python binding: all checks passed")
        return 0
    print("FAIL gpu_box2d python binding: %d check(s) failed" % fails)
    return 1


if __name__ == "__main__":
    sys.exit(main())
