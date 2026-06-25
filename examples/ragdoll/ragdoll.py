"""ragdoll.py. A jointed chain (a simple ragdoll limb) driven through the gpu-box2d
Python binding.

This example shows the joints in use outside any game: it builds a chain of box segments
connected by revolute joints, pinned at the top to a static anchor, and lets it swing
under gravity. The segments stay connected at their joints (the anchor distance holds)
while the chain settles, which is the ragdoll and rope behavior the revolute joint
provides.

It uses only the generic binding API (box bodies, revolute joints, the step, and the
numpy read-back), so it is a worked instance of driving the jointed engine from Python.

Run after building the binding (see bindings/README.md):
    python examples/ragdoll/ragdoll.py
"""
import sys
import numpy as np
import gpu_box2d as gb


def main():
    n_worlds = 16
    n_segments = 5
    seg_half_len = 0.4        # half-length of each segment along x
    seg_half_w = 0.1          # half-width
    batch = gb.Batch(n_worlds)
    for w in range(n_worlds):
        # a far-away ground edge so the chain swings freely
        batch.set_ground_edge(w, 0, -100.0, -100.0, 100.0, -100.0)

        # a static anchor box at the top (inv_mass 0 = immovable)
        anchor = batch.add_box(w, 0.0, 0.0, seg_half_w, seg_half_w, 0.0, 0.0,
                               gb.STATIC_BODY)

        slots = [anchor]
        prev = anchor
        prev_right_x = 0.0     # world x of the previous body's right edge / joint point
        for i in range(n_segments):
            # place the segment so its left end meets the previous body's joint point,
            # extending to the right
            cx = prev_right_x + seg_half_len
            seg = batch.add_box(w, cx, 0.0, seg_half_len, seg_half_w, 1.0, 8.0,
                                gb.DYNAMIC_BODY)
            # pin the segment's left end to the previous body's right end with a
            # revolute joint. Anchors are body-local.
            if prev == anchor:
                anchor_a = (0.0, 0.0)                 # anchor box local center
            else:
                anchor_a = (seg_half_len, 0.0)        # previous segment's right end
            anchor_b = (-seg_half_len, 0.0)           # this segment's left end
            batch.add_revolute_joint(w, prev, seg,
                                     anchor_a[0], anchor_a[1], anchor_b[0], anchor_b[1])
            slots.append(seg)
            prev = seg
            prev_right_x = cx + seg_half_len
    seg_slots = slots   # the body slot layout, the same in every world

    # record initial joint separations to confirm they hold
    def joint_gaps(positions):
        gaps = []
        # gap between consecutive segment centers should stay ~ seg_len (rigid links)
        for i in range(1, n_segments):
            a = positions[0, seg_slots[i], :]
            b = positions[0, seg_slots[i + 1], :]
            gaps.append(float(np.linalg.norm(b - a)))
        return gaps

    p0 = batch.positions()
    init_gaps = joint_gaps(p0)

    for _ in range(400):
        batch.step(1)

    p1 = batch.positions()
    final_gaps = joint_gaps(p1)

    # the chain should have swung down: the last segment's y drops well below 0
    tip_y = float(p1[0, seg_slots[-1], 1])
    print("ragdoll chain of %d segments, pinned at the top" % n_segments)
    print("  tip y after settling: %.4f (started at 0, should swing down)" % tip_y)
    print("  link lengths held: initial %s" % ["%.3f" % g for g in init_gaps])
    print("                      final  %s" % ["%.3f" % g for g in final_gaps])

    # links stay near their rest length (the joints hold the segments together)
    links_held = all(abs(f - i) < 0.05 for f, i in zip(final_gaps, init_gaps))
    swung = tip_y < -0.5

    # every world agrees (deterministic)
    tips = p1[:, seg_slots[-1], 1]
    agree = float(tips.std()) < 1e-5

    print("  links held: %s | chain swung down: %s | worlds agree: %s"
          % (links_held, swung, agree))

    if links_held and swung and agree:
        print("\nPASS ragdoll: jointed chain swings and holds its links, all worlds agree")
        return 0
    print("\nFAIL ragdoll: chain did not behave as expected")
    return 1


if __name__ == "__main__":
    sys.exit(main())
