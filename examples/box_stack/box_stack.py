"""box_stack.py. A stack of boxes settling, driven through the gpu-box2d Python binding.

This example shows the general engine outside the fruit-merge game: it builds a tower of
boxes in every world of a batch, steps the batch, and reports the settled heights. It
uses only the generic binding API (the static boundary, box bodies, the step, and the
numpy state read-back), so it is a worked instance of driving the engine from Python for
a non-game scene.

Run after building the binding (see bindings/README.md):
    python examples/box_stack/box_stack.py
"""
import sys
import numpy as np
import gpu_box2d as gb


def main():
    n_worlds = 64
    n_boxes = 5
    half = 0.5             # box half-extent
    gap = 0.02             # initial gap between boxes so they settle into contact

    batch = gb.Batch(n_worlds)
    box_slots = []
    for w in range(n_worlds):
        batch.set_ground_edge(w, 0, -10.0, 0.0, 10.0, 0.0)
        slots = []
        for i in range(n_boxes):
            y = half + i * (2 * half + gap)
            s = batch.add_box(w, 0.0, y, half, half, inv_mass=1.0, inv_i=6.0,
                              body_type=gb.DYNAMIC_BODY)
            slots.append(s)
        box_slots.append(slots)

    for _ in range(600):
        batch.step(1)

    pos = batch.positions()           # [n_worlds, max_bodies, 2]
    # heights of the stack in world 0, bottom to top
    heights = [float(pos[0, s, 1]) for s in box_slots[0]]
    print("world 0 settled box heights (bottom -> top):")
    for i, h in enumerate(heights):
        print("  box %d  y = %.4f" % (i, h))

    # every world should produce the same stack (deterministic, identical seeds)
    all_pos = pos[:, box_slots[0], 1]   # [n_worlds, n_boxes]
    spread = float(all_pos.std(axis=0).max())

    # the stack should be ordered bottom to top and roughly evenly spaced
    ordered = all(heights[i] < heights[i + 1] for i in range(n_boxes - 1))
    spacing = np.diff(heights)
    even = float(spacing.std()) < 0.05

    print("\nworlds agree (max std across worlds): %.6f" % spread)
    print("stack ordered bottom-to-top: %s" % ordered)
    print("spacing roughly even (std %.4f): %s" % (float(spacing.std()), even))

    ok = ordered and even and spread < 1e-5
    if ok:
        print("\nPASS box_stack: a %d-box tower settles, all %d worlds agree" % (n_boxes, n_worlds))
        return 0
    print("\nFAIL box_stack: stack did not settle as expected")
    return 1


if __name__ == "__main__":
    sys.exit(main())
