# gpu-box2d Python binding

Drive N independent, bit-faithful Box2D 2.3.0 worlds from Python and read per-world body state as numpy arrays. The binding is game-agnostic: it speaks bodies, joints, the static boundary, and the step, with no application logic baked in. An RL or simulation layer seeds worlds, steps them, and reads the state out as tensors.

The host build steps on the CPU and is bit-identical to a single-threaded host Box2D 2.3.0. The same driver compiles for CUDA and steps the same seeded state on the GPU through the SoA-global production path, so the API a Python user drives on a CPU drives the GPU the same way.

## Install

The extension is a small pybind11 module over the header-only core. Build it with the project on the include path:

```
pip install ./bindings
```

or build in place for a quick check:

```
python bindings/setup.py build_ext --inplace
```

The build sets the fidelity flags (`-ffp-contract=off`, and `-mfpmath=sse` on x86) so the host result matches the CPU Box2D reference. It needs `pybind11` and `numpy`.

## API

Everything lives on `gpu_box2d.Batch`. Slot 0 of every world is the static ground body; bodies you add take slots 1, 2, and up.

```python
import gpu_box2d as gb
import numpy as np

# 4096 independent worlds
batch = gb.Batch(n_worlds=4096)

for w in range(batch.n_worlds):
    # a flat floor segment as the static boundary
    batch.set_ground_edge(w, edge=0, ax=-10.0, ay=0.0, bx=10.0, by=0.0)
    # a dynamic circle dropped from above (inv_mass, inv_i; 0 means infinite)
    circle = batch.add_circle(w, px=0.0, py=3.0, radius=0.5,
                              inv_mass=1.0, inv_i=2.0, body_type=gb.DYNAMIC_BODY)
    # a dynamic box, if the build has polygons enabled
    box = batch.add_box(w, px=1.0, py=4.0, hx=0.5, hy=0.5,
                        inv_mass=1.0, inv_i=12.0, body_type=gb.DYNAMIC_BODY)

# step every world
for _ in range(300):
    batch.step(substeps=1)

# read the state out as numpy arrays
pos   = batch.positions()    # [n_worlds, max_bodies, 2]  (x, y)
ang   = batch.angles()       # [n_worlds, max_bodies]
vel   = batch.velocities()   # [n_worlds, max_bodies, 3]  (vx, vy, angular)
awake = batch.awake()        # [n_worlds, max_bodies]     uint8
count = batch.body_count()   # [n_worlds]                 int32, includes the ground slot
```

### Seeding

| Method | Effect |
|---|---|
| `set_ground_edge(world, edge, ax, ay, bx, by)` | Set a static ground-edge segment. The number of edge slots is `num_ground_edges`. |
| `add_circle(world, px, py, radius, inv_mass, inv_i, body_type)` | Add a circle body; returns its slot. |
| `add_box(world, px, py, hx, hy, inv_mass, inv_i, body_type)` | Add an axis-aligned box (when the build has polygons enabled). |
| `set_velocity(world, body, vx, vy, w)` | Set a body's linear and angular velocity. |
| `set_angle(world, body, angle)` | Set a body's orientation in radians. |
| `add_revolute_joint(world, body_a, body_b, anchor_ax, anchor_ay, anchor_bx, anchor_by)` | Pin two bodies with a revolute joint (when the build has joints enabled). |

`body_type` is `gb.STATIC_BODY` or `gb.DYNAMIC_BODY`. `inv_mass` and `inv_i` are the inverse mass and inverse rotational inertia, so a static or infinitely heavy body uses 0.

### Stepping and read-back

`step(substeps)` advances every world `substeps` times. The read-back methods return numpy arrays shaped by world then body, so a row indexes a world and a column indexes a body slot. Slots past a world's `body_count` read as 0.

### Introspection

| Property | Meaning |
|---|---|
| `n_worlds` | Worlds in the batch. |
| `max_bodies` | Per-world body capacity, including the ground slot. |
| `num_ground_edges` | Static ground-edge slots per world. |

## Determinism and fidelity

Identical seeds step to byte-identical state. The host build is bit-identical to a single-threaded host Box2D 2.3.0, the fidelity contract the whole library holds. `bindings/test_batch.py` exercises the binding end to end, and `test/gb_batch_test.cu` validates the underlying C++ driver against the standalone step at 0 ULP (it runs in the gate).

## Capacity

The per-world bounds are compile-time defines in the core (`GB_MAX_BODIES`, `GB_MAX_CONTACTS`, `GB_MAX_JOINTS`). Rebuild the extension with the defines raised to fit a denser scene. `max_bodies` reports the active value.
