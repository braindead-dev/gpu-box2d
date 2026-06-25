# API reference

This is the reference for the two ways to drive gpu-box2d: the C++ core (header-only, for embedding the engine in a CUDA or C++ program) and the Python binding (for driving N worlds from a training loop). Both speak the same bit-faithful Box2D 2.3.0 physics.

## C++ core

The core is header-only under `include/gpu_box2d/`. A program includes the assembled step and the launch helpers, allocates per-world state, and steps the batch.

### The step

```cpp
GB_HD void gb_world_step(GBWorld& w);
```

`gb_world_step` runs one `b2World::Step` on one world handle: collide (broad-phase and narrow-phase), solve (island assembly and the contact and joint solve), then the CCD pass. `GBWorld` is the per-world handle the active backend provides. The signature is frozen; the phases are the modules in `gb_step.cuh`.

### Backends and launch

The memory backend is a build-time choice behind the accessor macros, so physics call sites never change.

| Build | Backend | Launch |
|---|---|---|
| `-DGB_SOA_GLOBAL` (production) | transposed SoA, one thread per world | `gb_launch_thread_step(WorldPoolsSoA)` |
| default | contiguous `WorldShared`, one block per world | `gb_launch_block_step(WorldPools)` |

Under `-DGB_SOA_GLOBAL` the state lives in transposed global arrays indexed `slot*NW + world`, so a warp's 32 lanes read 32 consecutive addresses. The default backend keeps one `WorldShared` per world contiguous. Both are bit-identical; the macro changes addressing only.

### The accessor contract

Physics modules read and write world state through four macros, never raw arrays, so the backend stays swappable:

```cpp
BODY(w, field, slot)    // a body field, e.g. BODY(w, sweepCx, s)
CONT(w, field, idx)     // a contact field
EDGE(w, field, edge)    // a static ground-edge field
JOINT(w, field, slot)   // a joint field
SCAL(w, field)          // a per-world scalar, e.g. SCAL(w, bodyCount)
```

The field names are the `WorldShared` field set in `gb_pools.cuh`, the frozen interface. A circle body's radius is read through `gbCircleRadius(w, s)`; a body's shape tag through `gbBodyShape(w, s)`.

### Feature flags

```
-DGB_ENABLE_POLYGONS   per-body shape tag, polygon storage, the polygon narrow-phase,
                       and the edge-polygon collider
-DGB_ENABLE_JOINTS     the per-world revolute joint pool and the joint solve interleave
```

With both off, the assembled step is byte-for-byte the circle-and-edge engine. With a flag on, the matching storage appears behind the accessors and the step dispatches the new path.

### Application seams

An application plugs in without touching the physics, through two seams.

The contact listener, the begin-contact and end-contact hooks fired on touching transitions:

```cpp
#define GB_CONTACT_LISTENER_HOOKS 1
GB_HD inline void gbOnTouchBegin(GBWorld& w, int bodyA, int bodyB);
GB_HD inline void gbOnTouchEnd  (GBWorld& w, int bodyA, int bodyB);
```

The per-world field injection, application state added to the world struct, visible to the accessors by name:

```cpp
#define GB_WORLD_USER_FIELDS      /* fields added to WorldShared */
#define GB_WORLD_SOA_USER_FIELDS  /* the transposed mirror for the SoA backend */
```

`examples/fruit_merge/` is a worked instance: it records same-tier touches in its begin-contact hook and merges them after the step, with its own per-world fields injected through these hooks.

### Shapes and joints (header functions)

The narrow-phase and joint modules expose their phases as `GB_HD` functions, each 0-ULP against the Box2D 2.3.0 reference:

| Module | Header | Key functions |
|---|---|---|
| polygon | `gb_polygon.cuh` | `gbPolygonSetAsBox`, `gbPolygonSet`, `gbPolygonComputeMass`, `gbPolygonComputeAABB` |
| narrow-phase | `gb_collision.cuh` | `gbCollideCircles`, `gbCollideEdgeAndCircle`, `gbCollidePolygons`, `gbCollidePolygonAndCircle`, `gbCollideEdgeAndPolygon` |
| chain | `gb_chain_shape.cuh` | `gbChainCreateChain`, `gbChainCreateLoop`, `gbChainGetChildEdge` |
| revolute joint | `gb_revolute_joint.cuh` | `gbRevoluteFullInitVelocity`, `gbRevoluteFullSolveVelocity`, `gbRevoluteFullSolvePosition` |
| distance joint | `gb_distance_joint.cuh` | `gbDistanceInitVelocity`, `gbDistanceSolveVelocity`, `gbDistanceSolvePosition` |
| weld joint | `gb_weld_joint.cuh` | `gbWeldInitVelocity`, `gbWeldSolveVelocity`, `gbWeldSolvePosition` |
| prismatic joint | `gb_prismatic_joint.cuh` | `gbPrismaticInitVelocity`, `gbPrismaticSolveVelocity`, `gbPrismaticSolvePosition` |

Each joint phase takes the joint struct and a `GBIslandData&` (the per-island solve scratch) and matches the Box2D 2.3.0 evaluation order exactly.

## Batched driver (C++)

`bindings/gb_batch.cuh` is a generic driver over the `WorldShared` backend, the layer the Python binding wraps. It owns a `GBBatch` of `NW` worlds and seeds, steps, and reads them back.

```cpp
GBBatch b(n_worlds);
gbBatchSetGroundEdge(b, world, edge, ax, ay, bx, by);
int s   = gbBatchAddCircle(b, world, px, py, radius, invMass, invI, type);
int box = gbBatchAddBox   (b, world, px, py, hx, hy, invMass, invI, type);   // GB_ENABLE_POLYGONS
gbBatchSetVelocity(b, world, body, vx, vy, w);
gbBatchSetAngle(b, world, body, angle);
int j = gbBatchAddRevoluteJoint(b, world, bodyA, bodyB, ax, ay, bx, by);     // GB_ENABLE_JOINTS
gbBatchStep(b, substeps);
gbBatchGetPositions(b, out);   // float[NW * GB_MAX_BODIES * 2]
gbBatchGetAngles(b, out);      // float[NW * GB_MAX_BODIES]
gbBatchGetVelocities(b, out);  // float[NW * GB_MAX_BODIES * 3]
gbBatchGetAwake(b, out);       // unsigned char[NW * GB_MAX_BODIES]
gbBatchGetBodyCount(b, out);   // int[NW]
```

`type` is `GB_STATIC_BODY` or `GB_DYNAMIC_BODY`. Slot 0 of each world is the static ground body. The driver runs host-side; the same seeded state steps on the GPU through the SoA-global path.

## Python binding

`gpu_box2d.Batch` wraps the driver with numpy arrays. See [../bindings/README.md](../bindings/README.md) for install and the full method list. In brief:

```python
import gpu_box2d as gb
batch = gb.Batch(n_worlds)
batch.set_ground_edge(world, edge, ax, ay, bx, by)
batch.add_circle(world, px, py, radius, inv_mass, inv_i, body_type)
batch.add_box(world, px, py, hx, hy, inv_mass, inv_i, body_type)
batch.add_revolute_joint(world, body_a, body_b, ax, ay, bx, by)
batch.set_velocity(world, body, vx, vy, w)
batch.set_angle(world, body, angle)
batch.step(substeps)
pos = batch.positions()      # [n_worlds, max_bodies, 2]
ang = batch.angles()         # [n_worlds, max_bodies]
vel = batch.velocities()     # [n_worlds, max_bodies, 3]
awake = batch.awake()        # [n_worlds, max_bodies], uint8
count = batch.body_count()   # [n_worlds], int32
```

`gb.STATIC_BODY` and `gb.DYNAMIC_BODY` are the body-type constants. The read-back arrays are shaped by world then body; slots past a world's body count read as 0.

## Capacity and tuning

The per-world bounds are compile-time defines in `gb_pools.cuh` (`GB_MAX_BODIES`, `GB_MAX_CONTACTS`, `GB_MAX_JOINTS`) and `gb_polygon.cuh` (`GB_MAX_POLYGON_VERTICES`). Raising them grows the world struct; the shared-memory budget for the block-per-world backend is the constraint to track. See [extending.md](extending.md) for the budget note.
