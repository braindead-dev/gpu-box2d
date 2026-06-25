# Architecture

gpu-box2d runs many independent Box2D worlds on one GPU and keeps each world's result
bit-identical to a CPU build of Box2D 2.3.0. Three decisions make that possible: the
thread-per-world execution model, the SoA lane-equals-world memory layout, and the
serial-in-order solver rule. This document explains each and why it exists.

## Thread-per-world execution

One GPU thread simulates one world. A single kernel launch steps every world in the
batch: thread `world = blockIdx.x * blockDim.x + threadIdx.x` runs `gb_world_step` on
its own world. This is the production path.

The choice follows from the solver. Box2D's contact solver is sequential Gauss-Seidel
and is the dominant cost of a step (about 74 percent), and it resists parallelizing
across threads while preserving the floats (see the solver rule below). The
parallelism that is real is across worlds, so the execution model that wins gives each
world a thread and keeps every lane busy on an independent world. Two alternative
models that move the parallelism inside a world, block-per-world and a graph-colored
parallel solver, were built and measured and came in slower. See
[performance.md](performance.md) for the numbers and the reasoning.

The step and the launch helpers are in `include/gpu_box2d/gb_world.cuh`:

```
grid:   blocks of GB_BLOCK_THREADS threads, one thread per world
thread: world = blockIdx.x * blockDim.x + threadIdx.x
        if (world < NW) gb_world_step(handle_for(world))
```

The block-per-world shared-memory shell also lives in `gb_world.cuh`, guarded out of
the production build, as the host for the slower execution model.

## SoA lane-equals-world memory layout

Per-world state is stored as transposed structure-of-arrays. A field is indexed
`field[slot*NW + world]`, so for a fixed slot the values for consecutive worlds are
consecutive in memory. A warp is 32 threads running 32 consecutive worlds, so when
every lane reads the same field of the same slot, the warp reads 32 consecutive
addresses and the hardware coalesces them into one transaction. The contrast is a
per-world contiguous layout, where each lane strides a full per-world footprint and the
loads serialize. The SoA layout is `WorldPoolsSoA` in
`include/gpu_box2d/gb_soa_backend.cuh` and is selected with `-DGB_SOA_GLOBAL`.

### The accessor contract

Physics code never writes `w.field[slot]` directly. It goes through four macros:

```
BODY(w, field, s)   body field at slot s
CONT(w, field, c)   contact field at index c
EDGE(w, field, e)   ground edge field
SCAL(w, field)      per-world scalar
```

This indirection lets the memory backend change without editing a single physics call
site. The production backend (`-DGB_SOA_GLOBAL`) expands `BODY(w, sweepCx, s)` to the
transposed global access `p->sweepCx[s*NW + world]`. A second backend, used by the
block-per-world shell, makes `GBWorld == WorldShared`, a contiguous per-world POD, and
expands the same macro to a direct `w.sweepCx[s]`. Both backends compute the same
values in the same order. The macro changes addressing only. Keeping every module
behind these accessors is what makes the memory layout a build-time choice.

The field set is the general physics contract, defined as `WorldShared` in
`include/gpu_box2d/gb_pools.cuh`: bodies, contacts, and physics scalars. The defaults
(65 bodies, 128 contacts) keep the per-world working set small enough that the SoA
arrays stay bandwidth-friendly and the alternative shared-memory shell fits the 48 KB
default block budget with headroom. sm_86 and later offer a 100 KB opt-in for larger
bounds. An application adds its own per-world fields through `GB_WORLD_USER_FIELDS`
without editing the core struct.

## The serial-in-order solver rule

This is the rule that keeps the engine bit-identical, and it is the one most easily
broken by a well-meaning optimization.

Box2D's contact solver is sequential Gauss-Seidel. In each velocity iteration it walks
the island's contacts in a fixed order, and each contact's impulse solve reads the body
velocities as mutated by the previous contact. That read-after-write chain is the
definition of the result. Reordering the sweep changes the intermediate velocities, and
because the per-contact clamps (`max(impulse, 0)` for the normal,
`clamp(impulse, -maxFriction, maxFriction)` for friction) are nonlinear functions of
the running velocity, any reordered visitation produces different floats. Box2D v3's
colored SIMD solver is explicitly not bit-compatible with v2.3.0's serial sweep for
exactly this reason.

So the solver spine stays serial and in order:

- warm-start, the 8 velocity iterations, store-impulses, and the 3 position iterations
  run in a single fixed sweep,
- the DFS island assembly and the constraint load run in island order,
- the per-body and per-contact phases that are genuinely independent (integration,
  narrow-phase manifold compute, copy-back) touch disjoint slots and stay correct under
  any ordering.

In thread-per-world, the whole step including the solver runs on one thread, so the
serial order is automatic. The shared-memory shell preserves it by running the spine on
one lane. Either way, the solver math, the iteration order, and the float-fold order
match Box2D 2.3.0's `b2ContactSolver` bit-for-bit.

### The float-fold rule

Under `--fmad=false`, float addition is not associative, so a running serial fold gives
a different result than a tree reduction over the same values. Three values in Box2D's
step are computed by a running serial fold, and all three must stay serial or be reduced
in the exact same association order as the CPU loop:

1. the in-place velocity and position accumulation across the 8 velocity and 3 position
   sweeps,
2. `minSeparation`, the running `min` across contacts in `SolvePositionConstraints`,
   which drives the position early-exit,
3. `minSleepTime`, the running `min` across bodies in `b2Island::Solve`, which drives
   whether the island sleeps.

A parallel tree-reduce of any of these changes the association order and diverges by a
ULP, which then ripples through the connected island. Keep them serial. They scan at
most a few dozen contacts or bodies, so the cost is negligible.

## Pipeline

One world step is Box2D's `b2World::Step`, reproduced phase for phase:

```
Collide:  broad-phase pair generation (gb_broadphase.cuh)
          narrow-phase manifolds (gb_collision.cuh)
          persistent contact update with warm-start id carry
Solve:    DFS island assembly (gb_island.cuh)
          per island: integrate velocities, warm-start, 8 velocity iters,
          store impulses, integrate positions, 3 position iters, sleep
          (gb_island.cuh + gb_contact_solver.cuh)
SolveTOI: continuous collision for fast movers (gb_toi.cuh + gb_step.cuh)
```

The broad-phase reproduces Box2D's `b2DynamicTree` and `b2BroadPhase` so the
pair-creation order, and therefore the island contact-solve order, matches the CPU. The
CCD path is a required part of the engine. Box2D's TOI fires several times per settle on
dynamic-static contacts and moves positions enough to fork trajectories, so omitting it
diverges.

## Module map

| Header | Role |
|---|---|
| `gb_settings.cuh` | Box2D 2.3.0 constants (timestep, iteration counts, tolerances) |
| `gb_math.cuh` | vector, rotation, transform ops (b2Math) |
| `gb_pools.cuh` | the `WorldShared` field set and the accessor macros |
| `gb_soa_backend.cuh` | the transposed SoA lane-equals-world backend (production) |
| `gb_contact_types.cuh` | the shared manifold, constraint, block-matrix, and island types |
| `gb_world.cuh` | the step declaration, the thread-per-world launch, and the block shell |
| `gb_broadphase.cuh` | `b2DynamicTree` + `b2BroadPhase` |
| `gb_polygon.cuh` | `b2PolygonShape` (box, hull, mass, AABB) |
| `gb_collision.cuh` | narrow-phase manifolds (circle, edge, polygon) |
| `gb_contact_solver.cuh` | sequential-impulse solver, single-point and two-point block paths |
| `gb_island.cuh` | island assembly, integration, sleep |
| `gb_joint.cuh` | `b2RevoluteJoint` (point-to-point) |
| `gb_toi.cuh` | GJK distance + `b2TimeOfImpact` |
| `gb_step.cuh` | the assembly point that wires modules into the step |
| `gb_colored_solver.cuh` | the graph-colored parallel solver (alternative speed path) |

See [fidelity.md](fidelity.md) for how bit-identicality is verified, [performance.md](performance.md)
for throughput and the rejected execution models, and [extending.md](extending.md) for
adding shapes, the two-point solver, and joints.
