# Architecture

gpu-box2d runs many independent Box2D worlds on one GPU and keeps each world's
result bit-identical to a CPU build of Box2D 2.3.0. Three decisions make that
possible: the block-per-world execution model, the shared-memory arena, and the
serial-in-order solver rule. This document explains each and why it exists.

## Block-per-world

One CUDA block simulates one world. `blockIdx.x` is the world id. The block's
threads cooperate on that world's per-step work: a thread per body for
integration, a thread per contact for narrow-phase, a block-wide reduction for
sleep, and so on.

This maps the problem the way Brax and MJX map theirs. A world is the unit of
parallelism across the grid, and the bodies and contacts inside a world are the
unit of parallelism inside a block. The alternative, one thread per world, leaves
the heavy solver running on a single thread with all of its state in registers and
local memory, which caps occupancy and strides global memory by the full per-world
footprint on every access. Block-per-world puts the hot state in shared memory and
spreads the parallel phases across the block.

The execution shell is in `include/gpu_box2d/gb_world.cuh`:

```
grid:  one block per world
block: GB_BLOCK_THREADS lanes (sized to max(bodies, contacts))
flow:
  1. cooperatively load pools.world[blockIdx.x] global -> shared
  2. __syncthreads()
  3. run the step (parallel phases + serial spine, see below)
  4. cooperatively store shared -> global
```

## Shared-memory arena

Each world's working set lives in shared memory for the duration of its step. The
canonical layout is `WorldShared` in `include/gpu_box2d/gb_pools.cuh`: bodies,
contacts, merge pairs, and scalars as a single contiguous POD. The block loads its
world's `WorldShared` from global memory at block start, runs the step against the
shared copy, and stores it back at the end.

Shared memory is roughly two orders of magnitude faster than the uncoalesced global
access pattern that a thread-per-world layout produces. It is also where the solver
scratch lives, so the contact solver reads and writes shared memory instead of
thrashing local memory. The working set is sized against measured peaks (the
default bounds are 65 bodies, 128 contacts, 48 merge pairs) and fits the 48 KB
default shared budget with headroom. sm_86 and later offer a 100 KB opt-in for
larger bounds.

### The accessor contract

Physics code never writes `w.field[slot]` directly. It goes through four macros:

```
BODY(w, field, s)   body field at slot s
CONT(w, field, c)   contact field at index c
EDGE(w, field, e)   ground edge field
SCAL(w, field)      per-world scalar
```

This indirection lets the memory backend change without editing a single physics
call site. The default backend is `GBWorld == WorldShared`, a direct shared-memory
access. A second backend (`-DGB_SOA_GLOBAL`) reads transposed global arrays indexed
`slot*NW + world`, so a warp's 32 lanes read 32 consecutive addresses (coalesced
lane=world). Both backends compute the same values in the same order. The macro
changes addressing only. Keeping every module behind these accessors is what makes
the memory layout swappable.

## The serial-in-order solver rule

This is the rule that keeps the engine bit-identical, and it is the one most easily
broken by a well-meaning optimization.

Box2D's contact solver is sequential Gauss-Seidel. In each velocity iteration it
walks the island's contacts in a fixed order, and each contact's impulse solve
reads the body velocities as mutated by the previous contact. That read-after-write
chain is the definition of the result. Reordering the sweep changes the
intermediate velocities, and because the per-contact clamps (`max(impulse, 0)` for
the normal, `clamp(impulse, -maxFriction, maxFriction)` for friction) are nonlinear
functions of the running velocity, any reordered visitation produces different
floats. Box2D v3's colored SIMD solver is explicitly not bit-compatible with
v2.3.0's serial sweep for exactly this reason.

So the solver spine stays serial and in order, on one lane:

- warm-start, the 8 velocity iterations, store-impulses, and the 3 position
  iterations run on lane 0,
- the DFS island assembly and the constraint load run on lane 0,
- the parallel phases (broad-phase, narrow-phase, integrate velocities and
  positions, sleep checks, the merge narrow checks) run across the block on
  disjoint per-body and per-contact slots, with a `__syncthreads()` between phase
  groups.

This still wins. The arena and the solver scratch live in shared memory, register
pressure drops because state lives in shared memory, and the parallel phases use the
whole block. The solver math is arithmetically small. The cost that block-per-world
removes is the memory traffic and the scheduling overhead. The solver's flops were
never the bottleneck.

### The float-fold rule

Under `--fmad=false`, float addition is not associative, so a running serial fold
gives a different result than a tree reduction over the same values. Three values in
Box2D's step are computed by a running serial fold, and all three must stay serial
on lane 0 or be reduced in the exact same association order as the CPU loop:

1. the in-place velocity and position accumulation across the 8 velocity and 3
   position sweeps,
2. `minSeparation`, the running `min` across contacts in
   `SolvePositionConstraints`, which drives the position early-exit,
3. `minSleepTime`, the running `min` across bodies in `b2Island::Solve`, which
   drives whether the island sleeps.

A parallel tree-reduce of any of these changes the association order and diverges by
a ULP, which then ripples through the connected island. Keep them serial. They scan
at most a few dozen contacts or bodies, so the cost is negligible.

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
SolveTOI: continuous collision for fast movers (gb_toi.cuh)
```

The broad-phase reproduces Box2D's `b2DynamicTree` and `b2BroadPhase` so the
pair-creation order, and therefore the island contact-solve order, matches the CPU.
The CCD path is a required part of the engine. Box2D's TOI fires several times per
settle on dynamic-static contacts and moves positions enough to fork trajectories,
so omitting it diverges.

## Module map

| Header | Role |
|---|---|
| `gb_settings.cuh` | Box2D 2.3.0 constants (timestep, iteration counts, tolerances) |
| `gb_math.cuh` | vector, rotation, transform ops (b2Math) |
| `gb_pools.cuh` | the `WorldShared` layout and the accessor macros |
| `gb_world.cuh` | the block-per-world execution shell and the step declaration |
| `gb_broadphase.cuh` | `b2DynamicTree` + `b2BroadPhase` |
| `gb_collision.cuh` | narrow-phase manifolds (in development) |
| `gb_contact_solver.cuh` | sequential-impulse solver (in development) |
| `gb_island.cuh` | island assembly, integration, sleep (in development) |
| `gb_toi.cuh` | GJK distance + `b2TimeOfImpact` |
| `gb_step.cuh` | the assembly point that wires modules into the step |

See [fidelity.md](fidelity.md) for how bit-identicality is verified and
[extending.md](extending.md) for adding shapes, the two-point solver, and joints.
