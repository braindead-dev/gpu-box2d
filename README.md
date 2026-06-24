# gpu-box2d

A GPU-accelerated Box2D engine that runs thousands of independent, bit-faithful 2D
physics worlds in parallel on a single GPU.

## What it does

Reinforcement learning and large-scale simulation need to step many independent
physics worlds at once. Brax and MJX provide this for MuJoCo. gpu-box2d provides it
for Box2D. It runs thousands of separate Box2D worlds in parallel on one GPU, and it
keeps each world's result bit-identical to a CPU build of Box2D 2.3.0.

Bit-identicality is the point. A position-based or re-implemented 2D solver runs fast
on a GPU but produces different physics, which silently shifts the dynamics an agent
trains against. This engine ports Box2D 2.3.0 phase for phase and reproduces its
floats exactly, so a policy trained on the GPU batch behaves the same against the CPU
engine. You get the throughput of batched GPU simulation and the dynamics of the
reference engine at the same time.

The physics core is a general, header-only CUDA library. The fruit-merge game under
`examples/` is one application that lives entirely outside the core and plugs in
through a contact hook.

## Design

Four decisions define the engine. The first three are covered here; the fourth, the
fidelity guarantee, has its own section.

**Block-per-world.** One CUDA block simulates one world. The block's threads
cooperate on that world's work: a thread per body for integration, a thread per
contact for narrow-phase, a block reduction for sleep. This is how Brax and MJX think
about per-environment parallelism, applied to Box2D. A world is the unit of
parallelism across the grid; the bodies and contacts inside it are the unit of
parallelism inside a block.

**Shared-memory arena.** Each world's working set lives in shared memory for its
step. The block loads the world's state from global memory at block start, runs the
step against the shared copy, and stores it back. Shared memory is far faster than
the uncoalesced global access a thread-per-world layout produces, and it is where the
solver scratch lives, so the heavy solver reads shared memory rather than thrashing
local memory. Physics code reaches state through four accessor macros
(`BODY` / `CONT` / `EDGE` / `SCAL`), so the memory backend can change without
touching a single call site.

**Faithful sequential-impulse solver.** Box2D's contact solver is sequential
Gauss-Seidel. Each contact reads the body velocities as mutated by the previous
contact, in a fixed order. That read-after-write chain is the result, and the
per-contact clamps are nonlinear, so any reordered sweep produces different floats.
The solver spine therefore stays serial and in order on one lane, while the
embarrassingly-parallel phases (broad-phase, narrow-phase, integration, sleep) run
across the block. Three running float folds (the velocity accumulation, the
`minSeparation` fold, and the `minSleepTime` fold) also stay serial, because float
addition is not associative under the frozen flags and a parallel reduction would
change the bits. This is the single rule most easily broken by an optimization, and
it is documented in [docs/architecture.md](docs/architecture.md).

## The fidelity guarantee

A world simulated on the GPU produces the same floats as the same world simulated by
a CPU build of Box2D 2.3.0. Two things make that hold.

**The algorithm is Box2D 2.3.0**, ported phase for phase: the same solver, the same
contact order, the same island assembly order, the same broad-phase pair-creation
order, the same CCD path. Nothing is approximated.

**The floating-point environment is matched** with three compile flags:

```
--fmad=false      no fused multiply-add (matches CPU -ffp-contract=off)
-prec-div=true    IEEE division (matches -mfpmath=sse)
-prec-sqrt=true   IEEE square root
```

A CPU Box2D built with `-ffp-contract=off -mfpmath=sse` rounds every operation to
IEEE single precision with no fusion. These flags put the GPU in the same state, down
to the transcendentals. They are not a tuning knob. Relaxing any of them breaks the
bit match, so they are baked into the build target and the gate.

Verification runs two controls. Each module is compared in ULP against a golden CPU
build that links the real Box2D 2.3.0, and the GPU device path is compared against
the host path of the same source. Zero ULP device-versus-host proves the GPU adds no
floating-point drift of its own, which isolates any remaining difference to the
algorithm. Each module ships a 0-ULP micro-test and does not enter the assembled step
until it is green. See [docs/fidelity.md](docs/fidelity.md).

## Usage

The physics core is header-only. Include the assembled step and run one block per
world:

```cpp
#include "gpu_box2d/gb_world.cuh"   // block-per-world shell + gb_world_step
#include "gpu_box2d/gb_step.cuh"    // assembles the modules into gb_world_step

// pools.world is an array of WorldShared, one per world, in device memory.
WorldPools pools = /* allocate and seed NW worlds */;

// one kernel launch steps every world: load shared, step, store.
gb_launch_block_step(pools);        // <<<NW, GB_BLOCK_THREADS, sizeof(WorldShared)>>>
```

A game layer plugs in through the contact hook without touching the physics. The
fruit-merge example records same-tier touches in `fmBeginContact`, then merges them
after the step:

```cpp
#include "gpu_box2d/gb_pools.cuh"
#include "fruit_merge/fm_game.cuh"

int s = fmAddFruit(w, /*tier*/ 0, /*x*/ 0.0f, /*y*/ 5.0f, /*vy*/ 0.0f);  // create
// ... gb_world_step(w) settles the world, firing fmBeginContact on touches ...
int gained = fmProcessMerges(w);   // act on recorded pairs, score the merges
```

The core never learns what a fruit is. See
[examples/fruit_merge/](examples/fruit_merge/).

## Build

Requires CUDA 12.x and a GPU of compute capability 6.0 or later. Developed and
validated on an A10 (sm_86) with CUDA 12.8.

With CMake:

```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
ctest --test-dir build --output-on-failure
```

The `gpu_box2d` CMake target is an INTERFACE library that carries the include path
and the frozen flags, so a consumer that links it gets the correct floating-point
environment automatically.

The 0-ULP micro-test gate also runs from a script:

```
ARCH=86 ./test/run_gate.sh
```

## Status

This repository is early-stage. The validated foundation is here and proven; the full
parallel engine is in active development. The pieces below are green at 0 ULP against
Box2D 2.3.0.

| Component | Status |
|---|---|
| Bit-identical single-world physics (1-body drop, 2-body stack) | validated, 0 ULP over hundreds of substeps |
| Block-per-world execution model | validated, 0 ULP host-vs-device on single and two-body worlds |
| Broad-phase (`b2DynamicTree` + `b2BroadPhase`) | validated, exact proxyId and AddPair order |
| CCD / TOI (GJK distance + `b2TimeOfImpact`) | validated, 0 ULP on the fruit-wall scenario |
| Fidelity-test methodology (ULP gate + device-vs-host control) | validated |
| Narrow-phase manifolds | in development |
| Contact solver and island | in development (single and two-body 0 ULP; dense pile shows a 1-ULP device residual) |
| Block-parallel phases and multi-world step assembly | in development |
| Batched launcher and Python observation API | in development |
| Polygons, two-point block solver, joints | future work |

The accurate claim today: the bit-identical single-world physics, the block-per-world
model, the broad-phase, the CCD path, and the fidelity methodology are proven. The
contact solver, island, narrow-phase, and game-layer assembly are being built and
integrated behind the same 0-ULP gate.

## Roadmap

1. Land the narrow-phase, contact solver, and island modules at 0 ULP and assemble
   them into `gb_world_step`.
2. Close the dense-island 1-ULP residual to make dense single worlds bit-identical,
   not only distributionally faithful.
3. Move the parallel phases (broad-phase, narrow-phase, integration, sleep) to
   block-parallel while the solver spine stays serial.
4. Ship the batched launcher and the Python observation API for reinforcement
   learning.
5. Extend shape support: polygons, the two-point block solver, then joints. See
   [docs/extending.md](docs/extending.md).

## Extending

The core covers circles and static edges with the single-point solver and CCD.
[docs/extending.md](docs/extending.md) is the path to polygons, the two-point block
solver, and joints. The two rules that never bend: go through the accessors, and
match Box2D's evaluation order, not just its math. Every extension ships a 0-ULP
micro-test before it joins the step.

## Documentation

- [docs/architecture.md](docs/architecture.md): block-per-world, the shared-memory
  arena, and the serial-solver fidelity rule.
- [docs/fidelity.md](docs/fidelity.md): how bit-identicality is verified and what is
  verified today.
- [docs/extending.md](docs/extending.md): adding shapes, the two-point solver, and
  joints.

## License

MIT. See [LICENSE](LICENSE). This project ports algorithms from Box2D 2.3.0, which is
Copyright (c) Erin Catto and distributed under the MIT License; the Box2D license is
reproduced in [LICENSE](LICENSE).
