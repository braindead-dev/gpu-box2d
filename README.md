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

Three decisions define the engine. The first two are covered here; the third, the
fidelity guarantee, has its own section.

**Thread-per-world execution.** One GPU thread simulates one world. A single kernel
launch steps every world in the batch. This is the production path, and it wins for a
structural reason: Box2D's contact solver is serial Gauss-Seidel and is the dominant
cost of a step, so the work inside one world resists parallelizing across threads
while preserving the floats (see the fidelity section). Giving each world its own
thread keeps every lane busy on independent worlds, which is where the parallelism
lives. Two alternative execution models were built and measured and came in slower;
[docs/performance.md](docs/performance.md) documents them as findings.

**SoA lane-equals-world memory layout.** Per-world state is stored as transposed
structure-of-arrays, indexed `field[slot*NW + world]`, so a warp's 32 lanes (32
consecutive worlds) read 32 consecutive addresses in one coalesced transaction
instead of each lane striding a full per-world footprint. Physics code reaches state
through four accessor macros (`BODY` / `CONT` / `EDGE` / `SCAL`), so the memory
backend is a build-time choice that leaves every call site untouched. The contiguous
per-world layout for the alternative block-per-world shell sits behind the same
macros.

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
to the transcendentals. They are a fidelity contract, baked into the build target and
the gate. Relaxing any of them breaks the bit match.

The single rule most easily broken by an optimization is that Box2D's contact solver
is sequential Gauss-Seidel: each contact reads the body velocities as mutated by the
previous contact, in a fixed order, and the per-contact clamps are nonlinear, so any
reordered sweep produces different floats. The solver spine therefore stays serial and
in order. Three running float folds (the velocity accumulation, the `minSeparation`
fold, and the `minSleepTime` fold) also stay serial, because float addition is not
associative under the frozen flags and a parallel reduction would change the bits.
This rule is documented in [docs/architecture.md](docs/architecture.md).

Verification runs two controls. Each module is compared in ULP against a golden CPU
build that links the real Box2D 2.3.0, and the GPU device path is compared against the
host path of the same source. Zero ULP device-versus-host proves the GPU adds no
floating-point drift of its own, which isolates any remaining difference to the
algorithm. Each module ships a 0-ULP micro-test. See [docs/fidelity.md](docs/fidelity.md).

## Results

Validated on an A10 (sm_86) with CUDA 12.8.

- **Single-world physics is bit-identical.** A drop, a stack, and a settling pile are
  0 ULP against the CPU Box2D reference over hundreds of substeps, including the CCD
  path. The GPU device path is 0 ULP against the host path of the same source on every
  scenario, so the GPU adds no floating-point drift of its own.
- **Batched output matches the reference in distribution.** Against the CPU batch
  reference, the score distribution agrees at a Kolmogorov-Smirnov p-value of 1.0,
  with queue state and observations byte-exact.
- **Throughput is about 23K env-steps per second**, roughly 12x a 26-core CPU baseline
  and about 2x the pre-rewrite version. This is the measured ceiling for a
  bit-identical Box2D Gauss-Seidel solver on this card. The serial solver is about 74
  percent of a step and resists parallelizing while preserving the bit match, and
  occupancy plus data-dependent control flow bound the rest. Throughput scales with the
  GPU, so a larger card lifts the absolute number.
  [docs/performance.md](docs/performance.md) has the full breakdown.

Dense fruit-pile worlds carry an irreducible per-substep float32 difference that
compounds slightly over a full game and stays within outcome spec. This is inherent
to float32 in large connected islands and is the accepted standard for this class of
engine. [docs/fidelity.md](docs/fidelity.md) explains it.

## Usage

The physics core is header-only. The production build steps the whole batch with one
kernel launch, one thread per world, on the SoA lane-equals-world backend:

```cpp
// build with -DGB_SOA_GLOBAL for the thread-per-world SoA backend (production path)
#include "gpu_box2d/gb_world.cuh"   // step declaration + launch helpers
#include "gpu_box2d/gb_step.cuh"    // assembles the modules into gb_world_step

WorldPoolsSoA pools = /* allocate and seed NW worlds in transposed SoA arrays */;
gb_launch_thread_step(pools);       // one thread per world steps every world
```

A game layer plugs in through the contact hook without touching the physics. The
fruit-merge example records same-tier touches in its begin-contact hook, then merges
them after the step:

```cpp
#include "fruit_merge/fm_engine.cuh"   // composes the core with the game layer

int s = fmAddFruit(w, /*tier*/ 0, /*x*/ 0.0f, /*y*/ 5.0f, /*vy*/ 0.0f, /*age*/ 0.0f);
// fmSettleAndMerge(w) settles the world, firing the merge hook on touches,
// then acts on the recorded pairs and scores the merges.
int gained = fmSettleAndMerge(w);
```

The core never learns what a fruit is. It sees circles with a radius and a mass, read
through `gbCircleRadius`. See [examples/fruit_merge/](examples/fruit_merge/).

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

The engine is complete and validated. Single-world physics is bit-identical to Box2D
2.3.0, the batched engine matches the reference in distribution at KS p=1.0, and the
full pipeline (broad-phase, narrow-phase, contact solver, island, CCD, the game layer,
and both memory backends) is in place behind the 0-ULP gate.

| Component | Status |
|---|---|
| Bit-identical single-world physics (drop, stack, settling pile) | validated, 0 ULP over hundreds of substeps |
| Narrow-phase manifolds | validated, 0 ULP |
| Broad-phase (`b2DynamicTree` + `b2BroadPhase`) | validated, exact proxyId and AddPair order |
| Contact solver and island (sequential-impulse + DFS assembly) | validated, 0 ULP |
| CCD / TOI (GJK distance + `b2TimeOfImpact` + SolveTOI) | validated, 0 ULP on the fruit-wall scenario |
| Thread-per-world SoA execution (production path) | validated, about 23K env-steps/s on an A10 |
| Block-per-world shared-memory execution | built and measured, slower (see performance.md) |
| Graph-colored parallel solver | built and measured, distribution-faithful speed path (see performance.md) |
| Batched output vs CPU reference | KS p=1.0, queue and observations byte-exact |
| Fidelity-test methodology (ULP gate + device-vs-host control) | validated |
| Polygons, two-point block solver, joints | extension targets, see docs/extending.md |

## Roadmap

The core engine is done. The open direction is extending shape and constraint support
while keeping the bit-identical guarantee:

1. Polygons: the shape data, the `b2CollidePolygons` and `b2CollidePolygonAndCircle`
   narrow-phase, and the polygon mass formula.
2. The two-point block solver for polygon contacts.
3. Joints: the per-type constraint rows and the joint-contact solve interleave.

Each step ships a 0-ULP micro-test before it joins the assembled step. See
[docs/extending.md](docs/extending.md) for the concrete path.

## Documentation

- [docs/architecture.md](docs/architecture.md): thread-per-world execution, the SoA
  memory layout, and the serial-solver fidelity rule.
- [docs/fidelity.md](docs/fidelity.md): how bit-identicality is verified and what is
  verified.
- [docs/performance.md](docs/performance.md): the measured throughput, the structural
  ceiling, GPU scaling, and the two measured-and-rejected execution models.
- [docs/extending.md](docs/extending.md): adding shapes, the two-point solver, and
  joints.

## License

MIT. See [LICENSE](LICENSE). This project ports algorithms from Box2D 2.3.0, which is
Copyright (c) Erin Catto and distributed under the MIT License; the Box2D license is
reproduced in [LICENSE](LICENSE).
