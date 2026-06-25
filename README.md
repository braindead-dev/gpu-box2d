# gpu-box2d

A GPU-accelerated port of Box2D 2.3.0 that runs thousands of independent, bit-faithful 2D physics worlds in parallel on a single GPU. It handles circles, edges, and convex polygons, the single-point and two-point block contact solvers, continuous collision, and the revolute, distance, weld, and prismatic joints, each verified bit-for-bit against Box2D 2.3.0.

## What it does

Reinforcement learning and large-scale simulation need to step many independent physics worlds at once. Brax and MJX provide this for MuJoCo. gpu-box2d provides it for Box2D: it runs thousands of separate Box2D worlds in parallel on one GPU, and it keeps each world's result bit-identical to a CPU build of Box2D 2.3.0.

Bit-identicality is the point. A position-based or re-implemented 2D solver runs fast on a GPU but produces different physics, which silently shifts the dynamics an agent trains against. This engine ports Box2D 2.3.0 phase for phase and reproduces its floats exactly, so a policy trained on the GPU batch behaves the same against the CPU engine. You get the throughput of batched GPU simulation and the dynamics of the reference engine at the same time.

The physics core is a general, header-only CUDA library with no application logic baked in. It covers the shape, contact, and joint set a 2D rigid-body engine needs: circles, edges, and convex polygons; one-point and two-point contact manifolds with the block solver; the broad-phase, the island solver, continuous collision, and the revolute, distance, weld, and prismatic joints. An application plugs in through a generic contact listener and the per-world field hooks; the fruit-merge game under `examples/` is one such application and lives entirely outside the core.

## Design

Three decisions define the engine. The first two are covered here; the third, the fidelity guarantee, has its own section.

**Thread-per-world execution.** One GPU thread simulates one world. A single kernel launch steps every world in the batch. This is the production path, and it wins for a structural reason: Box2D's contact solver is serial Gauss-Seidel and is the dominant cost of a step, so the work inside one world resists parallelizing across threads while preserving the floats (see the fidelity section). Giving each world its own thread keeps every lane busy on independent worlds, which is where the parallelism lives. Two alternative execution models were built and measured and came in slower; [docs/performance.md](docs/performance.md) documents them as findings.

**SoA lane-equals-world memory layout.** Per-world state is stored as transposed structure-of-arrays, indexed `field[slot*NW + world]`, so a warp's 32 lanes (32 consecutive worlds) read 32 consecutive addresses in one coalesced transaction instead of each lane striding a full per-world footprint. Physics code reaches state through four accessor macros (`BODY` / `CONT` / `EDGE` / `SCAL`), so the memory backend is a build-time choice that leaves every call site untouched. The contiguous per-world layout for the alternative block-per-world shell sits behind the same macros.

## The fidelity guarantee

A world simulated on the GPU produces the same floats as the same world simulated by a CPU build of Box2D 2.3.0. Two things make that hold.

**The algorithm is Box2D 2.3.0**, ported phase for phase: the same solver, the same contact order, the same island assembly order, the same broad-phase pair-creation order, the same CCD path. Nothing is approximated.

**The floating-point environment is matched** with three compile flags:

```
--fmad=false      no fused multiply-add (matches CPU -ffp-contract=off)
-prec-div=true    IEEE division (matches -mfpmath=sse)
-prec-sqrt=true   IEEE square root
```

A CPU Box2D built with `-ffp-contract=off -mfpmath=sse` rounds every operation to IEEE single precision with no fusion. These flags put the GPU in the same state, down to the transcendentals. They are a fidelity contract, baked into the build target and the gate. Relaxing any of them breaks the bit match.

The single rule most easily broken by an optimization is that Box2D's contact solver is sequential Gauss-Seidel: each contact reads the body velocities as mutated by the previous contact, in a fixed order, and the per-contact clamps are nonlinear, so any reordered sweep produces different floats. The solver spine therefore stays serial and in order. Three running float folds (the velocity accumulation, the `minSeparation` fold, and the `minSleepTime` fold) also stay serial, because float addition is not associative under the frozen flags and a parallel reduction would change the bits. This rule is documented in [docs/architecture.md](docs/architecture.md).

Verification runs two controls. Each module is compared in ULP against a golden CPU build that links the real Box2D 2.3.0, and the GPU device path is compared against the host path of the same source. Zero ULP device-versus-host proves the GPU adds no floating-point drift of its own, which isolates any remaining difference to the algorithm. Each module ships a 0-ULP micro-test. See [docs/fidelity.md](docs/fidelity.md).

## Results

Validated on an A10 (sm_86) with CUDA 12.8.

- **Single-world physics is bit-identical.** A circle settling on a static edge, a stack of circles, and a settling pile are 0 ULP against the CPU Box2D reference over hundreds of substeps, including the CCD path. The GPU device path is 0 ULP against the host path of the same source on every scenario, so the GPU adds no floating-point drift of its own.
- **Polygons and the two-point block solver are bit-identical.** The polygon mass formula, the `b2CollidePolygons` two-point manifold, the `b2CollidePolygonAndCircle` manifold, and the two-point block-solver LCP cascade through the full velocity and position spine are 0 ULP against the Box2D 2.3.0 reference.
- **The joints are bit-identical.** A revolute joint with its motor and angle limit, a rigid rod and a soft spring on a distance joint, a welded bar on a weld joint, and a slider with a limit and a motor on a prismatic joint are each 0 ULP against the Box2D 2.3.0 reference over hundreds of substeps.
- **Batched output matches the reference in distribution.** Against the CPU batch reference of the example application, the output distribution agrees at a Kolmogorov-Smirnov p-value of 1.0, with per-world state byte-exact.
- **Throughput is about 23K env-steps per second**, roughly 12x a 26-core CPU baseline and about 2x the pre-rewrite version. This is the measured ceiling for a bit-identical Box2D Gauss-Seidel solver on this card. The serial solver is about 74 percent of a step and resists parallelizing while preserving the bit match, and occupancy plus data-dependent control flow bound the rest. Throughput scales with the GPU, so a larger card lifts the absolute number. [docs/performance.md](docs/performance.md) has the full breakdown.

Worlds with a large, densely packed connected island carry an irreducible per-substep float32 difference that stays within outcome spec. This is inherent to float32 in large connected islands and is the accepted standard for this class of engine. [docs/fidelity.md](docs/fidelity.md) explains it.

## Usage

The physics core is header-only. The production build steps the whole batch with one kernel launch, one thread per world, on the SoA lane-equals-world backend:

```cpp
// build with -DGB_SOA_GLOBAL for the thread-per-world SoA backend (production path)
#include "gpu_box2d/gb_world.cuh"   // step declaration + launch helpers
#include "gpu_box2d/gb_step.cuh"    // assembles the modules into gb_world_step

WorldPoolsSoA pools = /* allocate and seed NW worlds in transposed SoA arrays */;
gb_launch_thread_step(pools);       // one thread per world steps every world
```

An application plugs in through two seams, never touching the physics. It overrides the generic contact listener (`gbOnTouchBegin` / `gbOnTouchEnd`, the begin-contact and end-contact hooks) to react to touching transitions, and it injects its own per-world fields through `GB_WORLD_USER_FIELDS`. The fruit-merge example under [examples/fruit_merge/](examples/fruit_merge/) is a worked instance: it records same-tier touches in its begin-contact hook, then merges them after the step. The core never learns what a fruit is; it sees circles with a radius and a mass, read through `gbCircleRadius`.

## Build

Requires CUDA 12.x and a GPU of compute capability 6.0 or later. Developed and validated on an A10 (sm_86) with CUDA 12.8.

With CMake:

```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
ctest --test-dir build --output-on-failure
```

The `gpu_box2d` CMake target is an INTERFACE library that carries the include path and the frozen flags, so a consumer that links it gets the correct floating-point environment automatically.

The 0-ULP micro-test gate also runs from a script:

```
ARCH=86 ./test/run_gate.sh
```

The same micro-tests build and run host-mode on a CPU, which is the development path and needs no GPU:

```
CXX=clang++ ./test/run_gate_host.sh
```

## Status

The engine is complete and validated for circles, edges, and convex polygons, with single-point and two-point contact solving, continuous collision, and the revolute joint. Single-world physics is bit-identical to Box2D 2.3.0, and the full pipeline (broad-phase, narrow-phase, contact solver, island, CCD, and both memory backends) is in place behind the 0-ULP gate. The x86/CUDA gate (`test/run_gate.sh`) passes all green, ten micro-tests with zero red, on an A10 (sm_86) with CUDA 12.8. The same tests build and run host-mode on a CPU for development. See [docs/fidelity.md](docs/fidelity.md) for the gate output and the host-mode path.

| Component | Status |
|---|---|
| Bit-identical single-world physics (drop, stack, settling pile) | validated, 0 ULP over hundreds of substeps |
| Narrow-phase manifolds (circle, edge) | validated, 0 ULP |
| Polygon shape (`b2PolygonShape`: box, hull, mass, AABB) | validated, 0 ULP |
| Polygon narrow-phase (`b2CollidePolygons`, `b2CollidePolygonAndCircle`) | validated, 0 ULP on the one- and two-point manifolds |
| Edge-polygon narrow-phase (`b2CollideEdgeAndPolygon`, `b2EPCollider`) | validated, 0 ULP on the face-A and face-B manifolds |
| Two-point block solver (the LCP block path for polygon contacts) | validated, 0 ULP through the full velocity and position spine |
| Broad-phase (`b2DynamicTree` + `b2BroadPhase`) | validated, exact proxyId and AddPair order |
| Contact solver and island (sequential-impulse + DFS assembly) | validated, 0 ULP |
| CCD / TOI (GJK distance + `b2TimeOfImpact` + SolveTOI) | validated, 0 ULP on the circle-edge sweep |
| Revolute joint (`b2RevoluteJoint`, point-to-point + motor + limit) | validated, 0 ULP on a pendulum, a motor, and an angle limit over hundreds of substeps |
| Distance joint (`b2DistanceJoint`, rigid + soft) | validated, 0 ULP on a rod and a spring over hundreds of substeps |
| Weld joint (`b2WeldJoint`, 3x3, rigid + soft) | validated, 0 ULP on a welded bar over hundreds of substeps |
| Prismatic joint (`b2PrismaticJoint`, free + limit + motor) | validated, 0 ULP on a slider over hundreds of substeps |
| Polygons and the joint wired into the assembled `gb_world_step` | live, box-on-ground, box-on-box, circle-on-box, and a pinned pendulum settle through the step |
| Thread-per-world SoA execution (production path) | validated, about 23K env-steps/s on an A10 |
| Block-per-world shared-memory execution | built and measured, slower (see performance.md) |
| Graph-colored parallel solver | built and measured, distribution-faithful speed path (see performance.md) |
| x86/CUDA 0-ULP gate (`test/run_gate.sh`) | passes all green, 10 micro-tests, 0 red |

## Roadmap

The shape set now spans circles, edges, and convex polygons, with circle, edge-circle, polygon-polygon, polygon-circle, and the dedicated edge-polygon narrow-phase; the contact solver covers the one-point and two-point block paths, and the joint set covers the revolute, distance, weld, and prismatic joints. The polygon narrow-phase and the revolute joint solve are wired into the assembled `gb_world_step` behind the `GB_ENABLE_POLYGONS` and `GB_ENABLE_JOINTS` build flags, so a step over a mixed scene dispatches circle, edge, and polygon contacts and runs the joint solve in island order. The forward direction widens the constraint set and the launcher while holding the bit-identical guarantee.

1. A general batched launcher and Python bindings, so the engine drops into a training loop as a library.
2. The remaining joint paths: the revolute motor and angle-limit rows (the 3x3 path on top of the point-to-point joint that is already in, reusing the `GBMat33` ops the weld and prismatic joints carry), and the pulley and gear joints, each on the same accessor contract with a 0-ULP micro-test.
3. The chain shape (`b2ChainShape`) and the general edge with vertex0/vertex3 connectivity, so swept polygon soup works as a static collider. The edge-polygon narrow-phase already carries the adjacency path, so the chain reuses it directly.
4. Per-point warm-start id matching for polygon contacts, so a contact whose clip features change between substeps carries impulse by matching feature id per surviving point.

Two earlier roadmap items are now closed findings. Making dense connected islands bit-identical is unreachable: the residual is irreducible float32 rounding in large islands, reproduced by both the faithful broad-phase and the colored solver, so it is documented as a known property in [docs/fidelity.md](docs/fidelity.md). Block-parallelizing the phases was built and measured slower than thread-per-world and is documented as a rejected approach in [docs/performance.md](docs/performance.md).

Each new module ships a 0-ULP micro-test before it joins the assembled step. See [docs/extending.md](docs/extending.md) for the concrete path.

## Documentation

- [docs/architecture.md](docs/architecture.md): thread-per-world execution, the SoA memory layout, and the serial-solver fidelity rule.
- [docs/fidelity.md](docs/fidelity.md): how bit-identicality is verified and what is verified.
- [docs/performance.md](docs/performance.md): the measured throughput, the structural ceiling, GPU scaling, and the two measured-and-rejected execution models.
- [docs/extending.md](docs/extending.md): how polygons, the two-point solver, and the revolute joint were added, and the path for the next shape, solver row, and joint type.

## License

MIT. See [LICENSE](LICENSE). This project ports algorithms from Box2D 2.3.0, which is Copyright (c) Erin Catto and distributed under the MIT License; the Box2D license is reproduced in [LICENSE](LICENSE).
