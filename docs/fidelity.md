# Fidelity

The claim this engine makes is bit-identicality: a world simulated on the GPU produces
the same floats as the same world simulated by a CPU build of Box2D 2.3.0. This
document explains why that is achievable, how it is verified, and what is verified.

## Why bit-identical is possible

Two things have to line up: the algorithm and the floating-point environment.

**The algorithm** is Box2D 2.3.0, ported phase for phase. The solver is the same
sequential Gauss-Seidel sweep in the same contact order. The island assembly visits
bodies and contacts in the same order Box2D's prepend lists produce. The broad-phase
reproduces `b2DynamicTree` so the pair-creation order matches. Nothing is approximated
or re-derived. This is the difference from a position-based or re-implemented solver,
which converges to a different fixed point and diverges chaotically.

**The floating-point environment** is made to match with three compile flags:

```
--fmad=false      no fused multiply-add (matches CPU -ffp-contract=off)
-prec-div=true    IEEE division (matches -mfpmath=sse)
-prec-sqrt=true   IEEE square root
```

A CPU build of Box2D compiled with `-ffp-contract=off -mfpmath=sse` rounds every
operation to IEEE single precision with no fusion. The three nvcc flags put the GPU in
the same state: no FMA fusion, IEEE-rounded division and square root. Under these flags
the GPU evaluates the same expression tree to the same bits, including the
transcendentals `sinf`, `cosf`, and `sqrtf` used in the rotation and normalization math.

These flags are a fidelity contract. Relaxing any of them reintroduces fusion or a
faster but different division and breaks the bit match. They are set on the `gpu_box2d`
CMake target and in the gate script.

## The two controls

The methodology runs two comparisons.

**ULP versus CPU Box2D.** The same scenario runs on the GPU port and on a golden CPU
build that links the real Box2D 2.3.0. Every per-substep value (position, angle,
velocity, angular velocity, sleep flag) is compared in ULP. Zero ULP means the floats
are bit-identical. This is the primary gate.

**GPU device versus host.** The same device source runs on the GPU and, compiled in
host mode, on the CPU. When the GPU device and host agree at 0 ULP across every
scenario, the GPU floating-point environment is proven to match the CPU's, so any
difference between the GPU and the golden CPU Box2D is algorithmic and is found in the
port. The engine shows 0 ULP device-versus-host on every scenario, which isolates
fidelity work to the algorithm.

## Per-module micro-tests

Each module ships a test that compares its output against the Box2D 2.3.0 reference for
a fixed input and asserts 0 ULP. The ULP helper is:

```c
inline long ulpDiff(float a, float b){
    int ai = *(int*)&a, bi = *(int*)&b;
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return labs((long)ai - (long)bi);
}
```

A module enters the assembled step once its micro-test is green. This keeps the engine
verifiable while it is built, rather than validating only at the end.

| Module | Reference | Bit-exact? |
|---|---|---|
| math | b2Math ops | yes, same ops in the same order |
| collision | b2CollideCircles / b2CollideEdgeAndCircle / worldManifold | yes |
| polygon | b2PolygonShape mass / b2CollidePolygons / b2CollidePolygonAndCircle | yes |
| broad-phase | b2DynamicTree proxyId sequence + AddPair order | yes, integer-exact |
| contact solver | b2ContactSolver velocity + position iters (1-point and 2-point block) | yes, serial sweep |
| block solver | b2ContactSolver two-point LCP cascade + 2x2 block | yes |
| island | b2Island::Solve integrate / sleep + DFS order | yes |
| ccd | b2TimeOfImpact / b2Distance (GJK) | yes |
| joint | b2RevoluteJoint init / velocity / position (point-to-point) | yes |

## What is verified

The following are green at 0 ULP against the CPU Box2D reference:

- **Single-world physics.** A circle settling on a static edge, a stack of circles, and
  a settling pile are bit-identical over hundreds of substeps, including the CCD path.
- **Narrow-phase.** `b2CollideCircles`, `b2CollideEdgeAndCircle`, and the world
  manifold reproduce Box2D's manifolds and touching transitions exactly.
- **Polygons.** `test/gb_polygon_test.cu` reproduces the polygon mass formula, the
  `b2CollidePolygons` two-point manifold, and the `b2CollidePolygonAndCircle`
  manifold at 0 ULP, including the local normal, the plane point, both clip points,
  and the contact ids.
- **Two-point block solver.** `test/gb_block_solver_test.cu` drives the full velocity
  and position spine on a box resting on the ground and reproduces every body
  kinematic and both warm-start impulses at 0 ULP, exercising the 2x2 block matrix,
  its inverse, and the four-case LCP cascade.
- **Revolute joint.** `test/gb_joint_test.cu` swings a two-body pendulum over
  hundreds of substeps and reproduces the bob velocity, angular velocity, position,
  angle, and the warm-start impulse at 0 ULP.
- **Broad-phase.** `test/gb_broadphase_test.cu` produces the exact proxyId sequence and
  the exact AddPair order as Box2D for a fixed scene.
- **Contact solver and island.** `test/gb_island_test.cu` reproduces the velocity and
  position iterations, the sleep decision, and the DFS island order at 0 ULP on host and
  device.
- **CCD / TOI.** `test/gb_toi_test.cu` reproduces the GJK distance and the
  `b2TimeOfImpact` result at 0 ULP on a circle-edge continuous-collision sweep.
- **The execution model.** The GPU device path is 0 ULP against the host path of the
  same source on every scenario, so the execution model adds no drift of its own.

For a batched application, the example layer's output distribution agrees with its CPU
batch reference at a Kolmogorov-Smirnov p-value of 1.0, with per-world application state
byte-exact. That application logic lives outside the physics core; see
[../examples/fruit_merge](../examples/fruit_merge).

## The dense-world float32 note

Single isolated worlds and worlds with a handful of bodies are bit-identical. Worlds
with a large, densely packed connected island carry an irreducible per-substep float32
difference: a single contact can flip one substep earlier than the CPU because of
float32 rounding in the solve, which then forks the trajectory while staying bounded (no
NaN, no blowup). The island body order, the contact solve order, and the constraint
composition are byte-exact even for large islands, so the difference is float
sensitivity in the solve itself. The ordering is correct, and the residual is rounding.

This is inherent to float32 in large connected islands and is the accepted standard for
this class of engine. Brax and MJX make the same trade for the same reason. In this
regime worlds match in distribution, which is what batch reinforcement learning over
thousands of worlds requires, and the measured agreement is KS p=1.0 against the
reference. Closing it for dense single worlds would need a higher-precision contact
solve, which is outside the bit-identical-to-float32-Box2D scope this engine targets.

## Running the gate

```
ARCH=86 ./test/run_gate.sh
```

Set `ARCH` to your GPU's compute capability. The script builds each self-contained
micro-test with the frozen flags and asserts the 0-ULP pass line. CMake exposes the
same tests through CTest:

```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
ctest --test-dir build --output-on-failure
```
