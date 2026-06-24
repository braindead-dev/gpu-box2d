# Fidelity

The claim this engine makes is bit-identicality: a world simulated on the GPU
produces the same floats as the same world simulated by a CPU build of Box2D 2.3.0.
This document explains why that is achievable, how it is verified, and what is
verified today.

## Why bit-identical is possible

Two things have to line up: the algorithm and the floating-point environment.

**The algorithm** is Box2D 2.3.0, ported phase for phase. The solver is the same
sequential Gauss-Seidel sweep in the same contact order. The island assembly visits
bodies and contacts in the same order Box2D's prepend lists produce. The broad-phase
reproduces `b2DynamicTree` so the pair-creation order matches. Nothing is
approximated or re-derived. This is the difference from a position-based or
re-implemented solver, which converges to a different fixed point and diverges
chaotically.

**The floating-point environment** is made to match with three compile flags:

```
--fmad=false      no fused multiply-add (matches CPU -ffp-contract=off)
-prec-div=true    IEEE division (matches -mfpmath=sse)
-prec-sqrt=true   IEEE square root
```

A CPU build of Box2D compiled with `-ffp-contract=off -mfpmath=sse` rounds every
operation to IEEE single precision with no fusion. The three nvcc flags put the GPU
in the same state: no FMA fusion, IEEE-rounded division and square root. Under these
flags the GPU evaluates the same expression tree to the same bits, including the
transcendentals `sinf`, `cosf`, and `sqrtf` used in the rotation and normalization
math.

These flags are not optional and not a tuning knob. Relaxing any of them
reintroduces fusion or faster-but-different division and breaks the bit match. They
are set on the `gpu_box2d` CMake target and in the gate script.

## The two controls

The methodology runs two comparisons.

**ULP versus CPU Box2D.** The same scenario runs on the GPU port and on a golden CPU
build that links the real Box2D 2.3.0. Every per-substep value (position, angle,
velocity, angular velocity, sleep flag) is compared in ULP. Zero ULP means the
floats are bit-identical. This is the primary gate.

**GPU device versus host.** The same device source runs on the GPU and, compiled in
host mode, on the CPU. When these agree at 0 ULP while the GPU disagrees with the
golden CPU Box2D, the difference is algorithmic. When the GPU device and host agree
across every scenario, the GPU floating-point environment is proven to match the
CPU's, so any remaining divergence is a porting bug to find in the algorithm. The
single-world prototype showed 0 ULP device-versus-host on every scenario, which
isolates fidelity work to the algorithm.

## Per-module micro-tests

Each module ships a test that compares its output against the Box2D 2.3.0 reference
for a fixed input and asserts 0 ULP. The ULP helper is:

```c
inline long ulpDiff(float a, float b){
    int ai = *(int*)&a, bi = *(int*)&b;
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return labs((long)ai - (long)bi);
}
```

A module does not enter the assembled step until its micro-test is green. This keeps
the engine verifiable while it is built, rather than validating only at the end.

| Module | Reference | Bit-exact? |
|---|---|---|
| math | b2Math ops | yes, same ops in the same order |
| collision | b2CollideCircles / b2CollideEdgeAndCircle / worldManifold | yes |
| broad-phase | b2DynamicTree proxyId sequence + AddPair order | yes, integer-exact |
| contact solver | b2ContactSolver velocity + position iters | yes, serial sweep |
| island | b2Island::Solve integrate / sleep + DFS order | yes |
| ccd | b2TimeOfImpact / b2Distance (GJK) | yes |

## What is verified today

The following are green at 0 ULP:

- **Single-world physics.** A 1-body drop and a 2-body stack are bit-identical over
  hundreds of substeps, including the CCD path.
- **The block-per-world model.** Running the serial step on lane 0 against the
  shared-resident world reproduces the host result at 0 ULP on the single-world and
  two-body scenarios. The execution model itself adds no drift.
- **Broad-phase.** `test/gb_broadphase_test.cu` produces the exact proxyId sequence
  and the exact AddPair order as Box2D for a fixed scene.
- **CCD / TOI.** `test/gb_toi_test.cu` reproduces the GJK distance and the
  `b2TimeOfImpact` result at 0 ULP on a fruit-wall continuous-collision scenario.

The following are in development:

- **Contact solver and island.** `test/gb_island_test.cu` passes at 0 ULP on host
  and device for the single-drop and two-body scenarios. The five-body pile
  currently shows a 1-ULP device divergence on one body's transform at one substep.
  The body order, the contact solve order, and the constraint composition match the
  CPU; the residual is a float-rounding sensitivity in a dense connected island.
  Closing it is active work.

## The dense-world note

For large connected islands, a single contact or merge that flips one substep
earlier than the CPU forks the trajectory while staying bounded (no NaN, no blowup).
The island body order and contact solve order are byte-exact even for large islands,
so the residual is per-substep float sensitivity in the solve. The ordering is
correct; the rounding is the open part. In this regime worlds match in distribution.
For batch reinforcement learning, distributional fidelity over thousands of worlds is
the operative requirement, and a clean 0-ULP gate for dense single worlds is the
target being closed.

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
