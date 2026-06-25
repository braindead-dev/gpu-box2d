# Per-module micro-test template

Every module ships with a test that proves it matches the Box2D 2.3.0 reference
bit-for-bit before it joins the assembled step. No module enters the step without a
green micro-test. This is how the engine stays verifiable while it is built.

## The rule

- Each module is a header in `include/gpu_box2d/` written against the accessors
  (`BODY` / `CONT` / `EDGE` / `SCAL`) and `gb_math.cuh`. It never reads raw arrays
  and never adds globals.
- Each module gets a test in `test/<module>_test.cu` that:
  1. constructs a known input (bodies and contacts in a `WorldShared`),
  2. runs the module's device function,
  3. compares against the Box2D 2.3.0 reference value for the same input,
  4. asserts 0 ULP, or documents the exact tolerance and why.

## What the reference is, per module

| Module | Reference | Bit-exact? |
|---|---|---|
| math | b2Math.h ops (b2Mul / Cross / Dot / Normalize) | yes, same ops in the same order |
| collision | b2CollideCircles / b2CollideEdgeAndCircle / worldManifold | yes |
| polygon | b2PolygonShape mass / b2CollidePolygons / b2CollidePolygonAndCircle | yes |
| edge-polygon | b2CollideEdgeAndPolygon (b2EPCollider) | yes |
| chain | b2ChainShape CreateChain / CreateLoop / GetChildEdge | yes |
| broad-phase | b2DynamicTree proxyId sequence + AddPair order | yes, integer-exact |
| contact solver | b2ContactSolver single-point and two-point velocity + position iters | yes, serial sweep |
| island | b2Island::Solve integrate / sleep + DFS order | yes |
| ccd | b2TimeOfImpact / b2Distance (GJK) | yes |
| joint (revolute) | b2RevoluteJoint init / velocity / position (point-to-point, motor, limit) | yes |
| joint (distance) | b2DistanceJoint init / velocity / position (rigid + soft) | yes |
| joint (weld) | b2WeldJoint init / velocity / position (3x3, rigid + soft) | yes |
| joint (prismatic) | b2PrismaticJoint init / velocity / position (free + limit + motor) | yes |
| joint (pulley) | b2PulleyJoint init / velocity / position (two-body, ratio) | yes |

## Template

```cuda
// <module>_test.cu. Micro-test for gb_<module>. 0-ULP vs Box2D 2.3.0.
#include "gpu_box2d/gb_<module>.cuh"
#include <cstdio>

// 1. Reference value(s) for a fixed input, either hardcoded from a Box2D run or
//    computed by linking the real Box2D 2.3.0 in a separate translation unit.
// 2. Run the device function on the same input.
// 3. Compare bits.
__global__ void run(/* WorldShared* w, outputs */){ /* call gb_<fn> */ }

int main(){
    // build input, launch, copy back, diff vs reference
    // ULP check: int-cast both floats, assert the difference is 0
    // print "PASS <module>: 0 ULP" or "FAIL <module>: maxULP=N"
    return /* 0 if all 0-ULP */;
}
```

## ULP comparison helper

Use this exact helper for consistency across tests.

```cuda
inline long ulpDiff(float a, float b){
    int ai = *(int*)&a, bi = *(int*)&b;
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return labs((long)ai - (long)bi);
}
```

## Build flags (frozen, fidelity-critical)

```
nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 -Iinclude
```

`--fmad=false -prec-div=true -prec-sqrt=true` put the GPU floating-point environment
in the same state as a CPU build of Box2D with `-ffp-contract=off -mfpmath=sse`.
Changing these breaks bit-identicality. The GPU device path is 0-ULP against the
host path of the same code under these flags, which proves the environment matches.
