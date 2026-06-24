// gb_step.cuh. The assembled physics core, the single integration point.
// It wires the modules into gb_world_step:
//
//     gb_world_step(w) = Collide (broad-phase + narrow-phase)
//                        -> Solve (island assembly + contact solver)
//                        -> SolveTOI (CCD)
//
// Each module is a header under include/gpu_box2d/. A module is included here
// once its 0-ULP micro-test passes. The contact solver, the island, the
// narrow-phase, and the full SolveTOI driver are in active development, so this
// file is the stable seam that receives them as they validate. The signature of
// gb_world_step is frozen (declared in gb_world.cuh). Module internals can move to
// block-parallel without touching this wiring.
#pragma once
#include "gb_pools.cuh"

// ---- module includes (uncommented as each module's micro-test passes) -------
#include "gb_broadphase.cuh"        // broad-phase pair generation (validated, 0-ULP)
#include "gb_toi.cuh"               // GJK distance + b2TimeOfImpact (validated, 0-ULP)
// #include "gb_collision.cuh"      // narrow-phase manifolds (in development)
// #include "gb_contact_solver.cuh" // sequential Gauss-Seidel solver (in development)
// #include "gb_island.cuh"         // DFS island assembly + integrate + sleep (in development)

// ---- gb_world_step: the assembled step ---------------------------------------
// The body is filled in as the remaining modules validate and are included above.
// The signature is frozen (see gb_world.cuh). The block-per-world execution shell
// in gb_world.cuh calls this on a shared-resident world.
