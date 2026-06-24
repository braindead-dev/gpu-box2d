// gb_contact_types.cuh. The shared cross-module contact and solver interface.
// The narrow-phase, contact solver, island, and CCD headers all include this so they
// compose against one definition of the manifold, constraint, and island types and
// of the phase-function signatures. The types here (the fused single-point
// constraint and the one-point manifold) are the validated definitions. Their layout
// is fidelity-critical: every solver phase reads and writes these fields in Box2D's
// exact order, so changing them changes the floats.
#pragma once
#include "gb_pools.cuh"

// ---- manifold (1-point; all our shapes are circle+edge, so pointCount<=1) ------
struct GBManifold {
    int type;          // GB_MANIFOLD_CIRCLES / GB_MANIFOLD_FACE_A
    int pointCount;    // 0 or 1
    V2  localNormal;   // for face-A
    V2  localPoint;    // manifold.localPoint
    V2  pLocalPoint;   // points[0].localPoint
};
struct GBWorldManifold { V2 normal; V2 point0; };   // world-space (1 point)

// ---- fused velocity+position constraint (single point; localCenter==0) --------
struct GBVelConstraintPt {
    V2 rA, rB; float normalImpulse, tangentImpulse, normalMass, tangentMass, velocityBias;
};
struct GBConstraint {
    int   indexA, indexB;            // island-local body indices
    float invMassA, invMassB, invIA, invIB;
    float friction, restitution;
    V2    normal;
    GBVelConstraintPt p;             // single contact point (velocity)
    int   contactIdx;                // global contact slot (StoreImpulses)
    V2    localNormal, localPoint, pLocalPoint;   // position-solve
    int   type; float radiusA, radiusB;
};

// ---- per-island solve scratch (lives in shared memory in the block model) ------
struct GBIslandData {
    int        bodies[GB_MAX_BODIES];     // global body slots, island order
    int        contacts[GB_MAX_CONTACTS]; // global contact slots, island order
    int        bodyCount, contactCount;
    V2         posC[GB_MAX_BODIES];  float posA[GB_MAX_BODIES];
    V2         vel[GB_MAX_BODIES];   float velW[GB_MAX_BODIES];
    GBConstraint con[GB_MAX_CONTACTS];    // fused vel+pos constraints
};

// ============================================================================
// CROSS-MODULE PHASE-FUNCTION CONTRACT. Each phase is declared by its role here and
// defined by its owning header. The assembled step (gb_step.cuh) calls them. The
// signatures are stable.
//
//   gb_collision.cuh (narrow-phase):
//     void gbCollideCircles(GBManifold&, float rA, Xf xfA, float rB, Xf xfB);
//     void gbCollideEdgeAndCircle(GBManifold&, V2 A, V2 B, float edgeR, float circR, Xf xfA, Xf xfB);
//     void gbWorldManifoldInit(GBWorldManifold&, const GBManifold&, Xf xfA, float rA, Xf xfB, float rB);
//     void gbContactUpdate(GBWorld& w, int ci);     // narrow-phase + touching + listener hook
//
//   gb_broadphase.cuh: the GbBroadPhase/GbDynTree + gbBpUpdatePairs. The faithful
//     path uses the tree; the assembled step also offers a brute-force fat-AABB
//     gbCollidePhase. Either way it fills the world's contact pool and fires the
//     listener hook in Box2D m_contactList order.
//
//   gb_contact_solver.cuh:
//     void gbInitVelocityConstraints(GBWorld&, GBIslandData&);   // build GBConstraint per contact
//     void gbWarmStart(GBIslandData&);
//     void gbSolveVelocity(GBIslandData&);   // ONE velocity iteration (serial sweep, lane 0)
//     void gbStoreImpulses(GBWorld&, GBIslandData&);
//     bool gbSolvePosition(GBIslandData&);   // ONE position iteration; returns contactsOkay
//                                            //   [SERIAL: minSeparation fold on lane 0]
//
//   gb_island.cuh:
//     void gbWorldSolve(GBWorld& w);         // DFS island assembly + island Solve loop:
//        integrate-velocities -> warmstart -> 8x gbSolveVelocity -> store -> integrate-positions
//        -> up to 3x gbSolvePosition -> copy-back -> sleep
//        [SERIAL FOLDS on lane 0: in-place velocity accum, minSeparation, minSleepTime]
//
//   gb_toi.cuh: gbWorldSolveTOI(GBWorld&), which drives the above through the CCD path.
//
// THE 3 SERIAL FLOAT-FOLD RULE (non-associative under --fmad=false): the in-place
// velocity/position accumulation in gbSolveVelocity/gbSolvePosition, the
// minSeparation fold in gbSolvePosition, and the minSleepTime fold in the island
// sleep step must run serial and in order on lane 0. A parallel reduction would
// change the association order and break the 0-ULP match.
// ============================================================================
