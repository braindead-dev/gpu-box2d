// gb_test_iface.h, flat POD interface between the solver and island micro-test (the
// engine type universe: WorldShared, gb_math V2) and the CPU Box2D reference (the
// reference type universe: WorldArena and its own V2). Both universes define their own
// `struct V2`, `b2Dot`, and so on, so they cannot share a translation unit. This header
// pulls in neither, only plain floats and ints, and the reference lives in a separate
// .cu.
//
// The scenarios are plain Box2D cases: a circle settling on a static edge, two
// circles in contact, and a small pile of circles settling. Each body carries a
// userData int (b2Body::m_userData) that the reference adapter maps to a shape
// radius; the engine reads the radius directly through BODY(w,radius,s).
#pragma once
#include <cstdint>

#define GBT_MAX_BODIES   65
#define GBT_MAX_CONTACTS 128
#define GBT_N_EDGES      3

// The full physics state the solver consumes/produces, as flat arrays (no V2).
struct SolverState {
    // bodies
    float sweepCx[GBT_MAX_BODIES],  sweepCy[GBT_MAX_BODIES];
    float sweepC0x[GBT_MAX_BODIES], sweepC0y[GBT_MAX_BODIES];
    float sweepA[GBT_MAX_BODIES],   sweepA0[GBT_MAX_BODIES], sweepAlpha0[GBT_MAX_BODIES];
    float xfPx[GBT_MAX_BODIES], xfPy[GBT_MAX_BODIES], xfQs[GBT_MAX_BODIES], xfQc[GBT_MAX_BODIES];
    float velX[GBT_MAX_BODIES], velY[GBT_MAX_BODIES], angVel[GBT_MAX_BODIES];
    float invMass[GBT_MAX_BODIES], invI[GBT_MAX_BODIES];
    float radius[GBT_MAX_BODIES];
    int   userData[GBT_MAX_BODIES], bodyType[GBT_MAX_BODIES];
    float sleepTime[GBT_MAX_BODIES];
    unsigned char awake[GBT_MAX_BODIES], alive[GBT_MAX_BODIES];
    int   bodyCount;
    // edges
    float edgeAx[GBT_N_EDGES], edgeAy[GBT_N_EDGES], edgeBx[GBT_N_EDGES], edgeBy[GBT_N_EDGES];
    // contacts (manifold cache + warm-start)
    int   cBodyA[GBT_MAX_CONTACTS], cBodyB[GBT_MAX_CONTACTS], cEdge[GBT_MAX_CONTACTS];
    unsigned char cTouching[GBT_MAX_CONTACTS];
    float cFriction[GBT_MAX_CONTACTS], cRestitution[GBT_MAX_CONTACTS];
    int   cManifoldType[GBT_MAX_CONTACTS];
    float cLocalNormalX[GBT_MAX_CONTACTS], cLocalNormalY[GBT_MAX_CONTACTS];
    float cLocalPointX[GBT_MAX_CONTACTS],  cLocalPointY[GBT_MAX_CONTACTS];
    float cPointLocalX[GBT_MAX_CONTACTS],  cPointLocalY[GBT_MAX_CONTACTS];
    float cNormalImpulse[GBT_MAX_CONTACTS], cTangentImpulse[GBT_MAX_CONTACTS];
    float cToi[GBT_MAX_CONTACTS];
    int   cToiCount[GBT_MAX_CONTACTS];
    unsigned char cToiFlag[GBT_MAX_CONTACTS], cEnabled[GBT_MAX_CONTACTS];
    int   contactCount;
};

// One seed body for a scenario. sizeClass selects a circle radius from the
// reference adapter's radius table; x,y is the initial position and vx,vy the
// initial velocity.
struct SeedBody { int sizeClass; float x,y,vx,vy; };

// Implemented in gb_island_ref.cu (the Box2D 2.3.0 reference, in its own translation unit).
extern "C" {
    // initialize the persistent reference arena with these seeds.
    void ref_init(const SeedBody* seeds, int n);
    // run the reference collide on the reference arena; export post-collide pre-solve state.
    void ref_collide(SolverState* out);
    // run the reference worldSolve on the reference arena; export solved state.
    void ref_solve(SolverState* out);
}
