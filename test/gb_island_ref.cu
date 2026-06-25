// gb_island_ref.cu. CPU Box2D 2.3.0 reference for the solver and island micro-test, in
// its own translation unit (the only one that includes the Box2D 2.3.0 reference
// headers), so the reference type universe stays separate from the engine's gb_* types
// in the test TU. It exposes a flat-POD interface (gb_test_iface.h). The reference world
// persists across substeps so warm-start impulses carry exactly as the engine does.
//
// world_types.cuh and b2_step.cuh are the reference adapter headers: a thin CPU build of
// Box2D 2.3.0's collide and solve over a flat arena layout. Point your include path at
// your Box2D 2.3.0 reference build (see test/README.md).
//
// The reference adapter exposes its size index, radius table, and density under its own
// names. The aliases below bind those to the generic names this file uses, so the body
// reads in plain physics terms. To run against a different reference build, redefine the
// three aliases to match its arena.
#include "world_types.cuh"
#include "b2_step.cuh"          // collide + solve over the reference Box2D 2.3.0 build
#include "gb_test_iface.h"
#include <cstring>

// ---- reference-adapter aliases -------------------------------------------------
// REF_SIZECLASS  : the arena's per-body size index (-1 marks the static ground).
// refRadius(c)   : maps a size index to a circle radius.
// REF_DENSITY    : the area density used to derive mass from radius.
#define REF_SIZECLASS  tier
#define refRadius      tier_radius
#define REF_DENSITY    FRUIT_DENSITY

static WorldArena g_ref;        // persistent reference arena

static void exportState(SolverState* o){
    memset(o, 0, sizeof(SolverState));
    for (int i=0;i<MAX_BODIES;++i){
        o->sweepCx[i]=g_ref.sweepCx[i];   o->sweepCy[i]=g_ref.sweepCy[i];
        o->sweepC0x[i]=g_ref.sweepC0x[i]; o->sweepC0y[i]=g_ref.sweepC0y[i];
        o->sweepA[i]=g_ref.sweepA[i];     o->sweepA0[i]=g_ref.sweepA0[i]; o->sweepAlpha0[i]=g_ref.sweepAlpha0[i];
        o->xfPx[i]=g_ref.xfPx[i]; o->xfPy[i]=g_ref.xfPy[i]; o->xfQs[i]=g_ref.xfQs[i]; o->xfQc[i]=g_ref.xfQc[i];
        o->velX[i]=g_ref.velX[i]; o->velY[i]=g_ref.velY[i]; o->angVel[i]=g_ref.angVel[i];
        o->invMass[i]=g_ref.invMass[i]; o->invI[i]=g_ref.invI[i];
        // The arena's size index maps to a radius; carry both onto the engine's general
        // userData and radius fields.
        o->userData[i]=g_ref.REF_SIZECLASS[i]; o->bodyType[i]=g_ref.bodyType[i];
        o->radius[i]=g_ref.REF_SIZECLASS[i]>=0 ? refRadius(g_ref.REF_SIZECLASS[i]) : 0.0f;
        o->sleepTime[i]=g_ref.sleepTime[i]; o->awake[i]=g_ref.awake[i]; o->alive[i]=g_ref.alive[i];
    }
    o->bodyCount=g_ref.bodyCount;
    for (int e=0;e<N_EDGES;++e){ o->edgeAx[e]=g_ref.edgeAx[e]; o->edgeAy[e]=g_ref.edgeAy[e];
                                 o->edgeBx[e]=g_ref.edgeBx[e]; o->edgeBy[e]=g_ref.edgeBy[e]; }
    for (int c=0;c<MAX_CONTACTS;++c){
        o->cBodyA[c]=g_ref.cBodyA[c]; o->cBodyB[c]=g_ref.cBodyB[c]; o->cEdge[c]=g_ref.cEdge[c];
        o->cTouching[c]=g_ref.cTouching[c];
        o->cFriction[c]=g_ref.cFriction[c]; o->cRestitution[c]=g_ref.cRestitution[c];
        o->cManifoldType[c]=g_ref.cManifoldType[c];
        o->cLocalNormalX[c]=g_ref.cLocalNormalX[c]; o->cLocalNormalY[c]=g_ref.cLocalNormalY[c];
        o->cLocalPointX[c]=g_ref.cLocalPointX[c];   o->cLocalPointY[c]=g_ref.cLocalPointY[c];
        o->cPointLocalX[c]=g_ref.cPointLocalX[c];   o->cPointLocalY[c]=g_ref.cPointLocalY[c];
        o->cNormalImpulse[c]=g_ref.cNormalImpulse[c]; o->cTangentImpulse[c]=g_ref.cTangentImpulse[c];
        o->cToi[c]=g_ref.cToi[c]; o->cToiCount[c]=g_ref.cToiCount[c];
        o->cToiFlag[c]=g_ref.cToiFlag[c]; o->cEnabled[c]=g_ref.cEnabled[c];
    }
    o->contactCount=g_ref.contactCount;
}

extern "C" void ref_init(const SeedBody* seeds, int n){
    memset(&g_ref, 0, sizeof(WorldArena));
    g_ref.bodyType[GROUND_BODY]=0; g_ref.REF_SIZECLASS[GROUND_BODY]=-1;
    g_ref.invMass[GROUND_BODY]=0; g_ref.invI[GROUND_BODY]=0;
    g_ref.alive[GROUND_BODY]=1; g_ref.awake[GROUND_BODY]=0;
    syncTransform(g_ref, GROUND_BODY);
    // A container made of three static edges: a floor and two side walls. This is a
    // plain Box2D scene; the side walls keep settling piles bounded.
    g_ref.edgeAx[0]=-WALL_X_C; g_ref.edgeAy[0]=0; g_ref.edgeBx[0]= WALL_X_C; g_ref.edgeBy[0]=0;
    g_ref.edgeAx[1]=-WALL_X_C; g_ref.edgeAy[1]=0; g_ref.edgeBx[1]=-WALL_X_C; g_ref.edgeBy[1]=CONTAINER_H_C;
    g_ref.edgeAx[2]= WALL_X_C; g_ref.edgeAy[2]=0; g_ref.edgeBx[2]= WALL_X_C; g_ref.edgeBy[2]=CONTAINER_H_C;
    int bc=1;
    for (int i=0;i<n;++i){
        int s=bc++; int c=seeds[i].sizeClass; float r=refRadius(c);
        float mass=REF_DENSITY*B2_PI*r*r; float I=mass*(0.5f*r*r);
        g_ref.bodyType[s]=2; g_ref.REF_SIZECLASS[s]=c;
        g_ref.invMass[s]=mass>0.0f?1.0f/mass:0.0f; g_ref.invI[s]=I>0.0f?1.0f/I:0.0f;
        g_ref.alive[s]=1; g_ref.awake[s]=1; g_ref.sleepTime[s]=0.0f;
        g_ref.sweepCx[s]=seeds[i].x;  g_ref.sweepCy[s]=seeds[i].y;
        g_ref.sweepC0x[s]=seeds[i].x; g_ref.sweepC0y[s]=seeds[i].y;
        g_ref.velX[s]=seeds[i].vx; g_ref.velY[s]=seeds[i].vy;
        syncTransform(g_ref, s);
    }
    g_ref.bodyCount=bc; g_ref.contactCount=0; g_ref.pairCount=0; g_ref.stepComplete=1;
}

extern "C" void ref_collide(SolverState* out){
    collidePhase(g_ref);        // collide phase: identical input handed to both sides
    exportState(out);           // post-collide, pre-solve
}

extern "C" void ref_solve(SolverState* out){
    worldSolve(g_ref);          // the reference solver (b2Island / b2ContactSolver)
    exportState(out);           // solved ground truth
}
