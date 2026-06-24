// gb_pools.cuh. The frozen world-state contract. The SoA WorldState layout and
// the accessor abstraction every physics module codes against. Field semantics
// and accessor signatures are stable. Treat them as a published interface.
//
// ============================================================================
// THE KEY DESIGN: one physics codebase, two memory backends, zero call-site churn.
// ============================================================================
// The physics core reads and writes world state through FIELD MACROS, e.g.
// `BODY(w, sweepCx, s)`. A macro layer maps that to the right address for the
// active backend:
//   Thread-per-world SoA-global (-DGB_SOA_GLOBAL, the production default): GBWorld
//           is a thin handle {WorldPoolsSoA* p; int world;} and `BODY(w, sweepCx, s)`
//           expands to a transposed global access `p->sweepCx[s*NW + world]` so a
//           warp's 32 lanes (consecutive worlds) read 32 consecutive addresses
//           (coalesced lane=world). One CUDA thread runs one world's full step.
//           This is the measured high-throughput path (see docs/performance.md).
//   Block-per-world (default when GB_SOA_GLOBAL is not set): GBWorld == WorldShared,
//           a contiguous POD that lives in shared memory per block. `BODY(w, sweepCx,
//           s)` is a direct shared-memory access `w.sweepCx[s]`. This path is kept
//           as a documented alternative; it runs slower because the serial
//           Gauss-Seidel solver idles most of the block's lanes.
// Both backends are bit-identical: the macro changes ADDRESSING only, never the
// value computed or the order it is computed in.
//
// THE FROZEN INTERFACE is (1) the WorldShared field set and names, (2) the
// accessor macros BODY/CONT/EDGE/SCAL, (3) the per-world bounds, (4) the
// lane=world index rule (slot*NW+world). Modules build on these and never edit them.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"

// ---- per-world bounds (sized to measured peaks: 86 contacts / 34 bodies) -------
#ifndef GB_MAX_BODIES
#define GB_MAX_BODIES   65          // 64 dynamic circles + 1 static ground (slot 0)
#endif
#ifndef GB_MAX_CONTACTS
#define GB_MAX_CONTACTS 128         // measured peak 86
#endif
#ifndef GB_MAX_PAIRS
#define GB_MAX_PAIRS    48
#endif
#define GB_N_EDGES      3
#define GB_GROUND       0

// ============================================================================
// WorldShared. The canonical per-world field set as one contiguous POD. The
// block model holds one instance in shared memory per block. The SoA-global model
// mirrors each field as a transposed global array (length NW*MAX, index
// slot*NW+world). The field names below are the contract. Every module references
// them through the accessor macros. Physics fields and game-layer fields share
// this struct; game fields are marked and the physics core ignores them.
// ============================================================================
struct WorldShared {
    // bodies (b2Sweep + b2Transform + velocity + mass + flags). slot 0 = static ground.
    float  sweepCx[GB_MAX_BODIES],  sweepCy[GB_MAX_BODIES];   // m_sweep.c (== position)
    float  sweepC0x[GB_MAX_BODIES], sweepC0y[GB_MAX_BODIES];  // m_sweep.c0 (CCD anchor)
    float  sweepA[GB_MAX_BODIES],   sweepA0[GB_MAX_BODIES];   // m_sweep.a / a0
    float  sweepAlpha0[GB_MAX_BODIES];                        // m_sweep.alpha0
    float  xfPx[GB_MAX_BODIES], xfPy[GB_MAX_BODIES];          // m_xf.p
    float  xfQs[GB_MAX_BODIES], xfQc[GB_MAX_BODIES];          // m_xf.q (sin,cos)
    float  velX[GB_MAX_BODIES], velY[GB_MAX_BODIES], angVel[GB_MAX_BODIES];
    float  invMass[GB_MAX_BODIES], invI[GB_MAX_BODIES];
    float  radius[GB_MAX_BODIES];   // circle-shape radius. The general core reads
                                    // shape radius here; the game sets it at body
                                    // creation from its own size table.
    int    tier[GB_MAX_BODIES];     // GAME field (example: fruit tier). Physics ignores it.
    int    bodyType[GB_MAX_BODIES];
    float  sleepTime[GB_MAX_BODIES];
    unsigned char awake[GB_MAX_BODIES];
    unsigned char alive[GB_MAX_BODIES];
    int    bodyCount;
    // ground edge fixtures (static)
    float  edgeAx[GB_N_EDGES], edgeAy[GB_N_EDGES], edgeBx[GB_N_EDGES], edgeBy[GB_N_EDGES];
    // contacts (persistent; manifold cache + warm-start + CCD/TOI)
    int    cBodyA[GB_MAX_CONTACTS], cBodyB[GB_MAX_CONTACTS], cEdge[GB_MAX_CONTACTS];
    unsigned char cTouching[GB_MAX_CONTACTS];
    float  cFriction[GB_MAX_CONTACTS], cRestitution[GB_MAX_CONTACTS];
    int    cManifoldType[GB_MAX_CONTACTS];
    float  cLocalNormalX[GB_MAX_CONTACTS], cLocalNormalY[GB_MAX_CONTACTS];
    float  cLocalPointX[GB_MAX_CONTACTS],  cLocalPointY[GB_MAX_CONTACTS];
    float  cPointLocalX[GB_MAX_CONTACTS],  cPointLocalY[GB_MAX_CONTACTS];
    float  cNormalImpulse[GB_MAX_CONTACTS], cTangentImpulse[GB_MAX_CONTACTS];
    float  cToi[GB_MAX_CONTACTS];
    int    cToiCount[GB_MAX_CONTACTS];
    unsigned char cToiFlag[GB_MAX_CONTACTS], cEnabled[GB_MAX_CONTACTS];
    int    contactCount;
    // merge pairs (ContactListener-filled, insertion-ordered)
    int    pairLo[GB_MAX_PAIRS], pairHi[GB_MAX_PAIRS], pairA[GB_MAX_PAIRS], pairB[GB_MAX_PAIRS];
    int    pairCount;
    // physics scalars
    unsigned char stepComplete;
    // game scalars (example layer). The physics core ignores these. They can move
    // to global storage if shared-memory space gets tight.
    int      score, drops, maxTier, goldapples, cur, nxt;
    unsigned char dead;
    unsigned long long rng;
    float    age[GB_MAX_BODIES], outline[GB_MAX_BODIES];
};

// ============================================================================
// WorldPools. Persistent global batch state for the block-per-world backend. The
// launcher allocates it; the physics core never calls malloc. It stores an array of
// WorldShared, one per world, which the block-per-world kernel loads to shared
// memory. The production SoA-global backend uses WorldPoolsSoA (gb_soa_backend.cuh),
// which mirrors each field as a transposed global array behind the same accessor
// macros, so swapping backends changes storage rather than call sites.
// ============================================================================
struct WorldPools {
    int          NW;
    WorldShared* world;      // [NW] persistent per-world state
};

// ============================================================================
// THE ACCESSOR CONTRACT. Physics modules use these, never raw `w.field[s]`, so
// the memory backend can swap. GBWorld is the per-world handle the core receives.
// ============================================================================
#ifdef GB_SOA_GLOBAL
  // Production path. GBWorld is a thin handle {WorldPoolsSoA* p; int world;} and the
  // accessor macros index transposed global arrays at slot*NW+world (coalesced
  // lane=world). Defined in gb_soa_backend.cuh, which provides WorldPoolsSoA, GBWorld,
  // and the BODY/CONT/EDGE/SCAL macros with the same field names as WorldShared, so
  // physics call sites are unchanged.
  #include "gb_soa_backend.cuh"
#else
  // Alternative path. GBWorld is the WorldShared (shared-resident in the block model).
  typedef WorldShared GBWorld;
  // Field accessors. Direct here, but routed through macros so the SoA backend
  // can override addressing without touching any physics call site.
  #define BODY(w, field, s)  ((w).field[(s)])     // body field at slot s
  #define CONT(w, field, c)  ((w).field[(c)])     // contact field at index c
  #define EDGE(w, field, e)  ((w).field[(e)])     // ground edge field
  #define SCAL(w, field)     ((w).field)          // per-world scalar
#endif

// ---- general core shape accessor (game-agnostic) ----------------------------
// A circle body's radius. The general core reads radius through this accessor.
// The game layer sets BODY(w,radius,s) at body creation from its own size table.
// Edge fixtures use GB_POLYGON_RADIUS.
GB_HD inline float gbCircleRadius(GBWorld& w, int s){ return BODY(w, radius, s); }

// Shared-memory budget: WorldShared must fit the per-block shared arena. The
// 48 KB default holds it with headroom; sm_86 offers a 100 KB opt-in for larger
// bounds. Game scalars can move to global if shared space gets tight.
