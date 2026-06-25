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

// ---- per-world bounds ----------------------------------------------------------
// Defaults sized for a small-island workload; override at compile time to fit a
// denser scene. Larger bounds grow WorldShared (see the shared-budget note below).
#ifndef GB_MAX_BODIES
#define GB_MAX_BODIES   65          // dynamic bodies + 1 static ground (slot 0)
#endif
#ifndef GB_MAX_CONTACTS
#define GB_MAX_CONTACTS 128
#endif
#ifndef GB_MAX_JOINTS
#define GB_MAX_JOINTS   32          // revolute joints per world (with -DGB_ENABLE_JOINTS)
#endif
#define GB_N_EDGES      3           // static edge fixtures on the ground body
#define GB_GROUND       0           // body slot 0 is the static ground

// ============================================================================
// WorldShared. The canonical per-world field set as one contiguous POD. The
// block model holds one instance in shared memory per block. The SoA-global model
// mirrors each field as a transposed global array (length NW*MAX, index
// slot*NW+world). The field names below are the contract. Every module references
// them through the accessor macros. This is the general physics field set; an
// application adds its own per-world fields through GB_WORLD_USER_FIELDS (see the
// extension hook at the end of the struct).
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
    float  radius[GB_MAX_BODIES];   // circle-shape radius (b2CircleShape::m_radius),
                                    // read by the general core through gbCircleRadius.
    int    userData[GB_MAX_BODIES]; // b2Body::m_userData. The core never reads it.
    int    bodyType[GB_MAX_BODIES];
    float  sleepTime[GB_MAX_BODIES];
    unsigned char awake[GB_MAX_BODIES];
    unsigned char alive[GB_MAX_BODIES];
    int    bodyCount;
#ifdef GB_ENABLE_POLYGONS
    // ---- polygon shape storage (opt-in via -DGB_ENABLE_POLYGONS) --------------
    // shapeType selects the body's fixture shape: GB_SHAPE_CIRCLE (the default, 0) or
    // GB_SHAPE_POLYGON. When polygons are disabled this whole block is absent and the
    // struct is byte-identical to the circle-only layout, so the circle path is
    // unchanged. Polygon vertices/normals are packed x,y interleaved per body.
    int    shapeType[GB_MAX_BODIES];
    int    polyCount[GB_MAX_BODIES];
    float  polyRadius[GB_MAX_BODIES];
    float  polyCentroidX[GB_MAX_BODIES], polyCentroidY[GB_MAX_BODIES];
    float  polyVx[GB_MAX_BODIES*GB_MAX_POLYGON_VERTICES], polyVy[GB_MAX_BODIES*GB_MAX_POLYGON_VERTICES];
    float  polyNx[GB_MAX_BODIES*GB_MAX_POLYGON_VERTICES], polyNy[GB_MAX_BODIES*GB_MAX_POLYGON_VERTICES];
#endif
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
    // second manifold point (polygon contacts). cPointCount selects 1 or 2.
    int    cPointCount[GB_MAX_CONTACTS];
    float  cPointLocal2X[GB_MAX_CONTACTS], cPointLocal2Y[GB_MAX_CONTACTS];
    float  cNormalImpulse2[GB_MAX_CONTACTS], cTangentImpulse2[GB_MAX_CONTACTS];
    float  cToi[GB_MAX_CONTACTS];
    int    cToiCount[GB_MAX_CONTACTS];
    unsigned char cToiFlag[GB_MAX_CONTACTS], cEnabled[GB_MAX_CONTACTS];
    int    contactCount;
#ifdef GB_ENABLE_JOINTS
    // ---- revolute joint pool (opt-in via -DGB_ENABLE_JOINTS) ------------------
    // A revolute joint pins jBodyA and jBodyB at a shared world anchor. Anchors are
    // body-local; jImpulse carries the warm-start impulse across substeps. When joints
    // are disabled this block is absent and the layout is unchanged.
    int    jBodyA[GB_MAX_JOINTS], jBodyB[GB_MAX_JOINTS];
    float  jLocalAnchorAX[GB_MAX_JOINTS], jLocalAnchorAY[GB_MAX_JOINTS];
    float  jLocalAnchorBX[GB_MAX_JOINTS], jLocalAnchorBY[GB_MAX_JOINTS];
    float  jImpulseX[GB_MAX_JOINTS], jImpulseY[GB_MAX_JOINTS];
    int    jointCount;
#endif
    // physics scalars
    unsigned char stepComplete;
    // ---- application extension hook --------------------------------------
    // An application injects its own per-world fields (arrays sized to the
    // bounds, or scalars) by defining GB_WORLD_USER_FIELDS before including the
    // core. The physics core never reads them. Fields injected here are visible
    // to the BODY/CONT/EDGE/SCAL accessors by name, so an application layer
    // reaches its own state through the same contract the physics uses. See
    // examples/fruit_merge for a worked example.
#ifdef GB_WORLD_USER_FIELDS
    GB_WORLD_USER_FIELDS
#endif
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
  #define JOINT(w, field, j) ((w).field[(j)])     // joint field at slot j
  #define SCAL(w, field)     ((w).field)          // per-world scalar
#endif

// ---- shape-radius accessor --------------------------------------------------
// A circle body's radius (b2CircleShape::m_radius). The general core reads it
// through this accessor. An application sets BODY(w,radius,s) at body creation.
// Edge fixtures use GB_POLYGON_RADIUS.
GB_HD inline float gbCircleRadius(GBWorld& w, int s){ return BODY(w, radius, s); }

// ---- body shape-type accessor -----------------------------------------------
// A body's fixture shape: GB_SHAPE_CIRCLE or GB_SHAPE_POLYGON. With polygons
// disabled every body is a circle and this returns GB_SHAPE_CIRCLE with no field
// read, so the circle-only path compiles and runs identically.
GB_HD inline int gbBodyShape(GBWorld& w, int s){
#ifdef GB_ENABLE_POLYGONS
    return BODY(w, shapeType, s);
#else
    (void)w; (void)s; return GB_SHAPE_CIRCLE;
#endif
}

#ifdef GB_ENABLE_POLYGONS
// Flat slot for a body's vertex array entry: body*GB_MAX_POLYGON_VERTICES + vert.
// Indexed through the same BODY accessor, so it coalesces under the SoA backend.
GB_HD inline int gbPolyVertSlot(int body, int vert){
    return body * GB_MAX_POLYGON_VERTICES + vert;
}
#endif

// Shared-memory budget: WorldShared must fit the per-block shared arena. The
// 48 KB default holds it with headroom; sm_86 offers a 100 KB opt-in for larger
// bounds. Application fields injected through GB_WORLD_USER_FIELDS can move to
// global storage behind the accessors if shared space gets tight.
