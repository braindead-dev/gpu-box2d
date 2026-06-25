// gb_soa_backend.cuh. The GB_SOA_GLOBAL backend (the production thread-per-world
// path). Transposed SoA arrays indexed field[slot*NW + world], so a warp's 32 lanes
// (consecutive worlds) read 32 consecutive addresses (coalesced lane=world). The
// field names match WorldShared exactly, so the accessor macros swap the backing
// store without any module call-site changes. Included by gb_pools.cuh under
// -DGB_SOA_GLOBAL.
#pragma once

struct WorldPoolsSoA {
    int NW;
    float* sweepCx;   // [ MAX*NW ], body
    float* sweepCy;   // [ MAX*NW ], body
    float* sweepC0x;   // [ MAX*NW ], body
    float* sweepC0y;   // [ MAX*NW ], body
    float* sweepA;   // [ MAX*NW ], body
    float* sweepA0;   // [ MAX*NW ], body
    float* sweepAlpha0;   // [ MAX*NW ], body
    float* xfPx;   // [ MAX*NW ], body
    float* xfPy;   // [ MAX*NW ], body
    float* xfQs;   // [ MAX*NW ], body
    float* xfQc;   // [ MAX*NW ], body
    float* velX;   // [ MAX*NW ], body
    float* velY;   // [ MAX*NW ], body
    float* angVel;   // [ MAX*NW ], body
    float* invMass;   // [ MAX*NW ], body
    float* invI;   // [ MAX*NW ], body
    float* radius;   // [ MAX*NW ], body
    int* userData;   // [ MAX*NW ], body
    int* bodyType;   // [ MAX*NW ], body
    float* sleepTime;   // [ MAX*NW ], body
    unsigned char* awake;   // [ MAX*NW ], body
    unsigned char* alive;   // [ MAX*NW ], body
    int* bodyCount;   // [ NW ], scalar
#ifdef GB_ENABLE_POLYGONS
    // polygon shape storage (opt-in mirror of the WorldShared block).
    int* shapeType;   // [ MAX*NW ], body
    int* polyCount;   // [ MAX*NW ], body
    float* polyRadius;   // [ MAX*NW ], body
    float* polyCentroidX;   // [ MAX*NW ], body
    float* polyCentroidY;   // [ MAX*NW ], body
    float* polyVx;   // [ MAX*VERTS*NW ], body-vertex
    float* polyVy;   // [ MAX*VERTS*NW ], body-vertex
    float* polyNx;   // [ MAX*VERTS*NW ], body-vertex
    float* polyNy;   // [ MAX*VERTS*NW ], body-vertex
#endif
    float* edgeAx;   // [ MAX*NW ], edge
    float* edgeAy;   // [ MAX*NW ], edge
    float* edgeBx;   // [ MAX*NW ], edge
    float* edgeBy;   // [ MAX*NW ], edge
    int* cBodyA;   // [ MAX*NW ], cont
    int* cBodyB;   // [ MAX*NW ], cont
    int* cEdge;   // [ MAX*NW ], cont
    unsigned char* cTouching;   // [ MAX*NW ], cont
    float* cFriction;   // [ MAX*NW ], cont
    float* cRestitution;   // [ MAX*NW ], cont
    int* cManifoldType;   // [ MAX*NW ], cont
    float* cLocalNormalX;   // [ MAX*NW ], cont
    float* cLocalNormalY;   // [ MAX*NW ], cont
    float* cLocalPointX;   // [ MAX*NW ], cont
    float* cLocalPointY;   // [ MAX*NW ], cont
    float* cPointLocalX;   // [ MAX*NW ], cont
    float* cPointLocalY;   // [ MAX*NW ], cont
    float* cNormalImpulse;   // [ MAX*NW ], cont
    float* cTangentImpulse;   // [ MAX*NW ], cont
    int* cPointCount;   // [ MAX*NW ], cont (1 or 2)
    float* cPointLocal2X;   // [ MAX*NW ], cont
    float* cPointLocal2Y;   // [ MAX*NW ], cont
    float* cNormalImpulse2;   // [ MAX*NW ], cont
    float* cTangentImpulse2;   // [ MAX*NW ], cont
    float* cToi;   // [ MAX*NW ], cont
    int* cToiCount;   // [ MAX*NW ], cont
    unsigned char* cToiFlag;   // [ MAX*NW ], cont
    unsigned char* cEnabled;   // [ MAX*NW ], cont
    int* contactCount;   // [ NW ], scalar
#ifdef GB_ENABLE_JOINTS
    // revolute joint pool (opt-in mirror).
    int* jBodyA;   // [ MAXJ*NW ], joint
    int* jBodyB;   // [ MAXJ*NW ], joint
    float* jLocalAnchorAX;   // [ MAXJ*NW ], joint
    float* jLocalAnchorAY;   // [ MAXJ*NW ], joint
    float* jLocalAnchorBX;   // [ MAXJ*NW ], joint
    float* jLocalAnchorBY;   // [ MAXJ*NW ], joint
    float* jImpulseX;   // [ MAXJ*NW ], joint
    float* jImpulseY;   // [ MAXJ*NW ], joint
    int* jointCount;   // [ NW ], scalar
#endif
    unsigned char* stepComplete;   // [ NW ], scalar
    // ---- application extension hook (mirror of GB_WORLD_USER_FIELDS) ----------
    // An application injects a transposed pointer for each field it added to
    // WorldShared by defining GB_WORLD_SOA_USER_FIELDS. The names must match the
    // WorldShared field names so the BODY/CONT/EDGE/SCAL accessors resolve them.
#ifdef GB_WORLD_SOA_USER_FIELDS
    GB_WORLD_SOA_USER_FIELDS
#endif
};

struct GBWorld { WorldPoolsSoA* p; int world; };

// coalesced index: slot*NW + world
#define GBIDX(w, s) ((s)*(w).p->NW + (w).world)
#define BODY(w, field, s)  ((w).p->field[GBIDX(w,s)])
#define CONT(w, field, c)  ((w).p->field[GBIDX(w,c)])
#define EDGE(w, field, e)  ((w).p->field[GBIDX(w,e)])
#define JOINT(w, field, j) ((w).p->field[GBIDX(w,j)])
#define SCAL(w, field)     ((w).p->field[(w).world])