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
    int* tier;   // [ MAX*NW ], body
    int* bodyType;   // [ MAX*NW ], body
    float* sleepTime;   // [ MAX*NW ], body
    unsigned char* awake;   // [ MAX*NW ], body
    unsigned char* alive;   // [ MAX*NW ], body
    int* bodyCount;   // [ NW ], scalar
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
    float* cToi;   // [ MAX*NW ], cont
    int* cToiCount;   // [ MAX*NW ], cont
    unsigned char* cToiFlag;   // [ MAX*NW ], cont
    unsigned char* cEnabled;   // [ MAX*NW ], cont
    int* contactCount;   // [ NW ], scalar
    int* pairLo;   // [ MAX*NW ], pair
    int* pairHi;   // [ MAX*NW ], pair
    int* pairA;   // [ MAX*NW ], pair
    int* pairB;   // [ MAX*NW ], pair
    int* pairCount;   // [ NW ], scalar
    unsigned char* stepComplete;   // [ NW ], scalar
    int* score;   // [ NW ], scalar
    int* drops;   // [ NW ], scalar
    int* maxTier;   // [ NW ], scalar
    int* goldapples;   // [ NW ], scalar
    int* cur;   // [ NW ], scalar
    int* nxt;   // [ NW ], scalar
    unsigned char* dead;   // [ NW ], scalar
    unsigned long long* rng;   // [ NW ], scalar
    float* age;   // [ MAX*NW ], body
    float* outline;   // [ MAX*NW ], body
};

struct GBWorld { WorldPoolsSoA* p; int world; };

// coalesced index: slot*NW + world
#define GBIDX(w, s) ((s)*(w).p->NW + (w).world)
#define BODY(w, field, s)  ((w).p->field[GBIDX(w,s)])
#define CONT(w, field, c)  ((w).p->field[GBIDX(w,c)])
#define EDGE(w, field, e)  ((w).p->field[GBIDX(w,e)])
#define SCAL(w, field)     ((w).p->field[(w).world])