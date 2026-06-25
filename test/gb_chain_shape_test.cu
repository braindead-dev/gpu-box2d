// gb_chain_shape_test.cu. Micro-test for gb_chain_shape.cuh. 0-ULP versus a
// self-contained Box2D 2.3.0 reference of b2ChainShape's CreateChain / CreateLoop /
// GetChildEdge, Ref-prefixed, plus an integration check that a chain child edge drives
// gbCollideEdgeAndPolygon.
//
// Two parts:
//   1. Child-edge generation. For an open chain and a closed loop, every child edge's
//      vertex1, vertex2, vertex0, vertex3, and the hasVertex0 / hasVertex3 flags match
//      the Box2D reference bit-for-bit (positions at 0 ULP, flags exact). This is the
//      0-ULP claim.
//   2. Integration. A box resting on an interior child edge is collided through
//      gbCollideEdgeAndPolygon and produces a finite manifold, confirming the chain
//      wires into the adjacency-aware edge collider. The collider's bit-exactness
//      against Box2D is established in gb_collide_edge_polygon_test.cu.
//
// Build (frozen flags), self-contained, host or device:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -DGB_ENABLE_POLYGONS -Iinclude -Itest \
//        test/gb_chain_shape_test.cu -o test/gb_chain_shape_test
//   ./test/gb_chain_shape_test
//   Expected: PASS gb_chain_shape: 0 ULP
#include "gpu_box2d/gb_chain_shape.cuh"
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstdint>

inline long ulpDiff(float a, float b){
    int ai = *(int*)&a, bi = *(int*)&b;
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return labs((long)ai - (long)bi);
}

// ---------------------------------------------------------------------------
// Box2D 2.3.0 b2ChainShape reference (Ref-prefixed).
// ---------------------------------------------------------------------------
struct RVec { float x, y; };
static inline RVec rv(float x, float y){ RVec r; r.x=x; r.y=y; return r; }
#define R_MAXC 16
#define R_POLYGON_RADIUS (2.0f * 0.005f)

struct RChain { RVec vertices[R_MAXC + 1]; int count; RVec prev, next; bool hasPrev, hasNext; float radius; };
struct REdgeR { RVec v0, v1, v2, v3; bool has0, has3; };

static void rCreateChain(RChain& c, const RVec* vs, int count){
    c.count = count;
    for (int i = 0; i < count; ++i) c.vertices[i] = vs[i];
    c.prev = rv(0,0); c.next = rv(0,0); c.hasPrev = false; c.hasNext = false;
    c.radius = R_POLYGON_RADIUS;
}
static void rCreateLoop(RChain& c, const RVec* vs, int count){
    c.count = count + 1;
    for (int i = 0; i < count; ++i) c.vertices[i] = vs[i];
    c.vertices[count] = vs[0];
    c.prev = vs[count - 1]; c.next = vs[1]; c.hasPrev = true; c.hasNext = true;
    c.radius = R_POLYGON_RADIUS;
}
static void rGetChildEdge(const RChain& c, REdgeR& e, int index){
    e.v1 = c.vertices[index];
    e.v2 = c.vertices[index + 1];
    if (index > 0){ e.v0 = c.vertices[index - 1]; e.has0 = true; }
    else { e.v0 = c.prev; e.has0 = c.hasPrev; }
    if (index < c.count - 2){ e.v3 = c.vertices[index + 2]; e.has3 = true; }
    else { e.v3 = c.next; e.has3 = c.hasNext; }
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------
static int gFails = 0; static long gMax = 0;
static void chk(const char* what, int idx, float ref, float got){
    long u = ulpDiff(ref, got); if (u > gMax) gMax = u;
    if (u != 0){ if (gFails == 0) printf("  FAIL %-10s idx=%d ref=%.9g got=%.9g ulp=%ld\n", what, idx, ref, got, u); gFails = 1; }
}
static void chkFlag(const char* what, int idx, bool ref, bool got){
    if (ref != got){ if (gFails == 0) printf("  FAIL %-10s idx=%d ref=%d got=%d\n", what, idx, (int)ref, (int)got); gFails = 1; }
}

static void compareChildEdges(const char* name, const RChain& rc, const GBChainShape& gc){
    int rChildren = rc.count - 1;
    int gChildren = gbChainChildCount(gc);
    if (rChildren != gChildren){ printf("  FAIL %s child count ref=%d got=%d\n", name, rChildren, gChildren); gFails = 1; return; }
    for (int i = 0; i < rChildren; ++i){
        REdgeR re; rGetChildEdge(rc, re, i);
        GBEdgeShape ge; gbChainGetChildEdge(gc, ge, i);
        chk("v1.x", i, re.v1.x, ge.vertex1.x); chk("v1.y", i, re.v1.y, ge.vertex1.y);
        chk("v2.x", i, re.v2.x, ge.vertex2.x); chk("v2.y", i, re.v2.y, ge.vertex2.y);
        chkFlag("has0", i, re.has0, ge.hasVertex0);
        chkFlag("has3", i, re.has3, ge.hasVertex3);
        if (re.has0){ chk("v0.x", i, re.v0.x, ge.vertex0.x); chk("v0.y", i, re.v0.y, ge.vertex0.y); }
        if (re.has3){ chk("v3.x", i, re.v3.x, ge.vertex3.x); chk("v3.y", i, re.v3.y, ge.vertex3.y); }
    }
    printf("  %-14s %d child edges, %s\n", name, rChildren, gFails ? "DIVERGED" : "matched");
}

// Integration check: a chain child edge feeds gbCollideEdgeAndPolygon and produces a
// finite contact for a box resting on it. The collider's bit-exactness against Box2D is
// established separately in gb_collide_edge_polygon_test.cu; this confirms the chain
// wires into it and the adjacency-carrying child edge drives a manifold.
static void integration(const GBChainShape& gc, int childIndex){
    GBEdgeShape ge; gbChainGetChildEdge(gc, ge, childIndex);
    V2 mid = 0.5f*(ge.vertex1 + ge.vertex2);
    Rot id; id.s = 0.0f; id.c = 1.0f;
    Xf xfA; xfA.p = v2(0,0); xfA.q = id;
    GBPolygon box; gbPolygonSetAsBox(box, 0.5f, 0.5f);
    Xf xfB; xfB.p = v2(mid.x, mid.y + 0.5f); xfB.q = id;   // box bottom on the edge
    GBManifold m; m.pointCount = 0;
    gbCollideEdgeAndPolygon(m, ge, xfA, box, xfB);
    bool ok = m.pointCount > 0;
    for (int i = 0; i < m.pointCount; ++i){
        V2 p = i == 0 ? m.pLocalPoint : m.pLocalPoint2;
        if (!(isfinite(p.x) && isfinite(p.y))) ok = false;
    }
    if (!ok){ printf("  FAIL integration child %d produced no finite contact (count=%d)\n", childIndex, m.pointCount); gFails = 1; return; }
    printf("  integration   child %d: %d-point manifold via chain (finite)\n", childIndex, m.pointCount);
}

int main(){
    printf("Chain shape micro-test: gb_chain_shape vs Box2D 2.3.0 b2ChainShape\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    // An open chain: a stepped ground contour.
    {
        RVec rverts[5] = { rv(-4,0), rv(-2,0), rv(0,1), rv(2,0), rv(4,0) };
        V2   gverts[5] = { v2(-4,0), v2(-2,0), v2(0,1), v2(2,0), v2(4,0) };
        RChain rc; rCreateChain(rc, rverts, 5);
        GBChainShape gc; gbChainCreateChain(gc, gverts, 5);
        compareChildEdges("open chain", rc, gc);
        integration(gc, 1);   // the (-2,0)->(0,1) child edge, has both neighbors
    }
    // A closed loop: a diamond.
    {
        RVec rverts[4] = { rv(-2,0), rv(0,-2), rv(2,0), rv(0,2) };
        V2   gverts[4] = { v2(-2,0), v2(0,-2), v2(2,0), v2(0,2) };
        RChain rc; rCreateLoop(rc, rverts, 4);
        GBChainShape gc; gbChainCreateLoop(gc, gverts, 4);
        compareChildEdges("closed loop", rc, gc);
        integration(gc, 0);
    }

    if (!gFails){
        printf("PASS gb_chain_shape: 0 ULP (child-edge generation; edge-polygon integration finite), maxUlp=%ld\n", gMax);
        return 0;
    }
    printf("FAIL gb_chain_shape: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
