// gb_collide_edge_polygon_test.cu. Micro-test for gbCollideEdgeAndPolygon in
// gb_collision.cuh. 0-ULP versus a self-contained Box2D 2.3.0 reference of
// b2CollideEdgeAndPolygon (the b2EPCollider class), Ref-prefixed.
//
// The subject is the dedicated edge-vs-polygon narrow-phase that replaces the
// two-segment-polygon stand-in the assembled step used for a ground edge against a
// polygon body. The reference is a line-faithful copy of Box2D 2.3.0
// Collision/b2CollideEdge.cpp (b2EPCollider::Collide / ComputeEdgeSeparation /
// ComputePolygonSeparation) over the same fixed inputs.
//
// Scenarios:
//   1. A box resting flat on a horizontal ground edge (face-A, two points).
//   2. A box on a sloped edge (face-A, two points, rotated frame).
//   3. A box pressing a vertex into the edge so the polygon face wins (face-B).
//   4. A box separated from the edge (no contact, pointCount 0).
// Each compares the manifold type, point count, local normal, local point, both clip
// points, and both contact ids at 0 ULP.
//
// Build (frozen flags), self-contained, host or device:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -DGB_ENABLE_POLYGONS -Iinclude -Itest \
//        test/gb_collide_edge_polygon_test.cu -o test/gb_collide_edge_polygon_test
//   ./test/gb_collide_edge_polygon_test
//   Expected: PASS gb_collide_edge_polygon: 0 ULP
#include "gpu_box2d/gb_collision.cuh"
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
// Box2D 2.3.0 reference (Ref-prefixed). b2Math subset + b2EPCollider.
// ---------------------------------------------------------------------------
struct RVec { float x, y; };
static inline RVec rv(float x, float y){ RVec r; r.x=x; r.y=y; return r; }
static inline RVec operator+(RVec a, RVec b){ return rv(a.x+b.x, a.y+b.y); }
static inline RVec operator-(RVec a, RVec b){ return rv(a.x-b.x, a.y-b.y); }
static inline RVec operator*(float s, RVec a){ return rv(s*a.x, s*a.y); }
static inline RVec operator-(RVec a){ return rv(-a.x, -a.y); }
static inline float rDot(RVec a, RVec b){ return a.x*b.x + a.y*b.y; }
static inline float rCross(RVec a, RVec b){ return a.x*b.y - a.y*b.x; }
static inline float rMinF(float a, float b){ return a<b?a:b; }
static inline void rNormalize(RVec& v){
    float len = sqrtf(v.x*v.x + v.y*v.y);
    if (len < 1.19209290e-07f) return;
    float inv = 1.0f/len; v.x*=inv; v.y*=inv;
}
struct RRot { float s, c; };
struct RXf { RVec p; RRot q; };
static inline RVec rMulRV(RRot q, RVec v){ return rv(q.c*v.x - q.s*v.y, q.s*v.x + q.c*v.y); }
static inline RVec rMulXV(RXf t, RVec v){
    return rv((t.q.c*v.x - t.q.s*v.y) + t.p.x, (t.q.s*v.x + t.q.c*v.y) + t.p.y);
}
static inline RVec rMulTXV(RXf t, RVec v){
    float px = v.x - t.p.x, py = v.y - t.p.y;
    return rv(t.q.c*px + t.q.s*py, -t.q.s*px + t.q.c*py);
}
static inline RXf rMulTXX(RXf A, RXf B){
    RXf C;
    C.q.s = A.q.c*B.q.s - A.q.s*B.q.c;
    C.q.c = A.q.c*B.q.c + A.q.s*B.q.s;
    float px = B.p.x - A.p.x, py = B.p.y - A.p.y;
    C.p = rv(A.q.c*px + A.q.s*py, -A.q.s*px + A.q.c*py);
    return C;
}

#define R_MAXV 8
#define R_ANGULAR_SLOP (2.0f / 180.0f * 3.14159265359f)
#define R_POLYGON_RADIUS (2.0f * 0.005f)

struct RPolygon { RVec vertices[R_MAXV]; RVec normals[R_MAXV]; RVec centroid; int count; float radius; };
struct REdge { RVec v0, v1, v2, v3; bool has0, has3; };
struct RClipVertex { RVec v; int iA, iB, tA, tB; };
struct RManifoldPoint { RVec localPoint; int iA, iB, tA, tB; };
struct RManifold { int type; int pointCount; RVec localNormal, localPoint; RManifoldPoint points[2]; };
#define R_FACE_A 1
#define R_FACE_B 2
#define R_VERTEX 0
#define R_FACE   1

static int rClipSegmentToLine(RClipVertex vOut[2], const RClipVertex vIn[2],
                              RVec normal, float offset, int vertexIndexA){
    int numOut = 0;
    float distance0 = rDot(normal, vIn[0].v) - offset;
    float distance1 = rDot(normal, vIn[1].v) - offset;
    if (distance0 <= 0.0f) vOut[numOut++] = vIn[0];
    if (distance1 <= 0.0f) vOut[numOut++] = vIn[1];
    if (distance0 * distance1 < 0.0f){
        float interp = distance0 / (distance0 - distance1);
        vOut[numOut].v = vIn[0].v + interp * (vIn[1].v - vIn[0].v);
        vOut[numOut].iA = vertexIndexA;
        vOut[numOut].iB = vIn[0].iB;
        vOut[numOut].tA = R_VERTEX;
        vOut[numOut].tB = R_FACE;
        ++numOut;
    }
    return numOut;
}

struct REPAxis { int type; int index; float separation; };
#define R_AXIS_UNKNOWN 0
#define R_AXIS_EDGE_A  1
#define R_AXIS_EDGE_B  2
struct RTempPolygon { RVec vertices[R_MAXV]; RVec normals[R_MAXV]; int count; };
struct REPCollider {
    RTempPolygon polygonB; RXf xf; RVec centroidB;
    RVec v0, v1, v2, v3; RVec normal0, normal1, normal2; RVec normal;
    RVec lowerLimit, upperLimit; float radius; bool front;
};
static REPAxis rComputeEdgeSep(const REPCollider& c){
    REPAxis axis; axis.type = R_AXIS_EDGE_A; axis.index = c.front ? 0 : 1; axis.separation = FLT_MAX;
    for (int i = 0; i < c.polygonB.count; ++i){
        float s = rDot(c.normal, c.polygonB.vertices[i] - c.v1);
        if (s < axis.separation) axis.separation = s;
    }
    return axis;
}
static REPAxis rComputePolySep(const REPCollider& c){
    REPAxis axis; axis.type = R_AXIS_UNKNOWN; axis.index = -1; axis.separation = -FLT_MAX;
    RVec perp = rv(-c.normal.y, c.normal.x);
    for (int i = 0; i < c.polygonB.count; ++i){
        RVec n = -c.polygonB.normals[i];
        float s1 = rDot(n, c.polygonB.vertices[i] - c.v1);
        float s2 = rDot(n, c.polygonB.vertices[i] - c.v2);
        float s = rMinF(s1, s2);
        if (s > c.radius){ axis.type=R_AXIS_EDGE_B; axis.index=i; axis.separation=s; return axis; }
        if (rDot(n, perp) >= 0.0f){ if (rDot(n - c.upperLimit, c.normal) < -R_ANGULAR_SLOP) continue; }
        else                      { if (rDot(n - c.lowerLimit, c.normal) < -R_ANGULAR_SLOP) continue; }
        if (s > axis.separation){ axis.type=R_AXIS_EDGE_B; axis.index=i; axis.separation=s; }
    }
    return axis;
}
static void rCollide(RManifold* m, const REdge* edgeA, RXf xfA, const RPolygon* polyB, RXf xfB){
    REPCollider c;
    c.xf = rMulTXX(xfA, xfB);
    c.centroidB = rMulXV(c.xf, polyB->centroid);
    c.v0=edgeA->v0; c.v1=edgeA->v1; c.v2=edgeA->v2; c.v3=edgeA->v3;
    bool has0=edgeA->has0, has3=edgeA->has3;
    RVec edge1 = c.v2 - c.v1; rNormalize(edge1);
    c.normal1 = rv(edge1.y, -edge1.x);
    float offset1 = rDot(c.normal1, c.centroidB - c.v1);
    float offset0=0.0f, offset2=0.0f; bool convex1=false, convex2=false;
    if (has0){
        RVec edge0 = c.v1 - c.v0; rNormalize(edge0);
        c.normal0 = rv(edge0.y, -edge0.x);
        convex1 = rCross(edge0, edge1) >= 0.0f;
        offset0 = rDot(c.normal0, c.centroidB - c.v0);
    }
    if (has3){
        RVec edge2 = c.v3 - c.v2; rNormalize(edge2);
        c.normal2 = rv(edge2.y, -edge2.x);
        convex2 = rCross(edge1, edge2) > 0.0f;
        offset2 = rDot(c.normal2, c.centroidB - c.v2);
    }
    if (has0 && has3){
        if (convex1 && convex2){
            c.front = offset0>=0.0f||offset1>=0.0f||offset2>=0.0f;
            if (c.front){ c.normal=c.normal1; c.lowerLimit=c.normal0; c.upperLimit=c.normal2; }
            else        { c.normal=-c.normal1; c.lowerLimit=-c.normal1; c.upperLimit=-c.normal1; }
        } else if (convex1){
            c.front = offset0>=0.0f||(offset1>=0.0f&&offset2>=0.0f);
            if (c.front){ c.normal=c.normal1; c.lowerLimit=c.normal0; c.upperLimit=c.normal1; }
            else        { c.normal=-c.normal1; c.lowerLimit=-c.normal2; c.upperLimit=-c.normal1; }
        } else if (convex2){
            c.front = offset2>=0.0f||(offset0>=0.0f&&offset1>=0.0f);
            if (c.front){ c.normal=c.normal1; c.lowerLimit=c.normal1; c.upperLimit=c.normal2; }
            else        { c.normal=-c.normal1; c.lowerLimit=-c.normal1; c.upperLimit=-c.normal0; }
        } else {
            c.front = offset0>=0.0f&&offset1>=0.0f&&offset2>=0.0f;
            if (c.front){ c.normal=c.normal1; c.lowerLimit=c.normal1; c.upperLimit=c.normal1; }
            else        { c.normal=-c.normal1; c.lowerLimit=-c.normal2; c.upperLimit=-c.normal0; }
        }
    } else if (has0){
        if (convex1){
            c.front = offset0>=0.0f||offset1>=0.0f;
            if (c.front){ c.normal=c.normal1; c.lowerLimit=c.normal0; c.upperLimit=-c.normal1; }
            else        { c.normal=-c.normal1; c.lowerLimit=c.normal1; c.upperLimit=-c.normal1; }
        } else {
            c.front = offset0>=0.0f&&offset1>=0.0f;
            if (c.front){ c.normal=c.normal1; c.lowerLimit=c.normal1; c.upperLimit=-c.normal1; }
            else        { c.normal=-c.normal1; c.lowerLimit=c.normal1; c.upperLimit=-c.normal0; }
        }
    } else if (has3){
        if (convex2){
            c.front = offset1>=0.0f||offset2>=0.0f;
            if (c.front){ c.normal=c.normal1; c.lowerLimit=-c.normal1; c.upperLimit=c.normal2; }
            else        { c.normal=-c.normal1; c.lowerLimit=-c.normal1; c.upperLimit=c.normal1; }
        } else {
            c.front = offset1>=0.0f&&offset2>=0.0f;
            if (c.front){ c.normal=c.normal1; c.lowerLimit=-c.normal1; c.upperLimit=c.normal1; }
            else        { c.normal=-c.normal1; c.lowerLimit=-c.normal2; c.upperLimit=c.normal1; }
        }
    } else {
        c.front = offset1>=0.0f;
        if (c.front){ c.normal=c.normal1; c.lowerLimit=-c.normal1; c.upperLimit=-c.normal1; }
        else        { c.normal=-c.normal1; c.lowerLimit=c.normal1; c.upperLimit=c.normal1; }
    }
    c.polygonB.count = polyB->count;
    for (int i = 0; i < polyB->count; ++i){
        c.polygonB.vertices[i] = rMulXV(c.xf, polyB->vertices[i]);
        c.polygonB.normals[i]  = rMulRV(c.xf.q, polyB->normals[i]);
    }
    c.radius = 2.0f * R_POLYGON_RADIUS;
    m->pointCount = 0;
    REPAxis edgeAxis = rComputeEdgeSep(c);
    if (edgeAxis.type == R_AXIS_UNKNOWN) return;
    if (edgeAxis.separation > c.radius) return;
    REPAxis polygonAxis = rComputePolySep(c);
    if (polygonAxis.type != R_AXIS_UNKNOWN && polygonAxis.separation > c.radius) return;
    const float kRel = 0.98f, kAbs = 0.001f;
    REPAxis primaryAxis;
    if (polygonAxis.type == R_AXIS_UNKNOWN) primaryAxis = edgeAxis;
    else if (polygonAxis.separation > kRel * edgeAxis.separation + kAbs) primaryAxis = polygonAxis;
    else primaryAxis = edgeAxis;
    RClipVertex ie[2]; struct { int i1,i2; RVec v1,v2,normal,sideNormal1,sideNormal2; float sideOffset1,sideOffset2; } rf;
    if (primaryAxis.type == R_AXIS_EDGE_A){
        m->type = R_FACE_A;
        int bestIndex = 0; float bestValue = rDot(c.normal, c.polygonB.normals[0]);
        for (int i = 1; i < c.polygonB.count; ++i){
            float value = rDot(c.normal, c.polygonB.normals[i]);
            if (value < bestValue){ bestValue = value; bestIndex = i; }
        }
        int i1 = bestIndex; int i2 = i1 + 1 < c.polygonB.count ? i1 + 1 : 0;
        ie[0].v=c.polygonB.vertices[i1]; ie[0].iA=0; ie[0].iB=i1; ie[0].tA=R_FACE; ie[0].tB=R_VERTEX;
        ie[1].v=c.polygonB.vertices[i2]; ie[1].iA=0; ie[1].iB=i2; ie[1].tA=R_FACE; ie[1].tB=R_VERTEX;
        if (c.front){ rf.i1=0; rf.i2=1; rf.v1=c.v1; rf.v2=c.v2; rf.normal=c.normal1; }
        else        { rf.i1=1; rf.i2=0; rf.v1=c.v2; rf.v2=c.v1; rf.normal=-c.normal1; }
    } else {
        m->type = R_FACE_B;
        ie[0].v=c.v1; ie[0].iA=0; ie[0].iB=primaryAxis.index; ie[0].tA=R_VERTEX; ie[0].tB=R_FACE;
        ie[1].v=c.v2; ie[1].iA=0; ie[1].iB=primaryAxis.index; ie[1].tA=R_VERTEX; ie[1].tB=R_FACE;
        rf.i1=primaryAxis.index; rf.i2 = rf.i1 + 1 < c.polygonB.count ? rf.i1 + 1 : 0;
        rf.v1=c.polygonB.vertices[rf.i1]; rf.v2=c.polygonB.vertices[rf.i2]; rf.normal=c.polygonB.normals[rf.i1];
    }
    rf.sideNormal1 = rv(rf.normal.y, -rf.normal.x);
    rf.sideNormal2 = -rf.sideNormal1;
    rf.sideOffset1 = rDot(rf.sideNormal1, rf.v1);
    rf.sideOffset2 = rDot(rf.sideNormal2, rf.v2);
    RClipVertex cp1[2], cp2[2]; int np;
    np = rClipSegmentToLine(cp1, ie, rf.sideNormal1, rf.sideOffset1, rf.i1);
    if (np < 2) return;
    np = rClipSegmentToLine(cp2, cp1, rf.sideNormal2, rf.sideOffset2, rf.i2);
    if (np < 2) return;
    if (primaryAxis.type == R_AXIS_EDGE_A){ m->localNormal=rf.normal; m->localPoint=rf.v1; }
    else { m->localNormal=polyB->normals[rf.i1]; m->localPoint=polyB->vertices[rf.i1]; }
    int pointCount = 0;
    for (int i = 0; i < 2; ++i){
        float separation = rDot(rf.normal, cp2[i].v - rf.v1);
        if (separation <= c.radius){
            RManifoldPoint* cp = &m->points[pointCount];
            if (primaryAxis.type == R_AXIS_EDGE_A){
                cp->localPoint = rMulTXV(c.xf, cp2[i].v);
                cp->iA=cp2[i].iA; cp->iB=cp2[i].iB; cp->tA=cp2[i].tA; cp->tB=cp2[i].tB;
            } else {
                cp->localPoint = cp2[i].v;
                cp->tA=cp2[i].tB; cp->tB=cp2[i].tA; cp->iA=cp2[i].iB; cp->iB=cp2[i].iA;
            }
            ++pointCount;
        }
    }
    m->pointCount = pointCount;
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------
static int gFails = 0; static long gMax = 0;
static void chk(const char* what, int scen, float ref, float got){
    long u = ulpDiff(ref, got); if (u > gMax) gMax = u;
    if (u != 0){ if (gFails == 0) printf("  FAIL %-10s scen=%d ref=%.9g got=%.9g ulp=%ld\n", what, scen, ref, got, u); gFails = 1; }
}
static void chkInt(const char* what, int scen, int ref, int got){
    if (ref != got){ if (gFails == 0) printf("  FAIL %-10s scen=%d ref=%d got=%d\n", what, scen, ref, got); gFails = 1; }
}

// Build a box polygon in both universes from the same half-extents.
static void buildBoxR(RPolygon& p, float hx, float hy){
    p.count = 4; p.radius = R_POLYGON_RADIUS;
    p.vertices[0]=rv(-hx,-hy); p.vertices[1]=rv(hx,-hy); p.vertices[2]=rv(hx,hy); p.vertices[3]=rv(-hx,hy);
    p.normals[0]=rv(0,-1); p.normals[1]=rv(1,0); p.normals[2]=rv(0,1); p.normals[3]=rv(-1,0);
    p.centroid = rv(0,0);
}

static void runScenario(int scen, REdge edgeR, RXf xfA_r, RPolygon polyR, RXf xfB_r,
                        GBEdgeShape edgeG, Xf xfA_g, GBPolygon polyG, Xf xfB_g){
    RManifold rm; rm.pointCount = 0;
    rCollide(&rm, &edgeR, xfA_r, &polyR, xfB_r);
    GBManifold gm; gm.pointCount = 0;
    gbCollideEdgeAndPolygon(gm, edgeG, xfA_g, polyG, xfB_g);

    chkInt("count", scen, rm.pointCount, gm.pointCount);
    if (rm.pointCount == 0){ printf("  scenario %d: no contact (pointCount 0), matched\n", scen); return; }
    printf("  scenario %d: %s, %d point(s), matched\n", scen,
           rm.type == R_FACE_A ? "face-A (edge reference)" : "face-B (polygon reference)", rm.pointCount);
    chkInt("type", scen, rm.type == R_FACE_A ? GB_MANIFOLD_FACE_A : GB_MANIFOLD_FACE_B, gm.type);
    chk("lnormal.x", scen, rm.localNormal.x, gm.localNormal.x);
    chk("lnormal.y", scen, rm.localNormal.y, gm.localNormal.y);
    chk("lpoint.x", scen, rm.localPoint.x, gm.localPoint.x);
    chk("lpoint.y", scen, rm.localPoint.y, gm.localPoint.y);
    chk("p0.x", scen, rm.points[0].localPoint.x, gm.pLocalPoint.x);
    chk("p0.y", scen, rm.points[0].localPoint.y, gm.pLocalPoint.y);
    unsigned int rkey0 = ((unsigned)(rm.points[0].iA&0xff)) | ((unsigned)(rm.points[0].iB&0xff)<<8)
                       | ((unsigned)(rm.points[0].tA&0xff)<<16) | ((unsigned)(rm.points[0].tB&0xff)<<24);
    chkInt("id0", scen, (int)rkey0, (int)gm.id0);
    if (rm.pointCount > 1){
        chk("p1.x", scen, rm.points[1].localPoint.x, gm.pLocalPoint2.x);
        chk("p1.y", scen, rm.points[1].localPoint.y, gm.pLocalPoint2.y);
        unsigned int rkey1 = ((unsigned)(rm.points[1].iA&0xff)) | ((unsigned)(rm.points[1].iB&0xff)<<8)
                           | ((unsigned)(rm.points[1].tA&0xff)<<16) | ((unsigned)(rm.points[1].tB&0xff)<<24);
        chkInt("id1", scen, (int)rkey1, (int)gm.id1);
    }
}

int main(){
    printf("Edge-polygon micro-test: gbCollideEdgeAndPolygon vs Box2D 2.3.0 b2CollideEdgeAndPolygon\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    RRot id; id.s = 0.0f; id.c = 1.0f;
    Rot   gid; gid.s = 0.0f; gid.c = 1.0f;

    // Scenario 1: box flat on a horizontal ground edge from (-4,0) to (4,0).
    {
        REdge eR; eR.v1=rv(-4,0); eR.v2=rv(4,0); eR.v0=rv(0,0); eR.v3=rv(0,0); eR.has0=false; eR.has3=false;
        RXf xfA; xfA.p=rv(0,0); xfA.q=id;
        RPolygon pR; buildBoxR(pR, 0.5f, 0.5f);
        RXf xfB; xfB.p=rv(0.0f, 0.5f); xfB.q=id;   // box bottom at y=0
        GBEdgeShape eG; eG.vertex1=v2(-4,0); eG.vertex2=v2(4,0); eG.vertex0=v2(0,0); eG.vertex3=v2(0,0); eG.hasVertex0=false; eG.hasVertex3=false;
        Xf gxfA; gxfA.p=v2(0,0); gxfA.q=gid;
        GBPolygon pG; gbPolygonSetAsBox(pG, 0.5f, 0.5f);
        Xf gxfB; gxfB.p=v2(0.0f, 0.5f); gxfB.q=gid;
        runScenario(1, eR, xfA, pR, xfB, eG, gxfA, pG, gxfB);
    }
    // Scenario 2: box resting on a sloped edge (frame A rotated 0.3 rad). The box sits
    // on the edge surface in frame A at local (0, 0.5), transformed to world by xfA.
    {
        float ang = 0.3f; float s = sinf(ang), cs = cosf(ang);
        RRot rq; rq.s=s; rq.c=cs; Rot gq; gq.s=s; gq.c=cs;
        // world center of a box resting on the rotated edge at local (0, 0.49)
        float bx = cs*0.0f - s*0.49f, by = s*0.0f + cs*0.49f;
        REdge eR; eR.v1=rv(-4,0); eR.v2=rv(4,0); eR.v0=rv(0,0); eR.v3=rv(0,0); eR.has0=false; eR.has3=false;
        RXf xfA; xfA.p=rv(0,0); xfA.q=rq;
        RPolygon pR; buildBoxR(pR, 0.5f, 0.5f);
        RXf xfB; xfB.p=rv(bx, by); xfB.q=rq;
        GBEdgeShape eG; eG.vertex1=v2(-4,0); eG.vertex2=v2(4,0); eG.vertex0=v2(0,0); eG.vertex3=v2(0,0); eG.hasVertex0=false; eG.hasVertex3=false;
        Xf gxfA; gxfA.p=v2(0,0); gxfA.q=gq;
        GBPolygon pG; gbPolygonSetAsBox(pG, 0.5f, 0.5f);
        Xf gxfB; gxfB.p=v2(bx, by); gxfB.q=gq;
        runScenario(2, eR, xfA, pR, xfB, eG, gxfA, pG, gxfB);
    }
    // Scenario 3: a large box overhangs a short edge endpoint, rotated, so the
    // polygon face wins the separation test and the reference face is on the polygon
    // (face-B path).
    {
        float ang = 0.5f; RRot rq; rq.s=sinf(ang); rq.c=cosf(ang); Rot gq; gq.s=sinf(ang); gq.c=cosf(ang);
        REdge eR; eR.v1=rv(-0.5f,0); eR.v2=rv(0.5f,0); eR.v0=rv(0,0); eR.v3=rv(0,0); eR.has0=false; eR.has3=false;
        RXf xfA; xfA.p=rv(0,0); xfA.q=id;
        RPolygon pR; buildBoxR(pR, 1.0f, 1.0f);
        RXf xfB; xfB.p=rv(1.0f, 0.99f); xfB.q=rq;
        GBEdgeShape eG; eG.vertex1=v2(-0.5f,0); eG.vertex2=v2(0.5f,0); eG.vertex0=v2(0,0); eG.vertex3=v2(0,0); eG.hasVertex0=false; eG.hasVertex3=false;
        Xf gxfA; gxfA.p=v2(0,0); gxfA.q=gid;
        GBPolygon pG; gbPolygonSetAsBox(pG, 1.0f, 1.0f);
        Xf gxfB; gxfB.p=v2(1.0f, 0.99f); gxfB.q=gq;
        runScenario(3, eR, xfA, pR, xfB, eG, gxfA, pG, gxfB);
    }
    // Scenario 4: box clearly separated above the edge (no contact).
    {
        REdge eR; eR.v1=rv(-4,0); eR.v2=rv(4,0); eR.v0=rv(0,0); eR.v3=rv(0,0); eR.has0=false; eR.has3=false;
        RXf xfA; xfA.p=rv(0,0); xfA.q=id;
        RPolygon pR; buildBoxR(pR, 0.5f, 0.5f);
        RXf xfB; xfB.p=rv(0.0f, 2.0f); xfB.q=id;
        GBEdgeShape eG; eG.vertex1=v2(-4,0); eG.vertex2=v2(4,0); eG.vertex0=v2(0,0); eG.vertex3=v2(0,0); eG.hasVertex0=false; eG.hasVertex3=false;
        Xf gxfA; gxfA.p=v2(0,0); gxfA.q=gid;
        GBPolygon pG; gbPolygonSetAsBox(pG, 0.5f, 0.5f);
        Xf gxfB; gxfB.p=v2(0.0f, 2.0f); gxfB.q=gid;
        runScenario(4, eR, xfA, pR, xfB, eG, gxfA, pG, gxfB);
    }

    if (!gFails){
        printf("PASS gb_collide_edge_polygon: 0 ULP (face-A, face-B, separated), maxUlp=%ld\n", gMax);
        return 0;
    }
    printf("FAIL gb_collide_edge_polygon: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
