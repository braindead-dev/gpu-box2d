// gb_collision.cuh. The narrow-phase, written against the gb_pools accessor
// contract. The math is the validated narrow-phase, expressed against the gb_* type
// universe (GBManifold/GBWorldManifold from gb_contact_types.cuh, V2/Rot/Xf and ops
// from gb_math.cuh) and reading world state only through BODY/CONT/EDGE/SCAL.
//
// Line-faithful to Box2D 2.3.0:
//   Collision/b2CollideCircle.cpp  (b2CollideCircles)
//   Collision/b2CollideEdge.cpp    (b2CollideEdgeAndCircle, single-edge regions)
//   Collision/b2Collision.cpp      (b2WorldManifold::Initialize, 1-point)
//   Dynamics/b2Contact.cpp         (b2Contact::Update, touching flip, 1-point)
//
// This module covers circles, single-edges, and convex polygons. Circle and
// circle-edge manifolds are 1-point; polygon contacts produce one or two points
// through the clip path (b2CollidePolygons / b2CollidePolygonAndCircle). The
// two-point block solve that consumes them lives in gb_contact_solver.cuh.
//
// Polygon-source faithfulness (Box2D 2.3.0):
//   Collision/b2CollidePolygon.cpp (b2CollidePolygons, b2FindMaxSeparation,
//     b2EdgeSeparation, b2FindIncidentEdge)
//   Collision/b2CollideCircle.cpp  (b2CollidePolygonAndCircle)
//   Collision/b2Collision.cpp      (b2ClipSegmentToLine, b2WorldManifold 2-point)
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true (mirrors
// the CPU's -ffp-contract=off -mfpmath=sse). Changing these breaks bit-identicality.
#pragma once
#include "gpu_box2d/gb_pools.cuh"
#include "gpu_box2d/gb_contact_types.cuh"
#include "gpu_box2d/gb_polygon.cuh"

// =========================== Narrow-phase ===================================
// b2CollideCircles (b2CollideCircle.cpp:23). circle m_p == (0,0).
GB_HD inline void gbCollideCircles(GBManifold& m, float rA, Xf xfA, float rB, Xf xfB){
    m.pointCount = 0;
    V2 pA = b2MulTV(xfA, v2(0,0));
    V2 pB = b2MulTV(xfB, v2(0,0));
    V2 d = pB - pA;
    float distSqr = b2Dot(d,d);
    float radius = rA + rB;
    if (distSqr > radius*radius) return;
    m.type = GB_MANIFOLD_CIRCLES;
    m.localPoint  = v2(0,0);     // circleA->m_p
    m.localNormal = v2(0,0);
    m.pointCount = 1;
    m.pLocalPoint = v2(0,0);     // circleB->m_p
    m.id0 = 0;                   // points[0].id.key
}

// b2CollideEdgeAndCircle (b2CollideEdge.cpp:27). Edge has no vertex0/vertex3
// (single-segment edges), so the connectivity early-outs never trigger.
GB_HD inline void gbCollideEdgeAndCircle(GBManifold& m, V2 A, V2 B, float edgeR,
                                         float circR, Xf xfA, Xf xfB){
    m.pointCount = 0;
    V2 Q = b2MulTinvV(xfA, b2MulTV(xfB, v2(0,0)));   // circle m_p == 0
    V2 e = B - A;
    float u = b2Dot(e, B - Q);
    float v = b2Dot(e, Q - A);
    float radius = edgeR + circR;

    if (v <= 0.0f){                      // Region A (vertex1)
        V2 P = A;
        V2 d = Q - P;
        float dd = b2Dot(d,d);
        if (dd > radius*radius) return;
        // m_hasVertex0 == false => no connectivity check
        m.pointCount = 1; m.type = GB_MANIFOLD_CIRCLES;
        m.localNormal = v2(0,0); m.localPoint = P;
        m.pLocalPoint = v2(0,0); m.id0 = 0;
        return;
    }
    if (u <= 0.0f){                      // Region B (vertex2)
        V2 P = B;
        V2 d = Q - P;
        float dd = b2Dot(d,d);
        if (dd > radius*radius) return;
        // m_hasVertex3 == false => no connectivity check
        m.pointCount = 1; m.type = GB_MANIFOLD_CIRCLES;
        m.localNormal = v2(0,0); m.localPoint = P;
        m.pLocalPoint = v2(0,0); m.id0 = 0;
        return;
    }
    // Region AB (face)
    float den = b2Dot(e,e);
    V2 P = (1.0f/den) * (u*A + v*B);
    V2 d = Q - P;
    float dd = b2Dot(d,d);
    if (dd > radius*radius) return;
    V2 n = v2(-e.y, e.x);
    if (b2Dot(n, Q - A) < 0.0f) n = v2(-n.x, -n.y);
    b2Normalize(n);
    m.pointCount = 1; m.type = GB_MANIFOLD_FACE_A;
    m.localNormal = n; m.localPoint = A;
    m.pLocalPoint = v2(0,0); m.id0 = 0;
}

// b2WorldManifold::Initialize (b2Collision.cpp:22). Handles e_circles, e_faceA, and
// e_faceB, for one or two points. The clip points (pLocalPoint / pLocalPoint2) feed
// the face cases; point 0 alone is used when pointCount is 1.
#define GB_COLLISION_PROVIDED 1
GB_HD inline void gbWorldManifoldInit(GBWorldManifold& wm, const GBManifold& m,
                                      Xf xfA, float rA, Xf xfB, float rB){
    if (m.pointCount == 0) return;
    if (m.type == GB_MANIFOLD_CIRCLES){
        wm.normal = v2(1.0f, 0.0f);
        V2 pointA = b2MulTV(xfA, m.localPoint);
        V2 pointB = b2MulTV(xfB, m.pLocalPoint);
        if (b2DistanceSquared(pointA, pointB) > GB_EPSILON*GB_EPSILON){
            wm.normal = pointB - pointA;
            b2Normalize(wm.normal);
        }
        V2 cA = pointA + rA*wm.normal;
        V2 cB = pointB - rB*wm.normal;
        wm.point0 = 0.5f*(cA + cB);
    } else if (m.type == GB_MANIFOLD_FACE_A){
        wm.normal = b2MulRV(xfA.q, m.localNormal);
        V2 planePoint = b2MulTV(xfA, m.localPoint);
        V2 cp0 = b2MulTV(xfB, m.pLocalPoint);
        wm.point0 = 0.5f*((cp0 + (rA - b2Dot(cp0 - planePoint, wm.normal))*wm.normal)
                          + (cp0 - rB*wm.normal));
        if (m.pointCount > 1){
            V2 cp1 = b2MulTV(xfB, m.pLocalPoint2);
            wm.point1 = 0.5f*((cp1 + (rA - b2Dot(cp1 - planePoint, wm.normal))*wm.normal)
                              + (cp1 - rB*wm.normal));
        }
    } else { // GB_MANIFOLD_FACE_B
        wm.normal = b2MulRV(xfB.q, m.localNormal);
        V2 planePoint = b2MulTV(xfB, m.localPoint);
        V2 cp0 = b2MulTV(xfA, m.pLocalPoint);
        wm.point0 = 0.5f*((cp0 + (rB - b2Dot(cp0 - planePoint, wm.normal))*wm.normal)
                          + (cp0 - rA*wm.normal));
        if (m.pointCount > 1){
            V2 cp1 = b2MulTV(xfA, m.pLocalPoint2);
            wm.point1 = 0.5f*((cp1 + (rB - b2Dot(cp1 - planePoint, wm.normal))*wm.normal)
                              + (cp1 - rA*wm.normal));
        }
        wm.normal = -wm.normal;   // ensure normal points from A to B
    }
}

// =========================== Polygon narrow-phase ===========================
// b2ContactFeature packed into a key (indexA | indexB<<8 | typeA<<16 | typeB<<24),
// matching the byte layout of Box2D's b2ContactID. e_vertex=0, e_face=1.
#define GB_FEATURE_VERTEX 0
#define GB_FEATURE_FACE   1
GB_HD inline unsigned int gbFeatureKey(int indexA, int indexB, int typeA, int typeB){
    return ((unsigned int)(indexA & 0xff))
         | ((unsigned int)(indexB & 0xff) << 8)
         | ((unsigned int)(typeA  & 0xff) << 16)
         | ((unsigned int)(typeB  & 0xff) << 24);
}

// A clip vertex carries its world position and its contact-feature key parts.
struct GBClipVertex { V2 v; int indexA, indexB, typeA, typeB; };

// b2ClipSegmentToLine (b2Collision.cpp:198). Sutherland-Hodgman clip of a segment
// against a half-plane. Returns the number of output points (0, 1, or 2).
GB_HD inline int gbClipSegmentToLine(GBClipVertex vOut[2], const GBClipVertex vIn[2],
                                     V2 normal, float offset, int vertexIndexA){
    int numOut = 0;
    float distance0 = b2Dot(normal, vIn[0].v) - offset;
    float distance1 = b2Dot(normal, vIn[1].v) - offset;
    if (distance0 <= 0.0f) vOut[numOut++] = vIn[0];
    if (distance1 <= 0.0f) vOut[numOut++] = vIn[1];
    if (distance0 * distance1 < 0.0f){
        float interp = distance0 / (distance0 - distance1);
        vOut[numOut].v = vIn[0].v + interp * (vIn[1].v - vIn[0].v);
        vOut[numOut].indexA = vertexIndexA;
        vOut[numOut].indexB = vIn[0].indexB;
        vOut[numOut].typeA = GB_FEATURE_VERTEX;
        vOut[numOut].typeB = GB_FEATURE_FACE;
        ++numOut;
    }
    return numOut;
}

// b2EdgeSeparation (b2CollidePolygon.cpp:23). Separation of poly2 from edge1 of
// poly1 along edge1's world normal.
GB_HD inline float gbEdgeSeparation(const GBPolygon& poly1, Xf xf1, int edge1,
                                    const GBPolygon& poly2, Xf xf2){
    V2 normal1World = b2MulRV(xf1.q, poly1.normals[edge1]);
    V2 normal1 = b2MulTinvV_q(xf2.q, normal1World);
    int index = 0;
    float minDot = GB_MAXFLOAT;
    for (int i = 0; i < poly2.count; ++i){
        float dot = b2Dot(poly2.vertices[i], normal1);
        if (dot < minDot){ minDot = dot; index = i; }
    }
    V2 v1 = b2MulTV(xf1, poly1.vertices[edge1]);
    V2 v2w = b2MulTV(xf2, poly2.vertices[index]);
    return b2Dot(v2w - v1, normal1World);
}

// b2FindMaxSeparation (b2CollidePolygon.cpp:59). Best edge of poly1 and its
// separation against poly2. Writes the edge index.
GB_HD inline float gbFindMaxSeparation(int& edgeIndex,
                                       const GBPolygon& poly1, Xf xf1,
                                       const GBPolygon& poly2, Xf xf2){
    int count1 = poly1.count;
    V2 d = b2MulTV(xf2, poly2.centroid) - b2MulTV(xf1, poly1.centroid);
    V2 dLocal1 = b2MulTinvV_q(xf1.q, d);
    int edge = 0;
    float maxDot = -GB_MAXFLOAT;
    for (int i = 0; i < count1; ++i){
        float dot = b2Dot(poly1.normals[i], dLocal1);
        if (dot > maxDot){ maxDot = dot; edge = i; }
    }
    float s = gbEdgeSeparation(poly1, xf1, edge, poly2, xf2);
    int prevEdge = edge - 1 >= 0 ? edge - 1 : count1 - 1;
    float sPrev = gbEdgeSeparation(poly1, xf1, prevEdge, poly2, xf2);
    int nextEdge = edge + 1 < count1 ? edge + 1 : 0;
    float sNext = gbEdgeSeparation(poly1, xf1, nextEdge, poly2, xf2);
    int bestEdge; float bestSeparation; int increment;
    if (sPrev > s && sPrev > sNext){ increment=-1; bestEdge=prevEdge; bestSeparation=sPrev; }
    else if (sNext > s){ increment=1; bestEdge=nextEdge; bestSeparation=sNext; }
    else { edgeIndex = edge; return s; }
    for (;;){
        if (increment == -1) edge = bestEdge - 1 >= 0 ? bestEdge - 1 : count1 - 1;
        else                 edge = bestEdge + 1 < count1 ? bestEdge + 1 : 0;
        s = gbEdgeSeparation(poly1, xf1, edge, poly2, xf2);
        if (s > bestSeparation){ bestEdge = edge; bestSeparation = s; }
        else break;
    }
    edgeIndex = bestEdge;
    return bestSeparation;
}

// b2FindIncidentEdge (b2CollidePolygon.cpp:141). The edge of poly2 most anti-parallel
// to reference edge1 of poly1, as two clip vertices.
GB_HD inline void gbFindIncidentEdge(GBClipVertex c[2],
                                     const GBPolygon& poly1, Xf xf1, int edge1,
                                     const GBPolygon& poly2, Xf xf2){
    V2 normal1 = b2MulTinvV_q(xf2.q, b2MulRV(xf1.q, poly1.normals[edge1]));
    int index = 0;
    float minDot = GB_MAXFLOAT;
    for (int i = 0; i < poly2.count; ++i){
        float dot = b2Dot(normal1, poly2.normals[i]);
        if (dot < minDot){ minDot = dot; index = i; }
    }
    int i1 = index;
    int i2 = i1 + 1 < poly2.count ? i1 + 1 : 0;
    c[0].v = b2MulTV(xf2, poly2.vertices[i1]);
    c[0].indexA = edge1; c[0].indexB = i1;
    c[0].typeA = GB_FEATURE_FACE; c[0].typeB = GB_FEATURE_VERTEX;
    c[1].v = b2MulTV(xf2, poly2.vertices[i2]);
    c[1].indexA = edge1; c[1].indexB = i2;
    c[1].typeA = GB_FEATURE_FACE; c[1].typeB = GB_FEATURE_VERTEX;
}

// b2CollidePolygons (b2CollidePolygon.cpp:193). Reference-face selection, incident-edge
// clipping, and the up-to-two-point manifold. The normal points from A to B.
GB_HD inline void gbCollidePolygons(GBManifold& m,
                                    const GBPolygon& polyA, Xf xfA,
                                    const GBPolygon& polyB, Xf xfB){
    m.pointCount = 0;
    float totalRadius = polyA.radius + polyB.radius;
    int edgeA = 0;
    float separationA = gbFindMaxSeparation(edgeA, polyA, xfA, polyB, xfB);
    if (separationA > totalRadius) return;
    int edgeB = 0;
    float separationB = gbFindMaxSeparation(edgeB, polyB, xfB, polyA, xfA);
    if (separationB > totalRadius) return;

    const GBPolygon* poly1; const GBPolygon* poly2;
    Xf xf1, xf2; int edge1; int flip;
    const float k_relativeTol = 0.98f;
    const float k_absoluteTol = 0.001f;
    if (separationB > k_relativeTol * separationA + k_absoluteTol){
        poly1 = &polyB; poly2 = &polyA; xf1 = xfB; xf2 = xfA; edge1 = edgeB;
        m.type = GB_MANIFOLD_FACE_B; flip = 1;
    } else {
        poly1 = &polyA; poly2 = &polyB; xf1 = xfA; xf2 = xfB; edge1 = edgeA;
        m.type = GB_MANIFOLD_FACE_A; flip = 0;
    }

    GBClipVertex incidentEdge[2];
    gbFindIncidentEdge(incidentEdge, *poly1, xf1, edge1, *poly2, xf2);

    int count1 = poly1->count;
    int iv1 = edge1;
    int iv2 = edge1 + 1 < count1 ? edge1 + 1 : 0;
    V2 v11 = poly1->vertices[iv1];
    V2 v12 = poly1->vertices[iv2];
    V2 localTangent = v12 - v11; b2Normalize(localTangent);
    V2 localNormal = b2CrossVS(localTangent, 1.0f);
    V2 planePoint = 0.5f * (v11 + v12);
    V2 tangent = b2MulRV(xf1.q, localTangent);
    V2 normal = b2CrossVS(tangent, 1.0f);
    v11 = b2MulTV(xf1, v11);
    v12 = b2MulTV(xf1, v12);
    float frontOffset = b2Dot(normal, v11);
    float sideOffset1 = -b2Dot(tangent, v11) + totalRadius;
    float sideOffset2 = b2Dot(tangent, v12) + totalRadius;

    GBClipVertex clipPoints1[2];
    GBClipVertex clipPoints2[2];
    int np;
    np = gbClipSegmentToLine(clipPoints1, incidentEdge, -tangent, sideOffset1, iv1);
    if (np < 2) return;
    np = gbClipSegmentToLine(clipPoints2, clipPoints1, tangent, sideOffset2, iv2);
    if (np < 2) return;

    m.localNormal = localNormal;
    m.localPoint = planePoint;
    int pointCount = 0;
    for (int i = 0; i < GB_MAX_MANIFOLD_POINTS; ++i){
        float separation = b2Dot(normal, clipPoints2[i].v) - frontOffset;
        if (separation <= totalRadius){
            V2 localPt = b2MulTinvV(xf2, clipPoints2[i].v);
            int idxA = clipPoints2[i].indexA, idxB = clipPoints2[i].indexB;
            int tyA = clipPoints2[i].typeA, tyB = clipPoints2[i].typeB;
            unsigned int key = flip ? gbFeatureKey(idxB, idxA, tyB, tyA)
                                    : gbFeatureKey(idxA, idxB, tyA, tyB);
            if (pointCount == 0){ m.pLocalPoint = localPt; m.id0 = key; }
            else                { m.pLocalPoint2 = localPt; m.id1 = key; }
            ++pointCount;
        }
    }
    m.pointCount = pointCount;
}

// b2CollidePolygonAndCircle (b2CollideCircle.cpp:51). One-point face-A manifold with
// the circle at frame B (m_p == 0). The normal points from A to B.
GB_HD inline void gbCollidePolygonAndCircle(GBManifold& m,
                                            const GBPolygon& polyA, Xf xfA,
                                            float circR, Xf xfB){
    m.pointCount = 0;
    V2 c = b2MulTV(xfB, v2(0,0));
    V2 cLocal = b2MulTinvV(xfA, c);
    int normalIndex = 0;
    float separation = -GB_MAXFLOAT;
    float radius = polyA.radius + circR;
    int vertexCount = polyA.count;
    for (int i = 0; i < vertexCount; ++i){
        float s = b2Dot(polyA.normals[i], cLocal - polyA.vertices[i]);
        if (s > radius) return;
        if (s > separation){ separation = s; normalIndex = i; }
    }
    int vertIndex1 = normalIndex;
    int vertIndex2 = vertIndex1 + 1 < vertexCount ? vertIndex1 + 1 : 0;
    V2 v1 = polyA.vertices[vertIndex1];
    V2 v2v = polyA.vertices[vertIndex2];
    if (separation < GB_EPSILON){
        m.pointCount = 1; m.type = GB_MANIFOLD_FACE_A;
        m.localNormal = polyA.normals[normalIndex];
        m.localPoint = 0.5f * (v1 + v2v);
        m.pLocalPoint = v2(0,0); m.id0 = 0;
        return;
    }
    float u1 = b2Dot(cLocal - v1, v2v - v1);
    float u2 = b2Dot(cLocal - v2v, v1 - v2v);
    if (u1 <= 0.0f){
        if (b2DistanceSquared(cLocal, v1) > radius*radius) return;
        m.pointCount = 1; m.type = GB_MANIFOLD_FACE_A;
        V2 n = cLocal - v1; b2Normalize(n);
        m.localNormal = n; m.localPoint = v1;
        m.pLocalPoint = v2(0,0); m.id0 = 0;
    } else if (u2 <= 0.0f){
        if (b2DistanceSquared(cLocal, v2v) > radius*radius) return;
        m.pointCount = 1; m.type = GB_MANIFOLD_FACE_A;
        V2 n = cLocal - v2v; b2Normalize(n);
        m.localNormal = n; m.localPoint = v2v;
        m.pLocalPoint = v2(0,0); m.id0 = 0;
    } else {
        V2 faceCenter = 0.5f * (v1 + v2v);
        float sep = b2Dot(cLocal - faceCenter, polyA.normals[vertIndex1]);
        if (sep > radius) return;
        m.pointCount = 1; m.type = GB_MANIFOLD_FACE_A;
        m.localNormal = polyA.normals[vertIndex1];
        m.localPoint = faceCenter;
        m.pLocalPoint = v2(0,0); m.id0 = 0;
    }
}

// =========================== Edge-polygon narrow-phase ======================
// b2CollideEdgeAndPolygon (b2CollideEdge.cpp, b2EPCollider). An edge collides with a
// convex polygon accounting for edge adjacency: the edge can carry a preceding vertex
// (vertex0) and a following vertex (vertex3), which constrain the valid normal range so
// a polygon sliding across a chain of edges does not catch on an interior vertex. For a
// single-segment edge both flags are false and the routine reduces to the front/back
// face test. The port keeps the full adjacency logic so a chain shape reuses it.
//
// The manifold convention matches b2WorldManifold::Initialize: a face-A manifold stores
// the reference face in the edge frame (localNormal/localPoint) and each clip point in
// the polygon (B) frame; a face-B manifold stores the reference face in the polygon
// frame and each clip point in the edge (A) frame.

// b2EdgeShape (Collision/Shapes/b2EdgeShape.h). vertex1/vertex2 are the segment;
// vertex0/vertex3 are the optional adjacent vertices that set the normal range.
struct GBEdgeShape {
    V2   vertex0, vertex1, vertex2, vertex3;
    bool hasVertex0, hasVertex3;
};

// b2EPAxis. The best separating axis found by the collider.
struct GBEPAxis { int type; int index; float separation; };
#define GB_EP_AXIS_UNKNOWN 0
#define GB_EP_AXIS_EDGE_A  1
#define GB_EP_AXIS_EDGE_B  2

// b2TempPolygon. Polygon B expressed in the edge frame (frame A).
struct GBTempPolygon { V2 vertices[GB_MAX_POLYGON_VERTICES]; V2 normals[GB_MAX_POLYGON_VERTICES]; int count; };

// b2ReferenceFace. The face the incident edge is clipped against.
struct GBReferenceFace {
    int i1, i2;
    V2  v1, v2;
    V2  normal;
    V2  sideNormal1; float sideOffset1;
    V2  sideNormal2; float sideOffset2;
};

// b2MulT(b2Transform, b2Transform): C = A^-1 * B, used to express B in A's frame.
GB_HD inline Xf gbMulTxf(Xf A, Xf B){
    Xf C;
    // C.q = b2MulT(A.q, B.q)
    C.q.s = A.q.c * B.q.s - A.q.s * B.q.c;
    C.q.c = A.q.c * B.q.c + A.q.s * B.q.s;
    // C.p = b2MulT(A.q, B.p - A.p)
    float px = B.p.x - A.p.x, py = B.p.y - A.p.y;
    C.p = v2(A.q.c * px + A.q.s * py, -A.q.s * px + A.q.c * py);
    return C;
}

// The collider state (b2EPCollider members) carried through the phase functions.
struct GBEPCollider {
    GBTempPolygon polygonB;
    Xf   xf;
    V2   centroidB;
    V2   v0, v1, v2, v3;
    V2   normal0, normal1, normal2;
    V2   normal;
    V2   lowerLimit, upperLimit;
    float radius;
    bool front;
};

// b2EPCollider::ComputeEdgeSeparation.
GB_HD inline GBEPAxis gbEPComputeEdgeSeparation(const GBEPCollider& c){
    GBEPAxis axis;
    axis.type = GB_EP_AXIS_EDGE_A;
    axis.index = c.front ? 0 : 1;
    axis.separation = GB_MAXFLOAT;
    for (int i = 0; i < c.polygonB.count; ++i){
        float s = b2Dot(c.normal, c.polygonB.vertices[i] - c.v1);
        if (s < axis.separation) axis.separation = s;
    }
    return axis;
}

// b2EPCollider::ComputePolygonSeparation.
GB_HD inline GBEPAxis gbEPComputePolygonSeparation(const GBEPCollider& c){
    GBEPAxis axis;
    axis.type = GB_EP_AXIS_UNKNOWN;
    axis.index = -1;
    axis.separation = -GB_MAXFLOAT;
    V2 perp = v2(-c.normal.y, c.normal.x);
    for (int i = 0; i < c.polygonB.count; ++i){
        V2 n = -c.polygonB.normals[i];
        float s1 = b2Dot(n, c.polygonB.vertices[i] - c.v1);
        float s2 = b2Dot(n, c.polygonB.vertices[i] - c.v2);
        float s = b2MinF(s1, s2);
        if (s > c.radius){
            axis.type = GB_EP_AXIS_EDGE_B; axis.index = i; axis.separation = s;
            return axis;
        }
        if (b2Dot(n, perp) >= 0.0f){
            if (b2Dot(n - c.upperLimit, c.normal) < -GB_ANGULAR_SLOP) continue;
        } else {
            if (b2Dot(n - c.lowerLimit, c.normal) < -GB_ANGULAR_SLOP) continue;
        }
        if (s > axis.separation){
            axis.type = GB_EP_AXIS_EDGE_B; axis.index = i; axis.separation = s;
        }
    }
    return axis;
}

// b2EPCollider::Collide. Classifies the edge, sets the normal range, finds the best
// axis, clips the incident edge, and writes the manifold.
GB_HD inline void gbCollideEdgeAndPolygon(GBManifold& m, const GBEdgeShape& edgeA, Xf xfA,
                                          const GBPolygon& polygonB, Xf xfB){
    GBEPCollider c;
    c.xf = gbMulTxf(xfA, xfB);
    c.centroidB = b2MulTV(c.xf, polygonB.centroid);
    c.v0 = edgeA.vertex0; c.v1 = edgeA.vertex1; c.v2 = edgeA.vertex2; c.v3 = edgeA.vertex3;
    bool hasVertex0 = edgeA.hasVertex0;
    bool hasVertex3 = edgeA.hasVertex3;

    V2 edge1 = c.v2 - c.v1; b2Normalize(edge1);
    c.normal1 = v2(edge1.y, -edge1.x);
    float offset1 = b2Dot(c.normal1, c.centroidB - c.v1);
    float offset0 = 0.0f, offset2 = 0.0f;
    bool convex1 = false, convex2 = false;

    if (hasVertex0){
        V2 edge0 = c.v1 - c.v0; b2Normalize(edge0);
        c.normal0 = v2(edge0.y, -edge0.x);
        convex1 = b2Cross(edge0, edge1) >= 0.0f;
        offset0 = b2Dot(c.normal0, c.centroidB - c.v0);
    }
    if (hasVertex3){
        V2 edge2 = c.v3 - c.v2; b2Normalize(edge2);
        c.normal2 = v2(edge2.y, -edge2.x);
        convex2 = b2Cross(edge1, edge2) > 0.0f;
        offset2 = b2Dot(c.normal2, c.centroidB - c.v2);
    }

    if (hasVertex0 && hasVertex3){
        if (convex1 && convex2){
            c.front = offset0 >= 0.0f || offset1 >= 0.0f || offset2 >= 0.0f;
            if (c.front){ c.normal = c.normal1; c.lowerLimit = c.normal0; c.upperLimit = c.normal2; }
            else        { c.normal = -c.normal1; c.lowerLimit = -c.normal1; c.upperLimit = -c.normal1; }
        } else if (convex1){
            c.front = offset0 >= 0.0f || (offset1 >= 0.0f && offset2 >= 0.0f);
            if (c.front){ c.normal = c.normal1; c.lowerLimit = c.normal0; c.upperLimit = c.normal1; }
            else        { c.normal = -c.normal1; c.lowerLimit = -c.normal2; c.upperLimit = -c.normal1; }
        } else if (convex2){
            c.front = offset2 >= 0.0f || (offset0 >= 0.0f && offset1 >= 0.0f);
            if (c.front){ c.normal = c.normal1; c.lowerLimit = c.normal1; c.upperLimit = c.normal2; }
            else        { c.normal = -c.normal1; c.lowerLimit = -c.normal1; c.upperLimit = -c.normal0; }
        } else {
            c.front = offset0 >= 0.0f && offset1 >= 0.0f && offset2 >= 0.0f;
            if (c.front){ c.normal = c.normal1; c.lowerLimit = c.normal1; c.upperLimit = c.normal1; }
            else        { c.normal = -c.normal1; c.lowerLimit = -c.normal2; c.upperLimit = -c.normal0; }
        }
    } else if (hasVertex0){
        if (convex1){
            c.front = offset0 >= 0.0f || offset1 >= 0.0f;
            if (c.front){ c.normal = c.normal1; c.lowerLimit = c.normal0; c.upperLimit = -c.normal1; }
            else        { c.normal = -c.normal1; c.lowerLimit = c.normal1; c.upperLimit = -c.normal1; }
        } else {
            c.front = offset0 >= 0.0f && offset1 >= 0.0f;
            if (c.front){ c.normal = c.normal1; c.lowerLimit = c.normal1; c.upperLimit = -c.normal1; }
            else        { c.normal = -c.normal1; c.lowerLimit = c.normal1; c.upperLimit = -c.normal0; }
        }
    } else if (hasVertex3){
        if (convex2){
            c.front = offset1 >= 0.0f || offset2 >= 0.0f;
            if (c.front){ c.normal = c.normal1; c.lowerLimit = -c.normal1; c.upperLimit = c.normal2; }
            else        { c.normal = -c.normal1; c.lowerLimit = -c.normal1; c.upperLimit = c.normal1; }
        } else {
            c.front = offset1 >= 0.0f && offset2 >= 0.0f;
            if (c.front){ c.normal = c.normal1; c.lowerLimit = -c.normal1; c.upperLimit = c.normal1; }
            else        { c.normal = -c.normal1; c.lowerLimit = -c.normal2; c.upperLimit = c.normal1; }
        }
    } else {
        c.front = offset1 >= 0.0f;
        if (c.front){ c.normal = c.normal1; c.lowerLimit = -c.normal1; c.upperLimit = -c.normal1; }
        else        { c.normal = -c.normal1; c.lowerLimit = c.normal1; c.upperLimit = c.normal1; }
    }

    // Get polygonB in frameA.
    c.polygonB.count = polygonB.count;
    for (int i = 0; i < polygonB.count; ++i){
        c.polygonB.vertices[i] = b2MulTV(c.xf, polygonB.vertices[i]);
        c.polygonB.normals[i]  = b2MulRV(c.xf.q, polygonB.normals[i]);
    }
    c.radius = 2.0f * GB_POLYGON_RADIUS;
    m.pointCount = 0;

    GBEPAxis edgeAxis = gbEPComputeEdgeSeparation(c);
    if (edgeAxis.type == GB_EP_AXIS_UNKNOWN) return;
    if (edgeAxis.separation > c.radius) return;
    GBEPAxis polygonAxis = gbEPComputePolygonSeparation(c);
    if (polygonAxis.type != GB_EP_AXIS_UNKNOWN && polygonAxis.separation > c.radius) return;

    const float k_relativeTol = 0.98f;
    const float k_absoluteTol = 0.001f;
    GBEPAxis primaryAxis;
    if (polygonAxis.type == GB_EP_AXIS_UNKNOWN) primaryAxis = edgeAxis;
    else if (polygonAxis.separation > k_relativeTol * edgeAxis.separation + k_absoluteTol) primaryAxis = polygonAxis;
    else primaryAxis = edgeAxis;

    GBClipVertex ie[2];
    GBReferenceFace rf;
    if (primaryAxis.type == GB_EP_AXIS_EDGE_A){
        m.type = GB_MANIFOLD_FACE_A;
        int bestIndex = 0;
        float bestValue = b2Dot(c.normal, c.polygonB.normals[0]);
        for (int i = 1; i < c.polygonB.count; ++i){
            float value = b2Dot(c.normal, c.polygonB.normals[i]);
            if (value < bestValue){ bestValue = value; bestIndex = i; }
        }
        int i1 = bestIndex;
        int i2 = i1 + 1 < c.polygonB.count ? i1 + 1 : 0;
        ie[0].v = c.polygonB.vertices[i1];
        ie[0].indexA = 0; ie[0].indexB = i1; ie[0].typeA = GB_FEATURE_FACE; ie[0].typeB = GB_FEATURE_VERTEX;
        ie[1].v = c.polygonB.vertices[i2];
        ie[1].indexA = 0; ie[1].indexB = i2; ie[1].typeA = GB_FEATURE_FACE; ie[1].typeB = GB_FEATURE_VERTEX;
        if (c.front){ rf.i1 = 0; rf.i2 = 1; rf.v1 = c.v1; rf.v2 = c.v2; rf.normal = c.normal1; }
        else        { rf.i1 = 1; rf.i2 = 0; rf.v1 = c.v2; rf.v2 = c.v1; rf.normal = -c.normal1; }
    } else {
        m.type = GB_MANIFOLD_FACE_B;
        ie[0].v = c.v1;
        ie[0].indexA = 0; ie[0].indexB = primaryAxis.index; ie[0].typeA = GB_FEATURE_VERTEX; ie[0].typeB = GB_FEATURE_FACE;
        ie[1].v = c.v2;
        ie[1].indexA = 0; ie[1].indexB = primaryAxis.index; ie[1].typeA = GB_FEATURE_VERTEX; ie[1].typeB = GB_FEATURE_FACE;
        rf.i1 = primaryAxis.index;
        rf.i2 = rf.i1 + 1 < c.polygonB.count ? rf.i1 + 1 : 0;
        rf.v1 = c.polygonB.vertices[rf.i1];
        rf.v2 = c.polygonB.vertices[rf.i2];
        rf.normal = c.polygonB.normals[rf.i1];
    }
    rf.sideNormal1 = v2(rf.normal.y, -rf.normal.x);
    rf.sideNormal2 = -rf.sideNormal1;
    rf.sideOffset1 = b2Dot(rf.sideNormal1, rf.v1);
    rf.sideOffset2 = b2Dot(rf.sideNormal2, rf.v2);

    GBClipVertex clipPoints1[2];
    GBClipVertex clipPoints2[2];
    int np;
    np = gbClipSegmentToLine(clipPoints1, ie, rf.sideNormal1, rf.sideOffset1, rf.i1);
    if (np < GB_MAX_MANIFOLD_POINTS) return;
    np = gbClipSegmentToLine(clipPoints2, clipPoints1, rf.sideNormal2, rf.sideOffset2, rf.i2);
    if (np < GB_MAX_MANIFOLD_POINTS) return;

    if (primaryAxis.type == GB_EP_AXIS_EDGE_A){
        m.localNormal = rf.normal;
        m.localPoint  = rf.v1;
    } else {
        m.localNormal = polygonB.normals[rf.i1];
        m.localPoint  = polygonB.vertices[rf.i1];
    }
    int pointCount = 0;
    for (int i = 0; i < GB_MAX_MANIFOLD_POINTS; ++i){
        float separation = b2Dot(rf.normal, clipPoints2[i].v - rf.v1);
        if (separation <= c.radius){
            V2 localPt; unsigned int key;
            if (primaryAxis.type == GB_EP_AXIS_EDGE_A){
                localPt = b2MulTinvV(c.xf, clipPoints2[i].v);
                key = gbFeatureKey(clipPoints2[i].indexA, clipPoints2[i].indexB,
                                   clipPoints2[i].typeA, clipPoints2[i].typeB);
            } else {
                localPt = clipPoints2[i].v;
                key = gbFeatureKey(clipPoints2[i].indexB, clipPoints2[i].indexA,
                                   clipPoints2[i].typeB, clipPoints2[i].typeA);
            }
            if (pointCount == 0){ m.pLocalPoint = localPt; m.id0 = key; }
            else                { m.pLocalPoint2 = localPt; m.id1 = key; }
            ++pointCount;
        }
    }
    m.pointCount = pointCount;
}

// =========================== Per-contact helpers ============================
// Body transform from the cached xf fields, read via accessors.
GB_HD inline Xf gbBodyXf(GBWorld& w, int i){
    Xf t;
    t.p = v2(BODY(w, xfPx, i), BODY(w, xfPy, i));
    t.q.s = BODY(w, xfQs, i);
    t.q.c = BODY(w, xfQc, i);
    return t;
}

#ifdef GB_ENABLE_POLYGONS
// Read a body's polygon shape from the per-body polygon storage into a GBPolygon,
// all through the accessor contract. Only called when gbBodyShape(w,s) is
// GB_SHAPE_POLYGON.
GB_HD inline void gbLoadPolygon(GBWorld& w, int s, GBPolygon& p){
    p.count  = BODY(w, polyCount, s);
    p.radius = BODY(w, polyRadius, s);
    p.centroid = v2(BODY(w, polyCentroidX, s), BODY(w, polyCentroidY, s));
    for (int i = 0; i < p.count; ++i){
        int vs = gbPolyVertSlot(s, i);
        p.vertices[i] = v2(BODY(w, polyVx, vs), BODY(w, polyVy, vs));
        p.normals[i]  = v2(BODY(w, polyNx, vs), BODY(w, polyNy, vs));
    }
}
#endif

// gbContactUpdate. b2Contact::Update (b2Contact.cpp:161), 1-point path, on the
// accessor contract. Runs the narrow-phase for contact slot ci, sets enabled, caches
// the manifold, carries warm-start impulses, and flips cTouching.
//
// Contact key convention: cEdge < 0 is circle-circle (fixtureA = cBodyA's circle,
// fixtureB = cBodyB's circle); cEdge >= 0 is edge-circle (fixtureA = ground edge
// `cEdge`, fixtureB = cBodyB's circle). Radii read via the general accessor
// gbCircleRadius (BODY(w,radius,s)).
//
// CONTACT LISTENER HOOK. This is the generic b2ContactListener mechanism. On a
// touching transition gbContactUpdate calls gbOnTouchBegin (begin-contact) or
// gbOnTouchEnd (end-contact). The default definitions are no-ops; an application
// overrides them by defining GB_CONTACT_LISTENER_HOOKS and supplying its own
// gbOnTouchBegin / gbOnTouchEnd before this header is included. The hook carries
// no game meaning in the core and adds zero float ops to the narrow-phase, so the
// 0-ULP manifold and touching result hold.
#ifndef GB_CONTACT_LISTENER_HOOKS
GB_HD inline void gbOnTouchBegin(GBWorld&, int, int){}
GB_HD inline void gbOnTouchEnd(GBWorld&, int, int){}
#endif

// Run the narrow-phase for contact slot ci, choosing the manifold function from the
// fixture shapes. The circle-only build compiles the circle and edge cases alone, so
// the manifold it produces is byte-identical to the circle-only path.
GB_HD inline void gbNarrowPhase(GBWorld& w, int bodyA, int bodyB, int edge, GBManifold& m){
    m.pointCount = 0;
#ifdef GB_ENABLE_POLYGONS
    int shapeB = gbBodyShape(w, bodyB);
    if (edge < 0){
        int shapeA = gbBodyShape(w, bodyA);
        if (shapeA == GB_SHAPE_POLYGON && shapeB == GB_SHAPE_POLYGON){
            GBPolygon pA, pB; gbLoadPolygon(w, bodyA, pA); gbLoadPolygon(w, bodyB, pB);
            gbCollidePolygons(m, pA, gbBodyXf(w, bodyA), pB, gbBodyXf(w, bodyB));
        } else if (shapeA == GB_SHAPE_POLYGON){
            // polygon (A) vs circle (B): b2CollidePolygonAndCircle, polygon = fixtureA
            GBPolygon pA; gbLoadPolygon(w, bodyA, pA);
            gbCollidePolygonAndCircle(m, pA, gbBodyXf(w, bodyA),
                                      gbCircleRadius(w, bodyB), gbBodyXf(w, bodyB));
        } else if (shapeB == GB_SHAPE_POLYGON){
            // circle (A) vs polygon (B): polygon is fixtureA in Box2D's registry, so
            // the contact is (polygon=bodyB, circle=bodyA) with the key swapped by the
            // contact-creation order. Here bodyA is the circle, so collide with the
            // polygon as the reference and the manifold normal already points A to B.
            GBPolygon pB; gbLoadPolygon(w, bodyB, pB);
            gbCollidePolygonAndCircle(m, pB, gbBodyXf(w, bodyB),
                                      gbCircleRadius(w, bodyA), gbBodyXf(w, bodyA));
            if (m.pointCount > 0) m.type = GB_MANIFOLD_FACE_B;   // reference face on B
        } else {
            float rA = gbCircleRadius(w, bodyA), rB = gbCircleRadius(w, bodyB);
            gbCollideCircles(m, rA, gbBodyXf(w, bodyA), rB, gbBodyXf(w, bodyB));
        }
    } else {
        V2 A = v2(EDGE(w, edgeAx, edge), EDGE(w, edgeAy, edge));
        V2 B = v2(EDGE(w, edgeBx, edge), EDGE(w, edgeBy, edge));
        if (shapeB == GB_SHAPE_POLYGON){
            // edge (A) vs polygon (B): the dedicated b2CollideEdgeAndPolygon. A flat
            // ground edge is a single segment (no adjacency); a chain edge carries its
            // neighbors so the adjacency path activates.
            GBEdgeShape eA;
            eA.vertex1 = A; eA.vertex2 = B;
#ifdef GB_ENABLE_CHAIN
            eA.vertex0 = v2(EDGE(w, edgeV0x, edge), EDGE(w, edgeV0y, edge));
            eA.vertex3 = v2(EDGE(w, edgeV3x, edge), EDGE(w, edgeV3y, edge));
            eA.hasVertex0 = EDGE(w, edgeHasV0, edge) != 0;
            eA.hasVertex3 = EDGE(w, edgeHasV3, edge) != 0;
#else
            eA.vertex0 = v2(0,0); eA.vertex3 = v2(0,0);
            eA.hasVertex0 = false; eA.hasVertex3 = false;
#endif
            GBPolygon pB; gbLoadPolygon(w, bodyB, pB);
            gbCollideEdgeAndPolygon(m, eA, gbBodyXf(w, bodyA), pB, gbBodyXf(w, bodyB));
        } else {
            gbCollideEdgeAndCircle(m, A, B, GB_POLYGON_RADIUS, gbCircleRadius(w, bodyB),
                                   gbBodyXf(w, bodyA), gbBodyXf(w, bodyB));
        }
    }
#else
    if (edge < 0){
        // circle-circle: fixtureA = bodyA's circle, fixtureB = bodyB's circle
        float rA = gbCircleRadius(w, bodyA), rB = gbCircleRadius(w, bodyB);
        gbCollideCircles(m, rA, gbBodyXf(w, bodyA), rB, gbBodyXf(w, bodyB));
    } else {
        // edge-circle: fixtureA = ground edge, fixtureB = body's circle
        V2 A = v2(EDGE(w, edgeAx, edge), EDGE(w, edgeAy, edge));
        V2 B = v2(EDGE(w, edgeBx, edge), EDGE(w, edgeBy, edge));
        float circR = gbCircleRadius(w, bodyB);
        gbCollideEdgeAndCircle(m, A, B, GB_POLYGON_RADIUS, circR,
                               gbBodyXf(w, bodyA), gbBodyXf(w, bodyB));
    }
#endif
}

GB_HD inline void gbContactUpdate(GBWorld& w, int ci){
    CONT(w, cEnabled, ci) = 1;   // b2Contact::Update: m_flags |= e_enabledFlag
    bool wasTouching = CONT(w, cTouching, ci) != 0;
    int bodyA = CONT(w, cBodyA, ci), bodyB = CONT(w, cBodyB, ci), edge = CONT(w, cEdge, ci);
    GBManifold m; m.pointCount = 0;
    gbNarrowPhase(w, bodyA, bodyB, edge, m);
    bool touching = m.pointCount > 0;
    // warm-start id carry: a surviving touching contact keeps its impulse; a
    // non-touching one resets to 0. (Circle and edge manifolds have id 0; polygon
    // manifolds carry feature ids, which a later refinement can match per point.)
    if (touching){
        CONT(w, cManifoldType, ci) = m.type;
        CONT(w, cPointCount, ci) = m.pointCount;
        CONT(w, cLocalNormalX, ci) = m.localNormal.x; CONT(w, cLocalNormalY, ci) = m.localNormal.y;
        CONT(w, cLocalPointX,  ci) = m.localPoint.x;  CONT(w, cLocalPointY,  ci) = m.localPoint.y;
        CONT(w, cPointLocalX,  ci) = m.pLocalPoint.x; CONT(w, cPointLocalY,  ci) = m.pLocalPoint.y;
        if (m.pointCount > 1){
            CONT(w, cPointLocal2X, ci) = m.pLocalPoint2.x; CONT(w, cPointLocal2Y, ci) = m.pLocalPoint2.y;
        }
        // impulse carries from previous substep (cNormalImpulse/cTangentImpulse
        // are left intact since id.key matches). On first-touch they are 0.
    } else {
        CONT(w, cNormalImpulse, ci) = 0.0f; CONT(w, cTangentImpulse, ci) = 0.0f;
    }
    CONT(w, cTouching, ci) = touching ? 1 : 0;
    // b2Contact::Update: fire begin-contact / end-contact on touching transitions
    if (!wasTouching && touching)  gbOnTouchBegin(w, bodyA, bodyB);
    if ( wasTouching && !touching) gbOnTouchEnd(w, bodyA, bodyB);
}
