// gb_polygon.cuh. b2PolygonShape (Box2D 2.3.0), written against the gb_* type
// universe (V2 / Rot / Xf and ops from gb_math.cuh). A convex polygon stores its
// vertices, outward edge normals, centroid, count, and skin radius. This module
// covers shape construction, mass, and the tight AABB. The polygon narrow-phase
// lives in gb_collision.cuh and the two-point block solve in gb_contact_solver.cuh.
//
// Line-faithful to Box2D 2.3.0:
//   Collision/Shapes/b2PolygonShape.cpp  (SetAsBox / Set / ComputeMass / ComputeAABB)
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true (mirrors
// the CPU's -ffp-contract=off -mfpmath=sse). Changing these breaks bit-identicality.
#pragma once
#include "gb_settings.cuh"   // GB_MAX_POLYGON_VERTICES (b2_maxPolygonVertices)
#include "gb_math.cuh"

// A convex polygon shape. m_radius is the polytope skin (b2_polygonRadius for a
// shape built the standard way). Vertices wind counter-clockwise; normal i is the
// outward normal of the edge from vertex i to vertex i+1.
struct GBPolygon {
    V2    vertices[GB_MAX_POLYGON_VERTICES];
    V2    normals[GB_MAX_POLYGON_VERTICES];
    V2    centroid;
    int   count;
    float radius;
};

// Mass, center of mass, and rotational inertia about the body origin.
struct GBMassData { float mass; V2 center; float I; };

// b2PolygonShape::SetAsBox(hx, hy). An axis-aligned box of half-width hx and
// half-height hy centered at the local origin.
GB_HD inline void gbPolygonSetAsBox(GBPolygon& p, float hx, float hy){
    p.count = 4;
    p.vertices[0] = v2(-hx, -hy);
    p.vertices[1] = v2( hx, -hy);
    p.vertices[2] = v2( hx,  hy);
    p.vertices[3] = v2(-hx,  hy);
    p.normals[0] = v2(0.0f, -1.0f);
    p.normals[1] = v2(1.0f, 0.0f);
    p.normals[2] = v2(0.0f, 1.0f);
    p.normals[3] = v2(-1.0f, 0.0f);
    p.centroid = v2(0.0f, 0.0f);
    p.radius = GB_POLYGON_RADIUS;
}

// b2PolygonShape ComputeCentroid (file-local static in Box2D). Area-weighted
// centroid over a fan from the origin reference point.
GB_HD inline V2 gbPolygonComputeCentroid(const V2* vs, int count){
    V2 c = v2(0.0f, 0.0f);
    float area = 0.0f;
    V2 pRef = v2(0.0f, 0.0f);
    const float inv3 = 1.0f / 3.0f;
    for (int i = 0; i < count; ++i){
        V2 p1 = pRef;
        V2 p2 = vs[i];
        V2 p3 = i + 1 < count ? vs[i+1] : vs[0];
        V2 e1 = p2 - p1;
        V2 e2 = p3 - p1;
        float D = b2Cross(e1, e2);
        float triangleArea = 0.5f * D;
        area += triangleArea;
        c = c + (triangleArea * inv3) * (p1 + p2 + p3);
    }
    c = (1.0f / area) * c;
    return c;
}

// b2PolygonShape::Set(vertices, count). Builds the convex hull with the gift-wrap
// algorithm, derives outward normals, and computes the centroid. Reproduces the
// hull vertex order and the tie-breaks of Box2D 2.3.0 so downstream collision uses
// the same vertex and normal indices.
GB_HD inline void gbPolygonSet(GBPolygon& p, const V2* vertices, int count){
    if (count < 3){ gbPolygonSetAsBox(p, 1.0f, 1.0f); return; }
    int n = count < GB_MAX_POLYGON_VERTICES ? count : GB_MAX_POLYGON_VERTICES;

    V2 ps[GB_MAX_POLYGON_VERTICES];
    for (int i = 0; i < n; ++i) ps[i] = vertices[i];

    // Find the right-most point on the hull (ties broken by lower y).
    int i0 = 0;
    float x0 = ps[0].x;
    for (int i = 1; i < count; ++i){
        float x = ps[i].x;
        if (x > x0 || (x == x0 && ps[i].y < ps[i0].y)){ i0 = i; x0 = x; }
    }

    int hull[GB_MAX_POLYGON_VERTICES];
    int m = 0;
    int ih = i0;
    for (;;){
        hull[m] = ih;
        int ie = 0;
        for (int j = 1; j < n; ++j){
            if (ie == ih){ ie = j; continue; }
            V2 r = ps[ie] - ps[hull[m]];
            V2 vv = ps[j] - ps[hull[m]];
            float c = b2Cross(r, vv);
            if (c < 0.0f) ie = j;
            if (c == 0.0f && b2Dot(vv, vv) > b2Dot(r, r)) ie = j;
        }
        ++m;
        ih = ie;
        if (ie == i0) break;
    }

    p.count = m;
    for (int i = 0; i < m; ++i) p.vertices[i] = ps[hull[i]];
    for (int i = 0; i < m; ++i){
        int i1 = i;
        int i2 = i + 1 < m ? i + 1 : 0;
        V2 edge = p.vertices[i2] - p.vertices[i1];
        V2 nrm = b2CrossVS(edge, 1.0f);   // b2Cross(edge, 1.0f)
        b2Normalize(nrm);
        p.normals[i] = nrm;
    }
    p.centroid = gbPolygonComputeCentroid(p.vertices, m);
    p.radius = GB_POLYGON_RADIUS;
}

// b2PolygonShape::ComputeMass(density). Integrates over the polygon triangles to
// produce mass, center of mass, and inertia about the body origin.
GB_HD inline void gbPolygonComputeMass(const GBPolygon& p, GBMassData& md, float density){
    V2 center = v2(0.0f, 0.0f);
    float area = 0.0f;
    float I = 0.0f;
    V2 s = v2(0.0f, 0.0f);
    for (int i = 0; i < p.count; ++i) s = s + p.vertices[i];
    s = (1.0f / p.count) * s;
    const float k_inv3 = 1.0f / 3.0f;
    for (int i = 0; i < p.count; ++i){
        V2 e1 = p.vertices[i] - s;
        V2 e2 = i + 1 < p.count ? p.vertices[i+1] - s : p.vertices[0] - s;
        float D = b2Cross(e1, e2);
        float triangleArea = 0.5f * D;
        area += triangleArea;
        center = center + (triangleArea * k_inv3) * (e1 + e2);
        float ex1 = e1.x, ey1 = e1.y;
        float ex2 = e2.x, ey2 = e2.y;
        float intx2 = ex1*ex1 + ex2*ex1 + ex2*ex2;
        float inty2 = ey1*ey1 + ey2*ey1 + ey2*ey2;
        I += (0.25f * k_inv3 * D) * (intx2 + inty2);
    }
    md.mass = density * area;
    center = (1.0f / area) * center;
    md.center = center + s;
    md.I = density * I;
    md.I += md.mass * (b2Dot(md.center, md.center) - b2Dot(center, center));
}

// b2PolygonShape::ComputeAABB(xf). Tight world AABB over the rotated vertices,
// expanded by the skin radius. Output is the lower and upper corners.
GB_HD inline void gbPolygonComputeAABB(const GBPolygon& p, Xf xf, V2& lower, V2& upper){
    lower = b2MulTV(xf, p.vertices[0]);
    upper = lower;
    for (int i = 1; i < p.count; ++i){
        V2 v = b2MulTV(xf, p.vertices[i]);
        lower = v2(b2MinF(lower.x, v.x), b2MinF(lower.y, v.y));
        upper = v2(b2MaxF(upper.x, v.x), b2MaxF(upper.y, v.y));
    }
    V2 r = v2(p.radius, p.radius);
    lower = lower - r;
    upper = upper + r;
}
