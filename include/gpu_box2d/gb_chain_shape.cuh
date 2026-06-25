// gb_chain_shape.cuh. b2ChainShape (Box2D 2.3.0), written against the gb_* type
// universe. A chain shape is a sequence of connected edge segments, the standard static
// world boundary: a ground contour, a level outline, or a closed loop. It collides
// through its child edges, and each child edge carries the adjacent vertices so a body
// sliding across a vertex does not catch on the interior corner. That adjacency is
// exactly what gbCollideEdgeAndPolygon and gbCollideEdgeAndCircle already consume, so the
// chain is a thin generator over the validated edge collider.
//
// Line-faithful to Box2D 2.3.0:
//   Collision/Shapes/b2ChainShape.cpp  (CreateChain / CreateLoop / GetChildEdge)
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_collision.cuh"   // GBEdgeShape and the edge colliders

// b2ChainShape capacity. A chain of N vertices has N-1 child edges (open) or N child
// edges (loop, where the contour closes back to vertex 0).
#ifndef GB_MAX_CHAIN_VERTICES
#define GB_MAX_CHAIN_VERTICES 16
#endif

// A chain shape. vertices[0..count-1] are the contour points. prevVertex / nextVertex
// are the ghost vertices that give the first and last child edges their outer adjacency
// (set by CreateChain when the contour connects to neighboring shapes). For a loop, the
// chain stores count+1 vertices with vertices[count] == vertices[0] and the ghosts wrap
// around, which CreateLoop sets up.
struct GBChainShape {
    V2   vertices[GB_MAX_CHAIN_VERTICES + 1];
    int  count;            // number of vertices in `vertices`
    V2   prevVertex, nextVertex;
    bool hasPrevVertex, hasNextVertex;
    float radius;          // b2_polygonRadius
};

// b2ChainShape::CreateChain(vertices, count). An open chain through the given points.
// The end edges have no outer adjacency unless prev/next ghosts are set afterward.
GB_HD inline void gbChainCreateChain(GBChainShape& c, const V2* vertices, int count){
    int n = count <= GB_MAX_CHAIN_VERTICES ? count : GB_MAX_CHAIN_VERTICES;
    c.count = n;
    for (int i = 0; i < n; ++i) c.vertices[i] = vertices[i];
    c.prevVertex = v2(0.0f, 0.0f);
    c.nextVertex = v2(0.0f, 0.0f);
    c.hasPrevVertex = false;
    c.hasNextVertex = false;
    c.radius = GB_POLYGON_RADIUS;
}

// b2ChainShape::CreateLoop(vertices, count). A closed loop through the given points. The
// contour stores count+1 vertices (the last repeats the first), and the ghosts wrap so
// every child edge, including the closing one, has adjacency.
GB_HD inline void gbChainCreateLoop(GBChainShape& c, const V2* vertices, int count){
    int n = count <= GB_MAX_CHAIN_VERTICES ? count : GB_MAX_CHAIN_VERTICES;
    c.count = n + 1;
    for (int i = 0; i < n; ++i) c.vertices[i] = vertices[i];
    c.vertices[n] = vertices[0];
    c.prevVertex = vertices[n - 1];
    c.nextVertex = vertices[1];
    c.hasPrevVertex = true;
    c.hasNextVertex = true;
    c.radius = GB_POLYGON_RADIUS;
}

// b2ChainShape::GetChildEdge(edge, index). Build the child edge at `index`, wiring its
// vertex0 / vertex3 from the neighbors (or the ghosts at the ends). The result feeds the
// edge colliders directly.
GB_HD inline void gbChainGetChildEdge(const GBChainShape& c, GBEdgeShape& edge, int index){
    edge.vertex1 = c.vertices[index];
    edge.vertex2 = c.vertices[index + 1];
    if (index > 0){
        edge.vertex0 = c.vertices[index - 1];
        edge.hasVertex0 = true;
    } else {
        edge.vertex0 = c.prevVertex;
        edge.hasVertex0 = c.hasPrevVertex;
    }
    if (index < c.count - 2){
        edge.vertex3 = c.vertices[index + 2];
        edge.hasVertex3 = true;
    } else {
        edge.vertex3 = c.nextVertex;
        edge.hasVertex3 = c.hasNextVertex;
    }
}

// Number of child edges in a chain (count - 1).
GB_HD inline int gbChainChildCount(const GBChainShape& c){ return c.count - 1; }

#ifdef GB_ENABLE_CHAIN
// Load a chain into a world's static edge fixtures, one child edge per edge slot, with
// the adjacency carried so the edge-polygon collider sees the chain corners. The chain
// fills up to GB_N_EDGES child edges (the per-world edge capacity); a longer chain is
// truncated to that many edges. Unused edge slots collapse to a point so they create no
// spurious contacts. Returns the number of child edges written. Writes through the
// accessor contract, so it works on both memory backends.
GB_HD inline int gbWorldSetChain(GBWorld& w, const GBChainShape& c){
    int n = gbChainChildCount(c);
    if (n > GB_N_EDGES) n = GB_N_EDGES;
    for (int e = 0; e < n; ++e){
        GBEdgeShape ce; gbChainGetChildEdge(c, ce, e);
        EDGE(w, edgeAx, e) = ce.vertex1.x; EDGE(w, edgeAy, e) = ce.vertex1.y;
        EDGE(w, edgeBx, e) = ce.vertex2.x; EDGE(w, edgeBy, e) = ce.vertex2.y;
        EDGE(w, edgeV0x, e) = ce.vertex0.x; EDGE(w, edgeV0y, e) = ce.vertex0.y;
        EDGE(w, edgeV3x, e) = ce.vertex3.x; EDGE(w, edgeV3y, e) = ce.vertex3.y;
        EDGE(w, edgeHasV0, e) = ce.hasVertex0 ? 1 : 0;
        EDGE(w, edgeHasV3, e) = ce.hasVertex3 ? 1 : 0;
    }
    // collapse the unused edge slots to a point at the origin
    for (int e = n; e < GB_N_EDGES; ++e){
        EDGE(w, edgeAx, e) = 0.0f; EDGE(w, edgeAy, e) = 0.0f;
        EDGE(w, edgeBx, e) = 0.0f; EDGE(w, edgeBy, e) = 0.0f;
        EDGE(w, edgeV0x, e) = 0.0f; EDGE(w, edgeV0y, e) = 0.0f;
        EDGE(w, edgeV3x, e) = 0.0f; EDGE(w, edgeV3y, e) = 0.0f;
        EDGE(w, edgeHasV0, e) = 0; EDGE(w, edgeHasV3, e) = 0;
    }
    return n;
}
#endif
