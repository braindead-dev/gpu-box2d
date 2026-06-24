// gb_broadphase.cuh. The Box2D 2.3.0 b2DynamicTree and b2BroadPhase, written
// against the gb_pools.cuh accessor contract (BODY/CONT/EDGE/SCAL, GBWorld).
//
// STATUS: validated. The proxyId assignment sequence and the AddPair output order
// match Box2D 2.3.0 bit-for-bit (see test/gb_broadphase_test.cu, 0-ULP).
//
// The algorithm and every ordering decision match Box2D 2.3.0. The functions that
// read fixture geometry from a world use the accessor macros. DynTree and
// BroadPhase are self-contained structs that the caller manages per-world.
//
// Accessor usage:
//   BODY(w, sweepCx, s)  circle body position (x)
//   BODY(w, sweepCy, s)  circle body position (y)
//   BODY(w, tier, s)     tier index (for radius lookup)
//   EDGE(w, edgeAx, e) / edgeAy / edgeBx / edgeBy   ground edge endpoints
#pragma once
#include "gb_pools.cuh"   // GBWorld, accessor macros, GB_AABB_EXTENSION, etc.

// ---------------------------------------------------------------------------
// Sizing. Box2D starts the tree at capacity 16 and doubles. The backing array is
// fixed, and the 16/32/64 doubling schedule of m_nodeCapacity is emulated so the
// free-list rebuild order, and thus proxyId assignment, matches Box2D exactly.
// ---------------------------------------------------------------------------
#define GB_BP_NULL_NODE  (-1)
#define GB_BP_NODE_CAP   256     // >= worst-case: 2*MAX_FIXTURES + growth
#define GB_BP_MOVE_CAP   80      // <= MAX_FIXTURES moving proxies
#define GB_BP_PAIR_CAP   512     // <=67 fixtures => few new pairs per UpdatePairs

// ---------------------------------------------------------------------------
// AABB  (b2Collision.h)
// ---------------------------------------------------------------------------
struct GbAABB { V2 lo, hi; };

GB_HD inline float gbAabbPerimeter(const GbAABB& a){
    return 2.0f * ((a.hi.x - a.lo.x) + (a.hi.y - a.lo.y));
}
GB_HD inline GbAABB gbAabbCombine(const GbAABB& a, const GbAABB& b){
    GbAABB r;
    r.lo = v2(a.lo.x < b.lo.x ? a.lo.x : b.lo.x,
              a.lo.y < b.lo.y ? a.lo.y : b.lo.y);
    r.hi = v2(a.hi.x > b.hi.x ? a.hi.x : b.hi.x,
              a.hi.y > b.hi.y ? a.hi.y : b.hi.y);
    return r;
}
GB_HD inline bool gbAabbContains(const GbAABB& a, const GbAABB& b){
    return a.lo.x <= b.lo.x && a.lo.y <= b.lo.y
        && b.hi.x <= a.hi.x && b.hi.y <= a.hi.y;
}
GB_HD inline bool gbAabbOverlap(const GbAABB& a, const GbAABB& b){
    if (b.lo.x - a.hi.x > 0.0f || b.lo.y - a.hi.y > 0.0f) return false;
    if (a.lo.x - b.hi.x > 0.0f || a.lo.y - b.hi.y > 0.0f) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Tree node  (b2DynamicTree.h)
// union{parent,next} preserved as a single field: parent on the active path,
// next (free-list link) when the node is free (height == -1).
// ---------------------------------------------------------------------------
struct GbBPNode {
    GbAABB aabb;
    int    userData;   // fixture id (>=0) for leaves; -1 for internal/free
    int    parent;     // doubles as `next` on the free list
    int    child1;
    int    child2;
    int    height;     // leaf=0, free=-1
};

// ---------------------------------------------------------------------------
// DynTree  (b2DynamicTree, fixed arrays, growth schedule emulated)
// ---------------------------------------------------------------------------
struct GbDynTree {
    GbBPNode nodes[GB_BP_NODE_CAP];
    int root;
    int nodeCount;
    int nodeCapacity;     // logical capacity following the 16/32/64 doubling
    int freeList;
    int insertionCount;
};

GB_HD inline void gbTreeInit(GbDynTree& t){
    t.root = GB_BP_NULL_NODE;
    t.nodeCapacity = 16;
    t.nodeCount = 0;
    for (int i = 0; i < GB_BP_NODE_CAP; ++i){
        t.nodes[i].userData = -1;
        t.nodes[i].child1   = GB_BP_NULL_NODE;
        t.nodes[i].child2   = GB_BP_NULL_NODE;
        t.nodes[i].parent   = GB_BP_NULL_NODE;
        t.nodes[i].height   = -1;
        t.nodes[i].aabb.lo  = v2(0.f, 0.f);
        t.nodes[i].aabb.hi  = v2(0.f, 0.f);
    }
    // Link initial free list: nodes[0..capacity-2].next = i+1
    for (int i = 0; i < t.nodeCapacity - 1; ++i){
        t.nodes[i].parent = i + 1;
        t.nodes[i].height = -1;
    }
    t.nodes[t.nodeCapacity-1].parent = GB_BP_NULL_NODE;
    t.nodes[t.nodeCapacity-1].height = -1;
    t.freeList       = 0;
    t.insertionCount = 0;
}

GB_HD inline int gbAllocateNode(GbDynTree& t){
    if (t.freeList == GB_BP_NULL_NODE){
        // Double logical capacity, link new range onto free list.
        int oldCap = t.nodeCapacity;
        t.nodeCapacity *= 2;
        for (int i = oldCap; i < t.nodeCapacity - 1; ++i){
            t.nodes[i].parent = i + 1;
            t.nodes[i].height = -1;
        }
        t.nodes[t.nodeCapacity-1].parent = GB_BP_NULL_NODE;
        t.nodes[t.nodeCapacity-1].height = -1;
        t.freeList = oldCap;
    }
    int nodeId = t.freeList;
    t.freeList = t.nodes[nodeId].parent;
    t.nodes[nodeId].parent = GB_BP_NULL_NODE;
    t.nodes[nodeId].child1 = GB_BP_NULL_NODE;
    t.nodes[nodeId].child2 = GB_BP_NULL_NODE;
    t.nodes[nodeId].height = 0;
    t.nodes[nodeId].userData = -1;
    ++t.nodeCount;
    return nodeId;
}

GB_HD inline void gbFreeNode(GbDynTree& t, int nodeId){
    t.nodes[nodeId].parent = t.freeList;
    t.nodes[nodeId].height = -1;
    t.freeList = nodeId;
    --t.nodeCount;
}

GB_HD inline int gbBalance(GbDynTree& t, int iA){
    GbBPNode* A = &t.nodes[iA];
    if (A->child1 == GB_BP_NULL_NODE || A->height < 2) return iA;
    int iB = A->child1, iC = A->child2;
    GbBPNode* B = &t.nodes[iB];
    GbBPNode* C = &t.nodes[iC];
    int balance = C->height - B->height;

    if (balance > 1){   // Rotate C up
        int iF = C->child1, iG = C->child2;
        GbBPNode* F = &t.nodes[iF];
        GbBPNode* G = &t.nodes[iG];
        C->child1 = iA; C->parent = A->parent; A->parent = iC;
        if (C->parent != GB_BP_NULL_NODE){
            if (t.nodes[C->parent].child1 == iA) t.nodes[C->parent].child1 = iC;
            else                                  t.nodes[C->parent].child2 = iC;
        } else t.root = iC;
        if (F->height > G->height){
            C->child2 = iF; A->child2 = iG; G->parent = iA;
            A->aabb = gbAabbCombine(B->aabb, G->aabb);
            C->aabb = gbAabbCombine(A->aabb, F->aabb);
            A->height = 1 + (B->height > G->height ? B->height : G->height);
            C->height = 1 + (A->height > F->height ? A->height : F->height);
        } else {
            C->child2 = iG; A->child2 = iF; F->parent = iA;
            A->aabb = gbAabbCombine(B->aabb, F->aabb);
            C->aabb = gbAabbCombine(A->aabb, G->aabb);
            A->height = 1 + (B->height > F->height ? B->height : F->height);
            C->height = 1 + (A->height > G->height ? A->height : G->height);
        }
        return iC;
    }
    if (balance < -1){   // Rotate B up
        int iD = B->child1, iE = B->child2;
        GbBPNode* D = &t.nodes[iD];
        GbBPNode* E = &t.nodes[iE];
        B->child1 = iA; B->parent = A->parent; A->parent = iB;
        if (B->parent != GB_BP_NULL_NODE){
            if (t.nodes[B->parent].child1 == iA) t.nodes[B->parent].child1 = iB;
            else                                  t.nodes[B->parent].child2 = iB;
        } else t.root = iB;
        if (D->height > E->height){
            B->child2 = iD; A->child1 = iE; E->parent = iA;
            A->aabb = gbAabbCombine(C->aabb, E->aabb);
            B->aabb = gbAabbCombine(A->aabb, D->aabb);
            A->height = 1 + (C->height > E->height ? C->height : E->height);
            B->height = 1 + (A->height > D->height ? A->height : D->height);
        } else {
            B->child2 = iE; A->child1 = iD; D->parent = iA;
            A->aabb = gbAabbCombine(C->aabb, D->aabb);
            B->aabb = gbAabbCombine(A->aabb, E->aabb);
            A->height = 1 + (C->height > D->height ? C->height : D->height);
            B->height = 1 + (A->height > E->height ? A->height : E->height);
        }
        return iB;
    }
    return iA;
}

GB_HD inline void gbInsertLeaf(GbDynTree& t, int leaf){
    ++t.insertionCount;
    if (t.root == GB_BP_NULL_NODE){
        t.root = leaf;
        t.nodes[t.root].parent = GB_BP_NULL_NODE;
        return;
    }
    GbAABB leafAABB = t.nodes[leaf].aabb;
    int index = t.root;
    while (t.nodes[index].child1 != GB_BP_NULL_NODE){   // not a leaf
        int child1 = t.nodes[index].child1;
        int child2 = t.nodes[index].child2;
        float area = gbAabbPerimeter(t.nodes[index].aabb);
        GbAABB combinedAABB = gbAabbCombine(t.nodes[index].aabb, leafAABB);
        float combinedArea = gbAabbPerimeter(combinedAABB);
        float cost = 2.0f * combinedArea;
        float inheritanceCost = 2.0f * (combinedArea - area);
        float cost1, cost2;
        if (t.nodes[child1].child1 == GB_BP_NULL_NODE){   // child1 is a leaf
            GbAABB ab = gbAabbCombine(leafAABB, t.nodes[child1].aabb);
            cost1 = gbAabbPerimeter(ab) + inheritanceCost;
        } else {
            GbAABB ab = gbAabbCombine(leafAABB, t.nodes[child1].aabb);
            float oldArea = gbAabbPerimeter(t.nodes[child1].aabb);
            float newArea = gbAabbPerimeter(ab);
            cost1 = (newArea - oldArea) + inheritanceCost;
        }
        if (t.nodes[child2].child1 == GB_BP_NULL_NODE){   // child2 is a leaf
            GbAABB ab = gbAabbCombine(leafAABB, t.nodes[child2].aabb);
            cost2 = gbAabbPerimeter(ab) + inheritanceCost;
        } else {
            GbAABB ab = gbAabbCombine(leafAABB, t.nodes[child2].aabb);
            float oldArea = gbAabbPerimeter(t.nodes[child2].aabb);
            float newArea = gbAabbPerimeter(ab);
            cost2 = newArea - oldArea + inheritanceCost;
        }
        if (cost < cost1 && cost < cost2) break;
        if (cost1 < cost2) index = child1; else index = child2;
    }
    int sibling = index;
    int oldParent = t.nodes[sibling].parent;
    int newParent = gbAllocateNode(t);
    t.nodes[newParent].parent   = oldParent;
    t.nodes[newParent].userData = -1;
    t.nodes[newParent].aabb     = gbAabbCombine(leafAABB, t.nodes[sibling].aabb);
    t.nodes[newParent].height   = t.nodes[sibling].height + 1;
    if (oldParent != GB_BP_NULL_NODE){
        if (t.nodes[oldParent].child1 == sibling) t.nodes[oldParent].child1 = newParent;
        else                                       t.nodes[oldParent].child2 = newParent;
        t.nodes[newParent].child1 = sibling;
        t.nodes[newParent].child2 = leaf;
        t.nodes[sibling].parent   = newParent;
        t.nodes[leaf].parent      = newParent;
    } else {
        t.nodes[newParent].child1 = sibling;
        t.nodes[newParent].child2 = leaf;
        t.nodes[sibling].parent   = newParent;
        t.nodes[leaf].parent      = newParent;
        t.root = newParent;
    }
    index = t.nodes[leaf].parent;
    while (index != GB_BP_NULL_NODE){
        index = gbBalance(t, index);
        int c1 = t.nodes[index].child1;
        int c2 = t.nodes[index].child2;
        t.nodes[index].height = 1 + (t.nodes[c1].height > t.nodes[c2].height
                                      ? t.nodes[c1].height : t.nodes[c2].height);
        t.nodes[index].aabb   = gbAabbCombine(t.nodes[c1].aabb, t.nodes[c2].aabb);
        index = t.nodes[index].parent;
    }
}

GB_HD inline void gbRemoveLeaf(GbDynTree& t, int leaf){
    if (leaf == t.root){ t.root = GB_BP_NULL_NODE; return; }
    int parent      = t.nodes[leaf].parent;
    int grandParent = t.nodes[parent].parent;
    int sibling     = (t.nodes[parent].child1 == leaf)
                      ? t.nodes[parent].child2 : t.nodes[parent].child1;
    if (grandParent != GB_BP_NULL_NODE){
        if (t.nodes[grandParent].child1 == parent) t.nodes[grandParent].child1 = sibling;
        else                                        t.nodes[grandParent].child2 = sibling;
        t.nodes[sibling].parent = grandParent;
        gbFreeNode(t, parent);
        int index = grandParent;
        while (index != GB_BP_NULL_NODE){
            index = gbBalance(t, index);
            int c1 = t.nodes[index].child1;
            int c2 = t.nodes[index].child2;
            t.nodes[index].aabb   = gbAabbCombine(t.nodes[c1].aabb, t.nodes[c2].aabb);
            t.nodes[index].height = 1 + (t.nodes[c1].height > t.nodes[c2].height
                                          ? t.nodes[c1].height : t.nodes[c2].height);
            index = t.nodes[index].parent;
        }
    } else {
        t.root = sibling;
        t.nodes[sibling].parent = GB_BP_NULL_NODE;
        gbFreeNode(t, parent);
    }
}

// CreateProxy: inflate by GB_AABB_EXTENSION and insert.
GB_HD inline int gbCreateProxy(GbDynTree& t, const GbAABB& aabb, int userData){
    int proxyId = gbAllocateNode(t);
    t.nodes[proxyId].aabb.lo = v2(aabb.lo.x - GB_AABB_EXTENSION,
                                   aabb.lo.y - GB_AABB_EXTENSION);
    t.nodes[proxyId].aabb.hi = v2(aabb.hi.x + GB_AABB_EXTENSION,
                                   aabb.hi.y + GB_AABB_EXTENSION);
    t.nodes[proxyId].userData = userData;
    t.nodes[proxyId].height   = 0;
    gbInsertLeaf(t, proxyId);
    return proxyId;
}

GB_HD inline void gbDestroyProxy(GbDynTree& t, int proxyId){
    gbRemoveLeaf(t, proxyId);
    gbFreeNode(t, proxyId);
}

// MoveProxy: only re-insert if aabb is no longer contained in the fat node.
GB_HD inline bool gbMoveProxy(GbDynTree& t, int proxyId, const GbAABB& aabb, V2 displacement){
    if (gbAabbContains(t.nodes[proxyId].aabb, aabb)) return false;
    gbRemoveLeaf(t, proxyId);
    GbAABB b;
    b.lo = v2(aabb.lo.x - GB_AABB_EXTENSION, aabb.lo.y - GB_AABB_EXTENSION);
    b.hi = v2(aabb.hi.x + GB_AABB_EXTENSION, aabb.hi.y + GB_AABB_EXTENSION);
    float dx = GB_AABB_MULTIPLIER * displacement.x;
    float dy = GB_AABB_MULTIPLIER * displacement.y;
    if (dx < 0.0f) b.lo.x += dx; else b.hi.x += dx;
    if (dy < 0.0f) b.lo.y += dy; else b.hi.y += dy;
    t.nodes[proxyId].aabb = b;
    gbInsertLeaf(t, proxyId);
    return true;
}

// ---------------------------------------------------------------------------
// BroadPhase  (b2BroadPhase, fixed arrays)
// ---------------------------------------------------------------------------
struct GbBPPair { int proxyIdA, proxyIdB; };

struct GbBroadPhase {
    GbDynTree  tree;
    int        proxyCount;
    int        moveBuffer[GB_BP_MOVE_CAP];
    int        moveCount;
    GbBPPair   pairBuffer[GB_BP_PAIR_CAP];
    int        pairCount;
    int        queryProxyId;
    int        addA[GB_BP_PAIR_CAP];   // AddPair output (userData ids), in call order
    int        addB[GB_BP_PAIR_CAP];
    int        addCount;
};

GB_HD inline void gbBpInit(GbBroadPhase& bp){
    gbTreeInit(bp.tree);
    bp.proxyCount = 0;
    bp.moveCount  = 0;
    bp.pairCount  = 0;
    bp.addCount   = 0;
}

GB_HD inline void gbBpBufferMove(GbBroadPhase& bp, int proxyId){
    bp.moveBuffer[bp.moveCount++] = proxyId;
}
GB_HD inline void gbBpUnBufferMove(GbBroadPhase& bp, int proxyId){
    for (int i = 0; i < bp.moveCount; ++i)
        if (bp.moveBuffer[i] == proxyId) bp.moveBuffer[i] = GB_BP_NULL_NODE;
}

GB_HD inline int gbBpCreate(GbBroadPhase& bp, const GbAABB& aabb, int userData){
    int proxyId = gbCreateProxy(bp.tree, aabb, userData);
    ++bp.proxyCount;
    gbBpBufferMove(bp, proxyId);
    return proxyId;
}
GB_HD inline void gbBpDestroy(GbBroadPhase& bp, int proxyId){
    gbBpUnBufferMove(bp, proxyId);
    --bp.proxyCount;
    gbDestroyProxy(bp.tree, proxyId);
}
GB_HD inline void gbBpMove(GbBroadPhase& bp, int proxyId, const GbAABB& aabb, V2 displacement){
    bool buf = gbMoveProxy(bp.tree, proxyId, aabb, displacement);
    if (buf) gbBpBufferMove(bp, proxyId);
}

// QueryCallback: buffer (min,max) proxy pair.
GB_HD inline void gbBpQueryCallback(GbBroadPhase& bp, int proxyId){
    if (proxyId == bp.queryProxyId) return;
    int a = proxyId < bp.queryProxyId ? proxyId : bp.queryProxyId;
    int b = proxyId > bp.queryProxyId ? proxyId : bp.queryProxyId;
    bp.pairBuffer[bp.pairCount].proxyIdA = a;
    bp.pairBuffer[bp.pairCount].proxyIdB = b;
    ++bp.pairCount;
}

// DFS query of the tree with explicit stack.
GB_HD inline void gbBpQueryTree(GbBroadPhase& bp, const GbAABB& aabb){
    int stack[256];
    int sp = 0;
    stack[sp++] = bp.tree.root;
    while (sp > 0){
        int nodeId = stack[--sp];
        if (nodeId == GB_BP_NULL_NODE) continue;
        const GbBPNode& node = bp.tree.nodes[nodeId];
        if (gbAabbOverlap(node.aabb, aabb)){
            if (node.child1 == GB_BP_NULL_NODE){   // leaf
                gbBpQueryCallback(bp, nodeId);
            } else {
                stack[sp++] = node.child1;
                stack[sp++] = node.child2;
            }
        }
    }
}

// b2PairLessThan ordering.
GB_HD inline bool gbBpPairLess(const GbBPPair& p1, const GbBPPair& p2){
    if (p1.proxyIdA < p2.proxyIdA) return true;
    if (p1.proxyIdA == p2.proxyIdA) return p1.proxyIdB < p2.proxyIdB;
    return false;
}

// In-place insertion sort, identical algorithm to bpSortPairs.
GB_HD inline void gbBpSortPairs(GbBroadPhase& bp){
    for (int i = 1; i < bp.pairCount; ++i){
        GbBPPair key = bp.pairBuffer[i];
        int j = i - 1;
        while (j >= 0 && gbBpPairLess(key, bp.pairBuffer[j])){
            bp.pairBuffer[j+1] = bp.pairBuffer[j];
            --j;
        }
        bp.pairBuffer[j+1] = key;
    }
}

// UpdatePairs, produces addA/addB (userData ids) in AddPair call order.
// Bit-identical to b2BroadPhase::UpdatePairs from the Box2D 2.3.0 b2DynamicTree/b2BroadPhase port.
GB_HD inline void gbBpUpdatePairs(GbBroadPhase& bp){
    bp.pairCount = 0;
    bp.addCount  = 0;
    for (int i = 0; i < bp.moveCount; ++i){
        bp.queryProxyId = bp.moveBuffer[i];
        if (bp.queryProxyId == GB_BP_NULL_NODE) continue;
        GbAABB fatAABB = bp.tree.nodes[bp.queryProxyId].aabb;
        gbBpQueryTree(bp, fatAABB);
    }
    bp.moveCount = 0;
    gbBpSortPairs(bp);
    int i = 0;
    while (i < bp.pairCount){
        GbBPPair primary = bp.pairBuffer[i];
        bp.addA[bp.addCount] = bp.tree.nodes[primary.proxyIdA].userData;
        bp.addB[bp.addCount] = bp.tree.nodes[primary.proxyIdB].userData;
        ++bp.addCount;
        ++i;
        while (i < bp.pairCount){
            GbBPPair p = bp.pairBuffer[i];
            if (p.proxyIdA != primary.proxyIdA || p.proxyIdB != primary.proxyIdB) break;
            ++i;
        }
    }
}

// ---------------------------------------------------------------------------
// Shape AABB helpers using accessor macros (reads world state via BODY/EDGE).
// These are used by the world step to build proxy AABBs for CreateProxy /
// MoveProxy without reading raw WorldShared fields.
// radius must be supplied by caller (e.g., tier_radius(BODY(w,tier,s))).
// ---------------------------------------------------------------------------

// Tight AABB of a circle body at slot s (reads position via BODY accessor).
GB_HD inline GbAABB gbCircleAABB(const GBWorld& w, int s, float radius){
    float cx = BODY(w, sweepCx, s);
    float cy = BODY(w, sweepCy, s);
    GbAABB a;
    a.lo = v2(cx - radius, cy - radius);
    a.hi = v2(cx + radius, cy + radius);
    return a;
}

// Tight AABB of ground edge e (reads endpoints via EDGE accessor).
// polygonRadius == GB_POLYGON_RADIUS from gb_settings.cuh (edge skin = 0.01).
GB_HD inline GbAABB gbEdgeAABB(const GBWorld& w, int e, float polygonRadius){
    float ax = EDGE(w, edgeAx, e), ay = EDGE(w, edgeAy, e);
    float bx = EDGE(w, edgeBx, e), by = EDGE(w, edgeBy, e);
    GbAABB a;
    a.lo = v2((ax < bx ? ax : bx) - polygonRadius,
              (ay < by ? ay : by) - polygonRadius);
    a.hi = v2((ax > bx ? ax : bx) + polygonRadius,
              (ay > by ? ay : by) + polygonRadius);
    return a;
}

// Fat-AABB overlap of two proxies (used by caller for destruction check).
GB_HD inline bool gbProxyFatOverlap(const GbBroadPhase& bp, int pA, int pB){
    return gbAabbOverlap(bp.tree.nodes[pA].aabb, bp.tree.nodes[pB].aabb);
}

// ---------------------------------------------------------------------------
// userData encoding helpers.
//   edges:  -(e+1)    (-1, -2, -3 for edges 0, 1, 2)
//   bodies: slot s    (>=1; slot 0 is static ground, no proxy)
// ---------------------------------------------------------------------------
GB_HD inline int  gbBpUserOfEdge(int e){ return -(e+1); }
GB_HD inline bool gbBpIsEdge(int u)    { return u < 0; }
GB_HD inline int  gbBpEdgeOf(int u)    { return -u - 1; }
