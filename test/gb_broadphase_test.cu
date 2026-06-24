// gb_broadphase_test.cu, micro-test for gb_broadphase. 0-ULP vs CPU Box2D 2.3.0.
//
// Two checks (both required for GREEN):
//   (A) proxyId assignment, ground edges and N fruit proxies created in fixture
//       order must get exactly the same proxyIds as b2_broadphase.cuh's BroadPhase.
//       CPU reference: edges get 0,1,3; fruit i gets 5+2*i (verified in test_bp).
//
//   (B) AddPair set + ORDER, build the same fixed scene in both the reference
//       BroadPhase (b2_broadphase.cuh, reads raw structs) and the new GbBroadPhase
//       (gb_broadphase.cuh, reads via BODY/EDGE accessors on a WorldShared).
//       Call UpdatePairs once after all proxies are created. Compare addCount,
//       and then every (addA[k], addB[k]) pair in order, 0 ULP because these are
//       integers, not floats; the comparison is exact identity.
//
// Build (host-only, identical bits to CPU Box2D under frozen flags):
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest \
//        test/gb_broadphase_test.cu -o test/gb_broadphase_test
//
// Run:
//   ./test/gb_broadphase_test
//   Expected output: PASS gb_broadphase: proxyId exact, 0 ULP pair order
//
// The test is __host__-only (device compilation not needed for the 0-ULP comparison
// since the algorithm is __host__ __device__; the host path is sufficient and keeps
// build time minimal). The frozen flags ensure the float arithmetic is bit-identical
// to the CPU Box2D reference.
//
// ---- includes ---------------------------------------------------------------
// Reference: the faithful b2_broadphase.cuh (reads raw world_types WorldArena).
// Subject:   gb_broadphase.cuh (reads via BODY/EDGE on WorldShared).
// Both headers define overlapping type names, so we isolate them in separate
// compilation units via a thin wrapper approach, here we include the subject
// first (as the module under test) and replicate the reference logic inline
// using renamed types to avoid clashes.

// Subject under test
#include "gpu_box2d/gb_broadphase.cuh"    // GbBroadPhase, GbAABB, BODY, EDGE, ...

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Game geometry constants (world_types.cuh values, copy only what we need so
// we avoid pulling in the raw WorldArena / V2 redefinition from world_types.cuh)
// ---------------------------------------------------------------------------
#define TEST_WALL_X       3.75f
#define TEST_CONTAINER_H  9.5f
#define TEST_N_EDGES      3
// b2_polygonRadius = 2 * b2_linearSlop = 2 * 0.005 = 0.01
#define TEST_POLY_RADIUS  (2.0f * 0.005f)

// Fruit radii by tier (tier_radius from world_types.cuh, tiers 0-10)
static float tier_radius_ref(int tier){
    static const float r[11] = {
        0.24f, 0.32f, 0.40f, 0.56f, 0.72f,
        0.84f, 1.00f, 1.20f, 1.36f, 1.68f, 1.96f
    };
    return (tier >= 0 && tier < 11) ? r[tier] : 0.24f;
}

// ---------------------------------------------------------------------------
// ULP diff helper (per MICROTEST_TEMPLATE.md, not needed for integer pairs,
// but included for completeness so the pattern is present if float checks added)
// ---------------------------------------------------------------------------
inline long ulpDiff(float a, float b){
    int ai = *(int*)&a, bi = *(int*)&b;
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return labs((long)ai - (long)bi);
}

// ---------------------------------------------------------------------------
// Reference BroadPhase: replicate b2_broadphase.cuh logic with Ref-prefixed
// types (AABB/BPNode/DynTree/BroadPhase) so we can link both in one TU.
// The logic is a verbatim copy of b2_broadphase.cuh, that IS the reference.
// ---------------------------------------------------------------------------
#define REF_NULL  (-1)
#define REF_NCAP  256
#define REF_MCAP  80
#define REF_PCAP  512

struct RefAABB { float lox,loy,hix,hiy; };
static inline float refPerim(RefAABB a){ return 2.f*((a.hix-a.lox)+(a.hiy-a.loy)); }
static inline RefAABB refCombine(RefAABB a, RefAABB b){
    RefAABB r;
    r.lox=a.lox<b.lox?a.lox:b.lox; r.loy=a.loy<b.loy?a.loy:b.loy;
    r.hix=a.hix>b.hix?a.hix:b.hix; r.hiy=a.hiy>b.hiy?a.hiy:b.hiy;
    return r;
}
static inline bool refContains(RefAABB a, RefAABB b){
    return a.lox<=b.lox && a.loy<=b.loy && b.hix<=a.hix && b.hiy<=a.hiy;
}
static inline bool refOverlap(RefAABB a, RefAABB b){
    if(b.lox-a.hix>0.f||b.loy-a.hiy>0.f) return false;
    if(a.lox-b.hix>0.f||a.loy-b.hiy>0.f) return false;
    return true;
}
struct RefNode {
    RefAABB aabb; int userData,parent,child1,child2,height;
};
struct RefTree {
    RefNode nodes[REF_NCAP];
    int root,nodeCount,nodeCapacity,freeList,insertionCount;
};
static void refTreeInit(RefTree& t){
    t.root=REF_NULL; t.nodeCapacity=16; t.nodeCount=0;
    for(int i=0;i<REF_NCAP;++i){
        t.nodes[i].userData=-1; t.nodes[i].child1=REF_NULL;
        t.nodes[i].child2=REF_NULL; t.nodes[i].parent=REF_NULL;
        t.nodes[i].height=-1;
        t.nodes[i].aabb={0,0,0,0};
    }
    for(int i=0;i<t.nodeCapacity-1;++i){t.nodes[i].parent=i+1;t.nodes[i].height=-1;}
    t.nodes[t.nodeCapacity-1].parent=REF_NULL; t.nodes[t.nodeCapacity-1].height=-1;
    t.freeList=0; t.insertionCount=0;
}
static int refAllocNode(RefTree& t){
    if(t.freeList==REF_NULL){
        int oc=t.nodeCapacity; t.nodeCapacity*=2;
        for(int i=oc;i<t.nodeCapacity-1;++i){t.nodes[i].parent=i+1;t.nodes[i].height=-1;}
        t.nodes[t.nodeCapacity-1].parent=REF_NULL; t.nodes[t.nodeCapacity-1].height=-1;
        t.freeList=oc;
    }
    int id=t.freeList; t.freeList=t.nodes[id].parent;
    t.nodes[id].parent=REF_NULL; t.nodes[id].child1=REF_NULL;
    t.nodes[id].child2=REF_NULL; t.nodes[id].height=0; t.nodes[id].userData=-1;
    ++t.nodeCount; return id;
}
static void refFreeNode(RefTree& t,int id){
    t.nodes[id].parent=t.freeList; t.nodes[id].height=-1; t.freeList=id; --t.nodeCount;
}
static int refBalance(RefTree& t,int iA){
    RefNode*A=&t.nodes[iA];
    if(A->child1==REF_NULL||A->height<2) return iA;
    int iB=A->child1,iC=A->child2;
    RefNode*B=&t.nodes[iB],*C=&t.nodes[iC];
    int bal=C->height-B->height;
    if(bal>1){
        int iF=C->child1,iG=C->child2;
        RefNode*F=&t.nodes[iF],*G=&t.nodes[iG];
        C->child1=iA; C->parent=A->parent; A->parent=iC;
        if(C->parent!=REF_NULL){if(t.nodes[C->parent].child1==iA)t.nodes[C->parent].child1=iC;else t.nodes[C->parent].child2=iC;}else t.root=iC;
        if(F->height>G->height){C->child2=iF;A->child2=iG;G->parent=iA;A->aabb=refCombine(B->aabb,G->aabb);C->aabb=refCombine(A->aabb,F->aabb);A->height=1+(B->height>G->height?B->height:G->height);C->height=1+(A->height>F->height?A->height:F->height);}
        else{C->child2=iG;A->child2=iF;F->parent=iA;A->aabb=refCombine(B->aabb,F->aabb);C->aabb=refCombine(A->aabb,G->aabb);A->height=1+(B->height>F->height?B->height:F->height);C->height=1+(A->height>G->height?A->height:G->height);}
        return iC;
    }
    if(bal<-1){
        int iD=B->child1,iE=B->child2;
        RefNode*D=&t.nodes[iD],*E=&t.nodes[iE];
        B->child1=iA; B->parent=A->parent; A->parent=iB;
        if(B->parent!=REF_NULL){if(t.nodes[B->parent].child1==iA)t.nodes[B->parent].child1=iB;else t.nodes[B->parent].child2=iB;}else t.root=iB;
        if(D->height>E->height){B->child2=iD;A->child1=iE;E->parent=iA;A->aabb=refCombine(C->aabb,E->aabb);B->aabb=refCombine(A->aabb,D->aabb);A->height=1+(C->height>E->height?C->height:E->height);B->height=1+(A->height>D->height?A->height:D->height);}
        else{B->child2=iE;A->child1=iD;D->parent=iA;A->aabb=refCombine(C->aabb,D->aabb);B->aabb=refCombine(A->aabb,E->aabb);A->height=1+(C->height>D->height?C->height:D->height);B->height=1+(A->height>E->height?A->height:E->height);}
        return iB;
    }
    return iA;
}
static void refInsertLeaf(RefTree& t,int leaf){
    ++t.insertionCount;
    if(t.root==REF_NULL){t.root=leaf;t.nodes[t.root].parent=REF_NULL;return;}
    RefAABB lAABB=t.nodes[leaf].aabb;
    int idx=t.root;
    while(t.nodes[idx].child1!=REF_NULL){
        int c1=t.nodes[idx].child1,c2=t.nodes[idx].child2;
        float area=refPerim(t.nodes[idx].aabb);
        RefAABB cAABB=refCombine(t.nodes[idx].aabb,lAABB);
        float cArea=refPerim(cAABB);
        float cost=2.f*cArea, inh=2.f*(cArea-area);
        float cost1,cost2;
        if(t.nodes[c1].child1==REF_NULL){RefAABB ab=refCombine(lAABB,t.nodes[c1].aabb);cost1=refPerim(ab)+inh;}
        else{RefAABB ab=refCombine(lAABB,t.nodes[c1].aabb);float oa=refPerim(t.nodes[c1].aabb);cost1=(refPerim(ab)-oa)+inh;}
        if(t.nodes[c2].child1==REF_NULL){RefAABB ab=refCombine(lAABB,t.nodes[c2].aabb);cost2=refPerim(ab)+inh;}
        else{RefAABB ab=refCombine(lAABB,t.nodes[c2].aabb);float oa=refPerim(t.nodes[c2].aabb);cost2=refPerim(ab)-oa+inh;}
        if(cost<cost1&&cost<cost2) break;
        idx=(cost1<cost2)?c1:c2;
    }
    int sib=idx,oldP=t.nodes[sib].parent,newP=refAllocNode(t);
    t.nodes[newP].parent=oldP; t.nodes[newP].userData=-1;
    t.nodes[newP].aabb=refCombine(lAABB,t.nodes[sib].aabb);
    t.nodes[newP].height=t.nodes[sib].height+1;
    if(oldP!=REF_NULL){
        if(t.nodes[oldP].child1==sib)t.nodes[oldP].child1=newP;else t.nodes[oldP].child2=newP;
        t.nodes[newP].child1=sib;t.nodes[newP].child2=leaf;
        t.nodes[sib].parent=newP;t.nodes[leaf].parent=newP;
    }else{
        t.nodes[newP].child1=sib;t.nodes[newP].child2=leaf;
        t.nodes[sib].parent=newP;t.nodes[leaf].parent=newP;t.root=newP;
    }
    idx=t.nodes[leaf].parent;
    while(idx!=REF_NULL){
        idx=refBalance(t,idx);
        int c1=t.nodes[idx].child1,c2=t.nodes[idx].child2;
        t.nodes[idx].height=1+(t.nodes[c1].height>t.nodes[c2].height?t.nodes[c1].height:t.nodes[c2].height);
        t.nodes[idx].aabb=refCombine(t.nodes[c1].aabb,t.nodes[c2].aabb);
        idx=t.nodes[idx].parent;
    }
}
static void refRemoveLeaf(RefTree& t,int leaf){
    if(leaf==t.root){t.root=REF_NULL;return;}
    int par=t.nodes[leaf].parent,gpar=t.nodes[par].parent;
    int sib=(t.nodes[par].child1==leaf)?t.nodes[par].child2:t.nodes[par].child1;
    if(gpar!=REF_NULL){
        if(t.nodes[gpar].child1==par)t.nodes[gpar].child1=sib;else t.nodes[gpar].child2=sib;
        t.nodes[sib].parent=gpar; refFreeNode(t,par);
        int idx=gpar;
        while(idx!=REF_NULL){idx=refBalance(t,idx);int c1=t.nodes[idx].child1,c2=t.nodes[idx].child2;t.nodes[idx].aabb=refCombine(t.nodes[c1].aabb,t.nodes[c2].aabb);t.nodes[idx].height=1+(t.nodes[c1].height>t.nodes[c2].height?t.nodes[c1].height:t.nodes[c2].height);idx=t.nodes[idx].parent;}
    }else{t.root=sib;t.nodes[sib].parent=REF_NULL;refFreeNode(t,par);}
}
static int refCreateProxy(RefTree& t,RefAABB aabb,int ud){
    int id=refAllocNode(t);
    const float E=0.1f;
    t.nodes[id].aabb={aabb.lox-E,aabb.loy-E,aabb.hix+E,aabb.hiy+E};
    t.nodes[id].userData=ud; t.nodes[id].height=0; refInsertLeaf(t,id); return id;
}
struct RefPair{int a,b;};
struct RefBP {
    RefTree tree; int proxyCount;
    int moveBuffer[REF_MCAP],moveCount;
    RefPair pairBuffer[REF_PCAP]; int pairCount;
    int queryProxyId;
    int addA[REF_PCAP],addB[REF_PCAP],addCount;
};
static void refBpInit(RefBP& bp){
    refTreeInit(bp.tree); bp.proxyCount=0; bp.moveCount=0; bp.pairCount=0; bp.addCount=0;
}
static int refBpCreate(RefBP& bp,RefAABB aabb,int ud){
    int id=refCreateProxy(bp.tree,aabb,ud); ++bp.proxyCount;
    bp.moveBuffer[bp.moveCount++]=id; return id;
}
static void refQueryCB(RefBP& bp,int pid){
    if(pid==bp.queryProxyId) return;
    int a=pid<bp.queryProxyId?pid:bp.queryProxyId;
    int b=pid>bp.queryProxyId?pid:bp.queryProxyId;
    bp.pairBuffer[bp.pairCount].a=a; bp.pairBuffer[bp.pairCount].b=b; ++bp.pairCount;
}
static void refQueryTree(RefBP& bp,RefAABB aabb){
    int stack[256],sp=0;
    stack[sp++]=bp.tree.root;
    while(sp>0){
        int nid=stack[--sp]; if(nid==REF_NULL) continue;
        RefNode& n=bp.tree.nodes[nid];
        if(refOverlap(n.aabb,aabb)){
            if(n.child1==REF_NULL) refQueryCB(bp,nid);
            else{stack[sp++]=n.child1;stack[sp++]=n.child2;}
        }
    }
}
static void refSortPairs(RefBP& bp){
    for(int i=1;i<bp.pairCount;++i){
        RefPair key=bp.pairBuffer[i]; int j=i-1;
        while(j>=0&&(bp.pairBuffer[j].a>key.a||(bp.pairBuffer[j].a==key.a&&bp.pairBuffer[j].b>key.b))){
            bp.pairBuffer[j+1]=bp.pairBuffer[j]; --j;
        }
        bp.pairBuffer[j+1]=key;
    }
}
static void refBpUpdatePairs(RefBP& bp){
    bp.pairCount=0; bp.addCount=0;
    for(int i=0;i<bp.moveCount;++i){
        bp.queryProxyId=bp.moveBuffer[i];
        if(bp.queryProxyId==REF_NULL) continue;
        RefAABB fat=bp.tree.nodes[bp.queryProxyId].aabb; refQueryTree(bp,fat);
    }
    bp.moveCount=0; refSortPairs(bp);
    int i=0;
    while(i<bp.pairCount){
        RefPair pr=bp.pairBuffer[i];
        bp.addA[bp.addCount]=bp.tree.nodes[pr.a].userData;
        bp.addB[bp.addCount]=bp.tree.nodes[pr.b].userData;
        ++bp.addCount; ++i;
        while(i<bp.pairCount&&bp.pairBuffer[i].a==pr.a&&bp.pairBuffer[i].b==pr.b) ++i;
    }
}

// ---------------------------------------------------------------------------
// Fixed test scene: 3 edges + 5 fruits of varying tiers placed to guarantee
// some overlapping pairs at initial UpdatePairs.
// Tiers chosen so fruits are large enough to overlap after CreateProxy inflation.
// ---------------------------------------------------------------------------
static const int   N_FRUITS = 5;
static const int   FRUIT_TIERS[5]  = { 4, 4, 3, 5, 3 };
// Positions: two pairs overlapping (fruits 0&1, fruits 2&4), fruit 3 isolated.
static const float FRUIT_X[5] = { 0.0f,  0.3f, -2.0f,  2.5f, -1.7f };
static const float FRUIT_Y[5] = { 1.0f,  1.0f,  1.5f,  1.5f,  1.5f };

// Build a WorldShared with the test scene so gb_broadphase can read via BODY/EDGE.
static WorldShared buildWorld(){
    WorldShared w;
    memset(&w, 0, sizeof(w));
    // Ground body (slot 0) = static, no proxy.
    w.bodyCount = 1 + N_FRUITS;
    // Edges
    // floor: (-3.75,0)-(3.75,0)
    w.edgeAx[0]=-TEST_WALL_X; w.edgeAy[0]=0.f; w.edgeBx[0]=TEST_WALL_X;  w.edgeBy[0]=0.f;
    // left wall: (-3.75,0)-(-3.75,9.5)
    w.edgeAx[1]=-TEST_WALL_X; w.edgeAy[1]=0.f; w.edgeBx[1]=-TEST_WALL_X; w.edgeBy[1]=TEST_CONTAINER_H;
    // right wall: (3.75,0)-(3.75,9.5)
    w.edgeAx[2]=TEST_WALL_X;  w.edgeAy[2]=0.f; w.edgeBx[2]=TEST_WALL_X;  w.edgeBy[2]=TEST_CONTAINER_H;
    // Fruit bodies (slots 1..N_FRUITS)
    for(int i=0;i<N_FRUITS;++i){
        int s=i+1;
        w.sweepCx[s] = FRUIT_X[i];
        w.sweepCy[s] = FRUIT_Y[i];
        w.tier[s]    = FRUIT_TIERS[i];
        w.alive[s]   = 1;
        w.bodyType[s]= 2;  // GB_DYNAMIC_BODY
    }
    return w;
}

int main(){
    int fail = 0;

    // -----------------------------------------------------------------------
    // Build REFERENCE (RefBP, raw geometry constants)
    // -----------------------------------------------------------------------
    RefBP ref;
    refBpInit(ref);

    // Edge AABBs (tight + extension baked into refCreateProxy)
    // Tight AABB of edge: min(a,b)-polygonRadius .. max(a,b)+polygonRadius
    // Then refCreateProxy inflates by GB_AABB_EXTENSION=0.1.
    // To match gbEdgeAABB (tight = tight AABB with polyRadius included),
    // we pass the tight poly-radius AABB to refCreateProxy.
    float EP = TEST_POLY_RADIUS;
    int refEdge[3], refFruit[N_FRUITS];
    // floor
    { RefAABB a={-TEST_WALL_X-EP,-EP,TEST_WALL_X+EP,EP}; refEdge[0]=refBpCreate(ref,a,0); }
    // left wall
    { RefAABB a={-TEST_WALL_X-EP,-EP,-TEST_WALL_X+EP,TEST_CONTAINER_H+EP}; refEdge[1]=refBpCreate(ref,a,0); }
    // right wall
    { RefAABB a={TEST_WALL_X-EP,-EP,TEST_WALL_X+EP,TEST_CONTAINER_H+EP}; refEdge[2]=refBpCreate(ref,a,0); }
    // fruits
    for(int i=0;i<N_FRUITS;++i){
        float r=tier_radius_ref(FRUIT_TIERS[i]);
        float cx=FRUIT_X[i], cy=FRUIT_Y[i];
        RefAABB a={cx-r,cy-r,cx+r,cy+r};
        refFruit[i]=refBpCreate(ref,a,i+1);
    }
    refBpUpdatePairs(ref);

    // -----------------------------------------------------------------------
    // Build SUBJECT (GbBroadPhase, reads via BODY/EDGE on WorldShared)
    // -----------------------------------------------------------------------
    WorldShared w = buildWorld();
    GbBroadPhase gbp;
    gbBpInit(gbp);

    int gbEdge[3], gbFruit[N_FRUITS];
    for(int e=0;e<3;++e){
        GbAABB a = gbEdgeAABB(w, e, GB_POLYGON_RADIUS);
        gbEdge[e] = gbBpCreate(gbp, a, 0);
    }
    for(int i=0;i<N_FRUITS;++i){
        float r = tier_radius_ref(FRUIT_TIERS[i]);
        int s = i+1;
        GbAABB a = gbCircleAABB(w, s, r);
        gbFruit[i] = gbBpCreate(gbp, a, i+1);
    }
    gbBpUpdatePairs(gbp);

    // -----------------------------------------------------------------------
    // (A) proxyId assignment check
    // -----------------------------------------------------------------------
    printf("=== (A) proxyId assignment ===\n");
    bool okA = true;
    for(int e=0;e<3;++e){
        bool match = (refEdge[e] == gbEdge[e]);
        printf("  edge[%d]: ref=%d  gb=%d  %s\n",e,refEdge[e],gbEdge[e],match?"OK":"FAIL");
        if(!match) okA=false;
    }
    for(int i=0;i<N_FRUITS;++i){
        bool match = (refFruit[i] == gbFruit[i]);
        printf("  fruit[%d]: ref=%d  gb=%d  %s\n",i,refFruit[i],gbFruit[i],match?"OK":"FAIL");
        if(!match) okA=false;
    }
    if(!okA){
        printf("FAIL gb_broadphase: proxyId mismatch\n");
        fail=1;
    } else {
        printf("proxyId: all match (reference verified)\n");
    }

    // -----------------------------------------------------------------------
    // (B) AddPair count + order
    // -----------------------------------------------------------------------
    printf("\n=== (B) AddPair set + order ===\n");
    printf("  ref addCount=%d   gb addCount=%d\n", ref.addCount, gbp.addCount);
    bool okB = true;
    if(ref.addCount != gbp.addCount){
        printf("  FAIL: addCount mismatch\n");
        okB = false; fail = 1;
    } else {
        for(int k=0;k<ref.addCount;++k){
            bool match = (ref.addA[k]==gbp.addA[k] && ref.addB[k]==gbp.addB[k]);
            printf("  [%d] ref=(%d,%d)  gb=(%d,%d)  %s\n",
                   k, ref.addA[k],ref.addB[k], gbp.addA[k],gbp.addB[k],
                   match?"OK":"FAIL");
            if(!match){ okB=false; fail=1; }
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    if(!fail){
        printf("\nPASS gb_broadphase: proxyId exact, 0 ULP pair order\n");
        return 0;
    } else {
        printf("\nFAIL gb_broadphase: see above\n");
        return 1;
    }
}
