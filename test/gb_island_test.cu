// gb_island_test.cu. Solver and island micro-test. Proves gb_contact_solver.cuh + gb_island.cuh
// (the re-hosted b2ContactSolver + b2Island::Solve + b2World::Solve) match the CPU
// Box2D 2.3.0 reference solver to 0 ULP on multi-body scenarios.
//
// REFERENCE = Box2D 2.3.0 b2World::Solve, the instrumented 0-ULP port. It lives in
// gb_island_ref.cu, a separate translation unit, because the reference's type
// universe clashes with the engine's gb_* type universe. The test talks to it over
// the flat-POD gb_test_iface.h. The engine code (gbWorldSolve) runs on a WorldShared
// (the gb_pools layout) through the accessor macros, on host and on the device
// block-per-world shell.
//
// METHOD (isolates the solver): each substep,
//   1) ref_collide(): reference collide on the golden arena (identical input),
//      export the post-collide, pre-solve state,
//   2) load that state into a fresh WorldShared,
//   3) run gbWorldSolve on it on HOST and on the DEVICE block shell,
//   4) ref_solve(): reference worldSolve advances the golden arena, export solved truth,
//   5) diff every body+contact OUTPUT field: ref vs shared-host vs shared-device.
// Identical solver inputs every substep over a long settle => exercises island
// assembly (descending seed + descending incident contact), the 8 vel + 3 pos
// Gauss-Seidel sweeps, and ALL THREE float folds (vel-accum, minSeparation,
// minSleepTime) on single-drop / 2-body / 5-body pile.
//
// Build (frozen flags), two translation units. gb_island_ref.cu needs the Box2D
// 2.3.0 reference headers (see test/README.md):
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest -I<box2d-reference-dir> \
//        test/gb_island_test.cu test/gb_island_ref.cu -o test/gb_island_test
#include "gpu_box2d/gb_island.cuh"  // the engine: gbWorldSolve -> gb_contact_solver
#include "gb_test_iface.h"          // flat-POD bridge to the reference TU
#include <cstdio>
#include <cstring>
#include <cmath>

// ---- ULP comparison (the exact helper from MICROTEST_TEMPLATE.md) ----------
inline long ulpDiff(float a, float b){
    int ai = *(int*)&a, bi = *(int*)&b;
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return labs((long)ai - (long)bi);
}

// ---- load a flat SolverState into a WorldShared (the gb_pools layout). ------
inline void stateToShared(const SolverState& a, WorldShared& s){
    memset(&s, 0, sizeof(WorldShared));
    for (int i=0;i<GB_MAX_BODIES;++i){
        s.sweepCx[i]=a.sweepCx[i];   s.sweepCy[i]=a.sweepCy[i];
        s.sweepC0x[i]=a.sweepC0x[i]; s.sweepC0y[i]=a.sweepC0y[i];
        s.sweepA[i]=a.sweepA[i];     s.sweepA0[i]=a.sweepA0[i]; s.sweepAlpha0[i]=a.sweepAlpha0[i];
        s.xfPx[i]=a.xfPx[i]; s.xfPy[i]=a.xfPy[i]; s.xfQs[i]=a.xfQs[i]; s.xfQc[i]=a.xfQc[i];
        s.velX[i]=a.velX[i]; s.velY[i]=a.velY[i]; s.angVel[i]=a.angVel[i];
        s.invMass[i]=a.invMass[i]; s.invI[i]=a.invI[i];
        s.tier[i]=a.tier[i]; s.bodyType[i]=a.bodyType[i];
        s.sleepTime[i]=a.sleepTime[i]; s.awake[i]=a.awake[i]; s.alive[i]=a.alive[i];
    }
    s.bodyCount=a.bodyCount;
    for (int e=0;e<GB_N_EDGES;++e){ s.edgeAx[e]=a.edgeAx[e]; s.edgeAy[e]=a.edgeAy[e];
                                    s.edgeBx[e]=a.edgeBx[e]; s.edgeBy[e]=a.edgeBy[e]; }
    for (int c=0;c<GB_MAX_CONTACTS;++c){
        s.cBodyA[c]=a.cBodyA[c]; s.cBodyB[c]=a.cBodyB[c]; s.cEdge[c]=a.cEdge[c];
        s.cTouching[c]=a.cTouching[c];
        s.cFriction[c]=a.cFriction[c]; s.cRestitution[c]=a.cRestitution[c];
        s.cManifoldType[c]=a.cManifoldType[c];
        s.cLocalNormalX[c]=a.cLocalNormalX[c]; s.cLocalNormalY[c]=a.cLocalNormalY[c];
        s.cLocalPointX[c]=a.cLocalPointX[c];   s.cLocalPointY[c]=a.cLocalPointY[c];
        s.cPointLocalX[c]=a.cPointLocalX[c];   s.cPointLocalY[c]=a.cPointLocalY[c];
        s.cNormalImpulse[c]=a.cNormalImpulse[c]; s.cTangentImpulse[c]=a.cTangentImpulse[c];
        s.cToi[c]=a.cToi[c]; s.cToiCount[c]=a.cToiCount[c];
        s.cToiFlag[c]=a.cToiFlag[c]; s.cEnabled[c]=a.cEnabled[c];
    }
    s.contactCount=a.contactCount;
}

// ---- diff: SOLVER OUTPUTS (per-body kinematics + per-contact warm-start
// impulses + awake/sleep) between reference SolverState and candidate WorldShared. ----
struct DiffStat { long maxUlp; const char* worstName; int worstSlot; };
inline void chk(DiffStat& d, const char* name, int slot, float ref, float got){
    long u = ulpDiff(ref, got);
    if (u > d.maxUlp){ d.maxUlp=u; d.worstName=name; d.worstSlot=slot; }
}
inline DiffStat diffSolved(const SolverState& a, const WorldShared& s){
    DiffStat d{0,"-",-1};
    for (int i=0;i<a.bodyCount;++i){
        chk(d,"sweepCx",i,a.sweepCx[i],s.sweepCx[i]); chk(d,"sweepCy",i,a.sweepCy[i],s.sweepCy[i]);
        chk(d,"sweepA",i,a.sweepA[i],s.sweepA[i]);
        chk(d,"sweepC0x",i,a.sweepC0x[i],s.sweepC0x[i]); chk(d,"sweepC0y",i,a.sweepC0y[i],s.sweepC0y[i]);
        chk(d,"sweepA0",i,a.sweepA0[i],s.sweepA0[i]);
        chk(d,"velX",i,a.velX[i],s.velX[i]); chk(d,"velY",i,a.velY[i],s.velY[i]);
        chk(d,"angVel",i,a.angVel[i],s.angVel[i]);
        chk(d,"xfPx",i,a.xfPx[i],s.xfPx[i]); chk(d,"xfPy",i,a.xfPy[i],s.xfPy[i]);
        chk(d,"xfQs",i,a.xfQs[i],s.xfQs[i]); chk(d,"xfQc",i,a.xfQc[i],s.xfQc[i]);
        chk(d,"sleepTime",i,a.sleepTime[i],s.sleepTime[i]);
        if (a.awake[i]!=s.awake[i] && (long)1 > d.maxUlp){ d.maxUlp=1; d.worstName="awake"; d.worstSlot=i; }
    }
    for (int c=0;c<a.contactCount;++c){
        chk(d,"cNormalImpulse",c,a.cNormalImpulse[c],s.cNormalImpulse[c]);
        chk(d,"cTangentImpulse",c,a.cTangentImpulse[c],s.cTangentImpulse[c]);
    }
    return d;
}

// ---- device entry: run gbWorldSolve through the block-per-world shared-mem shell ---
// (serial spine on lane 0, the execution model from gb_world.cuh).
__global__ void kSolveBlock(WorldShared* g){
    extern __shared__ unsigned char smem[];
    WorldShared& w = *reinterpret_cast<WorldShared*>(smem);
    int* d = reinterpret_cast<int*>(&w);
    int* sg = reinterpret_cast<int*>(g);
    const int n = (int)(sizeof(WorldShared)/sizeof(int));
    for (int i=threadIdx.x;i<n;i+=blockDim.x) d[i]=sg[i];
    __syncthreads();
    if (threadIdx.x==0) gbWorldSolve(w);     // SERIAL spine on lane 0
    __syncthreads();
    for (int i=threadIdx.x;i<n;i+=blockDim.x) sg[i]=d[i];
}

int runScenario(const char* name, const SeedBody* seeds, int N, int nsub){
    ref_init(seeds, N);
    WorldShared* dW; cudaMalloc(&dW, sizeof(WorldShared));
    long maxUlpHost=0, maxUlpDev=0; const char* wnH="-"; int wsH=-1,subH=-1;
    const char* wnD="-"; int wsD=-1,subD=-1; int contactsSeen=0;

    for (int sub=0; sub<nsub; ++sub){
        SolverState pre;  ref_collide(&pre);   // post-collide / pre-solve
        if (pre.contactCount>contactsSeen) contactsSeen=pre.contactCount;
        WorldShared sh;   stateToShared(pre, sh);
        // host solve
        WorldShared shHost = sh; gbWorldSolve(shHost);
        // device solve (block shell, lane-0 serial spine)
        cudaMemcpy(dW, &sh, sizeof(WorldShared), cudaMemcpyHostToDevice);
        kSolveBlock<<<1,128,(int)sizeof(WorldShared)>>>(dW);
        cudaError_t e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){ fprintf(stderr,"[%s] CUDA ERR sub=%d: %s\n",name,sub,cudaGetErrorString(e)); cudaFree(dW); return 1; }
        WorldShared shDev; cudaMemcpy(&shDev, dW, sizeof(WorldShared), cudaMemcpyDeviceToHost);
        // reference solve = ground truth
        SolverState post; ref_solve(&post);
        // diff
        DiffStat dH = diffSolved(post, shHost);
        DiffStat dD = diffSolved(post, shDev);
        if (dH.maxUlp>maxUlpHost){ maxUlpHost=dH.maxUlp; wnH=dH.worstName; wsH=dH.worstSlot; subH=sub; }
        if (dD.maxUlp>maxUlpDev ){ maxUlpDev =dD.maxUlp; wnD=dD.worstName; wsD=dD.worstSlot; subD=sub; }
    }
    cudaFree(dW);
    bool pass = (maxUlpHost==0 && maxUlpDev==0);
    printf("  %-16s N=%d substeps=%d peakContacts=%d  HOST maxUlp=%ld  DEVICE maxUlp=%ld  -> %s\n",
           name, N, nsub, contactsSeen, maxUlpHost, maxUlpDev, pass?"PASS (0 ULP)":"FAIL");
    if (!pass){
        if(maxUlpHost) printf("      HOST   worst: field=%s slot=%d sub=%d ulp=%ld\n", wnH, wsH, subH, maxUlpHost);
        if(maxUlpDev)  printf("      DEVICE worst: field=%s slot=%d sub=%d ulp=%ld\n", wnD, wsD, subD, maxUlpDev);
    }
    return pass?0:1;
}

int main(){
    printf("Solver and island micro-test: gb_contact_solver.cuh + gb_island.cuh vs CPU Box2D 2.3.0 (b2_step::worldSolve)\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true | reference TU = Box2D 2.3.0 b2World::Solve\n\n");
    int fails=0;

    {   // s1: single-drop, one tier-2 fruit settling on the floor. 1-body island,
        // fruit-wall contact, vel+pos sweeps, sleep fold.
        SeedBody s[] = {{2, 0.0f, 0.55f, 0.0f, 0.0f}};
        fails += runScenario("s1_single_drop", s, 1, 200);
    }
    {   // s2: 2-body, two equal fruits stacked, touching each other + floor.
        // multi-body island, fruit-fruit + fruit-wall, Gauss-Seidel read-after-write
        // chain across contacts, all three folds.
        SeedBody s[] = {
            {2, 0.0f, 0.55f, 0.0f, 0.0f},
            {2, 0.0f, 1.50f, 0.0f, 0.0f},
        };
        fails += runScenario("s2_two_body", s, 2, 200);
    }
    {   // s3: 5-body pile, cluster settling into a multi-contact island. Heaviest
        // test of assembly order, 8 vel + 3 pos sweeps, minSeparation early-exit,
        // minSleepTime fold across 5 bodies.
        SeedBody s[] = {
            {2, -0.55f, 0.55f, 0.0f, 0.0f},
            {2,  0.55f, 0.55f, 0.0f, 0.0f},
            {2,  0.00f, 1.40f, 0.0f, 0.0f},
            {2, -0.55f, 2.30f, 0.0f, 0.0f},
            {2,  0.55f, 2.30f, 0.0f, 0.0f},
        };
        fails += runScenario("s3_five_pile", s, 5, 300);
    }

    printf("\n%s: solver/island re-host is %s vs CPU Box2D 2.3.0 on multi-body scenarios.\n",
           fails? "FAIL":"PASS", fails? "DIVERGENT":"0-ULP IDENTICAL");
    return fails?1:0;
}
