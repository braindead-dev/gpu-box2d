// gb_toi_test.cu, micro-test for gb_toi. 0-ULP vs CPU Box2D 2.3.0 (b2_toi.cuh).
// Scenario: tier-0 circle (r=0.25) falling toward the floor edge at y=0.
// The CPU reference is the b2_toi.cuh path; the GPU path is gb_toi.cuh.
// Both run on the same fixed input. Bit-identical math under --fmad=false.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cassert>

// ---- pull in both implementations -------------------------------------------
// Reference: original b2_toi.cuh (via the legacy headers it includes)
// We need b2_device.cuh -> world_types.cuh chain for the reference.
// Include gb_toi.cuh first (it uses GB_ names), then the reference.

// This test compares against the Box2D 2.3.0 CCD reference. The reference header is
// supplied when the narrow-phase and solver modules are assembled (see test/README.md).
// Build with: nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86
//             -Iinclude -Itest -I<box2d-reference-dir>
//             test/gb_toi_test.cu -o test/gb_toi_test

#include "gpu_box2d/gb_toi.cuh"   // the engine's CCD path (GB_ names)

// Include the Box2D 2.3.0 reference CCD path in its own namespace to avoid symbol
// collisions with the engine's GB_ type universe.
namespace ref {
#include "b2_toi.cuh"
}

// ULP diff helper (from MICROTEST_TEMPLATE.md, exact copy)
inline long ulpDiff(float a, float b){
    int ai = *(int*)&a, bi = *(int*)&b;
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return labs((long)ai - (long)bi);
}

// ---- GPU kernel: runs gb_toi on a fruit-wall scenario -----------------------
__global__ void runGBTOI(GBTOIOut* out_d,
                          GBDistOutput* dist_out_d,
                          // input proxy A: floor edge [(-3.75,0)-(3.75,0)], r=0.01
                          float eAx, float eAy, float eBx, float eBy,
                          // input proxy B: circle at (0, 0.5), r=0.25, moving to (0, -0.3)
                          float cx0, float cy0, float cx1, float cy1)
{
    // Build edge proxy (floor)
    GBDProxy pA;
    pA.v[0] = v2(eAx, eAy); pA.v[1] = v2(eBx, eBy);
    pA.count = 2; pA.radius = GB_POLYGON_RADIUS;

    // Build circle proxy (fruit, m_p == 0)
    GBDProxy pB;
    pB.v[0] = v2(0.0f, 0.0f); pB.count = 1; pB.radius = 0.25f;

    // Build sweeps: edge is static (alpha0=0, c0=c=(0,0), a=0)
    GBSweep sA;
    sA.localCenter = v2(0.0f, 0.0f);
    sA.c0 = v2(0.0f, 0.0f); sA.c = v2(0.0f, 0.0f);
    sA.a0 = 0.0f; sA.a = 0.0f; sA.alpha0 = 0.0f;

    // circle sweep: starts at (cx0,cy0), ends at (cx1,cy1), no rotation
    GBSweep sB;
    sB.localCenter = v2(0.0f, 0.0f);
    sB.c0 = v2(cx0, cy0); sB.c = v2(cx1, cy1);
    sB.a0 = 0.0f; sB.a = 0.0f; sB.alpha0 = 0.0f;

    // Run GJK distance at t=0
    Xf xfA, xfB;
    gbSweepGetTransform(sA, xfA, 0.0f);
    gbSweepGetTransform(sB, xfB, 0.0f);
    GBDistInput din; din.proxyA=pA; din.proxyB=pB; din.xfA=xfA; din.xfB=xfB; din.useRadii=false;
    GBSimplexCache cache; cache.count=0;
    gbDistanceGJK(*dist_out_d, cache, din);

    // Run TOI
    gbTOI(*out_d, pA, pB, sA, sB, 1.0f);
}

int main(){
    // ---- Define the exact scenario ------------------------------------------
    // Floor edge: from (-3.75, 0) to (3.75, 0), polygon radius = 0.01
    float eAx=-3.75f, eAy=0.0f, eBx=3.75f, eBy=0.0f;
    // Tier-0 circle (r=0.25): starts at (0, 0.5), moving to (0, -0.1) in one step.
    // This guarantees CCD fires (it would tunnel if not caught by TOI).
    float cx0=0.0f, cy0=0.5f, cx1=0.0f, cy1=-0.1f;

    // ---- CPU reference: run b2_toi.cuh path ---------------------------------
    // Build reference proxies
    ref::DProxy refPA, refPB;
    refPA.v[0] = ref::v2(eAx, eAy); refPA.v[1] = ref::v2(eBx, eBy);
    refPA.count = 2; refPA.radius = 2.0f * 0.005f; // B2_POLYGON_RADIUS

    refPB.v[0] = ref::v2(0.0f, 0.0f); refPB.count = 1; refPB.radius = 0.25f;

    ref::Sweep refSA, refSB;
    refSA.localCenter = ref::v2(0.0f, 0.0f);
    refSA.c0 = ref::v2(0.0f, 0.0f); refSA.c = ref::v2(0.0f, 0.0f);
    refSA.a0 = 0.0f; refSA.a = 0.0f; refSA.alpha0 = 0.0f;

    refSB.localCenter = ref::v2(0.0f, 0.0f);
    refSB.c0 = ref::v2(cx0, cy0); refSB.c = ref::v2(cx1, cy1);
    refSB.a0 = 0.0f; refSB.a = 0.0f; refSB.alpha0 = 0.0f;

    // GJK distance at t=0
    ref::Xf refXfA, refXfB;
    ref::sweepGetTransform(refSA, refXfA, 0.0f);
    ref::sweepGetTransform(refSB, refXfB, 0.0f);
    ref::DistInput refDin; refDin.proxyA=refPA; refDin.proxyB=refPB;
    refDin.xfA=refXfA; refDin.xfB=refXfB; refDin.useRadii=false;
    ref::SimplexCache refCache; refCache.count=0;
    ref::DistOutput refDout; ref::b2DistanceGJK(refDout, refCache, refDin);

    // TOI
    ref::TOIOut refTOI; ref::b2TOI(refTOI, refPA, refPB, refSA, refSB, 1.0f);

    printf("REF: dist=%.9g iters=%d toi_state=%d toi_t=%.9g\n",
           refDout.distance, refDout.iterations, refTOI.state, refTOI.t);

    // ---- GPU path -----------------------------------------------------------
    GBTOIOut* gpu_toi_d; cudaMalloc(&gpu_toi_d, sizeof(GBTOIOut));
    GBDistOutput* gpu_dist_d; cudaMalloc(&gpu_dist_d, sizeof(GBDistOutput));

    runGBTOI<<<1,1>>>(gpu_toi_d, gpu_dist_d,
                       eAx, eAy, eBx, eBy, cx0, cy0, cx1, cy1);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    GBTOIOut gpu_toi; cudaMemcpy(&gpu_toi, gpu_toi_d, sizeof(GBTOIOut), cudaMemcpyDeviceToHost);
    GBDistOutput gpu_dist; cudaMemcpy(&gpu_dist, gpu_dist_d, sizeof(GBDistOutput), cudaMemcpyDeviceToHost);

    printf("GPU: dist=%.9g iters=%d toi_state=%d toi_t=%.9g\n",
           gpu_dist.distance, gpu_dist.iterations, gpu_toi.state, gpu_toi.t);

    // ---- ULP comparison -----------------------------------------------------
    long ulp_dist     = ulpDiff(refDout.distance,    gpu_dist.distance);
    long ulp_pointAx  = ulpDiff(refDout.pointA.x,    gpu_dist.pointA.x);
    long ulp_pointAy  = ulpDiff(refDout.pointA.y,    gpu_dist.pointA.y);
    long ulp_pointBx  = ulpDiff(refDout.pointB.x,    gpu_dist.pointB.x);
    long ulp_pointBy  = ulpDiff(refDout.pointB.y,    gpu_dist.pointB.y);
    long ulp_toi_t    = ulpDiff(refTOI.t,            gpu_toi.t);
    int  state_match  = (refTOI.state == gpu_toi.state) ? 1 : 0;
    int  iters_match  = (refDout.iterations == gpu_dist.iterations) ? 1 : 0;

    long maxULP = ulp_dist;
    if (ulp_pointAx > maxULP) maxULP = ulp_pointAx;
    if (ulp_pointAy > maxULP) maxULP = ulp_pointAy;
    if (ulp_pointBx > maxULP) maxULP = ulp_pointBx;
    if (ulp_pointBy > maxULP) maxULP = ulp_pointBy;
    if (ulp_toi_t   > maxULP) maxULP = ulp_toi_t;

    printf("ULP: dist=%ld pAx=%ld pAy=%ld pBx=%ld pBy=%ld toi_t=%ld state=%s iters=%s\n",
           ulp_dist, ulp_pointAx, ulp_pointAy, ulp_pointBx, ulp_pointBy, ulp_toi_t,
           state_match?"MATCH":"MISMATCH", iters_match?"MATCH":"MISMATCH");

    bool pass = (maxULP == 0) && state_match && iters_match;
    if (pass)
        printf("PASS gb_toi: 0 ULP (GJK distance + b2TOI, fruit-wall CCD scenario)\n");
    else
        printf("FAIL gb_toi: maxULP=%ld state=%s iters=%s\n",
               maxULP, state_match?"ok":"MISMATCH", iters_match?"ok":"MISMATCH");

    cudaFree(gpu_toi_d); cudaFree(gpu_dist_d);
    return pass ? 0 : 1;
}
