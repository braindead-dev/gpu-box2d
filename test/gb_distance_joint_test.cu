// gb_distance_joint_test.cu. Micro-test for gb_distance_joint.cuh. 0-ULP versus a
// self-contained Box2D 2.3.0 reference of b2DistanceJoint's InitVelocityConstraints /
// SolveVelocityConstraints / SolvePositionConstraints, Ref-prefixed.
//
// Two scenarios, each run for many substeps through the full joint solve spine
// (integrate velocity, init, 8 velocity iterations, integrate position, 3 position
// iterations) on both the subject (gb_distance_joint.cuh) and the reference:
//   1. Rigid rod. A static pivot and a dynamic bob held at a fixed distance, swinging
//      under gravity. The position solve corrects the length.
//   2. Soft spring. The same bodies with frequencyHz and a damping ratio, so the bob
//      bounces on the spring (the position solve is skipped, the velocity row carries
//      the bias and gamma).
// Each substep compares the bob velocity, angular velocity, position, angle, and the
// warm-start impulse at 0 ULP.
//
// Build (frozen flags), self-contained, host or device:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_distance_joint_test.cu -o test/gb_distance_joint_test
//   ./test/gb_distance_joint_test
//   Expected: PASS gb_distance_joint: 0 ULP
#include "gpu_box2d/gb_distance_joint.cuh"
#include <cstdio>
#include <cmath>
#include <cstdint>

inline long ulpDiff(float a, float b){
    int ai = *(int*)&a, bi = *(int*)&b;
    if (ai < 0) ai = 0x80000000 - ai;
    if (bi < 0) bi = 0x80000000 - bi;
    return labs((long)ai - (long)bi);
}

// ---------------------------------------------------------------------------
// Box2D 2.3.0 b2DistanceJoint reference (Ref-prefixed).
// ---------------------------------------------------------------------------
struct RV2 { float x, y; };
static inline RV2 rv2(float x, float y){ RV2 r; r.x=x; r.y=y; return r; }
static inline RV2 operator+(RV2 a, RV2 b){ return rv2(a.x+b.x, a.y+b.y); }
static inline RV2 operator-(RV2 a, RV2 b){ return rv2(a.x-b.x, a.y-b.y); }
static inline RV2 operator*(float s, RV2 a){ return rv2(s*a.x, s*a.y); }
static inline float rDot(RV2 a, RV2 b){ return a.x*b.x + a.y*b.y; }
static inline float rCross(RV2 a, RV2 b){ return a.x*b.y - a.y*b.x; }
static inline RV2 rCrossSV(float s, RV2 a){ return rv2(-s*a.y, s*a.x); }
static inline float rLen(RV2 v){ return sqrtf(v.x*v.x + v.y*v.y); }
static inline float rNormalize(RV2& v){
    float len = rLen(v);
    if (len < 1.19209290e-07f) return 0.0f;
    float inv = 1.0f/len; v.x*=inv; v.y*=inv; return len;
}
static inline RV2 rMulRV(float s, float c, RV2 v){ return rv2(c*v.x - s*v.y, s*v.x + c*v.y); }
static inline float rClamp(float a, float lo, float hi){ return a<lo?lo:(a>hi?hi:a); }

#define R_PI 3.14159265359f
#define R_DT (1.0f/60.0f)
#define R_LINEAR_SLOP 0.005f
#define R_MAX_LINEAR_CORRECTION 0.2f

struct RBody { RV2 c, v; float a, w, invM, invI; };
struct RJoint {
    int ia, ib; RV2 lA, lB; float mA, mB, iA, iB;
    float length, frequencyHz, dampingRatio;
    RV2 u, rA, rB; float mass, gamma, bias, impulse;
};
static RBody rB[2];
static RJoint rJ;

static void rInit(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 cA=rB[ia].c; float aA=rB[ia].a;
    RV2 cB=rB[ib].c; float aB=rB[ib].a;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    float sA=sinf(aA), csA=cosf(aA), sB=sinf(aB), csB=cosf(aB);
    rJ.rA=rMulRV(sA,csA,rJ.lA); rJ.rB=rMulRV(sB,csB,rJ.lB);
    rJ.u = cB + rJ.rB - cA - rJ.rA;
    float length = rLen(rJ.u);
    if (length > R_LINEAR_SLOP) rJ.u = (1.0f/length) * rJ.u; else rJ.u = rv2(0,0);
    float crAu = rCross(rJ.rA, rJ.u);
    float crBu = rCross(rJ.rB, rJ.u);
    float invMass = mA + iA*crAu*crAu + mB + iB*crBu*crBu;
    rJ.mass = invMass != 0.0f ? 1.0f/invMass : 0.0f;
    if (rJ.frequencyHz > 0.0f){
        float C = length - rJ.length;
        float omega = 2.0f * R_PI * rJ.frequencyHz;
        float d = 2.0f * rJ.mass * rJ.dampingRatio * omega;
        float k = rJ.mass * omega * omega;
        float h = R_DT;
        rJ.gamma = h*(d + h*k);
        rJ.gamma = rJ.gamma != 0.0f ? 1.0f/rJ.gamma : 0.0f;
        rJ.bias = C * h * k * rJ.gamma;
        invMass += rJ.gamma;
        rJ.mass = invMass != 0.0f ? 1.0f/invMass : 0.0f;
    } else { rJ.gamma = 0.0f; rJ.bias = 0.0f; }
    RV2 P = rJ.impulse * rJ.u;
    vA = vA - mA*P; wA -= iA*rCross(rJ.rA, P);
    vB = vB + mB*P; wB += iB*rCross(rJ.rB, P);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static void rSolveVel(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    RV2 vpA = vA + rCrossSV(wA, rJ.rA);
    RV2 vpB = vB + rCrossSV(wB, rJ.rB);
    float Cdot = rDot(rJ.u, vpB - vpA);
    float impulse = -rJ.mass * (Cdot + rJ.bias + rJ.gamma * rJ.impulse);
    rJ.impulse += impulse;
    RV2 P = impulse * rJ.u;
    vA = vA - mA*P; wA -= iA*rCross(rJ.rA, P);
    vB = vB + mB*P; wB += iB*rCross(rJ.rB, P);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static bool rSolvePos(){
    if (rJ.frequencyHz > 0.0f) return true;
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 cA=rB[ia].c; float aA=rB[ia].a;
    RV2 cB=rB[ib].c; float aB=rB[ib].a;
    float sA=sinf(aA), csA=cosf(aA), sB=sinf(aB), csB=cosf(aB);
    RV2 rA=rMulRV(sA,csA,rJ.lA), rBv=rMulRV(sB,csB,rJ.lB);
    RV2 u = cB + rBv - cA - rA;
    float length = rNormalize(u);
    float C = length - rJ.length;
    C = rClamp(C, -R_MAX_LINEAR_CORRECTION, R_MAX_LINEAR_CORRECTION);
    float impulse = -rJ.mass * C;
    RV2 P = impulse * u;
    cA = cA - mA*P; aA -= iA*rCross(rA, P);
    cB = cB + mB*P; aB += iB*rCross(rBv, P);
    rB[ia].c=cA; rB[ia].a=aA; rB[ib].c=cB; rB[ib].a=aB;
    return fabsf(C) < R_LINEAR_SLOP;
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------
static const float BOB_INVM = 1.0f;
static const float BOB_INVI = 2.0f;
static const float GRAV = -9.81f;

static int gFails=0; static long gMax=0;
static void chk(const char* what, int sub, float ref, float got){
    long u=ulpDiff(ref,got); if(u>gMax) gMax=u;
    if(u!=0){ if(gFails==0) printf("  FAIL %-6s sub=%d ref=%.9g got=%.9g ulp=%ld\n", what, sub, ref, got, u); gFails=1; }
}

static void runCase(const char* name, float freq, float damp, int NSUB){
    float h = R_DT;
    // pivot (static) at origin; bob (dynamic) offset to the right at distance 1.5.
    RV2 bobPos = rv2(1.5f, 0.0f);
    rB[0].c=rv2(0,0); rB[0].v=rv2(0,0); rB[0].a=0.0f; rB[0].w=0.0f; rB[0].invM=0.0f; rB[0].invI=0.0f;
    rB[1].c=bobPos;   rB[1].v=rv2(0,0); rB[1].a=0.0f; rB[1].w=0.0f; rB[1].invM=BOB_INVM; rB[1].invI=BOB_INVI;
    rJ.ia=0; rJ.ib=1; rJ.lA=rv2(0,0); rJ.lB=rv2(0,0);
    rJ.mA=0.0f; rJ.mB=BOB_INVM; rJ.iA=0.0f; rJ.iB=BOB_INVI;
    rJ.length=1.5f; rJ.frequencyHz=freq; rJ.dampingRatio=damp; rJ.impulse=0.0f;

    GBIslandData isl;
    isl.bodyCount=2; isl.contactCount=0;
    isl.posC[0]=v2(0,0); isl.posA[0]=0.0f; isl.vel[0]=v2(0,0); isl.velW[0]=0.0f;
    isl.posC[1]=v2(bobPos.x,bobPos.y); isl.posA[1]=0.0f; isl.vel[1]=v2(0,0); isl.velW[1]=0.0f;
    GBDistanceJoint j;
    j.indexA=0; j.indexB=1; j.localAnchorA=v2(0,0); j.localAnchorB=v2(0,0);
    j.invMassA=0.0f; j.invMassB=BOB_INVM; j.invIA=0.0f; j.invIB=BOB_INVI;
    j.length=1.5f; j.frequencyHz=freq; j.dampingRatio=damp; j.impulse=0.0f;

    for (int sub=0; sub<NSUB; ++sub){
        // ---- reference ----
        rB[1].v = rB[1].v + h*rv2(0.0f, GRAV);
        rInit();
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) rSolveVel();
        for (int i=0;i<2;++i){ rB[i].c=rB[i].c+h*rB[i].v; rB[i].a += h*rB[i].w; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (rSolvePos()) break; }
        // ---- subject ----
        isl.vel[1] = isl.vel[1] + h*v2(0.0f, GRAV);
        gbDistanceInitVelocity(j, isl);
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) gbDistanceSolveVelocity(j, isl);
        for (int i=0;i<2;++i){ isl.posC[i]=isl.posC[i]+h*isl.vel[i]; isl.posA[i]=isl.posA[i]+h*isl.velW[i]; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (gbDistanceSolvePosition(j, isl)) break; }
        // ---- compare ----
        chk("vB.x", sub, rB[1].v.x, isl.vel[1].x);
        chk("vB.y", sub, rB[1].v.y, isl.vel[1].y);
        chk("wB",   sub, rB[1].w,   isl.velW[1]);
        chk("cB.x", sub, rB[1].c.x, isl.posC[1].x);
        chk("cB.y", sub, rB[1].c.y, isl.posC[1].y);
        chk("aB",   sub, rB[1].a,   isl.posA[1]);
        chk("imp",  sub, rJ.impulse, j.impulse);
    }
    printf("  %-12s %d substeps, %s\n", name, NSUB, gFails ? "DIVERGED" : "matched");
}

int main(){
    printf("Distance joint micro-test: gb_distance_joint vs Box2D 2.3.0 b2DistanceJoint\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    runCase("rigid rod", 0.0f, 0.0f, 240);
    runCase("soft spring", 4.0f, 0.5f, 240);

    if (!gFails){
        printf("PASS gb_distance_joint: 0 ULP (rigid rod + soft spring), maxUlp=%ld\n", gMax);
        return 0;
    }
    printf("FAIL gb_distance_joint: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
