// gb_joint_test.cu. Micro-test for gb_joint.cuh (the point-to-point revolute joint).
// 0-ULP versus a self-contained Box2D 2.3.0 reference of b2RevoluteJoint's
// point-to-point InitVelocityConstraints / SolveVelocityConstraints /
// SolvePositionConstraints.
//
// Scenario: a pendulum. Body A is a static pivot at the world origin; body B is a
// dynamic bob hanging from a revolute joint at the origin. The bob swings under
// gravity over many substeps. Each substep runs the full joint solve spine
// (integrate velocity, init, warm-start, 8 velocity iterations, integrate position,
// 3 position iterations) on both the subject (gb_joint.cuh) and the reference, then
// compares the bob's velocity, angular velocity, position, angle, and the warm-start
// impulse at 0 ULP.
//
// Build (frozen flags), self-contained:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_joint_test.cu -o test/gb_joint_test
//   ./test/gb_joint_test
//   Expected: PASS gb_joint: 0 ULP (revolute pendulum)
#include "gpu_box2d/gb_joint.cuh"
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
// Box2D 2.3.0 b2RevoluteJoint reference (point-to-point), Ref-prefixed.
// ---------------------------------------------------------------------------
struct RV2 { float x, y; };
static inline RV2 rv2(float x, float y){ RV2 r; r.x=x; r.y=y; return r; }
static inline RV2 operator+(RV2 a, RV2 b){ return rv2(a.x+b.x, a.y+b.y); }
static inline RV2 operator-(RV2 a, RV2 b){ return rv2(a.x-b.x, a.y-b.y); }
static inline RV2 operator*(float s, RV2 a){ return rv2(s*a.x, s*a.y); }
static inline RV2 operator-(RV2 a){ return rv2(-a.x, -a.y); }
static inline float rDot(RV2 a, RV2 b){ return a.x*b.x + a.y*b.y; }
static inline float rCross(RV2 a, RV2 b){ return a.x*b.y - a.y*b.x; }
static inline RV2 rCrossSV(float s, RV2 a){ return rv2(-s*a.y, s*a.x); }
static inline float rLen(RV2 v){ return sqrtf(v.x*v.x + v.y*v.y); }
static inline RV2 rMulRV(float s, float c, RV2 v){ return rv2(c*v.x - s*v.y, s*v.x + c*v.y); }

struct RMat22 { RV2 ex, ey; };
static RV2 rSolve(const RMat22& K, RV2 b){
    float a11=K.ex.x, a12=K.ey.x, a21=K.ex.y, a22=K.ey.y;
    float det=a11*a22-a12*a21; if(det!=0.0f) det=1.0f/det;
    return rv2(det*(a22*b.x-a12*b.y), det*(a11*b.y-a21*b.x));
}

struct RBody { RV2 c, v; float a, w, invM, invI; };
struct RJoint {
    int ia, ib; RV2 lA, lB; float mA, mB, iA, iB;
    RV2 rA, rB; RMat22 mass; RV2 impulse;
};
static RBody rB[2];
static RJoint rJ;

static void rInit(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    float aA=rB[ia].a, aB=rB[ib].a;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    float sA=sinf(aA), cA=cosf(aA), sB=sinf(aB), cB=cosf(aB);
    rJ.rA=rMulRV(sA,cA,rJ.lA); rJ.rB=rMulRV(sB,cB,rJ.lB);
    RMat22& K=rJ.mass;
    K.ex.x=mA+mB+iA*rJ.rA.y*rJ.rA.y+iB*rJ.rB.y*rJ.rB.y;
    K.ey.x=-iA*rJ.rA.x*rJ.rA.y-iB*rJ.rB.x*rJ.rB.y;
    K.ex.y=K.ey.x;
    K.ey.y=mA+mB+iA*rJ.rA.x*rJ.rA.x+iB*rJ.rB.x*rJ.rB.x;
    RV2 P=rJ.impulse;
    vA=vA-mA*P; wA-=iA*rCross(rJ.rA,P);
    vB=vB+mB*P; wB+=iB*rCross(rJ.rB,P);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static void rSolveVel(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    RV2 Cdot=vB+rCrossSV(wB,rJ.rB)-vA-rCrossSV(wA,rJ.rA);
    RV2 impulse=rSolve(rJ.mass,-Cdot);
    rJ.impulse=rJ.impulse+impulse;
    vA=vA-mA*impulse; wA-=iA*rCross(rJ.rA,impulse);
    vB=vB+mB*impulse; wB+=iB*rCross(rJ.rB,impulse);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static bool rSolvePos(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 cA=rB[ia].c; float aA=rB[ia].a;
    RV2 cB=rB[ib].c; float aB=rB[ib].a;
    float sA=sinf(aA), cAng=cosf(aA), sB=sinf(aB), cB2=cosf(aB);
    RV2 rA=rMulRV(sA,cAng,rJ.lA), rBv=rMulRV(sB,cB2,rJ.lB);
    RV2 C=cB+rBv-cA-rA;
    float posErr=rLen(C);
    RMat22 K;
    K.ex.x=mA+mB+iA*rA.y*rA.y+iB*rBv.y*rBv.y;
    K.ex.y=-iA*rA.x*rA.y-iB*rBv.x*rBv.y;
    K.ey.x=K.ex.y;
    K.ey.y=mA+mB+iA*rA.x*rA.x+iB*rBv.x*rBv.x;
    RV2 impulse=-1.0f*rSolve(K,C);
    cA=cA-mA*impulse; aA-=iA*rCross(rA,impulse);
    cB=cB+mB*impulse; aB+=iB*rCross(rBv,impulse);
    rB[ia].c=cA; rB[ia].a=aA; rB[ib].c=cB; rB[ib].a=aB;
    return posErr <= 0.005f;
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
    if(u!=0 && gFails==0){ printf("  FAIL %-6s sub=%d ref=%.9g got=%.9g ulp=%ld\n", what, sub, ref, got, u); gFails=1; }
    else if(u!=0) gFails=1;
}

int main(){
    printf("Revolute joint micro-test: gb_joint vs Box2D 2.3.0 b2RevoluteJoint (point-to-point)\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    float h = GB_DT;
    int NSUB = 240;

    // ---- seed both sides identically ---------------------------------------
    // pivot (static) at origin; bob (dynamic) offset to the right, joint anchor at
    // origin (world). localAnchorA = origin in A's frame = (0,0); localAnchorB is the
    // origin in B's frame = -bobPos.
    RV2 bobPos = rv2(1.5f, 0.0f);
    RV2 lAA = rv2(0.0f, 0.0f);
    RV2 lAB = rv2(-1.5f, 0.0f);

    rB[0].c=rv2(0,0); rB[0].v=rv2(0,0); rB[0].a=0.0f; rB[0].w=0.0f; rB[0].invM=0.0f; rB[0].invI=0.0f;
    rB[1].c=bobPos;   rB[1].v=rv2(0,0); rB[1].a=0.0f; rB[1].w=0.0f; rB[1].invM=BOB_INVM; rB[1].invI=BOB_INVI;
    rJ.ia=0; rJ.ib=1; rJ.lA=lAA; rJ.lB=lAB;
    rJ.mA=0.0f; rJ.mB=BOB_INVM; rJ.iA=0.0f; rJ.iB=BOB_INVI;
    rJ.impulse=rv2(0,0);

    GBIslandData isl;
    isl.bodyCount=2; isl.contactCount=0;
    isl.posC[0]=v2(0,0); isl.posA[0]=0.0f; isl.vel[0]=v2(0,0); isl.velW[0]=0.0f;
    isl.posC[1]=v2(bobPos.x,bobPos.y); isl.posA[1]=0.0f; isl.vel[1]=v2(0,0); isl.velW[1]=0.0f;
    GBRevoluteJoint j;
    j.indexA=0; j.indexB=1; j.localAnchorA=v2(0,0); j.localAnchorB=v2(-1.5f,0.0f);
    j.invMassA=0.0f; j.invMassB=BOB_INVM; j.invIA=0.0f; j.invIB=BOB_INVI;
    j.impulse=v2(0,0);

    for (int sub=0; sub<NSUB; ++sub){
        // ---- reference substep ---
        // integrate velocity (gravity on the dynamic bob)
        rB[1].v = rB[1].v + h*rv2(0.0f, GRAV);
        rInit();
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) rSolveVel();
        // integrate position
        for (int i=0;i<2;++i){ rB[i].c=rB[i].c+h*rB[i].v; rB[i].a += h*rB[i].w; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (rSolvePos()) break; }

        // ---- subject substep ---
        isl.vel[1] = isl.vel[1] + h*v2(0.0f, GRAV);
        gbRevoluteInitVelocity(j, isl);
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) gbRevoluteSolveVelocity(j, isl);
        for (int i=0;i<2;++i){ isl.posC[i]=isl.posC[i]+h*isl.vel[i]; isl.posA[i]=isl.posA[i]+h*isl.velW[i]; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (gbRevoluteSolvePosition(j, isl)) break; }

        // ---- compare ---
        chk("vB.x", sub, rB[1].v.x, isl.vel[1].x);
        chk("vB.y", sub, rB[1].v.y, isl.vel[1].y);
        chk("wB",   sub, rB[1].w,   isl.velW[1]);
        chk("cB.x", sub, rB[1].c.x, isl.posC[1].x);
        chk("cB.y", sub, rB[1].c.y, isl.posC[1].y);
        chk("aB",   sub, rB[1].a,   isl.posA[1]);
        chk("impx", sub, rJ.impulse.x, j.impulse.x);
        chk("impy", sub, rJ.impulse.y, j.impulse.y);
    }

    if (!gFails){
        printf("PASS gb_joint: 0 ULP (revolute pendulum, %d substeps), maxUlp=%ld\n", NSUB, gMax);
        return 0;
    }
    printf("FAIL gb_joint: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
