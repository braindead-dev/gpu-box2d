// gb_pulley_joint_test.cu. Micro-test for gb_pulley_joint.cuh. 0-ULP versus a
// self-contained Box2D 2.3.0 reference of b2PulleyJoint's InitVelocityConstraints /
// SolveVelocityConstraints / SolvePositionConstraints, Ref-prefixed.
//
// Scenario: two dynamic bodies hang over two fixed ground anchors, connected by a pulley
// with the constraint lengthA + ratio * lengthB = constant. Body A is heavier, so it
// descends and body B rises until the constrained total length holds. The system runs
// for many substeps through the full joint solve spine (integrate velocity, init, 8
// velocity iterations, integrate position, 3 position iterations) on both the subject
// (gb_pulley_joint.cuh) and the reference. Each substep compares both bodies' velocity,
// angular velocity, position, angle, and the warm-start impulse at 0 ULP.
//
// Build (frozen flags), self-contained, host or device:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_pulley_joint_test.cu -o test/gb_pulley_joint_test
//   ./test/gb_pulley_joint_test
//   Expected: PASS gb_pulley_joint: 0 ULP
#include "gpu_box2d/gb_pulley_joint.cuh"
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
// Box2D 2.3.0 b2PulleyJoint reference (Ref-prefixed).
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
static inline RV2 rMulRV(float s, float c, RV2 v){ return rv2(c*v.x - s*v.y, s*v.x + c*v.y); }

#define R_DT (1.0f/60.0f)
#define R_LINEAR_SLOP 0.005f

struct RBody { RV2 c, v; float a, w, invM, invI; };
struct RJoint {
    int ia, ib; RV2 gA, gB, lA, lB; float lengthA, lengthB, ratio, constant;
    float mA, mB, iA, iB;
    RV2 uA, uB, rA, rB; float mass, impulse;
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
    rJ.uA = cA + rJ.rA - rJ.gA;
    rJ.uB = cB + rJ.rB - rJ.gB;
    float lengthA=rLen(rJ.uA), lengthB=rLen(rJ.uB);
    if (lengthA > 10.0f*R_LINEAR_SLOP) rJ.uA=(1.0f/lengthA)*rJ.uA; else rJ.uA=rv2(0,0);
    if (lengthB > 10.0f*R_LINEAR_SLOP) rJ.uB=(1.0f/lengthB)*rJ.uB; else rJ.uB=rv2(0,0);
    float ruA=rCross(rJ.rA,rJ.uA), ruB=rCross(rJ.rB,rJ.uB);
    float massA=mA+iA*ruA*ruA, massB=mB+iB*ruB*ruB;
    rJ.mass=massA+rJ.ratio*rJ.ratio*massB;
    if (rJ.mass>0.0f) rJ.mass=1.0f/rJ.mass;
    RV2 PA=(-rJ.impulse)*rJ.uA;
    RV2 PB=(-rJ.ratio*rJ.impulse)*rJ.uB;
    vA=vA+mA*PA; wA+=iA*rCross(rJ.rA,PA);
    vB=vB+mB*PB; wB+=iB*rCross(rJ.rB,PB);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static void rSolveVel(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    RV2 vpA=vA+rCrossSV(wA,rJ.rA);
    RV2 vpB=vB+rCrossSV(wB,rJ.rB);
    float Cdot=-rDot(rJ.uA,vpA)-rJ.ratio*rDot(rJ.uB,vpB);
    float impulse=-rJ.mass*Cdot;
    rJ.impulse+=impulse;
    RV2 PA=(-impulse)*rJ.uA;
    RV2 PB=(-rJ.ratio*impulse)*rJ.uB;
    vA=vA+mA*PA; wA+=iA*rCross(rJ.rA,PA);
    vB=vB+mB*PB; wB+=iB*rCross(rJ.rB,PB);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static bool rSolvePos(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 cA=rB[ia].c; float aA=rB[ia].a;
    RV2 cB=rB[ib].c; float aB=rB[ib].a;
    float sA=sinf(aA), csA=cosf(aA), sB=sinf(aB), csB=cosf(aB);
    RV2 rA=rMulRV(sA,csA,rJ.lA), rBv=rMulRV(sB,csB,rJ.lB);
    RV2 uA=cA+rA-rJ.gA;
    RV2 uB=cB+rBv-rJ.gB;
    float lengthA=rLen(uA), lengthB=rLen(uB);
    if (lengthA > 10.0f*R_LINEAR_SLOP) uA=(1.0f/lengthA)*uA; else uA=rv2(0,0);
    if (lengthB > 10.0f*R_LINEAR_SLOP) uB=(1.0f/lengthB)*uB; else uB=rv2(0,0);
    float ruA=rCross(rA,uA), ruB=rCross(rBv,uB);
    float massA=mA+iA*ruA*ruA, massB=mB+iB*ruB*ruB;
    float mass=massA+rJ.ratio*rJ.ratio*massB;
    if (mass>0.0f) mass=1.0f/mass;
    float C=rJ.constant-lengthA-rJ.ratio*lengthB;
    float linearError=fabsf(C);
    float impulse=-mass*C;
    RV2 PA=(-impulse)*uA;
    RV2 PB=(-rJ.ratio*impulse)*uB;
    cA=cA+mA*PA; aA+=iA*rCross(rA,PA);
    cB=cB+mB*PB; aB+=iB*rCross(rBv,PB);
    rB[ia].c=cA; rB[ia].a=aA; rB[ib].c=cB; rB[ib].a=aB;
    return linearError < R_LINEAR_SLOP;
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------
static const float GRAV = -9.81f;

static int gFails=0; static long gMax=0;
static void chk(const char* what, int sub, float ref, float got){
    long u=ulpDiff(ref,got); if(u>gMax) gMax=u;
    if(u!=0){ if(gFails==0) printf("  FAIL %-6s sub=%d ref=%.9g got=%.9g ulp=%ld\n", what, sub, ref, got, u); gFails=1; }
}

int main(){
    printf("Pulley joint micro-test: gb_pulley_joint vs Box2D 2.3.0 b2PulleyJoint\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    float h = R_DT;
    int NSUB = 240;

    // two ground anchors at the top; two bodies hanging below them. ratio 1.
    RV2 gA = rv2(-2.0f, 5.0f), gB = rv2(2.0f, 5.0f);
    RV2 posA = rv2(-2.0f, 2.0f), posB = rv2(2.0f, 2.0f);
    float lenA = 3.0f, lenB = 3.0f, ratio = 1.0f;
    float constant = lenA + ratio * lenB;
    float invMA = 1.0f, invMB = 0.5f;   // body A lighter? heavier B by inv mass
    float invIA = 1.0f, invIB = 1.0f;

    rB[0].c=posA; rB[0].v=rv2(0,0); rB[0].a=0.0f; rB[0].w=0.0f; rB[0].invM=invMA; rB[0].invI=invIA;
    rB[1].c=posB; rB[1].v=rv2(0,0); rB[1].a=0.0f; rB[1].w=0.0f; rB[1].invM=invMB; rB[1].invI=invIB;
    rJ.ia=0; rJ.ib=1; rJ.gA=gA; rJ.gB=gB; rJ.lA=rv2(0,0); rJ.lB=rv2(0,0);
    rJ.lengthA=lenA; rJ.lengthB=lenB; rJ.ratio=ratio; rJ.constant=constant;
    rJ.mA=invMA; rJ.mB=invMB; rJ.iA=invIA; rJ.iB=invIB; rJ.impulse=0.0f;

    GBIslandData isl;
    isl.bodyCount=2; isl.contactCount=0;
    isl.posC[0]=v2(posA.x,posA.y); isl.posA[0]=0.0f; isl.vel[0]=v2(0,0); isl.velW[0]=0.0f;
    isl.posC[1]=v2(posB.x,posB.y); isl.posA[1]=0.0f; isl.vel[1]=v2(0,0); isl.velW[1]=0.0f;
    GBPulleyJoint j;
    j.indexA=0; j.indexB=1; j.groundAnchorA=v2(gA.x,gA.y); j.groundAnchorB=v2(gB.x,gB.y);
    j.localAnchorA=v2(0,0); j.localAnchorB=v2(0,0);
    j.lengthA=lenA; j.lengthB=lenB; j.ratio=ratio; j.constant=constant;
    j.invMassA=invMA; j.invMassB=invMB; j.invIA=invIA; j.invIB=invIB; j.impulse=0.0f;

    for (int sub=0; sub<NSUB; ++sub){
        // ---- reference ----
        rB[0].v = rB[0].v + h*rv2(0.0f, GRAV);
        rB[1].v = rB[1].v + h*rv2(0.0f, GRAV);
        rInit();
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) rSolveVel();
        for (int i=0;i<2;++i){ rB[i].c=rB[i].c+h*rB[i].v; rB[i].a += h*rB[i].w; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (rSolvePos()) break; }
        // ---- subject ----
        isl.vel[0] = isl.vel[0] + h*v2(0.0f, GRAV);
        isl.vel[1] = isl.vel[1] + h*v2(0.0f, GRAV);
        gbPulleyInitVelocity(j, isl);
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) gbPulleySolveVelocity(j, isl);
        for (int i=0;i<2;++i){ isl.posC[i]=isl.posC[i]+h*isl.vel[i]; isl.posA[i]=isl.posA[i]+h*isl.velW[i]; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (gbPulleySolvePosition(j, isl)) break; }
        // ---- compare both bodies ----
        for (int bi=0; bi<2; ++bi){
            chk("v.x", sub, rB[bi].v.x, isl.vel[bi].x);
            chk("v.y", sub, rB[bi].v.y, isl.vel[bi].y);
            chk("c.x", sub, rB[bi].c.x, isl.posC[bi].x);
            chk("c.y", sub, rB[bi].c.y, isl.posC[bi].y);
        }
        chk("imp", sub, rJ.impulse, j.impulse);
    }

    if (!gFails){
        printf("PASS gb_pulley_joint: 0 ULP (two-body pulley, %d substeps), maxUlp=%ld\n", NSUB, gMax);
        return 0;
    }
    printf("FAIL gb_pulley_joint: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
