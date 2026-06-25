// gb_gear_joint_test.cu. Micro-test for gb_gear_joint.cuh. 0-ULP versus a self-contained
// Box2D 2.3.0 reference of b2GearJoint's InitVelocityConstraints /
// SolveVelocityConstraints / SolvePositionConstraints, Ref-prefixed.
//
// Scenario: a meshed-gear pair. Two wheels (bodies A and B) each pinned by a revolute
// joint to a shared static ground (body C == body D == the ground), coupled by a gear
// joint with a ratio. Body A is given an initial spin; the gear transfers it to body B
// at the ratio. The full solve spine runs the two revolute joints and the gear together
// in Box2D's interleave (init all, then each velocity iteration solves the joints in
// order, then each position iteration), on both the subject and the reference. Each
// substep compares both wheels' angular velocity and angle, and the gear impulse, at
// 0 ULP.
//
// Coupling both revolute joints and the gear in the same spine is what a real island
// does, so the test exercises the gear in its realistic setting.
//
// Build (frozen flags), self-contained, host or device:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_gear_joint_test.cu -o test/gb_gear_joint_test
//   ./test/gb_gear_joint_test
//   Expected: PASS gb_gear_joint: 0 ULP
#include "gpu_box2d/gb_gear_joint.cuh"
#include "gpu_box2d/gb_joint.cuh"     // gbRevoluteInitVelocity / SolveVelocity / SolvePosition (point-to-point)
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
// Box2D 2.3.0 references (Ref-prefixed): the point-to-point revolute and the gear.
// ---------------------------------------------------------------------------
struct RV2 { float x, y; };
static inline RV2 rv2(float x, float y){ RV2 r; r.x=x; r.y=y; return r; }
static inline RV2 operator+(RV2 a, RV2 b){ return rv2(a.x+b.x, a.y+b.y); }
static inline RV2 operator-(RV2 a, RV2 b){ return rv2(a.x-b.x, a.y-b.y); }
static inline RV2 operator*(float s, RV2 a){ return rv2(s*a.x, s*a.y); }
static inline float rDot(RV2 a, RV2 b){ return a.x*b.x + a.y*b.y; }
static inline float rCross(RV2 a, RV2 b){ return a.x*b.y - a.y*b.x; }
static inline RV2 rCrossSV(float s, RV2 a){ return rv2(-s*a.y, s*a.x); }
static inline RV2 rMulRV(float s, float c, RV2 v){ return rv2(c*v.x - s*v.y, s*v.x + c*v.y); }

#define R_DT (1.0f/60.0f)
#define R_LINEAR_SLOP 0.005f

struct RBody { RV2 c, v; float a, w, invM, invI; };
static RBody rB[3];   // 0 = ground (C and D), 1 = wheel A, 2 = wheel B

// --- point-to-point revolute reference (matches gb_joint.cuh) ---
struct RMat22 { RV2 ex, ey; };
static RV2 rSolve22(const RMat22& K, RV2 b){
    float a11=K.ex.x,a12=K.ey.x,a21=K.ex.y,a22=K.ey.y;
    float det=a11*a22-a12*a21; if(det!=0.0f) det=1.0f/det;
    return rv2(det*(a22*b.x-a12*b.y), det*(a11*b.y-a21*b.x));
}
struct RRevolute { int ia, ib; RV2 lA, lB; float mA, mB, iA, iB; RV2 rA, rB; RMat22 mass; RV2 impulse; };
static void rRevInit(RRevolute& j){
    int ia=j.ia, ib=j.ib; float mA=j.mA, mB=j.mB, iA=j.iA, iB=j.iB;
    float aA=rB[ia].a, aB=rB[ib].a;
    RV2 vA=rB[ia].v; float wA=rB[ia].w; RV2 vB=rB[ib].v; float wB=rB[ib].w;
    float sA=sinf(aA), cA=cosf(aA), sB=sinf(aB), cB=cosf(aB);
    j.rA=rMulRV(sA,cA,j.lA); j.rB=rMulRV(sB,cB,j.lB);
    j.mass.ex.x=mA+mB+iA*j.rA.y*j.rA.y+iB*j.rB.y*j.rB.y;
    j.mass.ey.x=-iA*j.rA.x*j.rA.y-iB*j.rB.x*j.rB.y;
    j.mass.ex.y=j.mass.ey.x;
    j.mass.ey.y=mA+mB+iA*j.rA.x*j.rA.x+iB*j.rB.x*j.rB.x;
    RV2 P=j.impulse;
    vA=vA-mA*P; wA-=iA*rCross(j.rA,P); vB=vB+mB*P; wB+=iB*rCross(j.rB,P);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static void rRevSolveVel(RRevolute& j){
    int ia=j.ia, ib=j.ib; float mA=j.mA, mB=j.mB, iA=j.iA, iB=j.iB;
    RV2 vA=rB[ia].v; float wA=rB[ia].w; RV2 vB=rB[ib].v; float wB=rB[ib].w;
    RV2 Cdot=vB+rCrossSV(wB,j.rB)-vA-rCrossSV(wA,j.rA);
    RV2 impulse=rSolve22(j.mass, rv2(-Cdot.x,-Cdot.y));
    j.impulse=j.impulse+impulse;
    vA=vA-mA*impulse; wA-=iA*rCross(j.rA,impulse);
    vB=vB+mB*impulse; wB+=iB*rCross(j.rB,impulse);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static bool rRevSolvePos(RRevolute& j){
    int ia=j.ia, ib=j.ib; float mA=j.mA, mB=j.mB, iA=j.iA, iB=j.iB;
    RV2 cA=rB[ia].c; float aA=rB[ia].a; RV2 cB=rB[ib].c; float aB=rB[ib].a;
    float sA=sinf(aA), cA2=cosf(aA), sB=sinf(aB), cB2=cosf(aB);
    RV2 rA=rMulRV(sA,cA2,j.lA), rBv=rMulRV(sB,cB2,j.lB);
    RV2 C=cB+rBv-cA-rA;
    float posErr=sqrtf(C.x*C.x+C.y*C.y);
    RMat22 K;
    K.ex.x=mA+mB+iA*rA.y*rA.y+iB*rBv.y*rBv.y;
    K.ex.y=-iA*rA.x*rA.y-iB*rBv.x*rBv.y;
    K.ey.x=K.ex.y;
    K.ey.y=mA+mB+iA*rA.x*rA.x+iB*rBv.x*rBv.x;
    RV2 impulse=-1.0f*rSolve22(K,C);
    cA=cA-mA*impulse; aA-=iA*rCross(rA,impulse);
    cB=cB+mB*impulse; aB+=iB*rCross(rBv,impulse);
    rB[ia].c=cA; rB[ia].a=aA; rB[ib].c=cB; rB[ib].a=aB;
    return posErr <= R_LINEAR_SLOP;
}

// --- gear reference (revolute-revolute coupling) ---
struct RGear {
    int ia, ib, ic, id; float refA, refB; float ratio, constant;
    float mA, mB, mC, mD, iA, iB, iC, iD;
    RV2 JvAC, JvBD; float JwA, JwB, JwC, JwD; float mass; float impulse;
};
static void rGearInit(RGear& j){
    int ia=j.ia, ib=j.ib, ic=j.ic, id=j.id;
    float iA=j.iA, iB=j.iB, iC=j.iC, iD=j.iD;
    RV2 vA=rB[ia].v; float wA=rB[ia].w; RV2 vB=rB[ib].v; float wB=rB[ib].w;
    RV2 vC=rB[ic].v; float wC=rB[ic].w; RV2 vD=rB[id].v; float wD=rB[id].w;
    j.mass=0.0f;
    j.JvAC=rv2(0,0); j.JwA=1.0f; j.JwC=1.0f; j.mass+=iA+iC;
    j.JvBD=rv2(0,0); j.JwB=j.ratio; j.JwD=j.ratio; j.mass+=j.ratio*j.ratio*(iB+iD);
    j.mass = j.mass>0.0f ? 1.0f/j.mass : 0.0f;
    float mA=j.mA, mB=j.mB, mC=j.mC, mD=j.mD;
    vA=vA+(mA*j.impulse)*j.JvAC; wA+=iA*j.impulse*j.JwA;
    vB=vB+(mB*j.impulse)*j.JvBD; wB+=iB*j.impulse*j.JwB;
    vC=vC-(mC*j.impulse)*j.JvAC; wC-=iC*j.impulse*j.JwC;
    vD=vD-(mD*j.impulse)*j.JvBD; wD-=iD*j.impulse*j.JwD;
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
    rB[ic].v=vC; rB[ic].w=wC; rB[id].v=vD; rB[id].w=wD;
}
static void rGearSolveVel(RGear& j){
    int ia=j.ia, ib=j.ib, ic=j.ic, id=j.id;
    float mA=j.mA, mB=j.mB, mC=j.mC, mD=j.mD, iA=j.iA, iB=j.iB, iC=j.iC, iD=j.iD;
    RV2 vA=rB[ia].v; float wA=rB[ia].w; RV2 vB=rB[ib].v; float wB=rB[ib].w;
    RV2 vC=rB[ic].v; float wC=rB[ic].w; RV2 vD=rB[id].v; float wD=rB[id].w;
    float Cdot=rDot(j.JvAC,vA-vC)+rDot(j.JvBD,vB-vD);
    Cdot+=(j.JwA*wA-j.JwC*wC)+(j.JwB*wB-j.JwD*wD);
    float impulse=-j.mass*Cdot;
    j.impulse+=impulse;
    vA=vA+(mA*impulse)*j.JvAC; wA+=iA*impulse*j.JwA;
    vB=vB+(mB*impulse)*j.JvBD; wB+=iB*impulse*j.JwB;
    vC=vC-(mC*impulse)*j.JvAC; wC-=iC*impulse*j.JwC;
    vD=vD-(mD*impulse)*j.JvBD; wD-=iD*impulse*j.JwD;
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
    rB[ic].v=vC; rB[ic].w=wC; rB[id].v=vD; rB[id].w=wD;
}
static bool rGearSolvePos(RGear& j){
    int ia=j.ia, ib=j.ib, ic=j.ic, id=j.id;
    float mA=j.mA, mB=j.mB, mC=j.mC, mD=j.mD, iA=j.iA, iB=j.iB, iC=j.iC, iD=j.iD;
    RV2 cA=rB[ia].c; float aA=rB[ia].a; RV2 cB=rB[ib].c; float aB=rB[ib].a;
    RV2 cC=rB[ic].c; float aC=rB[ic].a; RV2 cD=rB[id].c; float aD=rB[id].a;
    float linearError=0.0f, mass=0.0f;
    RV2 JvAC=rv2(0,0), JvBD=rv2(0,0); float JwA=1.0f, JwC=1.0f, JwB=j.ratio, JwD=j.ratio;
    mass += iA+iC; mass += j.ratio*j.ratio*(iB+iD);
    float coordinateA = aA - aC - j.refA;
    float coordinateB = aB - aD - j.refB;
    float C=(coordinateA + j.ratio*coordinateB) - j.constant;
    float impulse=0.0f; if (mass>0.0f) impulse=-C/mass;
    cA=cA+(mA*impulse)*JvAC; aA+=iA*impulse*JwA;
    cB=cB+(mB*impulse)*JvBD; aB+=iB*impulse*JwB;
    cC=cC-(mC*impulse)*JvAC; aC-=iC*impulse*JwC;
    cD=cD-(mD*impulse)*JvBD; aD-=iD*impulse*JwD;
    rB[ia].c=cA; rB[ia].a=aA; rB[ib].c=cB; rB[ib].a=aB;
    rB[ic].c=cC; rB[ic].a=aC; rB[id].c=cD; rB[id].a=aD;
    (void)mass;
    return linearError < R_LINEAR_SLOP;
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------
static int gFails=0; static long gMax=0;
static void chk(const char* what, int sub, float ref, float got){
    long u=ulpDiff(ref,got); if(u>gMax) gMax=u;
    if(u!=0){ if(gFails==0) printf("  FAIL %-6s sub=%d ref=%.9g got=%.9g ulp=%ld\n", what, sub, ref, got, u); gFails=1; }
}

int main(){
    printf("Gear joint micro-test: gb_gear_joint vs Box2D 2.3.0 b2GearJoint (revolute-revolute)\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    float h = R_DT;
    int NSUB = 240;
    float ratio = 2.0f;

    // ground (slot 0, static); wheel A (slot 1) at (-1,0); wheel B (slot 2) at (1,0).
    // each wheel pinned to the ground at its own center via a revolute joint; wheel A
    // gets an initial spin, which the gear transfers to wheel B at the ratio.
    rB[0]=(RBody){ rv2(0,0), rv2(0,0), 0.0f, 0.0f, 0.0f, 0.0f };
    rB[1]=(RBody){ rv2(-1,0), rv2(0,0), 0.0f, 5.0f, 1.0f, 1.0f };
    rB[2]=(RBody){ rv2(1,0),  rv2(0,0), 0.0f, 0.0f, 1.0f, 1.0f };

    // reference joints
    RRevolute rev1 = { 0, 1, rv2(-1,0), rv2(0,0), 0.0f, 1.0f, 0.0f, 1.0f, {}, {}, rv2(0,0) };
    RRevolute rev2 = { 0, 2, rv2(1,0),  rv2(0,0), 0.0f, 1.0f, 0.0f, 1.0f, {}, {}, rv2(0,0) };
    RGear gear; gear.ia=1; gear.ib=2; gear.ic=0; gear.id=0; gear.refA=0.0f; gear.refB=0.0f;
    gear.ratio=ratio; gear.constant=0.0f;
    gear.mA=1.0f; gear.mB=1.0f; gear.mC=0.0f; gear.mD=0.0f;
    gear.iA=1.0f; gear.iB=1.0f; gear.iC=0.0f; gear.iD=0.0f; gear.impulse=0.0f;

    // subject island + joints
    GBIslandData isl; isl.bodyCount=3; isl.contactCount=0;
    for (int i=0;i<3;++i){ isl.posC[i]=v2(rB[i].c.x,rB[i].c.y); isl.posA[i]=rB[i].a; isl.vel[i]=v2(rB[i].v.x,rB[i].v.y); isl.velW[i]=rB[i].w; }
    GBRevoluteJoint grev1; grev1.indexA=0; grev1.indexB=1; grev1.localAnchorA=v2(-1,0); grev1.localAnchorB=v2(0,0);
    grev1.invMassA=0; grev1.invMassB=1; grev1.invIA=0; grev1.invIB=1; grev1.impulse=v2(0,0);
    GBRevoluteJoint grev2; grev2.indexA=0; grev2.indexB=2; grev2.localAnchorA=v2(1,0); grev2.localAnchorB=v2(0,0);
    grev2.invMassA=0; grev2.invMassB=1; grev2.invIA=0; grev2.invIB=1; grev2.impulse=v2(0,0);
    GBGearJoint ggear; ggear.indexA=1; ggear.indexB=2; ggear.indexC=0; ggear.indexD=0;
    ggear.typeA=GB_GEAR_REVOLUTE; ggear.typeB=GB_GEAR_REVOLUTE;
    ggear.referenceAngleA=0.0f; ggear.referenceAngleB=0.0f; ggear.ratio=ratio; ggear.constant=0.0f;
    ggear.invMassA=1; ggear.invMassB=1; ggear.invMassC=0; ggear.invMassD=0;
    ggear.invIA=1; ggear.invIB=1; ggear.invIC=0; ggear.invID=0; ggear.impulse=0.0f;

    for (int sub=0; sub<NSUB; ++sub){
        // no gravity here; the wheels only rotate, coupled by the gear.
        // ---- reference: init all, iterate velocity (joints in order), iterate position
        rRevInit(rev1); rRevInit(rev2); rGearInit(gear);
        for (int it=0; it<GB_VELOCITY_ITERS; ++it){ rRevSolveVel(rev1); rRevSolveVel(rev2); rGearSolveVel(gear); }
        for (int i=0;i<3;++i){ rB[i].c=rB[i].c+h*rB[i].v; rB[i].a += h*rB[i].w; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){
            bool ok = true;
            ok = rRevSolvePos(rev1) && ok; ok = rRevSolvePos(rev2) && ok; ok = rGearSolvePos(gear) && ok;
            if (ok) break;
        }
        // ---- subject
        gbRevoluteInitVelocity(grev1, isl); gbRevoluteInitVelocity(grev2, isl); gbGearInitVelocity(ggear, isl);
        for (int it=0; it<GB_VELOCITY_ITERS; ++it){ gbRevoluteSolveVelocity(grev1, isl); gbRevoluteSolveVelocity(grev2, isl); gbGearSolveVelocity(ggear, isl); }
        for (int i=0;i<3;++i){ isl.posC[i]=isl.posC[i]+h*isl.vel[i]; isl.posA[i]=isl.posA[i]+h*isl.velW[i]; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){
            bool ok = true;
            ok = gbRevoluteSolvePosition(grev1, isl) && ok;
            ok = gbRevoluteSolvePosition(grev2, isl) && ok;
            ok = gbGearSolvePosition(ggear, isl) && ok;
            if (ok) break;
        }
        // ---- compare both wheels and the gear impulse
        chk("wA", sub, rB[1].w, isl.velW[1]);
        chk("wB", sub, rB[2].w, isl.velW[2]);
        chk("aA", sub, rB[1].a, isl.posA[1]);
        chk("aB", sub, rB[2].a, isl.posA[2]);
        chk("gimp", sub, gear.impulse, ggear.impulse);
    }

    if (!gFails){
        printf("PASS gb_gear_joint: 0 ULP (revolute-revolute gear, %d substeps), maxUlp=%ld\n", NSUB, gMax);
        return 0;
    }
    printf("FAIL gb_gear_joint: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
