// gb_prismatic_joint_test.cu. Micro-test for gb_prismatic_joint.cuh. 0-ULP versus a
// self-contained Box2D 2.3.0 reference of b2PrismaticJoint's InitVelocityConstraints /
// SolveVelocityConstraints / SolvePositionConstraints, Ref-prefixed.
//
// Three scenarios, each run for many substeps through the full joint solve spine
// (integrate velocity, init, 8 velocity iterations, integrate position, 3 position
// iterations) on both the subject (gb_prismatic_joint.cuh) and the reference:
//   1. Free slider. A body slides on a vertical axis under gravity, no limit, no motor
//      (the 2x2 perpendicular-plus-angular path).
//   2. Limited slider. The same body with a translation limit, so it hits the lower
//      stop and the limit row activates (the 3x3 path).
//   3. Motorized slider. The same body with a motor driving it along the axis against
//      gravity (the motor row plus the 2x2 block).
// Each substep compares the body velocity, angular velocity, position, angle, the three
// impulse components, and the motor impulse at 0 ULP.
//
// Build (frozen flags), self-contained, host or device:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_prismatic_joint_test.cu -o test/gb_prismatic_joint_test
//   ./test/gb_prismatic_joint_test
//   Expected: PASS gb_prismatic_joint: 0 ULP
#include "gpu_box2d/gb_prismatic_joint.cuh"
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
// Box2D 2.3.0 b2PrismaticJoint reference (Ref-prefixed).
// ---------------------------------------------------------------------------
struct RV2 { float x, y; };
struct RV3 { float x, y, z; };
static inline RV2 rv2(float x, float y){ RV2 r; r.x=x; r.y=y; return r; }
static inline RV3 rv3(float x, float y, float z){ RV3 r; r.x=x; r.y=y; r.z=z; return r; }
static inline RV2 operator+(RV2 a, RV2 b){ return rv2(a.x+b.x, a.y+b.y); }
static inline RV2 operator-(RV2 a, RV2 b){ return rv2(a.x-b.x, a.y-b.y); }
static inline RV2 operator*(float s, RV2 a){ return rv2(s*a.x, s*a.y); }
static inline RV2 operator-(RV2 a){ return rv2(-a.x, -a.y); }
static inline RV3 operator+(RV3 a, RV3 b){ return rv3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline RV3 operator-(RV3 a, RV3 b){ return rv3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline float rDot(RV2 a, RV2 b){ return a.x*b.x + a.y*b.y; }
static inline float rCross(RV2 a, RV2 b){ return a.x*b.y - a.y*b.x; }
static inline RV2 rMulRV(float s, float c, RV2 v){ return rv2(c*v.x - s*v.y, s*v.x + c*v.y); }
static inline float rClamp(float a, float lo, float hi){ return a<lo?lo:(a>hi?hi:a); }
static inline float rMax(float a, float b){ return a>b?a:b; }
static inline float rMin(float a, float b){ return a<b?a:b; }

#define R_PI 3.14159265359f
#define R_DT (1.0f/60.0f)
#define R_LINEAR_SLOP 0.005f
#define R_ANGULAR_SLOP (2.0f/180.0f*R_PI)
#define R_MAX_LINEAR_CORRECTION 0.2f
#define R_INACTIVE 0
#define R_LOWER 1
#define R_UPPER 2
#define R_EQUAL 3

struct RMat33 { RV3 ex, ey, ez; };
static inline float rDot3(RV3 a, RV3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline RV3 rCross3(RV3 a, RV3 b){ return rv3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
static RV3 rSolve33(const RMat33& A, RV3 b){
    float det = rDot3(A.ex, rCross3(A.ey, A.ez));
    if (det != 0.0f) det = 1.0f/det;
    return rv3(det*rDot3(b, rCross3(A.ey, A.ez)), det*rDot3(A.ex, rCross3(b, A.ez)), det*rDot3(A.ex, rCross3(A.ey, b)));
}
static RV2 rSolve22(const RMat33& A, RV2 b){
    float a11=A.ex.x, a12=A.ey.x, a21=A.ex.y, a22=A.ey.y;
    float det=a11*a22-a12*a21; if(det!=0.0f) det=1.0f/det;
    return rv2(det*(a22*b.x-a12*b.y), det*(a11*b.y-a21*b.x));
}
struct RMat22 { RV2 ex, ey; };
static RV2 rSolveMat22(const RMat22& A, RV2 b){
    float a11=A.ex.x, a12=A.ey.x, a21=A.ex.y, a22=A.ey.y;
    float det=a11*a22-a12*a21; if(det!=0.0f) det=1.0f/det;
    return rv2(det*(a22*b.x-a12*b.y), det*(a11*b.y-a21*b.x));
}

struct RBody { RV2 c, v; float a, w, invM, invI; };
struct RJoint {
    int ia, ib; RV2 lA, lB, lXAxis, lYAxis; float refAngle; float mA, mB, iA, iB;
    int enableLimit; float lower, upper; int enableMotor; float maxMotorForce, motorSpeed;
    RV2 axis, perp; float s1, s2, a1, a2; RMat33 K; float motorMass; int limitState;
    RV3 impulse; float motorImpulse;
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
    RV2 rA=rMulRV(sA,csA,rJ.lA), rBv=rMulRV(sB,csB,rJ.lB);
    RV2 d=(cB-cA)+rBv-rA;
    rJ.axis=rMulRV(sA,csA,rJ.lXAxis);
    rJ.a1=rCross(d+rA, rJ.axis); rJ.a2=rCross(rBv, rJ.axis);
    rJ.motorMass=mA+mB+iA*rJ.a1*rJ.a1+iB*rJ.a2*rJ.a2;
    if (rJ.motorMass>0.0f) rJ.motorMass=1.0f/rJ.motorMass;
    rJ.perp=rMulRV(sA,csA,rJ.lYAxis);
    rJ.s1=rCross(d+rA, rJ.perp); rJ.s2=rCross(rBv, rJ.perp);
    float k11=mA+mB+iA*rJ.s1*rJ.s1+iB*rJ.s2*rJ.s2;
    float k12=iA*rJ.s1+iB*rJ.s2;
    float k13=iA*rJ.s1*rJ.a1+iB*rJ.s2*rJ.a2;
    float k22=iA+iB; if (k22==0.0f) k22=1.0f;
    float k23=iA*rJ.a1+iB*rJ.a2;
    float k33=mA+mB+iA*rJ.a1*rJ.a1+iB*rJ.a2*rJ.a2;
    rJ.K.ex=rv3(k11,k12,k13); rJ.K.ey=rv3(k12,k22,k23); rJ.K.ez=rv3(k13,k23,k33);
    if (rJ.enableLimit){
        float jt=rDot(rJ.axis, d);
        if (fabsf(rJ.upper-rJ.lower) < 2.0f*R_LINEAR_SLOP) rJ.limitState=R_EQUAL;
        else if (jt <= rJ.lower){ if (rJ.limitState!=R_LOWER){ rJ.limitState=R_LOWER; rJ.impulse.z=0.0f; } }
        else if (jt >= rJ.upper){ if (rJ.limitState!=R_UPPER){ rJ.limitState=R_UPPER; rJ.impulse.z=0.0f; } }
        else { rJ.limitState=R_INACTIVE; rJ.impulse.z=0.0f; }
    } else { rJ.limitState=R_INACTIVE; rJ.impulse.z=0.0f; }
    if (rJ.enableMotor==0) rJ.motorImpulse=0.0f;
    RV2 P=rJ.impulse.x*rJ.perp+(rJ.motorImpulse+rJ.impulse.z)*rJ.axis;
    float LA=rJ.impulse.x*rJ.s1+rJ.impulse.y+(rJ.motorImpulse+rJ.impulse.z)*rJ.a1;
    float LB=rJ.impulse.x*rJ.s2+rJ.impulse.y+(rJ.motorImpulse+rJ.impulse.z)*rJ.a2;
    vA=vA-mA*P; wA-=iA*LA; vB=vB+mB*P; wB+=iB*LB;
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static void rSolveVel(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    if (rJ.enableMotor && rJ.limitState!=R_EQUAL){
        float Cdot=rDot(rJ.axis, vB-vA)+rJ.a2*wB-rJ.a1*wA;
        float impulse=rJ.motorMass*(rJ.motorSpeed-Cdot);
        float oldImpulse=rJ.motorImpulse;
        float maxImpulse=R_DT*rJ.maxMotorForce;
        rJ.motorImpulse=rClamp(rJ.motorImpulse+impulse, -maxImpulse, maxImpulse);
        impulse=rJ.motorImpulse-oldImpulse;
        RV2 P=impulse*rJ.axis; float LA=impulse*rJ.a1; float LB=impulse*rJ.a2;
        vA=vA-mA*P; wA-=iA*LA; vB=vB+mB*P; wB+=iB*LB;
    }
    RV2 Cdot1;
    Cdot1.x=rDot(rJ.perp, vB-vA)+rJ.s2*wB-rJ.s1*wA;
    Cdot1.y=wB-wA;
    if (rJ.enableLimit && rJ.limitState!=R_INACTIVE){
        float Cdot2=rDot(rJ.axis, vB-vA)+rJ.a2*wB-rJ.a1*wA;
        RV3 Cdot=rv3(Cdot1.x, Cdot1.y, Cdot2);
        RV3 f1=rJ.impulse;
        RV3 df=rSolve33(rJ.K, rv3(-Cdot.x,-Cdot.y,-Cdot.z));
        rJ.impulse=rJ.impulse+df;
        if (rJ.limitState==R_LOWER) rJ.impulse.z=rMax(rJ.impulse.z, 0.0f);
        else if (rJ.limitState==R_UPPER) rJ.impulse.z=rMin(rJ.impulse.z, 0.0f);
        RV2 b=rv2(-Cdot1.x-(rJ.impulse.z-f1.z)*rJ.K.ez.x, -Cdot1.y-(rJ.impulse.z-f1.z)*rJ.K.ez.y);
        RV2 f2r=rSolve22(rJ.K, b);
        f2r.x+=f1.x; f2r.y+=f1.y;
        rJ.impulse.x=f2r.x; rJ.impulse.y=f2r.y;
        df=rJ.impulse-f1;
        RV2 P=df.x*rJ.perp+df.z*rJ.axis;
        float LA=df.x*rJ.s1+df.y+df.z*rJ.a1;
        float LB=df.x*rJ.s2+df.y+df.z*rJ.a2;
        vA=vA-mA*P; wA-=iA*LA; vB=vB+mB*P; wB+=iB*LB;
    } else {
        RV2 df=rSolve22(rJ.K, rv2(-Cdot1.x,-Cdot1.y));
        rJ.impulse.x+=df.x; rJ.impulse.y+=df.y;
        RV2 P=df.x*rJ.perp; float LA=df.x*rJ.s1+df.y; float LB=df.x*rJ.s2+df.y;
        vA=vA-mA*P; wA-=iA*LA; vB=vB+mB*P; wB+=iB*LB;
    }
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static bool rSolvePos(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 cA=rB[ia].c; float aA=rB[ia].a;
    RV2 cB=rB[ib].c; float aB=rB[ib].a;
    float sA=sinf(aA), csA=cosf(aA), sB=sinf(aB), csB=cosf(aB);
    RV2 rA=rMulRV(sA,csA,rJ.lA), rBv=rMulRV(sB,csB,rJ.lB);
    RV2 d=cB+rBv-cA-rA;
    RV2 axis=rMulRV(sA,csA,rJ.lXAxis);
    float a1=rCross(d+rA, axis); float a2=rCross(rBv, axis);
    RV2 perp=rMulRV(sA,csA,rJ.lYAxis);
    float s1=rCross(d+rA, perp); float s2=rCross(rBv, perp);
    RV2 C1; C1.x=rDot(perp, d); C1.y=aB-aA-rJ.refAngle;
    float linearError=fabsf(C1.x); float angularError=fabsf(C1.y);
    bool active=false; float C2=0.0f;
    if (rJ.enableLimit){
        float translation=rDot(axis, d);
        if (fabsf(rJ.upper-rJ.lower) < 2.0f*R_LINEAR_SLOP){
            C2=rClamp(translation, -R_MAX_LINEAR_CORRECTION, R_MAX_LINEAR_CORRECTION);
            linearError=rMax(linearError, fabsf(translation)); active=true;
        } else if (translation <= rJ.lower){
            C2=rClamp(translation-rJ.lower+R_LINEAR_SLOP, -R_MAX_LINEAR_CORRECTION, 0.0f);
            linearError=rMax(linearError, rJ.lower-translation); active=true;
        } else if (translation >= rJ.upper){
            C2=rClamp(translation-rJ.upper-R_LINEAR_SLOP, 0.0f, R_MAX_LINEAR_CORRECTION);
            linearError=rMax(linearError, translation-rJ.upper); active=true;
        }
    }
    RV3 impulse;
    if (active){
        float k11=mA+mB+iA*s1*s1+iB*s2*s2;
        float k12=iA*s1+iB*s2;
        float k13=iA*s1*a1+iB*s2*a2;
        float k22=iA+iB; if (k22==0.0f) k22=1.0f;
        float k23=iA*a1+iB*a2;
        float k33=mA+mB+iA*a1*a1+iB*a2*a2;
        RMat33 K; K.ex=rv3(k11,k12,k13); K.ey=rv3(k12,k22,k23); K.ez=rv3(k13,k23,k33);
        RV3 C=rv3(C1.x,C1.y,C2);
        impulse=rSolve33(K, rv3(-C.x,-C.y,-C.z));
    } else {
        float k11=mA+mB+iA*s1*s1+iB*s2*s2;
        float k12=iA*s1+iB*s2;
        float k22=iA+iB; if (k22==0.0f) k22=1.0f;
        RMat22 K; K.ex=rv2(k11,k12); K.ey=rv2(k12,k22);
        RV2 impulse1=rSolveMat22(K, rv2(-C1.x,-C1.y));
        impulse.x=impulse1.x; impulse.y=impulse1.y; impulse.z=0.0f;
    }
    RV2 P=impulse.x*perp+impulse.z*axis;
    float LA=impulse.x*s1+impulse.y+impulse.z*a1;
    float LB=impulse.x*s2+impulse.y+impulse.z*a2;
    cA=cA-mA*P; aA-=iA*LA; cB=cB+mB*P; aB+=iB*LB;
    rB[ia].c=cA; rB[ia].a=aA; rB[ib].c=cB; rB[ib].a=aB;
    return linearError<=R_LINEAR_SLOP && angularError<=R_ANGULAR_SLOP;
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------
static const float BODY_INVM = 1.0f;
static const float BODY_INVI = 1.0f;
static const float GRAV = -9.81f;

static int gFails=0; static long gMax=0;
static void chk(const char* what, int sub, float ref, float got){
    long u=ulpDiff(ref,got); if(u>gMax) gMax=u;
    if(u!=0){ if(gFails==0) printf("  FAIL %-6s sub=%d ref=%.9g got=%.9g ulp=%ld\n", what, sub, ref, got, u); gFails=1; }
}

struct Cfg { int enableLimit; float lower, upper; int enableMotor; float maxMotorForce, motorSpeed; };

static void runCase(const char* name, Cfg cfg, int NSUB){
    float h = R_DT;
    // base (static) at origin; body slides on the vertical y axis (localXAxis = (0,1)).
    RV2 bodyPos = rv2(0.0f, 0.0f);
    rB[0].c=rv2(0,0); rB[0].v=rv2(0,0); rB[0].a=0.0f; rB[0].w=0.0f; rB[0].invM=0.0f; rB[0].invI=0.0f;
    rB[1].c=bodyPos;   rB[1].v=rv2(0,0); rB[1].a=0.0f; rB[1].w=0.0f; rB[1].invM=BODY_INVM; rB[1].invI=BODY_INVI;
    rJ.ia=0; rJ.ib=1; rJ.lA=rv2(0,0); rJ.lB=rv2(0,0);
    rJ.lXAxis=rv2(0,1); rJ.lYAxis=rv2(-1,0); rJ.refAngle=0.0f;
    rJ.mA=0.0f; rJ.mB=BODY_INVM; rJ.iA=0.0f; rJ.iB=BODY_INVI;
    rJ.enableLimit=cfg.enableLimit; rJ.lower=cfg.lower; rJ.upper=cfg.upper;
    rJ.enableMotor=cfg.enableMotor; rJ.maxMotorForce=cfg.maxMotorForce; rJ.motorSpeed=cfg.motorSpeed;
    rJ.limitState=R_INACTIVE; rJ.impulse=rv3(0,0,0); rJ.motorImpulse=0.0f;

    GBIslandData isl;
    isl.bodyCount=2; isl.contactCount=0;
    isl.posC[0]=v2(0,0); isl.posA[0]=0.0f; isl.vel[0]=v2(0,0); isl.velW[0]=0.0f;
    isl.posC[1]=v2(bodyPos.x,bodyPos.y); isl.posA[1]=0.0f; isl.vel[1]=v2(0,0); isl.velW[1]=0.0f;
    GBPrismaticJoint j;
    j.indexA=0; j.indexB=1; j.localAnchorA=v2(0,0); j.localAnchorB=v2(0,0);
    j.localXAxisA=v2(0,1); j.localYAxisA=v2(-1,0); j.referenceAngle=0.0f;
    j.invMassA=0.0f; j.invMassB=BODY_INVM; j.invIA=0.0f; j.invIB=BODY_INVI;
    j.enableLimit=cfg.enableLimit; j.lowerTranslation=cfg.lower; j.upperTranslation=cfg.upper;
    j.enableMotor=cfg.enableMotor; j.maxMotorForce=cfg.maxMotorForce; j.motorSpeed=cfg.motorSpeed;
    j.limitState=GB_INACTIVE_LIMIT; j.impulse=v3(0,0,0); j.motorImpulse=0.0f;

    for (int sub=0; sub<NSUB; ++sub){
        // ---- reference ----
        rB[1].v = rB[1].v + h*rv2(0.0f, GRAV);
        rInit();
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) rSolveVel();
        for (int i=0;i<2;++i){ rB[i].c=rB[i].c+h*rB[i].v; rB[i].a += h*rB[i].w; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (rSolvePos()) break; }
        // ---- subject ----
        isl.vel[1] = isl.vel[1] + h*v2(0.0f, GRAV);
        gbPrismaticInitVelocity(j, isl);
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) gbPrismaticSolveVelocity(j, isl);
        for (int i=0;i<2;++i){ isl.posC[i]=isl.posC[i]+h*isl.vel[i]; isl.posA[i]=isl.posA[i]+h*isl.velW[i]; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (gbPrismaticSolvePosition(j, isl)) break; }
        // ---- compare ----
        chk("vB.x", sub, rB[1].v.x, isl.vel[1].x);
        chk("vB.y", sub, rB[1].v.y, isl.vel[1].y);
        chk("wB",   sub, rB[1].w,   isl.velW[1]);
        chk("cB.x", sub, rB[1].c.x, isl.posC[1].x);
        chk("cB.y", sub, rB[1].c.y, isl.posC[1].y);
        chk("aB",   sub, rB[1].a,   isl.posA[1]);
        chk("impx", sub, rJ.impulse.x, j.impulse.x);
        chk("impy", sub, rJ.impulse.y, j.impulse.y);
        chk("impz", sub, rJ.impulse.z, j.impulse.z);
        chk("mimp", sub, rJ.motorImpulse, j.motorImpulse);
    }
    printf("  %-18s %d substeps, %s\n", name, NSUB, gFails ? "DIVERGED" : "matched");
}

int main(){
    printf("Prismatic joint micro-test: gb_prismatic_joint vs Box2D 2.3.0 b2PrismaticJoint\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    Cfg freeSlider  = { 0, 0.0f, 0.0f, 0, 0.0f, 0.0f };
    Cfg limited     = { 1, -1.0f, 0.0f, 0, 0.0f, 0.0f };       // body falls to the lower stop
    Cfg motorized   = { 0, 0.0f, 0.0f, 1, 50.0f, 2.0f };       // motor drives up against gravity

    runCase("free slider", freeSlider, 240);
    runCase("limited slider", limited, 240);
    runCase("motorized slider", motorized, 240);

    if (!gFails){
        printf("PASS gb_prismatic_joint: 0 ULP (free + limit + motor), maxUlp=%ld\n", gMax);
        return 0;
    }
    printf("FAIL gb_prismatic_joint: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
