// gb_revolute_joint_test.cu. Micro-test for gb_revolute_joint.cuh (the full revolute
// joint with motor and limit). 0-ULP versus a self-contained Box2D 2.3.0 reference of
// b2RevoluteJoint's InitVelocityConstraints / SolveVelocityConstraints /
// SolvePositionConstraints, Ref-prefixed.
//
// Three scenarios, each run for many substeps through the full joint solve spine
// (integrate velocity, init, 8 velocity iterations, integrate position, 3 position
// iterations) on both the subject (gb_revolute_joint.cuh) and the reference:
//   1. Point-to-point pendulum. No motor, no limit (the 2x2 anchor path).
//   2. Motorized joint. A motor drives the relative angle against gravity (the motor
//      row plus the 2x2 anchor block).
//   3. Limited joint. The bob swings into the lower angle stop (the 3x3 limit path).
// Each substep compares the bob velocity, angular velocity, position, angle, the three
// impulse components, and the motor impulse at 0 ULP.
//
// Build (frozen flags), self-contained, host or device:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_revolute_joint_test.cu -o test/gb_revolute_joint_test
//   ./test/gb_revolute_joint_test
//   Expected: PASS gb_revolute_joint: 0 ULP
#include "gpu_box2d/gb_revolute_joint.cuh"
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
// Box2D 2.3.0 b2RevoluteJoint reference (full, Ref-prefixed).
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
static inline RV3 operator-(RV3 a){ return rv3(-a.x,-a.y,-a.z); }
static inline float rCross(RV2 a, RV2 b){ return a.x*b.y - a.y*b.x; }
static inline RV2 rCrossSV(float s, RV2 a){ return rv2(-s*a.y, s*a.x); }
static inline float rLen(RV2 v){ return sqrtf(v.x*v.x + v.y*v.y); }
static inline RV2 rMulRV(float s, float c, RV2 v){ return rv2(c*v.x - s*v.y, s*v.x + c*v.y); }
static inline float rClamp(float a, float lo, float hi){ return a<lo?lo:(a>hi?hi:a); }

#define R_PI 3.14159265359f
#define R_DT (1.0f/60.0f)
#define R_LINEAR_SLOP 0.005f
#define R_ANGULAR_SLOP (2.0f/180.0f*R_PI)
#define R_MAX_ANGULAR_CORRECTION (8.0f/180.0f*R_PI)
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
    int ia, ib; RV2 lA, lB; float refAngle; float mA, mB, iA, iB;
    int enableMotor; float maxMotorTorque, motorSpeed;
    int enableLimit; float lowerAngle, upperAngle;
    RV2 rA, rB; RMat33 mass; float motorMass; int limitState;
    RV3 impulse; float motorImpulse;
};
static RBody rB[2];
static RJoint rJ;

static void rInit(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    float aA=rB[ia].a, aB=rB[ib].a;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    float sA=sinf(aA), csA=cosf(aA), sB=sinf(aB), csB=cosf(aB);
    rJ.rA=rMulRV(sA,csA,rJ.lA); rJ.rB=rMulRV(sB,csB,rJ.lB);
    bool fixedRotation=(iA+iB==0.0f);
    RMat33& K=rJ.mass;
    K.ex.x=mA+mB+rJ.rA.y*rJ.rA.y*iA+rJ.rB.y*rJ.rB.y*iB;
    K.ey.x=-rJ.rA.y*rJ.rA.x*iA-rJ.rB.y*rJ.rB.x*iB;
    K.ez.x=-rJ.rA.y*iA-rJ.rB.y*iB;
    K.ex.y=K.ey.x;
    K.ey.y=mA+mB+rJ.rA.x*rJ.rA.x*iA+rJ.rB.x*rJ.rB.x*iB;
    K.ez.y=rJ.rA.x*iA+rJ.rB.x*iB;
    K.ex.z=K.ez.x; K.ey.z=K.ez.y; K.ez.z=iA+iB;
    rJ.motorMass=iA+iB; if (rJ.motorMass>0.0f) rJ.motorMass=1.0f/rJ.motorMass;
    if (rJ.enableMotor==0 || fixedRotation) rJ.motorImpulse=0.0f;
    if (rJ.enableLimit && !fixedRotation){
        float jointAngle=aB-aA-rJ.refAngle;
        if (fabsf(rJ.upperAngle-rJ.lowerAngle) < 2.0f*R_ANGULAR_SLOP) rJ.limitState=R_EQUAL;
        else if (jointAngle <= rJ.lowerAngle){ if (rJ.limitState!=R_LOWER) rJ.impulse.z=0.0f; rJ.limitState=R_LOWER; }
        else if (jointAngle >= rJ.upperAngle){ if (rJ.limitState!=R_UPPER) rJ.impulse.z=0.0f; rJ.limitState=R_UPPER; }
        else { rJ.limitState=R_INACTIVE; rJ.impulse.z=0.0f; }
    } else rJ.limitState=R_INACTIVE;
    RV2 P=rv2(rJ.impulse.x, rJ.impulse.y);
    vA=vA-mA*P; wA-=iA*(rCross(rJ.rA,P)+rJ.motorImpulse+rJ.impulse.z);
    vB=vB+mB*P; wB+=iB*(rCross(rJ.rB,P)+rJ.motorImpulse+rJ.impulse.z);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static void rSolveVel(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    bool fixedRotation=(iA+iB==0.0f);
    if (rJ.enableMotor && rJ.limitState!=R_EQUAL && !fixedRotation){
        float Cdot=wB-wA-rJ.motorSpeed;
        float impulse=-rJ.motorMass*Cdot;
        float oldImpulse=rJ.motorImpulse;
        float maxImpulse=R_DT*rJ.maxMotorTorque;
        rJ.motorImpulse=rClamp(rJ.motorImpulse+impulse, -maxImpulse, maxImpulse);
        impulse=rJ.motorImpulse-oldImpulse;
        wA-=iA*impulse; wB+=iB*impulse;
    }
    if (rJ.enableLimit && rJ.limitState!=R_INACTIVE && !fixedRotation){
        RV2 Cdot1=vB+rCrossSV(wB,rJ.rB)-vA-rCrossSV(wA,rJ.rA);
        float Cdot2=wB-wA;
        RV3 Cdot=rv3(Cdot1.x,Cdot1.y,Cdot2);
        RV3 impulse=-rSolve33(rJ.mass,Cdot);
        if (rJ.limitState==R_EQUAL) rJ.impulse=rJ.impulse+impulse;
        else if (rJ.limitState==R_LOWER){
            float newImpulse=rJ.impulse.z+impulse.z;
            if (newImpulse<0.0f){
                RV2 rhs=-Cdot1+rJ.impulse.z*rv2(rJ.mass.ez.x,rJ.mass.ez.y);
                RV2 reduced=rSolve22(rJ.mass,rhs);
                impulse.x=reduced.x; impulse.y=reduced.y; impulse.z=-rJ.impulse.z;
                rJ.impulse.x+=reduced.x; rJ.impulse.y+=reduced.y; rJ.impulse.z=0.0f;
            } else rJ.impulse=rJ.impulse+impulse;
        } else if (rJ.limitState==R_UPPER){
            float newImpulse=rJ.impulse.z+impulse.z;
            if (newImpulse>0.0f){
                RV2 rhs=-Cdot1+rJ.impulse.z*rv2(rJ.mass.ez.x,rJ.mass.ez.y);
                RV2 reduced=rSolve22(rJ.mass,rhs);
                impulse.x=reduced.x; impulse.y=reduced.y; impulse.z=-rJ.impulse.z;
                rJ.impulse.x+=reduced.x; rJ.impulse.y+=reduced.y; rJ.impulse.z=0.0f;
            } else rJ.impulse=rJ.impulse+impulse;
        }
        RV2 P=rv2(impulse.x,impulse.y);
        vA=vA-mA*P; wA-=iA*(rCross(rJ.rA,P)+impulse.z);
        vB=vB+mB*P; wB+=iB*(rCross(rJ.rB,P)+impulse.z);
    } else {
        RV2 Cdot=vB+rCrossSV(wB,rJ.rB)-vA-rCrossSV(wA,rJ.rA);
        RV2 impulse=rSolve22(rJ.mass, rv2(-Cdot.x,-Cdot.y));
        rJ.impulse.x+=impulse.x; rJ.impulse.y+=impulse.y;
        RV2 P=impulse;
        vA=vA-mA*P; wA-=iA*rCross(rJ.rA,P);
        vB=vB+mB*P; wB+=iB*rCross(rJ.rB,P);
    }
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static bool rSolvePos(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 cA=rB[ia].c; float aA=rB[ia].a;
    RV2 cB=rB[ib].c; float aB=rB[ib].a;
    float angularError=0.0f, positionError=0.0f;
    bool fixedRotation=(iA+iB==0.0f);
    if (rJ.enableLimit && rJ.limitState!=R_INACTIVE && !fixedRotation){
        float angle=aB-aA-rJ.refAngle;
        float limitImpulse=0.0f;
        if (rJ.limitState==R_EQUAL){
            float C=rClamp(angle-rJ.lowerAngle, -R_MAX_ANGULAR_CORRECTION, R_MAX_ANGULAR_CORRECTION);
            limitImpulse=-rJ.motorMass*C; angularError=fabsf(C);
        } else if (rJ.limitState==R_LOWER){
            float C=angle-rJ.lowerAngle; angularError=-C;
            C=rClamp(C+R_ANGULAR_SLOP, -R_MAX_ANGULAR_CORRECTION, 0.0f);
            limitImpulse=-rJ.motorMass*C;
        } else if (rJ.limitState==R_UPPER){
            float C=angle-rJ.upperAngle; angularError=C;
            C=rClamp(C-R_ANGULAR_SLOP, 0.0f, R_MAX_ANGULAR_CORRECTION);
            limitImpulse=-rJ.motorMass*C;
        }
        aA-=iA*limitImpulse; aB+=iB*limitImpulse;
    }
    float sA=sinf(aA), csA=cosf(aA), sB=sinf(aB), csB=cosf(aB);
    RV2 rA=rMulRV(sA,csA,rJ.lA), rBv=rMulRV(sB,csB,rJ.lB);
    RV2 C=cB+rBv-cA-rA;
    positionError=rLen(C);
    RMat22 K;
    K.ex.x=mA+mB+iA*rA.y*rA.y+iB*rBv.y*rBv.y;
    K.ex.y=-iA*rA.x*rA.y-iB*rBv.x*rBv.y;
    K.ey.x=K.ex.y;
    K.ey.y=mA+mB+iA*rA.x*rA.x+iB*rBv.x*rBv.x;
    RV2 impulse=-1.0f*rSolveMat22(K, C);
    cA=cA-mA*impulse; aA-=iA*rCross(rA,impulse);
    cB=cB+mB*impulse; aB+=iB*rCross(rBv,impulse);
    rB[ia].c=cA; rB[ia].a=aA; rB[ib].c=cB; rB[ib].a=aB;
    return positionError<=R_LINEAR_SLOP && angularError<=R_ANGULAR_SLOP;
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

struct Cfg { int enableMotor; float maxMotorTorque, motorSpeed; int enableLimit; float lower, upper; };

static void runCase(const char* name, Cfg cfg, int NSUB){
    float h = R_DT;
    RV2 bobPos = rv2(1.5f, 0.0f);
    rB[0].c=rv2(0,0); rB[0].v=rv2(0,0); rB[0].a=0.0f; rB[0].w=0.0f; rB[0].invM=0.0f; rB[0].invI=0.0f;
    rB[1].c=bobPos;   rB[1].v=rv2(0,0); rB[1].a=0.0f; rB[1].w=0.0f; rB[1].invM=BOB_INVM; rB[1].invI=BOB_INVI;
    rJ.ia=0; rJ.ib=1; rJ.lA=rv2(0,0); rJ.lB=rv2(-1.5f,0.0f); rJ.refAngle=0.0f;
    rJ.mA=0.0f; rJ.mB=BOB_INVM; rJ.iA=0.0f; rJ.iB=BOB_INVI;
    rJ.enableMotor=cfg.enableMotor; rJ.maxMotorTorque=cfg.maxMotorTorque; rJ.motorSpeed=cfg.motorSpeed;
    rJ.enableLimit=cfg.enableLimit; rJ.lowerAngle=cfg.lower; rJ.upperAngle=cfg.upper;
    rJ.limitState=R_INACTIVE; rJ.impulse=rv3(0,0,0); rJ.motorImpulse=0.0f;

    GBIslandData isl;
    isl.bodyCount=2; isl.contactCount=0;
    isl.posC[0]=v2(0,0); isl.posA[0]=0.0f; isl.vel[0]=v2(0,0); isl.velW[0]=0.0f;
    isl.posC[1]=v2(bobPos.x,bobPos.y); isl.posA[1]=0.0f; isl.vel[1]=v2(0,0); isl.velW[1]=0.0f;
    GBRevoluteJointFull j;
    j.indexA=0; j.indexB=1; j.localAnchorA=v2(0,0); j.localAnchorB=v2(-1.5f,0.0f); j.referenceAngle=0.0f;
    j.invMassA=0.0f; j.invMassB=BOB_INVM; j.invIA=0.0f; j.invIB=BOB_INVI;
    j.enableMotor=cfg.enableMotor; j.maxMotorTorque=cfg.maxMotorTorque; j.motorSpeed=cfg.motorSpeed;
    j.enableLimit=cfg.enableLimit; j.lowerAngle=cfg.lower; j.upperAngle=cfg.upper;
    j.limitState=GB_INACTIVE_LIMIT; j.impulse=v3(0,0,0); j.motorImpulse=0.0f;

    for (int sub=0; sub<NSUB; ++sub){
        rB[1].v = rB[1].v + h*rv2(0.0f, GRAV);
        rInit();
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) rSolveVel();
        for (int i=0;i<2;++i){ rB[i].c=rB[i].c+h*rB[i].v; rB[i].a += h*rB[i].w; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (rSolvePos()) break; }
        isl.vel[1] = isl.vel[1] + h*v2(0.0f, GRAV);
        gbRevoluteFullInitVelocity(j, isl);
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) gbRevoluteFullSolveVelocity(j, isl);
        for (int i=0;i<2;++i){ isl.posC[i]=isl.posC[i]+h*isl.vel[i]; isl.posA[i]=isl.posA[i]+h*isl.velW[i]; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (gbRevoluteFullSolvePosition(j, isl)) break; }
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
    printf("  %-20s %d substeps, %s\n", name, NSUB, gFails ? "DIVERGED" : "matched");
}

int main(){
    printf("Revolute joint micro-test (full): gb_revolute_joint vs Box2D 2.3.0 b2RevoluteJoint\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    Cfg p2p     = { 0, 0.0f, 0.0f, 0, 0.0f, 0.0f };
    Cfg motor   = { 1, 100.0f, 3.0f, 0, 0.0f, 0.0f };                       // motor drives rotation
    Cfg limited = { 0, 0.0f, 0.0f, 1, -0.25f*R_PI, 0.25f*R_PI };            // bob swings into a stop

    runCase("point-to-point", p2p, 240);
    runCase("motorized", motor, 240);
    runCase("angle-limited", limited, 240);

    if (!gFails){
        printf("PASS gb_revolute_joint: 0 ULP (point-to-point + motor + limit), maxUlp=%ld\n", gMax);
        return 0;
    }
    printf("FAIL gb_revolute_joint: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
