// gb_weld_joint_test.cu. Micro-test for gb_weld_joint.cuh. 0-ULP versus a self-contained
// Box2D 2.3.0 reference of b2WeldJoint's InitVelocityConstraints /
// SolveVelocityConstraints / SolvePositionConstraints, Ref-prefixed.
//
// Two scenarios, each run for many substeps through the full joint solve spine
// (integrate velocity, init, 8 velocity iterations, integrate position, 3 position
// iterations) on both the subject (gb_weld_joint.cuh) and the reference:
//   1. Rigid weld. A static base and a dynamic bar welded to it, so the bar holds its
//      anchor point and its angle under gravity (the 3x3 symmetric-inverse path).
//   2. Soft weld. The same bodies with frequencyHz and a damping ratio on the angular
//      row (the 2x2-inverse-plus-soft-angular path).
// Each substep compares the bar velocity, angular velocity, position, angle, and the
// three warm-start impulse components at 0 ULP.
//
// Build (frozen flags), self-contained, host or device:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_weld_joint_test.cu -o test/gb_weld_joint_test
//   ./test/gb_weld_joint_test
//   Expected: PASS gb_weld_joint: 0 ULP
#include "gpu_box2d/gb_weld_joint.cuh"
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
// Box2D 2.3.0 b2WeldJoint reference (Ref-prefixed).
// ---------------------------------------------------------------------------
struct RV2 { float x, y; };
struct RV3 { float x, y, z; };
static inline RV2 rv2(float x, float y){ RV2 r; r.x=x; r.y=y; return r; }
static inline RV3 rv3(float x, float y, float z){ RV3 r; r.x=x; r.y=y; r.z=z; return r; }
static inline RV2 operator+(RV2 a, RV2 b){ return rv2(a.x+b.x, a.y+b.y); }
static inline RV2 operator-(RV2 a, RV2 b){ return rv2(a.x-b.x, a.y-b.y); }
static inline RV2 operator*(float s, RV2 a){ return rv2(s*a.x, s*a.y); }
static inline RV3 operator+(RV3 a, RV3 b){ return rv3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline RV3 operator*(float s, RV3 a){ return rv3(s*a.x, s*a.y, s*a.z); }
static inline RV3 operator-(RV3 a){ return rv3(-a.x, -a.y, -a.z); }
static inline float rDot(RV2 a, RV2 b){ return a.x*b.x + a.y*b.y; }
static inline float rCross(RV2 a, RV2 b){ return a.x*b.y - a.y*b.x; }
static inline RV2 rCrossSV(float s, RV2 a){ return rv2(-s*a.y, s*a.x); }
static inline float rLen(RV2 v){ return sqrtf(v.x*v.x + v.y*v.y); }
static inline RV2 rMulRV(float s, float c, RV2 v){ return rv2(c*v.x - s*v.y, s*v.x + c*v.y); }

#define R_PI 3.14159265359f
#define R_DT (1.0f/60.0f)
#define R_LINEAR_SLOP 0.005f
#define R_ANGULAR_SLOP (2.0f/180.0f*R_PI)

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
static RMat33 rGetInverse22(const RMat33& M){
    float a=M.ex.x, b=M.ey.x, c=M.ex.y, d=M.ey.y;
    float det=a*d-b*c; if(det!=0.0f) det=1.0f/det;
    RMat33 R;
    R.ex.x= det*d; R.ey.x=-det*b; R.ex.z=0.0f;
    R.ex.y=-det*c; R.ey.y= det*a; R.ey.z=0.0f;
    R.ez.x=0.0f; R.ez.y=0.0f; R.ez.z=0.0f;
    return R;
}
static RMat33 rGetSymInverse33(const RMat33& M){
    float det = rDot3(M.ex, rCross3(M.ey, M.ez));
    if (det != 0.0f) det = 1.0f/det;
    float a11=M.ex.x, a12=M.ey.x, a13=M.ez.x, a22=M.ey.y, a23=M.ez.y, a33=M.ez.z;
    RMat33 R;
    R.ex.x=det*(a22*a33-a23*a23); R.ex.y=det*(a13*a23-a12*a33); R.ex.z=det*(a12*a23-a13*a22);
    R.ey.x=R.ex.y; R.ey.y=det*(a11*a33-a13*a13); R.ey.z=det*(a13*a12-a11*a23);
    R.ez.x=R.ex.z; R.ez.y=R.ey.z; R.ez.z=det*(a11*a22-a12*a12);
    return R;
}
static RV3 rMulM33V3(const RMat33& A, RV3 v){
    return rv3(v.x*A.ex.x+v.y*A.ey.x+v.z*A.ez.x, v.x*A.ex.y+v.y*A.ey.y+v.z*A.ez.y, v.x*A.ex.z+v.y*A.ey.z+v.z*A.ez.z);
}
static RV2 rMulM33V2(const RMat33& A, RV2 v){
    return rv2(A.ex.x*v.x+A.ey.x*v.y, A.ex.y*v.x+A.ey.y*v.y);
}

struct RBody { RV2 c, v; float a, w, invM, invI; };
struct RJoint {
    int ia, ib; RV2 lA, lB; float refAngle; float mA, mB, iA, iB;
    float frequencyHz, dampingRatio;
    RV2 rA, rB; RMat33 mass; float gamma, bias; RV3 impulse;
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
    RMat33 K;
    K.ex.x=mA+mB+rJ.rA.y*rJ.rA.y*iA+rJ.rB.y*rJ.rB.y*iB;
    K.ey.x=-rJ.rA.y*rJ.rA.x*iA-rJ.rB.y*rJ.rB.x*iB;
    K.ez.x=-rJ.rA.y*iA-rJ.rB.y*iB;
    K.ex.y=K.ey.x;
    K.ey.y=mA+mB+rJ.rA.x*rJ.rA.x*iA+rJ.rB.x*rJ.rB.x*iB;
    K.ez.y=rJ.rA.x*iA+rJ.rB.x*iB;
    K.ex.z=K.ez.x; K.ey.z=K.ez.y; K.ez.z=iA+iB;
    if (rJ.frequencyHz>0.0f){
        rJ.mass=rGetInverse22(K);
        float invM=iA+iB; float m=invM>0.0f?1.0f/invM:0.0f;
        float C=aB-aA-rJ.refAngle;
        float omega=2.0f*R_PI*rJ.frequencyHz;
        float d=2.0f*m*rJ.dampingRatio*omega;
        float k=m*omega*omega; float h=R_DT;
        rJ.gamma=h*(d+h*k); rJ.gamma=rJ.gamma!=0.0f?1.0f/rJ.gamma:0.0f;
        rJ.bias=C*h*k*rJ.gamma;
        invM+=rJ.gamma; rJ.mass.ez.z=invM!=0.0f?1.0f/invM:0.0f;
    } else if (K.ez.z==0.0f){
        rJ.mass=rGetInverse22(K); rJ.gamma=0.0f; rJ.bias=0.0f;
    } else {
        rJ.mass=rGetSymInverse33(K); rJ.gamma=0.0f; rJ.bias=0.0f;
    }
    RV2 P=rv2(rJ.impulse.x, rJ.impulse.y);
    vA=vA-mA*P; wA-=iA*(rCross(rJ.rA,P)+rJ.impulse.z);
    vB=vB+mB*P; wB+=iB*(rCross(rJ.rB,P)+rJ.impulse.z);
    rB[ia].v=vA; rB[ia].w=wA; rB[ib].v=vB; rB[ib].w=wB;
}
static void rSolveVel(){
    int ia=rJ.ia, ib=rJ.ib;
    float mA=rJ.mA, mB=rJ.mB, iA=rJ.iA, iB=rJ.iB;
    RV2 vA=rB[ia].v; float wA=rB[ia].w;
    RV2 vB=rB[ib].v; float wB=rB[ib].w;
    if (rJ.frequencyHz>0.0f){
        float Cdot2=wB-wA;
        float impulse2=-rJ.mass.ez.z*(Cdot2+rJ.bias+rJ.gamma*rJ.impulse.z);
        rJ.impulse.z+=impulse2; wA-=iA*impulse2; wB+=iB*impulse2;
        RV2 Cdot1=vB+rCrossSV(wB,rJ.rB)-vA-rCrossSV(wA,rJ.rA);
        RV2 impulse1=-1.0f*rMulM33V2(rJ.mass,Cdot1);
        rJ.impulse.x+=impulse1.x; rJ.impulse.y+=impulse1.y;
        RV2 P=impulse1;
        vA=vA-mA*P; wA-=iA*rCross(rJ.rA,P);
        vB=vB+mB*P; wB+=iB*rCross(rJ.rB,P);
    } else {
        RV2 Cdot1=vB+rCrossSV(wB,rJ.rB)-vA-rCrossSV(wA,rJ.rA);
        float Cdot2=wB-wA;
        RV3 Cdot=rv3(Cdot1.x,Cdot1.y,Cdot2);
        RV3 impulse=-1.0f*rMulM33V3(rJ.mass,Cdot);
        rJ.impulse=rJ.impulse+impulse;
        RV2 P=rv2(impulse.x,impulse.y);
        vA=vA-mA*P; wA-=iA*(rCross(rJ.rA,P)+impulse.z);
        vB=vB+mB*P; wB+=iB*(rCross(rJ.rB,P)+impulse.z);
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
    float positionError, angularError;
    RMat33 K;
    K.ex.x=mA+mB+rA.y*rA.y*iA+rBv.y*rBv.y*iB;
    K.ey.x=-rA.y*rA.x*iA-rBv.y*rBv.x*iB;
    K.ez.x=-rA.y*iA-rBv.y*iB;
    K.ex.y=K.ey.x;
    K.ey.y=mA+mB+rA.x*rA.x*iA+rBv.x*rBv.x*iB;
    K.ez.y=rA.x*iA+rBv.x*iB;
    K.ex.z=K.ez.x; K.ey.z=K.ez.y; K.ez.z=iA+iB;
    if (rJ.frequencyHz>0.0f){
        RV2 C1=cB+rBv-cA-rA;
        positionError=rLen(C1); angularError=0.0f;
        RV2 P=-1.0f*rSolve22(K,C1);
        cA=cA-mA*P; aA-=iA*rCross(rA,P);
        cB=cB+mB*P; aB+=iB*rCross(rBv,P);
    } else {
        RV2 C1=cB+rBv-cA-rA;
        float C2=aB-aA-rJ.refAngle;
        positionError=rLen(C1); angularError=fabsf(C2);
        RV3 C=rv3(C1.x,C1.y,C2);
        RV3 impulse;
        if (K.ez.z>0.0f) impulse=-rSolve33(K,C);
        else { RV2 i2=-1.0f*rSolve22(K,C1); impulse=rv3(i2.x,i2.y,0.0f); }
        RV2 P=rv2(impulse.x,impulse.y);
        cA=cA-mA*P; aA-=iA*(rCross(rA,P)+impulse.z);
        cB=cB+mB*P; aB+=iB*(rCross(rBv,P)+impulse.z);
    }
    rB[ia].c=cA; rB[ia].a=aA; rB[ib].c=cB; rB[ib].a=aB;
    return positionError<=R_LINEAR_SLOP && angularError<=R_ANGULAR_SLOP;
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------
static const float BAR_INVM = 1.0f;
static const float BAR_INVI = 2.0f;
static const float GRAV = -9.81f;

static int gFails=0; static long gMax=0;
static void chk(const char* what, int sub, float ref, float got){
    long u=ulpDiff(ref,got); if(u>gMax) gMax=u;
    if(u!=0){ if(gFails==0) printf("  FAIL %-6s sub=%d ref=%.9g got=%.9g ulp=%ld\n", what, sub, ref, got, u); gFails=1; }
}

static void runCase(const char* name, float freq, float damp, int NSUB){
    float h = R_DT;
    // base (static) at origin; bar (dynamic) with its anchor at the origin, body center
    // offset so the weld holds both the anchor and the angle.
    RV2 barPos = rv2(1.0f, 0.0f);
    rB[0].c=rv2(0,0); rB[0].v=rv2(0,0); rB[0].a=0.0f; rB[0].w=0.0f; rB[0].invM=0.0f; rB[0].invI=0.0f;
    rB[1].c=barPos;   rB[1].v=rv2(0,0); rB[1].a=0.0f; rB[1].w=0.0f; rB[1].invM=BAR_INVM; rB[1].invI=BAR_INVI;
    rJ.ia=0; rJ.ib=1; rJ.lA=rv2(0,0); rJ.lB=rv2(-1.0f,0.0f); rJ.refAngle=0.0f;
    rJ.mA=0.0f; rJ.mB=BAR_INVM; rJ.iA=0.0f; rJ.iB=BAR_INVI;
    rJ.frequencyHz=freq; rJ.dampingRatio=damp; rJ.impulse=rv3(0,0,0);

    GBIslandData isl;
    isl.bodyCount=2; isl.contactCount=0;
    isl.posC[0]=v2(0,0); isl.posA[0]=0.0f; isl.vel[0]=v2(0,0); isl.velW[0]=0.0f;
    isl.posC[1]=v2(barPos.x,barPos.y); isl.posA[1]=0.0f; isl.vel[1]=v2(0,0); isl.velW[1]=0.0f;
    GBWeldJoint j;
    j.indexA=0; j.indexB=1; j.localAnchorA=v2(0,0); j.localAnchorB=v2(-1.0f,0.0f); j.referenceAngle=0.0f;
    j.invMassA=0.0f; j.invMassB=BAR_INVM; j.invIA=0.0f; j.invIB=BAR_INVI;
    j.frequencyHz=freq; j.dampingRatio=damp; j.impulse=v3(0,0,0);

    for (int sub=0; sub<NSUB; ++sub){
        // ---- reference ----
        rB[1].v = rB[1].v + h*rv2(0.0f, GRAV);
        rInit();
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) rSolveVel();
        for (int i=0;i<2;++i){ rB[i].c=rB[i].c+h*rB[i].v; rB[i].a += h*rB[i].w; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (rSolvePos()) break; }
        // ---- subject ----
        isl.vel[1] = isl.vel[1] + h*v2(0.0f, GRAV);
        gbWeldInitVelocity(j, isl);
        for (int it=0; it<GB_VELOCITY_ITERS; ++it) gbWeldSolveVelocity(j, isl);
        for (int i=0;i<2;++i){ isl.posC[i]=isl.posC[i]+h*isl.vel[i]; isl.posA[i]=isl.posA[i]+h*isl.velW[i]; }
        for (int it=0; it<GB_POSITION_ITERS; ++it){ if (gbWeldSolvePosition(j, isl)) break; }
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
    }
    printf("  %-12s %d substeps, %s\n", name, NSUB, gFails ? "DIVERGED" : "matched");
}

int main(){
    printf("Weld joint micro-test: gb_weld_joint vs Box2D 2.3.0 b2WeldJoint\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    runCase("rigid weld", 0.0f, 0.0f, 240);
    runCase("soft weld", 4.0f, 0.7f, 240);

    if (!gFails){
        printf("PASS gb_weld_joint: 0 ULP (rigid + soft weld), maxUlp=%ld\n", gMax);
        return 0;
    }
    printf("FAIL gb_weld_joint: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
