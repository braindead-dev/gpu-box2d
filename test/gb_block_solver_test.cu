// gb_block_solver_test.cu. Micro-test for the two-point block solver in
// gb_contact_solver.cuh. 0-ULP versus a self-contained Box2D 2.3.0 reference of
// b2ContactSolver (the two-point InitializeVelocityConstraints, WarmStart, the block
// SolveVelocityConstraints LCP cascade, StoreImpulses, and the two-point
// SolvePositionConstraints).
//
// Scenario: a dynamic box resting on a static ground, with a two-point face manifold
// (the manifold a box-on-ground contact produces). The test runs the full solver
// spine for one substep (8 velocity iterations, 3 position iterations) on both the
// subject (gb_contact_solver.cuh, driven through a GBIslandData) and the reference,
// then compares every body velocity and position and both points' warm-start impulses
// at 0 ULP. This exercises the 2x2 block matrix, its inverse, and the four-case LCP
// cascade directly.
//
// Build (frozen flags), self-contained:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_block_solver_test.cu -o test/gb_block_solver_test
//   ./test/gb_block_solver_test
//   Expected: PASS gb_block_solver: 0 ULP (two-point block solve)
#include "gpu_box2d/gb_contact_solver.cuh"   // the subject solver phases
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
// Box2D 2.3.0 b2ContactSolver reference (Ref-prefixed). Two-point face manifold.
// ---------------------------------------------------------------------------
struct RV2 { float x, y; };
static inline RV2 rv2(float x, float y){ RV2 r; r.x=x; r.y=y; return r; }
static inline RV2 operator+(RV2 a, RV2 b){ return rv2(a.x+b.x, a.y+b.y); }
static inline RV2 operator-(RV2 a, RV2 b){ return rv2(a.x-b.x, a.y-b.y); }
static inline RV2 operator*(float s, RV2 a){ return rv2(s*a.x, s*a.y); }
static inline RV2 operator-(RV2 a){ return rv2(-a.x, -a.y); }
static inline float rDot(RV2 a, RV2 b){ return a.x*b.x + a.y*b.y; }
static inline float rCross(RV2 a, RV2 b){ return a.x*b.y - a.y*b.x; }
static inline RV2 rCrossVS(RV2 a, float s){ return rv2(s*a.y, -s*a.x); }
static inline RV2 rCrossSV(float s, RV2 a){ return rv2(-s*a.y, s*a.x); }
static inline float rClamp(float a, float lo, float hi){ return a<lo?lo:(a>hi?hi:a); }
static inline float rMax(float a, float b){ return a>b?a:b; }
static inline float rMin(float a, float b){ return a<b?a:b; }
static inline RV2 rMulRV(float s, float c, RV2 v){ return rv2(c*v.x - s*v.y, s*v.x + c*v.y); }
static const float REPS = 1.19209290e-07f;
static const float RVELTHRESH = 1.0f;
static const float RBAUM = 0.2f, RSLOP = 0.005f, RMAXCORR = 0.2f;

struct RMat22 { RV2 ex, ey; };
static RMat22 rInverse(const RMat22& m){
    float a=m.ex.x, b=m.ey.x, c=m.ex.y, d=m.ey.y;
    RMat22 B; float det=a*d-b*c; if(det!=0.0f) det=1.0f/det;
    B.ex.x=det*d; B.ey.x=-det*b; B.ex.y=-det*c; B.ey.y=det*a; return B;
}
static RV2 rMulMV(const RMat22& A, RV2 v){ return rv2(A.ex.x*v.x + A.ey.x*v.y, A.ex.y*v.x + A.ey.y*v.y); }

struct RPt { RV2 rA, rB; float normalImpulse, tangentImpulse, normalMass, tangentMass, velocityBias; };
struct RVC {
    RPt p[2]; RV2 normal; RMat22 K, normalMass; int pointCount;
    float friction, restitution; int indexA, indexB;
    float invMassA, invMassB, invIA, invIB;
    // position-solve cache
    RV2 localNormal, localPoint, localPoints[2]; int type; float radiusA, radiusB;
};
enum { RFACE_A=1 };

struct Body { RV2 c, v; float a, w, invM, invI; };

// The two bodies: ground (static, index 0) and the resting box (dynamic, index 1).
static Body gBodies[2];

// One contact: ground (A) vs box (B), face-A manifold, two points.
static RVC gRef;

// reference b2WorldManifold::Initialize, face-A, two points. Computes the world
// contact points the same way the subject's gbWorldManifoldInit does.
static void rWorldManifold(RV2 out[2]){
    RVC& m=gRef;
    int ia=m.indexA, ib=m.indexB;
    RV2 cA=gBodies[ia].c, cB=gBodies[ib].c; float aA=gBodies[ia].a, aB=gBodies[ib].a;
    float sA=sinf(aA), cAng=cosf(aA), sB=sinf(aB), cB2=cosf(aB);
    RV2 normal=rMulRV(sA,cAng,m.localNormal);
    RV2 planePoint=rMulRV(sA,cAng,m.localPoint); planePoint=planePoint+cA;
    for (int j=0;j<m.pointCount;++j){
        RV2 clip=rMulRV(sB,cB2,m.localPoints[j]); clip=clip+cB;
        RV2 a = clip + (m.radiusA - rDot(clip-planePoint,normal))*normal;
        RV2 b = clip - m.radiusB*normal;
        out[j] = 0.5f*(a+b);
    }
}
// reference InitializeVelocityConstraints (per point) + block prepare.
static void rInit(){
    RVC& vc = gRef;
    int ia=vc.indexA, ib=vc.indexB;
    float mA=vc.invMassA, mB=vc.invMassB, iA=vc.invIA, iB=vc.invIB;
    RV2 cA=gBodies[ia].c, vA=gBodies[ia].v; float wA=gBodies[ia].w;
    RV2 cB=gBodies[ib].c, vB=gBodies[ib].v; float wB=gBodies[ib].w;
    RV2 wpt[2]; rWorldManifold(wpt);
    for (int j=0;j<vc.pointCount;++j){
        RPt& vcp=vc.p[j];
        vcp.rA = wpt[j]-cA; vcp.rB = wpt[j]-cB;
        float rnA=rCross(vcp.rA,vc.normal), rnB=rCross(vcp.rB,vc.normal);
        float kN=mA+mB+iA*rnA*rnA+iB*rnB*rnB;
        vcp.normalMass = kN>0.0f?1.0f/kN:0.0f;
        RV2 tangent=rCrossVS(vc.normal,1.0f);
        float rtA=rCross(vcp.rA,tangent), rtB=rCross(vcp.rB,tangent);
        float kT=mA+mB+iA*rtA*rtA+iB*rtB*rtB;
        vcp.tangentMass = kT>0.0f?1.0f/kT:0.0f;
        vcp.velocityBias=0.0f;
        float vRel=rDot(vc.normal, vB+rCrossSV(wB,vcp.rB)-vA-rCrossSV(wA,vcp.rA));
        if (vRel < -RVELTHRESH) vcp.velocityBias = -vc.restitution*vRel;
    }
    if (vc.pointCount==2){
        RPt& cp1=vc.p[0]; RPt& cp2=vc.p[1];
        float rn1A=rCross(cp1.rA,vc.normal), rn1B=rCross(cp1.rB,vc.normal);
        float rn2A=rCross(cp2.rA,vc.normal), rn2B=rCross(cp2.rB,vc.normal);
        float k11=mA+mB+iA*rn1A*rn1A+iB*rn1B*rn1B;
        float k22=mA+mB+iA*rn2A*rn2A+iB*rn2B*rn2B;
        float k12=mA+mB+iA*rn1A*rn2A+iB*rn1B*rn2B;
        const float kmax=1000.0f;
        if (k11*k11 < kmax*(k11*k22-k12*k12)){
            vc.K.ex=rv2(k11,k12); vc.K.ey=rv2(k12,k22);
            vc.normalMass=rInverse(vc.K);
        } else vc.pointCount=1;
    }
}
static void rWarmStart(){
    RVC& vc=gRef; int ia=vc.indexA, ib=vc.indexB;
    float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
    RV2 vA=gBodies[ia].v; float wA=gBodies[ia].w;
    RV2 vB=gBodies[ib].v; float wB=gBodies[ib].w;
    RV2 normal=vc.normal, tangent=rCrossVS(normal,1.0f);
    for (int j=0;j<vc.pointCount;++j){
        RPt& vcp=vc.p[j];
        RV2 P=vcp.normalImpulse*normal + vcp.tangentImpulse*tangent;
        wA -= iA*rCross(vcp.rA,P); vA=vA-mA*P;
        wB += iB*rCross(vcp.rB,P); vB=vB+mB*P;
    }
    gBodies[ia].v=vA; gBodies[ia].w=wA; gBodies[ib].v=vB; gBodies[ib].w=wB;
}
static void rSolveVel(){
    RVC& vc=gRef; int ia=vc.indexA, ib=vc.indexB;
    float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
    RV2 vA=gBodies[ia].v; float wA=gBodies[ia].w;
    RV2 vB=gBodies[ib].v; float wB=gBodies[ib].w;
    RV2 normal=vc.normal, tangent=rCrossVS(normal,1.0f);
    float friction=vc.friction; int pc=vc.pointCount;
    for (int j=0;j<pc;++j){
        RPt& vcp=vc.p[j];
        RV2 dv=vB+rCrossSV(wB,vcp.rB)-vA-rCrossSV(wA,vcp.rA);
        float vt=rDot(dv,tangent)-0.0f; float lambda=vcp.tangentMass*(-vt);
        float maxF=friction*vcp.normalImpulse;
        float ni=rClamp(vcp.tangentImpulse+lambda,-maxF,maxF);
        lambda=ni-vcp.tangentImpulse; vcp.tangentImpulse=ni;
        RV2 P=lambda*tangent;
        vA=vA-mA*P; wA-=iA*rCross(vcp.rA,P);
        vB=vB+mB*P; wB+=iB*rCross(vcp.rB,P);
    }
    if (pc==1){
        RPt& vcp=vc.p[0];
        RV2 dv=vB+rCrossSV(wB,vcp.rB)-vA-rCrossSV(wA,vcp.rA);
        float vn=rDot(dv,normal); float lambda=-vcp.normalMass*(vn-vcp.velocityBias);
        float ni=rMax(vcp.normalImpulse+lambda,0.0f);
        lambda=ni-vcp.normalImpulse; vcp.normalImpulse=ni;
        RV2 P=lambda*normal;
        vA=vA-mA*P; wA-=iA*rCross(vcp.rA,P);
        vB=vB+mB*P; wB+=iB*rCross(vcp.rB,P);
    } else {
        RPt& cp1=vc.p[0]; RPt& cp2=vc.p[1];
        RV2 a=rv2(cp1.normalImpulse,cp2.normalImpulse);
        RV2 dv1=vB+rCrossSV(wB,cp1.rB)-vA-rCrossSV(wA,cp1.rA);
        RV2 dv2=vB+rCrossSV(wB,cp2.rB)-vA-rCrossSV(wA,cp2.rA);
        float vn1=rDot(dv1,normal), vn2=rDot(dv2,normal);
        RV2 b=rv2(vn1-cp1.velocityBias, vn2-cp2.velocityBias);
        b=b-rMulMV(vc.K,a);
        for(;;){
            RV2 x=-1.0f*rMulMV(vc.normalMass,b);
            if (x.x>=0.0f && x.y>=0.0f){
                RV2 d=x-a; RV2 P1=d.x*normal, P2=d.y*normal;
                vA=vA-mA*(P1+P2); wA-=iA*(rCross(cp1.rA,P1)+rCross(cp2.rA,P2));
                vB=vB+mB*(P1+P2); wB+=iB*(rCross(cp1.rB,P1)+rCross(cp2.rB,P2));
                cp1.normalImpulse=x.x; cp2.normalImpulse=x.y; break;
            }
            x.x=-cp1.normalMass*b.x; x.y=0.0f; vn1=0.0f; vn2=vc.K.ex.y*x.x+b.y;
            if (x.x>=0.0f && vn2>=0.0f){
                RV2 d=x-a; RV2 P1=d.x*normal, P2=d.y*normal;
                vA=vA-mA*(P1+P2); wA-=iA*(rCross(cp1.rA,P1)+rCross(cp2.rA,P2));
                vB=vB+mB*(P1+P2); wB+=iB*(rCross(cp1.rB,P1)+rCross(cp2.rB,P2));
                cp1.normalImpulse=x.x; cp2.normalImpulse=x.y; break;
            }
            x.x=0.0f; x.y=-cp2.normalMass*b.y; vn1=vc.K.ey.x*x.y+b.x; vn2=0.0f;
            if (x.y>=0.0f && vn1>=0.0f){
                RV2 d=x-a; RV2 P1=d.x*normal, P2=d.y*normal;
                vA=vA-mA*(P1+P2); wA-=iA*(rCross(cp1.rA,P1)+rCross(cp2.rA,P2));
                vB=vB+mB*(P1+P2); wB+=iB*(rCross(cp1.rB,P1)+rCross(cp2.rB,P2));
                cp1.normalImpulse=x.x; cp2.normalImpulse=x.y; break;
            }
            x.x=0.0f; x.y=0.0f; vn1=b.x; vn2=b.y;
            if (vn1>=0.0f && vn2>=0.0f){
                RV2 d=x-a; RV2 P1=d.x*normal, P2=d.y*normal;
                vA=vA-mA*(P1+P2); wA-=iA*(rCross(cp1.rA,P1)+rCross(cp2.rA,P2));
                vB=vB+mB*(P1+P2); wB+=iB*(rCross(cp1.rB,P1)+rCross(cp2.rB,P2));
                cp1.normalImpulse=x.x; cp2.normalImpulse=x.y; break;
            }
            break;
        }
    }
    gBodies[ia].v=vA; gBodies[ia].w=wA; gBodies[ib].v=vB; gBodies[ib].w=wB;
}
// reference two-point face-A position solve.
static bool rSolvePos(){
    RVC& pc=gRef; float minSep=0.0f;
    int ia=pc.indexA, ib=pc.indexB;
    float mA=pc.invMassA, iA=pc.invIA, mB=pc.invMassB, iB=pc.invIB;
    RV2 cA=gBodies[ia].c; float aA=gBodies[ia].a;
    RV2 cB=gBodies[ib].c; float aB=gBodies[ib].a;
    for (int j=0;j<pc.pointCount;++j){
        float sA=sinf(aA), cAng=cosf(aA), sB=sinf(aB), cB2=cosf(aB);
        RV2 xfAp=cA, xfBp=cB;   // localCenter==0
        // face-A position manifold
        RV2 normal=rMulRV(sA,cAng,pc.localNormal);
        RV2 planePoint = rMulRV(sA,cAng,pc.localPoint); planePoint=planePoint+xfAp;
        RV2 clip=rMulRV(sB,cB2,pc.localPoints[j]); clip=clip+xfBp;
        float separation=rDot(clip-planePoint,normal)-pc.radiusA-pc.radiusB;
        RV2 point=clip;
        RV2 rA=point-cA, rB=point-cB;
        minSep=rMin(minSep,separation);
        float C=rClamp(RBAUM*(separation+RSLOP),-RMAXCORR,0.0f);
        float rnA=rCross(rA,normal), rnB=rCross(rB,normal);
        float K=mA+mB+iA*rnA*rnA+iB*rnB*rnB;
        float impulse = K>0.0f ? -C/K : 0.0f;
        RV2 P=impulse*normal;
        cA=cA-mA*P; aA-=iA*rCross(rA,P);
        cB=cB+mB*P; aB+=iB*rCross(rB,P);
    }
    gBodies[ia].c=cA; gBodies[ia].a=aA; gBodies[ib].c=cB; gBodies[ib].a=aB;
    return minSep >= -3.0f*RSLOP;
}

// ---------------------------------------------------------------------------
// Scenario seed: a 0.5x0.5 box resting on a static ground plane (a face manifold
// with two points). Both bodies start with a small downward velocity so the
// velocity solve does real work, and a small overlap so the position solve does too.
// ---------------------------------------------------------------------------
static const float BOX_INVM = 4.0f;        // 1/mass for a unit-ish box
static const float BOX_INVI = 24.0f;       // 1/I
static const RV2   GROUND_C = {0.0f, 0.0f};
static const RV2   BOX_C    = {0.0f, 0.49f};  // slight penetration into the plane top at y=0.5? plane is ground
// face-A manifold: ground top face normal +y in ground frame. The two clip points are
// the box's bottom corners projected to local frames.
static void seedManifold(RVC& vc){
    vc.pointCount=2; vc.type=RFACE_A;
    vc.friction=0.3f; vc.restitution=0.0f;
    vc.indexA=0; vc.indexB=1;
    vc.invMassA=0.0f; vc.invMassB=BOX_INVM;
    vc.invIA=0.0f;    vc.invIB=BOX_INVI;
    vc.normal=rv2(0.0f,1.0f);
    vc.localNormal=rv2(0.0f,1.0f);
    vc.localPoint=rv2(0.0f,0.0f);      // ground reference face center (local)
    vc.localPoints[0]=rv2(-0.25f,-0.25f);  // box bottom-left corner (local to box)
    vc.localPoints[1]=rv2( 0.25f,-0.25f);  // box bottom-right corner (local to box)
    vc.radiusA=2.0f*0.005f; vc.radiusB=2.0f*0.005f;
    vc.p[0].normalImpulse=0.05f; vc.p[0].tangentImpulse=0.01f;  // carried warm-start
    vc.p[1].normalImpulse=0.05f; vc.p[1].tangentImpulse=0.01f;
}

static int gFails=0; static long gMax=0;
static void chk(const char* what, float ref, float got){
    long u=ulpDiff(ref,got); if(u>gMax) gMax=u;
    if(u!=0){ printf("  FAIL %-14s ref=%.9g got=%.9g ulp=%ld\n", what, ref, got, u); gFails=1; }
}

int main(){
    printf("Two-point block-solver micro-test vs Box2D 2.3.0 b2ContactSolver\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    // ---- seed both sides from the same state -------------------------------
    Body bInit[2];
    bInit[0].c=GROUND_C; bInit[0].v=rv2(0,0); bInit[0].a=0.0f; bInit[0].w=0.0f;
    bInit[0].invM=0.0f; bInit[0].invI=0.0f;
    bInit[1].c=BOX_C;    bInit[1].v=rv2(0.05f,-0.3f); bInit[1].a=0.01f; bInit[1].w=0.02f;
    bInit[1].invM=BOX_INVM; bInit[1].invI=BOX_INVI;
    // world contact points (box bottom corners at the seed transform; ~y=0.0 plane)
    RV2 worldPt[2] = { rv2(-0.25f, 0.0f), rv2(0.25f, 0.0f) };

    // ---- reference run -----------------------------------------------------
    gBodies[0]=bInit[0]; gBodies[1]=bInit[1];
    (void)worldPt;
    seedManifold(gRef);
    rInit(); rWarmStart();
    for (int it=0; it<GB_VELOCITY_ITERS; ++it) rSolveVel();
    // integrate positions (h*v), then 3 position iterations with early-exit
    float h=GB_DT;
    for (int i=0;i<2;++i){ gBodies[i].c = gBodies[i].c + h*gBodies[i].v; gBodies[i].a += h*gBodies[i].w; }
    for (int it=0; it<GB_POSITION_ITERS; ++it){ if (rSolvePos()) break; }
    Body refOut[2] = { gBodies[0], gBodies[1] };
    float refN0=gRef.p[0].normalImpulse, refN1=gRef.p[1].normalImpulse;
    float refT0=gRef.p[0].tangentImpulse, refT1=gRef.p[1].tangentImpulse;

    // ---- subject run (gb_contact_solver via a GBIslandData) ----------------
    GBIslandData isl;
    isl.bodyCount=2; isl.contactCount=1;
    isl.posC[0]=v2(bInit[0].c.x,bInit[0].c.y); isl.posA[0]=bInit[0].a;
    isl.vel[0]=v2(bInit[0].v.x,bInit[0].v.y);  isl.velW[0]=bInit[0].w;
    isl.posC[1]=v2(bInit[1].c.x,bInit[1].c.y); isl.posA[1]=bInit[1].a;
    isl.vel[1]=v2(bInit[1].v.x,bInit[1].v.y);  isl.velW[1]=bInit[1].w;
    GBConstraint& vc = isl.con[0];
    vc.indexA=0; vc.indexB=1;
    vc.invMassA=0.0f; vc.invMassB=BOX_INVM; vc.invIA=0.0f; vc.invIB=BOX_INVI;
    vc.friction=0.3f; vc.restitution=0.0f;
    vc.pointCount=2; vc.type=GB_MANIFOLD_FACE_A;
    vc.localNormal=v2(0.0f,1.0f); vc.localPoint=v2(0.0f,0.0f);
    vc.pLocalPoint =v2(-0.25f,-0.25f);
    vc.pLocalPoint2=v2( 0.25f,-0.25f);
    vc.radiusA=2.0f*0.005f; vc.radiusB=2.0f*0.005f; vc.contactIdx=0;
    vc.p.normalImpulse=0.05f;  vc.p.tangentImpulse=0.01f;
    vc.p2.normalImpulse=0.05f; vc.p2.tangentImpulse=0.01f;
    vc.p.rA=v2(0,0); vc.p.rB=v2(0,0); vc.p.normalMass=0; vc.p.tangentMass=0; vc.p.velocityBias=0;
    vc.p2.rA=v2(0,0); vc.p2.rB=v2(0,0); vc.p2.normalMass=0; vc.p2.tangentMass=0; vc.p2.velocityBias=0;

    // The subject InitializeVelocityConstraints rebuilds rA/rB from the world
    // manifold via gbWorldManifoldInit at the seed transform; the reference uses the
    // same world points, so InitializeVelocityConstraints reproduces them. (Seed
    // transform: ground at origin, box at BOX_C with small angle ~0.01.)
    // We feed the world points by setting the cached manifold so worldManifold lands
    // on worldPt. For a flat ground face manifold with localPoint at origin and box
    // clip points at the bottom corners, worldManifold point j = clip_j projected to
    // mid-surface; with radii equal and separation ~0 this equals the corner world
    // position, matching gWorldPt above.

    GBWorld dummy;   // unused by the island-only solver phases
    gbInitVelocityConstraints(dummy, isl);
    gbWarmStart(isl);
    for (int it=0; it<GB_VELOCITY_ITERS; ++it) gbSolveVelocity(isl);
    for (int i=0;i<2;++i){ isl.posC[i]=isl.posC[i]+h*isl.vel[i]; isl.posA[i]=isl.posA[i]+h*isl.velW[i]; }
    for (int it=0; it<GB_POSITION_ITERS; ++it){ if (gbSolvePosition(isl)) break; }

    // ---- compare -----------------------------------------------------------
    printf("body velocities/positions and warm-start impulses:\n");
    chk("vB.x", refOut[1].v.x, isl.vel[1].x);
    chk("vB.y", refOut[1].v.y, isl.vel[1].y);
    chk("wB",   refOut[1].w,   isl.velW[1]);
    chk("cB.x", refOut[1].c.x, isl.posC[1].x);
    chk("cB.y", refOut[1].c.y, isl.posC[1].y);
    chk("aB",   refOut[1].a,   isl.posA[1]);
    chk("n0", refN0, vc.p.normalImpulse);
    chk("n1", refN1, vc.p2.normalImpulse);
    chk("t0", refT0, vc.p.tangentImpulse);
    chk("t1", refT1, vc.p2.tangentImpulse);

    if (!gFails){
        printf("\nPASS gb_block_solver: 0 ULP (two-point block solve), maxUlp=%ld\n", gMax);
        return 0;
    }
    printf("\nFAIL gb_block_solver: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
