// gb_polygon_test.cu. Micro-test for gb_polygon.cuh and the polygon narrow-phase in
// gb_collision.cuh. 0-ULP versus a self-contained Box2D 2.3.0 reference.
//
// The test embeds the Box2D 2.3.0 polygon reference (Ref-prefixed types and
// functions: b2PolygonShape::ComputeMass / SetAsBox, b2CollidePolygons,
// b2CollidePolygonAndCircle, and their support routines) so it builds and runs from
// this repository alone. Each subject value is compared in ULP against the reference
// value for the same fixed input.
//
// Checks:
//   (A) box mass formula:      gbPolygonComputeMass vs the reference, 0 ULP.
//   (B) box-on-box manifold:   gbCollidePolygons two-point manifold vs the reference,
//       0 ULP on the local normal, plane point, both clip points, and the ids.
//   (C) polygon-circle:        gbCollidePolygonAndCircle one-point manifold, 0 ULP.
//
// Build (frozen flags), self-contained:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -Iinclude -Itest test/gb_polygon_test.cu -o test/gb_polygon_test
//   ./test/gb_polygon_test
//   Expected: PASS gb_polygon: 0 ULP (mass, polygon-polygon, polygon-circle)
//
// The algorithm is __host__ __device__, so the host path is sufficient for the
// bit-exact comparison under the frozen flags.
#include "gpu_box2d/gb_collision.cuh"   // gbPolygon*, gbCollidePolygons, gbCollidePolygonAndCircle
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
// Box2D 2.3.0 polygon reference (Ref-prefixed). A faithful copy of the upstream
// source for the functions under test.
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
static inline float rDistSq(RV2 a, RV2 b){ RV2 c=a-b; return rDot(c,c); }
static inline float rLen(RV2 v){ return sqrtf(v.x*v.x + v.y*v.y); }
static inline void rNorm(RV2& v){
    float len = rLen(v);
    if (len < 1.19209290e-07f) return;
    float inv = 1.0f/len; v.x*=inv; v.y*=inv;
}
struct RRot { float s, c; };
static inline RRot rRot(float a){ RRot q; q.s=sinf(a); q.c=cosf(a); return q; }
struct RXf { RV2 p; RRot q; };
static inline RV2 rMulRV(RRot q, RV2 v){ return rv2(q.c*v.x - q.s*v.y, q.s*v.x + q.c*v.y); }
static inline RV2 rMulTRV(RRot q, RV2 v){ return rv2(q.c*v.x + q.s*v.y, -q.s*v.x + q.c*v.y); }
static inline RV2 rMulXV(RXf t, RV2 v){
    return rv2((t.q.c*v.x - t.q.s*v.y) + t.p.x, (t.q.s*v.x + t.q.c*v.y) + t.p.y);
}
static inline RV2 rMulTXV(RXf t, RV2 v){
    float px=v.x-t.p.x, py=v.y-t.p.y;
    return rv2(t.q.c*px + t.q.s*py, -t.q.s*px + t.q.c*py);
}
static const float RMAXFLOAT = 3.40282347e+38f;
static const float REPSILON  = 1.19209290e-07f;
static const float RPOLYRAD  = 2.0f * 0.005f;

#define RMAXV 8
struct RPoly { RV2 vertices[RMAXV], normals[RMAXV], centroid; int count; float radius; };

static void rSetAsBox(RPoly& p, float hx, float hy){
    p.count=4;
    p.vertices[0]=rv2(-hx,-hy); p.vertices[1]=rv2(hx,-hy);
    p.vertices[2]=rv2(hx,hy);   p.vertices[3]=rv2(-hx,hy);
    p.normals[0]=rv2(0,-1); p.normals[1]=rv2(1,0); p.normals[2]=rv2(0,1); p.normals[3]=rv2(-1,0);
    p.centroid=rv2(0,0); p.radius=RPOLYRAD;
}
struct RMass { float mass; RV2 center; float I; };
static void rComputeMass(const RPoly& p, RMass& md, float density){
    RV2 center=rv2(0,0); float area=0.0f, I=0.0f; RV2 s=rv2(0,0);
    for (int i=0;i<p.count;++i) s = s + p.vertices[i];
    s = (1.0f/p.count) * s;
    const float k_inv3 = 1.0f/3.0f;
    for (int i=0;i<p.count;++i){
        RV2 e1 = p.vertices[i]-s;
        RV2 e2 = i+1<p.count ? p.vertices[i+1]-s : p.vertices[0]-s;
        float D = rCross(e1,e2);
        float ta = 0.5f*D; area += ta;
        center = center + (ta*k_inv3)*(e1+e2);
        float ex1=e1.x, ey1=e1.y, ex2=e2.x, ey2=e2.y;
        float intx2 = ex1*ex1 + ex2*ex1 + ex2*ex2;
        float inty2 = ey1*ey1 + ey2*ey1 + ey2*ey2;
        I += (0.25f*k_inv3*D)*(intx2+inty2);
    }
    md.mass = density*area;
    center = (1.0f/area)*center;
    md.center = center + s;
    md.I = density*I;
    md.I += md.mass*(rDot(md.center,md.center) - rDot(center,center));
}
struct RClip { RV2 v; int iA,iB,tA,tB; };
static int rClipSeg(RClip vOut[2], const RClip vIn[2], RV2 normal, float offset, int vidxA){
    int numOut=0;
    float d0=rDot(normal,vIn[0].v)-offset, d1=rDot(normal,vIn[1].v)-offset;
    if (d0<=0.0f) vOut[numOut++]=vIn[0];
    if (d1<=0.0f) vOut[numOut++]=vIn[1];
    if (d0*d1<0.0f){
        float interp=d0/(d0-d1);
        vOut[numOut].v = vIn[0].v + interp*(vIn[1].v-vIn[0].v);
        vOut[numOut].iA=vidxA; vOut[numOut].iB=vIn[0].iB; vOut[numOut].tA=0; vOut[numOut].tB=1;
        ++numOut;
    }
    return numOut;
}
static float rEdgeSep(const RPoly& p1, RXf xf1, int e1, const RPoly& p2, RXf xf2){
    RV2 n1w = rMulRV(xf1.q, p1.normals[e1]);
    RV2 n1 = rMulTRV(xf2.q, n1w);
    int idx=0; float minDot=RMAXFLOAT;
    for (int i=0;i<p2.count;++i){ float d=rDot(p2.vertices[i],n1); if(d<minDot){minDot=d; idx=i;} }
    RV2 v1=rMulXV(xf1,p1.vertices[e1]); RV2 v2=rMulXV(xf2,p2.vertices[idx]);
    return rDot(v2-v1,n1w);
}
static float rFindMaxSep(int& ei, const RPoly& p1, RXf xf1, const RPoly& p2, RXf xf2){
    int c1=p1.count;
    RV2 d = rMulXV(xf2,p2.centroid) - rMulXV(xf1,p1.centroid);
    RV2 dL = rMulTRV(xf1.q,d);
    int edge=0; float maxDot=-RMAXFLOAT;
    for (int i=0;i<c1;++i){ float dt=rDot(p1.normals[i],dL); if(dt>maxDot){maxDot=dt; edge=i;} }
    float s=rEdgeSep(p1,xf1,edge,p2,xf2);
    int pe=edge-1>=0?edge-1:c1-1; float sp=rEdgeSep(p1,xf1,pe,p2,xf2);
    int ne=edge+1<c1?edge+1:0;    float sn=rEdgeSep(p1,xf1,ne,p2,xf2);
    int be; float bs; int inc;
    if (sp>s && sp>sn){ inc=-1; be=pe; bs=sp; }
    else if (sn>s){ inc=1; be=ne; bs=sn; }
    else { ei=edge; return s; }
    for(;;){
        if(inc==-1) edge=be-1>=0?be-1:c1-1; else edge=be+1<c1?be+1:0;
        s=rEdgeSep(p1,xf1,edge,p2,xf2);
        if(s>bs){ be=edge; bs=s; } else break;
    }
    ei=be; return bs;
}
static void rFindIncident(RClip c[2], const RPoly& p1, RXf xf1, int e1, const RPoly& p2, RXf xf2){
    RV2 n1 = rMulTRV(xf2.q, rMulRV(xf1.q, p1.normals[e1]));
    int idx=0; float minDot=RMAXFLOAT;
    for (int i=0;i<p2.count;++i){ float d=rDot(n1,p2.normals[i]); if(d<minDot){minDot=d; idx=i;} }
    int i1=idx, i2=i1+1<p2.count?i1+1:0;
    c[0].v=rMulXV(xf2,p2.vertices[i1]); c[0].iA=e1; c[0].iB=i1; c[0].tA=1; c[0].tB=0;
    c[1].v=rMulXV(xf2,p2.vertices[i2]); c[1].iA=e1; c[1].iB=i2; c[1].tA=1; c[1].tB=0;
}
enum { RMAN_CIRCLES=0, RMAN_FACEA=1, RMAN_FACEB=2 };
struct RMan { int type, pointCount; RV2 localNormal, localPoint, pLocal[2]; unsigned id[2]; };
static unsigned rKey(int iA,int iB,int tA,int tB){
    return ((unsigned)(iA&0xff)) | ((unsigned)(iB&0xff)<<8)
         | ((unsigned)(tA&0xff)<<16) | ((unsigned)(tB&0xff)<<24);
}
static void rCollidePolys(RMan& m, const RPoly& pA, RXf xfA, const RPoly& pB, RXf xfB){
    m.pointCount=0;
    float tr = pA.radius + pB.radius;
    int eA=0; float sA=rFindMaxSep(eA,pA,xfA,pB,xfB); if(sA>tr) return;
    int eB=0; float sB=rFindMaxSep(eB,pB,xfB,pA,xfA); if(sB>tr) return;
    const RPoly* p1; const RPoly* p2; RXf x1,x2; int e1; int flip;
    const float kr=0.98f, ka=0.001f;
    if (sB > kr*sA + ka){ p1=&pB; p2=&pA; x1=xfB; x2=xfA; e1=eB; m.type=RMAN_FACEB; flip=1; }
    else { p1=&pA; p2=&pB; x1=xfA; x2=xfB; e1=eA; m.type=RMAN_FACEA; flip=0; }
    RClip inc[2]; rFindIncident(inc,*p1,x1,e1,*p2,x2);
    int c1=p1->count; int iv1=e1, iv2=e1+1<c1?e1+1:0;
    RV2 v11=p1->vertices[iv1], v12=p1->vertices[iv2];
    RV2 lt=v12-v11; rNorm(lt);
    RV2 ln=rCrossVS(lt,1.0f);
    RV2 pp=0.5f*(v11+v12);
    RV2 tan=rMulRV(x1.q,lt); RV2 nrm=rCrossVS(tan,1.0f);
    v11=rMulXV(x1,v11); v12=rMulXV(x1,v12);
    float fo=rDot(nrm,v11);
    float so1=-rDot(tan,v11)+tr, so2=rDot(tan,v12)+tr;
    RClip cp1[2], cp2[2]; int np;
    np=rClipSeg(cp1,inc,-tan,so1,iv1); if(np<2) return;
    np=rClipSeg(cp2,cp1,tan,so2,iv2); if(np<2) return;
    m.localNormal=ln; m.localPoint=pp;
    int pc=0;
    for (int i=0;i<2;++i){
        float sep=rDot(nrm,cp2[i].v)-fo;
        if(sep<=tr){
            RV2 lp=rMulTXV(x2,cp2[i].v);
            unsigned key = flip ? rKey(cp2[i].iB,cp2[i].iA,cp2[i].tB,cp2[i].tA)
                                : rKey(cp2[i].iA,cp2[i].iB,cp2[i].tA,cp2[i].tB);
            m.pLocal[pc]=lp; m.id[pc]=key; ++pc;
        }
    }
    m.pointCount=pc;
}
static void rCollidePolyCircle(RMan& m, const RPoly& pA, RXf xfA, float cr, RXf xfB){
    m.pointCount=0;
    RV2 c=rMulXV(xfB,rv2(0,0)); RV2 cL=rMulTXV(xfA,c);
    int ni=0; float sep=-RMAXFLOAT; float radius=pA.radius+cr;
    for (int i=0;i<pA.count;++i){ float s=rDot(pA.normals[i],cL-pA.vertices[i]); if(s>radius) return; if(s>sep){sep=s; ni=i;} }
    int v1i=ni, v2i=v1i+1<pA.count?v1i+1:0;
    RV2 v1=pA.vertices[v1i], v2=pA.vertices[v2i];
    if (sep<REPSILON){
        m.pointCount=1; m.type=RMAN_FACEA; m.localNormal=pA.normals[ni];
        m.localPoint=0.5f*(v1+v2); m.pLocal[0]=rv2(0,0); m.id[0]=0; return;
    }
    float u1=rDot(cL-v1,v2-v1), u2=rDot(cL-v2,v1-v2);
    if (u1<=0.0f){
        if (rDistSq(cL,v1)>radius*radius) return;
        m.pointCount=1; m.type=RMAN_FACEA; RV2 n=cL-v1; rNorm(n);
        m.localNormal=n; m.localPoint=v1; m.pLocal[0]=rv2(0,0); m.id[0]=0;
    } else if (u2<=0.0f){
        if (rDistSq(cL,v2)>radius*radius) return;
        m.pointCount=1; m.type=RMAN_FACEA; RV2 n=cL-v2; rNorm(n);
        m.localNormal=n; m.localPoint=v2; m.pLocal[0]=rv2(0,0); m.id[0]=0;
    } else {
        RV2 fc=0.5f*(v1+v2); float s2=rDot(cL-fc,pA.normals[v1i]);
        if (s2>radius) return;
        m.pointCount=1; m.type=RMAN_FACEA; m.localNormal=pA.normals[v1i];
        m.localPoint=fc; m.pLocal[0]=rv2(0,0); m.id[0]=0;
    }
}

// ---------------------------------------------------------------------------
// Test harness.
// ---------------------------------------------------------------------------
static int gFails = 0;
static long gMax = 0;
static void chk(const char* what, float ref, float got){
    long u = ulpDiff(ref, got);
    if (u > gMax) gMax = u;
    if (u != 0){ printf("  FAIL %-22s ref=%.9g got=%.9g ulp=%ld\n", what, ref, got, u); gFails=1; }
}
static void chkInt(const char* what, long ref, long got){
    if (ref != got){ printf("  FAIL %-22s ref=%ld got=%ld\n", what, ref, got); gFails=1; }
}

// Build matching subject/reference boxes from the same half-extents.
static void makeBoxSubject(GBPolygon& p, float hx, float hy){ gbPolygonSetAsBox(p, hx, hy); }
static void makeBoxRef(RPoly& p, float hx, float hy){ rSetAsBox(p, hx, hy); }

static Xf  sXf(float x, float y, float a){ Xf  t; t.p=v2(x,y);  t.q=rotSet(a); return t; }
static RXf rXf(float x, float y, float a){ RXf t; t.p=rv2(x,y); t.q=rRot(a);   return t; }

int main(){
    printf("Polygon micro-test: gb_polygon + gbCollidePolygons/AndCircle vs Box2D 2.3.0\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    // (A) box mass formula.
    {
        GBPolygon sp; makeBoxSubject(sp, 0.5f, 0.7f);
        RPoly rp; makeBoxRef(rp, 0.5f, 0.7f);
        GBMassData sm; gbPolygonComputeMass(sp, sm, 2.5f);
        RMass rm; rComputeMass(rp, rm, 2.5f);
        printf("(A) box mass (hx=0.5 hy=0.7 density=2.5)\n");
        chk("mass", rm.mass, sm.mass);
        chk("center.x", rm.center.x, sm.center.x);
        chk("center.y", rm.center.y, sm.center.y);
        chk("I", rm.I, sm.I);
    }

    // (B) box on box: a small box resting at an angle on a wide ground box,
    // overlapping to produce a two-point face manifold.
    {
        GBPolygon sg; makeBoxSubject(sg, 5.0f, 0.5f);   // ground box
        RPoly    rg; makeBoxRef(rg, 5.0f, 0.5f);
        GBPolygon sb; makeBoxSubject(sb, 0.5f, 0.5f);   // resting box
        RPoly    rb; makeBoxRef(rb, 0.5f, 0.5f);
        Xf  sgx=sXf(0,0,0), sbx=sXf(0.1f, 0.94f, 0.02f);
        RXf rgx=rXf(0,0,0), rbx=rXf(0.1f, 0.94f, 0.02f);
        GBManifold sm; gbCollidePolygons(sm, sg, sgx, sb, sbx);
        RMan rm; rCollidePolys(rm, rg, rgx, rb, rbx);
        printf("(B) box-on-box manifold (pointCount ref=%d got=%d)\n", rm.pointCount, sm.pointCount);
        chkInt("pointCount", rm.pointCount, sm.pointCount);
        chkInt("type", rm.type, sm.type);
        chk("localNormal.x", rm.localNormal.x, sm.localNormal.x);
        chk("localNormal.y", rm.localNormal.y, sm.localNormal.y);
        chk("localPoint.x", rm.localPoint.x, sm.localPoint.x);
        chk("localPoint.y", rm.localPoint.y, sm.localPoint.y);
        if (rm.pointCount == sm.pointCount && sm.pointCount >= 1){
            chk("p0.x", rm.pLocal[0].x, sm.pLocalPoint.x);
            chk("p0.y", rm.pLocal[0].y, sm.pLocalPoint.y);
            chkInt("id0", (long)rm.id[0], (long)sm.id0);
        }
        if (rm.pointCount == sm.pointCount && sm.pointCount >= 2){
            chk("p1.x", rm.pLocal[1].x, sm.pLocalPoint2.x);
            chk("p1.y", rm.pLocal[1].y, sm.pLocalPoint2.y);
            chkInt("id1", (long)rm.id[1], (long)sm.id1);
        }
    }

    // (C) polygon vs circle: a circle resting on the top face of a box.
    {
        GBPolygon sb; makeBoxSubject(sb, 1.0f, 0.5f);
        RPoly    rb; makeBoxRef(rb, 1.0f, 0.5f);
        Xf  sbx=sXf(0,0,0), scx=sXf(0.2f, 0.9f, 0.0f);
        RXf rbx=rXf(0,0,0), rcx=rXf(0.2f, 0.9f, 0.0f);
        float circR=0.45f;
        GBManifold sm; gbCollidePolygonAndCircle(sm, sb, sbx, circR, scx);
        RMan rm; rCollidePolyCircle(rm, rb, rbx, circR, rcx);
        printf("(C) polygon-circle manifold (pointCount ref=%d got=%d)\n", rm.pointCount, sm.pointCount);
        chkInt("pointCount", rm.pointCount, sm.pointCount);
        chkInt("type", rm.type, sm.type);
        chk("localNormal.x", rm.localNormal.x, sm.localNormal.x);
        chk("localNormal.y", rm.localNormal.y, sm.localNormal.y);
        chk("localPoint.x", rm.localPoint.x, sm.localPoint.x);
        chk("localPoint.y", rm.localPoint.y, sm.localPoint.y);
        if (rm.pointCount == sm.pointCount && sm.pointCount >= 1){
            chk("p0.x", rm.pLocal[0].x, sm.pLocalPoint.x);
            chk("p0.y", rm.pLocal[0].y, sm.pLocalPoint.y);
        }
    }

    if (!gFails){
        printf("\nPASS gb_polygon: 0 ULP (mass, polygon-polygon, polygon-circle), maxUlp=%ld\n", gMax);
        return 0;
    }
    printf("\nFAIL gb_polygon: see above (maxUlp=%ld)\n", gMax);
    return 1;
}
