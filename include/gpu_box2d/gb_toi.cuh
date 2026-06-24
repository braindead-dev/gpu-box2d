// gb_toi.cuh. The continuous-collision path: GJK distance (b2Distance) and
// b2TimeOfImpact, written against the gb_pools.cuh accessor contract
// (BODY/CONT/EDGE/SCAL, GBWorld).
//
// STATUS: validated. The GJK distance and b2TimeOfImpact results match Box2D
// 2.3.0 bit-for-bit on the fruit-wall CCD scenario (see test/gb_toi_test.cu,
// 0-ULP). Box2D's CCD fires on dynamic-static contacts during settling and is
// outcome-affecting, so a faithful engine must include it.
//
// Self-contained: includes only gb_pools.cuh (which pulls in gb_math.cuh and
// gb_settings.cuh). The full SolveTOI driver (gbIslandSolveTOI / gbWorldSolveTOI)
// needs the narrow-phase, solver, and island modules and is gated behind
// #ifdef GB_TOI_FULL. Include those headers, then define GB_TOI_FULL.
//
// Compile flags (FROZEN): --fmad=false -prec-div=true -prec-sqrt=true
#pragma once
#include "gb_pools.cuh"

// ---------------------------------------------------------------------------
// b2Sweep, per-body sweep for CCD interpolation.
// localCenter == (0,0) for all circles and for the static edge ground body.
// ---------------------------------------------------------------------------
struct GBSweep {
    V2 localCenter;
    V2 c0, c;
    float a0, a;
    float alpha0;
};

GB_HD inline void gbSweepGetTransform(const GBSweep& s, Xf& xf, float beta){
    xf.p = (1.0f-beta)*s.c0 + beta*s.c;
    float angle = (1.0f-beta)*s.a0 + beta*s.a;
    xf.q = rotSet(angle);
    xf.p = xf.p - b2MulRV(xf.q, s.localCenter);
}
GB_HD inline void gbSweepAdvance(GBSweep& s, float alpha){
    float beta = (alpha - s.alpha0) / (1.0f - s.alpha0);
    s.c0 = (1.0f-beta)*s.c0 + beta*s.c;
    s.a0 = (1.0f-beta)*s.a0 + beta*s.a;
    s.alpha0 = alpha;
}
GB_HD inline void gbSweepNormalize(GBSweep& s){
    float twoPi = 2.0f * GB_PI;
    float d = twoPi * floorf(s.a0 / twoPi);
    s.a0 -= d; s.a -= d;
}

// ---------------------------------------------------------------------------
// Distance proxy, edge (2 verts) or circle (1 vert, m_p==0).
// ---------------------------------------------------------------------------
struct GBDProxy { V2 v[2]; int count; float radius; };

GB_HD inline int gbProxySupport(const GBDProxy& p, V2 d){
    int best=0; float bv=b2Dot(p.v[0], d);
    for (int i=1;i<p.count;++i){ float val=b2Dot(p.v[i],d); if(val>bv){bv=val;best=i;} }
    return best;
}

// Build proxy from world fields via accessors.
// edge >= 0 and isA==true: the static edge side (ground fixture).
// otherwise: circle (fruit body, m_p == 0).
// fruitTier: call site supplies BODY(w,tier,body) to avoid needing tier_radius here.
GB_HD inline GBDProxy gbContactProxy(const GBWorld& w, int body, int edge, bool isA, float radius){
    GBDProxy p;
    if (edge >= 0 && isA){
        p.v[0] = v2(EDGE(w,edgeAx,edge), EDGE(w,edgeAy,edge));
        p.v[1] = v2(EDGE(w,edgeBx,edge), EDGE(w,edgeBy,edge));
        p.count = 2; p.radius = GB_POLYGON_RADIUS;
    } else {
        p.v[0] = v2(0.0f, 0.0f); p.count = 1; p.radius = radius;
    }
    return p;
}

// Read sweep from world body slot via accessors.
GB_HD inline GBSweep gbBodySweep(const GBWorld& w, int i){
    GBSweep s;
    s.localCenter = v2(0.0f, 0.0f);
    s.c0    = v2(BODY(w,sweepC0x,i), BODY(w,sweepC0y,i));
    s.c     = v2(BODY(w,sweepCx,i),  BODY(w,sweepCy,i));
    s.a0    = BODY(w,sweepA0,i);
    s.a     = BODY(w,sweepA,i);
    s.alpha0= BODY(w,sweepAlpha0,i);
    return s;
}

// Write back c0/a0/alpha0 after gbSweepAdvance.
GB_HD inline void gbWriteSweepAdvance(GBWorld& w, int i, const GBSweep& s){
    BODY(w,sweepC0x,i)   = s.c0.x;
    BODY(w,sweepC0y,i)   = s.c0.y;
    BODY(w,sweepA0,i)    = s.a0;
    BODY(w,sweepAlpha0,i)= s.alpha0;
}

// SynchronizeTransform: xf.q = Rot(sweepA); xf.p = sweepC (localCenter==0).
// The contact solver defines an identical helper. Guarded so the two compose in one
// translation unit: whichever header is included first defines it, the other yields.
#ifndef GB_SYNC_TRANSFORM_PROVIDED
#define GB_SYNC_TRANSFORM_PROVIDED 1
GB_HD inline void gbSyncTransform(GBWorld& w, int i){
    Rot q = rotSet(BODY(w,sweepA,i));
    BODY(w,xfQs,i) = q.s;
    BODY(w,xfQc,i) = q.c;
    BODY(w,xfPx,i) = BODY(w,sweepCx,i);
    BODY(w,xfPy,i) = BODY(w,sweepCy,i);
}
#endif

// ---------------------------------------------------------------------------
// GJK distance (faithful b2Distance). All math byte-identical to the Box2D 2.3.0 CCD path.
// ---------------------------------------------------------------------------
struct GBSimplexCache { float metric; int count; int indexA[3], indexB[3]; };
struct GBSVert { V2 wA, wB, w; float a; int indexA, indexB; };
struct GBSimplex {
    GBSVert v1, v2, v3; int count;
    GB_HD GBSVert& at(int i){ return i==0?v1:(i==1?v2:v3); }
};

GB_HD inline float gbSimplexMetric(GBSimplex& s){
    if (s.count==2) return b2Length(s.v1.w - s.v2.w);
    if (s.count==3) return b2Cross(s.v2.w - s.v1.w, s.v3.w - s.v1.w);
    return 0.0f;
}
GB_HD inline void gbSimplexReadCache(GBSimplex& s, const GBSimplexCache& cache,
        const GBDProxy& pA, Xf xfA, const GBDProxy& pB, Xf xfB){
    s.count = cache.count;
    for (int i=0;i<s.count;++i){
        GBSVert& v = s.at(i);
        v.indexA = cache.indexA[i]; v.indexB = cache.indexB[i];
        v.wA = b2MulTV(xfA, pA.v[v.indexA]);
        v.wB = b2MulTV(xfB, pB.v[v.indexB]);
        v.w = v.wB - v.wA; v.a = 0.0f;
    }
    if (s.count > 1){
        float metric1 = cache.metric, metric2 = gbSimplexMetric(s);
        if (metric2 < 0.5f*metric1 || 2.0f*metric1 < metric2 || metric2 < GB_EPSILON) s.count=0;
    }
    if (s.count == 0){
        GBSVert& v = s.at(0);
        v.indexA=0; v.indexB=0;
        v.wA = b2MulTV(xfA, pA.v[0]); v.wB = b2MulTV(xfB, pB.v[0]);
        v.w = v.wB - v.wA; v.a = 1.0f; s.count=1;
    }
}
GB_HD inline void gbSimplexWriteCache(GBSimplex& s, GBSimplexCache& cache){
    cache.metric = gbSimplexMetric(s); cache.count = s.count;
    for (int i=0;i<s.count;++i){ cache.indexA[i]=s.at(i).indexA; cache.indexB[i]=s.at(i).indexB; }
}
GB_HD inline V2 gbSimplexSearchDir(GBSimplex& s){
    if (s.count==1) return -s.v1.w;
    V2 e12 = s.v2.w - s.v1.w;
    float sgn = b2Cross(e12, -s.v1.w);
    if (sgn > 0.0f) return b2CrossSV(1.0f, e12);
    else            return b2CrossVS(e12, 1.0f);
}
GB_HD inline V2 gbSimplexClosest(GBSimplex& s){
    if (s.count==1) return s.v1.w;
    if (s.count==2) return s.v1.a*s.v1.w + s.v2.a*s.v2.w;
    return v2(0.0f,0.0f);
}
GB_HD inline void gbSimplexWitness(GBSimplex& s, V2& pA, V2& pB){
    if (s.count==1){ pA=s.v1.wA; pB=s.v1.wB; }
    else if (s.count==2){ pA=s.v1.a*s.v1.wA + s.v2.a*s.v2.wA; pB=s.v1.a*s.v1.wB + s.v2.a*s.v2.wB; }
    else { pA=s.v1.a*s.v1.wA + s.v2.a*s.v2.wA + s.v3.a*s.v3.wA; pB=pA; }
}
GB_HD inline void gbSimplexSolve2(GBSimplex& s){
    V2 w1=s.v1.w, w2=s.v2.w, e12=w2-w1;
    float d12_2 = -b2Dot(w1,e12);
    if (d12_2 <= 0.0f){ s.v1.a=1.0f; s.count=1; return; }
    float d12_1 = b2Dot(w2,e12);
    if (d12_1 <= 0.0f){ s.v2.a=1.0f; s.count=1; s.v1=s.v2; return; }
    float inv = 1.0f/(d12_1+d12_2);
    s.v1.a=d12_1*inv; s.v2.a=d12_2*inv; s.count=2;
}
GB_HD inline void gbSimplexSolve3(GBSimplex& s){
    V2 w1=s.v1.w, w2=s.v2.w, w3=s.v3.w;
    V2 e12=w2-w1; float w1e12=b2Dot(w1,e12), w2e12=b2Dot(w2,e12);
    float d12_1=w2e12, d12_2=-w1e12;
    V2 e13=w3-w1; float w1e13=b2Dot(w1,e13), w3e13=b2Dot(w3,e13);
    float d13_1=w3e13, d13_2=-w1e13;
    V2 e23=w3-w2; float w2e23=b2Dot(w2,e23), w3e23=b2Dot(w3,e23);
    float d23_1=w3e23, d23_2=-w2e23;
    float n123=b2Cross(e12,e13);
    float d123_1=n123*b2Cross(w2,w3), d123_2=n123*b2Cross(w3,w1), d123_3=n123*b2Cross(w1,w2);
    if (d12_2<=0.0f && d13_2<=0.0f){ s.v1.a=1.0f; s.count=1; return; }
    if (d12_1>0.0f && d12_2>0.0f && d123_3<=0.0f){ float inv=1.0f/(d12_1+d12_2); s.v1.a=d12_1*inv; s.v2.a=d12_2*inv; s.count=2; return; }
    if (d13_1>0.0f && d13_2>0.0f && d123_2<=0.0f){ float inv=1.0f/(d13_1+d13_2); s.v1.a=d13_1*inv; s.v3.a=d13_2*inv; s.count=2; s.v2=s.v3; return; }
    if (d12_1<=0.0f && d23_2<=0.0f){ s.v2.a=1.0f; s.count=1; s.v1=s.v2; return; }
    if (d13_1<=0.0f && d23_1<=0.0f){ s.v3.a=1.0f; s.count=1; s.v1=s.v3; return; }
    if (d23_1>0.0f && d23_2>0.0f && d123_1<=0.0f){ float inv=1.0f/(d23_1+d23_2); s.v2.a=d23_1*inv; s.v3.a=d23_2*inv; s.count=2; s.v1=s.v3; return; }
    float inv=1.0f/(d123_1+d123_2+d123_3); s.v1.a=d123_1*inv; s.v2.a=d123_2*inv; s.v3.a=d123_3*inv; s.count=3;
}

struct GBDistInput { GBDProxy proxyA, proxyB; Xf xfA, xfB; bool useRadii; };
struct GBDistOutput { V2 pointA, pointB; float distance; int iterations; };

GB_HD inline void gbDistanceGJK(GBDistOutput& out, GBSimplexCache& cache, const GBDistInput& in){
    const GBDProxy& pA=in.proxyA; const GBDProxy& pB=in.proxyB;
    Xf xfA=in.xfA, xfB=in.xfB;
    GBSimplex simplex; gbSimplexReadCache(simplex, cache, pA, xfA, pB, xfB);
    const int k_maxIters=20;
    int saveA[3], saveB[3], saveCount=0;
    float distanceSqr1=GB_MAXFLOAT, distanceSqr2=distanceSqr1;
    int iter=0;
    while (iter < k_maxIters){
        saveCount = simplex.count;
        for (int i=0;i<saveCount;++i){ saveA[i]=simplex.at(i).indexA; saveB[i]=simplex.at(i).indexB; }
        if (simplex.count==2) gbSimplexSolve2(simplex);
        else if (simplex.count==3) gbSimplexSolve3(simplex);
        if (simplex.count==3) break;
        V2 p = gbSimplexClosest(simplex);
        distanceSqr2 = b2Dot(p,p);
        distanceSqr1 = distanceSqr2;
        V2 d = gbSimplexSearchDir(simplex);
        if (b2Dot(d,d) < GB_EPSILON*GB_EPSILON) break;
        GBSVert& vert = simplex.at(simplex.count);
        vert.indexA = gbProxySupport(pA, b2MulTinvV_q(xfA.q, -d));
        vert.wA = b2MulTV(xfA, pA.v[vert.indexA]);
        vert.indexB = gbProxySupport(pB, b2MulTinvV_q(xfB.q, d));
        vert.wB = b2MulTV(xfB, pB.v[vert.indexB]);
        vert.w = vert.wB - vert.wA;
        ++iter;
        bool dup=false;
        for (int i=0;i<saveCount;++i){ if(vert.indexA==saveA[i] && vert.indexB==saveB[i]){dup=true;break;} }
        if (dup) break;
        ++simplex.count;
    }
    gbSimplexWitness(simplex, out.pointA, out.pointB);
    out.distance = b2Length(out.pointA - out.pointB);
    out.iterations = iter;
    gbSimplexWriteCache(simplex, cache);
    if (in.useRadii){
        float rA=pA.radius, rB=pB.radius;
        if (out.distance > rA+rB && out.distance > GB_EPSILON){
            out.distance -= rA+rB;
            V2 normal = out.pointB - out.pointA; b2Normalize(normal);
            out.pointA = out.pointA + rA*normal;
            out.pointB = out.pointB - rB*normal;
        } else {
            V2 p2 = 0.5f*(out.pointA + out.pointB);
            out.pointA=p2; out.pointB=p2; out.distance=0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Separation function (faithful b2SeparationFunction).
// ---------------------------------------------------------------------------
enum GBSepType { GB_SEP_POINTS=0, GB_SEP_FACEA=1, GB_SEP_FACEB=2 };
struct GBSepFn { GBDProxy proxyA, proxyB; GBSweep sweepA, sweepB; int type; V2 localPoint, axis; };

GB_HD inline float gbSepInit(GBSepFn& f, const GBSimplexCache& cache,
        const GBDProxy& pA, const GBSweep& sA, const GBDProxy& pB, const GBSweep& sB, float t1){
    f.proxyA=pA; f.proxyB=pB; f.sweepA=sA; f.sweepB=sB;
    int count = cache.count;
    Xf xfA, xfB; gbSweepGetTransform(sA, xfA, t1); gbSweepGetTransform(sB, xfB, t1);
    if (count==1){
        f.type=GB_SEP_POINTS;
        V2 pa = b2MulTV(xfA, pA.v[cache.indexA[0]]);
        V2 pb = b2MulTV(xfB, pB.v[cache.indexB[0]]);
        f.axis = pb - pa;
        float s = b2Normalize(f.axis);
        return s;
    } else if (cache.indexA[0]==cache.indexA[1]){
        f.type=GB_SEP_FACEB;
        V2 b1=pB.v[cache.indexB[0]], b2v=pB.v[cache.indexB[1]];
        f.axis = b2CrossVS(b2v-b1, 1.0f); b2Normalize(f.axis);
        V2 normal = b2MulRV(xfB.q, f.axis);
        f.localPoint = 0.5f*(b1+b2v);
        V2 pb = b2MulTV(xfB, f.localPoint);
        V2 pa = b2MulTV(xfA, pA.v[cache.indexA[0]]);
        float s = b2Dot(pa-pb, normal);
        if (s<0.0f){ f.axis=-f.axis; s=-s; }
        return s;
    } else {
        f.type=GB_SEP_FACEA;
        V2 a1=pA.v[cache.indexA[0]], a2=pA.v[cache.indexA[1]];
        f.axis = b2CrossVS(a2-a1, 1.0f); b2Normalize(f.axis);
        V2 normal = b2MulRV(xfA.q, f.axis);
        f.localPoint = 0.5f*(a1+a2);
        V2 pa = b2MulTV(xfA, f.localPoint);
        V2 pb = b2MulTV(xfB, pB.v[cache.indexB[0]]);
        float s = b2Dot(pb-pa, normal);
        if (s<0.0f){ f.axis=-f.axis; s=-s; }
        return s;
    }
}
GB_HD inline float gbSepFindMin(const GBSepFn& f, int& indexA, int& indexB, float t){
    Xf xfA, xfB; gbSweepGetTransform(f.sweepA, xfA, t); gbSweepGetTransform(f.sweepB, xfB, t);
    if (f.type==GB_SEP_POINTS){
        V2 axisA = b2MulTinvV_q(xfA.q,  f.axis);
        V2 axisB = b2MulTinvV_q(xfB.q, -f.axis);
        indexA = gbProxySupport(f.proxyA, axisA);
        indexB = gbProxySupport(f.proxyB, axisB);
        V2 pa = b2MulTV(xfA, f.proxyA.v[indexA]);
        V2 pb = b2MulTV(xfB, f.proxyB.v[indexB]);
        return b2Dot(pb-pa, f.axis);
    } else if (f.type==GB_SEP_FACEA){
        V2 normal = b2MulRV(xfA.q, f.axis);
        V2 pa = b2MulTV(xfA, f.localPoint);
        V2 axisB = b2MulTinvV_q(xfB.q, -normal);
        indexA=-1; indexB=gbProxySupport(f.proxyB, axisB);
        V2 pb = b2MulTV(xfB, f.proxyB.v[indexB]);
        return b2Dot(pb-pa, normal);
    } else {
        V2 normal = b2MulRV(xfB.q, f.axis);
        V2 pb = b2MulTV(xfB, f.localPoint);
        V2 axisA = b2MulTinvV_q(xfA.q, -normal);
        indexB=-1; indexA=gbProxySupport(f.proxyA, axisA);
        V2 pa = b2MulTV(xfA, f.proxyA.v[indexA]);
        return b2Dot(pa-pb, normal);
    }
}
GB_HD inline float gbSepEval(const GBSepFn& f, int indexA, int indexB, float t){
    Xf xfA, xfB; gbSweepGetTransform(f.sweepA, xfA, t); gbSweepGetTransform(f.sweepB, xfB, t);
    if (f.type==GB_SEP_POINTS){
        V2 pa = b2MulTV(xfA, f.proxyA.v[indexA]);
        V2 pb = b2MulTV(xfB, f.proxyB.v[indexB]);
        return b2Dot(pb-pa, f.axis);
    } else if (f.type==GB_SEP_FACEA){
        V2 normal = b2MulRV(xfA.q, f.axis);
        V2 pa = b2MulTV(xfA, f.localPoint);
        V2 pb = b2MulTV(xfB, f.proxyB.v[indexB]);
        return b2Dot(pb-pa, normal);
    } else {
        V2 normal = b2MulRV(xfB.q, f.axis);
        V2 pb = b2MulTV(xfB, f.localPoint);
        V2 pa = b2MulTV(xfA, f.proxyA.v[indexA]);
        return b2Dot(pa-pb, normal);
    }
}

// ---------------------------------------------------------------------------
// b2TimeOfImpact (faithful). 0-ULP identical math to Box2D 2.3.0 b2TimeOfImpact.
// ---------------------------------------------------------------------------
enum GBTOIState { GB_TOI_UNKNOWN=0, GB_TOI_FAILED, GB_TOI_OVERLAPPED, GB_TOI_TOUCHING, GB_TOI_SEPARATED };
struct GBTOIOut { int state; float t; };

GB_HD inline void gbTOI(GBTOIOut& out, const GBDProxy& proxyA, const GBDProxy& proxyB,
                        GBSweep sweepA, GBSweep sweepB, float tMax){
    out.state = GB_TOI_UNKNOWN; out.t = tMax;
    gbSweepNormalize(sweepA); gbSweepNormalize(sweepB);
    float totalRadius = proxyA.radius + proxyB.radius;
    float target = b2MaxF(GB_LINEAR_SLOP, totalRadius - 3.0f*GB_LINEAR_SLOP);
    float tolerance = 0.25f*GB_LINEAR_SLOP;
    float t1 = 0.0f;
    const int k_maxIterations = 20;
    int iter = 0;
    GBSimplexCache cache; cache.count=0;
    for(;;){
        Xf xfA, xfB; gbSweepGetTransform(sweepA, xfA, t1); gbSweepGetTransform(sweepB, xfB, t1);
        GBDistInput din; din.proxyA=proxyA; din.proxyB=proxyB; din.xfA=xfA; din.xfB=xfB; din.useRadii=false;
        GBDistOutput dout; gbDistanceGJK(dout, cache, din);
        if (dout.distance <= 0.0f){ out.state=GB_TOI_OVERLAPPED; out.t=0.0f; break; }
        if (dout.distance < target + tolerance){ out.state=GB_TOI_TOUCHING; out.t=t1; break; }
        GBSepFn fcn; gbSepInit(fcn, cache, proxyA, sweepA, proxyB, sweepB, t1);
        bool done=false; float t2=tMax; int pushBackIter=0;
        for(;;){
            int indexA, indexB;
            float s2 = gbSepFindMin(fcn, indexA, indexB, t2);
            if (s2 > target + tolerance){ out.state=GB_TOI_SEPARATED; out.t=tMax; done=true; break; }
            if (s2 > target - tolerance){ t1=t2; break; }
            float s1 = gbSepEval(fcn, indexA, indexB, t1);
            if (s1 < target - tolerance){ out.state=GB_TOI_FAILED; out.t=t1; done=true; break; }
            if (s1 <= target + tolerance){ out.state=GB_TOI_TOUCHING; out.t=t1; done=true; break; }
            int rootIterCount=0; float a1=t1, a2=t2;
            for(;;){
                float t;
                if (rootIterCount & 1) t = a1 + (target - s1)*(a2 - a1)/(s2 - s1);
                else                   t = 0.5f*(a1 + a2);
                ++rootIterCount;
                float s = gbSepEval(fcn, indexA, indexB, t);
                if (b2AbsF(s - target) < tolerance){ t2=t; break; }
                if (s > target){ a1=t; s1=s; } else { a2=t; s2=s; }
                if (rootIterCount==50) break;
            }
            ++pushBackIter;
            if (pushBackIter == 16) break;
        }
        ++iter;
        if (done) break;
        if (iter == k_maxIterations){ out.state=GB_TOI_FAILED; out.t=t1; break; }
    }
}

// ---------------------------------------------------------------------------
// GB_TOI_FULL: the world/island SolveTOI driver. Gated because it needs the
// narrow-phase, solver, and island modules included first, plus the shape-radius
// lookup. Include those headers, then define GB_TOI_FULL, then re-include this one.
// ---------------------------------------------------------------------------
#ifdef GB_TOI_FULL

// gbIslandSolveTOI and gbWorldSolveTOI follow here. They depend on types and
// functions from the modules still in development:
//   - GbIslandData / GbConstraint (gb_island.cuh)
//   - gbWorldManifoldInit / gbCollideCircles / gbCollideEdgeAndCircle (gb_collision.cuh)
//   - gbContactUpdate (gb_collision.cuh)
//   - gbWorldSolve / gbCollidePhase (the assembled step)
//   - the shape-radius lookup for gbContactProxy
// The integration point (gb_step.cuh) supplies these once those modules validate.

#endif // GB_TOI_FULL
