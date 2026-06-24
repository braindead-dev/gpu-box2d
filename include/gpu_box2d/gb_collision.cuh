// gb_collision.cuh. The narrow-phase, written against the gb_pools accessor
// contract. The math is the validated narrow-phase, expressed against the gb_* type
// universe (GBManifold/GBWorldManifold from gb_contact_types.cuh, V2/Rot/Xf and ops
// from gb_math.cuh) and reading world state only through BODY/CONT/EDGE/SCAL.
//
// Line-faithful to Box2D 2.3.0:
//   Collision/b2CollideCircle.cpp  (b2CollideCircles)
//   Collision/b2CollideEdge.cpp    (b2CollideEdgeAndCircle, single-edge regions)
//   Collision/b2Collision.cpp      (b2WorldManifold::Initialize, 1-point)
//   Dynamics/b2Contact.cpp         (b2Contact::Update, touching flip, 1-point)
//
// This module covers the circle and single-edge shapes, where every manifold is
// 1-point. The 2-point block path (polygon contacts) is a separate extension; see
// docs/extending.md.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true (mirrors
// the CPU's -ffp-contract=off -mfpmath=sse). Changing these breaks bit-identicality.
#pragma once
#include "gpu_box2d/gb_pools.cuh"
#include "gpu_box2d/gb_contact_types.cuh"

// =========================== Narrow-phase ===================================
// b2CollideCircles (b2CollideCircle.cpp:23). circle m_p == (0,0).
GB_HD inline void gbCollideCircles(GBManifold& m, float rA, Xf xfA, float rB, Xf xfB){
    m.pointCount = 0;
    V2 pA = b2MulTV(xfA, v2(0,0));
    V2 pB = b2MulTV(xfB, v2(0,0));
    V2 d = pB - pA;
    float distSqr = b2Dot(d,d);
    float radius = rA + rB;
    if (distSqr > radius*radius) return;
    m.type = GB_MANIFOLD_CIRCLES;
    m.localPoint  = v2(0,0);     // circleA->m_p
    m.localNormal = v2(0,0);
    m.pointCount = 1;
    m.pLocalPoint = v2(0,0);     // circleB->m_p
}

// b2CollideEdgeAndCircle (b2CollideEdge.cpp:27). Edge has no vertex0/vertex3
// (single-segment edges), so the connectivity early-outs never trigger.
GB_HD inline void gbCollideEdgeAndCircle(GBManifold& m, V2 A, V2 B, float edgeR,
                                         float circR, Xf xfA, Xf xfB){
    m.pointCount = 0;
    V2 Q = b2MulTinvV(xfA, b2MulTV(xfB, v2(0,0)));   // circle m_p == 0
    V2 e = B - A;
    float u = b2Dot(e, B - Q);
    float v = b2Dot(e, Q - A);
    float radius = edgeR + circR;

    if (v <= 0.0f){                      // Region A (vertex1)
        V2 P = A;
        V2 d = Q - P;
        float dd = b2Dot(d,d);
        if (dd > radius*radius) return;
        // m_hasVertex0 == false => no connectivity check
        m.pointCount = 1; m.type = GB_MANIFOLD_CIRCLES;
        m.localNormal = v2(0,0); m.localPoint = P;
        m.pLocalPoint = v2(0,0);
        return;
    }
    if (u <= 0.0f){                      // Region B (vertex2)
        V2 P = B;
        V2 d = Q - P;
        float dd = b2Dot(d,d);
        if (dd > radius*radius) return;
        // m_hasVertex3 == false => no connectivity check
        m.pointCount = 1; m.type = GB_MANIFOLD_CIRCLES;
        m.localNormal = v2(0,0); m.localPoint = P;
        m.pLocalPoint = v2(0,0);
        return;
    }
    // Region AB (face)
    float den = b2Dot(e,e);
    V2 P = (1.0f/den) * (u*A + v*B);
    V2 d = Q - P;
    float dd = b2Dot(d,d);
    if (dd > radius*radius) return;
    V2 n = v2(-e.y, e.x);
    if (b2Dot(n, Q - A) < 0.0f) n = v2(-n.x, -n.y);
    b2Normalize(n);
    m.pointCount = 1; m.type = GB_MANIFOLD_FACE_A;
    m.localNormal = n; m.localPoint = A;
    m.pLocalPoint = v2(0,0);
}

// b2WorldManifold::Initialize (b2Collision.cpp:22). 1-point only.
GB_HD inline void gbWorldManifoldInit(GBWorldManifold& wm, const GBManifold& m,
                                      Xf xfA, float rA, Xf xfB, float rB){
    if (m.pointCount == 0) return;
    if (m.type == GB_MANIFOLD_CIRCLES){
        wm.normal = v2(1.0f, 0.0f);
        V2 pointA = b2MulTV(xfA, m.localPoint);
        V2 pointB = b2MulTV(xfB, m.pLocalPoint);
        if (b2DistanceSquared(pointA, pointB) > GB_EPSILON*GB_EPSILON){
            wm.normal = pointB - pointA;
            b2Normalize(wm.normal);
        }
        V2 cA = pointA + rA*wm.normal;
        V2 cB = pointB - rB*wm.normal;
        wm.point0 = 0.5f*(cA + cB);
    } else { // GB_MANIFOLD_FACE_A
        wm.normal = b2MulRV(xfA.q, m.localNormal);
        V2 planePoint = b2MulTV(xfA, m.localPoint);
        V2 clipPoint = b2MulTV(xfB, m.pLocalPoint);
        V2 cA = clipPoint + (rA - b2Dot(clipPoint - planePoint, wm.normal))*wm.normal;
        V2 cB = clipPoint - rB*wm.normal;
        wm.point0 = 0.5f*(cA + cB);
    }
}

// =========================== Per-contact helpers ============================
// Body transform from the cached xf fields, read via accessors.
GB_HD inline Xf gbBodyXf(GBWorld& w, int i){
    Xf t;
    t.p = v2(BODY(w, xfPx, i), BODY(w, xfPy, i));
    t.q.s = BODY(w, xfQs, i);
    t.q.c = BODY(w, xfQc, i);
    return t;
}

// gbContactUpdate. b2Contact::Update (b2Contact.cpp:161), 1-point path, on the
// accessor contract. Runs the narrow-phase for contact slot ci, sets enabled, caches
// the manifold, carries warm-start impulses, and flips cTouching.
//
// Contact key convention: cEdge < 0 is circle-circle (fixtureA = cBodyA's circle,
// fixtureB = cBodyB's circle); cEdge >= 0 is edge-circle (fixtureA = ground edge
// `cEdge`, fixtureB = cBodyB's circle). Radii read via the general accessor
// gbCircleRadius (BODY(w,radius,s)).
//
// CONTACT LISTENER HOOK. This is the generic b2ContactListener mechanism. On a
// touching transition gbContactUpdate calls gbOnTouchBegin (begin-contact) or
// gbOnTouchEnd (end-contact). The default definitions are no-ops; an application
// overrides them by defining GB_CONTACT_LISTENER_HOOKS and supplying its own
// gbOnTouchBegin / gbOnTouchEnd before this header is included. The hook carries
// no game meaning in the core and adds zero float ops to the narrow-phase, so the
// 0-ULP manifold and touching result hold.
#ifndef GB_CONTACT_LISTENER_HOOKS
GB_HD inline void gbOnTouchBegin(GBWorld&, int, int){}
GB_HD inline void gbOnTouchEnd(GBWorld&, int, int){}
#endif

GB_HD inline void gbContactUpdate(GBWorld& w, int ci){
    CONT(w, cEnabled, ci) = 1;   // b2Contact::Update: m_flags |= e_enabledFlag
    bool wasTouching = CONT(w, cTouching, ci) != 0;
    int bodyA = CONT(w, cBodyA, ci), bodyB = CONT(w, cBodyB, ci), edge = CONT(w, cEdge, ci);
    GBManifold m; m.pointCount = 0;
    if (edge < 0){
        // circle-circle: fixtureA = bodyA's circle, fixtureB = bodyB's circle
        float rA = gbCircleRadius(w, bodyA), rB = gbCircleRadius(w, bodyB);
        gbCollideCircles(m, rA, gbBodyXf(w, bodyA), rB, gbBodyXf(w, bodyB));
    } else {
        // edge-circle: fixtureA = ground edge, fixtureB = body's circle
        V2 A = v2(EDGE(w, edgeAx, edge), EDGE(w, edgeAy, edge));
        V2 B = v2(EDGE(w, edgeBx, edge), EDGE(w, edgeBy, edge));
        float circR = gbCircleRadius(w, bodyB);
        gbCollideEdgeAndCircle(m, A, B, GB_POLYGON_RADIUS, circR,
                               gbBodyXf(w, bodyA), gbBodyXf(w, bodyB));
    }
    bool touching = m.pointCount > 0;
    // warm-start id carry: all our manifolds have id.key == 0, so a surviving
    // touching contact keeps its impulse; a non-touching one resets to 0 anyway.
    if (touching){
        CONT(w, cManifoldType, ci) = m.type;
        CONT(w, cLocalNormalX, ci) = m.localNormal.x; CONT(w, cLocalNormalY, ci) = m.localNormal.y;
        CONT(w, cLocalPointX,  ci) = m.localPoint.x;  CONT(w, cLocalPointY,  ci) = m.localPoint.y;
        CONT(w, cPointLocalX,  ci) = m.pLocalPoint.x; CONT(w, cPointLocalY,  ci) = m.pLocalPoint.y;
        // impulse carries from previous substep (cNormalImpulse/cTangentImpulse
        // are left intact since id.key matches). On first-touch they are 0.
    } else {
        CONT(w, cNormalImpulse, ci) = 0.0f; CONT(w, cTangentImpulse, ci) = 0.0f;
    }
    CONT(w, cTouching, ci) = touching ? 1 : 0;
    // b2Contact::Update: fire begin-contact / end-contact on touching transitions
    if (!wasTouching && touching)  gbOnTouchBegin(w, bodyA, bodyB);
    if ( wasTouching && !touching) gbOnTouchEnd(w, bodyA, bodyB);
}
