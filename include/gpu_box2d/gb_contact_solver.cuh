// gb_contact_solver.cuh. The sequential-impulse contact solver (Box2D 2.3.0
// b2ContactSolver), single-point manifold path, written against the gb_pools
// accessor contract (BODY/CONT/EDGE/SCAL).
//
// THE FIDELITY RULE (see docs/architecture.md): the warm-start, the 8 velocity
// iterations, StoreImpulses, and the 3 position iterations are SEQUENTIAL
// GAUSS-SEIDEL sweeps. Each contact reads the body velocities AS MUTATED by the
// previous contact (a read-after-write chain). The running float folds, namely the
// in-place velocity/position accumulation and the minSeparation min-fold in the
// position solve, are NON-ASSOCIATIVE under --fmad=false. So every function here
// runs SERIAL, IN-ORDER, ON LANE 0. No tree-reduce, no graph-coloring, no Jacobi.
// The solver math, the iteration order, and the float-fold order match Box2D
// 2.3.0's b2ContactSolver bit-for-bit. Parallelizing this changes the floats.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
//
// CONSTRAINT INTERFACE: this header owns GbConstraint (the fused velocity+position
// constraint) and GbIslandData (the per-island solve scratch). gb_island.cuh
// includes this header, fills the island, and drives the phases below.
#pragma once
#include "gb_pools.cuh"     // accessor contract + WorldShared field set
#include "gb_settings.cuh"  // Box2D constants (iters, baumgarte, slop, tolerances)
#include "gb_math.cuh"      // V2 / Rot / Xf / b2* ops

#if defined(B2_GPU_DUMP) || defined(B2_GPU_CLIST)
#include <cstdio>
#endif

// ============================================================================
// COLLISION DEPENDENCY (gb_collision.cuh, in development).
// The solver needs the 1-point manifold cache type and b2WorldManifold::Initialize.
// gb_collision.cuh will own these. Until it lands they are provided here as pure
// float math (identical to Box2D 2.3.0 b2Collision.cpp) behind a guard: the
// collision module defines GB_COLLISION_PROVIDED before this header is reached,
// and these blocks then yield to the canonical definitions.
// ============================================================================
#ifndef GB_COLLISION_PROVIDED
#define GB_COLLISION_PROVIDED 1

// 1-point manifold (the only kind fruit-merge ever produces). Box2D 2.3.0 b2Collision.h.
struct GbManifold {
    int   type;         // ManifoldType (GB_MANIFOLD_*)
    int   pointCount;   // 0 or 1
    V2    localNormal;  // for FACE_A
    V2    localPoint;   // manifold.localPoint
    V2    pLocalPoint;  // points[0].localPoint
};

// b2WorldManifold::Initialize (b2Collision.cpp:22), 1-point. Box2D 2.3.0 b2Collision.cpp.
struct GbWorldManifold { V2 normal; V2 point0; };
GB_HD inline void gbWorldManifoldInit(GbWorldManifold& wm, const GbManifold& m,
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
    } else { // FACE_A
        wm.normal = b2MulRV(xfA.q, m.localNormal);
        V2 planePoint = b2MulTV(xfA, m.localPoint);
        V2 clipPoint  = b2MulTV(xfB, m.pLocalPoint);
        V2 cA = clipPoint + (rA - b2Dot(clipPoint - planePoint, wm.normal))*wm.normal;
        V2 cB = clipPoint - rB*wm.normal;
        wm.point0 = 0.5f*(cA + cB);
    }
}
#endif // GB_COLLISION_PROVIDED

// ============================================================================
// CONSTRAINT INTERFACE. Fused velocity+position constraint, exactly as
// Box2D 2.3.0 b2ContactConstraint. localCenterA/B are (0,0) for circles so they are omitted.
// ============================================================================
struct GbVelConstraintPt {
    V2    rA, rB;
    float normalImpulse, tangentImpulse;
    float normalMass, tangentMass, velocityBias;
};
struct GbConstraint {
    int   indexA, indexB;          // island-local body indices
    float invMassA, invMassB, invIA, invIB;
    float friction, restitution;
    V2    normal;
    GbVelConstraintPt p;           // single contact point (velocity)
    int   contactIdx;              // global contact slot (for StoreImpulses)
    // position-solve fields:
    V2    localNormal, localPoint, pLocalPoint;
    int   type; float radiusA, radiusB;
};

// Per-island solve scratch. One fused constraint array (vel+pos). Sized to the
// per-world bounds; lives in shared memory (block model) so the solver reads/writes
// shared, not local. the Box2D 2.3.0 island solve scratch.
struct GbIslandData {
    int     bodies[GB_MAX_BODIES];     // global body slots, island order
    int     contacts[GB_MAX_CONTACTS]; // global contact slots, island order
    int     bodyCount, contactCount;
    V2      posC[GB_MAX_BODIES];  float posA[GB_MAX_BODIES];
    V2      vel[GB_MAX_BODIES];   float velW[GB_MAX_BODIES];
    GbConstraint con[GB_MAX_CONTACTS]; // fused vel+pos constraints
};

// circle fixture radius for a body slot (the shape-radius lookup). tier->radius.
// Owned by the game layer in spirit; defined here against the accessor so the solver
// is self-contained. tier_radius lives in the shared world types (game-agnostic radii).
#ifndef GB_FRUIT_RADIUS_PROVIDED
#define GB_FRUIT_RADIUS_PROVIDED 1
GB_HD inline float gbBodyRadius(GBWorld& w, int i){
    // size-by-tier table for the fruit-merge example. (Game radii; kept inline so the solver and island build
    // standalone. The game layer may override by defining GB_FRUIT_RADIUS_PROVIDED.)
    const float R[12] = {0.25f,0.28f,0.5f,0.525f,0.66f,0.84f,0.975f,
                         1.2f,1.32f,1.65f,1.95f,2.2f};
    return R[BODY(w, tier, i)];
}
#endif

// SynchronizeTransform: m_xf.q = Rot(a); m_xf.p = c - Mul(q, localCenter=0) == c.
// Box2D 2.3.0 b2Body::SynchronizeTransform, re-hosted onto accessors.
GB_HD inline void gbSyncTransform(GBWorld& w, int i){
    Rot q = rotSet(BODY(w, sweepA, i));
    BODY(w, xfQs, i) = q.s; BODY(w, xfQc, i) = q.c;
    BODY(w, xfPx, i) = BODY(w, sweepCx, i);   // - Mul(q, localCenter=0)
    BODY(w, xfPy, i) = BODY(w, sweepCy, i);
}

// ============================================================================
// b2ContactSolver, the SERIAL Gauss-Seidel phases. Each operates on the island
// scratch `isl` and the world `w` (for warm-start carry + radii). LANE 0 ONLY.
// ============================================================================

// ---- InitializeVelocityConstraints (position-dependent portions) -----------
// b2ContactSolver::InitializeVelocityConstraints. Builds the world manifold from the
// persisted 1-point cache, computes rA/rB/normalMass/tangentMass/velocityBias.
// Per-contact independent of OTHER contacts' state, but the body-velocity reads
// (vRel) come from the island vel buffers as warm-started; kept serial-in-order with
// the rest of the solver to be conservative + identical to Box2D 2.3.0.
GB_HD inline void gbInitVelocityConstraints(GBWorld& w, GbIslandData& isl){
    int cc = isl.contactCount;
    for (int i = 0; i < cc; ++i){
        GbConstraint& vc = isl.con[i];
        GbConstraint& pc = isl.con[i];   // fused: same object
        int ia = vc.indexA, ib = vc.indexB;
        float mA=vc.invMassA, mB=vc.invMassB, iA=vc.invIA, iB=vc.invIB;
        V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
        V2 vA = isl.vel[ia];  float wA = isl.velW[ia];
        V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
        V2 vB = isl.vel[ib];  float wB = isl.velW[ib];
        Xf xfA, xfB;
        xfA.q = rotSet(aA); xfB.q = rotSet(aB);
        xfA.p = cA - b2MulRV(xfA.q, v2(0,0));
        xfB.p = cB - b2MulRV(xfB.q, v2(0,0));
        // build manifold object from the persisted cache to feed worldManifold
        GbManifold man; man.pointCount=1; man.type=pc.type;
        man.localNormal=pc.localNormal; man.localPoint=pc.localPoint; man.pLocalPoint=pc.pLocalPoint;
        GbWorldManifold wm; gbWorldManifoldInit(wm, man, xfA, pc.radiusA, xfB, pc.radiusB);
        vc.normal = wm.normal;
        GbVelConstraintPt& vcp = vc.p;
        vcp.rA = wm.point0 - cA; vcp.rB = wm.point0 - cB;
        float rnA = b2Cross(vcp.rA, vc.normal);
        float rnB = b2Cross(vcp.rB, vc.normal);
        float kNormal = mA + mB + iA*rnA*rnA + iB*rnB*rnB;
        vcp.normalMass = kNormal > 0.0f ? 1.0f/kNormal : 0.0f;
        V2 tangent = b2CrossVS(vc.normal, 1.0f);
        float rtA = b2Cross(vcp.rA, tangent);
        float rtB = b2Cross(vcp.rB, tangent);
        float kTangent = mA + mB + iA*rtA*rtA + iB*rtB*rtB;
        vcp.tangentMass = kTangent > 0.0f ? 1.0f/kTangent : 0.0f;
        vcp.velocityBias = 0.0f;
        float vRel = b2Dot(vc.normal, vB + b2CrossSV(wB, vcp.rB) - vA - b2CrossSV(wA, vcp.rA));
        if (vRel < -GB_VELOCITY_THRESHOLD) vcp.velocityBias = -vc.restitution * vRel;
    }
}

// ---- WarmStart -------------------------------------------------------------
// b2ContactSolver::WarmStart. Applies the carried impulse to the island velocities.
// SERIAL: each contact mutates the shared island vel buffers in order.
GB_HD inline void gbWarmStart(GbIslandData& isl){
    int cc = isl.contactCount;
    for (int i = 0; i < cc; ++i){
        GbConstraint& vc = isl.con[i];
        int ia=vc.indexA, ib=vc.indexB;
        float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
        V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
        V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
        V2 normal=vc.normal; V2 tangent=b2CrossVS(normal,1.0f);
        GbVelConstraintPt& vcp=vc.p;
        V2 P = vcp.normalImpulse*normal + vcp.tangentImpulse*tangent;
        wA -= iA*b2Cross(vcp.rA, P); vA = vA - mA*P;
        wB += iB*b2Cross(vcp.rB, P); vB = vB + mB*P;
        isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
    }
}

// ---- SolveVelocityConstraints (8 iterations) -------------------------------
// b2ContactSolver::SolveVelocityConstraints, called GB_VELOCITY_ITERS times.
// SERIAL GAUSS-SEIDEL: tangent (friction) solve THEN normal solve, per contact,
// in fixed contact order. Each contact reads vA/vB AS MUTATED by the previous one.
// The in-place velocity accumulation is the first of the THREE float folds.
GB_HD inline void gbSolveVelocityConstraints(GbIslandData& isl){
    int cc = isl.contactCount;
    for (int iter = 0; iter < GB_VELOCITY_ITERS; ++iter){
        for (int i = 0; i < cc; ++i){
            GbConstraint& vc = isl.con[i];
            int ia=vc.indexA, ib=vc.indexB;
            float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
            V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
            V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
            V2 normal=vc.normal; V2 tangent=b2CrossVS(normal,1.0f);
            float friction=vc.friction;
            GbVelConstraintPt& vcp=vc.p;
            // tangent (friction) first
            {
                V2 dv = vB + b2CrossSV(wB, vcp.rB) - vA - b2CrossSV(wA, vcp.rA);
                float vt = b2Dot(dv, tangent) - 0.0f; // tangentSpeed 0
                float lambda = vcp.tangentMass * (-vt);
                float maxFriction = friction * vcp.normalImpulse;
                float newImp = b2ClampF(vcp.tangentImpulse + lambda, -maxFriction, maxFriction);
                lambda = newImp - vcp.tangentImpulse;
                vcp.tangentImpulse = newImp;
                V2 P = lambda*tangent;
                vA = vA - mA*P; wA -= iA*b2Cross(vcp.rA, P);
                vB = vB + mB*P; wB += iB*b2Cross(vcp.rB, P);
            }
            // normal (1-point)
            {
                V2 dv = vB + b2CrossSV(wB, vcp.rB) - vA - b2CrossSV(wA, vcp.rA);
                float vn = b2Dot(dv, normal);
                float lambda = -vcp.normalMass * (vn - vcp.velocityBias);
                float newImp = b2MaxF(vcp.normalImpulse + lambda, 0.0f);
                lambda = newImp - vcp.normalImpulse;
                vcp.normalImpulse = newImp;
                V2 P = lambda*normal;
                vA = vA - mA*P; wA -= iA*b2Cross(vcp.rA, P);
                vB = vB + mB*P; wB += iB*b2Cross(vcp.rB, P);
            }
            isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
        }
    }
}

// ---- StoreImpulses (carry warm-start to next substep) ----------------------
// b2ContactSolver::StoreImpulses. Writes the converged impulses back to the
// persistent per-contact warm-start slots.
GB_HD inline void gbStoreImpulses(GBWorld& w, GbIslandData& isl){
    int cc = isl.contactCount;
    for (int i = 0; i < cc; ++i){
        int ci = isl.con[i].contactIdx;
        CONT(w, cNormalImpulse,  ci) = isl.con[i].p.normalImpulse;
        CONT(w, cTangentImpulse, ci) = isl.con[i].p.tangentImpulse;
#if defined(B2_GPU_DUMP) && !defined(__CUDA_ARCH__)
        fprintf(stderr,"  IMP i=%d nrm=%.8f tan=%.8f\n", i,
                isl.con[i].p.normalImpulse, isl.con[i].p.tangentImpulse);
#endif
    }
}

// ---- SolvePositionConstraints (<=3 iterations, early-exit) -----------------
// b2ContactSolver::SolvePositionConstraints, called up to GB_POSITION_ITERS times.
// SERIAL GAUSS-SEIDEL: each contact reads cA/cB AS MUTATED by the previous one.
// The `minSeparation` running min-fold (the SECOND of the three folds) drives the
// `contactsOkay` early-exit and MUST stay serial-in-order on lane 0. Returns true
// iff the position solve converged (positionSolved), needed by sleep management.
GB_HD inline bool gbSolvePositionConstraints(GbIslandData& isl){
    int cc = isl.contactCount;
    bool positionSolved = false;
    for (int iter = 0; iter < GB_POSITION_ITERS; ++iter){
        float minSeparation = 0.0f;
        for (int i = 0; i < cc; ++i){
            GbConstraint& pc = isl.con[i];
            int ia=pc.indexA, ib=pc.indexB;
            float mA=pc.invMassA, iA=pc.invIA, mB=pc.invMassB, iB=pc.invIB;
            V2 cA=isl.posC[ia]; float aA=isl.posA[ia];
            V2 cB=isl.posC[ib]; float aB=isl.posA[ib];
            // single point j=0
            Xf xfA, xfB;
            xfA.q=rotSet(aA); xfB.q=rotSet(aB);
            xfA.p = cA - b2MulRV(xfA.q, v2(0,0));
            xfB.p = cB - b2MulRV(xfB.q, v2(0,0));
            // b2PositionSolverManifold::Initialize
            V2 normal, point; float separation;
            if (pc.type == GB_MANIFOLD_CIRCLES){
                V2 pointA = b2MulTV(xfA, pc.localPoint);
                V2 pointB = b2MulTV(xfB, pc.pLocalPoint);
                normal = pointB - pointA; b2Normalize(normal);
                point = 0.5f*(pointA + pointB);
                separation = b2Dot(pointB - pointA, normal) - pc.radiusA - pc.radiusB;
            } else { // FACE_A
                normal = b2MulRV(xfA.q, pc.localNormal);
                V2 planePoint = b2MulTV(xfA, pc.localPoint);
                V2 clipPoint  = b2MulTV(xfB, pc.pLocalPoint);
                separation = b2Dot(clipPoint - planePoint, normal) - pc.radiusA - pc.radiusB;
                point = clipPoint;
            }
            V2 rA = point - cA; V2 rB = point - cB;
            minSeparation = b2MinF(minSeparation, separation);
#if defined(B2_GPU_DUMP) && !defined(__CUDA_ARCH__)
            fprintf(stderr,"  PS i=%d type=%d sep=%.8f normal=(%.6f,%.6f) point=(%.6f,%.6f)\n",
                    i, pc.type, separation, normal.x, normal.y, point.x, point.y);
#endif
            float C = b2ClampF(GB_BAUMGARTE*(separation + GB_LINEAR_SLOP),
                               -GB_MAX_LINEAR_CORRECTION, 0.0f);
            float rnA = b2Cross(rA, normal); float rnB = b2Cross(rB, normal);
            float K = mA + mB + iA*rnA*rnA + iB*rnB*rnB;
            float impulse = K > 0.0f ? -C/K : 0.0f;
            V2 P = impulse*normal;
            cA = cA - mA*P; aA -= iA*b2Cross(rA, P);
            cB = cB + mB*P; aB += iB*b2Cross(rB, P);
            isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;
        }
        bool contactsOkay = minSeparation >= -3.0f*GB_LINEAR_SLOP;
        if (contactsOkay){ positionSolved = true; break; }
    }
    return positionSolved;
}
