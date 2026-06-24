// gb_contact_solver.cuh. b2ContactSolver (Box2D 2.3.0), single-point manifold path,
// on the gb_pools accessor contract (BODY/CONT/EDGE/SCAL) and the cross-module types
// in gb_contact_types.cuh (GBConstraint / GBIslandData / GBManifold / GBWorldManifold).
//
// THE 3-SERIAL-FLOAT-FOLD RULE (see gb_contact_types.cuh): under --fmad=false float
// reduction is non-associative. The warm-start, each velocity iteration,
// StoreImpulses, and each position iteration are sequential Gauss-Seidel sweeps. Every
// contact reads the body velocities as mutated by the previous contact (a
// read-after-write chain). The running folds owned here are (1) the in-place
// velocity/position accumulation in the sweeps and (2) the minSeparation min-fold in
// gbSolvePosition. Both run serial, in order, on lane 0. A tree-reduce, graph-coloring,
// or Jacobi sweep would change the floats. The solver math, the iteration order, and
// the float-fold order match Box2D's b2ContactSolver bit-for-bit.
//
// The signatures are stable. Note the granularity contract:
//   gbSolveVelocity(isl) = ONE velocity iteration  (the island loops it GB_VELOCITY_ITERS times)
//   gbSolvePosition(isl) = ONE position iteration, returns contactsOkay for this pass
//                          (the island loops it up to GB_POSITION_ITERS times, early-exits).
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_pools.cuh"            // accessor contract + WorldShared field set
#include "gb_settings.cuh"         // Box2D constants (iters, baumgarte, slop, tolerances)
#include "gb_math.cuh"             // V2 / Rot / Xf / b2* ops
#include "gb_contact_types.cuh"    // shared types: GBManifold/GBConstraint/GBIslandData

#if defined(B2_GPU_DUMP) || defined(B2_GPU_CLIST)
#include <cstdio>
#endif

// ============================================================================
// COLLISION DEPENDENCY: gbWorldManifoldInit is owned by gb_collision.cuh. When that
// header is included first it defines GB_COLLISION_PROVIDED and its definition wins.
// This guarded copy (identical pure math) lets the solver compile standalone.
// ============================================================================
#ifndef GB_COLLISION_PROVIDED
#define GB_COLLISION_PROVIDED 1
// b2WorldManifold::Initialize (b2Collision.cpp:22), 1-point.
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
// SynchronizeTransform: xf.q = Rot(sweepA); xf.p = sweepC (localCenter==0). This is a
// shared per-body helper; gb_toi.cuh defines an identical one. Guarded so the two
// compose in one translation unit: whichever header defines it first wins, the other
// yields. The definitions are identical either way.
// ============================================================================
#ifndef GB_SYNC_TRANSFORM_PROVIDED
#define GB_SYNC_TRANSFORM_PROVIDED 1
GB_HD inline void gbSyncTransform(GBWorld& w, int i){
    Rot q = rotSet(BODY(w, sweepA, i));
    BODY(w, xfQs, i) = q.s; BODY(w, xfQc, i) = q.c;
    BODY(w, xfPx, i) = BODY(w, sweepCx, i);   // - Mul(q, localCenter=0)
    BODY(w, xfPy, i) = BODY(w, sweepCy, i);
}
#endif // GB_SYNC_TRANSFORM_PROVIDED

// Circle fixture radius - the core reads it via the accessor
// gbCircleRadius(w,s) == BODY(w,radius,s) (gb_pools.cuh). Edge fixtures use
// GB_POLYGON_RADIUS. (See gb_pools.cuh::gbCircleRadius.)

// ============================================================================
// b2ContactSolver - the SERIAL Gauss-Seidel phases (LANE 0 ONLY). Each operates on
// the island scratch `isl` (+ the world `w` for warm-start carry + radii).
// ============================================================================

// ---- InitializeVelocityConstraints (position-dependent portions) -----------
// b2ContactSolver::InitializeVelocityConstraints. Builds the world manifold from the
// persisted 1-point cache, computes rA/rB/normalMass/tangentMass/velocityBias.
GB_HD inline void gbInitVelocityConstraints(GBWorld& w, GBIslandData& isl){
    int cc = isl.contactCount;
    for (int i = 0; i < cc; ++i){
        GBConstraint& vc = isl.con[i];
        GBConstraint& pc = isl.con[i];   // fused: same object
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
        GBManifold man; man.pointCount=1; man.type=pc.type;
        man.localNormal=pc.localNormal; man.localPoint=pc.localPoint; man.pLocalPoint=pc.pLocalPoint;
        GBWorldManifold wm; gbWorldManifoldInit(wm, man, xfA, pc.radiusA, xfB, pc.radiusB);
        vc.normal = wm.normal;
        GBVelConstraintPt& vcp = vc.p;
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
// b2ContactSolver::WarmStart. SERIAL: each contact mutates the island vel buffers in order.
GB_HD inline void gbWarmStart(GBIslandData& isl){
    int cc = isl.contactCount;
    for (int i = 0; i < cc; ++i){
        GBConstraint& vc = isl.con[i];
        int ia=vc.indexA, ib=vc.indexB;
        float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
        V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
        V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
        V2 normal=vc.normal; V2 tangent=b2CrossVS(normal,1.0f);
        GBVelConstraintPt& vcp=vc.p;
        V2 P = vcp.normalImpulse*normal + vcp.tangentImpulse*tangent;
        wA -= iA*b2Cross(vcp.rA, P); vA = vA - mA*P;
        wB += iB*b2Cross(vcp.rB, P); vB = vB + mB*P;
        isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
    }
}

// ---- SolveVelocityConstraints, ONE iteration -------------------------------
// b2ContactSolver::SolveVelocityConstraints (one pass). The island loops it
// GB_VELOCITY_ITERS times. SERIAL GAUSS-SEIDEL: tangent (friction) solve then normal
// solve, per contact, in fixed contact order, each contact reading vA/vB as mutated by
// the previous one. The in-place velocity accumulation is fold 1 of the three.
GB_HD inline void gbSolveVelocity(GBIslandData& isl){
    int cc = isl.contactCount;
    for (int i = 0; i < cc; ++i){
        GBConstraint& vc = isl.con[i];
        int ia=vc.indexA, ib=vc.indexB;
        float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
        V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
        V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
        V2 normal=vc.normal; V2 tangent=b2CrossVS(normal,1.0f);
        float friction=vc.friction;
        GBVelConstraintPt& vcp=vc.p;
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

// ---- StoreImpulses (carry warm-start to next substep) ----------------------
// b2ContactSolver::StoreImpulses. Writes converged impulses to the persistent slots.
GB_HD inline void gbStoreImpulses(GBWorld& w, GBIslandData& isl){
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

// ---- SolvePositionConstraints, ONE iteration -------------------------------
// b2ContactSolver::SolvePositionConstraints (one pass). The island loops it up to
// GB_POSITION_ITERS times, early-exiting when the returned contactsOkay is true.
// SERIAL GAUSS-SEIDEL: each contact reads cA/cB as mutated by the previous one. The
// `minSeparation` running min-fold (fold 2 of the three) is computed within this single
// pass and drives the returned `contactsOkay`. Lane 0 only.
GB_HD inline bool gbSolvePosition(GBIslandData& isl){
    int cc = isl.contactCount;
    float minSeparation = 0.0f;
    for (int i = 0; i < cc; ++i){
        GBConstraint& pc = isl.con[i];
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
    // contactsOkay (b2ContactSolver.cpp): the position solve converged this pass.
    return minSeparation >= -3.0f*GB_LINEAR_SLOP;
}
