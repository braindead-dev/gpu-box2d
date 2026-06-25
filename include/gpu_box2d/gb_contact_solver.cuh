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
// b2WorldManifold::Initialize (b2Collision.cpp:22). Handles e_circles, e_faceA, and
// e_faceB, for one or two points. Identical to the copy in gb_collision.cuh.
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
    } else if (m.type == GB_MANIFOLD_FACE_A){
        wm.normal = b2MulRV(xfA.q, m.localNormal);
        V2 planePoint = b2MulTV(xfA, m.localPoint);
        V2 cp0 = b2MulTV(xfB, m.pLocalPoint);
        wm.point0 = 0.5f*((cp0 + (rA - b2Dot(cp0 - planePoint, wm.normal))*wm.normal)
                          + (cp0 - rB*wm.normal));
        if (m.pointCount > 1){
            V2 cp1 = b2MulTV(xfB, m.pLocalPoint2);
            wm.point1 = 0.5f*((cp1 + (rA - b2Dot(cp1 - planePoint, wm.normal))*wm.normal)
                              + (cp1 - rB*wm.normal));
        }
    } else { // FACE_B
        wm.normal = b2MulRV(xfB.q, m.localNormal);
        V2 planePoint = b2MulTV(xfB, m.localPoint);
        V2 cp0 = b2MulTV(xfA, m.pLocalPoint);
        wm.point0 = 0.5f*((cp0 + (rB - b2Dot(cp0 - planePoint, wm.normal))*wm.normal)
                          + (cp0 - rA*wm.normal));
        if (m.pointCount > 1){
            V2 cp1 = b2MulTV(xfA, m.pLocalPoint2);
            wm.point1 = 0.5f*((cp1 + (rB - b2Dot(cp1 - planePoint, wm.normal))*wm.normal)
                              + (cp1 - rA*wm.normal));
        }
        wm.normal = -wm.normal;
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

// Per-point helper: build rA/rB/normalMass/tangentMass/velocityBias at one point.
GB_HD inline void gbInitVelPoint(GBVelConstraintPt& vcp, V2 worldPoint, V2 normal,
                                 V2 cA, V2 vA, float wA, V2 cB, V2 vB, float wB,
                                 float mA, float mB, float iA, float iB, float restitution){
    vcp.rA = worldPoint - cA; vcp.rB = worldPoint - cB;
    float rnA = b2Cross(vcp.rA, normal);
    float rnB = b2Cross(vcp.rB, normal);
    float kNormal = mA + mB + iA*rnA*rnA + iB*rnB*rnB;
    vcp.normalMass = kNormal > 0.0f ? 1.0f/kNormal : 0.0f;
    V2 tangent = b2CrossVS(normal, 1.0f);
    float rtA = b2Cross(vcp.rA, tangent);
    float rtB = b2Cross(vcp.rB, tangent);
    float kTangent = mA + mB + iA*rtA*rtA + iB*rtB*rtB;
    vcp.tangentMass = kTangent > 0.0f ? 1.0f/kTangent : 0.0f;
    vcp.velocityBias = 0.0f;
    float vRel = b2Dot(normal, vB + b2CrossSV(wB, vcp.rB) - vA - b2CrossSV(wA, vcp.rA));
    if (vRel < -GB_VELOCITY_THRESHOLD) vcp.velocityBias = -restitution * vRel;
}

// ---- InitializeVelocityConstraints (position-dependent portions) -----------
// b2ContactSolver::InitializeVelocityConstraints. Builds the world manifold from the
// persisted cache, computes rA/rB/normalMass/tangentMass/velocityBias per point, and
// prepares the 2x2 block matrix when the contact has two points.
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
        GBManifold man; man.pointCount=vc.pointCount; man.type=pc.type;
        man.localNormal=pc.localNormal; man.localPoint=pc.localPoint;
        man.pLocalPoint=pc.pLocalPoint; man.pLocalPoint2=pc.pLocalPoint2;
        GBWorldManifold wm; gbWorldManifoldInit(wm, man, xfA, pc.radiusA, xfB, pc.radiusB);
        vc.normal = wm.normal;
        gbInitVelPoint(vc.p, wm.point0, vc.normal, cA, vA, wA, cB, vB, wB,
                       mA, mB, iA, iB, vc.restitution);
        if (vc.pointCount == 2)
            gbInitVelPoint(vc.p2, wm.point1, vc.normal, cA, vA, wA, cB, vB, wB,
                           mA, mB, iA, iB, vc.restitution);

        // Two-point block matrix (b2ContactSolver.cpp:216). Build K and invert it,
        // falling back to a single point if the condition number is poor.
        if (vc.pointCount == 2){
            float rn1A = b2Cross(vc.p.rA,  vc.normal);
            float rn1B = b2Cross(vc.p.rB,  vc.normal);
            float rn2A = b2Cross(vc.p2.rA, vc.normal);
            float rn2B = b2Cross(vc.p2.rB, vc.normal);
            float k11 = mA + mB + iA*rn1A*rn1A + iB*rn1B*rn1B;
            float k22 = mA + mB + iA*rn2A*rn2A + iB*rn2B*rn2B;
            float k12 = mA + mB + iA*rn1A*rn2A + iB*rn1B*rn2B;
            const float k_maxConditionNumber = 1000.0f;
            if (k11*k11 < k_maxConditionNumber*(k11*k22 - k12*k12)){
                vc.K.ex = v2(k11, k12); vc.K.ey = v2(k12, k22);
                vc.normalMass22 = gbMat22GetInverse(vc.K);
            } else {
                vc.pointCount = 1;   // constraints redundant; use one
            }
        }
    }
}

// ---- WarmStart -------------------------------------------------------------
// b2ContactSolver::WarmStart. SERIAL: each contact mutates the island vel buffers in
// order; within a contact each point applies its cached impulse in point order.
GB_HD inline void gbWarmStart(GBIslandData& isl){
    int cc = isl.contactCount;
    for (int i = 0; i < cc; ++i){
        GBConstraint& vc = isl.con[i];
        int ia=vc.indexA, ib=vc.indexB;
        float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
        V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
        V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
        V2 normal=vc.normal; V2 tangent=b2CrossVS(normal,1.0f);
        for (int j = 0; j < vc.pointCount; ++j){
            GBVelConstraintPt& vcp = (j == 0) ? vc.p : vc.p2;
            V2 P = vcp.normalImpulse*normal + vcp.tangentImpulse*tangent;
            wA -= iA*b2Cross(vcp.rA, P); vA = vA - mA*P;
            wB += iB*b2Cross(vcp.rB, P); vB = vB + mB*P;
        }
        isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
    }
}

// ---- SolveVelocityConstraints, ONE iteration -------------------------------
// b2ContactSolver::SolveVelocityConstraints (one pass). The island loops it
// GB_VELOCITY_ITERS times. SERIAL GAUSS-SEIDEL: tangent (friction) solve per point,
// then the normal solve, per contact, in fixed contact order, each contact reading
// vA/vB as mutated by the previous one. The in-place velocity accumulation is fold 1
// of the three. A two-point contact runs the block LCP through the fixed four-case
// cascade; the branch taken is part of the result.
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
        int pointCount = vc.pointCount;

        // tangent (friction) first, for each point in order
        for (int j = 0; j < pointCount; ++j){
            GBVelConstraintPt& vcp = (j == 0) ? vc.p : vc.p2;
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

        if (pointCount == 1){
            GBVelConstraintPt& vcp = vc.p;
            V2 dv = vB + b2CrossSV(wB, vcp.rB) - vA - b2CrossSV(wA, vcp.rA);
            float vn = b2Dot(dv, normal);
            float lambda = -vcp.normalMass * (vn - vcp.velocityBias);
            float newImp = b2MaxF(vcp.normalImpulse + lambda, 0.0f);
            lambda = newImp - vcp.normalImpulse;
            vcp.normalImpulse = newImp;
            V2 P = lambda*normal;
            vA = vA - mA*P; wA -= iA*b2Cross(vcp.rA, P);
            vB = vB + mB*P; wB += iB*b2Cross(vcp.rB, P);
        } else {
            // Two-point block solver (b2ContactSolver.cpp:368). Total-enumeration LCP
            // over the four complementarity cases, applied in Box2D order.
            GBVelConstraintPt& cp1 = vc.p;
            GBVelConstraintPt& cp2 = vc.p2;
            V2 a = v2(cp1.normalImpulse, cp2.normalImpulse);
            V2 dv1 = vB + b2CrossSV(wB, cp1.rB) - vA - b2CrossSV(wA, cp1.rA);
            V2 dv2 = vB + b2CrossSV(wB, cp2.rB) - vA - b2CrossSV(wA, cp2.rA);
            float vn1 = b2Dot(dv1, normal);
            float vn2 = b2Dot(dv2, normal);
            V2 b = v2(vn1 - cp1.velocityBias, vn2 - cp2.velocityBias);
            b = b - gbMulMV(vc.K, a);
            for (;;){
                // Case 1: vn1 = vn2 = 0
                V2 x = -1.0f * gbMulMV(vc.normalMass22, b);
                if (x.x >= 0.0f && x.y >= 0.0f){
                    V2 d = x - a;
                    V2 P1 = d.x*normal; V2 P2 = d.y*normal;
                    vA = vA - mA*(P1 + P2);
                    wA -= iA*(b2Cross(cp1.rA, P1) + b2Cross(cp2.rA, P2));
                    vB = vB + mB*(P1 + P2);
                    wB += iB*(b2Cross(cp1.rB, P1) + b2Cross(cp2.rB, P2));
                    cp1.normalImpulse = x.x; cp2.normalImpulse = x.y;
                    break;
                }
                // Case 2: vn1 = 0, x2 = 0
                x.x = -cp1.normalMass * b.x; x.y = 0.0f;
                vn1 = 0.0f; vn2 = vc.K.ex.y * x.x + b.y;
                if (x.x >= 0.0f && vn2 >= 0.0f){
                    V2 d = x - a;
                    V2 P1 = d.x*normal; V2 P2 = d.y*normal;
                    vA = vA - mA*(P1 + P2);
                    wA -= iA*(b2Cross(cp1.rA, P1) + b2Cross(cp2.rA, P2));
                    vB = vB + mB*(P1 + P2);
                    wB += iB*(b2Cross(cp1.rB, P1) + b2Cross(cp2.rB, P2));
                    cp1.normalImpulse = x.x; cp2.normalImpulse = x.y;
                    break;
                }
                // Case 3: vn2 = 0, x1 = 0
                x.x = 0.0f; x.y = -cp2.normalMass * b.y;
                vn1 = vc.K.ey.x * x.y + b.x; vn2 = 0.0f;
                if (x.y >= 0.0f && vn1 >= 0.0f){
                    V2 d = x - a;
                    V2 P1 = d.x*normal; V2 P2 = d.y*normal;
                    vA = vA - mA*(P1 + P2);
                    wA -= iA*(b2Cross(cp1.rA, P1) + b2Cross(cp2.rA, P2));
                    vB = vB + mB*(P1 + P2);
                    wB += iB*(b2Cross(cp1.rB, P1) + b2Cross(cp2.rB, P2));
                    cp1.normalImpulse = x.x; cp2.normalImpulse = x.y;
                    break;
                }
                // Case 4: x1 = x2 = 0
                x.x = 0.0f; x.y = 0.0f;
                vn1 = b.x; vn2 = b.y;
                if (vn1 >= 0.0f && vn2 >= 0.0f){
                    V2 d = x - a;
                    V2 P1 = d.x*normal; V2 P2 = d.y*normal;
                    vA = vA - mA*(P1 + P2);
                    wA -= iA*(b2Cross(cp1.rA, P1) + b2Cross(cp2.rA, P2));
                    vB = vB + mB*(P1 + P2);
                    wB += iB*(b2Cross(cp1.rB, P1) + b2Cross(cp2.rB, P2));
                    cp1.normalImpulse = x.x; cp2.normalImpulse = x.y;
                    break;
                }
                // No solution: give up (matches Box2D).
                break;
            }
        }
        isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
    }
}

// ---- StoreImpulses (carry warm-start to next substep) ----------------------
// b2ContactSolver::StoreImpulses. Writes converged impulses to the persistent slots.
// Point 0 maps to the contact's primary impulse fields; point 1 (when present) maps
// to the secondary impulse fields. A 1-point contact touches only the primary fields,
// so the single-point carry stays byte-identical.
GB_HD inline void gbStoreImpulses(GBWorld& w, GBIslandData& isl){
    int cc = isl.contactCount;
    for (int i = 0; i < cc; ++i){
        GBConstraint& vc = isl.con[i];
        int ci = vc.contactIdx;
        CONT(w, cNormalImpulse,  ci) = vc.p.normalImpulse;
        CONT(w, cTangentImpulse, ci) = vc.p.tangentImpulse;
        if (vc.pointCount == 2){
            CONT(w, cNormalImpulse2,  ci) = vc.p2.normalImpulse;
            CONT(w, cTangentImpulse2, ci) = vc.p2.tangentImpulse;
        }
#if defined(B2_GPU_DUMP) && !defined(__CUDA_ARCH__)
        fprintf(stderr,"  IMP i=%d nrm=%.8f tan=%.8f\n", i,
                vc.p.normalImpulse, vc.p.tangentImpulse);
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
        // b2ContactSolver::SolvePositionConstraints, per point in order.
        for (int j = 0; j < pc.pointCount; ++j){
            Xf xfA, xfB;
            xfA.q=rotSet(aA); xfB.q=rotSet(aB);
            xfA.p = cA - b2MulRV(xfA.q, v2(0,0));
            xfB.p = cB - b2MulRV(xfB.q, v2(0,0));
            // b2PositionSolverManifold::Initialize
            V2 normal, point; float separation;
            V2 clipLocal = (j == 0) ? pc.pLocalPoint : pc.pLocalPoint2;
            if (pc.type == GB_MANIFOLD_CIRCLES){
                V2 pointA = b2MulTV(xfA, pc.localPoint);
                V2 pointB = b2MulTV(xfB, clipLocal);
                normal = pointB - pointA; b2Normalize(normal);
                point = 0.5f*(pointA + pointB);
                separation = b2Dot(pointB - pointA, normal) - pc.radiusA - pc.radiusB;
            } else if (pc.type == GB_MANIFOLD_FACE_A){
                normal = b2MulRV(xfA.q, pc.localNormal);
                V2 planePoint = b2MulTV(xfA, pc.localPoint);
                V2 clipPoint  = b2MulTV(xfB, clipLocal);
                separation = b2Dot(clipPoint - planePoint, normal) - pc.radiusA - pc.radiusB;
                point = clipPoint;
            } else { // FACE_B
                normal = b2MulRV(xfB.q, pc.localNormal);
                V2 planePoint = b2MulTV(xfB, pc.localPoint);
                V2 clipPoint  = b2MulTV(xfA, clipLocal);
                separation = b2Dot(clipPoint - planePoint, normal) - pc.radiusA - pc.radiusB;
                point = clipPoint;
                normal = -normal;   // ensure normal points from A to B
            }
            V2 rA = point - cA; V2 rB = point - cB;
            minSeparation = b2MinF(minSeparation, separation);
#if defined(B2_GPU_DUMP) && !defined(__CUDA_ARCH__)
            fprintf(stderr,"  PS i=%d j=%d type=%d sep=%.8f normal=(%.6f,%.6f) point=(%.6f,%.6f)\n",
                    i, j, pc.type, separation, normal.x, normal.y, point.x, point.y);
#endif
            float C = b2ClampF(GB_BAUMGARTE*(separation + GB_LINEAR_SLOP),
                               -GB_MAX_LINEAR_CORRECTION, 0.0f);
            float rnA = b2Cross(rA, normal); float rnB = b2Cross(rB, normal);
            float K = mA + mB + iA*rnA*rnA + iB*rnB*rnB;
            float impulse = K > 0.0f ? -C/K : 0.0f;
            V2 P = impulse*normal;
            cA = cA - mA*P; aA -= iA*b2Cross(rA, P);
            cB = cB + mB*P; aB += iB*b2Cross(rB, P);
        }
        isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;
    }
    // contactsOkay (b2ContactSolver.cpp): the position solve converged this pass.
    return minSeparation >= -3.0f*GB_LINEAR_SLOP;
}
