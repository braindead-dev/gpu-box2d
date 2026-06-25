// gb_joint.cuh. b2RevoluteJoint (Box2D 2.3.0), point-to-point case, written against
// the gb_* type universe. A revolute joint pins two bodies at a shared world anchor,
// so the bodies rotate freely about it while their anchor points stay coincident.
//
// This header ports the point-to-point revolute joint: the two-degree-of-freedom
// position constraint solved with the 2x2 mass matrix in both the velocity solve and
// the position solve. The motor and angle-limit rows (the 3x3 path) are a documented
// follow-on; see docs/extending.md. The math and the evaluation order match Box2D
// 2.3.0 b2RevoluteJoint::InitVelocityConstraints / SolveVelocityConstraints /
// SolvePositionConstraints for a joint with motor and limit disabled.
//
// Box2D-source faithfulness:
//   Dynamics/Joints/b2RevoluteJoint.cpp (point-to-point branch)
//
// A joint solves alongside contacts in b2Island::Solve. Box2D solves joints before
// contacts in joint-list order within each velocity and each position iteration, and
// the joint velocity solve runs once per velocity iteration. The per-island driver
// that interleaves joints and contacts is the integration point; this header provides
// the per-joint phases that the driver calls.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_types.cuh"   // GBMat22, GBIslandData, V2/Rot/Xf

// b2Mat22::Solve. Solves K * x = b for x without forming the inverse. Same float
// operations and ordering as Box2D 2.3.0.
GB_HD inline V2 gbMat22Solve(const GBMat22& K, V2 b){
    float a11 = K.ex.x, a12 = K.ey.x, a21 = K.ex.y, a22 = K.ey.y;
    float det = a11*a22 - a12*a21;
    if (det != 0.0f) det = 1.0f / det;
    return v2(det*(a22*b.x - a12*b.y), det*(a11*b.y - a21*b.x));
}

// GBRevoluteJoint is defined in gb_contact_types.cuh so the per-island scratch can
// carry an array of joints. The phases below operate on it. Anchors are body-local;
// localCenter is 0 in the gb_* body model, so the world anchor arm is
// rA = Rot(aA) * localAnchorA.

// b2RevoluteJoint::InitVelocityConstraints (point-to-point). Builds rA/rB and the 2x2
// mass matrix, then applies the warm-start impulse to the island velocity buffers.
GB_HD inline void gbRevoluteInitVelocity(GBRevoluteJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    float aA = isl.posA[ia], aB = isl.posA[ib];
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    j.rA = b2MulRV(qA, j.localAnchorA);   // localCenter == 0
    j.rB = b2MulRV(qB, j.localAnchorB);
    GBMat22& K = j.mass;
    K.ex.x = mA + mB + iA*j.rA.y*j.rA.y + iB*j.rB.y*j.rB.y;
    K.ey.x = -iA*j.rA.x*j.rA.y - iB*j.rB.x*j.rB.y;
    K.ex.y = K.ey.x;
    K.ey.y = mA + mB + iA*j.rA.x*j.rA.x + iB*j.rB.x*j.rB.x;
    // warm-start (m_motorImpulse and m_impulse.z are 0 in the point-to-point case)
    V2 P = j.impulse;
    vA = vA - mA*P; wA -= iA*b2Cross(j.rA, P);
    vB = vB + mB*P; wB += iB*b2Cross(j.rB, P);
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2RevoluteJoint::SolveVelocityConstraints (point-to-point), ONE iteration. The
// island driver loops it GB_VELOCITY_ITERS times alongside the contacts.
GB_HD inline void gbRevoluteSolveVelocity(GBRevoluteJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    V2 Cdot = vB + b2CrossSV(wB, j.rB) - vA - b2CrossSV(wA, j.rA);
    V2 impulse = gbMat22Solve(j.mass, -Cdot);
    j.impulse = j.impulse + impulse;
    vA = vA - mA*impulse; wA -= iA*b2Cross(j.rA, impulse);
    vB = vB + mB*impulse; wB += iB*b2Cross(j.rB, impulse);
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2RevoluteJoint::SolvePositionConstraints (point-to-point), ONE iteration. Returns
// true when the position error is within the linear slop, the joint's contribution to
// the island position early-exit.
GB_HD inline bool gbRevoluteSolvePosition(GBRevoluteJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    V2 rA = b2MulRV(qA, j.localAnchorA);
    V2 rB = b2MulRV(qB, j.localAnchorB);
    V2 C = cB + rB - cA - rA;
    float positionError = b2Length(C);
    GBMat22 K;
    K.ex.x = mA + mB + iA*rA.y*rA.y + iB*rB.y*rB.y;
    K.ex.y = -iA*rA.x*rA.y - iB*rB.x*rB.y;
    K.ey.x = K.ex.y;
    K.ey.y = mA + mB + iA*rA.x*rA.x + iB*rB.x*rB.x;
    V2 impulse = -1.0f * gbMat22Solve(K, C);
    cA = cA - mA*impulse; aA -= iA*b2Cross(rA, impulse);
    cB = cB + mB*impulse; aB += iB*b2Cross(rB, impulse);
    isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;
    return positionError <= GB_LINEAR_SLOP;
}
