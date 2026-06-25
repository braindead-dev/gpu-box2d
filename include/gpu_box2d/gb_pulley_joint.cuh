// gb_pulley_joint.cuh. b2PulleyJoint (Box2D 2.3.0), written against the gb_* type
// universe. A pulley joint connects two bodies over two fixed ground anchors with the
// constraint lengthA + ratio * lengthB = constant, so pulling one body down lets the
// other rise. It is the block-and-tackle and counterweight model.
//
// This header ports the single constraint row: the effective mass over the two pulley
// arms, the velocity solve, and the position solve that holds the total length. The math
// and the evaluation order match Box2D 2.3.0 b2PulleyJoint::InitVelocityConstraints /
// SolveVelocityConstraints / SolvePositionConstraints.
//
// Box2D-source faithfulness:
//   Dynamics/Joints/b2PulleyJoint.cpp
//
// The ground anchors are world-space points. localCenter is 0 in the gb_* body model,
// so the world anchor arm is rA = Rot(aA) * localAnchorA.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_types.cuh"   // GBIslandData, V2/Rot

// A pulley joint. groundAnchorA / groundAnchorB are fixed world points (the pulley
// wheels). lengthA / lengthB are the rope segment rest lengths. ratio scales the B side
// (a block-and-tackle advantage). constant = lengthA + ratio * lengthB is the conserved
// total. m_impulse is the accumulated impulse (warm-start).
struct GBPulleyJoint {
    int   indexA, indexB;
    V2    groundAnchorA, groundAnchorB;
    V2    localAnchorA, localAnchorB;
    float lengthA, lengthB;
    float ratio, constant;
    float invMassA, invMassB, invIA, invIB;
    int   jointIdx;
    // solver state (set by InitVelocityConstraints)
    V2    uA, uB, rA, rB;
    float mass;
    float impulse;
};

// b2PulleyJoint::InitVelocityConstraints.
GB_HD inline void gbPulleyInitVelocity(GBPulleyJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    j.rA = b2MulRV(qA, j.localAnchorA);   // localCenter == 0
    j.rB = b2MulRV(qB, j.localAnchorB);

    j.uA = cA + j.rA - j.groundAnchorA;
    j.uB = cB + j.rB - j.groundAnchorB;
    float lengthA = b2Length(j.uA);
    float lengthB = b2Length(j.uB);
    if (lengthA > 10.0f * GB_LINEAR_SLOP) j.uA = (1.0f / lengthA) * j.uA;
    else j.uA = v2(0.0f, 0.0f);
    if (lengthB > 10.0f * GB_LINEAR_SLOP) j.uB = (1.0f / lengthB) * j.uB;
    else j.uB = v2(0.0f, 0.0f);

    float ruA = b2Cross(j.rA, j.uA);
    float ruB = b2Cross(j.rB, j.uB);
    float massA = mA + iA*ruA*ruA;
    float massB = mB + iB*ruB*ruB;
    j.mass = massA + j.ratio*j.ratio*massB;
    if (j.mass > 0.0f) j.mass = 1.0f / j.mass;

    // warm start (dtRatio == 1 in steady DT)
    V2 PA = (-j.impulse) * j.uA;
    V2 PB = (-j.ratio * j.impulse) * j.uB;
    vA = vA + mA*PA; wA += iA*b2Cross(j.rA, PA);
    vB = vB + mB*PB; wB += iB*b2Cross(j.rB, PB);
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2PulleyJoint::SolveVelocityConstraints, ONE iteration.
GB_HD inline void gbPulleySolveVelocity(GBPulleyJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    V2 vpA = vA + b2CrossSV(wA, j.rA);
    V2 vpB = vB + b2CrossSV(wB, j.rB);
    float Cdot = -b2Dot(j.uA, vpA) - j.ratio * b2Dot(j.uB, vpB);
    float impulse = -j.mass * Cdot;
    j.impulse += impulse;
    V2 PA = (-impulse) * j.uA;
    V2 PB = (-j.ratio * impulse) * j.uB;
    vA = vA + mA*PA; wA += iA*b2Cross(j.rA, PA);
    vB = vB + mB*PB; wB += iB*b2Cross(j.rB, PB);
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2PulleyJoint::SolvePositionConstraints, ONE iteration. Returns true when the total
// length error is within the linear slop.
GB_HD inline bool gbPulleySolvePosition(GBPulleyJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    V2 rA = b2MulRV(qA, j.localAnchorA);
    V2 rB = b2MulRV(qB, j.localAnchorB);

    V2 uA = cA + rA - j.groundAnchorA;
    V2 uB = cB + rB - j.groundAnchorB;
    float lengthA = b2Length(uA);
    float lengthB = b2Length(uB);
    if (lengthA > 10.0f * GB_LINEAR_SLOP) uA = (1.0f / lengthA) * uA;
    else uA = v2(0.0f, 0.0f);
    if (lengthB > 10.0f * GB_LINEAR_SLOP) uB = (1.0f / lengthB) * uB;
    else uB = v2(0.0f, 0.0f);

    float ruA = b2Cross(rA, uA);
    float ruB = b2Cross(rB, uB);
    float massA = mA + iA*ruA*ruA;
    float massB = mB + iB*ruB*ruB;
    float mass = massA + j.ratio*j.ratio*massB;
    if (mass > 0.0f) mass = 1.0f / mass;

    float C = j.constant - lengthA - j.ratio * lengthB;
    float linearError = b2AbsF(C);
    float impulse = -mass * C;
    V2 PA = (-impulse) * uA;
    V2 PB = (-j.ratio * impulse) * uB;
    cA = cA + mA*PA; aA += iA*b2Cross(rA, PA);
    cB = cB + mB*PB; aB += iB*b2Cross(rB, PB);
    isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;

    return linearError < GB_LINEAR_SLOP;
}
