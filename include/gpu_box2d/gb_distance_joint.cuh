// gb_distance_joint.cuh. b2DistanceJoint (Box2D 2.3.0), written against the gb_* type
// universe. A distance joint holds two body-local anchor points at a fixed separation,
// a rigid rod between them. With a positive frequency it becomes a soft spring (the
// suspension model in Box2D's car demo); with frequency 0 it is the rigid rod, which
// is the common case and the one the position solve corrects.
//
// This header ports both paths: the rigid constraint (single velocity row plus a
// position correction) and the soft spring (frequency and damping ratio fold into a
// bias and a gamma so the velocity row alone carries the constraint, and the position
// solve is skipped). The math and the evaluation order match Box2D 2.3.0
// b2DistanceJoint::InitVelocityConstraints / SolveVelocityConstraints /
// SolvePositionConstraints.
//
// Box2D-source faithfulness:
//   Dynamics/Joints/b2DistanceJoint.cpp
//
// localCenter is 0 in the gb_* body model, so the world anchor arm is
// rA = Rot(aA) * localAnchorA.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_types.cuh"   // GBIslandData, V2/Rot/Xf

// A distance joint. m_length is the rest separation. m_frequencyHz == 0 is the rigid
// rod; m_frequencyHz > 0 plus m_dampingRatio is the soft spring. m_gamma and m_bias are
// derived in InitVelocityConstraints for the soft case (0 for the rigid case).
struct GBDistanceJoint {
    int   indexA, indexB;
    V2    localAnchorA, localAnchorB;
    float invMassA, invMassB, invIA, invIB;
    float length;
    float frequencyHz, dampingRatio;
    int   jointIdx;
    // solver state (set by InitVelocityConstraints)
    V2    u, rA, rB;
    float mass, gamma, bias;
    float impulse;            // accumulated impulse (warm-start)
};

// b2DistanceJoint::InitVelocityConstraints. Builds the unit axis u, the effective mass,
// and the soft-constraint gamma/bias, then applies the warm-start impulse.
GB_HD inline void gbDistanceInitVelocity(GBDistanceJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    j.rA = b2MulRV(qA, j.localAnchorA);   // localCenter == 0
    j.rB = b2MulRV(qB, j.localAnchorB);
    j.u = cB + j.rB - cA - j.rA;

    float length = b2Length(j.u);
    if (length > GB_LINEAR_SLOP) j.u = (1.0f / length) * j.u;
    else j.u = v2(0.0f, 0.0f);

    float crAu = b2Cross(j.rA, j.u);
    float crBu = b2Cross(j.rB, j.u);
    float invMass = mA + iA*crAu*crAu + mB + iB*crBu*crBu;
    j.mass = invMass != 0.0f ? 1.0f / invMass : 0.0f;

    if (j.frequencyHz > 0.0f){
        float C = length - j.length;
        float omega = 2.0f * GB_PI * j.frequencyHz;
        float d = 2.0f * j.mass * j.dampingRatio * omega;          // damping coefficient
        float k = j.mass * omega * omega;                          // spring stiffness
        float h = GB_DT;
        j.gamma = h * (d + h * k);
        j.gamma = j.gamma != 0.0f ? 1.0f / j.gamma : 0.0f;
        j.bias = C * h * k * j.gamma;
        invMass += j.gamma;
        j.mass = invMass != 0.0f ? 1.0f / invMass : 0.0f;
    } else {
        j.gamma = 0.0f;
        j.bias = 0.0f;
    }

    // warm start (dtRatio == 1 in steady DT)
    V2 P = j.impulse * j.u;
    vA = vA - mA*P; wA -= iA*b2Cross(j.rA, P);
    vB = vB + mB*P; wB += iB*b2Cross(j.rB, P);
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2DistanceJoint::SolveVelocityConstraints, ONE iteration.
GB_HD inline void gbDistanceSolveVelocity(GBDistanceJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    V2 vpA = vA + b2CrossSV(wA, j.rA);
    V2 vpB = vB + b2CrossSV(wB, j.rB);
    float Cdot = b2Dot(j.u, vpB - vpA);
    float impulse = -j.mass * (Cdot + j.bias + j.gamma * j.impulse);
    j.impulse += impulse;
    V2 P = impulse * j.u;
    vA = vA - mA*P; wA -= iA*b2Cross(j.rA, P);
    vB = vB + mB*P; wB += iB*b2Cross(j.rB, P);
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2DistanceJoint::SolvePositionConstraints, ONE iteration. The soft spring carries no
// position correction (returns true). The rigid rod corrects the length error and
// returns whether it is within the linear slop.
GB_HD inline bool gbDistanceSolvePosition(GBDistanceJoint& j, GBIslandData& isl){
    if (j.frequencyHz > 0.0f) return true;   // soft constraint has no position error
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    V2 rA = b2MulRV(qA, j.localAnchorA);
    V2 rB = b2MulRV(qB, j.localAnchorB);
    V2 u = cB + rB - cA - rA;
    float length = b2Normalize(u);
    float C = length - j.length;
    C = b2ClampF(C, -GB_MAX_LINEAR_CORRECTION, GB_MAX_LINEAR_CORRECTION);
    float impulse = -j.mass * C;
    V2 P = impulse * u;
    cA = cA - mA*P; aA -= iA*b2Cross(rA, P);
    cB = cB + mB*P; aB += iB*b2Cross(rB, P);
    isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;
    return b2AbsF(C) < GB_LINEAR_SLOP;
}
