// gb_weld_joint.cuh. b2WeldJoint (Box2D 2.3.0), written against the gb_* type universe.
// A weld joint fixes two bodies together: it holds a shared anchor point coincident
// (the two linear rows) and the relative angle constant (the angular row), so the pair
// moves as one rigid body. With a positive frequency the angular row becomes a soft
// torsional spring, the breakable-structure model.
//
// This header ports the 3x3 path: the linear anchor constraint and the angular
// constraint solved together with the 3x3 effective-mass matrix in the velocity solve
// and the position solve. The rigid case uses the symmetric 3x3 inverse; the soft case
// inverts the 2x2 linear block and carries the angular row through a bias and gamma.
// The math and the evaluation order match Box2D 2.3.0 b2WeldJoint::
// InitVelocityConstraints / SolveVelocityConstraints / SolvePositionConstraints.
//
// Box2D-source faithfulness:
//   Dynamics/Joints/b2WeldJoint.cpp
//
// localCenter is 0 in the gb_* body model, so the world anchor arm is
// rA = Rot(aA) * localAnchorA.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_types.cuh"   // GBIslandData, GBMat33 + ops, V2/V3/Rot

// A weld joint. m_referenceAngle is the relative angle (aB - aA) held by the joint.
// m_frequencyHz == 0 is the rigid weld; m_frequencyHz > 0 plus m_dampingRatio is the
// soft torsional spring on the angular row. m_impulse is the accumulated 3-impulse
// (x, y linear; z angular) carried for warm-start.
struct GBWeldJoint {
    int   indexA, indexB;
    V2    localAnchorA, localAnchorB;
    float referenceAngle;
    float invMassA, invMassB, invIA, invIB;
    float frequencyHz, dampingRatio;
    int   jointIdx;
    // solver state (set by InitVelocityConstraints)
    V2    rA, rB;
    GBMat33 mass;
    float gamma, bias;
    V3    impulse;
};

// b2WeldJoint::InitVelocityConstraints. Builds the arms, the 3x3 K matrix, the
// effective-mass matrix (3x3 inverse for the rigid case, 2x2 inverse plus the soft
// angular row otherwise), and applies the warm-start impulse.
GB_HD inline void gbWeldInitVelocity(GBWeldJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    float aA = isl.posA[ia], aB = isl.posA[ib];
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    j.rA = b2MulRV(qA, j.localAnchorA);   // localCenter == 0
    j.rB = b2MulRV(qB, j.localAnchorB);

    GBMat33 K;
    K.ex.x = mA + mB + j.rA.y*j.rA.y*iA + j.rB.y*j.rB.y*iB;
    K.ey.x = -j.rA.y*j.rA.x*iA - j.rB.y*j.rB.x*iB;
    K.ez.x = -j.rA.y*iA - j.rB.y*iB;
    K.ex.y = K.ey.x;
    K.ey.y = mA + mB + j.rA.x*j.rA.x*iA + j.rB.x*j.rB.x*iB;
    K.ez.y = j.rA.x*iA + j.rB.x*iB;
    K.ex.z = K.ez.x;
    K.ey.z = K.ez.y;
    K.ez.z = iA + iB;

    if (j.frequencyHz > 0.0f){
        j.mass = gbMat33GetInverse22(K);
        float invM = iA + iB;
        float m = invM > 0.0f ? 1.0f / invM : 0.0f;
        float C = aB - aA - j.referenceAngle;
        float omega = 2.0f * GB_PI * j.frequencyHz;
        float d = 2.0f * m * j.dampingRatio * omega;
        float k = m * omega * omega;
        float h = GB_DT;
        j.gamma = h * (d + h * k);
        j.gamma = j.gamma != 0.0f ? 1.0f / j.gamma : 0.0f;
        j.bias = C * h * k * j.gamma;
        invM += j.gamma;
        j.mass.ez.z = invM != 0.0f ? 1.0f / invM : 0.0f;
    } else if (K.ez.z == 0.0f){
        j.mass = gbMat33GetInverse22(K);
        j.gamma = 0.0f;
        j.bias = 0.0f;
    } else {
        j.mass = gbMat33GetSymInverse33(K);
        j.gamma = 0.0f;
        j.bias = 0.0f;
    }

    // warm start (dtRatio == 1 in steady DT)
    V2 P = v2(j.impulse.x, j.impulse.y);
    vA = vA - mA*P; wA -= iA*(b2Cross(j.rA, P) + j.impulse.z);
    vB = vB + mB*P; wB += iB*(b2Cross(j.rB, P) + j.impulse.z);
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2WeldJoint::SolveVelocityConstraints, ONE iteration.
GB_HD inline void gbWeldSolveVelocity(GBWeldJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];

    if (j.frequencyHz > 0.0f){
        float Cdot2 = wB - wA;
        float impulse2 = -j.mass.ez.z * (Cdot2 + j.bias + j.gamma * j.impulse.z);
        j.impulse.z += impulse2;
        wA -= iA * impulse2;
        wB += iB * impulse2;
        V2 Cdot1 = vB + b2CrossSV(wB, j.rB) - vA - b2CrossSV(wA, j.rA);
        V2 impulse1 = -1.0f * gbMulM33V2(j.mass, Cdot1);
        j.impulse.x += impulse1.x;
        j.impulse.y += impulse1.y;
        V2 P = impulse1;
        vA = vA - mA*P; wA -= iA*b2Cross(j.rA, P);
        vB = vB + mB*P; wB += iB*b2Cross(j.rB, P);
    } else {
        V2 Cdot1lin = vB + b2CrossSV(wB, j.rB) - vA - b2CrossSV(wA, j.rA);
        float Cdot2 = wB - wA;
        V3 Cdot = v3(Cdot1lin.x, Cdot1lin.y, Cdot2);
        V3 impulse = -1.0f * gbMulM33V3(j.mass, Cdot);
        j.impulse = j.impulse + impulse;
        V2 P = v2(impulse.x, impulse.y);
        vA = vA - mA*P; wA -= iA*(b2Cross(j.rA, P) + impulse.z);
        vB = vB + mB*P; wB += iB*(b2Cross(j.rB, P) + impulse.z);
    }
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2WeldJoint::SolvePositionConstraints, ONE iteration. Returns true when the linear
// error is within the linear slop and the angular error is within the angular slop.
GB_HD inline bool gbWeldSolvePosition(GBWeldJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    V2 rA = b2MulRV(qA, j.localAnchorA);
    V2 rB = b2MulRV(qB, j.localAnchorB);

    float positionError, angularError;
    GBMat33 K;
    K.ex.x = mA + mB + rA.y*rA.y*iA + rB.y*rB.y*iB;
    K.ey.x = -rA.y*rA.x*iA - rB.y*rB.x*iB;
    K.ez.x = -rA.y*iA - rB.y*iB;
    K.ex.y = K.ey.x;
    K.ey.y = mA + mB + rA.x*rA.x*iA + rB.x*rB.x*iB;
    K.ez.y = rA.x*iA + rB.x*iB;
    K.ex.z = K.ez.x;
    K.ey.z = K.ez.y;
    K.ez.z = iA + iB;

    if (j.frequencyHz > 0.0f){
        V2 C1 = cB + rB - cA - rA;
        positionError = b2Length(C1);
        angularError = 0.0f;
        V2 P = -1.0f * gbMat33Solve22(K, C1);
        cA = cA - mA*P; aA -= iA*b2Cross(rA, P);
        cB = cB + mB*P; aB += iB*b2Cross(rB, P);
    } else {
        V2 C1 = cB + rB - cA - rA;
        float C2 = aB - aA - j.referenceAngle;
        positionError = b2Length(C1);
        angularError = b2AbsF(C2);
        V3 C = v3(C1.x, C1.y, C2);
        V3 impulse;
        if (K.ez.z > 0.0f){
            impulse = -1.0f * gbMat33Solve33(K, C);
        } else {
            V2 impulse2 = -1.0f * gbMat33Solve22(K, C1);
            impulse = v3(impulse2.x, impulse2.y, 0.0f);
        }
        V2 P = v2(impulse.x, impulse.y);
        cA = cA - mA*P; aA -= iA*(b2Cross(rA, P) + impulse.z);
        cB = cB + mB*P; aB += iB*(b2Cross(rB, P) + impulse.z);
    }
    isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;
    return positionError <= GB_LINEAR_SLOP && angularError <= GB_ANGULAR_SLOP;
}
