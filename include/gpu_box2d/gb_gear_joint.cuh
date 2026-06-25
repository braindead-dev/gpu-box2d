// gb_gear_joint.cuh. b2GearJoint (Box2D 2.3.0), written against the gb_* type universe.
// A gear joint couples two other joints (each revolute or prismatic) with a ratio, so
// turning one drives the other: the meshed-gears and rack-and-pinion model. The coupled
// coordinate is coordinateA + ratio * coordinateB = constant, where each coordinate is a
// joint angle (revolute) or a translation along an axis (prismatic).
//
// This header ports the four-body constraint: the Jacobian over the two driven bodies
// and the two reference (ground) bodies, the single-row velocity solve, and the position
// solve. The math and the evaluation order match Box2D 2.3.0 b2GearJoint::
// InitVelocityConstraints / SolveVelocityConstraints / SolvePositionConstraints for both
// the revolute-revolute and the prismatic couplings.
//
// Box2D-source faithfulness:
//   Dynamics/Joints/b2GearJoint.cpp
//
// localCenter is 0 in the gb_* body model, so a body-local anchor maps to the world arm
// rX = Rot(aX) * localAnchorX. The four bodies are indexA (driven A), indexB (driven B),
// indexC (joint1 ground), indexD (joint2 ground).
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_types.cuh"   // GBIslandData, V2/Rot

// Coupled joint type tags, matching b2JointType for the two cases a gear couples.
#define GB_GEAR_REVOLUTE  0
#define GB_GEAR_PRISMATIC 1

// A gear joint. typeA / typeB select how each side reads its coordinate. For a revolute
// side, referenceAngle is the joint's reference angle. For a prismatic side, localAxis
// and localAnchor on the driven and ground bodies define the slide. ratio scales side B.
// constant is coordinateA + ratio * coordinateB at construction. m_impulse warm-starts.
struct GBGearJoint {
    int   indexA, indexB, indexC, indexD;
    int   typeA, typeB;
    // revolute reference angles (used when the matching side is revolute)
    float referenceAngleA, referenceAngleB;
    // prismatic data (used when the matching side is prismatic)
    V2    localAnchorA, localAnchorB, localAnchorC, localAnchorD;
    V2    localAxisC, localAxisD;
    float ratio, constant;
    // inverse mass / inertia of the four bodies
    float invMassA, invMassB, invMassC, invMassD;
    float invIA, invIB, invIC, invID;
    int   jointIdx;
    // solver state (set by InitVelocityConstraints)
    V2    JvAC, JvBD;
    float JwA, JwB, JwC, JwD;
    float mass;
    float impulse;
};

// b2GearJoint::InitVelocityConstraints. Builds the Jacobian and the effective mass, then
// applies the warm-start impulse to all four bodies.
GB_HD inline void gbGearInitVelocity(GBGearJoint& j, GBIslandData& isl){
    int ia=j.indexA, ib=j.indexB, ic=j.indexC, id=j.indexD;
    float mA=j.invMassA, mB=j.invMassB, mC=j.invMassC, mD=j.invMassD;
    float iA=j.invIA, iB=j.invIB, iC=j.invIC, iD=j.invID;
    float aA=isl.posA[ia], aB=isl.posA[ib], aC=isl.posA[ic], aD=isl.posA[id];
    V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
    V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
    V2 vC=isl.vel[ic]; float wC=isl.velW[ic];
    V2 vD=isl.vel[id]; float wD=isl.velW[id];
    Rot qA=rotSet(aA), qB=rotSet(aB), qC=rotSet(aC), qD=rotSet(aD);

    j.mass = 0.0f;
    if (j.typeA == GB_GEAR_REVOLUTE){
        j.JvAC = v2(0.0f, 0.0f);
        j.JwA = 1.0f; j.JwC = 1.0f;
        j.mass += iA + iC;
    } else {
        V2 u = b2MulRV(qC, j.localAxisC);
        V2 rC = b2MulRV(qC, j.localAnchorC);   // localCenter == 0
        V2 rA = b2MulRV(qA, j.localAnchorA);
        j.JvAC = u; j.JwC = b2Cross(rC, u); j.JwA = b2Cross(rA, u);
        j.mass += mC + mA + iC*j.JwC*j.JwC + iA*j.JwA*j.JwA;
    }
    if (j.typeB == GB_GEAR_REVOLUTE){
        j.JvBD = v2(0.0f, 0.0f);
        j.JwB = j.ratio; j.JwD = j.ratio;
        j.mass += j.ratio*j.ratio*(iB + iD);
    } else {
        V2 u = b2MulRV(qD, j.localAxisD);
        V2 rD = b2MulRV(qD, j.localAnchorD);
        V2 rB = b2MulRV(qB, j.localAnchorB);
        j.JvBD = j.ratio * u; j.JwD = j.ratio*b2Cross(rD, u); j.JwB = j.ratio*b2Cross(rB, u);
        j.mass += j.ratio*j.ratio*(mD + mB) + iD*j.JwD*j.JwD + iB*j.JwB*j.JwB;
    }
    j.mass = j.mass > 0.0f ? 1.0f / j.mass : 0.0f;

    // warm start (dtRatio == 1 in steady DT)
    vA = vA + (mA*j.impulse)*j.JvAC; wA += iA*j.impulse*j.JwA;
    vB = vB + (mB*j.impulse)*j.JvBD; wB += iB*j.impulse*j.JwB;
    vC = vC - (mC*j.impulse)*j.JvAC; wC -= iC*j.impulse*j.JwC;
    vD = vD - (mD*j.impulse)*j.JvBD; wD -= iD*j.impulse*j.JwD;
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
    isl.vel[ic]=vC; isl.velW[ic]=wC; isl.vel[id]=vD; isl.velW[id]=wD;
}

// b2GearJoint::SolveVelocityConstraints, ONE iteration.
GB_HD inline void gbGearSolveVelocity(GBGearJoint& j, GBIslandData& isl){
    int ia=j.indexA, ib=j.indexB, ic=j.indexC, id=j.indexD;
    float mA=j.invMassA, mB=j.invMassB, mC=j.invMassC, mD=j.invMassD;
    float iA=j.invIA, iB=j.invIB, iC=j.invIC, iD=j.invID;
    V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
    V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
    V2 vC=isl.vel[ic]; float wC=isl.velW[ic];
    V2 vD=isl.vel[id]; float wD=isl.velW[id];

    float Cdot = b2Dot(j.JvAC, vA - vC) + b2Dot(j.JvBD, vB - vD);
    Cdot += (j.JwA*wA - j.JwC*wC) + (j.JwB*wB - j.JwD*wD);
    float impulse = -j.mass * Cdot;
    j.impulse += impulse;

    vA = vA + (mA*impulse)*j.JvAC; wA += iA*impulse*j.JwA;
    vB = vB + (mB*impulse)*j.JvBD; wB += iB*impulse*j.JwB;
    vC = vC - (mC*impulse)*j.JvAC; wC -= iC*impulse*j.JwC;
    vD = vD - (mD*impulse)*j.JvBD; wD -= iD*impulse*j.JwD;
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
    isl.vel[ic]=vC; isl.velW[ic]=wC; isl.vel[id]=vD; isl.velW[id]=wD;
}

// b2GearJoint::SolvePositionConstraints, ONE iteration. Box2D rebuilds the Jacobian,
// forms the coupled coordinate error, applies the correction to all four bodies, and
// reports solved (the linear error it tracks is 0). Returns true.
GB_HD inline bool gbGearSolvePosition(GBGearJoint& j, GBIslandData& isl){
    int ia=j.indexA, ib=j.indexB, ic=j.indexC, id=j.indexD;
    float mA=j.invMassA, mB=j.invMassB, mC=j.invMassC, mD=j.invMassD;
    float iA=j.invIA, iB=j.invIB, iC=j.invIC, iD=j.invID;
    V2 cA=isl.posC[ia]; float aA=isl.posA[ia];
    V2 cB=isl.posC[ib]; float aB=isl.posA[ib];
    V2 cC=isl.posC[ic]; float aC=isl.posA[ic];
    V2 cD=isl.posC[id]; float aD=isl.posA[id];
    Rot qA=rotSet(aA), qB=rotSet(aB), qC=rotSet(aC), qD=rotSet(aD);

    float linearError = 0.0f;
    V2 JvAC, JvBD; float JwA, JwB, JwC, JwD; float mass = 0.0f;
    float coordinateA, coordinateB;

    if (j.typeA == GB_GEAR_REVOLUTE){
        JvAC = v2(0.0f, 0.0f); JwA = 1.0f; JwC = 1.0f;
        mass += iA + iC;
        coordinateA = aA - aC - j.referenceAngleA;
    } else {
        V2 u = b2MulRV(qC, j.localAxisC);
        V2 rC = b2MulRV(qC, j.localAnchorC);
        V2 rA = b2MulRV(qA, j.localAnchorA);
        JvAC = u; JwC = b2Cross(rC, u); JwA = b2Cross(rA, u);
        mass += mC + mA + iC*JwC*JwC + iA*JwA*JwA;
        V2 pC = j.localAnchorC;   // localCenter == 0
        V2 pA = b2MulTinvV_q(qC, rA + (cA - cC));
        coordinateA = b2Dot(pA - pC, j.localAxisC);
    }
    if (j.typeB == GB_GEAR_REVOLUTE){
        JvBD = v2(0.0f, 0.0f); JwB = j.ratio; JwD = j.ratio;
        mass += j.ratio*j.ratio*(iB + iD);
        coordinateB = aB - aD - j.referenceAngleB;
    } else {
        V2 u = b2MulRV(qD, j.localAxisD);
        V2 rD = b2MulRV(qD, j.localAnchorD);
        V2 rB = b2MulRV(qB, j.localAnchorB);
        JvBD = j.ratio * u; JwD = j.ratio*b2Cross(rD, u); JwB = j.ratio*b2Cross(rB, u);
        mass += j.ratio*j.ratio*(mD + mB) + iD*JwD*JwD + iB*JwB*JwB;
        V2 pD = j.localAnchorD;
        V2 pB = b2MulTinvV_q(qD, rB + (cB - cD));
        coordinateB = b2Dot(pB - pD, j.localAxisD);
    }

    float C = (coordinateA + j.ratio * coordinateB) - j.constant;
    float impulse = 0.0f;
    if (mass > 0.0f) impulse = -C / mass;

    cA = cA + (mA*impulse)*JvAC; aA += iA*impulse*JwA;
    cB = cB + (mB*impulse)*JvBD; aB += iB*impulse*JwB;
    cC = cC - (mC*impulse)*JvAC; aC -= iC*impulse*JwC;
    cD = cD - (mD*impulse)*JvBD; aD -= iD*impulse*JwD;
    isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;
    isl.posC[ic]=cC; isl.posA[ic]=aC; isl.posC[id]=cD; isl.posA[id]=aD;

    return linearError < GB_LINEAR_SLOP;
}
