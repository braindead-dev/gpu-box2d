// gb_revolute_joint.cuh. b2RevoluteJoint (Box2D 2.3.0), the full joint with the motor
// and the angle limit, written against the gb_* type universe. A revolute joint pins two
// bodies at a shared world anchor so they rotate about it; the motor drives the relative
// angle toward a target speed, and the limit bounds the relative angle between a lower
// and an upper stop.
//
// This header ports the 3x3 path: the 2x2 point-to-point anchor block plus the angular
// motor and limit row. The point-to-point-only case lives in gb_joint.cuh and is wired
// into the assembled step; this module is the complete joint, validated standalone, for
// the motor-and-limit cases. The math and the evaluation order match Box2D 2.3.0
// b2RevoluteJoint::InitVelocityConstraints / SolveVelocityConstraints /
// SolvePositionConstraints.
//
// Box2D-source faithfulness:
//   Dynamics/Joints/b2RevoluteJoint.cpp
//
// localCenter is 0 in the gb_* body model, so the world anchor arm is
// rA = Rot(aA) * localAnchorA.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_types.cuh"   // GBIslandData, GBMat33 + ops, gbMat22SolveV, V2/V3/Rot

// b2LimitState reused from the prismatic joint set.
#ifndef GB_INACTIVE_LIMIT
#define GB_INACTIVE_LIMIT 0
#define GB_AT_LOWER_LIMIT 1
#define GB_AT_UPPER_LIMIT 2
#define GB_EQUAL_LIMITS   3
#endif

// A full revolute joint. referenceAngle is the relative angle at construction.
// m_impulse is (anchor.x, anchor.y, angular-limit); m_motorImpulse is the motor.
struct GBRevoluteJointFull {
    int   indexA, indexB;
    V2    localAnchorA, localAnchorB;
    float referenceAngle;
    float invMassA, invMassB, invIA, invIB;
    int   enableMotor; float maxMotorTorque, motorSpeed;
    int   enableLimit; float lowerAngle, upperAngle;
    int   jointIdx;
    // solver state (set by InitVelocityConstraints)
    V2    rA, rB;
    GBMat33 mass;
    float motorMass;
    int   limitState;
    // accumulated impulses (warm-start)
    V3    impulse;
    float motorImpulse;
};

// b2RevoluteJoint::InitVelocityConstraints (full).
GB_HD inline void gbRevoluteFullInitVelocity(GBRevoluteJointFull& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    float aA = isl.posA[ia], aB = isl.posA[ib];
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    j.rA = b2MulRV(qA, j.localAnchorA);   // localCenter == 0
    j.rB = b2MulRV(qB, j.localAnchorB);

    bool fixedRotation = (iA + iB == 0.0f);
    GBMat33& K = j.mass;
    K.ex.x = mA + mB + j.rA.y*j.rA.y*iA + j.rB.y*j.rB.y*iB;
    K.ey.x = -j.rA.y*j.rA.x*iA - j.rB.y*j.rB.x*iB;
    K.ez.x = -j.rA.y*iA - j.rB.y*iB;
    K.ex.y = K.ey.x;
    K.ey.y = mA + mB + j.rA.x*j.rA.x*iA + j.rB.x*j.rB.x*iB;
    K.ez.y = j.rA.x*iA + j.rB.x*iB;
    K.ex.z = K.ez.x;
    K.ey.z = K.ez.y;
    K.ez.z = iA + iB;

    j.motorMass = iA + iB;
    if (j.motorMass > 0.0f) j.motorMass = 1.0f / j.motorMass;
    if (j.enableMotor == 0 || fixedRotation) j.motorImpulse = 0.0f;

    if (j.enableLimit && !fixedRotation){
        float jointAngle = aB - aA - j.referenceAngle;
        if (b2AbsF(j.upperAngle - j.lowerAngle) < 2.0f * GB_ANGULAR_SLOP){
            j.limitState = GB_EQUAL_LIMITS;
        } else if (jointAngle <= j.lowerAngle){
            if (j.limitState != GB_AT_LOWER_LIMIT) j.impulse.z = 0.0f;
            j.limitState = GB_AT_LOWER_LIMIT;
        } else if (jointAngle >= j.upperAngle){
            if (j.limitState != GB_AT_UPPER_LIMIT) j.impulse.z = 0.0f;
            j.limitState = GB_AT_UPPER_LIMIT;
        } else {
            j.limitState = GB_INACTIVE_LIMIT;
            j.impulse.z = 0.0f;
        }
    } else {
        j.limitState = GB_INACTIVE_LIMIT;
    }

    // warm start (dtRatio == 1 in steady DT)
    V2 P = v2(j.impulse.x, j.impulse.y);
    vA = vA - mA*P; wA -= iA*(b2Cross(j.rA, P) + j.motorImpulse + j.impulse.z);
    vB = vB + mB*P; wB += iB*(b2Cross(j.rB, P) + j.motorImpulse + j.impulse.z);
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2RevoluteJoint::SolveVelocityConstraints (full), ONE iteration.
GB_HD inline void gbRevoluteFullSolveVelocity(GBRevoluteJointFull& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    bool fixedRotation = (iA + iB == 0.0f);

    // motor
    if (j.enableMotor && j.limitState != GB_EQUAL_LIMITS && !fixedRotation){
        float Cdot = wB - wA - j.motorSpeed;
        float impulse = -j.motorMass * Cdot;
        float oldImpulse = j.motorImpulse;
        float maxImpulse = GB_DT * j.maxMotorTorque;
        j.motorImpulse = b2ClampF(j.motorImpulse + impulse, -maxImpulse, maxImpulse);
        impulse = j.motorImpulse - oldImpulse;
        wA -= iA * impulse;
        wB += iB * impulse;
    }

    if (j.enableLimit && j.limitState != GB_INACTIVE_LIMIT && !fixedRotation){
        V2 Cdot1 = vB + b2CrossSV(wB, j.rB) - vA - b2CrossSV(wA, j.rA);
        float Cdot2 = wB - wA;
        V3 Cdot = v3(Cdot1.x, Cdot1.y, Cdot2);
        V3 impulse = -1.0f * gbMat33Solve33(j.mass, Cdot);
        if (j.limitState == GB_EQUAL_LIMITS){
            j.impulse = j.impulse + impulse;
        } else if (j.limitState == GB_AT_LOWER_LIMIT){
            float newImpulse = j.impulse.z + impulse.z;
            if (newImpulse < 0.0f){
                V2 rhs = -1.0f*Cdot1 + j.impulse.z * v2(j.mass.ez.x, j.mass.ez.y);
                V2 reduced = gbMat33Solve22(j.mass, rhs);
                impulse.x = reduced.x;
                impulse.y = reduced.y;
                impulse.z = -j.impulse.z;
                j.impulse.x += reduced.x;
                j.impulse.y += reduced.y;
                j.impulse.z = 0.0f;
            } else {
                j.impulse = j.impulse + impulse;
            }
        } else if (j.limitState == GB_AT_UPPER_LIMIT){
            float newImpulse = j.impulse.z + impulse.z;
            if (newImpulse > 0.0f){
                V2 rhs = -1.0f*Cdot1 + j.impulse.z * v2(j.mass.ez.x, j.mass.ez.y);
                V2 reduced = gbMat33Solve22(j.mass, rhs);
                impulse.x = reduced.x;
                impulse.y = reduced.y;
                impulse.z = -j.impulse.z;
                j.impulse.x += reduced.x;
                j.impulse.y += reduced.y;
                j.impulse.z = 0.0f;
            } else {
                j.impulse = j.impulse + impulse;
            }
        }
        V2 P = v2(impulse.x, impulse.y);
        vA = vA - mA*P; wA -= iA*(b2Cross(j.rA, P) + impulse.z);
        vB = vB + mB*P; wB += iB*(b2Cross(j.rB, P) + impulse.z);
    } else {
        // point-to-point 2x2
        V2 Cdot = vB + b2CrossSV(wB, j.rB) - vA - b2CrossSV(wA, j.rA);
        V2 impulse = gbMat33Solve22(j.mass, -1.0f*Cdot);
        j.impulse.x += impulse.x;
        j.impulse.y += impulse.y;
        V2 P = impulse;
        vA = vA - mA*P; wA -= iA*b2Cross(j.rA, P);
        vB = vB + mB*P; wB += iB*b2Cross(j.rB, P);
    }
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2RevoluteJoint::SolvePositionConstraints (full), ONE iteration. Returns true when the
// linear and angular errors are within slop.
GB_HD inline bool gbRevoluteFullSolvePosition(GBRevoluteJointFull& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    float angularError = 0.0f, positionError = 0.0f;
    bool fixedRotation = (iA + iB == 0.0f);

    // angular limit
    if (j.enableLimit && j.limitState != GB_INACTIVE_LIMIT && !fixedRotation){
        float angle = aB - aA - j.referenceAngle;
        float limitImpulse = 0.0f;
        if (j.limitState == GB_EQUAL_LIMITS){
            float C = b2ClampF(angle - j.lowerAngle, -GB_MAX_ANGULAR_CORRECTION, GB_MAX_ANGULAR_CORRECTION);
            limitImpulse = -j.motorMass * C;
            angularError = b2AbsF(C);
        } else if (j.limitState == GB_AT_LOWER_LIMIT){
            float C = angle - j.lowerAngle;
            angularError = -C;
            C = b2ClampF(C + GB_ANGULAR_SLOP, -GB_MAX_ANGULAR_CORRECTION, 0.0f);
            limitImpulse = -j.motorMass * C;
        } else if (j.limitState == GB_AT_UPPER_LIMIT){
            float C = angle - j.upperAngle;
            angularError = C;
            C = b2ClampF(C - GB_ANGULAR_SLOP, 0.0f, GB_MAX_ANGULAR_CORRECTION);
            limitImpulse = -j.motorMass * C;
        }
        aA -= iA * limitImpulse;
        aB += iB * limitImpulse;
    }

    // point-to-point
    Rot qA = rotSet(aA), qB = rotSet(aB);
    V2 rA = b2MulRV(qA, j.localAnchorA);
    V2 rB = b2MulRV(qB, j.localAnchorB);
    V2 C = cB + rB - cA - rA;
    positionError = b2Length(C);
    GBMat22 K;
    K.ex.x = mA + mB + iA*rA.y*rA.y + iB*rB.y*rB.y;
    K.ex.y = -iA*rA.x*rA.y - iB*rB.x*rB.y;
    K.ey.x = K.ex.y;
    K.ey.y = mA + mB + iA*rA.x*rA.x + iB*rB.x*rB.x;
    V2 impulse = -1.0f * gbMat22SolveV(K, C);
    cA = cA - mA*impulse; aA -= iA*b2Cross(rA, impulse);
    cB = cB + mB*impulse; aB += iB*b2Cross(rB, impulse);
    isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;

    return positionError <= GB_LINEAR_SLOP && angularError <= GB_ANGULAR_SLOP;
}
