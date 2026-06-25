// gb_prismatic_joint.cuh. b2PrismaticJoint (Box2D 2.3.0), written against the gb_* type
// universe. A prismatic joint lets two bodies slide along a shared axis while holding
// the relative angle fixed: it is the slider, the piston, and the suspension strut. It
// constrains the two perpendicular-to-axis degrees of freedom and the relative angle,
// and leaves translation along the axis free, optionally bounded by a limit and driven
// by a motor.
//
// This header ports the full joint: the 2x2 perpendicular-plus-angular block (rows that
// are always active), the motor row along the axis, and the limit row that activates the
// axis constraint at the lower or upper stop. The math and the evaluation order match
// Box2D 2.3.0 b2PrismaticJoint::InitVelocityConstraints / SolveVelocityConstraints /
// SolvePositionConstraints.
//
// Box2D-source faithfulness:
//   Dynamics/Joints/b2PrismaticJoint.cpp
//
// localCenter is 0 in the gb_* body model, so the world anchor arm is
// rA = Rot(aA) * localAnchorA and the axis is Rot(aA) * localXAxisA.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_types.cuh"   // GBIslandData, GBMat33 + ops, V2/V3/Rot

// b2LimitState.
#define GB_INACTIVE_LIMIT 0
#define GB_AT_LOWER_LIMIT 1
#define GB_AT_UPPER_LIMIT 2
#define GB_EQUAL_LIMITS   3

// A prismatic joint. localXAxisA is the slide axis in A's frame; localYAxisA is its
// perpendicular. referenceAngle is the relative angle (aB - aA) the joint holds.
// m_impulse is (perpendicular, angular, axis-limit); m_motorImpulse is the axis motor.
struct GBPrismaticJoint {
    int   indexA, indexB;
    V2    localAnchorA, localAnchorB;
    V2    localXAxisA, localYAxisA;
    float referenceAngle;
    float invMassA, invMassB, invIA, invIB;
    int   enableLimit; float lowerTranslation, upperTranslation;
    int   enableMotor; float maxMotorForce, motorSpeed;
    int   jointIdx;
    // solver state (set by InitVelocityConstraints)
    V2    axis, perp;
    float s1, s2, a1, a2;
    GBMat33 K;
    float motorMass;
    int   limitState;
    // accumulated impulses (warm-start)
    V3    impulse;
    float motorImpulse;
};

// b2PrismaticJoint::InitVelocityConstraints.
GB_HD inline void gbPrismaticInitVelocity(GBPrismaticJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    V2 rA = b2MulRV(qA, j.localAnchorA);   // localCenter == 0
    V2 rB = b2MulRV(qB, j.localAnchorB);
    V2 d = (cB - cA) + rB - rA;

    // motor and limit axis
    j.axis = b2MulRV(qA, j.localXAxisA);
    j.a1 = b2Cross(d + rA, j.axis);
    j.a2 = b2Cross(rB, j.axis);
    j.motorMass = mA + mB + iA*j.a1*j.a1 + iB*j.a2*j.a2;
    if (j.motorMass > 0.0f) j.motorMass = 1.0f / j.motorMass;

    // perpendicular constraint
    j.perp = b2MulRV(qA, j.localYAxisA);
    j.s1 = b2Cross(d + rA, j.perp);
    j.s2 = b2Cross(rB, j.perp);
    float k11 = mA + mB + iA*j.s1*j.s1 + iB*j.s2*j.s2;
    float k12 = iA*j.s1 + iB*j.s2;
    float k13 = iA*j.s1*j.a1 + iB*j.s2*j.a2;
    float k22 = iA + iB;
    if (k22 == 0.0f) k22 = 1.0f;   // both bodies fixed rotation, prevent singularity
    float k23 = iA*j.a1 + iB*j.a2;
    float k33 = mA + mB + iA*j.a1*j.a1 + iB*j.a2*j.a2;
    j.K.ex = v3(k11, k12, k13);
    j.K.ey = v3(k12, k22, k23);
    j.K.ez = v3(k13, k23, k33);

    // limit state
    if (j.enableLimit){
        float jointTranslation = b2Dot(j.axis, d);
        if (b2AbsF(j.upperTranslation - j.lowerTranslation) < 2.0f * GB_LINEAR_SLOP){
            j.limitState = GB_EQUAL_LIMITS;
        } else if (jointTranslation <= j.lowerTranslation){
            if (j.limitState != GB_AT_LOWER_LIMIT){ j.limitState = GB_AT_LOWER_LIMIT; j.impulse.z = 0.0f; }
        } else if (jointTranslation >= j.upperTranslation){
            if (j.limitState != GB_AT_UPPER_LIMIT){ j.limitState = GB_AT_UPPER_LIMIT; j.impulse.z = 0.0f; }
        } else {
            j.limitState = GB_INACTIVE_LIMIT; j.impulse.z = 0.0f;
        }
    } else {
        j.limitState = GB_INACTIVE_LIMIT; j.impulse.z = 0.0f;
    }

    if (j.enableMotor == 0) j.motorImpulse = 0.0f;

    // warm start (dtRatio == 1 in steady DT)
    V2 P = j.impulse.x*j.perp + (j.motorImpulse + j.impulse.z)*j.axis;
    float LA = j.impulse.x*j.s1 + j.impulse.y + (j.motorImpulse + j.impulse.z)*j.a1;
    float LB = j.impulse.x*j.s2 + j.impulse.y + (j.motorImpulse + j.impulse.z)*j.a2;
    vA = vA - mA*P; wA -= iA*LA;
    vB = vB + mB*P; wB += iB*LB;
    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2PrismaticJoint::SolveVelocityConstraints, ONE iteration.
GB_HD inline void gbPrismaticSolveVelocity(GBPrismaticJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 vA = isl.vel[ia]; float wA = isl.velW[ia];
    V2 vB = isl.vel[ib]; float wB = isl.velW[ib];

    // motor row
    if (j.enableMotor && j.limitState != GB_EQUAL_LIMITS){
        float Cdot = b2Dot(j.axis, vB - vA) + j.a2*wB - j.a1*wA;
        float impulse = j.motorMass * (j.motorSpeed - Cdot);
        float oldImpulse = j.motorImpulse;
        float maxImpulse = GB_DT * j.maxMotorForce;
        j.motorImpulse = b2ClampF(j.motorImpulse + impulse, -maxImpulse, maxImpulse);
        impulse = j.motorImpulse - oldImpulse;
        V2 P = impulse * j.axis;
        float LA = impulse * j.a1;
        float LB = impulse * j.a2;
        vA = vA - mA*P; wA -= iA*LA;
        vB = vB + mB*P; wB += iB*LB;
    }

    V2 Cdot1;
    Cdot1.x = b2Dot(j.perp, vB - vA) + j.s2*wB - j.s1*wA;
    Cdot1.y = wB - wA;

    if (j.enableLimit && j.limitState != GB_INACTIVE_LIMIT){
        // solve the prismatic and limit constraints together (3x3)
        float Cdot2 = b2Dot(j.axis, vB - vA) + j.a2*wB - j.a1*wA;
        V3 Cdot = v3(Cdot1.x, Cdot1.y, Cdot2);
        V3 f1 = j.impulse;
        V3 df = gbMat33Solve33(j.K, -Cdot);
        j.impulse = j.impulse + df;
        if (j.limitState == GB_AT_LOWER_LIMIT) j.impulse.z = b2MaxF(j.impulse.z, 0.0f);
        else if (j.limitState == GB_AT_UPPER_LIMIT) j.impulse.z = b2MinF(j.impulse.z, 0.0f);
        // f2(1:2) = invK(1:2,1:2) * (-Cdot(1:2) - K(1:2,3) * (f2(3) - f1(3))) + f1(1:2)
        V2 b = v2(-Cdot1.x - (j.impulse.z - f1.z) * j.K.ez.x,
                  -Cdot1.y - (j.impulse.z - f1.z) * j.K.ez.y);
        V2 f2r = gbMat33Solve22(j.K, b);
        f2r.x += f1.x; f2r.y += f1.y;
        j.impulse.x = f2r.x; j.impulse.y = f2r.y;
        df = j.impulse - f1;
        V2 P = df.x*j.perp + df.z*j.axis;
        float LA = df.x*j.s1 + df.y + df.z*j.a1;
        float LB = df.x*j.s2 + df.y + df.z*j.a2;
        vA = vA - mA*P; wA -= iA*LA;
        vB = vB + mB*P; wB += iB*LB;
    } else {
        // limit is inactive, just solve the 2x2 prismatic constraint
        V2 df = gbMat33Solve22(j.K, -Cdot1);
        j.impulse.x += df.x;
        j.impulse.y += df.y;
        V2 P = df.x * j.perp;
        float LA = df.x*j.s1 + df.y;
        float LB = df.x*j.s2 + df.y;
        vA = vA - mA*P; wA -= iA*LA;
        vB = vB + mB*P; wB += iB*LB;
    }

    isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
}

// b2PrismaticJoint::SolvePositionConstraints, ONE iteration. Returns true when the
// linear and angular errors are within slop, including the active limit error.
GB_HD inline bool gbPrismaticSolvePosition(GBPrismaticJoint& j, GBIslandData& isl){
    int ia = j.indexA, ib = j.indexB;
    float mA = j.invMassA, mB = j.invMassB, iA = j.invIA, iB = j.invIB;
    V2 cA = isl.posC[ia]; float aA = isl.posA[ia];
    V2 cB = isl.posC[ib]; float aB = isl.posA[ib];
    Rot qA = rotSet(aA), qB = rotSet(aB);
    V2 rA = b2MulRV(qA, j.localAnchorA);
    V2 rB = b2MulRV(qB, j.localAnchorB);
    V2 d = cB + rB - cA - rA;

    V2 axis = b2MulRV(qA, j.localXAxisA);
    float a1 = b2Cross(d + rA, axis);
    float a2 = b2Cross(rB, axis);
    V2 perp = b2MulRV(qA, j.localYAxisA);
    float s1 = b2Cross(d + rA, perp);
    float s2 = b2Cross(rB, perp);

    V3 impulse;
    V2 C1;
    C1.x = b2Dot(perp, d);
    C1.y = aB - aA - j.referenceAngle;

    float linearError = b2AbsF(C1.x);
    float angularError = b2AbsF(C1.y);

    bool active = false;
    float C2 = 0.0f;
    if (j.enableLimit){
        float translation = b2Dot(axis, d);
        if (b2AbsF(j.upperTranslation - j.lowerTranslation) < 2.0f * GB_LINEAR_SLOP){
            C2 = b2ClampF(translation, -GB_MAX_LINEAR_CORRECTION, GB_MAX_LINEAR_CORRECTION);
            linearError = b2MaxF(linearError, b2AbsF(translation));
            active = true;
        } else if (translation <= j.lowerTranslation){
            C2 = b2ClampF(translation - j.lowerTranslation + GB_LINEAR_SLOP, -GB_MAX_LINEAR_CORRECTION, 0.0f);
            linearError = b2MaxF(linearError, j.lowerTranslation - translation);
            active = true;
        } else if (translation >= j.upperTranslation){
            C2 = b2ClampF(translation - j.upperTranslation - GB_LINEAR_SLOP, 0.0f, GB_MAX_LINEAR_CORRECTION);
            linearError = b2MaxF(linearError, translation - j.upperTranslation);
            active = true;
        }
    }

    if (active){
        float k11 = mA + mB + iA*s1*s1 + iB*s2*s2;
        float k12 = iA*s1 + iB*s2;
        float k13 = iA*s1*a1 + iB*s2*a2;
        float k22 = iA + iB;
        if (k22 == 0.0f) k22 = 1.0f;
        float k23 = iA*a1 + iB*a2;
        float k33 = mA + mB + iA*a1*a1 + iB*a2*a2;
        GBMat33 K;
        K.ex = v3(k11, k12, k13);
        K.ey = v3(k12, k22, k23);
        K.ez = v3(k13, k23, k33);
        V3 C = v3(C1.x, C1.y, C2);
        impulse = gbMat33Solve33(K, -C);
    } else {
        float k11 = mA + mB + iA*s1*s1 + iB*s2*s2;
        float k12 = iA*s1 + iB*s2;
        float k22 = iA + iB;
        if (k22 == 0.0f) k22 = 1.0f;
        GBMat22 K;
        K.ex = v2(k11, k12);
        K.ey = v2(k12, k22);
        V2 impulse1 = gbMat22SolveV(K, -1.0f*C1);
        impulse.x = impulse1.x;
        impulse.y = impulse1.y;
        impulse.z = 0.0f;
    }

    V2 P = impulse.x*perp + impulse.z*axis;
    float LA = impulse.x*s1 + impulse.y + impulse.z*a1;
    float LB = impulse.x*s2 + impulse.y + impulse.z*a2;
    cA = cA - mA*P; aA -= iA*LA;
    cB = cB + mB*P; aB += iB*LB;
    isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;

    return linearError <= GB_LINEAR_SLOP && angularError <= GB_ANGULAR_SLOP;
}
