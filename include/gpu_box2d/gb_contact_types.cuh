// gb_contact_types.cuh. The shared cross-module contact and solver interface.
// The narrow-phase, contact solver, island, and CCD headers all include this so they
// compose against one definition of the manifold, constraint, and island types and
// of the phase-function signatures. The types here (the fused single-point
// constraint and the one-point manifold) are the validated definitions. Their layout
// is fidelity-critical: every solver phase reads and writes these fields in Box2D's
// exact order, so changing them changes the floats.
#pragma once
#include "gb_pools.cuh"

// ---- manifold (up to GB_MAX_MANIFOLD_POINTS points) ----------------------------
// Circles and circle-edge produce one point. Polygon contacts produce one or two.
// pointCount drives every loop, so a 1-point manifold uses point 0 alone and the
// 1-point path stays byte-identical. localPoint and pLocalPoint name point 0 for
// back-compatibility with the single-point modules; pLocalPoint2 and the contact
// ids carry the second clip point.
struct GBManifold {
    int type;          // GB_MANIFOLD_CIRCLES / GB_MANIFOLD_FACE_A / GB_MANIFOLD_FACE_B
    int pointCount;    // 0, 1, or 2
    V2  localNormal;   // for face-A / face-B
    V2  localPoint;    // manifold.localPoint (reference face center or circleA center)
    V2  pLocalPoint;   // points[0].localPoint
    V2  pLocalPoint2;  // points[1].localPoint
    unsigned int id0, id1;   // points[0].id.key, points[1].id.key (warm-start match)
};
// world-space contact points (up to GB_MAX_MANIFOLD_POINTS)
struct GBWorldManifold { V2 normal; V2 point0; V2 point1; };

// ---- fused velocity+position constraint (up to two points; localCenter==0) -----
struct GBVelConstraintPt {
    V2 rA, rB; float normalImpulse, tangentImpulse, normalMass, tangentMass, velocityBias;
};
// 2x2 block matrix (b2Mat22), column-major like Box2D: ex=(ex.x,ex.y), ey=(ey.x,ey.y).
struct GBMat22 { V2 ex, ey; };

// b2Mat22::GetInverse. The same float operations and ordering as Box2D 2.3.0.
GB_HD inline GBMat22 gbMat22GetInverse(const GBMat22& m){
    float a = m.ex.x, b = m.ey.x, c = m.ex.y, d = m.ey.y;
    GBMat22 B;
    float det = a*d - b*c;
    if (det != 0.0f) det = 1.0f / det;
    B.ex.x =  det*d;  B.ey.x = -det*b;
    B.ex.y = -det*c;  B.ey.y =  det*a;
    return B;
}
// b2Mul(b2Mat22, b2Vec2).
GB_HD inline V2 gbMulMV(const GBMat22& A, V2 v){
    return v2(A.ex.x*v.x + A.ey.x*v.y, A.ex.y*v.x + A.ey.y*v.y);
}

// 3x3 matrix (b2Mat33), column-major (ex, ey, ez are columns). Used by the 3x3 joint
// solves (weld, and the revolute motor/limit path). The ops below are line-faithful to
// Box2D 2.3.0 b2Mat33.
struct GBMat33 { V3 ex, ey, ez; };

// b2Mat33::Solve33. Solves A * x = b for a 3x3 system.
GB_HD inline V3 gbMat33Solve33(const GBMat33& A, V3 b){
    float det = b2Dot3(A.ex, b2Cross3(A.ey, A.ez));
    if (det != 0.0f) det = 1.0f / det;
    V3 x;
    x.x = det * b2Dot3(b, b2Cross3(A.ey, A.ez));
    x.y = det * b2Dot3(A.ex, b2Cross3(b, A.ez));
    x.z = det * b2Dot3(A.ex, b2Cross3(A.ey, b));
    return x;
}
// b2Mat33::Solve22. Solves the upper-left 2x2 block for a 2-vector b.
GB_HD inline V2 gbMat33Solve22(const GBMat33& A, V2 b){
    float a11 = A.ex.x, a12 = A.ey.x, a21 = A.ex.y, a22 = A.ey.y;
    float det = a11*a22 - a12*a21;
    if (det != 0.0f) det = 1.0f / det;
    return v2(det*(a22*b.x - a12*b.y), det*(a11*b.y - a21*b.x));
}
// b2Mat33::GetInverse22. Inverse of the upper-left 2x2 block; the third row/column of
// the result is zeroed.
GB_HD inline GBMat33 gbMat33GetInverse22(const GBMat33& M){
    float a = M.ex.x, b = M.ey.x, c = M.ex.y, d = M.ey.y;
    float det = a*d - b*c;
    if (det != 0.0f) det = 1.0f / det;
    GBMat33 R;
    R.ex.x =  det*d; R.ey.x = -det*b; R.ex.z = 0.0f;
    R.ex.y = -det*c; R.ey.y =  det*a; R.ey.z = 0.0f;
    R.ez.x = 0.0f;   R.ez.y = 0.0f;   R.ez.z = 0.0f;
    return R;
}
// b2Mul(b2Mat33, b2Vec3): full 3x3 matrix times a 3-vector.
GB_HD inline V3 gbMulM33V3(const GBMat33& A, V3 v){
    return v3(v.x*A.ex.x + v.y*A.ey.x + v.z*A.ez.x,
              v.x*A.ex.y + v.y*A.ey.y + v.z*A.ez.y,
              v.x*A.ex.z + v.y*A.ey.z + v.z*A.ez.z);
}
// b2Mul22(b2Mat33, b2Vec2): the upper-left 2x2 block times a 2-vector.
GB_HD inline V2 gbMulM33V2(const GBMat33& A, V2 v){
    return v2(A.ex.x*v.x + A.ey.x*v.y, A.ex.y*v.x + A.ey.y*v.y);
}
// b2Mat33::GetSymInverse33. Inverse of a symmetric 3x3 matrix.
GB_HD inline GBMat33 gbMat33GetSymInverse33(const GBMat33& M){
    float det = b2Dot3(M.ex, b2Cross3(M.ey, M.ez));
    if (det != 0.0f) det = 1.0f / det;
    float a11 = M.ex.x, a12 = M.ey.x, a13 = M.ez.x;
    float a22 = M.ey.y, a23 = M.ez.y, a33 = M.ez.z;
    GBMat33 R;
    R.ex.x = det*(a22*a33 - a23*a23);
    R.ex.y = det*(a13*a23 - a12*a33);
    R.ex.z = det*(a12*a23 - a13*a22);
    R.ey.x = R.ex.y;
    R.ey.y = det*(a11*a33 - a13*a13);
    R.ey.z = det*(a13*a12 - a11*a23);
    R.ez.x = R.ex.z;
    R.ez.y = R.ey.z;
    R.ez.z = det*(a11*a22 - a12*a12);
    return R;
}
struct GBConstraint {
    int   indexA, indexB;            // island-local body indices
    float invMassA, invMassB, invIA, invIB;
    float friction, restitution;
    V2    normal;
    GBVelConstraintPt p;             // contact point 0 (velocity)
    GBVelConstraintPt p2;            // contact point 1 (velocity); used when pointCount==2
    int   pointCount;                // 1 or 2 (set to 1 if the block solve is ill-conditioned)
    GBMat22 K;                       // block velocity matrix (two-point solve)
    GBMat22 normalMass22;            // inverse of K (two-point solve)
    int   contactIdx;                // global contact slot (StoreImpulses)
    V2    localNormal, localPoint, pLocalPoint;   // position-solve, point 0
    V2    pLocalPoint2;                            // position-solve, point 1
    int   type; float radiusA, radiusB;
};

// ---- revolute joint, point-to-point (the solver phases live in gb_joint.cuh) ----
// The struct lives here so the island scratch can carry an array of joints. Anchors
// are body-local; localCenter is 0 in the gb_* body model, so the world anchor arm is
// rA = Rot(aA) * localAnchorA.
struct GBRevoluteJoint {
    int indexA, indexB;            // island-local body indices
    V2  localAnchorA, localAnchorB;
    float invMassA, invMassB, invIA, invIB;
    int jointIdx;                  // global joint slot (impulse store-back)
    V2  rA, rB;                    // solver arms (set by InitVelocityConstraints)
    GBMat22 mass;                  // 2x2 point-to-point effective mass
    V2  impulse;                   // accumulated point-to-point impulse (warm-start)
};

// ---- per-island solve scratch (lives in shared memory in the block model) ------
struct GBIslandData {
    int        bodies[GB_MAX_BODIES];     // global body slots, island order
    int        contacts[GB_MAX_CONTACTS]; // global contact slots, island order
    int        bodyCount, contactCount;
    V2         posC[GB_MAX_BODIES];  float posA[GB_MAX_BODIES];
    V2         vel[GB_MAX_BODIES];   float velW[GB_MAX_BODIES];
    GBConstraint con[GB_MAX_CONTACTS];    // fused vel+pos constraints
#ifdef GB_ENABLE_JOINTS
    int          joints[GB_MAX_JOINTS];   // global joint slots, island order
    int          jointCount;
    GBRevoluteJoint jnt[GB_MAX_JOINTS];   // island-local joint scratch
#endif
};

// ============================================================================
// CROSS-MODULE PHASE-FUNCTION CONTRACT. Each phase is declared by its role here and
// defined by its owning header. The assembled step (gb_step.cuh) calls them. The
// signatures are stable.
//
//   gb_collision.cuh (narrow-phase):
//     void gbCollideCircles(GBManifold&, float rA, Xf xfA, float rB, Xf xfB);
//     void gbCollideEdgeAndCircle(GBManifold&, V2 A, V2 B, float edgeR, float circR, Xf xfA, Xf xfB);
//     void gbCollidePolygons(GBManifold&, const GBPolygon& A, Xf xfA, const GBPolygon& B, Xf xfB);
//     void gbCollidePolygonAndCircle(GBManifold&, const GBPolygon& A, Xf xfA, float circR, Xf xfB);
//     void gbWorldManifoldInit(GBWorldManifold&, const GBManifold&, Xf xfA, float rA, Xf xfB, float rB);
//     void gbContactUpdate(GBWorld& w, int ci);     // narrow-phase + touching + listener hook
//
//   gb_broadphase.cuh: the GbBroadPhase/GbDynTree + gbBpUpdatePairs. The faithful
//     path uses the tree; the assembled step also offers a brute-force fat-AABB
//     gbCollidePhase. Either way it fills the world's contact pool and fires the
//     listener hook in Box2D m_contactList order.
//
//   gb_contact_solver.cuh (single-point and two-point block paths):
//     void gbInitVelocityConstraints(GBWorld&, GBIslandData&);   // build GBConstraint per contact
//     void gbWarmStart(GBIslandData&);
//     void gbSolveVelocity(GBIslandData&);   // ONE velocity iteration (serial sweep, lane 0)
//     void gbStoreImpulses(GBWorld&, GBIslandData&);
//     bool gbSolvePosition(GBIslandData&);   // ONE position iteration; returns contactsOkay
//                                            //   [SERIAL: minSeparation fold on lane 0]
//
//   gb_joint.cuh (revolute, point-to-point):
//     void gbRevoluteInitVelocity(GBRevoluteJoint&, GBIslandData&);
//     void gbRevoluteSolveVelocity(GBRevoluteJoint&, GBIslandData&);   // ONE velocity iteration
//     bool gbRevoluteSolvePosition(GBRevoluteJoint&, GBIslandData&);   // ONE position iteration
//
//   gb_island.cuh:
//     void gbWorldSolve(GBWorld& w);         // DFS island assembly + island Solve loop:
//        integrate-velocities -> warmstart -> 8x gbSolveVelocity -> store -> integrate-positions
//        -> up to 3x gbSolvePosition -> copy-back -> sleep
//        [SERIAL FOLDS on lane 0: in-place velocity accum, minSeparation, minSleepTime]
//
//   gb_toi.cuh: gbWorldSolveTOI(GBWorld&), which drives the above through the CCD path.
//
// THE 3 SERIAL FLOAT-FOLD RULE (non-associative under --fmad=false): the in-place
// velocity/position accumulation in gbSolveVelocity/gbSolvePosition, the
// minSeparation fold in gbSolvePosition, and the minSleepTime fold in the island
// sleep step must run serial and in order on lane 0. A parallel reduction would
// change the association order and break the 0-ULP match.
// ============================================================================
