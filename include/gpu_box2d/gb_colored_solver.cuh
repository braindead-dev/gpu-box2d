// gb_colored_solver.cuh. A graph-colored parallel contact solver.
// =============================================================================
// MEASURED-AND-REJECTED ALTERNATIVE. This is one of two alternative execution models
// the project built and measured against the production thread-per-world engine. It is
// kept for the record and documented in docs/performance.md; the production path does
// not use it.
//
// It is not bit-identical to Box2D. Graph coloring reorders the Gauss-Seidel sweep into
// Jacobi-within-color and Gauss-Seidel-across-colors. That breaks the serial-solver
// wall (the solver is about 74% of a step): within one color, contacts touch disjoint
// bodies, so all contacts of a color solve in parallel (one thread per contact) with
// conflict-free velocity writes, and colors run sequentially with __syncthreads between
// them. Because the sweep is reordered, validity for training is judged by distribution
// fidelity (a KS test) while the bit-identical engine
// (gb_island.cuh/gb_contact_solver.cuh) remains the ULP reference, untouched.
//
// Why it lost: it needs a block-per-world host (the per-world arena and island scratch
// live in shared memory, about 32 KB of the 48 KB block budget), and that host collapses
// occupancy. Measured at about 5.1K env-steps/s, below the production thread-per-world
// path at about 23K. Breaking the serial wall did not pay for the host it required.
//
// EXECUTION: one CUDA block per world. Block threads cooperate:
//   - lane 0: DFS island assembly + greedy contact coloring + collide bookkeeping
//   - all threads: parallel-by-color velocity and position iterations, parallel per-body
//     integration and copy-back.
// Build flags: nvcc --fmad=false -prec-div=true -prec-sqrt=true.
// =============================================================================
#pragma once
#include "gpu_box2d/gb_pools.cuh"
#include "gpu_box2d/gb_settings.cuh"
#include "gpu_box2d/gb_math.cuh"
#include "gpu_box2d/gb_contact_types.cuh"
// This alternative solver reuses two helpers from the bit-identical core:
// gbWorldManifoldInit (from gb_collision.cuh, the canonical owner; the define below
// suppresses the fallback copy in gb_contact_solver.cuh) and gbSyncTransform (from
// gb_contact_solver.cuh). Including them here makes this header self-contained.
#define GB_COLLISION_PROVIDED 1
#include "gpu_box2d/gb_collision.cuh"
#include "gpu_box2d/gb_contact_solver.cuh"

#ifndef GB_MAX_COLORS
#define GB_MAX_COLORS 16   // greedy coloring of <=128 contacts on <=65 bodies: low degree
#endif

// Per-island coloring metadata (lives in shared with the island scratch).
struct GBColoring {
    int color[GB_MAX_CONTACTS];        // color of each island contact
    int order[GB_MAX_CONTACTS];        // contacts sorted by color (stable within color)
    int colorStart[GB_MAX_COLORS+1];   // prefix offsets into order[] per color
    int numColors;
};

// -----------------------------------------------------------------------------
// Greedy graph coloring (lane 0, serial - cheap: once per island-solve, O(cc^2/bound)).
// Two contacts conflict iff they share a DYNAMIC body. Static ground (indexA may map to
// a static body) does not create a conflict (its velocity is never written), so we only
// conflict on dynamic indices. We detect "dynamic" via invMass>0 stored on the constraint.
// -----------------------------------------------------------------------------
GB_HD inline void gbColorIsland(const GBIslandData& isl, GBColoring& col){
    int cc = isl.contactCount;
    // bitmask of colors used by conflicting already-colored contacts (<=GB_MAX_COLORS bits)
    int maxc = 0;
    for (int i = 0; i < cc; ++i){
        const GBConstraint& ci = isl.con[i];
        // dynamic island-body indices this contact writes
        int a = (ci.invMassA > 0.0f) ? ci.indexA : -1;
        int b = (ci.invMassB > 0.0f) ? ci.indexB : -1;
        unsigned used = 0u;
        for (int j = 0; j < i; ++j){
            const GBConstraint& cj = isl.con[j];
            int ja = (cj.invMassA > 0.0f) ? cj.indexA : -1;
            int jb = (cj.invMassB > 0.0f) ? cj.indexB : -1;
            bool conflict = (a>=0 && (a==ja || a==jb)) || (b>=0 && (b==ja || b==jb));
            if (conflict) used |= (1u << col.color[j]);
        }
        int c = 0;
        while (c < GB_MAX_COLORS && (used & (1u<<c))) ++c;
        if (c >= GB_MAX_COLORS) c = GB_MAX_COLORS - 1; // clamp (degrades to some serial within color; rare)
        col.color[i] = c;
        if (c+1 > maxc) maxc = c+1;
    }
    col.numColors = maxc;
    // counting sort contacts into order[] by color (stable)
    for (int c = 0; c <= maxc; ++c) col.colorStart[c] = 0;
    for (int i = 0; i < cc; ++i) col.colorStart[col.color[i]+1]++;
    for (int c = 0; c < maxc; ++c) col.colorStart[c+1] += col.colorStart[c];
    int cursor[GB_MAX_COLORS];
    for (int c = 0; c < maxc; ++c) cursor[c] = col.colorStart[c];
    for (int i = 0; i < cc; ++i){
        int c = col.color[i];
        col.order[cursor[c]++] = i;
    }
}

// -----------------------------------------------------------------------------
// Parallel-by-color VELOCITY iteration. Each thread handles one contact within the
// current color; same-color contacts touch disjoint dynamic bodies so the in-place
// vel writes are conflict-free. Math is verbatim from gbSolveVelocity (one contact).
// Caller loops colors with __syncthreads between them, and loops this GB_VELOCITY_ITERS.
// -----------------------------------------------------------------------------
GB_HD inline void gbSolveVelocityContact(GBIslandData& isl, int i){
    GBConstraint& vc = isl.con[i];
    int ia=vc.indexA, ib=vc.indexB;
    float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
    V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
    V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
    V2 normal=vc.normal; V2 tangent=b2CrossVS(normal,1.0f);
    float friction=vc.friction;
    GBVelConstraintPt& vcp=vc.p;
    { // tangent (friction)
        V2 dv = vB + b2CrossSV(wB, vcp.rB) - vA - b2CrossSV(wA, vcp.rA);
        float vt = b2Dot(dv, tangent);
        float lambda = vcp.tangentMass * (-vt);
        float maxFriction = friction * vcp.normalImpulse;
        float newImp = b2ClampF(vcp.tangentImpulse + lambda, -maxFriction, maxFriction);
        lambda = newImp - vcp.tangentImpulse;
        vcp.tangentImpulse = newImp;
        V2 P = lambda*tangent;
        vA = vA - mA*P; wA -= iA*b2Cross(vcp.rA, P);
        vB = vB + mB*P; wB += iB*b2Cross(vcp.rB, P);
    }
    { // normal
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

// Parallel-by-color POSITION iteration (one contact). Math verbatim from gbSolvePosition.
// We DROP the minSeparation early-exit (always run GB_POSITION_ITERS passes) - the
// position correction is order-sensitive anyway; running a fixed 3 is distribution-fine.
GB_HD inline void gbSolvePositionContact(GBIslandData& isl, int i){
    GBConstraint& pc = isl.con[i];
    int ia=pc.indexA, ib=pc.indexB;
    float mA=pc.invMassA, iA=pc.invIA, mB=pc.invMassB, iB=pc.invIB;
    V2 cA=isl.posC[ia]; float aA=isl.posA[ia];
    V2 cB=isl.posC[ib]; float aB=isl.posA[ib];
    Xf xfA, xfB;
    xfA.q=rotSet(aA); xfB.q=rotSet(aB);
    xfA.p = cA - b2MulRV(xfA.q, v2(0,0));
    xfB.p = cB - b2MulRV(xfB.q, v2(0,0));
    V2 normal, point; float separation;
    if (pc.type == GB_MANIFOLD_CIRCLES){
        V2 pointA = b2MulTV(xfA, pc.localPoint);
        V2 pointB = b2MulTV(xfB, pc.pLocalPoint);
        normal = pointB - pointA; b2Normalize(normal);
        point = 0.5f*(pointA + pointB);
        separation = b2Dot(pointB - pointA, normal) - pc.radiusA - pc.radiusB;
    } else {
        normal = b2MulRV(xfA.q, pc.localNormal);
        V2 planePoint = b2MulTV(xfA, pc.localPoint);
        V2 clipPoint  = b2MulTV(xfB, pc.pLocalPoint);
        separation = b2Dot(clipPoint - planePoint, normal) - pc.radiusA - pc.radiusB;
        point = clipPoint;
    }
    V2 rA = point - cA; V2 rB = point - cB;
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

// One contact's init (position-dependent constraint portions). Verbatim from
// gbInitVelocityConstraints body, parallel-safe (each contact writes only its own con[i]).
GB_HD inline void gbInitVelContact(GBWorld& w, GBIslandData& isl, int i){
    GBConstraint& vc = isl.con[i];
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
    GBManifold man; man.pointCount=1; man.type=vc.type;
    man.localNormal=vc.localNormal; man.localPoint=vc.localPoint; man.pLocalPoint=vc.pLocalPoint;
    GBWorldManifold wm; gbWorldManifoldInit(wm, man, xfA, vc.radiusA, xfB, vc.radiusB);
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

// One contact's warm-start. It writes vel[ia] and vel[ib], so it is conflict-free only
// within a color and must be applied per-color (the same disjoint-body guarantee).
GB_HD inline void gbWarmStartContact(GBIslandData& isl, int i){
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

// =============================================================================
// BLOCK-PARALLEL colored island solve. Called by all threads of the block.
// `tid` = threadIdx.x, `nt` = blockDim.x. The island data + coloring live in SHARED
// memory (passed by reference). Lane 0 must have already done DFS assembly + the
// per-contact constraint LOAD (the cheap serial part) + coloring before this is called.
//   Phases (all threads cooperate, __syncthreads between):
//   1. integrate velocities (per-body)
//   2. init velocity constraints (per-contact, independent writes)
//   3. warm start (per-color)
//   4. GB_VELOCITY_ITERS x [ per-color velocity solve ]
//   5. store impulses (per-contact) + integrate positions (per-body)
//   6. GB_POSITION_ITERS x [ per-color position solve ]
//   7. copy back + sync transform (per-body); sleep decided by lane 0
// =============================================================================
GB_HD inline void gbIslandSolveColored(GBWorld& w, GBIslandData& isl, GBColoring& col,
                                       bool allowSleep, int tid, int nt){
#ifdef __CUDA_ARCH__
    float h = GB_DT;
    int bc = isl.bodyCount, cc = isl.contactCount;

    // 1) integrate velocities + init body state buffers (per-body parallel)
    for (int i = tid; i < bc; i += nt){
        int bi = isl.bodies[i];
        V2 c = v2(BODY(w, sweepCx, bi), BODY(w, sweepCy, bi));
        float a = BODY(w, sweepA, bi);
        V2 vv = v2(BODY(w, velX, bi), BODY(w, velY, bi));
        float ww = BODY(w, angVel, bi);
        BODY(w, sweepC0x, bi) = BODY(w, sweepCx, bi); BODY(w, sweepC0y, bi) = BODY(w, sweepCy, bi);
        BODY(w, sweepA0, bi)  = BODY(w, sweepA, bi);
        if (BODY(w, bodyType, bi) == GB_DYNAMIC_BODY)
            vv = vv + h*v2(0.0f, GB_GRAVITY_Y);
        isl.posC[i] = c; isl.posA[i] = a; isl.vel[i] = vv; isl.velW[i] = ww;
    }
    __syncthreads();

    // 2) init velocity constraints (per-contact parallel - writes only con[i])
    for (int i = tid; i < cc; i += nt) gbInitVelContact(w, isl, i);
    __syncthreads();

    // 3) warm start (per-color: within a color, disjoint bodies)
    for (int c = 0; c < col.numColors; ++c){
        int lo = col.colorStart[c], hi = col.colorStart[c+1];
        for (int k = lo + tid; k < hi; k += nt) gbWarmStartContact(isl, col.order[k]);
        __syncthreads();
    }

    // 4) velocity iterations (per-color, fixed GB_VELOCITY_ITERS)
    for (int it = 0; it < GB_VELOCITY_ITERS; ++it){
        for (int c = 0; c < col.numColors; ++c){
            int lo = col.colorStart[c], hi = col.colorStart[c+1];
            for (int k = lo + tid; k < hi; k += nt) gbSolveVelocityContact(isl, col.order[k]);
            __syncthreads();
        }
    }

    // 5) store impulses (per-contact) + integrate positions (per-body)
    for (int i = tid; i < cc; i += nt){
        int ci = isl.con[i].contactIdx;
        CONT(w, cNormalImpulse,  ci) = isl.con[i].p.normalImpulse;
        CONT(w, cTangentImpulse, ci) = isl.con[i].p.tangentImpulse;
    }
    for (int i = tid; i < bc; i += nt){
        V2 c=isl.posC[i]; float a=isl.posA[i];
        V2 vv=isl.vel[i]; float ww=isl.velW[i];
        V2 tr = h*vv;
        if (b2Dot(tr,tr) > GB_MAX_TRANSLATION_SQ){ float ratio=GB_MAX_TRANSLATION/b2Length(tr); vv=ratio*vv; }
        float rot = h*ww;
        if (rot*rot > GB_MAX_ROTATION_SQ){ float ratio=GB_MAX_ROTATION/b2AbsF(rot); ww*=ratio; }
        c = c + h*vv; a += h*ww;
        isl.posC[i]=c; isl.posA[i]=a; isl.vel[i]=vv; isl.velW[i]=ww;
    }
    __syncthreads();

    // 6) position iterations (per-color, fixed GB_POSITION_ITERS - no early-exit)
    for (int it = 0; it < GB_POSITION_ITERS; ++it){
        for (int c = 0; c < col.numColors; ++c){
            int lo = col.colorStart[c], hi = col.colorStart[c+1];
            for (int k = lo + tid; k < hi; k += nt) gbSolvePositionContact(isl, col.order[k]);
            __syncthreads();
        }
    }

    // 7) copy back + sync transform (per-body)
    for (int i = tid; i < bc; i += nt){
        int bi = isl.bodies[i];
        BODY(w, sweepCx, bi)=isl.posC[i].x; BODY(w, sweepCy, bi)=isl.posC[i].y;
        BODY(w, sweepA, bi) =isl.posA[i];
        BODY(w, velX, bi)=isl.vel[i].x; BODY(w, velY, bi)=isl.vel[i].y; BODY(w, angVel, bi)=isl.velW[i];
        gbSyncTransform(w, bi);
    }
    __syncthreads();

    // sleep: lane 0 (cheap reduction over <=65 bodies)
    if (allowSleep && tid == 0){
        float minSleepTime = GB_MAXFLOAT;
        const float linTolSqr = GB_LINEAR_SLEEP_TOL*GB_LINEAR_SLEEP_TOL;
        const float angTolSqr = GB_ANGULAR_SLEEP_TOL*GB_ANGULAR_SLEEP_TOL;
        for (int i = 0; i < bc; ++i){
            int bi = isl.bodies[i];
            if (BODY(w, bodyType, bi)==GB_STATIC_BODY) continue;
            float av = BODY(w, angVel, bi);
            V2 lv = v2(BODY(w, velX, bi), BODY(w, velY, bi));
            if (av*av > angTolSqr || b2Dot(lv,lv) > linTolSqr){
                BODY(w, sleepTime, bi)=0.0f; minSleepTime=0.0f;
            } else {
                BODY(w, sleepTime, bi) += h;
                minSleepTime = b2MinF(minSleepTime, BODY(w, sleepTime, bi));
            }
        }
        if (minSleepTime >= GB_TIME_TO_SLEEP){
            for (int i = 0; i < bc; ++i){
                int bi = isl.bodies[i];
                BODY(w, awake, bi)=0; BODY(w, sleepTime, bi)=0.0f;
                BODY(w, velX, bi)=0.0f; BODY(w, velY, bi)=0.0f; BODY(w, angVel, bi)=0.0f;
            }
        }
    }
    __syncthreads();
#endif
}

// =============================================================================
// gbWorldSolveColored - block-parallel world solve. Lane 0 does the (cheap, serial)
// DFS island assembly + constraint LOAD + greedy coloring per island; all threads then
// run the colored island solve. Islands handled sequentially (one block, one world).
// `isl`/`col` are SHARED-memory scratch passed in. tid/nt = threadIdx.x/blockDim.x.
// =============================================================================
GB_HD inline void gbWorldSolveColored(GBWorld& w, GBIslandData& isl, GBColoring& col,
                                      int tid, int nt,
                                      unsigned char* bodyInIsland,    // shared [GB_MAX_BODIES]
                                      unsigned char* contactInIsland, // shared [GB_MAX_CONTACTS]
                                      int* stack){                    // shared [GB_MAX_BODIES]
#ifdef __CUDA_ARCH__
    int bc0 = SCAL(w, bodyCount), cc0 = SCAL(w, contactCount);
    for (int i = tid; i < bc0; i += nt) bodyInIsland[i] = 0;
    for (int i = tid; i < cc0; i += nt) contactInIsland[i] = 0;
    __syncthreads();

    // Iterate seeds in reverse creation order (matches the bit-identical island order;
    // coloring still reorders WITHIN an island, but island membership is the same).
    for (int seed = bc0 - 1; seed >= 0; --seed){
        __syncthreads();
        if (bodyInIsland[seed] || !BODY(w, alive, seed) || !BODY(w, awake, seed)
            || BODY(w, bodyType, seed)==GB_STATIC_BODY) continue;

        // lane 0: DFS-assemble this island + load constraints + color
        if (tid == 0){
            isl.bodyCount=0; isl.contactCount=0;
            int sc=0; stack[sc++]=seed; bodyInIsland[seed]=1;
            while (sc > 0){
                int b = stack[--sc];
                isl.bodies[isl.bodyCount++] = b;
                BODY(w, awake, b)=1;
                if (BODY(w, bodyType, b)==GB_STATIC_BODY) continue;
                for (int ci = cc0-1; ci >= 0; --ci){
                    if (contactInIsland[ci]) continue;
                    int ca=CONT(w, cBodyA, ci), cb=CONT(w, cBodyB, ci);
                    if (ca != b && cb != b) continue;
                    if (!CONT(w, cTouching, ci)) continue;
                    isl.contacts[isl.contactCount++] = ci;
                    contactInIsland[ci]=1;
                    int other = (ca==b)? cb : ca;
                    if (bodyInIsland[other]) continue;
                    stack[sc++]=other; bodyInIsland[other]=1;
                }
            }
            // load per-contact constraints (island-local indices, radii, warm-start)
            int bc = isl.bodyCount, cc = isl.contactCount;
            for (int i = 0; i < cc; ++i){
                int ci = isl.contacts[i];
                GBConstraint& vc = isl.con[i];
                int bodyA = CONT(w, cBodyA, ci), bodyB = CONT(w, cBodyB, ci), edge = CONT(w, cEdge, ci);
                int ia=-1, ib=-1;
                for (int k=0;k<bc;++k){ if(isl.bodies[k]==bodyA) ia=k; if(isl.bodies[k]==bodyB) ib=k; }
                float radiusA, radiusB;
                if (edge < 0){ radiusA = gbCircleRadius(w,bodyA); radiusB = gbCircleRadius(w,bodyB); }
                else         { radiusA = GB_POLYGON_RADIUS;       radiusB = gbCircleRadius(w,bodyB); }
                vc.friction = CONT(w, cFriction, ci); vc.restitution = CONT(w, cRestitution, ci);
                vc.indexA = ia; vc.indexB = ib;
                vc.invMassA = BODY(w, invMass, bodyA); vc.invMassB = BODY(w, invMass, bodyB);
                vc.invIA = BODY(w, invI, bodyA);       vc.invIB = BODY(w, invI, bodyB);
                vc.contactIdx = ci;
                vc.p.normalImpulse  = CONT(w, cNormalImpulse, ci);
                vc.p.tangentImpulse = CONT(w, cTangentImpulse, ci);
                vc.p.rA = v2(0,0); vc.p.rB = v2(0,0);
                vc.p.normalMass=0; vc.p.tangentMass=0; vc.p.velocityBias=0;
                vc.localNormal=v2(CONT(w, cLocalNormalX, ci), CONT(w, cLocalNormalY, ci));
                vc.localPoint =v2(CONT(w, cLocalPointX, ci),  CONT(w, cLocalPointY, ci));
                vc.pLocalPoint=v2(CONT(w, cPointLocalX, ci),  CONT(w, cPointLocalY, ci));
                vc.type=CONT(w, cManifoldType, ci); vc.radiusA=radiusA; vc.radiusB=radiusB;
            }
            gbColorIsland(isl, col);
        }
        __syncthreads();

        // all threads: colored parallel island solve
        gbIslandSolveColored(w, isl, col, /*allowSleep*/true, tid, nt);

        // post-solve: free static bodies for other islands (lane 0)
        if (tid == 0)
            for (int i=0;i<isl.bodyCount;++i){
                int bi = isl.bodies[i];
                if (BODY(w, bodyType, bi)==GB_STATIC_BODY) bodyInIsland[bi]=0;
            }
        __syncthreads();
    }
#endif
}
