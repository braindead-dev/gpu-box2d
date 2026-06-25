// gb_island.cuh. b2Island::Solve (integrate / warm-start / iterate / sleep) and
// b2World::Solve (DFS island assembly), on the gb_pools accessor contract
// (BODY/CONT/EDGE/SCAL) and the cross-module types in gb_contact_types.cuh. Drives the
// per-iteration solver phases in gb_contact_solver.cuh.
//
// THE 3-SERIAL-FLOAT-FOLD RULE (see gb_contact_types.cuh): the DFS island assembly, the
// constraint load, and the solver phases run in order on lane 0. The three running
// float folds are non-associative under --fmad=false: (1) in-place velocity/position
// accumulation in the sweeps, (2) the minSeparation min-fold in gbSolvePosition, and
// (3) the minSleepTime min-fold in the sleep step (here). A tree-reduce would change
// the floats. Body and contact iteration order match Box2D b2Island::Solve:
//   * the DFS body-seed loop iterates m_bodyList in reverse creation order
//     (descending slot).
//   * each body's incident contacts iterate its edge list in reverse creation order
//     (descending contact index).
// Both iterate descending to match the CPU island body order and contact solve order.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_pools.cuh"
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_types.cuh"     // shared types + phase-function contract
#include "gb_contact_solver.cuh"    // the per-iteration Gauss-Seidel phases
#ifdef GB_ENABLE_JOINTS
#include "gb_joint.cuh"             // revolute joint phases (opt-in)
#endif

// ============================================================================
// b2Island::Solve, faithful (single-point contacts only). Lane 0, serial in order.
// The serial spine is integrate-velocities -> load constraints ->
// gbInitVelocityConstraints -> gbWarmStart -> GB_VELOCITY_ITERS x gbSolveVelocity ->
// gbStoreImpulses -> integrate-positions -> up to GB_POSITION_ITERS x gbSolvePosition
// (early-exit) -> copy-back and sync -> sleep (minSleepTime fold).
// ============================================================================
GB_HD inline void gbIslandSolve(GBWorld& w, GBIslandData& isl, bool allowSleep){
    float h = GB_DT;
    int bc = isl.bodyCount, cc = isl.contactCount;

    // ---- Integrate velocities + damping; init body state buffers. -----------
    for (int i = 0; i < bc; ++i){
        int bi = isl.bodies[i];
        V2 c = v2(BODY(w, sweepCx, bi), BODY(w, sweepCy, bi));
        float a = BODY(w, sweepA, bi);
        V2 vv = v2(BODY(w, velX, bi), BODY(w, velY, bi));
        float ww = BODY(w, angVel, bi);
        // store positions for CCD
        BODY(w, sweepC0x, bi) = BODY(w, sweepCx, bi); BODY(w, sweepC0y, bi) = BODY(w, sweepCy, bi);
        BODY(w, sweepA0, bi)  = BODY(w, sweepA, bi);
        if (BODY(w, bodyType, bi) == GB_DYNAMIC_BODY){
            vv = vv + h*( v2(0.0f, GB_GRAVITY_Y) /*gravityScale=1, force=0*/ );
            // w += h * invI * torque(0) -> unchanged; damping 0 -> no change
        }
        isl.posC[i] = c; isl.posA[i] = a;
        isl.vel[i] = vv; isl.velW[i] = ww;
    }

    // ---- Load velocity constraints (b2ContactSolver ctor: copy + warm-start) -
    // Fills the FROZEN GBConstraint array from the world contact pool, island order.
    for (int i = 0; i < cc; ++i){
        int ci = isl.contacts[i];
        GBConstraint& vc = isl.con[i];
        GBConstraint& pc = isl.con[i];   // fused: same object
        int bodyA = CONT(w, cBodyA, ci), bodyB = CONT(w, cBodyB, ci), edge = CONT(w, cEdge, ci);
        // island-local indices
        int ia=-1, ib=-1;
        for (int k=0;k<bc;++k){ if(isl.bodies[k]==bodyA) ia=k; if(isl.bodies[k]==bodyB) ib=k; }
        float radiusA, radiusB;
#ifdef GB_ENABLE_POLYGONS
        // shape-aware skin radius: a polygon fixture uses GB_POLYGON_RADIUS, a circle
        // uses its m_radius, and the ground edge uses GB_POLYGON_RADIUS.
        radiusB = gbBodyShape(w,bodyB)==GB_SHAPE_POLYGON ? GB_POLYGON_RADIUS : gbCircleRadius(w,bodyB);
        if (edge < 0) radiusA = gbBodyShape(w,bodyA)==GB_SHAPE_POLYGON ? GB_POLYGON_RADIUS : gbCircleRadius(w,bodyA);
        else          radiusA = GB_POLYGON_RADIUS;
#else
        if (edge < 0){ radiusA = gbCircleRadius(w,bodyA); radiusB = gbCircleRadius(w,bodyB); }
        else         { radiusA = GB_POLYGON_RADIUS;       radiusB = gbCircleRadius(w,bodyB); }
#endif

        vc.friction = CONT(w, cFriction, ci); vc.restitution = CONT(w, cRestitution, ci);
        vc.indexA = ia; vc.indexB = ib;
        vc.invMassA = BODY(w, invMass, bodyA); vc.invMassB = BODY(w, invMass, bodyB);
        vc.invIA = BODY(w, invI, bodyA);       vc.invIB = BODY(w, invI, bodyB);
        vc.contactIdx = ci;
        // point count: a cached manifold of 2 points (polygon contact) drives the
        // two-point block path; everything else is a single point. A contact whose
        // cache predates polygon support reads back 0, which maps to 1.
        int pcount = CONT(w, cPointCount, ci);
        vc.pointCount = pcount == 2 ? 2 : 1;
        // warm-start impulse (dtRatio == 1 in steady DT)
        vc.p.normalImpulse  = CONT(w, cNormalImpulse, ci);   // warmStarting=true, dtRatio=1
        vc.p.tangentImpulse = CONT(w, cTangentImpulse, ci);
        vc.p.rA = v2(0,0); vc.p.rB = v2(0,0);
        vc.p.normalMass=0; vc.p.tangentMass=0; vc.p.velocityBias=0;
        if (vc.pointCount == 2){
            vc.p2.normalImpulse  = CONT(w, cNormalImpulse2, ci);
            vc.p2.tangentImpulse = CONT(w, cTangentImpulse2, ci);
            vc.p2.rA = v2(0,0); vc.p2.rB = v2(0,0);
            vc.p2.normalMass=0; vc.p2.tangentMass=0; vc.p2.velocityBias=0;
        }

        pc.indexA=ia; pc.indexB=ib;
        pc.invMassA=vc.invMassA; pc.invMassB=vc.invMassB;
        pc.invIA=vc.invIA; pc.invIB=vc.invIB;
        pc.localNormal=v2(CONT(w, cLocalNormalX, ci), CONT(w, cLocalNormalY, ci));
        pc.localPoint =v2(CONT(w, cLocalPointX, ci),  CONT(w, cLocalPointY, ci));
        pc.pLocalPoint=v2(CONT(w, cPointLocalX, ci),  CONT(w, cPointLocalY, ci));
        pc.pLocalPoint2=v2(CONT(w, cPointLocal2X, ci), CONT(w, cPointLocal2Y, ci));
        pc.type=CONT(w, cManifoldType, ci); pc.radiusA=radiusA; pc.radiusB=radiusB;
    }

#ifdef GB_ENABLE_JOINTS
    // ---- Load joint constraints (island-local scratch from the joint pool) ---
    int jc = isl.jointCount;
    for (int i = 0; i < jc; ++i){
        int ji = isl.joints[i];
        GBRevoluteJoint& jn = isl.jnt[i];
        int bodyA = JOINT(w, jBodyA, ji), bodyB = JOINT(w, jBodyB, ji);
        int ia=-1, ib=-1;
        for (int k=0;k<bc;++k){ if(isl.bodies[k]==bodyA) ia=k; if(isl.bodies[k]==bodyB) ib=k; }
        jn.indexA=ia; jn.indexB=ib; jn.jointIdx=ji;
        jn.localAnchorA=v2(JOINT(w, jLocalAnchorAX, ji), JOINT(w, jLocalAnchorAY, ji));
        jn.localAnchorB=v2(JOINT(w, jLocalAnchorBX, ji), JOINT(w, jLocalAnchorBY, ji));
        jn.invMassA=BODY(w, invMass, bodyA); jn.invMassB=BODY(w, invMass, bodyB);
        jn.invIA=BODY(w, invI, bodyA);       jn.invIB=BODY(w, invI, bodyB);
        jn.impulse=v2(JOINT(w, jImpulseX, ji), JOINT(w, jImpulseY, ji));   // dtRatio==1
    }
#endif

    // ---- serial solver spine (lane 0) --------------------------------------
    // b2Island::Solve order: init contacts, warm-start contacts, init joints; then
    // each velocity iteration solves joints first (joint-list order) then contacts.
    gbInitVelocityConstraints(w, isl);                       // position-dependent portions
    gbWarmStart(isl);                                        // apply carried impulse
#ifdef GB_ENABLE_JOINTS
    for (int i = 0; i < jc; ++i) gbRevoluteInitVelocity(isl.jnt[i], isl);
#endif
    for (int it = 0; it < GB_VELOCITY_ITERS; ++it){          // 8 Gauss-Seidel velocity iters
#ifdef GB_ENABLE_JOINTS
        for (int i = 0; i < jc; ++i) gbRevoluteSolveVelocity(isl.jnt[i], isl);
#endif
        gbSolveVelocity(isl);
    }
    gbStoreImpulses(w, isl);                                 // carry warm-start to next substep
#ifdef GB_ENABLE_JOINTS
    for (int i = 0; i < jc; ++i){                            // store joint warm-start impulse
        int ji = isl.jnt[i].jointIdx;
        JOINT(w, jImpulseX, ji) = isl.jnt[i].impulse.x;
        JOINT(w, jImpulseY, ji) = isl.jnt[i].impulse.y;
    }
#endif

    // ---- Integrate positions ------------------------------------------------
    for (int i = 0; i < bc; ++i){
        V2 c=isl.posC[i]; float a=isl.posA[i];
        V2 vv=isl.vel[i]; float ww=isl.velW[i];
        V2 translation = h*vv;
        if (b2Dot(translation, translation) > GB_MAX_TRANSLATION_SQ){
            float ratio = GB_MAX_TRANSLATION / b2Length(translation);
            vv = ratio*vv;
        }
        float rotation = h*ww;
        if (rotation*rotation > GB_MAX_ROTATION_SQ){
            float ratio = GB_MAX_ROTATION / b2AbsF(rotation);
            ww *= ratio;
        }
        c = c + h*vv; a += h*ww;
        isl.posC[i]=c; isl.posA[i]=a; isl.vel[i]=vv; isl.velW[i]=ww;
    }

    // ---- position solve: up to GB_POSITION_ITERS iters, early-exit ----------
    // (b2Island::Solve: positionSolved when a pass reports contactsOkay; the
    //  minSeparation fold lives inside each gbSolvePosition pass.)
    bool positionSolved = false;
    for (int it = 0; it < GB_POSITION_ITERS; ++it){
        bool contactsOkay = gbSolvePosition(isl);
#ifdef GB_ENABLE_JOINTS
        // b2Island::Solve: solve joints' position after contacts, in joint-list order,
        // and exit only when both are within tolerance.
        bool jointsOkay = true;
        for (int i = 0; i < jc; ++i)
            jointsOkay = gbRevoluteSolvePosition(isl.jnt[i], isl) && jointsOkay;
        if (contactsOkay && jointsOkay){ positionSolved = true; break; }
#else
        if (contactsOkay){ positionSolved = true; break; }
#endif
    }

    // ---- Copy state back to bodies + SynchronizeTransform -------------------
    for (int i = 0; i < bc; ++i){
        int bi = isl.bodies[i];
        BODY(w, sweepCx, bi)=isl.posC[i].x; BODY(w, sweepCy, bi)=isl.posC[i].y;
        BODY(w, sweepA, bi) =isl.posA[i];
        BODY(w, velX, bi)=isl.vel[i].x; BODY(w, velY, bi)=isl.vel[i].y; BODY(w, angVel, bi)=isl.velW[i];
        gbSyncTransform(w, bi);
    }

    // ---- Sleep management (minSleepTime min-fold - the THIRD serial fold) ----
    if (allowSleep){
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
        if (minSleepTime >= GB_TIME_TO_SLEEP && positionSolved){
            for (int i = 0; i < bc; ++i){
                int bi = isl.bodies[i];
                // SetAwake(false): zero velocity/sleepTime
                BODY(w, awake, bi)=0; BODY(w, sleepTime, bi)=0.0f;
                BODY(w, velX, bi)=0.0f; BODY(w, velY, bi)=0.0f; BODY(w, angVel, bi)=0.0f;
            }
        }
    }
}

// ============================================================================
// b2World::Solve. DFS island assembly over the contact graph plus the island solve.
// Serial, lane 0. Iteration order is byte-identical (descending body-seed plus
// descending incident-contact, matching the CPU prepend lists).
// ============================================================================
GB_HD inline void gbWorldSolve(GBWorld& w){
    // clear island flags (use awake as the wake set; track visited via a local mask)
    unsigned char bodyInIsland[GB_MAX_BODIES];
    unsigned char contactInIsland[GB_MAX_CONTACTS];
    for (int i=0;i<SCAL(w, bodyCount);++i) bodyInIsland[i]=0;
    for (int i=0;i<SCAL(w, contactCount);++i) contactInIsland[i]=0;
#ifdef GB_ENABLE_JOINTS
    unsigned char jointInIsland[GB_MAX_JOINTS];
    for (int i=0;i<SCAL(w, jointCount);++i) jointInIsland[i]=0;
#endif

    int stack[GB_MAX_BODIES]; // DFS stack of body slots
    GBIslandData isl;

    // seed in reverse creation order (newest body first, matching m_bodyList)
    for (int seed = SCAL(w, bodyCount)-1; seed >= 0; --seed){
        if (bodyInIsland[seed]) continue;
        if (!BODY(w, alive, seed)) continue;
        if (!BODY(w, awake, seed)) continue;             // IsAwake()==false skip
        if (BODY(w, bodyType, seed)==GB_STATIC_BODY) continue;  // static seed skip

        // reset island
        isl.bodyCount=0; isl.contactCount=0;
#ifdef GB_ENABLE_JOINTS
        isl.jointCount=0;
#endif
        int sc=0; stack[sc++]=seed; bodyInIsland[seed]=1;

        while (sc > 0){
            int b = stack[--sc];
            // add body to island
            isl.bodies[isl.bodyCount++] = b;
            // SetAwake(true) - keep awake (it is, since seed/neighbors are awake)
            BODY(w, awake, b)=1;
            if (BODY(w, bodyType, b)==GB_STATIC_BODY) continue; // don't propagate across static

            // search contacts connected to b in reverse creation order
            // (newest-first, matching the body's prepend contact-edge list)
            for (int ci = SCAL(w, contactCount)-1; ci >= 0; --ci){
                if (contactInIsland[ci]) continue;
                int ca=CONT(w, cBodyA, ci), cb=CONT(w, cBodyB, ci);
                if (ca != b && cb != b) continue;        // edge incident to b
                if (!CONT(w, cTouching, ci)) continue;    // IsTouching()==false skip
                // add contact
                isl.contacts[isl.contactCount++] = ci;
                contactInIsland[ci]=1;
                int other = (ca==b)? cb : ca;
                if (bodyInIsland[other]) continue;
                stack[sc++]=other; bodyInIsland[other]=1;
            }
#ifdef GB_ENABLE_JOINTS
            // search joints connected to b in reverse creation order (b2World::Solve
            // walks the joint edge list after the contact edge list).
            for (int ji = SCAL(w, jointCount)-1; ji >= 0; --ji){
                if (jointInIsland[ji]) continue;
                int ja=JOINT(w, jBodyA, ji), jb=JOINT(w, jBodyB, ji);
                if (ja != b && jb != b) continue;
                isl.joints[isl.jointCount++] = ji;
                jointInIsland[ji]=1;
                int other = (ja==b)? jb : ja;
                if (bodyInIsland[other]) continue;
                stack[sc++]=other; bodyInIsland[other]=1;
            }
#endif
        }

#if defined(B2_GPU_DUMP) && !defined(__CUDA_ARCH__)
        {
            fprintf(stderr, "ISLAND bodies=%d contacts=%d | bodyorder:",
                    isl.bodyCount, isl.contactCount);
            for (int i=0;i<isl.bodyCount;++i)
                fprintf(stderr, " %d(type%d)", isl.bodies[i], BODY(w, bodyType, isl.bodies[i]));
            fprintf(stderr, " | contactorder:");
            for (int i=0;i<isl.contactCount;++i){
                int ci=isl.contacts[i];
                fprintf(stderr, " (%d,%d)%s", CONT(w, cBodyA, ci), CONT(w, cBodyB, ci),
                        CONT(w, cEdge, ci)>=0?"E":"");
            }
            fprintf(stderr, "\n");
        }
#endif
        gbIslandSolve(w, isl, /*allowSleep*/true);

        // post-solve: allow static bodies to participate in other islands
        for (int i=0;i<isl.bodyCount;++i){
            int bi = isl.bodies[i];
            if (BODY(w, bodyType, bi)==GB_STATIC_BODY) bodyInIsland[bi]=0;
        }
    }
}
