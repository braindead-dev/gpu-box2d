// gb_island.cuh. b2Island::Solve (integrate, warm-start, iterate, sleep) and
// b2World::Solve (DFS island assembly), written against the gb_pools accessor
// contract (BODY/CONT/EDGE/SCAL). Uses the constraint interface in
// gb_contact_solver.cuh.
//
// STATUS: in development. The single-world and two-body micro-tests pass at 0 ULP
// on host and device; the dense five-body pile currently shows a 1-ULP device
// divergence (see test/gb_island_test.cu and docs/fidelity.md).
//
// THE FIDELITY RULE (see docs/architecture.md): the DFS island assembly, the
// constraint load, and the serial solver phases run IN-ORDER ON LANE 0. The three
// running float folds, namely (1) the in-place velocity/position accumulation in
// the sweeps, (2) the minSeparation min-fold in the position solve, and
// (3) the minSleepTime min-fold in sleep management (here), are non-associative
// under --fmad=false. No tree-reduce.
//
// Body and contact ITERATION ORDER match Box2D 2.3.0 b2Island::Solve:
//   * the DFS body-seed loop iterates m_bodyList, which is reverse creation order
//     (descending slot).
//   * each body's incident contacts iterate its edge list, also reverse creation
//     order (descending contact index).
// Body slot equals creation order (ground = 0 first); contact index increases with
// creation. Both loops descend to match the CPU island body order and contact
// solve order.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true.
#pragma once
#include "gb_pools.cuh"
#include "gb_settings.cuh"
#include "gb_math.cuh"
#include "gb_contact_solver.cuh"   // GbConstraint, GbIslandData, the solver phases

// ============================================================================
// b2Island::Solve, faithful (single-point contacts only). LANE 0, serial-in-order.
// Re-host of Box2D 2.3.0 b2Island::Solve. The serial spine is integrate-velocities ->
// load constraints -> InitializeVelocityConstraints  -> WarmStart  ->
// 8 velocity iters  -> StoreImpulses  -> integrate-positions ->
// 3 position iters  -> copy-back+sync -> sleep (minSleepTime fold).
// ============================================================================
GB_HD inline void gbIslandSolve(GBWorld& w, GbIslandData& isl, bool allowSleep){
    float h = GB_DT;
    int bc = isl.bodyCount, cc = isl.contactCount;

    // ---- Integrate velocities + damping; init body state buffers. -----------
    // (Box2D 2.3.0 b2Island::Solve: integrate v += h*(g + invM*F); store sweep.c0/a0 for CCD.)
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
            // w += h * invI * torque(0) -> unchanged
            // damping 0 -> clamp(1 - h*0,0,1)=1 -> no change
        }
        isl.posC[i] = c; isl.posA[i] = a;
        isl.vel[i] = vv; isl.velW[i] = ww;
    }

    // ---- Load velocity constraints (b2ContactSolver ctor: copy + warm-start) -
    // Fills the fused GbConstraint array from the world contact pool, in island
    // contact order. (The position-dependent portions are filled by
    // gbInitVelocityConstraints below.)
    for (int i = 0; i < cc; ++i){
        int ci = isl.contacts[i];
        GbConstraint& vc = isl.con[i];
        GbConstraint& pc = isl.con[i];   // fused: same object
        int bodyA = CONT(w, cBodyA, ci), bodyB = CONT(w, cBodyB, ci), edge = CONT(w, cEdge, ci);
        // island-local indices
        int ia=-1, ib=-1;
        for (int k=0;k<bc;++k){ if(isl.bodies[k]==bodyA) ia=k; if(isl.bodies[k]==bodyB) ib=k; }
        float radiusA, radiusB;
        if (edge < 0){ radiusA = gbBodyRadius(w,bodyA); radiusB = gbBodyRadius(w,bodyB); }
        else         { radiusA = GB_POLYGON_RADIUS;     radiusB = gbBodyRadius(w,bodyB); }

        vc.friction = CONT(w, cFriction, ci); vc.restitution = CONT(w, cRestitution, ci);
        vc.indexA = ia; vc.indexB = ib;
        vc.invMassA = BODY(w, invMass, bodyA); vc.invMassB = BODY(w, invMass, bodyB);
        vc.invIA = BODY(w, invI, bodyA);       vc.invIB = BODY(w, invI, bodyB);
        vc.contactIdx = ci;
        // warm-start impulse (dtRatio == 1 in steady DT)
        vc.p.normalImpulse  = CONT(w, cNormalImpulse, ci);   // warmStarting=true, dtRatio=1
        vc.p.tangentImpulse = CONT(w, cTangentImpulse, ci);
        vc.p.rA = v2(0,0); vc.p.rB = v2(0,0);
        vc.p.normalMass=0; vc.p.tangentMass=0; vc.p.velocityBias=0;

        pc.indexA=ia; pc.indexB=ib;
        pc.invMassA=vc.invMassA; pc.invMassB=vc.invMassB;
        pc.invIA=vc.invIA; pc.invIB=vc.invIB;
        pc.localNormal=v2(CONT(w, cLocalNormalX, ci), CONT(w, cLocalNormalY, ci));
        pc.localPoint =v2(CONT(w, cLocalPointX, ci),  CONT(w, cLocalPointY, ci));
        pc.pLocalPoint=v2(CONT(w, cPointLocalX, ci),  CONT(w, cPointLocalY, ci));
        pc.type=CONT(w, cManifoldType, ci); pc.radiusA=radiusA; pc.radiusB=radiusB;
    }

    // ---- SERIAL SOLVER SPINE (lane 0) --------------------------------------
    gbInitVelocityConstraints(w, isl);   // position-dependent portions
    gbWarmStart(isl);                    // apply carried impulse
    gbSolveVelocityConstraints(isl);     // 8 Gauss-Seidel velocity iterations
    gbStoreImpulses(w, isl);             // carry warm-start to next substep

    // ---- Integrate positions ------------------------------------------------
    // (Box2D 2.3.0 b2Island::Solve: clamp translation/rotation, advance c/a.)
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

    // ---- position solve (3 Gauss-Seidel iters + minSeparation min-fold) ---
    bool positionSolved = gbSolvePositionConstraints(isl);

    // ---- Copy state back to bodies + SynchronizeTransform -------------------
    for (int i = 0; i < bc; ++i){
        int bi = isl.bodies[i];
        BODY(w, sweepCx, bi)=isl.posC[i].x; BODY(w, sweepCy, bi)=isl.posC[i].y;
        BODY(w, sweepA, bi) =isl.posA[i];
        BODY(w, velX, bi)=isl.vel[i].x; BODY(w, velY, bi)=isl.vel[i].y; BODY(w, angVel, bi)=isl.velW[i];
        gbSyncTransform(w, bi);
    }

    // ---- Sleep management (minSleepTime min-fold, the THIRD serial fold) ----
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
// b2World::Solve. DFS island assembly over the contact graph + island solve.
// Re-host of Box2D 2.3.0 b2World::Solve. SERIAL, LANE 0. Iteration order byte-identical
// (descending body-seed + descending incident-contact, matching the CPU prepend lists).
// ============================================================================
GB_HD inline void gbWorldSolve(GBWorld& w){
    // clear island flags (use awake as the wake set; track visited via a local mask)
    unsigned char bodyInIsland[GB_MAX_BODIES];
    unsigned char contactInIsland[GB_MAX_CONTACTS];
    for (int i=0;i<SCAL(w, bodyCount);++i) bodyInIsland[i]=0;
    for (int i=0;i<SCAL(w, contactCount);++i) contactInIsland[i]=0;

    int stack[GB_MAX_BODIES]; // DFS stack of body slots
    GbIslandData isl;

    // seed in reverse creation order (newest body first, matching m_bodyList)
    for (int seed = SCAL(w, bodyCount)-1; seed >= 0; --seed){
        if (bodyInIsland[seed]) continue;
        if (!BODY(w, alive, seed)) continue;
        if (!BODY(w, awake, seed)) continue;             // IsAwake()==false skip
        if (BODY(w, bodyType, seed)==GB_STATIC_BODY) continue;  // static seed skip

        // reset island
        isl.bodyCount=0; isl.contactCount=0;
        int sc=0; stack[sc++]=seed; bodyInIsland[seed]=1;

        while (sc > 0){
            int b = stack[--sc];
            // add body to island
            isl.bodies[isl.bodyCount++] = b;
            // SetAwake(true), keep awake (it is, since seed/neighbors are awake)
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
