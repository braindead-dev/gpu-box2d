// gb_toi_orchestration.inc. gbIslandSolveTOI and gbWorldSolveTOI, the SolveTOI
// integration glue, ported byte-faithfully from the validated CCD path onto the
// accessor contract. Included by gb_step.cuh after all modules. Uses the GJK/TOI
// primitives from gb_toi.cuh and the island assembly from gb_island.cuh.
#pragma once
GB_HD inline void gbIslandSolveTOI(GBWorld& w, GBIslandData& isl, float subDt,
                                 int toiIndexA, int toiIndexB){
    int bc=isl.bodyCount, cc=isl.contactCount;
    // init body state from current sweeps
    for (int i=0;i<bc;++i){
        int bi=isl.bodies[i];
        isl.posC[i]=v2(BODY(w,sweepCx,bi), BODY(w,sweepCy,bi)); isl.posA[i]=BODY(w,sweepA,bi);
        isl.vel[i]=v2(BODY(w,velX,bi), BODY(w,velY,bi)); isl.velW[i]=BODY(w,angVel,bi);
    }
    // build position constraints (same as discrete init, minus warm-start)
    for (int i=0;i<cc;++i){
        int ci=isl.contacts[i]; GBConstraint& pc=isl.con[i];
        int bodyA=CONT(w,cBodyA,ci), bodyB=CONT(w,cBodyB,ci), edge=CONT(w,cEdge,ci);
        int ia=-1, ib=-1;
        for (int k=0;k<bc;++k){ if(isl.bodies[k]==bodyA) ia=k; if(isl.bodies[k]==bodyB) ib=k; }
        float radiusA, radiusB;
        if (edge<0){ radiusA=gbCircleRadius(w,bodyA); radiusB=gbCircleRadius(w,bodyB); }
        else       { radiusA=GB_POLYGON_RADIUS;    radiusB=gbCircleRadius(w,bodyB); }
        pc.indexA=ia; pc.indexB=ib;
        pc.invMassA=BODY(w,invMass,bodyA); pc.invMassB=BODY(w,invMass,bodyB);
        pc.invIA=BODY(w,invI,bodyA); pc.invIB=BODY(w,invI,bodyB);
        pc.localNormal=v2(CONT(w,cLocalNormalX,ci), CONT(w,cLocalNormalY,ci));
        pc.localPoint =v2(CONT(w,cLocalPointX,ci),  CONT(w,cLocalPointY,ci));
        pc.pLocalPoint=v2(CONT(w,cPointLocalX,ci),  CONT(w,cPointLocalY,ci));
        pc.type=CONT(w,cManifoldType,ci); pc.radiusA=radiusA; pc.radiusB=radiusB;
    }
    // Solve TOI position constraints (20 iters, only toiIndex bodies have mass)
    for (int iter=0; iter<20; ++iter){
        float minSeparation=0.0f;
        for (int i=0;i<cc;++i){
            GBConstraint& pc=isl.con[i];
            int ia=pc.indexA, ib=pc.indexB;
            float mA=0.0f, iA=0.0f, mB=0.0f, iB=0.0f;
            if (ia==toiIndexA || ia==toiIndexB){ mA=pc.invMassA; iA=pc.invIA; }
            if (ib==toiIndexA || ib==toiIndexB){ mB=pc.invMassB; iB=pc.invIB; }
            V2 cA=isl.posC[ia]; float aA=isl.posA[ia];
            V2 cB=isl.posC[ib]; float aB=isl.posA[ib];
            Xf xfA, xfB; xfA.q=rotSet(aA); xfB.q=rotSet(aB);
            xfA.p = cA - b2MulRV(xfA.q, v2(0,0));
            xfB.p = cB - b2MulRV(xfB.q, v2(0,0));
            V2 normal, point; float separation;
            if (pc.type==GB_MANIFOLD_CIRCLES){
                V2 pA=b2MulTV(xfA, pc.localPoint); V2 pB=b2MulTV(xfB, pc.pLocalPoint);
                normal=pB-pA; b2Normalize(normal); point=0.5f*(pA+pB);
                separation=b2Dot(pB-pA, normal)-pc.radiusA-pc.radiusB;
            } else {
                normal=b2MulRV(xfA.q, pc.localNormal);
                V2 planePoint=b2MulTV(xfA, pc.localPoint);
                V2 clipPoint =b2MulTV(xfB, pc.pLocalPoint);
                separation=b2Dot(clipPoint-planePoint, normal)-pc.radiusA-pc.radiusB; point=clipPoint;
            }
            V2 rA=point-cA, rB=point-cB;
            minSeparation=b2MinF(minSeparation, separation);
            float C=b2ClampF(0.75f/*toiBaumgarte*/*(separation+GB_LINEAR_SLOP), -GB_MAX_LINEAR_CORRECTION, 0.0f);
            float rnA=b2Cross(rA,normal), rnB=b2Cross(rB,normal);
            float K=mA+mB+iA*rnA*rnA+iB*rnB*rnB;
            float impulse=K>0.0f?-C/K:0.0f;
            V2 P=impulse*normal;
            cA=cA-mA*P; aA-=iA*b2Cross(rA,P); cB=cB+mB*P; aB+=iB*b2Cross(rB,P);
            isl.posC[ia]=cA; isl.posA[ia]=aA; isl.posC[ib]=cB; isl.posA[ib]=aB;
        }
        if (minSeparation >= -1.5f*GB_LINEAR_SLOP) break;
    }
    // leap of faith: store c0/a0 for the two TOI bodies
    {
        int ba=isl.bodies[toiIndexA], bb=isl.bodies[toiIndexB];
        BODY(w,sweepC0x,ba)=isl.posC[toiIndexA].x; BODY(w,sweepC0y,ba)=isl.posC[toiIndexA].y; BODY(w,sweepA0,ba)=isl.posA[toiIndexA];
        BODY(w,sweepC0x,bb)=isl.posC[toiIndexB].x; BODY(w,sweepC0y,bb)=isl.posC[toiIndexB].y; BODY(w,sweepA0,bb)=isl.posA[toiIndexB];
    }
    // init velocity constraints (no warm start) + solve velocity
    for (int i=0;i<cc;++i){
        GBConstraint& vc=isl.con[i]; GBConstraint& pc=isl.con[i];
        int ci=isl.contacts[i];
        vc.friction=CONT(w,cFriction,ci); vc.restitution=CONT(w,cRestitution,ci);
        vc.indexA=pc.indexA; vc.indexB=pc.indexB;
        vc.invMassA=pc.invMassA; vc.invMassB=pc.invMassB; vc.invIA=pc.invIA; vc.invIB=pc.invIB;
        vc.contactIdx=ci;
        vc.p.normalImpulse=0.0f; vc.p.tangentImpulse=0.0f; // no warm start
        int ia=vc.indexA, ib=vc.indexB;
        V2 cA=isl.posC[ia]; float aA=isl.posA[ia]; V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
        V2 cB=isl.posC[ib]; float aB=isl.posA[ib]; V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
        Xf xfA,xfB; xfA.q=rotSet(aA); xfB.q=rotSet(aB);
        xfA.p=cA-b2MulRV(xfA.q, v2(0,0)); xfB.p=cB-b2MulRV(xfB.q, v2(0,0));
        GBManifold man; man.pointCount=1; man.type=pc.type;
        man.localNormal=pc.localNormal; man.localPoint=pc.localPoint; man.pLocalPoint=pc.pLocalPoint;
        GBWorldManifold wm; gbWorldManifoldInit(wm, man, xfA, pc.radiusA, xfB, pc.radiusB);
        vc.normal=wm.normal;
        vc.p.rA=wm.point0-cA; vc.p.rB=wm.point0-cB;
        float rnA=b2Cross(vc.p.rA,vc.normal), rnB=b2Cross(vc.p.rB,vc.normal);
        float kN=vc.invMassA+vc.invMassB+vc.invIA*rnA*rnA+vc.invIB*rnB*rnB;
        vc.p.normalMass=kN>0.0f?1.0f/kN:0.0f;
        V2 tangent=b2CrossVS(vc.normal,1.0f);
        float rtA=b2Cross(vc.p.rA,tangent), rtB=b2Cross(vc.p.rB,tangent);
        float kT=vc.invMassA+vc.invMassB+vc.invIA*rtA*rtA+vc.invIB*rtB*rtB;
        vc.p.tangentMass=kT>0.0f?1.0f/kT:0.0f;
        vc.p.velocityBias=0.0f;
        float vRel=b2Dot(vc.normal, vB+b2CrossSV(wB,vc.p.rB)-vA-b2CrossSV(wA,vc.p.rA));
        if (vRel < -GB_VELOCITY_THRESHOLD) vc.p.velocityBias=-vc.restitution*vRel;
    }
    for (int iter=0; iter<GB_VELOCITY_ITERS; ++iter){
        for (int i=0;i<cc;++i){
            GBConstraint& vc=isl.con[i];
            int ia=vc.indexA, ib=vc.indexB;
            float mA=vc.invMassA, iA=vc.invIA, mB=vc.invMassB, iB=vc.invIB;
            V2 vA=isl.vel[ia]; float wA=isl.velW[ia];
            V2 vB=isl.vel[ib]; float wB=isl.velW[ib];
            V2 normal=vc.normal; V2 tangent=b2CrossVS(normal,1.0f); float friction=vc.friction;
            GBVelConstraintPt& vcp=vc.p;
            { V2 dv=vB+b2CrossSV(wB,vcp.rB)-vA-b2CrossSV(wA,vcp.rA);
              float vt=b2Dot(dv,tangent); float lambda=vcp.tangentMass*(-vt);
              float maxF=friction*vcp.normalImpulse;
              float newImp=b2ClampF(vcp.tangentImpulse+lambda,-maxF,maxF);
              lambda=newImp-vcp.tangentImpulse; vcp.tangentImpulse=newImp;
              V2 P=lambda*tangent; vA=vA-mA*P; wA-=iA*b2Cross(vcp.rA,P); vB=vB+mB*P; wB+=iB*b2Cross(vcp.rB,P); }
            { V2 dv=vB+b2CrossSV(wB,vcp.rB)-vA-b2CrossSV(wA,vcp.rA);
              float vn=b2Dot(dv,normal); float lambda=-vcp.normalMass*(vn-vcp.velocityBias);
              float newImp=b2MaxF(vcp.normalImpulse+lambda,0.0f);
              lambda=newImp-vcp.normalImpulse; vcp.normalImpulse=newImp;
              V2 P=lambda*normal; vA=vA-mA*P; wA-=iA*b2Cross(vcp.rA,P); vB=vB+mB*P; wB+=iB*b2Cross(vcp.rB,P); }
            isl.vel[ia]=vA; isl.velW[ia]=wA; isl.vel[ib]=vB; isl.velW[ib]=wB;
        }
    }
    // integrate positions over subDt + sync bodies
    float h=subDt;
    for (int i=0;i<bc;++i){
        int bi=isl.bodies[i];
        V2 c=isl.posC[i]; float a=isl.posA[i]; V2 vv=isl.vel[i]; float ww=isl.velW[i];
        V2 tr=h*vv;
        if (b2Dot(tr,tr) > GB_MAX_TRANSLATION_SQ){ float r=GB_MAX_TRANSLATION/b2Length(tr); vv=r*vv; }
        float rot=h*ww;
        if (rot*rot > GB_MAX_ROTATION_SQ){ float r=GB_MAX_ROTATION/b2AbsF(rot); ww*=r; }
        c=c+h*vv; a+=h*ww;
        BODY(w,sweepCx,bi)=c.x; BODY(w,sweepCy,bi)=c.y; BODY(w,sweepA,bi)=a;
        BODY(w,velX,bi)=vv.x; BODY(w,velY,bi)=vv.y; BODY(w,angVel,bi)=ww;
        gbSyncTransform(w, bi);
    }
}

// Build a distance proxy for one side of a contact at the current transform.

GB_HD inline void gbWorldSolveTOI(GBWorld& w){
    if (SCAL(w,stepComplete)){
        for (int b=0;b<SCAL(w,bodyCount);++b) BODY(w,sweepAlpha0,b)=0.0f;
        for (int c=0;c<SCAL(w,contactCount);++c){ CONT(w,cToiFlag,c)=0; CONT(w,cToiCount,c)=0; CONT(w,cToi,c)=1.0f; }
    }
    unsigned char bodyInIsland[GB_MAX_BODIES];
    unsigned char contactInIsland[GB_MAX_CONTACTS];
    GBIslandData isl;

    for(;;){
        int minContact=-1; float minAlpha=1.0f;
        // iterate contacts in m_contactList order (newest-first = reverse index)
        for (int ci=SCAL(w,contactCount)-1; ci>=0; --ci){
            if (!CONT(w,cEnabled,ci)) continue;
            if (CONT(w,cToiCount,ci) > 8 /*b2_maxSubSteps*/) continue;
            float alpha=1.0f;
            if (CONT(w,cToiFlag,ci)){
                alpha=CONT(w,cToi,ci);
            } else {
                int bodyA=CONT(w,cBodyA,ci), bodyB=CONT(w,cBodyB,ci), edge=CONT(w,cEdge,ci);
                int typeA=BODY(w,bodyType,bodyA), typeB=BODY(w,bodyType,bodyB);
                bool activeA = BODY(w,awake,bodyA) && typeA!=0;
                bool activeB = BODY(w,awake,bodyB) && typeB!=0;
                if (!activeA && !activeB) continue;
                bool collideA = (typeA!=2);   // not bullet, dynamic? collide if static/kinematic
                bool collideB = (typeB!=2);
                if (!collideA && !collideB) continue;  // two dynamic non-bullets -> skip
                GBSweep sA=gbBodySweep(w,bodyA), sB=gbBodySweep(w,bodyB);
                float alpha0=sA.alpha0;
                if (sA.alpha0 < sB.alpha0){ alpha0=sB.alpha0; gbSweepAdvance(sA,alpha0); gbWriteSweepAdvance(w,bodyA,sA); }
                else if (sB.alpha0 < sA.alpha0){ alpha0=sA.alpha0; gbSweepAdvance(sB,alpha0); gbWriteSweepAdvance(w,bodyB,sB); }
                GBDProxy pA=gbContactProxy(w,bodyA,edge,true, gbCircleRadius(w,bodyA));
                GBDProxy pB=gbContactProxy(w,bodyB,edge,false, gbCircleRadius(w,bodyB));
                GBTOIOut out; gbTOI(out, pA, pB, sA, sB, 1.0f);
                float beta=out.t;
                if (out.state==GB_TOI_TOUCHING) alpha=b2MinF(alpha0 + (1.0f-alpha0)*beta, 1.0f);
                else alpha=1.0f;
                CONT(w,cToi,ci)=alpha; CONT(w,cToiFlag,ci)=1;
            }
            if (alpha < minAlpha){ minContact=ci; minAlpha=alpha; }
        }
        if (minContact<0 || 1.0f - 10.0f*GB_EPSILON < minAlpha){ SCAL(w,stepComplete)=1; break; }

        int ci=minContact;
        int bodyA=CONT(w,cBodyA,ci), bodyB=CONT(w,cBodyB,ci), edge=CONT(w,cEdge,ci);
        // backup sweeps
        GBSweep backup1=gbBodySweep(w,bodyA), backup2=gbBodySweep(w,bodyB);
        // advance both bodies to minAlpha
        GBSweep sA=backup1, sB=backup2;
        gbSweepAdvance(sA, minAlpha); gbWriteSweepAdvance(w,bodyA,sA);
        // also set current c/a from interpolation done in Advance? Advance only moves c0/a0.
        // b2Body::Advance also sets sweep.c = sweep.c0 etc.; replicate:
        { GBSweep t=gbBodySweep(w,bodyA); Xf xf; gbSweepGetTransform(t,xf,minAlpha);
          // b2Body::Advance: m_sweep.Advance(alpha); then c=c0,a=a0; sync transform
        }
        // Faithful b2Body::Advance: sweep.Advance(alpha); c=c0; a=a0; SynchronizeTransform
        auto bodyAdvance = [&](int bi, float alpha){
            GBSweep s=gbBodySweep(w,bi); gbSweepAdvance(s,alpha); gbWriteSweepAdvance(w,bi,s);
            BODY(w,sweepCx,bi)=s.c0.x; BODY(w,sweepCy,bi)=s.c0.y; BODY(w,sweepA,bi)=s.a0;
            gbSyncTransform(w,bi);
        };
        // redo cleanly using bodyAdvance for both
        BODY(w,sweepC0x,bodyA)=backup1.c0.x; BODY(w,sweepC0y,bodyA)=backup1.c0.y; BODY(w,sweepA0,bodyA)=backup1.a0; BODY(w,sweepAlpha0,bodyA)=backup1.alpha0;
        bodyAdvance(bodyA, minAlpha);
        bodyAdvance(bodyB, minAlpha);

        // The TOI contact likely has some new contact points -> update manifold + touching
        gbContactUpdate(w, ci);
        CONT(w,cToiFlag,ci)=0; ++CONT(w,cToiCount,ci);

        if (!CONT(w,cEnabled,ci) || !CONT(w,cTouching,ci)){
            // restore
            CONT(w,cEnabled,ci)=0;
            BODY(w,sweepC0x,bodyA)=backup1.c0.x; BODY(w,sweepC0y,bodyA)=backup1.c0.y; BODY(w,sweepA0,bodyA)=backup1.a0; BODY(w,sweepAlpha0,bodyA)=backup1.alpha0;
            BODY(w,sweepCx,bodyA)=backup1.c.x; BODY(w,sweepCy,bodyA)=backup1.c.y; BODY(w,sweepA,bodyA)=backup1.a; gbSyncTransform(w,bodyA);
            BODY(w,sweepC0x,bodyB)=backup2.c0.x; BODY(w,sweepC0y,bodyB)=backup2.c0.y; BODY(w,sweepA0,bodyB)=backup2.a0; BODY(w,sweepAlpha0,bodyB)=backup2.alpha0;
            BODY(w,sweepCx,bodyB)=backup2.c.x; BODY(w,sweepCy,bodyB)=backup2.c.y; BODY(w,sweepA,bodyB)=backup2.a; gbSyncTransform(w,bodyB);
            continue;
        }
        BODY(w,awake,bodyA)=1; BODY(w,awake,bodyB)=1;

        // Build TOI island: bodyA, bodyB, contact ci. Then neighbours (static/bullet only).
        for (int i=0;i<SCAL(w,bodyCount);++i) bodyInIsland[i]=0;
        for (int i=0;i<SCAL(w,contactCount);++i) contactInIsland[i]=0;
        isl.bodyCount=0; isl.contactCount=0;
        // add order matches CPU: Add(bA); Add(bB); Add(minContact)
        int localA=isl.bodyCount; isl.bodies[isl.bodyCount++]=bodyA; bodyInIsland[bodyA]=1;
        int localB=isl.bodyCount; isl.bodies[isl.bodyCount++]=bodyB; bodyInIsland[bodyB]=1;
        isl.contacts[isl.contactCount++]=ci; contactInIsland[ci]=1;
        // neighbours of bodyA, bodyB (only if dynamic body iterates)
        int seeds[2]={bodyA, bodyB};
        for (int si=0; si<2; ++si){
            int body=seeds[si];
            if (BODY(w,bodyType,body)!=2) continue;  // only dynamic bodies expand
            // iterate this body's contacts (reverse index = newest first)
            for (int ce=SCAL(w,contactCount)-1; ce>=0; --ce){
                if (contactInIsland[ce]) continue;
                int ca=CONT(w,cBodyA,ce), cb=CONT(w,cBodyB,ce);
                if (ca!=body && cb!=body) continue;
                int other=(ca==body)?cb:ca;
                // only add static/kinematic/bullet bodies (skip dynamic-dynamic)
                if (BODY(w,bodyType,other)==2) continue;
                // tentatively advance other to minAlpha
                GBSweep backup=gbBodySweep(w,other);
                if (!bodyInIsland[other]) bodyAdvance(other, minAlpha);
                gbContactUpdate(w, ce);
                if (!CONT(w,cEnabled,ce) || !CONT(w,cTouching,ce)){
                    // restore other
                    BODY(w,sweepC0x,other)=backup.c0.x; BODY(w,sweepC0y,other)=backup.c0.y; BODY(w,sweepA0,other)=backup.a0; BODY(w,sweepAlpha0,other)=backup.alpha0;
                    BODY(w,sweepCx,other)=backup.c.x; BODY(w,sweepCy,other)=backup.c.y; BODY(w,sweepA,other)=backup.a; gbSyncTransform(w,other);
                    continue;
                }
                contactInIsland[ce]=1; isl.contacts[isl.contactCount++]=ce;
                if (bodyInIsland[other]) continue;
                bodyInIsland[other]=1;
                if (BODY(w,bodyType,other)!=0) BODY(w,awake,other)=1;
                isl.bodies[isl.bodyCount++]=other;
            }
        }

        float subDt=(1.0f-minAlpha)*GB_DT;
        gbIslandSolveTOI(w, isl, subDt, localA, localB);

        // invalidate TOI flags on displaced dynamic bodies' contacts
        for (int i=0;i<isl.bodyCount;++i){
            int bi=isl.bodies[i];
            if (BODY(w,bodyType,bi)!=2) continue;
            for (int ce=0; ce<SCAL(w,contactCount); ++ce){
                if (CONT(w,cBodyA,ce)==bi || CONT(w,cBodyB,ce)==bi){ CONT(w,cToiFlag,ce)=0; }
            }
        }
        // (FindNewContacts handled by next substep's collidePhase)
    }
}

// ============================================================================
// The CCD/TOI pass completes b2World::Step (Collide -> Solve -> SolveTOI). The CCD
// path is required: Box2D's TOI fires on dynamic-static contacts during settling and
// moves positions enough to fork trajectories, so a faithful engine includes it.
// ============================================================================
