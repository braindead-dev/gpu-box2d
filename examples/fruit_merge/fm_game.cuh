// fm_game.cuh. The fruit-merge game layer, on the gb_pools accessor contract.
//
// This is the example game logic, separated from the general gpu-box2d physics core.
// The split is what keeps the core publishable:
//   * the game (this file) owns merge, spawn, death, score, queue, RNG, and the
//     ContactListener-style merge hook;
//   * the physics core owns the step and reads a circle's radius via gbCircleRadius
//     (BODY(w,radius,s)), a general field the game populates at body creation, so the
//     core never needs the game's tier map.
//
// CONTRACT DISCIPLINE:
//   * Touch WorldShared only through the accessor macros BODY/CONT/EDGE/SCAL and
//     gbCircleRadius (gb_pools.cuh). Use no raw `w.field[s]` and add no new globals.
//   * Call the step `gb_world_step(GBWorld&)` (gb_world.cuh). The game never reaches
//     into the broad-phase, narrow-phase, solver, or TOI internals; physics is a black
//     box behind the step.
//   * At body creation, set BODY(w,radius,s) from fm_tier_radius (the game's map); the
//     physics modules then read that radius. BODY(w,tier,s) stays a game-only field
//     the physics ignores.
//   * The merge-pair iteration order in fmProcessContacts is load-bearing for
//     determinism and is preserved byte-for-byte from the validated CPU reference. Do
//     not reorder it.
//
// This is the validated CPU game logic with arena field access rewritten onto the
// accessor contract and the physics call routed through gb_world_step.
#pragma once
#include "gpu_box2d/gb_pools.cuh"
#include "gpu_box2d/gb_math.cuh"
#include "gpu_box2d/gb_world.cuh"   // gb_world_step(GBWorld&) - the frozen step

// ============================================================================
// GAME CONSTANTS (the CPU reference / config.py). These are GAME data - they live in the
// example layer; gb_settings.cuh stays game-agnostic.
// Exact values from the CPU reference (the validated CPU/GPU contract).
// ============================================================================
#define FM_N_TIERS          12       // fruit tiers 0..11
#define FM_GOLDAPPLE        11       // tier 11 = GoldApple (terminal merge)
#define FM_N_BINS           16       // action bins
#define FM_N_SPAWNABLE      5        // SPAWNABLE_TIERS = [0,1,2,3,4]
#define FM_SPAWN_DELAY      0.5f     // merge-eligible only when age > this
#define FM_DEATH_TIME       4.0f     // outline > this => dead
#define FM_INTER_DROP_TIME  1.3f     // outline bump applied to top fruits at settle end
#define FM_DEAD_Y           7.75f    // death line (the container height is 9.5)
#define FM_SPAWN_Y          8.75f    // drop spawn height
#define FM_MAX_SETTLE_STEPS 300      // settle substep cap
#define FM_WALL_X           3.75f    // half-width of the container
#define FM_DT               (1.0f/60.0f)
#define FM_DENSITY          1.0f     // fruit material density (the CPU reference)

// Radii by tier (the CPU reference RADII). The GAME's tier->radius map; written into the
// GENERAL physics field BODY(w,radius,s) at spawn so the core stays game-agnostic.
GB_HD inline float fm_tier_radius(int tier){
    const float R[FM_N_TIERS] = {0.25f,0.28f,0.5f,0.525f,0.66f,0.84f,0.975f,
                                 1.2f,1.32f,1.65f,1.95f,2.2f};
    return R[tier];
}
// Source-indexed FruitScore (the CPU reference MERGE_SCORE): merging two tier-t fruits
// gains S[t]. Length N_TIERS-1 (no score for merging into the terminal tier index).
GB_HD inline int fm_merge_score(int tier){
    const int S[FM_N_TIERS-1] = {1,3,7,9,13,21,27,34,44,62,90};
    return S[tier];
}
// splitmix64 - per-world deterministic RNG. Bit-identical to the CPU reference sm64.
GB_HD inline unsigned long long fm_sm64(unsigned long long& s){
    unsigned long long z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
GB_HD inline int fm_rand_tier(unsigned long long& s){ return (int)(fm_sm64(s) % FM_N_SPAWNABLE); }

// ============================================================================
// THE MERGE HOOK (ContactListener). The PHYSICS step fires these on touching
// transitions, in contact order, exactly as Box2D's b2ContactListener::Begin/End
// drive the CPU MergeListener (the CPU reference). They are GAME logic (tier eligibility,
// the insertion-ordered same-tier pair store) plugged into the general step.
//
// The pair store (pairLo/pairHi/pairA/pairB/pairCount) is a frozen WorldShared field
// set; the order is INSERTION ORDER - an existing key updates in place (keeps its
// position), a new key appends. fmProcessContacts consumes a snapshot in this order.
// Key = (min,max) body slot. pairA/pairB keep fixtureA/fixtureB body slots.
// ============================================================================
GB_HD inline void fmMergeBeginContact(GBWorld& w, int bodyA, int bodyB){
    if (bodyA==GB_GROUND || bodyB==GB_GROUND) return;             // wall contact: no merge
    if (!BODY(w,alive,bodyA) || !BODY(w,alive,bodyB)) return;
    int ta=BODY(w,tier,bodyA), tb=BODY(w,tier,bodyB);
    if (ta==tb && ta < FM_GOLDAPPLE){
        int lo = bodyA<bodyB?bodyA:bodyB, hi = bodyA<bodyB?bodyB:bodyA;
        for (int i=0;i<SCAL(w,pairCount);++i)
            if (CONT(w,pairLo,i)==lo && CONT(w,pairHi,i)==hi){     // existing key: update in place
                CONT(w,pairA,i)=bodyA; CONT(w,pairB,i)=bodyB; return;
            }
        if (SCAL(w,pairCount) < GB_MAX_PAIRS){                     // new key: append (insertion order)
            int i=SCAL(w,pairCount)++;
            CONT(w,pairLo,i)=lo; CONT(w,pairHi,i)=hi; CONT(w,pairA,i)=bodyA; CONT(w,pairB,i)=bodyB;
        }
    }
}
GB_HD inline void fmMergeEndContact(GBWorld& w, int bodyA, int bodyB){
    if (bodyA==GB_GROUND || bodyB==GB_GROUND) return;
    int lo = bodyA<bodyB?bodyA:bodyB, hi = bodyA<bodyB?bodyB:bodyA;
    for (int i=0;i<SCAL(w,pairCount);++i)
        if (CONT(w,pairLo,i)==lo && CONT(w,pairHi,i)==hi){
            for (int k=i+1;k<SCAL(w,pairCount);++k){               // erase, shift down (preserve order)
                CONT(w,pairLo,k-1)=CONT(w,pairLo,k); CONT(w,pairHi,k-1)=CONT(w,pairHi,k);
                CONT(w,pairA,k-1) =CONT(w,pairA,k);  CONT(w,pairB,k-1) =CONT(w,pairB,k);
            }
            --SCAL(w,pairCount); return;
        }
}
// find/erase used by fmProcessContacts (mirror mergeFind / mergeEraseAt).
GB_HD inline int fmMergeFind(const GBWorld& w, int lo, int hi){
    for (int i=0;i<SCAL(w,pairCount);++i)
        if (CONT(w,pairLo,i)==lo && CONT(w,pairHi,i)==hi) return i;
    return -1;
}
GB_HD inline void fmMergeEraseAt(GBWorld& w, int idx){
    for (int k=idx+1;k<SCAL(w,pairCount);++k){
        CONT(w,pairLo,k-1)=CONT(w,pairLo,k); CONT(w,pairHi,k-1)=CONT(w,pairHi,k);
        CONT(w,pairA,k-1) =CONT(w,pairA,k);  CONT(w,pairB,k-1) =CONT(w,pairB,k);
    }
    --SCAL(w,pairCount);
}

// ============================================================================
// fmSyncTransform - refresh the cached transform (m_xf) from the sweep for a body.
// The game writes a new body's position directly into the sweep, so it must seed the
// transform cache (Box2D's b2Body ctor does SynchronizeTransform). localCenter==0 for
// our circles, so xf.p == sweep.c and xf.q == rot(sweep.a). This touches only frozen
// body fields via accessors (the physics step recomputes it during the step proper).
// ============================================================================
GB_HD inline void fmSyncTransform(GBWorld& w, int s){
    Rot q = rotSet(BODY(w,sweepA,s));
    BODY(w,xfQs,s)=q.s; BODY(w,xfQc,s)=q.c;
    BODY(w,xfPx,s)=BODY(w,sweepCx,s);   // - Mul(q, localCenter=0)
    BODY(w,xfPy,s)=BODY(w,sweepCy,s);
}

// ============================================================================
// addFruit - append a new dynamic circle body at the next free slot (mirrors
// the CPU reference add_fruit). Sets mass from the tier and, per the
// frozen contract, SETS BODY(w,radius,s) from the GAME's tier map so the physics
// modules (collision/CCD/solver) can read it via gbCircleRadius. tier stays GAME-only.
// ============================================================================
GB_HD inline int fmAddFruit(GBWorld& w, int tier, float x, float y, float vy, float age0){
    int s = SCAL(w,bodyCount)++;
    float r = fm_tier_radius(tier);
    float mass = FM_DENSITY * GB_PI * r * r;            // b2CircleShape::ComputeMass
    float I    = mass * (0.5f * r * r);
    BODY(w,bodyType,s)=GB_DYNAMIC_BODY; BODY(w,tier,s)=tier;
    BODY(w,radius,s)=r;                                 // GENERAL physics field (publishability)
    BODY(w,invMass,s)= mass>0.0f ? 1.0f/mass : 0.0f;
    BODY(w,invI,s)   = I>0.0f ? 1.0f/I : 0.0f;
    BODY(w,alive,s)=1; BODY(w,awake,s)=1; BODY(w,sleepTime,s)=0.0f;
    BODY(w,sweepCx,s)=x; BODY(w,sweepCy,s)=y; BODY(w,sweepA,s)=0.0f;
    BODY(w,sweepC0x,s)=x; BODY(w,sweepC0y,s)=y; BODY(w,sweepA0,s)=0.0f; BODY(w,sweepAlpha0,s)=0.0f;
    BODY(w,velX,s)=0.0f; BODY(w,velY,s)=vy; BODY(w,angVel,s)=0.0f;
    BODY(w,age,s)=age0; BODY(w,outline,s)=0.0f;
    fmSyncTransform(w, s);
    return s;
}

// ============================================================================
// fmDestroyFruit - deactivate a body and remove its contacts, firing EndContact on
// each touching incident contact first (mirrors b2World::DestroyBody +
// the CPU reference destroyFruit). Keeps the merge-pair store consistent.
// ============================================================================
GB_HD inline void fmDestroyFruit(GBWorld& w, int body){
    // EndContact for each touching contact incident to this body.
    for (int i=0;i<SCAL(w,contactCount);++i){
        int a=CONT(w,cBodyA,i), b=CONT(w,cBodyB,i);
        if (a!=body && b!=body) continue;
        if (CONT(w,cTouching,i)) fmMergeEndContact(w, a, b);
    }
    // compact-remove all contacts incident to body (same field set as gb_pools contacts).
    int n=0;
    for (int i=0;i<SCAL(w,contactCount);++i){
        if (CONT(w,cBodyA,i)==body || CONT(w,cBodyB,i)==body) continue;
        if (n!=i){
            CONT(w,cBodyA,n)=CONT(w,cBodyA,i); CONT(w,cBodyB,n)=CONT(w,cBodyB,i); CONT(w,cEdge,n)=CONT(w,cEdge,i);
            CONT(w,cTouching,n)=CONT(w,cTouching,i);
            CONT(w,cFriction,n)=CONT(w,cFriction,i); CONT(w,cRestitution,n)=CONT(w,cRestitution,i);
            CONT(w,cManifoldType,n)=CONT(w,cManifoldType,i);
            CONT(w,cLocalNormalX,n)=CONT(w,cLocalNormalX,i); CONT(w,cLocalNormalY,n)=CONT(w,cLocalNormalY,i);
            CONT(w,cLocalPointX,n)=CONT(w,cLocalPointX,i); CONT(w,cLocalPointY,n)=CONT(w,cLocalPointY,i);
            CONT(w,cPointLocalX,n)=CONT(w,cPointLocalX,i); CONT(w,cPointLocalY,n)=CONT(w,cPointLocalY,i);
            CONT(w,cNormalImpulse,n)=CONT(w,cNormalImpulse,i); CONT(w,cTangentImpulse,n)=CONT(w,cTangentImpulse,i);
            CONT(w,cToi,n)=CONT(w,cToi,i); CONT(w,cToiCount,n)=CONT(w,cToiCount,i);
            CONT(w,cToiFlag,n)=CONT(w,cToiFlag,i); CONT(w,cEnabled,n)=CONT(w,cEnabled,i);
        }
        ++n;
    }
    SCAL(w,contactCount)=n;
    BODY(w,alive,body)=0; BODY(w,awake,body)=0;
}

// ============================================================================
// fmCompactFruits - after merges, keep alive fruits in CREATION ORDER, rebuild slots
// 1..k, and remap contacts + merge pairs (mirror fruits.swap(keep) / compactFruits).
// This keeps n_fruits()/obs iteration in the same order as the CPU fruit vector.
// Ground stays slot 0. (Uses GB_MAX_BODIES for the remap scratch - frozen bound.)
// ============================================================================
GB_HD inline void fmCompactFruits(GBWorld& w){
    int remap[GB_MAX_BODIES];
    for (int i=0;i<GB_MAX_BODIES;++i) remap[i]=-1;
    remap[GB_GROUND]=GB_GROUND;
    int dst=1;
    for (int s=1;s<SCAL(w,bodyCount);++s){
        if (!BODY(w,alive,s)) continue;
        remap[s]=dst;
        if (dst!=s){
            BODY(w,sweepCx,dst)=BODY(w,sweepCx,s); BODY(w,sweepCy,dst)=BODY(w,sweepCy,s);
            BODY(w,sweepC0x,dst)=BODY(w,sweepC0x,s); BODY(w,sweepC0y,dst)=BODY(w,sweepC0y,s);
            BODY(w,sweepA,dst)=BODY(w,sweepA,s); BODY(w,sweepA0,dst)=BODY(w,sweepA0,s); BODY(w,sweepAlpha0,dst)=BODY(w,sweepAlpha0,s);
            BODY(w,xfPx,dst)=BODY(w,xfPx,s); BODY(w,xfPy,dst)=BODY(w,xfPy,s); BODY(w,xfQs,dst)=BODY(w,xfQs,s); BODY(w,xfQc,dst)=BODY(w,xfQc,s);
            BODY(w,velX,dst)=BODY(w,velX,s); BODY(w,velY,dst)=BODY(w,velY,s); BODY(w,angVel,dst)=BODY(w,angVel,s);
            BODY(w,invMass,dst)=BODY(w,invMass,s); BODY(w,invI,dst)=BODY(w,invI,s);
            BODY(w,radius,dst)=BODY(w,radius,s);                 // general physics field moves with the body
            BODY(w,tier,dst)=BODY(w,tier,s); BODY(w,bodyType,dst)=BODY(w,bodyType,s);
            BODY(w,sleepTime,dst)=BODY(w,sleepTime,s); BODY(w,awake,dst)=BODY(w,awake,s); BODY(w,alive,dst)=1;
            BODY(w,age,dst)=BODY(w,age,s); BODY(w,outline,dst)=BODY(w,outline,s);
        }
        ++dst;
    }
    SCAL(w,bodyCount)=dst;
    // remap contacts
    for (int i=0;i<SCAL(w,contactCount);++i){
        CONT(w,cBodyA,i) = remap[CONT(w,cBodyA,i)];
        CONT(w,cBodyB,i) = remap[CONT(w,cBodyB,i)];
    }
    // remap merge pairs (recompute lo/hi from remapped slots)
    for (int i=0;i<SCAL(w,pairCount);++i){
        int a=remap[CONT(w,pairA,i)], b=remap[CONT(w,pairB,i)];
        CONT(w,pairA,i)=a; CONT(w,pairB,i)=b;
        CONT(w,pairLo,i)=a<b?a:b; CONT(w,pairHi,i)=a<b?b:a;
    }
}

// ============================================================================
// fmProcessContacts. Iterate a snapshot of the insertion-ordered pair list and merge
// eligible pairs (matches the CPU reference process_contacts). Returns score gained.
//
// THE PAIR-ITERATION ORDER IS LOAD-BEARING: it is the snapshot order, which is
// insertion order. The merged-set guard and the in-place mid-point double-precision
// math are reproduced exactly so the float results match the CPU. Do not reorder the
// iteration or change the math.
// ============================================================================
GB_HD inline int fmProcessContacts(GBWorld& w){
    int gained=0;
    // snapshot of pairs (copy) - mirrors list(pairs.items())
    int snapLo[GB_MAX_PAIRS], snapHi[GB_MAX_PAIRS], snapA[GB_MAX_PAIRS], snapB[GB_MAX_PAIRS];
    int snapN=SCAL(w,pairCount);
    for (int i=0;i<snapN;++i){
        snapLo[i]=CONT(w,pairLo,i); snapHi[i]=CONT(w,pairHi,i);
        snapA[i]=CONT(w,pairA,i);   snapB[i]=CONT(w,pairB,i);
    }
    int merged[GB_MAX_PAIRS*2]; int mergedN=0;
    bool anyMerge=false;
    for (int p=0;p<snapN;++p){
        int a=snapA[p], b=snapB[p], lo=snapLo[p], hi=snapHi[p];
        if (!(BODY(w,alive,a) && BODY(w,alive,b))){
            int idx=fmMergeFind(w,lo,hi); if (idx>=0) fmMergeEraseAt(w,idx);
            continue;
        }
        if (BODY(w,age,a) <= FM_SPAWN_DELAY || BODY(w,age,b) <= FM_SPAWN_DELAY) continue;
        bool inM=false;
        for (int m=0;m<mergedN;++m) if (merged[m]==a || merged[m]==b){ inM=true; break; }
        if (inM) continue;
        int idx=fmMergeFind(w,lo,hi); if (idx>=0) fmMergeEraseAt(w,idx);
        int ti=BODY(w,tier,a);
        float pax=BODY(w,sweepCx,a), pay=BODY(w,sweepCy,a);
        float pbx=BODY(w,sweepCx,b), pby=BODY(w,sweepCy,b);
        float mx = ((double)pax + (double)pbx)/2.0;   // midpoint (double precision, matching the CPU reference)
        float my = ((double)pay + (double)pby)/2.0;
        merged[mergedN++]=a; merged[mergedN++]=b;
        BODY(w,alive,a)=0; BODY(w,alive,b)=0;
        fmDestroyFruit(w, a); fmDestroyFruit(w, b);   // fires EndContact, removes contacts
        int nt=ti+1;
        gained += fm_merge_score(ti);
        if (nt > SCAL(w,maxTier)) SCAL(w,maxTier)=nt;
        if (nt >= FM_GOLDAPPLE){ SCAL(w,goldapples) += 1; }
        else { fmAddFruit(w, nt, mx, my, 0.0f, 0.0f); }
        anyMerge=true;
    }
    if (anyMerge) fmCompactFruits(w);
    return gained;
}

// awake_count (matches the CPU reference).
GB_HD inline int fmAwakeCount(const GBWorld& w){
    int c=0;
    for (int s=1;s<SCAL(w,bodyCount);++s) if (BODY(w,alive,s) && BODY(w,awake,s)) ++c;
    return c;
}

// ============================================================================
// settle_and_merge (matches the CPU reference). Cap
// FM_MAX_SETTLE_STEPS; each substep: STEP physics, age+=DT, process_contacts, outline
// update. Break when nothing awake & no pending merges. Post-settle: top fruits accrue
// INTER_DROP_TIME outline. Returns total score gained.
//
// The physics is the frozen step gb_world_step(w) - the game NEVER reaches into it.
// gb_world_step is responsible for firing the merge hook (fmMergeBegin/EndContact),
// exactly as the CPU collidePhase fires the CPU MergeListener.
// ============================================================================
GB_HD inline int fmSettleAndMerge(GBWorld& w){
    int gained=0;
    for (int step=0; step<FM_MAX_SETTLE_STEPS; ++step){
        gb_world_step(w);
        for (int s=1;s<SCAL(w,bodyCount);++s) if (BODY(w,alive,s)) BODY(w,age,s) += FM_DT;
        int g = fmProcessContacts(w);
        gained += g;
        for (int s=1;s<SCAL(w,bodyCount);++s){
            if (!BODY(w,alive,s)) continue;
            float top = BODY(w,sweepCy,s) + fm_tier_radius(BODY(w,tier,s));
            if (top > FM_DEAD_Y) BODY(w,outline,s) += FM_DT;
            else                 BODY(w,outline,s) = 0.0f;
        }
        if (g==0 && SCAL(w,pairCount)==0 && fmAwakeCount(w)==0) break;
    }
    // post-settle: top fruits accrue INTER_DROP_TIME outline (the CPU reference)
    for (int s=1;s<SCAL(w,bodyCount);++s){
        if (!BODY(w,alive,s)) continue;
        float top = BODY(w,sweepCy,s) + fm_tier_radius(BODY(w,tier,s));
        if (top > FM_DEAD_Y) BODY(w,outline,s) += FM_INTER_DROP_TIME;
    }
    return gained;
}

// ---- one settle SUBSTEP (physics + age + merge + outline). For the activity-
// compacted batched driver : run worlds substep-synchronously, drop finished ones
// each substep. Sets done once settled (nothing awake, no pending merges). ----------
GB_HD inline int fmSettleOneSubstep(GBWorld& w, bool& done){
    gb_world_step(w);
    for (int s=1;s<SCAL(w,bodyCount);++s) if (BODY(w,alive,s)) BODY(w,age,s) += FM_DT;
    int g = fmProcessContacts(w);
    for (int s=1;s<SCAL(w,bodyCount);++s){
        if (!BODY(w,alive,s)) continue;
        float top = BODY(w,sweepCy,s) + fm_tier_radius(BODY(w,tier,s));
        if (top > FM_DEAD_Y) BODY(w,outline,s) += FM_DT;
        else                 BODY(w,outline,s) = 0.0f;
    }
    done = (g==0 && SCAL(w,pairCount)==0 && fmAwakeCount(w)==0);
    return g;
}
// post-settle outline bump (the CPU reference), applied once when a world finishes.
GB_HD inline void fmSettleFinalize(GBWorld& w){
    for (int s=1;s<SCAL(w,bodyCount);++s){
        if (!BODY(w,alive,s)) continue;
        float top = BODY(w,sweepCy,s) + fm_tier_radius(BODY(w,tier,s));
        if (top > FM_DEAD_Y) BODY(w,outline,s) += FM_INTER_DROP_TIME;
    }
}

// check_dead (matches the CPU reference).
GB_HD inline bool fmCheckDead(const GBWorld& w){
    for (int s=1;s<SCAL(w,bodyCount);++s)
        if (BODY(w,alive,s) && BODY(w,outline,s) > FM_DEATH_TIME) return true;
    return false;
}

// bin -> world x for the current tier (matches the CPU reference bin mapping).
GB_HD inline float fmBinToX(int cur, int action_bin){
    float r = fm_tier_radius(cur);
    float usable = FM_WALL_X - r;
    int bin = action_bin<0?0:(action_bin>=FM_N_BINS?FM_N_BINS-1:action_bin);
    return -usable + (2.0f*usable) * ((float)bin/(float)(FM_N_BINS-1));
}

// ============================================================================
// fmGameStepX - drop cur at explicit world x (clamped), settle, advance queue.
// Mirrors the CPU reference step_x. Used by lookahead.
// ============================================================================
GB_HD inline int fmGameStepX(GBWorld& w, float world_x, int next_nxt){
    int cur=SCAL(w,cur);
    float r = fm_tier_radius(cur);
    float x = world_x;
    float loX=-FM_WALL_X+r, hiX=FM_WALL_X-r;
    if (x<loX) x=loX; else if (x>hiX) x=hiX;
    fmAddFruit(w, cur, x, FM_SPAWN_Y, -r, 0.0f);
    int gained = fmSettleAndMerge(w);
    SCAL(w,score) += gained;
    SCAL(w,drops) += 1;
    SCAL(w,cur) = SCAL(w,nxt); SCAL(w,nxt) = next_nxt;
    SCAL(w,dead) = fmCheckDead(w) ? 1 : 0;
    return gained;
}

// ============================================================================
// fmGameStep - one env-step: drop cur at bin x, settle, advance queue.
// Mirrors the CPU reference step.
// ============================================================================
GB_HD inline int fmGameStep(GBWorld& w, int action_bin, int next_nxt){
    float x = fmBinToX(SCAL(w,cur), action_bin);
    return fmGameStepX(w, x, next_nxt);
}

// ============================================================================
// fmInitEmptyWorld - reset a world to an empty board (mirror game_ref.cu initEmpty /
// the CPU reference initEmptyWorld + FruitWorld::reset). Ground = slot 0, 3 edge fixtures
// (floor + 2 walls). Game scalars zeroed; queue left to the caller (respawn sets it).
// ============================================================================
GB_HD inline void fmInitEmptyWorld(GBWorld& w){
    for (int i=0;i<GB_MAX_BODIES;++i){
        BODY(w,alive,i)=0; BODY(w,awake,i)=0; BODY(w,tier,i)=-1; BODY(w,bodyType,i)=GB_STATIC_BODY;
        BODY(w,invMass,i)=0.0f; BODY(w,invI,i)=0.0f; BODY(w,sleepTime,i)=0.0f;
        BODY(w,age,i)=0.0f; BODY(w,outline,i)=0.0f; BODY(w,sweepAlpha0,i)=0.0f;
        BODY(w,radius,i)=0.0f;
    }
    SCAL(w,contactCount)=0; SCAL(w,pairCount)=0; SCAL(w,stepComplete)=1;
    // ground (static, slot 0)
    BODY(w,bodyType,GB_GROUND)=GB_STATIC_BODY; BODY(w,tier,GB_GROUND)=-1;
    BODY(w,alive,GB_GROUND)=1; BODY(w,awake,GB_GROUND)=0;
    BODY(w,sweepCx,GB_GROUND)=0.0f; BODY(w,sweepCy,GB_GROUND)=0.0f; BODY(w,sweepA,GB_GROUND)=0.0f;
    BODY(w,sweepC0x,GB_GROUND)=0.0f; BODY(w,sweepC0y,GB_GROUND)=0.0f; BODY(w,sweepA0,GB_GROUND)=0.0f;
    fmSyncTransform(w, GB_GROUND);
    // 3 ground edges: floor, left wall, right wall (the container geometry).
    EDGE(w,edgeAx,0)=-FM_WALL_X; EDGE(w,edgeAy,0)=0.0f;        EDGE(w,edgeBx,0)= FM_WALL_X; EDGE(w,edgeBy,0)=0.0f;
    EDGE(w,edgeAx,1)=-FM_WALL_X; EDGE(w,edgeAy,1)=0.0f;        EDGE(w,edgeBx,1)=-FM_WALL_X; EDGE(w,edgeBy,1)=9.5f;
    EDGE(w,edgeAx,2)= FM_WALL_X; EDGE(w,edgeAy,2)=0.0f;        EDGE(w,edgeBx,2)= FM_WALL_X; EDGE(w,edgeBy,2)=9.5f;
    SCAL(w,bodyCount)=1;
    SCAL(w,score)=0; SCAL(w,drops)=0; SCAL(w,maxTier)=0; SCAL(w,goldapples)=0;
    SCAL(w,cur)=0; SCAL(w,nxt)=0; SCAL(w,dead)=0;
}

// ============================================================================
// fmRespawnWorld - reseed + respawn for batch index i (mirror the CPU reference
// reseed and respawn). Seeds the per-world RNG from the
// master seed, BURNS ONE to decorrelate, resets the board, draws cur+nxt. Bit-exact.
// ============================================================================
GB_HD inline void fmRespawnWorld(GBWorld& w, unsigned long long master, int i){
    unsigned long long s = master ^ (0xD1B54A32D192ED03ULL * (unsigned long long)(i+1));
    (void)fm_sm64(s);                       // burn one
    fmInitEmptyWorld(w);
    int c = fm_rand_tier(s);
    int n = fm_rand_tier(s);
    SCAL(w,cur)=c; SCAL(w,nxt)=n;
    SCAL(w,rng)=s;
}

// ============================================================================
// fmBatchStep - the committed env-step for one world in a batch (mirror the CPU reference
// kStep body). Dead worlds are FROZEN: gained=0, no RNG draw, queue/state untouched.
// Live worlds draw next_nxt from the per-world RNG, then step. Returns gained.
// ============================================================================
GB_HD inline int fmBatchStep(GBWorld& w, int action_bin){
    if (SCAL(w,dead)) return 0;                          // frozen, like gpu_env
    int a=action_bin; if(a<0)a=0; else if(a>=FM_N_BINS)a=FM_N_BINS-1;
    unsigned long long s=SCAL(w,rng);
    int next_nxt = fm_rand_tier(s);                      // RNG advances ONLY for live worlds
    SCAL(w,rng)=s;
    return fmGameStep(w, a, next_nxt);
}
