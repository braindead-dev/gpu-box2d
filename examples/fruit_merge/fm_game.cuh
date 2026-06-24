// fm_game.cuh. The fruit-merge game layer for the gpu-box2d core. This is the
// flagship example: a complete game built on top of the physics engine without
// touching the contact solver or the island internals. The game talks to the
// physics through three seams only:
//
//   1. body create / destroy (slot activate / deactivate) for spawning and merging,
//   2. a ContactListener hook (begin/end touch) that mirrors Box2D's
//      b2ContactListener, where the merge rule lives,
//   3. the frozen WorldShared accessors (BODY/CONT/EDGE/SCAL) for reading state.
//
// The physics core never knows what a fruit is. The game sets a body's radius and
// its own tier index at creation, listens for same-tier touches, and replaces two
// touching fruits with one of the next tier. This separation is the point: any game
// that needs faithful 2D rigid-body physics replaces this file and keeps the core.
//
// STATUS: this header shows the integration shape (the hook, the merge rule, the
// spawn and death logic) against the frozen contract. The end-to-end batched
// launcher is assembled once the narrow-phase, solver, and island modules validate
// (see the repository roadmap). The merge ordering and RNG notes below are the
// determinism rules the batched build reproduces bit-for-bit.
#pragma once
#include "gpu_box2d/gb_pools.cuh"

// ----------------------------------------------------------------------------
// Game constants. These belong to the fruit-merge game, not to the physics core.
// ----------------------------------------------------------------------------
#define FM_N_TIERS        12
#define FM_N_BINS         16          // action bins across the play width
#define FM_N_SPAWNABLE    5           // the queue draws tiers from {0..4}
#define FM_WALL_X         3.75f       // play half-width
#define FM_CONTAINER_H    9.5f        // container height (top of the side walls)
#define FM_DEAD_Y         7.75f       // a fruit resting above this for DEATH_TIME ends the game
#define FM_DEATH_TIME     4.0f        // seconds above the death line before the game ends
#define FM_FRUIT_DENSITY  1.0f        // b2CircleShape density for mass computation

// Radius by tier. The game sets BODY(w,radius,s) from this at body creation; the
// physics core reads the radius back through gbCircleRadius and never sees a tier.
GB_HD inline float fmTierRadius(int tier){
    const float R[FM_N_TIERS] = {0.25f,0.28f,0.5f,0.525f,0.66f,0.84f,0.975f,
                                 1.2f,1.32f,1.65f,1.95f,2.2f};
    return R[tier];
}

// Score gained when a pair merges INTO this tier (index is the resulting tier).
GB_HD inline int fmFruitScore(int tier){
    const int S[FM_N_TIERS] = {1,3,7,9,13,21,27,34,44,62,90,0};
    return S[tier];
}

// ----------------------------------------------------------------------------
// Deterministic RNG. splitmix64, matching the CPU batch engine bit-for-bit so a
// seeded game reproduces its spawn sequence exactly on CPU and GPU.
// ----------------------------------------------------------------------------
GB_HD inline unsigned long long fmSplitmix64(unsigned long long& s){
    unsigned long long z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
GB_HD inline int fmRandTier(unsigned long long& s){
    return (int)(fmSplitmix64(s) % (unsigned long long)FM_N_SPAWNABLE);
}

// ----------------------------------------------------------------------------
// Body create / destroy. The game owns these; the physics core treats a "create"
// as activating a body slot and a "destroy" as deactivating it. The merge replaces
// two bodies with one, which is two destroys and one create.
// ----------------------------------------------------------------------------

// Create a circle fruit body in a fresh slot. Sets the radius and tier (game
// fields), the mass from the circle shape, and the initial kinematic state.
GB_HD inline int fmAddFruit(GBWorld& w, int tier, float x, float y, float vy){
    int s = SCAL(w, bodyCount)++;
    float r = fmTierRadius(tier);
    float mass = FM_FRUIT_DENSITY * GB_PI * r * r;   // b2CircleShape::ComputeMass
    float I    = mass * (0.5f * r * r);
    BODY(w, bodyType, s) = GB_DYNAMIC_BODY;
    BODY(w, tier, s)     = tier;
    BODY(w, radius, s)   = r;
    BODY(w, invMass, s)  = mass > 0.0f ? 1.0f / mass : 0.0f;
    BODY(w, invI, s)     = I    > 0.0f ? 1.0f / I    : 0.0f;
    BODY(w, alive, s) = 1; BODY(w, awake, s) = 1; BODY(w, sleepTime, s) = 0.0f;
    BODY(w, sweepCx, s) = x;  BODY(w, sweepCy, s) = y;  BODY(w, sweepA, s) = 0.0f;
    BODY(w, sweepC0x, s) = x; BODY(w, sweepC0y, s) = y; BODY(w, sweepA0, s) = 0.0f;
    BODY(w, sweepAlpha0, s) = 0.0f;
    BODY(w, velX, s) = 0.0f; BODY(w, velY, s) = vy; BODY(w, angVel, s) = 0.0f;
    BODY(w, age, s) = 0.0f; BODY(w, outline, s) = 0.0f;
    return s;
}

// Deactivate a fruit body slot. The physics core skips dead slots in every phase.
GB_HD inline void fmDestroyFruit(GBWorld& w, int body){
    BODY(w, alive, body) = 0;
    BODY(w, awake, body) = 0;
}

// ----------------------------------------------------------------------------
// The ContactListener hook. The physics core calls fmBeginContact when two
// fixtures start touching and fmEndContact when they stop, mirroring Box2D's
// b2ContactListener::BeginContact / EndContact. The game records same-tier pairs
// here in INSERTION ORDER; it does not act on them yet. Acting during the contact
// callback would mutate the body set mid-solve, so merges run after the step.
//
// Merge-pair ordering is load-bearing for determinism: the pair list is keyed by
// (min,max) body slot, a new key appends, an existing key updates in place, and an
// end-touch erases. The post-step merge pass iterates this list in insertion order.
// ----------------------------------------------------------------------------
GB_HD inline int fmMergeFind(GBWorld& w, int lo, int hi){
    for (int i = 0; i < SCAL(w, pairCount); ++i)
        if (BODY(w, pairLo, i) == lo && BODY(w, pairHi, i) == hi) return i;
    return -1;
}

GB_HD inline void fmBeginContact(GBWorld& w, int bodyA, int bodyB){
    if (!BODY(w, alive, bodyA) || !BODY(w, alive, bodyB)) return;
    if (BODY(w, tier, bodyA) != BODY(w, tier, bodyB)) return;   // only same-tier merges
    int lo = bodyA < bodyB ? bodyA : bodyB;
    int hi = bodyA < bodyB ? bodyB : bodyA;
    if (fmMergeFind(w, lo, hi) >= 0) return;                    // already recorded
    int i = SCAL(w, pairCount)++;
    BODY(w, pairLo, i) = lo; BODY(w, pairHi, i) = hi;
    BODY(w, pairA, i)  = bodyA; BODY(w, pairB, i) = bodyB;
}

GB_HD inline void fmEndContact(GBWorld& w, int bodyA, int bodyB){
    int lo = bodyA < bodyB ? bodyA : bodyB;
    int hi = bodyA < bodyB ? bodyB : bodyA;
    int idx = fmMergeFind(w, lo, hi);
    if (idx < 0) return;
    for (int k = idx + 1; k < SCAL(w, pairCount); ++k){
        BODY(w, pairLo, k-1) = BODY(w, pairLo, k); BODY(w, pairHi, k-1) = BODY(w, pairHi, k);
        BODY(w, pairA, k-1)  = BODY(w, pairA, k);  BODY(w, pairB, k-1)  = BODY(w, pairB, k);
    }
    SCAL(w, pairCount)--;
}

// ----------------------------------------------------------------------------
// The post-step merge pass. Runs after gb_world_step settles the world. Iterates
// the recorded same-tier pairs in insertion order, and for each still-live pair
// replaces the two fruits with one of the next tier at their midpoint, scoring the
// new tier. A fruit that already merged this pass is skipped (its slot is dead).
// ----------------------------------------------------------------------------
GB_HD inline int fmProcessMerges(GBWorld& w){
    int gained = 0;
    int n = SCAL(w, pairCount);
    for (int i = 0; i < n; ++i){
        int a = BODY(w, pairA, i), b = BODY(w, pairB, i);
        if (!BODY(w, alive, a) || !BODY(w, alive, b)) continue;     // consumed already
        if (BODY(w, tier, a) != BODY(w, tier, b)) continue;
        int tier = BODY(w, tier, a);
        if (tier + 1 >= FM_N_TIERS) continue;                       // top tier does not merge
        float mx = 0.5f * (BODY(w, sweepCx, a) + BODY(w, sweepCx, b));
        float my = 0.5f * (BODY(w, sweepCy, a) + BODY(w, sweepCy, b));
        fmDestroyFruit(w, a);
        fmDestroyFruit(w, b);
        int ns = fmAddFruit(w, tier + 1, mx, my, 0.0f);
        gained += fmFruitScore(tier + 1);
        if (tier + 1 > SCAL(w, maxTier)) SCAL(w, maxTier) = tier + 1;
        (void)ns;   // the batched build also updates gold-apple counts here
    }
    SCAL(w, pairCount) = 0;
    SCAL(w, score) += gained;
    return gained;
}
