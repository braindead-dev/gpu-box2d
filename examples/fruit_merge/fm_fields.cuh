// fm_fields.cuh. The fruit-merge per-world fields, injected into the core world
// state through the application extension hooks (gb_pools.cuh ::
// GB_WORLD_USER_FIELDS and gb_soa_backend.cuh :: GB_WORLD_SOA_USER_FIELDS).
//
// Include this BEFORE the core (gb_pools.cuh / gb_step.cuh). It adds the game's
// own per-body and per-world fields to WorldShared and to the transposed SoA
// mirror, so the game layer reaches them through the same BODY / CONT / SCAL
// accessors the physics uses. The physics core never reads any of these fields.
#pragma once

// The merge-pair pool capacity (a game concept; the core does not define it).
#ifndef GB_MAX_PAIRS
#define GB_MAX_PAIRS 48
#endif

// Fields added to the contiguous WorldShared POD.
//   tier / age / outline : per-body game state (indexed by body slot).
//   pair* / pairCount    : the insertion-ordered merge-pair store filled by the
//                          contact listener and consumed after the step.
//   score .. rng         : per-world game scalars.
#define GB_WORLD_USER_FIELDS                                                    \
    int   tier[GB_MAX_BODIES];                                                  \
    float age[GB_MAX_BODIES], outline[GB_MAX_BODIES];                          \
    int   pairLo[GB_MAX_PAIRS], pairHi[GB_MAX_PAIRS];                          \
    int   pairA[GB_MAX_PAIRS], pairB[GB_MAX_PAIRS];                            \
    int   pairCount;                                                           \
    int   score, drops, maxTier, goldapples, cur, nxt;                        \
    unsigned char dead;                                                        \
    unsigned long long rng;

// The transposed SoA mirror. Each name matches a WorldShared field above, so the
// BODY / CONT / SCAL accessors resolve them under the SoA-global backend. Body
// and pair fields are length MAX*NW; scalars are length NW.
#define GB_WORLD_SOA_USER_FIELDS                                                \
    int* tier;                                                                  \
    float* age; float* outline;                                                \
    int* pairLo; int* pairHi; int* pairA; int* pairB;                          \
    int* pairCount;                                                            \
    int* score; int* drops; int* maxTier; int* goldapples; int* cur; int* nxt; \
    unsigned char* dead;                                                       \
    unsigned long long* rng;
