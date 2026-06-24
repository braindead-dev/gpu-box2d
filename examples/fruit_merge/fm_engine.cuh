// fm_engine.cuh. The full fruit-merge engine: the physics core wired to the game
// layer. This is the example app's top-level header. It composes the general
// gpu-box2d physics core (gb_step.cuh) with the fruit-merge game layer (fm_game.cuh
// for merge, spawn, death, and score, plus fm_obs.cuh for observations) through two
// seams: the application field hooks (fm_fields.cuh adds the game's own per-world
// state to the core world struct) and the contact listener hook
// (gbOnTouchBegin / gbOnTouchEnd), which the game overrides to drive its
// insertion-ordered merge-pair store.
//
// Include order is load-bearing:
//   1. fm_fields.cuh defines GB_WORLD_USER_FIELDS / GB_WORLD_SOA_USER_FIELDS, which
//      the core world struct picks up. Include it before any core header.
//   2. forward-declare the game's merge listener (defined in fm_game.cuh) so it is
//      visible before the hook.
//   3. define GB_CONTACT_LISTENER_HOOKS and the gbOnTouchBegin / gbOnTouchEnd hooks
//      to call the merge listener, before gb_step.cuh pulls in the narrow-phase, so
//      the narrow-phase uses these hooks in place of its no-op stubs. The hooks add
//      zero float ops to the narrow-phase, so the 0-ULP manifold result holds.
//   4. gb_step.cuh, the assembled physics core (defines gb_world_step).
//   5. fm_game.cuh and fm_obs.cuh, the game step (fmGameStep drives gb_world_step).
#pragma once

// ---- 1) inject the game's per-world fields into the core world state ----------
#include "fm_fields.cuh"

#include "gpu_box2d/gb_pools.cuh"
#include "gpu_box2d/gb_contact_types.cuh"
#include "gpu_box2d/gb_math.cuh"

// ---- 2) forward-declare the game's merge listener (defined in fm_game.cuh) ----
GB_HD inline void fmMergeBeginContact(GBWorld& w, int bodyA, int bodyB);
GB_HD inline void fmMergeEndContact(GBWorld& w, int bodyA, int bodyB);

// ---- 3) wire the contact listener seam: the core's hooks call the merge listener.
// (declared before gb_step.cuh so the narrow-phase sees GB_CONTACT_LISTENER_HOOKS.)
#define GB_CONTACT_LISTENER_HOOKS 1
GB_HD inline void gbOnTouchBegin(GBWorld& w, int a, int b){ fmMergeBeginContact(w, a, b); }
GB_HD inline void gbOnTouchEnd  (GBWorld& w, int a, int b){ fmMergeEndContact(w, a, b); }

// ---- 4) the assembled physics core (gb_world_step = collide -> solve -> TOI) ----
#include "gpu_box2d/gb_step.cuh"

// ---- 5) the game layer (fmGameStep drives gb_world_step + merge/spawn/death/obs) ----
#include "fm_game.cuh"
#include "fm_obs.cuh"
