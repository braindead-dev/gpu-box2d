// fm_engine.cuh. The full fruit-merge engine: the physics core wired to the game
// layer. This is the example app's top-level header. It composes the general
// gpu-box2d physics core (gb_step.cuh) with the fruit-merge game layer (fm_game.cuh
// for merge, spawn, death, and score, plus fm_obs.cuh for observations) through one
// seam: the contact touch hook (gbOnTouchBegin / gbOnTouchEnd), which the game
// overrides to drive its insertion-ordered merge-pair store.
//
// Include order is load-bearing:
//   1. forward-declare the game's merge listener (defined in fm_game.cuh) so it is
//      visible before the hook.
//   2. define GB_GAME_TOUCH_HOOKS and the gbOnTouchBegin / gbOnTouchEnd hooks to call
//      the merge listener, before gb_step.cuh pulls in the narrow-phase, so the
//      narrow-phase uses these hooks in place of its no-op stubs. The hooks add zero
//      float ops to the narrow-phase, so the 0-ULP manifold result holds.
//   3. gb_step.cuh, the assembled physics core (defines gb_world_step).
//   4. fm_game.cuh and fm_obs.cuh, the game step (fmGameStep drives gb_world_step).
#pragma once
#include "gpu_box2d/gb_pools.cuh"
#include "gpu_box2d/gb_contact_types.cuh"
#include "gpu_box2d/gb_math.cuh"

// ---- 1) forward-declare the game's merge listener (defined in fm_game.cuh) ----
GB_HD inline void fmMergeBeginContact(GBWorld& w, int bodyA, int bodyB);
GB_HD inline void fmMergeEndContact(GBWorld& w, int bodyA, int bodyB);

// ---- 2) wire the touch-hook seam: the core's hooks call the merge listener ----
// (declared before gb_step.cuh so the narrow-phase sees GB_GAME_TOUCH_HOOKS.)
#define GB_GAME_TOUCH_HOOKS 1
GB_HD inline void gbOnTouchBegin(GBWorld& w, int a, int b){ fmMergeBeginContact(w, a, b); }
GB_HD inline void gbOnTouchEnd  (GBWorld& w, int a, int b){ fmMergeEndContact(w, a, b); }

// ---- 3) the assembled physics core (gb_world_step = collide -> solve -> TOI) ----
#include "gpu_box2d/gb_step.cuh"

// ---- 4) the game layer (fmGameStep drives gb_world_step + merge/spawn/death/obs) ----
#include "fm_game.cuh"
#include "fm_obs.cuh"
