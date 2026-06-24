# Fruit merge example

This is the flagship example for gpu-box2d: a complete physics-driven game built
on the engine core with the game logic fully separated from the physics.

## What it shows

A fruit-merge game (drop a circle into a container, equal-tier circles that touch
combine into the next tier, the game ends when fruit stacks above a death line) is
a clean test of a 2D rigid-body engine. It needs circles, static walls, stacking,
settling, sleep, and continuous collision for fast drops. Every one of those is
physics. The merge rule, the score, the spawn queue, and the death check are game.

`fm_game.cuh` keeps that split strict. The game touches the engine through three
seams and nothing else:

1. **Body create and destroy.** `fmAddFruit` activates a body slot and sets its
   radius and tier; `fmDestroyFruit` deactivates one. A merge is two destroys and
   one create.
2. **A ContactListener hook.** `fmMergeBeginContact` and `fmMergeEndContact` mirror
   Box2D's `b2ContactListener`. The assembled step fires them on touching transitions
   through the core's `gbOnTouchBegin` / `gbOnTouchEnd` seam, which `fm_engine.cuh`
   wires to the game. The merge rule lives here. It records same-tier touching pairs in
   insertion order and acts on them after the step, so the body set never changes
   mid-solve.
3. **The frozen accessors.** The game reads and writes world state through
   `BODY` / `CONT` / `EDGE` / `SCAL`, the same contract the physics modules use.

The physics core never learns what a fruit is. It sees circles with a radius and a
mass. Swap `fm_game.cuh` for your own game layer and the engine is unchanged.

## Determinism

The example reproduces a reference CPU implementation bit-for-bit when seeded. Two
rules carry that:

- **RNG.** `fm_sm64` is splitmix64 with the reference constants, so the spawn
  sequence matches across CPU and GPU for a given seed.
- **Merge order.** The merge-pair list is keyed by `(min,max)` body slot. A new key
  appends, an existing key updates in place, an end-touch erases, and the post-step
  pass iterates in insertion order. That fixed order is what makes the same seed
  produce the same game.

## Files

- `fm_game.cuh`: the game layer (constants, RNG, the merge hook, body create and
  destroy, the settle-and-merge env-step, batch reset and respawn).
- `fm_obs.cuh`: the observation encoding for reinforcement learning, byte-exact to the
  CPU reference.
- `fm_engine.cuh`: the top-level header that composes the physics core with the game
  layer through the touch hook. Include this to get the full fruit-merge engine.

## Status

Complete and validated. The game layer drives the assembled physics core, and the
end-to-end output (per-drop score, queue, death, and observations) is exact against the
CPU reference, with the batched score distribution matching at KS p=1.0. The game logic
is fully separated from the physics: the core sees circles with a radius and a mass and
never learns what a fruit is, so swapping `fm_game.cuh` for another game layer leaves
the engine unchanged.
