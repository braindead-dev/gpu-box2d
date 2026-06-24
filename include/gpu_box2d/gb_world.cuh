// gb_world.cuh. The world step plus the block-per-world execution shell. This
// defines how a kernel runs one Box2D world per CUDA block with the per-world state
// resident in shared memory. The physics phases (broad-phase, narrow-phase,
// integrate, solve, sleep, merge) plug into this shell and the gb_pools accessor
// contract.
//
// STATUS: the block-per-world execution model is validated. A world's state loads
// global -> shared at block start, the serial step runs on lane 0 reading shared,
// and the result stores shared -> global. This path is 0-ULP on the single-world
// and two-body scenarios (see test/gb_island_test.cu). The block-parallel phases
// and the multi-world step assembly are in active development.
//
// EXECUTION MODEL:
//   grid:  one block per world (blockIdx.x = world id)
//   block: GB_BLOCK_THREADS lanes (sized to max(bodies, contacts))
//   shared: one WorldShared per block plus the island solve scratch
//   flow:
//     1. cooperatively load pools.world[blockIdx.x] -> shared (all lanes, strided)
//     2. __syncthreads()
//     3. run the step:
//          - parallel phases: each lane handles a body or a contact
//          - serial spine (lane 0): DFS island assembly, WarmStart, 8 velocity
//            iters, 3 position iters, the 3 min-folds (minSeparation, minSleepTime,
//            and the in-place velocity accumulation), and the merge pass. These
//            stay serial-in-order (see docs/architecture.md).
//          - __syncthreads() between phase groups
//     4. cooperatively store shared -> pools.world[blockIdx.x]
//
// FIDELITY: the serial spine reads and writes the shared arena in Box2D's exact
// order, so the floats match the CPU bit-for-bit. The parallel phases touch only
// disjoint per-body and per-contact slots. The 3 serial folds stay on lane 0.
#pragma once
#include "gb_pools.cuh"

// Block size. Must be >= max(MAX_BODIES, typical contactCount) so the parallel
// phases cover all elements in one pass; otherwise they loop with stride.
#ifndef GB_BLOCK_THREADS
#define GB_BLOCK_THREADS 128
#endif

// ---- cooperative global<->shared transfer of one WorldShared (all lanes) -------
// WorldShared is a POD; copy it word-by-word strided across the block's threads.
GB_HD inline void gbLoadShared(WorldShared& dst, const WorldShared& src){
#ifdef __CUDA_ARCH__
    const int* s = reinterpret_cast<const int*>(&src);
    int*       d = reinterpret_cast<int*>(&dst);
    const int  n = (int)(sizeof(WorldShared)/sizeof(int));
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
#else
    dst = src;
#endif
}
GB_HD inline void gbStoreShared(WorldShared& dst, const WorldShared& src){
    gbLoadShared(dst, src);   // symmetric word copy
}

// ============================================================================
// THE STEP (declaration). gb_world_step runs one b2World::Step on a shared-resident
// world. It first runs as the serial step on lane 0; phases then move to
// block-parallel inside the module headers. The signature is frozen: phases are
// internal.
//
//   void gb_world_step(GBWorld& w);                  // collide -> solve -> TOI
//
// In the default (block/shared) build, GBWorld == WorldShared (the shared
// instance). gb_world_step is defined by the assembled core (gb_step.cuh). It is
// declared here so the launcher and the example can call it.
// ============================================================================
GB_HD void gb_world_step(GBWorld& w);   // defined by the assembled physics core

// ---- the block-per-world step kernel (the shell) ------------------------------
// One block per world: load shared, step, store. The example launcher uses this
// or a settle-loop variant. The block-parallel phases live inside gb_world_step
// and use threadIdx.x; the shell itself is stable.
#ifdef __CUDACC__
__global__ inline void gb_kBlockStep(WorldPools pools){
    extern __shared__ unsigned char gb_smem[];
    WorldShared& w = *reinterpret_cast<WorldShared*>(gb_smem);
    int world = blockIdx.x;
    if (world >= pools.NW) return;
    gbLoadShared(w, pools.world[world]);
    __syncthreads();
    gb_world_step(w);                       // serial spine on lane 0 + parallel phases
    __syncthreads();
    gbStoreShared(pools.world[world], w);
}
// launch helper: shared bytes = sizeof(WorldShared) (plus island scratch once the
// parallel phases land)
inline void gb_launch_block_step(WorldPools pools){
    int shmem = (int)sizeof(WorldShared);
    gb_kBlockStep<<<pools.NW, GB_BLOCK_THREADS, shmem>>>(pools);
}
#endif
