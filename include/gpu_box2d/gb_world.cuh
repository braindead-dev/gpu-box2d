// gb_world.cuh. The world step declaration plus both execution shells. It defines how
// a kernel runs one Box2D world per launch element, behind the gb_pools accessor
// contract.
//
// TWO EXECUTION MODELS, both validated, both bit-identical to the CPU:
//
//   Thread-per-world + SoA-global (-DGB_SOA_GLOBAL, the production default). One CUDA
//   thread runs one world's full step. State lives in transposed global arrays indexed
//   slot*NW+world, so a warp's 32 lanes (consecutive worlds) read 32 consecutive
//   addresses (coalesced). This is the measured high-throughput path. The reason it
//   wins is structural: Box2D's contact solver is a serial Gauss-Seidel sweep that
//   dominates the step and cannot parallelize without changing the floats. With one
//   thread per world, every lane does useful serial work. (See docs/performance.md:
//   about 23K env-steps/s on an A10, against about 2.7K for the block-per-world shell
//   below, which idles most of a block's lanes on the same serial solver.)
//
//   Block-per-world + shared memory (default when GB_SOA_GLOBAL is not set). One CUDA
//   block runs one world, with the per-world state resident in shared memory. The
//   block loads the world global->shared, runs the step against the shared copy, and
//   stores it back. This path is kept as a documented alternative.
//
// FIDELITY: in both models the serial spine reads and writes the world state in
// Box2D's exact order, so the floats match the CPU bit-for-bit. The 3 serial folds
// (minSeparation, minSleepTime, the in-place velocity accumulation) stay serial. See
// docs/architecture.md.
#pragma once
#include "gb_pools.cuh"

// ============================================================================
// THE STEP (declaration). gb_world_step runs one b2World::Step on a world handle.
// The signature is frozen; the internal phases are the modules. GBWorld is the
// per-world handle the active backend provides (a SoA handle under GB_SOA_GLOBAL,
// the shared-resident WorldShared otherwise). gb_world_step is defined by the
// assembled core (gb_step.cuh). It is declared here so the launcher and example
// can call it.
//
//   void gb_world_step(GBWorld& w);                  // collide -> solve -> TOI
// ============================================================================
GB_HD void gb_world_step(GBWorld& w);   // defined by the assembled physics core

#ifdef GB_SOA_GLOBAL
// ============================================================================
// Production path: thread-per-world on the SoA-global backend. One thread steps one
// world. The handle {pools, world} costs nothing; the accessor macros do the
// transposed indexing. The launcher allocates the WorldPoolsSoA arrays.
// ============================================================================
#ifdef __CUDACC__
__global__ inline void gb_kThreadStep(WorldPoolsSoA pools){
    int world = blockIdx.x * blockDim.x + threadIdx.x;
    if (world >= pools.NW) return;
    GBWorld w; w.p = &pools; w.world = world;
    gb_world_step(w);
}
// One thread per world. 256 threads per block is a good default; tune per GPU.
inline void gb_launch_thread_step(WorldPoolsSoA pools, int threadsPerBlock = 256){
    int blocks = (pools.NW + threadsPerBlock - 1) / threadsPerBlock;
    gb_kThreadStep<<<blocks, threadsPerBlock>>>(pools);
}
#endif // __CUDACC__

#else // !GB_SOA_GLOBAL
// ============================================================================
// Alternative path: block-per-world with the per-world state in shared memory.
//   grid:  one block per world (blockIdx.x = world id)
//   block: GB_BLOCK_THREADS lanes (sized to max(bodies, contacts))
//   shared: one WorldShared per block plus the island solve scratch
//   flow:
//     1. cooperatively load pools.world[blockIdx.x] global -> shared
//     2. __syncthreads()
//     3. run the step (serial spine on lane 0, parallel phases across the block)
//     4. cooperatively store shared -> global
// ============================================================================

// Block size. Must be >= max(MAX_BODIES, typical contactCount) so the parallel phases
// cover all elements in one pass; otherwise they loop with stride.
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

// ---- the block-per-world step kernel (the shell) ------------------------------
// One block per world: load shared, step, store. The block-parallel phases live
// inside gb_world_step and use threadIdx.x; the shell itself is stable.
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
// launch helper: shared bytes = sizeof(WorldShared) (plus island scratch when the
// block-parallel phases are enabled)
inline void gb_launch_block_step(WorldPools pools){
    int shmem = (int)sizeof(WorldShared);
    gb_kBlockStep<<<pools.NW, GB_BLOCK_THREADS, shmem>>>(pools);
}
#endif // __CUDACC__

#endif // GB_SOA_GLOBAL
