// gb_batch_cuda.cuh. The CUDA device path for the batched-world driver. It uploads the
// seeded WorldShared array to the transposed SoA-global layout, launches the
// thread-per-world step kernel, and downloads the result, so the same Batch API a user
// drives on a CPU steps on the GPU. The seeding API (gbBatchAddCircle and the rest in
// gb_batch.cuh) is unchanged; only the step dispatch differs.
//
// This header is compiled by nvcc with -DGB_SOA_GLOBAL (the production backend). It is a
// separate translation unit from the host driver, because the SoA-global and the block
// backends select different accessor macros and cannot coexist in one compile. The host
// build (gb_batch.cuh on its own) keeps working with no GPU; this header adds the device
// path when a CUDA toolchain is present.
//
// The transpose is driven by a single field list (GB_BATCH_SOA_FIELDS), so the SoA
// allocation and the up and down copies stay in sync with the field set. The same field
// list is exercised host-mode by test/gb_batch_transpose_test.cu, which round-trips a
// WorldShared through the flat SoA layout and asserts identity, so the transpose is
// validated without a GPU.
//
// Build flags (FROZEN): nvcc --fmad=false -prec-div=true -prec-sqrt=true -DGB_SOA_GLOBAL.
#pragma once

// The transposed index used by the SoA-global backend: slot*NW + world.
#ifndef GB_SOA_INDEX
#define GB_SOA_INDEX(slot, world, NW) ((slot) * (NW) + (world))
#endif

// The per-world field list. Each entry is X(field, elemType, perWorldLen), where
// perWorldLen is the number of elements of that field per world (a scalar is 1). This is
// the single source of truth the transpose iterates. It mirrors the WorldShared field set
// and its opt-in blocks, so it tracks the same build flags.
#define GB_BATCH_SOA_FIELDS(X) \
    X(sweepCx, float, GB_MAX_BODIES)  X(sweepCy, float, GB_MAX_BODIES) \
    X(sweepC0x, float, GB_MAX_BODIES) X(sweepC0y, float, GB_MAX_BODIES) \
    X(sweepA, float, GB_MAX_BODIES)   X(sweepA0, float, GB_MAX_BODIES) \
    X(sweepAlpha0, float, GB_MAX_BODIES) \
    X(xfPx, float, GB_MAX_BODIES) X(xfPy, float, GB_MAX_BODIES) \
    X(xfQs, float, GB_MAX_BODIES) X(xfQc, float, GB_MAX_BODIES) \
    X(velX, float, GB_MAX_BODIES) X(velY, float, GB_MAX_BODIES) X(angVel, float, GB_MAX_BODIES) \
    X(invMass, float, GB_MAX_BODIES) X(invI, float, GB_MAX_BODIES) \
    X(radius, float, GB_MAX_BODIES) \
    X(userData, int, GB_MAX_BODIES) X(bodyType, int, GB_MAX_BODIES) \
    X(sleepTime, float, GB_MAX_BODIES) \
    X(awake, unsigned char, GB_MAX_BODIES) X(alive, unsigned char, GB_MAX_BODIES) \
    GB_BATCH_SOA_FIELDS_POLY(X) \
    X(edgeAx, float, GB_N_EDGES) X(edgeAy, float, GB_N_EDGES) \
    X(edgeBx, float, GB_N_EDGES) X(edgeBy, float, GB_N_EDGES) \
    GB_BATCH_SOA_FIELDS_CHAIN(X) \
    X(cBodyA, int, GB_MAX_CONTACTS) X(cBodyB, int, GB_MAX_CONTACTS) X(cEdge, int, GB_MAX_CONTACTS) \
    X(cTouching, unsigned char, GB_MAX_CONTACTS) \
    X(cFriction, float, GB_MAX_CONTACTS) X(cRestitution, float, GB_MAX_CONTACTS) \
    X(cManifoldType, int, GB_MAX_CONTACTS) \
    X(cLocalNormalX, float, GB_MAX_CONTACTS) X(cLocalNormalY, float, GB_MAX_CONTACTS) \
    X(cLocalPointX, float, GB_MAX_CONTACTS) X(cLocalPointY, float, GB_MAX_CONTACTS) \
    X(cPointLocalX, float, GB_MAX_CONTACTS) X(cPointLocalY, float, GB_MAX_CONTACTS) \
    X(cNormalImpulse, float, GB_MAX_CONTACTS) X(cTangentImpulse, float, GB_MAX_CONTACTS) \
    X(cPointCount, int, GB_MAX_CONTACTS) \
    X(cPointLocal2X, float, GB_MAX_CONTACTS) X(cPointLocal2Y, float, GB_MAX_CONTACTS) \
    X(cNormalImpulse2, float, GB_MAX_CONTACTS) X(cTangentImpulse2, float, GB_MAX_CONTACTS) \
    X(cToi, float, GB_MAX_CONTACTS) X(cToiCount, int, GB_MAX_CONTACTS) \
    X(cToiFlag, unsigned char, GB_MAX_CONTACTS) X(cEnabled, unsigned char, GB_MAX_CONTACTS) \
    GB_BATCH_SOA_FIELDS_JOINT(X)

// Scalar fields (one element per world) live outside the array field list.
#define GB_BATCH_SOA_SCALARS(X) \
    X(bodyCount, int) X(contactCount, int) X(stepComplete, unsigned char) \
    GB_BATCH_SOA_SCALARS_JOINT(X)

#ifdef GB_ENABLE_POLYGONS
#define GB_BATCH_SOA_FIELDS_POLY(X) \
    X(shapeType, int, GB_MAX_BODIES) X(polyCount, int, GB_MAX_BODIES) \
    X(polyRadius, float, GB_MAX_BODIES) \
    X(polyCentroidX, float, GB_MAX_BODIES) X(polyCentroidY, float, GB_MAX_BODIES) \
    X(polyVx, float, GB_MAX_BODIES*GB_MAX_POLYGON_VERTICES) X(polyVy, float, GB_MAX_BODIES*GB_MAX_POLYGON_VERTICES) \
    X(polyNx, float, GB_MAX_BODIES*GB_MAX_POLYGON_VERTICES) X(polyNy, float, GB_MAX_BODIES*GB_MAX_POLYGON_VERTICES)
#else
#define GB_BATCH_SOA_FIELDS_POLY(X)
#endif

#ifdef GB_ENABLE_CHAIN
#define GB_BATCH_SOA_FIELDS_CHAIN(X) \
    X(edgeV0x, float, GB_N_EDGES) X(edgeV0y, float, GB_N_EDGES) \
    X(edgeV3x, float, GB_N_EDGES) X(edgeV3y, float, GB_N_EDGES) \
    X(edgeHasV0, unsigned char, GB_N_EDGES) X(edgeHasV3, unsigned char, GB_N_EDGES)
#else
#define GB_BATCH_SOA_FIELDS_CHAIN(X)
#endif

#ifdef GB_ENABLE_JOINTS
#define GB_BATCH_SOA_FIELDS_JOINT(X) \
    X(jBodyA, int, GB_MAX_JOINTS) X(jBodyB, int, GB_MAX_JOINTS) \
    X(jLocalAnchorAX, float, GB_MAX_JOINTS) X(jLocalAnchorAY, float, GB_MAX_JOINTS) \
    X(jLocalAnchorBX, float, GB_MAX_JOINTS) X(jLocalAnchorBY, float, GB_MAX_JOINTS) \
    X(jImpulseX, float, GB_MAX_JOINTS) X(jImpulseY, float, GB_MAX_JOINTS)
#define GB_BATCH_SOA_SCALARS_JOINT(X) X(jointCount, int)
#else
#define GB_BATCH_SOA_FIELDS_JOINT(X)
#define GB_BATCH_SOA_SCALARS_JOINT(X)
#endif

// ---------------------------------------------------------------------------
// Host-side transpose between a contiguous WorldShared array and the flat SoA arrays.
// These are pure functions over CPU pointers, so they are validated host-mode by
// test/gb_batch_transpose_test.cu. The device path (below) calls them on staging buffers
// before and after the kernel launch.
// ---------------------------------------------------------------------------

// Copy field `src[world].field[i]` to `dst[i*NW + world]` for every world and element.
#define GB_TRANSPOSE_UP_ONE(field, type, len) \
    do { for (int wi = 0; wi < NW; ++wi) for (int i = 0; i < (int)(len); ++i) \
        soa.field[GB_SOA_INDEX(i, wi, NW)] = host[wi].field[i]; } while (0);
#define GB_TRANSPOSE_DOWN_ONE(field, type, len) \
    do { for (int wi = 0; wi < NW; ++wi) for (int i = 0; i < (int)(len); ++i) \
        host[wi].field[i] = soa.field[GB_SOA_INDEX(i, wi, NW)]; } while (0);
#define GB_TRANSPOSE_UP_SCALAR(field, type) \
    do { for (int wi = 0; wi < NW; ++wi) soa.field[wi] = host[wi].field; } while (0);
#define GB_TRANSPOSE_DOWN_SCALAR(field, type) \
    do { for (int wi = 0; wi < NW; ++wi) host[wi].field = soa.field[wi]; } while (0);

// soa here is a WorldPoolsSoA whose pointers address host (CPU) staging buffers. The
// device path allocates these on the GPU and uses cudaMemcpy; the transpose itself runs
// host-side on the staging buffers.
inline void gbBatchTransposeUp(const WorldShared* host, WorldPoolsSoA& soa, int NW){
    GB_BATCH_SOA_FIELDS(GB_TRANSPOSE_UP_ONE)
    GB_BATCH_SOA_SCALARS(GB_TRANSPOSE_UP_SCALAR)
}
inline void gbBatchTransposeDown(WorldShared* host, const WorldPoolsSoA& soa, int NW){
    GB_BATCH_SOA_FIELDS(GB_TRANSPOSE_DOWN_ONE)
    GB_BATCH_SOA_SCALARS(GB_TRANSPOSE_DOWN_SCALAR)
}

#if defined(__CUDACC__) && defined(GB_SOA_GLOBAL)
// ---------------------------------------------------------------------------
// Device path. Allocate the SoA arrays on the GPU, transpose-up on a host staging copy,
// upload, launch the thread-per-world step, download, and transpose-down. Compiled only
// by nvcc with the SoA-global backend.
// ---------------------------------------------------------------------------
#include <cstdlib>
#include <cstring>

// Allocate every SoA array on the device and a matching host staging buffer.
struct GBBatchDevice {
    WorldPoolsSoA dev;     // device pointers
    WorldPoolsSoA stage;   // host staging pointers (same field names)
    int NW;
};

inline void gbBatchDeviceAlloc(GBBatchDevice& d, int NW){
    d.NW = NW;
    d.dev.NW = NW; d.stage.NW = NW;
#define GB_ALLOC_ONE(field, type, len) \
    do { size_t n = (size_t)(len) * NW; \
         cudaMalloc((void**)&d.dev.field, n*sizeof(type)); \
         d.stage.field = (type*)malloc(n*sizeof(type)); } while (0);
#define GB_ALLOC_SCALAR(field, type) \
    do { size_t n = (size_t)NW; \
         cudaMalloc((void**)&d.dev.field, n*sizeof(type)); \
         d.stage.field = (type*)malloc(n*sizeof(type)); } while (0);
    GB_BATCH_SOA_FIELDS(GB_ALLOC_ONE)
    GB_BATCH_SOA_SCALARS(GB_ALLOC_SCALAR)
#undef GB_ALLOC_ONE
#undef GB_ALLOC_SCALAR
}

inline void gbBatchDeviceFree(GBBatchDevice& d){
#define GB_FREE_ONE(field, type, len) do { cudaFree(d.dev.field); free(d.stage.field); } while (0);
#define GB_FREE_SCALAR(field, type)   do { cudaFree(d.dev.field); free(d.stage.field); } while (0);
    GB_BATCH_SOA_FIELDS(GB_FREE_ONE)
    GB_BATCH_SOA_SCALARS(GB_FREE_SCALAR)
#undef GB_FREE_ONE
#undef GB_FREE_SCALAR
}

// Copy the staging buffers host -> device (after transpose-up) or device -> host (before
// transpose-down).
inline void gbBatchDeviceUpload(GBBatchDevice& d){
    int NW = d.NW;
#define GB_UP_ONE(field, type, len) cudaMemcpy(d.dev.field, d.stage.field, (size_t)(len)*NW*sizeof(type), cudaMemcpyHostToDevice);
#define GB_UP_SCALAR(field, type)   cudaMemcpy(d.dev.field, d.stage.field, (size_t)NW*sizeof(type), cudaMemcpyHostToDevice);
    GB_BATCH_SOA_FIELDS(GB_UP_ONE)
    GB_BATCH_SOA_SCALARS(GB_UP_SCALAR)
#undef GB_UP_ONE
#undef GB_UP_SCALAR
}
inline void gbBatchDeviceDownload(GBBatchDevice& d){
    int NW = d.NW;
#define GB_DOWN_ONE(field, type, len) cudaMemcpy(d.stage.field, d.dev.field, (size_t)(len)*NW*sizeof(type), cudaMemcpyDeviceToHost);
#define GB_DOWN_SCALAR(field, type)   cudaMemcpy(d.stage.field, d.dev.field, (size_t)NW*sizeof(type), cudaMemcpyDeviceToHost);
    GB_BATCH_SOA_FIELDS(GB_DOWN_ONE)
    GB_BATCH_SOA_SCALARS(GB_DOWN_SCALAR)
#undef GB_DOWN_ONE
#undef GB_DOWN_SCALAR
}

// Step the batch on the GPU. Transpose the host worlds into the staging buffers, upload,
// launch the thread-per-world kernel `substeps` times, download, and transpose back. The
// per-world result is the same physics the host driver produces, since the SoA-global
// backend is bit-identical to the block backend (see docs/fidelity.md).
inline void gbBatchStepDevice(GBBatchDevice& d, WorldShared* host, int substeps){
    gbBatchTransposeUp(host, d.stage, d.NW);
    gbBatchDeviceUpload(d);
    for (int s = 0; s < substeps; ++s) gb_launch_thread_step(d.dev);
    cudaDeviceSynchronize();
    gbBatchDeviceDownload(d);
    gbBatchTransposeDown(host, d.stage, d.NW);
}
#endif // __CUDACC__ && GB_SOA_GLOBAL
