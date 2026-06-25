// gb_batch_transpose_test.cu. Host-mode validation of the SoA transpose that the CUDA
// batch path (bindings/gb_batch_cuda.cuh) uses to upload and download worlds. The
// transpose maps a contiguous WorldShared array to and from the transposed SoA layout
// (field[slot*NW + world]) that the GPU thread-per-world backend reads. This test
// round-trips a seeded batch through the flat SoA arrays and asserts the result is
// byte-identical to the input, so the transpose is validated without a GPU.
//
// The transpose is the one piece the CUDA path adds; the kernel it wraps is the same
// gb_world_step the host path runs, and the SoA backend is bit-identical to the block
// backend (docs/fidelity.md). Validating the transpose here means the device path moves
// the right bytes in and out.
//
// Build (host), with the same feature flags the binding uses:
//   clang++ -O2 -x c++ -DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS -DGB_ENABLE_CHAIN \
//           -Iinclude -Ibindings test/gb_batch_transpose_test.cu -o test/gb_batch_transpose_test
//   ./test/gb_batch_transpose_test
//   Expected: PASS gb_batch_transpose
#include "gpu_box2d/gb_pools.cuh"        // block backend: WorldShared + block accessors
#include "gpu_box2d/gb_soa_backend.cuh"  // the WorldPoolsSoA struct (macros gated off here)
#include "gb_batch_cuda.cuh"             // gbBatchTranspose{Up,Down} + the field list
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Allocate the flat SoA arrays as plain host buffers and free them, using the same field
// list the transpose iterates, so the test covers exactly the transposed fields.
static void allocSoA(WorldPoolsSoA& soa, int NW){
    soa.NW = NW;
#define A_ONE(field, type, len) soa.field = (type*)calloc((size_t)(len)*NW, sizeof(type));
#define A_SCALAR(field, type)   soa.field = (type*)calloc((size_t)NW, sizeof(type));
    GB_BATCH_SOA_FIELDS(A_ONE)
    GB_BATCH_SOA_SCALARS(A_SCALAR)
#undef A_ONE
#undef A_SCALAR
}
static void freeSoA(WorldPoolsSoA& soa){
#define F_ONE(field, type, len) free(soa.field);
#define F_SCALAR(field, type)   free(soa.field);
    GB_BATCH_SOA_FIELDS(F_ONE)
    GB_BATCH_SOA_SCALARS(F_SCALAR)
#undef F_ONE
#undef F_SCALAR
}

int main(){
    printf("Batch transpose round-trip test: WorldShared <-> SoA (field[slot*NW+world])\n\n");

    const int NW = 7;
    std::vector<WorldShared> worlds(NW), original(NW);

    // Fill each world with a distinct, deterministic pattern across every field, so a
    // mis-transposed field (wrong stride, wrong world) shows up as a mismatch.
    for (int wi = 0; wi < NW; ++wi){
        unsigned char* p = reinterpret_cast<unsigned char*>(&worlds[wi]);
        for (size_t b = 0; b < sizeof(WorldShared); ++b)
            p[b] = (unsigned char)((b * 31u + wi * 131u + 7u) & 0xff);
    }
    original = worlds;

    WorldPoolsSoA soa; allocSoA(soa, NW);

    // up: WorldShared array -> flat SoA arrays
    gbBatchTransposeUp(worlds.data(), soa, NW);
    // clobber the source to prove the down copy restores from the SoA arrays
    std::memset(worlds.data(), 0, NW * sizeof(WorldShared));
    // down: flat SoA arrays -> WorldShared array
    gbBatchTransposeDown(worlds.data(), soa, NW);

    // every transposed field must round-trip. Compare field by field through the list so
    // a failure names the field (a raw memcmp would not, and untransposed padding bytes
    // are not part of the round-trip).
    // Compare by raw bytes, so a float field seeded with arbitrary bytes (a possible NaN)
    // round-trips correctly. A value compare would falsely fail since NaN != NaN.
    int fails = 0;
    long checked = 0;
#define CMP_ONE(field, type, len) \
    do { for (int wi = 0; wi < NW; ++wi) for (int i = 0; i < (int)(len); ++i){ \
        ++checked; \
        if (memcmp(&worlds[wi].field[i], &original[wi].field[i], sizeof(type)) != 0){ \
            if (fails == 0) printf("  FAIL field %-16s world %d index %d\n", #field, wi, i); fails = 1; } } } while (0);
#define CMP_SCALAR(field, type) \
    do { for (int wi = 0; wi < NW; ++wi){ ++checked; \
        if (memcmp(&worlds[wi].field, &original[wi].field, sizeof(type)) != 0){ \
            if (fails == 0) printf("  FAIL scalar %-16s world %d\n", #field, wi); fails = 1; } } } while (0);
    GB_BATCH_SOA_FIELDS(CMP_ONE)
    GB_BATCH_SOA_SCALARS(CMP_SCALAR)
#undef CMP_ONE
#undef CMP_SCALAR

    freeSoA(soa);

    printf("  checked %ld field elements across %d worlds\n", checked, NW);
    if (!fails){
        printf("PASS gb_batch_transpose: WorldShared round-trips through the SoA layout byte-exact\n");
        return 0;
    }
    printf("FAIL gb_batch_transpose: see above\n");
    return 1;
}
