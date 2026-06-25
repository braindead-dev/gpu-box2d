// gb_batch_test.cu. Micro-test for the batched-world driver (bindings/gb_batch.cuh). It
// checks two things: the batch produces correct physics (a dropped circle settles on the
// floor), and the batch adds no drift, the per-world result is bit-identical to driving
// gb_world_step on a standalone WorldShared with the same seed.
//
// The driver is the C++ layer the Python binding wraps. This test validates it without
// Python, so the gate covers the driver on any machine. The binding itself is exercised
// by bindings/test_batch.py.
//
// Build (frozen flags), self-contained, host:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS -Iinclude -Ibindings \
//        test/gb_batch_test.cu -o test/gb_batch_test
//   ./test/gb_batch_test
//   Expected: PASS gb_batch: 0 ULP (driver matches standalone step)
#include "gb_batch.cuh"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

static long bitDiff(float a, float b){
    return labs((long)(*(int*)&a) - (long)(*(int*)&b));
}

int main(){
    printf("Batch driver micro-test: gb_batch (bindings) vs standalone gb_world_step\n");
    printf("flags: --fmad=false -prec-div=true -prec-sqrt=true\n\n");

    int fails = 0;

    // 1) physics sanity: 8 worlds, each a circle dropped onto a flat floor settles near
    //    the radius height.
    {
        const int NW = 8;
        const float R = 0.5f;
        GBBatch b(NW);
        for (int w = 0; w < NW; ++w){
            gbBatchSetGroundEdge(b, w, 0, -10.0f, 0.0f, 10.0f, 0.0f);
            gbBatchAddCircle(b, w, 0.05f*w, 3.0f, R, 1.0f, 2.0f, GB_DYNAMIC_BODY);
        }
        for (int s = 0; s < 300; ++s) gbBatchStep(b, 1);
        std::vector<float> pos(NW*GB_MAX_BODIES*2);
        gbBatchGetPositions(b, pos.data());
        for (int w = 0; w < NW; ++w){
            float y = pos[(w*GB_MAX_BODIES + 1)*2 + 1];
            if (!(fabsf(y - R) < 0.05f)){ printf("  FAIL world %d circle rests at y=%.6f\n", w, y); fails = 1; }
        }
        if (!fails) printf("  ok   8 circles settle near the floor (y ~ %.2f)\n", R);
    }

    // 2) bit-identical: the batch driver and a standalone WorldShared step to the same
    //    bits over 300 steps. This proves the driver adds no floating-point drift.
    {
        GBBatch b(1);
        gbBatchSetGroundEdge(b, 0, 0, -10.0f, 0.0f, 10.0f, 0.0f);
        gbBatchAddCircle(b, 0, 0.0f, 3.0f, 0.5f, 1.0f, 2.0f, GB_DYNAMIC_BODY);
        WorldShared standalone = b.worlds[0];
        long maxbits = 0;
        for (int s = 0; s < 300; ++s){
            gbBatchStep(b, 1);
            gb_world_step(standalone);
        }
        for (int sl = 0; sl < GB_MAX_BODIES; ++sl){
            maxbits = std::max(maxbits, bitDiff(b.worlds[0].sweepCx[sl], standalone.sweepCx[sl]));
            maxbits = std::max(maxbits, bitDiff(b.worlds[0].sweepCy[sl], standalone.sweepCy[sl]));
            maxbits = std::max(maxbits, bitDiff(b.worlds[0].sweepA[sl],  standalone.sweepA[sl]));
            maxbits = std::max(maxbits, bitDiff(b.worlds[0].velX[sl],    standalone.velX[sl]));
            maxbits = std::max(maxbits, bitDiff(b.worlds[0].velY[sl],    standalone.velY[sl]));
        }
        if (maxbits != 0){ printf("  FAIL batch-vs-standalone max ULP = %ld\n", maxbits); fails = 1; }
        else printf("  ok   batch driver is bit-identical to standalone step (0 ULP, 300 steps)\n");
    }

    // 3) a box settles on the floor through the polygon path.
    {
        GBBatch b(1);
        gbBatchSetGroundEdge(b, 0, 0, -10.0f, 0.0f, 10.0f, 0.0f);
        gbBatchAddBox(b, 0, 0.0f, 3.0f, 0.5f, 0.5f, 1.0f, 12.0f, GB_DYNAMIC_BODY);
        for (int s = 0; s < 300; ++s) gbBatchStep(b, 1);
        std::vector<float> pos(GB_MAX_BODIES*2);
        gbBatchGetPositions(b, pos.data());
        float y = pos[1*2 + 1];
        if (!(fabsf(y - 0.5f) < 0.05f)){ printf("  FAIL box rests at y=%.6f\n", y); fails = 1; }
        else printf("  ok   box settles on the floor (y=%.4f)\n", y);
    }

    if (!fails){
        printf("PASS gb_batch: 0 ULP (driver matches standalone step), physics correct\n");
        return 0;
    }
    printf("FAIL gb_batch: see above\n");
    return 1;
}
