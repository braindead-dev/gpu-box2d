// gb_bench.cu. A throughput benchmark for the batched-world driver. It seeds a fixed
// scene per world (a small pile of circles and boxes dropping onto a floor), steps the
// whole batch a fixed number of times, and reports steps per second and world-steps per
// second across a sweep of world counts.
//
// This host build measures the CPU driver, which is bit-identical to a single-threaded
// host Box2D 2.3.0. It establishes the per-world cost and the scaling shape; the GPU
// number comes from the same driver compiled with nvcc on the SoA-global path and is
// reported in docs/performance.md. The benchmark itself is the same code on either
// target.
//
// Build (host):
//   clang++ -O2 -ffp-contract=off -Iinclude -Ibindings -DGB_ENABLE_POLYGONS \
//           -DGB_ENABLE_JOINTS bench/gb_bench.cu -o bench/gb_bench
//   ./bench/gb_bench
//   (add -mfpmath=sse on x86 for the exact IEEE single-precision environment)
#include "gb_batch.cuh"
#include <cstdio>
#include <chrono>
#include <vector>

// Seed one world with a deterministic scene: a flat floor, a column of circles, and a
// couple of boxes, so every world does comparable work.
static void seedScene(GBBatch& b, int world){
    gbBatchSetGroundEdge(b, world, 0, -20.0f, 0.0f, 20.0f, 0.0f);
    // a column of 8 circles dropping in
    for (int i = 0; i < 8; ++i){
        float x = -1.0f + 0.13f * i;
        float y = 1.0f + 0.6f * i;
        gbBatchAddCircle(b, world, x, y, 0.3f, 1.0f, 4.0f, GB_DYNAMIC_BODY);
    }
    // two boxes
    gbBatchAddBox(b, world, 1.5f, 1.0f, 0.4f, 0.4f, 1.0f, 8.0f, GB_DYNAMIC_BODY);
    gbBatchAddBox(b, world, 2.0f, 2.0f, 0.4f, 0.4f, 1.0f, 8.0f, GB_DYNAMIC_BODY);
}

static double benchOne(int NW, int steps){
    GBBatch b(NW);
    for (int w = 0; w < NW; ++w) seedScene(b, w);
    // a few warmup steps so the bodies are in contact before timing
    for (int s = 0; s < 10; ++s) gbBatchStep(b, 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s) gbBatchStep(b, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    return secs;
}

int main(int argc, char** argv){
    int steps = 200;
    if (argc > 1) steps = atoi(argv[1]);

    printf("gpu-box2d throughput benchmark (host build, CPU)\n");
    printf("scene per world: 8 circles + 2 boxes on a floor; %d steps timed\n\n", steps);
    printf("%10s  %12s  %16s  %16s\n", "worlds", "wall (s)", "batch-steps/s", "world-steps/s");
    printf("%10s  %12s  %16s  %16s\n", "------", "--------", "-------------", "-------------");

    int counts[] = { 1, 16, 64, 256, 1024, 4096 };
    for (int ci = 0; ci < (int)(sizeof(counts)/sizeof(counts[0])); ++ci){
        int NW = counts[ci];
        double secs = benchOne(NW, steps);
        double batchStepsPerSec = steps / secs;
        double worldStepsPerSec = (double)steps * NW / secs;
        printf("%10d  %12.4f  %16.1f  %16.0f\n", NW, secs, batchStepsPerSec, worldStepsPerSec);
    }

    printf("\nNote: a world-step is one gb_world_step on one world (collide + solve + TOI).\n");
    printf("This is the CPU host driver. The GPU number from the same driver on the\n");
    printf("SoA-global path is reported in docs/performance.md.\n");
    return 0;
}
