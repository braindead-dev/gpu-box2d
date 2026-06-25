# Benchmark

A throughput benchmark for the batched-world driver. It seeds a fixed scene per world (a column of circles and a couple of boxes settling on a floor), steps the whole batch a fixed number of times, and reports batch-steps per second and world-steps per second across a sweep of world counts.

## Run

```
CXX=clang++ ./bench/run_bench.sh        # 200 timed steps by default
CXX=clang++ ./bench/run_bench.sh 400    # 400 timed steps
```

This host build measures the CPU driver, which is bit-identical to a single-threaded host Box2D 2.3.0, so it establishes the per-world cost and the scaling shape without a GPU. A world-step is one `gb_world_step` (collide, solve, TOI) on one world. World-steps per second stays flat as the world count grows, which is the embarrassingly-parallel-across-worlds property the GPU path exploits.

The measured host table and the relationship to the GPU throughput are in [../docs/performance.md](../docs/performance.md).
