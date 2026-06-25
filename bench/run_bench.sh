#!/usr/bin/env bash
# run_bench.sh. Build and run the throughput benchmark host-mode (CPU). It sweeps the
# world count and reports batch-steps/s and world-steps/s for a fixed scene per world.
#
# This measures the CPU host driver, which is bit-identical to a single-threaded host
# Box2D 2.3.0. The GPU number comes from the same driver compiled with nvcc on the
# SoA-global path; see docs/performance.md.
#
# Usage:
#   ./bench/run_bench.sh            # uses ${CXX:-c++}, 200 timed steps
#   CXX=clang++ ./bench/run_bench.sh 400
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CXX="${CXX:-c++}"
STEPS="${1:-200}"

FLAGS="-O2 -x c++ -ffp-contract=off"
# x86 gets the exact IEEE single-precision environment; arm64 rounds to it without the switch.
case "$(uname -m)" in
  x86_64|amd64) FLAGS="$FLAGS -mfpmath=sse" ;;
esac

BIN="$HERE/gb_bench"
$CXX $FLAGS -DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS \
  -I"$ROOT/include" -I"$ROOT/bindings" "$HERE/gb_bench.cu" -o "$BIN" || {
    echo "benchmark failed to build"; exit 1;
}
"$BIN" "$STEPS"
