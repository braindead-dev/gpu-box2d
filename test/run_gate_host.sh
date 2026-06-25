#!/usr/bin/env bash
# run_gate_host.sh. Build and run the per-module 0-ULP micro-tests on the CPU with a
# host C++ compiler. Each micro-test is a self-contained translation unit whose subject
# and reference math is host_device, so it runs on any machine with no GPU.
#
# This is the development gate. It confirms the algorithm and the evaluation order. The
# definitive bit match is the x86/CUDA gate (run_gate.sh), which holds the GPU and the
# CPU reference in the same IEEE single-precision state through the frozen nvcc flags.
# On an x86 host you can add -mfpmath=sse -ffp-contract=off to match that state here too;
# arm64 has no such switch, so an arm64 host run validates the algorithm rather than the
# exact bits.
#
# Usage:
#   ./test/run_gate_host.sh            # uses ${CXX:-c++}
#   CXX=clang++ ./test/run_gate_host.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
INC="$ROOT/include"
CXX="${CXX:-c++}"
FLAGS="-O2 -x c++ -ffp-contract=off"
# On x86, -mfpmath=sse completes the IEEE single-precision environment that matches the
# CPU Box2D reference. arm64 rounds to IEEE single precision without that switch. Each
# micro-test embeds its reference in the same translation unit, so the 0-ULP comparison
# holds on either architecture; the flags align the host environment with the gate.
case "$(uname -m)" in
  x86_64|amd64|i386|i686) FLAGS="$FLAGS -mfpmath=sse" ;;
esac

PASS=0
FAIL=0
ok(){ echo "  GREEN  $1"; PASS=$((PASS+1)); }
bad(){ echo "  RED    $1"; FAIL=$((FAIL+1)); }

# build_run <module> <pass-grep> <extra-flags>. The source is test/<module>_test.cu.
build_run(){
  local mod="$1" passline="$2" extra="${3:-}"
  local bin="$HERE/${mod}_test_host"
  if $CXX $FLAGS $extra -I"$INC" -I"$HERE" "$HERE/${mod}_test.cu" -o "$bin" 2>/dev/null; then
    if "$bin" 2>&1 | grep -q "$passline"; then
      ok "$mod"
    else
      bad "$mod diverged"
    fi
  else
    bad "$mod failed to build"
  fi
}

echo "=== gpu-box2d host-mode micro-test gate (CXX=$CXX) ==="

build_run gb_broadphase          "PASS gb_broadphase"
build_run gb_polygon             "PASS gb_polygon"
build_run gb_collide_edge_polygon "PASS gb_collide_edge_polygon" "-DGB_ENABLE_POLYGONS"
build_run gb_chain_shape         "PASS gb_chain_shape" "-DGB_ENABLE_POLYGONS"
build_run gb_block_solver        "PASS gb_block_solver"
build_run gb_joint               "PASS gb_joint"
build_run gb_revolute_joint      "PASS gb_revolute_joint"
build_run gb_distance_joint      "PASS gb_distance_joint"
build_run gb_weld_joint          "PASS gb_weld_joint"
build_run gb_prismatic_joint     "PASS gb_prismatic_joint"
build_run gb_pulley_joint        "PASS gb_pulley_joint"
build_run gb_gear_joint          "PASS gb_gear_joint"
build_run gb_wired_step          "PASS gb_wired_step" "-DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS"
build_run gb_chain_step          "PASS gb_chain_step" "-DGB_ENABLE_POLYGONS -DGB_ENABLE_CHAIN"
build_run gb_batch               "PASS gb_batch" "-DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS -I$ROOT/bindings"
build_run gb_batch_transpose     "PASS gb_batch_transpose" "-DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS -DGB_ENABLE_CHAIN -I$ROOT/bindings"

echo "================================================================"
echo "HOST GATE SUMMARY: $PASS green, $FAIL red"
if [ "$FAIL" -eq 0 ]; then
  echo "ALL GREEN."
  exit 0
else
  echo "GATE RED. A module diverged from the Box2D 2.3.0 reference."
  exit 1
fi
