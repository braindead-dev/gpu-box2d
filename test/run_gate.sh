#!/usr/bin/env bash
# run_gate.sh. Build and run the per-module 0-ULP micro-tests. Each test compares
# a module's output against Box2D 2.3.0 and asserts 0 ULP. Exit 0 means every gate
# is green. A non-zero exit means a module diverged from the reference.
#
# The frozen flags make the GPU floating-point environment match a CPU build of
# Box2D 2.3.0, which is what makes the comparison meaningful. Do not change them.
#   --fmad=false -prec-div=true -prec-sqrt=true
#
# Usage:
#   ARCH=86 ./test/run_gate.sh        # sm_86 (A10, A100-class); set ARCH for yours
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
INC="$ROOT/include"
ARCH="${ARCH:-86}"
FLAGS="-O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_${ARCH}"

PASS=0
FAIL=0
ok(){ echo "  GREEN  $1"; PASS=$((PASS+1)); }
bad(){ echo "  RED    $1"; FAIL=$((FAIL+1)); }

echo "=== gpu-box2d micro-test gate (arch sm_${ARCH}, frozen flags) ==="

# Broad-phase: self-contained (embeds its own Box2D reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_broadphase_test.cu" -o "$HERE/gb_broadphase_test" 2>/dev/null; then
  if "$HERE/gb_broadphase_test" 2>&1 | grep -q "PASS gb_broadphase"; then
    ok "gb_broadphase (proxyId + AddPair order, 0 ULP)"
  else
    bad "gb_broadphase diverged"
  fi
else
  bad "gb_broadphase failed to build"
fi

# Polygon narrow-phase + mass: self-contained (embeds its Box2D 2.3.0 polygon
# reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_polygon_test.cu" -o "$HERE/gb_polygon_test" 2>/dev/null; then
  if "$HERE/gb_polygon_test" 2>&1 | grep -q "PASS gb_polygon"; then
    ok "gb_polygon (mass + polygon-polygon + polygon-circle, 0 ULP)"
  else
    bad "gb_polygon diverged"
  fi
else
  bad "gb_polygon failed to build"
fi

# Edge-polygon narrow-phase: self-contained (embeds its Box2D 2.3.0
# b2CollideEdgeAndPolygon reference). Validated, 0-ULP.
if nvcc $FLAGS -DGB_ENABLE_POLYGONS -I"$INC" -I"$HERE" "$HERE/gb_collide_edge_polygon_test.cu" -o "$HERE/gb_collide_edge_polygon_test" 2>/dev/null; then
  if "$HERE/gb_collide_edge_polygon_test" 2>&1 | grep -q "PASS gb_collide_edge_polygon"; then
    ok "gb_collide_edge_polygon (b2CollideEdgeAndPolygon, face-A + face-B, 0 ULP)"
  else
    bad "gb_collide_edge_polygon diverged"
  fi
else
  bad "gb_collide_edge_polygon failed to build"
fi

# Chain shape: self-contained (embeds its Box2D 2.3.0 b2ChainShape reference).
# Validated, 0-ULP on the child-edge generation; integration with the edge collider.
if nvcc $FLAGS -DGB_ENABLE_POLYGONS -I"$INC" -I"$HERE" "$HERE/gb_chain_shape_test.cu" -o "$HERE/gb_chain_shape_test" 2>/dev/null; then
  if "$HERE/gb_chain_shape_test" 2>&1 | grep -q "PASS gb_chain_shape"; then
    ok "gb_chain_shape (b2ChainShape child edges, 0 ULP)"
  else
    bad "gb_chain_shape diverged"
  fi
else
  bad "gb_chain_shape failed to build"
fi

# Two-point block solver: self-contained (embeds its Box2D 2.3.0 b2ContactSolver
# reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_block_solver_test.cu" -o "$HERE/gb_block_solver_test" 2>/dev/null; then
  if "$HERE/gb_block_solver_test" 2>&1 | grep -q "PASS gb_block_solver"; then
    ok "gb_block_solver (two-point LCP block solve, 0 ULP)"
  else
    bad "gb_block_solver diverged"
  fi
else
  bad "gb_block_solver failed to build"
fi

# Revolute joint (point-to-point): self-contained (embeds its Box2D 2.3.0
# b2RevoluteJoint reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_joint_test.cu" -o "$HERE/gb_joint_test" 2>/dev/null; then
  if "$HERE/gb_joint_test" 2>&1 | grep -q "PASS gb_joint"; then
    ok "gb_joint (revolute pendulum, 0 ULP)"
  else
    bad "gb_joint diverged"
  fi
else
  bad "gb_joint failed to build"
fi

# Distance joint (rigid rod + soft spring): self-contained (embeds its Box2D 2.3.0
# b2DistanceJoint reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_distance_joint_test.cu" -o "$HERE/gb_distance_joint_test" 2>/dev/null; then
  if "$HERE/gb_distance_joint_test" 2>&1 | grep -q "PASS gb_distance_joint"; then
    ok "gb_distance_joint (rigid rod + soft spring, 0 ULP)"
  else
    bad "gb_distance_joint diverged"
  fi
else
  bad "gb_distance_joint failed to build"
fi

# Weld joint (rigid + soft, the 3x3 path): self-contained (embeds its Box2D 2.3.0
# b2WeldJoint reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_weld_joint_test.cu" -o "$HERE/gb_weld_joint_test" 2>/dev/null; then
  if "$HERE/gb_weld_joint_test" 2>&1 | grep -q "PASS gb_weld_joint"; then
    ok "gb_weld_joint (rigid + soft weld, 3x3 path, 0 ULP)"
  else
    bad "gb_weld_joint diverged"
  fi
else
  bad "gb_weld_joint failed to build"
fi

# Revolute joint, full (point-to-point + motor + limit): self-contained (embeds its
# Box2D 2.3.0 b2RevoluteJoint reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_revolute_joint_test.cu" -o "$HERE/gb_revolute_joint_test" 2>/dev/null; then
  if "$HERE/gb_revolute_joint_test" 2>&1 | grep -q "PASS gb_revolute_joint"; then
    ok "gb_revolute_joint (point-to-point + motor + limit, 0 ULP)"
  else
    bad "gb_revolute_joint diverged"
  fi
else
  bad "gb_revolute_joint failed to build"
fi

# Prismatic joint (free + limit + motor): self-contained (embeds its Box2D 2.3.0
# b2PrismaticJoint reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_prismatic_joint_test.cu" -o "$HERE/gb_prismatic_joint_test" 2>/dev/null; then
  if "$HERE/gb_prismatic_joint_test" 2>&1 | grep -q "PASS gb_prismatic_joint"; then
    ok "gb_prismatic_joint (free + limit + motor, 0 ULP)"
  else
    bad "gb_prismatic_joint diverged"
  fi
else
  bad "gb_prismatic_joint failed to build"
fi

# Pulley joint (two-body, ratio): self-contained (embeds its Box2D 2.3.0
# b2PulleyJoint reference). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_pulley_joint_test.cu" -o "$HERE/gb_pulley_joint_test" 2>/dev/null; then
  if "$HERE/gb_pulley_joint_test" 2>&1 | grep -q "PASS gb_pulley_joint"; then
    ok "gb_pulley_joint (two-body pulley, 0 ULP)"
  else
    bad "gb_pulley_joint diverged"
  fi
else
  bad "gb_pulley_joint failed to build"
fi

# Gear joint (revolute-revolute coupling): self-contained (embeds its Box2D 2.3.0
# b2GearJoint and b2RevoluteJoint references). Validated, 0-ULP.
if nvcc $FLAGS -I"$INC" -I"$HERE" "$HERE/gb_gear_joint_test.cu" -o "$HERE/gb_gear_joint_test" 2>/dev/null; then
  if "$HERE/gb_gear_joint_test" 2>&1 | grep -q "PASS gb_gear_joint"; then
    ok "gb_gear_joint (revolute-revolute gear, 0 ULP)"
  else
    bad "gb_gear_joint diverged"
  fi
else
  bad "gb_gear_joint failed to build"
fi

# Wired-step integration: polygons and the revolute joint driven through the
# assembled gb_world_step (built with the polygon and joint features on).
if nvcc $FLAGS -DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS -I"$INC" -I"$HERE" "$HERE/gb_wired_step_test.cu" -o "$HERE/gb_wired_step_test" 2>/dev/null; then
  if "$HERE/gb_wired_step_test" 2>&1 | grep -q "PASS gb_wired_step"; then
    ok "gb_wired_step (polygons + joint live in gb_world_step)"
  else
    bad "gb_wired_step diverged"
  fi
else
  bad "gb_wired_step failed to build"
fi

# Batch driver (the C++ layer the Python binding wraps): self-contained. Validates the
# driver produces correct physics and is bit-identical to the standalone step.
if nvcc $FLAGS -DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS -I"$INC" -I"$ROOT/bindings" -I"$HERE" "$HERE/gb_batch_test.cu" -o "$HERE/gb_batch_test" 2>/dev/null; then
  if "$HERE/gb_batch_test" 2>&1 | grep -q "PASS gb_batch"; then
    ok "gb_batch (batched driver matches standalone step, 0 ULP)"
  else
    bad "gb_batch diverged"
  fi
else
  bad "gb_batch failed to build"
fi

# The CCD (gb_toi) and solver/island (gb_contact_solver + gb_island) micro-tests
# compile against the Box2D 2.3.0 reference translation unit. That reference is
# wired in once the narrow-phase and solver modules are assembled. Until then,
# build and run them from the development tree.

echo "================================================================"
echo "GATE SUMMARY: $PASS green, $FAIL red"
if [ "$FAIL" -eq 0 ]; then
  echo "ALL GREEN."
  exit 0
else
  echo "GATE RED. A module diverged from Box2D 2.3.0."
  exit 1
fi
