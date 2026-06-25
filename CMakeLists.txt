cmake_minimum_required(VERSION 3.18)
project(gpu_box2d LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# gpu-box2d is a header-only CUDA library. Consumers link the INTERFACE target
# gpu_box2d, which carries the include path and the fidelity-critical compile
# flags. Those flags make the GPU floating-point environment match a CPU build
# of Box2D 2.3.0 (-ffp-contract=off -mfpmath=sse), which is what makes results
# bit-identical across platforms. See docs/fidelity.md.
# ----------------------------------------------------------------------------

# Default to an Ampere data-center target. Override with
# -DCMAKE_CUDA_ARCHITECTURES=86 (or your arch) at configure time.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# The frozen flags. Do not relax these; they are why the engine is bit-identical.
#   --fmad=false     disable fused multiply-add (matches -ffp-contract=off)
#   -prec-div=true   IEEE division (matches -mfpmath=sse)
#   -prec-sqrt=true  IEEE square root
set(GPU_BOX2D_FROZEN_FLAGS --fmad=false -prec-div=true -prec-sqrt=true)

add_library(gpu_box2d INTERFACE)
target_include_directories(gpu_box2d INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_compile_options(gpu_box2d INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:${GPU_BOX2D_FROZEN_FLAGS}>)
target_compile_features(gpu_box2d INTERFACE cxx_std_14)

# The production backend is thread-per-world on the transposed SoA layout, selected
# with -DGB_SOA_GLOBAL. Consumers that want it set this option; the micro-tests build
# against the default (contiguous per-world) backend because they drive a WorldShared
# directly. See docs/architecture.md.
option(GPU_BOX2D_SOA_GLOBAL "Use the thread-per-world SoA-global production backend" OFF)
if(GPU_BOX2D_SOA_GLOBAL)
  target_compile_definitions(gpu_box2d INTERFACE GB_SOA_GLOBAL)
endif()

# ----------------------------------------------------------------------------
# Tests. Each module ships a 0-ULP micro-test. The broad-phase test is fully
# self-contained (it embeds its own Box2D 2.3.0 reference) and runs by default.
# The CCD and solver/island tests compare against a CPU build of Box2D 2.3.0,
# supplied on the include path; enable them with GPU_BOX2D_BUILD_REF_TESTS and
# point GPU_BOX2D_REF_INCLUDE at your reference build. See test/README.md.
# ----------------------------------------------------------------------------
option(GPU_BOX2D_BUILD_TESTS "Build the self-contained 0-ULP micro-tests" ON)
option(GPU_BOX2D_BUILD_REF_TESTS "Build the tests that need a Box2D 2.3.0 reference" OFF)
set(GPU_BOX2D_REF_INCLUDE "" CACHE PATH "Include dir of the Box2D 2.3.0 reference build")

if(GPU_BOX2D_BUILD_TESTS)
  enable_testing()

  add_executable(gb_broadphase_test test/gb_broadphase_test.cu)
  target_link_libraries(gb_broadphase_test PRIVATE gpu_box2d)
  target_include_directories(gb_broadphase_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_broadphase COMMAND gb_broadphase_test)
  set_tests_properties(gb_broadphase PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_broadphase")

  # Polygon narrow-phase and mass formula. Self-contained (embeds its Box2D 2.3.0
  # polygon reference). Validated, 0 ULP.
  add_executable(gb_polygon_test test/gb_polygon_test.cu)
  target_link_libraries(gb_polygon_test PRIVATE gpu_box2d)
  target_include_directories(gb_polygon_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_polygon COMMAND gb_polygon_test)
  set_tests_properties(gb_polygon PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_polygon")

  # Edge-polygon narrow-phase. Self-contained (embeds its Box2D 2.3.0
  # b2CollideEdgeAndPolygon reference). Validated, 0 ULP.
  add_executable(gb_collide_edge_polygon_test test/gb_collide_edge_polygon_test.cu)
  target_link_libraries(gb_collide_edge_polygon_test PRIVATE gpu_box2d)
  target_include_directories(gb_collide_edge_polygon_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_compile_definitions(gb_collide_edge_polygon_test PRIVATE GB_ENABLE_POLYGONS)
  add_test(NAME gb_collide_edge_polygon COMMAND gb_collide_edge_polygon_test)
  set_tests_properties(gb_collide_edge_polygon PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_collide_edge_polygon")

  # Chain shape. Self-contained (embeds its Box2D 2.3.0 b2ChainShape reference).
  # Validated, 0 ULP on the child-edge generation.
  add_executable(gb_chain_shape_test test/gb_chain_shape_test.cu)
  target_link_libraries(gb_chain_shape_test PRIVATE gpu_box2d)
  target_include_directories(gb_chain_shape_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_compile_definitions(gb_chain_shape_test PRIVATE GB_ENABLE_POLYGONS)
  add_test(NAME gb_chain_shape COMMAND gb_chain_shape_test)
  set_tests_properties(gb_chain_shape PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_chain_shape")

  # Two-point block solver. Self-contained (embeds its Box2D 2.3.0 b2ContactSolver
  # reference). Validated, 0 ULP.
  add_executable(gb_block_solver_test test/gb_block_solver_test.cu)
  target_link_libraries(gb_block_solver_test PRIVATE gpu_box2d)
  target_include_directories(gb_block_solver_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_block_solver COMMAND gb_block_solver_test)
  set_tests_properties(gb_block_solver PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_block_solver")

  # Revolute joint (point-to-point). Self-contained (embeds its Box2D 2.3.0
  # b2RevoluteJoint reference). Validated, 0 ULP.
  add_executable(gb_joint_test test/gb_joint_test.cu)
  target_link_libraries(gb_joint_test PRIVATE gpu_box2d)
  target_include_directories(gb_joint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_joint COMMAND gb_joint_test)
  set_tests_properties(gb_joint PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_joint")

  # Revolute joint, full (point-to-point + motor + limit). Self-contained (embeds its
  # Box2D 2.3.0 b2RevoluteJoint reference). Validated, 0 ULP.
  add_executable(gb_revolute_joint_test test/gb_revolute_joint_test.cu)
  target_link_libraries(gb_revolute_joint_test PRIVATE gpu_box2d)
  target_include_directories(gb_revolute_joint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_revolute_joint COMMAND gb_revolute_joint_test)
  set_tests_properties(gb_revolute_joint PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_revolute_joint")

  # Distance joint (rigid rod + soft spring). Self-contained (embeds its Box2D 2.3.0
  # b2DistanceJoint reference). Validated, 0 ULP.
  add_executable(gb_distance_joint_test test/gb_distance_joint_test.cu)
  target_link_libraries(gb_distance_joint_test PRIVATE gpu_box2d)
  target_include_directories(gb_distance_joint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_distance_joint COMMAND gb_distance_joint_test)
  set_tests_properties(gb_distance_joint PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_distance_joint")

  # Weld joint (rigid + soft, the 3x3 path). Self-contained (embeds its Box2D 2.3.0
  # b2WeldJoint reference). Validated, 0 ULP.
  add_executable(gb_weld_joint_test test/gb_weld_joint_test.cu)
  target_link_libraries(gb_weld_joint_test PRIVATE gpu_box2d)
  target_include_directories(gb_weld_joint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_weld_joint COMMAND gb_weld_joint_test)
  set_tests_properties(gb_weld_joint PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_weld_joint")

  # Prismatic joint (free + limit + motor). Self-contained (embeds its Box2D 2.3.0
  # b2PrismaticJoint reference). Validated, 0 ULP.
  add_executable(gb_prismatic_joint_test test/gb_prismatic_joint_test.cu)
  target_link_libraries(gb_prismatic_joint_test PRIVATE gpu_box2d)
  target_include_directories(gb_prismatic_joint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_prismatic_joint COMMAND gb_prismatic_joint_test)
  set_tests_properties(gb_prismatic_joint PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_prismatic_joint")

  # Pulley joint (two-body, ratio). Self-contained (embeds its Box2D 2.3.0
  # b2PulleyJoint reference). Validated, 0 ULP.
  add_executable(gb_pulley_joint_test test/gb_pulley_joint_test.cu)
  target_link_libraries(gb_pulley_joint_test PRIVATE gpu_box2d)
  target_include_directories(gb_pulley_joint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_pulley_joint COMMAND gb_pulley_joint_test)
  set_tests_properties(gb_pulley_joint PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_pulley_joint")

  # Gear joint (revolute-revolute coupling). Self-contained (embeds its Box2D 2.3.0
  # b2GearJoint and b2RevoluteJoint references). Validated, 0 ULP.
  add_executable(gb_gear_joint_test test/gb_gear_joint_test.cu)
  target_link_libraries(gb_gear_joint_test PRIVATE gpu_box2d)
  target_include_directories(gb_gear_joint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_gear_joint COMMAND gb_gear_joint_test)
  set_tests_properties(gb_gear_joint PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_gear_joint")

  # Wired-step integration: polygons and the revolute joint driven through the
  # assembled gb_world_step. Built with the polygon and joint features enabled.
  add_executable(gb_wired_step_test test/gb_wired_step_test.cu)
  target_link_libraries(gb_wired_step_test PRIVATE gpu_box2d)
  target_include_directories(gb_wired_step_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_compile_definitions(gb_wired_step_test PRIVATE GB_ENABLE_POLYGONS GB_ENABLE_JOINTS)
  add_test(NAME gb_wired_step COMMAND gb_wired_step_test)
  set_tests_properties(gb_wired_step PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_wired_step")

  # Chain-step integration: the chain shape wired into gb_world_step as a per-world
  # collider. Built with the polygon and chain features enabled.
  add_executable(gb_chain_step_test test/gb_chain_step_test.cu)
  target_link_libraries(gb_chain_step_test PRIVATE gpu_box2d)
  target_include_directories(gb_chain_step_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_compile_definitions(gb_chain_step_test PRIVATE GB_ENABLE_POLYGONS GB_ENABLE_CHAIN)
  add_test(NAME gb_chain_step COMMAND gb_chain_step_test)
  set_tests_properties(gb_chain_step PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_chain_step")

  # Batch driver (the C++ layer the Python binding wraps). Self-contained. Validates
  # the driver against the standalone step at 0 ULP and checks the physics.
  add_executable(gb_batch_test test/gb_batch_test.cu)
  target_link_libraries(gb_batch_test PRIVATE gpu_box2d)
  target_include_directories(gb_batch_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_CURRENT_SOURCE_DIR}/bindings)
  target_compile_definitions(gb_batch_test PRIVATE GB_ENABLE_POLYGONS GB_ENABLE_JOINTS)
  add_test(NAME gb_batch COMMAND gb_batch_test)
  set_tests_properties(gb_batch PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_batch")

  if(GPU_BOX2D_BUILD_REF_TESTS AND GPU_BOX2D_REF_INCLUDE)
    add_executable(gb_toi_test test/gb_toi_test.cu)
    target_link_libraries(gb_toi_test PRIVATE gpu_box2d)
    target_include_directories(gb_toi_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test ${GPU_BOX2D_REF_INCLUDE})
    add_test(NAME gb_toi COMMAND gb_toi_test)
    set_tests_properties(gb_toi PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

    add_executable(gb_island_test test/gb_island_test.cu test/gb_island_ref.cu)
    target_link_libraries(gb_island_test PRIVATE gpu_box2d)
    target_include_directories(gb_island_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test ${GPU_BOX2D_REF_INCLUDE})
    add_test(NAME gb_island COMMAND gb_island_test)
    set_tests_properties(gb_island PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
  endif()
endif()
