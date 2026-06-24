cmake_minimum_required(VERSION 3.18)
project(gpu_box2d LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# gpu-box2d is a header-only CUDA library. Consumers link the INTERFACE target
# gpu_box2d, which carries the include path and the fidelity-critical compile
# flags. Those flags make the GPU floating-point environment match a CPU build
# of Box2D 2.3.0 (-ffp-contract=off -mfpmath=sse), which is what makes results
# bit-identical across platforms. See docs/fidelity.md.
# ----------------------------------------------------------------------------

# Default to a modern data-center / Ampere target. Override with
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
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(gpu_box2d INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:${GPU_BOX2D_FROZEN_FLAGS}>)
target_compile_features(gpu_box2d INTERFACE cxx_std_14)

# ----------------------------------------------------------------------------
# Tests. Each module ships a 0-ULP micro-test. The broad-phase test is fully
# self-contained (it embeds its own Box2D reference) and runs by default. The
# CCD and solver/island tests need the Box2D 2.3.0 reference translation unit,
# which is wired in once the narrow-phase and solver modules are assembled.
# ----------------------------------------------------------------------------
option(GPU_BOX2D_BUILD_TESTS "Build the 0-ULP micro-tests" ON)

if(GPU_BOX2D_BUILD_TESTS)
  enable_testing()

  add_executable(gb_broadphase_test test/gb_broadphase_test.cu)
  target_link_libraries(gb_broadphase_test PRIVATE gpu_box2d)
  target_include_directories(gb_broadphase_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_test(NAME gb_broadphase COMMAND gb_broadphase_test)
  set_tests_properties(gb_broadphase PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS gb_broadphase")
endif()
