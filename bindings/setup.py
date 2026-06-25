# setup.py. Build the gpu-box2d Python extension (host-mode, CPU). The extension wraps
# the batched world driver (bindings/gb_batch.cuh) so an RL or simulation layer drives N
# independent Box2D 2.3.0 worlds from Python and reads per-world state as numpy arrays.
#
# This host build steps on the CPU and is bit-identical to a single-threaded host Box2D
# 2.3.0 (the fidelity contract). A CUDA build of the same driver steps on the GPU through
# the SoA-global production path; that target is built with nvcc and is outside this
# host setup.
#
# Build and install into the active environment:
#   pip install ./bindings
# or build in place for a quick check:
#   python bindings/setup.py build_ext --inplace
#
# The fidelity flags match a CPU build of Box2D with -ffp-contract=off. On x86 add
# -mfpmath=sse for the exact IEEE single-precision environment; arm64 rounds to IEEE
# single precision without that switch.
import os
import sys
import platform

from setuptools import setup, Extension
import pybind11

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INCLUDE = os.path.join(ROOT, "include")

extra_compile_args = ["-O2", "-std=c++14", "-ffp-contract=off"]
# On x86, -mfpmath=sse completes the IEEE single-precision environment that matches the
# CPU Box2D reference. arm64 has no such switch and rounds to IEEE single precision.
machine = platform.machine().lower()
if machine in ("x86_64", "amd64", "i386", "i686"):
    extra_compile_args.append("-mfpmath=sse")

# The driver exposes boxes (polygons) and joints by default.
define_macros = [("GB_ENABLE_POLYGONS", "1"), ("GB_ENABLE_JOINTS", "1")]

ext = Extension(
    name="gpu_box2d",
    sources=[os.path.join(HERE, "gb_pybind.cpp")],
    include_dirs=[pybind11.get_include(), INCLUDE, HERE],
    define_macros=define_macros,
    extra_compile_args=extra_compile_args,
    language="c++",
)

setup(
    name="gpu_box2d",
    version="0.1.0",
    description="Batched, bit-faithful Box2D 2.3.0 worlds with a numpy state API",
    long_description="Drive N independent Box2D 2.3.0 physics worlds and read per-world "
                     "body state as numpy arrays. The host build is bit-identical to a "
                     "single-threaded host Box2D 2.3.0; a CUDA build steps the same state "
                     "on the GPU.",
    ext_modules=[ext],
    python_requires=">=3.7",
    install_requires=["numpy"],
)
