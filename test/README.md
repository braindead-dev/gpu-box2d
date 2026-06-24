# Tests

Each module ships a micro-test that compares its output against Box2D 2.3.0 and
asserts 0 ULP. See [MICROTEST_TEMPLATE.md](MICROTEST_TEMPLATE.md) for the pattern and
[../docs/fidelity.md](../docs/fidelity.md) for the methodology.

## Self-contained tests

These build and run from this repository alone. They embed the Box2D 2.3.0 reference
they compare against.

- **`gb_broadphase_test.cu`** compares the `b2DynamicTree` proxyId sequence and the
  `b2BroadPhase::UpdatePairs` AddPair order against an inline reference copy of the
  Box2D 2.3.0 logic. Validated, 0 ULP. This is the test CMake and the gate run.

Build and run it directly:

```
nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
     -Iinclude -Itest test/gb_broadphase_test.cu -o test/gb_broadphase_test
./test/gb_broadphase_test
```

Or through CMake / CTest:

```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
ctest --test-dir build --output-on-failure
```

## Tests that need the Box2D 2.3.0 reference translation unit

These compare against the upstream Box2D 2.3.0 reference compiled into a separate
translation unit. They are wired in once the narrow-phase and solver modules are
assembled and the reference TU is part of the build.

- **`gb_toi_test.cu`** compares the GJK distance and `b2TimeOfImpact` result against
  the Box2D 2.3.0 CCD reference on a fruit-wall continuous-collision scenario.
  Validated, 0 ULP.
- **`gb_island_test.cu`** with **`gb_island_ref.cu`** compares the contact solver and
  island assembly against `b2World::Solve`. `gb_test_iface.h` is the flat-POD bridge
  between the test's type universe and the reference's. The single-drop and two-body
  scenarios pass at 0 ULP on host and device; the five-body pile currently shows a
  1-ULP device residual (see ../docs/fidelity.md).
