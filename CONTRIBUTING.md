# Contributing

gpu-box2d is a bit-faithful port of Box2D 2.3.0. The one rule that shapes every contribution is that the engine reproduces Box2D 2.3.0's floats exactly. A change is correct when its output matches the reference at 0 ULP, proven by a micro-test. This guide is how to add to the engine while keeping that guarantee.

## The module-and-micro-test workflow

Every capability is a header module under `include/gpu_box2d/` plus a self-contained micro-test under `test/` that compares it against an embedded Box2D 2.3.0 reference and asserts 0 ULP. A module joins the assembled step only after its micro-test is green.

To add a module:

1. **Write the header** in `include/gpu_box2d/`, against the accessor macros (`BODY` / `CONT` / `EDGE` / `JOINT` / `SCAL`) and `gb_math.cuh`. Match Box2D 2.3.0's math and its evaluation order. Never read raw arrays; never add globals.
2. **Write the micro-test** in `test/<module>_test.cu`, following `test/MICROTEST_TEMPLATE.md`. Embed a line-faithful Box2D 2.3.0 reference of the same routine in the same translation unit, run both over a fixed input, and assert 0 ULP with the standard `ulpDiff` helper.
3. **Build with the frozen flags** and run it. Host-mode is the development path; the x86/CUDA gate is the definitive one.
4. **Wire it into the gate.** Add the test to `test/run_gate.sh`, `test/run_gate_host.sh`, and `CMakeLists.txt`. When green, include the header in `gb_step.cuh` if it joins the assembled step, and re-run the full gate.

The two seams an application uses (the contact listener and the per-world field injection) are documented in [docs/api.md](docs/api.md); a contribution that changes them is rare and must keep the seam generic, with no game concept in the core.

## The two rules that never bend

1. **Go through the accessors.** New code reaches world state with `BODY` / `CONT` / `EDGE` / `JOINT` / `SCAL`, never `w.field[slot]`. This is what keeps the SoA-global and block-per-world backends swappable.
2. **Match Box2D's evaluation order along with its math.** A new manifold, solver path, or assembly step visits elements in the same order Box2D 2.3.0 does, and any running float fold stays serial and in order. Reordering changes the floats even when the math is right. See [docs/architecture.md](docs/architecture.md).

## Running the gate

Host-mode (no GPU, the development path):

```
CXX=clang++ ./test/run_gate_host.sh
```

The definitive x86/CUDA gate with the frozen flags:

```
ARCH=86 ./test/run_gate.sh
```

Or through CTest:

```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
ctest --test-dir build --output-on-failure
```

A contribution keeps the gate green. The CI in `.github/workflows/ci.yml` runs the host gate and the Python binding on every push and pull request.

## The frozen build flags

These put the floating-point environment in the state that matches the CPU Box2D reference. Do not relax them.

```
--fmad=false -prec-div=true -prec-sqrt=true          # nvcc
-ffp-contract=off -mfpmath=sse                       # host x86
```

On arm64 there is no `-mfpmath=sse`; the gate's self-contained tests still compare 0 ULP because each test's subject and reference are compiled together. The definitive bit match is the x86/CUDA gate.

## Style

Prose and code, including comments, follow a few hard rules:

- No em-dashes, no non-ASCII characters.
- No contrast-by-negation ("X, not Y"); state it positively.
- Concise. Documentation soft-wrapped, one logical line per paragraph.

Grep-verify clean after each batch of edits:

```
grep -rnP "[^\x00-\x7F]" include test docs README.md   # non-ASCII
grep -rn "\xE2\x80\x94" .                                 # em-dash
```

## Commits

Keep commits clean, professional, and incremental, one capability per commit or a small group. Write what changed and why the fidelity claim holds. No session footers, no co-author trailers. The commit author is the project author.

## What a good contribution looks like

A new joint type, for example: a header with the three phases (`InitVelocityConstraints`, `SolveVelocityConstraints`, `SolvePositionConstraints`) ported in Box2D order; a micro-test that swings or slides the joint over hundreds of substeps and reproduces every body kinematic and impulse at 0 ULP against an embedded reference; the test wired into all three gate definitions; and the docs (the fidelity table, the extending guide) updated to match. Every claim in the docs maps to a passing test.
