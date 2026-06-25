# Performance

This document states what the engine measures, why that number is the ceiling for a
bit-identical Box2D solver on this hardware, how it scales with the GPU, and the two
alternative execution models that were built, measured, and came in slower. All
numbers are from an A10 (sm_86) with CUDA 12.8.

## The measured result

The production engine runs at about **23K env-steps per second** on an A10. An
env-step is one drop-and-settle of a world (the example application's unit of work),
which is tens to a few hundred physics substeps. That is:

- roughly **12x** a 26-core CPU baseline running the same Box2D 2.3.0 algorithm, and
- about **2x** the pre-rewrite GPU version.

Fidelity holds at this throughput. Single-world physics is 0 ULP against the CPU
reference, and the batched output matches the reference score distribution at a
Kolmogorov-Smirnov p-value of 1.0. The throughput comes entirely from the memory layout
and the execution model; the physics is untouched.

The two gains over the pre-rewrite version are the coalesced SoA lane-equals-world
memory layout (a warp's 32 lanes read 32 consecutive addresses in one transaction)
and a smaller per-thread working set (a tighter arena and a fused velocity-and-position
constraint that lowered register and local-memory pressure). Both reduce memory
traffic without touching a single float in the solver.

## Why 23K is the ceiling here

The number is bounded by the structure of a bit-identical Box2D step. It is a
structural ceiling, with no unfinished optimization left on the table.

**The solver is serial and is most of the step.** Box2D's contact solver is sequential
Gauss-Seidel: each contact reads the body velocities as mutated by the previous
contact, in a fixed order. That read-after-write chain, combined with the nonlinear
per-contact clamps, is the reason the port is bit-identical. It also means the solver
cannot be split across threads or reordered while staying bit-identical. The solver and
its serial folds are about **74 percent of a step**. Amdahl's law caps any speedup that
leaves the solver serial, and the solver must stay serial to keep the floats.

**The rest is occupancy- and control-flow-bound.** The non-solver work (broad-phase,
narrow-phase, integration, sleep, CCD) is parallel across worlds, but the per-thread
register footprint of the fully inlined solver plus CCD caps occupancy, and the
data-dependent control flow (variable island sizes, variable settle lengths, the TOI
event loop) keeps lanes from running in lockstep. These bound the remaining 26 percent.

So 23K is the measured ceiling for a bit-identical Box2D Gauss-Seidel solver on an A10.
Pushing past it on the same card would require either breaking the bit match (a
different solver) or a faster card.

## GPU scaling

Throughput scales with the GPU. The work is embarrassingly parallel across worlds, and
the A10 number is set by how much silicon is available to run independent worlds
concurrently. A card with more streaming multiprocessors and more memory bandwidth runs
proportionally more worlds at once. An H100 has roughly 3x the relevant silicon of an
A10, so the absolute env-steps-per-second rises accordingly while the per-world physics
stays bit-identical. The ceiling is set by the serial solver fraction within one world;
the absolute number is set by the card.

## Two execution models that were measured and lost

Both of these are real, built, and measured. They are documented here because they are
useful to anyone building GPU physics, and because the production choice only makes
sense next to the alternatives it beat.

### Block-per-world: 2.7K env-steps/s

Assign one CUDA block to each world and cooperate the block's threads on that world's
work, with the per-world state resident in shared memory. This is the natural structure
for batched physics and is how Brax and MJX frame per-environment parallelism, so it was
the first candidate. The shell is in `include/gpu_box2d/gb_world.cuh`.

It runs at **2.7K env-steps/s**, about an order of magnitude below thread-per-world. The
reason is the serial solver. The solver spine has to run on one lane to stay
bit-identical, so during the solver (most of the step) the block's other lanes idle.
A block-per-world layout therefore wastes most of its threads exactly when the step
spends most of its time. Shared-memory residency does help the parallel phases, but it
cannot compensate for idling the block through the dominant serial phase.

### Graph-colored parallel solver: 5.1K env-steps/s

Coloring the contact graph lets contacts that touch disjoint bodies solve in parallel
within a color, with colors processed in sequence. This breaks the serial Gauss-Seidel
wall: the solver itself becomes parallel. The implementation is in
`include/gpu_box2d/gb_colored_solver.cuh`.

It runs at **5.1K env-steps/s**, faster than block-per-world but still well below the
serial thread-per-world engine. The coloring reorders the sweep, so this solver is
distribution-faithful rather than bit-identical: it is validated by KS while the
bit-identical serial engine remains the ULP reference. The throughput is limited by its
host. A colored solver needs the block's threads cooperating on one world, so it only
runs inside the block-per-world execution model, and that model collapses occupancy for
the same reason block-per-world does on its own. Breaking the serial wall in the solver
did not pay off, because the execution model it requires gives back more than it gains.

### What the two findings say

The serial Gauss-Seidel solver is the load-bearing constraint. Keeping it serial and
giving each world its own thread (so the parallelism is across worlds, where it is real)
beats both moving the parallelism inside a world (block-per-world) and parallelizing the
solver itself (graph coloring). For a bit-identical Box2D port, thread-per-world is the
right execution model, and the path to higher absolute throughput is a larger GPU.

## Reproducing

The throughput numbers come from the batched launcher stepping a fixed number of worlds
and timing the kernel, best-of over several runs on an uncontended GPU. The fidelity
gates that must stay green alongside any performance change are in `test/run_gate.sh`
and `docs/fidelity.md`.
