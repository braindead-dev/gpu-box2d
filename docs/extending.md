# Extending toward full Box2D

The current core covers circles and static edges, the single-point contact solver,
and the CCD path. Box2D supports polygons, two-point manifolds, and joints. This
document is the path to adding them while keeping the bit-identical guarantee.

## The two rules that never bend

Before any extension, two constraints hold for every line of new code.

1. **Go through the accessors.** New code reads and writes world state with
   `BODY` / `CONT` / `EDGE` / `SCAL`, never raw `w.field[slot]`. This is what keeps
   the memory backend swappable. A module that reaches around the accessors breaks
   the SoA-global backend silently.
2. **Match Box2D's evaluation order along with its math.** The result is the
   algorithm and the evaluation order together. A new manifold, solver path, or
   assembly step must visit elements in the same order Box2D 2.3.0 does, and any
   running fold stays serial and in order (see [architecture.md](architecture.md)).
   Reordering changes the floats even when the math is right.

Every extension ships with a 0-ULP micro-test against the Box2D 2.3.0 reference
before it joins the assembled step. The template is in `test/MICROTEST_TEMPLATE.md`.

## Adding polygons

Polygons add a shape type, narrow-phase functions, and a mass formula.

- **Shape data.** A polygon needs its vertices, normals, and centroid. Store them in
  `WorldShared` as fixed-capacity arrays indexed by body slot, behind new `BODY`
  accessors. Keep a per-body shape tag (circle or polygon) so the narrow-phase can
  dispatch.
- **Narrow-phase.** Port `b2CollidePolygons` and `b2CollidePolygonAndCircle` from
  Box2D 2.3.0 into `gb_collision.cuh`. These produce up to two manifold points and a
  reference face, which is what motivates the two-point solver below. The clipping
  order and the incident-edge selection must match Box2D exactly.
- **Mass.** `b2PolygonShape::ComputeMass` integrates over the polygon. Port it as
  written; the centroid and rotational inertia feed the solver's `invMass` and
  `invI`.
- **Broad-phase.** Polygons need a tight AABB from their rotated vertices. Add a
  `gbPolygonAABB` helper next to `gbCircleAABB` in `gb_broadphase.cuh`. The rest of
  the tree is shape-agnostic and needs no change.

## The two-point solver

Circles produce one contact point. Polygons produce up to two, and Box2D solves a
two-point manifold with a block solver that handles both points together when it can.

- **Constraint.** Extend `GBConstraint` in `gb_contact_types.cuh` to carry two
  points. The fused velocity and position fields become arrays of two.
- **Block solve.** Port `b2ContactSolver::SolveVelocityConstraints`'s two-point block
  path. It attempts a 2x2 solve of both normal impulses and falls back to two
  sequential single-point solves through a fixed cascade of cases. Reproduce the
  cascade in order; the fallback branch taken is part of the result.
- **Order.** The two points within a contact solve in Box2D's order, and contacts
  still solve in island order. The serial-in-order solver rule covers both.
- **Test.** A box resting on the ground exercises the two-point block solver
  directly. Compare every body's post-solve kinematics and both warm-start impulses
  at 0 ULP.

## Joints

Joints add constraint rows that solve alongside contacts in `b2Island::Solve`.

- **Storage.** Add a joint pool to `WorldShared` (anchors, reference angles, per-type
  parameters) behind `JOINT` accessors in the same style as `CONT`.
- **Solve.** Each joint type (revolute, prismatic, distance, weld, and so on) has its
  own `InitVelocityConstraints`, `SolveVelocityConstraints`, and
  `SolvePositionConstraints`. Box2D solves joints and contacts in a fixed interleave
  within each velocity and position iteration. Reproduce that interleave; joints
  before contacts, in joint-list order, then contacts in contact order.
- **Island assembly.** Extend the DFS in `gb_island.cuh` to walk joint edges as well
  as contact edges so joint-connected bodies land in the same island, matching
  Box2D's graph traversal.
- **Test.** A two-body pendulum on a revolute joint, settled over many substeps,
  compared at 0 ULP.

## Growing the bounds

Polygons and joints enlarge the per-world working set. The bounds are compile-time
defines in `gb_pools.cuh` (`GB_MAX_BODIES`, `GB_MAX_CONTACTS`). Raising them grows
`WorldShared`. If the result exceeds the 48 KB default shared budget, use the 100 KB
shared opt-in on sm_86 and later, or move cold per-world state (infrequently touched
fields) to global storage behind the accessors. The shared budget is the design
constraint to track. Because every access goes through the accessors, moving a field
between shared and global is a backend change that leaves every call site untouched.

## Checklist for a new module

1. Write the header in `include/gpu_box2d/`, against the accessors, matching Box2D
   2.3.0 math and order.
2. Write `test/<module>_test.cu` that compares against the Box2D 2.3.0 reference and
   asserts 0 ULP.
3. Build with the frozen flags (`--fmad=false -prec-div=true -prec-sqrt=true`).
4. When the micro-test is green, include the header in `gb_step.cuh` and re-run the
   full gate. If a gate reddens, revert the single include and find the divergence.
