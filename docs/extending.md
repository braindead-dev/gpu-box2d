# Extending toward full Box2D

The core covers circles, edges, and convex polygons, the single-point and two-point
contact solvers, continuous collision, and the revolute joint. This document records
how the polygon and joint layers were added on the accessor contract, and gives the
concrete path for the next shape, solver row, and joint type. Every step keeps the
bit-identical guarantee.

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

## Polygons (shipped)

Polygon support adds a shape type, narrow-phase functions, and a mass formula. It
lives in `gb_polygon.cuh` and the polygon section of `gb_collision.cuh`.

- **Shape data.** `GBPolygon` in `gb_polygon.cuh` holds the vertices, outward edge
  normals, centroid, count, and skin radius. `gbPolygonSetAsBox` builds a box,
  `gbPolygonSet` builds a convex hull with the gift-wrap algorithm in Box2D's vertex
  order, and `gbPolygonComputeMass` integrates the triangles for mass, center, and
  inertia. `gb_polygon_test.cu` checks the box mass formula at 0 ULP.
- **Narrow-phase.** `gbCollidePolygons` ports `b2CollidePolygons`: reference-face
  selection through `gbFindMaxSeparation` and `gbEdgeSeparation`, incident-edge
  selection through `gbFindIncidentEdge`, and the two-sided clip through
  `gbClipSegmentToLine`. It produces up to two manifold points and a reference face.
  `gbCollidePolygonAndCircle` ports the circle case. The clip order, the
  reference-face choice, and the contact-id features match Box2D exactly.
  `gb_polygon_test.cu` checks the two-point box-on-box manifold and the
  polygon-circle manifold at 0 ULP, including the local normal, the plane point, both
  clip points, and the ids.
- **Broad-phase.** A polygon needs a tight AABB from its rotated vertices.
  `gbPolygonComputeAABB` in `gb_polygon.cuh` produces it. The tree in
  `gb_broadphase.cuh` is shape-agnostic and needs no change.

## The two-point block solver (shipped)

Circles produce one contact point. Polygons produce up to two, and Box2D solves a
two-point manifold with a block solver that handles both points together when the
matrix is well conditioned. The solver in `gb_contact_solver.cuh` carries both paths.

- **Constraint.** `GBConstraint` in `gb_contact_types.cuh` carries two velocity
  points (`p`, `p2`), the 2x2 block matrix `K`, its inverse `normalMass22`, and a
  `pointCount` of 1 or 2. A 1-point contact uses point 0 alone, so the single-point
  result stays byte-identical.
- **Block solve.** `gbSolveVelocity` runs the friction solve per point, then for a
  two-point contact runs the total-enumeration LCP through the fixed four-case
  cascade from `b2ContactSolver::SolveVelocityConstraints`. The fallback branch taken
  is part of the result, so the cascade runs in Box2D order.
- **Two-point position solve.** `gbSolvePosition` solves both points in order with
  the face-A and face-B position manifolds, feeding the same `minSeparation` fold.
- **Test.** `gb_block_solver_test.cu` drives the full velocity and position spine on
  a box resting on the ground (a two-point face manifold) and compares every body
  kinematic and both warm-start impulses against the Box2D 2.3.0 reference at 0 ULP.

## The revolute joint (shipped, point-to-point)

`gb_joint.cuh` ports `b2RevoluteJoint`'s point-to-point case: the two-degree-of-
freedom anchor constraint solved with the 2x2 mass matrix in both the velocity solve
and the position solve.

- **Storage.** `GBRevoluteJoint` carries the island-local body indices, the body-local
  anchors, the inverse mass and inertia, the solver arms `rA` / `rB`, the 2x2 mass
  matrix, and the accumulated impulse for warm-start.
- **Solve.** `gbRevoluteInitVelocity` builds the arms and the mass matrix and applies
  the warm-start impulse; `gbRevoluteSolveVelocity` runs one velocity iteration
  through `gbMat22Solve`; `gbRevoluteSolvePosition` runs one position iteration and
  returns whether the position error is within the linear slop.
- **Test.** `gb_joint_test.cu` swings a two-body pendulum over hundreds of substeps
  and compares the bob velocity, angular velocity, position, angle, and the
  warm-start impulse against the Box2D 2.3.0 reference at 0 ULP.

## Adding the next module

The same path extends the engine further.

- **More joint types.** The revolute motor and angle-limit rows add the 3x3 path
  (`b2Mat33::Solve22` / `Solve33`) on top of the point-to-point joint. Prismatic,
  distance, weld, and pulley each have their own `InitVelocityConstraints`,
  `SolveVelocityConstraints`, and `SolvePositionConstraints`. Box2D solves joints
  before contacts within each velocity and position iteration, in joint-list order;
  reproduce that interleave in the per-island driver.
- **Island assembly with joints.** Extend the DFS in `gb_island.cuh` to walk joint
  edges as well as contact edges, so joint-connected bodies land in the same island
  in Box2D's graph-traversal order.
- **More shapes.** The chain shape and the general edge (with vertex0/vertex3
  connectivity) follow the polygon pattern: shape data behind new accessors, a
  narrow-phase ported in order, and a tight AABB helper.

## Growing the bounds

Polygons and joints enlarge the per-world working set. The bounds are compile-time
defines in `gb_pools.cuh` (`GB_MAX_BODIES`, `GB_MAX_CONTACTS`) and `gb_polygon.cuh`
(`GB_MAX_POLYGON_VERTICES`). Raising them grows `WorldShared`. If the result exceeds
the 48 KB default shared budget, use the 100 KB shared opt-in on sm_86 and later, or
move cold per-world state (infrequently touched fields) to global storage behind the
accessors. The shared budget is the design constraint to track. Because every access
goes through the accessors, moving a field between shared and global is a backend
change that leaves every call site untouched.

## Checklist for a new module

1. Write the header in `include/gpu_box2d/`, against the accessors, matching Box2D
   2.3.0 math and order.
2. Write `test/<module>_test.cu` that compares against the Box2D 2.3.0 reference and
   asserts 0 ULP.
3. Build with the frozen flags (`--fmad=false -prec-div=true -prec-sqrt=true`).
4. When the micro-test is green, include the header in `gb_step.cuh` and re-run the
   full gate. If a gate reddens, revert the single include and find the divergence.
