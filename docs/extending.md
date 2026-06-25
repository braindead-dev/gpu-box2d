# Extending toward full Box2D

The core covers circles, edges, and convex polygons, the single-point and two-point
contact solvers, continuous collision, and the revolute joint. The polygon and joint
paths are wired into the assembled `gb_world_step` behind two build flags. This
document records how the polygon and joint layers were added on the accessor contract,
and gives the concrete path for the next shape, solver row, and joint type. Every step
keeps the bit-identical guarantee.

## Build flags for the broader feature set

The circle-and-edge core is the default build, and it is byte-for-byte the original
engine. The polygon and joint paths are opt-in so a circle-only consumer pays nothing
for them in struct size or code:

```
-DGB_ENABLE_POLYGONS   per-body shape tag, polygon storage, and the polygon dispatch
                       in the narrow-phase, the broad-phase AABB, and the solver radius
-DGB_ENABLE_JOINTS     the per-world revolute joint pool, the joint edge walk in island
                       assembly, and the joint solve interleave
-DGB_ENABLE_CHAIN      the per-edge chain adjacency (vertex0/vertex3 and the has-flags),
                       so a static edge that belongs to a chain collides with its
                       neighbors known to the edge-polygon collider
```

With all flags off, the assembled step compiles and runs exactly as the circle-and-edge
engine, and the per-substep floats are unchanged. With a flag on, the matching storage
appears in `WorldShared` and its SoA mirror behind the accessor macros, and the step
dispatches the new path when a body carries that shape, the world holds a joint, or an
edge carries chain adjacency.

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

## The revolute motor and angle limit (shipped)

`gb_revolute_joint.cuh` ports the full `b2RevoluteJoint`: the point-to-point anchor
block plus the angular motor and limit row, the 3x3 path. The motor drives the relative
angle toward a target speed, clamped to the maximum motor torque; the limit bounds the
relative angle between a lower and an upper stop, which turns the solve into the 3x3
path with the limit impulse clamped by sign at the active stop. The point-to-point-only
header `gb_joint.cuh` stays wired into the assembled step, so that path is unchanged;
this module is the complete joint validated standalone. `gb_revolute_joint_test.cu` runs
a point-to-point pendulum, a motorized joint, and an angle-limited joint over hundreds
of substeps and reproduces every body kinematic plus the three impulse components and
the motor impulse at 0 ULP.

## The distance and weld joints (shipped)

Two more joint types follow the same module pattern: a header against the accessor
contract with init, velocity, and position phases, and a 0-ULP micro-test against the
Box2D 2.3.0 reference.

- **Distance joint.** `gb_distance_joint.cuh` ports `b2DistanceJoint`. The rigid case is
  a single velocity row plus a length-correcting position solve; the soft case folds a
  frequency and damping ratio into a bias and gamma so the velocity row alone carries
  the spring and the position solve is skipped. `gb_distance_joint_test.cu` runs a rigid
  rod and a soft spring at 0 ULP.
- **Weld joint.** `gb_weld_joint.cuh` ports `b2WeldJoint`, the first 3x3 joint. It holds
  the shared anchor (two linear rows) and the relative angle (the angular row) together
  with the 3x3 effective-mass matrix. The rigid case uses the symmetric 3x3 inverse; the
  soft case inverts the 2x2 linear block and carries the angular row through a bias and
  gamma. `gb_weld_joint_test.cu` runs a rigid and a soft weld at 0 ULP, exercising both
  paths.

The 3x3 machinery the weld joint needs lives in `gb_contact_types.cuh`: `GBMat33` with
`Solve33`, `Solve22`, `GetInverse22`, and `GetSymInverse33`, line-faithful to
`b2Mat33`. The revolute motor and angle-limit rows reuse the same matrix.

## The prismatic joint (shipped)

`gb_prismatic_joint.cuh` ports `b2PrismaticJoint`, the slider. It constrains the two
perpendicular-to-axis degrees of freedom and the relative angle and leaves translation
along the axis free, with an optional limit and motor. It carries the most paths of the
joints so far:

- **The 2x2 block** holds the perpendicular and angular rows, always active.
- **The motor row** drives the body along the axis, clamped to the maximum motor force.
- **The limit row** activates the axis constraint at the lower or upper stop, which
  turns the solve into the 3x3 path that solves the limit and the perpendicular block
  together, with the limit impulse clamped by sign.

`gb_prismatic_joint_test.cu` runs a free slider, a slider that falls to its lower limit,
and a motorized slider over hundreds of substeps, and reproduces the body velocity,
angular velocity, position, angle, the three impulse components, and the motor impulse
at 0 ULP, so all three paths are covered.

## Wired into the assembled step (shipped)

With `GB_ENABLE_POLYGONS` and `GB_ENABLE_JOINTS` the assembled `gb_world_step` drives
the new paths end to end.

- **Shape dispatch.** `gbContactUpdate` reads each fixture's shape tag through
  `gbBodyShape` and calls `gbCollideCircles`, `gbCollidePolygonAndCircle`,
  `gbCollidePolygons`, or `gbCollideEdgeAndPolygon` accordingly, writing the manifold
  point count and the second clip point into the contact cache. A ground edge against a
  polygon body uses the dedicated `b2CollideEdgeAndPolygon` port.
- **Broad-phase and solver radius.** The brute-force broad-phase AABB uses the rotated
  polygon vertex bound for a polygon body, and the island constraint load uses the
  polygon skin radius. Both branch on the shape tag and leave the circle path
  untouched.
- **Block solver.** A cached two-point manifold sets the contact's point count to two,
  which routes the contact through the two-point block path in `gbSolveVelocity` and
  the two-point position solve.
- **Joints.** The DFS in `gb_island.cuh` walks joint edges after contact edges, so
  joint-connected bodies share an island, and the island solver runs the revolute
  phases in Box2D's interleave: init contacts, warm-start contacts, init joints; each
  velocity iteration solves joints then contacts; each position iteration solves
  contacts then joints, exiting when both are within tolerance.
- **Test.** `gb_wired_step_test.cu` builds a `WorldShared` through the accessor
  contract and steps it: a box settles on the floor through a two-point manifold, a
  box stacks on a box, a circle rests on a box, and a body pinned by a revolute joint
  swings while holding its anchor distance. This proves the dispatch is live; the
  per-module 0-ULP tests establish the underlying fidelity.

## The edge-polygon narrow-phase (shipped)

`gbCollideEdgeAndPolygon` in `gb_collision.cuh` ports `b2CollideEdgeAndPolygon` (the
`b2EPCollider` class), so a ground edge against a polygon body produces a bit-exact
manifold. It replaces the two-segment-polygon stand-in the wired step used before.

- **Shape data.** `GBEdgeShape` carries the segment (vertex1, vertex2) and the optional
  adjacent vertices (vertex0, vertex3) with their has-flags. The ground edges are single
  segments, so both flags are false.
- **Collide.** The port keeps the full edge-adjacency path: it classifies the edge,
  sets the valid normal range from the adjacent vertices, computes the edge-axis and
  polygon-axis separations, picks the primary axis with the same hysteresis as
  `b2CollidePolygons`, clips the incident edge, and writes a face-A (edge reference) or
  face-B (polygon reference) manifold. The manifold convention matches
  `b2WorldManifold::Initialize`, so the existing world-manifold and solver paths consume
  it unchanged. Keeping the adjacency logic in place means a chain shape reuses this
  routine directly.
- **Test.** `gb_collide_edge_polygon_test.cu` runs a box flat on a horizontal edge
  (face-A, two points), a box on a sloped edge (face-A, two points, rotated frame), a
  large box overhanging a short edge so the polygon face wins (face-B, one point), and a
  separated box (no contact), and reproduces the manifold type, point count, local
  normal, local point, both clip points, and both contact ids at 0 ULP.

## The pulley joint (shipped)

`gb_pulley_joint.cuh` ports `b2PulleyJoint`. It connects two bodies over two fixed
ground anchors with the constraint `lengthA + ratio * lengthB = constant`, so one body
descending lets the other rise, the block-and-tackle and counterweight model. It is a
single constraint row over the two pulley arms: an effective mass that folds both arms
and the ratio, a velocity solve, and a position solve that holds the total length.
`gb_pulley_joint_test.cu` hangs two bodies over two ground anchors and reproduces both
bodies' velocity, position, and the warm-start impulse at 0 ULP over hundreds of
substeps.

## The chain shape (shipped)

`gb_chain_shape.cuh` ports `b2ChainShape`, the standard static world boundary: a ground
contour, a level outline, or a closed loop. A chain is a sequence of connected edge
segments, and it collides through its child edges. `gbChainGetChildEdge` builds the
child edge at an index and wires its vertex0 / vertex3 from the neighbors (or the ghost
vertices at the ends), which is exactly the adjacency the edge-polygon and edge-circle
colliders already consume, so the chain is a thin generator over the validated edge
collider. `gbChainCreateChain` builds an open chain and `gbChainCreateLoop` a closed
loop. `gb_chain_shape_test.cu` reproduces the child-edge generation at 0 ULP against the
Box2D 2.3.0 reference for an open chain and a loop, and confirms a child edge drives the
edge-polygon collider end to end.

With `-DGB_ENABLE_CHAIN` the chain is a live per-world static collider. Each static edge
fixture carries its chain neighbors (vertex0 / vertex3 and the has-flags), so the
edge-polygon collider sees the chain corners. `gbWorldSetChain(w, chain)` loads a chain
into a world's edge fixtures, one child edge per slot with the adjacency filled, through
the accessor contract. The assembled step then collides bodies against the chain child
edges, with the adjacency keeping a body sliding across an interior chain vertex from
catching on the corner. `gb_chain_step_test.cu` settles a box on a flat chain at the
same height as a flat floor and catches a box in a V-shaped chain valley, both through
`gb_world_step`. With the flag off the per-edge layout is byte-identical, so the
non-chain path is unchanged.

## The gear joint (shipped)

`gb_gear_joint.cuh` ports `b2GearJoint`, the last joint type. It couples two other
joints (each revolute or prismatic) with a ratio, so turning one drives the other: the
meshed-gears and rack-and-pinion model. The coupled coordinate is
`coordinateA + ratio * coordinateB = constant`, where each coordinate is a joint angle
(revolute) or a translation (prismatic). It is a four-body constraint over the two driven
bodies and the two reference bodies, with a single-row velocity solve and a position
solve. `gb_gear_joint_test.cu` couples two wheels (each on a revolute joint to a shared
ground) with a ratio and runs the two revolute joints and the gear together in island
order, reproducing both wheels' angular velocity and angle and the gear impulse at 0 ULP
over hundreds of substeps.

## Adding the next module

The same path extends the engine further.

- **Per-point warm-start id matching for polygon contacts.** A polygon contact whose
  clip features change between substeps could carry impulse by matching the contact
  feature id per surviving point, the warm-start refinement Box2D applies. The feature
  ids are already produced by the narrow-phase; the remaining work is the per-point match
  in the contact cache.

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
