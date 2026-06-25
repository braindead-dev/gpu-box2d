// gb_settings.cuh. Box2D 2.3.0 global constants (b2Settings.h). The FIDELITY
// CONTRACT: every value here is the exact CPU-engine constant. Do not change.
// Part of the general gpu-box2d core (game-agnostic).
#pragma once

#ifdef __CUDACC__
  #define GB_HD __host__ __device__
#else
  #define GB_HD
#endif

// ---- math/epsilon ----
#define GB_PI                    3.14159265359f
#define GB_EPSILON               1.19209290e-07f   // FLT_EPSILON
#define GB_MAXFLOAT              3.40282347e+38f    // FLT_MAX

// ---- step ----
#define GB_DT                    (1.0f/60.0f)
#define GB_VELOCITY_ITERS        8
#define GB_POSITION_ITERS        3
#define GB_GRAVITY_Y             (-9.81f)

// ---- solver tolerances (b2Settings.h) ----
#define GB_LINEAR_SLOP           0.005f
#define GB_POLYGON_RADIUS        (2.0f * GB_LINEAR_SLOP)   // edge skin = 0.01
#define GB_BAUMGARTE             0.2f
#define GB_TOI_BAUMGARTE         0.75f
#define GB_MAX_LINEAR_CORRECTION 0.2f
#define GB_VELOCITY_THRESHOLD    1.0f
#define GB_MAX_TRANSLATION       2.0f
#define GB_MAX_TRANSLATION_SQ    (GB_MAX_TRANSLATION * GB_MAX_TRANSLATION)
#define GB_MAX_ROTATION          (0.5f * GB_PI)
#define GB_MAX_ROTATION_SQ       (GB_MAX_ROTATION * GB_MAX_ROTATION)

// ---- sleep ----
#define GB_TIME_TO_SLEEP         0.5f
#define GB_LINEAR_SLEEP_TOL      0.01f
#define GB_ANGULAR_SLEEP_TOL     (2.0f / 180.0f * GB_PI)

// ---- broad-phase ----
#define GB_AABB_EXTENSION        0.1f
#define GB_AABB_MULTIPLIER       2.0f
#define GB_MAX_SUBSTEPS          8           // b2_maxSubSteps (CCD)

// ---- body types (match b2BodyType) ----
#define GB_STATIC_BODY           0
#define GB_DYNAMIC_BODY          2

// ---- manifold/shape tags (de-virtualized) ----
#define GB_SHAPE_CIRCLE          0
#define GB_SHAPE_EDGE            1
#define GB_SHAPE_POLYGON         2
#define GB_MANIFOLD_CIRCLES      0
#define GB_MANIFOLD_FACE_A       1
#define GB_MANIFOLD_FACE_B       2

// ---- manifold capacity (b2_maxManifoldPoints) ----
#define GB_MAX_MANIFOLD_POINTS   2

// ---- polygon capacity (b2_maxPolygonVertices) ----
#ifndef GB_MAX_POLYGON_VERTICES
#define GB_MAX_POLYGON_VERTICES  8
#endif
