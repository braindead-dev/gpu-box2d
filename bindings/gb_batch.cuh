// gb_batch.cuh. A generic batched-world driver over the gpu-box2d core. It owns an
// array of WorldShared (one per world), seeds bodies and the static world boundary,
// steps every world through gb_world_step, and exposes per-world body state as flat
// arrays. The API is game-agnostic: it speaks bodies (position, angle, velocity, mass,
// shape) and the step, with no application concept baked in. An RL or simulation layer
// drives N worlds through it and reads the state out as tensors.
//
// Backend. This driver uses the contiguous WorldShared block backend, so it runs on a
// host CPU with no GPU (the step is GB_HD and runs host-side). The same field set and
// accessor contract back the SoA-global production path, so a CUDA build can step the
// same seeded state on the device; see the note at gbBatchStep. The binding layer
// (gb_pybind.cpp) wraps this driver for Python.
//
// Build flags (FROZEN for fidelity): nvcc --fmad=false -prec-div=true -prec-sqrt=true,
// or the host equivalent -ffp-contract=off -mfpmath=sse on x86.
#pragma once
#include "gpu_box2d/gb_step.cuh"
#include <vector>
#include <cstring>

// A batched set of NW independent Box2D worlds. Each world is a WorldShared with up to
// GB_MAX_BODIES bodies, GB_N_EDGES static ground edges, and (with the joint feature)
// GB_MAX_JOINTS joints. Slot 0 is the static ground body by the core's convention.
struct GBBatch {
    int NW;
    std::vector<WorldShared> worlds;   // [NW]

    explicit GBBatch(int nWorlds) : NW(nWorlds), worlds(nWorlds){
        for (int i = 0; i < NW; ++i) gbBatchResetWorld(worlds[i]);
    }

    // Zero a world to a clean empty state: one static ground body at slot 0, no other
    // bodies, no contacts, no joints, ready to step.
    static void gbBatchResetWorld(WorldShared& w){
        std::memset(&w, 0, sizeof(WorldShared));
        w.bodyCount = 1;            // slot 0 = static ground
        w.bodyType[GB_GROUND] = GB_STATIC_BODY;
        w.alive[GB_GROUND] = 1;
        w.awake[GB_GROUND] = 0;
        w.contactCount = 0;
        w.stepComplete = 1;
#ifdef GB_ENABLE_JOINTS
        w.jointCount = 0;
#endif
        // ground edges default to a single flat floor segment; an application overrides
        // them with gbBatchSetGroundEdge. The unused edge slots collapse to a point so
        // they create no spurious contacts.
        for (int e = 0; e < GB_N_EDGES; ++e){
            w.edgeAx[e] = 0.0f; w.edgeAy[e] = 0.0f; w.edgeBx[e] = 0.0f; w.edgeBy[e] = 0.0f;
        }
    }
};

// ---- world setup (host-side seeding) ---------------------------------------
// Set a static ground edge (e in [0, GB_N_EDGES)) for one world. The edges are the
// world's static boundary; the core collides bodies against them.
inline void gbBatchSetGroundEdge(GBBatch& b, int world, int e, float ax, float ay, float bx, float by){
    WorldShared& w = b.worlds[world];
    w.edgeAx[e] = ax; w.edgeAy[e] = ay; w.edgeBx[e] = bx; w.edgeBy[e] = by;
}

// Add a circle body to a world and return its body slot. type is GB_STATIC_BODY or
// GB_DYNAMIC_BODY. invMass / invI are the inverse mass and inverse rotational inertia
// (0 for a static or infinitely heavy body). Returns -1 if the world is full.
inline int gbBatchAddCircle(GBBatch& b, int world, float px, float py, float radius,
                            float invMass, float invI, int type){
    WorldShared& w = b.worlds[world];
    int s = w.bodyCount;
    if (s >= GB_MAX_BODIES) return -1;
    w.bodyCount = s + 1;
    w.sweepCx[s] = px; w.sweepCy[s] = py;
    w.sweepC0x[s] = px; w.sweepC0y[s] = py;
    w.sweepA[s] = 0.0f; w.sweepA0[s] = 0.0f; w.sweepAlpha0[s] = 0.0f;
    w.xfPx[s] = px; w.xfPy[s] = py; w.xfQs[s] = 0.0f; w.xfQc[s] = 1.0f;
    w.velX[s] = 0.0f; w.velY[s] = 0.0f; w.angVel[s] = 0.0f;
    w.invMass[s] = invMass; w.invI[s] = invI;
    w.radius[s] = radius;
    w.bodyType[s] = type;
    w.sleepTime[s] = 0.0f;
    w.awake[s] = (type == GB_STATIC_BODY) ? 0 : 1;
    w.alive[s] = 1;
#ifdef GB_ENABLE_POLYGONS
    w.shapeType[s] = GB_SHAPE_CIRCLE;
#endif
    return s;
}

#ifdef GB_ENABLE_POLYGONS
// Add a box (axis-aligned in body frame) body and return its slot. hx / hy are the
// half-extents. Builds the polygon shape (vertices, normals, centroid, radius) the same
// way b2PolygonShape::SetAsBox does, then writes it into the per-body polygon storage.
inline int gbBatchAddBox(GBBatch& b, int world, float px, float py, float hx, float hy,
                         float invMass, float invI, int type){
    WorldShared& w = b.worlds[world];
    int s = w.bodyCount;
    if (s >= GB_MAX_BODIES) return -1;
    w.bodyCount = s + 1;
    w.sweepCx[s] = px; w.sweepCy[s] = py;
    w.sweepC0x[s] = px; w.sweepC0y[s] = py;
    w.sweepA[s] = 0.0f; w.sweepA0[s] = 0.0f; w.sweepAlpha0[s] = 0.0f;
    w.xfPx[s] = px; w.xfPy[s] = py; w.xfQs[s] = 0.0f; w.xfQc[s] = 1.0f;
    w.velX[s] = 0.0f; w.velY[s] = 0.0f; w.angVel[s] = 0.0f;
    w.invMass[s] = invMass; w.invI[s] = invI;
    w.radius[s] = 0.0f;
    w.bodyType[s] = type;
    w.sleepTime[s] = 0.0f;
    w.awake[s] = (type == GB_STATIC_BODY) ? 0 : 1;
    w.alive[s] = 1;
    w.shapeType[s] = GB_SHAPE_POLYGON;
    GBPolygon p; gbPolygonSetAsBox(p, hx, hy);
    w.polyCount[s] = p.count;
    w.polyRadius[s] = p.radius;
    w.polyCentroidX[s] = p.centroid.x; w.polyCentroidY[s] = p.centroid.y;
    for (int i = 0; i < p.count; ++i){
        int vs = gbPolyVertSlot(s, i);
        w.polyVx[vs] = p.vertices[i].x; w.polyVy[vs] = p.vertices[i].y;
        w.polyNx[vs] = p.normals[i].x;  w.polyNy[vs] = p.normals[i].y;
    }
    return s;
}
#endif

// Set a body's linear and angular velocity (after seeding, before stepping).
inline void gbBatchSetVelocity(GBBatch& b, int world, int body, float vx, float vy, float w_ang){
    WorldShared& w = b.worlds[world];
    w.velX[body] = vx; w.velY[body] = vy; w.angVel[body] = w_ang;
    if (w.bodyType[body] != GB_STATIC_BODY) w.awake[body] = 1;
}

// Set a body's angle (orientation). Updates the cached transform rotation.
inline void gbBatchSetAngle(GBBatch& b, int world, int body, float angle){
    WorldShared& w = b.worlds[world];
    w.sweepA[body] = angle; w.sweepA0[body] = angle;
    w.xfQs[body] = sinf(angle); w.xfQc[body] = cosf(angle);
}

#ifdef GB_ENABLE_JOINTS
// Add a revolute joint pinning bodyA and bodyB at the given body-local anchors. Returns
// the joint slot, or -1 if the world's joint pool is full.
inline int gbBatchAddRevoluteJoint(GBBatch& b, int world, int bodyA, int bodyB,
                                   float anchorAx, float anchorAy, float anchorBx, float anchorBy){
    WorldShared& w = b.worlds[world];
    int j = w.jointCount;
    if (j >= GB_MAX_JOINTS) return -1;
    w.jointCount = j + 1;
    w.jBodyA[j] = bodyA; w.jBodyB[j] = bodyB;
    w.jLocalAnchorAX[j] = anchorAx; w.jLocalAnchorAY[j] = anchorAy;
    w.jLocalAnchorBX[j] = anchorBx; w.jLocalAnchorBY[j] = anchorBy;
    w.jImpulseX[j] = 0.0f; w.jImpulseY[j] = 0.0f;
    return j;
}
#endif

// ---- stepping --------------------------------------------------------------
// Step every world `substeps` times. On a CPU build this runs the step host-side per
// world in a plain loop; the step is the same gb_world_step the production SoA-global
// kernel runs, so the per-world result is identical to a single-threaded host Box2D.
// On a CUDA build the same seeded WorldShared array uploads to WorldPoolsSoA and steps
// with gb_launch_thread_step; the seeding API above is unchanged, so a Python user who
// drives this driver on a CPU drives the GPU the same way.
inline void gbBatchStep(GBBatch& b, int substeps = 1){
    for (int s = 0; s < substeps; ++s){
        for (int i = 0; i < b.NW; ++i){
            gb_world_step(b.worlds[i]);
        }
    }
}

// ---- state read-back -------------------------------------------------------
// Fill a flat [NW * GB_MAX_BODIES * 2] array of body positions (x, y), row-major by
// world then body. Slots past a world's bodyCount are written as 0.
inline void gbBatchGetPositions(const GBBatch& b, float* out){
    for (int i = 0; i < b.NW; ++i){
        const WorldShared& w = b.worlds[i];
        for (int s = 0; s < GB_MAX_BODIES; ++s){
            int base = (i * GB_MAX_BODIES + s) * 2;
            if (s < w.bodyCount){ out[base+0] = w.sweepCx[s]; out[base+1] = w.sweepCy[s]; }
            else                { out[base+0] = 0.0f;         out[base+1] = 0.0f; }
        }
    }
}
// Fill a flat [NW * GB_MAX_BODIES] array of body angles.
inline void gbBatchGetAngles(const GBBatch& b, float* out){
    for (int i = 0; i < b.NW; ++i){
        const WorldShared& w = b.worlds[i];
        for (int s = 0; s < GB_MAX_BODIES; ++s)
            out[i*GB_MAX_BODIES + s] = (s < w.bodyCount) ? w.sweepA[s] : 0.0f;
    }
}
// Fill a flat [NW * GB_MAX_BODIES * 3] array of (vx, vy, angVel) per body.
inline void gbBatchGetVelocities(const GBBatch& b, float* out){
    for (int i = 0; i < b.NW; ++i){
        const WorldShared& w = b.worlds[i];
        for (int s = 0; s < GB_MAX_BODIES; ++s){
            int base = (i * GB_MAX_BODIES + s) * 3;
            if (s < w.bodyCount){ out[base+0]=w.velX[s]; out[base+1]=w.velY[s]; out[base+2]=w.angVel[s]; }
            else                { out[base+0]=0.0f;      out[base+1]=0.0f;      out[base+2]=0.0f; }
        }
    }
}
// Fill a flat [NW * GB_MAX_BODIES] array of awake flags (1 awake, 0 asleep).
inline void gbBatchGetAwake(const GBBatch& b, unsigned char* out){
    for (int i = 0; i < b.NW; ++i){
        const WorldShared& w = b.worlds[i];
        for (int s = 0; s < GB_MAX_BODIES; ++s)
            out[i*GB_MAX_BODIES + s] = (s < w.bodyCount) ? w.awake[s] : 0;
    }
}
// Fill a flat [NW] array of each world's live body count (includes the ground slot).
inline void gbBatchGetBodyCount(const GBBatch& b, int* out){
    for (int i = 0; i < b.NW; ++i) out[i] = b.worlds[i].bodyCount;
}
