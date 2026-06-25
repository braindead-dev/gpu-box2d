// gb_chain_step_test.cu. Integration test for the chain shape wired into the assembled
// step (-DGB_ENABLE_POLYGONS -DGB_ENABLE_CHAIN). It proves the chain is live as a
// per-world static collider: gbWorldSetChain loads a chain into the world's edge
// fixtures with adjacency, and gb_world_step collides bodies against the chain child
// edges through the adjacency-aware edge-polygon collider.
//
// The child-edge generation is 0-ULP validated in gb_chain_shape_test. This test checks
// that the assembled step uses it and settles to a physically correct rest.
//
// Checks:
//   (A) a flat chain (three collinear segments) settles a box at the same height as a
//       flat floor, so wiring the chain does not change flat-ground behavior.
//   (B) a V-shaped chain (a valley) catches a box dropped from one side and settles it
//       near the bottom of the valley, so the chain collides as a contour and the
//       adjacency carries across the interior vertices.
//
// Build (frozen flags), self-contained:
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -DGB_ENABLE_POLYGONS -DGB_ENABLE_CHAIN -Iinclude -Itest \
//        test/gb_chain_step_test.cu -o test/gb_chain_step_test
//   ./test/gb_chain_step_test
//   Expected: PASS gb_chain_step
#include "gpu_box2d/gb_step.cuh"
#include "gpu_box2d/gb_chain_shape.cuh"
#include "gpu_box2d/gb_polygon.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>

static int gFails = 0;
static void expect(const char* what, bool ok, double got){
    if (!ok){ printf("  FAIL %-32s value=%.6f\n", what, got); gFails = 1; }
    else     printf("  ok   %-32s value=%.6f\n", what, got);
}

static void syncXf(WorldShared& w, int s){
    float a = w.sweepA[s];
    w.xfQs[s] = sinf(a); w.xfQc[s] = cosf(a);
    w.xfPx[s] = w.sweepCx[s]; w.xfPy[s] = w.sweepCy[s];
}

static void setGround(WorldShared& w){
    w.bodyType[GB_GROUND] = GB_STATIC_BODY;
    w.shapeType[GB_GROUND] = GB_SHAPE_CIRCLE;
    w.invMass[GB_GROUND] = 0.0f; w.invI[GB_GROUND] = 0.0f;
    w.alive[GB_GROUND] = 1; w.awake[GB_GROUND] = 0;
    w.xfQc[GB_GROUND] = 1.0f; w.xfQs[GB_GROUND] = 0.0f;
}

static void setBox(WorldShared& w, int s, float hx, float hy, float x, float y, float density){
    GBPolygon p; gbPolygonSetAsBox(p, hx, hy);
    GBMassData md; gbPolygonComputeMass(p, md, density);
    w.bodyType[s] = GB_DYNAMIC_BODY;
    w.shapeType[s] = GB_SHAPE_POLYGON;
    w.polyCount[s] = p.count; w.polyRadius[s] = p.radius;
    w.polyCentroidX[s] = p.centroid.x; w.polyCentroidY[s] = p.centroid.y;
    for (int i = 0; i < p.count; ++i){
        int vs = gbPolyVertSlot(s, i);
        w.polyVx[vs] = p.vertices[i].x; w.polyVy[vs] = p.vertices[i].y;
        w.polyNx[vs] = p.normals[i].x;  w.polyNy[vs] = p.normals[i].y;
    }
    w.invMass[s] = md.mass > 0.0f ? 1.0f/md.mass : 0.0f;
    w.invI[s]    = md.I > 0.0f ? 1.0f/md.I : 0.0f;
    w.alive[s] = 1; w.awake[s] = 1; w.sleepTime[s] = 0.0f;
    w.sweepCx[s]=x; w.sweepCy[s]=y; w.sweepC0x[s]=x; w.sweepC0y[s]=y;
    w.sweepA[s]=0.0f; w.sweepA0[s]=0.0f;
    w.velX[s]=0.0f; w.velY[s]=0.0f; w.angVel[s]=0.0f;
    syncXf(w, s);
}

int main(){
    printf("Chain-step integration test: chain wired into gb_world_step\n\n");

    // ---- (A) flat chain settles a box like a flat floor ---------------------
    {
        WorldShared w; std::memset(&w, 0, sizeof(WorldShared));
        setGround(w);
        // a flat chain: four collinear vertices -> three collinear child edges
        GBChainShape chain;
        V2 verts[4] = { v2(-5,0), v2(-1.67f,0), v2(1.67f,0), v2(5,0) };
        gbChainCreateChain(chain, verts, 4);
        int nEdges = gbWorldSetChain(w, chain);
        w.bodyCount = 2;
        setBox(w, 1, 0.5f, 0.5f, 0.0f, 3.0f, 1.0f);
        for (int s = 0; s < 400; ++s) gb_world_step(w);
        float boxY = w.sweepCy[1];
        printf("(A) flat chain (%d child edges)\n", nEdges);
        expect("box rests on the flat chain", fabsf(boxY - 0.5f) < 0.05f, boxY);
        expect("box stays finite", std::isfinite(boxY), boxY);
    }

    // ---- (B) V-shaped chain catches a box in the valley ---------------------
    {
        WorldShared w; std::memset(&w, 0, sizeof(WorldShared));
        setGround(w);
        // a valley: down to the middle then up. Three child edges meet at two interior
        // vertices, where the adjacency keeps a sliding box from catching.
        GBChainShape chain;
        V2 verts[4] = { v2(-4.0f, 3.0f), v2(-1.5f, 0.0f), v2(1.5f, 0.0f), v2(4.0f, 3.0f) };
        gbChainCreateChain(chain, verts, 4);
        int nEdges = gbWorldSetChain(w, chain);
        w.bodyCount = 2;
        // drop the box above the left slope so it slides toward the bottom
        setBox(w, 1, 0.4f, 0.4f, -2.0f, 4.0f, 1.0f);
        for (int s = 0; s < 600; ++s) gb_world_step(w);
        float boxX = w.sweepCx[1];
        float boxY = w.sweepCy[1];
        printf("(B) V-shaped chain (%d child edges)\n", nEdges);
        // the box should come to rest in the flat bottom of the valley (|x| < 1.5) and
        // above the floor (y near the box half-extent plus skin)
        expect("box settles in the valley bottom", fabsf(boxX) < 1.5f, boxX);
        expect("box rests near the valley floor", boxY > 0.3f && boxY < 1.0f, boxY);
        expect("box stays finite", std::isfinite(boxX) && std::isfinite(boxY), boxY);
    }

    if (!gFails){
        printf("\nPASS gb_chain_step: chain is live as a per-world collider in gb_world_step\n");
        return 0;
    }
    printf("\nFAIL gb_chain_step: see above\n");
    return 1;
}
