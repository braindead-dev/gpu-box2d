// gb_wired_step_test.cu. Integration test for the assembled step with polygons and
// joints wired in (-DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS). It proves the dispatch is
// live: gb_world_step drives the polygon narrow-phase, the two-point block solver, and
// the revolute joint solve, on a WorldShared built through the accessor contract.
//
// The per-module 0-ULP fidelity is established by gb_polygon_test, gb_block_solver_test,
// and gb_joint_test. This test checks that the assembled step activates those paths and
// settles to a stable, physically correct rest, so the support is part of the engine.
//
// Checks:
//   (A) a box resting on the ground produces a two-point polygon contact through
//       gbContactUpdate and the box settles to rest above the floor.
//   (B) a body pinned to the ground by a revolute joint stays at its anchor distance
//       while swinging under gravity (the joint constraint holds).
//
// Build (frozen flags), self-contained (no reference needed):
//   nvcc -O2 --fmad=false -prec-div=true -prec-sqrt=true -arch=sm_86 \
//        -DGB_ENABLE_POLYGONS -DGB_ENABLE_JOINTS -Iinclude -Itest \
//        test/gb_wired_step_test.cu -o test/gb_wired_step_test
//   ./test/gb_wired_step_test
//   Expected: PASS gb_wired_step
#include "gpu_box2d/gb_step.cuh"
#include "gpu_box2d/gb_polygon.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>

static int gFails = 0;
static void expect(const char* what, bool ok, double got){
    if (!ok){ printf("  FAIL %-28s value=%.6f\n", what, got); gFails = 1; }
    else     printf("  ok   %-28s value=%.6f\n", what, got);
}

// set a body as a static circle ground at slot 0 with three edges (floor + walls).
static void setGround(WorldShared& w){
    w.bodyType[GB_GROUND] = GB_STATIC_BODY;
    w.shapeType[GB_GROUND] = GB_SHAPE_CIRCLE;   // ground collides through its edge fixtures
    w.invMass[GB_GROUND] = 0.0f; w.invI[GB_GROUND] = 0.0f;
    w.alive[GB_GROUND] = 1; w.awake[GB_GROUND] = 0;
    w.xfQc[GB_GROUND] = 1.0f; w.xfQs[GB_GROUND] = 0.0f;
    // floor at y=0 from x=-5 to x=5
    w.edgeAx[0]=-5.0f; w.edgeAy[0]=0.0f; w.edgeBx[0]=5.0f; w.edgeBy[0]=0.0f;
    w.edgeAx[1]=-5.0f; w.edgeAy[1]=0.0f; w.edgeBx[1]=-5.0f; w.edgeBy[1]=6.0f;
    w.edgeAx[2]= 5.0f; w.edgeAy[2]=0.0f; w.edgeBx[2]= 5.0f; w.edgeBy[2]=6.0f;
}

static void syncXf(WorldShared& w, int s){
    float a = w.sweepA[s];
    w.xfQs[s] = sinf(a); w.xfQc[s] = cosf(a);
    w.xfPx[s] = w.sweepCx[s]; w.xfPy[s] = w.sweepCy[s];
}

// place a box body (polygon) at slot s.
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

// place a circle body at slot s.
static void setCircle(WorldShared& w, int s, float r, float x, float y){
    w.bodyType[s] = GB_DYNAMIC_BODY;
    w.shapeType[s] = GB_SHAPE_CIRCLE;
    w.radius[s] = r;
    float mass = 3.14159265359f * r * r;   // unit-density disk mass
    w.invMass[s] = 1.0f/mass; w.invI[s] = 1.0f/(mass*0.5f*r*r);
    w.alive[s]=1; w.awake[s]=1; w.sleepTime[s]=0.0f;
    w.sweepCx[s]=x; w.sweepCy[s]=y; w.sweepC0x[s]=x; w.sweepC0y[s]=y;
    w.sweepA[s]=0.0f; w.sweepA0[s]=0.0f; w.velX[s]=0; w.velY[s]=0; w.angVel[s]=0;
    syncXf(w, s);
}

int main(){
    printf("Wired-step integration test: polygons + revolute joint through gb_world_step\n\n");

    // (A) box settles on the ground via a two-point polygon contact.
    {
        WorldShared w; memset(&w, 0, sizeof(w));
        setGround(w);
        setBox(w, 1, 0.5f, 0.5f, 0.0f, 0.7f, 1.0f);   // box just above the floor
        w.bodyCount = 2; w.contactCount = 0; w.stepComplete = 1;
        w.jointCount = 0;

        int sawTwoPoint = 0;
        float minY = 1e9f;
        for (int step = 0; step < 400; ++step){
            gb_world_step(w);
            for (int c = 0; c < w.contactCount; ++c)
                if (w.cTouching[c] && w.cPointCount[c] == 2) sawTwoPoint = 1;
        }
        float restY = w.sweepCy[1];
        // box half-height 0.5 plus the polygon skin: the rest center sits just above
        // 0.5 (the floor face plus both skins), with no penetration and no tunneling.
        bool stable = std::isfinite(restY) && restY > 0.5f && restY < 0.53f;
        (void)minY;
        printf("(A) box-on-ground\n");
        expect("two-point contact seen", sawTwoPoint == 1, sawTwoPoint);
        expect("box rests on the floor face", stable, restY);
        expect("box stays finite", std::isfinite(restY), restY);
    }

    // (A2) box stacked on a box: the body-body polygon-polygon path settles to a stack.
    {
        WorldShared w; memset(&w, 0, sizeof(w));
        setGround(w);
        setBox(w, 1, 0.6f, 0.4f, 0.0f, 0.42f, 1.0f);   // lower box on the floor
        setBox(w, 2, 0.5f, 0.5f, 0.0f, 1.45f, 1.0f);   // upper box on the lower box
        w.bodyCount = 3; w.contactCount = 0; w.stepComplete = 1; w.jointCount = 0;
        int sawPolyPoly = 0;
        for (int step = 0; step < 500; ++step){
            gb_world_step(w);
            for (int c = 0; c < w.contactCount; ++c)
                if (w.cTouching[c] && w.cEdge[c] < 0 && w.cPointCount[c] == 2) sawPolyPoly = 1;
        }
        float lower = w.sweepCy[1], upper = w.sweepCy[2];
        printf("(A2) box-on-box stack\n");
        expect("polygon-polygon contact seen", sawPolyPoly == 1, sawPolyPoly);
        expect("lower box rests", std::isfinite(lower) && lower > 0.4f && lower < 0.46f, lower);
        expect("upper box above lower", std::isfinite(upper) && upper > lower + 0.7f, upper);
    }

    // (A3) circle resting on a box: the polygon-circle body-body path settles.
    {
        WorldShared w; memset(&w, 0, sizeof(w));
        setGround(w);
        setBox(w, 1, 1.0f, 0.3f, 0.0f, 0.32f, 1.0f);   // wide low box on the floor
        setCircle(w, 2, 0.4f, 0.0f, 1.1f);             // circle dropped onto it
        w.bodyCount = 3; w.contactCount = 0; w.stepComplete = 1; w.jointCount = 0;
        for (int step = 0; step < 500; ++step) gb_world_step(w);
        float boxY = w.sweepCy[1], cy = w.sweepCy[2];
        printf("(A3) circle-on-box\n");
        // circle rest center near box-top (0.32+0.3=0.62) + circle radius 0.4 ~= 1.02
        expect("circle rests on box", std::isfinite(cy) && cy > boxY + 0.5f && cy < boxY + 0.8f, cy);
        expect("box rests on floor", std::isfinite(boxY) && boxY > 0.3f && boxY < 0.36f, boxY);
    }

    // (B) revolute joint holds anchor distance while the body swings.
    {
        WorldShared w; memset(&w, 0, sizeof(w));
        setGround(w);
        // dynamic bob (circle) at (1.5, 3.0); pivot anchor at the world point (0, 3.0).
        setCircle(w, 1, 0.3f, 1.5f, 3.0f);
        w.bodyCount = 2; w.contactCount = 0; w.stepComplete = 1;
        // joint: bodyA = ground (static slot 0), bodyB = bob. Anchor world (0,3).
        // localAnchorA = anchor - groundPos = (0,3) - (0,0) = (0,3).
        // localAnchorB = anchor - bobPos   = (0,3) - (1.5,3) = (-1.5,0).
        w.jointCount = 1;
        w.jBodyA[0]=GB_GROUND; w.jBodyB[0]=1;
        w.jLocalAnchorAX[0]=0.0f; w.jLocalAnchorAY[0]=3.0f;
        w.jLocalAnchorBX[0]=-1.5f; w.jLocalAnchorBY[0]=0.0f;
        w.jImpulseX[0]=0.0f; w.jImpulseY[0]=0.0f;
        // ground must be awake-eligible as a joint neighbor; keep it static (no seed).

        float maxErr = 0.0f;
        float anchorX=0.0f, anchorY=3.0f, armLen=1.5f;
        for (int step = 0; step < 300; ++step){
            gb_world_step(w);
            float dx = w.sweepCx[1]-anchorX, dy = w.sweepCy[1]-anchorY;
            float dist = sqrtf(dx*dx + dy*dy);
            float err = fabsf(dist - armLen);
            if (err > maxErr) maxErr = err;
        }
        printf("(B) revolute pendulum\n");
        // the point-to-point joint holds the bob at ~armLen from the pivot. Box2D's
        // soft position solve permits a small bounded drift (a few linear slops).
        expect("anchor distance held", std::isfinite(maxErr) && maxErr < 0.05f, maxErr);
        expect("bob moved (swing happened)", std::isfinite(w.sweepCy[1]) && w.sweepCy[1] < 2.99f, w.sweepCy[1]);
    }

    if (!gFails){ printf("\nPASS gb_wired_step: polygons and the revolute joint are live in gb_world_step\n"); return 0; }
    printf("\nFAIL gb_wired_step: see above\n");
    return 1;
}
