// gb_step.cuh. The assembled physics core, the single integration point. It wires the
// validated modules into gb_world_step:
//     gb_world_step(w) = gbCollidePhase (broad-phase + narrow-phase)
//                        -> gbWorldSolve (island assembly + contact solver)
//                        -> gbWorldSolveTOI (CCD)
// Every module is 0-ULP micro-tested. This file owns only the glue: the broad-phase
// pairing loop (gbCollidePhase) that fills the contact pool, and the gb_world_step
// composition. Everything else lives in the modules.
#pragma once
#include "gpu_box2d/gb_pools.cuh"
#include "gpu_box2d/gb_contact_types.cuh"
#include "gpu_box2d/gb_math.cuh"

// gb_collision.cuh is the canonical owner of gbWorldManifoldInit; suppress the
// fallback copy in gb_contact_solver.cuh so there is exactly one definition.
#define GB_COLLISION_PROVIDED 1

// ---- module includes --------------------------------------------------------
// Order matters for the shared-helper guards: gb_collision provides
// gbWorldManifoldInit; gb_contact_solver provides gbSyncTransform (sets
// GB_SYNC_TRANSFORM_PROVIDED), so gb_toi's copy yields.
#include "gpu_box2d/gb_collision.cuh"       // gbCollideCircles/EdgeAndCircle/worldManifold/contactUpdate
#include "gpu_box2d/gb_broadphase.cuh"      // GbDynTree/GbBroadPhase (faithful pair order; brute path here)
#include "gpu_box2d/gb_contact_solver.cuh"  // per-iteration Gauss-Seidel phases (serial spine)
#include "gpu_box2d/gb_island.cuh"          // gbWorldSolve (DFS + island Solve + folds)
#include "gpu_box2d/gb_toi.cuh"             // GJK + b2TOI geometry primitives (gbTOI/gbContactProxy/...)

// ---- contact materials (Box2D mixing, reference-exact). Fruit and wall defaults.
// (Per-fixture material fields are a later refinement; these values match the
//  validated CPU reference, so the assembled core stays bit-faithful.)
#define GB_FRUIT_FRICTION    0.265f
#define GB_FRUIT_RESTITUTION 0.02f
#define GB_WALL_FRICTION     0.2f
#define GB_WALL_RESTITUTION  0.0f

// ---- fat-AABB helpers (brute-force broad-phase, fat-AABB-equivalent to the tree) --
GB_HD inline void gbCircleAABB(GBWorld& w, int i, float& lo_x, float& lo_y, float& hi_x, float& hi_y){
    float r = gbCircleRadius(w, i);
    float cx = BODY(w,sweepCx,i), cy = BODY(w,sweepCy,i);
    lo_x=cx-r-GB_AABB_EXTENSION; lo_y=cy-r-GB_AABB_EXTENSION;
    hi_x=cx+r+GB_AABB_EXTENSION; hi_y=cy+r+GB_AABB_EXTENSION;
}
GB_HD inline void gbEdgeAABB(GBWorld& w, int e, float& lo_x, float& lo_y, float& hi_x, float& hi_y){
    float ax=EDGE(w,edgeAx,e), ay=EDGE(w,edgeAy,e), bx=EDGE(w,edgeBx,e), by=EDGE(w,edgeBy,e);
    lo_x=b2MinF(ax,bx)-GB_POLYGON_RADIUS-GB_AABB_EXTENSION; lo_y=b2MinF(ay,by)-GB_POLYGON_RADIUS-GB_AABB_EXTENSION;
    hi_x=b2MaxF(ax,bx)+GB_POLYGON_RADIUS+GB_AABB_EXTENSION; hi_y=b2MaxF(ay,by)+GB_POLYGON_RADIUS+GB_AABB_EXTENSION;
}
GB_HD inline bool gbAabbOvl(float a_lo_x,float a_lo_y,float a_hi_x,float a_hi_y,
                            float b_lo_x,float b_lo_y,float b_hi_x,float b_hi_y){
    return !(b_lo_x>a_hi_x || b_lo_y>a_hi_y || a_lo_x>b_hi_x || a_lo_y>b_hi_y);
}
GB_HD inline int gbFindContact(GBWorld& w, int bodyA, int bodyB, int edge){
    for (int i=0;i<SCAL(w,contactCount);++i)
        if (CONT(w,cBodyA,i)==bodyA && CONT(w,cBodyB,i)==bodyB && CONT(w,cEdge,i)==edge) return i;
    return -1;
}

// ============================================================================
// gbCollidePhase. b2ContactManager Collide + FindNewContacts (brute-force fat-AABB),
// byte-identical to the validated collide phase: create new overlapping pairs
// (fruit-fruit in a<b order, then fruit-wall), destroy separated, then run
// gbContactUpdate on each surviving contact.
// ============================================================================
GB_HD inline void gbCollidePhase(GBWorld& w){
    int bc = SCAL(w,bodyCount);
    // 1) new contacts - fruit-fruit
    for (int a=1;a<bc;++a){
        if (!BODY(w,alive,a)) continue;
        float al,at,ar,ab; gbCircleAABB(w,a,al,at,ar,ab);
        for (int b=a+1;b<bc;++b){
            if (!BODY(w,alive,b)) continue;
            float bl,bt,br,bb; gbCircleAABB(w,b,bl,bt,br,bb);
            if (!gbAabbOvl(al,at,ar,ab, bl,bt,br,bb)) continue;
            if (gbFindContact(w,a,b,-1)>=0) continue;
            int ci=SCAL(w,contactCount)++;
            CONT(w,cBodyA,ci)=a; CONT(w,cBodyB,ci)=b; CONT(w,cEdge,ci)=-1; CONT(w,cTouching,ci)=0;
            CONT(w,cFriction,ci)=b2MixFriction(GB_FRUIT_FRICTION,GB_FRUIT_FRICTION);
            CONT(w,cRestitution,ci)=b2MixRestitution(GB_FRUIT_RESTITUTION,GB_FRUIT_RESTITUTION);
            CONT(w,cNormalImpulse,ci)=0.0f; CONT(w,cTangentImpulse,ci)=0.0f;
            CONT(w,cEnabled,ci)=1; CONT(w,cToi,ci)=1.0f; CONT(w,cToiCount,ci)=0; CONT(w,cToiFlag,ci)=0;
        }
    }
    // fruit-wall
    for (int e=0;e<GB_N_EDGES;++e){
        float el,et,er,eb; gbEdgeAABB(w,e,el,et,er,eb);
        for (int b=1;b<bc;++b){
            if (!BODY(w,alive,b)) continue;
            float bl,bt,br,bb; gbCircleAABB(w,b,bl,bt,br,bb);
            if (!gbAabbOvl(el,et,er,eb, bl,bt,br,bb)) continue;
            if (gbFindContact(w,GB_GROUND,b,e)>=0) continue;
            int ci=SCAL(w,contactCount)++;
            CONT(w,cBodyA,ci)=GB_GROUND; CONT(w,cBodyB,ci)=b; CONT(w,cEdge,ci)=e; CONT(w,cTouching,ci)=0;
            CONT(w,cFriction,ci)=b2MixFriction(GB_WALL_FRICTION,GB_FRUIT_FRICTION);
            CONT(w,cRestitution,ci)=b2MixRestitution(GB_WALL_RESTITUTION,GB_FRUIT_RESTITUTION);
            CONT(w,cNormalImpulse,ci)=0.0f; CONT(w,cTangentImpulse,ci)=0.0f;
            CONT(w,cEnabled,ci)=1; CONT(w,cToi,ci)=1.0f; CONT(w,cToiCount,ci)=0; CONT(w,cToiFlag,ci)=0;
        }
    }
    // 2) destroy separated (fat-AABB no longer overlaps), compact-keep
    int n=0;
    for (int i=0;i<SCAL(w,contactCount);++i){
        int a=CONT(w,cBodyA,i), b=CONT(w,cBodyB,i), e=CONT(w,cEdge,i);
        bool keep;
        if (e<0){
            float al,at,ar,ab; gbCircleAABB(w,a,al,at,ar,ab);
            float bl,bt,br,bb; gbCircleAABB(w,b,bl,bt,br,bb);
            keep = BODY(w,alive,a) && BODY(w,alive,b) && gbAabbOvl(al,at,ar,ab, bl,bt,br,bb);
        } else {
            float el,et,er,eb; gbEdgeAABB(w,e,el,et,er,eb);
            float bl,bt,br,bb; gbCircleAABB(w,b,bl,bt,br,bb);
            keep = BODY(w,alive,b) && gbAabbOvl(el,et,er,eb, bl,bt,br,bb);
        }
        if (!keep) continue;
        if (n!=i){
            CONT(w,cBodyA,n)=CONT(w,cBodyA,i); CONT(w,cBodyB,n)=CONT(w,cBodyB,i); CONT(w,cEdge,n)=CONT(w,cEdge,i);
            CONT(w,cTouching,n)=CONT(w,cTouching,i);
            CONT(w,cFriction,n)=CONT(w,cFriction,i); CONT(w,cRestitution,n)=CONT(w,cRestitution,i);
            CONT(w,cManifoldType,n)=CONT(w,cManifoldType,i);
            CONT(w,cLocalNormalX,n)=CONT(w,cLocalNormalX,i); CONT(w,cLocalNormalY,n)=CONT(w,cLocalNormalY,i);
            CONT(w,cLocalPointX,n)=CONT(w,cLocalPointX,i); CONT(w,cLocalPointY,n)=CONT(w,cLocalPointY,i);
            CONT(w,cPointLocalX,n)=CONT(w,cPointLocalX,i); CONT(w,cPointLocalY,n)=CONT(w,cPointLocalY,i);
            CONT(w,cNormalImpulse,n)=CONT(w,cNormalImpulse,i); CONT(w,cTangentImpulse,n)=CONT(w,cTangentImpulse,i);
            CONT(w,cToi,n)=CONT(w,cToi,i); CONT(w,cToiCount,n)=CONT(w,cToiCount,i);
            CONT(w,cToiFlag,n)=CONT(w,cToiFlag,i); CONT(w,cEnabled,n)=CONT(w,cEnabled,i);
        }
        ++n;
    }
    SCAL(w,contactCount)=n;
    // 3) update each surviving contact (narrow-phase + touching + warm-start carry + hook)
    for (int i=0;i<SCAL(w,contactCount);++i) gbContactUpdate(w, i);
}

// ============================================================================
// TOI ORCHESTRATION (gbIslandSolveTOI + gbWorldSolveTOI), the world/island SolveTOI
// path. gb_toi.cuh provides the GJK/TOI geometry primitives (gbTOI, gbContactProxy,
// gbBodySweep, gbSweepAdvance, gbWriteSweepAdvance); this orchestration is the glue
// that drives them with the assembled solver, island, and collision. Ported
// byte-faithfully from the validated CCD path onto the accessor contract.
// ============================================================================
#include "gpu_box2d/gb_toi_orchestration.inc"   // gbIslandSolveTOI + gbWorldSolveTOI

// ============================================================================
// gb_world_step - one b2World::Step: Collide -> Solve -> SolveTOI.
// The FROZEN step signature (declared in gb_world.cuh). This is the assembled core.
// ============================================================================
GB_HD inline void gb_world_step(GBWorld& w){
    gbCollidePhase(w);
    if (SCAL(w,stepComplete)) gbWorldSolve(w);
    gbWorldSolveTOI(w);
}
