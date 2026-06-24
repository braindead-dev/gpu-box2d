// fm_obs.cuh. The observation encoding for the fruit-merge game layer.
//
// Byte-exact to the CPU reference observation fill and its GPU twin. The RL training
// pipeline reads these buffers directly, so a single differing float silently corrupts
// training; this is a byte-for-byte contract.
//
// CONTRACT DISCIPLINE: touches WorldShared ONLY through the frozen accessor macros
// (BODY/SCAL, gb_pools.cuh). No raw arrays, no new globals. Reads GAME fields
// (tier/age/outline) and physics fields (sweepCx/y, velX/y) through the same macros;
// uses the GAME tier->radius map (fm_tier_radius) for the r feature, exactly as the
// CPU does (RADII[tier]).
//
// LOAD-BEARING DETAILS (must not drift):
//   * Normalization denominators: x/FM_WALL_X (3.75), y/FM_DEAD_Y (7.75, the death
//     line; the container height is 9.5), r/2.0, outline/FM_DEATH_TIME (4.0).
//   * feats column order [x/3.75, y/7.75, clamp(vx/10,+/-1), clamp(vy/10,+/-1), r/2,
//     outline/4]. counts divided by 8.0 (incl. zeros). glob = [nf/40, max top, max ol].
//   * mask: 1 = padding, 0 = live (== gpu_env ~alive). Pre-zero tiers/feats/counts;
//     pre-set all mask = 1.
//   * Fruit iteration is in LIVE-CREATION ORDER (slot order 1..bodyCount), capped at
//     min(n_fruits, MAXF). fm_game keeps fruits compacted in creation order, so slot
//     order == the CPU fruit-vector order.
#pragma once
#include "gpu_box2d/gb_pools.cuh"
#include "fm_game.cuh"   // fm_tier_radius + FM_* constants (shared game data)

// obs sizes (mirror the CPU reference: MAXF=64 fruit slots, N_TIERS=12).
#define FM_MAXF     64
#define FM_OBS_NT   FM_N_TIERS   // 12

// clamp helper (mirror the CPU reference clampf exactly).
GB_HD inline float fmClampf(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }

// ============================================================================
// fmFillObs - fill one world's observation slice. Pointers are this world's rows:
//   trow [MAXF]      int32   tiers   (padding = 0)
//   frow [MAXF*6]    float   feats   (padding rows = 0)
//   mrow [MAXF]      uint8   mask    (1 = padding, 0 = live)
//   crow [NT]        float   counts  (per tier / 8.0)
//   g    [3]         float   glob    [nf/40, max top, max ol]
// Identical math and order to the CPU reference observation fill. NT is the tier count
// (FM_OBS_NT = 12).
// ============================================================================
GB_HD inline void fmFillObs(const GBWorld& w, int MAXF, int NT,
                            int* trow, float* frow, unsigned char* mrow,
                            float* crow, float* g){
    for (int j=0;j<MAXF;++j){ trow[j]=0; mrow[j]=1; }   // pre-zero tiers, mask=padding
    for (int j=0;j<MAXF*6;++j) frow[j]=0.0f;
    for (int t=0;t<NT;++t) crow[t]=0.0f;
    int nf=0; float max_top=0.0f, max_ol=0.0f;
    for (int s=1;s<SCAL(w,bodyCount) && nf<MAXF;++s){    // live-creation order, capped at MAXF
        if (!BODY(w,alive,s)) continue;
        int t=BODY(w,tier,s);
        float x=BODY(w,sweepCx,s), y=BODY(w,sweepCy,s);
        float vx=BODY(w,velX,s), vy=BODY(w,velY,s), ol=BODY(w,outline,s);
        float r=fm_tier_radius(t);
        trow[nf]=t; mrow[nf]=0;
        float* f=frow + (size_t)nf*6;
        f[0]=x/FM_WALL_X;  f[1]=y/FM_DEAD_Y;
        f[2]=fmClampf(vx/10.0f,-1.0f,1.0f); f[3]=fmClampf(vy/10.0f,-1.0f,1.0f);
        f[4]=r/2.0f; f[5]=ol/FM_DEATH_TIME;
        crow[t]+=1.0f;
        float top=(y+r)/FM_DEAD_Y; if(top>max_top)max_top=top;
        float olf=ol/FM_DEATH_TIME; if(olf>max_ol)max_ol=olf;
        ++nf;
    }
    for (int t=0;t<NT;++t) crow[t]/=8.0f;
    g[0]=(float)nf/40.0f; g[1]=max_top; g[2]=max_ol;
}

// n_fruits (live fruit count, slot order). Mirror FruitWorld::n_fruits via alive scan.
GB_HD inline int fmNFruits(const GBWorld& w){
    int nf=0;
    for (int s=1;s<SCAL(w,bodyCount);++s) if (BODY(w,alive,s)) ++nf;
    return nf;
}
