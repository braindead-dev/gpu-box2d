// gb_math.cuh. Box2D 2.3.0 b2Math (b2Vec2/b2Rot/b2Transform) ops, faithful.
// General gpu-box2d core. Float32 throughout; compile with --fmad=false to match
// the CPU's -ffp-contract=off -mfpmath=sse (no FMA fusion ->  IEEE rounding parity).
#pragma once
#include "gb_settings.cuh"
#include <cmath>

// 2D vector
struct V2 { float x, y; };
GB_HD inline V2 v2(float x, float y){ V2 r; r.x=x; r.y=y; return r; }
GB_HD inline V2 operator+(V2 a, V2 b){ return v2(a.x+b.x, a.y+b.y); }
GB_HD inline V2 operator-(V2 a, V2 b){ return v2(a.x-b.x, a.y-b.y); }
GB_HD inline V2 operator*(float s, V2 a){ return v2(s*a.x, s*a.y); }
GB_HD inline V2 operator-(V2 a){ return v2(-a.x, -a.y); }
GB_HD inline float b2Dot(V2 a, V2 b){ return a.x*b.x + a.y*b.y; }
GB_HD inline float b2Cross(V2 a, V2 b){ return a.x*b.y - a.y*b.x; }
GB_HD inline V2 b2CrossVS(V2 a, float s){ return v2(s*a.y, -s*a.x); }   // b2Cross(vec,scalar)
GB_HD inline V2 b2CrossSV(float s, V2 a){ return v2(-s*a.y, s*a.x); }   // b2Cross(scalar,vec)
GB_HD inline float b2DistanceSquared(V2 a, V2 b){ V2 c=a-b; return b2Dot(c,c); }
GB_HD inline float b2ClampF(float a, float lo, float hi){ return a<lo?lo:(a>hi?hi:a); }
GB_HD inline float b2MaxF(float a, float b){ return a>b?a:b; }
GB_HD inline float b2MinF(float a, float b){ return a<b?a:b; }
GB_HD inline float b2AbsF(float a){ return a<0.0f?-a:a; }
GB_HD inline float b2Length(V2 v){ return sqrtf(v.x*v.x + v.y*v.y); }
GB_HD inline float b2Normalize(V2& v){
    float length = b2Length(v);
    if (length < GB_EPSILON) return 0.0f;
    float invLength = 1.0f / length;
    v.x *= invLength; v.y *= invLength;
    return length;
}

// 3D vector (b2Vec3), used by the 3x3 joint solves.
struct V3 { float x, y, z; };
GB_HD inline V3 v3(float x, float y, float z){ V3 r; r.x=x; r.y=y; r.z=z; return r; }
GB_HD inline V3 operator+(V3 a, V3 b){ return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
GB_HD inline V3 operator-(V3 a, V3 b){ return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
GB_HD inline V3 operator*(float s, V3 a){ return v3(s*a.x, s*a.y, s*a.z); }
GB_HD inline V3 operator-(V3 a){ return v3(-a.x, -a.y, -a.z); }
GB_HD inline float b2Dot3(V3 a, V3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
GB_HD inline V3 b2Cross3(V3 a, V3 b){ return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }

// rotation (sin,cos) + transform
struct Rot { float s, c; };
GB_HD inline Rot rotSet(float angle){ Rot q; q.s=sinf(angle); q.c=cosf(angle); return q; }
struct Xf { V2 p; Rot q; };

GB_HD inline V2 b2MulRV(Rot q, V2 v){ return v2(q.c*v.x - q.s*v.y, q.s*v.x + q.c*v.y); }       // b2Mul(Rot,Vec)
GB_HD inline V2 b2MulTinvV_q(Rot q, V2 v){ return v2(q.c*v.x + q.s*v.y, -q.s*v.x + q.c*v.y); } // b2MulT(Rot,Vec)
GB_HD inline V2 b2MulTV(Xf T, V2 v){                                                            // b2Mul(Transform,Vec)
    return v2((T.q.c*v.x - T.q.s*v.y) + T.p.x, (T.q.s*v.x + T.q.c*v.y) + T.p.y);
}
GB_HD inline V2 b2MulTinvV(Xf T, V2 v){                                                         // b2MulT(Transform,Vec)
    float px=v.x-T.p.x, py=v.y-T.p.y;
    return v2(T.q.c*px + T.q.s*py, -T.q.s*px + T.q.c*py);
}
GB_HD inline float b2MixFriction(float f1, float f2){ return sqrtf(f1*f2); }
GB_HD inline float b2MixRestitution(float r1, float r2){ return r1>r2?r1:r2; }
