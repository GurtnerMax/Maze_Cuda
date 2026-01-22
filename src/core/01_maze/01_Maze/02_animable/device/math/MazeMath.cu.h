#pragma once
#include "cudas.h"

// Labels
// 0 : mur
// INF : couloir non visité
// +k : distance depuis Start
// -k : distance depuis Goal
static constexpr int LABEL_INF = 0x3fffffff;

// Directions (0..7), 255 = none, 254 = sentinel start/goal
__device__ __forceinline__ unsigned char packDir(int k) { return (unsigned char)k; }

// HSV (h in [0,1), s in [0,1], v in [0,1]) -> RGB
__device__ __forceinline__ float3 hsv2rgb(float h, float s, float v)
{
    float hh = h * 6.0f;
    int   i  = (int)hh;
    float f  = hh - (float)i;

    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));

    float r, g, b;

    switch (i)
    {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
    return make_float3(r, g, b);
}
