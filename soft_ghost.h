#pragma once
// Acoustic reflectors for rotating 6-sided ghosts.
//
// Never stamp a moving/rotating ghost as a binary 1/2 in the hard CUDA
// obstacle mask. A newly solid cell used to reset p_next to zero,
// generating nonphysical pressure pops while objects moved.
//
// Instead: quarter-cell supersampling produces fractional 0..1 occupancy;
// CPU exponential smoothing (see smoothGhostCoverage) limits changes
// between display frames; the FDTD stencil blends neighbor gradients
// toward the zero-normal-gradient hard-wall condition continuously.
// This is a stable, approximate time-varying acoustic impedance, not
// a moving-rigid-body CFD model or a mathematically exact scattering solver.
#include "source_model.h"
#include <algorithm>
#include <cmath>

namespace echo {
inline float rotatingGhostCoverage(float dx,float dy,float rotation,
                                   float radiusCm=10.0f) {
    if(radiusCm<=0 || !std::isfinite(dx)||!std::isfinite(dy)||
       !std::isfinite(rotation))return 0.0f;
    // 2x2 sub-cell coverage: a polygon edge never changes a full cell
    // from 0 to 1 on an infinitesimal rotation.
    float amount=0.0f;
    for(int y=0;y<2;++y)for(int x=0;x<2;++x)
        amount+=insideNgon(dx+(x?0.25f:-0.25f),
                           dy+(y?0.25f:-0.25f),radiusCm,6,rotation)
                    ?0.25f:0.0f;
    return amount;
}
inline float smoothGhostCoverage(float previous,float target,float dtSeconds,
                                 float responseSeconds=0.055f) {
    if(!std::isfinite(previous)||!std::isfinite(target)||
       !std::isfinite(dtSeconds)||responseSeconds<=0.0f)
        return 0.0f;
    const float fraction=1.0f-std::exp(
        -std::max(0.0f,dtSeconds)/responseSeconds);
    return std::clamp(previous+
        fraction*(std::clamp(target,0.0f,1.0f)-
                  std::clamp(previous,0.0f,1.0f)),0.0f,1.0f);
}
#ifdef __CUDACC__
#define ECHO_GHOST_HD __host__ __device__
#else
#define ECHO_GHOST_HD
#endif
ECHO_GHOST_HD inline float softGhostNeighbour(
    float center,float neighbor,float hereCoverage,
    float neighborCoverage) {
    // Symmetric edge permeability. At full coverage the ghost is ~98%
    // reflecting but remains a fluid cell with pressure history intact.
    // CFL stability is retained: every edge weight is in [0.02,1].
    const float h=fminf(1.0f,fmaxf(0.0f,hereCoverage));
    const float n=fminf(1.0f,fmaxf(0.0f,neighborCoverage));
    const float transparency=1.0f-0.98f*fmaxf(h,n);
    return center+transparency*(neighbor-center);
}
#undef ECHO_GHOST_HD
} // namespace echo
