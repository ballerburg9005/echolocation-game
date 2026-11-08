#pragma once
// Optional *receiver-only* incident-field subtraction.
//
// Linear acoustics permits p_total = p_free + p_scattered. We compute the
// player's free-field click with the IDENTICAL 2D finite-difference stencil,
// bilinear ears and temporal waveform, then subtract it at the microphones.
// The actual FDTD pressure grid remains unchanged: it still contains full
// direct + reflected waves, including the waves that drive reflection.
// This is the mathematically coherent "echo-only" approach, not a ray skip,
// delay-gate, or physical hearing model.
//
// Limitations: receiver/source motion during the ~3ms kernel response,
// lattice rounding, and moving scatterers cause mismatch; it is not zero
// direct leakage guaranteed. CPU reference work must be profiled on hardware.
#include "source_model.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace echo {
struct DirectResponse {
    std::vector<float> left,right;
};
inline DirectResponse freeFieldDirect(const PhasedSource& source,
    float micLx,float micLy,float micRx,float micRy,
    int duration,float inputStep,float c2,float decay,
    float amplitude,float playbackGain,bool phased,
    int frames=158) {
    DirectResponse out;
    if(duration<2||frames<1||inputStep<=0)return out;
    constexpr int N=164;
    constexpr int centre=N/2;
    // Translate world coordinates: place the *actual quantized* front source
    // at the center of the homogeneous reference grid.
    const float sx=(float)centre-(float)source.frontX;
    const float sy=(float)centre-(float)source.frontY;
    const float lx=micLx+sx,ly=micLy+sy;
    const float rx=micRx+sx,ry=micRy+sy;
    const int rearX=source.backX+centre-source.frontX;
    const int rearY=source.backY+centre-source.frontY;
    if(lx<5||ly<5||rx<5||ry<5||
       lx>N-6||ly>N-6||rx>N-6||ry>N-6 ||
       (phased&&(rearX<5||rearY<5||rearX>=N-5||rearY>=N-5)))
        return out;
    out.left.reserve(frames);out.right.reserve(frames);
    std::vector<float> previous(N*N,0),current(N*N,0),next(N*N,0);
    for(int t=0;t<frames;++t) {
        for(int y=1;y<N-1;++y)for(int x=1;x<N-1;++x) {
            const int idx=y*N+x;
            next[idx]=waveStep(current[idx],previous[idx],
                current[idx-1],current[idx+1],
                current[idx-N],current[idx+N],c2)*decay;
        }
        next[centre*N+centre]+=
            amplitude*clickSample(t,duration,inputStep)*decay;
        if(phased && source.enabled) {
            next[rearY*N+rearX]-=
                amplitude*clickSample(t-source.backDelay,duration,
                                      inputStep)*decay;
        }
        out.left.push_back(bilinearSample(next.data(),N,lx,ly)*playbackGain);
        out.right.push_back(bilinearSample(next.data(),N,rx,ry)*playbackGain);
        previous.swap(current);
        current.swap(next);
    }
    return out;
}
// Subtraction is defined on the *simulation* timeline, before resampling;
// this preserves exact step alignment and keeps UI/game tones unmodified.
inline float subtractDirect(float total,const std::vector<float>& reference,
                            uint64_t sourceStart,uint64_t sampleStep) {
    if(sourceStart==UINT64_MAX||sampleStep<sourceStart)return total;
    const uint64_t offset=sampleStep-sourceStart;
    if(offset>=reference.size())return total;
    return total-reference[(size_t)offset];
}
} // namespace echo
