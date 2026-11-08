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
inline DirectResponse freeFieldDirect(const BeamArray& beam,
    float micLx,float micLy,float micRx,float micRy,
    int duration,float inputStep,float c2,float decay,
    float amplitude,float playbackGain,bool phased,
    int frames=158) {
    DirectResponse out;
    if(!beam.valid||duration<2||frames<1||inputStep<=0)return out;
    constexpr int N=164;
    constexpr int centre=N/2;
    // Translate world coordinates: place the *actual quantized* front source
    // at the center of the homogeneous reference grid.
    const float sx=(float)centre-(float)beam.elements[1].frontX;
    const float sy=(float)centre-(float)beam.elements[1].frontY;
    const float lx=micLx+sx,ly=micLy+sy;
    const float rx=micRx+sx,ry=micRy+sy;
    if(lx<5||ly<5||rx<5||ry<5||
       lx>N-6||ly>N-6||rx>N-6||ry>N-6)
        return out;
    for(int k=0;k<3;++k) {
        if(beam.weights[k]<=0.0f)continue;
        const auto& e=beam.elements[k];
        const int frontX=e.frontX+centre-beam.elements[1].frontX;
        const int frontY=e.frontY+centre-beam.elements[1].frontY;
        const int rearX=e.backX+centre-beam.elements[1].frontX;
        const int rearY=e.backY+centre-beam.elements[1].frontY;
        if(frontX<5||frontY<5||frontX>=N-5||frontY>=N-5||
           (phased&&beam.directional&&
            (rearX<5||rearY<5||rearX>=N-5||rearY>=N-5)))
            return out;
    }
    out.left.reserve(frames);out.right.reserve(frames);
    std::vector<float> previous(N*N,0),current(N*N,0),next(N*N,0);
    for(int t=0;t<frames;++t) {
        for(int y=1;y<N-1;++y)for(int x=1;x<N-1;++x) {
            const int idx=y*N+x;
            next[idx]=waveStep(current[idx],previous[idx],
                current[idx-1],current[idx+1],
                current[idx-N],current[idx+N],c2)*decay;
        }
        for(int k=0;k<3;++k) {
            if(beam.weights[k]<=0.0f)continue;
            const auto& e=beam.elements[k];
            const int frontX=e.frontX+centre-beam.elements[1].frontX;
            const int frontY=e.frontY+centre-beam.elements[1].frontY;
            const int rearX=e.backX+centre-beam.elements[1].frontX;
            const int rearY=e.backY+centre-beam.elements[1].frontY;
            // Match the precise subcell source footprint in the CUDA
            // solver, including the fractionally delayed opposite sign
            // pulse. Otherwise experimental J subtraction invents pops
            // because its reference disagrees with the actual emitter.
            const float sourceX=e.continuous?e.frontSubX+sx:(float)frontX;
            const float sourceY=e.continuous?e.frontSubY+sy:(float)frontY;
            const int ix=(int)std::floor(sourceX);
            const int iy=(int)std::floor(sourceY);
            for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
                next[(iy+dy)*N+(ix+dx)]+=
                    beam.weights[k]*amplitude*decay*
                    sourceWeight(sourceX,sourceY,ix+dx,iy+dy)*
                    clickSample(t,duration,inputStep);
            if(phased && beam.directional && e.enabled) {
                const float positionX=e.continuous?
                    e.backSubX+sx:(float)rearX;
                const float positionY=e.continuous?
                    e.backSubY+sy:(float)rearY;
                const int bx=(int)std::floor(positionX);
                const int by=(int)std::floor(positionY);
                const float backClick=e.continuous?
                    subcellClickSample((float)t-e.backDelaySub,
                                       duration,inputStep):
                    clickSample(t-e.backDelay,duration,inputStep);
                for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
                    next[(by+dy)*N+(bx+dx)]-=
                        beam.weights[k]*amplitude*decay*
                        sourceWeight(positionX,positionY,bx+dx,by+dy)*
                        backClick;
            }
        }
        out.left.push_back(bilinearSample(next.data(),N,lx,ly)*playbackGain);
        out.right.push_back(bilinearSample(next.data(),N,rx,ry)*playbackGain);
        previous.swap(current);
        current.swap(next);
    }
    return out;
}
// Preserve the earlier two-element source for isolated numerical A/B tests.
inline DirectResponse freeFieldDirect(const PhasedSource& source,
    float micLx,float micLy,float micRx,float micRy,
    int duration,float inputStep,float c2,float decay,
    float amplitude,float playbackGain,bool phased,
    int frames=158) {
    BeamArray beam;
    beam.elements[1]=source;
    beam.valid=true;
    beam.directional=phased&&source.enabled;
    return freeFieldDirect(beam,micLx,micLy,micRx,micRy,duration,inputStep,
                           c2,decay,amplitude,playbackGain,phased,frames);
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
