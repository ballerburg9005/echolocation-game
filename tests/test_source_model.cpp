#include "../source_model.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// Free-space 2D FDTD, same CFL/laplacian/source as the CUDA solver. Compare
// forward and rear microphone ENERGY, not just an unvalidated directivity
// slogan. No wall reflections reach these probes before test ends.
static float freeFieldEnergy(bool paired,bool rearProbe) {
    constexpr int N=180,steps=155;
    std::vector<float> old(N*N,0),now(N*N,0),next(N*N,0);
    const int front=90*N+96,back=90*N+90;
    const int probe=90*N+(rearProbe?55:125);
    const int duration=(int)std::lround(
        echo::kClickDurationSeconds/echo::kStepSeconds);
    const int delay=(int)std::lround(
        6.0f*echo::kCellMetres/
        (echo::kSpeedOfSound*echo::kStepSeconds));
    float energy=0.0f;
    for(int step=0;step<steps;++step) {
        for(int y=1;y<N-1;++y)for(int x=1;x<N-1;++x) {
            const int idx=y*N+x;
            next[idx]=echo::waveStep(now[idx],old[idx],
                now[idx-1],now[idx+1],now[idx-N],now[idx+N],
                echo::kCourantSquared)*0.9995f;
        }
        next[front]+=echo::phasedPlayerClick(step,duration,
                                             echo::kStepSeconds,false,delay);
        if(paired)next[back]+=echo::phasedPlayerClick(step,duration,
                                             echo::kStepSeconds,true,delay);
        energy+=next[probe]*next[probe];
        old.swap(now);
        now.swap(next);
    }
    return energy;
}

int main(){
    const int N=180;
    std::vector<float> mask(N*N,0);
    auto east=echo::phasedSource(N,mask.data(),90,90,0);
    auto north=echo::phasedSource(N,mask.data(),90,90,
                                 echo::kPi/2);
    assert(east.enabled && north.enabled);
    assert(east.frontX>east.backX && east.backDelay>=7);
    assert(north.frontY>north.backY && north.backDelay>=7);
    assert(echo::phasedPlayerClick(4,100,echo::kStepSeconds,true,9)==0.0f);
    assert(echo::phasedPlayerClick(30,100,echo::kStepSeconds,false,9)>-1.1f);
    mask[east.frontY*N+east.frontX]=1;
    assert(!echo::phasedSource(N,mask.data(),90,90,0).enabled);

    const float monoBack=freeFieldEnergy(false,true);
    const float phasedBack=freeFieldEnergy(true,true);
    const float phasedFront=freeFieldEnergy(true,false);
    assert(monoBack>1e-7f);
    // This measures cancellation IN THE SIMULATED PRESSURE FIELD,
    // without assuming a physically impossible zero-direct-sound ray.
    std::printf("phased 2D FDTD energy: back=%g monoBack=%g front=%g\n",
                phasedBack,monoBack,phasedFront);
    assert(phasedBack<monoBack*0.70f);
    assert(phasedFront>phasedBack*1.25f);

    std::vector<float> materials(80*80,0);
    std::vector<float> emap(80*80,0);
    assert(echo::insideNgon(0,0,8,16));
    assert(!echo::insideNgon(9,0,8,16));
    assert(!echo::insideNgon(0,8,8,16,echo::kPi/16));
    for(int y=0;y<80;++y)for(int x=0;x<80;++x)
        if(echo::insideNgon(x-40,y-40,8,16))
            materials[y*80+x]=2;
    assert(echo::placeWorldEmitter(emap,materials.data(),80,
            40,40,echo::kGhostTracker,8));
    int nonzero=0;
    for(size_t i=0;i<emap.size();++i)if(emap[i]>0) {
        assert(materials[i]<=0);
        assert((int)emap[i]==echo::kGhostTracker);
        ++nonzero;
    }
    assert(nonzero>0&&nonzero<=4);
    assert(echo::toneFrequency(echo::kGhostTracker)!=
           echo::toneFrequency(echo::kSvgItem));
    assert(echo::fdtdWorldTone(0,100,echo::kStepSeconds)==0);
    assert(std::isfinite(echo::fdtdWorldTone(
        echo::kGhostTracker+0.5f,100,echo::kStepSeconds)));
    std::puts("phased FDTD source, n-gon reflectors and in-field material tones passed");
}
