#include "../source_model.h"
#include <array>
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

// Numerical polar test at a 42cm observation ring. A plain front/back
// pair rejects the back but radiates strongly sideways; the new three-across
// aperture must actually reduce SIDE energy in the same FDTD stencil.
static std::array<double,3> directionalEnergies(
    const echo::BeamArray& beam,float heading=0.0f) {
    constexpr int N=180,steps=173;
    std::vector<float> old(N*N,0),now(N*N,0),next(N*N,0);
    const int duration=(int)std::lround(
        echo::kClickDurationSeconds/echo::kStepSeconds);
    std::array<double,3> energy={0.0,0.0,0.0};
    for(int step=0;step<steps;++step) {
        for(int y=1;y<N-1;++y)for(int x=1;x<N-1;++x) {
            const int i=y*N+x;
            next[i]=echo::waveStep(now[i],old[i],
                now[i-1],now[i+1],now[i-N],now[i+N],
                echo::kCourantSquared)*0.9995f;
        }
        for(int k=0;k<3;++k) {
            if(beam.weights[k]<=0)continue;
            const auto& e=beam.elements[k];
            next[e.frontY*N+e.frontX]+=
                beam.weights[k]*echo::clickSample(
                    step,duration,echo::kStepSeconds);
            if(beam.directional)
                next[e.backY*N+e.backX]-=
                    beam.weights[k]*echo::clickSample(
                        step-e.backDelay,duration,echo::kStepSeconds);
        }
        for(int angleIndex=0;angleIndex<3;++angleIndex) {
            const float angle=heading+
                (angleIndex==1?echo::kPi/2.0f:
                 angleIndex==2?echo::kPi:0.0f);
            const float sx=90.0f+42.0f*std::cos(angle);
            const float sy=90.0f+42.0f*std::sin(angle);
            const double pressure=echo::bilinearSample(
                next.data(),N,sx,sy);
            energy[angleIndex]+=pressure*pressure;
        }
        old.swap(now);now.swap(next);
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

    // The six actual source cells must fit in a 13cm player collision
    // radius and be bilaterally symmetric, even when rotated.
    mask[east.frontY*N+east.frontX]=0.0f;
    const auto narrow=echo::playerBeam(N,mask.data(),90,90,0,true);
    const auto mono=echo::playerBeam(N,mask.data(),90,90,0,false);
    assert(narrow.valid && narrow.directional);
    assert(mono.valid && !mono.directional);
    assert(narrow.weights[0]==0.25f&&narrow.weights[1]==0.50f&&
           narrow.weights[2]==0.25f);
    assert(mono.weights[1]==1.0f);
    for(int k=0;k<3;++k) {
        const auto& e=narrow.elements[k];
        assert(e.enabled && e.backDelay>0);
        assert(std::hypot(e.frontX-90.0f,e.frontY-90.0f)<13.0f);
        assert(std::hypot(e.backX-90.0f,e.backY-90.0f)<13.0f);
    }
    assert(narrow.elements[0].frontY<narrow.elements[1].frontY);
    assert(narrow.elements[2].frontY>narrow.elements[1].frontY);
    auto twoElement=narrow;
    twoElement.weights[0]=twoElement.weights[2]=0.0f;
    twoElement.weights[1]=1.0f;
    const auto focused=directionalEnergies(narrow);
    const auto wide=directionalEnergies(twoElement);
    std::printf("FDTD 42cm angular energy: narrow front=%g side=%g rear=%g; 2-source side=%g\n",
        focused[0],focused[1],focused[2],wide[1]);
    assert(focused[0]>1e-5);
    assert(focused[1]/focused[0]<0.18); // genuine side cancellation
    assert(focused[1]<wide[1]*0.30);
    assert(focused[2]<focused[0]*0.06);
    assert(focused[0]>wide[0]*0.70);
    // Rotating the player should rotate the *acoustic beam*, not leave
    // a fixed-forward visual shape with a broad sideways actual wave.
    for(float heading:{echo::kPi/8.0f,echo::kPi/4.0f,
                       echo::kPi/2.0f,3.0f*echo::kPi/4.0f}) {
        const auto turned=echo::playerBeam(
            N,mask.data(),90.0f,90.0f,heading,true);
        assert(turned.valid && turned.directional);
        const auto energy=directionalEnergies(turned,heading);
        assert(energy[0]>1e-5);
        assert(energy[1]/energy[0]<0.22);
        assert(energy[2]/energy[0]<0.08);
    }
    // Avoid a clipped aperture when one of the outer sources would
    // be embedded in an SVG solid; fallback to safe central pair.
    auto obstructed=mask;
    obstructed[narrow.elements[0].frontY*N+
               narrow.elements[0].frontX]=1.0f;
    const auto fallback=echo::playerBeam(N,obstructed.data(),90,90,0,true);
    assert(fallback.valid && fallback.directional);
    assert(fallback.weights[1]==1.0f && fallback.weights[0]==0.0f);

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
    // At 1cm pixel resolution a 16-sided, 16cm target looked circular.
    // A 20cm hexagon must show distinct flat edges and true 6-fold
    // rotational symmetry (not a shifted circle/square composite).
    assert(echo::insideNgon(10,0,10,6,0));
    assert(echo::insideNgon(0,8,10,6,0));
    assert(!echo::insideNgon(0,10,10,6,0));
    for(int y=-8;y<=8;++y)for(int x=-8;x<=8;++x) {
        const bool a=echo::insideNgon((float)x,(float)y,7.0f,8,
                                      echo::kPi/8.0f);
        const bool opposite=echo::insideNgon((float)-x,(float)-y,
                                      7.0f,8,echo::kPi/8.0f);
        assert(a==opposite);
    }
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
    // Atmospheric spaceship drones were caused by persistent unmodulated
    // in-field oscillators. Only short, tapered pings may radiate now.
    int audible=0,silent=0;
    for(uint64_t step=0;step<180000ULL;step+=113ULL) {
        const float sample=echo::fdtdWorldTone(
            echo::kGhostTracker+0.72f,step,echo::kStepSeconds);
        if(std::fabs(sample)>0.00001f)++audible;
        else ++silent;
    }
    assert(audible>5);
    assert(silent>audible*15);
    // Nothing emitted at zero signal strength.
    assert(!echo::placeWorldEmitter(emap,materials.data(),80,
            40,40,echo::kGhostTracker,8.0f,0.0f));
    std::puts("phased FDTD source, n-gon reflectors and in-field material tones passed");
}
