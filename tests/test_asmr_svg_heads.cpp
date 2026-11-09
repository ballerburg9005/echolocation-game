#include "../asmr_svg_heads.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
int main(){
    const auto simple=asmr::svg::load("listener.svg");
    const auto canal=asmr::svg::load("listener_with_ear_channel.svg");
    const auto artist=asmr::svg::load("speaker.svg");
    assert(simple.valid()&&canal.valid()&&artist.valid());
    assert(simple.loops.size()==1);
    assert(canal.loops.size()==2);
    assert(artist.loops.size()==1);
    assert(std::fabs(simple.widthMm-129.82309f)<0.001f);
    assert(std::fabs(canal.widthMm-170.22807f)<0.001f);
    assert(std::fabs(artist.heightMm-130.03357f)<0.001f);
    assert(simple.loops[0].size()>200);
    assert(canal.loops[0].size()>200);
    assert(artist.loops[0].size()>200);
    assert(std::fabs(asmr::micSpacing(true)-0.080f)<1e-6f);
    assert(std::fabs(asmr::micSpacing(false)-0.096f)<1e-6f);
    assert(asmr::kMouthOffset<0.060f);
    for(float dx:{0.010f,0.005f,0.0035f,0.0015f}){
        const int n=asmr::cells(dx);
        for(int channel=0;channel<2;++channel){
            const auto& shape=channel?canal:simple;
            auto mask=asmr::svg::listenerMask(n,dx,shape,channel!=0);
            const auto left=asmr::leftEar(channel!=0);
            const auto right=asmr::rightEar(channel!=0);
            assert(asmr::probeInAir(mask,n,left,dx));
            assert(asmr::probeInAir(mask,n,right,dx));
            const int mx=(int)(asmr::kListenerX/dx);
            const int my=(int)(asmr::kListenerY/dx);
            assert(mask[(size_t)my*n+mx]>0); // stitched solid core
            const int outer=(int)(0.014f/dx);
            assert(mask[(size_t)outer*n+outer]==0);
        }
    }
    const auto texture=asmr::svg::speakerTexture(artist);
    assert(texture.size()==(size_t)asmr::svg::kSpeakerTexture*
                           asmr::svg::kSpeakerTexture);
    const int mid=asmr::svg::kSpeakerTexture/2;
    assert(texture[(size_t)mid*asmr::svg::kSpeakerTexture+mid]>0.5f);
    const int forward=(int)((.5f*asmr::svg::kSpeakerTexture*
                        asmr::svg::kSpeakerTexel-
                        asmr::kMouthOffset)/asmr::svg::kSpeakerTexel);
    assert(texture[(size_t)forward*asmr::svg::kSpeakerTexture+mid]==0.0f);
    // A two-sided arbitrary head shape must be oriented by geometric
    // rotation, not replaced by a circle or a rendering-only marker.
    const auto facing=asmr::svg::inWorld(
        artist,{1.50f,1.0f},echo::kPi,true);
    assert(facing.size()==1&&facing[0].size()>200);
    std::printf("SVG ASMR heads: simple %zu vertices, ear channels "
        "%zu+%zu, speaker %zu; 8cm/9.6cm calibrated mics "
        "and mouth in fluid at 2m scale passed\n",
        simple.loops[0].size(),canal.loops[0].size(),
        canal.loops[1].size(),artist.loops[0].size());
}
