#include "../asmr_svg_geometry.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <queue>
#include <vector>

static bool reachable(const std::vector<float>& mask,int n,float dx,
                       asmr::Vec2 point){
    // The ear channel must be physically open to the external room,
    // not a sealed empty pit that would trap a virtual microphone.
    const int x=(int)(point.x/dx),y=(int)(point.y/dx);
    if(x<2||y<2||x>=n-2||y>=n-2)return false;
    const int start=y*n+x;
    if(mask[start]>0)return false;
    const int outside=(n/4)*n+n/4;
    if(mask[outside]>0)return false;
    std::vector<unsigned char> seen(mask.size(),0);
    std::queue<int> todo;
    seen[start]=1;todo.push(start);
    while(!todo.empty()){
        const int v=todo.front();todo.pop();
        if(v==outside)return true;
        for(int offset:{-1,1,-n,n}){
            const int to=v+offset;
            if(to<0||to>=n*n||seen[to]||mask[to]>0)continue;
            seen[to]=1;todo.push(to);
        }
    }
    return false;
}
int main(){
    auto simple=asmr::listenerSilhouette(asmr::HeadStyle::Simple);
    auto channels=asmr::listenerSilhouette(asmr::HeadStyle::Channels);
    auto pinnaV2=asmr::listenerSilhouette(asmr::HeadStyle::PinnaV2);
    auto speaker=asmr::speakerSilhouette();
    assert(simple.present()&&channels.present()&&
           pinnaV2.present()&&speaker.present());
    assert(std::fabs(simple.widthMetres-0.12982309f)<1e-5f);
    assert(std::fabs(channels.widthMetres-0.17022807f)<1e-5f);
    assert(std::fabs(speaker.widthMetres-0.13448267f)<1e-5f);
    assert(std::fabs(pinnaV2.widthMetres-0.18534811f)<1e-5f);
    assert(std::fabs(pinnaV2.heightMetres-0.13685854f)<1e-5f);
    assert(pinnaV2.pinnaV2&&pinnaV2.channels);
    // Two user-authored left and right paths, with noticeable pinna
    // recesses rather than the previous circular head outline.
    assert(pinnaV2.solid(0,0));
    assert(!pinnaV2.solid(-0.086f,0.0f));
    assert(!pinnaV2.solid(0.086f,0.0f));
    assert(channels.channels&&!simple.channels);
    // The exact subject-provided speaker recess is on its local -Y.
    assert(!speaker.solid(0,-asmr::kMouthOffset));
    assert(speaker.solid(0,0));
    // The ear opening in channel SVG is 3mm above the drawn centre:
    // the author's 8cm separation should land in its OPEN groove.
    assert(!channels.solid(-0.04f,asmr::kChannelEarY));
    assert(!channels.solid(0.04f,asmr::kChannelEarY));
    std::fprintf(stderr,
        "[SVG VERIFY] simple (%d,%d) alpha: -48mm=%d +48mm=%d "
        "left=-0.06:%d,-0.05:%d,-0.04:%d right=0.06:%d "
        "channel [-40mm,+40mm] mid= %d,%d shifted=%d,%d\n",
        simple.w,simple.h,
        (int)simple.solid(-.048f,0),(int)simple.solid(.048f,0),
        (int)simple.solid(-.060f,0),
        (int)simple.solid(-.050f,0),(int)simple.solid(-.040f,0),
        (int)simple.solid(.060f,0),
        (int)channels.solid(-.040f,0),
        (int)channels.solid(.040f,0),
        (int)channels.solid(-.040f,asmr::kChannelEarY),
        (int)channels.solid(.040f,asmr::kChannelEarY));
    // Original simple recess actually ENDS ~2mm outside the required
    // ±4.8cm microphone positions; the physical acoustic mask
    // adds a 3mm connecting microphone port at precisely that x.
    assert(simple.solid(-.048f,0));
    assert(simple.solid(.048f,0));
    assert(std::fabs(asmr::micSpacing(true)-0.08f)<1e-6f);
    assert(std::fabs(asmr::micSpacing(false)-0.096f)<1e-6f);

    for(float dx:{0.0035f,0.0030f,0.0025f,0.0015f}) {
        const int n=asmr::cells(dx);
        const auto a=asmr::svgRoomMask(n,dx,simple);
        const auto b=asmr::svgRoomMask(n,dx,channels);
        const auto c=asmr::svgRoomMask(n,dx,pinnaV2);
        assert(a.size()==(size_t)n*n&&b.size()==a.size());
        assert(c.size()==a.size());
        assert(asmr::validateMicrophones(a,n,dx,false,simple));
        assert(asmr::validateMicrophones(b,n,dx,true,channels));
        assert(asmr::validateMicrophones(c,n,dx,true,pinnaV2));
        assert(reachable(b,n,dx,asmr::leftEar(true)));
        assert(reachable(b,n,dx,asmr::rightEar(true)));
        assert(reachable(c,n,dx,asmr::leftEar(true)));
        assert(reachable(c,n,dx,asmr::rightEar(true)));
        assert(reachable(a,n,dx,asmr::leftEar(false)));
        assert(reachable(a,n,dx,asmr::rightEar(false)));
        assert(b[n/2*n+n/2]==1.0f); // TWO halves join as solid
        assert(c[n/2*n+n/2]==1.0f); // v2 mirrored halves join
        assert(a[n/4*n+n/4]==0.0f); // no phantom 17cm circle
        assert(b[0]==echo::kOuterBoundaryMaterial);
        const int sourceX=(int)std::lround(
            (asmr::kListenerX+0.5f-asmr::kMouthOffset)/dx);
        assert(b[n/2*n+sourceX]==0.0f);
    }
    const auto speakerForward=asmr::faceListener({1.5f,1.0f});
    const auto m=asmr::mouth({1.5f,1.0f});
    assert(!asmr::headSolidAt(
        speaker,{1.5f,1.0f},m,speakerForward));
    std::printf("Exact Inkscape SVG masks passed: 129.82mm simple, "
      "170.23mm channels, 185.35mm new pinna V2, 134.48mm sender; "
      "8.0cm/9.6cm real-air ear spacing, open channel and source recess\n");
}
