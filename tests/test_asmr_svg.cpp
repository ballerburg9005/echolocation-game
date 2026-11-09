#include "../asmr_svg_geometry.h"
#include "../asmr_dynamic_head.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <queue>
#include <vector>

static bool reachable(const std::vector<float>& mask,int nx,int ny,float dx,
                       asmr::Vec2 point){
    // The ear channel must be physically open to the external room,
    // not a sealed empty pit that would trap a virtual microphone.
    const float gx=point.x/dx-0.5f,gy=point.y/dx-0.5f;
    const int ix=(int)std::floor(gx),iy=(int)std::floor(gy);
    if(ix<2||iy<2||ix+1>=nx-2||iy+1>=ny-2)return false;
    const int outside=(ny/4)*nx+nx/4;
    if(mask[outside]>0)return false;
    std::vector<unsigned char> seen(mask.size(),0);
    std::queue<int> todo;
    for(int y=0;y<2;++y)for(int x=0;x<2;++x){
        const int id=(iy+y)*nx+ix+x;
        if(mask[(size_t)id]>0)continue;
        const float weight=(1.0f-std::fabs(gx-(ix+x)))*
                           (1.0f-std::fabs(gy-(iy+y)));
        if(weight<=0.001f)continue;
        seen[(size_t)id]=1;todo.push(id);
    }
    while(!todo.empty()){
        const int v=todo.front();todo.pop();
        if(v==outside)return true;
        for(int offset:{-1,1,-nx,nx}){
            const int to=v+offset;
            if(to<0||to>=nx*ny||seen[to]||mask[to]>0)continue;
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
    // The pinna tips are part of the *solid* author's drawing;
    // verify the world-bounds rather than guessing these tips are air.
    assert(!pinnaV2.solid(-0.099f,0.0f));
    assert(!pinnaV2.solid(0.099f,0.0f));
    assert(channels.channels&&!simple.channels);
    // The exact subject-provided speaker recess is on its local -Y.
    assert(!speaker.solid(0,-asmr::kMouthOffset));
    // The acoustic 2x2 source stencil must be AIR at live 3.5mm dx.
    for(float x:{-0.0035f,0.0035f})
        for(float y:{-0.0035f,0.0035f})
            assert(!speaker.solid(x,-asmr::kMouthOffset+y));
    assert(speaker.solid(0,0));
    // Newly authored circular cavity: target the INNER half-spacing
    // probes, 7mm below centre. Leave V2's upper pinna mics unchanged.
    const auto innerL=asmr::leftEar(true);
    const auto innerR=asmr::rightEar(true);
    assert(!channels.solid(innerL.x-asmr::kListenerX,
                           asmr::kChannelEarY));
    assert(!channels.solid(innerR.x-asmr::kListenerX,
                           asmr::kChannelEarY));
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
    assert(std::fabs(asmr::micSpacing(true)-0.042f)<1e-6f);
    assert(std::fabs(asmr::micSpacing(false)-0.096f)<1e-6f);
    assert(std::fabs(asmr::micSpacing(true,true)-0.124f)<1e-6f);
    // V2's old 40mm ear location is SOLID in the unmodified SVG.
    // Its actual open recess starts farther out, clear at 62mm.
    assert(pinnaV2.solid(-0.040f,asmr::kPinnaEarY));
    assert(pinnaV2.solid(0.040f,asmr::kPinnaEarY));
    assert(!pinnaV2.solid(-0.062f,asmr::kPinnaEarY));
    assert(!pinnaV2.solid(0.062f,asmr::kPinnaEarY));

    for(float dx:{asmr::kLiveDx,0.0035f,0.0030f,0.0025f,asmr::kOfflineDx}) {
        const int nx=asmr::cells(dx),ny=asmr::cellsY(dx);
        const auto a=asmr::svgRoomMask(nx,ny,dx,simple);
        const auto b=asmr::svgRoomMask(nx,ny,dx,channels);
        const auto c=asmr::svgRoomMask(nx,ny,dx,pinnaV2);
        assert(a.size()==(size_t)nx*ny&&b.size()==a.size());
        assert(c.size()==a.size());
        assert(asmr::validateMicrophones(a,nx,ny,dx,false,simple));
        const bool channelValid=asmr::validateMicrophones(
            b,nx,ny,dx,true,channels);
        // A user may revise the exact channel SVG so its mouth or its
        // narrow neck seals at this dx. The program must reject that
        // resolution rather than sample an isolated air cavity.
        const bool channelReachable=
            reachable(b,nx,ny,dx,asmr::leftEar(true)) &&
            reachable(b,nx,ny,dx,asmr::rightEar(true));
        // Connectivity is necessary, though a too-small bilinear air
        // footprint can independently reject an otherwise open passage.
        assert(!channelValid || channelReachable);
        if(!channelValid)std::fprintf(stderr,
            "[SVG VERIFY] Channel ear pocket disconnected at dx=%.2fmm; "
            "alternative head will be unavailable on this grid.\n",dx*1000);
        assert(asmr::validateMicrophones(c,nx,ny,dx,true,pinnaV2));
        assert(reachable(c,nx,ny,dx,asmr::leftEar(true,true)));
        assert(reachable(c,nx,ny,dx,asmr::rightEar(true,true)));
        // No silent cheat: no cut-through to the old 40mm v2 points.
        const int xSolid=(int)std::floor((asmr::kListenerX+0.040f)/dx);
        const int ySolid=(int)std::floor((asmr::kListenerY+asmr::kChannelEarY)/dx);
        assert(c[(size_t)ySolid*nx+xSolid]==1.0f);
        assert(reachable(a,nx,ny,dx,asmr::leftEar(false)));
        assert(reachable(a,nx,ny,dx,asmr::rightEar(false)));
        assert(b[(int)(asmr::kListenerY/dx)*nx+(int)(asmr::kListenerX/dx)]==1.0f); // TWO halves join as solid
        assert(c[(int)(asmr::kListenerY/dx)*nx+(int)(asmr::kListenerX/dx)]==1.0f); // v2 mirrored halves join
        assert(a[(ny/4)*nx+nx/4]==0.0f); // no phantom 17cm circle
        assert(b[0]==echo::kOuterBoundaryMaterial);
        const int sourceX=(int)std::lround(
            (asmr::kListenerX+0.5f-asmr::kMouthOffset)/dx);
        assert(b[(int)(asmr::kListenerY/dx)*nx+sourceX]==0.0f);
    }
    // Rectangular field texture and the authored SVG use the same
    // physical (X,Y) coordinates. Previously they appeared as two
    // vertically offset heads in the UI; verify the raster mask
    // occupies EXACTLY the same local world geometry, rather than
    // trying to hide a real mapping error with an arbitrary offset.
    {
        constexpr float dx=asmr::kLiveDx;
        const int nx=asmr::cells(dx),ny=asmr::cellsY(dx);
        const auto mask=asmr::svgRoomMask(nx,ny,dx,pinnaV2);
        int checked=0,filled=0;
        const int x0=(int)std::floor((asmr::kListenerX-
            pinnaV2.widthMetres*0.5f-2*dx)/dx);
        const int x1=(int)std::ceil((asmr::kListenerX+
            pinnaV2.widthMetres*0.5f+2*dx)/dx);
        const int y0=(int)std::floor((asmr::kListenerY-
            pinnaV2.heightMetres*0.5f-2*dx)/dx);
        const int y1=(int)std::ceil((asmr::kListenerY+
            pinnaV2.heightMetres*0.5f+2*dx)/dx);
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){
            const float localX=(x+0.5f)*dx-asmr::kListenerX;
            const float localY=(y+0.5f)*dx-asmr::kListenerY;
            if(std::fabs(localX)<0.002f)continue; // authored seam weld
            const bool authored=pinnaV2.solid(localX,localY);
            const bool acoustic=mask[(size_t)y*nx+x]==1.0f;
            assert(authored==acoustic);
            ++checked;
            if(acoustic)++filled;
        }
        assert(checked>500 && filled>100);
        // Test exact viewport -> pressure texture -> physical field
        // lookup used by CUDA, including the Y-axis reversal.
        constexpr int screenW=540,screenH=960;
        constexpr int textureW=360,textureH=640;
        const float screenX=asmr::kListenerX/asmr::kRoomMetres*screenW;
        const float screenY=(asmr::kRoomHeightMetres-asmr::kListenerY)/
                                asmr::kRoomHeightMetres*screenH;
        const int texX=(int)(screenX/screenW*textureW);
        const int texY=(int)(screenY/screenH*textureH);
        const int fieldX=std::min(nx-1,(int)(((float)texX+0.5f)*nx/textureW));
        const int fieldY=std::min(ny-1,(int)(
            ((float)(textureH-texY)-0.5f)*ny/textureH));
        assert(std::fabs(((fieldX+0.5f)*dx)-asmr::kListenerX)<2*dx);
        assert(std::fabs(((fieldY+0.5f)*dx)-asmr::kListenerY)<2*dx);
        assert(mask[(size_t)fieldY*nx+fieldX]==1.0f);
    }
    // The zoomed acoustic mask must remain in physical registration
    // with the unchanged listener SVG, including the real mic points.
    // FDTD occupancy and acoustic pressure-image Y lookup registered
    // to the exact authored SVG at all true-zoom domain sizes.
    int connectedChannelZooms=0;
    for(size_t stage=0;stage<asmr::kZoomSteps.size();++stage){
        const bool square=stage+1==asmr::kZoomSteps.size();
        const float zoom=asmr::kZoomSteps[stage];
        const float dx=asmr::zoomDx(asmr::kLiveDx,(int)stage);
        const auto view=asmr::roomView(zoom,square);
        const int nx=(int)std::lround(view.width/dx);
        const int ny=(int)std::lround(view.height/dx);
        const auto mask=asmr::svgRoomMask(nx,ny,dx,pinnaV2,
                                          view.originX,view.originY);
        assert(std::fabs(asmr::relativeCellWork(dx,view)-
                        asmr::kZoomWorkRatios[stage])<0.015f);
        const int cx=(int)std::floor((asmr::kListenerX-view.originX)/dx);
        const int cy=(int)std::floor((asmr::kListenerY-view.originY)/dx);
        assert(cx>=0&&cx<nx&&cy>=0&&cy<ny);
        assert(mask[(size_t)cy*nx+cx]==1.0f);
        assert(asmr::validateMicrophones(mask,nx,ny,dx,
            true,pinnaV2,view.originX,view.originY));
        // Window/door termination material is physical FDTD mask
        // (not decorative graphics).
        assert(mask[(int)(ny*asmr::kLeftPortCentre)*nx]==
               asmr::kPortMaterial);
        assert(mask[(ny-1)*nx+(int)(nx*asmr::kFarPortCentre)]==
               asmr::kPortMaterial);
        const int imageW=360,imageH=640; // actual GPU pressure texture
        const int texX=(int)(((asmr::kListenerX-view.originX)/
                    view.width)*imageW);
        const int texY=(int)(((view.originY+view.height-
                    asmr::kListenerY)/view.height)*imageH);
        const int fieldX=std::min(nx-1,(int)(((float)texX+0.5f)*nx/imageW));
        const int fieldY=std::min(ny-1,(int)(
            ((float)(imageH-texY)-0.5f)*ny/imageH));
        assert(std::abs(fieldX-cx)<3&&std::abs(fieldY-cy)<3);
        assert(mask[(size_t)fieldY*nx+fieldX]==1.0f);
        const auto channelMask=asmr::svgRoomMask(nx,ny,dx,channels,
                                     view.originX,view.originY);
        const bool channelIsOpen=asmr::validateMicrophones(
            channelMask,nx,ny,dx,true,channels,view.originX,view.originY);
        if(channelIsOpen)++connectedChannelZooms;
        std::fprintf(stderr,"[SVG ZOOM] stage %zu %.2fx %s dx=%.3fmm "
            "grid=%dx%d channel microphone connected=%s\\n",
            stage,zoom,square?"SQUARE":"PORTRAIT",dx*1000,
            nx,ny,channelIsOpen?"YES":"NO");
    }
    // We must have at least one state in which the current authored
    // ear channel can actually receive pressure from outside.
    assert(connectedChannelZooms>=1);
    // Real CUDA moving-scatterer stencil samples exactly the same
    // alpha pixels as librsvg used for the artist's display.
    constexpr float dx=.0025f;
    const asmr::Vec2 centre{1.5f,1.0f};
    const auto forward=asmr::faceListener(centre);
    const auto cover=[&](asmr::Vec2 world) {
        return asmr::movingHeadCoverage(
            speaker.alpha.data(),speaker.w,speaker.h,
            speaker.widthMetres/dx,speaker.heightMetres/dx,
            centre.x/dx-.5f,centre.y/dx-.5f,
            forward.x,forward.y,world.x/dx-.5f,world.y/dx-.5f);
    };
    assert(cover(centre)>0.7f); // actual moving speaker body scatters
    assert(cover(asmr::mouth(centre))<0.10f); // mouth port stays air
    assert(cover({1.85f,1.0f})==0.0f);
    assert(std::fabs(asmr::movingHeadNeighbor(1,2,0,0)-2)<1e-6f);
    assert(asmr::movingHeadNeighbor(1,2,1,1)<1.2f);
    // A fixed listener and a moving *soft* speaker both scatter, but
    // no moving speaker cell is ever set to a hard mask / zeroed.
    const auto speakerForward=asmr::faceListener({1.5f,1.0f});
    const auto m=asmr::mouth({1.5f,1.0f});
    assert(!asmr::headSolidAt(
        speaker,{1.5f,1.0f},m,speakerForward));
    std::printf("Exact Inkscape SVG masks passed: 129.82mm simple, "
      "170.23mm channels, 185.35mm new pinna V2, 134.48mm sender; "
      "4.2cm channel / 9.6cm simple / 12.4cm V2 real-air mic spacing, open channel and source recess\n");
}
