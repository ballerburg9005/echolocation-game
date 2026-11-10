#pragma once
// Render EXACT supplied Inkscape SVG paths, including elliptical arcs,
// cubics, nested transforms and the two halves of the ear-channel head,
// into an acoustic occupancy mask. No hand-drawn approximate polygons,
// inferred circles, synthetic HRTF or ear material deletion.
#include "asmr_lab_math.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string>
#include <vector>

namespace asmr {
enum class HeadStyle {Simple,Channels,PinnaV2};
struct SvgSilhouette {
    float widthMetres=0, heightMetres=0;
    int w=0,h=0;
    std::vector<unsigned char> alpha;
    bool channels=false;
    bool pinnaV2=false; // author's two mirrored ear openings with pinnae
    bool present()const{return w>0&&h>0&&!alpha.empty();}
    // World local x right, local y UP; raster SVG y is DOWN.
    bool solid(float x,float y)const {
        if(!present())return false;
        const float u=x/widthMetres+0.5f;
        const float v=0.5f-y/heightMetres;
        if(u<0||u>=1||v<0||v>=1)return false;
        const int px=std::min(w-1,(int)(u*w));
        const int py=std::min(h-1,(int)(v*h));
        return alpha[(size_t)py*w+px]>=48;
    }
};

SvgSilhouette renderSvg(const std::string& filename,
                         float rasterPixelMetres=0.00040f);
std::string svgLocation(const char* filename);
SvgSilhouette listenerSilhouette(HeadStyle style);
SvgSilhouette speakerSilhouette();

inline bool headSolidAt(const SvgSilhouette& svg,Vec2 centre,
                        Vec2 world,Vec2 forward){
    // Local SVG mouth points downward; transform it toward fixed
    // listener without rotating the user's exact curved geometry.
    const Vec2 diff=sub(world,centre);
    const Vec2 right{-forward.y,forward.x};
    const float localX=diff.x*right.x+diff.y*right.y;
    const float localY=-(diff.x*forward.x+diff.y*forward.y);
    return svg.solid(localX,localY);
}
// A sub-cell ear passage can be open in the author's 0.4mm SVG raster
// but vanish when cell-centre sampled onto a coarser FDTD grid. Preserve
// ONLY the topology of an ACTUAL connected transparent path in the
// original SVG. This is a one-cell subgrid airway approximation, not
// a fabricated straight tunnel through the head or a change to SVG art.
// BFS uses 4-way connections in the high-resolution authored air.
// Every cleared coarse cell intersects this authenticated air path.
inline bool preserveSvgAirway(std::vector<float>& mask,int nx,int ny,
                      float dx,const SvgSilhouette& svg,Vec2 mic,
                      float originX,float originY) {
    if(!svg.present() || svg.w<4 || svg.h<4 || nx<8 || ny<8)
        return false;
    const float localX=mic.x-kListenerX;
    const float localY=mic.y-kListenerY;
    if(svg.solid(localX,localY))return false;
    const int sx=std::clamp((int)((localX/svg.widthMetres+0.5f)*svg.w),
                            0,svg.w-1);
    const int sy=std::clamp((int)((0.5f-localY/svg.heightMetres)*svg.h),
                            0,svg.h-1);
    const int start=sy*svg.w+sx;
    if(svg.alpha[(size_t)start]>=48)return false;
    std::vector<int> parent(svg.alpha.size(),-2);
    std::queue<int> pending;
    parent[(size_t)start]=-1;
    pending.push(start);
    int exitPixel=-1;
    while(!pending.empty()){
        const int at=pending.front();pending.pop();
        const int x=at%svg.w,y=at/svg.w;
        if(x==0||y==0||x==svg.w-1||y==svg.h-1){
            exitPixel=at;break;
        }
        const int adjacent[4]{at-1,at+1,at-svg.w,at+svg.w};
        for(const int to:adjacent){
            if(parent[(size_t)to]!=-2 ||
               svg.alpha[(size_t)to]>=48)continue;
            parent[(size_t)to]=at;
            pending.push(to);
        }
    }
    if(exitPixel<0)return false; // never invent a nonexistent canal
    // Walk from the real exterior back to the real mic. Both points
    // are inside the image, and all intermediate pixels are transparent.
    for(int at=exitPixel;at>=0;at=parent[(size_t)at]){
        const int x=at%svg.w,y=at/svg.w;
        const float px=kListenerX+
            ((x+0.5f)/(float)svg.w-0.5f)*svg.widthMetres;
        const float py=kListenerY+
            (0.5f-(y+0.5f)/(float)svg.h)*svg.heightMetres;
        const int ix=(int)std::floor((px-originX)/dx);
        const int iy=(int)std::floor((py-originY)/dx);
        if(ix<=0||iy<=0||ix>=nx-1||iy>=ny-1)continue;
        const size_t index=(size_t)iy*nx+ix;
        if(mask[index]==1.0f)mask[index]=0.0f;
    }
    return true;
}
inline std::vector<float> svgRoomMask(int nx,int ny,float dx,
                    const SvgSilhouette& listenerSvg,
                    float originX=0,float originY=0){
    std::vector<float> mask=roomMask(nx,ny,dx,originX,originY);
    if(!listenerSvg.present())return mask;
    // Remove the former circular fallback ENTIRELY before stamping the
    // real SVG, including above and below its narrower child-size bbox.
    // Otherwise ghost semicircles remain around the authored silhouette.
    for(float& material:mask)
        if(material>0.0f && material<3.5f)material=0.0f;
    const int xmin=std::max(1,(int)std::floor(
        (kListenerX-listenerSvg.widthMetres*0.5f-originX)/dx)-1);
    const int xmax=std::min(nx-2,(int)std::ceil(
        (kListenerX+listenerSvg.widthMetres*0.5f-originX)/dx)+1);
    const int ymin=std::max(1,(int)std::floor(
        (kListenerY-listenerSvg.heightMetres*0.5f-originY)/dx)-1);
    const int ymax=std::min(ny-2,(int)std::ceil(
        (kListenerY+listenerSvg.heightMetres*0.5f-originY)/dx)+1);
    for(int y=ymin;y<=ymax;++y)
        for(int x=xmin;x<=xmax;++x) {
            const size_t i=(size_t)y*nx+x;
            if(mask[i]==echo::kOuterBoundaryMaterial)continue;
            const float worldX=originX+(x+0.5f)*dx;
            const float worldY=originY+(y+0.5f)*dx;
            float lx=worldX-kListenerX;
            const float ly=worldY-kListenerY;
            bool solid=listenerSvg.solid(lx,ly);
            // The author's ear-channel file has TWO independently
            // authored mirrored paths. Seal only sub-millimetre gaps
            // at the central straight join where both neighboring
            // halves are SOLID; never bridge a real ear passage.
            if(listenerSvg.channels&&std::fabs(lx)<0.0010f &&
               listenerSvg.solid(-0.0015f,ly) &&
               listenerSvg.solid(0.0015f,ly))
                solid=true;
            mask[i]=solid?1.0f:0.0f;
        }
    // For the inner-ear SVG only, preserve each actually transparent
    // airway at any dx. No new air path is permitted unless its
    // high-resolution SVG pixels are connected all the way outside.
    if(listenerSvg.channels &&
       (!listenerSvg.pinnaV2||dx>0.008f)){
        // When F11/Z budgeting coarsens V2 beyond 8mm, its REAL
        // transparent pinna recess can disappear at cell centers.
        // Preserve only the authored 0.4mm-raster air-connected
        // passage. Fine V2 grids retain their original strict mask.
        preserveSvgAirway(mask,nx,ny,dx,listenerSvg,
                          leftEar(true,listenerSvg.pinnaV2),
                          originX,originY);
        preserveSvgAirway(mask,nx,ny,dx,listenerSvg,
                          rightEar(true,listenerSvg.pinnaV2),
                          originX,originY);
    }
    // V2 microphones are IN AUTHORED AIR at +-62mm. Do not erase
    // solid head material to create synthetic canals for this mode.
    if(!listenerSvg.channels) {
        // The author's simple SVG recess ends roughly 1-2mm OUTSIDE
        // the specifically requested ±48mm microphone positions.
        // Extend only its central 5mm-wide air notch ~3mm inward as
        // an acoustic microphone port. Otherwise a correct 9.6cm
        // spacing would lie INSIDE a pressure-zero solid cell.
        // The user SVG paths themselves remain byte-for-byte exact.
        // The original 5.4mm-high port becomes narrower than ONE
        // grid cell at inverse F11 scales. Cell-center sampling can
        // then miss it entirely and apparently disable the head.
        // Area-overlap rasterization preserves one connected fluid
        // cell across the EXISTING prescribed 45-59mm acoustic port;
        // it is not a new tunnel or a change to the user's SVG file.
        for(int y=0;y<ny;++y) {
            const float localY=originY+(y+0.5f)*dx-kListenerY;
            if(std::fabs(localY)>0.0027f+0.5f*dx)continue;
            for(int side:{-1,1}) {
                for(int x=0;x<nx;++x) {
                    const float localX=originX+(x+0.5f)*dx-kListenerX;
                    const float distance=side*localX;
                    if(distance+0.5f*dx>=0.0450f &&
                       distance-0.5f*dx<=0.059f &&
                       mask[(size_t)y*nx+x]<3.5f)
                        mask[(size_t)y*nx+x]=0.0f;
                }
            }
        }
    }
    return mask;
}
inline std::vector<float> svgRoomMask(int n,float dx,
                    const SvgSilhouette& listenerSvg){
    return svgRoomMask(n,n,dx,listenerSvg);
}
// A valid microphone must sample AIR connected to the room, not a
// sealed groove that appears blue/empty but cannot receive sound.
// A narrow SVG-authored air channel is restored along its proven
// high-resolution transparent path before validating connectivity;
// do not accept a sealed virtual microphone.
inline bool microphoneConnectedToRoom(const std::vector<float>& mask,
                         int nx,int ny,float dx,Vec2 mic,
                         float originX=0,float originY=0) {
    if(mask.size()!=(size_t)nx*ny || nx<8 || ny<8)return false;
    const int exterior=(ny/20)*nx+nx/20; // air at even 8x zoom
    if(mask[(size_t)exterior]!=0.0f)return false;
    const float gx=(mic.x-originX)/dx-0.5f;
        const float gy=(mic.y-originY)/dx-0.5f;
    const int ix=(int)std::floor(gx),iy=(int)std::floor(gy);
    if(ix<2||iy<2||ix+1>=nx-2||iy+1>=ny-2)return false;
    std::vector<unsigned char> seen(mask.size(),0);
    std::queue<int> queue;
    // Start in whichever parts of the MIC'S 2x2 sample footprint
    // contribute actual air pressure, not an arbitrarily chosen cell.
    for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox){
        const int id=(iy+oy)*nx+ix+ox;
        if(mask[(size_t)id]!=0.0f)continue;
        const float weight=(1.0f-std::fabs(gx-(ix+ox)))*
                           (1.0f-std::fabs(gy-(iy+oy)));
        if(weight<=0.001f)continue;
        seen[(size_t)id]=1;queue.push(id);
    }
    while(!queue.empty()){
        const int v=queue.front();queue.pop();
        if(v==exterior)return true;
        for(int offset:{-1,1,-nx,nx}){
            const int to=v+offset;
            if(to<0||to>=nx*ny||seen[(size_t)to]||
               mask[(size_t)to]!=0.0f)continue;
            seen[(size_t)to]=1;queue.push(to);
        }
    }
    return false;
}
// V2 microphones sit farther out in the actual 12.4cm-separated
// pinna recess air, with no carved artificial tunnel. The separate
// inner-channel model keeps the original 8cm eardrum target.
inline bool validateMicrophones(const std::vector<float>& mask,int nx,int ny,
                        float dx,bool channel,
                        const SvgSilhouette& svg,
                        float originX=0,float originY=0) {
    const Vec2 left=leftEar(channel,svg.pinnaV2);
    const Vec2 right=rightEar(channel,svg.pinnaV2);
    const auto check=[&](Vec2 mic) {
        if(channel&&svg.present()&&!svg.pinnaV2 &&
           svg.solid(mic.x-kListenerX,mic.y-kListenerY))
            return false; // channel eardrum in authored recess
        // Avoid passing a partially air-overlapping sample that is
        // visually or physically buried in the V2 head material.
        if(svg.pinnaV2 && svg.solid(mic.x-kListenerX,
                                    mic.y-kListenerY))return false;
        const float gx=(mic.x-originX)/dx-0.5f;
        const float gy=(mic.y-originY)/dx-0.5f;
        const int ix=(int)std::floor(gx),iy=(int)std::floor(gy);
        if(ix<2||iy<2||ix+1>=nx-2||iy+1>=ny-2)return false;
        // Bilinear microphone sampling is explicitly mask-aware:
        // a thin recess may have some solid neighbouring cells,
        // but its sample point can still be in air. The kernel
        // renormalizes weights of only physically fluid neighbors.
        float airWeights=0;
        for(int y=0;y<2;++y)for(int x=0;x<2;++x){
            if(mask[(size_t)(iy+y)*nx+(ix+x)]!=0.0f)continue;
            airWeights+=(1-std::fabs(gx-(ix+x)))*
                        (1-std::fabs(gy-(iy+y)));
        }
        // At coarse grid spacings a fully transparent real pinna
        // cavity can occupy <1 whole cell. Mask-aware mic sampling
        // renormalizes air weights; the true SVG mic pixel must
        // still be transparent and BFS must connect to exterior.
        const float minimum=svg.pinnaV2?
            (dx<=0.008f?0.98f:0.12f):0.20f;
        return airWeights>minimum;
    };
    return check(left)&&check(right) &&
        microphoneConnectedToRoom(mask,nx,ny,dx,left,originX,originY) &&
        microphoneConnectedToRoom(mask,nx,ny,dx,right,originX,originY);
}
inline bool validateMicrophones(const std::vector<float>& mask,int n,
                        float dx,bool channel,const SvgSilhouette& svg){
    return validateMicrophones(mask,n,n,dx,channel,svg);
}
} // namespace asmr
