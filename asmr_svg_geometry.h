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
inline std::vector<float> svgRoomMask(int n,float dx,
                    const SvgSilhouette& listenerSvg){
    std::vector<float> mask=roomMask(n,dx);
    if(!listenerSvg.present())return mask;
    // Remove the former circular fallback ENTIRELY before stamping the
    // real SVG, including above and below its narrower child-size bbox.
    // Otherwise ghost semicircles remain around the authored silhouette.
    for(float& material:mask)
        if(material>0.0f && material<3.5f)material=0.0f;
    const int xmin=std::max(1,(int)std::floor(
        (kListenerX-listenerSvg.widthMetres*0.5f)/dx)-1);
    const int xmax=std::min(n-2,(int)std::ceil(
        (kListenerX+listenerSvg.widthMetres*0.5f)/dx)+1);
    const int ymin=std::max(1,(int)std::floor(
        (kListenerY-listenerSvg.heightMetres*0.5f)/dx)-1);
    const int ymax=std::min(n-2,(int)std::ceil(
        (kListenerY+listenerSvg.heightMetres*0.5f)/dx)+1);
    for(int y=ymin;y<=ymax;++y)
        for(int x=xmin;x<=xmax;++x) {
            const size_t i=(size_t)y*n+x;
            if(mask[i]==echo::kOuterBoundaryMaterial)continue;
            const float worldX=(x+0.5f)*dx,worldY=(y+0.5f)*dx;
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
    // V2 microphones are IN AUTHORED AIR at +-62mm. Do not erase
    // solid head material to create synthetic canals for this mode.
    if(!listenerSvg.channels) {
        // The author's simple SVG recess ends roughly 1-2mm OUTSIDE
        // the specifically requested ±48mm microphone positions.
        // Extend only its central 5mm-wide air notch ~3mm inward as
        // an acoustic microphone port. Otherwise a correct 9.6cm
        // spacing would lie INSIDE a pressure-zero solid cell.
        // The user SVG paths themselves remain byte-for-byte exact.
        for(int y=0;y<n;++y) {
            const float localY=(y+0.5f)*dx-kListenerY;
            if(std::fabs(localY)>0.0027f)continue;
            for(int side:{-1,1}) {
                for(int x=0;x<n;++x) {
                    const float localX=(x+0.5f)*dx-kListenerX;
                    const float distance=side*localX;
                    if(distance>=0.0450f&&distance<=0.059f &&
                       mask[(size_t)y*n+x]<3.5f)
                        mask[(size_t)y*n+x]=0.0f;
                }
            }
        }
    }
    return mask;
}
// V2 microphones sit farther out in the actual 12.4cm-separated
// pinna recess air, with no carved artificial tunnel. The separate
// inner-channel model keeps the original 8cm eardrum target.
inline bool validateMicrophones(const std::vector<float>& mask,int n,
                        float dx,bool channel,
                        const SvgSilhouette& svg) {
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
        const float gx=mic.x/dx-0.5f,gy=mic.y/dx-0.5f;
        const int ix=(int)std::floor(gx),iy=(int)std::floor(gy);
        if(ix<2||iy<2||ix+1>=n-2||iy+1>=n-2)return false;
        // Bilinear microphone sampling is explicitly mask-aware:
        // a thin recess may have some solid neighbouring cells,
        // but its sample point can still be in air. The kernel
        // renormalizes weights of only physically fluid neighbors.
        float airWeights=0;
        for(int y=0;y<2;++y)for(int x=0;x<2;++x){
            if(mask[(size_t)(iy+y)*n+(ix+x)]!=0.0f)continue;
            airWeights+=(1-std::fabs(gx-(ix+x)))*
                        (1-std::fabs(gy-(iy+y)));
        }
        return airWeights>(svg.pinnaV2?0.98f:0.20f);
    };
    return check(left)&&check(right);
}
} // namespace asmr
