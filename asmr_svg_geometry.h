#pragma once
// Render EXACT supplied Inkscape SVG paths, including elliptical arcs,
// cubics, nested transforms and the two halves of the ear-channel head,
// into an acoustic occupancy mask. No hand-drawn approximate polygons,
// inferred circles, synthetic HRTF or ear material deletion.
#include "asmr_lab_math.h"
#include <librsvg/rsvg.h>
#include <cairo.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace asmr {
enum class HeadStyle {Simple,Channels};
struct SvgSilhouette {
    float widthMetres=0, heightMetres=0;
    int w=0,h=0;
    std::vector<unsigned char> alpha;
    bool channels=false;
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

inline SvgSilhouette renderSvg(const std::string& filename,
                              float rasterPixelMetres=0.00040f){
    SvgSilhouette out;
    GError* error=nullptr;
    RsvgHandle* handle=rsvg_handle_new_from_file(filename.c_str(),&error);
    if(!handle) {
        std::fprintf(stderr,"[SVG HEAD] %s: %s\n",
            filename.c_str(),error?error->message:"cannot open");
        if(error)g_error_free(error);
        return out;
    }
    gboolean hw=FALSE,hh=FALSE,hv=FALSE;
    RsvgLength width{},height{};
    RsvgRectangle view{};
    rsvg_handle_get_intrinsic_dimensions(handle,&hw,&width,
                                         &hh,&height,&hv,&view);
    // In these user-supplied SVGs viewBox units are mm and width/height
    // are physically specified in mm, hence 1 viewBox unit = 1mm.
    if(!hv||view.width<50||view.height<50||
       view.width>500||view.height>500) {
        std::fprintf(stderr,
            "[SVG HEAD] Missing mm viewBox in %s\n",filename.c_str());
        g_object_unref(handle);
        return out;
    }
    out.widthMetres=(float)(view.width*0.001);
    out.heightMetres=(float)(view.height*0.001);
    out.w=std::max(32,(int)std::ceil(out.widthMetres/rasterPixelMetres));
    out.h=std::max(32,(int)std::ceil(out.heightMetres/rasterPixelMetres));
    cairo_surface_t* surface=cairo_image_surface_create(
        CAIRO_FORMAT_ARGB32,out.w,out.h);
    cairo_t* cr=cairo_create(surface);
    cairo_set_operator(cr,CAIRO_OPERATOR_CLEAR);
    cairo_paint(cr);
    cairo_set_operator(cr,CAIRO_OPERATOR_OVER);
    const RsvgRectangle target{
        0.0,0.0,(double)out.w,(double)out.h
    };
    const gboolean ok=rsvg_handle_render_document(handle,cr,
                                                    &target,&error);
    cairo_destroy(cr);
    if(ok){
        cairo_surface_flush(surface);
        const int stride=cairo_image_surface_get_stride(surface);
        const auto* pixels=cairo_image_surface_get_data(surface);
        out.alpha.assign((size_t)out.w*out.h,0);
        for(int y=0;y<out.h;++y)
            for(int x=0;x<out.w;++x) {
                const unsigned char* p=pixels+y*stride+x*4;
                // Native endian ARGB32: stored A in byte 3 on x86.
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                const unsigned char a=p[3];
#else
                const unsigned char a=p[0];
#endif
                out.alpha[(size_t)y*out.w+x]=a;
            }
    }else{
        std::fprintf(stderr,"[SVG HEAD] render error in %s: %s\n",
            filename.c_str(),error?error->message:"unknown");
        out=SvgSilhouette{};
    }
    if(error)g_error_free(error);
    cairo_surface_destroy(surface);
    g_object_unref(handle);
    return out;
}

inline std::string svgLocation(const char* filename) {
    // The user keeps their authored SVGs under HOME; prefer those,
    // falling back to exact copies committed with the experiment.
    const char* home=std::getenv("HOME");
    if(home&&*home){
        std::string direct=std::string(home)+"/"+filename;
        if(FILE* file=std::fopen(direct.c_str(),"rb")){
            std::fclose(file);return direct;
        }
    }
    return std::string("asmr_shapes/")+filename;
}
inline SvgSilhouette listenerSilhouette(HeadStyle style){
    SvgSilhouette shape=renderSvg(svgLocation(
        style==HeadStyle::Channels?
        "listener_with_ear_channel.svg":"listener.svg"));
    shape.channels=style==HeadStyle::Channels;
    return shape;
}
inline SvgSilhouette speakerSilhouette(){
    return renderSvg(svgLocation("speaker.svg"));
}
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
            if(mask[i]==kOuterBoundaryMaterial)continue;
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
    return mask;
}
// These positions are deliberately specified by this user's 8cm child
// eardrum-to-eardrum experiment, not generic adult HRTF anthropology.
// For the simple head, recess microphones are 8cm + 0.8*2cm = 9.6cm
// apart. In channel mode, sample at x=±4cm deep in the OPEN canals.
inline bool validateMicrophones(const std::vector<float>& mask,int n,
                        float dx,bool channel,
                        const SvgSilhouette& svg) {
    const Vec2 left=leftEar(channel),right=rightEar(channel);
    const auto check=[&](Vec2 mic) {
        if(svg.present() &&
           svg.solid(mic.x-kListenerX,mic.y-kListenerY))
            return false; // user-specified eardrum must be in AIR
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
        return airWeights>0.20f;
    };
    return check(left)&&check(right);
}
} // namespace asmr
