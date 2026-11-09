#include "asmr_svg_geometry.h"
#include <librsvg/rsvg.h>
#include <cairo.h>

// Third-party GLib/Cairo headers must be compiled by g++, not nvcc:
// CUDA's host_defines.h conflicts with glib's __has_attribute macros.
namespace asmr {
SvgSilhouette renderSvg(const std::string& filename,
                              float rasterPixelMetres){
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

std::string svgLocation(const char* filename) {
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
SvgSilhouette listenerSilhouette(HeadStyle style){
    const char* filename=style==HeadStyle::PinnaV2?"listener_v2.svg":
                         style==HeadStyle::Channels?
                         "listener_with_ear_channel.svg":"listener.svg";
    SvgSilhouette shape=renderSvg(svgLocation(filename));
    shape.channels=style!=HeadStyle::Simple;
    shape.pinnaV2=style==HeadStyle::PinnaV2;
    return shape;
}
SvgSilhouette speakerSilhouette(){
    return renderSvg(svgLocation("speaker.svg"));
}

} // namespace asmr
