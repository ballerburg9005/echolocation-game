#pragma once
// Acoustically moving SPEAKER silhouette. Same physical SVG alpha as
// the 2D actor texture, not a proxy circle or a newly clipped hard mask.
// Moving impedance lives in the ORIGINAL pressure stencil with no
// pressure-cell resets. Inputs are acoustic GRID coordinates.
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define ASMR_HEAD_HD __host__ __device__
#else
#define ASMR_HEAD_HD
#endif
namespace asmr {
ASMR_HEAD_HD inline float movingHeadCoverage(
    const unsigned char* alpha,int width,int height,
    float widthInCells,float heightInCells,
    float centreX,float centreY,float forwardX,float forwardY,
    float x,float y) {
    if(!alpha||width<2||height<2||widthInCells<=0||heightInCells<=0)
        return 0.0f;
    const float dx=x-centreX,dy=y-centreY;
    // Avoid coordinate transforms and four alpha loads on almost every
    // CUDA thread in the 2m acoustic room: only the ~15cm actor box.
    const float halfDiagonal=0.5f*sqrtf(
        widthInCells*widthInCells+heightInCells*heightInCells)+1.5f;
    if(fabsf(dx)>halfDiagonal||fabsf(dy)>halfDiagonal)return 0.0f;
    // Speaker SVG front opening points along local -Y. It rotates
    // exactly with the visual head to face the listener.
    const float localX=-forwardY*dx+forwardX*dy;
    const float localY=-forwardX*dx-forwardY*dy;
    const float u=(localX/widthInCells+0.5f)*width-0.5f;
    const float v=(0.5f-localY/heightInCells)*height-0.5f;
    if(u<0.0f||v<0.0f||u>width-1.0f||v>height-1.0f)
        return 0.0f;
    const int ix=(int)floorf(u),iy=(int)floorf(v);
    const int jx=ix+1<width?ix+1:ix;
    const int jy=iy+1<height?iy+1:iy;
    const float fx=u-ix,fy=v-iy;
    const float a=(1-fx)*alpha[iy*width+ix]+
                    fx*alpha[iy*width+jx];
    const float b=(1-fx)*alpha[jy*width+ix]+
                    fx*alpha[jy*width+jx];
    // The user's SVG has opacity=0.5, so fully filled pixels are
    // alpha~128, not 255. Normalize without falsely weakening geometry.
    const float occupancy=(a*(1-fy)+b*fy)/128.0f;
    return fminf(1.0f,fmaxf(0.0f,occupancy));
}
ASMR_HEAD_HD inline float movingHeadNeighbor(float center,float neighbor,
                                             float coverageHere,
                                             float coverageThere) {
    const float face=fmaxf(coverageHere,coverageThere);
    const float transmissivity=fmaxf(0.10f,1.0f-0.90f*face);
    return center+(neighbor-center)*transmissivity;
}
} // namespace asmr
#undef ASMR_HEAD_HD
