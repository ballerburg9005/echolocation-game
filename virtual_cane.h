#pragma once
// Virtual white cane: three short grid probes with distinct stereo taps.
// This is an OPTIONAL sonification/navigation aid, NOT a simulated physical
// cane, force-feedback system, echo, or binaural impulse-response model.
#include <cmath>
#include <algorithm>
#include <array>

namespace echo {
struct CaneHit {
    float distanceCm=0;
    float x=0,y=0;
    bool hit=false;
};

inline CaneHit castCaneRay(const float* mask,int side,float px,float py,
                           float bearing,float maximum=100.0f) {
    CaneHit out;
    if(!mask || side<2 || !std::isfinite(px) || !std::isfinite(py) ||
       !std::isfinite(bearing) || !std::isfinite(maximum) || maximum<=0)
        return out;
    const float vx=std::cos(bearing);
    const float vy=std::sin(bearing);
    out.x=px;out.y=py;
    for(float t=0.0f;t<=maximum;t+=0.5f) {
        const float x=px+vx*t,y=py+vy*t;
        const int xx=(int)std::floor(x),yy=(int)std::floor(y);
        if(xx<0||yy<0||xx>=side||yy>=side||
           mask[yy*side+xx]>0.0f) {
            return CaneHit{t,x,y,true};
        }
        out=CaneHit{t,x,y,false};
    }
    return out;
}
inline std::array<CaneHit,3> sweepCane(const float* mask,int side,
                                       float x,float y,float facing,
                                       float range=100.0f) {
    constexpr float spread=0.4188790204786391f; // 24 degrees
    // Order left, center, right: positive mathematical y is left of facing.
    return {{
        castCaneRay(mask,side,x,y,facing+spread,range),
        castCaneRay(mask,side,x,y,facing,range),
        castCaneRay(mask,side,x,y,facing-spread,range)
    }};
}
} // namespace echo
