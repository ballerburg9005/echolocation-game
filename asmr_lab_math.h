#pragma once
// Portable geometry, recorded paths and sample-rate constraints for the
// independent ASMR experiment. CPU tests have no CUDA or SDL dependency.
#include "acoustics.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace asmr {
constexpr float kRoomMetres=2.0f;
constexpr float kListenerX=1.0f;
constexpr float kListenerY=1.0f;
constexpr float kHeadRadius=0.085f;
constexpr float kArtistRadius=0.085f;
constexpr float kEarOffset=0.105f;  // 21cm binaural distance, outside head
constexpr float kMouthOffset=0.075f;
constexpr float kHeadClearance=0.025f;
constexpr float kMinimumSeparation=kHeadRadius+kArtistRadius+kHeadClearance;
constexpr float kRoomClearance=kArtistRadius+0.025f;
constexpr float kLiveDx=0.005f;    // 400² cells / 6.86 kHz at 10 cells per wave
constexpr float kMediumDx=0.008f;  // 250² cells, graceful fallback on weaker GPUs
constexpr float kSafeDx=0.010f;    // original 200² low-power preview
constexpr float kOfflineDx=0.002f;// 1000², 17.15kHz @ 10 cells per wave
constexpr float kLivePassband=5500.0f;
constexpr float kMediumPassband=4000.0f;
constexpr float kSafePassband=2800.0f;
constexpr float kOfflinePassband=15000.0f;
constexpr int kOutputRate=48000;
constexpr int kGraphBatch=126; // multiple of 3 for GPU field ring to repeat in a CUDA Graph
constexpr float kOutsideReflection=0.32f;
constexpr float kHeadReflection=0.74f;

struct Vec2 {float x=0,y=0;};
struct PathPoint {double t=0; Vec2 centre;};
inline Vec2 add(Vec2 a,Vec2 b){return {a.x+b.x,a.y+b.y};}
inline Vec2 sub(Vec2 a,Vec2 b){return {a.x-b.x,a.y-b.y};}
inline Vec2 scale(Vec2 a,float t){return {a.x*t,a.y*t};}
inline float length(Vec2 a){return std::hypot(a.x,a.y);}
inline Vec2 listener(){return {kListenerX,kListenerY};}
inline Vec2 faceListener(Vec2 centre) {
    const Vec2 toward=sub(listener(),centre);
    const float n=length(toward);
    return n>1e-6f?scale(toward,1.0f/n):Vec2{1,0};
}
inline Vec2 mouth(Vec2 centre){
    return add(centre,scale(faceListener(centre),kMouthOffset));
}
// Source location is a continuous path through the world: smoothing
// happens on the physics timeline, not by repeatedly moving backward
// and forward within every GPU audio batch. The mouse's visual 1:1
// position and collision geometry are unaffected. Source smoothing
// removes motion-induced numerical zipper noise, not room echoes.
inline Vec2 advanceSource(Vec2 current,Vec2 target,float dt,
                          float responseSeconds=0.012f,
                          float maxMetresPerSecond=2.25f) {
    if(dt<=0.0f)return current;
    const float alpha=1.0f-std::exp(-dt/responseSeconds);
    Vec2 motion=scale(sub(target,current),alpha);
    const float distance=length(motion);
    const float maximum=maxMetresPerSecond*dt;
    if(distance>maximum && distance>0.0f)
        motion=scale(motion,maximum/distance);
    return add(current,motion);
}

inline bool validArtist(Vec2 p) {
    return std::isfinite(p.x)&&std::isfinite(p.y)&&
        p.x>=kRoomClearance-1e-5f&&p.y>=kRoomClearance-1e-5f&&
        p.x<=kRoomMetres-kRoomClearance+1e-5f&&
        p.y<=kRoomMetres-kRoomClearance+1e-5f&&
        length(sub(p,listener()))>=kMinimumSeparation-1e-5f;
}
inline Vec2 restrictCentre(Vec2 p) {
    p.x=std::clamp(p.x,kRoomClearance,kRoomMetres-kRoomClearance);
    p.y=std::clamp(p.y,kRoomClearance,kRoomMetres-kRoomClearance);
    const Vec2 v=sub(p,listener());
    const float d=length(v);
    if(d<kMinimumSeparation){
        const Vec2 unit=d>1e-6f?scale(v,1.0f/d):Vec2{1,0};
        p=add(listener(),scale(unit,kMinimumSeparation));
    }
    return p;
}
inline Vec2 moveArtist(Vec2 origin,Vec2 target) {
    // A 1:1 mouse-to-world destination, except the circle-collision
    // constraints. Substep the full segment to prevent tunnelling through
    // the stationary listener's head when the mouse teleports across it.
    Vec2 current=restrictCentre(origin);
    target=restrictCentre(target);
    const Vec2 delta=sub(target,current);
    const int steps=std::max(1,(int)std::ceil(length(delta)/0.006f));
    const Vec2 inc=scale(delta,1.0f/steps);
    for(int i=0;i<steps;++i){
        const Vec2 next=add(current,inc);
        if(validArtist(next)){current=next;continue;}
        const Vec2 slideX=restrictCentre({next.x,current.y});
        const Vec2 slideY=restrictCentre({current.x,next.y});
        // Only accept a slide if it is within one substep of current.
        const bool goodX=length(sub(slideX,current))<=0.009f;
        const bool goodY=length(sub(slideY,current))<=0.009f;
        if(goodX&&(!goodY||
            length(sub(slideX,next))<=length(sub(slideY,next))))
            current=slideX;
        else if(goodY)current=slideY;
    }
    return current;
}
inline Vec2 interpolate(const std::vector<PathPoint>& points,double t){
    if(points.empty())return {1.45f,1.0f};
    if(t<=points.front().t)return points.front().centre;
    if(t>=points.back().t)return points.back().centre;
    auto it=std::lower_bound(points.begin(),points.end(),t,
        [](const PathPoint& p,double at){return p.t<at;});
    if(it==points.begin())return it->centre;
    const auto& a=*(it-1);
    const auto& b=*it;
    const float alpha=(float)std::clamp((t-a.t)/(b.t-a.t),0.0,1.0);
    return add(scale(a.centre,1-alpha),scale(b.centre,alpha));
}
struct PathSelection {
    double start=0, duration=0;
    std::vector<PathPoint> points;
};
inline PathSelection selectLast(const std::vector<PathPoint>& record,
                                double now,double seconds){
    PathSelection out;
    if(record.empty()||seconds<=0)return out;
    out.start=std::max(record.front().t,now-seconds);
    out.duration=std::max(0.0,now-out.start);
    out.points.push_back({out.start,interpolate(record,out.start)});
    for(const auto& point:record)
        if(point.t>out.start && point.t<now)out.points.push_back(point);
    out.points.push_back({now,interpolate(record,now)});
    return out;
}
inline int cells(float dx){return (int)std::lround(kRoomMetres/dx);}
inline double stepSeconds(float dx){
    return (double)dx/((double)echo::kSpeedOfSound*
                          (double)echo::kSqrtTwo);
}
inline float tenCellFrequency(float dx){
    return echo::kSpeedOfSound/(10.0f*dx);
}
// Fixed listener head shadow and rubber-like room boundary, both
// represented directly in the SAME FDTD grid. Artist remains a
// collision-checked visual body so it never causes hard wall pops.
inline std::vector<float> roomMask(int n,float dx){
    std::vector<float> mask((size_t)n*n,0.0f);
    const int border=std::max(2,(int)std::ceil(0.012f/dx));
    for(int y=0;y<n;++y)for(int x=0;x<n;++x){
        const int id=y*n+x;
        if(x<border||y<border||x>=n-border||y>=n-border){
            mask[id]=echo::kOuterBoundaryMaterial;continue;
        }
        const float px=((float)x+0.5f)*dx;
        const float py=((float)y+0.5f)*dx;
        if(std::hypot(px-kListenerX,py-kListenerY)<=kHeadRadius)
            mask[id]=1.0f;
    }
    return mask;
}
inline bool probeInAir(const std::vector<float>& mask,int n,
                       Vec2 p,float dx) {
    const float gx=p.x/dx,gy=p.y/dx;
    const int ix=(int)std::floor(gx),iy=(int)std::floor(gy);
    if(ix<2||iy<2||ix+1>=n-2||iy+1>=n-2)return false;
    return mask[iy*n+ix]==0 && mask[iy*n+ix+1]==0 &&
           mask[(iy+1)*n+ix]==0 && mask[(iy+1)*n+ix+1]==0;
}
inline Vec2 leftEar(){return {kListenerX-kEarOffset,kListenerY};}
inline Vec2 rightEar(){return {kListenerX+kEarOffset,kListenerY};}
inline std::vector<float> lowpass(const std::vector<float>& input,
                                  float cutoff,int sampleRate,
                                  int halfWidth=40){
    if(input.empty())return {};
    const float nyquist=0.5f*sampleRate;
    cutoff=std::clamp(cutoff,20.0f,nyquist*0.96f);
    const int taps=2*halfWidth+1;
    std::vector<float> weights(taps,0),out(input.size(),0);
    double norm=0;
    for(int j=-halfWidth;j<=halfWidth;++j) {
        const double f=(double)cutoff/sampleRate;
        const double h=j==0?2.0*f:std::sin(2.0*echo::kPi*f*j)/
                             (echo::kPi*j);
        const double window=0.5+0.5*std::cos(
            (double)echo::kPi*j/(halfWidth+1));
        weights[j+halfWidth]=(float)(h*window);
        norm+=h*window;
    }
    for(float& w:weights)w=(float)(w/norm);
    for(size_t i=0;i<input.size();++i) {
        double sum=0;
        for(int j=-halfWidth;j<=halfWidth;++j) {
            const int64_t index=(int64_t)i+j;
            if(index>=0 && index<(int64_t)input.size())
                sum+=weights[j+halfWidth]*input[(size_t)index];
        }
        out[i]=(float)sum;
    }
    return out;
}
} // namespace asmr
