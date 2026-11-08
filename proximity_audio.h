#pragma once
// Strictly local non-FDTD discoverability hints. Never continuously chime at
// a target hundreds of centimetres away or through an opaque SVG wall.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace echo {
inline float markerRadiusCm(char type) {
    return type=='O'?24.0f:type=='I'?30.0f:35.0f;
}
inline float proximityStrength(float distanceCm,float radiusCm) {
    if(!std::isfinite(distanceCm)||distanceCm<0.0f||
       !std::isfinite(radiusCm)||radiusCm<=0.0f||
       distanceCm>=radiusCm)return 0.0f;
    const float near=1.0f-distanceCm/radiusCm;
    return near*near; // no abrupt level jump at range boundary
}
// Passability check intentionally models only hard occlusion. It does NOT
// constitute wave diffraction, binaural hearing or physical propagation.
inline bool hasClearSight(const float* walls,int side,
                          float ax,float ay,float bx,float by,
                          float targetClearanceCm=5.0f) {
    if(!walls||side<2)return false;
    const float dx=bx-ax,dy=by-ay;
    const float range=std::hypot(dx,dy);
    if(!std::isfinite(range))return false;
    if(range<0.01f)return true;
    const float maxT=std::max(0.0f,range-targetClearanceCm);
    for(float t=1.0f;t<maxT;t+=1.0f) {
        const int x=(int)std::floor(ax+dx*t/range);
        const int y=(int)std::floor(ay+dy*t/range);
        if(x<0||y<0||x>=side||y>=side||
           walls[y*side+x]>0.0f)return false;
    }
    return true;
}
// Decide which objects are allowed to emit a discoverable FDTD cue, not
// whether an FDTD wave is allowed to pass through a wall. Continuous
// oscillators used to fill an entire 5m room with a stationary drone; at
// this 1cm/2D grid there is also much less distance spreading than in 3D.
// These proximity-limited, line-of-sight-verified pulses are deliberately
// gameplay cues, not a physical inverse-square law or diffraction model.
inline float localEmitterGain(const float* staticWalls,int side,
    float listenerX,float listenerY,float sourceX,float sourceY,
    float maxDistanceCm) {
    if(!staticWalls||side<2||maxDistanceCm<=0.0f)return 0.0f;
    const float distance=std::hypot(sourceX-listenerX,
                                    sourceY-listenerY);
    if(distance>=maxDistanceCm ||
       !hasClearSight(staticWalls,side,listenerX,listenerY,
                      sourceX,sourceY,0.0f))return 0.0f;
    const float strength=1.0f-distance/maxDistanceCm;
    return strength*strength; // smooth fade at range threshold
}

// Allow a quiet chime on first discovery and rare reminders while still
// close. Crossing the outer hysteresis radius rearms the discovery event.
struct ProximityGate {
    bool inside=false;
    double lastTime=-1000;
    bool shouldChime(float distanceCm,float radiusCm,double now) {
        if(!std::isfinite(distanceCm)||radiusCm<=0)return false;
        if(distanceCm>=radiusCm+6.0f)inside=false;
        if(distanceCm>=radiusCm)return false;
        const bool initial=!inside;
        inside=true;
        if((initial && now-lastTime>0.35) ||
           (!initial && now-lastTime>=4.5)) {
            lastTime=now;
            return true;
        }
        return false;
    }
};
inline std::vector<float> localChime(int sampleRate,char type,
                                    float leftGain,float rightGain,
                                    bool rear=false) {
    if(sampleRate<=0)return {};
    const int count=type=='O'?1:type=='I'?2:3;
    const float noteTime=0.13f;
    const int n=(int)std::ceil(sampleRate*count*noteTime);
    std::vector<float> result(2*n,0.0f);
    const float base=type=='O'?390.0f:type=='I'?750.0f:1050.0f;
    for(int i=0;i<n;++i) {
        const float t=(float)i/sampleRate;
        const int note=std::min(count-1,(int)(t/noteTime));
        const float within=t-note*noteTime;
        if(within>=0.075f)continue;
        const float env=std::sin(3.14159265358979323846f*within/0.075f);
        const float frequency=(base+note*(type=='O'?0.0f:120.0f))*
                              (rear?0.74f:1.0f);
        const float v=env*env*std::sin(
            2.0f*3.14159265358979323846f*frequency*within);
        result[2*i]=v*leftGain;
        result[2*i+1]=v*rightGain;
    }
    return result;
}
// Reusable stereo bearing. Positive angular offset means LEFT of heading,
// not right. Extra near-ear weighting avoids identical levels everywhere.
struct LocalCue {
    float left=0,right=0;
    bool rear=false;
};
inline LocalCue localBearing(float playerX,float playerY,float heading,
                             float targetX,float targetY,float strength) {
    if(strength<=0)return {};
    const float dx=targetX-playerX,dy=targetY-playerY;
    const float d=std::max(0.001f,std::hypot(dx,dy));
    const float angle=std::atan2(dy,dx)-heading;
    const float sinSide=std::sin(angle);
    const float halfEarCm=8.0f;
    const float dl=std::sqrt(std::max(0.0f,d*d+halfEarCm*halfEarCm-
                                 2*d*halfEarCm*sinSide));
    const float dr=std::sqrt(std::max(0.0f,d*d+halfEarCm*halfEarCm+
                                 2*d*halfEarCm*sinSide));
    const float nearest=std::min(dl,dr);
    const float maxGain=std::min(0.095f,0.11f*strength);
    return {maxGain*(nearest+5.0f)/(dl+5.0f),
            maxGain*(nearest+5.0f)/(dr+5.0f),std::cos(angle)<0.0f};
}
}
