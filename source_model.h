#pragma once
// Portable geometry and signals for FDTD emitters. No SDL, OpenGL or CUDA
// dependency in CPU tests; functions used by kernels compile host/device.
//
// Player sound: a two-point phase-shifted array. A rear NEGATIVE click is
// delayed by the acoustic transit time across the sources. This cancels the
// outward wave BEHIND the player in free field, not the reflected wave from
// arbitrary objects. It is directional FDTD, not an occluding head/wedge,
// ray tracer, HRTF, direct-path subtraction or perfect echo isolation.
#include "acoustics.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef __CUDACC__
#define ECHO_SOURCE_HD __host__ __device__
#else
#define ECHO_SOURCE_HD
#endif

namespace echo {
// The original microphones sat almost exactly left/right of the source.
// A two-element forward beam has poor lateral suppression there; worse,
// near-field interference can INCREASE direct pressure at those positions.
// Keep the ears behind all source elements, inside the 13cm player
// collision circle: 6.5cm back and 4.5cm lateral (~7.9cm radial).
// Narrower 9cm inter-ear spacing makes the two dots sit close to the wedge.
// This is not rigid head scattering; it is deliberate receiver placement.
struct ReceiverLayout {
    float leftX=0,leftY=0,rightX=0,rightY=0;
};
inline ReceiverLayout shieldedReceivers(float x,float y,float heading) {
    const float fx=std::cos(heading),fy=std::sin(heading);
    const float lx=-fy,ly=fx;
    return {x-6.5f*fx+4.5f*lx,y-6.5f*fy+4.5f*ly,
            x-6.5f*fx-4.5f*lx,y-6.5f*fy-4.5f*ly};
}
struct PhasedSource {
    int frontX=0,frontY=0,backX=0,backY=0,backDelay=0;
    bool enabled=false;
};
inline PhasedSource phasedSource(int side,const float* walls,
    float centerX,float centerY,float heading) {
    PhasedSource out;
    if(!walls||side<20)return out;
    const float fx=std::cos(heading),fy=std::sin(heading);
    // Keep BOTH sources within the player's collision ring. The forward
    // point is close to the old 5.7cm source; the other is ~0.3cm behind
    // the pivot, not physically behind the ear. Grid rounding matters.
    const float frontDistance=6.0f;
    const float backDistance=0.0f;
    out.frontX=(int)std::lround(centerX+frontDistance*fx);
    out.frontY=(int)std::lround(centerY+frontDistance*fy);
    out.backX=(int)std::lround(centerX+backDistance*fx);
    out.backY=(int)std::lround(centerY+backDistance*fy);
    auto freeCell=[&](int x,int y){
        return x>1&&y>1&&x<side-2&&y<side-2&&
               walls[y*side+x]<=0.0f;
    };
    if(!freeCell(out.frontX,out.frontY))return out;
    if(!freeCell(out.backX,out.backY) ||
       (out.frontX==out.backX && out.frontY==out.backY))return out;
    const float separation=std::hypot((float)(out.frontX-out.backX),
                                       (float)(out.frontY-out.backY));
    out.backDelay=(int)std::lround(
        separation*kCellMetres/(kSpeedOfSound*kStepSeconds));
    out.enabled=out.backDelay>0;
    return out;
}
// A two-element endfire source suppresses the BACK but leaves much of
// the SIDE of a 2D wavefront intact. A simple 3-across aperture
// (0.25/0.50/0.25 at -8/0/+8 cm lateral displacement) steers the
// surviving radiation into a narrower forward lobe at 1.8 kHz.
// Each column is still a paired front/phase-delayed-negative-rear FDTD
// pressure source, NOT an opaque wedge or a ray-only acoustic filter.
// All six source cells fit inside the 13cm player collision circle.
struct BeamArray {
    PhasedSource elements[3];
    float weights[3]={0.0f,1.0f,0.0f};
    bool directional=false;
    bool valid=false;
};
inline BeamArray playerBeam(int side,const float* walls,
    float x,float y,float heading,bool narrow=true) {
    BeamArray array;
    const auto central=phasedSource(side,walls,x,y,heading);
    if(!central.enabled)return array; // leave the main CUDA click inactive
    array.elements[1]=central;
    array.valid=true;
    if(!narrow)return array; // B-key A/B: original single monopole
    array.directional=true;
    const float forwardX=std::cos(heading),forwardY=std::sin(heading);
    const float sideX=-forwardY,sideY=forwardX;
    PhasedSource sides[2];
    bool allGood=true;
    for(int k=0;k<2;++k){
        const float offset=k==0?-8.0f:8.0f;
        auto &p=sides[k];
        p.frontX=(int)std::lround(x+6.0f*forwardX+offset*sideX);
        p.frontY=(int)std::lround(y+6.0f*forwardY+offset*sideY);
        p.backX=(int)std::lround(x+offset*sideX);
        p.backY=(int)std::lround(y+offset*sideY);
        auto freeCell=[&](int cx,int cy){
            return cx>1&&cy>1&&cx<side-2&&cy<side-2&&
                   walls[cy*side+cx]<=0.0f;
        };
        p.enabled=freeCell(p.frontX,p.frontY)&&freeCell(p.backX,p.backY);
        if(p.enabled){
            const float separation=std::hypot(
                (float)(p.frontX-p.backX),(float)(p.frontY-p.backY));
            p.backDelay=(int)std::lround(
                separation*kCellMetres/(kSpeedOfSound*kStepSeconds));
            p.enabled=p.backDelay>0;
        }
        allGood=allGood&&p.enabled;
    }
    if(!allGood)return array; // fallback: central endfire, no clipped array
    array.elements[0]=sides[0];
    array.elements[2]=sides[1];
    array.weights[0]=0.25f;array.weights[1]=0.50f;
    array.weights[2]=0.25f;
    return array;
}

// Every source contributes to the SAME linear pressure update: the delayed
// back element cancels the source's backward wave by phase interference.
// Its emitted pulse may extend beyond the front click by backDelay steps.
ECHO_SOURCE_HD inline float phasedPlayerClick(
    int pulseStep,int duration,float dt,bool back,int delay) {
    return back?-clickSample(pulseStep-delay,duration,dt):
                clickSample(pulseStep,duration,dt);
}

// Polygon occupancy used by reflective, visually faceted game objects.
// Even-sided vertices permit stable CPU masks with exactly flat edges.
inline bool insideNgon(float dx,float dy,float radius,int edges,
                       float rotation=0.0f) {
    if(radius<=0||edges<3)return false;
    const float angle=std::atan2(dy,dx)-rotation;
    const float sector=2.0f*kPi/(float)edges;
    // Angle relative to closest polygon vertex, in [-sector/2,+sector/2].
    const float difference=std::fabs(std::remainder(angle,sector));
    const float angleFromEdgeNormal=0.5f*sector-difference;
    const float maximum=radius*std::cos(kPi/(float)edges)/
                        std::cos(angleFromEdgeNormal);
    return dx*dx+dy*dy <= maximum*maximum;
}

// One encoded float per grid point: integer part = emitter class;
// fractional part = local 2x2 source weight. The CUDA kernel only branches
// on nonzero tiles, not through every enemy/item for every grid cell.
enum : int {
    kGhostTracker=1,kGhostSprinter=2,kGhostSentinel=3,
    kShardFirst=4,kShardLast=15,kArcadeExit=16,
    kSvgGoal=17,kSvgItem=18,kSvgObstacle=19
};
ECHO_SOURCE_HD inline float toneFrequency(int id) {
    if(id==kGhostTracker)return 660.0f;
    if(id==kGhostSprinter)return 840.0f;
    if(id==kGhostSentinel)return 530.0f;
    if(id>=kShardFirst && id<=kShardLast)
        return 1020.0f+(id-kShardFirst)*14.0f;
    if(id==kArcadeExit)return 1240.0f;
    if(id==kSvgGoal)return 1180.0f;
    if(id==kSvgItem)return 990.0f;
    if(id==kSvgObstacle)return 720.0f;
    return 0.0f;
}
ECHO_SOURCE_HD inline float fdtdWorldTone(float encoded,
                                           uint64_t step,float dt) {
    if(encoded<1.0f)return 0.0f;
    const int id=(int)encoded;
    const float weight=(encoded-(float)id)/0.9f;
    if(weight<=0.0f)return 0.0f;
    const float frequency=toneFrequency(id);
    // Brief, fully tapered pulses rather than perpetually driven resonant
    // sine waves (the reported 'spaceship engine' spanning a 5m level).
    // Every object has a different phase so its ping is distinguishable.
    // Energy is injected INTO FDTD; the location gate is applied on the CPU.
    const double t=(double)step*(double)dt;
    const double cycle=fmod(t+0.157*(double)id,2.15);
    constexpr float duration=0.085f;
    if(cycle>=(double)duration)return 0.0f;
    const float envelope=0.5f-0.5f*cosf(
        2.0f*kPi*(float)cycle/duration);
    const double phase=fmod(t*(double)frequency,1.0);
    const float amplitude=0.15f*weight*envelope;
    return amplitude*sinf(2.0f*kPi*(float)phase);
}

// Place one emitter using bilinear weights in the nearest *fluid* cells.
// Source tiles must not be INSIDE a solid ghost, candy, or static wall.
// Move no harder wall cells in order to make an audible target.
inline bool placeWorldEmitter(std::vector<float>& encoded,
                               const float* mask,int side,
                               float x,float y,int id,float radiusCm=0,
                               float strength=1.0f) {
    if(id<1||id>kSvgObstacle||!mask||side<4||
       strength<=0.0f||!std::isfinite(strength)||
       encoded.size()!=(size_t)side*side)return false;
    strength=std::min(1.0f,strength);
    const float directions[12][2]={
        {1,0},{0,1},{-1,0},{0,-1},{.70710678f,.70710678f},
        {-.70710678f,.70710678f},{-.70710678f,-.70710678f},
        {.70710678f,-.70710678f},{.92387953f,.38268343f},
        {-.38268343f,.92387953f},{-.92387953f,-.38268343f},
        {.38268343f,-.92387953f}};
    float sourceX=x,sourceY=y;
    bool found=false;
    for(int ring=0;ring<9 && !found;++ring) {
        const float r=radiusCm+(float)ring*2.0f+2.0f;
        const int dirs=(radiusCm>0||ring>0)?12:1;
        for(int d=0;d<dirs;++d) {
            const float sx=(radiusCm>0||ring>0)?x+directions[d][0]*r:x;
            const float sy=(radiusCm>0||ring>0)?y+directions[d][1]*r:y;
            const int px=(int)std::floor(sx),py=(int)std::floor(sy);
            if(px<=1||py<=1||px>=side-2||py>=side-2)continue;
            if(mask[py*side+px]>0 ||
               mask[py*side+px+1]>0 ||
               mask[(py+1)*side+px]>0 ||
               mask[(py+1)*side+px+1]>0)continue;
            sourceX=sx;sourceY=sy;found=true;break;
        }
    }
    if(!found)return false;
    const int ix=(int)std::floor(sourceX),iy=(int)std::floor(sourceY);
    const float fx=sourceX-ix,fy=sourceY-iy;
    for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
        const float w=(dx?fx:1-fx)*(dy?fy:1-fy)*strength;
        if(w<0.001f)continue;
        const int index=(iy+dy)*side+(ix+dx);
        const float prior=encoded[index];
        // Overlap is rare; preserve whichever source is locally strongest.
        if(prior==0.0f || prior-std::floor(prior) < w*0.9f)
            encoded[index]=(float)id+0.9f*w;
    }
    return true;
}
} // namespace echo
#undef ECHO_SOURCE_HD
