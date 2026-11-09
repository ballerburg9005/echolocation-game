#pragma once
// Portable geometry, recorded paths and sample-rate constraints for the
// independent ASMR experiment. CPU tests have no CUDA or SDL dependency.
#include "acoustics.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace asmr {
// Preserve the prior square solver's ~284k CELLS while reflowing it
// as a 400x711 portrait grid. 2m wide x (2m*16/9) tall at 5mm is
// physically longer but does NOT multiply the number of simulation cells.
constexpr float kRoomMetres=2.0f; // physical horizontal width
constexpr float kRoomHeightMetres=kRoomMetres*16.0f/9.0f; // 9:16 portrait
constexpr float kListenerX=kRoomMetres*0.5f;
constexpr float kListenerY=kRoomHeightMetres/3.0f; // EXACT bottom third
// Leaves 1/3 (1.19m) behind and 2/3 (2.37m) in front (+Y).
// These are explicitly user-specified experimental CHILD HEAD distances,
// not a claim that every child has the same eardrum spacing.
constexpr float kEardrumSeparation=0.080f; // previous 8cm experimental pair
constexpr float kNewChannelMicSeparation=0.039f; // +/-19.5mm: inner cavity pair
constexpr float kEarCanalLength=0.020f;
constexpr float kChannelEarY=0.002f; // inner cavity: 1mm DOWN from previous +3mm
constexpr float kPinnaEarY=0.003f; // V2 stays 3mm ABOVE, unchanged

constexpr float kOuterMicSeparation=
    kEardrumSeparation+0.8f*kEarCanalLength; // 9.6cm simple notch mics
constexpr float kHeadRadius=0.105f; // encloses 185mm pinna-v2 lateral extent with clearance
constexpr float kArtistRadius=0.070f; // ~13cm diameter SVG sender
constexpr float kEarOffset=kOuterMicSeparation*0.5f;
constexpr float kMouthOffset=0.055f; // 5.5cm: clear of the speaker's recess edge
constexpr float kHeadClearance=0.018f;
constexpr float kMinimumSeparation=kHeadRadius+kArtistRadius+kHeadClearance;
constexpr float kRoomClearance=kArtistRadius+0.018f;
// Live 400x711 ~= the original 533x533 budget. Offline 1000x1778
// ~= the previous 1333x1333 offline budget: BOTH preserve area in
// cell counts while physical room spans 2x3.56m. FDTD step time and
// accurate acoustic bandwidth must change with dx to obey the CFL rule.
constexpr float kLiveDx=0.0050f;  // 400x711 in 9:16 room; ~25% less live cell-work
constexpr float kBalancedDx=0.0045f; // optional higher-quality live preview
constexpr float kPreviousDx=0.00375f; // optional former live preset
constexpr float kMediumDx=0.0030f; // 667², 11.4kHz
constexpr float kUltraDx=0.0025f;  // 800², 13.7kHz
constexpr float kSafeDx=0.0050f;   // 400², 6.86kHz
constexpr float kLowDx=0.0080f;    // 250² fallback
constexpr float kLowestDx=0.0100f; // 200² fallback
constexpr float kOfflineDx=0.0020f;// 1000x1778 ~= 1333² cells; 17.15kHz at 10 cells / wave
constexpr float kLivePassband=6000.0f; // <6.86kHz ten-cell numerical limit
constexpr float kBalancedPassband=6800.0f;
constexpr float kPreviousPassband=8500.0f;
constexpr float kMediumPassband=9800.0f;
constexpr float kUltraPassband=12000.0f;
constexpr float kSafePassband=5500.0f;
constexpr float kLowPassband=4000.0f;
constexpr float kLowestPassband=2800.0f;
constexpr float kOfflinePassband=16000.0f; // below 17.15kHz 10-cell limit
constexpr int kOutputRate=48000;
// Keep one short window on the CUDA device, not the complete MP3.
// The 2-second overlap means a 126-step graph batch can cross a
// source-window boundary while the entire MP3 stays in host RAM.
constexpr uint64_t kMp3WindowFrames=60ULL*kOutputRate;
constexpr uint64_t kMp3WindowStride=58ULL*kOutputRate;
inline uint64_t mp3WindowStart(double seconds,uint64_t fullFrames){
    if(fullFrames<2||!std::isfinite(seconds)||seconds<0.0)return 0;
    if(fullFrames<=kMp3WindowFrames)return 0;
    double pos=std::fmod(seconds*(double)kOutputRate,
                         (double)fullFrames);
    if(pos<0)pos+=(double)fullFrames;
    return ((uint64_t)pos/kMp3WindowStride)*kMp3WindowStride;
}

constexpr int kGraphBatch=126; // multiple of 3 for GPU field ring to repeat in a CUDA Graph
constexpr float kHeadReflection=0.74f;
constexpr float kOutsideReflection=0.82f;
constexpr float kPortMaterial=5.0f;
constexpr float kPortReflection=0.035f;
// Port extents are PHYSICAL metre values so they grow in relative
// screen size as the acoustic room is cropped. Shares parameters
// with the SDL wall drawing; all ports terminate sound.
constexpr float kLeftPortCentre=0.66f;  // fraction from bottom
constexpr float kLeftPortHeight=0.28f; // metres
constexpr float kRightPortCentre=0.245f;
constexpr float kRightPortHeight=0.32f;
constexpr float kFarPortCentre=0.285f; // fraction from left
constexpr float kFarPortWidth=0.22f; // metres
// This ASMR-specific impedance leaves the original arcade's wall
// physics untouched. All materials use the same 5-point stencil.
#ifdef __CUDACC__
__host__ __device__
#endif
inline float boundaryDamping(float left,float right,
                                     float below,float above){
    const float values[4]{left,right,below,above};
    int head=0,wall=0,port=0;
    for(int i=0;i<4;++i){
        const float m=values[i];
        if(m>=4.5f)++port;
        else if(m>=3.5f)++wall;
        else if(m>0)++head;
    }
    return echo::wallDamping(kHeadReflection,echo::kCourant,head)+
           echo::wallDamping(kOutsideReflection,echo::kCourant,wall)+
           echo::wallDamping(kPortReflection,echo::kCourant,port);
}

struct Vec2 {float x=0,y=0;};
struct PathPoint {double t=0; Vec2 centre;};
inline Vec2 add(Vec2 a,Vec2 b){return {a.x+b.x,a.y+b.y};}
inline Vec2 sub(Vec2 a,Vec2 b){return {a.x-b.x,a.y-b.y};}
inline Vec2 scale(Vec2 a,float t){return {a.x*t,a.y*t};}
inline float length(Vec2 a){return std::hypot(a.x,a.y);}
// 2D point-source pressure falls roughly as r^-0.5, whereas 3D
// free-field pressure falls as r^-1 (inverse-square INTENSITY).
// F12 changes ONLY captured stereo microphone pressure, not the
// wave propagation or phase. 1x adds the sqrt(r0/r) correction
// needed to approximate 3D amplitude; 20x exaggerates that
// additional attenuation 20-fold in its exponent. 0x is raw FDTD.
// Never add gain at close range or divide by an unbounded r.
constexpr std::array<int,11> kDistanceFalloffSteps{
    0,1,2,3,4,6,8,10,12,16,20};
// 10cm is INSIDE the minimum permitted mouth-to-ear approach.
// A 20cm dead zone made even 20x completely ineffective near a
// 0.5x listener. Scale this reference together with F11 geometry.
constexpr float kFalloffReferenceMetres=0.10f;
inline float distanceFalloffGain(float distanceMetres,int strength,
                                 float physicalScale=1.0f){
    if(strength<=0)return 1.0f;
    if(!std::isfinite(distanceMetres)||!std::isfinite(physicalScale)||
       physicalScale<=0.0f)return 0.0f;
    // All listener/mic/source geometry is scaled by F11. The
    // attenuation reference MUST scale as well: fixed 20cm made
    // F12 completely inactive inside the 0.1x/0.5x rooms.
    // Ratio now depends on physical source-to-ear distance
    // relative to the actual, uniformly scaled 10cm reference.
    const float reference=kFalloffReferenceMetres*physicalScale;
    const float distance=std::max(reference,
                                  std::max(0.0f,distanceMetres));
    const float ratio=reference/distance;
    return std::pow(ratio,0.5f*std::min(strength,20));
}

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
        p.y<=kRoomHeightMetres-kRoomClearance+1e-5f&&
        length(sub(p,listener()))>=kMinimumSeparation-1e-5f;
}
inline Vec2 restrictCentre(Vec2 p) {
    p.x=std::clamp(p.x,kRoomClearance,kRoomMetres-kRoomClearance);
    p.y=std::clamp(p.y,kRoomClearance,kRoomHeightMetres-kRoomClearance);
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
    if(points.empty())return {kListenerX+0.45f,kListenerY};
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
// Exact requested clip durations are measured in stereo FRAMES, never
// CUDA acoustic time steps. The interpolation resampler can emit 1-2
// extra frames when an acoustic batch overshoots the target time;
// truncate them, and pad if a partial history was selected.
inline size_t clipFrames(int seconds,int sampleRate=kOutputRate) {
    return seconds>0&&sampleRate>0?
        (size_t)seconds*(size_t)sampleRate:0;
}
inline void fitStereoClip(std::vector<float>& samples,int seconds,
                          int sampleRate=kOutputRate) {
    samples.resize(clipFrames(seconds,sampleRate)*2,0.0f);
}
// SDL's queued-byte count is the only timing source while replaying.
// Count across enqueued loop copies, then wrap the playback marker at
// exactly the clip's stereo PCM frame length.
inline double playbackSeconds(uint64_t queuedTotalFrames,
                              uint64_t pendingFrames,
                              size_t clipFrameCount,
                              int sampleRate=kOutputRate) {
    if(clipFrameCount==0||sampleRate<=0)return 0.0;
    const uint64_t played=queuedTotalFrames>=pendingFrames?
        queuedTotalFrames-pendingFrames:0;
    return (double)(played%clipFrameCount)/sampleRate;
}
// Five physical zooms, with deliberately smaller simulation volumes.
// Solver cell-work per simulated second scales as area/dx^3; prior
// same-cell-count zoom increased it *linearly* with zoom, causing
// audible real-time underruns. These presets target 85%,81%,76%,68.4%
// and 27.2% of the ORIGINAL full-size 5.00mm solver cell-work, while
// genuinely refining the acoustic grid near the listener.
// The final view is an actual 0.75m square; there is room to orbit
// the head since its center has 0.375m to all four walls.
// F11 scales the ENTIRE physical laboratory: room, listener/source
// silhouettes, microphones, mouth, openings, and movement.
// Smaller-than-life worlds need more CFL steps per real second.
// Enlarge normalized cells by approximately (1/(0.60*S))^(1/3),
// lowering the estimated small-room cell-updates/second by 40%
// compared with the previous compensated work budget.
// The substantial spatial resolution tradeoff is intentional.
// Navigation order: 1 -> 2 -> 3 -> 4 -> .1 -> .25 -> .5 -> .75 -> 1.
constexpr std::array<float,8> kPhysicsScales{
    1.00f,2.00f,3.00f,4.00f,0.10f,0.25f,0.50f,0.75f};
constexpr int kDefaultPhysicsScaleIndex=0;
constexpr float kSmallRoomWorkFraction=0.60f;
constexpr float kSmallRoomRasterGuard=1.0201f;
constexpr float kSquareMp3Gain=0.75f;
inline int nextPhysicalScaleIndex(int index,bool reverse=false){
    const int count=(int)kPhysicsScales.size();
    return (index+count+(reverse?-1:1))%count;
}
inline float physicalMetres(float designMetres,float scale){
    return designMetres*scale;
}
inline float physicalScaleDxFactor(float scale){
    // Cell updates per real second ∝ area/(S*normalized_dx^3).
    // Previous small-room work: 1/(1.01^3) of the 1x work.
    // New small-room target: kSmallRoomWorkFraction of that.
    // An additional ~1% spacing guard keeps the authored V2 pinna
    // microphone sample fully in air at 0.1x / Z=1.25, without
    // changing any head artwork or increasing the work budget.
    return scale<1.0f?
        kSmallRoomRasterGuard*
            std::cbrt(1.0f/(scale*kSmallRoomWorkFraction)):
        1.0f;
}
inline float physicalScaleDesignDx(float zoomDx,float scale){
    return zoomDx*physicalScaleDxFactor(scale);
}
inline double physicalStepSeconds(float designDx,float scale){
    return (double)(designDx*scale)/
            ((double)echo::kSpeedOfSound*(double)echo::kSqrtTwo);
}
constexpr std::array<float,5> kZoomSteps{1.00f,1.25f,1.55f,2.00f,2.6666667f};
constexpr std::array<float,5> kZoomWorkRatios{0.85f,0.81f,0.76f,0.684f,0.272f};
constexpr float kSquareRoomSide=0.75f;
// Explicit, testable navigation state: H, Z and Q may each change
// ONLY their own axis. Cycles are stable in both directions, with
// no hidden cursor that can get out of sync after a rejected change.
struct LabSelection {int head=0,quality=0,zoom=0;};
inline LabSelection nextSelection(LabSelection current,char key,
                                  bool reverse=false,int qualities=8){
    const int step=reverse?-1:1;
    auto cycle=[&](int index,int count){
        return (index+count+step)%count;
    };
    if(key=='H')current.head=cycle(current.head,3);
    else if(key=='Z')current.zoom=cycle(current.zoom,(int)kZoomSteps.size());
    else if(key=='Q'&&qualities>0)
        current.quality=cycle(current.quality,qualities);
    return current;
}
struct RoomView {
    float zoom=1, width=kRoomMetres,height=kRoomHeightMetres;
    float originX=0,originY=0;
};
inline RoomView roomView(float zoom=1.0f,bool square=false){
    if(square){
        return {zoom,kSquareRoomSide,kSquareRoomSide,
                kListenerX-kSquareRoomSide*0.5f,
                kListenerY-kSquareRoomSide*0.5f};
    }
    const float scale=std::clamp(zoom,1.0f,2.0f);
    const float width=kRoomMetres/scale;
    const float height=kRoomHeightMetres/scale;
    return {scale,width,height,kListenerX-width*0.5f,
            kListenerY-height/3.0f};
}
// Four widely separated sources approximate a diffuse external
// noise field for the optional N diagnostic. They are actual pressure
// emitters inside the FDTD grid, not sound added after the microphones.
// Deliberately well away from the head at every square/portrait Z.
inline Vec2 ambientSource(RoomView view,int index){
    constexpr float positions[4][2]{
        {0.14f,0.18f},{0.85f,0.22f},
        {0.18f,0.83f},{0.82f,0.79f}};
    index=std::clamp(index,0,3);
    return {view.originX+positions[index][0]*view.width,
            view.originY+positions[index][1]*view.height};
}
constexpr int kAmbientSourceCount=4;
constexpr int kAmbientNoiseSeconds=8;
constexpr float kAmbientInjection=0.14f; // relative to MP3 excitation
// Offline normalizer would otherwise raise a faint environmental noise
// probe all the way to full-scale headphone output. Keep its finished
// WAV comfortably below 0dBFS, AFTER the usual normalization step.
constexpr float kAmbientReplayGain=0.25f;
// Full-size baseline is the PREVIOUS 400x711 @5mm, not the more
// expensive Q quality variants. Keep Q's relative dx factor, while
// reducing its geometry workload by the same proportion at each Z.
inline float zoomDx(float baseDx,int stage){
    stage=std::clamp(stage,0,(int)kZoomSteps.size()-1);
    const RoomView view=roomView(
        kZoomSteps[(size_t)stage],stage==(int)kZoomSteps.size()-1);
    const float fraction=(view.width*view.height)/
                         (kRoomMetres*kRoomHeightMetres);
    return baseDx*std::cbrt(fraction/kZoomWorkRatios[(size_t)stage]);
}
inline float relativeCellWork(float dx,RoomView view,
                              float baselineDx=kLiveDx){
    // Relative acoustic cell-updates/sec, ignoring fixed launch overhead.
    return (view.width*view.height)/(kRoomMetres*kRoomHeightMetres)*
           std::pow(baselineDx/dx,3.0f);
}
inline float physicalScaleWork(float designDx,RoomView view,
                               float scale,
                               float baselineDx=kLiveDx){
    // Relative cell-update work per REAL second, not per graph.
    return relativeCellWork(designDx,view,baselineDx)/scale;
}
// Bound the TOTAL real-time 2D FDTD cost regardless of the
// combination of Z crop, F11 metres, and Q baseline resolution.
// Area alone is not the budget: acoustic steps/second also scale
// as 1/(physical scale * normalized dx). One cubic-root rule
// handles all room sizes, including the full 9:16 view.
// Ceiling relative to the original 2x3.56m @ 5mm full-room solver.
// A 0.5x square already ran at ~15% of that baseline on the user's
// hardware; limit larger rooms to at most 19% (approx. +25% work).
constexpr float kLiveCellWorkCeiling=0.19f;
inline float budgetedLiveDx(float qualityDx,int zoomStage,
                            float physicalScale){
    const float nominal=physicalScaleDesignDx(
        zoomDx(qualityDx,zoomStage),physicalScale);
    const RoomView v=roomView(kZoomSteps[(size_t)
        std::clamp(zoomStage,0,(int)kZoomSteps.size()-1)],
        zoomStage==(int)kZoomSteps.size()-1);
    const float work=physicalScaleWork(nominal,v,physicalScale);
    const float adjustment=std::cbrt(
        std::max(1.0f,work/kLiveCellWorkCeiling));
    return nominal*adjustment;
}

inline bool validArtistInView(Vec2 p,RoomView view){
    return validArtist(p) &&
        p.x>=view.originX+kRoomClearance-1e-5f &&
        p.x<=view.originX+view.width-kRoomClearance+1e-5f &&
        p.y>=view.originY+kRoomClearance-1e-5f &&
        p.y<=view.originY+view.height-kRoomClearance+1e-5f;
}
inline Vec2 restrictCentreInView(Vec2 p,RoomView view){
    const float minX=view.originX+kRoomClearance;
    const float maxX=view.originX+view.width-kRoomClearance;
    const float minY=view.originY+kRoomClearance;
    const float maxY=view.originY+view.height-kRoomClearance;
    p.x=std::clamp(p.x,minX,maxX);
    p.y=std::clamp(p.y,minY,maxY);
    if(validArtistInView(p,view))return p;
    // Search valid head positions on the close front/back arcs.
    const Vec2 c=listener();
    Vec2 best=p;float bestDist=1e20f;
    for(int i=0;i<360;++i){
        const float angle=2.0f*echo::kPi*i/360.0f;
        Vec2 q{c.x+(kMinimumSeparation+0.001f)*std::cos(angle),
               c.y+(kMinimumSeparation+0.001f)*std::sin(angle)};
        if(!validArtistInView(q,view))continue;
        const float d=length(sub(q,p));
        if(d<bestDist){bestDist=d;best=q;}
    }
    return bestDist<1e19f?best:p;
}
inline Vec2 moveArtistInView(Vec2 current,Vec2 target,RoomView view){
    // Restrict to zoomed physical walls while preserving continuous
    // collision sliding against the stationary head.
    current=restrictCentreInView(current,view);
    target=restrictCentreInView(target,view);
    Vec2 diff=sub(target,current);
    const int steps=std::max(1,(int)std::ceil(length(diff)/0.006f));
    const Vec2 d=scale(diff,1.0f/steps);
    for(int i=0;i<steps;++i){
        const Vec2 next=add(current,d);
        if(validArtistInView(next,view))current=next;
        else{
            const Vec2 slideX={next.x,current.y};
            const Vec2 slideY={current.x,next.y};
            if(validArtistInView(slideX,view))current=slideX;
            else if(validArtistInView(slideY,view))current=slideY;
        }
    }
    return current;
}
inline int cells(float dx){return (int)std::lround(kRoomMetres/dx);} // X cells
inline int cellsY(float dx){return (int)std::lround(kRoomHeightMetres/dx);}
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
inline std::vector<float> roomMask(int nx,int ny,float dx,
                                   float originX=0,float originY=0){
    std::vector<float> mask((size_t)nx*ny,0.0f);
    const int border=std::max(2,(int)std::ceil(0.012f/dx));
    for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){
        const int id=y*nx+x;
        if(x<border||y<border||x>=nx-border||y>=ny-border){
            // Three small sound-absorbing slots across the perimeter;
            // all remaining walls are substantially reflective.
            const float u=((float)x+0.5f)/nx;
            const float v=((float)y+0.5f)/ny;
            const float realH=ny*dx,realW=nx*dx;
            const bool port=(x<border &&
                    std::fabs(v-kLeftPortCentre)*realH<
                                0.5f*kLeftPortHeight)||
                (x>=nx-border &&
                    std::fabs(v-kRightPortCentre)*realH<
                                0.5f*kRightPortHeight)||
                (y>=ny-border &&
                    std::fabs(u-kFarPortCentre)*realW<
                                0.5f*kFarPortWidth);
            mask[id]=port?kPortMaterial:
                         echo::kOuterBoundaryMaterial;
            continue;
        }
        const float px=originX+((float)x+0.5f)*dx;
        const float py=originY+((float)y+0.5f)*dx;
        if(std::hypot(px-kListenerX,py-kListenerY)<=kHeadRadius)
            mask[id]=1.0f;
    }
    // Circle-only fallback: mics must be in a connected air notch
    // inside the bounds, never sampled from a zero-pressure solid.
    for(int sign:{-1,1}){
        // Include the two sample cells on the inward-facing side of
        // the 2x2 bilinear microphone footprint, not only the
        // exact mic centre.
        const float a=kListenerX+sign*(kEarOffset-2.0f*dx);
        const float b=kListenerX+sign*(kHeadRadius+.007f);
        const int x0=std::max(0,(int)std::floor((std::min(a,b)-originX)/dx-.5f));
        const int x1=std::min(nx-1,(int)std::ceil((std::max(a,b)-originX)/dx-.5f));
        const int y0=std::max(0,(int)std::floor(
            (kListenerY-std::max(.004f,1.5f*dx)-originY)/dx-.5f));
        const int y1=std::min(ny-1,(int)std::ceil(
            (kListenerY+std::max(.004f,1.5f*dx)-originY)/dx-.5f));
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x)
            if(mask[(size_t)y*nx+x]==1.0f)mask[(size_t)y*nx+x]=0.0f;
    }
    return mask;
}
// The 3-argument overloads support older square-only geometry checks.
inline std::vector<float> roomMask(int n,float dx){
    return roomMask(n,n,dx);
}
inline bool probeInAir(const std::vector<float>& mask,int nx,int ny,
                       Vec2 p,float dx,float originX=0,float originY=0) {
    const float gx=(p.x-originX)/dx,gy=(p.y-originY)/dx;
    const int ix=(int)std::floor(gx),iy=(int)std::floor(gy);
    if(ix<2||iy<2||ix+1>=nx-2||iy+1>=ny-2)return false;
    return mask[(size_t)iy*nx+ix]==0 &&
           mask[(size_t)iy*nx+ix+1]==0 &&
           mask[(size_t)(iy+1)*nx+ix]==0 &&
           mask[(size_t)(iy+1)*nx+ix+1]==0;
}
inline bool probeInAir(const std::vector<float>& mask,int n,
                       Vec2 p,float dx){return probeInAir(mask,n,n,p,dx);}
// The v2 open pinna recess is air around x=+-62mm at y=+3mm.
// These OUTER ear probes differ from the 8cm eardrum spacing, still
// available in the separately selectable ear-channel head (H).
constexpr float kPinnaMicSeparation=0.124f;
inline float micSpacing(bool earChannels=false,bool pinnaV2=false){
    return pinnaV2?kPinnaMicSeparation:
        earChannels?kNewChannelMicSeparation:kOuterMicSeparation;
}
inline float micOffsetY(bool earChannels=false,bool pinnaV2=false){
    return pinnaV2?kPinnaEarY:earChannels?kChannelEarY:0.0f;
}
inline Vec2 leftEar(bool earChannels=false,bool pinnaV2=false){
    return {kListenerX-0.5f*micSpacing(earChannels,pinnaV2),
            kListenerY+micOffsetY(earChannels,pinnaV2)};
}
inline Vec2 rightEar(bool earChannels=false,bool pinnaV2=false){
    return {kListenerX+0.5f*micSpacing(earChannels,pinnaV2),
            kListenerY+micOffsetY(earChannels,pinnaV2)};
}
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
