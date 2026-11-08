#pragma once
// Non-FDTD wall-contact *game cue*. Determines the actual solid cell touched
// around the player's collision circle, then produces ear-specific arrival
// times and near-ear level differences. Unlike simple sine-wave panning, a
// front or rear contact changes spectral content. This is a headphone cue,
// NOT a calibrated human HRTF or measured physical sound rendering.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace echo {
constexpr float kContactPi=3.14159265358979323846f;
struct WallContact {
    bool valid=false;
    float x=0,y=0,relative=0;
    float distanceCm=0;
};
inline WallContact findWallContact(const float* walls,int side,
    float px,float py,float heading,
    float wallDirX,float wallDirY,float radiusCm=13.0f) {
    if(!walls||side<3||!std::isfinite(px)||!std::isfinite(py)||
       radiusCm<=0)return {};
    WallContact best;
    float score=std::numeric_limits<float>::infinity();
    const float directionLength=std::hypot(wallDirX,wallDirY);
    const int reach=(int)std::ceil(radiusCm+5.0f);
    const int x0=std::max(0,(int)std::floor(px)-reach);
    const int x1=std::min(side-1,(int)std::ceil(px)+reach);
    const int y0=std::max(0,(int)std::floor(py)-reach);
    const int y1=std::min(side-1,(int)std::ceil(py)+reach);
    for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x) {
        if(walls[y*side+x]<=0.0f)continue;
        const float dx=(float)x-px,dy=(float)y-py;
        const float d=std::hypot(dx,dy);
        if(d<1.0e-5f||d>radiusCm+4.0f)continue;
        float alignment=1.0f;
        if(directionLength>1.0e-4f)
            alignment=(dx*wallDirX+dy*wallDirY)/(d*directionLength);
        if(alignment < 0.15f)continue;
        const float candidate=std::fabs(d-radiusCm)+
                               2.0f*(1.0f-alignment);
        if(candidate<score) {
            score=candidate;
            best.valid=true;best.x=(float)x;best.y=(float)y;
            best.distanceCm=d;
            best.relative=std::atan2(dy,dx)-heading;
            best.relative=std::atan2(std::sin(best.relative),
                                     std::cos(best.relative));
        }
    }
    return best;
}
struct EarContact {
    float leftDistanceCm=0,rightDistanceCm=0;
    float leftGain=0,rightGain=0;
    int leftDelaySamples=0,rightDelaySamples=0;
    bool behind=false;
};
inline EarContact contactEars(float relative,float distanceCm,
                              int sampleRate,float earHalfSpacingCm=8.0f) {
    EarContact out;
    if(sampleRate<=0 || distanceCm<=0 || !std::isfinite(relative))return out;
    // Ear positions are +/- 8cm around the head, within the collision ring.
    const float side=std::sin(relative);
    const float leftSq=distanceCm*distanceCm+earHalfSpacingCm*earHalfSpacingCm-
                       2*distanceCm*earHalfSpacingCm*side;
    const float rightSq=distanceCm*distanceCm+earHalfSpacingCm*earHalfSpacingCm+
                        2*distanceCm*earHalfSpacingCm*side;
    out.leftDistanceCm=std::sqrt(std::max(0.0f,leftSq));
    out.rightDistanceCm=std::sqrt(std::max(0.0f,rightSq));
    const float nearDistance=std::min(out.leftDistanceCm,out.rightDistanceCm);
    // 5cm softening avoids an infinite gain at the virtual ear location.
    out.leftGain=0.16f*(nearDistance+5.0f)/
                 (out.leftDistanceCm+5.0f);
    out.rightGain=0.16f*(nearDistance+5.0f)/
                  (out.rightDistanceCm+5.0f);
    out.leftDelaySamples=(int)std::lround(
        ((out.leftDistanceCm-nearDistance)/34300.0f)*sampleRate);
    out.rightDelaySamples=(int)std::lround(
        ((out.rightDistanceCm-nearDistance)/34300.0f)*sampleRate);
    out.behind=std::cos(relative)<0.0f;
    return out;
}
// Fully deterministic, tightly faded tap. Rear is duller/longer so sounds
// exactly in front and behind don't collapse to the same phantom center.
inline std::vector<float> renderContactTap(const EarContact& ears,
                                            int sampleRate) {
    if(sampleRate<=0)return {};
    const int count=(int)std::ceil(sampleRate*0.065);
    std::vector<float> output(2*count,0.0f);
    const float pitch=ears.behind?535.0f:1100.0f;
    const float harmonic=ears.behind?0.08f:0.35f;
    for(int channel=0;channel<2;++channel) {
        const int delay=channel==0?ears.leftDelaySamples:ears.rightDelaySamples;
        const float gain=channel==0?ears.leftGain:ears.rightGain;
        for(int i=delay;i<count;++i) {
            const float t=(float)(i-delay)/sampleRate;
            const float onset=1.0f-std::exp(-t/0.0017f);
            const float decay=std::exp(-t/(ears.behind?0.019f:0.013f));
            const float wave=std::sin(2.0f*kContactPi*pitch*t)+
                             harmonic*std::sin(2.0f*kContactPi*2.1f*pitch*t);
            output[i*2+channel]=gain*onset*decay*wave;
        }
    }
    return output;
}
struct ContactTrigger {
    bool wasTouching=false;
    float lastAngle=0;
    double lastTime=-100.0;
    bool shouldPlay(bool collided,float movementCm,float angle,
                    double nowSeconds) {
        if(!collided){wasTouching=false;return false;}
        const float difference=std::atan2(std::sin(angle-lastAngle),
                                           std::cos(angle-lastAngle));
        const bool fresh=!wasTouching || std::fabs(difference)>0.90f;
        wasTouching=true;
        if(!fresh || movementCm<0.1f || nowSeconds-lastTime<0.16)
            return false;
        lastAngle=angle;lastTime=nowSeconds;
        return true;
    }
};
}
