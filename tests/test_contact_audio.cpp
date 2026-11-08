#include "../contact_audio.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
static float energy(const std::vector<float>& audio,int channel){
    float total=0;
    for(size_t i=0;i<audio.size()/2;++i)
        total+=std::fabs(audio[i*2+channel]);
    return total;
}
int main(){
    const int N=120;
    std::vector<float> grid(N*N,0.0f);
    for(int y=2;y<N-2;++y)grid[y*N+63]=1.0f;
    // Player facing east: left is grid +Y and right is -Y.
    grid[63*N+50]=1.0f;
    grid[37*N+50]=1.0f;
    const auto left=echo::findWallContact(grid.data(),N,50,50,0,
                                           0,1,13.0f);
    const auto right=echo::findWallContact(grid.data(),N,50,50,0,
                                            0,-1,13.0f);
    assert(left.valid && right.valid);
    assert(left.relative>1.4f && right.relative<-1.4f);
    const auto leftEars=echo::contactEars(left.relative,left.distanceCm,48000);
    const auto rightEars=echo::contactEars(right.relative,right.distanceCm,48000);
    assert(leftEars.leftGain>leftEars.rightGain*1.8f);
    assert(leftEars.leftDelaySamples==0 && leftEars.rightDelaySamples>0);
    assert(rightEars.rightGain>rightEars.leftGain*1.8f);
    assert(rightEars.rightDelaySamples==0 && rightEars.leftDelaySamples>0);
    const auto leftAudio=echo::renderContactTap(leftEars,48000);
    const auto rightAudio=echo::renderContactTap(rightEars,48000);
    assert(!leftAudio.empty() && !rightAudio.empty());
    assert(energy(leftAudio,0)>energy(leftAudio,1)*1.7f);
    assert(energy(rightAudio,1)>energy(rightAudio,0)*1.7f);
    // Centered front and rear have identical interaural levels, but the
    // different spectral envelope makes them distinguishable.
    const auto front=echo::contactEars(0,13,48000);
    const auto back=echo::contactEars(3.14159265358979f,13,48000);
    assert(!front.behind && back.behind);
    // The cue's ear coordinates now agree with FDTD sampling:
    // -6.5cm behind pivot, 4.5cm lateral (9cm ear spacing).
    const float expectedFront=std::hypot(13.0f+6.5f,4.5f);
    const float expectedRear=std::hypot(-13.0f+6.5f,4.5f);
    assert(std::fabs(front.leftDistanceCm-expectedFront)<0.001f);
    assert(std::fabs(back.leftDistanceCm-expectedRear)<0.001f);
    assert(front.leftDistanceCm>back.leftDistanceCm*2.0f);
    assert(std::fabs(front.leftGain-front.rightGain)<1e-5f);
    assert(std::fabs(back.leftGain-back.rightGain)<1e-5f);
    const auto frontAudio=echo::renderContactTap(front,48000);
    const auto rearAudio=echo::renderContactTap(back,48000);
    float difference=0;
    for(size_t i=0;i<frontAudio.size();++i)
        difference+=std::fabs(frontAudio[i]-rearAudio[i]);
    assert(difference>1);
    // Proximity and collision onset are not enough to constantly beep
    // while holding the forward key against one wall.
    echo::ContactTrigger gate;
    assert(!gate.shouldPlay(false,1,0,1.0));
    assert(gate.shouldPlay(true,1,0,1.1));
    assert(!gate.shouldPlay(true,1,0,1.9));
    assert(gate.shouldPlay(true,1,1.5f,2.0));
    assert(!gate.shouldPlay(false,1,1.5f,3.0));
    assert(gate.shouldPlay(true,1,1.5f,3.2));
    std::puts("wall contact tests: true contact-point azimuth, ILD, ITD, rear cue and no repeated beeps");
}
