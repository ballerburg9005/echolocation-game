#include "../direct_reference.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main(){
    constexpr int N=164;
    std::vector<float> empty(N*N,0);
    const auto pair=echo::phasedSource(N,empty.data(),82,82,0.0f);
    assert(pair.enabled);
    const auto ears=echo::shieldedReceivers(82,82,0);
    const float lX=ears.leftX,lY=ears.leftY;
    const float rX=ears.rightX,rY=ears.rightY;
    assert(lX<82 && rX<82 && lY>82 && rY<82);
    assert(std::hypot(lX-82,lY-82)<13);
    assert(std::hypot(rX-82,rY-82)<13);
    const int duration=(int)std::lround(
        echo::kClickDurationSeconds/echo::kStepSeconds);
    const auto direct=echo::freeFieldDirect(pair,lX,lY,rX,rY,
        duration,echo::kStepSeconds,echo::kCourantSquared,0.9995f,
        750.0f,0.01f,true);
    assert(direct.left.size()==158);
    assert(direct.right.size()==158);
    float total=0;
    for(size_t i=0;i<direct.left.size();++i)
        total+=direct.left[i]*direct.left[i];
    assert(total>1e-9f);
    constexpr uint64_t onset=1000;
    for(size_t i=0;i<direct.left.size();++i){
        const float residual=echo::subtractDirect(
            direct.left[i],direct.left,onset,onset+i);
        assert(std::fabs(residual)<1e-6f);
        // An echo/scattered component is LEFT after direct cancellation.
        const float scattered=0.2f*std::sin((float)i*0.1f);
        const float expected=echo::subtractDirect(
            direct.left[i]+scattered,direct.left,onset,onset+i);
        assert(std::fabs(expected-scattered)<1e-5f);
    }
    assert(echo::subtractDirect(1.0f,direct.left,onset,onset-1)==1.0f);
    assert(echo::subtractDirect(1.0f,direct.left,onset,onset+300)==1.0f);
    assert(echo::subtractDirect(1.0f,direct.left,UINT64_MAX,onset)==1.0f);
    const auto omni=echo::freeFieldDirect(pair,lX,lY,rX,rY,
        duration,echo::kStepSeconds,echo::kCourantSquared,0.9995f,
        750.0f,0.01f,false);
    assert(omni.left.size()==direct.left.size());
    double difference=0.0;
    for(size_t i=0;i<direct.left.size();++i)
        difference+=std::fabs(direct.left[i]-omni.left[i]);
    assert(difference>0.0001);
    // The earlier microphones were lateral to the phased array and saw
    // essentially no suppression (even a direct-field increase). The
    // player must keep its ears BEHIND both sources, not simply beside them.
    float phasedEnergy=0,monoEnergy=0;
    for(size_t i=0;i<direct.left.size();++i) {
        phasedEnergy+=direct.left[i]*direct.left[i]+
                      direct.right[i]*direct.right[i];
        monoEnergy+=omni.left[i]*omni.left[i]+
                    omni.right[i]*omni.right[i];
    }
    std::printf("actual-ear phased/mono direct pressure energy: %.4f\n",
                phasedEnergy/monoEnergy);
    assert(monoEnergy>0.00001f);
    assert(phasedEnergy<monoEnergy*0.30f);
    // Validate the real ear positions when turning as well as facing
    // along a grid axis. FDTD grid anisotropy / source quantization
    // must not turn a diagonal heading into an amplified direct click.
    for(float heading: {echo::kPi/8,echo::kPi/4,echo::kPi/2}) {
        const auto oriented=echo::phasedSource(N,empty.data(),82,82,heading);
        assert(oriented.enabled);
        const auto layout=echo::shieldedReceivers(82,82,heading);
        const auto aimed=echo::freeFieldDirect(oriented,
            layout.leftX,layout.leftY,layout.rightX,layout.rightY,
            duration,echo::kStepSeconds,echo::kCourantSquared,
            0.9995f,750.0f,0.01f,true);
        const auto plain=echo::freeFieldDirect(oriented,
            layout.leftX,layout.leftY,layout.rightX,layout.rightY,
            duration,echo::kStepSeconds,echo::kCourantSquared,
            0.9995f,750.0f,0.01f,false);
        float aE=0.0f,mE=0.0f;
        for(size_t i=0;i<aimed.left.size();++i) {
            aE+=aimed.left[i]*aimed.left[i]+
                aimed.right[i]*aimed.right[i];
            mE+=plain.left[i]*plain.left[i]+
                plain.right[i]*plain.right[i];
        }
        assert(mE>0.000001f && aE/mE<0.30f);
    }
    // The J-key reference must include EVERY source in the narrower beam,
    // otherwise subtraction leaves a fake six-emitter direct click.
    const auto tight=echo::shieldedReceivers(82,82,0.0f);
    const auto beam=echo::playerBeam(N,empty.data(),82,82,0.0f,true);
    const auto monoBeam=echo::playerBeam(N,empty.data(),82,82,0.0f,false);
    assert(beam.valid && beam.directional && monoBeam.valid);
    const auto focused=echo::freeFieldDirect(beam,
        tight.leftX,tight.leftY,tight.rightX,tight.rightY,
        duration,echo::kStepSeconds,echo::kCourantSquared,0.9995f,
        750.0f,0.01f,true);
    const auto single=echo::freeFieldDirect(monoBeam,
        tight.leftX,tight.leftY,tight.rightX,tight.rightY,
        duration,echo::kStepSeconds,echo::kCourantSquared,0.9995f,
        750.0f,0.01f,false);
    assert(focused.left.size()==158 && single.left.size()==158);
    double focusedEnergy=0.0,singleEnergy=0.0;
    for(size_t i=0;i<focused.left.size();++i) {
        const float residual=echo::subtractDirect(
            focused.left[i],focused.left,1000,1000+i);
        assert(std::fabs(residual)<1e-6f);
        focusedEnergy+=focused.left[i]*focused.left[i]+
                       focused.right[i]*focused.right[i];
        singleEnergy+=single.left[i]*single.left[i]+
                      single.right[i]*single.right[i];
    }
    std::printf("six-source array direct ear energy ratio %.5f\n",
                 focusedEnergy/singleEnergy);
    assert(singleEnergy>1e-6 && focusedEnergy<singleEnergy*0.20);
    std::puts("scattered-field monitoring: real 6-source FDTD direct reference preserves echoes and shields rear ears");
}
