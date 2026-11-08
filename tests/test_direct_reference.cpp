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
    const float lX=82,lY=90,rX=82,rY=74;
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
    std::puts("scattered-field monitoring: FDTD direct path subtracts without erasing residual echoes");
}
