#include "../virtual_cane.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const int N=120;
    std::vector<float> mask(N*N,0.0f);
    for(int i=0;i<N;++i){
        mask[i]=1;mask[(N-1)*N+i]=1;
        mask[i*N]=1;mask[i*N+N-1]=1;
    }
    for(int y=1;y<N-1;++y)mask[y*N+72]=1;
    auto right=echo::castCaneRay(mask.data(),N,35,60,0.0f,80);
    assert(right.hit && std::abs(right.distanceCm-37.0f)<1.0f);
    auto left=echo::castCaneRay(mask.data(),N,35,60,3.14159265358979323846f,40);
    assert(left.hit && left.distanceCm>30);
    auto north=echo::castCaneRay(mask.data(),N,35,60,1.57079632679f,15);
    assert(!north.hit && north.distanceCm>14);
    const auto sweep=echo::sweepCane(mask.data(),N,35,60,0.0f,80);
    assert(sweep[0].hit&&sweep[1].hit&&sweep[2].hit);
    assert(sweep[0].distanceCm>sweep[1].distanceCm);
    assert(sweep[2].distanceCm>sweep[1].distanceCm);
    assert(!echo::castCaneRay(mask.data(),N,35,60,0,-1).hit);
    std::puts("virtual cane tests passed: left/center/right, ray distance, wall stops, finite guards");
}
