#include "../asmr_lab_math.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// Reference CPU reproduction of the pressure update used in asmrPressureStep.
// A uniform, stationary air pressure is a valid zero-frequency solution:
// air attenuation must damp VELOCITY, not create DC oscillation.
static void testUniformPressure() {
    const float dt = (float)asmr::physicalStepSeconds(0.0087f, 1.0f);
    const float bulk = asmr::bulkDamping(dt);
    assert(bulk > 0.0f && bulk < 0.001f);
    float p=1.0f, previous=1.0f;
    for(int t=0;t<10000;++t) {
        const float next=echo::impedanceWaveStep(
            p,previous,p,p,p,p,echo::kCourantSquared,bulk);
        assert(std::isfinite(next));
        previous=p;p=next;
    }
    assert(std::fabs(p-1.0f)<1e-4f);
}

struct PulseResponse {
    float incoming=0.0f,reflected=0.0f;
};
// Normally incident wide Gaussian pressure pulse on a locally reactive
// wall. A 1D cross section uses exactly the same finite-difference
// pressure stencil and impedance calculation as the 2D CUDA solver.
// This measures the REAL numerical reflection, not merely a constant.
static PulseResponse testWallPulse(float material) {
    constexpr int n=1400, wall=850, probe=610, steps=1600;
    constexpr float width=22.0f, center=400.0f;
    const float dt=(float)asmr::physicalStepSeconds(0.0087f,1.0f);
    std::vector<float> now(n),previous(n),next(n);
    for(int i=0;i<n;++i) {
        const float x=((float)i-center)/width;
        const float xp=((float)i-center+echo::kCourant)/width;
        now[i]=std::exp(-x*x);
        previous[i]=std::exp(-xp*xp);
    }
    PulseResponse response;
    for(int t=0;t<steps;++t) {
        std::fill(next.begin(),next.end(),0.0f);
        for(int i=1;i<wall;++i) {
            const float centreP=now[i];
            const float right=i==wall-1?centreP:now[i+1];
            const float wallDamping=asmr::boundaryDamping(
                0.0f,i==wall-1?material:0.0f,0.0f,0.0f);
            next[i]=echo::impedanceWaveStep(
                centreP,previous[i],now[i-1],right,centreP,centreP,
                echo::kCourantSquared,
                wallDamping+asmr::bulkDamping(dt));
        }
        if(t>=100&&t<550)
            response.incoming=std::max(response.incoming,
                std::fabs(now[probe]));
        if(t>=650&&t<1500)
            response.reflected=std::max(response.reflected,
                std::fabs(now[probe]));
        previous.swap(now);now.swap(next);
    }
    assert(response.incoming>0.9f);
    assert(response.reflected<response.incoming);
    return response;
}

int main() {
    testUniformPressure();
    const PulseResponse wall=testWallPulse(echo::kOuterBoundaryMaterial);
    const PulseResponse port=testWallPulse(asmr::kPortMaterial);
    const float wallR=wall.reflected/wall.incoming;
    const float portR=port.reflected/port.incoming;
    // Grid dispersion and absorption mean the measured numbers
    // are approximate, not asserted to equal their nominal constants.
    assert(std::fabs(wallR-asmr::kOutsideReflection)<0.08f);
    assert(std::fabs(portR-asmr::kPortReflection)<0.03f);
    assert(portR<0.10f && wallR>portR*5.0f);
    assert(asmr::kUltraOfflineDx<asmr::kOfflineDx);
    assert(asmr::kUltraOfflinePassband<0.5f*asmr::kOutputRate);
    assert(asmr::kUltraOfflinePassband<
           asmr::tenCellFrequency(asmr::kUltraOfflineDx));
    const float ultraWork=std::pow(
        asmr::kOfflineDx/asmr::kUltraOfflineDx,3.0f);
    assert(ultraWork>2.0f && ultraWork<3.0f);
    std::printf(
        "ASMR air damping preserves DC; wall pulse R=%.3f, "
        "window R=%.3f; ultra offline work %.2fx: PASS\n",
        wallR,portR,ultraWork);
}
