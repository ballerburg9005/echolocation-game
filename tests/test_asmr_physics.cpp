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

// After a single impulse, a 2D room with partially absorbing walls and
// three low-reflection ports must lose wave energy rather than accumulate
// pressure without a continuing source. Include a rigid head obstruction.
static void testRoomRingdown() {
    constexpr int nx=84,ny=132,steps=1200;
    const float dt=(float)asmr::physicalStepSeconds(0.010f,1.0f);
    const float c2=echo::kCourantSquared;
    std::vector<float> mask((size_t)nx*ny,0.0f);
    for(int y=0;y<ny;++y)for(int x=0;x<nx;++x) {
        const int id=y*nx+x;
        if(x<2||x>=nx-2||y<2||y>=ny-2)
            mask[id]=echo::kOuterBoundaryMaterial;
        if((x<2&&std::abs(y-86)<7)||
           (x>=nx-2&&std::abs(y-33)<8)||
           (y>=ny-2&&std::abs(x-24)<6))
            mask[id]=asmr::kPortMaterial;
        if((x-nx/2)*(x-nx/2)+(y-ny/3)*(y-ny/3)<64)
            mask[id]=1.0f;
    }
    std::vector<float> previous((size_t)nx*ny),
                       now(previous),next(previous);
    for(int y=0;y<ny;++y)for(int x=0;x<nx;++x) {
        const int id=y*nx+x;
        const float r2=(float)((x-20)*(x-20)+(y-55)*(y-55));
        now[id]=mask[id]==0.0f?std::exp(-r2/(2.0f*3.4f*3.4f)):0.0f;
    }
    previous=now;
    const auto energy=[&]() {
        double kinetic=0.0,potential=0.0;
        for(int y=0;y<ny-1;++y)for(int x=0;x<nx-1;++x) {
            const int id=y*nx+x;
            if(mask[id]>0.0f)continue;
            const double velocity=now[id]-previous[id];
            kinetic+=velocity*velocity;
            const double ddx=now[id+1]-now[id];
            const double ddy=now[id+nx]-now[id];
            potential+=(ddx*ddx+ddy*ddy)*c2;
        }
        return kinetic+potential;
    };
    const double initialEnergy=energy();
    double midpointEnergy=0.0;
    assert(initialEnergy>0.5);
    for(int t=0;t<steps;++t){
        std::fill(next.begin(),next.end(),0.0f);
        for(int y=1;y<ny-1;++y)for(int x=1;x<nx-1;++x) {
            const int id=y*nx+x;
            if(mask[id]>0.0f)continue;
            const float p=now[id];
            const float mL=mask[id-1],mR=mask[id+1];
            const float mD=mask[id-nx],mU=mask[id+nx];
            const float l=mL>0.0f?p:now[id-1];
            const float r=mR>0.0f?p:now[id+1];
            const float d=mD>0.0f?p:now[id-nx];
            const float u=mU>0.0f?p:now[id+nx];
            const float damping=asmr::boundaryDamping(
                mL,mR,mD,mU)+asmr::bulkDamping(dt);
            next[id]=echo::impedanceWaveStep(
                p,previous[id],l,r,d,u,c2,damping);
            assert(std::isfinite(next[id]));
        }
        previous.swap(now);now.swap(next);
        if(t==599)midpointEnergy=energy();
    }
    const double finalEnergy=energy();
    assert(midpointEnergy<initialEnergy*0.50);
    assert(finalEnergy<midpointEnergy*0.50);
    std::printf("2D room impulse energy proxy: initial %.4g, "
                "600 steps %.4g, 1200 steps %.4g\n",
                initialEnergy,midpointEnergy,finalEnergy);
}

int main() {
    testRoomRingdown();
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
