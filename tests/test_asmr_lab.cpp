#include "../asmr_lab_math.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main(){
    static_assert(asmr::kGraphBatch%3==0,
                  "CUDA graph must preserve triple-buffer identity");
    assert(asmr::cells(asmr::kLiveDx)==400);
    assert(asmr::cells(asmr::kMediumDx)==250);
    assert(asmr::cells(asmr::kSafeDx)==200);
    assert(asmr::cells(asmr::kOfflineDx)==1000);
    assert(asmr::tenCellFrequency(asmr::kLiveDx)>6800.0f);
    assert(asmr::tenCellFrequency(asmr::kOfflineDx)>17000.0f);
    assert(asmr::kOfflinePassband>=8000.0f);
    assert(asmr::kOfflinePassband<asmr::tenCellFrequency(asmr::kOfflineDx));

    for(float dx:{asmr::kLiveDx,asmr::kMediumDx,
                  asmr::kSafeDx,asmr::kOfflineDx}){
        const int n=asmr::cells(dx);
        const auto mask=asmr::roomMask(n,dx);
        assert(mask.size()==(size_t)n*n);
        assert(mask[0]==echo::kOuterBoundaryMaterial);
        assert(mask[n/2*n+n/2]==1.0f);
        assert(mask[(n/2-1)*n+(n/2-1)]==1.0f);
        assert(asmr::probeInAir(mask,n,asmr::leftEar(),dx));
        assert(asmr::probeInAir(mask,n,asmr::rightEar(),dx));
        const auto mouth=asmr::mouth({1.42f,1.0f});
        assert(asmr::probeInAir(mask,n,mouth,dx));
        assert(asmr::probeInAir(mask,n,{0.75f,1.0f},dx));
    }

    // Body-circle constraints apply even when the cursor jumps from one
    // end of the window to the other: no tunnelling through the listener.
    asmr::Vec2 p{1.50f,1.0f};
    assert(asmr::validArtist(p));
    p=asmr::moveArtist(p,{0.40f,1.0f});
    assert(asmr::validArtist(p));
    assert(p.x>asmr::kListenerX);
    for(int a=0;a<=360;++a){
        const float radians=2*echo::kPi*a/360.0f;
        const asmr::Vec2 desired{
            1.0f+0.10f*std::cos(radians),
            1.0f+0.10f*std::sin(radians)};
        const auto restricted=asmr::restrictCentre(desired);
        assert(asmr::validArtist(restricted));
        const auto mouth=asmr::mouth(restricted);
        assert(asmr::length(asmr::sub(mouth,asmr::listener()))>
               asmr::kHeadRadius+0.01f);
        assert(asmr::length(asmr::sub(mouth,restricted))<
               asmr::kArtistRadius);
    }
    // One physical source timeline must advance monotonically across
    // dozens of CUDA graph batches; the old code repeatedly swept the
    // mouth from old mouse position to new within each mini-batch.
    // This produces artificial grain when the mouse is moving.
    const asmr::Vec2 target{1.40f,1.50f};
    asmr::Vec2 smoothed{1.52f,1.0f};
    float previousDistance=asmr::length(asmr::sub(smoothed,target));
    float previousStep=0.0f;
    const float dt=(float)asmr::stepSeconds(asmr::kLiveDx);
    for(int batch=0;batch<25;++batch)
        for(int step=0;step<asmr::kGraphBatch;++step) {
            const auto updated=asmr::advanceSource(
                smoothed,target,dt,0.012f,2.25f);
            const float displacement=asmr::length(
                asmr::sub(updated,smoothed));
            assert(displacement<=2.25f*dt+1e-6f);
            const float distance=asmr::length(
                asmr::sub(updated,target));
            assert(distance<=previousDistance+1e-6f);
            previousDistance=distance;
            previousStep=displacement;
            smoothed=updated;
        }
    assert(previousDistance<0.45f);
    assert(previousStep>=0.0f);
    assert(asmr::length(asmr::sub(
        asmr::advanceSource(smoothed,target,0.0f),smoothed))<1e-7f);
    const auto roomEdge=asmr::moveArtist({1.60f,1.3f},{5,-2});
    assert(asmr::validArtist(roomEdge));

    std::vector<asmr::PathPoint> points{
        {0.0,{1.5f,1.0f}},
        {2.0,{1.5f,1.5f}},
        {4.0,{1.0f,1.5f}},
        {9.0,{0.5f,1.5f}},
        {15.0,{1.5f,1.0f}},
        {22.0,{1.45f,0.8f}}
    };
    const auto five=asmr::selectLast(points,22.0,5.0);
    const auto ten=asmr::selectLast(points,22.0,10.0);
    const auto twenty=asmr::selectLast(points,22.0,20.0);
    assert(std::fabs(five.start-17.0)<1e-9);
    assert(std::fabs(five.duration-5.0)<1e-9);
    assert(std::fabs(ten.start-12.0)<1e-9);
    assert(std::fabs(twenty.start-2.0)<1e-9);
    assert(five.points.front().t==five.start);
    assert(twenty.points.back().t==22.0);
    assert(std::fabs(asmr::interpolate(points,3.0).x-1.25f)<1e-6f);
    assert(std::fabs(asmr::interpolate(points,3.0).y-1.5f)<1e-6f);
    const auto partial=asmr::selectLast(points,3.0,10.0);
    assert(partial.duration==3.0);
    assert(partial.points.front().t==0.0);
    assert(asmr::selectLast({},5.0,5.0).points.empty());

    // Both filters must retain audible speech and reject unusably
    // high frequencies in the discovery mode, while offline can
    // reproduce >8kHz content. Avoid dependence on a file upload.
    const int rate=asmr::kOutputRate,N=4800;
    std::vector<float> low(N),high(N);
    for(int i=0;i<N;++i){
        const float t=(float)i/rate;
        low[i]=std::sin(2*echo::kPi*1000.0f*t);
        high[i]=std::sin(2*echo::kPi*10000.0f*t);
    }
    const auto liveLow=asmr::lowpass(low,asmr::kLivePassband,rate);
    const auto liveHigh=asmr::lowpass(high,asmr::kLivePassband,rate);
    const auto offlineHigh=asmr::lowpass(high,asmr::kOfflinePassband,rate);
    auto rms=[](const std::vector<float>& data){
        double sum=0;
        for(size_t i=300;i+300<data.size();++i)
            sum+=(double)data[i]*data[i];
        return std::sqrt(sum/(data.size()-600));
    };
    assert(rms(liveLow)>0.5);
    assert(rms(liveHigh)<0.03);
    assert(rms(offlineHigh)>0.45);
    std::puts("ASMR lab: true 5/8/10mm live and 2mm offline FDTD geometry, contiguous source timeline, 5/10/20s paths, 5.5/15kHz filters passed");
}
