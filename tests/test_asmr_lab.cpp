#include "../asmr_lab_math.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <cstdio>
#include <vector>

int main(){
    static_assert(asmr::kGraphBatch%3==0,
                  "CUDA graph must preserve triple-buffer identity");
    // F12 alters ONLY stereo sample amplitude, not CUDA wave updates.
    // Raw 2D spreading is roughly pressure~1/sqrt(r); adding the
    // 1x correction sqrt(r_ref/r) yields 3D-like pressure~1/r,
    // hence inverse-square INTENSITY. 0x must be byte-identical.
    static_assert(asmr::kDistanceFalloffSteps.size()==11,
                  "F12 must expose the full 0x-20x cycle");
    assert(asmr::kDistanceFalloffSteps.front()==0);
    assert(asmr::kDistanceFalloffSteps.back()==20);
    const float near=asmr::kFalloffReferenceMetres;
    assert(std::fabs(asmr::distanceFalloffGain(0.8f,0)-1.0f)<1e-6f);
    assert(std::fabs(asmr::distanceFalloffGain(near,20)-1.0f)<1e-6f);
    assert(std::fabs(asmr::distanceFalloffGain(0.8f,1)-
                     std::sqrt(near/0.8f))<1e-6f);
    assert(std::fabs(asmr::distanceFalloffGain(0.8f,2)-
                     near/0.8f)<1e-6f);
    assert(std::fabs(asmr::distanceFalloffGain(near*0.5f,20)-1.0f)<1e-6f);
    assert(asmr::distanceFalloffGain(
        std::numeric_limits<float>::quiet_NaN(),20)==0.0f);
    assert(std::fabs(std::sqrt(near/0.8f)*
        asmr::distanceFalloffGain(0.8f,1)-near/0.8f)<1e-6f);
    float lastGain=1.0f;
    for(const int strength:asmr::kDistanceFalloffSteps){
        const float gain=asmr::distanceFalloffGain(0.8f,strength);
        assert(std::isfinite(gain)&&gain>=0.0f&&gain<=lastGain);
        lastGain=gain;
    }
    for(size_t i=1;i<asmr::kDistanceFalloffSteps.size();++i)
        assert(asmr::kDistanceFalloffSteps[i]>
               asmr::kDistanceFalloffSteps[i-1]);
    // At any selected strength, greater range must never get louder.
    for(int strength:asmr::kDistanceFalloffSteps)
        assert(asmr::distanceFalloffGain(2.0f,strength)<=
               asmr::distanceFalloffGain(0.5f,strength));
    // Explicit F11 navigation: start at 1x, grow 2/3/4x, then
    // shrink to 0.1/0.25/0.5/0.75x, wrapping back to 1x.
    constexpr std::array<float,8> expectedScales{
        1.0f,2.0f,3.0f,4.0f,0.1f,0.25f,0.5f,0.75f};
    static_assert(asmr::kPhysicsScales.size()==8);
    assert(asmr::kDefaultPhysicsScaleIndex==0);
    assert(asmr::kPhysicsScales==expectedScales);
    assert(asmr::kPhysicsScales[(size_t)
        asmr::nextPhysicalScaleIndex(
            asmr::kDefaultPhysicsScaleIndex,true)]==0.75f);
    assert(asmr::kPhysicsScales[(size_t)
        asmr::nextPhysicalScaleIndex(3)]==0.1f);
    assert(asmr::kSmallRoomWorkFraction==0.60f);
    for(int i=0;i<8;++i){
        assert(asmr::nextPhysicalScaleIndex(i)==(i+1)%8);
        assert(asmr::nextPhysicalScaleIndex(i,true)==(i+7)%8);
        assert(asmr::nextPhysicalScaleIndex(
            asmr::nextPhysicalScaleIndex(i),true)==i);
    }
    assert(asmr::kSquareMp3Gain==0.75f);
    assert(0.55f*asmr::kSquareMp3Gain<0.55f);
    // Large worlds preserve the normalized grid count; small worlds
    // deliberately reduce normalized Nx,Ny. Their physical dx is
    // enlarged by the cube-root compensation so that the reduced
    // area balances more CFL steps/sec. All metric geometry scales,
    // independently of the numerical mesh chosen.
    for(size_t z=0;z<asmr::kZoomSteps.size();++z){
        const bool square=z+1==asmr::kZoomSteps.size();
        const auto view=asmr::roomView(asmr::kZoomSteps[z],square);
        const float baseDx=asmr::zoomDx(asmr::kLiveDx,(int)z);
        const int baseNx=(int)std::lround(view.width/baseDx);
        const int baseNy=(int)std::lround(view.height/baseDx);
        const float baseWork=asmr::physicalScaleWork(baseDx,view,1.0f);
        for(float physical:asmr::kPhysicsScales){
            const float factor=asmr::physicalScaleDxFactor(physical);
            const float designDx=asmr::physicalScaleDesignDx(
                baseDx,physical);
            const int nx=(int)std::lround(view.width/designDx);
            const int ny=(int)std::lround(view.height/designDx);
            const float work=asmr::physicalScaleWork(
                designDx,view,physical);
            assert(std::isfinite(work));
            assert(nx>=40&&ny>=40&&nx<=baseNx&&ny<=baseNy);
            const float expectedRatio=physical<1?
                asmr::kSmallRoomWorkFraction/
                std::pow(asmr::kSmallRoomRasterGuard,3.0f):
                1.0f/physical;
            assert(std::fabs(work-baseWork*expectedRatio)
                   <baseWork*0.001f);
            if(physical<1.0f){
                // Previous small-world preset used 1.01x raster
                // guard and full baseline work. Require >=40% cut.
                const float previousWork=baseWork/
                    (1.01f*1.01f*1.01f);
                assert(work<=previousWork*0.6001f);
            }
            if(physical<1){
                assert(factor>1.0f&&nx<baseNx&&ny<baseNy);
            }else{
                assert(factor==1.0f&&nx==baseNx&&ny==baseNy);
            }
            const double dt=asmr::physicalStepSeconds(
                designDx,physical);
            assert(std::fabs(
                dt/asmr::physicalStepSeconds(baseDx,1.0f)-
                physical*factor)<0.0001);
            const float cutoff=asmr::tenCellFrequency(
                designDx*physical);
            assert(std::fabs(cutoff-
                asmr::tenCellFrequency(baseDx)/(physical*factor))<
                std::max(0.1f,cutoff*0.0001f));
            const float waveDistance=asmr::length(asmr::sub(
                asmr::mouth({asmr::kListenerX+0.50f,asmr::kListenerY}),
                asmr::leftEar(true,true)));
            assert(std::fabs(
                ((waveDistance*physical)/echo::kSpeedOfSound)/
                (waveDistance/echo::kSpeedOfSound)-physical)<1e-5f);
            assert(std::fabs(asmr::physicalMetres(
                view.width,physical)-view.width*physical)<1e-5f);
        }
    }
    assert(std::fabs(asmr::physicalMetres(
        asmr::kRoomHeightMetres,0.1f)-0.3555556f)<0.00001f);
    assert(std::fabs(asmr::physicalMetres(
        asmr::kRoomMetres,0.1f)-0.2f)<1e-6f);
    assert(std::fabs(asmr::physicalMetres(
        asmr::kRoomHeightMetres,4.0f)-14.22222f)<0.0001f);
    assert(std::fabs(asmr::physicalMetres(
        asmr::kRoomMetres,4.0f)-8.0f)<1e-6f);
    assert(std::fabs(asmr::physicalMetres(
        asmr::kRoomMetres,0.25f)-0.5f)<1e-6f);
    assert(std::fabs(asmr::physicalMetres(
        asmr::kRoomMetres,0.75f)-1.5f)<1e-6f);
    // The unified Z × F11 × Q budget MUST prevent a large portrait
    // world from silently doing 3-5x more work than the working square.
    // This is an analytic cell-update ceiling; fixed launch overhead
    // and real GPU runtime are separate measurements.
    assert(asmr::kLiveCellWorkCeiling<=0.20f);
    for(const float physical:asmr::kPhysicsScales)
        for(size_t z=0;z<asmr::kZoomSteps.size();++z){
            const bool square=z+1==asmr::kZoomSteps.size();
            const auto view=asmr::roomView(
                asmr::kZoomSteps[z],square);
            for(const float quality:{0.0025f,0.00375f,0.0050f,0.008f}){
                const float nominal=asmr::physicalScaleDesignDx(
                    asmr::zoomDx(quality,(int)z),physical);
                const float dx=asmr::budgetedLiveDx(
                    quality,(int)z,physical);
                const float work=asmr::physicalScaleWork(dx,view,physical);
                assert(dx>=nominal*0.99999f);
                assert(work<=asmr::kLiveCellWorkCeiling*1.0002f);
                assert(std::isfinite(work));
                if(asmr::physicalScaleWork(nominal,view,physical)>
                    asmr::kLiveCellWorkCeiling)
                    assert(std::fabs(work-
                        asmr::kLiveCellWorkCeiling)<0.0001f);
            }
        }
    // N diagnostic excites the SAME FDTD physics from four fixed,
    // distinct, air-borne positions; it is not noise mixed at the ears.
    // All four positions must stay well outside the listener and walls
    // for all five Z stages and all scaled physical worlds.
    static_assert(asmr::kAmbientSourceCount==4);
    assert(asmr::kAmbientNoiseSeconds>=5);
    assert(asmr::kAmbientInjection>0.0f);
    assert(asmr::kAmbientInjection<0.25f);
    assert(asmr::kAmbientReplayGain==0.25f);
    for(size_t z=0;z<asmr::kZoomSteps.size();++z){
        const auto view=asmr::roomView(
            asmr::kZoomSteps[z],
            z+1==asmr::kZoomSteps.size());
        std::array<asmr::Vec2,4> sources{};
        for(int i=0;i<asmr::kAmbientSourceCount;++i){
            sources[(size_t)i]=asmr::ambientSource(view,i);
            const auto p=sources[(size_t)i];
            assert(p.x>view.originX+0.1f*view.width);
            assert(p.x<view.originX+0.9f*view.width);
            assert(p.y>view.originY+0.1f*view.height);
            assert(p.y<view.originY+0.9f*view.height);
            assert(asmr::length(asmr::sub(p,asmr::listener()))>
                   asmr::kMinimumSeparation+0.02f);
        }
        for(int i=0;i<4;++i)
            for(int j=i+1;j<4;++j)
                assert(asmr::length(
                    asmr::sub(sources[(size_t)i],sources[(size_t)j]))>
                    0.15f);
    }
    // Ports are physically different from reflecting wall material.
    assert(asmr::kOutsideReflection>0.8f);
    assert(asmr::kPortReflection<0.04f);
    assert(asmr::boundaryDamping(asmr::kPortMaterial,0,0,0)>
           asmr::boundaryDamping(
               echo::kOuterBoundaryMaterial,0,0,0));
    // F12 bug regression: at F11=0.5x the near-reference
    // also scales to 10cm. A 20x gain must NOT be identically
    // 1 across the entire smaller room.
    for(const float scale:{0.1f,0.25f,0.5f,0.75f,1.0f,2.0f,4.0f}){
        const float near=asmr::kFalloffReferenceMetres*scale;
        assert(std::fabs(asmr::distanceFalloffGain(
            near,20,scale)-1.0f)<1e-6f);
        assert(asmr::distanceFalloffGain(near*2.0f,20,scale)<0.002f);
        assert(std::fabs(asmr::distanceFalloffGain(
            near*4.0f,1,scale)-0.5f)<1e-6f);
        assert(asmr::distanceFalloffGain(
            near*0.5f,20,scale)==1.0f);
    }
    assert(asmr::distanceFalloffGain(0.18f,20,0.5f)<0.0001f);
    assert(asmr::distanceFalloffGain(0.09f,20,0.5f)<0.01f);
    assert(asmr::distanceFalloffGain(0.18f,20,1.0f)<0.01f);
    // A source well above the physical reference must be audibly
    // attenuated at F12=20 even at minimum safe approach.
    assert(asmr::kFalloffReferenceMetres<asmr::kMinimumSeparation);
    const float closestV2Ear=asmr::kMinimumSeparation-
        asmr::kMouthOffset-0.5f*asmr::micSpacing(true,true);
    assert(closestV2Ear>asmr::kFalloffReferenceMetres);
    assert(asmr::distanceFalloffGain(
        closestV2Ear*0.5f,20,0.5f)<0.05f);
    // Whole-MP3 streaming: no 60-second looping. The device window
    // remains 60 seconds; its 58s stride includes an interpolation
    // overlap and only the *full track* end wraps song time.
    const uint64_t songFrames=210ULL*asmr::kOutputRate;
    assert(asmr::mp3WindowStart(0.0,songFrames)==0);
    assert(asmr::mp3WindowStart(59.0,songFrames)==
           asmr::kMp3WindowStride);
    assert(asmr::mp3WindowStart(117.0,songFrames)==
           2*asmr::kMp3WindowStride);
    assert(asmr::mp3WindowStart(175.0,songFrames)==
           3*asmr::kMp3WindowStride);
    assert(asmr::mp3WindowStart(209.99,songFrames)==
           3*asmr::kMp3WindowStride);
    assert(asmr::mp3WindowStart(210.0,songFrames)==0);
    assert(asmr::mp3WindowStart(65.0,
        43ULL*asmr::kOutputRate)==0);
    assert(asmr::cells(asmr::kLiveDx)==400);
    assert(asmr::cellsY(asmr::kLiveDx)==711);
    // Every zoom is a REAL smaller FDTD room and finer dx, yet its
    // acoustic work budget declines rather than exploding with zoom.
    for(size_t i=0;i<asmr::kZoomSteps.size();++i){
        const bool square=i+1==asmr::kZoomSteps.size();
        const float zoom=asmr::kZoomSteps[i];
        const auto view=asmr::roomView(zoom,square);
        const float dx=asmr::zoomDx(asmr::kLiveDx,(int)i);
        const int nx=(int)std::lround(view.width/dx);
        const int ny=(int)std::lround(view.height/dx);
        const float work=asmr::relativeCellWork(dx,view);
        assert(std::fabs(work-asmr::kZoomWorkRatios[i])<0.015f);
        assert(dx<=asmr::kLiveDx*1.06f);
        assert(nx>100&&ny>100);
        assert(std::fabs((asmr::kListenerX-view.originX)/view.width-
                         0.5f)<1e-5f);
        assert(std::fabs((asmr::kListenerY-view.originY)/view.height-
                         (square?0.5f:1.0f/3))<1e-5f);
        assert((std::fabs(view.width-view.height)<1e-5f)==square);
        const asmr::Vec2 nearFront{asmr::kListenerX,
            asmr::kListenerY+asmr::kMinimumSeparation+0.002f};
        assert(asmr::validArtistInView(nearFront,view));
        const auto clamped=asmr::restrictCentreInView(
            {asmr::kListenerX,asmr::kListenerY},view);
        assert(asmr::validArtistInView(clamped,view));
        if(square){
            assert(std::fabs(view.width-0.75f)<1e-5f);
            for(int a=0;a<360;++a){
                const float theta=2.0f*echo::kPi*a/360;
                const asmr::Vec2 orbit{
                    asmr::kListenerX+(asmr::kMinimumSeparation+0.01f)*
                                       std::cos(theta),
                    asmr::kListenerY+(asmr::kMinimumSeparation+0.01f)*
                                       std::sin(theta)};
                assert(asmr::validArtistInView(orbit,view));
            }
        }
    }
    // A UI choice is a triple, never three independently drifting
    // cursors. Pressing H/Z/Q must advance exactly one dimension and
    // wrap in both directions without skipping any intermediate item.
    for(int head=0;head<3;++head)
        for(int zoom=0;zoom<(int)asmr::kZoomSteps.size();++zoom)
            for(int quality=0;quality<8;++quality){
                const asmr::LabSelection current{head,quality,zoom};
                const auto nextH=asmr::nextSelection(current,'H');
                assert(nextH.head==(head+1)%3);
                assert(nextH.zoom==zoom&&nextH.quality==quality);
                const auto prevH=asmr::nextSelection(current,'H',true);
                assert(prevH.head==(head+2)%3);
                assert(prevH.zoom==zoom&&prevH.quality==quality);
                const auto nextZ=asmr::nextSelection(current,'Z');
                assert(nextZ.zoom==(zoom+1)%5);
                assert(nextZ.head==head&&nextZ.quality==quality);
                const auto prevZ=asmr::nextSelection(current,'Z',true);
                assert(prevZ.zoom==(zoom+4)%5);
                assert(prevZ.head==head&&prevZ.quality==quality);
                const auto nextQ=asmr::nextSelection(current,'Q');
                assert(nextQ.quality==(quality+1)%8);
                assert(nextQ.head==head&&nextQ.zoom==zoom);
                const auto prevQ=asmr::nextSelection(current,'Q',true);
                assert(prevQ.quality==(quality+7)%8);
                assert(prevQ.head==head&&prevQ.zoom==zoom);
            }
    assert(std::fabs(asmr::kZoomWorkRatios[3]-0.684f)<1e-5f);
    assert(std::fabs(asmr::kZoomWorkRatios[4]-0.272f)<1e-5f);
    assert(asmr::kZoomWorkRatios[4]/
           0.68f <0.401f); // exact 60% fewer square grid updates
    assert(asmr::kZoomWorkRatios.front()<0.86f);
    assert(asmr::kZoomWorkRatios.back()<=0.85f*0.80f+0.001f);
    assert(std::fabs(asmr::micSpacing(true)-0.039f)<1e-6f);
    assert(std::fabs(asmr::micOffsetY(true)-0.002f)<1e-6f);
    assert(std::fabs(asmr::micOffsetY(true,true)-0.003f)<1e-6f);
    // Reflecting walls lose less pressure than the three door slots.
    assert(asmr::kOutsideReflection>0.78f);
    assert(asmr::kPortReflection<0.05f);
    assert(asmr::boundaryDamping(asmr::kPortMaterial,0,0,0)>
           asmr::boundaryDamping(echo::kOuterBoundaryMaterial,0,0,0));
    // Aspect conversion preserves total live cells within 0.2%.
    assert(std::abs(asmr::cells(asmr::kLiveDx)*
                asmr::cellsY(asmr::kLiveDx)-533*533)<600);
    // Position is EXACTLY the lower third, leaving twice the
    // listener-to-back-wall distance in the forward (+Y) direction.
    assert(std::fabs(asmr::kListenerY-asmr::kRoomHeightMetres/3.0f)<1e-6f);
    assert(std::fabs(asmr::kListenerX-asmr::kRoomMetres/2.0f)<1e-6f);
    assert(std::fabs(asmr::kRoomHeightMetres/
                     asmr::kRoomMetres-16.0/9.0)<1e-6);
    // New 9:16 room with 5mm preview uses about 25% fewer cell updates
    // per simulated second than the previous 533x533 at 5mm.
    constexpr float formerDx=0.00375f;
    constexpr int formerCells=533;
    const double updateRatio=
        (double)asmr::cells(asmr::kLiveDx)*
        asmr::cellsY(asmr::kLiveDx)*formerDx/
        ((double)formerCells*formerCells*asmr::kLiveDx);
    assert(updateRatio>0.73 && updateRatio<0.78);
    assert(asmr::validArtist({1.0f,3.20f}));
    assert(!asmr::validArtist({1.0f,asmr::kRoomHeightMetres}));
    assert(asmr::cells(asmr::kMediumDx)==667);
    assert(asmr::cells(asmr::kUltraDx)==800);
    assert(asmr::cells(asmr::kSafeDx)==400);
    assert(asmr::cells(asmr::kLowDx)==250);
    assert(asmr::cells(asmr::kLowestDx)==200);
    assert(asmr::cells(asmr::kOfflineDx)==1000);
    assert(asmr::cellsY(asmr::kOfflineDx)==1778);
    // Offline also retains its OLD 1333x1333 total grid cells.
    assert(std::abs(asmr::cells(asmr::kOfflineDx)*
                    asmr::cellsY(asmr::kOfflineDx)-1333*1333)<2000);
    assert(asmr::tenCellFrequency(asmr::kLiveDx)>6800.0f);
    assert(asmr::kLivePassband<asmr::tenCellFrequency(asmr::kLiveDx));
    assert(asmr::tenCellFrequency(asmr::kOfflineDx)>17000.0f);
    assert(asmr::kOfflinePassband<24000.0f);
    assert(asmr::kOfflinePassband>=8000.0f);
    assert(asmr::kOfflinePassband<asmr::tenCellFrequency(asmr::kOfflineDx));

    for(float dx:{asmr::kLiveDx,asmr::kMediumDx,
                  asmr::kUltraDx,asmr::kSafeDx,
                  asmr::kLowDx,asmr::kLowestDx,
                  asmr::kOfflineDx}){
        const int nx=asmr::cells(dx),ny=asmr::cellsY(dx);
        const auto mask=asmr::roomMask(nx,ny,dx);
        assert(mask.size()==(size_t)nx*ny);
        assert(mask[0]==echo::kOuterBoundaryMaterial);
        const int hx=(int)(asmr::kListenerX/dx);
        const int hy=(int)(asmr::kListenerY/dx);
        assert(mask[(size_t)hy*nx+hx]==1.0f);
        assert(mask[(size_t)(hy-1)*nx+hx-1]==1.0f);
        assert(asmr::probeInAir(mask,nx,ny,asmr::leftEar(),dx));
        assert(asmr::probeInAir(mask,nx,ny,asmr::rightEar(),dx));
        const auto mouth=asmr::mouth({1.42f,asmr::kListenerY});
        assert(asmr::probeInAir(mask,nx,ny,mouth,dx));
        assert(asmr::probeInAir(mask,nx,ny,{0.75f,asmr::kListenerY},dx));
    }

    // Body-circle constraints apply even when the cursor jumps from one
    // end of the window to the other: no tunnelling through the listener.
    asmr::Vec2 p{1.50f,asmr::kListenerY};
    assert(asmr::validArtist(p));
    p=asmr::moveArtist(p,{0.40f,asmr::kListenerY});
    assert(asmr::validArtist(p));
    assert(p.x>asmr::kListenerX);
    for(int a=0;a<=360;++a){
        const float radians=2*echo::kPi*a/360.0f;
        const asmr::Vec2 desired{
            1.0f+0.10f*std::cos(radians),
            asmr::kListenerY+0.10f*std::sin(radians)};
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
    asmr::Vec2 smoothed{1.52f,asmr::kListenerY};
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
    // A 3150-step interval is ~24ms at 5mm; a max-velocity
    // source correctly covers only a few centimetres in that time.
    assert(previousDistance<
        asmr::length(asmr::sub(asmr::Vec2{1.52f,asmr::kListenerY},target))-0.03f);
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
    // Exact 5/10/20s replay irrespective of acoustic FDTD dt,
    // GPU graph overrun, or interpolator sample count.
    for(int seconds:{5,10,20}) {
        const size_t count=asmr::clipFrames(seconds);
        assert(count==(size_t)seconds*48000);
        std::vector<float> oversized(count*2+14,0.5f);
        asmr::fitStereoClip(oversized,seconds);
        assert(oversized.size()==count*2);
        std::vector<float> undersized(count,0.1f);
        asmr::fitStereoClip(undersized,seconds);
        assert(undersized.size()==count*2);
        assert(undersized.back()==0.0f);
        assert(std::fabs(asmr::playbackSeconds(count,0,count)-
                         0.0)<1e-9); // exact wrap
        assert(std::fabs(asmr::playbackSeconds(
            count+24000,0,count)-0.5)<1e-9);
        assert(std::fabs(asmr::playbackSeconds(
            count+24000,24000,count)-0.0)<1e-9);
    }
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

    // Verify that changing the *FDTD* timestep cannot change pitch or
    // recorded duration: acoustic 440Hz at 5mm -> 48kHz stereo.
    {
        const double step=asmr::stepSeconds(asmr::kLiveDx);
        echo::StereoLinearResampler resampler(step,asmr::kOutputRate);
        std::vector<float> frames;
        constexpr double sourceHz=440.0, duration=0.15;
        const int steps=(int)std::ceil(duration/step);
        for(int i=0;i<steps;++i){
            const float sound=(float)std::sin(2.0*echo::kPi*sourceHz*i*step);
            resampler.push(sound,sound,frames);
        }
        const size_t stereoFrames=frames.size()/2;
        assert(stereoFrames>asmr::kOutputRate*0.149);
        assert(stereoFrames<asmr::kOutputRate*0.151);
        double errorSum=0.0;
        for(size_t i=256;i+256<stereoFrames;++i){
            const double exact=std::sin(2.0*echo::kPi*sourceHz*i/
                                        asmr::kOutputRate);
            const double diff=frames[2*i]-exact;
            errorSum+=diff*diff;
            assert(std::fabs(frames[2*i]-frames[2*i+1])<1e-7);
        }
        assert(std::sqrt(errorSum/(stereoFrames-512))<0.005);
    }
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
    assert(rms(liveHigh)<0.18); // 10kHz above the 6kHz lowpass
    // A 14kHz component can be represented on the same-cell-budget
    // offline grid, but remains rejected by the economical live grid.
    std::vector<float> veryHigh(N);
    for(int i=0;i<N;++i)
        veryHigh[i]=std::sin(2*echo::kPi*14000.0f*i/rate);
    const auto preview14=asmr::lowpass(veryHigh,
        asmr::kLivePassband,rate);
    const auto offline14=asmr::lowpass(veryHigh,
        asmr::kOfflinePassband,rate);
    assert(rms(preview14)<0.08);
    assert(rms(offline14)>0.35);
    assert(rms(offlineHigh)>0.45);
    std::puts("ASMR lab: 5/4.5/3.75/3/2.5/6/8/10mm live, 2mm offline 2x3.56m FDTD, exact ~284k/1.78M grid budgets, correct 440Hz-to-48kHz clock, 6k/16k filters passed");
}
