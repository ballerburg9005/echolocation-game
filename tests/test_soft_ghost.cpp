#include "../soft_ghost.h"
#include "../echo_arcade.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static std::vector<float> rotatedCoverage(float rotation){
    constexpr int N=96;
    std::vector<float> staticWalls(N*N,0.0f),coverage(N*N,0.0f);
    for(int i=0;i<N;++i) {
        staticWalls[i]=staticWalls[(N-1)*N+i]=1.0f;
        staticWalls[i*N]=staticWalls[i*N+N-1]=1.0f;
    }
    echo::EchoArcade arcade(staticWalls.data(),N,24,24,77);
    assert(!arcade.enemies.empty());
    for(size_t i=1;i<arcade.enemies.size();++i)
        arcade.enemies[i].eaten=true;
    arcade.enemies[0].x=54.0f;
    arcade.enemies[0].y=50.0f;
    arcade.enemies[0].visualSpin=rotation;
    arcade.stampGhostCoverage(coverage.data());
    return coverage;
}

static std::vector<float> waveAfterPulse(
    const std::vector<float>& ghost,float materialStrength=1.0f) {
    constexpr int N=96,steps=180;
    std::vector<float> previous(N*N,0),now(N*N,0),next(N*N,0);
    const int duration=(int)std::lround(
        echo::kClickDurationSeconds/echo::kStepSeconds);
    for(int step=0;step<steps;++step) {
        for(int y=1;y<N-1;++y)
            for(int x=1;x<N-1;++x) {
                const int idx=y*N+x;
                const float c=now[idx],h=ghost[idx]*materialStrength;
                const float l=echo::softGhostNeighbour(
                    c,now[idx-1],h,ghost[idx-1]*materialStrength);
                const float r=echo::softGhostNeighbour(
                    c,now[idx+1],h,ghost[idx+1]*materialStrength);
                const float d=echo::softGhostNeighbour(
                    c,now[idx-N],h,ghost[idx-N]*materialStrength);
                const float u=echo::softGhostNeighbour(
                    c,now[idx+N],h,ghost[idx+N]*materialStrength);
                next[idx]=echo::waveStep(
                    c,previous[idx],l,r,d,u,
                    echo::kCourantSquared)*0.9995f;
            }
        next[50*N+29]+=75.0f*echo::clickSample(
            step,duration,echo::kStepSeconds);
        previous.swap(now);now.swap(next);
        for(int i=0;i<N*N;++i)
            assert(std::isfinite(now[i]) && std::fabs(now[i])<500.0f);
    }
    return now;
}

int main(){
    constexpr int N=96;
    const auto a=rotatedCoverage(0.0f);
    const auto b=rotatedCoverage(echo::kPi/6.0f);
    assert(a.size()==N*N && b.size()==N*N);
    float geometryChange=0.0f; int partial=0;float areaA=0,areaB=0;
    for(int i=0;i<N*N;++i) {
        assert(a[i]>=0&&a[i]<=1&&b[i]>=0&&b[i]<=1);
        if((a[i]>0&&a[i]<1)||(b[i]>0&&b[i]<1))++partial;
        geometryChange+=std::fabs(a[i]-b[i]);
        areaA+=a[i];areaB+=b[i];
    }
    assert(partial>4);
    assert(geometryChange>5.0f);
    assert(std::fabs(areaA-areaB)<areaA*0.08f);
    // Rotational invariance: a regular hexagon repeats every 60 degrees.
    const auto periodic=rotatedCoverage(echo::kPi/3.0f);
    for(int i=0;i<N*N;++i)
        assert(std::fabs(a[i]-periodic[i])<1e-5f);

    // Ghost edges fade between frames, rather than snapping a full
    // raster cell from pressure-carrying fluid to 0-pressure solid.
    float state=0;
    for(int i=0;i<60;++i) {
        const float next=echo::smoothGhostCoverage(
            state,1.0f,1.0f/60.0f);
        assert(next>=state && next-state<0.3f);
        state=next;
    }
    assert(state>0.99f);
    const float fading=echo::smoothGhostCoverage(
        state,0.0f,1.0f/60.0f);
    assert(fading>0.70f && fading<state);

    // Fully covered rotating ghost edges reflect rather than silently
    // transmitting all pressure, while keeping interior pressure alive.
    assert(echo::softGhostNeighbour(10.0f,20.0f,0,0)==20.0f);
    assert(std::fabs(
        echo::softGhostNeighbour(10.0f,20.0f,1,1)-10.2f)<1e-4f);
    assert(echo::softGhostNeighbour(10.0f,20.0f,0.5f,0.5f)>10.0f);

    // Orientation changes the actual propagated FDTD pressure, not only
    // a rendered sprite/outline.
    const auto pressureA=waveAfterPulse(a);
    const auto pressureB=waveAfterPulse(b);
    double difference=0.0;
    for(int i=0;i<N*N;++i)
        difference+=std::fabs(pressureA[i]-pressureB[i]);
    std::printf("rotating soft ghost FDTD pressure difference: %.6f\n",
                difference);
    assert(difference>0.1);
    // Tiny candies use the *same* time-varying FDTD ghost interface,
    // still with 8 faces and no hard p_next reset. Rotating an octagon
    // must alter the wave solution, not merely the camera mesh.
    std::vector<float> candy0(N*N,0.0f),candy1(N*N,0.0f);
    constexpr int cx=54,cy=50;
    float candyArea=0.0f,shiftedArea=0.0f,edgeChange=0.0f;
    for(int y=35;y<65;++y)for(int x=39;x<69;++x) {
        const int idx=y*N+x;
        const float dx=(x+0.5f)-cx,dy=(y+0.5f)-cy;
        candy0[idx]=echo::rotatingCandyCoverage(dx,dy,0.0f);
        candy1[idx]=echo::rotatingCandyCoverage(dx,dy,echo::kPi/8.0f);
        candyArea+=candy0[idx];shiftedArea+=candy1[idx];
        edgeChange+=std::fabs(candy0[idx]-candy1[idx]);
    }
    assert(echo::kCandyRadiusCm==5.5f);
    assert(candyArea>60.0f && candyArea<120.0f);
    assert(std::fabs(candyArea-shiftedArea)<candyArea*0.12f);
    assert(edgeChange>1.0f);
    assert(echo::rotatingCandyCoverage(7,0,0)==0.0f);
    assert(echo::rotatingCandyCoverage(0,0,0)==1.0f);
    // 8-fold rotational symmetry is preserved.
    for(int y=-5;y<=5;++y)for(int x=-5;x<=5;++x)
        assert(std::fabs(echo::rotatingCandyCoverage((float)x,(float)y,0)-
                echo::rotatingCandyCoverage((float)x,(float)y,echo::kPi/4))
                <1e-5f);
    // Candy uses the SAME reflecting FDTD impedance as rotating ghosts.
    // Only its geometry and distinct lower voice differ.
    const auto coinPressure0=waveAfterPulse(candy0,1.0f);
    const auto coinPressure1=waveAfterPulse(candy1,1.0f);
    float propagatedDifference=0.0f;
    for(size_t i=0;i<coinPressure0.size();++i)
        propagatedDifference+=std::fabs(coinPressure0[i]-coinPressure1[i]);
    assert(propagatedDifference>0.1f);
    // Material decoding reuses one float per cell and the existing
    // FDTD stencil. Smaller candy is gently reflective, not ghost-strong.
    assert(echo::effectiveSoftCoverage(-0.75f)==0.75f);
    assert(std::fabs(echo::effectiveSoftCoverage(-1.75f)-0.75f)<1e-5f);
    assert(echo::effectiveSoftCoverage(0)==0.0f);
    assert(echo::effectiveSoftCoverage(2.0f)==0.0f);
    assert(echo::effectiveSoftCoverage(-1.75f)==
           echo::effectiveSoftCoverage(-0.75f));
    std::printf("rotating candy FDTD wave difference %.4f, 8-gon area %.2f cells\\n",
                propagatedDifference,candyArea);
    std::puts("rotating hexagons: fractional geometric coverage, smooth impedance and distinct FDTD echoes passed");
}
