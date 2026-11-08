#include "../proximity_audio.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
int main(){
    const int N=100;
    std::vector<float> walls(N*N,0.0f);
    for(int y=0;y<N;++y)walls[y*N+50]=1.0f;
    assert(!echo::hasClearSight(walls.data(),N,20,30,80,30));
    assert(echo::hasClearSight(walls.data(),N,20,30,35,30));
    assert(echo::proximityStrength(90,30)==0.0f);
    assert(echo::proximityStrength(31,30)==0.0f);
    assert(echo::proximityStrength(15,30)>0.24f);
    assert(echo::proximityStrength(2,30)>echo::proximityStrength(15,30));
    assert(echo::markerRadiusCm('O')<echo::markerRadiusCm('W'));
    echo::ProximityGate g;
    assert(!g.shouldChime(100,30,1));
    assert(g.shouldChime(20,30,1));
    assert(!g.shouldChime(19,30,2));
    assert(!g.shouldChime(19,30,5));
    assert(g.shouldChime(19,30,5.6));
    assert(!g.shouldChime(31,30,6));
    assert(!g.shouldChime(38,30,7));
    assert(g.shouldChime(20,30,8));
    const auto left=echo::localBearing(30,30,0,30,45,0.5f);
    const auto right=echo::localBearing(30,30,0,30,15,0.5f);
    assert(left.left>left.right*1.5f);
    assert(right.right>right.left*1.5f);
    const auto rear=echo::localBearing(30,30,0,15,30,0.5f);
    assert(rear.rear);
    const auto cue=echo::localChime(48000,'I',left.left,left.right);
    assert(cue.size()>10000 && cue.size()%2==0);
    std::puts("proximity cues: silent beyond short range, blocked behind walls, hysteresis, stereo side");
}
