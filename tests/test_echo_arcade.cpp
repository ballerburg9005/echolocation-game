#include "../echo_arcade.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main(){
    const int N=120;
    std::vector<float> mask(N*N,0);
    for(int i=0;i<N;++i) {
        mask[i]=mask[(N-1)*N+i]=1;
        mask[i*N]=mask[i*N+N-1]=1;
    }
    // Wall with passable opening in the middle, for BFS pathfinding.
    for(int y=1;y<N-1;++y)if(y<49||y>71)mask[y*N+N/2]=1;
    echo::EchoArcade game(mask.data(),N,23,25,128);
    assert(!game.enemies.empty());
    assert(!game.shards.empty());
    assert(game.remaining==(int)game.shards.size());
    for(const auto& e:game.enemies)
        assert(!echo::circleGridContact(mask.data(),N,e.x,e.y,11).hit);
    for(const auto& s:game.shards)
        assert(!echo::circleGridContact(mask.data(),N,s.x,s.y,11).hit);
    std::vector<float> dynamic=mask;
    game.stampEnemies(dynamic.data());
    int reflectors=0;
    for(float value:dynamic)if(value==2.0f)++reflectors;
    assert(reflectors>0);
    game.hearClick(game.enemies[0].x,game.enemies[0].y);
    assert(game.enemies[0].alert>0.0f);
    // Items collect exactly once.
    auto s=game.shards[0];
    auto ev=game.update(0.016f,s.x,s.y);
    assert(ev.pickups==1 && game.score==100);
    auto ev2=game.update(0.016f,s.x,s.y);
    assert(ev2.pickups==0);
    // Exit can't be reached before required pickups.
    assert(!game.reachExit(game.exitX,game.exitY));
    for(const auto& shard:game.shards)game.update(0,shard.x,shard.y);
    assert(game.remaining==0);
    assert(game.reachExit(game.exitX,game.exitY));
    assert(game.won);
    game.reset();
    assert(!game.won && game.lives==3 && game.remaining>0);
    std::puts("echo arcade tests passed: deterministic reachable spawns, audio attraction, pickups, exit, reset and reflectors");
}
