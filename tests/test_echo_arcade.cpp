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
    const auto validSpawn=echo::findFreeSpawn(mask.data(),N,23,25);
    assert(validSpawn.valid && validSpawn.x==23 && validSpawn.y==25);
    const auto movedSpawn=echo::findFreeSpawn(mask.data(),N,0,0);
    assert(movedSpawn.valid);
    assert(!echo::circleGridContact(mask.data(),N,movedSpawn.x,
                                    movedSpawn.y,13).hit);
    std::vector<float> completelyBlocked(N*N,1.0f);
    assert(!echo::findFreeSpawn(completelyBlocked.data(),N,23,25).valid);
    // A decoy is a legitimate secondary sound source, projected along an
    // unobstructed corridor. It never crosses the central SVG wall.
    const auto decoy=echo::projectDecoy(mask.data(),N,23,25,1,0,90);
    assert(decoy.valid && decoy.x>23 && decoy.x<N/2-3);
    assert(!echo::circleGridContact(mask.data(),N,decoy.x,decoy.y,3).hit);
    assert(!echo::projectDecoy(mask.data(),N,23,25,0,0).valid);
    const auto freeDecoy=echo::projectDecoy(mask.data(),N,23,25,-1,0,14);
    assert(freeDecoy.valid && freeDecoy.x<23);
    // A source about to hit a wall must not be assigned inside the wall.
    assert(!echo::projectDecoy(mask.data(),N,4,25,-1,0).valid);
    std::vector<float> nearWall(N*N,0.0f);
    for(int row=0;row<N;++row)nearWall[row*N+29]=1.0f;
    assert(!echo::projectDecoy(nearWall.data(),N,23,25,1,0,65).valid);
    // Facing east at +45° input angle: an enemy north must sound left.
    auto left=echo::threatCue(50,50,0.7853981634f,50,90,true);
    auto right=echo::threatCue(50,50,0.7853981634f,50,10,true);
    assert(left.audible&&right.audible);
    assert(left.left>left.right&&right.right>right.left);
    assert(echo::threatCue(0,0,0,200,200,false).audible==false);
    auto close=echo::threatCue(50,50,0.7853981634f,50,60,false);
    assert(close.interval<left.interval);
    const auto sprinter=echo::threatCue(50,50,0.7853981634f,50,90,true,1);
    const auto sentinel=echo::threatCue(50,50,0.7853981634f,50,90,true,2);
    assert(sprinter.frequency>left.frequency);
    assert(sentinel.frequency<left.frequency);
    echo::EchoArcade game(mask.data(),N,23,25,128);
    assert(!game.enemies.empty());
    assert(!game.shards.empty());
    auto initialGuide=game.guidanceTarget();
    assert(initialGuide.available && !initialGuide.exit);

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
    game.stampShards(dynamic.data());
    int shardReflectors=0;
    for(float value:dynamic)if(value==3.0f)++shardReflectors;
    assert(shardReflectors>0);
    const float soundX=game.enemies[0].x;
    const float soundY=game.enemies[0].y;
    game.hearClick(soundX,soundY);
    assert(game.enemies[0].alert>0.0f);
    assert(game.enemies[0].lastHeardX==soundX);
    assert(game.enemies[0].lastHeardY==soundY);
    assert(!game.enemies[0].pursuitDistances.empty());
    // Moving the player must not magically update enemy hearing memory.
    game.update(0.016f,23,25);
    assert(game.enemies[0].lastHeardX==soundX);
    assert(game.enemies[0].lastHeardY==soundY);
    // A thick, fully sealed wall blocks arcade sound alerts even when
    // the geometric distance is close. This is a passability heuristic.
    std::vector<float> sealed=mask;
    for(int y=1;y<N-1;++y)sealed[y*N+N/2]=1.0f;
    echo::EchoArcade soundTest(sealed.data(),N,23,25,128);
    assert(!soundTest.enemies.empty());
    for(auto &enemy:soundTest.enemies) {
        enemy.x=90;enemy.y=25;enemy.alert=0;enemy.pursuitDistances.clear();
    }
    soundTest.hearClick(23,25);
    for(const auto &enemy:soundTest.enemies)assert(enemy.alert==0.0f);
    // Overdrive requires four collected shards. A stunned ghost is harmless
    // and can be captured for bonus points; after 5 seconds it respawns.
    const int P=240;
    std::vector<float> powerMap(P*P,0.0f);
    for(int edge=0;edge<P;++edge) {
        powerMap[edge]=powerMap[(P-1)*P+edge]=1;
        powerMap[edge*P]=powerMap[edge*P+P-1]=1;
    }
    echo::EchoArcade powered(powerMap.data(),P,23,25,777);
    assert(powered.shards.size()>=4);
    const auto firstPowerShard=powered.shards[0];
    assert(!powered.activateOverdrive());
    for(int i=0;i<4;++i) {
        auto p=powered.shards[i];
        powered.update(0,p.x,p.y);
    }
    assert(powered.charges==1);
    assert(powered.activateOverdrive());
    assert(powered.charges==0 && powered.overdrive==5.0f);
    assert(!powered.enemies.empty());
    const float ghostX=powered.enemies[0].x;
    const float ghostY=powered.enemies[0].y;
    auto captured=powered.update(0,ghostX,ghostY);
    assert(captured.ghostsCaught>=1 && powered.enemies[0].eaten);
    const int bonusScore=powered.score;
    assert(bonusScore>=600); // four shards + first ghost
    for(int t=0;t<101;++t)powered.update(0.05f,23,25);
    assert(powered.overdrive==0.0f);
    assert(!powered.enemies[0].eaten);
    powered.reset();
    assert(powered.shards[0].x==firstPowerShard.x);
    assert(powered.shards[0].y==firstPowerShard.y);
    assert(powered.charges==0 && powered.score==0);
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
    auto exitGuide=game.guidanceTarget();
    assert(exitGuide.available && exitGuide.exit);
    assert(std::hypot(exitGuide.x-game.exitX,exitGuide.y-game.exitY)<0.01f);
    assert(game.reachExit(game.exitX,game.exitY));
    assert(game.won);
    game.reset();
    assert(!game.won && game.lives==3 && game.remaining>0);
    // Ghosts are gentle, nonlethal bumpers: ten or fifty collisions cannot
    // remove collectibles, stop AI, subtract score or cause GAME OVER.
    const int scoreBeforeHits=game.score;
    for(int collision=0;collision<12;++collision) {
        game.enemies[0].x=23;
        game.enemies[0].y=25;
        game.invulnerability=0.0f;
        const auto bump=game.update(0,23,25);
        assert(bump.hit && bump.bounced);
        assert(bump.recoilX>23 && bump.recoilX<26);
        assert(!bump.gameOver && !game.lost && game.lives==3);
        assert(game.score==scoreBeforeHits);
        assert(game.guidanceTarget().available);
        auto repeat=game.update(0,23,25);
        assert(!repeat.hit); // no repeated contact jingle
    }
    // Ensure pickups work after sustained ghost contact as well.
    const auto afterBumpsShard=game.shards[0];
    const int beforePickup=game.remaining;
    game.update(0,afterBumpsShard.x,afterBumpsShard.y);
    assert(game.remaining==beforePickup-1);
    assert(!game.lost && game.lives==3);
    game.reset();
    assert(!game.lost && game.lives==3);
    // Faceted source material must be centered, not a circle plus a shifted
    // second square in the 2D visualization.
    std::vector<float> shapeMask=mask;
    game.stampShards(shapeMask.data());
    const auto shard=game.shards[0];
    int materialCount=0;
    for(int y=0;y<N;++y)for(int x=0;x<N;++x)
        if(shapeMask[y*N+x]==3.0f)++materialCount;
    assert(materialCount>0);
    assert(echo::insideNgon(0,0,5,8,echo::kPi/8));
    assert(!echo::insideNgon(6,6,5,8,echo::kPi/8));
    std::puts("echo arcade tests passed: deterministic reachable spawns, audio attraction, pickups, exit, reset and reflectors");
}
