#pragma once
// Self-contained, deterministic echo-arcade world. No SDL, CUDA or OpenGL.
// Coordinates are the same 1-cm grid used by the 2D FDTD simulation.
// Enemies have physical occupancy; the game layer decides when to stamp
// their disks onto the wave mask. AI awareness is a *game mechanic*,
// not a physical acoustic sensing calculation.
#include "collision.h"
#include "source_model.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>
#include <limits>

namespace echo {

// Choose a collision-safe start on arbitrary SVG levels. Search at 5-cm
// intervals, minimizing distance from the intended (legacy) start position.
// Unlike silently teleporting on every restart this runs only at level load.
struct SafeSpawn { float x=0,y=0; bool valid=false; };
inline SafeSpawn findFreeSpawn(const float* walls,int size,float desiredX,
                              float desiredY,float radius=13.0f) {
    if(!walls||size<3||!std::isfinite(desiredX)||!std::isfinite(desiredY))
        return {};
    if(!circleGridContact(walls,size,desiredX,desiredY,radius).hit &&
       desiredX>=radius+1 && desiredY>=radius+1 &&
       desiredX<(float)size-radius-1 &&
       desiredY<(float)size-radius-1)
        return {desiredX,desiredY,true};
    SafeSpawn result;
    float bestDistance=std::numeric_limits<float>::infinity();
    // Sample legal centres; if the radius does not fit even once, report
    // failure instead of placing the player inside a blocking wall.
    for(float y=radius+2.0f;y<size-radius-2.0f;y+=5.0f)
        for(float x=radius+2.0f;x<size-radius-2.0f;x+=5.0f) {
            const float dx=x-desiredX,dy=y-desiredY;
            const float d2=dx*dx+dy*dy;
            if(d2<bestDistance &&
               !circleGridContact(walls,size,x,y,radius).hit) {
                result={x,y,true};
                bestDistance=d2;
            }
        }
    return result;
}

// Decoy emitter placement: a sound source projected along the player's
// facing direction, blocked by actual SVG geometry (no teleport through walls).
// This is used for a *real* second FDTD source in the main CUDA kernel.
struct DecoyPlacement { float x=0,y=0; bool valid=false; };
inline DecoyPlacement projectDecoy(const float* walls,int size,float startX,
                                   float startY,float dirX,float dirY,
                                   float maxRangeCm=85.0f) {
    DecoyPlacement result;
    if(!walls||size<3||maxRangeCm<12.0f)return result;
    const float norm=std::hypot(dirX,dirY);
    if(norm<0.00001f)return result;
    dirX/=norm;dirY/=norm;
    // Sampling every 2cm with a 3cm clearance keeps the click emitter off
    // blocked cells and prevents jumping across a thin wall.
    for(float distance=0.0f;distance<=maxRangeCm;distance+=2.0f) {
        const float x=startX+dirX*distance;
        const float y=startY+dirY*distance;
        if(circleGridContact(walls,size,x,y,3.0f).hit)break;
        if(distance>=10.0f)result={x,y,true};
    }
    return result;
}

struct Enemy {
    float x=0, y=0;
    float spawnX=0, spawnY=0;
    float speed=17.0f; // cm/s; cannot outrun player forever
    float alert=0.0f;
    float wanderTimer=0.0f;
    unsigned direction=0;
    // Arcade archetypes: 0 tracker, 1 sprinter, 2 sentinel.
    unsigned archetype=0;
    bool eaten=false;
    float lastHeardX=0,lastHeardY=0;
    std::vector<int> pursuitDistances; // shortest paths to last audible click
};
// Assistive proximity heartbeat for enemies. This is explicitly game UI
// sonification, not part of the modeled wave field. Facing angle uses the
// same -45-degree offset as the original player's sound funnel.
struct ThreatCue {
    float left=0,right=0,frequency=0,interval=0;
    bool audible=false;
};
inline ThreatCue threatCue(float px,float py,float angle,
                           float ex,float ey,bool alerted,
                           unsigned archetype=0) {
    const float distance=std::hypot(ex-px,ey-py);
    // Optional UI heartbeat is deliberately local. Gameplay itself must
    // still allow listening to FDTD echoes from much farther away.
    if(distance>=55.0f)return {};
    const float closeness=1.0f-std::max(0.0f,distance)/55.0f;
    const float bearing=std::atan2(ey-py,ex-px)-(angle-0.7853981633974483f);
    const float pan=std::sin(bearing);
    const float amplitude=0.12f*closeness*closeness;
    ThreatCue t;
    t.left=amplitude*std::sqrt(0.5f*(1.0f+pan));
    t.right=amplitude*std::sqrt(0.5f*(1.0f-pan));
    // Distinct pitches: tracker middle, sprinter high, sentinel low.
    const float voice=archetype==1?105.0f:archetype==2?-65.0f:0.0f;
    t.frequency=235.0f+voice+120.0f*closeness+(alerted?65.0f:0.0f);
    t.interval=std::max(0.17f,(0.9f-0.65f*closeness)*(alerted?0.8f:1.0f));
    t.audible=true;
    return t;
}

struct Shard {
    float x=0, y=0;
    bool collected=false;
};
struct GuidanceTarget { float x=0,y=0; bool exit=false,available=false; };
struct ArcadeEvent {
    int pickups=0;
    int chargesGained=0;
    int ghostsCaught=0;
    bool hit=false;
    bool bounced=false;
    float recoilX=0,recoilY=0;
    bool gameOver=false;
    bool cleared=false;
    bool enemyAlert=false;
};

class EchoArcade {
public:
    explicit EchoArcade(const float* mask, int side,
                        float startX, float startY, uint32_t seed=0xEC401202)
        : grid_(mask), side_(side), homeX_(startX), homeY_(startY),
          rng_(seed ? seed : 0xEC401202u), seed_(seed ? seed : 0xEC401202u),
          navSide_((side + kNavStep-1)/kNavStep),
          walkable_(navSide_*navSide_,0), distances_(walkable_.size(),-1) {
        reset();
    }
    void reset() {
        rng_=seed_;
        score=0; lives=3; charges=0; shardsCollected=0; combo=0;
        overdrive=0.0f; won=false; lost=false; invulnerability=0.0f;
        playerX=homeX_; playerY=homeY_; enemies.clear(); shards.clear();
        for(int y=0;y<navSide_;++y)for(int x=0;x<navSide_;++x){
            walkable_[y*navSide_+x] = !circleGridContact(
                grid_,side_,coord(x),coord(y),kAgentRadius).hit;
        }
        buildDistances(homeX_, homeY_);
        struct Node {float d; int idx;};
        std::vector<Node> choices;
        for(int i=0;i<(int)distances_.size();++i) if(distances_[i]>=0 &&
            distances_[i]*kNavStep >= 65)
            choices.push_back({(float)distances_[i],i});
        std::sort(choices.begin(),choices.end(),[](const Node&a,const Node&b){
            if(a.d!=b.d) return a.d>b.d;
            return a.idx<b.idx;
        });
        // A generated exit always exists on open/reachable maps.
        if(!choices.empty()){
            const int i=choices[0].idx;
            exitX=coord(i%navSide_); exitY=coord(i/navSide_);
        }else {exitX=homeX_;exitY=homeY_;}
        // Spread the ghosts apart, choose reachable points far from start.
        for(const auto &choice: choices) {
            if(enemies.size()>=3) break;
            const float x=coord(choice.idx%navSide_), y=coord(choice.idx/navSide_);
            if (std::hypot(x-exitX,y-exitY)<25) continue;
            bool spaced=true;
            for(const Enemy&e:enemies)if(std::hypot(e.x-x,e.y-y)<75)spaced=false;
            if(spaced) {
                Enemy e; e.x=e.spawnX=x; e.y=e.spawnY=y;
                e.direction=(unsigned)(enemies.size()%4);
                e.archetype=(unsigned)enemies.size();
                e.speed=e.archetype==0?17.0f:e.archetype==1?24.0f:12.5f;
                enemies.push_back(e);
            }
        }
        // Deterministic scatter; require separation and sufficient player
        // distance so early play is navigable without sight.
        if (!choices.empty()) {
            for(int attempt=0;attempt<5000 && shards.size()<12;++attempt){
                int id=choices[nextRand()%choices.size()].idx;
                float x=coord(id%navSide_), y=coord(id/navSide_);
                if (std::hypot(x-homeX_,y-homeY_)<45 ||
                    std::hypot(x-exitX,y-exitY)<22) continue;
                bool spaced=true;
                for(const Shard&s:shards) if(std::hypot(x-s.x,y-s.y)<27)spaced=false;
                if(spaced) shards.push_back({x,y,false});
            }
        }
        remaining=(int)shards.size();
        navTimer_=0;
    }

    // Arcade sound detection follows reachable passages rather than
    // Euclidean range through thick maze walls. This is 5cm-grid geodesic
    // *gameplay*, not a measured sound pressure / diffraction prediction.
    void hearClick(float x,float y) {
        if(won||overdrive>0.0f) return;
        std::vector<int> soundField(walkable_.size(),-1);
        computeDistances(x,y,soundField);
        for(Enemy &enemy:enemies) {
            const int enemyNode=nodeAt(enemy.x,enemy.y);
            const int distanceCells=soundField[enemyNode];
            const float maxRange=enemy.archetype==1?125.0f:
                                 enemy.archetype==2?230.0f:180.0f;
            if(distanceCells<0 || distanceCells*kNavStep>maxRange)continue;
            enemy.alert=std::max(enemy.alert,
                enemy.archetype==1?3.2f:enemy.archetype==2?6.0f:4.5f);
            enemy.lastHeardX=x;
            enemy.lastHeardY=y;
            enemy.pursuitDistances=soundField;
        }
    }

    // Earn one charge for every four shards. An overdrive pulse is an
    // optional arcade power-up: enemy stun / capture are *game mechanics*,
    // while the caller also emits a normal physical FDTD click.
    bool activateOverdrive() {
        if(won||charges<=0||overdrive>0.0f)return false;
        --charges;
        overdrive=5.0f;
        combo=0;
        for(Enemy &e:enemies) {
            e.alert=0.0f;
            e.pursuitDistances.clear();
        }
        return true;
    }

    ArcadeEvent update(float dt,float x,float y) {
        ArcadeEvent result;
        if(won) return result;
        dt=std::max(0.0f,std::min(dt,0.05f));
        playerX=x;playerY=y;
        const float previousOverdrive=overdrive;
        overdrive=std::max(0.0f,overdrive-dt);
        if(previousOverdrive>0 && overdrive==0.0f) {
            for(Enemy &e:enemies) {
                if(e.eaten) {e.x=e.spawnX;e.y=e.spawnY;}
                e.eaten=false;
            }
        }
        invulnerability=std::max(0.0f,invulnerability-dt);
        // Refresh player-directed BFS at 2 Hz, not once per frame.
        navTimer_-=dt;
        if(navTimer_<=0) {
            buildDistances(x,y);
            navTimer_=0.5f;
        }
        for(Shard &s:shards) {
            if(!s.collected && std::hypot(s.x-x,s.y-y)<15.0f){
                s.collected=true; result.pickups++; remaining--; score+=100;
                ++shardsCollected;
                if(shardsCollected%4==0 && charges<3) {
                    ++charges; ++result.chargesGained;
                }
            }
        }
        for(Enemy &e:enemies) {
            if(e.eaten)continue;
            if(overdrive>0.0f) {
                // Catching a ghost earns a Pac-Man-style increasing combo,
                // instead of taking damage. It disappears until overdrive ends.
                if(std::hypot(e.x-x,e.y-y)<24.0f) {
                    e.eaten=true;
                    ++result.ghostsCaught;
                    score+=200*(1<<std::min(combo,3));
                    ++combo;
                }
                continue;
            }
            if(e.alert>0) {
                e.alert=std::max(0.0f,e.alert-dt);
                result.enemyAlert=true;
            }
            float tx=0,ty=0;
            if(e.alert>0) {
                const int here=nodeAt(e.x,e.y);
                int best=here;
                const auto &field=e.pursuitDistances.empty()?distances_:e.pursuitDistances;
                int bestDist=(here>=0 && here<(int)field.size())?field[here]:-1;
                const int xx=here%navSide_, yy=here/navSide_;
                const int neighbors[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
                for(const auto&d:neighbors){
                    int nx=xx+d[0],ny=yy+d[1];
                    if(nx<0||ny<0||nx>=navSide_||ny>=navSide_)continue;
                    int n=ny*navSide_+nx;
                    if(field[n]>=0 && (bestDist<0||field[n]<bestDist)){
                        best=n;bestDist=field[n];
                    }
                }
                if(best!=here) {
                    tx=coord(best%navSide_)-e.x;ty=coord(best/navSide_)-e.y;
                } else {tx=e.lastHeardX-e.x;ty=e.lastHeardY-e.y;}
            } else {
                e.wanderTimer-=dt;
                if(e.wanderTimer<=0) {
                    e.direction=nextRand()%4;
                    e.wanderTimer=1.0f+(nextRand()%150)/100.0f;
                }
                const float directions[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
                tx=directions[e.direction][0]; ty=directions[e.direction][1];
            }
            float norm=std::hypot(tx,ty);
            if(norm>0.00001f) {
                const float speed=e.speed*(e.alert>0?1.22f:0.65f);
                const auto next=slideCircle(grid_,side_,e.x,e.y,
                        tx/norm*speed*dt,ty/norm*speed*dt,kAgentRadius);
                e.x=next.x;e.y=next.y;
                if(next.collided && !next.slid && e.alert<=0)e.wanderTimer=0;
            }
            // A ghost is a gentle solid bumper, NOT an enemy that can
            // end the game. Never decrement lives, scores or disable pickups.
            // Resolve small overlaps continuously; only one short feedback
            // event is sent per cooldown, not a repeated death jingle.
            const float gap=std::hypot(e.x-playerX,e.y-playerY);
            if(gap<22.0f) {
                float awayX=playerX-e.x,awayY=playerY-e.y;
                float length=std::hypot(awayX,awayY);
                if(length<0.001f) {
                    awayX=playerX-homeX_;awayY=playerY-homeY_;
                    length=std::hypot(awayX,awayY);
                }
                if(length<0.001f){awayX=1.0f;awayY=0.0f;length=1.0f;}
                const float push=std::min(1.5f,22.0f-gap);
                const auto moved=slideCircle(grid_,side_,
                    playerX,playerY,awayX/length*push,
                    awayY/length*push,13.0f);
                if(std::hypot(moved.x-playerX,moved.y-playerY)>0.01f){
                    playerX=moved.x;playerY=moved.y;
                    result.bounced=true;
                    result.recoilX=playerX;
                    result.recoilY=playerY;
                }
                if(invulnerability<=0.0f) {
                    result.hit=true; // tiny contact feedback, not injury
                    invulnerability=0.9f;
                }
            }
        }
        return result;
    }

    // When a map supplies an explicit W landmark, make it the arcade exit.
    // Invalid or blocked marker positions leave the generated exit in place.
    // Optional arcade guidance: select the *reachable* nearest shard by
    // shortest-path cost in the collision-grid BFS. This cue is not physical
    // acoustics and can be disabled with F6 to play by raw echoes.
    GuidanceTarget guidanceTarget() const {
        if(won)return {};
        if(remaining==0)return GuidanceTarget{exitX,exitY,true,true};
        int bestDistance=std::numeric_limits<int>::max();
        const Shard* closest=nullptr;
        for(const Shard& shard:shards) {
            if(shard.collected)continue;
            int n=nodeAt(shard.x,shard.y);
            if(distances_[n]>=0 && distances_[n]<bestDistance) {
                bestDistance=distances_[n];
                closest=&shard;
            }
        }
        if(!closest)return {};
        return GuidanceTarget{closest->x,closest->y,false,true};
    }

    void setExit(float x,float y) {
        if(x>=1 && y>=1 && x<side_-1 && y<side_-1 &&
           !circleGridContact(grid_,side_,x,y,kAgentRadius).hit) {
            exitX=x;exitY=y;
        }
    }

    bool reachExit(float px,float py) {
        if(!won&&remaining==0&&std::hypot(px-exitX,py-exitY)<18) {
            won=true; score+=500;return true;
        }
        return false;
    }
    // Shards are small reflecting targets in addition to their distinct
    // gameplay tones. Their material id 3 is a marker in the FDTD mask:
    // the CUDA boundary kernel decides the coefficient for that interface.
    // These are collectible (not movement-blocking) to preserve reachability.
    void stampShards(float* dynamicMask) const {
        // A finished round stops AI/movement, not the acoustic scenery.
        for(const Shard &s:shards){
            if(s.collected)continue;
            // A single centred 14cm octagon, with no separate square
            // glyph painted on top. Bigger than the old 10cm item, so
            // its facets remain visible on a one-centimetre grid.
            const int minX=std::max(1,(int)std::floor(s.x-7.0f));
            const int maxX=std::min(side_-2,(int)std::ceil(s.x+7.0f));
            const int minY=std::max(1,(int)std::floor(s.y-7.0f));
            const int maxY=std::min(side_-2,(int)std::ceil(s.y+7.0f));
            for(int y=minY;y<=maxY;++y)for(int x=minX;x<=maxX;++x){
                float dx=(x+0.5f)-s.x,dy=(y+0.5f)-s.y;
                if(insideNgon(dx,dy,7.0f,8,0.3926990817f)&&
                   grid_[y*side_+x]<=0 &&
                    dynamicMask[y*side_+x]<1.5f)
                    dynamicMask[y*side_+x]=3.0f;
            }
        }
    }
    void stampEnemies(float* dynamicMask) const {
        // Ghost reflectors must not disappear when the last life is lost.
        for(const Enemy&e:enemies){
            if(e.eaten)continue;
            // At 8cm radius/16 edges the grid raster looked like a
            // circle. Use a plainly faceted 20cm hexagonal reflector.
            const int minX=std::max(1,(int)std::floor(e.x-10));
            const int maxX=std::min(side_-2,(int)std::ceil(e.x+10));
            const int minY=std::max(1,(int)std::floor(e.y-10));
            const int maxY=std::min(side_-2,(int)std::ceil(e.y+10));
            for(int yy=minY; yy<=maxY;++yy)for(int xx=minX;xx<=maxX;++xx){
                const float dx=(xx+0.5f)-e.x,dy=(yy+0.5f)-e.y;
                if(insideNgon(dx,dy,10.0f,6,
                     (float)e.archetype*0.5235987756f) &&
                     grid_[yy*side_+xx]<=0)
                    dynamicMask[yy*side_+xx]=2.0f;
            }
        }
    }
    // lives and lost are retained only for backwards-compatible callers;
    // ghost collisions are nonlethal and never modify either field.
    int score=0,lives=3,remaining=0,charges=0,shardsCollected=0,combo=0;
    float overdrive=0.0f;
    bool won=false,lost=false;
    float invulnerability=0;
    float playerX=0,playerY=0,exitX=0,exitY=0;
    std::vector<Enemy> enemies;
    std::vector<Shard> shards;

private:
    static constexpr int kNavStep=5;
    static constexpr float kAgentRadius=11.0f;
    const float* grid_;
    int side_;
    float homeX_,homeY_;
    uint32_t rng_,seed_;
    int navSide_;
    std::vector<uint8_t> walkable_;
    std::vector<int> distances_;
    float navTimer_=0;

    float coord(int cell)const{return std::min((float)side_-1,(float)cell*kNavStep+2.5f);}
    int nodeAt(float x,float y)const{
        int cx=std::max(0,std::min(navSide_-1,(int)(x/kNavStep)));
        int cy=std::max(0,std::min(navSide_-1,(int)(y/kNavStep)));
        return cy*navSide_+cx;
    }
    uint32_t nextRand() {
        // Fixed-seed xorshift; reproducible and safe from modulo-by-zero.
        rng_^=rng_<<13; rng_^=rng_>>17; rng_^=rng_<<5; return rng_;
    }
    void computeDistances(float x,float y,std::vector<int>& field) const {
        std::fill(field.begin(),field.end(),-1);
        int root=nodeAt(x,y);
        if(!walkable_[root]) {
            // Nearest legal nav cell (for a valid player position close to
            // discrete Nav grid obstacle boundaries).
            float best=std::numeric_limits<float>::infinity();
            for(int i=0;i<(int)walkable_.size();++i)if(walkable_[i]) {
                const float dx=coord(i%navSide_)-x,dy=coord(i/navSide_)-y;
                const float d=dx*dx+dy*dy;
                if(d<best){best=d;root=i;}
            }
            if(!walkable_[root])return;
        }
        std::queue<int> pending;pending.push(root);field[root]=0;
        while(!pending.empty()){
            int n=pending.front();pending.pop();
            int cx=n%navSide_,cy=n/navSide_;
            const int directions[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
            for(const auto &d:directions) {
                int xx=cx+d[0], yy=cy+d[1];
                if(xx<0||xx>=navSide_||yy<0||yy>=navSide_)continue;
                int next=yy*navSide_+xx;
                if(walkable_[next]&&field[next]<0){
                    field[next]=field[n]+1;pending.push(next);
                }
            }
        }
    }
    void buildDistances(float x,float y){
        computeDistances(x,y,distances_);
    }
};

} // namespace echo
