#include "../acoustics.h"
#include "../collision.h"
#include "../svg_raster.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <queue>
#include <string>
#include <vector>

// Parse the real shipped SVG's supported primitive rectangles and ellipses,
// then verify the new jagged room is reachable with the SAME 13cm player
// collider (not merely with an abstract grid-point path).
int main(){
    constexpr int N=500, step=5;
    std::ifstream svg("echo_arena.svg");
    assert(svg.good());
    std::vector<float> walls(N*N,0.0f);
    std::string line;
    int rectangles=0,ellipses=0,zigzagParts=0;
    while(std::getline(svg,line)){
        float x=0,y=0,w=0,h=0;
        if(std::sscanf(line.c_str(),
            " <rect x=\"%f\" y=\"%f\" width=\"%f\" height=\"%f\"",
            &x,&y,&w,&h)==4){
            assert(echo::rasterRect(walls.data(),N,x,y,w,h,0,true)>0);
            ++rectangles;
            if(x>=385 && x<=433 && y>=340 && y<420)++zigzagParts;
        }
        float cx=0,cy=0,rr=0;
        if(std::sscanf(line.c_str(),
            " <circle cx=\"%f\" cy=\"%f\" r=\"%f\"",
            &cx,&cy,&rr)==3){
            assert(echo::rasterEllipse(walls.data(),N,cx,cy,
                                       rr,rr,0,true)>0);
            ++ellipses;
        }
        float rx=0,ry=0;
        if(std::sscanf(line.c_str(),
            " <ellipse cx=\"%f\" cy=\"%f\" rx=\"%f\" ry=\"%f\"",
            &cx,&cy,&rx,&ry)==4){
            assert(echo::rasterEllipse(walls.data(),N,cx,cy,
                                       rx,ry,0,true)>0);
            ++ellipses;
        }
    }
    assert(rectangles>=17 && zigzagParts==5 && ellipses>=5);
    echo::stampOuterBoundary(walls.data(),N);
    assert(walls[0]==echo::kOuterBoundaryMaterial);
    assert(walls[250*N+1]==echo::kOuterBoundaryMaterial);
    assert(walls[250*N+2]==0);
    // Midpoints on the new stepped reflector are true FDTD hard solids.
    assert(walls[(N-354)*N+400]>0);
    assert(walls[(N-379)*N+425]>0);
    assert(walls[(N-400)*N+395]>0);

    // Sweep a conservative 5cm-centre graph with the actual player
    // collision routine: doorways and the route around the zigzag must
    // remain traversable, with no spawn trap or compulsory tight pinch.
    constexpr int S=N/step;
    std::vector<unsigned char> available(S*S,0),visited(S*S,0);
    for(int y=0;y<S;++y)for(int x=0;x<S;++x){
        const float px=2.5f+step*x,py=2.5f+step*y;
        available[y*S+x]=!echo::circleGridContact(
            walls.data(),N,px,py,13.0f).hit;
    }
    auto index=[](float x,float y) {
        return (int)((y-2.5f)/step)*S+(int)((x-2.5f)/step);
    };
    const int start=index(52.5f,152.5f),goal=index(467.5f,107.5f);
    assert(available[start] && available[goal]);
    std::queue<int> queue;
    queue.push(start);visited[start]=1;
    const int direction[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    while(!queue.empty()){
        const int v=queue.front();queue.pop();
        const int x=v%S,y=v/S;
        for(const auto& d:direction){
            const int nx=x+d[0],ny=y+d[1];
            if(nx<0||ny<0||nx>=S||ny>=S)continue;
            const int u=ny*S+nx;
            if(available[u]&&!visited[u]){
                visited[u]=1;queue.push(u);
            }
        }
    }
    assert(visited[goal]);
    std::puts("echo arena: five actual zigzag reflector steps, softened outer ring and clear 13cm-radius route");
}
