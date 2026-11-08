#include "../visual_mesh.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
int main() {
    const int N=8;
    std::vector<float> mask(N*N,0);
    auto a=echo::extrudeMask(mask.data(),N);
    assert(a.size()==12); // floor and ceiling
    for(const auto &v:a)assert(std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z));
    mask[3*N+3]=1;
    auto b=echo::extrudeMask(mask.data(),N);
    assert(b.size()==12+6+4*6);
    // Roof and ceiling must not overlap in the same plane (z-fighting).
    float ceiling=-1.0f, roof=-1.0f;
    for(size_t i=0;i<b.size();++i) {
        if(i>=6 && i<12)ceiling=std::max(ceiling,b[i].y);
        if(i>=12 && i<18)roof=std::max(roof,b[i].y);
    }
    assert(ceiling>roof+0.01f); // top and four separate wall faces
    assert(b.size()%3==0);
    // Adjacent solids must not expose internal walls to the camera.
    mask[3*N+4]=1;
    auto c=echo::extrudeMask(mask.data(),N);
    assert(c.size()==12+6+6*6); // merged roof plus six exterior faces
    // OpenGL view basis: looking east, screen-right points +Z.
    // Increasing bearing is a LEFT turn: an eastward fixed landmark should
    // move to screen-right as the camera turns toward north (+grid Y).
    const float north=echo::visualForwardZ(3.14159265358979323846f/2.0f);
    assert(north < -0.99f);
    assert(echo::visualZFromGridY(100) < echo::visualZFromGridY(20));
    // The player and walls use the SAME grid-to-world orientation.
    bool negativeWallZ=false;
    for(const auto &v:c)if(v.z < -0.03f)negativeWallZ=true;
    assert(negativeWallZ);
    std::puts("3D visual mesh tests passed: room, acoustic-mask walls, exterior faces and merged roofs");
}
