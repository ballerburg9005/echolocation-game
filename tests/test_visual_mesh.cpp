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
    assert(b.size()==12+6+4*6); // top and four separate wall faces
    assert(b.size()%3==0);
    // Adjacent solids must not expose internal walls to the camera.
    mask[3*N+4]=1;
    auto c=echo::extrudeMask(mask.data(),N);
    assert(c.size()==12+6+6*6); // merged roof plus six exterior faces
    std::puts("3D visual mesh tests passed: room, acoustic-mask walls, exterior faces and merged roofs");
}
