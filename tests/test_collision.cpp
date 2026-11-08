#include "../collision.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const int n = 40;
    std::vector<float> grid(n*n, 0.0f);
    for (int y = 0; y < n; ++y) grid[y*n+20] = 1.0f;

    auto hit = echo::circleGridContact(grid.data(), n, 18.6f, 10.0f, 1.5f);
    assert(hit.hit && hit.normalX < -0.90f && std::fabs(hit.normalY) < 0.1f);
    auto clear = echo::circleGridContact(grid.data(), n, 12.0f, 10.0f, 1.5f);
    assert(!clear.hit);

    // Diagonal movement into a vertical wall must continue up the wall.
    auto moved = echo::slideCircle(grid.data(), n, 17.0f, 10.0f,
                                   4.0f, 7.0f, 1.5f);
    assert(moved.collided && moved.slid);
    assert(moved.x < 19.0f && moved.y > 14.0f);
    assert(moved.wallX > 0.8f);

    // Large steps may not tunnel through a 1-cell wall.
    auto blocked = echo::slideCircle(grid.data(), n, 10.0f, 10.0f,
                                     30.0f, 0.0f, 1.5f);
    assert(blocked.collided && blocked.x < 19.0f);

    // Outer grid limits are solid even if the level lacks explicit borders.
    auto edge = echo::slideCircle(grid.data(), n, 5.0f, 5.0f,
                                  -30.0f, 0.0f, 1.5f);
    assert(edge.collided && edge.x > 0.0f);
    std::puts("collision tests passed: contact normals, sliding, tunneling, world boundaries");
}
