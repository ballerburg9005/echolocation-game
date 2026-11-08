#include "../svg_raster.h"
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    constexpr int N = 50;
    std::vector<float> mask(N*N, 0.0f);
    assert(echo::rasterRect(mask.data(), N, 10, 10, 4, 6, 0, true) == 24);
    // SVG top-left: upper SVG rows map to higher game-space y indexes.
    assert(mask[(N-11)*N+10] == 1.0f);
    assert(mask[(N-17)*N+10] == 0.0f);
    assert(mask[(N-11)*N+14] == 0.0f);

    std::fill(mask.begin(), mask.end(), 0.0f);
    assert(echo::rasterRect(mask.data(), N, 10, 10, 4, 6, 0, false) == 0);
    assert(echo::rasterRect(mask.data(), N, 10, 10, 4, 6, 2, false) > 0);
    assert(mask[(N-13)*N+12] == 0.0f); // hollow stroke

    std::fill(mask.begin(), mask.end(), 0.0f);
    assert(echo::rasterEllipse(mask.data(), N, 20, 20, 4, 6, 0, true) > 0);
    assert(mask[(N-21)*N+20] == 1.0f);
    assert(mask[(N-21)*N+27] == 0.0f);
    std::fill(mask.begin(), mask.end(), 0.0f);
    assert(echo::rasterEllipse(mask.data(), N, 20, 20, 4, 6, 2, false) > 0);
    assert(mask[(N-21)*N+20] == 0.0f);
    assert(echo::rasterRect(mask.data(), N, 0, 0, -10, 10, 0, true) == 0);
    std::puts("SVG raster tests passed: rect/ellipse, fill/stroke, clipping and Y flip");
}
