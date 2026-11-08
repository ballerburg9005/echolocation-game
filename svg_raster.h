#pragma once
// Minimal pixel-space SVG primitive rasterization. Coordinates use SVG's
// top-left origin; the game grid uses bottom-left. Neither arbitrary path
// geometry nor element/group transforms are handled by this module.
#include <algorithm>
#include <cmath>

namespace echo {

inline int rasterRect(float* mask, int size, float x, float y, float width,
                      float height, float stroke, bool filled) {
    if (!(width > 0 && height > 0) || !std::isfinite(x) || !std::isfinite(y)
        || !std::isfinite(width) || !std::isfinite(height)) return 0;
    const float half = std::max(0.0f, stroke) * 0.5f;
    // Clamp while still floating-point: converting huge SVG coordinates
    // directly to int is undefined and can make raster loops unbounded.
    const int minX = (int)std::max(0.0f, std::min((float)size, std::floor(x-half)));
    const int maxX = (int)std::max(-1.0f, std::min((float)(size-1), std::ceil(x+width+half)));
    const int minY = (int)std::max(0.0f, std::min((float)size, std::floor(size-(y+height+half))));
    const int maxY = (int)std::max(-1.0f, std::min((float)(size-1), std::ceil(size-(y-half))));
    int marked = 0;
    for (int iy = minY; iy <= maxY; ++iy) {
        const float sy = size - iy - 0.5f;
        for (int ix = minX; ix <= maxX; ++ix) {
            const float sx = ix + 0.5f;
            const bool insideFill = filled &&
                sx >= x && sx <= x+width && sy >= y && sy <= y+height;
            const bool insideOuter = half > 0 &&
                sx >= x-half && sx <= x+width+half &&
                sy >= y-half && sy <= y+height+half;
            const bool insideInner =
                sx > x+half && sx < x+width-half &&
                sy > y+half && sy < y+height-half;
            if (insideFill || (insideOuter && !insideInner)) {
                mask[iy*size+ix] = 1.0f;
                ++marked;
            }
        }
    }
    return marked;
}

inline int rasterEllipse(float* mask, int size, float cx, float cy,
                         float rx, float ry, float stroke, bool filled) {
    if (!(rx > 0 && ry > 0) || !std::isfinite(cx) || !std::isfinite(cy)
        || !std::isfinite(rx) || !std::isfinite(ry)) return 0;
    const float half = std::max(0.0f, stroke)*0.5f;
    const int minX = (int)std::max(0.0f, std::min((float)size, std::floor(cx-rx-half)));
    const int maxX = (int)std::max(-1.0f, std::min((float)(size-1), std::ceil(cx+rx+half)));
    const int minY = (int)std::max(0.0f, std::min((float)size, std::floor(size-(cy+ry+half))));
    const int maxY = (int)std::max(-1.0f, std::min((float)(size-1), std::ceil(size-(cy-ry-half))));
    int marked = 0;
    for (int iy = minY; iy <= maxY; ++iy) {
        const float sy = size-iy-0.5f;
        for (int ix = minX; ix <= maxX; ++ix) {
            const float sx = ix+0.5f, dx=sx-cx, dy=sy-cy;
            const bool insideFill = filled &&
                dx*dx/(rx*rx) + dy*dy/(ry*ry) <= 1.0f;
            const bool insideOuter = half > 0 &&
                dx*dx/((rx+half)*(rx+half)) +
                dy*dy/((ry+half)*(ry+half)) <= 1.0f;
            const bool insideInner = rx>half && ry>half &&
                dx*dx/((rx-half)*(rx-half)) +
                dy*dy/((ry-half)*(ry-half)) < 1.0f;
            if (insideFill || (insideOuter && !insideInner)) {
                mask[iy*size+ix] = 1.0f;
                ++marked;
            }
        }
    }
    return marked;
}
} // namespace echo
