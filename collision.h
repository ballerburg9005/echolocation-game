#pragma once
// CPU-only, resolution-independent player/obstacle collision and wall sliding.
#include <algorithm>
#include <cmath>

namespace echo {

struct GridContact {
    bool hit = false;
    float normalX = 0.0f; // from obstacle toward player
    float normalY = 0.0f;
};

inline GridContact circleGridContact(const float* mask, int size,
                                     float cx, float cy, float radius) {
    GridContact out;
    float sumX = 0.0f, sumY = 0.0f;
    const int x0 = (int)std::floor(cx - radius);
    const int x1 = (int)std::ceil(cx + radius);
    const int y0 = (int)std::floor(cy - radius);
    const int y1 = (int)std::ceil(cy + radius);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            if (x >= 0 && x < size && y >= 0 && y < size &&
                mask[y * size + x] <= 0.0f) continue;
            const float dx = cx - x, dy = cy - y;
            const float d2 = dx*dx + dy*dy;
            if (d2 > radius*radius) continue;
            out.hit = true;
            const float distance = std::sqrt(d2);
            if (distance > 0.00001f) {
                const float penetration = radius - distance + 0.5f;
                sumX += (dx/distance) * penetration;
                sumY += (dy/distance) * penetration;
            }
        }
    }
    const float length = std::sqrt(sumX*sumX + sumY*sumY);
    if (length > 0.00001f) {
        out.normalX = sumX / length;
        out.normalY = sumY / length;
    }
    return out;
}

struct SlideResult {
    float x, y;
    bool collided;
    bool slid;
    float wallX, wallY; // unit direction toward the latest obstacle
};

inline SlideResult slideCircle(const float* mask, int size,
                              float startX, float startY,
                              float deltaX, float deltaY, float radius) {
    SlideResult result{startX, startY, false, false, 0.0f, 0.0f};
    const float distance = std::sqrt(deltaX*deltaX + deltaY*deltaY);
    // Substep to avoid tunneling through thin SVG walls on slow frames.
    const int count = std::max(1, (int)std::ceil(distance / 2.0f));
    const float vx = deltaX / count, vy = deltaY / count;
    for (int i = 0; i < count; ++i) {
        const auto full = circleGridContact(mask, size, result.x + vx,
                                            result.y + vy, radius);
        if (!full.hit) {
            result.x += vx;
            result.y += vy;
            continue;
        }
        result.collided = true;
        if (full.normalX != 0.0f || full.normalY != 0.0f) {
            result.wallX = -full.normalX;
            result.wallY = -full.normalY;
            const float into = vx * full.normalX + vy * full.normalY;
            // Remove only the component aimed *into* the wall.
            const float dx = vx - std::min(0.0f, into) * full.normalX;
            const float dy = vy - std::min(0.0f, into) * full.normalY;
            if ((dx*dx + dy*dy) > 0.000001f &&
                !circleGridContact(mask, size, result.x + dx,
                                   result.y + dy, radius).hit) {
                result.x += dx;
                result.y += dy;
                result.slid = true;
                continue;
            }
        }
        // Axis fallbacks handle corners and non-axis-aligned obstacle normals.
        bool moved = false;
        if (vx != 0.0f &&
            !circleGridContact(mask, size, result.x + vx, result.y, radius).hit) {
            result.x += vx;
            moved = true;
        }
        if (vy != 0.0f &&
            !circleGridContact(mask, size, result.x, result.y + vy, radius).hit) {
            result.y += vy;
            moved = true;
        }
        result.slid |= moved;
    }
    return result;
}
} // namespace echo
