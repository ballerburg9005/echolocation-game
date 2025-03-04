#include <SDL2/SDL.h>
#include <SDL2/SDL_main.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <tinyxml2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include "graphics_setup.h"
#include "render_3d_view.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

static float g_DX              = 0.01f;
static float g_DT;
static float g_pulseIntervalSec= 0.3f;
static float g_decayFactor     = 1.0f-(0.0005f*0.5f);
static float g_reflection      = 0.9f;
static float g_timeFudgeFactor = 0.985f;
static float g_pulseFreq       = 10000.f;

static int   g_sampleRate      = 48000;
static float g_c2;
static int   g_pulseIntervalSteps;
static int   g_pulseDurationSteps;

static float g_waveVolumeScale = 0.05f;

static const float SPEED_OF_SOUND = 343.0f;
int   g_width, g_height;
static int WINDOW_WIDTH = 1000;  // Fixed: 2x500px for 2D views
static int WINDOW_HEIGHT = 1000; // Fixed: 500px 3D + 500px 2D
static const int AUDIO_BATCH = 4096;
__constant__ float ZOOM_FACTOR = 4.0f;

// Define these globally with external linkage
std::vector<Rect> g_rects;
std::vector<Circle> g_circles;
std::vector<ToneMarker> g_toneMarkers;
static ToneMarker* d_toneMarkers = nullptr;
static int g_numToneMarkers = 0;

float g_player_pivot_x = 0.f, g_player_pivot_y = 0.f, g_player_angle = -(float)M_PI / 4.f;

// --------------------------------------------------------------------------------
// NEW STATIC MASK (for block skipping) and ACTIVE BLOCK LIST
// --------------------------------------------------------------------------------
static float* h_staticMask = nullptr;  // Holds permanent occluders loaded from SVG
static std::vector<int> g_activeBlockList; // List of tile indices that are not fully occluded
static int* d_activeBlockList = nullptr;   // Device copy of above
static int g_activeBlockCount = 0;         // Number of active tiles
static int g_numBlocksX = 0;               // (g_width + 31)/32
static int g_numBlocksY = 0;               // (g_height + 31)/32

// Each block is 32×32 threads
static const int BLOCK_SIZE_X = 32;
static const int BLOCK_SIZE_Y = 32;

// --------------------------------------------------------------------------------

static void recalcDerivedParams(bool print = false)
{
    g_DT = (g_DX / (SPEED_OF_SOUND * 1.41421356237f));
    g_c2 = (SPEED_OF_SOUND * g_DT / g_DX) * (SPEED_OF_SOUND * g_DT / g_DX);

    if (g_pulseIntervalSec < 0.1f) g_pulseIntervalSec = 0.1f;
    if (g_pulseIntervalSec > 5.f)  g_pulseIntervalSec = 5.f;
    g_pulseIntervalSteps = (int)(g_pulseIntervalSec / g_DT);

    g_pulseDurationSteps = (int)(0.0005f / g_DT);
    if (g_pulseDurationSteps < 1) g_pulseDurationSteps = 1;

    if (g_reflection < 0.f) g_reflection = 0.f;
    if (g_reflection > 1.f) g_reflection = 1.f;

    if (g_decayFactor < 0.f) g_decayFactor = 0.f;
    if (g_decayFactor > 1.f) g_decayFactor = 1.f;

    if (g_timeFudgeFactor < 0.9f) g_timeFudgeFactor = 0.9f;
    if (g_timeFudgeFactor > 1.0f) g_timeFudgeFactor = 1.0f;

    if (g_pulseFreq < 60.f) g_pulseFreq = 1000.f;
    if (g_pulseFreq > 20000.f) g_pulseFreq = 20000.f;

    if (print) {
        printf("[PARAMS] Pulse=%.2fs  Decay=%.6f  Reflect=%.3f  Fudge=%.4f  Freq=%.0f Hz\n",
               g_pulseIntervalSec, g_decayFactor, g_reflection, g_timeFudgeFactor, g_pulseFreq);
    }
}

struct Player {
    float pivot_x;
    float pivot_y;
    float angle;
    Player& operator=(const Player& other) {
        pivot_x = other.pivot_x; pivot_y = other.pivot_y; angle = other.angle; return *this;
    }
};

extern float g_player_pivot_x, g_player_pivot_y, g_player_angle;
static Player g_player = {0.f, 0.f, -(float)M_PI / 4.f};
static float g_mic_l_x, g_mic_l_y;
static float g_mic_r_x, g_mic_r_y;
static float g_wedge_x, g_wedge_y;
static float g_pulse_x, g_pulse_y;

static void updatePlayerComponents(Player &p)
{
    float c = cosf(p.angle), s = sinf(p.angle);
    float lx = 6.f, ly = 5.5f, rx = -6.f, ry = -5.5f, wx = -1.f, wy = 1.f, px = 4.f, py = -4.f;

    g_mic_l_x = p.pivot_x + lx * c - ly * s;
    g_mic_l_y = p.pivot_y + lx * s + ly * c;
    g_mic_r_x = p.pivot_x + rx * c - ry * s;
    g_mic_r_y = p.pivot_y + rx * s + ry * c;
    g_wedge_x = p.pivot_x + wx * c - wy * s;
    g_wedge_y = p.pivot_y + wx * s + wy * c;
    g_pulse_x = p.pivot_x + px * c - py * s;
    g_pulse_y = p.pivot_y + px * s + py * c;
}

// SVG parsing code is unchanged, except we fill "h_staticMask" (not changed each frame)
static void parseSVGElement(tinyxml2::XMLElement* elem, float* h_mask)
{
    using namespace tinyxml2;
    while (elem) {
        const char* name = elem->Name();
        if (strcmp(name, "rect") == 0) {
            Rect r;
            elem->QueryFloatAttribute("x", &r.x);
            elem->QueryFloatAttribute("y", &r.y);
            elem->QueryFloatAttribute("width", &r.width);
            elem->QueryFloatAttribute("height", &r.height);
            
            float y_top = g_height - (r.y + r.height);
            float y_bottom = g_height - r.y;
            
            int x0 = (int)floorf(r.x);
            int y0 = (int)floorf(y_top);
            int x1 = (int)ceilf(r.x + r.width);
            int y1 = (int)ceilf(y_bottom);

            printf("[SVG] Rect: x=%.1f, y=%.1f, w=%.1f, h=%.1f (grid: %d,%d to %d,%d)\n",
                   r.x, r.y, r.width, r.height, x0, y0, x1, y1);

            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    if (x >= 0 && x < g_width && y >= 0 && y < g_height) {
                        h_mask[y * g_width + x] = 1.f;
                    }
                }
            }
            r.y = y_top;
            g_rects.push_back(Rect{r.x, r.y, r.width, r.height});
        }
        else if (strcmp(name, "path") == 0) {
            const char* type = elem->Attribute("sodipodi:type");
            if (type && (strcmp(type, "arc") == 0 || strcmp(type, "circle") == 0)) {
                Circle c;
                elem->QueryFloatAttribute("sodipodi:cx", &c.cx);
                elem->QueryFloatAttribute("sodipodi:cy", &c.cy);
                elem->QueryFloatAttribute("sodipodi:rx", &c.r);
                
                c.cy = g_height - c.cy;
                
                int cx = (int)roundf(c.cx);
                int cy = (int)roundf(c.cy);
                int r = (int)ceilf(c.r);
                
                printf("[SVG] Circle: cx=%.1f, cy=%.1f, r=%.1f\n", c.cx, c.cy, c.r);

                for (int y = cy - r; y <= cy + r; y++) {
                    for (int x = cx - r; x <= cx + r; x++) {
                        float dx = x - c.cx;
                        float dy = y - c.cy;
                        if (dx * dx + dy * dy <= c.r * c.r) {
                            if (x >= 0 && x < g_width && y >= 0 && y < g_height) {
                                h_mask[y * g_width + x] = 1.f;
                            }
                        }
                    }
                }
                g_circles.push_back(Circle{c.cx, c.cy, c.r});
            }
        }
        else if (strcmp(name, "text") == 0) {
            XMLElement* tspan = elem->FirstChildElement("tspan");
            const char* text = tspan ? tspan->GetText() : elem->GetText();
            if (text && (strstr(text, "O") || strstr(text, "W") || strstr(text, "I"))) {
                ToneMarker tm;
                elem->QueryFloatAttribute("x", &tm.x);
                elem->QueryFloatAttribute("y", &tm.y);
                tm.y = g_height - tm.y;
                if (strstr(text, "O")) tm.type = 'O';
                else if (strstr(text, "W")) tm.type = 'W';
                else if (strstr(text, "I")) tm.type = 'I';
                tm.isPlaying = false;
                printf("[SVG] Tone Marker (%c): x=%.1f, y=%.1f\n", tm.type, tm.x, tm.y);
                g_toneMarkers.push_back(ToneMarker{tm.x, tm.y, tm.type, tm.isPlaying});
            }
        }

        parseSVGElement(elem->FirstChildElement(), h_mask);
        elem = elem->NextSiblingElement();
    }
}

static bool loadSVG(const char* filename, float* &h_mask)
{
    using namespace tinyxml2;
    XMLDocument doc;
    if (doc.LoadFile(filename) != XML_SUCCESS) {
        fprintf(stderr, "Failed to load SVG file: %s\n", filename);
        return false;
    }

    XMLElement* svg = doc.FirstChildElement("svg");
    if (!svg) {
        fprintf(stderr, "No <svg> root element found\n");
        return false;
    }

    float width, height;
    svg->QueryFloatAttribute("width", &width);
    svg->QueryFloatAttribute("height", &height);
    g_width = (int)ceilf(width);
    g_height = (int)ceilf(height);

    h_mask = (float*)calloc(g_width * g_height, sizeof(float));
    memset(h_mask, 0, g_width * g_height * sizeof(float));
    
    int w = 2;
    for (int y = 0; y < g_height; y++) {
        for (int x = 0; x < g_width; x++) {
            if (x < w || x >= g_width - w || y < w || y >= g_height - w) {
                h_mask[y * g_width + x] = 1.f;
            }
        }
    }
    printf("[SVG] Added boundaries: %dx%d\n", g_width, g_height);

    parseSVGElement(svg->FirstChildElement(), h_mask);

    g_player.pivot_x = g_width * 0.1f;
    g_player.pivot_y = g_height * 0.3f;
    updatePlayerComponents(g_player);

    printf("[SVG] Loaded %d rects, %d circles, %d tone markers\n", 
           (int)g_rects.size(), (int)g_circles.size(), (int)g_toneMarkers.size());
    return true;
}

// This builds the dynamic mask each frame by copying static mask and then
// "drawing" the wedge. We do NOT use dynamic changes for block skipping.
static void updateDynamicMask(const float* h_static, float* h_dynamic,
                              float wedge_x, float wedge_y, float pivot_x, float pivot_y) 
{
    memcpy(h_dynamic, h_static, g_width * g_height * sizeof(float));
    float fwd_angle = atan2f(g_pulse_y - pivot_y, g_pulse_x - pivot_x);
    float wedge_half = 38.0f * (float)M_PI / 180.0f;
    int wedge_length = 10;

    for (int i = 0; i <= wedge_length; i++) {
        float cell1_x = wedge_x + i * cosf(fwd_angle - wedge_half);
        float cell1_y = wedge_y + i * sinf(fwd_angle - wedge_half);
        float cell2_x = wedge_x + i * cosf(fwd_angle + wedge_half);
        float cell2_y = wedge_y + i * sinf(fwd_angle + wedge_half);

        int ix1 = (int)roundf(cell1_x);
        int iy1 = (int)roundf(cell1_y);
        int ix2 = (int)roundf(cell2_x);
        int iy2 = (int)roundf(cell2_y);

        if (ix1 >= 0 && ix1 < g_width && iy1 >= 0 && iy1 < g_height)
            h_dynamic[iy1 * g_width + ix1] = 1.0f;
        if (ix2 >= 0 && ix2 < g_width && iy2 >= 0 && iy2 < g_height)
            h_dynamic[iy2 * g_width + ix2] = 1.0f;
    }
}

// --------------------------------------------------------------------------------
// Build a list of "active" blocks from the *static* mask only.
// If a 32×32 tile is fully occluded (mask=1 everywhere), skip it entirely.
// --------------------------------------------------------------------------------
static void buildActiveBlockList(const float* h_mask, int width, int height,
                                 int blockSizeX, int blockSizeY,
                                 std::vector<int>& activeBlocks)
{
    activeBlocks.clear();
    int nbx = (width  + blockSizeX - 1) / blockSizeX;
    int nby = (height + blockSizeY - 1) / blockSizeY;

    for (int by = 0; by < nby; by++) {
        for (int bx = 0; bx < nbx; bx++) {
            bool fullyOccluded = true;
            int startX = bx * blockSizeX;
            int startY = by * blockSizeY;
            for (int yy = 0; yy < blockSizeY && (startY + yy) < height; yy++) {
                for (int xx = 0; xx < blockSizeX && (startX + xx) < width; xx++) {
                    int gx = startX + xx;
                    int gy = startY + yy;
                    if (h_mask[gy * width + gx] == 0.0f) {
                        fullyOccluded = false;
                        break;
                    }
                }
                if (!fullyOccluded) break;
            }
            if (!fullyOccluded) {
                // Keep this tile
                int tileId = by * nbx + bx;
                activeBlocks.push_back(tileId);
            }
        }
    }
}

// --------------------------------------------------------------------------------
// Tiled FDTD update kernel:
// We skip entire tiles that are fully occluded in the *static* mask. For partial
// tiles, we still do the wave update. We still do reflection using the *dynamic*
// mask that can include the wedge.
//
// (We replicate the shared-memory approach from the original "updatePressure"
//  kernel, adapted to a per-tile approach. Each CUDA block handles one tile.)
// --------------------------------------------------------------------------------
__global__ void updatePressureTiled(float* p_next, const float* p,
                                    const float* p_prev, const float* mask,
                                    float c2, int width, int height,
                                    float reflection, const int* activeBlocks,
                                    int numBlocksX)
{
    // Which tile are we processing?
    int tileIndex = blockIdx.x; // each block in 1D
    int tileId    = activeBlocks[tileIndex];

    // Decode (bx,by)
    int by = tileId / numBlocksX;
    int bx = tileId % numBlocksX;

    // Each tile is 32×32
    // We use shared memory with a halo as in the original code
    __shared__ float s_p[BLOCK_SIZE_Y+2][BLOCK_SIZE_X+2];

    // global x,y for this thread
    int localX = threadIdx.x; // 0..31
    int localY = threadIdx.y; // 0..31
    int x = bx * BLOCK_SIZE_X + localX;
    int y = by * BLOCK_SIZE_Y + localY;

    // If we are out-of-bounds, just exit
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    // We copy from p[] to shared memory with a 1-cell halo, so we read neighbors
    // s_p[threadIdx.y+1][threadIdx.x+1] = p[idx], plus edges
    // First load the center
    s_p[localY+1][localX+1] = p[idx];

    // Load left/right neighbors if needed
    if (localX == 0 && x > 0) {
        s_p[localY+1][0] = p[idx - 1];
    }
    if (localX == BLOCK_SIZE_X-1 && x < width-1) {
        s_p[localY+1][BLOCK_SIZE_X+1] = p[idx + 1];
    }
    // Load top/bottom neighbors if needed
    if (localY == 0 && y > 0) {
        s_p[0][localX+1] = p[idx - width];
    }
    if (localY == BLOCK_SIZE_Y-1 && y < height-1) {
        s_p[BLOCK_SIZE_Y+1][localX+1] = p[idx + width];
    }

    __syncthreads();

    // If on outer boundary, just skip big updates
    if (x < 1 || x >= width-1 || y < 1 || y >= height-1) {
        return;
    }

    // If occluded in dynamic mask: reflection
    if (mask[idx] > 0.f) {
        p_next[idx] = -reflection * s_p[localY+1][localX+1];
        return;
    }

    // Otherwise, do normal wave update
    float lap = s_p[localY+1][localX+2]   // x+1
              + s_p[localY+1][localX]     // x-1
              + s_p[localY+2][localX+1]   // y+1
              + s_p[localY][localX+1]     // y-1
              - 4.f * s_p[localY+1][localX+1];

    p_next[idx] = 2.f * s_p[localY+1][localX+1] - p_prev[idx] + c2 * lap;
}

__global__ void applyDecay(float* p, float decay, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = width * height;
    if (idx < totalSize) {
        p[idx] *= decay;
    }
}

__global__ void addPulse(float* p, float sx, float sy, float amplitude,
                         int t, int interval, int duration, float dt,
                         int width, int height, float freq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        int adj = t - 1000;
        if (adj >= 0 && (adj % interval) < duration) {
            float t_adj = (adj % duration) * dt;
            float val = amplitude * sinf(2.f * freq * t_adj);
            int ix = (int)roundf(sx), iy = (int)roundf(sy);
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                atomicAdd(&p[iy * width + ix], val);
            }
        }
    }
}

__global__ void captureAndReduce(const float* p, float lx, float ly,
                                 float rx, float ry, int sampleIndex,
                                 float* audioL, float* audioR, float waveVolume,
                                 int width, int height)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int ilx = (int)roundf(lx), ily = (int)roundf(ly);
        int irx = (int)roundf(rx), iry = (int)roundf(ry);
        if (ilx < 0) ilx = 0; if (ilx >= width) ilx = width - 1;
        if (ily < 0) ily = 0; if (ily >= height) ily = height - 1;
        if (irx < 0) irx = 0; if (irx >= width) irx = width - 1;
        if (iry < 0) iry = 0; if (iry >= height) iry = height - 1;

        float vl = p[ily * width + ilx];
        float vr = p[iry * width + irx];
        audioL[sampleIndex] = vl * waveVolume;
        audioR[sampleIndex] = vr * waveVolume;
    }
}

// The updateVBO kernel remains unchanged except for referencing the dynamic mask
__global__ void updateVBO(float* vbo_data, float* p, float* mask,
                          int width, int height, int showObj, int t,
                          float mlx, float mly, float mrx, float mry,
                          float px, float py, float wx, float wy,
                          float player_x, float player_y, float player_angle,
                          const ToneMarker* toneMarkers, int numToneMarkers)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    int vbo_idx = idx * 3;
    int fp_idx = (width * height + idx) * 3;

    float pxOut_static = 2.f * x / width - 1.f;
    float pyOut_static = 2.f * y / height - 1.f;
    vbo_data[vbo_idx] = pxOut_static * 0.5f - 0.5f;
    vbo_data[vbo_idx + 1] = pyOut_static;

    float dx = x - player_x;
    float dy = y - player_y;
    float c = cosf(-player_angle - (float)M_PI / 4.f + (float)M_PI);
    float s = sinf(-player_angle - (float)M_PI / 4.f + (float)M_PI);
    float rx = dx * c - dy * s;
    float ry = dy * c + dx * s;
    float pxOut_fp = (rx * ZOOM_FACTOR) / (width / 2.0f);
    float pyOut_fp = (ry * ZOOM_FACTOR) / (height / 2.0f);
    float fp_x = pxOut_fp * 0.5f + 0.5f;

    if (vbo_data[vbo_idx] < 0.f) {
        float wave = (t < 1000) ? 0.f : p[idx];
        float dx_mic = x - mlx, dy_mic = y - mly;
        if (dx_mic * dx_mic + dy_mic * dy_mic <= 4.f) {
            vbo_data[vbo_idx + 2] = -2.f;
        } else {
            dx_mic = x - mrx; dy_mic = y - mry;
            if (dx_mic * dx_mic + dy_mic * dy_mic <= 4.f) {
                vbo_data[vbo_idx + 2] = -2.f;
            } else {
                float dpx = x - px, dpy = y - py;
                if (dpx * dpx + dpy * dpy <= 1.f) {
                    vbo_data[vbo_idx + 2] = -3.f;
                } else {
                    bool toneMarker = false;
                    for (int i = 0; i < numToneMarkers; i++) {
                        float dtx = x - toneMarkers[i].x;
                        float dty = y - toneMarkers[i].y;
                        if (dtx * dtx + dty * dty <= 25.f) {
                            if (toneMarkers[i].type == 'O') vbo_data[vbo_idx + 2] = -5.f;
                            else if (toneMarkers[i].type == 'W') vbo_data[vbo_idx + 2] = -6.f;
                            else if (toneMarkers[i].type == 'I') vbo_data[vbo_idx + 2] = -7.f;
                            toneMarker = true;
                            break;
                        }
                    }
                    if (!toneMarker) {
                        if (showObj && mask[idx] > 0.f) {
                            vbo_data[vbo_idx + 2] = -1.f;
                        } else {
                            vbo_data[vbo_idx + 2] = tanhf(wave * 0.01f);
                        }
                    }
                }
            }
        }
    } else {
        vbo_data[vbo_idx + 2] = -10.f;
    }

    if (fp_x >= 0.f && fp_x <= 1.f) {
        vbo_data[fp_idx] = fp_x;
        vbo_data[fp_idx + 1] = pyOut_fp;
        float wave = (t < 1000) ? 0.f : p[idx];
        float dx_mic = x - mlx, dy_mic = y - mly;
        if (dx_mic * dx_mic + dy_mic * dy_mic <= 4.f) {
            vbo_data[fp_idx + 2] = -2.f;
        } else {
            dx_mic = x - mrx; dy_mic = y - mry;
            if (dx_mic * dx_mic + dy_mic * dy_mic <= 4.f) {
                vbo_data[fp_idx + 2] = -2.f;
            } else {
                float dpx = x - px, dpy = y - py;
                if (dpx * dpx + dpy * dpy <= 1.f) {
                    vbo_data[fp_idx + 2] = -3.f;
                } else {
                    bool toneMarker = false;
                    for (int i = 0; i < numToneMarkers; i++) {
                        float dtx = x - toneMarkers[i].x;
                        float dty = y - toneMarkers[i].y;
                        if (dtx * dtx + dty * dty <= 25.f) {
                            if (toneMarkers[i].type == 'O') vbo_data[fp_idx + 2] = -5.f;
                            else if (toneMarkers[i].type == 'W') vbo_data[fp_idx + 2] = -6.f;
                            else if (toneMarkers[i].type == 'I') vbo_data[fp_idx + 2] = -7.f;
                            toneMarker = true;
                            break;
                        }
                    }
                    if (!toneMarker) {
                        if (showObj && mask[idx] > 0.f) {
                            vbo_data[fp_idx + 2] = -1.f;
                        } else {
                            vbo_data[fp_idx + 2] = tanhf(wave * 0.01f);
                        }
                    }
                }
            }
        }
    } else {
        vbo_data[fp_idx + 2] = -10.f;
    }
}

// Collisions & player movement code is unchanged
static bool checkPlayerCollision(float px, float py, const float* h_mask,
                                 float& collisionAngle, float playerAngle,
                                 float& normalX, float& normalY)
{
    const float radius = 10.0f;
    int ixmin = (int)fmaxf(0, floorf(px - radius));
    int ixmax = (int)fminf(g_width - 1, ceilf(px + radius));
    int iymin = (int)fmaxf(0, floorf(py - radius));
    int iymax = (int)fminf(g_height - 1, ceilf(py + radius));
    float r2 = radius * radius;

    bool collided = false;
    const int nS = 720;
    float bestScore = -1.f, bestAngle = 0.f;
    float bestNX = 0.f, bestNY = 0.f;

    for (int i = 0; i < nS; i++) {
        float a = -M_PI + i * (2.0 * M_PI / nS);
        float sc = 0.f;
        float sumNX = 0.f, sumNY = 0.f;
        int collisionCount = 0;
        for (float r = 0.f; r <= radius; r += 0.25f) {
            float gx = px + r * cos(a);
            float gy = py + r * sin(a);
            int ix = (int)roundf(gx), iy = (int)roundf(gy);
            if (ix >= 0 && ix < g_width && iy >= 0 && iy < g_height) {
                if (h_mask[iy * g_width + ix] > 0.f) {
                    sc += 1.f;
                    float nx = cos(a + M_PI);
                    float ny = sin(a + M_PI);
                    sumNX += nx;
                    sumNY += ny;
                    collisionCount++;
                }
            }
        }
        if (sc > bestScore) {
            bestScore = sc;
            bestAngle = a;
            if (collisionCount > 0) {
                bestNX = sumNX / collisionCount;
                bestNY = sumNY / collisionCount;
                float len = sqrtf(bestNX * bestNX + bestNY * bestNY);
                if (len > 1e-6f) {
                    bestNX /= len;
                    bestNY /= len;
                }
            }
        }
    }

    if (bestScore > 0.f) {
        normalX = bestNX;
        normalY = bestNY;
        collisionAngle = atan2f(normalY, normalX) - playerAngle;
        if (collisionAngle < -M_PI) collisionAngle += 2.0 * M_PI;
        if (collisionAngle > M_PI) collisionAngle -= 2.0 * M_PI;
    } else {
        normalX = normalY = 0.f;
        collisionAngle = 0.f;
    }

    for (int y = iymin; y <= iymax; y++) {
        for (int x = ixmin; x <= ixmax; x++) {
            float dx = x - px, dy = y - py;
            if (dx * dx + dy * dy <= r2 && h_mask[y * g_width + x] > 0.f) {
                collided = true;
                break;
            }
        }
        if (collided) break;
    }

    return collided;
}

int main(int argc, char* argv[])
{
    const char* svgFile = (argc >= 2) ? argv[1] : "drawing.svg";
    // Originally, we loaded h_mask. We'll now call that "h_staticMask."
    // We'll also have a separate h_dynamic for wedge updates each frame.
    if (!loadSVG(svgFile, h_staticMask)) {
        fprintf(stderr, "Could not load %s, proceeding with default 500x500\n", svgFile);
        g_width = 500;
        g_height = 500;
        h_staticMask = (float*)calloc(g_width*g_height, sizeof(float));
        int w = 2;
        for (int y = 0; y < g_height; y++) {
            for (int x = 0; x < g_width; x++) {
                if (x < w || x >= g_width - w || y < w || y >= g_height - w) {
                    h_staticMask[y * g_width + x] = 1.f;
                }
            }
        }
        g_player.pivot_x = g_width * 0.1f;
        g_player.pivot_y = g_height * 0.3f;
        updatePlayerComponents(g_player);
        printf("[DEFAULT] Running with empty room and boundaries: %dx%d\n", g_width, g_height);
    }

    g_numToneMarkers = g_toneMarkers.size();
    if (g_numToneMarkers > 0) {
        CUDA_CHECK(cudaMalloc(&d_toneMarkers, g_numToneMarkers * sizeof(ToneMarker)));
        CUDA_CHECK(cudaMemcpy(d_toneMarkers, g_toneMarkers.data(), 
                              g_numToneMarkers * sizeof(ToneMarker), 
                              cudaMemcpyHostToDevice));
    }

    // Build active block list (static geometry only)
    g_numBlocksX = (g_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    g_numBlocksY = (g_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;

    buildActiveBlockList(h_staticMask, g_width, g_height,
                         BLOCK_SIZE_X, BLOCK_SIZE_Y, g_activeBlockList);

    g_activeBlockCount = (int)g_activeBlockList.size();
    printf("[BLOCK SKIP] Total tiles: %d x %d = %d\n",
           g_numBlocksX, g_numBlocksY, g_numBlocksX*g_numBlocksY);
    printf("[BLOCK SKIP] Active tiles: %d\n", g_activeBlockCount);

    // Copy active block list to device
    CUDA_CHECK(cudaMalloc(&d_activeBlockList, g_activeBlockCount*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_activeBlockList, g_activeBlockList.data(),
                          g_activeBlockCount*sizeof(int), cudaMemcpyHostToDevice));

    float* d_mask;
    CUDA_CHECK(cudaMalloc(&d_mask, g_width * g_height * sizeof(float)));

    float *d_p, *d_p_prev, *d_p_next;
    CUDA_CHECK(cudaMalloc(&d_p,      g_width * g_height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_prev, g_width * g_height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_next, g_width * g_height * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p,      0, g_width*g_height*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p_prev, 0, g_width*g_height*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p_next, 0, g_width*g_height*sizeof(float)));

    float *d_audioL, *d_audioR;
    CUDA_CHECK(cudaMalloc(&d_audioL, AUDIO_BATCH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_audioR, AUDIO_BATCH * sizeof(float)));
    float* h_audioBuffer = (float*)malloc(AUDIO_BATCH * 2 * sizeof(float));

    GraphicsContext gfx;
    Render3DContext ctx3d;
    initGraphics(gfx, WINDOW_WIDTH, WINDOW_HEIGHT);
    init3DRendering(ctx3d, WINDOW_WIDTH, WINDOW_HEIGHT);

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = 48000;
    want.format = AUDIO_F32SYS;
    want.channels = 2;
    want.samples = 1024;
    SDL_AudioDeviceID simDev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
    if (!simDev) {
        fprintf(stderr, "Failed open sim audio:%s\n", SDL_GetError());
        exit(1);
    }
    SDL_PauseAudioDevice(simDev, 0);

    SDL_AudioSpec beepSpec;
    SDL_zero(beepSpec);
    beepSpec.freq = have.freq;
    beepSpec.format = AUDIO_F32SYS;
    beepSpec.channels = 2;
    beepSpec.samples = 1024;
    SDL_AudioDeviceID beepDev = SDL_OpenAudioDevice(NULL, 0, &beepSpec, &have, 0);
    if (!beepDev) {
        fprintf(stderr, "Failed beep audio:%s\n", SDL_GetError());
        exit(1);
    }
    SDL_PauseAudioDevice(beepDev, 0);

    g_sampleRate = have.freq;
    recalcDerivedParams(true);

    // We'll keep the old 2D block dims for the VBO update
    dim3 thr(32, 32);
    dim3 blk((g_width + 31) / 32, (g_height + 31) / 32);
    int threads1d = 1024;
    int blocks1d = (g_width * g_height + threads1d - 1) / threads1d;

    uint64_t perfFreq = SDL_GetPerformanceFrequency();
    uint64_t lastCount = SDL_GetPerformanceCounter();
    double simTime = 0.0;
    bool quit = false;
    int showObjects = 1;
    int audioBatchIndex = 0;

    const int beepSamples = (int)(0.1f * g_sampleRate);
    float* beepBuf = (float*)malloc(beepSamples * 2 * sizeof(float));
    double lastBeep = 0.0;
    double beepGap = 0.4;
    double lastSlideTime = 0.0;
    bool sliding = false;
    bool wasSliding = false;
    double lastDebugTime = 0.0;
    double lastSyncTime = 0.0;

    // We'll allocate a separate host dynamic mask each frame
    float* h_dynamicMask = (float*)malloc(g_width*g_height*sizeof(float));

    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED) {
                WINDOW_WIDTH  = e.window.data1; 
                WINDOW_HEIGHT = e.window.data2; 
                printf("[RESIZE] Window width updated to %d\n", WINDOW_WIDTH);
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_1: g_pulseIntervalSec -= 0.1f; recalcDerivedParams(true); break;
                    case SDLK_2: g_pulseIntervalSec += 0.1f; recalcDerivedParams(true); break;
                    case SDLK_3: {
                        float diff = 1.f - g_decayFactor;
                        diff *= 0.5f;
                        if (diff > 0.999999f) diff = 0.999999f;
                        g_decayFactor = 1.f - diff;
                        recalcDerivedParams(true);
                    } break;
                    case SDLK_4: {
                        float diff = 1.f - g_decayFactor;
                        diff *= 2.f;
                        if (diff > 0.999999f) diff = 0.999999f;
                        g_decayFactor = 1.f - diff;
                        recalcDerivedParams(true);
                    } break;
                    case SDLK_5: {
                        float diff = 1.f - g_reflection;
                        diff *= 0.5f;
                        if (diff > 0.9999f) diff = 0.9999f;
                        g_reflection = 1.f - diff;
                        recalcDerivedParams(true);
                    } break;
                    case SDLK_6: {
                        float diff = 1.f - g_reflection;
                        diff *= 2.f;
                        if (diff > 0.9999f) diff = 0.9999f;
                        g_reflection = 1.f - diff;
                        recalcDerivedParams(true);
                    } break;
                    case SDLK_7: g_timeFudgeFactor -= 0.001f; recalcDerivedParams(true); break;
                    case SDLK_8: g_timeFudgeFactor += 0.001f; recalcDerivedParams(true); break;
                    case SDLK_KP_MINUS: g_pulseFreq -= 1000.f; recalcDerivedParams(true); break;
                    case SDLK_KP_PLUS: g_pulseFreq += 1000.f; recalcDerivedParams(true); break;
                    case SDLK_KP_DIVIDE: g_pulseFreq /= 1.1f; recalcDerivedParams(true); break;
                    case SDLK_KP_MULTIPLY: g_pulseFreq *= 1.1f; recalcDerivedParams(true); break;
                    case SDLK_SPACE: showObjects = !showObjects; break;
                }
            }
        }

        uint64_t now = SDL_GetPerformanceCounter();
        float dtFrame = (float)((now - lastCount) / (double)perfFreq);
        lastCount = now;

        // Build the dynamic mask each frame by copying h_staticMask
        // and adding the wedge
        updateDynamicMask(h_staticMask, h_dynamicMask,
                          g_wedge_x, g_wedge_y, g_player.pivot_x, g_player.pivot_y);
        CUDA_CHECK(cudaMemcpy(d_mask, h_dynamicMask,
                              g_width*g_height*sizeof(float), cudaMemcpyHostToDevice));

        // Player movement
        const Uint8* ks = SDL_GetKeyboardState(NULL);
        Player newP = g_player;
        float moveSpeed = 50.f;
        float rotSpeed = (float)M_PI / 2.f;
        float strafeSpeed = moveSpeed;

        if (ks[SDL_SCANCODE_LEFT] || ks[SDL_SCANCODE_A]) newP.angle += rotSpeed * dtFrame;
        if (ks[SDL_SCANCODE_RIGHT] || ks[SDL_SCANCODE_D]) newP.angle -= rotSpeed * dtFrame;

        updatePlayerComponents(newP);

        float fx = g_pulse_x - newP.pivot_x;
        float fy = g_pulse_y - newP.pivot_y;
        float ln = sqrtf(fx * fx + fy * fy);
        if (ln > 1e-6f) { fx /= ln; fy /= ln; }
        float dx = 0.f, dy = 0.f;
        if (ks[SDL_SCANCODE_UP] || ks[SDL_SCANCODE_W]) {
            dx += moveSpeed * dtFrame * fx;
            dy += moveSpeed * dtFrame * fy;
        }
        if (ks[SDL_SCANCODE_DOWN] || ks[SDL_SCANCODE_S]) {
            dx -= moveSpeed * dtFrame * fx;
            dy -= moveSpeed * dtFrame * fy;
        }

        float sx_ = -fy;
        float sy_ = fx;
        if (ks[SDL_SCANCODE_Q] || ks[SDL_SCANCODE_RCTRL]) {
            dx += strafeSpeed * dtFrame * sx_;
            dy += strafeSpeed * dtFrame * sy_;
        }
        if (ks[SDL_SCANCODE_E] || ks[SDL_SCANCODE_KP_0]) {
            dx -= strafeSpeed * dtFrame * sx_;
            dy -= strafeSpeed * dtFrame * sy_;
        }

        float moveMag = sqrtf(dx * dx + dy * dy);
        float maxStep = moveSpeed * dtFrame;
        if (moveMag > maxStep) {
            dx = dx * maxStep / moveMag;
            dy = dy * maxStep / moveMag;
            moveMag = maxStep;
        }

        float collisionAngle = 0.f, normalX = 0.f, normalY = 0.f;
        newP.pivot_x += dx;
        newP.pivot_y += dy;
        bool collided = checkPlayerCollision(newP.pivot_x, newP.pivot_y,
                                             h_staticMask, collisionAngle,
                                             newP.angle, normalX, normalY);

        if (collided) {
            float dot = dx * normalX + dy * normalY;
            float projX = dot * normalX;
            float projY = dot * normalY;
            float slideX = dx - projX;
            float slideY = dy - projY;
            float slideMag = sqrtf(slideX * slideX + slideY * slideY);
            if (slideMag > maxStep) {
                slideX = slideX * maxStep / slideMag;
                slideY = slideY * maxStep / slideMag;
                slideMag = maxStep;
            }

            float newX = g_player.pivot_x + slideX;
            float newY = g_player.pivot_y + slideY;
            float tempAngle, tempNX, tempNY;

            if (!checkPlayerCollision(newX, newY, h_staticMask,
                                      tempAngle, newP.angle, tempNX, tempNY)) {
                newP.pivot_x = newX;
                newP.pivot_y = newY;
                collisionAngle = tempAngle;
                normalX = tempNX;
                normalY = tempNY;
                sliding = (slideMag > 0.01f);
            } else {
                float pushX = g_player.pivot_x;
                float pushY = g_player.pivot_y;
                float stepSize = 0.1f;
                int maxSteps = 50;
                float tolerance = 0.1f;
                for (int i = 0; i < maxSteps && checkPlayerCollision(pushX, pushY, h_staticMask,
                                                                     tempAngle, newP.angle,
                                                                     tempNX, tempNY); i++) {
                    pushX -= (normalX + tolerance * slideX) * stepSize;
                    pushY -= (normalY + tolerance * slideY) * stepSize;
                }
                newP.pivot_x = pushX;
                newP.pivot_y = pushY;
                sliding = false;
            }

            newP.pivot_x = g_player.pivot_x * 0.1f + newP.pivot_x * 0.9f;
            newP.pivot_y = g_player.pivot_y * 0.1f + newP.pivot_y * 0.9f;
        } else {
            sliding = false;
        }

        updatePlayerComponents(newP);
        g_player = newP;
        updatePlayerComponents(g_player);

        g_player_pivot_x = g_player.pivot_x;
        g_player_pivot_y = g_player.pivot_y;
        g_player_angle = g_player.angle;

        double nowT = simTime;
        if (!wasSliding && collided && nowT - lastBeep > beepGap) {
            float playerDirX = cosf(g_player.angle);
            float playerDirY = sinf(g_player.angle);
            float relativeAngle = atan2f(normalY, normalX) - atan2f(playerDirY, playerDirX);
            if (relativeAngle < -M_PI) relativeAngle += 2.0 * M_PI;
            if (relativeAngle > M_PI) relativeAngle -= 2.0 * M_PI;
            float pan = sinf(relativeAngle);
            float lv = 0.5f - 0.5f * pan;
            float rv = 0.5f + 0.5f * pan;
            for (int i = 0; i < beepSamples; i++) {
                float t = (float)i / (float)g_sampleRate;
                float s = sinf(2.f * 440.f * t) * 0.2f;
                beepBuf[2 * i] = s * lv;
                beepBuf[2 * i + 1] = s * rv;
            }
            SDL_QueueAudio(beepDev, beepBuf, beepSamples * 2 * sizeof(float));
            lastBeep = nowT;
        }

        if (sliding && nowT - lastSlideTime > 0.1) {
            float pan = sinf(collisionAngle);
            float lv = 0.5f - 0.5f * pan;
            float rv = 0.5f + 0.5f * pan;
            int slideSamples = beepSamples;
            float* slideBuf = (float*)malloc(slideSamples * 2 * sizeof(float));
            for (int i = 0; i < slideSamples; i++) {
                float t = (float)i / (float)g_sampleRate;
                float s = sinf(2.f * 330.f * t) * 0.05f;
                slideBuf[2 * i] = s * lv;
                slideBuf[2 * i + 1] = s * rv;
            }
            SDL_QueueAudio(beepDev, slideBuf, slideSamples * 2 * sizeof(float));
            free(slideBuf);
            lastSlideTime = nowT;
        }

        for (int i = 0; i < g_numToneMarkers; i++) {
            ToneMarker tm = g_toneMarkers[i];
            float dx_ = tm.x - g_player.pivot_x;
            float dy_ = tm.y - g_player.pivot_y;
            float dist = sqrtf(dx_ * dx_ + dy_ * dy_);
            if (dist < 50.f) {
                if (!tm.isPlaying) {
                    tm.isPlaying = true;
                    printf("[TONE DEBUG] Started playing marker %c at (%.1f, %.1f), Dist=%.1f\n",
                           tm.type, tm.x, tm.y, dist);
                    g_toneMarkers[i].isPlaying = true;
                }
                float angle = atan2f(dy_, dx_) - g_player.angle;
                float pan = sinf(angle);
                float atten = fmaxf(0.2f, 1.f - dist / 50.f);
                float freq = (tm.type == 'O') ? 440.f : (tm.type == 'W') ? 660.f : 880.f;
                int samplesToPlay = (int)(dtFrame * g_sampleRate);
                float* markerBuf = (float*)malloc(samplesToPlay * 2 * sizeof(float));
                for (int j = 0; j < samplesToPlay; j++) {
                    float t = (float)j / (float)g_sampleRate;
                    float sample = sinf(2.f * freq * t) * 0.2f * atten;
                    markerBuf[2 * j] = sample * (0.5f - 0.5f * pan);
                    markerBuf[2 * j + 1] = sample * (0.5f + 0.5f * pan);
                }
                SDL_QueueAudio(beepDev, markerBuf, samplesToPlay * 2 * sizeof(float));
                free(markerBuf);
            } else {
                if (tm.isPlaying) {
                    tm.isPlaying = false;
                    printf("[TONE DEBUG] Stopped marker %c (Dist=%.1f)\n", tm.type, dist);
                    g_toneMarkers[i].isPlaying = false;
                }
            }
        }

        wasSliding = sliding;

        int rawSteps = (int)ceilf(dtFrame / g_DT);
        int steps = (int)(rawSteps * g_timeFudgeFactor);
        if (steps < 1) steps = 1;
        if (steps > 1000) steps = 1000;

        const float targetQueuedSec = 0.03f;
        Uint32 queued = SDL_GetQueuedAudioSize(simDev);
        float queuedSec = (float)queued / (sizeof(float) * 2 * (float)g_sampleRate);
        if (queuedSec > 0.06f) {
            SDL_ClearQueuedAudio(simDev);
            printf("[SYNC] Cleared queue to avoid lag (%.3f s)\n", queuedSec);
        } else if (simTime - lastSyncTime >= 1.0) {
            if (queuedSec < 0.02f && g_timeFudgeFactor < 1.0f) {
                g_timeFudgeFactor += 0.001f;
                recalcDerivedParams();
                printf("[SYNC] Increased fudge factor to %.4f (QueuedSec=%.3f)\n", g_timeFudgeFactor, queuedSec);
                lastSyncTime = simTime;
            } else if (queuedSec > 0.04f && g_timeFudgeFactor > 0.9f) {
                g_timeFudgeFactor -= 0.001f;
                recalcDerivedParams();
                printf("[SYNC] Decreased fudge factor to %.4f (QueuedSec=%.3f)\n", g_timeFudgeFactor, queuedSec);
                lastSyncTime = simTime;
            }
        }

        static unsigned long simStep2 = 0;
        for (int i = 0; i < steps; i++) {
            // NEW: Launch updatePressureTiled with the list of active blocks
            dim3 tileThreads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            if (g_activeBlockCount > 0) {
                updatePressureTiled<<<g_activeBlockCount, tileThreads>>>(
                    d_p_next, d_p, d_p_prev, d_mask,
                    g_c2, g_width, g_height, g_reflection,
                    d_activeBlockList, g_numBlocksX
                );
                CUDA_CHECK(cudaGetLastError());
            }

            addPulse<<<1, 1>>>(d_p_next, g_pulse_x, g_pulse_y, 500.0f,
                               simStep2, g_pulseIntervalSteps, g_pulseDurationSteps,
                               g_DT, g_width, g_height, g_pulseFreq);
            CUDA_CHECK(cudaGetLastError());

            applyDecay<<<blocks1d, threads1d>>>(d_p_next, g_decayFactor, g_width, g_height);
            CUDA_CHECK(cudaGetLastError());

            captureAndReduce<<<1, 32>>>(d_p_next, g_mic_l_x, g_mic_l_y, g_mic_r_x, g_mic_r_y,
                                        audioBatchIndex, d_audioL, d_audioR,
                                        g_waveVolumeScale, g_width, g_height);
            CUDA_CHECK(cudaGetLastError());

            audioBatchIndex++;
            simStep2++;
            simTime += g_DT;

            float* tmp = d_p_prev;
            d_p_prev = d_p;
            d_p = d_p_next;
            d_p_next = tmp;

            if (audioBatchIndex >= AUDIO_BATCH) break;
        }

        if (audioBatchIndex > 0) {
            float* hostL = (float*)malloc(audioBatchIndex * sizeof(float));
            float* hostR = (float*)malloc(audioBatchIndex * sizeof(float));
            CUDA_CHECK(cudaMemcpy(hostL, d_audioL, audioBatchIndex * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostR, d_audioR, audioBatchIndex * sizeof(float), cudaMemcpyDeviceToHost));

            for (int k = 0; k < audioBatchIndex; k++) {
                h_audioBuffer[2 * k] = hostL[k];
                h_audioBuffer[2 * k + 1] = hostR[k];
            }
            SDL_QueueAudio(simDev, h_audioBuffer, audioBatchIndex * 2 * sizeof(float));
            CUDA_CHECK(cudaMemcpy(d_toneMarkers, g_toneMarkers.data(), 
                                  g_numToneMarkers * sizeof(ToneMarker), 
                                  cudaMemcpyHostToDevice));
            free(hostL);
            free(hostR);
            audioBatchIndex = 0;

            queued = SDL_GetQueuedAudioSize(simDev);
            queuedSec = (float)queued / (sizeof(float) * 2 * (float)g_sampleRate);

            if (simTime - lastDebugTime >= 5.0) {
                printf("[DEBUG] SimTime=%.2f  Steps=%d  QueuedSec=%.3f\n", simTime, steps, queuedSec);
                lastDebugTime = simTime;
            }
        }

        float* vbo_data; size_t vbo_size;
        CUDA_CHECK(cudaGraphicsMapResources(1, &gfx.cuda_vbo_resource, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vbo_data, &vbo_size, gfx.cuda_vbo_resource));

        updateVBO<<<blk, thr>>>(vbo_data, d_p, d_mask, g_width, g_height, showObjects, simStep2,
                                g_mic_l_x, g_mic_l_y, g_mic_r_x, g_mic_r_y,
                                g_pulse_x, g_pulse_y, g_wedge_x, g_wedge_y,
                                g_player.pivot_x, g_player.pivot_y, g_player.angle,
                                d_toneMarkers, g_numToneMarkers);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &gfx.cuda_vbo_resource, 0));

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render 3D view in top half
        glViewport(0, g_height, WINDOW_WIDTH, WINDOW_HEIGHT - g_height); // Top ~500px
        render3DView(ctx3d, WINDOW_WIDTH, WINDOW_HEIGHT-g_height);

        // Render 2D views in bottom half
        glViewport(0, 0, g_width*2, g_height);
        glUseProgram(gfx.shader);
        glBindVertexArray(gfx.vao);
        glDrawArrays(GL_POINTS, 0, g_width * g_height * 2);

        SDL_GL_SwapWindow(gfx.window);
    }

    SDL_CloseAudioDevice(beepDev);
    SDL_CloseAudioDevice(simDev);

    if (d_toneMarkers) CUDA_CHECK(cudaFree(d_toneMarkers));
    CUDA_CHECK(cudaGraphicsUnregisterResource(gfx.cuda_vbo_resource));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_prev));
    CUDA_CHECK(cudaFree(d_p_next));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_audioL));
    CUDA_CHECK(cudaFree(d_audioR));
    if (d_activeBlockList) CUDA_CHECK(cudaFree(d_activeBlockList));

    free(h_staticMask);
    free(h_dynamicMask);
    free(h_audioBuffer);
    free(beepBuf);

    cleanupGraphics(gfx);
    cleanup3DRendering(ctx3d);
    return 0;
}

