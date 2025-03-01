// echolocation_game_dynamic_params_lag_fix_svg.cu
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <tinyxml2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>

#ifndef PI
#define PI 3.14159265358979323846f
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
static float g_pulseIntervalSec= 1.0f;
static float g_decayFactor     = 0.9995f;
static float g_reflectionCoeff = 0.9f;
static float g_timeFudgeFactor = 0.985f;

static int   g_sampleRate      = 48000;
static float g_c2;
static int   g_pulseIntervalSteps;
static int   g_pulseDurationSteps;

static float g_waveVolumeScale = 0.05f;
static float g_toneVolumeScale = 1.0f;

static const float SPEED_OF_SOUND = 343.0f;
static const int   SIZE = 500;
static const int   WINDOW_WIDTH = SIZE * 2;
static const int   WINDOW_HEIGHT = SIZE;
static const int   PULSE_DELAY = 1000;
static const float PULSE_AMPLITUDE = 750.0f;
static const int   AUDIO_BATCH = 4096;
__constant__ float ZOOM_FACTOR = 4.0f;

struct Rect { float x, y, width, height; };
struct Circle { float cx, cy, r; };
struct ToneMarker { 
    float x, y; 
    char type; // 'O' for obstacle, 'W' for win, 'I' for item
    int sampleOffset; // Tracks WAV playback progress
    float* audioData; // Host-side WAV data
    int audioLen;     // Length of WAV data
    bool isPlaying;   // Playback state
};

static std::vector<Rect> g_rects;
static std::vector<Circle> g_circles;
static std::vector<ToneMarker> g_toneMarkers;
static ToneMarker* d_toneMarkers = nullptr;
static int g_numToneMarkers = 0;

static float* h_audioO = nullptr;
static float* h_audioW = nullptr;
static float* h_audioI = nullptr;
static int audioLenO = 0;
static int audioLenW = 0;
static int audioLenI = 0;

static void recalcDerivedParams()
{
    g_DT = g_DX / (SPEED_OF_SOUND * 1.41421356237f);
    g_c2 = (SPEED_OF_SOUND * g_DT / g_DX) * (SPEED_OF_SOUND * g_DT / g_DX);

    if (g_pulseIntervalSec < 0.1f) g_pulseIntervalSec = 0.1f;
    if (g_pulseIntervalSec > 5.f)  g_pulseIntervalSec = 5.f;
    g_pulseIntervalSteps = (int)(g_pulseIntervalSec / g_DT);

    g_pulseDurationSteps = (int)(0.0005f / g_DT);
    if (g_pulseDurationSteps < 1) g_pulseDurationSteps = 1;

    if (g_reflectionCoeff < 0.f) g_reflectionCoeff = 0.f;
    if (g_reflectionCoeff > 1.f) g_reflectionCoeff = 1.f;

    if (g_decayFactor < 0.f) g_decayFactor = 0.f;
    if (g_decayFactor > 1.f) g_decayFactor = 1.f;

    if (g_timeFudgeFactor < 0.9f) g_timeFudgeFactor = 0.9f;
    if (g_timeFudgeFactor > 1.0f) g_timeFudgeFactor = 1.0f;

    printf("[PARAMS] Pulse=%.2fs  Decay=%.6f  Reflect=%.3f  Fudge=%.4f\n",
           g_pulseIntervalSec, g_decayFactor, g_reflectionCoeff, g_timeFudgeFactor);
}

static void initAudio()
{
    SDL_AudioSpec wavSpec;
    Uint8* wavBuffer;
    Uint32 wavLength;

    // Load obstacle.wav
    if (!SDL_LoadWAV("obstacle.wav", &wavSpec, &wavBuffer, &wavLength)) {
        fprintf(stderr, "Failed to load obstacle.wav: %s\n", SDL_GetError());
        exit(1);
    }
    if (wavSpec.format != AUDIO_F32SYS) {
        audioLenO = wavLength / sizeof(int16_t);
        h_audioO = (float*)malloc(audioLenO * sizeof(float));
        int16_t* src = (int16_t*)wavBuffer;
        for (int i = 0; i < audioLenO; i++) {
            h_audioO[i] = src[i] / 32768.0f;
        }
        SDL_FreeWAV(wavBuffer);
    } else {
        audioLenO = wavLength / sizeof(float);
        h_audioO = (float*)malloc(audioLenO * sizeof(float));
        memcpy(h_audioO, wavBuffer, audioLenO * sizeof(float));
        SDL_FreeWAV(wavBuffer);
    }

    // Load win.wav
    if (!SDL_LoadWAV("win.wav", &wavSpec, &wavBuffer, &wavLength)) {
        fprintf(stderr, "Failed to load win.wav: %s\n", SDL_GetError());
        exit(1);
    }
    if (wavSpec.format != AUDIO_F32SYS) {
        audioLenW = wavLength / sizeof(int16_t);
        h_audioW = (float*)malloc(audioLenW * sizeof(float));
        int16_t* src = (int16_t*)wavBuffer;
        for (int i = 0; i < audioLenW; i++) {
            h_audioW[i] = src[i] / 32768.0f;
        }
        SDL_FreeWAV(wavBuffer);
    } else {
        audioLenW = wavLength / sizeof(float);
        h_audioW = (float*)malloc(audioLenW * sizeof(float));
        memcpy(h_audioW, wavBuffer, audioLenW * sizeof(float));
        SDL_FreeWAV(wavBuffer);
    }

    // Load powerup.wav
    if (!SDL_LoadWAV("powerup.wav", &wavSpec, &wavBuffer, &wavLength)) {
        fprintf(stderr, "Failed to load powerup.wav: %s\n", SDL_GetError());
        exit(1);
    }
    if (wavSpec.format != AUDIO_F32SYS) {
        audioLenI = wavLength / sizeof(int16_t);
        h_audioI = (float*)malloc(audioLenI * sizeof(float));
        int16_t* src = (int16_t*)wavBuffer;
        for (int i = 0; i < audioLenI; i++) {
            h_audioI[i] = src[i] / 32768.0f;
        }
        SDL_FreeWAV(wavBuffer);
    } else {
        audioLenI = wavLength / sizeof(float);
        h_audioI = (float*)malloc(audioLenI * sizeof(float));
        memcpy(h_audioI, wavBuffer, audioLenI * sizeof(float));
        SDL_FreeWAV(wavBuffer);
    }

    printf("[AUDIO] Loaded WAV files: O=%d samples, W=%d samples, I=%d samples\n", audioLenO, audioLenW, audioLenI);
}

struct Player {
    float pivot_x;
    float pivot_y;
    float angle;
};

static Player g_player = {50.f, 150.f, -PI / 4.f};
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
            
            float y_top = SIZE - (r.y + r.height);
            float y_bottom = SIZE - r.y;
            
            int x0 = (int)floorf(r.x);
            int y0 = (int)floorf(y_top);
            int x1 = (int)ceilf(r.x + r.width);
            int y1 = (int)ceilf(y_bottom);

            printf("[SVG] Rect: x=%.1f, y=%.1f, w=%.1f, h=%.1f (grid: %d,%d to %d,%d)\n",
                   r.x, r.y, r.width, r.height, x0, y0, x1, y1);

            for (int y = y0; y <= y1; y++) {
                for (int x = x0; x <= x1; x++) {
                    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
                        h_mask[y * SIZE + x] = 1.f;
                    }
                }
            }
            r.y = y_top;
            g_rects.push_back(r);
        }
        else if (strcmp(name, "path") == 0) {
            const char* type = elem->Attribute("sodipodi:type");
            if (type && (strcmp(type, "arc") == 0 || strcmp(type, "circle") == 0)) {
                Circle c;
                elem->QueryFloatAttribute("sodipodi:cx", &c.cx);
                elem->QueryFloatAttribute("sodipodi:cy", &c.cy);
                elem->QueryFloatAttribute("sodipodi:rx", &c.r);
                
                c.cy = SIZE - c.cy;
                
                int cx = (int)roundf(c.cx);
                int cy = (int)roundf(c.cy);
                int r = (int)ceilf(c.r);
                
                printf("[SVG] Circle: cx=%.1f, cy=%.1f, r=%.1f\n", c.cx, c.cy, c.r);

                for (int y = cy - r; y <= cy + r; y++) {
                    for (int x = cx - r; x <= cx + r; x++) {
                        float dx = x - c.cx;
                        float dy = y - c.cy;
                        if (dx * dx + dy * dy <= c.r * c.r) {
                            if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
                                h_mask[y * SIZE + x] = 1.f;
                            }
                        }
                    }
                }
                g_circles.push_back(c);
            }
        }
        else if (strcmp(name, "text") == 0) {
            XMLElement* tspan = elem->FirstChildElement("tspan");
            const char* text = tspan ? tspan->GetText() : elem->GetText();
            if (text && (strstr(text, "O") || strstr(text, "W") || strstr(text, "I"))) {
                ToneMarker tm;
                elem->QueryFloatAttribute("x", &tm.x);
                elem->QueryFloatAttribute("y", &tm.y);
                tm.y = SIZE - tm.y;
                if (strstr(text, "O")) {
                    tm.type = 'O';
                    tm.audioData = h_audioO;
                    tm.audioLen = audioLenO;
                } else if (strstr(text, "W")) {
                    tm.type = 'W';
                    tm.audioData = h_audioW;
                    tm.audioLen = audioLenW;
                } else if (strstr(text, "I")) {
                    tm.type = 'I';
                    tm.audioData = h_audioI;
                    tm.audioLen = audioLenI;
                }
                tm.sampleOffset = 0;
                tm.isPlaying = false;
                printf("[SVG] Tone Marker (%c): x=%.1f, y=%.1f\n", tm.type, tm.x, tm.y);
                g_toneMarkers.push_back(tm);
            }
        }

        parseSVGElement(elem->FirstChildElement(), h_mask);
        elem = elem->NextSiblingElement();
    }
}

static bool loadSVG(const char* filename, float* h_mask)
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

    memset(h_mask, 0, SIZE * SIZE * sizeof(float));
    
    int w = 2;
    for (int y = 0; y < SIZE; y++) {
        for (int x = 0; x < SIZE; x++) {
            if (x < w || x >= SIZE - w || y < w || y >= SIZE - w) {
                h_mask[y * SIZE + x] = 1.f;
            }
        }
    }
    printf("[SVG] Added boundaries\n");

    parseSVGElement(svg->FirstChildElement(), h_mask);

    printf("[SVG] Loaded %d rects, %d circles, %d tone markers\n", 
           (int)g_rects.size(), (int)g_circles.size(), (int)g_toneMarkers.size());
    return true;
}

static void updateDynamicMask(const float* h_static, float* h_dynamic, float wedge_x, float wedge_y, float pivot_x, float pivot_y) {
    memcpy(h_dynamic, h_static, SIZE * SIZE * sizeof(float));
    float fwd_angle = atan2f(g_pulse_y - pivot_y, g_pulse_x - pivot_x);
    float wedge_half = 38.0f * PI / 180.0f;
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

        if (ix1 >= 0 && ix1 < SIZE && iy1 >= 0 && iy1 < SIZE)
            h_dynamic[iy1 * SIZE + ix1] = 1.0f;
        if (ix2 >= 0 && ix2 < SIZE && iy2 >= 0 && iy2 < SIZE)
            h_dynamic[iy2 * SIZE + ix2] = 1.0f;
    }
}

__global__ void updatePressure(float* p_next, const float* p, const float* p_prev,
                               const float* mask, float c2, int size, float reflection)
{
    __shared__ float s_p[34][34];
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
    int x = blockIdx.x * 32 + (tx - 1);
    int y = blockIdx.y * 32 + (ty - 1);
    if (x >= size || y >= size) return;
    int idx = y * size + x;
    s_p[ty][tx] = p[idx];

    if (tx == 1  && x > 0      && y < size) s_p[ty][0] = p[idx - 1];
    if (tx == 32 && x < size - 1 && y < size) s_p[ty][33] = p[idx + 1];
    if (ty == 1  && y > 0      && x < size) s_p[0][tx] = p[idx - size];
    if (ty == 32 && y < size - 1 && x < size) s_p[33][tx] = p[idx + size];
    __syncthreads();

    if (x < 1 || x >= size - 1 || y < 1 || y >= size - 1) return;

    if (mask[idx] > 0.f) {
        p_next[idx] = -reflection * s_p[ty][tx];
        return;
    }
    float lap = s_p[ty][tx + 1] + s_p[ty][tx - 1] +
                s_p[ty + 1][tx] + s_p[ty - 1][tx] -
                4.f * s_p[ty][tx];
    p_next[idx] = 2.f * s_p[ty][tx] - p_prev[idx] + c2 * lap;
}

__global__ void applyDecay(float* p, float decay)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE * SIZE) {
        p[idx] *= decay;
    }
}

__global__ void addPulse(float* p, float sx, float sy, float amplitude,
                         int t, int interval, int duration, float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        int adj = t - 1000;
        if (adj >= 0 && (adj % interval) < duration) {
            float val = amplitude * sinf(2.f * PI * 500.f * ((adj % duration)) * dt);
            int ix = (int)roundf(sx), iy = (int)roundf(sy);
            if (ix >= 0 && ix < SIZE && iy >= 0 && iy < SIZE) {
                atomicAdd(&p[iy * SIZE + ix], val);
            }
        }
    }
}

__global__ void captureAndReduce(const float* p, float lx, float ly,
                                 float rx, float ry, int sampleIndex,
                                 float* audioL, float* audioR, float waveVolume)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int ilx = (int)roundf(lx), ily = (int)roundf(ly);
        int irx = (int)roundf(rx), iry = (int)roundf(ry);
        if (ilx < 0) ilx = 0; if (ilx >= SIZE) ilx = SIZE - 1;
        if (ily < 0) ily = 0; if (ily >= SIZE) ily = SIZE - 1;
        if (irx < 0) irx = 0; if (irx >= SIZE) irx = SIZE - 1;
        if (iry < 0) iry = 0; if (iry >= SIZE) iry = SIZE - 1;

        float vl = p[ily * SIZE + ilx];
        float vr = p[iry * SIZE + irx];
        audioL[sampleIndex] = vl * waveVolume;
        audioR[sampleIndex] = vr * waveVolume;
    }
}

__global__ void updateVBO(float* vbo_data, float* p, float* mask, int size, int showObj, int t,
                          float mlx, float mly, float mrx, float mry,
                          float px, float py, float wx, float wy,
                          float player_x, float player_y, float player_angle,
                          const ToneMarker* toneMarkers, int numToneMarkers)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size) return;
    int idx = y * size + x;
    int vbo_idx = idx * 3;
    int fp_idx = (SIZE * SIZE + idx) * 3;

    float pxOut_static = 2.f * x / size - 1.f;
    float pyOut_static = 2.f * y / size - 1.f;
    vbo_data[vbo_idx] = pxOut_static * 0.5f - 0.5f;
    vbo_data[vbo_idx + 1] = pyOut_static;

    float dx = x - player_x;
    float dy = y - player_y;
    float c = cosf(-player_angle - PI / 4.f + PI), s = sinf(-player_angle - PI / 4.f + PI);
    float rx = dx * c - dy * s;
    float ry = dx * s + dy * c;
    float pxOut_fp = (rx * ZOOM_FACTOR) / (size / 2.0f);
    float pyOut_fp = (ry * ZOOM_FACTOR) / (size / 2.0f);
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

static bool checkPlayerCollision(float px, float py, const float* h_mask, float& collisionAngle, float playerAngle)
{
    int base = (int)ceilf(sqrtf(6.f * 6.f + 5.5f * 5.5f));
    int radius = base + 10 - 5;
    int ixmin = (int)fmaxf(0, floorf(px - radius));
    int ixmax = (int)fminf(SIZE - 1, ceilf(px + radius));
    int iymin = (int)fmaxf(0, floorf(py - radius));
    int iymax = (int)fminf(SIZE - 1, ceilf(py + radius));
    int r2 = radius * radius;

    bool collided = false;
    const int nS = 360, nR = 5;
    float best = -1.f, bestAngle = 0.f;

    for (int i = 0; i < nS; i++) {
        float a = -PI + i * (2.f * PI / nS);
        float sc = 0.f;
        for (int j = 0; j < nR; j++) {
            float rr = (radius - 5) + j * (5.f / (nR - 1));
            float gx = px + rr * cosf(a);
            float gy = py + rr * sinf(a);
            int ix = (int)roundf(gx), iy = (int)roundf(gy);
            if (ix >= 0 && ix < SIZE && iy >= 0 && iy < SIZE) {
                if (h_mask[iy * SIZE + ix] > 0.f) {
                    sc += 1.f;
                }
            }
        }
        if (sc > best) {
            best = sc;
            bestAngle = a;
        }
    }

    collisionAngle = bestAngle - playerAngle + PI / 2.f;
    if (collisionAngle < -PI) collisionAngle += 2.f * PI;
    if (collisionAngle > PI) collisionAngle -= 2.f * PI;

    for (int y = iymin; y <= iymax; y++) {
        for (int x = ixmin; x <= ixmax; x++) {
            float dx = x - px, dy = y - py;
            if (dx * dx + dy * dy <= r2 && h_mask[y * SIZE + x] > 0.f) {
                collided = true;
                break;
            }
        }
    }

    return collided;
}

struct GraphicsContext {
    SDL_Window* window;
    SDL_GLContext gl_context;
    GLuint vbo, vao, shader;
    cudaGraphicsResource* cuda_vbo_resource;
};

static const char* vertexShaderSource = R"(
#version 330 core
layout(location=0) in vec2 pos;
layout(location=1) in float pressure;
out vec3 color;
void main(){
    gl_Position = vec4(pos, 0, 1);
    if (pressure < -9.0)      color = vec3(0, 0, 0);       // Black (invisible)
    else if (pressure < -6.5) color = vec3(1, 1, 0);       // Gold for 'I'
    else if (pressure < -5.5) color = vec3(0, 1, 1);       // Cyan for 'W'
    else if (pressure < -4.5) color = vec3(1, 0, 1);       // Purple for 'O'
    else if (pressure < -3.5) color = vec3(0.5, 0, 0.5);   // Dark purple (unused)
    else if (pressure < -2.5) color = vec3(1, 0, 0);       // Red (pulse)
    else if (pressure < -1.5) color = vec3(0, 1, 0);       // Green (mic)
    else if (pressure < -0.5) color = vec3(1, 1, 0);       // Yellow (objects)
    else {
        float r = (pressure > 0 ? pressure : 0);
        float b = (pressure < 0 ? -pressure : 0);
        color = vec3(r, 0, b);
    }
}
)";

static const char* fragmentShaderSource = R"(
#version 330 core
in vec3 color;
out vec4 fragColor;
void main(){
    fragColor = vec4(color, 1);
}
)";

static void checkGLError(const char* where) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) {
        fprintf(stderr, "GL error at %s: %d\n", where, (int)e);
        exit(1);
    }
}

static void initGraphics(GraphicsContext &gfx)
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) < 0) {
        fprintf(stderr, "SDL_Init error:%s\n", SDL_GetError());
        exit(1);
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    gfx.window = SDL_CreateWindow("Echolocation, Dual View (SVG)",
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_OPENGL);
    if (!gfx.window) {
        fprintf(stderr, "SDL_CreateWindow error:%s\n", SDL_GetError());
        exit(1);
    }
    gfx.gl_context = SDL_GL_CreateContext(gfx.window);
    if (!gfx.gl_context) {
        fprintf(stderr, "SDL_GL_CreateContext error:%s\n", SDL_GetError());
        exit(1);
    }
    glewExperimental = GL_TRUE;
    glewInit();
    glClearColor(0, 0, 0, 1);

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSource, NULL);
    glCompileShader(vs); checkGLError("VertexCompile");

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSource, NULL);
    glCompileShader(fs); checkGLError("FragmentCompile");

    gfx.shader = glCreateProgram();
    glAttachShader(gfx.shader, vs);
    glAttachShader(gfx.shader, fs);
    glLinkProgram(gfx.shader);
    checkGLError("ShaderLink");
    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenBuffers(1, &gfx.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, gfx.vbo);
    glBufferData(GL_ARRAY_BUFFER, SIZE * SIZE * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&gfx.cuda_vbo_resource, gfx.vbo,
                                            cudaGraphicsMapFlagsWriteDiscard));

    glGenVertexArrays(1, &gfx.vao);
    glBindVertexArray(gfx.vao);
    glBindBuffer(GL_ARRAY_BUFFER, gfx.vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
}

int main(int argc, char** argv)
{
    const char* svgFile = (argc >= 2) ? argv[1] : "drawing.svg";
    float* h_mask = (float*)calloc(SIZE * SIZE, sizeof(float));
    bool svgLoaded = false;

    if (!loadSVG(svgFile, h_mask)) {
        fprintf(stderr, "Could not load %s, proceeding with empty room\n", svgFile);
        memset(h_mask, 0, SIZE * SIZE * sizeof(float));
        int w = 2;
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                if (x < w || x >= SIZE - w || y < w || y >= SIZE - w) {
                    h_mask[y * SIZE + x] = 1.f;
                }
            }
        }
        printf("[DEFAULT] Running with empty room and boundaries\n");
    } else {
        svgLoaded = true;
    }

    g_numToneMarkers = g_toneMarkers.size();
    if (g_numToneMarkers > 0) {
        CUDA_CHECK(cudaMalloc(&d_toneMarkers, g_numToneMarkers * sizeof(ToneMarker)));
        CUDA_CHECK(cudaMemcpy(d_toneMarkers, g_toneMarkers.data(), 
                            g_numToneMarkers * sizeof(ToneMarker), 
                            cudaMemcpyHostToDevice));
    }

    float *d_mask;
    CUDA_CHECK(cudaMalloc(&d_mask, SIZE * SIZE * sizeof(float)));

    float *d_p, *d_p_prev, *d_p_next;
    CUDA_CHECK(cudaMalloc(&d_p, SIZE * SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_prev, SIZE * SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p_next, SIZE * SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p, 0, SIZE * SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p_prev, 0, SIZE * SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_p_next, 0, SIZE * SIZE * sizeof(float)));

    float *d_audioL, *d_audioR;
    CUDA_CHECK(cudaMalloc(&d_audioL, AUDIO_BATCH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_audioR, AUDIO_BATCH * sizeof(float)));
    float* h_audioBuffer = (float*)malloc(AUDIO_BATCH * 2 * sizeof(float));

    GraphicsContext gfx;
    initGraphics(gfx);
    initAudio();

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

    SDL_AudioSpec markerSpec;
    SDL_zero(markerSpec);
    markerSpec.freq = have.freq;
    markerSpec.format = AUDIO_F32SYS;
    markerSpec.channels = 2;
    markerSpec.samples = 1024;
    SDL_AudioDeviceID markerDev = SDL_OpenAudioDevice(NULL, 0, &markerSpec, &have, 0);
    if (!markerDev) {
        fprintf(stderr, "Failed marker audio:%s\n", SDL_GetError());
        exit(1);
    }
    SDL_PauseAudioDevice(markerDev, 0);

    g_sampleRate = have.freq;
    recalcDerivedParams();

    dim3 thr(32, 32);
    dim3 blk((SIZE + 31) / 32, (SIZE + 31) / 32);
    int threads1d = 1024;
    int blocks1d = (SIZE * SIZE + threads1d - 1) / threads1d;

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

    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_1: g_pulseIntervalSec -= 0.1f; recalcDerivedParams(); break;
                    case SDLK_2: g_pulseIntervalSec += 0.1f; recalcDerivedParams(); break;
                    case SDLK_3: {
                        float diff = 1.f - g_decayFactor;
                        diff *= 0.5f;
                        if (diff > 0.999999f) diff = 0.999999f;
                        g_decayFactor = 1.f - diff;
                        recalcDerivedParams();
                    } break;
                    case SDLK_4: {
                        float diff = 1.f - g_decayFactor;
                        diff *= 2.f;
                        if (diff > 0.999999f) diff = 0.999999f;
                        g_decayFactor = 1.f - diff;
                        recalcDerivedParams();
                    } break;
                    case SDLK_5: {
                        float diff = 1.f - g_reflectionCoeff;
                        diff *= 0.5f;
                        if (diff > 0.9999f) diff = 0.9999f;
                        g_reflectionCoeff = 1.f - diff;
                        recalcDerivedParams();
                    } break;
                    case SDLK_6: {
                        float diff = 1.f - g_reflectionCoeff;
                        diff *= 2.f;
                        if (diff > 0.9999f) diff = 0.9999f;
                        g_reflectionCoeff = 1.f - diff;
                        recalcDerivedParams();
                    } break;
                    case SDLK_7: g_timeFudgeFactor -= 0.001f; recalcDerivedParams(); break;
                    case SDLK_8: g_timeFudgeFactor += 0.001f; recalcDerivedParams(); break;
                    case SDLK_SPACE: showObjects = !showObjects; break;
                }
            }
        }

        uint64_t now = SDL_GetPerformanceCounter();
        float dtFrame = (float)((now - lastCount) / (double)perfFreq);
        lastCount = now;

        float* h_dynamic = (float*)malloc(SIZE * SIZE * sizeof(float));
        updateDynamicMask(h_mask, h_dynamic, g_wedge_x, g_wedge_y, g_player.pivot_x, g_player.pivot_y);
        CUDA_CHECK(cudaMemcpy(d_mask, h_dynamic, SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice));
        free(h_dynamic);

        const Uint8* ks = SDL_GetKeyboardState(NULL);
        Player newP = g_player;
        float moveSpeed = 50.f;
        float rotSpeed = PI / 2.f;
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

        float sx = -fy;
        float sy = fx;
        if (ks[SDL_SCANCODE_Q] || ks[SDL_SCANCODE_RCTRL]) {
            dx += strafeSpeed * dtFrame * sx;
            dy += strafeSpeed * dtFrame * sy;
        }
        if (ks[SDL_SCANCODE_E] || ks[SDL_SCANCODE_KP_0]) {
            dx -= strafeSpeed * dtFrame * sx;
            dy -= strafeSpeed * dtFrame * sy;
        }

        float moveMag = sqrtf(dx * dx + dy * dy);
        float maxStep = 13.5f;
        if (moveMag > maxStep) {
            dx = dx * maxStep / moveMag;
            dy = dy * maxStep / moveMag;
            moveMag = maxStep;
        }

        float collisionAngle = 0.f;
        newP.pivot_x += dx;
        newP.pivot_y += dy;
        bool collided = checkPlayerCollision(newP.pivot_x, newP.pivot_y, h_mask, collisionAngle, newP.angle);

        if (collided) {
            float normalAngle = collisionAngle + newP.angle - PI / 2.f;
            float normalX = cosf(normalAngle);
            float normalY = sinf(normalAngle);
            float dot = dx * normalX + dy * normalY;

            float tangentAngle = normalAngle + PI / 2.f;
            float tangentX = cosf(tangentAngle);
            float tangentY = sinf(tangentAngle);
            float slideDot = dx * tangentX + dy * tangentY;
            if (fabsf(slideDot) > maxStep) slideDot = (slideDot > 0 ? maxStep : -maxStep);

            float newX = g_player.pivot_x + slideDot * tangentX;
            float newY = g_player.pivot_y + slideDot * tangentY;
            float tempAngle;

            if (!checkPlayerCollision(newX, newY, h_mask, tempAngle, newP.angle)) {
                newP.pivot_x = newX;
                newP.pivot_y = newY;
                collisionAngle = tempAngle;
                sliding = (fabsf(slideDot) > 0.01f);
            } else {
                tangentAngle = normalAngle - PI / 2.f;
                tangentX = cosf(tangentAngle);
                tangentY = sinf(tangentAngle);
                slideDot = dx * tangentX + dy * tangentY;
                if (fabsf(slideDot) > maxStep) slideDot = (slideDot > 0 ? maxStep : -maxStep);

                newX = g_player.pivot_x + slideDot * tangentX;
                newY = g_player.pivot_y + slideDot * tangentY;

                if (!checkPlayerCollision(newX, newY, h_mask, tempAngle, newP.angle)) {
                    newP.pivot_x = newX;
                    newP.pivot_y = newY;
                    collisionAngle = tempAngle;
                    sliding = (fabsf(slideDot) > 0.01f);
                } else {
                    float pushX = g_player.pivot_x;
                    float pushY = g_player.pivot_y;
                    float stepSize = 1.0f;
                    int maxSteps = 10;
                    for (int i = 0; i < maxSteps && checkPlayerCollision(pushX, pushY, h_mask, tempAngle, newP.angle); i++) {
                        pushX -= normalX * stepSize;
                        pushY -= normalY * stepSize;
                    }
                    newP.pivot_x = pushX;
                    newP.pivot_y = pushY;
                    sliding = false;
                }
            }

            newP.pivot_x = g_player.pivot_x * 0.2f + newP.pivot_x * 0.8f;
            newP.pivot_y = g_player.pivot_y * 0.2f + newP.pivot_y * 0.8f;
        } else {
            sliding = false;
        }

        updatePlayerComponents(newP);
        g_player = newP;
        updatePlayerComponents(g_player);

        double nowT = simTime;
        if (!wasSliding && collided && nowT - lastBeep > beepGap) {
            float pan = sinf(collisionAngle);
            float lv = 0.5f + 0.5f * pan;
            float rv = 0.5f - 0.5f * pan;
            for (int i = 0; i < beepSamples; i++) {
                float t = (float)i / (float)g_sampleRate;
                float s = sinf(2.f * PI * 440.f * t) * 0.2f;
                beepBuf[2 * i] = s * lv;
                beepBuf[2 * i + 1] = s * rv;
            }
            SDL_QueueAudio(beepDev, beepBuf, beepSamples * 2 * sizeof(float));
            lastBeep = nowT;
        }

        if (sliding && nowT - lastSlideTime > 0.1) {
            float pan = sinf(collisionAngle);
            float lv = 0.5f + 0.5f * pan;
            float rv = 0.5f - 0.5f * pan;
            int slideSamples = beepSamples;
            float* slideBuf = (float*)malloc(slideSamples * 2 * sizeof(float));
            for (int i = 0; i < slideSamples; i++) {
                float t = (float)i / (float)g_sampleRate;
                float s = sinf(2.f * PI * 330.f * t) * 0.05f;
                slideBuf[2 * i] = s * lv;
                slideBuf[2 * i + 1] = s * rv;
            }
            SDL_QueueAudio(beepDev, slideBuf, slideSamples * 2 * sizeof(float));
            free(slideBuf);
            lastSlideTime = nowT;
        }

        // Handle marker audio playback
        for (int i = 0; i < g_numToneMarkers; i++) {
            ToneMarker& tm = g_toneMarkers[i];
            float dx = tm.x - g_player.pivot_x;
            float dy = tm.y - g_player.pivot_y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 50.f) {
                if (!tm.isPlaying) {
                    tm.isPlaying = true;
                    tm.sampleOffset = 0;
                    printf("[MARKER DEBUG] Started %c at (%.1f, %.1f), Player at (%.1f, %.1f), Dist=%.1f\n",
                           tm.type, tm.x, tm.y, g_player.pivot_x, g_player.pivot_y, dist);
                }
                if (tm.sampleOffset < tm.audioLen) {
                    float angle = atan2f(dy, dx) - g_player.angle;
                    float pan = sinf(angle);
                    float atten = fmaxf(0.2f, 1.f - dist / 50.f);
                    int samplesToPlay = (int)(dtFrame * g_sampleRate);
                    if (tm.sampleOffset + samplesToPlay > tm.audioLen) {
                        samplesToPlay = tm.audioLen - tm.sampleOffset;
                    }
                    float* markerBuf = (float*)malloc(samplesToPlay * 2 * sizeof(float));
                    for (int j = 0; j < samplesToPlay; j++) {
                        float sample = tm.audioData[tm.sampleOffset + j] * atten * g_toneVolumeScale;
                        markerBuf[2 * j] = sample * (0.5f + 0.5f * pan);
                        markerBuf[2 * j + 1] = sample * (0.5f - 0.5f * pan);
                    }
                    SDL_QueueAudio(markerDev, markerBuf, samplesToPlay * 2 * sizeof(float));
                    tm.sampleOffset += samplesToPlay;
                    free(markerBuf);
                }
                if (tm.sampleOffset >= tm.audioLen) {
                    tm.sampleOffset = 0; // Loop
                }
            } else {
                if (tm.isPlaying) {
                    tm.isPlaying = false;
                    printf("[MARKER DEBUG] Stopped %c at (%.1f, %.1f), Player at (%.1f, %.1f), Dist=%.1f\n",
                           tm.type, tm.x, tm.y, g_player.pivot_x, g_player.pivot_y, dist);
                }
            }
        }

        wasSliding = sliding;

        int rawSteps = (int)ceilf(dtFrame / g_DT);
        int steps = (int)(rawSteps * g_timeFudgeFactor);
        if (steps < 1) steps = 1;
        if (steps > 1000) steps = 1000;

        static unsigned long simStep2 = 0;
        for (int i = 0; i < steps; i++) {
            updatePressure<<<blk, thr>>>(d_p_next, d_p, d_p_prev, d_mask,
                                         g_c2, SIZE, g_reflectionCoeff);
            CUDA_CHECK(cudaGetLastError());

            addPulse<<<1, 1>>>(d_p_next, g_pulse_x, g_pulse_y, PULSE_AMPLITUDE,
                               simStep2, g_pulseIntervalSteps, g_pulseDurationSteps,
                               g_DT);
            CUDA_CHECK(cudaGetLastError());

            applyDecay<<<blocks1d, threads1d>>>(d_p_next, g_decayFactor);
            CUDA_CHECK(cudaGetLastError());

            captureAndReduce<<<1, 32>>>(d_p_next, g_mic_l_x, g_mic_l_y,
                                      g_mic_r_x, g_mic_r_y, audioBatchIndex,
                                      d_audioL, d_audioR, g_waveVolumeScale);
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

            const float BYTES_PER_FRAME = (float)(sizeof(float) * 2);
            Uint32 queued = SDL_GetQueuedAudioSize(simDev);
            float queuedSec = (float)queued / (BYTES_PER_FRAME * (float)g_sampleRate);

            if (queuedSec > 0.06f) {
                SDL_ClearQueuedAudio(simDev);
                printf("Queue reset to avoid lag. (%.2f s)\n", queuedSec);
            }

            if (simTime - lastDebugTime >= 5.0) {
                printf("[DEBUG] SimTime=%.2f  Steps=%d  QueuedSec=%.3f\n", simTime, steps, queuedSec);
                lastDebugTime = simTime;
            }
        }

        float* vbo_data; size_t vbo_size;
        CUDA_CHECK(cudaGraphicsMapResources(1, &gfx.cuda_vbo_resource, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vbo_data, &vbo_size, gfx.cuda_vbo_resource));

        updateVBO<<<blk, thr>>>(vbo_data, d_p, d_mask, SIZE, showObjects, simStep2,
                                g_mic_l_x, g_mic_l_y, g_mic_r_x, g_mic_r_y,
                                g_pulse_x, g_pulse_y, g_wedge_x, g_wedge_y,
                                g_player.pivot_x, g_player.pivot_y, g_player.angle,
                                d_toneMarkers, g_numToneMarkers);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &gfx.cuda_vbo_resource, 0));

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(gfx.shader);
        glBindVertexArray(gfx.vao);
        glDrawArrays(GL_POINTS, 0, SIZE * SIZE * 2);
        SDL_GL_SwapWindow(gfx.window);
    }

    SDL_CloseAudioDevice(beepDev);
    SDL_CloseAudioDevice(simDev);
    SDL_CloseAudioDevice(markerDev);

    if (d_toneMarkers) CUDA_CHECK(cudaFree(d_toneMarkers));
    free(h_audioO);
    free(h_audioW);
    free(h_audioI);
    CUDA_CHECK(cudaGraphicsUnregisterResource(gfx.cuda_vbo_resource));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_prev));
    CUDA_CHECK(cudaFree(d_p_next));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_audioL));
    CUDA_CHECK(cudaFree(d_audioR));
    free(h_mask);
    free(h_audioBuffer);
    free(beepBuf);

    glDeleteBuffers(1, &gfx.vbo);
    glDeleteVertexArrays(1, &gfx.vao);
    glDeleteProgram(gfx.shader);
    SDL_GL_DeleteContext(gfx.gl_context);
    SDL_DestroyWindow(gfx.window);
    SDL_Quit();
    return 0;
}
