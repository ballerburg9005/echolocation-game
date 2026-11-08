// echolocation_game_dynamic_params_lag_fix_svg.cu
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
#include <algorithm>
#include "acoustics.h"
#include "collision.h"
#include "svg_raster.h"
#include "echo_arcade.h"
#include "source_model.h"
#include "direct_reference.h"
#include "virtual_cane.h"
#include "contact_audio.h"
#include "proximity_audio.h"
#include "view3d.h"
#include <climits>
#include <ctype.h>

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

static const float g_DX = echo::kCellMetres;
static const float g_DT = echo::kStepSeconds;
static float g_pulseIntervalSec= 1.0f;
static float g_decayFactor     = 0.9995f;
static float g_reflectionCoeff = 0.9f;

static int   g_sampleRate      = 48000;
static float g_c2;
static int   g_pulseIntervalSteps;
static int   g_pulseDurationSteps;

static float g_waveVolumeScale = 0.01f; // start quietly with headphones
static float g_visualScale=0.018f; // shader only, never pressure physics

static const float SPEED_OF_SOUND = echo::kSpeedOfSound;
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
    bool isPlaying; // Playback state
    bool collected; // Item markers are one-shot
};

static std::vector<Rect> g_rects;
static std::vector<Circle> g_circles;
static std::vector<ToneMarker> g_toneMarkers;
static ToneMarker* d_toneMarkers = nullptr;
static int g_numToneMarkers = 0;

static void recalcDerivedParams()
{
    g_c2 = (SPEED_OF_SOUND * g_DT / g_DX) * (SPEED_OF_SOUND * g_DT / g_DX);

    if (g_pulseIntervalSec < 0.1f) g_pulseIntervalSec = 0.1f;
    if (g_pulseIntervalSec > 5.f)  g_pulseIntervalSec = 5.f;
    g_pulseIntervalSteps = std::max(1, (int)lroundf(g_pulseIntervalSec / g_DT));

    g_pulseDurationSteps = (int)lroundf(echo::kClickDurationSeconds / g_DT);
    if (g_pulseDurationSteps < 1) g_pulseDurationSteps = 1;

    if (g_reflectionCoeff < 0.f) g_reflectionCoeff = 0.f;
    if (g_reflectionCoeff > 1.f) g_reflectionCoeff = 1.f;

    if (g_decayFactor < 0.f) g_decayFactor = 0.f;
    if (g_decayFactor > 1.f) g_decayFactor = 1.f;

    g_waveVolumeScale = fmaxf(0.001f, fminf(0.08f, g_waveVolumeScale));
    printf("[PARAMS] Pulse=%.2fs  Decay/step=%.6f  Wall reflection=%.3f  Volume=%.4f\n",
           g_pulseIntervalSec, g_decayFactor, g_reflectionCoeff, g_waveVolumeScale);
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

// SVG stroke/fill parsing for basic shapes. Full cascading CSS, inheritance,
// transforms and path tessellation are deliberately not emulated here.
static bool svgFilled(const tinyxml2::XMLElement* elem) {
    const char* fill = elem->Attribute("fill");
    bool filled = !fill || strcmp(fill, "none") != 0;
    const char* style = elem->Attribute("style");
    if (style) {
        const char* v = strstr(style, "fill:");
        if (v) filled = strncmp(v + 5, "none", 4) != 0;
    }
    return filled;
}

static float svgStrokeWidth(const tinyxml2::XMLElement* elem) {
    const char* stroke = elem->Attribute("stroke");
    bool stroked = stroke && strcmp(stroke, "none") != 0;
    const char* style = elem->Attribute("style");
    if (style) {
        const char* v = strstr(style, "stroke:");
        if (v) stroked = strncmp(v + 7, "none", 4) != 0;
    }
    if (!stroked) return 0.0f;
    float width = 1.0f;
    elem->QueryFloatAttribute("stroke-width", &width);
    if (style) {
        const char* v = strstr(style, "stroke-width:");
        if (v) width = strtof(v + 13, nullptr);
    }
    return fmaxf(0.0f, fminf(50.0f, width));
}

static void parseSVGElement(tinyxml2::XMLElement* elem, float* h_mask)
{
    using namespace tinyxml2;
    static bool warnedTransforms = false;
    while (elem) {
        const char* name = elem->Name();
        // Definitions are not objects in the drawing.
        if (strcmp(name, "defs") == 0 || strcmp(name, "metadata") == 0) {
            elem = elem->NextSiblingElement();
            continue;
        }
        if (elem->Attribute("transform") && !warnedTransforms) {
            fprintf(stderr, "[SVG] WARNING: transforms are unsupported and ignored\n");
            warnedTransforms = true;
        }
        if (strcmp(name, "rect") == 0) {
            Rect r{};
            elem->QueryFloatAttribute("x", &r.x);
            elem->QueryFloatAttribute("y", &r.y);
            elem->QueryFloatAttribute("width", &r.width);
            elem->QueryFloatAttribute("height", &r.height);
            const int count = echo::rasterRect(h_mask, SIZE, r.x, r.y,
                r.width, r.height, svgStrokeWidth(elem), svgFilled(elem));
            if (count > 0) {
                printf("[SVG] Rect x=%.2f y=%.2f w=%.2f h=%.2f (%d cells)\n",
                       r.x, r.y, r.width, r.height, count);
                r.y = SIZE - (r.y + r.height);
                g_rects.push_back(r);
            }
        } else if (strcmp(name, "circle") == 0 || strcmp(name, "ellipse") == 0 ||
                   strcmp(name, "path") == 0) {
            const bool nativeCircle = strcmp(name, "circle") == 0;
            const bool nativeEllipse = strcmp(name, "ellipse") == 0;
            const char* sodipodiType = elem->Attribute("sodipodi:type");
            const bool inkscapeArc = strcmp(name, "path") == 0 &&
                sodipodiType && (strcmp(sodipodiType, "arc") == 0 ||
                                 strcmp(sodipodiType, "circle") == 0);
            if (nativeCircle || nativeEllipse || inkscapeArc) {
                float cx = 0.0f, cy = 0.0f, rx = 0.0f, ry = 0.0f;
                if (inkscapeArc) {
                    elem->QueryFloatAttribute("sodipodi:cx", &cx);
                    elem->QueryFloatAttribute("sodipodi:cy", &cy);
                    elem->QueryFloatAttribute("sodipodi:rx", &rx);
                    elem->QueryFloatAttribute("sodipodi:ry", &ry);
                } else {
                    elem->QueryFloatAttribute("cx", &cx);
                    elem->QueryFloatAttribute("cy", &cy);
                    if (nativeCircle) {
                        elem->QueryFloatAttribute("r", &rx);
                        ry = rx;
                    } else {
                        elem->QueryFloatAttribute("rx", &rx);
                        elem->QueryFloatAttribute("ry", &ry);
                    }
                }
                if (inkscapeArc && ry <= 0.0f) ry = rx;
                const int count = echo::rasterEllipse(h_mask, SIZE, cx, cy,
                    rx, ry, svgStrokeWidth(elem), svgFilled(elem));
                if (count > 0) {
                    g_circles.push_back(Circle{cx, SIZE - cy, fmaxf(rx, ry)});
                    printf("[SVG] Ellipse cx=%.2f cy=%.2f rx=%.2f ry=%.2f (%d cells)\n",
                           cx, cy, rx, ry, count);
                }
            }
        } else if (strcmp(name, "text") == 0) {
            XMLElement* tspan = elem->FirstChildElement("tspan");
            const char* t = tspan ? tspan->GetText() : elem->GetText();
            if (t) {
                while (*t && isspace((unsigned char)*t)) ++t;
                const char type = *t;
                // Text is a marker only when its content is a single code.
                if ((type == 'O' || type == 'I' || type == 'W') &&
                    (t[1] == '\0' || isspace((unsigned char)t[1]))) {
                    ToneMarker tm{};
                    elem->QueryFloatAttribute("x", &tm.x);
                    elem->QueryFloatAttribute("y", &tm.y);
                    tm.y = SIZE - tm.y;
                    tm.type = type;
                    g_toneMarkers.push_back(tm);
                    printf("[SVG] %c marker x=%.1f y=%.1f\n", tm.type, tm.x, tm.y);
                }
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

// No moving player-funnel solids in the FDTD boundary field. The changing
// geometry was causing pressure discontinuities audible as movement rattles.
// Render the funnel as a visual overlay instead.
static void updateDynamicMask(const float* staticMask,float* dynamicMask) {
    memcpy(dynamicMask,staticMask,SIZE*SIZE*sizeof(float));
}

// Fused pressure / source / damping kernel. 2 launches rather than 4 per
// 20.6us step removes substantial driver overhead on consumer GPUs.
// This barrier-free stencil also works on partial CUDA blocks (SIZE=500).
__global__ void updatePressure(float* p_next, const float* p, const float* p_prev,
                               const float* mask, const float* worldEmitters,
                               float c2, int size, float reflection,
                               int sourceX, int sourceY,
                               int rearX, int rearY, int rearDelay,
                               float amplitude,
                               unsigned long long step, int interval,
                               int duration, float dt, float decay,
                               bool automaticPulse, uint64_t manualPulseStart,
                               int decoyX, int decoyY, uint64_t decoyStart)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size) return;
    const int idx = y * size + x;
    if (x == 0 || y == 0 || x == size - 1 || y == size - 1 || mask[idx] > 0.0f) {
        p_next[idx] = 0.0f;
        return;
    }

    const float center = p[idx], previous = p_prev[idx];
    const bool wallL = mask[idx - 1] > 0.0f;
    const bool wallR = mask[idx + 1] > 0.0f;
    const bool wallD = mask[idx - size] > 0.0f;
    const bool wallU = mask[idx + size] > 0.0f;
    const float left  = wallL ? center : p[idx - 1];
    const float right = wallR ? center : p[idx + 1];
    const float down  = wallD ? center : p[idx - size];
    const float up    = wallU ? center : p[idx + size];
    const int adjacentWalls = (int)wallL + (int)wallR + (int)wallD + (int)wallU;
    // Ghosts are moving pressure-reflecting boundaries with a distinct local
    // coefficient; the static SVG wall coefficient remains player-adjustable.
    const bool touchesShard=(wallL && mask[idx-1]>2.5f) ||
                            (wallR && mask[idx+1]>2.5f) ||
                            (wallD && mask[idx-size]>2.5f) ||
                            (wallU && mask[idx+size]>2.5f);
    const bool touchesGhost=(wallL && mask[idx-1]>1.5f) ||
                            (wallR && mask[idx+1]>1.5f) ||
                            (wallD && mask[idx-size]>1.5f) ||
                            (wallU && mask[idx+size]>1.5f);
    const float localReflection=touchesShard?0.62f:
                                touchesGhost?0.82f:reflection;
    const float damping = echo::wallDamping(localReflection,
                                             sqrtf(c2),adjacentWalls);
    float next = echo::impedanceWaveStep(center, previous, left, right,
                                        down, up, c2, damping);

    if (x == sourceX && y == sourceY) {
        const int pulseStep=echo::pulseIndex(
            step,interval,duration,automaticPulse,manualPulseStart);
        if(pulseStep>=0)
            next+=amplitude*echo::phasedPlayerClick(
                pulseStep,duration,dt,false,rearDelay);
    }
    // Delayed rear canceling source: backward-field interference without a
    // moving solid/head. Reflected energy remains in the SAME wave field.
    if (x == rearX && y == rearY && rearDelay>0 &&
        step >= (uint64_t)rearDelay) {
        const int backStep=echo::pulseIndex(
            step-(uint64_t)rearDelay,interval,duration,
            automaticPulse,manualPulseStart);
        if(backStep>=0)
            next+=amplitude*echo::phasedPlayerClick(
                backStep,duration,dt,true,0);
    }
    // An actual remote source in the *same* FDTD solver: the decoy is
    // not merely a stereo UI chirp or visually painted distraction.
    if(x==decoyX && y==decoyY &&
       decoyStart!=UINT64_MAX && step>=decoyStart &&
       step-decoyStart<(uint64_t)duration) {
        const int decoyStep=(int)(step-decoyStart);
        next+=amplitude*0.95f*echo::clickSample(decoyStep,duration,dt);
    }
    // Continuous room-object sources travel through/refelect from the
    // same FDTD geometry as clicks, unlike the old global UI beacons.
    const float emitter=worldEmitters[idx];
    if(emitter>=1.0f)
        next+=echo::fdtdWorldTone(emitter,step,dt);
    p_next[idx] = next * decay;
}

__global__ void captureAndReduce(const float* p, float lx, float ly,
                                 float rx, float ry, int sampleIndex,
                                 float* audioL, float* audioR, float waveVolume)
{
    if (blockIdx.x || threadIdx.x) return;
    audioL[sampleIndex] = echo::bilinearSample(p, SIZE, lx, ly) * waveVolume;
    audioR[sampleIndex] = echo::bilinearSample(p, SIZE, rx, ry) * waveVolume;
}

__global__ void updateVBO(float* vbo_data, float* p, float* mask, int size, int showObj, int t,
                          float visualScale,
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
    const bool displayFunnel=echo::visualFunnelPixel((float)x,(float)y,
        wx,wy,px,py,player_x,player_y);

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
                        if (toneMarkers[i].collected) continue;
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
                        if (displayFunnel) {
                            vbo_data[vbo_idx + 2] = -1.f;
                        } else if (showObj && mask[idx] > 2.5f) {
                            vbo_data[vbo_idx + 2] = -7.f;
                        } else if (showObj && mask[idx] > 1.5f) {
                            vbo_data[vbo_idx + 2] = -8.f;
                        } else if (showObj && mask[idx] > 0.f) {
                            vbo_data[vbo_idx + 2] = -1.f;
                        } else {
                            vbo_data[vbo_idx + 2] = echo::displayPressure(wave, visualScale);
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
                        if (toneMarkers[i].collected) continue;
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
                        if (displayFunnel) {
                            vbo_data[fp_idx + 2] = -1.f;
                        } else if (showObj && mask[idx] > 2.5f) {
                            vbo_data[fp_idx + 2] = -7.f;
                        } else if (showObj && mask[idx] > 1.5f) {
                            vbo_data[fp_idx + 2] = -8.f;
                        } else if (showObj && mask[idx] > 0.f) {
                            vbo_data[fp_idx + 2] = -1.f;
                        } else {
                            vbo_data[fp_idx + 2] = echo::displayPressure(wave, visualScale);
                        }
                    }
                }
            }
        }
    } else {
        vbo_data[fp_idx + 2] = -10.f;
    }
}

struct GraphicsContext {
    SDL_Window* window;
    SDL_GLContext gl_context;
    GLuint vbo, vao, shader;
    GLuint glyphVAO, glyphVBO;
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
    else if (pressure < -7.5) color = vec3(1, 0.18, 0.42); // Ghost reflector
    else if (pressure < -6.5) color = vec3(1, 1, 0);       // Gold for 'I'
    else if (pressure < -5.5) color = vec3(0, 1, 1);       // Cyan for 'W'
    else if (pressure < -4.5) color = vec3(1, 0, 1);       // Purple for 'O'
    else if (pressure < -3.5) color = vec3(0.5, 0, 0.5);   // Dark purple (unused)
    else if (pressure < -2.5) color = vec3(1, 0, 0);       // Red (pulse)
    else if (pressure < -1.5) color = vec3(0, 1, 0);       // Green (mic)
    else if (pressure < -0.5) color = vec3(1, 1, 0);       // Yellow (objects)
    else {
        // Display-only logarithmic exposure restores a broad gradation of
        // visible wave strengths. The old tanh clipped most large pulses
        // to identical saturated red/blue, causing hard contour bands.
        float v=clamp(abs(pressure)/0.45,0.0,1.0);
        float middle=0.13*sqrt(v);
        color=pressure>=0.0 ?
            vec3(v,middle,0.055*v) :
            vec3(0.055*v,middle,v);
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
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS,1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES,4);

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
    glEnable(GL_MULTISAMPLE);
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

    // Small CPU-drawn map glyphs do not alter the acoustic occupancy mask.
    // They show collectible shards/exit without creating fake reflectors.
    glGenVertexArrays(1, &gfx.glyphVAO);
    glGenBuffers(1, &gfx.glyphVBO);
    glBindVertexArray(gfx.glyphVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gfx.glyphVBO);
    glBufferData(GL_ARRAY_BUFFER, 128 * 3 * sizeof(float), nullptr,
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3*sizeof(float),
                          (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

// Event jingles are deliberately distinct from the wave-simulated clicks.
// They indicate game rules; they should not be interpreted as 3D acoustics.
static void playArcadeFeedback(SDL_AudioDeviceID device,int sampleRate,int kind)
{
    const float duration=kind==5?0.52f:kind==6?0.24f:kind==7?0.32f:
                         kind==4?0.24f:kind==2?0.66f:
                         kind==3?0.60f:kind==1?0.34f:0.20f;
    const int count=(int)(duration*sampleRate);
    std::vector<float> samples(count*2,0.0f);
    for(int i=0;i<count;++i) {
        const float t=(float)i/sampleRate;
        const int note=kind==7?(int)(t/0.16f):
                       kind==2?(int)(t/0.22f):
                       kind==3?(int)(t/0.20f):
                       kind==1?(int)(t/0.17f):0;
        const float frequency=kind==5?(460.0f+1080.0f*t/duration):
                              kind==6?(1130.0f-540.0f*t/duration):
                              kind==7?(930.0f+240.0f*(float)note):
                              kind==4?(1040.0f+800.0f*t/duration):
                              kind==2?(note==0?620.0f:note==1?830.0f:1240.0f):
                              kind==3?(note==0?490.0f:note==1?350.0f:210.0f):
                              kind==1?(note==0?350.0f:230.0f):1050.0f;
        const float noteLength=kind==7?0.16f:kind==2?0.22f:
                               kind==3?0.20f:kind==1?0.17f:duration;
        const float phase=fmodf(t,noteLength)/noteLength;
        const float envelope=sinf(PI*phase);
        const float v=0.18f*envelope*envelope*sinf(2.0f*PI*frequency*t);
        samples[2*i]=v;samples[2*i+1]=v;
    }
    SDL_QueueAudio(device,samples.data(),(Uint32)(samples.size()*sizeof(float)));
}

// Virtual cane assistance: three successive 75ms left/centre/right taps.
// The tap pitch indicates obstacle distance and loudness indicates contact;
// these stereo cues are separate from the 2D wave simulation.
static void playCaneScan(SDL_AudioDeviceID device,int sampleRate,
                         const float* staticMask,int size,
                         float px,float py,float heading)
{
    if(SDL_GetQueuedAudioSize(device)>
       (Uint32)(0.45f*sampleRate*2*sizeof(float)))return;
    const auto rays=echo::sweepCane(staticMask,size,px,py,heading);
    const int samples=(int)(0.39f*sampleRate);
    std::vector<float> output(2*samples,0.0f);
    const float panLeft[3]={0.96f,0.70f,0.18f};
    const float panRight[3]={0.18f,0.70f,0.96f};
    for(int i=0;i<3;++i) {
        const float nearby=rays[i].hit?
            fmaxf(0.0f,1.0f-fminf(rays[i].distanceCm,100.0f)/100.0f):0.0f;
        const float frequency=rays[i].hit?510.0f+840.0f*nearby:245.0f;
        const float loudness=rays[i].hit?0.09f+0.075f*nearby:0.029f;
        const int first=(int)(i*0.13f*sampleRate);
        const int n=(int)(0.075f*sampleRate);
        for(int j=0;j<n && first+j<samples;++j){
            const float t=(float)j/sampleRate;
            const float env=sinf(PI*(float)j/(float)n);
            const float sample=loudness*env*env*sinf(2.0f*PI*frequency*t);
            output[2*(first+j)]+=sample*panLeft[i];
            output[2*(first+j)+1]+=sample*panRight[i];
        }
    }
    SDL_QueueAudio(device,output.data(),(Uint32)(output.size()*sizeof(float)));
}

int main(int argc, char* argv[])
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

    const echo::SafeSpawn initialSpawn=echo::findFreeSpawn(
        h_mask,SIZE,g_player.pivot_x,g_player.pivot_y,13.0f);
    if(!initialSpawn.valid) {
        fprintf(stderr,"No safe player spawn on SVG level %s\n",svgFile);
        free(h_mask);
        return EXIT_FAILURE;
    }
    if(initialSpawn.x!=g_player.pivot_x || initialSpawn.y!=g_player.pivot_y)
        printf("[LEVEL] Moved blocked player start to (%.1f, %.1f)\n",
               initialSpawn.x,initialSpawn.y);
    g_player.pivot_x=initialSpawn.x;
    g_player.pivot_y=initialSpawn.y;
    const Player initialPlayer=g_player;
    updatePlayerComponents(g_player);

    g_numToneMarkers = g_toneMarkers.size();
    // Object discovery tones are now physical FDTD world emitters; the
    // old nonphysical proximity chimes are deliberately disabled.
    if (g_numToneMarkers > 0) {
        CUDA_CHECK(cudaMalloc(&d_toneMarkers, g_numToneMarkers * sizeof(ToneMarker)));
        CUDA_CHECK(cudaMemcpy(d_toneMarkers, g_toneMarkers.data(), 
                            g_numToneMarkers * sizeof(ToneMarker), 
                            cudaMemcpyHostToDevice));
    }

    float *d_mask;
    CUDA_CHECK(cudaMalloc(&d_mask, SIZE * SIZE * sizeof(float)));
    float *d_worldEmitters;
    CUDA_CHECK(cudaMalloc(&d_worldEmitters,SIZE*SIZE*sizeof(float)));
    std::vector<float> h_worldEmitters(SIZE*SIZE,0.0f);

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
    std::vector<float> h_audioL(AUDIO_BATCH), h_audioR(AUDIO_BATCH);
    std::vector<float> h_audioOut;
    h_audioOut.reserve(AUDIO_BATCH * 2 + 8);
    float* h_dynamic = (float*)malloc(SIZE * SIZE * sizeof(float));
    if (!h_dynamic) { fprintf(stderr, "Out of host memory\n"); exit(1); }

    GraphicsContext gfx;
    initGraphics(gfx);
    echo::View3D view3d;
    const bool has3D = echo::init3D(view3d,h_mask,SIZE);
    if(!has3D)fprintf(stderr,"[3D] Disabled: shader/program unavailable\n");
    bool show3D=has3D, full3D=false, arcadeEnabled=true;

    echo::EchoArcade arcade(h_mask,SIZE,g_player.pivot_x,g_player.pivot_y);
    for(const ToneMarker& tm:g_toneMarkers){
        if(tm.type=='W'){arcade.setExit(tm.x,tm.y);break;}
    }
    std::vector<echo::VisualOrb> orbs;
    orbs.reserve(128);
    std::vector<float> mapGlyphs;
    mapGlyphs.reserve(128*3);
    double lastArcadeMessage=0.0;
    double lastThreatCue=0.0;
    double lastGuidanceCue=0.0;
    // Large-range artificial beacons were confusing and sounded through walls.
    // Opt-in aids are quiet by default; short-range SVG discoveries still work.
    bool dangerAudio=false, guidanceAudio=false;
    std::printf("[ARCADE] %d ghosts, %d shards; reach the exit after collecting all.\n",
                (int)arcade.enemies.size(),arcade.remaining);

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = 48000;
    want.format = AUDIO_F32SYS; // Back to 32-bit float
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
    beepSpec.format = AUDIO_F32SYS; // Back to 32-bit float
    beepSpec.channels = 2;
    beepSpec.samples = 1024;
    SDL_AudioDeviceID beepDev = SDL_OpenAudioDevice(NULL, 0, &beepSpec, &have, 0);
    if (!beepDev) {
        fprintf(stderr, "Failed beep audio:%s\n", SDL_GetError());
        exit(1);
    }
    SDL_PauseAudioDevice(beepDev, 0);

    g_sampleRate = have.freq;
    recalcDerivedParams();
    echo::StereoLinearResampler audioResampler(g_DT, g_sampleRate);
    updatePlayerComponents(g_player);
    // Mic positions are continuous across GPU batches: sampling an
    // instantaneous new position once per SDL frame can zipper/rattle in
    // the standing pressure field even when the wall grid is unchanged.
    float previousMicLX=g_mic_l_x,previousMicLY=g_mic_l_y;
    float previousMicRX=g_mic_r_x,previousMicRY=g_mic_r_y;

    dim3 thr(32, 32);
    dim3 blk((SIZE + 31) / 32, (SIZE + 31) / 32);

    uint64_t perfFreq = SDL_GetPerformanceFrequency();
    uint64_t lastCount = SDL_GetPerformanceCounter();
    double simTime = 0.0;
    bool quit = false;
    int showObjects = 1;
    int audioBatchIndex = 0;

    echo::ContactTrigger wallTouch;
    bool gameWon = false;
    int collectedItems = 0;
    double lastDebugTime = 0.0;
    unsigned long long simStep2=0;
    uint64_t manualPulseAt=UINT64_MAX;
    uint64_t decoyPulseAt=UINT64_MAX;
    int decoyPulseX=0,decoyPulseY=0;
    double lastDecoySimTime=-100.0;
    constexpr double decoyCooldownSeconds=9.0;
    bool automaticPulse=true;
    bool phasedClick=true, worldTones=true;
    bool echoOnlyMonitor=false; // opt-in until on-hardware listening tests
    echo::DirectResponse lastDirect;
    uint64_t directPulseStart=UINT64_MAX;

    std::puts("[CONTROLS] F2 3D/2D | F3 immersive 3D | F4 arcade on/off | R restart");
    std::puts("[CONTROLS] W/S move | A/D rotate | Q/E strafe | arrows also work");
    std::puts("[CONTROLS] F click | T auto-pings/stealth | G FDTD decoy (9s cooldown)");
    std::puts("[CONTROLS] B phased/isotropic FDTD click | N world tones | J echo-only monitor (experimental)");
    std::puts("[CONTROLS] C obstacle scan | H spend charge for 5s ghost overdrive");
    std::puts("[CONTROLS] 1/2 pulse interval");
    std::puts("[CONTROLS] F5 short-range threat aid | F6 nearby-target aid (both OFF by default)");
    std::puts("[CONTROLS] 3/4 decay | 5/6 reflections | 7/8 volume | 9/0 wave DISPLAY exposure");

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
                    case SDLK_7: g_waveVolumeScale -= 0.002f; recalcDerivedParams(); break;
                    case SDLK_8: g_waveVolumeScale += 0.002f; recalcDerivedParams(); break;
                    case SDLK_9:
                        g_visualScale=fmaxf(0.001f,g_visualScale/1.25f);
                        break;
                    case SDLK_0:
                        g_visualScale=fminf(0.08f,g_visualScale*1.25f);
                        break;
                    case SDLK_SPACE: showObjects = !showObjects; break;
                    case SDLK_F2: if(has3D){show3D=!show3D;full3D=false;} break;
                    case SDLK_F3: if(has3D) {
                        show3D=true;full3D=!full3D;
                    } break;
                    case SDLK_F4: arcadeEnabled=!arcadeEnabled; break;
                    case SDLK_F5: dangerAudio=!dangerAudio; break;
                    case SDLK_F6: guidanceAudio=!guidanceAudio; break;
                    case SDLK_t:
                        if(!e.key.repeat)automaticPulse=!automaticPulse;
                        break;
                    case SDLK_b:
                        if(!e.key.repeat)phasedClick=!phasedClick;
                        printf("[SOURCE] %s FDTD player click\n",
                                phasedClick?"forward-phased":"isotropic");
                        break;
                    case SDLK_n:
                        if(!e.key.repeat)worldTones=!worldTones;
                        printf("[SOURCES] Room-object FDTD tones %s\n",
                                worldTones?"ON":"OFF");
                        break;
                    case SDLK_j:
                        if(!e.key.repeat) {
                            echoOnlyMonitor=!echoOnlyMonitor;
                            directPulseStart=UINT64_MAX;
                            lastDirect={};
                            printf("[MONITOR] %s direct-field subtraction (experimental)\n",
                                echoOnlyMonitor?"Echo-only":"Full pressure");
                        }
                        break;
                    case SDLK_f:
                        if(!e.key.repeat) {
                            manualPulseAt=simStep2;
                            if(arcadeEnabled)arcade.hearClick(g_pulse_x,g_pulse_y);
                        }
                        break;
                    case SDLK_h:
                        if(!e.key.repeat && arcadeEnabled &&
                           arcade.activateOverdrive()) {
                            // Unlike the extra power-up game rules, this
                            // activation click excites the actual wave field.
                            manualPulseAt=simStep2;
                            playArcadeFeedback(beepDev,g_sampleRate,5);
                        }
                        break;
                    case SDLK_c:
                        if(!e.key.repeat)playCaneScan(beepDev,g_sampleRate,
                            h_mask,SIZE,g_player.pivot_x,g_player.pivot_y,
                            g_player.angle-PI/4.0f);
                        break;
                    case SDLK_g:
                        if(!e.key.repeat && arcadeEnabled && !arcade.won &&
                           !arcade.lost &&
                           simTime-lastDecoySimTime>=decoyCooldownSeconds) {
                            const echo::DecoyPlacement d=echo::projectDecoy(
                                h_mask,SIZE,g_player.pivot_x,g_player.pivot_y,
                                g_pulse_x-g_player.pivot_x,
                                g_pulse_y-g_player.pivot_y);
                            if(d.valid) {
                                decoyPulseX=(int)lroundf(d.x);
                                decoyPulseY=(int)lroundf(d.y);
                                decoyPulseAt=simStep2;
                                lastDecoySimTime=simTime;
                                arcade.hearClick(d.x,d.y);
                                playArcadeFeedback(beepDev,g_sampleRate,4);
                            }
                        }
                        break;
                    case SDLK_r:
                        arcade.reset();
                        for(const ToneMarker& tm:g_toneMarkers)if(tm.type=='W'){
                            arcade.setExit(tm.x,tm.y);break;
                        }
                        gameWon=false;collectedItems=0;simStep2=0;simTime=0;
                        manualPulseAt=UINT64_MAX;decoyPulseAt=UINT64_MAX;
                        directPulseStart=UINT64_MAX;
                        lastDirect={};
                        lastDecoySimTime=-100;automaticPulse=true;
                        g_player=initialPlayer;
                        updatePlayerComponents(g_player);
                        previousMicLX=g_mic_l_x; previousMicLY=g_mic_l_y;
                        previousMicRX=g_mic_r_x; previousMicRY=g_mic_r_y;
                        for(ToneMarker &tm:g_toneMarkers)tm.collected=false;
                        // FDTD emitters are reconstructed from reset objects.
                        if(g_numToneMarkers>0)CUDA_CHECK(cudaMemcpy(
                            d_toneMarkers,g_toneMarkers.data(),
                            g_numToneMarkers*sizeof(ToneMarker),cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemset(d_p,0,SIZE*SIZE*sizeof(float)));
                        CUDA_CHECK(cudaMemset(d_p_prev,0,SIZE*SIZE*sizeof(float)));
                        CUDA_CHECK(cudaMemset(d_p_next,0,SIZE*SIZE*sizeof(float)));
                        audioResampler=echo::StereoLinearResampler(g_DT,g_sampleRate);
                        SDL_ClearQueuedAudio(simDev);
                        SDL_ClearQueuedAudio(beepDev);
                        wallTouch=echo::ContactTrigger{};
                        lastThreatCue=0;lastGuidanceCue=0;
                        lastCount=SDL_GetPerformanceCounter();
                        break;
                    case SDLK_ESCAPE: quit = true; break;
                }
            }
        }

        uint64_t now = SDL_GetPerformanceCounter();
        // Bound interaction movement after stalls; simulation remains audio-paced.
        float dtFrame = fminf(0.05f, (float)((now - lastCount) / (double)perfFreq));
        lastCount = now;

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

        // Conservative 13 cm player radius, substepped motion and tangent/
        // per-axis fallbacks. This replaces the 360-ray heuristic, which
        // sometimes pushed the player through obstacles or stopped sliding.
        const echo::SlideResult motion = echo::slideCircle(
            h_mask, SIZE, g_player.pivot_x, g_player.pivot_y,
            dx, dy, 13.0f);
        newP.pivot_x = motion.x;
        newP.pivot_y = motion.y;
        const bool collided = motion.collided;
        const float heading = newP.angle - PI / 4.0f;
        // Use the collision's actual nearest obstructing wall cell. Previous
        // code relied on a smeared aggregate normal and only panned a 440Hz
        // tone, which was directionally ambiguous (front sounded like back).
        const echo::WallContact contact=echo::findWallContact(
            h_mask,SIZE,newP.pivot_x,newP.pivot_y,heading,
            motion.wallX,motion.wallY,13.0f);

        g_player = newP;
        updatePlayerComponents(g_player);
        echo::ArcadeEvent arcadeEvent;
        if(arcadeEnabled) {
            arcadeEvent=arcade.update(dtFrame,g_player.pivot_x,g_player.pivot_y);
            if(arcadeEvent.bounced) {
                g_player.pivot_x=arcadeEvent.recoilX;
                g_player.pivot_y=arcadeEvent.recoilY;
                updatePlayerComponents(g_player);
                // Recoil is a discrete game event, not ordinary continuous
                // movement; reset the interpolation anchor rather than
                // dragging microphones through a 16cm wall in one batch.
                previousMicLX=g_mic_l_x;previousMicLY=g_mic_l_y;
                previousMicRX=g_mic_r_x;previousMicRY=g_mic_r_y;
            }
            if(arcade.reachExit(g_player.pivot_x,g_player.pivot_y)) {
                arcadeEvent.cleared=true;
                gameWon=true;
            }
        }
        updateDynamicMask(h_mask,h_dynamic);
        // Moving enemies really alter the *CUDA wave boundary*, not merely
        // the visuals. Static maze remains unchanged for CPU collisions.
        if(arcadeEnabled) {
            arcade.stampEnemies(h_dynamic);
            arcade.stampShards(h_dynamic);
        }
        CUDA_CHECK(cudaMemcpy(d_mask, h_dynamic, SIZE * SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
        // CPU builds only the sparse FDTD source map once per *frame*.
        // The pressure kernel sees O(1) map access, not O(object count)
        // comparisons for every one of ~48,500 steps each second.
        std::fill(h_worldEmitters.begin(),h_worldEmitters.end(),0.0f);
        if(worldTones) {
            if(arcadeEnabled) {
                for(const echo::Enemy& en:arcade.enemies) {
                    if(en.eaten)continue;
                    echo::placeWorldEmitter(h_worldEmitters,h_dynamic,SIZE,
                        en.x,en.y,(int)en.archetype+echo::kGhostTracker,9.0f);
                }
                for(size_t n=0;n<arcade.shards.size();++n) {
                    const auto &sh=arcade.shards[n];
                    if(sh.collected)continue;
                    echo::placeWorldEmitter(h_worldEmitters,h_dynamic,SIZE,
                        sh.x,sh.y,echo::kShardFirst+(int)(n%12),6.0f);
                }
                echo::placeWorldEmitter(h_worldEmitters,h_dynamic,SIZE,
                    arcade.exitX,arcade.exitY,echo::kArcadeExit,0.0f);
            }
            for(const ToneMarker &tm:g_toneMarkers) {
                if(tm.collected)continue;
                if(arcadeEnabled && tm.type=='W')continue;
                const int id=tm.type=='W'?echo::kSvgGoal:
                         tm.type=='I'?echo::kSvgItem:
                         echo::kSvgObstacle;
                echo::placeWorldEmitter(h_worldEmitters,h_dynamic,SIZE,
                                       tm.x,tm.y,id,0.0f);
            }
        }
        CUDA_CHECK(cudaMemcpy(d_worldEmitters,h_worldEmitters.data(),
            SIZE*SIZE*sizeof(float),cudaMemcpyHostToDevice));

        double nowT = now / (double)perfFreq;
        if(arcadeEnabled) {
            if(arcadeEvent.pickups>0)playArcadeFeedback(beepDev,g_sampleRate,0);
            if(arcadeEvent.chargesGained>0)playArcadeFeedback(beepDev,g_sampleRate,7);
            if(arcadeEvent.ghostsCaught>0)playArcadeFeedback(beepDev,g_sampleRate,6);
            if(arcadeEvent.hit)playArcadeFeedback(beepDev,g_sampleRate,
                                                   arcadeEvent.gameOver?3:1);
            if(arcadeEvent.cleared)playArcadeFeedback(beepDev,g_sampleRate,2);
            if(nowT-lastArcadeMessage>0.35) {
                char title[256];
                std::snprintf(title,sizeof(title),
                    "ECHO ARCADE | %s | Score %d | Lives %d | Shards %d | Charges %d | R restart | B beam N tones",
                    arcade.won?"LEVEL CLEAR":arcade.lost?"GAME OVER":
                    arcade.overdrive>0?"GHOSTS FROZEN":"HUNT THE ECHO",
                    arcade.score,arcade.lives,arcade.remaining,arcade.charges);
                SDL_SetWindowTitle(gfx.window,title);
                lastArcadeMessage=nowT;
            }
        }
        // Extra nonphysical audio HUD: nearest approaching ghost produces
        // a low stereo heartbeat that speeds up with proximity. F5 turns it
        // off if the player wants to train with only wave reflections.
        if(arcadeEnabled&&dangerAudio&&!arcade.won&&!arcade.lost&&arcade.overdrive<=0.0f) {
            echo::ThreatCue nearest;
            float bestDistance=1.0e9f;
            for(const echo::Enemy &en:arcade.enemies) {
                if(en.eaten)continue;
                const float d=std::hypot(en.x-g_player.pivot_x,
                                         en.y-g_player.pivot_y);
                if(d<bestDistance) {
                    bestDistance=d;
                    nearest=echo::threatCue(g_player.pivot_x,
                           g_player.pivot_y,g_player.angle,en.x,en.y,
                           en.alert>0,en.archetype);
                }
            }
            if(nearest.audible && nowT-lastThreatCue>=nearest.interval &&
               SDL_GetQueuedAudioSize(beepDev) <
                   (Uint32)(0.18*g_sampleRate*2*sizeof(float))) {
                const int n=(int)(0.085*g_sampleRate);
                std::vector<float> heartbeat(2*n);
                for(int j=0;j<n;++j) {
                    const float t=(float)j/g_sampleRate;
                    const float env=sinf(PI*(float)j/(float)n);
                    const float beat=(sinf(2.0f*PI*nearest.frequency*t) +
                                      0.22f*sinf(4.0f*PI*nearest.frequency*t))*
                                     env*env;
                    heartbeat[2*j]=nearest.left*beat;
                    heartbeat[2*j+1]=nearest.right*beat;
                }
                SDL_QueueAudio(beepDev,heartbeat.data(),
                               (Uint32)(heartbeat.size()*sizeof(float)));
                lastThreatCue=nowT;
            }
        }

        // A distinct high-pitched optional "next target" beacon allows
        // navigation without vision. It is game UI (direct stereo bearing),
        // not a simulated sound propagation source; F6 disables the aid.
        if(arcadeEnabled&&guidanceAudio&&!arcade.won&&!arcade.lost &&
           nowT-lastGuidanceCue>=4.5 &&
           SDL_GetQueuedAudioSize(beepDev) <
               (Uint32)(0.15*g_sampleRate*2*sizeof(float))) {
            const echo::GuidanceTarget target=arcade.guidanceTarget();
            if(target.available &&
               std::hypot(target.x-g_player.pivot_x,
                          target.y-g_player.pivot_y)<35.0f &&
               echo::hasClearSight(h_mask,SIZE,g_player.pivot_x,
                        g_player.pivot_y,target.x,target.y)) {
                const float dx=target.x-g_player.pivot_x;
                const float dy=target.y-g_player.pivot_y;
                const float bearing=atan2f(dy,dx)-
                    (g_player.angle-PI/4.0f);
                const float pan=sinf(bearing);
                const float left=sqrtf(0.5f*(1.0f+pan));
                const float right=sqrtf(0.5f*(1.0f-pan));
                const int notes=target.exit?3:2;
                const int n=(int)(notes*0.13f*g_sampleRate);
                std::vector<float> beacon(2*n,0.0f);
                const float distance=std::hypot(dx,dy);
                const float gain=0.09f*echo::proximityStrength(distance,35.0f);
                for(int j=0;j<n;++j) {
                    const float t=(float)j/g_sampleRate;
                    const int note=std::min(notes-1,(int)(t/0.13f));
                    const float within=fmodf(t,0.13f);
                    if(within>0.08f)continue;
                    const float env=sinf(PI*within/0.08f);
                    const float frequency=(target.exit?1190.0f:810.0f)+
                                          note*(target.exit?170.0f:110.0f);
                    const float chirp=gain*env*env*
                        sinf(2.0f*PI*frequency*within);
                    beacon[2*j]=chirp*left;
                    beacon[2*j+1]=chirp*right;
                }
                SDL_QueueAudio(beepDev,beacon.data(),
                               (Uint32)(beacon.size()*sizeof(float)));
                lastGuidanceCue=nowT;
            }
        }

        // A short single contact tap, only on collision onset or a genuinely
        // new contact direction. Explicit ear-to-hit distances generate
        // level asymmetry and sub-millisecond interaural delay, while rear
        // contacts have softer spectral content. No recurring sliding beeps.
        if(wallTouch.shouldPlay(collided && contact.valid,
                 hypotf(dx,dy),contact.relative,nowT) &&
           SDL_GetQueuedAudioSize(beepDev) <
               (Uint32)(0.12f*g_sampleRate*2*sizeof(float))) {
            const auto ears=echo::contactEars(contact.relative,
                                             contact.distanceCm,g_sampleRate);
            auto tap=echo::renderContactTap(ears,g_sampleRate);
            if(!tap.empty())
                SDL_QueueAudio(beepDev,tap.data(),
                               (Uint32)(tap.size()*sizeof(float)));
        }

        // O/I/W landmarks now emit their location-specific, extremely quiet
        // tones INSIDE the 2D FDTD field. Only the game-rule collection / win
        // events remain direct UI audio (and the wall-contact tactile cue).
        for(int i=0;i<g_numToneMarkers;++i) {
            ToneMarker &tm=g_toneMarkers[i];
            if(tm.collected)continue;
            const float dist=std::hypot(tm.x-g_player.pivot_x,
                                        tm.y-g_player.pivot_y);
            if(dist<13.0f && tm.type=='I') {
                tm.collected=true;
                ++collectedItems;
                printf("[GAME] SVG item collected: %d\n",collectedItems);
                CUDA_CHECK(cudaMemcpy(d_toneMarkers,g_toneMarkers.data(),
                    g_numToneMarkers*sizeof(ToneMarker),cudaMemcpyHostToDevice));
                playArcadeFeedback(beepDev,g_sampleRate,0);
            } else if(dist<13.0f && tm.type=='W' && !gameWon &&
                      (!arcadeEnabled || arcade.remaining==0)) {
                gameWon=true;
                printf("[GAME] SVG goal reached\n");
                if(!arcadeEnabled)playArcadeFeedback(beepDev,g_sampleRate,2);
            }
        }

        // SDL's output clock, not render-frame delta or an arbitrary fudge
        // factor, determines how many real physical FDTD steps are needed.
        const int queuedFrames = (int)(SDL_GetQueuedAudioSize(simDev) /
                                       (2 * sizeof(float)));
        const int targetFrames = (int)(0.060 * g_sampleRate);
        const int missingFrames = std::max(0, targetFrames - queuedFrames);
        const int steps = std::min(1024, std::min(AUDIO_BATCH,
            (int)ceil((missingFrames + (missingFrames ? 2 : 0)) /
                      (g_sampleRate * (double)g_DT))));

        const auto pair=echo::phasedSource(SIZE,h_mask,
            g_player.pivot_x,g_player.pivot_y,
            g_player.angle-PI/4.0f);
        const int sourceX=pair.enabled?pair.frontX:(int)lroundf(g_pulse_x);
        const int sourceY=pair.enabled?pair.frontY:(int)lroundf(g_pulse_y);
        const int rearX=(pair.enabled && phasedClick)?pair.backX:-1;
        const int rearY=(pair.enabled && phasedClick)?pair.backY:-1;
        const int rearDelay=(pair.enabled && phasedClick)?pair.backDelay:0;
        for (int i = 0; i < steps; ++i) {
            const float alpha=(float)(i+1)/(float)steps;
            const float micLX=previousMicLX+(g_mic_l_x-previousMicLX)*alpha;
            const float micLY=previousMicLY+(g_mic_l_y-previousMicLY)*alpha;
            const float micRX=previousMicRX+(g_mic_r_x-previousMicRX)*alpha;
            const float micRY=previousMicRY+(g_mic_r_y-previousMicRY)*alpha;
            // Keep the actual wave grid intact. Option J subtracts an
            // independently FDTD-simulated homogeneous incident field
            // ONLY from the headphones, after GPU sampling. It therefore
            // leaves reflections to be generated normally by the grid.
            if(echoOnlyMonitor && echo::pulseIndex(
                simStep2,g_pulseIntervalSteps,g_pulseDurationSteps,
                automaticPulse,manualPulseAt)==0) {
                echo::PhasedSource refSource;
                refSource.frontX=sourceX;refSource.frontY=sourceY;
                refSource.backX=rearX;refSource.backY=rearY;
                refSource.backDelay=rearDelay;
                refSource.enabled=rearDelay>0;
                lastDirect=echo::freeFieldDirect(
                    refSource,micLX,micLY,micRX,micRY,
                    g_pulseDurationSteps,g_DT,g_c2,g_decayFactor,
                    PULSE_AMPLITUDE,g_waveVolumeScale,
                    phasedClick && pair.enabled);
                directPulseStart=simStep2;
            }
            if (arcadeEnabled && automaticPulse && simStep2>=1000ULL &&
                (simStep2-1000ULL)%(unsigned long long)g_pulseIntervalSteps==0ULL)
                arcade.hearClick(g_pulse_x,g_pulse_y);
            updatePressure<<<blk, thr>>>(d_p_next, d_p, d_p_prev,
                                         d_mask,d_worldEmitters,
                                         g_c2, SIZE, g_reflectionCoeff,
                                         sourceX, sourceY,
                                         rearX,rearY,rearDelay,PULSE_AMPLITUDE,
                                         simStep2, g_pulseIntervalSteps,
                                         g_pulseDurationSteps, g_DT, g_decayFactor,
                                         automaticPulse,manualPulseAt,
                                         decoyPulseX,decoyPulseY,decoyPulseAt);
            CUDA_CHECK(cudaGetLastError());

            captureAndReduce<<<1, 1>>>(d_p_next, micLX, micLY,
                                       micRX, micRY, audioBatchIndex,
                                       d_audioL, d_audioR, g_waveVolumeScale);
            CUDA_CHECK(cudaGetLastError());

            ++audioBatchIndex;
            ++simStep2;
            simTime += g_DT;

            float* tmp = d_p_prev;
            d_p_prev = d_p;
            d_p = d_p_next;
            d_p_next = tmp;
        }

        if(steps>0) {
            previousMicLX=g_mic_l_x; previousMicLY=g_mic_l_y;
            previousMicRX=g_mic_r_x; previousMicRY=g_mic_r_y;
        }
        if(manualPulseAt!=UINT64_MAX &&
           simStep2>=manualPulseAt+(uint64_t)g_pulseDurationSteps)
            manualPulseAt=UINT64_MAX;
        if(decoyPulseAt!=UINT64_MAX &&
           simStep2>=decoyPulseAt+(uint64_t)g_pulseDurationSteps)
            decoyPulseAt=UINT64_MAX;

        if (audioBatchIndex > 0) {
            CUDA_CHECK(cudaMemcpy(h_audioL.data(), d_audioL,
                                  audioBatchIndex * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_audioR.data(), d_audioR,
                                  audioBatchIndex * sizeof(float), cudaMemcpyDeviceToHost));
            h_audioOut.clear();
            const uint64_t firstStep=simStep2-(uint64_t)audioBatchIndex;
            for (int k = 0; k < audioBatchIndex; ++k) {
                const uint64_t physicalStep=firstStep+(uint64_t)k;
                const float l=echoOnlyMonitor?
                    echo::subtractDirect(h_audioL[k],lastDirect.left,
                                         directPulseStart,physicalStep):
                    h_audioL[k];
                const float r=echoOnlyMonitor?
                    echo::subtractDirect(h_audioR[k],lastDirect.right,
                                         directPulseStart,physicalStep):
                    h_audioR[k];
                audioResampler.push(l,r,h_audioOut);
            }
            // A soft limiter prevents a numerical transient from delivering
            // arbitrary full-scale peaks to headphones.
            for (float &sample : h_audioOut) {
                sample = isfinite(sample) ? 0.8f * tanhf(sample) : 0.0f;
            }
            if (!h_audioOut.empty() &&
                SDL_QueueAudio(simDev, h_audioOut.data(),
                               (Uint32)(h_audioOut.size() * sizeof(float))) != 0) {
                fprintf(stderr, "SDL_QueueAudio error: %s\n", SDL_GetError());
            }
            audioBatchIndex = 0;
        }
        if (simTime - lastDebugTime >= 5.0) {
            float queuedSeconds = SDL_GetQueuedAudioSize(simDev) /
                                  (2.0f * sizeof(float) * g_sampleRate);
            printf("[DEBUG] SimTime=%.2f Steps=%d AudioQueue=%.3fs\n",
                   simTime, steps, queuedSeconds);
            lastDebugTime = simTime;
        }

        float* vbo_data; size_t vbo_size;
        CUDA_CHECK(cudaGraphicsMapResources(1, &gfx.cuda_vbo_resource, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vbo_data, &vbo_size, gfx.cuda_vbo_resource));

        updateVBO<<<blk, thr>>>(vbo_data, d_p, d_mask, SIZE, showObjects, simStep2,
                                g_visualScale,
                                g_mic_l_x, g_mic_l_y, g_mic_r_x, g_mic_r_y,
                                g_pulse_x, g_pulse_y, g_wedge_x, g_wedge_y,
                                g_player.pivot_x, g_player.pivot_y, g_player.angle,
                                d_toneMarkers, g_numToneMarkers);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &gfx.cuda_vbo_resource, 0));

        if(!full3D || !show3D) {
            glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);
            glClearColor(0,0,0,1);
            glClear(GL_COLOR_BUFFER_BIT);
            glDisable(GL_DEPTH_TEST);
            glUseProgram(gfx.shader);
            glBindVertexArray(gfx.vao);
            glPointSize(1.0f);
            glDrawArrays(GL_POINTS,0,SIZE*SIZE*2);

            if(arcadeEnabled) {
                mapGlyphs.clear();
                // Glyphs live exclusively on the 2D map: none are added to
                // the wave mask (unlike the actual reflecting ghosts).
                auto addGlyph=[&](float x,float y,float code) {
                    mapGlyphs.push_back((2*x/SIZE-1)*0.5f-0.5f);
                    mapGlyphs.push_back(2*y/SIZE-1);
                    mapGlyphs.push_back(code);
                };
                for(const echo::Enemy &en:arcade.enemies)
                    if(!en.eaten)
                        addGlyph(en.x,en.y,arcade.overdrive>0?-6.0f:-8.0f);
                for(const echo::Shard &shard:arcade.shards)
                    if(!shard.collected)addGlyph(shard.x,shard.y,-7.0f);
                addGlyph(arcade.exitX,arcade.exitY,-6.0f);
                if(lastDecoySimTime>=0 &&
                   simTime-lastDecoySimTime<1.2)
                    addGlyph((float)decoyPulseX,(float)decoyPulseY,-6.0f);
                glBindVertexArray(gfx.glyphVAO);
                glBindBuffer(GL_ARRAY_BUFFER,gfx.glyphVBO);
                glBufferData(GL_ARRAY_BUFFER,
                    (GLsizeiptr)(mapGlyphs.size()*sizeof(float)),
                    mapGlyphs.data(),GL_DYNAMIC_DRAW);
                glPointSize(8.0f);
                glDrawArrays(GL_POINTS,0,(GLsizei)(mapGlyphs.size()/3));
                glPointSize(1.0f);
            }
        }

        if(has3D && show3D) {
            orbs.clear();
            if(arcadeEnabled) {
                for(const echo::Enemy &en:arcade.enemies) {
                    if(en.eaten)continue;
                    const bool alert=en.alert>0.0f;
                    const float idleR=en.archetype==0?0.67f:
                                      en.archetype==1?1.0f:0.24f;
                    const float idleG=en.archetype==0?0.21f:
                                      en.archetype==1?0.58f:0.68f;
                    const float idleB=en.archetype==0?0.95f:
                                      en.archetype==1?0.12f:1.0f;
                    orbs.push_back(echo::VisualOrb{en.x,en.y,0.105f,1.12f,
                        arcade.overdrive>0?0.17f:alert?1.0f:idleR,
                        arcade.overdrive>0?0.90f:alert?0.12f:idleG,
                        arcade.overdrive>0?1.0f:alert?0.24f:idleB,
                        arcade.overdrive>0?2.0f:alert?1.6f:0.8f,16});
                }
                for(const echo::Shard &shard:arcade.shards) {
                    if(shard.collected)continue;
                    orbs.push_back(echo::VisualOrb{shard.x,shard.y,0.065f,0.85f,
                        1.0f,0.78f,0.16f,1.1f,8});
                }
                if(lastDecoySimTime>=0 && simTime-lastDecoySimTime<1.2)
                    orbs.push_back(echo::VisualOrb{(float)decoyPulseX,
                        (float)decoyPulseY,0.09f,0.65f,0.24f,0.94f,1.0f,2.0f});
                orbs.push_back(echo::VisualOrb{arcade.exitX,arcade.exitY,0.17f,1.12f,
                    arcade.remaining==0?0.22f:0.62f,
                    arcade.remaining==0?1.0f:0.32f,0.85f,1.7f});
            }
            for(const ToneMarker &tm:g_toneMarkers) {
                if(tm.collected || (arcadeEnabled&&tm.type=='W'))continue;
                const float r=tm.type=='W'?0.1f:0.06f;
                orbs.push_back(echo::VisualOrb{tm.x,tm.y,r,0.95f,
                    tm.type=='W'?0.2f:tm.type=='I'?1.0f:0.9f,
                    tm.type=='W'?1.0f:tm.type=='I'?0.8f:0.3f,
                    tm.type=='W'?0.9f:tm.type=='I'?0.1f:1.0f,0.75f});
            }
            echo::render3D(view3d,full3D?0:SIZE,0,
                           full3D?WINDOW_WIDTH:SIZE,WINDOW_HEIGHT,
                           g_player.pivot_x,g_player.pivot_y,
                           g_player.angle,orbs);
        }
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
    CUDA_CHECK(cudaFree(d_worldEmitters));
    CUDA_CHECK(cudaFree(d_audioL));
    CUDA_CHECK(cudaFree(d_audioR));
    free(h_mask);
    free(h_dynamic);

    echo::release3D(view3d);
    glDeleteBuffers(1, &gfx.glyphVBO);
    glDeleteVertexArrays(1, &gfx.glyphVAO);
    glDeleteBuffers(1, &gfx.vbo);
    glDeleteVertexArrays(1, &gfx.vao);
    glDeleteProgram(gfx.shader);
    SDL_GL_DeleteContext(gfx.gl_context);
    SDL_DestroyWindow(gfx.window);
    SDL_Quit();
    return 0;
}
