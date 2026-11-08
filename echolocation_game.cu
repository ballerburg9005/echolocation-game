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

// Fused pressure / source / damping kernel. 2 launches rather than 4 per
// 20.6us step removes substantial driver overhead on consumer GPUs.
// This barrier-free stencil also works on partial CUDA blocks (SIZE=500).
__global__ void updatePressure(float* p_next, const float* p, const float* p_prev,
                               const float* mask, float c2, int size, float reflection,
                               int sourceX, int sourceY, float amplitude,
                               unsigned long long step, int interval,
                               int duration, float dt, float decay,
                               bool automaticPulse, uint64_t manualPulseStart)
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
    const bool touchesGhost=(wallL && mask[idx-1]>1.5f) ||
                            (wallR && mask[idx+1]>1.5f) ||
                            (wallD && mask[idx-size]>1.5f) ||
                            (wallU && mask[idx+size]>1.5f);
    const float damping = echo::wallDamping(
        touchesGhost ? 0.82f : reflection, sqrtf(c2), adjacentWalls);
    float next = echo::impedanceWaveStep(center, previous, left, right,
                                        down, up, c2, damping);

    if (x == sourceX && y == sourceY) {
        const int pulseStep=echo::pulseIndex(
            step,interval,duration,automaticPulse,manualPulseStart);
        if(pulseStep>=0)
            next+=amplitude*echo::clickSample(pulseStep,duration,dt);
    }
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
                        if (showObj && mask[idx] > 1.5f) {
                            vbo_data[vbo_idx + 2] = -8.f;
                        } else if (showObj && mask[idx] > 0.f) {
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
                        if (showObj && mask[idx] > 1.5f) {
                            vbo_data[fp_idx + 2] = -8.f;
                        } else if (showObj && mask[idx] > 0.f) {
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
    const float duration=kind==2?0.66f:kind==3?0.60f:kind==1?0.34f:0.20f;
    const int count=(int)(duration*sampleRate);
    std::vector<float> samples(count*2,0.0f);
    for(int i=0;i<count;++i) {
        const float t=(float)i/sampleRate;
        const int note=kind==2?(int)(t/0.22f):
                       kind==3?(int)(t/0.20f):
                       kind==1?(int)(t/0.17f):0;
        const float frequency=kind==2?(note==0?620.0f:note==1?830.0f:1240.0f):
                              kind==3?(note==0?490.0f:note==1?350.0f:210.0f):
                              kind==1?(note==0?350.0f:230.0f):1050.0f;
        const float noteLength=kind==2?0.22f:kind==3?0.20f:kind==1?0.17f:duration;
        const float phase=fmodf(t,noteLength)/noteLength;
        const float envelope=sinf(PI*phase);
        const float v=0.18f*envelope*envelope*sinf(2.0f*PI*frequency*t);
        samples[2*i]=v;samples[2*i+1]=v;
    }
    SDL_QueueAudio(device,samples.data(),(Uint32)(samples.size()*sizeof(float)));
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
    bool dangerAudio=true;
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

    dim3 thr(32, 32);
    dim3 blk((SIZE + 31) / 32, (SIZE + 31) / 32);

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
    bool gameWon = false;
    int collectedItems = 0;
    double lastDebugTime = 0.0;
    unsigned long long simStep2=0;
    uint64_t manualPulseAt=UINT64_MAX;
    bool automaticPulse=true;

    std::puts("[CONTROLS] F2 3D/2D | F3 immersive 3D | F4 arcade on/off | R restart");
    std::puts("[CONTROLS] W/S move | A/D rotate | Q/E strafe | arrows also work");
    std::puts("[CONTROLS] F manually click | T toggle auto-click/stealth | 1/2 interval");
    std::puts("[CONTROLS] F5 toggle enemy danger heartbeat | 3/4 decay | 5/6 reflections | 7/8 volume");

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
                    case SDLK_SPACE: showObjects = !showObjects; break;
                    case SDLK_F2: if(has3D){show3D=!show3D;full3D=false;} break;
                    case SDLK_F3: if(has3D) {
                        show3D=true;full3D=!full3D;
                    } break;
                    case SDLK_F4: arcadeEnabled=!arcadeEnabled; break;
                    case SDLK_F5: dangerAudio=!dangerAudio; break;
                    case SDLK_t:
                        if(!e.key.repeat)automaticPulse=!automaticPulse;
                        break;
                    case SDLK_f:
                        if(!e.key.repeat) {
                            manualPulseAt=simStep2;
                            if(arcadeEnabled)arcade.hearClick(g_pulse_x,g_pulse_y);
                        }
                        break;
                    case SDLK_r:
                        arcade.reset();
                        for(const ToneMarker& tm:g_toneMarkers)if(tm.type=='W'){
                            arcade.setExit(tm.x,tm.y);break;
                        }
                        gameWon=false;collectedItems=0;simStep2=0;simTime=0;
                        manualPulseAt=UINT64_MAX;automaticPulse=true;
                        g_player=Player{50.0f,150.0f,-PI/4.0f};
                        updatePlayerComponents(g_player);
                        for(ToneMarker &tm:g_toneMarkers)tm.collected=false;
                        if(g_numToneMarkers>0)CUDA_CHECK(cudaMemcpy(
                            d_toneMarkers,g_toneMarkers.data(),
                            g_numToneMarkers*sizeof(ToneMarker),cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemset(d_p,0,SIZE*SIZE*sizeof(float)));
                        CUDA_CHECK(cudaMemset(d_p_prev,0,SIZE*SIZE*sizeof(float)));
                        CUDA_CHECK(cudaMemset(d_p_next,0,SIZE*SIZE*sizeof(float)));
                        audioResampler=echo::StereoLinearResampler(g_DT,g_sampleRate);
                        SDL_ClearQueuedAudio(simDev);
                        SDL_ClearQueuedAudio(beepDev);
                        lastBeep=0;lastSlideTime=0;lastThreatCue=0;wasSliding=false;
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
        sliding = motion.slid;
        // Bearing to obstacle in the same heading convention as the funnel.
        const float heading = newP.angle - PI / 4.0f;
        const float collisionAngle = atan2f(motion.wallY, motion.wallX) - heading;

        g_player = newP;
        updatePlayerComponents(g_player);
        echo::ArcadeEvent arcadeEvent;
        if(arcadeEnabled) {
            arcadeEvent=arcade.update(dtFrame,g_player.pivot_x,g_player.pivot_y);
            if(arcade.reachExit(g_player.pivot_x,g_player.pivot_y)) {
                arcadeEvent.cleared=true;
                gameWon=true;
            }
        }
        updateDynamicMask(h_mask, h_dynamic, g_wedge_x, g_wedge_y,
                          g_player.pivot_x, g_player.pivot_y);
        // Moving enemies really alter the *CUDA wave boundary*, not merely
        // the visuals. Static maze remains unchanged for CPU collisions.
        if(arcadeEnabled) arcade.stampEnemies(h_dynamic);
        CUDA_CHECK(cudaMemcpy(d_mask, h_dynamic, SIZE * SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));

        double nowT = now / (double)perfFreq;
        if(arcadeEnabled) {
            if(arcadeEvent.pickups>0)playArcadeFeedback(beepDev,g_sampleRate,0);
            if(arcadeEvent.hit)playArcadeFeedback(beepDev,g_sampleRate,
                                                   arcadeEvent.gameOver?3:1);
            if(arcadeEvent.cleared)playArcadeFeedback(beepDev,g_sampleRate,2);
            if(nowT-lastArcadeMessage>0.35) {
                char title[256];
                std::snprintf(title,sizeof(title),
                    "ECHO ARCADE | %s | Score %d | Lives %d | Shards %d | F2 view F3 immersive R restart",
                    arcade.won?"LEVEL CLEAR":arcade.lost?"GAME OVER":"HUNT THE ECHO",
                    arcade.score,arcade.lives,arcade.remaining);
                SDL_SetWindowTitle(gfx.window,title);
                lastArcadeMessage=nowT;
            }
        }
        // Extra nonphysical audio HUD: nearest approaching ghost produces
        // a low stereo heartbeat that speeds up with proximity. F5 turns it
        // off if the player wants to train with only wave reflections.
        if(arcadeEnabled&&dangerAudio&&!arcade.won&&!arcade.lost) {
            echo::ThreatCue nearest;
            float bestDistance=1.0e9f;
            for(const echo::Enemy &en:arcade.enemies) {
                const float d=std::hypot(en.x-g_player.pivot_x,
                                         en.y-g_player.pivot_y);
                if(d<bestDistance) {
                    bestDistance=d;
                    nearest=echo::threatCue(g_player.pivot_x,
                           g_player.pivot_y,g_player.angle,en.x,en.y,en.alert>0);
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

        if (!wasSliding && collided && nowT - lastBeep > beepGap) {
            float pan = sinf(collisionAngle);
            float lv = 0.5f + 0.5f * pan;
            float rv = 0.5f - 0.5f * pan;
            for (int i = 0; i < beepSamples; i++) {
                float t = (float)i / (float)g_sampleRate;
                float s = sinf(2.f * PI * 440.f * t) * 0.2f; // 32-bit float
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
            for (int i = 0; i < slideSamples; i++) {
                float t = (float)i / (float)g_sampleRate;
                float s = sinf(2.f * PI * 330.f * t) * 0.05f; // 32-bit float
                beepBuf[2 * i] = s * lv;
                beepBuf[2 * i + 1] = s * rv;
            }
            if (SDL_GetQueuedAudioSize(beepDev) <
                (Uint32)(0.15 * g_sampleRate * 2 * sizeof(float)))
                SDL_QueueAudio(beepDev, beepBuf, slideSamples * 2 * sizeof(float));
            lastSlideTime = nowT;
        }

        // Continuous, directional rhythms distinguish obstacle/collectible/goal
        // without relying on the screen. Global clock avoids phase resets.
        const int cueSamples = std::max(0, (int)lroundf(dtFrame * g_sampleRate));
        std::vector<float> markerMix(cueSamples * 2, 0.0f);
        bool markersAudible = false;
        for (int i = 0; i < g_numToneMarkers; ++i) {
            ToneMarker &tm = g_toneMarkers[i];
            if (tm.collected) continue;
            const float deltaX = tm.x - g_player.pivot_x;
            const float deltaY = tm.y - g_player.pivot_y;
            const float dist = sqrtf(deltaX*deltaX + deltaY*deltaY);
            if (dist < 13.0f && tm.type == 'I') {
                tm.collected = true;
                ++collectedItems;
                printf("[GAME] Item collected! Total: %d\n", collectedItems);
                CUDA_CHECK(cudaMemcpy(d_toneMarkers, g_toneMarkers.data(),
                          g_numToneMarkers * sizeof(ToneMarker), cudaMemcpyHostToDevice));
                // Two ascending notes confirm a collectible without vision.
                const int n = (int)(0.35f * g_sampleRate);
                std::vector<float> tune(n*2);
                for (int j = 0; j < n; ++j) {
                    const float within = fmodf((float)j / g_sampleRate, 0.175f);
                    const float freq = j < n/2 ? 650.0f : 900.0f;
                    const float envelope = sinf(PI * within / 0.175f);
                    const float s = 0.16f * envelope * envelope *
                                    sinf(2.0f * PI * freq * within);
                    tune[2*j] = s; tune[2*j+1] = s;
                }
                SDL_QueueAudio(beepDev, tune.data(),
                               (Uint32)(tune.size()*sizeof(float)));
                continue;
            }
            if (dist < 13.0f && tm.type == 'W' && !gameWon &&
                (!arcadeEnabled || arcade.remaining==0)) {
                gameWon = true;
                printf("[GAME] Goal reached! Items collected: %d\n", collectedItems);
                SDL_SetWindowTitle(gfx.window, "Echolocation - Goal reached!");
                const int n = (int)(0.60f * g_sampleRate);
                std::vector<float> tune(n*2);
                for (int j = 0; j < n; ++j) {
                    const int note = (int)((3LL * j) / n);
                    const float within = fmodf((float)j / g_sampleRate, 0.20f);
                    const float envelope = sinf(PI * within / 0.20f);
                    const float freq = note == 0 ? 660.0f :
                                       note == 1 ? 880.0f : 1320.0f;
                    const float s = 0.20f * envelope * envelope *
                                    sinf(2.0f * PI * freq * within);
                    tune[2*j] = s; tune[2*j+1] = s;
                }
                SDL_QueueAudio(beepDev, tune.data(),
                               (Uint32)(tune.size()*sizeof(float)));
            }
            if (dist >= 85.0f) {
                tm.isPlaying = false;
                continue;
            }
            tm.isPlaying = true;
            markersAudible = true;
            const float angle = atan2f(deltaY, deltaX) -
                                (g_player.angle - PI/4.0f);
            const float pan = sinf(angle);
            const float leftGain = sqrtf(0.5f * (1.0f + pan));
            const float rightGain = sqrtf(0.5f * (1.0f - pan));
            const float rangeGain = 0.16f * powf(1.0f - dist / 85.0f, 2.0f);
            const float frequency = tm.type == 'O' ? 360.0f :
                                    tm.type == 'I' ? 720.0f : 1080.0f;
            const int count = tm.type == 'O' ? 1 : tm.type == 'I' ? 2 : 3;
            for (int j = 0; j < cueSamples; ++j) {
                const double t = nowT + (double)j / g_sampleRate;
                const double cycle = fmod(t, 0.75);
                const int event = (int)(cycle / 0.15);
                const double within = cycle - event*0.15;
                if (event >= count || within >= 0.09) continue;
                const float envelope = sinf(PI * (float)(within/0.09));
                const float sample = rangeGain * envelope * envelope *
                               sinf(2.0f * PI * frequency * (float)within);
                markerMix[2*j] += sample * leftGain;
                markerMix[2*j+1] += sample * rightGain;
            }
        }
        if (markersAudible && !markerMix.empty() &&
            SDL_GetQueuedAudioSize(beepDev) <
                (Uint32)(0.10 * g_sampleRate * 2 * sizeof(float))) {
            for (float &sample : markerMix) sample = 0.75f * tanhf(sample);
            SDL_QueueAudio(beepDev, markerMix.data(),
                           (Uint32)(markerMix.size() * sizeof(float)));
        }

        wasSliding = sliding;

        // SDL's output clock, not render-frame delta or an arbitrary fudge
        // factor, determines how many real physical FDTD steps are needed.
        const int queuedFrames = (int)(SDL_GetQueuedAudioSize(simDev) /
                                       (2 * sizeof(float)));
        const int targetFrames = (int)(0.060 * g_sampleRate);
        const int missingFrames = std::max(0, targetFrames - queuedFrames);
        const int steps = std::min(1024, std::min(AUDIO_BATCH,
            (int)ceil((missingFrames + (missingFrames ? 2 : 0)) /
                      (g_sampleRate * (double)g_DT))));

        const int sourceX = (int)roundf(g_pulse_x);
        const int sourceY = (int)roundf(g_pulse_y);
        for (int i = 0; i < steps; ++i) {
            if (arcadeEnabled && automaticPulse && simStep2>=1000ULL &&
                (simStep2-1000ULL)%(unsigned long long)g_pulseIntervalSteps==0ULL)
                arcade.hearClick(g_pulse_x,g_pulse_y);
            updatePressure<<<blk, thr>>>(d_p_next, d_p, d_p_prev, d_mask,
                                         g_c2, SIZE, g_reflectionCoeff,
                                         sourceX, sourceY, PULSE_AMPLITUDE,
                                         simStep2, g_pulseIntervalSteps,
                                         g_pulseDurationSteps, g_DT, g_decayFactor,
                                         automaticPulse,manualPulseAt);
            CUDA_CHECK(cudaGetLastError());

            captureAndReduce<<<1, 1>>>(d_p_next, g_mic_l_x, g_mic_l_y,
                                       g_mic_r_x, g_mic_r_y, audioBatchIndex,
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

        if(manualPulseAt!=UINT64_MAX &&
           simStep2>=manualPulseAt+(uint64_t)g_pulseDurationSteps)
            manualPulseAt=UINT64_MAX;

        if (audioBatchIndex > 0) {
            CUDA_CHECK(cudaMemcpy(h_audioL.data(), d_audioL,
                                  audioBatchIndex * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_audioR.data(), d_audioR,
                                  audioBatchIndex * sizeof(float), cudaMemcpyDeviceToHost));
            h_audioOut.clear();
            for (int k = 0; k < audioBatchIndex; ++k) {
                audioResampler.push(h_audioL[k], h_audioR[k], h_audioOut);
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
                    if(!arcade.won&&!arcade.lost)addGlyph(en.x,en.y,-8.0f);
                for(const echo::Shard &shard:arcade.shards)
                    if(!shard.collected)addGlyph(shard.x,shard.y,-7.0f);
                addGlyph(arcade.exitX,arcade.exitY,-6.0f);
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
                    if(arcade.lost||arcade.won)break;
                    const bool alert=en.alert>0.0f;
                    orbs.push_back(echo::VisualOrb{en.x,en.y,0.105f,1.12f,
                        alert?1.0f:0.55f,alert?0.16f:0.26f,
                        alert?0.25f:0.95f,alert?1.6f:0.8f});
                }
                for(const echo::Shard &shard:arcade.shards) {
                    if(shard.collected)continue;
                    orbs.push_back(echo::VisualOrb{shard.x,shard.y,0.065f,0.85f,
                        1.0f,0.78f,0.16f,1.1f});
                }
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
    CUDA_CHECK(cudaFree(d_audioL));
    CUDA_CHECK(cudaFree(d_audioR));
    free(h_mask);
    free(h_dynamic);
    free(beepBuf);

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
