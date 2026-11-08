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
                if (strstr(text, "O")) tm.type = 'O';
                else if (strstr(text, "W")) tm.type = 'W';
                else if (strstr(text, "I")) tm.type = 'I';
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

// No block-level barriers: partial tiles in a 500x500 grid cannot deadlock.
// Solids are excluded from the pressure field; neighboring fluid cells use a
// rigid/impedance ghost pressure, rather than a sign-flipped "wall pressure".
__global__ void updatePressure(float* p_next, const float* p, const float* p_prev,
                               const float* mask, float c2, int size, float reflection)
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
    const float ghost = echo::wallGhost(center, previous, reflection, sqrtf(c2));
    const float left  = mask[idx - 1]    > 0.0f ? ghost : p[idx - 1];
    const float right = mask[idx + 1]    > 0.0f ? ghost : p[idx + 1];
    const float down  = mask[idx - size] > 0.0f ? ghost : p[idx - size];
    const float up    = mask[idx + size] > 0.0f ? ghost : p[idx + size];
    p_next[idx] = echo::waveStep(center, previous, left, right, down, up, c2);
}

__global__ void applyDecay(float* p, float decay)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE * SIZE) {
        p[idx] *= decay;
    }
}

__global__ void addPulse(float* p, const float* mask, float sx, float sy,
                         float amplitude, unsigned long long t,
                         int interval, int duration, float dt)
{
    if (blockIdx.x || threadIdx.x || t < 1000ULL || interval < 1) return;
    const unsigned long long elapsed = t - 1000ULL;
    const int pulseStep = (int)(elapsed % (unsigned long long)interval);
    if (pulseStep >= duration) return;
    const int ix = (int)roundf(sx), iy = (int)roundf(sy);
    if (ix < 1 || ix >= SIZE - 1 || iy < 1 || iy >= SIZE - 1) return;
    const int idx = iy * SIZE + ix;
    if (mask[idx] > 0.0f) return;
    p[idx] += amplitude * echo::clickSample(pulseStep, duration, dt);
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
                    case SDLK_7: g_waveVolumeScale -= 0.002f; recalcDerivedParams(); break;
                    case SDLK_8: g_waveVolumeScale += 0.002f; recalcDerivedParams(); break;
                    case SDLK_SPACE: showObjects = !showObjects; break;
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
        updateDynamicMask(h_mask, h_dynamic, g_wedge_x, g_wedge_y,
                          g_player.pivot_x, g_player.pivot_y);
        CUDA_CHECK(cudaMemcpy(d_mask, h_dynamic, SIZE * SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));

        double nowT = now / (double)perfFreq;
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
            float* slideBuf = (float*)malloc(slideSamples * 2 * sizeof(float));
            for (int i = 0; i < slideSamples; i++) {
                float t = (float)i / (float)g_sampleRate;
                float s = sinf(2.f * PI * 330.f * t) * 0.05f; // 32-bit float
                slideBuf[2 * i] = s * lv;
                slideBuf[2 * i + 1] = s * rv;
            }
            SDL_QueueAudio(beepDev, slideBuf, slideSamples * 2 * sizeof(float));
            free(slideBuf);
            lastSlideTime = nowT;
        }

        // Handle marker beep playback
        for (int i = 0; i < g_numToneMarkers; i++) {
            ToneMarker& tm = g_toneMarkers[i];
            float dx = tm.x - g_player.pivot_x;
            float dy = tm.y - g_player.pivot_y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 50.f) {
                if (!tm.isPlaying) {
                    tm.isPlaying = true;
                    printf("[TONE DEBUG] Started playing marker %c at (%.1f, %.1f), Player at (%.1f, %.1f), Dist=%.1f\n",
                           tm.type, tm.x, tm.y, g_player.pivot_x, g_player.pivot_y, dist);
                }
                float angle = atan2f(dy, dx) - g_player.angle;
                float pan = sinf(angle);
                float atten = fmaxf(0.2f, 1.f - dist / 50.f);
                float freq = (tm.type == 'O') ? 440.f : (tm.type == 'W') ? 660.f : 880.f; // Different freqs for O, W, I
                int samplesToPlay = (int)(dtFrame * g_sampleRate);
                float* markerBuf = (float*)malloc(samplesToPlay * 2 * sizeof(float));
                for (int j = 0; j < samplesToPlay; j++) {
                    float t = (float)j / (float)g_sampleRate;
                    float sample = sinf(2.f * PI * freq * t) * 0.2f * atten; // 32-bit float
                    markerBuf[2 * j] = sample * (0.5f + 0.5f * pan);
                    markerBuf[2 * j + 1] = sample * (0.5f - 0.5f * pan);
                }
                SDL_QueueAudio(beepDev, markerBuf, samplesToPlay * 2 * sizeof(float));
                free(markerBuf);
            } else {
                if (tm.isPlaying) {
                    tm.isPlaying = false;
                    printf("[TONE DEBUG] Stopped marker %c at (%.1f, %.1f), Player at (%.1f, %.1f), Dist=%.1f (too far)\n",
                           tm.type, tm.x, tm.y, g_player.pivot_x, g_player.pivot_y, dist);
                }
            }
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

        static unsigned long long simStep2 = 0;
        for (int i = 0; i < steps; ++i) {
            updatePressure<<<blk, thr>>>(d_p_next, d_p, d_p_prev, d_mask,
                                         g_c2, SIZE, g_reflectionCoeff);
            CUDA_CHECK(cudaGetLastError());

            addPulse<<<1, 1>>>(d_p_next, d_mask, g_pulse_x, g_pulse_y,
                              PULSE_AMPLITUDE, simStep2, g_pulseIntervalSteps,
                              g_pulseDurationSteps, g_DT);
            CUDA_CHECK(cudaGetLastError());

            applyDecay<<<blocks1d, threads1d>>>(d_p_next, g_decayFactor);
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
                sample = 0.8f * tanhf(sample);
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

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(gfx.shader);
        glBindVertexArray(gfx.vao);
        glDrawArrays(GL_POINTS, 0, SIZE * SIZE * 2);
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

    glDeleteBuffers(1, &gfx.vbo);
    glDeleteVertexArrays(1, &gfx.vao);
    glDeleteProgram(gfx.shader);
    SDL_GL_DeleteContext(gfx.gl_context);
    SDL_DestroyWindow(gfx.window);
    SDL_Quit();
    return 0;
}
