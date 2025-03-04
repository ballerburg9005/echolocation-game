#ifndef GRAPHICS_SETUP_H
#define GRAPHICS_SETUP_H

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "render_3d_view.h" // Added for 3D rendering support
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
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
    if (pressure < -9.0)      color = vec3(0, 0, 0);
    else if (pressure < -6.5) color = vec3(1, 1, 0);
    else if (pressure < -5.5) color = vec3(0, 1, 1);
    else if (pressure < -4.5) color = vec3(1, 0, 1);
    else if (pressure < -3.5) color = vec3(0.5, 0, 0.5);
    else if (pressure < -2.5) color = vec3(1, 0, 0);
    else if (pressure < -1.5) color = vec3(0, 1, 0);
    else if (pressure < -0.5) color = vec3(1, 1, 0);
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

static void initGraphics(GraphicsContext &gfx, int WINDOW_WIDTH, int WINDOW_HEIGHT) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) < 0) {
        fprintf(stderr, "SDL_Init error:%s\n", SDL_GetError());
        exit(1);
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    gfx.window = SDL_CreateWindow("FDTD based Echolocation Game Prototype", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);  // Add SDL_WINDOW_RESIZABLE

/*    gfx.window = SDL_CreateWindow("Echolocation, Dual View (SVG)",
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_OPENGL);
*/

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
    glBufferData(GL_ARRAY_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

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

static void cleanupGraphics(GraphicsContext &gfx) {
    glDeleteBuffers(1, &gfx.vbo);
    glDeleteVertexArrays(1, &gfx.vao);
    glDeleteProgram(gfx.shader);
    SDL_GL_DeleteContext(gfx.gl_context);
    SDL_DestroyWindow(gfx.window);
    SDL_Quit();
}

#endif // GRAPHICS_SETUP_H
