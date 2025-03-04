#ifndef RENDER_3D_VIEW_H
#define RENDER_3D_VIEW_H

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdio.h>
#include <vector>
#include "graphics_setup.h"

struct Rect { float x, y, width, height; };
struct Circle { float cx, cy, r; };
struct ToneMarker { float x, y; char type; bool isPlaying; };

extern int g_width, g_height;
extern float g_player_pivot_x, g_player_pivot_y, g_player_angle;
extern std::vector<Rect> g_rects;
extern std::vector<Circle> g_circles;
extern std::vector<ToneMarker> g_toneMarkers;

struct Render3DContext {
    GLuint vao[4], vbo[4], shader;
    GLint u_mvp, u_lightPos[5], u_viewPos;
    int numRoomVerts, numRectVerts, numCircleVerts, numToneVerts;
    GLint u_roomWidth;  // Added
    GLint u_roomDepth;  // Added
};

static const char* vertexShader3D = R"(
#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in vec3 normal;
uniform mat4 mvp;
out vec3 vPos;
out vec3 vNormal;
void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    vPos = pos;
    vNormal = normal;
}
)";



static const char* fragmentShader3D = R"(
#version 330 core
in vec3 vPos;
in vec3 vNormal;
uniform vec3 lightPos[5]; // Kept for compatibility, overridden by grid
uniform vec3 viewPos;
uniform float roomWidth;  // New uniform: width in meters (e.g., 5.0)
uniform float roomDepth;  // New uniform: depth in meters (e.g., 5.0)
out vec4 fragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0)), u.x), u.y);
}
float perlin(vec2 p, float freq) {
    float n = 0.0;
    n += 0.5 * noise(p * freq);
    n += 0.25 * noise(p * freq * 2.0);
    n += 0.125 * noise(p * freq * 4.0);
    return n;
}

void main() {
    vec3 baseColor;
    if (vPos.y < 0.1) 
        baseColor = vec3(0.6, 0.2, 0.2);          // Floor: reddish-brown carpet
    else if (vPos.y > 1.9) 
        baseColor = vec3(0.9, 0.9, 0.85);         // Ceiling: off-white
    else if (vPos.y > 0.9 && vPos.y < 1.1 && length(vPos.xz - vec2(roomWidth/2.0, roomDepth/2.0)) < 0.5) { // Tone markers, centered
        if (vPos.x > 0.0 && vPos.x < 0.1)
            baseColor = vec3(0.0, 1.0, 1.0); // O: cyan
        else if (vPos.x > 0.1 && vPos.x < 0.2)
            baseColor = vec3(1.0, 1.0, 0.0); // W: yellow
        else 
            baseColor = vec3(1.0, 0.0, 1.0); // I: magenta
    } 
    else if (vPos.y > 0.1 && vPos.y < 1.9)
        baseColor = vec3(0.8, 0.7, 0.4);         // Walls: soft yellow
    else 
        baseColor = vec3(0.4, 0.6, 0.4);         // Objects: soft green

    // UV mapping
    vec2 uv;
    if (vPos.y < 0.1 || vPos.y > 1.9) {
        uv = vec2(vPos.x, vPos.z);
    } else {
        if (abs(vNormal.x) > abs(vNormal.z))
            uv = vec2(vPos.z, vPos.y);
        else
            uv = vec2(vPos.x, vPos.y);
    }

    // 4x finer bump maps
    float freq = (vPos.y > 0.1 && vPos.y < 1.9) ? 60.0 : 20.0; // 60.0 * 4 and 20.0 * 4
    float texture = perlin(uv, freq);
    vec3 color = baseColor * (1.0 + texture * 0.4);
    float bump = texture * 0.1;
    vec3 norm = normalize(vNormal + vec3(bump - 0.015, bump - 0.015, bump - 0.015));

    // Metallic effect: simple random metallicity based on position
    float metallic = clamp(perlin(vPos.xz * 10.0, 5.0), 0.0, 1.0) * 0.5; // 0.0 to 0.5 range
    vec3 specColor = mix(vec3(1.0), baseColor, 1.0 - metallic); // Metallic surfaces reflect white

    // Dynamic light grid: 8x8 lights, scaled to room size
    vec3 lighting = vec3(0.0);
    float ao = 1.0 - smoothstep(0.0, 0.5, length(vPos - vec3(roomWidth/2.0, 1.0, roomDepth/2.0)) / max(roomWidth, roomDepth));
    const int gridSize = 8;
    float spacingX = roomWidth / float(gridSize);  // e.g., 5.0 / 8 = 0.625
    float spacingZ = roomDepth / float(gridSize);  // e.g., 5.0 / 8 = 0.625
    const float height = 1.8;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            vec3 lightPosGrid = vec3(float(i) * spacingX, height, float(j) * spacingZ);
            vec3 lightDir = normalize(lightPosGrid - vPos);
            vec3 viewDir = normalize(viewPos - vPos);
            vec3 halfway = normalize(lightDir + viewDir);
            float dist = length(lightPosGrid - vPos);
            float atten = 1.0 / (1.0 + 0.06 * dist + 0.015 * dist * dist); // Brighter falloff
            float diff = max(dot(norm, lightDir), 0.0) * atten * 0.04;     // Brighter diffuse
            float spec = pow(max(dot(norm, halfway), 0.0), 64.0) * atten * 0.015 * metallic; // Metallic specular
            lighting += color * diff + specColor * spec;
        }
    }

    // Brighter ambient
    vec3 ambient = color * 0.35;
    color = ambient + lighting * ao;

    fragColor = vec4(color, 1.0);
}
)";


static void init3DRendering(Render3DContext& ctx, int windowWidth, int windowHeight) {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShader3D, NULL);
    glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShader3D, NULL);
    glCompileShader(fs);
    ctx.shader = glCreateProgram();
    glAttachShader(ctx.shader, vs);
    glAttachShader(ctx.shader, fs);
    glLinkProgram(ctx.shader);
    glDeleteShader(vs);
    glDeleteShader(fs);

    // Set up uniform locations
    ctx.u_mvp = glGetUniformLocation(ctx.shader, "mvp");
    char buf[16];
    for (int i = 0; i < 5; i++) {
        snprintf(buf, sizeof(buf), "lightPos[%d]", i);
        ctx.u_lightPos[i] = glGetUniformLocation(ctx.shader, buf);
    }
    ctx.u_viewPos = glGetUniformLocation(ctx.shader, "viewPos");
    ctx.u_roomWidth = glGetUniformLocation(ctx.shader, "roomWidth");  // New uniform for width
    ctx.u_roomDepth = glGetUniformLocation(ctx.shader, "roomDepth");  // New uniform for depth

    // Calculate room dimensions in meters
    float widthMeters = (float)g_width / 100.0f;  // e.g., 500 / 100 = 5.0
    float heightMeters = (float)g_height / 100.0f; // e.g., 500 / 100 = 5.0
    float roomHeight = 2.0f;

    // Set uniform values (shader must be active)
    glUseProgram(ctx.shader);
    glUniform1f(ctx.u_roomWidth, widthMeters);
    glUniform1f(ctx.u_roomDepth, heightMeters);
    glUseProgram(0);  // Reset program state

    // Room geometry (already dynamic with widthMeters, heightMeters, roomHeight)
    float roomVerts[] = {
        0, 0, heightMeters, 0, 1, 0,  widthMeters, 0, heightMeters, 0, 1, 0,  widthMeters, 0, 0, 0, 1, 0,
        0, 0, heightMeters, 0, 1, 0,  widthMeters, 0, 0, 0, 1, 0,  0, 0, 0, 0, 1, 0,
        0, roomHeight, heightMeters, 0, -1, 0,  widthMeters, roomHeight, heightMeters, 0, -1, 0,  widthMeters, roomHeight, 0, 0, -1, 0,
        0, roomHeight, heightMeters, 0, -1, 0,  widthMeters, roomHeight, 0, 0, -1, 0,  0, roomHeight, 0, 0, -1, 0,
        0, 0, heightMeters, 0, 0, 1,  widthMeters, 0, heightMeters, 0, 0, 1,  widthMeters, roomHeight, heightMeters, 0, 0, 1,
        0, 0, heightMeters, 0, 0, 1,  widthMeters, roomHeight, heightMeters, 0, 0, 1,  0, roomHeight, heightMeters, 0, 0, 1,
        0, 0, 0, 0, 0, -1,  widthMeters, 0, 0, 0, 0, -1,  widthMeters, roomHeight, 0, 0, 0, -1,
        0, 0, 0, 0, 0, -1,  widthMeters, roomHeight, 0, 0, 0, -1,  0, roomHeight, 0, 0, 0, -1,
        0, 0, heightMeters, 1, 0, 0,  0, roomHeight, heightMeters, 1, 0, 0,  0, roomHeight, 0, 1, 0, 0,
        0, 0, heightMeters, 1, 0, 0,  0, roomHeight, 0, 1, 0, 0,  0, 0, 0, 1, 0, 0,
        widthMeters, 0, heightMeters, -1, 0, 0,  widthMeters, roomHeight, heightMeters, -1, 0, 0,  widthMeters, roomHeight, 0, -1, 0, 0,
        widthMeters, 0, heightMeters, -1, 0, 0,  widthMeters, roomHeight, 0, -1, 0, 0,  widthMeters, 0, 0, -1, 0, 0
    };
    ctx.numRoomVerts = 36;

    // Rectangle geometry
    std::vector<float> rectVerts;
    for (const auto& r : g_rects) {
        float x0 = r.x / 100.0f, z0 = heightMeters - (r.y + r.height) / 100.0f;
        float x1 = (r.x + r.width) / 100.0f, z1 = heightMeters - r.y / 100.0f;
        float verts[] = {
            x0, 0, z0, 0, 0, 1,  x1, 0, z0, 0, 0, 1,  x1, roomHeight, z0, 0, 0, 1,
            x0, 0, z0, 0, 0, 1,  x1, roomHeight, z0, 0, 0, 1,  x0, roomHeight, z0, 0, 0, 1,
            x0, 0, z1, 0, 0, -1,  x1, 0, z1, 0, 0, -1,  x1, roomHeight, z1, 0, 0, -1,
            x0, 0, z1, 0, 0, -1,  x1, roomHeight, z1, 0, 0, -1,  x0, roomHeight, z1, 0, 0, -1,
            x0, 0, z0, 1, 0, 0,  x0, roomHeight, z0, 1, 0, 0,  x0, roomHeight, z1, 1, 0, 0,
            x0, 0, z0, 1, 0, 0,  x0, roomHeight, z1, 1, 0, 0,  x0, 0, z1, 1, 0, 0,
            x1, 0, z0, -1, 0, 0,  x1, roomHeight, z0, -1, 0, 0,  x1, roomHeight, z1, -1, 0, 0,
            x1, 0, z0, -1, 0, 0,  x1, roomHeight, z1, -1, 0, 0,  x1, 0, z1, -1, 0, 0,
            x0, roomHeight, z0, 0, 1, 0,  x1, roomHeight, z0, 0, 1, 0,  x1, roomHeight, z1, 0, 1, 0,
            x0, roomHeight, z0, 0, 1, 0,  x1, roomHeight, z1, 0, 1, 0,  x0, roomHeight, z1, 0, 1, 0
        };
        rectVerts.insert(rectVerts.end(), verts, verts + 5 * 6 * 6);
    }
    ctx.numRectVerts = rectVerts.size() / 6;

    // Circle geometry
    std::vector<float> circleVerts;
    for (const auto& c : g_circles) {
        float cx = c.cx / 100.0f, cz = heightMeters - c.cy / 100.0f, r = c.r / 100.0f;
        for (int i = 0; i < 12; i++) {
            float a0 = i * 2.0f * M_PI / 12.0f;
            float a1 = (i + 1) * 2.0f * M_PI / 12.0f;
            float x0 = cx + r * cosf(a0), z0 = cz + r * sinf(a0);
            float x1 = cx + r * cosf(a1), z1 = cz + r * sinf(a1);
            float nx0 = cosf(a0), nz0 = sinf(a0);
            float nx1 = cosf(a1), nz1 = sinf(a1);
            float verts[] = {
                x0, 0, z0, nx0, 0, nz0,  x1, 0, z1, nx1, 0, nz1,  x1, roomHeight, z1, nx1, 0, nz1,
                x0, 0, z0, nx0, 0, nz0,  x1, roomHeight, z1, nx1, 0, nz1,  x0, roomHeight, z0, nx0, 0, nz0,
                cx, roomHeight, cz, 0, 1, 0,  x0, roomHeight, z0, 0, 1, 0,  x1, roomHeight, z1, 0, 1, 0,
                cx, 0, cz, 0, -1, 0,  x1, 0, z1, 0, -1, 0,  x0, 0, z0, 0, -1, 0
            };
            circleVerts.insert(circleVerts.end(), verts, verts + 12 * 6);
        }
    }
    ctx.numCircleVerts = circleVerts.size() / 6;

    // Tone marker geometry
    std::vector<float> toneVerts;
    for (const auto& t : g_toneMarkers) {
        float x = t.x / 100.0f, z = heightMeters - t.y / 100.0f;
        float r = 0.1f;
        float y_center = 1.0f;
        for (int i = 0; i < 12; i++) {
            float phi0 = i * 2.0f * M_PI / 12.0f;
            float phi1 = (i + 1) * 2.0f * M_PI / 12.0f;
            for (int j = 0; j < 12; j++) {
                float theta0 = j * M_PI / 12.0f - M_PI / 2.0f;
                float theta1 = (j + 1) * M_PI / 12.0f - M_PI / 2.0f;
                float x00 = x + r * cosf(phi0) * cosf(theta0);
                float z00 = z + r * sinf(phi0) * cosf(theta0);
                float y00 = y_center + r * sinf(theta0);
                float x01 = x + r * cosf(phi0) * cosf(theta1);
                float z01 = z + r * sinf(phi0) * cosf(theta1);
                float y01 = y_center + r * sinf(theta1);
                float x10 = x + r * cosf(phi1) * cosf(theta0);
                float z10 = z + r * sinf(phi1) * cosf(theta0);
                float y10 = y_center + r * sinf(theta0);
                float x11 = x + r * cosf(phi1) * cosf(theta1);
                float z11 = z + r * sinf(phi1) * cosf(theta1);
                float y11 = y_center + r * sinf(theta1);
                float nx00 = cosf(phi0) * cosf(theta0);
                float nz00 = sinf(phi0) * cosf(theta0);
                float ny00 = sinf(theta0);
                float nx01 = cosf(phi0) * cosf(theta1);
                float nz01 = sinf(phi0) * cosf(theta1);
                float ny01 = sinf(theta1);
                float nx10 = cosf(phi1) * cosf(theta0);
                float nz10 = sinf(phi1) * cosf(theta0);
                float ny10 = sinf(theta0);
                float nx11 = cosf(phi1) * cosf(theta1);
                float nz11 = sinf(phi1) * cosf(theta1);
                float ny11 = sinf(theta1);
                float verts[] = {
                    x00, y00, z00, nx00, ny00, nz00,  x10, y10, z10, nx10, ny10, nz10,  x11, y11, z11, nx11, ny11, nz11,
                    x00, y00, z00, nx00, ny00, nz00,  x11, y11, z11, nx11, ny11, nz11,  x01, y01, z01, nx01, ny01, nz01
                };
                toneVerts.insert(toneVerts.end(), verts, verts + 6 * 6);
            }
        }
    }
    ctx.numToneVerts = toneVerts.size() / 6;

    // Buffer setup
    float* buffers[] = { roomVerts, rectVerts.data(), circleVerts.data(), toneVerts.data() };
    int sizes[] = { ctx.numRoomVerts, ctx.numRectVerts, ctx.numCircleVerts, ctx.numToneVerts };
    for (int i = 0; i < 4; i++) {
        glGenVertexArrays(1, &ctx.vao[i]);
        glBindVertexArray(ctx.vao[i]);
        glGenBuffers(1, &ctx.vbo[i]);
        glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo[i]);
        glBufferData(GL_ARRAY_BUFFER, sizes[i] * 6 * sizeof(float), buffers[i], GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
    }
}


static void render3DView(Render3DContext& ctx, int windowWidth, int windowHeight) {
    glViewport(0, g_height, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(ctx.shader);

    float x = g_player_pivot_x / 100.0f;
    float z = (float)g_height / 100.0f - g_player_pivot_y / 100.0f;
    float y = 1.0f;
    glm::mat4 proj = glm::perspective(glm::radians(80.0f), (float)windowWidth / (float)windowHeight, 0.1f, 50.0f);
    proj = glm::scale(proj, glm::vec3(1.0f, 0.5f, 1.0f));
    glm::mat4 view = glm::lookAt(
        glm::vec3(x, y, z),
        glm::vec3(x - sinf(g_player_angle + glm::radians(225.0f)), y, z - cosf(g_player_angle + glm::radians(225.0f))),
        glm::vec3(0, 1, 0)
    );
    glm::mat4 mvp = proj * view;
    glUniformMatrix4fv(ctx.u_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

    glm::vec3 lightPos[5] = {
        glm::vec3(0.5f, 1.5f, 0.5f),
        glm::vec3(0.5f, 1.5f, 4.5f),
        glm::vec3(4.5f, 1.5f, 0.5f),
        glm::vec3(4.5f, 1.5f, 4.5f),
        glm::vec3(2.5f, 1.5f, 2.5f)
    };
    for (int i = 0; i < 5; i++) {
        glUniform3fv(ctx.u_lightPos[i], 1, glm::value_ptr(lightPos[i]));
    }
    glUniform3fv(ctx.u_viewPos, 1, glm::value_ptr(glm::vec3(x, y, z)));

    glBindVertexArray(ctx.vao[0]); // Room
    glDrawArrays(GL_TRIANGLES, 0, ctx.numRoomVerts);
    glBindVertexArray(ctx.vao[1]); // Rects
    glDrawArrays(GL_TRIANGLES, 0, ctx.numRectVerts);
    glBindVertexArray(ctx.vao[2]); // Circles
    glDrawArrays(GL_TRIANGLES, 0, ctx.numCircleVerts);
    glBindVertexArray(ctx.vao[3]); // Tone markers
    glDrawArrays(GL_TRIANGLES, 0, ctx.numToneVerts);
    glDisable(GL_DEPTH_TEST);
}

static void cleanup3DRendering(Render3DContext& ctx) {
    for (int i = 0; i < 4; i++) {
        glDeleteBuffers(1, &ctx.vbo[i]);
        glDeleteVertexArrays(1, &ctx.vao[i]);
    }
    glDeleteProgram(ctx.shader);
}

#endif // RENDER_3D_VIEW_H
