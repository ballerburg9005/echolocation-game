#pragma once
// Restored immersive 3D *visualization* inspired by experimental/render_3d_view.h.
// Unlike that branch this renderer uses actual rasterized 2D acoustic walls,
// so every visible wall aligns with the FDTD collision/reflector mask.
// This is 2.5D extrusion, not a 3D acoustic solver.
#include "visual_mesh.h"
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

namespace echo {
struct VisualOrb {
    float x,y;           // FDTD grid coordinates (centimetres)
    float radius=0.08f;   // metres
    float height=0.85f;   // vertical centre above floor (metres)
    float r=1,g=1,b=1;
    float glow=0.7f;
};
struct View3D {
    GLuint program=0;
    GLuint wallsVAO=0,wallsVBO=0;
    GLuint sphereVAO=0,sphereVBO=0;
    GLsizei wallVertices=0,sphereVertices=0;
    GLint uVP=-1,uOffset=-1,uScale=-1,uTint=-1,uGlow=-1,uEye=-1;
    bool ready=false;
};

inline const char* viewVertexShader() {return R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
uniform mat4 uViewProjection;
uniform vec3 uOffset;
uniform float uScale;
uniform vec3 uTint;
out vec3 vWorld;
out vec3 vNormal;
out vec3 vColor;
void main() {
    vWorld = aPos*uScale + uOffset;
    vNormal = normalize(aNormal);
    vColor = aColor*uTint;
    gl_Position = uViewProjection*vec4(vWorld,1.0);
}
)GLSL";}
inline const char* viewFragmentShader(){return R"GLSL(
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
uniform vec3 uEye;
uniform float uGlow;
out vec4 fragColor;
float hash31(vec3 p){
    p=fract(p*0.1031);
    p+=dot(p,p.yzx+33.33);
    return fract((p.x+p.y)*p.z);
}
void main(){
    vec3 n=normalize(vNormal);
    vec3 toEye=uEye-vWorld;
    float dist=max(length(toEye),0.001);
    vec3 lightDir=toEye/dist;
    float diffuse=max(dot(n,lightDir),0.0);
    float falloff=1.0/(1.0+0.22*dist*dist);
    float fine=hash31(floor(vWorld*26.0));
    float textureFactor=mix(0.92,1.12,fine);
    vec3 base=vColor*textureFactor;
    // Readable floor perspective grid without textures or shadow maps.
    if(abs(n.y)>0.95 && vWorld.y<0.04){
        vec2 grid=abs(fract(vWorld.xz*2.0)-0.5);
        float line=1.0-step(0.02,min(grid.x,grid.y));
        base=mix(base,base*1.50,line*0.35);
    }
    float spec=pow(max(dot(reflect(-lightDir,n),lightDir),0.0),24.0);
    vec3 lit=base*(0.32+1.65*diffuse*falloff) +
             vec3(0.09,0.18,0.28)*spec*falloff +
             base*uGlow;
    vec3 fogColor=vec3(0.015,0.026,0.050);
    float fog=clamp((dist-1.6)/7.5,0.0,0.88);
    vec3 result=mix(lit,fogColor,fog);
    fragColor=vec4(pow(clamp(result,0.0,1.0),vec3(0.9)),1.0);
}
)GLSL";}

inline GLuint compile3DShader(GLenum type,const char* source){
    GLuint shader=glCreateShader(type);
    glShaderSource(shader,1,&source,nullptr);
    glCompileShader(shader);
    GLint ok=0;glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    if(!ok) {
        char buf[2048]={0};
        glGetShaderInfoLog(shader,sizeof(buf),nullptr,buf);
        std::fprintf(stderr,"[3D] Shader compilation failed: %s\n",buf);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}
inline void upload3DMesh(GLuint& vao,GLuint& vbo,
                         const std::vector<VisualVertex>& vertices) {
    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,
        (GLsizeiptr)(vertices.size()*sizeof(VisualVertex)),
        vertices.empty()?nullptr:vertices.data(),GL_STATIC_DRAW);
    const GLsizei stride=sizeof(VisualVertex);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,stride,(void*)0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,stride,(void*)(3*sizeof(float)));
    glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,stride,(void*)(6*sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}
inline std::vector<VisualVertex> orbMesh(){
    std::vector<VisualVertex> v;
    const int around=18,bands=10;
    for(int y=0;y<bands;++y)for(int x=0;x<around;++x){
        const float p0=3.14159265358979323846f*(float)y/bands;
        const float p1=3.14159265358979323846f*(float)(y+1)/bands;
        const float a0=6.28318530717958647692f*(float)x/around;
        const float a1=6.28318530717958647692f*(float)(x+1)/around;
        auto vertex=[](float a,float p) {
            float nx=std::sin(p)*std::cos(a);
            float ny=std::cos(p);
            float nz=std::sin(p)*std::sin(a);
            return VisualVertex{nx,ny,nz,nx,ny,nz,1,1,1};
        };
        const auto a=vertex(a0,p0),b=vertex(a0,p1);
        const auto c=vertex(a1,p1),d=vertex(a1,p0);
        triangle(v,a,b,c);
        triangle(v,a,c,d);
    }
    return v;
}
inline bool init3D(View3D& out,const float* mask,int size){
    GLuint vs=compile3DShader(GL_VERTEX_SHADER,viewVertexShader());
    GLuint fs=compile3DShader(GL_FRAGMENT_SHADER,viewFragmentShader());
    if(!vs||!fs){if(vs)glDeleteShader(vs);if(fs)glDeleteShader(fs);return false;}
    out.program=glCreateProgram();
    glAttachShader(out.program,vs);
    glAttachShader(out.program,fs);
    glLinkProgram(out.program);
    glDeleteShader(vs);glDeleteShader(fs);
    GLint ok=0;glGetProgramiv(out.program,GL_LINK_STATUS,&ok);
    if(!ok){
        char buf[2048]={0};
        glGetProgramInfoLog(out.program,sizeof(buf),nullptr,buf);
        std::fprintf(stderr,"[3D] Shader linking failed: %s\n",buf);
        glDeleteProgram(out.program);out.program=0;return false;
    }
    out.uVP=glGetUniformLocation(out.program,"uViewProjection");
    out.uOffset=glGetUniformLocation(out.program,"uOffset");
    out.uScale=glGetUniformLocation(out.program,"uScale");
    out.uTint=glGetUniformLocation(out.program,"uTint");
    out.uGlow=glGetUniformLocation(out.program,"uGlow");
    out.uEye=glGetUniformLocation(out.program,"uEye");
    const auto walls=extrudeMask(mask,size);
    const auto spheres=orbMesh();
    out.wallVertices=(GLsizei)walls.size();
    out.sphereVertices=(GLsizei)spheres.size();
    upload3DMesh(out.wallsVAO,out.wallsVBO,walls);
    upload3DMesh(out.sphereVAO,out.sphereVBO,spheres);
    out.ready=true;
    std::printf("[3D] Raster-extruded maze: %d triangles, orbs: %d triangles\n",
                out.wallVertices/3,out.sphereVertices/3);
    return true;
}
inline void render3D(const View3D& view,int viewportX,int viewportY,
                     int viewportWidth,int viewportHeight,
                     float playerX,float playerY,float angle,
                     const std::vector<VisualOrb>& orbs) {
    if(!view.ready||viewportWidth<=0||viewportHeight<=0)return;
    glViewport(viewportX,viewportY,viewportWidth,viewportHeight);
    glEnable(GL_SCISSOR_TEST);
    glScissor(viewportX,viewportY,viewportWidth,viewportHeight);
    glClearColor(0.012f,0.025f,0.044f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    // The player's actual acoustic forward vector points at angle - 45 deg.
    const float bearing=angle-3.14159265358979323846f/4.0f;
    const glm::vec3 camera(playerX*.01f,1.17f,playerY*.01f);
    const glm::vec3 look(camera.x+std::cos(bearing),camera.y,
                         camera.z+std::sin(bearing));
    const glm::mat4 projection=glm::perspective(glm::radians(77.0f),
        (float)viewportWidth/viewportHeight,0.025f,40.0f);
    const glm::mat4 viewMatrix=glm::lookAt(camera,look,glm::vec3(0,1,0));
    const glm::mat4 vp=projection*viewMatrix;
    glUseProgram(view.program);
    glUniformMatrix4fv(view.uVP,1,GL_FALSE,glm::value_ptr(vp));
    glUniform3fv(view.uEye,1,glm::value_ptr(camera));
    glUniform3f(view.uOffset,0,0,0);
    glUniform1f(view.uScale,1.0f);
    glUniform3f(view.uTint,1,1,1);
    glUniform1f(view.uGlow,0);
    glBindVertexArray(view.wallsVAO);
    glDrawArrays(GL_TRIANGLES,0,view.wallVertices);

    glBindVertexArray(view.sphereVAO);
    for(const VisualOrb& orb:orbs){
        glUniform3f(view.uOffset,orb.x*.01f,orb.height,orb.y*.01f);
        glUniform1f(view.uScale,orb.radius);
        glUniform3f(view.uTint,orb.r,orb.g,orb.b);
        glUniform1f(view.uGlow,orb.glow);
        glDrawArrays(GL_TRIANGLES,0,view.sphereVertices);
    }
    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
}
inline void release3D(View3D& view) {
    if(view.wallsVAO)glDeleteVertexArrays(1,&view.wallsVAO);
    if(view.wallsVBO)glDeleteBuffers(1,&view.wallsVBO);
    if(view.sphereVAO)glDeleteVertexArrays(1,&view.sphereVAO);
    if(view.sphereVBO)glDeleteBuffers(1,&view.sphereVBO);
    if(view.program)glDeleteProgram(view.program);
    view=View3D{};
}
} // namespace echo
