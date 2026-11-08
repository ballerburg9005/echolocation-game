#pragma once
// CPU-side 2.5D mesh derived *directly* from the same occupancy mask that
// drives wave propagation. Visual extrusion is NOT a 3D wave solver.
#include <vector>
#include <cmath>
#include <cstddef>

namespace echo {
struct VisualVertex {
    float x,y,z;
    float nx,ny,nz;
    float r,g,b;
};
inline void triangle(std::vector<VisualVertex>& mesh,
                     VisualVertex a,VisualVertex b,VisualVertex c) {
    mesh.push_back(a);mesh.push_back(b);mesh.push_back(c);
}
inline void quad(std::vector<VisualVertex>& mesh,
                 float x0,float y0,float z0, float x1,float y1,float z1,
                 float x2,float y2,float z2, float x3,float y3,float z3,
                 float nx,float ny,float nz,float r,float g,float b) {
    VisualVertex a{x0,y0,z0,nx,ny,nz,r,g,b};
    VisualVertex c{x1,y1,z1,nx,ny,nz,r,g,b};
    VisualVertex d{x2,y2,z2,nx,ny,nz,r,g,b};
    VisualVertex e{x3,y3,z3,nx,ny,nz,r,g,b};
    triangle(mesh,a,c,d);triangle(mesh,a,d,e);
}
// OpenGL uses a right-handed XYZ camera, while grid Y increases towards
// mathematical left. Put grid-Y on negative Z; positive player rotation
// now turns LEFT in both the 2D diagnostic and first-person 3D camera.
inline float visualZFromGridY(float gridY,float cellMetres=0.01f) {
    return -gridY*cellMetres;
}
inline float visualForwardZ(float heading) {
    return -std::sin(heading);
}

inline std::vector<VisualVertex> extrudeMask(const float* mask,int size,
                                             float cellMetres=0.01f,
                                             float heightMetres=2.5f) {
    std::vector<VisualVertex> out;
    if(!mask||size<2||cellMetres<=0||heightMetres<=0)return out;
    const float L=size*cellMetres,H=heightMetres;
    // Dark floor, lighter ceiling. No automatic floor-wall quads at
    // arbitrary room positions, so the maze remains grounded in SVG cells.
    quad(out,0,0,0,L,0,0,L,0,L,0,0,L,0,1,0,0.09f,0.13f,0.18f);
    // Ceiling cannot be coplanar with wall and pillar roofs. It caused
    // z-fighting: dark/brilliant stripes across the tops of round objects.
    const float ceiling=H+0.025f;
    quad(out,0,ceiling,0,0,ceiling,L,L,ceiling,L,L,ceiling,0,
         0,-1,0,0.13f,0.17f,0.22f);
    auto occupied=[&](int x,int y) {
        return x>=0&&y>=0&&x<size&&y<size && mask[y*size+x]>0.0f;
    };
    for(int y=0;y<size;++y) {
        // Merge adjacent top faces by horizontal runs to reduce GPU memory.
        int x=0;
        while(x<size) {
            if(!occupied(x,y)){++x;continue;}
            const int x0=x;
            const bool outer=mask[y*size+x]>=3.5f;
            while(x<size&&occupied(x,y)&&
                  (mask[y*size+x]>=3.5f)==outer)++x;
            const float a=x0*cellMetres,b=x*cellMetres;
            const float z0=y*cellMetres,z1=(y+1)*cellMetres;
            quad(out,a,H,z0,b,H,z0,b,H,z1,a,H,z1,
                 0,1,0,outer?0.20f:0.35f,
                 outer?0.31f:0.51f,outer?0.48f:0.66f);
        }
        for(int x=0;x<size;++x) {
            if(!occupied(x,y))continue;
            const float a=x*cellMetres,b=(x+1)*cellMetres;
            const float c=y*cellMetres,d=(y+1)*cellMetres;
            // Muted blue rubber-like outer ring, distinct from the
            // bright, sharp interior masonry. The same mask material
            // determines the actual FDTD boundary reflectivity.
            const bool outer=mask[y*size+x]>=3.5f;
            if(!occupied(x-1,y))
                quad(out,a,0,c,a,0,d,a,H,d,a,H,c,
                     -1,0,0,outer?0.16f:0.25f,
                     outer?0.32f:0.53f,outer?0.49f:0.65f);
            if(!occupied(x+1,y))
                quad(out,b,0,d,b,0,c,b,H,c,b,H,d,
                     1,0,0,outer?0.16f:0.25f,
                     outer?0.32f:0.53f,outer?0.49f:0.65f);
            if(!occupied(x,y-1))
                quad(out,b,0,c,a,0,c,a,H,c,b,H,c,
                     0,0,-1,outer?0.17f:0.29f,
                     outer?0.34f:0.56f,outer?0.50f:0.68f);
            if(!occupied(x,y+1))
                quad(out,a,0,d,b,0,d,b,H,d,a,H,d,
                     0,0,1,outer?0.17f:0.29f,
                     outer?0.34f:0.56f,outer?0.50f:0.68f);
        }
    }
    // Mirror the horizontal grid axis into a right-handed GL world.
    // Position, normals, camera and moving orbs must transform together.
    for (VisualVertex &v:out) {
        v.z=-v.z;
        v.nz=-v.nz;
    }
    return out;
}
} // namespace echo
