#pragma once
#include "asmr_lab_math.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Exact user-provided millimetre SVG contours, including Inkscape
// translation/rotation and relative elliptical arcs / cubic Beziers.
// This CPU geometry is rasterized into the TRUE 2D FDTD material field.
namespace asmr::svg {
struct Shape {
    std::vector<std::vector<Vec2>> loops;
    float viewX=0,viewY=0,widthMm=0,heightMm=0;
    bool valid()const{return !loops.empty()&&widthMm>0&&heightMm>0;}
};
inline std::string attribute(const std::string& tag,const std::string& name){
    // Match an entire XML attribute name. Searching just "d=" also
    // matches the suffix of "id=", silently losing every SVG path.
    const std::string key=" "+name+"=";
    const auto found=tag.find(key);
    if(found==std::string::npos)return {};
    size_t i=found+key.size();
    if(i>=tag.size())return {};
    const char quote=tag[i++];
    if(quote!='\''&&quote!='"')return {};
    const auto j=tag.find(quote,i);
    return j==std::string::npos?std::string():tag.substr(i,j-i);
}
inline float transformNumber(const std::string& value,
                             const std::string& command,int index=0){
    const auto start=value.find(command+"(");
    if(start==std::string::npos)return 0;
    const auto end=value.find(')',start);
    if(end==std::string::npos)return 0;
    std::string nums=value.substr(start+command.size()+1,
                                  end-start-command.size()-1);
    std::replace(nums.begin(),nums.end(),',',' ');
    std::istringstream in(nums);
    float f=0;
    for(int k=0;k<=index;++k)if(!(in>>f))return 0;
    return f;
}
inline bool getNumber(const char*& p,float& value){
    while(*p==','||*p==' '||*p=='\t'||*p=='\r'||*p=='\n')++p;
    if(!*p||((*p<'0'||*p>'9')&&*p!='-'&&*p!='+'&&*p!='.'))
        return false;
    char* end=nullptr;
    value=std::strtof(p,&end);
    if(end==p || !std::isfinite(value))return false;
    p=end;return true;
}
inline void appendArc(std::vector<Vec2>& points,Vec2 a,Vec2 b,
                      float rx,float ry,float degrees,bool large,bool sweep){
    rx=std::fabs(rx);ry=std::fabs(ry);
    if(rx<1e-6f||ry<1e-6f||length(sub(a,b))<1e-5f){
        points.push_back(b);return;
    }
    const double phi=degrees*echo::kPi/180.0;
    const double co=std::cos(phi),si=std::sin(phi);
    const double dx=(a.x-b.x)/2.0,dy=(a.y-b.y)/2.0;
    const double xx=co*dx+si*dy,yy=-si*dx+co*dy;
    double rX=rx,rY=ry;
    const double lambda=xx*xx/(rX*rX)+yy*yy/(rY*rY);
    if(lambda>1.0){const double k=std::sqrt(lambda);rX*=k;rY*=k;}
    const double denom=rX*rX*yy*yy+rY*rY*xx*xx;
    const double factor=denom>1e-16?
        (large==sweep?-1.0:1.0)*std::sqrt(std::max(0.0,
        (rX*rX*rY*rY-denom)/denom)):0.0;
    const double cxp=factor*rX*yy/rY;
    const double cyp=-factor*rY*xx/rX;
    const double cx=co*cxp-si*cyp+(a.x+b.x)/2.0;
    const double cy=si*cxp+co*cyp+(a.y+b.y)/2.0;
    const double ux=(xx-cxp)/rX,uy=(yy-cyp)/rY;
    const double vx=(-xx-cxp)/rX,vy=(-yy-cyp)/rY;
    const double first=std::atan2(uy,ux);
    double delta=std::atan2(ux*vy-uy*vx,ux*vx+uy*vy);
    if(!sweep&&delta>0)delta-=2.0*echo::kPi;
    if(sweep&&delta<0)delta+=2.0*echo::kPi;
    const int steps=std::clamp((int)std::ceil(
        std::fabs(delta)*std::max(rX,rY)/0.6),2,700);
    for(int j=1;j<=steps;++j){
        const double angle=first+delta*j/steps;
        const double u=rX*std::cos(angle),v=rY*std::sin(angle);
        points.push_back({(float)(co*u-si*v+cx),
                          (float)(si*u+co*v+cy)});
    }
    points.back()=b;
}
inline std::vector<Vec2> parsePath(const std::string& path){
    std::vector<Vec2> points;
    const char* p=path.c_str();
    char cmd=0;
    Vec2 here{},start{};
    for(int iter=0;*p&&iter<5000;++iter){
        while(*p==','||*p==' '||*p=='\n'||*p=='\t'||*p=='\r')++p;
        if(!*p)break;
        if((*p>='A'&&*p<='Z')||(*p>='a'&&*p<='z')){
            cmd=*p++;
            if(cmd=='z'||cmd=='Z'){
                if(!points.empty())points.push_back(start);
                here=start;cmd=0;
            }
            continue;
        }
        const bool rel=cmd>='a'&&cmd<='z';
        const char c=rel?char(cmd-'a'+'A'):cmd;
        float x=0,y=0,a=0,b=0,r=0,s=0,t=0;
        const auto pair=[&](float& u,float& v){
            return getNumber(p,u)&&getNumber(p,v);
        };
        if(c=='M'||c=='L'){
            if(!pair(x,y))break;
            here=rel?add(here,{x,y}):Vec2{x,y};
            if(c=='M'){start=here;cmd=rel?'l':'L';}
            points.push_back(here);
        }else if(c=='C'){
            if(!pair(x,y)||!pair(a,b)||!pair(r,s))break;
            const Vec2 p0=here;
            const Vec2 p1=rel?add(p0,{x,y}):Vec2{x,y};
            const Vec2 p2=rel?add(p0,{a,b}):Vec2{a,b};
            const Vec2 p3=rel?add(p0,{r,s}):Vec2{r,s};
            const float total=length(sub(p1,p0))+length(sub(p2,p1))+
                              length(sub(p3,p2));
            const int count=std::clamp((int)std::ceil(total/0.6f),4,500);
            for(int k=1;k<=count;++k){
                const float q=(float)k/count,u=1.0f-q;
                points.push_back({u*u*u*p0.x+3*u*u*q*p1.x+
                    3*u*q*q*p2.x+q*q*q*p3.x,
                    u*u*u*p0.y+3*u*u*q*p1.y+
                    3*u*q*q*p2.y+q*q*q*p3.y});
            }
            here=p3;
        }else if(c=='A'){
            if(!getNumber(p,x)||!getNumber(p,y)||!getNumber(p,a)||
               !getNumber(p,b)||!getNumber(p,r)||!getNumber(p,s)||
               !getNumber(p,t))break;
            const Vec2 to=rel?add(here,{s,t}):Vec2{s,t};
            appendArc(points,here,to,x,y,a,b>0.5f,r>0.5f);
            here=to;
        }else if(c=='H'||c=='V'){
            if(!getNumber(p,x))break;
            if(c=='H')here.x=rel?here.x+x:x;
            else here.y=rel?here.y+x:x;
            points.push_back(here);
        }else{
            break; // refuse unsupported SVG path command
        }
    }
    return points;
}
inline Shape load(const std::string& filename){
    Shape shape;
    std::ifstream file(filename);
    if(!file)return shape;
    const std::string text((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    const auto open=text.find("<svg"),end=text.find('>',open);
    if(open==std::string::npos||end==std::string::npos)return shape;
    std::istringstream vb(attribute(text.substr(open,end-open),"viewBox"));
    if(!(vb>>shape.viewX>>shape.viewY>>shape.widthMm>>shape.heightMm)||
       !std::isfinite(shape.widthMm)||!std::isfinite(shape.heightMm)||
       shape.widthMm<20||shape.heightMm<20||
       shape.widthMm>300||shape.heightMm>300)return {};
    const auto group=text.find("<g");
    const auto endG=text.find('>',group);
    const auto groupTransform=group!=std::string::npos&&
        endG!=std::string::npos?
        attribute(text.substr(group,endG-group),"transform"):"";
    const float tx=transformNumber(groupTransform,"translate");
    const float ty=transformNumber(groupTransform,"translate",1);
    size_t cursor=0;
    while((cursor=text.find("<path",cursor))!=std::string::npos){
        const auto endTag=text.find('>',cursor);
        if(endTag==std::string::npos)break;
        const auto tag=text.substr(cursor,endTag-cursor);
        auto points=parsePath(attribute(tag,"d"));
        const float deg=transformNumber(attribute(tag,"transform"),
                                        "rotate")*echo::kPi/180.0f;
        const float co=std::cos(deg),si=std::sin(deg);
        for(Vec2& point:points){
            const float x=point.x,y=point.y;
            point={co*x-si*y+tx,si*x+co*y+ty};
        }
        if(points.size()>=4)shape.loops.push_back(std::move(points));
        cursor=endTag+1;
    }
    return shape;
}
inline std::vector<std::vector<Vec2>> inWorld(
    const Shape& shape,Vec2 centre,float faceAngle=0.0f,
    bool artist=false){
    std::vector<std::vector<Vec2>> out;
    out.reserve(shape.loops.size());
    // Speaker's SVG recess is at the bottom of its viewBox.
    const float angle=artist?faceAngle+echo::kPi/2.0f:0.0f;
    const float co=std::cos(angle),si=std::sin(angle);
    for(const auto& poly:shape.loops){
        auto& world=out.emplace_back();
        world.reserve(poly.size());
        for(const Vec2 point:poly){
            const float x=(point.x-shape.viewX-shape.widthMm*0.5f)*.001f;
            const float y=-(point.y-shape.viewY-shape.heightMm*0.5f)*.001f;
            world.push_back({centre.x+co*x-si*y,centre.y+si*x+co*y});
        }
    }
    return out;
}
inline void rasterPolygons(std::vector<float>& mask,int n,float dx,
                          const std::vector<std::vector<Vec2>>& polys,
                          float material=1.0f){
    if((size_t)n*n!=mask.size())return;
    for(const auto& poly:polys){
        if(poly.size()<3)continue;
        float lowest=1000,highest=-1000;
        for(const auto& pt:poly){
            lowest=std::min(lowest,pt.y);
            highest=std::max(highest,pt.y);
        }
        const int minY=std::max(0,(int)std::floor(lowest/dx-.5f));
        const int maxY=std::min(n-1,(int)std::ceil(highest/dx-.5f));
        std::vector<float> crossings;crossings.reserve(64);
        for(int y=minY;y<=maxY;++y){
            crossings.clear();
            const float yy=(y+.5f)*dx;
            for(size_t i=0;i<poly.size();++i){
                const auto a=poly[i],b=poly[(i+1)%poly.size()];
                if((a.y<=yy&&b.y>yy)||(b.y<=yy&&a.y>yy))
                    crossings.push_back(a.x+(yy-a.y)*
                                        (b.x-a.x)/(b.y-a.y));
            }
            std::sort(crossings.begin(),crossings.end());
            for(size_t k=1;k<crossings.size();k+=2){
                const int x0=std::max(0,(int)std::ceil(
                        crossings[k-1]/dx-.5f));
                const int x1=std::min(n-1,(int)std::floor(
                        crossings[k]/dx-.5f));
                for(int x=x0;x<=x1;++x)
                    mask[(size_t)y*n+x]=material;
            }
        }
    }
}
inline std::vector<float> listenerMask(int n,float dx,
                                      const Shape& model,bool canals){
    auto mask=roomMask(n,dx);
    if(!model.valid())return mask;
    for(float& value:mask)if(value==1.0f)value=0.0f;
    rasterPolygons(mask,n,dx,inWorld(model,listener()),1.0f);
    // Repair only a possible sub-3mm vertical seam between the two
    // independently exported SVG halves. Never bridge side canals.
    if(canals){
        const int mid=(int)(kListenerX/dx);
        const int half=std::max(1,(int)std::ceil(.002f/dx));
        for(int y=0;y<n;++y){
            const float wy=(y+.5f)*dx;
            if(std::fabs(wy-kListenerY)>.049f||
               mid-half-1<0||mid+half+1>=n)continue;
            if(mask[(size_t)y*n+mid-half-1]>0.0f&&
               mask[(size_t)y*n+mid+half+1]>0.0f)
                for(int x=mid-half;x<=mid+half;++x)
                    mask[(size_t)y*n+x]=1.0f;
        }
    }
    // Ear mics must remain in an AIR pocket, not inside hard cells.
    // Millimetre-width lateral channels lead from the external notches
    // to precise kid-sized microphones; never carve the whole centre.
    const float half=0.5f*(canals?kEardrumSeparation:kOuterMicSeparation);
    for(int sign:{-1,1}){
        const float a=kListenerX+sign*(half-2.0f*dx);
        const float b=kListenerX+sign*(model.widthMm*.0005f+.008f);
        const int x0=std::max(0,(int)std::floor(std::min(a,b)/dx-.5f));
        const int x1=std::min(n-1,(int)std::ceil(std::max(a,b)/dx-.5f));
        const int y0=std::max(0,(int)std::floor(
            (kListenerY-std::max(.0035f,1.5f*dx))/dx-.5f));
        const int y1=std::min(n-1,(int)std::ceil(
            (kListenerY+std::max(.0035f,1.5f*dx))/dx-.5f));
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x)
            if(mask[(size_t)y*n+x]==1.0f)
                mask[(size_t)y*n+x]=0.0f;
    }
    return mask;
}
constexpr int kSpeakerTexture=192;
constexpr float kSpeakerTexel=.001f;
inline std::vector<float> speakerTexture(const Shape& model){
    std::vector<float> raster((size_t)kSpeakerTexture*
                             kSpeakerTexture,0.0f);
    if(model.valid())rasterPolygons(raster,kSpeakerTexture,
        kSpeakerTexel,inWorld(model,
            {kSpeakerTexture*kSpeakerTexel*.5f,
             kSpeakerTexture*kSpeakerTexel*.5f}),1.0f);
    return raster;
}
} // namespace asmr::svg
