// Standalone ASMR room experiment. No changes to the arcade executable.
// Source is always ~/test.mp3: decoded into mono PCM inside the real FDTD room.
// Listener head is a REAL stationary 2D FDTD obstruction with two ears.
// Artist is a collision-safe moving visual head; its mouth is a real
// moving FDTD pressure emitter facing the listener on every path sample.
// Live: 5mm grid, 5.5kHz band (Q cycles 5/8/10mm for weaker GPUs).
// Offline: 2mm grid, 15kHz band to preserve >8kHz spatial effects, BEFORE
// stutter-free 48kHz stereo WAV playback and writing an output file.
#include "asmr_lab_math.h"
#include "asmr_svg_geometry.h"
#include "asmr_mp3_audio.h"
#include "source_model.h"
#include <cuda_runtime.h>
#include <SDL2/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#define LAB_CUDA(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
    std::fprintf(stderr,"ASMR CUDA %s:%d: %s\n",__FILE__,__LINE__, \
                 cudaGetErrorString(err));std::exit(2);}}while(0)

static constexpr int kWindow=800;
static constexpr int kMaxWavSeconds=60;
static constexpr int kMaxWavFrames=asmr::kOutputRate*kMaxWavSeconds;
static constexpr float kInputStrength=3.0f;
static constexpr int kPressureImage=800; // full screen, no smoothing upscaled 400px tiles


// At the edge of the sculpted ear recess, a bilinear microphone's
// surrounding lattice can include some SOLID pressure cells. Weight
// only AIR neighbors: otherwise the stereo gain changes with dx,
 // creating spurious ear asymmetry even at fixed physical spacing.
__device__ float sampleEarAir(const float* pressure,const float* mask,
                              int n,float gx,float gy) {
    const int x=(int)floorf(gx),y=(int)floorf(gy);
    if(x<1||y<1||x+1>=n-1||y+1>=n-1)return 0.0f;
    float accum=0.0f,total=0.0f;
    for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox){
        const int index=(y+oy)*n+x+ox;
        if(mask[index]>0.0f)continue;
        const float weight=(1.0f-fabsf(gx-(float)(x+ox)))*
                           (1.0f-fabsf(gy-(float)(y+oy)));
        accum+=weight*pressure[index];
        total+=weight;
    }
    return total>0.001f?accum/total:0.0f;
}
__device__ float roomWaveSample(const float* pcm,int length,double seconds) {
    if(length<2||seconds<0.0)return 0.0f;
    const double absolute=seconds*(double)asmr::kOutputRate;
    double f=fmod(absolute,(double)length);
    if(f<0.0)f+=(double)length;
    const int i=(int)f;
    const int j=i+1<length?i+1:0;
    const float alpha=(float)(f-(double)i);
    const float normal=pcm[i]*(1.0f-alpha)+pcm[j]*alpha;
    // Do not insert a harsh loop click if a short test.wav ends abruptly.
    constexpr int crossfade=960; // 20ms at 48kHz
    if(length<crossfade*2 || f<length-crossfade)return normal;
    const float fraction=(float)((f-(length-crossfade))/crossfade);
    const float atStart=(float)(f-(length-crossfade));
    const int a=(int)atStart;
    const float local=atStart-a;
    const float beginning=pcm[a]*(1.0f-local)+pcm[a+1]*local;
    return normal*(1.0f-fraction)+beginning*fraction;
}

__global__ void asmrPressureStep(
    float* next,const float* now,const float* previous,
    const float* mask,const float* pcm,const int* pcmLength,
    const float2* mouthLocations,const uint64_t* blockBase,
    float* outStereo,int localStep,int n,float dx,float dt,float decay,
    float halfMicSeparation,float micYOffset,double absoluteStart,
    double recordingDuration,bool offline)
{
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=n||y>=n)return;
    const int idx=y*n+x;
    // Read the preceding, completed pressure field at BOTH ear points.
    // Unlike separate output kernels, this doesn't double graph launches.
    // The p field is read-only for the entirety of this CUDA kernel.
    if(x==0&&y==0){
        const float lx=(asmr::kListenerX-halfMicSeparation)/dx-0.5f;
        const float rx=(asmr::kListenerX+halfMicSeparation)/dx-0.5f;
        const float ey=(asmr::kListenerY+micYOffset)/dx-0.5f;
        outStereo[localStep*2]=sampleEarAir(now,mask,n,lx,ey);
        outStereo[localStep*2+1]=sampleEarAir(now,mask,n,rx,ey);
    }
    if(x==0||y==0||x==n-1||y==n-1||mask[idx]>0.0f) {
        next[idx]=0.0f;return;
    }
    const float center=now[idx];
    const float l=mask[idx-1]>0.0f?center:now[idx-1];
    const float r=mask[idx+1]>0.0f?center:now[idx+1];
    const float d=mask[idx-n]>0.0f?center:now[idx-n];
    const float u=mask[idx+n]>0.0f?center:now[idx+n];
    const float damping=echo::mixedWallDamping(
        mask[idx-1],mask[idx+1],mask[idx-n],mask[idx+n],
        asmr::kHeadReflection,echo::kCourant);
    float pressure=echo::impedanceWaveStep(
        center,previous[idx],l,r,d,u,echo::kCourantSquared,damping);

    const float2 mouth=mouthLocations[localStep];
    // Only the 2x2 source footprint evaluates WAV interpolation.
    if(fabsf((float)x-mouth.x)<1.0f&&
       fabsf((float)y-mouth.y)<1.0f) {
        const double elapsed=(double)(*blockBase+(uint64_t)localStep)*dt;
        if(!offline||elapsed<recordingDuration) {
            const float amplitude=roomWaveSample(
                pcm,*pcmLength,absoluteStart+elapsed);
            float envelope=1.0f;
            if(offline) {
                const float fadeIn=fminf(1.0f,(float)(elapsed/0.035));
                const float fadeOut=fminf(1.0f,(float)(
                    (recordingDuration-elapsed)/0.06));
                envelope=fmaxf(0.0f,fminf(fadeIn,fadeOut));
            }
            // Unit-sum bilinear injection avoids 1cm / 2.5mm
            // mouth-cell jumping during the recorded motion.
            pressure+=kInputStrength*amplitude*envelope*
                echo::sourceWeight(mouth.x,mouth.y,x,y);
        }
    }
    next[idx]=pressure*decay;
}

// This is the SAME field advanced by asmrPressureStep and sampled by
// both listener microphones. It is not ray-traced, generated sine rings,
// an HRTF display, or an image synthesizer.
__global__ void pressureImageKernel(const float* p,const float* mask,
    uint32_t* out,int n,int imageSize,float exposure) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=imageSize||y>=imageSize)return;
    const int sx=min(n-1,(int)(((float)x+0.5f)*n/imageSize));
    const int sy=min(n-1,(int)(((float)(imageSize-y)-0.5f)*n/imageSize));
    const int index=sy*n+sx;
    const float material=mask[index];
    const float pressure=p[index];
    int r=9,g=14,b=21;
    if(material>=3.5f){r=34;g=72;b=96;}
    else if(material>0.0f){r=32;g=85;b=135;}
    else if(isfinite(pressure)) {
        // A high-contrast signed pressure view; do NOT log-compress
        // into smooth broad blobs, which concealed interference fringes.
        // This is only a GPU-to-display transfer function.
        const float tone=fminf(1.0f,
            sqrtf(fabsf(pressure)*exposure*32.0f));
        if(pressure>=0.0f) {
            r=9+(int)(100.0f*tone);
            g=14+(int)(210.0f*tone);
            b=21+(int)(232.0f*tone);
        }else{
            r=9+(int)(237.0f*tone);
            g=14+(int)(100.0f*tone);
            b=21+(int)(174.0f*tone);
        }
    }
    // SDL_PIXELFORMAT_RGBA32 means little-endian ABGR8888 on x86,
    // so this word lands in host memory as 4 bytes R,G,B,A.
    out[y*imageSize+x]=0xff000000u |
        ((uint32_t)b<<16)|((uint32_t)g<<8)|(uint32_t)r;
}

class FdtdRoom {
public:
    explicit FdtdRoom(float resolution,
                      const asmr::SvgSilhouette& head,
                      bool channels):
      dx_(resolution),n_(asmr::cells(resolution)),
      dt_((float)asmr::stepSeconds(resolution)),
      halfMicSeparation_(asmr::micSpacing(channels)*0.5f),
      micYOffset_(channels?asmr::kChannelEarY:0.0f){
        const auto mask=asmr::svgRoomMask(n_,dx_,head);
        if(!asmr::validateMicrophones(mask,n_,dx_,channels,head)) {
            std::fprintf(stderr,
                "[ASMR ERROR] Mic at x = %.4f / %.4f m is inside a "
                "SOLID svg cell or blocked bilinear footprint, "
                "resolution %.4fm. Refusing invalid ear samples.\n",
                asmr::kListenerX-halfMicSeparation_,
                asmr::kListenerX+halfMicSeparation_,dx_);
            std::exit(3);
        }
        std::printf("[SVG FDTD] Actual head mask: %.1fx%.1fmm, "
                    "%d cells across room, mics %.1fmm apart, "
                    "dx=%.2fmm, dt=%.2fus, room=%.1fx%.1fm.\n",
                    head.widthMetres*1000,head.heightMetres*1000,
                    n_,halfMicSeparation_*2000,dx_*1000,
                    dt_*1e6,asmr::kRoomMetres,asmr::kRoomMetres);
        const size_t size=mask.size()*sizeof(float);
        for(auto& p:fields_) {LAB_CUDA(cudaMalloc(&p,size));
                              LAB_CUDA(cudaMemset(p,0,size));}
        LAB_CUDA(cudaMalloc(&deviceMask_,size));
        LAB_CUDA(cudaMemcpy(deviceMask_,mask.data(),size,
                            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMalloc(&deviceMouth_,sizeof(float2)*asmr::kGraphBatch));
        LAB_CUDA(cudaMalloc(&deviceStep_,sizeof(uint64_t)));
        LAB_CUDA(cudaMalloc(&deviceOutput_,sizeof(float)*2*asmr::kGraphBatch));
        LAB_CUDA(cudaMalloc(&deviceAudio_,sizeof(float)*kMaxWavFrames));
        LAB_CUDA(cudaMalloc(&deviceAudioFrames_,sizeof(int)));
        LAB_CUDA(cudaMalloc(&deviceImage_,
                           (size_t)kPressureImage*kPressureImage*
                           sizeof(uint32_t)));
        int zero=0;
        LAB_CUDA(cudaMemcpy(deviceAudioFrames_,&zero,sizeof(int),
                            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaStreamCreate(&stream_));
        const dim3 threads(16,16);
        const dim3 blocks((n_+15)/16,(n_+15)/16);
        const float decay=std::exp(-dt_*0.6f);
        LAB_CUDA(cudaStreamBeginCapture(stream_,
                    cudaStreamCaptureModeGlobal));
        for(int t=0;t<asmr::kGraphBatch;++t){
            // Three pressure buffers return to the SAME orientation
            // after 126 time steps (126 is divisible by three).
            asmrPressureStep<<<blocks,threads,0,stream_>>>(
                fields_[(t+2)%3],fields_[(t+1)%3],fields_[t%3],
                deviceMask_,deviceAudio_,deviceAudioFrames_,
                deviceMouth_,deviceStep_,deviceOutput_,t,
                n_,dx_,dt_,decay,halfMicSeparation_,micYOffset_,
                0.0,1e12,false);
        }
        LAB_CUDA(cudaStreamEndCapture(stream_,&graph_));
        LAB_CUDA(cudaGraphInstantiate(&execution_,graph_,0));
        LAB_CUDA(cudaStreamSynchronize(stream_));
    }

    // Capture embeds mode/time parameters. A separate CUDA Graph is
    // instantiated when entering offline mode to give it exactly the
    // correct source-time origin and 5/10/20s fade-out envelope.
    void setRenderTiming(double absoluteStart,double length,bool offline){
        if(execution_){LAB_CUDA(cudaGraphExecDestroy(execution_));execution_=nullptr;}
        if(graph_){LAB_CUDA(cudaGraphDestroy(graph_));graph_=nullptr;}
        const dim3 threads(16,16),blocks((n_+15)/16,(n_+15)/16);
        const float decay=std::exp(-dt_*0.6f);
        LAB_CUDA(cudaStreamBeginCapture(stream_,
                    cudaStreamCaptureModeGlobal));
        for(int t=0;t<asmr::kGraphBatch;++t)
            asmrPressureStep<<<blocks,threads,0,stream_>>>(
                fields_[(t+2)%3],fields_[(t+1)%3],fields_[t%3],
                deviceMask_,deviceAudio_,deviceAudioFrames_,
                deviceMouth_,deviceStep_,deviceOutput_,t,
                n_,dx_,dt_,decay,halfMicSeparation_,micYOffset_,
                absoluteStart,length,offline);
        LAB_CUDA(cudaStreamEndCapture(stream_,&graph_));
        LAB_CUDA(cudaGraphInstantiate(&execution_,graph_,0));
        LAB_CUDA(cudaStreamSynchronize(stream_));
    }

    void reset(){
        const size_t bytes=(size_t)n_*n_*sizeof(float);
        for(auto& p:fields_)LAB_CUDA(cudaMemset(p,0,bytes));
    }
    void uploadAudio(const std::vector<float>& source){
        const int amount=(int)std::min(source.size(),
                                      (size_t)kMaxWavFrames);
        if(amount>0)LAB_CUDA(cudaMemcpy(
            deviceAudio_,source.data(),(size_t)amount*sizeof(float),
            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(deviceAudioFrames_,&amount,sizeof(int),
                            cudaMemcpyHostToDevice));
    }
    std::array<float,asmr::kGraphBatch*2> process(
        const std::array<float2,asmr::kGraphBatch>& mouths,
        uint64_t firstStep){
        LAB_CUDA(cudaMemcpy(deviceMouth_,mouths.data(),
            mouths.size()*sizeof(float2),cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(deviceStep_,&firstStep,
            sizeof(uint64_t),cudaMemcpyHostToDevice));
        LAB_CUDA(cudaGraphLaunch(execution_,stream_));
        LAB_CUDA(cudaStreamSynchronize(stream_));
        std::array<float,asmr::kGraphBatch*2> result{};
        LAB_CUDA(cudaMemcpy(result.data(),deviceOutput_,
            result.size()*sizeof(float),cudaMemcpyDeviceToHost));
        return result;
    }
    void pressureImage(std::vector<uint32_t>& pixels,float exposure){
        pixels.resize((size_t)kPressureImage*kPressureImage);
        const dim3 thr(16,16);
        const dim3 blk((kPressureImage+15)/16,
                       (kPressureImage+15)/16);
        pressureImageKernel<<<blk,thr,0,stream_>>>(
            fields_[1],deviceMask_,deviceImage_,n_,
            kPressureImage,exposure);
        LAB_CUDA(cudaGetLastError());
        LAB_CUDA(cudaStreamSynchronize(stream_));
        LAB_CUDA(cudaMemcpy(pixels.data(),deviceImage_,
            pixels.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    }
    float dx()const{return dx_;}
    float dt()const{return dt_;}
    ~FdtdRoom(){
        if(execution_)cudaGraphExecDestroy(execution_);
        if(graph_)cudaGraphDestroy(graph_);
        if(stream_)cudaStreamDestroy(stream_);
        for(auto& p:fields_)if(p)cudaFree(p);
        if(deviceMask_)cudaFree(deviceMask_);
        if(deviceMouth_)cudaFree(deviceMouth_);
        if(deviceStep_)cudaFree(deviceStep_);
        if(deviceOutput_)cudaFree(deviceOutput_);
        if(deviceAudio_)cudaFree(deviceAudio_);
        if(deviceAudioFrames_)cudaFree(deviceAudioFrames_);
        if(deviceImage_)cudaFree(deviceImage_);
    }
    FdtdRoom(const FdtdRoom&)=delete;
    FdtdRoom& operator=(const FdtdRoom&)=delete;
private:
    float dx_=0,dt_=0;
    float halfMicSeparation_=0,micYOffset_=0;
    int n_=0;
    float* fields_[3]{};
    float* deviceMask_=nullptr,*deviceAudio_=nullptr;
    float2* deviceMouth_=nullptr;
    uint64_t* deviceStep_=nullptr;
    float* deviceOutput_=nullptr;
    int* deviceAudioFrames_=nullptr;
    uint32_t* deviceImage_=nullptr;
    cudaStream_t stream_=nullptr;
    cudaGraph_t graph_=nullptr;
    cudaGraphExec_t execution_=nullptr;
};

static bool loadSourceAudio(const std::string& filename,
                            std::vector<float>& full,
                            std::vector<float>& live,
                            float liveCutoff) {
    full.clear();live.clear();
    std::vector<float> source;
    std::string error;
    if(!asmr::decodeMP3(filename,source,error,asmr::kOutputRate,
                         kMaxWavSeconds)) {
        std::fprintf(stderr,"[MP3] %s: %s (path recording still works).\n",
                     filename.c_str(),error.c_str());
        return false;
    }
    full=asmr::lowpass(source,asmr::kOfflinePassband,asmr::kOutputRate);
    live=asmr::lowpass(source,liveCutoff,asmr::kOutputRate);
    std::printf("[MP3] Loaded %s (%.2fs, mono 48kHz). "
                "LIVE %.0fHz, OFFLINE %.0fHz. Input enters CUDA FDTD.\n",
                filename.c_str(),source.size()/
                (double)asmr::kOutputRate,liveCutoff,
                asmr::kOfflinePassband);
    return true;
}

// GPU source positions are tracked continuously across ALL batches.
// The old preview swept previousMouse->currentMouse AGAIN for EACH
// batch (around 50 times per frame), audibly adding zipper/hiss artifacts.
// The artist remains 1:1 under the cursor; only its physical mouth
// pressure source is smoothly motion-limited through simulation time.
static std::array<float2,asmr::kGraphBatch> mouthBatch(
    const std::vector<asmr::PathPoint>& track,double start,
    double dt,uint64_t baseStep,float dx,
    asmr::Vec2& physicalSource,const asmr::Vec2& liveTarget,
    bool offline) {
    std::array<float2,asmr::kGraphBatch> result{};
    for(int k=0;k<asmr::kGraphBatch;++k) {
        const double elapsed=(double)(baseStep+(uint64_t)k)*dt;
        const asmr::Vec2 target=offline?
            asmr::interpolate(track,start+elapsed):liveTarget;
        physicalSource=asmr::advanceSource(
            physicalSource,target,(float)dt,0.012f,2.25f);
        const auto m=asmr::mouth(physicalSource);
        result[k]=make_float2(m.x/dx-0.5f,m.y/dx-0.5f);
    }
    return result;
}

static int screenX(float x){
    return (int)std::lround(x/asmr::kRoomMetres*kWindow);
}
static int screenY(float y){
    return (int)std::lround((asmr::kRoomMetres-y)/
                             asmr::kRoomMetres*kWindow);
}
static void color(SDL_Renderer* r,Uint8 red,Uint8 green,Uint8 blue){
    SDL_SetRenderDrawColor(r,red,green,blue,255);
}
static SDL_Texture* headTexture(SDL_Renderer* renderer,
                                     const asmr::SvgSilhouette& shape,
                                     Uint8 red,Uint8 green,Uint8 blue) {
    if(!shape.present())return nullptr;
    SDL_Texture* result=SDL_CreateTexture(renderer,SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STATIC,shape.w,shape.h);
    if(!result)return nullptr;
    std::vector<uint32_t> pixels((size_t)shape.w*shape.h,0);
    for(size_t j=0;j<pixels.size();++j) {
        const uint32_t opacity=shape.alpha[j];
        pixels[j]=(opacity<<24)|((uint32_t)blue<<16)|
                  ((uint32_t)green<<8)|red;
    }
    SDL_UpdateTexture(result,nullptr,pixels.data(),
                      shape.w*sizeof(uint32_t));
    SDL_SetTextureBlendMode(result,SDL_BLENDMODE_BLEND);
    return result;
}
static void drawHeadSvg(SDL_Renderer* renderer,SDL_Texture* texture,
                        const asmr::SvgSilhouette& shape,
                        asmr::Vec2 centre,double rotation) {
    if(!texture||!shape.present())return;
    const int w=screenX(shape.widthMetres);
    const int h=screenX(shape.heightMetres);
    const SDL_Rect destination{screenX(centre.x)-w/2,
                               screenY(centre.y)-h/2,w,h};
    SDL_RenderCopyEx(renderer,texture,nullptr,&destination,rotation,
                     nullptr,SDL_FLIP_NONE);
}
static void circle(SDL_Renderer* r,int cx,int cy,int radius,bool filled){
    if(filled){
        for(int y=-radius;y<=radius;++y){
            const int width=(int)std::sqrt((float)(radius*radius-y*y));
            SDL_RenderDrawLine(r,cx-width,cy+y,cx+width,cy+y);
        }
    }else{
        for(int step=0;step<144;++step){
            const float a=2*echo::kPi*step/144.0f;
            const float b=2*echo::kPi*(step+1)/144.0f;
            SDL_RenderDrawLine(r,cx+(int)(radius*std::cos(a)),
                cy+(int)(radius*std::sin(a)),
                cx+(int)(radius*std::cos(b)),
                cy+(int)(radius*std::sin(b)));
        }
    }
}
static void drawRoom(SDL_Renderer* renderer,
    const std::vector<asmr::PathPoint>& record,
    asmr::Vec2 artist,asmr::Vec2 physicalSource,
    float progress,bool processing,bool playing,
    float playbackFraction,bool mouseMode,
    SDL_Texture* waves,bool showPressure,
    const asmr::SvgSilhouette& listenerShape,
    const asmr::SvgSilhouette& speakerShape,
    SDL_Texture* listenerTexture,SDL_Texture* speakerTexture,
    bool channels){
    color(renderer,9,14,21);SDL_RenderClear(renderer);
    // Full-size texture with nearest physical pressure sampled and
    // colored on the GPU from the current REAL wave grid.
    if(showPressure && waves)SDL_RenderCopy(renderer,waves,nullptr,nullptr);
    color(renderer,39,78,113);
    for(int thickness=0;thickness<7;++thickness){
        const SDL_Rect r{thickness,thickness,
            kWindow-2*thickness,kWindow-2*thickness};
        SDL_RenderDrawRect(renderer,&r);
    }
    // The movement trail is the path that will be rendered afterwards.
    color(renderer,66,85,89);
    for(size_t i=1;i<record.size();++i)
        if(record.back().t-record[i].t<22.0)
            SDL_RenderDrawLine(renderer,
                screenX(record[i-1].centre.x),
                screenY(record[i-1].centre.y),
                screenX(record[i].centre.x),
                screenY(record[i].centre.y));
    color(renderer,39,117,189);
    if(listenerTexture)
        drawHeadSvg(renderer,listenerTexture,listenerShape,
                    asmr::listener(),0.0);
    else
        circle(renderer,screenX(asmr::kListenerX),
            screenY(asmr::kListenerY),screenX(asmr::kHeadRadius),
            !(showPressure&&waves));
    color(renderer,105,203,252);
    circle(renderer,screenX(asmr::leftEar(channels).x),
           screenY(asmr::leftEar(channels).y),3,true);
    color(renderer,245,158,245);
    circle(renderer,screenX(asmr::rightEar(channels).x),
           screenY(asmr::rightEar(channels).y),3,true);
    // Artist head always points mouth toward the centre of the listener.
    color(renderer,236,141,107);
    if(speakerTexture) {
        // SVG mouth notch faces DOWN on the authored image. Rotate
        // it precisely to the fixed listener; the source coordinates
        // are the matching local -Y recess in world centimetres.
        const asmr::Vec2 f=asmr::faceListener(artist);
        const double degrees=std::atan2(-f.x,-f.y)*180.0/echo::kPi;
        drawHeadSvg(renderer,speakerTexture,speakerShape,
                    artist,degrees);
    }else
        circle(renderer,screenX(artist.x),screenY(artist.y),
            screenX(asmr::kArtistRadius),!(showPressure&&waves));
    // The yellow spot is the *actual interpolated CUDA source*,
    // not a decorative mouth or artificial stereo panning.
    const asmr::Vec2 m=asmr::mouth(physicalSource);
    color(renderer,255,231,126);
    circle(renderer,screenX(m.x),screenY(m.y),5,true);
    // Mouth orientation and 20-30cm peripersonal space reference.
    color(renderer,58,110,132);
    circle(renderer,screenX(asmr::kListenerX),
           screenY(asmr::kListenerY),screenX(0.30f),false);
    // 25cm physical ruler: 100 pixels in a 2m wide 800px room.
    color(renderer,186,184,160);
    SDL_RenderDrawLine(renderer,24,kWindow-20,
                       24+screenX(0.25f),kWindow-20);
    SDL_RenderDrawLine(renderer,24,kWindow-25,24,kWindow-15);
    SDL_RenderDrawLine(renderer,24+screenX(0.25f),kWindow-25,
                       24+screenX(0.25f),kWindow-15);
    if(processing){
        color(renderer,88,91,106);
        SDL_Rect bg{80,kWindow-43,kWindow-160,14};
        SDL_RenderFillRect(renderer,&bg);
        color(renderer,70,222,174);
        SDL_Rect bar=bg;
        bar.w=(int)std::lround(bg.w*std::clamp(progress,0.0f,1.0f));
        SDL_RenderFillRect(renderer,&bar);
    }else if(playing){
        SDL_Rect bar{80,kWindow-43,kWindow-160,14};
        color(renderer,88,91,106);
        SDL_RenderFillRect(renderer,&bar);
        color(renderer,70,222,174);
        bar.w=(int)std::lround(bar.w*
            std::clamp(playbackFraction,0.0f,1.0f));
        SDL_RenderFillRect(renderer,&bar);
        // A dot at the active phase is visible even with a tiny
        // fraction, and snaps back to the start on every loop.
        color(renderer,255,243,140);
        circle(renderer,80+bar.w,kWindow-36,6,true);
    }
    if(mouseMode){
        color(renderer,251,241,169);
        SDL_Rect marker{13,13,10,10};
        SDL_RenderFillRect(renderer,&marker);
    }
    SDL_RenderPresent(renderer);
}
static void writeLE16(std::ofstream& f,uint16_t value){
    f.put((char)(value&255));f.put((char)((value>>8)&255));
}
static void writeLE32(std::ofstream& f,uint32_t value){
    writeLE16(f,(uint16_t)(value&65535));
    writeLE16(f,(uint16_t)(value>>16));
}
static bool writeWav(const std::string& file,
                     const std::vector<float>& audio){
    std::ofstream out(file,std::ios::binary);
    if(!out)return false;
    const uint32_t size=(uint32_t)audio.size()*sizeof(int16_t);
    out.write("RIFF",4);writeLE32(out,36+size);
    out.write("WAVEfmt ",8);writeLE32(out,16);
    writeLE16(out,1);writeLE16(out,2);
    writeLE32(out,asmr::kOutputRate);
    writeLE32(out,asmr::kOutputRate*2*2);
    writeLE16(out,4);writeLE16(out,16);
    out.write("data",4);writeLE32(out,size);
    for(float sample:audio){
        const int scaled=(int)std::lround(
            std::clamp(sample,-1.0f,1.0f)*32767.0f);
        writeLE16(out,(uint16_t)(int16_t)scaled);
    }
    return out.good();
}
static void writeTrajectory(const std::string& file,
    const asmr::PathSelection& selected){
    std::ofstream out(file);
    if(!out)return;
    out<<"seconds,artist_x_m,artist_y_m,mouth_x_m,mouth_y_m,artist_angle_rad\n";
    out<<std::setprecision(9);
    for(const auto& p:selected.points){
        const auto m=asmr::mouth(p.centre);
        const auto d=asmr::faceListener(p.centre);
        out<<(p.t-selected.start)<<","<<p.centre.x<<","
           <<p.centre.y<<","<<m.x<<","<<m.y<<","
           <<std::atan2(d.y,d.x)<<"\n";
    }
}
static void makeSafePlayback(std::vector<float>& audio){
    float largest=0;
    for(float v:audio)if(std::isfinite(v))
        largest=std::max(largest,std::fabs(v));
    const float gain=largest>1e-8f?
        std::min(12.0f,0.38f/largest):0.0f;
    const size_t frames=audio.size()/2;
    for(size_t i=0;i<frames;++i){
        const float fadeIn=(float)std::min(1.0,
            (double)i/(asmr::kOutputRate*0.035));
        const float fadeOut=(float)std::min(1.0,
            (double)(frames-i)/(asmr::kOutputRate*0.060));
        for(int ear=0;ear<2;++ear){
            float& v=audio[i*2+ear];
            v=std::isfinite(v)?
                gain*v*std::min(fadeIn,fadeOut):0.0f;
        }
    }
}
struct RenderJob {
    asmr::PathSelection selected;
    int window=0;
    uint64_t current=0,total=0;
    std::vector<float> samples;
    asmr::Vec2 sourceCursor{};
    std::unique_ptr<echo::StereoLinearResampler> resampler;
};
enum class Mode {Preview,Render,Playback};

int main(){
    const char* home=std::getenv("HOME");
    const std::string filename=std::string(home&&*home?home:".")+
                               "/test.mp3";
    if(SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO)<0){
        std::fprintf(stderr,"[SDL] %s\n",SDL_GetError());return 1;
    }
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices<1){
        std::fprintf(stderr,"[ASMR] Requires an NVIDIA CUDA GPU.\n");
        SDL_Quit();return 1;
    }
    SDL_Window* window=SDL_CreateWindow(
        "ASMR FDTD Lab",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,
        kWindow,kWindow,0);
    if(!window){std::fprintf(stderr,"[SDL] %s\n",SDL_GetError());
                SDL_Quit();return 1;}
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY,"0"); // nearest FDTD cells
    SDL_Renderer* renderer=SDL_CreateRenderer(window,-1,
        SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
    if(!renderer)renderer=SDL_CreateRenderer(window,-1,SDL_RENDERER_SOFTWARE);
    if(!renderer){SDL_DestroyWindow(window);SDL_Quit();return 1;}
    SDL_Texture* waveTexture=SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGBA32,SDL_TEXTUREACCESS_STREAMING,
        kPressureImage,kPressureImage);
    if(!waveTexture)
        std::fprintf(stderr,"[VISUAL] Cannot display CUDA pressure map: %s\n",
                     SDL_GetError());
    std::vector<uint32_t> pressurePixels;
    SDL_AudioSpec desired{},obtained{};
    desired.freq=asmr::kOutputRate;
    desired.format=AUDIO_F32SYS;
    desired.channels=2;
    desired.samples=1024;
    SDL_AudioDeviceID device=SDL_OpenAudioDevice(
        nullptr,0,&desired,&obtained,0);
    if(!device) {
        std::fprintf(stderr,"[SDL] Audio open: %s\n",SDL_GetError());
        SDL_DestroyRenderer(renderer);SDL_DestroyWindow(window);
        SDL_Quit();return 1;
    }
    SDL_PauseAudioDevice(device,0);
    asmr::SvgSilhouette speakerShape=asmr::speakerSilhouette();
    asmr::SvgSilhouette listenerSimple=asmr::listenerSilhouette(
        asmr::HeadStyle::Simple);
    asmr::SvgSilhouette listenerChannels=asmr::listenerSilhouette(
        asmr::HeadStyle::Channels);
    asmr::SvgSilhouette listenerV2=asmr::listenerSilhouette(
        asmr::HeadStyle::PinnaV2);
    if(!speakerShape.present()||!listenerSimple.present()||
       !listenerChannels.present()||!listenerV2.present()) {
        std::fprintf(stderr,"[FDTD SVG] Missing listener_v2.svg "
            "or speaker.svg. Run from the checkout with asmr_shapes/, "
            "or put the exact drawings in HOME.\n");
        return 4;
    }
    // Default is the supplied 185.35mm mirrored PINNA silhouette,
    // not the old 17cm circle or the previous channel drawing.
    int headIndex=0; // 0 new V2, 1 explicit canal, 2 original simple
    bool channels=true; // v2 eardrums are 8cm apart in open ports
    const auto* head=&listenerV2;
    SDL_Texture* speakerSprite=headTexture(
        renderer,speakerShape,237,141,98);
    SDL_Texture* simpleSprite=headTexture(
        renderer,listenerSimple,38,132,218);
    SDL_Texture* channelsSprite=headTexture(
        renderer,listenerChannels,38,132,218);
    SDL_Texture* pinnaV2Sprite=headTexture(
        renderer,listenerV2,38,132,218);
    const std::array<float,6> qualityDx{
        asmr::kLiveDx,asmr::kMediumDx,asmr::kUltraDx,
        asmr::kSafeDx,asmr::kLowDx,asmr::kLowestDx};
    const std::array<float,6> qualityBand{
        asmr::kLivePassband,asmr::kMediumPassband,asmr::kUltraPassband,
        asmr::kSafePassband,asmr::kLowPassband,
        asmr::kLowestPassband};
    int qualityIndex=0;
    float liveCutoff=qualityBand[qualityIndex];
    std::vector<float> fullAudio,liveAudio;
    bool waveLoaded=loadSourceAudio(filename,fullAudio,liveAudio,
                                    liveCutoff);

    auto live=std::make_unique<FdtdRoom>(
        qualityDx[qualityIndex],*head,channels);
    live->uploadAudio(liveAudio);
    std::unique_ptr<FdtdRoom> highQuality;
    echo::StereoLinearResampler liveResampler(live->dt(),
                                                asmr::kOutputRate);
    std::vector<float> queued,blockSamples;
    queued.reserve(asmr::kGraphBatch*4*96);
    blockSamples.reserve(asmr::kGraphBatch*4);
    uint64_t previewStep=0;
    uint64_t clippedPreview=0,shortQueueWarnings=0;
    uint64_t audioDiagTime=SDL_GetTicks64();
    asmr::Vec2 artist{1.50f,1.0f},sourceCursor=artist;
    bool mouseMode=true,previewEnabled=true,quit=false;
    bool showPressure=true;
    float pressureExposure=1.0f;
    Mode mode=Mode::Preview;
    SDL_WarpMouseInWindow(window,screenX(artist.x),screenY(artist.y));
    std::vector<asmr::PathPoint> record{{0.0,artist}};
    record.reserve(6000);
    double recordingClock=0;
    RenderJob job;
    std::vector<float> lastPlayback;
    asmr::PathSelection lastPlaybackPath;
    uint64_t playbackEnqueuedCycles=0;
    double playbackPhase=0.0;
    int lastWindow=5;
    auto startReplay=[&]() {
        if(lastPlayback.empty())return;
        SDL_ClearQueuedAudio(device);
        if(SDL_QueueAudio(device,lastPlayback.data(),
             (Uint32)(lastPlayback.size()*sizeof(float)))!=0)
            std::fprintf(stderr,"[PLAYBACK] Queue failed: %s\n",
                         SDL_GetError());
        playbackEnqueuedCycles=1;
        playbackPhase=0.0;
        mode=Mode::Playback;
    };
    uint64_t lastFrame=SDL_GetPerformanceCounter();
    const double perfFrequency=(double)SDL_GetPerformanceFrequency();

    std::printf("\n[ASMR LAB] Room 2x2m. A fixed blue listener head has two "
       "FDTD ears; moving orange artist aims mouth at its centre.\n");
    std::printf("[CONTROLS] Mouse follows cursor 1:1 (M toggles mouse). "
       "WASD/arrows move with mouse disabled.\n");
    std::printf("[CONTROLS] F5 = render last 5s, F6 = 10s, F7 = 20s "
       "in HIGH quality, then play the complete finished WAV.\n");
    std::printf("[CONTROLS] P loop replay, SPACE return to live, "
       "L reload ~/test.mp3, T toggle live, "
       "H cycle V2 pinna/channel/simple, Q 3.5/3/2.5/5/8/10mm, "
       "V pressure waves, 9/0 exposure, "
       "R reset, Esc exit.\n");
    std::printf("[SPECTRUM] Live dx=%.1fmm ~%.0fHz (10 cells/lambda), "
       "offline dx=%.1fmm ~%.0fHz with %.0fHz input lowpass.\n",
       asmr::kLiveDx*1000,asmr::tenCellFrequency(asmr::kLiveDx),
       asmr::kOfflineDx*1000,asmr::tenCellFrequency(asmr::kOfflineDx),
       asmr::kOfflinePassband);
    std::printf("[WARNING] 2D FDTD gives natural two-ear timing/level"
       " differences and planar head shadow, NOT realistic pinnae/HRTFs. "
       "High-quality 20s processing may take substantial GPU time.\n");
    while(!quit) {
        const uint64_t currentCounter=SDL_GetPerformanceCounter();
        // The recorded path has to follow REAL wall time even if GPU
        // preview frames lag. Only input movement distance uses the
        // capped delta, to avoid collision tunnelling after stalls.
        const double wallSeconds=std::max(0.0,
            (currentCounter-lastFrame)/perfFrequency);
        const float frameDt=(float)std::min(0.05,wallSeconds);
        lastFrame=currentCounter;
        SDL_Event event;
        int captureSeconds=0;
        bool reload=false;
        while(SDL_PollEvent(&event)){
            if(event.type==SDL_QUIT)quit=true;
            if(event.type!=SDL_KEYDOWN||event.key.repeat)continue;
            const SDL_Keycode key=event.key.keysym.sym;
            if(key==SDLK_ESCAPE)quit=true;
            if(key==SDLK_m) {
                mouseMode=!mouseMode;
                if(mouseMode)SDL_WarpMouseInWindow(
                    window,screenX(artist.x),screenY(artist.y));
            }
            if(key==SDLK_F5||key==SDLK_5)captureSeconds=5;
            if(key==SDLK_F6||key==SDLK_1)captureSeconds=10;
            if(key==SDLK_F7||key==SDLK_2)captureSeconds=20;
            if(key==SDLK_l)reload=true;
            if(key==SDLK_v)showPressure=!showPressure;
            if(key==SDLK_9)pressureExposure=std::max(
                0.025f,pressureExposure/1.4f);
            if(key==SDLK_0)pressureExposure=std::min(
                500.0f,pressureExposure*1.4f);
            if(key==SDLK_t) {
                previewEnabled=!previewEnabled;
                SDL_ClearQueuedAudio(device);
                live->reset();previewStep=0;
                sourceCursor=artist;
                liveResampler=echo::StereoLinearResampler(
                    live->dt(),asmr::kOutputRate);
            }
            if(key==SDLK_h && mode==Mode::Preview){
                const int proposed=(headIndex+1)%3;
                const asmr::SvgSilhouette& shape=proposed==0?
                    listenerV2:proposed==1?listenerChannels:listenerSimple;
                const bool proposedChannels=proposed!=2;
                const auto mask=asmr::svgRoomMask(
                    asmr::cells(live->dx()),live->dx(),shape);
                if(!asmr::validateMicrophones(mask,
                    asmr::cells(live->dx()),live->dx(),
                    proposedChannels,shape)) {
                    std::fprintf(stderr,"[SVG HEAD] Mode %d at %.2fmm "
                        "blocks the eardrum microphone. Press Q "
                        "for finer resolution.\n",proposed,
                        live->dx()*1000);
                }else{
                    headIndex=proposed;
                    channels=proposedChannels;
                    head=&shape;
                    SDL_ClearQueuedAudio(device);
                    highQuality.reset();live.reset();
                    live=std::make_unique<FdtdRoom>(
                        qualityDx[qualityIndex],*head,channels);
                    live->uploadAudio(liveAudio);
                    previewStep=0;sourceCursor=artist;
                    liveResampler=echo::StereoLinearResampler(
                        live->dt(),asmr::kOutputRate);
                    std::printf("[SVG HEAD] %s, physically open mics "
                        "%.1fcm apart.\n",
                        headIndex==0?"listener_v2 pinna":
                        headIndex==1?"ear-channel alternative":
                        "original simple alternative",
                        asmr::micSpacing(channels)*100);
                }
            }
            if(key==SDLK_q && mode==Mode::Preview){
                // Never silently use 10mm cells for a 3mm-wide
                // ear groove if its microphone is blocked at this
                // resolution; skip any such unsupported grid.
                const int old=qualityIndex;
                for(size_t trial=0;trial<qualityDx.size();++trial) {
                    qualityIndex=(qualityIndex+1)%
                        (int)qualityDx.size();
                    const float dx=qualityDx[qualityIndex];
                    const int side=asmr::cells(dx);
                    const auto mask=asmr::svgRoomMask(side,dx,*head);
                    if(asmr::validateMicrophones(
                        mask,side,dx,channels,*head))
                        break;
                }
                if(old!=qualityIndex) {
                    liveCutoff=qualityBand[qualityIndex];
                    if(waveLoaded)liveAudio=asmr::lowpass(
                        fullAudio,liveCutoff,asmr::kOutputRate);
                    SDL_ClearQueuedAudio(device);
                    live.reset();
                    live=std::make_unique<FdtdRoom>(
                        qualityDx[qualityIndex],*head,channels);
                    live->uploadAudio(liveAudio);
                    previewStep=0;sourceCursor=artist;
                    liveResampler=echo::StereoLinearResampler(
                        live->dt(),asmr::kOutputRate);
                    std::printf("[QUALITY] Full 2x2m FDTD grid "
                        "dx=%.2fmm, %dx%d acoustic cells, "
                        "spatial 10-cell band %.0fHz, "
                        "input MP3 LP %.0fHz\n",
                        live->dx()*1000,
                        asmr::cells(live->dx()),
                        asmr::cells(live->dx()),
                        asmr::tenCellFrequency(live->dx()),
                        liveCutoff);
                }
            }
            if(key==SDLK_r){
                record.clear();recordingClock=0;
                record.push_back({0,artist});
                previewStep=0;live->reset();
                sourceCursor=artist;
                liveResampler=echo::StereoLinearResampler(
                    live->dt(),asmr::kOutputRate);
                SDL_ClearQueuedAudio(device);
                mode=Mode::Preview;
                std::printf("[PATH] Recording reset.\n");
            }
            if(key==SDLK_p&&!lastPlayback.empty()){
                startReplay();
                std::printf("[PLAYBACK] Looping last processed %ds "
                    "with visible path. SPACE returns to live.\n",
                    lastWindow);
            }
            if(key==SDLK_SPACE && mode==Mode::Playback){
                SDL_ClearQueuedAudio(device);
                mode=Mode::Preview;
                live->reset();previewStep=0;
                sourceCursor=artist;
                liveResampler=echo::StereoLinearResampler(
                    live->dt(),asmr::kOutputRate);
            }
        }
        if(reload){
            waveLoaded=loadSourceAudio(filename,fullAudio,liveAudio,
                                       liveCutoff);
            live->uploadAudio(liveAudio);
            if(highQuality)highQuality->uploadAudio(fullAudio);
            if(mode==Mode::Preview){
                live->reset();previewStep=0;
                sourceCursor=artist;
                liveResampler=echo::StereoLinearResampler(
                    live->dt(),asmr::kOutputRate);
                SDL_ClearQueuedAudio(device);
            }
        }
        if(captureSeconds>0) {
            const double now=record.empty()?0:record.back().t;
            job.selected=asmr::selectLast(record,now,captureSeconds);
            if(job.selected.duration<0.15||!waveLoaded){
                std::fprintf(stderr,
                    "[RENDER] Need >=0.15 seconds of path AND readable %s.\n",
                    filename.c_str());
            } else {
                if(!highQuality)
                    highQuality=std::make_unique<FdtdRoom>(
                        asmr::kOfflineDx,*head,channels);
                highQuality->uploadAudio(fullAudio);
                highQuality->setRenderTiming(job.selected.start,
                    (double)captureSeconds,true);
                highQuality->reset();
                job.window=captureSeconds;
                job.current=0;
                job.sourceCursor=asmr::interpolate(
                    job.selected.points,job.selected.start);
                // The render is EXACTLY the requested length in
                // 48kHz stereo frames. Two extra acoustic steps ensure
                // the resampler never runs short of its final frame.
                job.total=(uint64_t)std::ceil(
                    (double)job.window/highQuality->dt())+2;
                job.samples.clear();
                job.samples.reserve(
                    asmr::clipFrames(job.window)*2+16);
                job.resampler=std::make_unique<echo::StereoLinearResampler>(
                    highQuality->dt(),asmr::kOutputRate);
                SDL_ClearQueuedAudio(device);
                mode=Mode::Render;
                std::printf("[RENDER] Captured last %.2fs; "
                  "offline %.1fmm / %.0fHz. %.0f million grid steps.\n",
                  job.selected.duration,asmr::kOfflineDx*1000,
                  asmr::kOfflinePassband,
                  (double)job.total*asmr::cells(asmr::kOfflineDx)*
                      asmr::cells(asmr::kOfflineDx)/1e6);
            }
        }
        if(mode==Mode::Preview){
            if(mouseMode){
                int x=0,y=0;
                SDL_GetMouseState(&x,&y);
                const asmr::Vec2 wanted{
                    asmr::kRoomMetres*
                        ((float)x+0.5f)/kWindow,
                    asmr::kRoomMetres*
                        (1.0f-((float)y+0.5f)/kWindow)};
                artist=asmr::moveArtist(artist,wanted);
            }else{
                const Uint8* keys=SDL_GetKeyboardState(nullptr);
                const float step=frameDt*0.40f;
                float dx=0,dy=0;
                if(keys[SDL_SCANCODE_W]||keys[SDL_SCANCODE_UP])dy+=step;
                if(keys[SDL_SCANCODE_S]||keys[SDL_SCANCODE_DOWN])dy-=step;
                if(keys[SDL_SCANCODE_A]||keys[SDL_SCANCODE_LEFT])dx-=step;
                if(keys[SDL_SCANCODE_D]||keys[SDL_SCANCODE_RIGHT])dx+=step;
                artist=asmr::moveArtist(artist,
                    {artist.x+dx,artist.y+dy});
            }
            recordingClock+=wallSeconds;
            record.push_back({recordingClock,artist});
            // Maintain an uninterrupted 60s circular time history.
            while(record.size()>2 &&
                  record[1].t<recordingClock-60.0)
                record.erase(record.begin());

            if(previewEnabled&&waveLoaded){
                const int queuedFrames=(int)(
                    SDL_GetQueuedAudioSize(device)/(2*sizeof(float)));
                if(previewStep>0 && queuedFrames<24)
                    ++shortQueueWarnings;
                int need=std::max(0,(int)(0.075*asmr::kOutputRate)-
                                    queuedFrames);
                int batches=0;
                queued.clear();
                // The physical source position persists ACROSS the
                // CUDA batches. Previously the same mouse sweep was
                // replayed from start to finish for every mini-batch.
                while(need>0 && batches<96){
                    const auto points=mouthBatch(
                        {},0,live->dt(),previewStep,live->dx(),
                        sourceCursor,artist,false);
                    const auto audio=live->process(points,previewStep);
                    previewStep+=asmr::kGraphBatch;
                    blockSamples.clear();
                    for(int n=0;n<asmr::kGraphBatch;++n)
                        liveResampler.push(audio[n*2],audio[n*2+1],
                                           blockSamples);
                    queued.insert(queued.end(),blockSamples.begin(),
                                  blockSamples.end());
                    need-=(int)blockSamples.size()/2;
                    ++batches;
                }
                // Preserve actual FDTD pressure differences linearly.
                // No always-on tanh distortion/harmonic generator.
                for(float& v:queued) {
                    v=std::isfinite(v)?0.55f*v:0.0f;
                    if(std::fabs(v)>0.85f)++clippedPreview;
                    v=std::clamp(v,-0.85f,0.85f);
                }
                if(!queued.empty()&&SDL_QueueAudio(
                    device,queued.data(),
                    (Uint32)(queued.size()*sizeof(float)))!=0)
                    std::fprintf(stderr,"[SDL AUDIO] %s\n",
                                 SDL_GetError());
            }
        }else if(mode==Mode::Render){
            // A bounded number of expensive GPU graph batches per UI
            // frame keeps progress and Escape responsive. No audio is
            // queued until the ENTIRE high-resolution render is ready.
            for(int batch=0;batch<4 && job.current<job.total;++batch){
                const auto sources=mouthBatch(
                    job.selected.points,job.selected.start,
                    highQuality->dt(),job.current,
                    highQuality->dx(),job.sourceCursor,artist,true);
                const auto samples=highQuality->process(
                    sources,job.current);
                const uint64_t count=std::min(
                    (uint64_t)asmr::kGraphBatch,job.total-job.current);
                for(uint64_t i=0;i<count;++i)
                    job.resampler->push(
                        samples[2*i],samples[2*i+1],job.samples);
                job.current+=count;
            }
            if(job.current>=job.total){
                // DO NOT derive the playback length from the number
                // of FDTD steps, block boundaries or float dt.
                // Both saved WAV and SDL loop are N*48000 frames.
                asmr::fitStereoClip(job.samples,job.window);
                makeSafePlayback(job.samples);
                char soundPath[80],trackPath[80];
                std::snprintf(soundPath,sizeof(soundPath),
                              "asmr_last_%02ds.wav",job.window);
                std::snprintf(trackPath,sizeof(trackPath),
                              "asmr_last_%02ds_path.csv",job.window);
                if(!writeWav(soundPath,job.samples))
                    std::fprintf(stderr,
                        "[RENDER] Could not save %s\n",soundPath);
                writeTrajectory(trackPath,job.selected);
                lastWindow=job.window;
                lastPlaybackPath=job.selected;
                lastPlayback.swap(job.samples);
                startReplay();
                std::printf("[RENDER] Complete. Saved %s and %s; "
                    "now looping EXACTLY %.2fs of pre-rendered stereo. "
                    "SPACE stops.\n",soundPath,trackPath,
                    (double)lastPlayback.size()/
                    (2*asmr::kOutputRate));
            }
        }else if(mode==Mode::Playback && !lastPlayback.empty()){
            const uint64_t frameCount=lastPlayback.size()/2;
            const uint64_t pendingBefore=SDL_GetQueuedAudioSize(device)/
                (2*sizeof(float));
            // Queue the next complete PCM loop before the current
            // copy runs out: no GPU render or gap at the wrap point.
            if(pendingBefore < (uint64_t)asmr::kOutputRate/4) {
                if(SDL_QueueAudio(device,lastPlayback.data(),
                    (Uint32)(lastPlayback.size()*sizeof(float)))==0)
                    ++playbackEnqueuedCycles;
            }
            const uint64_t pending=SDL_GetQueuedAudioSize(device)/
                (2*sizeof(float));
            playbackPhase=asmr::playbackSeconds(
                playbackEnqueuedCycles*frameCount,
                pending,frameCount);
        }
        if(SDL_GetTicks64()-audioDiagTime>=5000){
            std::printf("[ASMR FDTD] LIVE dx=%.1fmm cells=%dx%d "
                        "pressure steps=%llu queue=%.1fms "
                        "near-empty queues=%llu safety-clipped samples=%llu "
                        "(V toggles REAL CUDA wave field)\n",
                live->dx()*1000,
                asmr::cells(live->dx()),asmr::cells(live->dx()),
                (unsigned long long)previewStep,
                1000.0*SDL_GetQueuedAudioSize(device)/
                    (2.0*sizeof(float)*asmr::kOutputRate),
                (unsigned long long)shortQueueWarnings,
                (unsigned long long)clippedPreview);
            shortQueueWarnings=0;clippedPreview=0;
            audioDiagTime=SDL_GetTicks64();
        }
        const float progress=mode==Mode::Render&&job.total>0?
            (float)((double)job.current/job.total):0.0f;
        if(showPressure && waveTexture){
            // This GPU pressure snapshot is from the exact field
            // that produces the ear samples. Render the offline wave
            // grid too while a slow 15kHz capture is processing.
            FdtdRoom* active=((mode==Mode::Render||
                               mode==Mode::Playback) && highQuality)?
                highQuality.get():live.get();
            active->pressureImage(pressurePixels,pressureExposure);
            SDL_UpdateTexture(waveTexture,nullptr,
                pressurePixels.data(),kPressureImage*sizeof(uint32_t));
        }
        const asmr::Vec2 visibleArtist=mode==Mode::Render?
            job.sourceCursor:mode==Mode::Playback?
            asmr::interpolate(lastPlaybackPath.points,
                              lastPlaybackPath.start+playbackPhase):artist;
        const asmr::Vec2 visibleSource=mode==Mode::Render?
            job.sourceCursor:mode==Mode::Playback?
            visibleArtist:sourceCursor;
        const auto& visibleTrail=mode==Mode::Playback?
            lastPlaybackPath.points:record;
        drawRoom(renderer,visibleTrail,visibleArtist,visibleSource,
                 progress,mode==Mode::Render,mode==Mode::Playback,
                 lastWindow>0?(float)(playbackPhase/lastWindow):0.0f,
                 mouseMode,waveTexture,showPressure,
                 *head,speakerShape,
                 headIndex==0?pinnaV2Sprite:
                     headIndex==1?channelsSprite:simpleSprite,
                 speakerSprite,channels);
        char title[255];
        std::snprintf(title,sizeof(title),
            "ASMR FDTD | %s | MP3:%s | path %.1fs | REPLAY %.2f/%.0fs | %.0fHz / %.0fHz | P loop SPACE stop",
            mode==Mode::Render?"HIGH-RES RENDER":mode==Mode::Playback?
                "VISIBLE LOOP":"LIVE RECORDING",
            waveLoaded?"READY":"MISSING",
            record.empty()?0:record.back().t-record.front().t,
            playbackPhase,(double)lastWindow,
            liveCutoff,asmr::kOfflinePassband);
        SDL_SetWindowTitle(window,title);
        SDL_Delay(6);
    }
    SDL_ClearQueuedAudio(device);
    SDL_CloseAudioDevice(device);
    highQuality.reset();
    if(speakerSprite)SDL_DestroyTexture(speakerSprite);
    if(simpleSprite)SDL_DestroyTexture(simpleSprite);
    if(channelsSprite)SDL_DestroyTexture(channelsSprite);
    if(pinnaV2Sprite)SDL_DestroyTexture(pinnaV2Sprite);
    if(waveTexture)SDL_DestroyTexture(waveTexture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
