// Standalone ASMR room experiment. No changes to the arcade executable.
// Source defaults to ~/test.mp3, or --mp3 FILE; decoded to mono PCM.
// Listener head is a REAL stationary 2D FDTD obstruction with two ears.
// Artist is a collision-safe moving visual head; its mouth is a real
// moving FDTD pressure emitter facing the listener on every path sample.
// Live true-zoom presets target 85%->27.2% of the OLD 400x711@5mm
// acoustic cell-update budget; default 1x dx is now ~5.28mm.
// Offline starts near 2.11mm with the same physical zoom and a finer
// independent mesh; neither simulates an unchanged square-room grid.
// both paths resample physically timed FDTD samples to 48kHz stereo
// WAV before playback. The GPU cannot change playback speed or pitch.
#include "asmr_lab_math.h"
#include "asmr_svg_geometry.h"
#include "asmr_dynamic_head.h"
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

static constexpr int kWindowW=540;
static constexpr int kWindowH=960; // 9:16 portrait
static constexpr int kMaxWavSeconds=60;
static constexpr int kMaxWavFrames=asmr::kOutputRate*kMaxWavSeconds;
static constexpr float kInputStrength=3.0f;
static constexpr int kPressureW=360, kPressureH=640; // GPU field visual only
static constexpr int kReplayW=180, kReplayH=320; // replay visual only
static constexpr int kReplayPressureFps=24; // no effect on FDTD/audio


// At the edge of the sculpted ear recess, a bilinear microphone's
// surrounding lattice can include some SOLID pressure cells. Weight
// only AIR neighbors: otherwise the stereo gain changes with dx,
 // creating spurious ear asymmetry even at fixed physical spacing.
__device__ float sampleEarAir(const float* pressure,const float* mask,
                              int nx,int ny,float gx,float gy) {
    const int x=(int)floorf(gx),y=(int)floorf(gy);
    if(x<1||y<1||x+1>=nx-1||y+1>=ny-1)return 0.0f;
    float accum=0.0f,total=0.0f;
    for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox){
        const int index=(y+oy)*nx+x+ox;
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
    const float4* mouthLocations,const uint64_t* blockBase,
    const float* ambientPCM,const int* ambientLength,
    const int* ambientOn,const float2* ambientPositions,
    float* outStereo,int localStep,int nx,int ny,float dx,float dt,float decay,
    float originX,float originY,float halfMicSeparation,float micYOffset,double absoluteStart,
    double recordingDuration,bool offline,
    const unsigned char* speakerPixels,int speakerW,int speakerH,
    float speakerWidthCells,float speakerHeightCells)
{
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=nx||y>=ny)return;
    const int idx=y*nx+x;
    // Read the preceding, completed pressure field at BOTH ear points.
    // Unlike separate output kernels, this doesn't double graph launches.
    // The p field is read-only for the entirety of this CUDA kernel.
    if(x==0&&y==0){
        const float lx=(asmr::kListenerX-originX-halfMicSeparation)/dx-0.5f;
        const float rx=(asmr::kListenerX-originX+halfMicSeparation)/dx-0.5f;
        const float ey=(asmr::kListenerY-originY+micYOffset)/dx-0.5f;
        outStereo[localStep*2]=sampleEarAir(now,mask,nx,ny,lx,ey);
        outStereo[localStep*2+1]=sampleEarAir(now,mask,nx,ny,rx,ey);
    }
    if(x==0||y==0||x==nx-1||y==ny-1||mask[idx]>0.0f) {
        next[idx]=0.0f;return;
    }
    const float center=now[idx];
    const float4 mouth=mouthLocations[localStep];
    // The source location and sender centre are passed at EVERY FDTD
    // timestep, and give the forward direction without a sqrt.
    const float invMouthCells=dx/asmr::kMouthOffset;
    const float fx=(mouth.x-mouth.z)*invMouthCells;
    const float fy=(mouth.y-mouth.w)*invMouthCells;
    float b0=0,bL=0,bR=0,bD=0,bU=0;
    const float extent=0.5f*sqrtf(
        speakerWidthCells*speakerWidthCells+
        speakerHeightCells*speakerHeightCells)+2.0f;
    if(fabsf((float)x-mouth.z)<extent &&
       fabsf((float)y-mouth.w)<extent) {
        b0=asmr::movingHeadCoverage(speakerPixels,speakerW,speakerH,
           speakerWidthCells,speakerHeightCells,mouth.z,mouth.w,fx,fy,x,y);
        bL=asmr::movingHeadCoverage(speakerPixels,speakerW,speakerH,
           speakerWidthCells,speakerHeightCells,mouth.z,mouth.w,fx,fy,x-1,y);
        bR=asmr::movingHeadCoverage(speakerPixels,speakerW,speakerH,
           speakerWidthCells,speakerHeightCells,mouth.z,mouth.w,fx,fy,x+1,y);
        bD=asmr::movingHeadCoverage(speakerPixels,speakerW,speakerH,
           speakerWidthCells,speakerHeightCells,mouth.z,mouth.w,fx,fy,x,y-1);
        bU=asmr::movingHeadCoverage(speakerPixels,speakerW,speakerH,
           speakerWidthCells,speakerHeightCells,mouth.z,mouth.w,fx,fy,x,y+1);
    }
    // No hard moving wall, no wiping pressure histories. This is
    // smooth, symmetric impedance scattering by actual SVG geometry.
    const float l=mask[idx-1]>0?center:asmr::movingHeadNeighbor(
        center,now[idx-1],b0,bL);
    const float r=mask[idx+1]>0?center:asmr::movingHeadNeighbor(
        center,now[idx+1],b0,bR);
    const float d=mask[idx-nx]>0?center:asmr::movingHeadNeighbor(
        center,now[idx-nx],b0,bD);
    const float u=mask[idx+nx]>0?center:asmr::movingHeadNeighbor(
        center,now[idx+nx],b0,bU);
    const float damping=asmr::boundaryDamping(
        mask[idx-1],mask[idx+1],mask[idx-nx],mask[idx+nx]);
    float pressure=echo::impedanceWaveStep(
        center,previous[idx],l,r,d,u,echo::kCourantSquared,damping);

    // Only the 2x2 mouth footprint decodes the injected PCM.
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
    // Diagnostic N: four distant, incoherent broadband emitters
    // genuinely excite the acoustic FIELD. Both pinnae and canal
    // receive the propagated/scattered field, not fake colored noise
    // mixed at the ear output. Only ~16 cells per stencil do PCM work.
    for(int source=0;source<asmr::kAmbientSourceCount;++source){
        const float2 p=ambientPositions[source];
        if(fabsf((float)x-p.x)>=1.0f||
           fabsf((float)y-p.y)>=1.0f||
           *ambientOn==0)continue;
        const double elapsed=(double)(*blockBase+
                                      (uint64_t)localStep)*dt;
        float envelope=1.0f;
        if(offline){
            if(elapsed>=recordingDuration)continue;
            const float rise=fminf(1.0f,(float)(elapsed/0.035));
            const float fall=fminf(1.0f,(float)(
                (recordingDuration-elapsed)/0.06));
            envelope=fmaxf(0.0f,fminf(rise,fall));
        }
        const float noise=roomWaveSample(ambientPCM,*ambientLength,
            absoluteStart+elapsed+2.0137*source);
        pressure+=kInputStrength*asmr::kAmbientInjection*
            noise*envelope*echo::sourceWeight(p.x,p.y,x,y);
    }
    next[idx]=pressure*decay;
}

// This is the SAME field advanced by asmrPressureStep and sampled by
// both listener microphones. It is not ray-traced, generated sine rings,
// an HRTF display, or an image synthesizer.
__global__ void pressureImageKernel(const float* p,const float* mask,
    uint32_t* out,int nx,int ny,int imageW,int imageH,float exposure) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=imageW||y>=imageH)return;
    const int sx=min(nx-1,(int)(((float)x+0.5f)*nx/imageW));
    const int sy=min(ny-1,(int)(((float)(imageH-y)-0.5f)*ny/imageH));
    const int index=sy*nx+sx;
    const float material=mask[index];
    const float pressure=p[index];
    int r=9,g=14,b=21;
    if(material>=4.5f){r=40;g=174;b=145;} // acoustic window/door
    else if(material>=3.5f){r=98;g=111;b=135;} // strong wall
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
    out[y*imageW+x]=0xff000000u |
        ((uint32_t)b<<16)|((uint32_t)g<<8)|(uint32_t)r;
}

class FdtdRoom {
public:
    explicit FdtdRoom(float resolution,
                      const asmr::SvgSilhouette& head,
                      bool channels,
                      const asmr::SvgSilhouette& speaker,
                      float zoom=1.0f,bool square=false,
                      float physicsScale=1.0f):
      dx_(resolution),
      dt_((float)asmr::physicalStepSeconds(resolution,physicsScale)),
      physicsScale_(physicsScale),
      nx_((int)std::lround(
          asmr::roomView(zoom,square).width/resolution)),
      ny_((int)std::lround(
          asmr::roomView(zoom,square).height/resolution)),
      view_(asmr::roomView(zoom,square)),
      halfMicSeparation_(asmr::micSpacing(channels,head.pinnaV2)*0.5f),
      micYOffset_(asmr::micOffsetY(channels,head.pinnaV2)){
        if(!speaker.present()) {
            std::fprintf(stderr,"[FDTD] Missing speaker SVG.\n");
            std::exit(4);
        }
        speakerW_=speaker.w;speakerH_=speaker.h;
        speakerWidthCells_=speaker.widthMetres/dx_;
        speakerHeightCells_=speaker.heightMetres/dx_;
        const auto mask=asmr::svgRoomMask(nx_,ny_,dx_,head,
            view_.originX,view_.originY);
        if(!asmr::validateMicrophones(mask,nx_,ny_,dx_,channels,head,
            view_.originX,view_.originY)) {
            std::fprintf(stderr,
                "[ASMR ERROR] Mic at x = %.4f / %.4f m is inside a "
                "SOLID SVG cell, blocked bilinear footprint, or "
                "disconnected air cavity, resolution %.4fm. "
                "Refusing invalid ear samples.\n",
                asmr::kListenerX-halfMicSeparation_,
                asmr::kListenerX+halfMicSeparation_,dx_);
            std::exit(3);
        }
        std::printf("[SVG FDTD] Actual head mask: %.1fx%.1fmm, "
                    "%dx%d cells across room, mics %.1fmm apart, "
                    "dx=%.2fmm, dt=%.2fus, room=%.1fx%.1fm.\n",
                    head.widthMetres*1000,head.heightMetres*1000,
                    nx_,ny_,halfMicSeparation_*2000,dx_*1000,
                    dt_*1e6,view_.width,view_.height);
        std::printf("[F11 PHYSICAL] room %.2fx%.2fm, "
            "listener SVG %.1fx%.1fmm, mics %.1fmm apart, "
            "real dx=%.2fmm, dt=%.2fus, scale=%gx; "
            "grid remains %dx%d.\n",
            view_.width*physicsScale_,view_.height*physicsScale_,
            head.widthMetres*physicsScale_*1000,
            head.heightMetres*physicsScale_*1000,
            halfMicSeparation_*2000*physicsScale_,
            dx_*physicsScale_*1000,dt_*1e6,
            physicsScale_,nx_,ny_);
        const size_t size=mask.size()*sizeof(float);
        for(auto& p:fields_) {LAB_CUDA(cudaMalloc(&p,size));
                              LAB_CUDA(cudaMemset(p,0,size));}
        LAB_CUDA(cudaMalloc(&deviceMask_,size));
        LAB_CUDA(cudaMemcpy(deviceMask_,mask.data(),size,
                            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMalloc(&deviceMouth_,sizeof(float4)*asmr::kGraphBatch));
        LAB_CUDA(cudaMalloc(&deviceSpeaker_,speaker.alpha.size()));
        LAB_CUDA(cudaMemcpy(deviceSpeaker_,speaker.alpha.data(),
                            speaker.alpha.size(),cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMalloc(&deviceStep_,sizeof(uint64_t)));
        LAB_CUDA(cudaMalloc(&deviceOutput_,sizeof(float)*2*asmr::kGraphBatch));
        LAB_CUDA(cudaMalloc(&deviceAudio_,sizeof(float)*kMaxWavFrames));
        LAB_CUDA(cudaMalloc(&deviceAudioFrames_,sizeof(int)));
        LAB_CUDA(cudaMalloc(&deviceAmbientPCM_,
            sizeof(float)*asmr::kOutputRate*asmr::kAmbientNoiseSeconds));
        LAB_CUDA(cudaMalloc(&deviceAmbientFrames_,sizeof(int)));
        LAB_CUDA(cudaMalloc(&deviceAmbientOn_,sizeof(int)));
        LAB_CUDA(cudaMalloc(&deviceAmbientPositions_,
            sizeof(float2)*asmr::kAmbientSourceCount));
        std::array<float2,asmr::kAmbientSourceCount> locations{};
        for(int i=0;i<asmr::kAmbientSourceCount;++i){
            const auto pos=asmr::ambientSource(view_,i);
            locations[(size_t)i]=make_float2(
                (pos.x-view_.originX)/dx_-0.5f,
                (pos.y-view_.originY)/dx_-0.5f);
        }
        LAB_CUDA(cudaMemcpy(deviceAmbientPositions_,locations.data(),
            sizeof(locations),cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMalloc(&deviceImage_,
                           (size_t)kPressureW*kPressureH*
                           sizeof(uint32_t)));
        int zero=0;
        LAB_CUDA(cudaMemcpy(deviceAudioFrames_,&zero,sizeof(int),
                            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(deviceAmbientFrames_,&zero,sizeof(int),
                            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(deviceAmbientOn_,&zero,sizeof(int),
                            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaStreamCreate(&stream_));
        const dim3 threads(16,16);
        const dim3 blocks((nx_+15)/16,(ny_+15)/16);
        const float decay=std::exp(-dt_*0.6f);
        LAB_CUDA(cudaStreamBeginCapture(stream_,
                    cudaStreamCaptureModeGlobal));
        for(int t=0;t<asmr::kGraphBatch;++t){
            // Three pressure buffers return to the SAME orientation
            // after 126 time steps (126 is divisible by three).
            asmrPressureStep<<<blocks,threads,0,stream_>>>(
                fields_[(t+2)%3],fields_[(t+1)%3],fields_[t%3],
                deviceMask_,deviceAudio_,deviceAudioFrames_,
                deviceMouth_,deviceStep_,
                deviceAmbientPCM_,deviceAmbientFrames_,
                deviceAmbientOn_,deviceAmbientPositions_,
                deviceOutput_,t,
                nx_,ny_,dx_,dt_,decay,view_.originX,view_.originY,
                halfMicSeparation_,micYOffset_,
                0.0,1e12,false,deviceSpeaker_,speakerW_,speakerH_,
                speakerWidthCells_,speakerHeightCells_);
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
        const dim3 threads(16,16),blocks((nx_+15)/16,(ny_+15)/16);
        const float decay=std::exp(-dt_*0.6f);
        LAB_CUDA(cudaStreamBeginCapture(stream_,
                    cudaStreamCaptureModeGlobal));
        for(int t=0;t<asmr::kGraphBatch;++t)
            asmrPressureStep<<<blocks,threads,0,stream_>>>(
                fields_[(t+2)%3],fields_[(t+1)%3],fields_[t%3],
                deviceMask_,deviceAudio_,deviceAudioFrames_,
                deviceMouth_,deviceStep_,
                deviceAmbientPCM_,deviceAmbientFrames_,
                deviceAmbientOn_,deviceAmbientPositions_,
                deviceOutput_,t,
                nx_,ny_,dx_,dt_,decay,view_.originX,view_.originY,
                halfMicSeparation_,micYOffset_,
                absoluteStart,length,offline,
                deviceSpeaker_,speakerW_,speakerH_,
                speakerWidthCells_,speakerHeightCells_);
        LAB_CUDA(cudaStreamEndCapture(stream_,&graph_));
        LAB_CUDA(cudaGraphInstantiate(&execution_,graph_,0));
        LAB_CUDA(cudaStreamSynchronize(stream_));
    }

    void reset(){
        const size_t bytes=(size_t)nx_*ny_*sizeof(float);
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
    void uploadAmbient(const std::vector<float>& pcm,bool enabled){
        const int amount=(int)std::min(pcm.size(),
            (size_t)asmr::kOutputRate*asmr::kAmbientNoiseSeconds);
        if(amount>0)LAB_CUDA(cudaMemcpy(
            deviceAmbientPCM_,pcm.data(),(size_t)amount*sizeof(float),
            cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(deviceAmbientFrames_,&amount,sizeof(int),
            cudaMemcpyHostToDevice));
        setAmbientEnabled(enabled);
    }
    void setAmbientEnabled(bool enabled){
        const int value=enabled?1:0;
        LAB_CUDA(cudaMemcpy(deviceAmbientOn_,&value,sizeof(int),
            cudaMemcpyHostToDevice));
    }
    std::array<float,asmr::kGraphBatch*2> process(
        const std::array<float4,asmr::kGraphBatch>& mouths,
        uint64_t firstStep){
        // Each source step is a float4: mouth XY + artist centre XY.
        // Copying float2 here uploaded only HALF of the positions!
        static_assert(sizeof(float4)==4*sizeof(float));
        LAB_CUDA(cudaMemcpy(deviceMouth_,mouths.data(),
            sizeof(mouths),cudaMemcpyHostToDevice));
        LAB_CUDA(cudaMemcpy(deviceStep_,&firstStep,
            sizeof(uint64_t),cudaMemcpyHostToDevice));
        LAB_CUDA(cudaGraphLaunch(execution_,stream_));
        LAB_CUDA(cudaStreamSynchronize(stream_));
        std::array<float,asmr::kGraphBatch*2> result{};
        LAB_CUDA(cudaMemcpy(result.data(),deviceOutput_,
            result.size()*sizeof(float),cudaMemcpyDeviceToHost));
        return result;
    }
    void pressureImage(std::vector<uint32_t>& pixels,float exposure,
                       int imageW=kPressureW,int imageH=kPressureH){
        imageW=std::clamp(imageW,1,kPressureW);
        imageH=std::clamp(imageH,1,kPressureH);
        pixels.resize((size_t)imageW*imageH);
        const dim3 thr(16,16);
        const dim3 blk((imageW+15)/16,(imageH+15)/16);
        pressureImageKernel<<<blk,thr,0,stream_>>>(
            fields_[1],deviceMask_,deviceImage_,nx_,ny_,
            imageW,imageH,exposure);
        LAB_CUDA(cudaGetLastError());
        LAB_CUDA(cudaStreamSynchronize(stream_));
        LAB_CUDA(cudaMemcpy(pixels.data(),deviceImage_,
            pixels.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    }
    float dx()const{return dx_;} // normalized coordinate cell
    float physicalDx()const{return dx_*physicsScale_;}
    float physicsScale()const{return physicsScale_;}
    float dt()const{return dt_;} // real physical seconds at F11 scale
    asmr::RoomView view()const{return view_;}
    int nx()const{return nx_;}
    int ny()const{return ny_;}
    ~FdtdRoom(){
        if(execution_)cudaGraphExecDestroy(execution_);
        if(graph_)cudaGraphDestroy(graph_);
        if(stream_)cudaStreamDestroy(stream_);
        for(auto& p:fields_)if(p)cudaFree(p);
        if(deviceMask_)cudaFree(deviceMask_);
        if(deviceMouth_)cudaFree(deviceMouth_);
        if(deviceSpeaker_)cudaFree(deviceSpeaker_);
        if(deviceStep_)cudaFree(deviceStep_);
        if(deviceOutput_)cudaFree(deviceOutput_);
        if(deviceAudio_)cudaFree(deviceAudio_);
        if(deviceAudioFrames_)cudaFree(deviceAudioFrames_);
        if(deviceAmbientPCM_)cudaFree(deviceAmbientPCM_);
        if(deviceAmbientFrames_)cudaFree(deviceAmbientFrames_);
        if(deviceAmbientOn_)cudaFree(deviceAmbientOn_);
        if(deviceAmbientPositions_)cudaFree(deviceAmbientPositions_);
        if(deviceImage_)cudaFree(deviceImage_);
    }
    FdtdRoom(const FdtdRoom&)=delete;
    FdtdRoom& operator=(const FdtdRoom&)=delete;
private:
    float dx_=0,dt_=0,physicsScale_=1;
    int nx_=0,ny_=0;
    asmr::RoomView view_{};
    float halfMicSeparation_=0,micYOffset_=0;
    float speakerWidthCells_=0,speakerHeightCells_=0;
    int speakerW_=0,speakerH_=0;
    float* fields_[3]{};
    float* deviceMask_=nullptr,*deviceAudio_=nullptr;
    float4* deviceMouth_=nullptr;
    unsigned char* deviceSpeaker_=nullptr;
    uint64_t* deviceStep_=nullptr;
    float* deviceOutput_=nullptr;
    int* deviceAudioFrames_=nullptr;
    float* deviceAmbientPCM_=nullptr;
    int* deviceAmbientFrames_=nullptr,*deviceAmbientOn_=nullptr;
    float2* deviceAmbientPositions_=nullptr;
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
static std::array<float4,asmr::kGraphBatch> mouthBatch(
    const std::vector<asmr::PathPoint>& track,double start,
    double dt,uint64_t baseStep,float dx,
    asmr::Vec2& physicalSource,const asmr::Vec2& liveTarget,
    bool offline,float originX=0,float originY=0) {
    std::array<float4,asmr::kGraphBatch> result{};
    for(int k=0;k<asmr::kGraphBatch;++k) {
        const double elapsed=(double)(baseStep+(uint64_t)k)*dt;
        const asmr::Vec2 target=offline?
            asmr::interpolate(track,start+elapsed):liveTarget;
        physicalSource=asmr::advanceSource(
            physicalSource,target,(float)dt,0.012f,2.25f);
        const auto m=asmr::mouth(physicalSource);
        result[k]=make_float4((m.x-originX)/dx-0.5f,
                              (m.y-originY)/dx-0.5f,
                              (physicalSource.x-originX)/dx-0.5f,
                              (physicalSource.y-originY)/dx-0.5f);
    }
    return result;
}

static asmr::RoomView gView=asmr::roomView();
// Draw the real physical aspect of the FDTD domain. A square acoustic
// room gets a centered 540x540 viewport, NOT stretched over 540x960.
static SDL_Rect roomViewport(){
    const float ratio=gView.width/gView.height;
    constexpr float screenRatio=(float)kWindowW/kWindowH;
    if(ratio>screenRatio+0.0001f){
        const int h=(int)std::lround(kWindowW*gView.height/gView.width);
        return {0,(kWindowH-h)/2,kWindowW,h};
    }
    const int w=(int)std::lround(kWindowH*ratio);
    return {(kWindowW-w)/2,0,w,kWindowH};
}
static int pixelsForMetres(float distance){
    const SDL_Rect v=roomViewport();
    return (int)std::lround(distance/gView.width*v.w);
}
static int screenX(float x){
    const SDL_Rect v=roomViewport();
    return v.x+(int)std::lround((x-gView.originX)/gView.width*v.w);
}
static int screenY(float y){
    const SDL_Rect v=roomViewport();
    return v.y+(int)std::lround((gView.originY+gView.height-y)/
                                gView.height*v.h);
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
    const int w=pixelsForMetres(shape.widthMetres);
    const int h=pixelsForMetres(shape.heightMetres);
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
    asmr::Vec2 target, // exact mouse/keyboard target, not acoustic shape
    float progress,bool processing,bool playing,
    float playbackFraction,bool mouseMode,
    SDL_Texture* waves,bool showPressure,bool showSvgGuide,
    const asmr::SvgSilhouette& listenerShape,
    const asmr::SvgSilhouette& speakerShape,
    SDL_Texture* listenerTexture,SDL_Texture* speakerTexture,
    bool channels){
    color(renderer,9,14,21);SDL_RenderClear(renderer);
    const SDL_Rect viewport=roomViewport();
    // GPU pressure and the SVG/mic/cursor share ONE physical viewport.
    if(showPressure && waves)
        SDL_RenderCopy(renderer,waves,nullptr,&viewport);
    // Model *one* physical reflecting perimeter, not seven nested
    // outlines that look like a crowd of extra walls when zoomed.
    const int wallPx=std::max(2,pixelsForMetres(0.012f));
    color(renderer,98,111,135);
    SDL_Rect walls[4]{
       {viewport.x,viewport.y,wallPx,viewport.h},
       {viewport.x+viewport.w-wallPx,viewport.y,wallPx,viewport.h},
       {viewport.x,viewport.y,viewport.w,wallPx},
       {viewport.x,viewport.y+viewport.h-wallPx,viewport.w,wallPx}
    };
    for(const auto& rect:walls)SDL_RenderFillRect(renderer,&rect);
    // Same exact normalized wall segments as the CUDA material mask;
    // fill across the wall thickness at every portrait/square zoom.
    color(renderer,40,174,145);
    const int leftH=std::max(2,pixelsForMetres(asmr::kLeftPortHeight));
    const int rightH=std::max(2,pixelsForMetres(asmr::kRightPortHeight));
    const int farW=std::max(2,pixelsForMetres(asmr::kFarPortWidth));
    SDL_Rect ports[3]{
       {viewport.x,
        viewport.y+(int)std::lround(
            (1.0f-asmr::kLeftPortCentre)*viewport.h)-leftH/2,
        wallPx,leftH},
       {viewport.x+viewport.w-wallPx,
        viewport.y+(int)std::lround(
            (1.0f-asmr::kRightPortCentre)*viewport.h)-rightH/2,
        wallPx,rightH},
       {viewport.x+(int)std::lround(
            asmr::kFarPortCentre*viewport.w)-farW/2,
        viewport.y,farW,wallPx}
    };
    for(const auto& port:ports)SDL_RenderFillRect(renderer,&port);
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
    // Overlay the *authored* smooth SVG as a low-opacity co-registration
    // guide, on top of the independent CUDA grid material. If a head
    // is misregistered, pressing B makes the two outlines obvious
    // instead of removing the author's drawing altogether.
    if(listenerTexture){
        const bool pressureVisible=showPressure && waves;
        if(!pressureVisible || showSvgGuide){
            SDL_SetTextureAlphaMod(listenerTexture,
                pressureVisible?78:255);
            drawHeadSvg(renderer,listenerTexture,listenerShape,
                        asmr::listener(),0.0);
            SDL_SetTextureAlphaMod(listenerTexture,255);
        }
    }else if(!showPressure || !waves){
        circle(renderer,screenX(asmr::kListenerX),
            screenY(asmr::kListenerY),pixelsForMetres(asmr::kHeadRadius),true);
    }
    // Physical listener centre: matches where the CUDA SVG mask is
    // stamped. The 2 microphone dots below are also world-registered.
    color(renderer,245,239,195);
    const int cx=screenX(asmr::kListenerX);
    const int cy=screenY(asmr::kListenerY);
    SDL_RenderDrawLine(renderer,cx-4,cy,cx+4,cy);
    SDL_RenderDrawLine(renderer,cx,cy-4,cx,cy+4);
    color(renderer,105,203,252);
    circle(renderer,screenX(asmr::leftEar(channels,listenerShape.pinnaV2).x),
           screenY(asmr::leftEar(channels,listenerShape.pinnaV2).y),3,true);
    color(renderer,245,158,245);
    circle(renderer,screenX(asmr::rightEar(channels,listenerShape.pinnaV2).x),
           screenY(asmr::rightEar(channels,listenerShape.pinnaV2).y),3,true);
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
            pixelsForMetres(asmr::kArtistRadius),!(showPressure&&waves));
    // The yellow spot is the *actual interpolated CUDA source*,
    // not a decorative mouth or artificial stereo panning.
    const asmr::Vec2 m=asmr::mouth(physicalSource);
    color(renderer,255,231,126);
    circle(renderer,screenX(m.x),screenY(m.y),2,true);
    // Mouse position stays 1:1 as a small hollow target marker,
    // distinct from the acoustically rate-limited SVG speaker.
    // This exposes source smoothing instead of making it look like
    // a 200px ghost of the speaker or a pressure rendering error.
    if(!processing&&!playing &&
       asmr::length(asmr::sub(target,artist))>0.012f){
        color(renderer,255,244,172);
        circle(renderer,screenX(target.x),screenY(target.y),5,false);
    }
    // Mouth orientation and 20-30cm peripersonal space reference.
    color(renderer,58,110,132);
    circle(renderer,screenX(asmr::kListenerX),
           screenY(asmr::kListenerY),pixelsForMetres(0.30f),false);
    // 25cm ruler: scaled to the physical 2m horizontal width.
    color(renderer,186,184,160);
    SDL_RenderDrawLine(renderer,24,kWindowH-20,
                       24+pixelsForMetres(0.25f),kWindowH-20);
    SDL_RenderDrawLine(renderer,24,kWindowH-25,24,kWindowH-15);
    SDL_RenderDrawLine(renderer,24+pixelsForMetres(0.25f),kWindowH-25,
                       24+pixelsForMetres(0.25f),kWindowH-15);
    if(processing){
        color(renderer,88,91,106);
        SDL_Rect bg{80,kWindowH-43,kWindowW-160,14};
        SDL_RenderFillRect(renderer,&bg);
        color(renderer,70,222,174);
        SDL_Rect bar=bg;
        bar.w=(int)std::lround(bg.w*std::clamp(progress,0.0f,1.0f));
        SDL_RenderFillRect(renderer,&bar);
    }else if(playing){
        SDL_Rect bar{80,kWindowH-43,kWindowW-160,14};
        color(renderer,88,91,106);
        SDL_RenderFillRect(renderer,&bar);
        color(renderer,70,222,174);
        bar.w=(int)std::lround(bar.w*
            std::clamp(playbackFraction,0.0f,1.0f));
        SDL_RenderFillRect(renderer,&bar);
        // A dot at the active phase is visible even with a tiny
        // fraction, and snaps back to the start on every loop.
        color(renderer,255,243,140);
        circle(renderer,80+bar.w,kWindowH-36,6,true);
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
// Post-solver attenuation audition. Work on already normalized,
 // unmodified 48kHz stereo PCM so the offline peak normalizer does
// NOT undo distance changes. Geometry uses the smoothed physical
// mouth path actually supplied to FDTD, never the raw mouse target.
// Every 2ms, evaluate the two microphone distances and interpolate
// gains between samples; this avoids per-audio-frame powf overhead.
static std::vector<float> distanceShapedClip(
    const std::vector<float>& reference,
    const std::vector<asmr::PathPoint>& physicalPath,
    bool channel,bool pinnaV2,int strength,
    float physicsScale=1.0f){
    std::vector<float> result=reference;
    if(strength<=0||physicalPath.empty())return result;
    const size_t frames=result.size()/2;
    const auto earL=asmr::leftEar(channel,pinnaV2);
    const auto earR=asmr::rightEar(channel,pinnaV2);
    size_t pathIndex=0;
    auto at=[&](size_t frame){
        const double time=(double)frame/asmr::kOutputRate;
        while(pathIndex+1<physicalPath.size() &&
              physicalPath[pathIndex+1].t<=time)++pathIndex;
        asmr::Vec2 source=physicalPath[pathIndex].centre;
        if(pathIndex+1<physicalPath.size()){
            const auto& a=physicalPath[pathIndex];
            const auto& b=physicalPath[pathIndex+1];
            const double denom=b.t-a.t;
            const float alpha=denom>1e-9?
                (float)std::clamp((time-a.t)/denom,0.0,1.0):0.0f;
            source=asmr::add(source,asmr::scale(
                asmr::sub(b.centre,source),alpha));
        }
        const auto m=asmr::mouth(source);
        return std::array<float,2>{
            asmr::distanceFalloffGain(
                asmr::length(asmr::sub(m,earL))*physicsScale,strength),
            asmr::distanceFalloffGain(
                asmr::length(asmr::sub(m,earR))*physicsScale,strength)};
    };
    constexpr size_t segment=96; // 2ms at 48kHz
    for(size_t first=0;first<frames;first+=segment){
        const size_t last=std::min(frames,first+segment);
        const auto a=at(first);
        const auto b=at(last);
        const float inv=1.0f/(float)(last-first);
        for(size_t i=first;i<last;++i){
            const float t=(float)(i-first)*inv;
            result[2*i]*=a[0]+(b[0]-a[0])*t;
            result[2*i+1]*=a[1]+(b[1]-a[1])*t;
        }
    }
    return result;
}
struct RenderJob {
    asmr::PathSelection selected;
    int window=0;
    uint64_t current=0,total=0;
    std::vector<float> samples;
    // The displayed replay head must follow the *same smoothed physical
    // centre* that fed moving CUDA impedance, not the unfiltered mouse path.
    std::vector<asmr::PathPoint> physicalPath;
    // Precomputed actual FDTD pressure snapshots for looping visual playback.
    std::vector<std::vector<uint32_t>> waveFrames;
    size_t nextWaveFrame=0;
    asmr::Vec2 sourceCursor{};
    std::unique_ptr<echo::StereoLinearResampler> resampler;
};
enum class Mode {Preview,Render,Playback};

static void printUsage(const char* executable){
    std::printf(
        "Usage: %s [--mp3 FILE] [--help]\n"
        "\nInput:\n"
        "  --mp3 FILE     Use FILE as the MP3 source instead of ~/test.mp3\n"
        "                 (absolute or relative to the working directory).\n"
        "  --help, -h     Print help without starting SDL or CUDA.\n"
        "\nControls:\n"
        "  Mouse         Move the speaker target within the acoustic room\n"
        "  M             Toggle mouse / keyboard movement\n"
        "  WASD, arrows  Move speaker in keyboard mode\n"
        "  H / Shift+H   Next / previous listener SVG; keeps zoom\n"
        "  Q / Shift+Q   Next / previous compatible grid quality\n"
        "  Z / Shift+Z   Next / previous FDTD zoom; keeps listener\n"
        "                1.00/1.25/1.55/2.00/SQUARE 0.75m\n"
        "                last view is a 0.75m square, still orbitable\n"
        "                (~85% -> 27.2% of original full-room cell-work)\n"
        "  F11 / Shift+F11 Cycle real physical scale in both directions\n"
        "                0.1 / 0.2 / 0.5 / 1 / 2 / 5 / 10x\n"
        "                Shift+F11 from 1x = 0.5x smaller room\n"
        "                small scales coarsen the GPU grid to cap work\n"
        "  F12           Cycle distance falloff: raw, 1,2,3,4,6,8,\n"
        "                10,12,16,20x extra 2D-to-3D pressure loss\n"
        "  Shift+F12     Previous distance-falloff strength\n"
        "                0x = original 2D waves, 1x = ~3D (1/r)\n"
        "                20x is exaggerated, not physically realistic\n"
        "  V             Toggle pressure field visualization\n"
        "  B             Toggle translucent original SVG on acoustic mask\n"
        "  9, 0          Decrease / increase pressure exposure\n"
        "  D             Toggle real-solver diagnostic chirp\n"
        "  T             Toggle live FDTD sound\n"
        "  L             Reload the selected MP3 source\n"
        "  R             Reset recorded path and solver\n"
        "  F5, F6, F7    Offline render/replay last 5, 10, 20 seconds\n"
        "  P             Replay last completed offline recording\n"
        "  Space         Stop replay, return to live mode\n"
        "  Esc           Quit\n"
        "\nAt high zoom the GPU may not meet real-time audio deadlines.\n",
        executable);
}

int main(int argc,char** argv){
    const char* home=std::getenv("HOME");
    std::string filename=std::string(home&&*home?home:".")+"/test.mp3";
    for(int i=1;i<argc;++i){
        const std::string arg=argv[i];
        if(arg=="--help"||arg=="-h"){printUsage(argv[0]);return 0;}
        if(arg=="--mp3"){
            if(i+1>=argc||argv[i+1][0]=='\0'||
               std::strncmp(argv[i+1],"--",2)==0){
                std::fprintf(stderr,"[ARG] --mp3 requires a file path.\n");
                printUsage(argv[0]);return 2;
            }
            filename=argv[++i];
        }else{
            std::fprintf(stderr,"[ARG] Unknown argument: %s\n",arg.c_str());
            printUsage(argv[0]);return 2;
        }
    }
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
        kWindowW,kWindowH,0);
    if(!window){std::fprintf(stderr,"[SDL] %s\n",SDL_GetError());
                SDL_Quit();return 1;}
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY,"0"); // nearest FDTD cells
    SDL_Renderer* renderer=SDL_CreateRenderer(window,-1,
        SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
    if(!renderer)renderer=SDL_CreateRenderer(window,-1,SDL_RENDERER_SOFTWARE);
    if(!renderer){SDL_DestroyWindow(window);SDL_Quit();return 1;}
    SDL_Texture* waveTexture=SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGBA32,SDL_TEXTUREACCESS_STREAMING,
        kPressureW,kPressureH);
    if(!waveTexture)
        std::fprintf(stderr,"[VISUAL] Cannot display CUDA pressure map: %s\n",
                     SDL_GetError());
    SDL_Texture* replayWaveTexture=SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGBA32,SDL_TEXTUREACCESS_STREAMING,
        kReplayW,kReplayH);
    if(!replayWaveTexture)
        std::fprintf(stderr,"[VISUAL] Replay pressure texture: %s\n",
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
    // No SDL format changes are permitted: the simulator outputs
    // exactly 48 kHz floating-point interleaved stereo. Never change
    // the device rate based on the CUDA solver's physical timestep.
    if(obtained.freq!=asmr::kOutputRate||obtained.channels!=2||
       obtained.format!=AUDIO_F32SYS){
        std::fprintf(stderr,
            "[AUDIO CLOCK] Unexpected device format: %d Hz, %d channels "
            "(SDL format %u). Refusing mismatched playback.\n",
            obtained.freq,(int)obtained.channels,
            (unsigned)obtained.format);
        SDL_CloseAudioDevice(device);
        SDL_DestroyRenderer(renderer);SDL_DestroyWindow(window);
        SDL_Quit();return 1;
    }
    std::printf("[AUDIO CLOCK] SDL output=%d Hz, %d channels, "
                "32-bit float PCM; FDTD temporal samples are "
                "resampled in PHYSICAL seconds to that fixed rate.\n",
                obtained.freq,(int)obtained.channels);
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
    int headIndex=0; // single source of truth for active H model
    bool channels=true; // V2 uses its own 12.4cm open pinna probes
    const auto* head=&listenerV2;
    SDL_Texture* speakerSprite=headTexture(
        renderer,speakerShape,237,141,98);
    SDL_Texture* simpleSprite=headTexture(
        renderer,listenerSimple,38,132,218);
    SDL_Texture* channelsSprite=headTexture(
        renderer,listenerChannels,38,132,218);
    SDL_Texture* pinnaV2Sprite=headTexture(
        renderer,listenerV2,38,132,218);
    const std::array<float,8> qualityDx{
        asmr::kLiveDx,asmr::kBalancedDx,asmr::kPreviousDx,
        asmr::kMediumDx,asmr::kUltraDx,
        0.006f,asmr::kLowDx,asmr::kLowestDx};
    const std::array<float,8> qualityBand{
        asmr::kLivePassband,asmr::kBalancedPassband,
        asmr::kPreviousPassband,asmr::kMediumPassband,
        asmr::kUltraPassband,5000.0f,
        asmr::kLowPassband,asmr::kLowestPassband};
    int qualityIndex=0;
    int falloffIndex=0; // F12: 0x = raw FDTD microphone pressure
    auto falloffStrength=[&](){return asmr::kDistanceFalloffSteps[(size_t)falloffIndex];};
    // Zoom area and dx are budgeted together: higher physical
    // magnification yet progressively FEWER cell updates per second.
    const auto& zoomSteps=asmr::kZoomSteps;
    int zoomIndex=0;
    int physicsScaleIndex=asmr::kDefaultPhysicsScaleIndex; // 1x default, independent H/Q/Z/F12
    auto physicsScaleValue=[&](){
        return asmr::kPhysicsScales[(size_t)physicsScaleIndex];};
    auto zoomValue=[&](){return zoomSteps[zoomIndex];};
    auto squareMode=[&](){return zoomIndex==4;};
    auto activeView=[&](){
        return asmr::roomView(zoomValue(),squareMode());};
    auto liveDx=[&](){
        return asmr::physicalScaleDesignDx(
            asmr::zoomDx(qualityDx[qualityIndex],zoomIndex),
            physicsScaleValue());
    };
    auto liveBand=[&](){return std::min(
        std::min(19000.0f,
            qualityBand[qualityIndex]*qualityDx[qualityIndex]/
                (liveDx()*physicsScaleValue())),
        0.90f*asmr::tenCellFrequency(
            liveDx()*physicsScaleValue()));};
    float liveCutoff=liveBand();
    std::vector<float> fullAudio,liveAudio;
    bool waveLoaded=loadSourceAudio(filename,fullAudio,liveAudio,
                                    liveCutoff);
    // Explicit D-key diagnostic: a ONE-SECOND CHIRP injected through
    // exactly the same moving-mouth CUDA FDTD solver as the MP3.
    // It never substitutes for ~/test.mp3 unless the user toggles it.
    std::vector<float> diagnosticAudio(asmr::kOutputRate);
    for(size_t i=0;i<diagnosticAudio.size();++i) {
        const double t=(double)i/asmr::kOutputRate;
        const double env=std::sin(echo::kPi*t);
        const double phase=2.0*echo::kPi*(320.0*t+1200.0*t*t);
        diagnosticAudio[i]=(float)(0.18*env*env*std::sin(phase));
    }
    bool diagnosticTone=false;

    auto live=std::make_unique<FdtdRoom>(
        liveDx(),*head,channels,speakerShape,
        zoomValue(),squareMode(),physicsScaleValue());
    gView=live->view();
    live->uploadAudio(diagnosticTone?diagnosticAudio:liveAudio);
    std::unique_ptr<FdtdRoom> highQuality;
    echo::StereoLinearResampler liveResampler(live->dt(),
                                                asmr::kOutputRate);
    std::vector<float> queued,blockSamples;
    queued.reserve(asmr::kGraphBatch*4*96);
    blockSamples.reserve(asmr::kGraphBatch*4);
    uint64_t previewStep=0;
    uint64_t clippedPreview=0,shortQueueWarnings=0;
    float previewSignalPeak=0.0f;
    // Acoustic seconds generated and actual GPU+transfer time over
    // each diagnostic window; report both to distinguish underruns
    // from filter/head acoustic coloration or an output-rate issue.
    uint64_t previewStepsWindow=0;
    double previewComputeSecondsWindow=0.0;
    uint64_t audioDiagTime=SDL_GetTicks64();
    asmr::Vec2 artist{asmr::kListenerX+0.50f,asmr::kListenerY},
                sourceCursor=artist;
    bool mouseMode=true,previewEnabled=true,quit=false;
    bool showPressure=true,showSvgGuide=true;
    asmr::RoomView lastPlaybackView=asmr::roomView();
    uint64_t lastWaveRefresh=0; // 30Hz signed-pressure display; never audio
    float pressureExposure=1.0f;
    Mode mode=Mode::Preview;
    SDL_WarpMouseInWindow(window,screenX(artist.x),screenY(artist.y));
    std::vector<asmr::PathPoint> record{{0.0,artist}};
    record.reserve(6000);
    double recordingClock=0;
    RenderJob job;
    std::vector<float> lastPlayback;
    std::vector<float> lastPlaybackBase; // normalized but NO F12 shaping
    std::vector<std::vector<uint32_t>> lastPlaybackWaves;
    int displayedPlaybackWave=-1;
    asmr::PathSelection lastPlaybackPath;
    std::vector<asmr::PathPoint> lastPlaybackPhysicalPath;
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
        displayedPlaybackWave=-1;
        mode=Mode::Playback;
    };
    uint64_t lastFrame=SDL_GetPerformanceCounter();
    const double perfFrequency=(double)SDL_GetPerformanceFrequency();
    bool primeAudioAfterSwitch=false;
    std::string selectorNotice;
    uint64_t selectorNoticeUntil=0;
    size_t previewFadeFrames=2400; // 50ms attack after solver state reset
    // One atomic transition for ALL H, Q and Z changes. In particular,
    // neither H nor Z is ever allowed to mutate the other dimension.
    // Invalid channel air simply refuses that candidate configuration.
    const auto candidateValid=[&](int h,int q,int z,
                                 float physicalScale){
        const auto& shape=h==0?listenerV2:
                          h==1?listenerChannels:listenerSimple;
        const auto view=asmr::roomView(
            zoomSteps[(size_t)z],z==(int)zoomSteps.size()-1);
        const float dx=asmr::physicalScaleDesignDx(
            asmr::zoomDx(qualityDx[(size_t)q],z),physicalScale);
        const int nx=(int)std::lround(view.width/dx);
        const int ny=(int)std::lround(view.height/dx);
        const auto mask=asmr::svgRoomMask(nx,ny,dx,shape,
                                        view.originX,view.originY);
        return asmr::validateMicrophones(mask,nx,ny,dx,h!=2,
                              shape,view.originX,view.originY);
    };
    const auto selectConfiguration=[&](int h,int q,int z,
                                      const char* changed,
                                      int requestedPhysics=-1)->bool{
        const int physicalIndex=requestedPhysics<0?
            physicsScaleIndex:requestedPhysics;
        const float physicalScale=
            asmr::kPhysicsScales[(size_t)physicalIndex];
        if(!candidateValid(h,q,z,physicalScale)){
            char message[256];
            std::snprintf(message,sizeof(message),
                "[%s] BLOCKED %s at Z %.2fx%s Q%d (mic not connected); "
                "CURRENT %s Z %.2fx",
                changed,h==0?"V2":h==1?"CHANNEL":"SIMPLE",
                zoomSteps[(size_t)z],z==4?" SQUARE":"",q,
                headIndex==0?"V2":headIndex==1?"CHANNEL":"SIMPLE",
                zoomValue());
            selectorNotice=message;
            selectorNoticeUntil=SDL_GetTicks64()+4000;
            std::fprintf(stderr,"%s\n",message);
            return false;
        }
        if(h==headIndex&&q==qualityIndex&&z==zoomIndex&&
           physicalIndex==physicsScaleIndex)return true;
        selectorNoticeUntil=0;
        const auto* newHead=h==0?&listenerV2:
                             h==1?&listenerChannels:&listenerSimple;
        const float dx=asmr::physicalScaleDesignDx(
            asmr::zoomDx(qualityDx[(size_t)q],z),physicalScale);
        const float nextCutoff=std::min(
            std::min(19000.0f,
                qualityBand[(size_t)q]*qualityDx[(size_t)q]/
                    (dx*physicalScale)),
            0.90f*asmr::tenCellFrequency(dx*physicalScale));
        std::vector<float> newAudio=waveLoaded?
            asmr::lowpass(fullAudio,nextCutoff,asmr::kOutputRate):
            std::vector<float>{};
        // SDL is paused during CUDA graph construction, preventing
        // an empty live queue from generating crackling while switching.
        SDL_PauseAudioDevice(device,1);
        SDL_ClearQueuedAudio(device);
        auto next=std::make_unique<FdtdRoom>(
            dx,*newHead,h!=2,speakerShape,zoomSteps[(size_t)z],
            z==(int)zoomSteps.size()-1,physicalScale);
        const auto preparedDiagnostic=diagnosticTone?
            asmr::lowpass(diagnosticAudio,nextCutoff,asmr::kOutputRate):
            std::vector<float>{};
        next->uploadAudio(diagnosticTone?preparedDiagnostic:newAudio);
        highQuality.reset();
        live=std::move(next);
        const bool changedRoom=z!=zoomIndex ||
                               physicalIndex!=physicsScaleIndex;
        headIndex=h;qualityIndex=q;zoomIndex=z;
        physicsScaleIndex=physicalIndex;
        head=newHead;channels=h!=2;
        liveCutoff=nextCutoff;liveAudio=std::move(newAudio);
        gView=live->view();
        artist=asmr::restrictCentreInView(artist,gView);
        sourceCursor=artist;
        previewStep=0;
        liveResampler=echo::StereoLinearResampler(
            live->dt(),asmr::kOutputRate);
        previewFadeFrames=0;
        primeAudioAfterSwitch=previewEnabled&&
                              (waveLoaded||diagnosticTone);
        if(!primeAudioAfterSwitch)SDL_PauseAudioDevice(device,0);
        if(changedRoom){
            recordingClock=0.0;record.clear();
            record.push_back({0.0,artist});
            SDL_WarpMouseInWindow(window,
                screenX(artist.x),screenY(artist.y));
        }
        // A completed WAV used the previous head/room, and must not
        // be replayed with the newly selected visible geometry.
        lastPlayback.clear();lastPlaybackBase.clear();
        lastPlaybackWaves.clear();
        lastPlaybackPhysicalPath.clear();
        std::fprintf(stderr,
            "[%s] ACTIVE head=%s Z=%.2fx%s Q=%d dx=%.3fmm "
            "room=%.3fx%.3fm cells=%dx%d expected FDTD "
            "cell-work=%.1f%% of old 5mm full room, "
            "F11 %gx physical, dx %.2fmm, LP %.0fHz.\n",
            changed,h==0?"V2":h==1?"CHANNEL":"SIMPLE",
            zoomValue(),squareMode()?" SQUARE":"",qualityIndex,
            live->dx()*1000,gView.width,gView.height,
            live->nx(),live->ny(),
            100.0f*asmr::physicalScaleWork(
                live->dx(),gView,physicsScaleValue()),
            physicsScaleValue(),live->physicalDx()*1000,
            liveCutoff);
        return true;
    };

    std::printf("\n[ASMR LAB] Room 2.0x3.56m (portrait 9:16). A fixed blue listener head has two "
       "FDTD ears; moving orange artist aims mouth at its centre.\n");
    std::printf("[CONTROLS] Mouse follows cursor 1:1 (M toggles mouse). "
       "WASD/arrows move with mouse disabled.\n");
    std::printf("[CONTROLS] F5 = render last 5s, F6 = 10s, F7 = 20s "
       "in HIGH quality, then play the complete finished WAV.\n");
    std::printf("[CONTROLS] P loop replay, SPACE return to live, "
       "L reload selected MP3, D FDTD test chirp, T toggle live, "
       "H V2/channel/simple, Q quality, Z 1/1.25/1.55/2/square, "
       "F11 physical metres 0.1/0.2/0.5/1/2/5/10x, "
       "F12 distance 0x-20x (Shift reverses), "
       "V pressure waves, B authored mask guide, 9/0 exposure, "
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
            if((key==SDLK_h||key==SDLK_z||key==SDLK_q)&&
               mode!=Mode::Preview){
                selectorNotice=mode==Mode::Playback?
                    "[H/Z/Q] REPLAY ACTIVE: press SPACE to return to live":
                    "[H/Z/Q] OFFLINE RENDER ACTIVE: navigation locked";
                selectorNoticeUntil=SDL_GetTicks64()+4000;
            }
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
            if(key==SDLK_F12){
                const int oldStrength=falloffStrength();
                const bool reverse=(event.key.keysym.mod&KMOD_SHIFT)!=0;
                const int count=(int)asmr::kDistanceFalloffSteps.size();
                falloffIndex=(falloffIndex+count+(reverse?-1:1))%count;
                const int strength=falloffStrength();
                std::fprintf(stderr,
                    "[F12 DISTANCE] %dx  (old %dx). 0x=raw 2D FDTD, "
                    "1x=approx 3D amplitude 1/r, "
                    "20x=exaggerated. Reference %.0fcm.\n",
                    strength,oldStrength,
                    asmr::kFalloffReferenceMetres*100.0f);
                if(mode==Mode::Playback && !lastPlaybackBase.empty()){
                    // Reapply instantly to the *existing* offline FDTD
                    // recording, without rerendering a single CUDA step
                    // or restarting the WAV timeline at t=0.
                    const size_t frames=lastPlaybackBase.size()/2;
                    const size_t atFrame=std::min(frames-1,
                        (size_t)std::max(0.0,std::floor(
                            playbackPhase*asmr::kOutputRate)));
                    auto updated=distanceShapedClip(lastPlaybackBase,
                        lastPlaybackPhysicalPath,channels,head->pinnaV2,
                        strength,physicsScaleValue());
                    std::vector<float> tail(
                        updated.begin()+2*atFrame,updated.end());
                    // An instantaneous change of exponent otherwise
                    // clicks: crossfade 20ms from the previous PCM.
                    const size_t fade=std::min(tail.size()/2,
                        (size_t)asmr::kOutputRate/50);
                    for(size_t n=0;n<fade;++n){
                        const float fraction=(float)n/(float)fade;
                        for(int ear=0;ear<2;++ear){
                            tail[2*n+ear]=
                                (1.0f-fraction)*
                                    lastPlayback[2*(atFrame+n)+ear]+
                                fraction*tail[2*n+ear];
                        }
                    }
                    SDL_PauseAudioDevice(device,1);
                    SDL_ClearQueuedAudio(device);
                    lastPlayback=std::move(updated);
                    playbackEnqueuedCycles=1;
                    if(SDL_QueueAudio(device,tail.data(),
                        (Uint32)(tail.size()*sizeof(float)))!=0)
                        std::fprintf(stderr,
                            "[F12 DISTANCE] SDL audio queue: %s\n",
                            SDL_GetError());
                    if(tail.size()/2<
                       (size_t)asmr::kOutputRate/4){
                        if(SDL_QueueAudio(device,lastPlayback.data(),
                            (Uint32)(lastPlayback.size()*sizeof(float)))==0)
                            ++playbackEnqueuedCycles;
                    }
                    playbackPhase=(double)atFrame/
                                      asmr::kOutputRate;
                    displayedPlaybackWave=-1;
                    SDL_PauseAudioDevice(device,0);
                    char wavPath[80];
                    std::snprintf(wavPath,sizeof(wavPath),
                        "asmr_last_%02ds.wav",lastWindow);
                    if(!writeWav(wavPath,lastPlayback))
                        std::fprintf(stderr,
                            "[F12 DISTANCE] Could not update %s\n",
                            wavPath);
                }else if(mode==Mode::Preview &&
                         previewEnabled&&(waveLoaded||diagnosticTone)){
                    // Change gain immediately, not after the 160ms of
                    // audio queued under the previous F12 setting.
                    // Do NOT reset/rewind the live wavefield.
                    SDL_PauseAudioDevice(device,1);
                    SDL_ClearQueuedAudio(device);
                    previewFadeFrames=0;
                    primeAudioAfterSwitch=true;
                }
                char notice[128];
                std::snprintf(notice,sizeof(notice),
                    "F12 DISTANCE %dx | %s",
                    strength,strength==0?"RAW 2D":
                        strength==1?"~3D 1/r PRESSURE":
                                     "EXAGGERATED FALL-OFF");
                selectorNotice=notice;
                selectorNoticeUntil=SDL_GetTicks64()+2300;
            }
            if(key==SDLK_v)showPressure=!showPressure;
            if(key==SDLK_b)showSvgGuide=!showSvgGuide;
            if(key==SDLK_9)pressureExposure=std::max(
                0.025f,pressureExposure/1.4f);
            if(key==SDLK_0)pressureExposure=std::min(
                500.0f,pressureExposure*1.4f);
            if(key==SDLK_d && mode==Mode::Preview) {
                diagnosticTone=!diagnosticTone;
                const auto filteredDiagnostic=diagnosticTone?
                    asmr::lowpass(diagnosticAudio,liveCutoff,
                                  asmr::kOutputRate):
                    std::vector<float>{};
                live->uploadAudio(diagnosticTone?
                    filteredDiagnostic:liveAudio);
                live->reset();previewStep=0;sourceCursor=artist;
                liveResampler=echo::StereoLinearResampler(
                    live->dt(),asmr::kOutputRate);
                SDL_ClearQueuedAudio(device);
                std::printf("[FDTD PROBE] %s: 320-2720Hz chirp goes "
                    "through the REAL moving-source FDTD path, "
                    "even if ~/test.mp3 is missing.\n",
                    diagnosticTone?"ON":"OFF");
            }
            if(key==SDLK_t) {
                previewEnabled=!previewEnabled;
                SDL_ClearQueuedAudio(device);
                live->reset();previewStep=0;
                sourceCursor=artist;
                liveResampler=echo::StereoLinearResampler(
                    live->dt(),asmr::kOutputRate);
            }
            if(key==SDLK_F11){
                if(mode==Mode::Preview){
                    const bool reverse=(event.key.keysym.mod&KMOD_SHIFT)!=0;
                    const int next=asmr::nextPhysicalScaleIndex(
                        physicsScaleIndex,reverse);
                    if(selectConfiguration(headIndex,qualityIndex,zoomIndex,
                                           "F11",next)){
                        char notice[200];
                        std::snprintf(notice,sizeof(notice),
                            "F11 PHYSICS %gx | ROOM %.2fx%.2fm | "
                            "REAL DX %.2fmm | BAND %.0fHz | Q%d Z %.2fx",
                            physicsScaleValue(),
                            live->view().width*physicsScaleValue(),
                            live->view().height*physicsScaleValue(),
                            live->physicalDx()*1000,
                            liveCutoff,qualityIndex,zoomValue());
                        selectorNotice=notice;
                        selectorNoticeUntil=SDL_GetTicks64()+3500;
                    }
                }else{
                    selectorNotice="F11: press SPACE / return to LIVE first";
                    selectorNoticeUntil=SDL_GetTicks64()+3500;
                }
            }
            if(key==SDLK_h && mode==Mode::Preview){
                const bool reverse=(event.key.keysym.mod&KMOD_SHIFT)!=0;
                const auto next=asmr::nextSelection(
                    {headIndex,qualityIndex,zoomIndex},'H',reverse);
                selectConfiguration(next.head,next.quality,next.zoom,"H");
            }
            if(key==SDLK_q && mode==Mode::Preview){
                const bool reverse=(event.key.keysym.mod&KMOD_SHIFT)!=0;
                auto next=asmr::LabSelection{headIndex,qualityIndex,zoomIndex};
                bool applied=false;
                for(size_t trial=0;trial<qualityDx.size()-1;++trial){
                    next=asmr::nextSelection(next,'Q',reverse,
                                            (int)qualityDx.size());
                    if(selectConfiguration(next.head,next.quality,
                                           next.zoom,"Q")){
                        applied=true;break;
                    }
                }
                if(!applied)std::fprintf(stderr,
                    "[Q] No compatible quality preset for this H/Z.\n");
            }
            if(key==SDLK_z && mode==Mode::Preview){
                const bool reverse=(event.key.keysym.mod&KMOD_SHIFT)!=0;
                const auto next=asmr::nextSelection(
                    {headIndex,qualityIndex,zoomIndex},'Z',reverse);
                selectConfiguration(next.head,next.quality,next.zoom,"Z");
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
            const auto filteredDiagnostic=diagnosticTone?
                asmr::lowpass(diagnosticAudio,liveCutoff,
                              asmr::kOutputRate):
                std::vector<float>{};
            live->uploadAudio(diagnosticTone?filteredDiagnostic:liveAudio);
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
                        asmr::physicalScaleDesignDx(
                            asmr::zoomDx(asmr::kOfflineDx,zoomIndex),
                            physicsScaleValue()),
                        *head,channels,
                        speakerShape,zoomValue(),squareMode(),
                        physicsScaleValue());
                const float offlineCutoff=std::min(
                    asmr::kOfflinePassband,
                    0.90f*asmr::tenCellFrequency(
                        highQuality->physicalDx()));
                const auto offlineAudio=asmr::lowpass(
                    fullAudio,offlineCutoff,asmr::kOutputRate);
                highQuality->uploadAudio(offlineAudio);
                std::printf("[F11 OFFLINE] physical dx %.2fmm, "
                    "band-limited MP3 to %.0fHz, F11 %gx.\n",
                    highQuality->physicalDx()*1000,offlineCutoff,
                    physicsScaleValue());
                highQuality->setRenderTiming(job.selected.start,
                    (double)captureSeconds,true);
                highQuality->reset();
                job.window=captureSeconds;
                job.current=0;
                job.sourceCursor=asmr::interpolate(
                    job.selected.points,job.selected.start);
                job.physicalPath.clear();
                job.physicalPath.push_back({0.0,job.sourceCursor});
                // The render is EXACTLY the requested length in
                // 48kHz stereo frames. Two extra acoustic steps ensure
                // the resampler never runs short of its final frame.
                job.total=(uint64_t)std::ceil(
                    (double)job.window/highQuality->dt())+2;
                job.samples.clear();
                job.samples.reserve(
                    asmr::clipFrames(job.window)*2+16);
                job.waveFrames.clear();job.nextWaveFrame=0;
                job.waveFrames.reserve((size_t)job.window*
                                       kReplayPressureFps);
                job.resampler=std::make_unique<echo::StereoLinearResampler>(
                    highQuality->dt(),asmr::kOutputRate);
                SDL_ClearQueuedAudio(device);
                mode=Mode::Render;
                gView=highQuality->view();
                std::printf("[RENDER] Captured last %.2fs; "
                  "offline %.1fmm / %.0fHz. %.0f million grid steps.\n",
                  job.selected.duration,highQuality->dx()*1000,
                  asmr::kOfflinePassband,
                  (double)job.total*highQuality->nx()*
                      highQuality->ny()/1e6);
            }
        }
        if(mode==Mode::Preview){
            if(mouseMode){
                int x=0,y=0;
                SDL_GetMouseState(&x,&y);
                const SDL_Rect viewport=roomViewport();
                const float u=std::clamp(
                    ((float)(x-viewport.x)+0.5f)/viewport.w,0.0f,1.0f);
                const float v=std::clamp(
                    ((float)(y-viewport.y)+0.5f)/viewport.h,0.0f,1.0f);
                const asmr::Vec2 wanted{
                    gView.originX+gView.width*u,
                    gView.originY+gView.height*(1.0f-v)};
                artist=asmr::moveArtistInView(artist,wanted,gView);
            }else{
                const Uint8* keys=SDL_GetKeyboardState(nullptr);
                const float step=frameDt*0.40f;
                float dx=0,dy=0;
                if(keys[SDL_SCANCODE_W]||keys[SDL_SCANCODE_UP])dy+=step;
                if(keys[SDL_SCANCODE_S]||keys[SDL_SCANCODE_DOWN])dy-=step;
                if(keys[SDL_SCANCODE_A]||keys[SDL_SCANCODE_LEFT])dx-=step;
                if(keys[SDL_SCANCODE_D]||keys[SDL_SCANCODE_RIGHT])dx+=step;
                artist=asmr::moveArtistInView(artist,
                    {artist.x+dx,artist.y+dy},gView);
            }
            recordingClock+=wallSeconds;
            record.push_back({recordingClock,artist});
            // Maintain an uninterrupted 60s circular time history.
            while(record.size()>2 &&
                  record[1].t<recordingClock-60.0)
                record.erase(record.begin());

            if(previewEnabled&&(waveLoaded||diagnosticTone)){
                const int queuedFrames=(int)(
                    SDL_GetQueuedAudioSize(device)/(2*sizeof(float)));
                if(previewStep>0 && queuedFrames<24)
                    ++shortQueueWarnings;
                // 160ms jitter cushion. Recompute graph-batch budget
                // from physical dt: at finely resolved zoom a fixed
                // 144 batches could NEVER refill 120ms and caused
                // consistent crackles independent of raw GPU speed.
                constexpr double kAudioTargetSeconds=0.160;
                int need=std::max(0,(int)(kAudioTargetSeconds*
                                    asmr::kOutputRate)-queuedFrames);
                const int maxBatches=(int)std::ceil(
                    kAudioTargetSeconds/
                    (live->dt()*asmr::kGraphBatch))+8;
                int batches=0;
                queued.clear();
                // The physical source position persists ACROSS the
                // CUDA batches. Previously the same mouse sweep was
                // replayed from start to finish for every mini-batch.
                while(need>0 && batches<maxBatches){
                    const auto points=mouthBatch(
                        {},0,live->dt(),previewStep,live->dx(),
                        sourceCursor,artist,false,
                        live->view().originX,live->view().originY);
                    const uint64_t gpuStart=SDL_GetPerformanceCounter();
                    const auto audio=live->process(points,previewStep);
                    const uint64_t gpuEnd=SDL_GetPerformanceCounter();
                    previewComputeSecondsWindow+=
                        (gpuEnd-gpuStart)/perfFrequency;
                    previewStepsWindow+=asmr::kGraphBatch;
                    previewStep+=asmr::kGraphBatch;
                    blockSamples.clear();
                    // The wavefield is unchanged. Amplify neither
                    // stereo channel; only add frequency-independent
                    // distance-dependent loss to the sampled pressure.
                    // Mouth travels <~3mm within one 126-step graph,
                    // so a midpoint gain avoids 200k powf calls/sec.
                    const auto& mid=points[asmr::kGraphBatch/2];
                    const asmr::Vec2 physicalMouth{
                        live->view().originX+
                            (mid.x+0.5f)*live->dx(),
                        live->view().originY+
                            (mid.y+0.5f)*live->dx()};
                    const float falloffL=asmr::distanceFalloffGain(
                        asmr::length(asmr::sub(physicalMouth,
                            asmr::leftEar(channels,head->pinnaV2)))*
                            physicsScaleValue(),
                        falloffStrength());
                    const float falloffR=asmr::distanceFalloffGain(
                        asmr::length(asmr::sub(physicalMouth,
                            asmr::rightEar(channels,head->pinnaV2)))*
                            physicsScaleValue(),
                        falloffStrength());
                    for(int n=0;n<asmr::kGraphBatch;++n)
                        liveResampler.push(
                            audio[n*2]*falloffL,
                            audio[n*2+1]*falloffR,blockSamples);
                    queued.insert(queued.end(),blockSamples.begin(),
                                  blockSamples.end());
                    need-=(int)blockSamples.size()/2;
                    ++batches;
                }
                // Preserve actual FDTD pressure differences linearly.
                // No always-on tanh distortion/harmonic generator.
                for(size_t frame=0;frame<queued.size()/2;++frame){
                    const float fade=std::min(1.0f,
                        (float)previewFadeFrames/
                        (0.050f*asmr::kOutputRate));
                    if(previewFadeFrames<asmr::kOutputRate)++previewFadeFrames;
                    for(int ch=0;ch<2;++ch){
                        float& v=queued[frame*2+ch];
                        // Requested square-room MP3 output trim:
                        // 75% amplitude relative to all other zooms.
                        const float squareGain=squareMode()?
                            asmr::kSquareMp3Gain:1.0f;
                        v=std::isfinite(v)?
                            0.55f*squareGain*fade*v:0.0f;
                        if(std::fabs(v)>0.85f)++clippedPreview;
                        v=std::clamp(v,-0.85f,0.85f);
                        previewSignalPeak=std::max(
                            previewSignalPeak,std::fabs(v));
                    }
                }
                if(!queued.empty()){
                    if(SDL_QueueAudio(device,queued.data(),
                        (Uint32)(queued.size()*sizeof(float)))!=0)
                        std::fprintf(stderr,"[SDL AUDIO] %s\n",
                                     SDL_GetError());
                    else if(primeAudioAfterSwitch){
                        // Unpause only AFTER a full new solver-time
                        // audio queue is ready, never mid-rebuild.
                        SDL_PauseAudioDevice(device,0);
                        primeAudioAfterSwitch=false;
                    }
                }
            }
        }else if(mode==Mode::Render){
            // A bounded number of expensive GPU graph batches per UI
            // frame keeps progress and Escape responsive. No audio is
            // queued until the ENTIRE high-resolution render is ready.
            for(int batch=0;batch<4 && job.current<job.total;++batch){
                const auto sources=mouthBatch(
                    job.selected.points,job.selected.start,
                    highQuality->dt(),job.current,
                    highQuality->dx(),job.sourceCursor,artist,true,
                    highQuality->view().originX,
                    highQuality->view().originY);
                const auto samples=highQuality->process(
                    sources,job.current);
                const uint64_t count=std::min(
                    (uint64_t)asmr::kGraphBatch,job.total-job.current);
                for(uint64_t i=0;i<count;++i)
                    job.resampler->push(
                        samples[2*i],samples[2*i+1],job.samples);
                job.current+=count;
                job.physicalPath.push_back({
                    (double)job.current*highQuality->dt(),
                    job.sourceCursor});
            }
            // Keep 24 fps REAL FDTD pressure snapshots separate from
            // untouched high-resolution stereo sound computation.
            const size_t targetWaveFrames=(size_t)job.window*
                                          kReplayPressureFps;
            const double simulatedTime=job.current*highQuality->dt();
            if(job.nextWaveFrame<targetWaveFrames &&
               simulatedTime >= (double)job.nextWaveFrame/
                                kReplayPressureFps){
                std::vector<uint32_t> frame;
                highQuality->pressureImage(frame,pressureExposure,
                                            kReplayW,kReplayH);
                job.waveFrames.push_back(std::move(frame));
                ++job.nextWaveFrame;
            }
            if(job.current>=job.total){
                // DO NOT derive the playback length from the number
                // of FDTD steps, block boundaries or float dt.
                // Both saved WAV and SDL loop are N*48000 frames.
                asmr::fitStereoClip(job.samples,job.window);
                makeSafePlayback(job.samples);
                if(squareMode()){
                    // Apply 75% AFTER offline peak normalization so
                    // the limiter cannot silently undo it.
                    for(float& sample:job.samples)
                        sample*=asmr::kSquareMp3Gain;
                }
                // F12 must run AFTER normalizing the unshaped FDTD
                // output, or the old automatic gain would cancel the
                // entire effect for far-away recorded movement.
                lastPlaybackBase=job.samples;
                job.samples=distanceShapedClip(lastPlaybackBase,
                    job.physicalPath,channels,head->pinnaV2,
                    falloffStrength(),physicsScaleValue());
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
                lastPlaybackView=highQuality->view();
                lastPlaybackPhysicalPath.swap(job.physicalPath);
                lastPlayback.swap(job.samples);
                while(job.waveFrames.size()<targetWaveFrames)
                    job.waveFrames.push_back(job.waveFrames.empty()?
                        std::vector<uint32_t>((size_t)kReplayW*
                            kReplayH,0xff150e09u):
                        job.waveFrames.back());
                lastPlaybackWaves.swap(job.waveFrames);
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
        const uint64_t audioDiagNow=SDL_GetTicks64();
        if(audioDiagNow-audioDiagTime>=5000){
            const double windowWallSeconds=
                (audioDiagNow-audioDiagTime)/1000.0;
            const double generatedSeconds=previewStepsWindow*live->dt();
            const double liveDelivery=generatedSeconds/
                std::max(0.001,windowWallSeconds);
            const double computeCapacity=generatedSeconds/
                std::max(0.000001,previewComputeSecondsWindow);
            std::printf("[ASMR FDTD] LIVE physical dx=%.2fmm F11 %.0fx cells=%dx%d "
                        "pressure steps=%llu queue=%.1fms "
                        "near-empty queues=%llu safety-clipped samples=%llu "
                        "generated/wall=%.2fx solver/compute=%.2fx "
                        "earPeak=%.5g MP3=%s preview=%s probe=%s "
                        "(V toggles REAL CUDA wave field)\n",
                live->physicalDx()*1000,physicsScaleValue(),
                live->nx(),live->ny(),
                (unsigned long long)previewStep,
                1000.0*SDL_GetQueuedAudioSize(device)/
                    (2.0*sizeof(float)*asmr::kOutputRate),
                (unsigned long long)shortQueueWarnings,
                (unsigned long long)clippedPreview,
                liveDelivery,computeCapacity,
                previewSignalPeak,waveLoaded?"loaded":"missing",
                previewEnabled?"on":"off",
                diagnosticTone?"chirp":"off");
            // METRES describes the CURRENT physically cropped FDTD
            // world, not just its 1x window. The source-to-ear flight
            // times still use true metres and 343 m/s at every zoom.
            const auto actualMouth=asmr::mouth(sourceCursor);
            const auto left=asmr::leftEar(channels,head->pinnaV2);
            const auto right=asmr::rightEar(channels,head->pinnaV2);
            const float dL=physicsScaleValue()*
                asmr::length(asmr::sub(actualMouth,left));
            const float dR=physicsScaleValue()*
                asmr::length(asmr::sub(actualMouth,right));
            const float dCentre=physicsScaleValue()*asmr::length(
                asmr::sub(actualMouth,asmr::listener()));
            std::printf("[METRES F11 %.0fx] Room=%.3fx%.3fm, "
                "listener=(%.3f,%.3f), mouse=(%.3f,%.3f), "
                "acousticHead=(%.3f,%.3f) mouth=(%.3f,%.3f), "
                "mouseLag=%.3fm centreRange=%.3fm, "
                "micSeparation=%.3fm, L=%.3fm/%.2fms, "
                "R=%.3fm/%.2fms; physical dx=%.4fm dt=%.2fus\n",
                physicsScaleValue(),
                live->view().width*physicsScaleValue(),
                live->view().height*physicsScaleValue(),
                asmr::kListenerX*physicsScaleValue(),
                asmr::kListenerY*physicsScaleValue(),
                artist.x*physicsScaleValue(),artist.y*physicsScaleValue(),
                sourceCursor.x*physicsScaleValue(),
                sourceCursor.y*physicsScaleValue(),
                actualMouth.x*physicsScaleValue(),
                actualMouth.y*physicsScaleValue(),
                asmr::length(asmr::sub(artist,sourceCursor))*
                    physicsScaleValue(),dCentre,
                asmr::micSpacing(channels,head->pinnaV2)*
                    physicsScaleValue(),
                dL,1000.0f*dL/echo::kSpeedOfSound,
                dR,1000.0f*dR/echo::kSpeedOfSound,
                live->physicalDx(),live->dt()*1e6f);
            shortQueueWarnings=0;clippedPreview=0;
            previewSignalPeak=0.0f;
            previewStepsWindow=0;
            previewComputeSecondsWindow=0.0;
            audioDiagTime=audioDiagNow;
        }
        gView=mode==Mode::Render&&highQuality?highQuality->view():
              mode==Mode::Playback?lastPlaybackView:live->view();
        const float progress=mode==Mode::Render&&job.total>0?
            (float)((double)job.current/job.total):0.0f;
        SDL_Texture* displayedWaves=waveTexture;
        if(showPressure && mode==Mode::Playback &&
           replayWaveTexture && !lastPlaybackWaves.empty()){
            // SDL-consumed audio frames also determine the visual
            // frame, so wave snapshots loop with the PCM exactly.
            const int index=std::clamp((int)std::floor(
                playbackPhase*kReplayPressureFps),0,
                (int)lastPlaybackWaves.size()-1);
            if(index!=displayedPlaybackWave) {
                SDL_UpdateTexture(replayWaveTexture,nullptr,
                    lastPlaybackWaves[index].data(),
                    kReplayW*sizeof(uint32_t));
                displayedPlaybackWave=index;
            }
            displayedWaves=replayWaveTexture;
        }else if(showPressure && waveTexture &&
                 SDL_GetTicks64()-lastWaveRefresh>=33){
            // GPU visualization at <=30Hz, never decimate the solver.
            FdtdRoom* active=(mode==Mode::Render && highQuality)?
                highQuality.get():live.get();
            active->pressureImage(pressurePixels,pressureExposure);
            SDL_UpdateTexture(waveTexture,nullptr,
                pressurePixels.data(),kPressureW*sizeof(uint32_t));
            lastWaveRefresh=SDL_GetTicks64();
        }
        // Never render the physical moving speaker at the cursor
        // target while the CUDA impedance silhouette still occupies
        // the smoothed source position. They must coincide.
        const bool livePhysicsActive=previewEnabled&&
                                     (waveLoaded||diagnosticTone);
        const asmr::Vec2 visibleArtist=mode==Mode::Render?
            job.sourceCursor:mode==Mode::Playback?
            asmr::interpolate(lastPlaybackPhysicalPath,playbackPhase):
            livePhysicsActive?sourceCursor:artist;
        const asmr::Vec2 visibleSource=mode==Mode::Render?
            job.sourceCursor:mode==Mode::Playback?
            visibleArtist:livePhysicsActive?sourceCursor:artist;
        const auto& visibleTrail=mode==Mode::Playback?
            lastPlaybackPath.points:record;
        drawRoom(renderer,visibleTrail,visibleArtist,visibleSource,
                 artist,progress,mode==Mode::Render,mode==Mode::Playback,
                 lastWindow>0?(float)(playbackPhase/lastWindow):0.0f,
                 mouseMode,displayedWaves,showPressure,showSvgGuide,
                 *head,speakerShape,
                 headIndex==0?pinnaV2Sprite:
                     headIndex==1?channelsSprite:simpleSprite,
                 speakerSprite,channels);
        char title[255];
        std::snprintf(title,sizeof(title),
            "ASMR %s | F11 %gx (%.2fx%.2fm) | "
            "Z %.2fx%s Q%d real dx %.1fmm LP %.0fHz | "
            "head:%s | F12 %dx | range %.1fm | work~%.0f%% | MP3:%s",
            mode==Mode::Render?"OFFLINE":mode==Mode::Playback?
                "WAV LOOP":"LIVE",
            physicsScaleValue(),
            gView.width*physicsScaleValue(),
            gView.height*physicsScaleValue(),
            gView.zoom,
            std::fabs(gView.width-gView.height)<0.01f?" SQUARE":"",
            qualityIndex,
            mode==Mode::Render&&highQuality?
                highQuality->physicalDx()*1000:live->physicalDx()*1000,
            mode==Mode::Render&&highQuality?
                std::min(asmr::kOfflinePassband,
                    0.90f*asmr::tenCellFrequency(
                        highQuality->physicalDx())):liveCutoff,
            headIndex==0?"V2":headIndex==1?"CHANNEL":"SIMPLE",
            falloffStrength(),
            physicsScaleValue()*asmr::length(asmr::sub(
                asmr::mouth(visibleSource),asmr::listener())),
            100.0f*asmr::physicalScaleWork(
                live->dx(),live->view(),physicsScaleValue()),
            diagnosticTone?"TEST":waveLoaded?"READY":"MISSING");
        SDL_SetWindowTitle(window,
            SDL_GetTicks64()<selectorNoticeUntil?
                selectorNotice.c_str():title);
        SDL_Delay(6);
    }
    SDL_ClearQueuedAudio(device);
    SDL_CloseAudioDevice(device);
    highQuality.reset();
    if(speakerSprite)SDL_DestroyTexture(speakerSprite);
    if(simpleSprite)SDL_DestroyTexture(simpleSprite);
    if(channelsSprite)SDL_DestroyTexture(channelsSprite);
    if(pinnaV2Sprite)SDL_DestroyTexture(pinnaV2Sprite);
    if(replayWaveTexture)SDL_DestroyTexture(replayWaveTexture);
    if(waveTexture)SDL_DestroyTexture(waveTexture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
