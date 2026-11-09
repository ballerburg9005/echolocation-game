// Standalone ASMR room experiment. No changes to the arcade executable.
// Default soundtrack: test.wav, optionally overridden by argv[1].
// Listener head is a REAL stationary 2D FDTD obstruction with two ears.
// Artist is a collision-safe moving visual head; its mouth is a real
// moving FDTD pressure emitter facing the listener on every path sample.
// Live: 1cm grid, ~2.8kHz source lowpass for cheaper live discovery.
// Offline: 2.5mm grid, 12kHz lowpass for 8kHz+ cues, processed BEFORE
// stutter-free 48kHz stereo WAV playback and writing an output file.
#include "asmr_lab_math.h"
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
    double absoluteStart,double recordingDuration,bool offline)
{
    const int x=blockIdx.x*blockDim.x+threadIdx.x;
    const int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=n||y>=n)return;
    const int idx=y*n+x;
    // Read the preceding, completed pressure field at BOTH ear points.
    // Unlike separate output kernels, this doesn't double graph launches.
    // The p field is read-only for the entirety of this CUDA kernel.
    if(x==0&&y==0){
        const float lx=(asmr::kListenerX-asmr::kEarOffset)/dx-0.5f;
        const float rx=(asmr::kListenerX+asmr::kEarOffset)/dx-0.5f;
        const float ey=asmr::kListenerY/dx-0.5f;
        outStereo[localStep*2]=echo::bilinearSample(now,n,lx,ey);
        outStereo[localStep*2+1]=echo::bilinearSample(now,n,rx,ey);
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

class FdtdRoom {
public:
    explicit FdtdRoom(float resolution):dx_(resolution),
      n_(asmr::cells(resolution)),dt_((float)asmr::stepSeconds(resolution)){
        const auto mask=asmr::roomMask(n_,dx_);
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
                n_,dx_,dt_,decay,0.0,1e12,false);
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
                n_,dx_,dt_,decay,absoluteStart,length,offline);
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
    }
    FdtdRoom(const FdtdRoom&)=delete;
    FdtdRoom& operator=(const FdtdRoom&)=delete;
private:
    float dx_=0,dt_=0;
    int n_=0;
    float* fields_[3]{};
    float* deviceMask_=nullptr,*deviceAudio_=nullptr;
    float2* deviceMouth_=nullptr;
    uint64_t* deviceStep_=nullptr;
    float* deviceOutput_=nullptr;
    int* deviceAudioFrames_=nullptr;
    cudaStream_t stream_=nullptr;
    cudaGraph_t graph_=nullptr;
    cudaGraphExec_t execution_=nullptr;
};

static bool loadWavFile(const std::string& filename,
                        std::vector<float>& full,
                        std::vector<float>& live){
    full.clear();live.clear();
    SDL_AudioSpec spec{};
    Uint8* input=nullptr;
    Uint32 inputBytes=0;
    if(!SDL_LoadWAV(filename.c_str(),&spec,&input,&inputBytes)){
        std::fprintf(stderr,
            "[WAV] Cannot load %s: %s. Path capture remains available.\n",
            filename.c_str(),SDL_GetError());
        return false;
    }
    SDL_AudioCVT cvt{};
    const int conversion=SDL_BuildAudioCVT(&cvt,
        spec.format,spec.channels,spec.freq,
        AUDIO_F32SYS,1,asmr::kOutputRate);
    if(conversion<0){
        SDL_FreeWAV(input);
        std::fprintf(stderr,"[WAV] Unsupported PCM conversion\n");
        return false;
    }
    cvt.len=(int)inputBytes;
    cvt.buf=(Uint8*)SDL_malloc(inputBytes*(size_t)std::max(1,cvt.len_mult));
    if(!cvt.buf){SDL_FreeWAV(input);return false;}
    std::memcpy(cvt.buf,input,inputBytes);
    SDL_FreeWAV(input);
    if(cvt.needed&&SDL_ConvertAudio(&cvt)!=0){
        std::fprintf(stderr,"[WAV] Convert failed: %s\n",SDL_GetError());
        SDL_free(cvt.buf);return false;
    }
    if(!cvt.needed)cvt.len_cvt=cvt.len;
    const float* data=(const float*)cvt.buf;
    const size_t count=std::min((size_t)cvt.len_cvt/sizeof(float),
                                (size_t)kMaxWavFrames);
    std::vector<float> normalized(count,0);
    for(size_t i=0;i<count;++i) {
        const float sample=data[i];
        normalized[i]=std::isfinite(sample)?
            std::clamp(sample,-1.0f,1.0f):0.0f;
    }
    SDL_free(cvt.buf);
    if(count<128)return false;
    if((size_t)cvt.len_cvt/sizeof(float)>count)
        std::fprintf(stderr,"[WAV] Limited input to first %d seconds\n",
                      kMaxWavSeconds);
    full=asmr::lowpass(normalized,asmr::kOfflinePassband,asmr::kOutputRate);
    live=asmr::lowpass(normalized,asmr::kLivePassband,asmr::kOutputRate);
    std::printf("[WAV] Loaded %s, %.2fs, mono 48kHz. "
        "Preview LP %.0fHz; offline LP %.0fHz.\n",
        filename.c_str(),(double)count/asmr::kOutputRate,
        asmr::kLivePassband,asmr::kOfflinePassband);
    return true;
}

static std::array<float2,asmr::kGraphBatch> mouthBatch(
    const std::vector<asmr::PathPoint>& track,double start,
    double dt,uint64_t baseStep,float dx,
    const asmr::Vec2 liveFrom,const asmr::Vec2 liveTo,bool offline){
    std::array<float2,asmr::kGraphBatch> result{};
    for(int k=0;k<asmr::kGraphBatch;++k){
        const double elapsed=(double)(baseStep+(uint64_t)k)*dt;
        asmr::Vec2 centre;
        if(offline)
            centre=asmr::interpolate(track,start+elapsed);
        else {
            const float alpha=(float)(k+1)/asmr::kGraphBatch;
            centre=asmr::add(asmr::scale(liveFrom,1-alpha),
                             asmr::scale(liveTo,alpha));
        }
        const auto m=asmr::mouth(centre);
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
    asmr::Vec2 artist,float progress,bool processing,bool playing,
    bool mouseMode){
    color(renderer,9,14,21);SDL_RenderClear(renderer);
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
    circle(renderer,screenX(asmr::kListenerX),
        screenY(asmr::kListenerY),
        screenX(asmr::kHeadRadius),true);
    color(renderer,105,203,252);
    circle(renderer,screenX(asmr::leftEar().x),
           screenY(asmr::leftEar().y),6,true);
    color(renderer,245,158,245);
    circle(renderer,screenX(asmr::rightEar().x),
           screenY(asmr::rightEar().y),6,true);
    // Artist head always points mouth toward the centre of the listener.
    color(renderer,236,141,107);
    circle(renderer,screenX(artist.x),screenY(artist.y),
        screenX(asmr::kArtistRadius),true);
    const asmr::Vec2 m=asmr::mouth(artist);
    color(renderer,255,231,126);
    circle(renderer,screenX(m.x),screenY(m.y),5,true);
    // Mouth orientation and 20-30cm peripersonal space reference.
    color(renderer,58,110,132);
    circle(renderer,screenX(asmr::kListenerX),
           screenY(asmr::kListenerY),screenX(0.30f),false);
    if(processing){
        color(renderer,88,91,106);
        SDL_Rect bg{80,kWindow-43,kWindow-160,14};
        SDL_RenderFillRect(renderer,&bg);
        color(renderer,70,222,174);
        SDL_Rect bar=bg;
        bar.w=(int)std::lround(bg.w*std::clamp(progress,0.0f,1.0f));
        SDL_RenderFillRect(renderer,&bar);
    }else if(playing){
        color(renderer,70,222,174);
        SDL_Rect bar{80,kWindow-43,kWindow-160,8};
        SDL_RenderFillRect(renderer,&bar);
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
    std::unique_ptr<echo::StereoLinearResampler> resampler;
};
enum class Mode {Preview,Render,Playback};

int main(int argc,char** argv){
    const std::string filename=argc>1?argv[1]:"test.wav";
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
    SDL_Renderer* renderer=SDL_CreateRenderer(window,-1,
        SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
    if(!renderer)renderer=SDL_CreateRenderer(window,-1,SDL_RENDERER_SOFTWARE);
    if(!renderer){SDL_DestroyWindow(window);SDL_Quit();return 1;}
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
    std::vector<float> fullAudio,liveAudio;
    bool waveLoaded=loadWavFile(filename,fullAudio,liveAudio);

    FdtdRoom live(asmr::kLiveDx);
    live.uploadAudio(liveAudio);
    std::unique_ptr<FdtdRoom> highQuality;
    echo::StereoLinearResampler liveResampler(live.dt(),
                                                asmr::kOutputRate);
    std::vector<float> queued;
    queued.reserve(asmr::kGraphBatch*4);
    uint64_t previewStep=0;
    asmr::Vec2 artist{1.50f,1.0f},previousArtist=artist;
    bool mouseMode=true,previewEnabled=true,quit=false;
    Mode mode=Mode::Preview;
    SDL_WarpMouseInWindow(window,screenX(artist.x),screenY(artist.y));
    std::vector<asmr::PathPoint> record{{0.0,artist}};
    record.reserve(6000);
    double recordingClock=0;
    RenderJob job;
    std::vector<float> lastPlayback;
    int lastWindow=5;
    uint64_t lastFrame=SDL_GetPerformanceCounter();
    const double perfFrequency=(double)SDL_GetPerformanceFrequency();

    std::printf("\n[ASMR LAB] Room 2x2m. A fixed blue listener head has two "
       "FDTD ears; moving orange artist aims mouth at its centre.\n");
    std::printf("[CONTROLS] Mouse follows cursor 1:1 (M toggles mouse). "
       "WASD/arrows move with mouse disabled.\n");
    std::printf("[CONTROLS] F5 = render last 5s, F6 = 10s, F7 = 20s "
       "in HIGH quality, then play the complete finished WAV.\n");
    std::printf("[CONTROLS] P = replay latest processed clip, "
       "L = reload WAV, T = live-preview on/off, R = reset path, Esc = exit.\n");
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
            if(key==SDLK_t)previewEnabled=!previewEnabled;
            if(key==SDLK_r){
                record.clear();recordingClock=0;
                record.push_back({0,artist});
                previewStep=0;live.reset();
                liveResampler=echo::StereoLinearResampler(
                    live.dt(),asmr::kOutputRate);
                SDL_ClearQueuedAudio(device);
                mode=Mode::Preview;
                std::printf("[PATH] Recording reset.\n");
            }
            if(key==SDLK_p&&!lastPlayback.empty()){
                mode=Mode::Playback;
                SDL_ClearQueuedAudio(device);
                SDL_QueueAudio(device,lastPlayback.data(),
                    (Uint32)(lastPlayback.size()*sizeof(float)));
                std::printf("[PLAYBACK] Replaying last processed %ds.\n",
                            lastWindow);
            }
        }
        if(reload){
            waveLoaded=loadWavFile(filename,fullAudio,liveAudio);
            live.uploadAudio(liveAudio);
            if(highQuality)highQuality->uploadAudio(fullAudio);
            if(mode==Mode::Preview){
                live.reset();previewStep=0;
                liveResampler=echo::StereoLinearResampler(
                    live.dt(),asmr::kOutputRate);
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
                        asmr::kOfflineDx);
                highQuality->uploadAudio(fullAudio);
                highQuality->setRenderTiming(job.selected.start,
                    job.selected.duration,true);
                highQuality->reset();
                job.window=captureSeconds;
                job.current=0;
                job.total=(uint64_t)std::ceil(
                    (job.selected.duration+0.22)/highQuality->dt());
                job.samples.clear();
                job.samples.reserve((size_t)(
                    (job.selected.duration+0.3)*asmr::kOutputRate)*2);
                job.resampler=std::make_unique<echo::StereoLinearResampler>(
                    highQuality->dt(),asmr::kOutputRate);
                SDL_ClearQueuedAudio(device);
                mode=Mode::Render;
                std::printf("[RENDER] Captured last %.2fs; "
                  "offline %.1fmm / %.0fHz. %.0f million grid steps.\n",
                  job.selected.duration,asmr::kOfflineDx*1000,
                  asmr::kOfflinePassband,
                  (double)job.total*800.0*800.0/1e6);
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
                int need=std::max(0,(int)(0.085*asmr::kOutputRate)-
                                    queuedFrames);
                int batches=0;
                while(need>0 && batches<64){
                    const auto points=mouthBatch({},0,live.dt(),
                        previewStep,live.dx(),previousArtist,artist,false);
                    const auto audio=live.process(points,previewStep);
                    previewStep+=asmr::kGraphBatch;
                    queued.clear();
                    for(int n=0;n<asmr::kGraphBatch;++n)
                        liveResampler.push(audio[n*2],audio[n*2+1],
                                           queued);
                    for(float& v:queued)
                        v=0.42f*std::tanh(2.0f*v); // safety, preview only
                    if(!queued.empty())
                        SDL_QueueAudio(device,queued.data(),
                            (Uint32)(queued.size()*sizeof(float)));
                    need-=(int)queued.size()/2;
                    ++batches;
                }
            }
            previousArtist=artist;
        }else if(mode==Mode::Render){
            // A bounded number of expensive GPU graph batches per UI
            // frame keeps progress and Escape responsive. No audio is
            // queued until the ENTIRE high-resolution render is ready.
            for(int batch=0;batch<4 && job.current<job.total;++batch){
                const auto sources=mouthBatch(
                    job.selected.points,job.selected.start,
                    highQuality->dt(),job.current,
                    highQuality->dx(),artist,artist,true);
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
                lastPlayback.swap(job.samples);
                SDL_ClearQueuedAudio(device);
                SDL_QueueAudio(device,lastPlayback.data(),
                    (Uint32)(lastPlayback.size()*sizeof(float)));
                mode=Mode::Playback;
                std::printf("[RENDER] Complete. Saved %s and %s; "
                    "now playing %.2fs entirely preprocessed audio.\n",
                    soundPath,trackPath,
                    (double)lastPlayback.size()/
                    (2*asmr::kOutputRate));
            }
        }else if(mode==Mode::Playback){
            if(SDL_GetQueuedAudioSize(device)==0){
                mode=Mode::Preview;
                live.reset();previewStep=0;
                liveResampler=echo::StereoLinearResampler(
                    live.dt(),asmr::kOutputRate);
                previousArtist=artist;
            }
        }
        const float progress=mode==Mode::Render&&job.total>0?
            (float)((double)job.current/job.total):0.0f;
        drawRoom(renderer,record,artist,progress,
                 mode==Mode::Render,mode==Mode::Playback,mouseMode);
        char title[255];
        std::snprintf(title,sizeof(title),
            "ASMR Room | %s | WAV:%s | path %.1fs | live %.0fHz / render %.0fHz | F5/F6/F7 = last 5/10/20s | M mouse | L reload | P replay",
            mode==Mode::Render?"RENDERING":mode==Mode::Playback?
                "PLAYING FINISHED WAV":"RECORDING",
            waveLoaded?"READY":"MISSING",
            record.empty()?0:record.back().t-record.front().t,
            asmr::kLivePassband,asmr::kOfflinePassband);
        SDL_SetWindowTitle(window,title);
        SDL_Delay(6);
    }
    SDL_ClearQueuedAudio(device);
    SDL_CloseAudioDevice(device);
    highQuality.reset();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
