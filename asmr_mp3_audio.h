#pragma once
// Decode an MP3 using the system libmpg123 library, not a shell command,
// temporary WAV file, or direct-to-headphone sound path. The only audio
// source for this experiment is ~/test.mp3; decoded mono PCM feeds CUDA FDTD.
#include <SDL2/SDL.h>
#include <mpg123.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace asmr {
// Decode the *entire* input. SDL_AudioStream converts/resamples each
// small mpg123 block instead of using one signed-int-sized SDL_AudioCVT
// buffer; a 4-minute MP3 must not silently become a 60-second loop.
// secondsLimit>0 is an explicit optional caller limit, not a default.
inline bool decodeMP3(const std::string& filename,
                      std::vector<float>& pcm,
                      std::string& error,
                      int outputRate=48000,int secondsLimit=0) {
    pcm.clear();
    error.clear();
    if(mpg123_init()!=MPG123_OK){
        error="libmpg123 initialization failed";return false;
    }
    int code=MPG123_OK;
    mpg123_handle* mh=mpg123_new(nullptr,&code);
    if(!mh){
        error="libmpg123 decoder allocation failed";
        mpg123_exit();return false;
    }
    const auto closeDecoder=[&](){
        mpg123_delete(mh);
        mpg123_exit();
    };
    if(mpg123_open_fixed(mh,filename.c_str(),MPG123_STEREO,
                         MPG123_ENC_SIGNED_16)!=MPG123_OK){
        error=std::string("Cannot open MP3: ")+mpg123_strerror(mh);
        closeDecoder();return false;
    }
    long nativeRate=0;
    int channels=0,encoding=0;
    if(mpg123_getformat(mh,&nativeRate,&channels,&encoding)!=MPG123_OK||
       nativeRate<8000||nativeRate>192000||channels!=2||
       encoding!=MPG123_ENC_SIGNED_16||outputRate<=0){
        error="Unsupported MP3 sample rate, format, or output rate";
        closeDecoder();return false;
    }
    SDL_AudioStream* stream=SDL_NewAudioStream(
        AUDIO_S16SYS,(Uint8)channels,(int)nativeRate,
        AUDIO_F32SYS,1,outputRate);
    if(!stream){
        error=std::string("MP3 resampler: ")+SDL_GetError();
        closeDecoder();return false;
    }
    std::array<unsigned char,16384> input{};
    std::array<float,8192> output{};
    const size_t frameBytes=(size_t)channels*sizeof(int16_t);
    const uint64_t maximum=secondsLimit>0?
        (uint64_t)nativeRate*(uint64_t)secondsLimit:UINT64_MAX;
    uint64_t nativeFrames=0;
    bool valid=true;
    const auto drain=[&]()->bool{
        while(SDL_AudioStreamAvailable(stream)>0){
            const int received=SDL_AudioStreamGet(
                stream,output.data(),(int)(output.size()*sizeof(float)));
            if(received<0){
                error=std::string("MP3 resampler read: ")+SDL_GetError();
                return false;
            }
            if(received==0)break;
            if(received%(int)sizeof(float)!=0){
                error="MP3 resampler emitted partial float";return false;
            }
            const size_t frames=(size_t)received/sizeof(float);
            for(size_t i=0;i<frames;++i)
                pcm.push_back(std::isfinite(output[i])?
                    std::clamp(output[i],-1.0f,1.0f):0.0f);
        }
        return true;
    };
    while(nativeFrames<maximum){
        size_t received=0;
        const int status=mpg123_read(mh,input.data(),
                                     input.size(),&received);
        const size_t frames=std::min(
            received/frameBytes,(size_t)std::min(
                maximum-nativeFrames,(uint64_t)(received/frameBytes)));
        if(frames>0){
            if(SDL_AudioStreamPut(stream,input.data(),
                (int)(frames*frameBytes))<0){
                error=std::string("MP3 resampler write: ")+SDL_GetError();
                valid=false;break;
            }
            nativeFrames+=frames;
            if(!drain()){valid=false;break;}
        }
        if(status==MPG123_DONE)break;
        if(status!=MPG123_OK){
            error=std::string("MP3 decoding failed: ")+
                  mpg123_strerror(mh);
            valid=false;break;
        }
        if(received==0)break;
    }
    if(valid&&SDL_AudioStreamFlush(stream)<0){
        error=std::string("MP3 resampler flush: ")+SDL_GetError();
        valid=false;
    }
    if(valid)valid=drain();
    SDL_FreeAudioStream(stream);
    closeDecoder();
    if(!valid)return false;
    if(pcm.size()<128){
        error="MP3 decoded to fewer than 128 audio frames";
        pcm.clear();return false;
    }
    return true;
}
} // namespace asmr
