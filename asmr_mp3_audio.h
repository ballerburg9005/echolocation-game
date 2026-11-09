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
inline bool decodeMP3(const std::string& filename,
                      std::vector<float>& pcm,
                      std::string& error,
                      int outputRate=48000,int secondsLimit=60) {
    pcm.clear();
    error.clear();
    if(mpg123_init()!=MPG123_OK) {
        error="libmpg123 initialization failed"; return false;
    }
    int code=MPG123_OK;
    mpg123_handle* mh=mpg123_new(nullptr,&code);
    if(!mh){
        error="libmpg123 decoder allocation failed";
        mpg123_exit();return false;
    }
    // Preserve native MP3 sample rate, decode standard signed-16bit
    // stereo with libmpg123; SDL then handles 44.1->48kHz conversion
    // and mono downmix, avoiding any external ffmpeg requirement.
    if(mpg123_open_fixed(mh,filename.c_str(),MPG123_STEREO,
                         MPG123_ENC_SIGNED_16)!=MPG123_OK){
        error=std::string("Cannot open MP3: ")+mpg123_strerror(mh);
        mpg123_delete(mh);mpg123_exit();return false;
    }
    long nativeRate=0;
    int channels=0,encoding=0;
    if(mpg123_getformat(mh,&nativeRate,&channels,&encoding)!=MPG123_OK ||
       nativeRate<8000||nativeRate>192000||channels!=2||
       encoding!=MPG123_ENC_SIGNED_16){
        error="Unsupported decoded MP3 rate or sample format";
        mpg123_delete(mh);mpg123_exit();return false;
    }
    const size_t frameBytes=channels*sizeof(int16_t);
    const size_t maximum=(size_t)nativeRate*
        (size_t)secondsLimit*frameBytes;
    std::vector<unsigned char> bytes;
    bytes.reserve(std::min(maximum,(size_t)8*1024*1024));
    std::array<unsigned char,16384> buffer{};
    bool valid=true;
    while(bytes.size()<maximum) {
        size_t received=0;
        const size_t capacity=std::min(buffer.size(),
                                      maximum-bytes.size());
        const int status=mpg123_read(mh,buffer.data(),
                                     capacity,&received);
        if(received)
            bytes.insert(bytes.end(),buffer.begin(),
                         buffer.begin()+received);
        if(status==MPG123_DONE)break;
        if(status!=MPG123_OK){
            error=std::string("MP3 decoding failed: ")+
                  mpg123_strerror(mh);
            valid=false;break;
        }
        if(received==0)break;
    }
    mpg123_delete(mh);
    mpg123_exit();
    bytes.resize(bytes.size()/frameBytes*frameBytes);
    if(!valid||bytes.size()<frameBytes*128)return false;
    SDL_AudioCVT cvt{};
    const int conversion=SDL_BuildAudioCVT(&cvt,
        AUDIO_S16SYS,channels,(int)nativeRate,
        AUDIO_F32SYS,1,outputRate);
    if(conversion<0){
        error=std::string("SDL sample conversion failed: ")+
              SDL_GetError();
        return false;
    }
    cvt.len=(int)bytes.size();
    cvt.buf=(Uint8*)SDL_malloc((size_t)cvt.len*
                              (size_t)std::max(1,cvt.len_mult));
    if(!cvt.buf){
        error="Insufficient memory to resample MP3";return false;
    }
    std::memcpy(cvt.buf,bytes.data(),bytes.size());
    if(cvt.needed&&SDL_ConvertAudio(&cvt)!=0){
        error=std::string("SDL MP3 resampling failed: ")+
              SDL_GetError();
        SDL_free(cvt.buf);return false;
    }
    if(!cvt.needed)cvt.len_cvt=cvt.len;
    const size_t sampleCount=std::min(
        (size_t)cvt.len_cvt/sizeof(float),
        (size_t)outputRate*(size_t)secondsLimit);
    const float* samples=(const float*)cvt.buf;
    pcm.resize(sampleCount);
    for(size_t i=0;i<sampleCount;++i)
        pcm[i]=std::isfinite(samples[i])?
            std::clamp(samples[i],-1.0f,1.0f):0.0f;
    SDL_free(cvt.buf);
    return pcm.size()>=128;
}
} // namespace asmr
