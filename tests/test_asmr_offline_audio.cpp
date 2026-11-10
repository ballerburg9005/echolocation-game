#include "../asmr_offline_audio.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static void testSpikeCannotMuteSong(){
    constexpr int fs=48000,seconds=10;
    std::vector<float> audio((size_t)fs*seconds*2);
    for(int n=0;n<fs*seconds;++n){
        const float t=(float)n/fs;
        const float v=0.001f*std::sin(2*3.14159265358979323846f*440*t);
        audio[2*n]=v;audio[2*n+1]=v;
    }
    audio[2*fs]=1e9f; // one numerical spike after initial fade
    const auto result=asmr::normalizeOfflineAudio(audio);
    assert(result.before.peak>1e8);
    assert(result.before.rms>1000);
    assert(result.after.nonFinite==0);
    assert(result.after.peak<=0.85+1e-6);
    assert(result.gain>=1);
    // The rest of the test signal must not disappear when one sample
    // dominates the maximum. Measured far from the transient.
    const size_t at=(size_t)fs*5*2;
    double energy=0;
    for(size_t i=at;i<at+fs*2;++i)
        energy+=(double)audio[i]*audio[i];
    const double rms=std::sqrt(energy/(fs*2));
    assert(rms>0.01);
    assert(result.clipped>=1);
}
static void testRealSilenceRemainsSilent(){
    std::vector<float> audio(96000,0.0f);
    const auto result=asmr::normalizeOfflineAudio(audio);
    assert(result.gain==0);
    assert(result.after.peak==0.0);
    assert(result.after.rms==0.0);
    assert(result.after.activeFraction==0);
}
static void testNaNSafe(){
    std::vector<float> audio(96000,0.002f);
    audio[40000]=std::numeric_limits<float>::quiet_NaN();
    audio[40001]=std::numeric_limits<float>::infinity();
    const auto result=asmr::normalizeOfflineAudio(audio);
    assert(result.before.nonFinite==2);
    assert(result.after.nonFinite==0);
    assert(audio[40000]==0.0f&&audio[40001]==0.0f);
    assert(result.after.peak<=0.85);
}
int main(){
    testSpikeCannotMuteSong();
    testRealSilenceRemainsSilent();
    testNaNSafe();
    std::puts("Offline ASMR: robust normalization, spikes, silence, NaN: PASS");
}
