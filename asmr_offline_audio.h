#pragma once
// CPU-reference diagnostics and safe normalization for offline ASMR audio.
// Do not let a single numerical transient mute an entire WAV recording.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace asmr {
struct OfflineAudioStats {
    double peak=0.0;
    double rms=0.0;
    double activeFraction=0.0; // samples exceeding 1e-5 absolute
    uint64_t nonFinite=0;
    uint64_t samples=0;
};
inline OfflineAudioStats offlineAudioStats(const std::vector<float>& audio) {
    OfflineAudioStats s;
    s.samples=audio.size();
    if(audio.empty())return s;
    long double squared=0.0;
    uint64_t active=0;
    for(const float v:audio) {
        if(!std::isfinite(v)){++s.nonFinite;continue;}
        const double amplitude=std::fabs((double)v);
        s.peak=std::max(s.peak,amplitude);
        squared+=(long double)v*v;
        if(amplitude>1e-5)++active;
    }
    s.rms=std::sqrt((double)(squared/audio.size()));
    s.activeFraction=(double)active/audio.size();
    return s;
}
struct OfflineAudioNormalization {
    OfflineAudioStats before;
    OfflineAudioStats after;
    double referencePeak=0.0;
    double gain=0.0;
    uint64_t clipped=0;
};
// Some numerically pathological recordings contain one giant startup spike.
// Previous global-maximum normalization scaled the entire recording to
// near-zero, with only a little click at the beginning. Use a robust
// high-percentile pressure amplitude instead of that isolated maximum.
// Retain ONE linear gain for the entire recording, with rare outliers
// safety-clipped, not broadband dynamic compression or artificial EQ.
inline OfflineAudioNormalization normalizeOfflineAudio(
        std::vector<float>& audio,int sampleRate=48000) {
    OfflineAudioNormalization result;
    result.before=offlineAudioStats(audio);
    if(audio.empty()||sampleRate<=0)return result;
    std::vector<float> magnitudes;
    magnitudes.reserve(audio.size()/16+1);
    for(size_t i=0;i<audio.size();i+=16)
        if(std::isfinite(audio[i]))
            magnitudes.push_back(std::fabs(audio[i]));
    if(!magnitudes.empty()) {
        const size_t rank=(magnitudes.size()-1)*997/1000;
        std::nth_element(magnitudes.begin(),
                         magnitudes.begin()+rank,magnitudes.end());
        result.referencePeak=magnitudes[rank];
    }
    // Use the high-percentile amplitude when a genuine signal
    // occupies the timeline. Raw RMS can be dominated by the single
    // enormous spike we are explicitly trying to reject.
    // For very sparse signals fall back conservatively; the caller
    // diagnoses almost-empty audio rather than amplifying numerical dust.
    const double reference=result.referencePeak>1e-9?
        result.referencePeak:result.before.rms*1.5;
    // Keep actual silence silent; never amplify floating-point dust.
    if(reference>1e-10 && std::isfinite(reference))
        result.gain=std::min(64.0,0.38/reference);
    const size_t frames=audio.size()/2;
    for(size_t i=0;i<frames;++i) {
        const double fadeIn=std::min(1.0,(double)i/(sampleRate*0.035));
        const double fadeOut=std::min(1.0,
                        (double)(frames-i)/(sampleRate*0.060));
        for(int ear=0;ear<2;++ear) {
            float& v=audio[2*i+ear];
            const double value=std::isfinite(v)?
                (double)v*result.gain*std::min(fadeIn,fadeOut):0.0;
            if(value>0.85||value< -0.85)++result.clipped;
            v=(float)std::clamp(value,-0.85,0.85);
        }
    }
    result.after=offlineAudioStats(audio);
    return result;
}
} // namespace asmr
