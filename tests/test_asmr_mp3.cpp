#include "../asmr_mp3_audio.h"
#include "../asmr_lab_math.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Run with CI-generated 75-second clean and locally corrupted MP3.
// The corruption deliberately damages MPEG side info near 12kB,
// reproducing the user's invalid-header/resynchronization warning.
// Crucially, recovery must not mistake a damaged frame for EOF.
int main(int argc,char** argv){
    if(argc<3){
        std::puts("MP3 integration: skipped (supply clean and "
                  "corrupted fixture paths; CI runs both)");
        return 0;
    }
    for(int i=1;i<=2;++i){
        std::vector<float> pcm;
        std::string error;
        if(!asmr::decodeMP3(argv[i],pcm,error)){
            std::fprintf(stderr,
                "MP3 decode fixture %d FAILED: %s\n",
                i,error.c_str());
            return 1;
        }
        const double seconds=pcm.size()/48000.0;
        if(!(seconds>72.0&&seconds<76.0)){
            std::fprintf(stderr,
                "MP3 fixture %d truncated: %.2f of ~75 seconds\n",
                i,seconds);
            return 2;
        }
        // Source window beyond 60s accesses the REAL continuing
        // song; the MP3 and IIR preprocessing are not truncated.
        assert(asmr::mp3WindowStart(64.0,pcm.size())==
               asmr::kMp3WindowStride);
        auto window=std::vector<float>(pcm.begin(),
            pcm.begin()+std::min(pcm.size(),(size_t)48000*2));
        auto filtered=asmr::lowpassSourceWindow(
            std::move(window),5200.0f,48000);
        for(float sample:filtered)assert(std::isfinite(sample));
        std::fprintf(stderr,
            "[MP3 TEST] fixture %d decoded %.2fs across 60s "
            "boundary; first cache filter passed.\n",i,seconds);
    }
    std::puts("MP3 integration: full-length decode and recovery passed");
}
