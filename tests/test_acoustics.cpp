#include "../acoustics.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static bool near(float a, float b, float tolerance = 0.00001f) {
    return std::fabs(a - b) <= tolerance;
}

int main() {
    // Two-dimensional leapfrog CFL must be at or below 1/sqrt(2).
    assert(echo::kCourant > 0.0f);
    assert(echo::kCourant <= 1.0f / echo::kSqrtTwo + 0.000001f);
    assert(near(echo::kCourantSquared, 0.5f, 0.000001f));

    // A uniform pressure field is stationary without an initial velocity.
    assert(near(echo::waveStep(4, 4, 4, 4, 4, 4,
                               echo::kCourantSquared), 4));

    // Centered-time impedance damping: hard walls have none; R=0 is
    // maximally absorptive but must remain numerically stable.
    assert(near(echo::wallDamping(1.0f, echo::kCourant, 2), 0.0f));
    assert(near(echo::wallDamping(0.0f, echo::kCourant, 2), echo::kCourant));
    assert(echo::wallDamping(0.5f, echo::kCourant, 1) >
           echo::wallDamping(0.9f, echo::kCourant, 1));
    assert(near(echo::impedanceWaveStep(4, 4, 4, 4, 4, 4,
                  echo::kCourantSquared, 0.25f), 4));

    float field[4] = {0, 1, 2, 3};
    assert(near(echo::bilinearSample(field, 2, 0.5f, 0.5f), 1.5f));
    assert(near(echo::bilinearSample(field, 2, -1.0f, 0.0f), 0.0f));
    assert(near(echo::bilinearSample(field, 2, 2.0f, 2.0f), 3.0f));

    int clickSteps = (int)std::round(echo::kClickDurationSeconds / echo::kStepSeconds);
    assert(clickSteps > 10);
    assert(near(echo::clickSample(0, clickSteps, echo::kStepSeconds), 0.0f));
    assert(near(echo::clickSample(clickSteps - 1, clickSteps, echo::kStepSeconds), 0.0f, 0.000001f));
    assert(near(echo::clickSample(clickSteps, clickSteps, echo::kStepSeconds), 0.0f));
    float peak = 0.0f;
    for (int i = 0; i < clickSteps; ++i) {
        peak = std::fmax(peak, std::fabs(echo::clickSample(i, clickSteps, echo::kStepSeconds)));
    }
    assert(peak > 0.6f && peak <= 1.0f);

    // Manual and automatic pulses use identical physical steps with no
    // simulator clock distortion; manual takes precedence when simultaneous.
    assert(echo::pulseIndex(999,500,100,true,UINT64_MAX)==-1);
    assert(echo::pulseIndex(1000,500,100,true,UINT64_MAX)==0);
    assert(echo::pulseIndex(1100,500,100,true,UINT64_MAX)==-1);
    assert(echo::pulseIndex(1000,500,100,false,UINT64_MAX)==-1);
    assert(echo::pulseIndex(1000,500,100,false,1000)==0);
    assert(echo::pulseIndex(1008,500,100,false,1000)==8);
    assert(echo::pulseIndex(1100,500,100,false,1000)==-1);
    assert(echo::pulseIndex(1001,500,100,true,1000)==1);

    // Smooth wide-range pressure visualization is bounded away from the
    // sentinel codes -2..-10 even for large negative pressure spikes.
    const float low=echo::displayPressure(10.0f,0.018f);
    const float high=echo::displayPressure(100.0f,0.018f);
    assert(low>0 && high>low && high<0.99f);
    assert(near(echo::displayPressure(-10,0.018f),-low));
    assert(echo::displayPressure(-1000000,0.018f)>-0.5f);
    assert(echo::displayPressure(1000000,0.018f)<0.5f);
    assert(echo::displayPressure(-50000,0.018f)>-0.5f);
    // Funnel is only a visual outline, not a moving binary acoustic wall:
    // exact ray points render as edges; the back/side of the cone do not.
    assert(echo::visualFunnelPixel(5,5,0,0,10,0,0,0)==false);
    const float halfAngle=16.0f*echo::kPi/180.0f;
    assert(echo::visualFunnelPixel(5.0f*std::cos(halfAngle),
        5.0f*std::sin(halfAngle),0,0,10,0,0,0));
    // A previous 38-degree half-angle was visibly far too open.
    const float oldHalfAngle=38.0f*echo::kPi/180.0f;
    assert(!echo::visualFunnelPixel(5.0f*std::cos(oldHalfAngle),
        5.0f*std::sin(oldHalfAngle),0,0,10,0,0,0));
    assert(!echo::visualFunnelPixel(-5,0,0,0,10,0,0,0));
    // The forward cyan wedge is VISUAL ONLY and fully inside the real
    // 13cm collision disk, so it can neither hit nor visually clip
    // into an SVG wall while the player remains legally placed.
    int wedgePixels=0;
    for(int y=-14;y<=14;++y)for(int x=-14;x<=14;++x) {
        if(!echo::visualFunnelPixel((float)x,(float)y,
                                    -4.5f,0,6.0f,0,0,0))
            continue;
        ++wedgePixels;
        assert(std::hypot((float)x,(float)y)<13.0f);
    }
    assert(wedgePixels>8);

    // One second of 48.5 kHz FDTD must become 48 kHz stereo without drifting.
    echo::StereoLinearResampler resampler(echo::kStepSeconds, 48000);
    std::vector<float> rendered;
    int inputFrames = (int)std::floor(1.0 / echo::kStepSeconds) + 1;
    for (int i = 0; i < inputFrames; ++i) {
        double time = i * (double)echo::kStepSeconds;
        float signal = (float)std::sin(2.0 * echo::kPi * 1000.0 * time);
        resampler.push(signal, -signal, rendered);
    }
    assert(rendered.size() % 2 == 0);
    assert(std::abs((int)resampler.outputFrames() - 48000) <= 1);
    assert(resampler.inputFrames() == (uint64_t)inputFrames);
    float maxError = 0.0f;
    for (size_t i = 0; i < rendered.size() / 2; ++i) {
        float expected = (float)std::sin(2.0 * echo::kPi * 1000.0 * i / 48000.0);
        maxError = std::fmax(maxError, std::fabs(rendered[2*i] - expected));
        assert(near(rendered[2*i], -rendered[2*i+1], 0.000001f));
    }
    assert(maxError < 0.004f);
    // Evolve a closed 2D room to catch boundary instabilities that algebraic
    // one-step checks miss. Explicit backward-time ghost loss diverges at R=0.
    constexpr int N = 60, STEPS = 600;
    float finalEnergies[4] = {};
    const float reflectionValues[4] = {0.0f, 0.5f, 0.9f, 1.0f};
    for (int m = 0; m < 4; ++m) {
        const float reflection = reflectionValues[m];
        std::vector<float> current(N*N, 0.0f), previous(N*N, 0.0f);
        std::vector<float> next(N*N, 0.0f);
        for (int y = 1; y < N-1; ++y) {
            for (int x = 1; x < N-1; ++x) {
                const float r2 = (float)((x-N/2)*(x-N/2) + (y-N/2)*(y-N/2));
                current[y*N+x] = std::exp(-r2/8.0f);
            }
        }
        previous = current; // zero initial time derivative
        for (int t = 0; t < STEPS; ++t) {
            for (int y = 1; y < N-1; ++y) {
                for (int x = 1; x < N-1; ++x) {
                    const int i = y*N+x;
                    const bool wl = x==1, wr = x==N-2;
                    const bool wd = y==1, wu = y==N-2;
                    const float center = current[i];
                    const float damping = echo::wallDamping(reflection,
                                       echo::kCourant, (int)wl+(int)wr+(int)wd+(int)wu);
                    next[i] = echo::impedanceWaveStep(
                        center, previous[i],
                        wl ? center : current[i-1], wr ? center : current[i+1],
                        wd ? center : current[i-N], wu ? center : current[i+N],
                        echo::kCourantSquared, damping);
                    assert(std::isfinite(next[i]));
                    assert(std::fabs(next[i]) < 10.0f);
                }
            }
            previous.swap(current);
            current.swap(next);
        }
        float energy = 0.0f;
        for (float p : current) energy += p*p;
        finalEnergies[m] = energy;
    }
    assert(finalEnergies[0] < finalEnergies[3] * 0.2f);
    assert(finalEnergies[1] < finalEnergies[3] * 0.5f);
    std::printf("acoustics tests passed: Courant, stable impedance walls R=0..1, bilinear microphones, click and 48 kHz timing (max tone error %g)\n", maxError);
}
