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

    // Hard walls enforce a zero-gradient ghost; lower R causes actual loss.
    assert(near(echo::wallGhost(2.0f, 1.0f, 1.0f, echo::kCourant), 2.0f));
    assert(echo::wallGhost(2.0f, 1.0f, 0.0f, echo::kCourant) < 2.0f);
    assert(near(echo::wallGhost(2.0f, 2.0f, 0.5f, echo::kCourant), 2.0f));

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
    std::printf("acoustics tests passed: Courant, rigid/absorbing walls, bilinear microphones, click and 48 kHz timing (max tone error %g)\n", maxError);
}
