#pragma once

// Shared, dependency-free numerical primitives. This header is also compiled by
// the CPU-only tests, so changes to CUDA math have a reproducible reference.
#include <math.h>
#include <stdint.h>
#include <vector>

#ifdef __CUDACC__
#define ECHO_HD __host__ __device__
#else
#define ECHO_HD
#endif

namespace echo {
constexpr float kSpeedOfSound = 343.0f; // metres per second
constexpr float kCellMetres = 0.01f;
constexpr float kSqrtTwo = 1.4142135623730951f;
constexpr float kStepSeconds = kCellMetres / (kSpeedOfSound * kSqrtTwo);
constexpr float kCourant = kSpeedOfSound * kStepSeconds / kCellMetres;
constexpr float kCourantSquared = kCourant * kCourant;
constexpr float kClickFrequencyHz = 1800.0f;
constexpr float kClickDurationSeconds = 0.002f;
constexpr float kPi = 3.14159265358979323846f;

// Local, centered-time impedance boundary approximation. A rigid wall
// (R=1) contributes zero damping; softer walls contribute a centered-time
// velocity term proportional to (1-R)/(1+R) for each solid neighbor.
//
// A *backward*-time ghost pressure can be unstable at R=0 on this CFL grid.
// Instead, the damping is evaluated at (p_next-p_prev)/(2dt), making the
// update semi-implicit in velocity while retaining an explicit per-cell solve.
// The ghost's spatial pressure is always the fluid cell (Neumann geometry).
ECHO_HD inline float wallDamping(float reflection, float courant,
                                 int adjacentWalls) {
    const float r = fmaxf(0.0f, fminf(1.0f, reflection));
    const float impedance = (1.0f - r) / (1.0f + r);
    return 0.5f * courant * impedance * adjacentWalls;
}

ECHO_HD inline float waveStep(float p, float prev, float left, float right,
                              float down, float up, float courantSq) {
    return 2.0f * p - prev +
           courantSq * (left + right + down + up - 4.0f * p);
}

ECHO_HD inline float impedanceWaveStep(float p, float prev, float left,
                                       float right, float down, float up,
                                       float courantSq, float damping) {
    return (2.0f*p - (1.0f-damping)*prev +
            courantSq*(left+right+down+up-4.0f*p)) / (1.0f+damping);
}

// Gameplay can use automatic pings, manual pings, or both. A disabled
// automatic clock must *not* stop simulation time or change the sound speed.
ECHO_HD inline int pulseIndex(uint64_t step, int interval, int duration,
                              bool automatic, uint64_t manualStart) {
    if (manualStart != UINT64_MAX && step >= manualStart &&
        step - manualStart < (uint64_t)duration)
        return (int)(step - manualStart);
    if (automatic && step >= 1000ULL && interval > 0) {
        uint64_t since=step-1000ULL;
        int within=(int)(since % (uint64_t)interval);
        if(within<duration)return within;
    }
    return -1;
}

// Hann-tapered burst: zero at both ends, avoiding the old discontinuous
// quarter-cycle pulse. The chosen carrier is resolved by ~19 cells/wavelength.
ECHO_HD inline float clickSample(int index, int duration, float dt) {
    if (index < 0 || duration < 2 || index >= duration) return 0.0f;
    const float phase = 2.0f * kPi * index / (duration - 1);
    const float envelope = 0.5f - 0.5f * cosf(phase);
    return envelope * sinf(2.0f * kPi * kClickFrequencyHz * index * dt);
}

// Logarithmic DISPLAY-only pressure encoding. Negative values below -1
// are reserved for SDL/OpenGL glyph metadata; never let a strong wave
// become an invisible sentinel or a ghost marker. Wave physics unchanged.
ECHO_HD inline float displayPressure(float pressure,float exposure) {
    const float magnitude=fabsf(pressure)*fmaxf(0.000001f,exposure);
    const float mapped=log1pf(5.0f*magnitude)/logf(51.0f);
    // Wave values must remain within (-0.5,+0.5): -1 through -10
    // are glyph IDs, and the GPU marker shader tests pressure<-0.5.
    const float limit=0.45f*fminf(1.0f,mapped);
    return copysignf(limit,pressure);
}

// The visible sound funnel is an overlay, not a moving solid obstacle.
// Rotating/translating a binary funnel mask each frame abruptly changes
// boundary conditions and creates spurious rattling in a wave solver.
ECHO_HD inline bool visualFunnelPixel(float x,float y,
    float wedgeX,float wedgeY,float pulseX,float pulseY,
    float playerX,float playerY) {
    const float heading=atan2f(pulseY-playerY,pulseX-playerX);
    const float dx=x-wedgeX,dy=y-wedgeY;
    const float aperture=38.0f*kPi/180.0f;
    for(int branch=-1;branch<=1;branch+=2) {
        const float angle=heading+branch*aperture;
        const float forward=dx*cosf(angle)+dy*sinf(angle);
        const float off=dx*sinf(angle)-dy*cosf(angle);
        if(forward>=0.0f&&forward<=10.0f&&fabsf(off)<=0.65f)return true;
    }
    return false;
}

// Bilinear receiver sampling prevents 1-cell jumps in virtual microphone
// position from introducing artificial broadband clicks.
ECHO_HD inline float bilinearSample(const float* p, int size, float x, float y) {
    x = fmaxf(0.0f, fminf((float)(size - 1), x));
    y = fmaxf(0.0f, fminf((float)(size - 1), y));
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = x0 + 1 < size ? x0 + 1 : x0;
    int y1 = y0 + 1 < size ? y0 + 1 : y0;
    float fx = x - x0, fy = y - y0;
    float a = p[y0 * size + x0] * (1.0f - fx) + p[y0 * size + x1] * fx;
    float b = p[y1 * size + x0] * (1.0f - fx) + p[y1 * size + x1] * fx;
    return a * (1.0f - fy) + b * fy;
}

// The simulation advances at ~48,508 steps/s, NOT exactly 48 kHz. Resample
// with a persistent phase across GPU batches so physics time and sound playback
// cannot silently drift apart. One output is an interleaved stereo frame.
class StereoLinearResampler {
public:
    explicit StereoLinearResampler(double inputStepSeconds, int outputRate)
        : inputStep_(inputStepSeconds), outputRate_(outputRate),
          inputIndex_(0), outputIndex_(0), previousL_(0), previousR_(0) {}

    void push(float left, float right, std::vector<float>& output) {
        if (inputIndex_ == 0) {
            output.push_back(left);
            output.push_back(right);
            previousL_ = left;
            previousR_ = right;
            ++inputIndex_;
            ++outputIndex_;
            return;
        }
        const double t0 = (inputIndex_ - 1) * inputStep_;
        const double t1 = inputIndex_ * inputStep_;
        while ((double)outputIndex_ / outputRate_ <= t1 + 1e-12) {
            const double t = (double)outputIndex_ / outputRate_;
            float alpha = (float)((t - t0) / inputStep_);
            alpha = fmaxf(0.0f, fminf(1.0f, alpha));
            output.push_back(previousL_ + alpha * (left - previousL_));
            output.push_back(previousR_ + alpha * (right - previousR_));
            ++outputIndex_;
        }
        previousL_ = left;
        previousR_ = right;
        ++inputIndex_;
    }

    uint64_t outputFrames() const { return outputIndex_; }
    uint64_t inputFrames() const { return inputIndex_; }

private:
    double inputStep_;
    double outputRate_;
    uint64_t inputIndex_, outputIndex_;
    float previousL_, previousR_;
};
} // namespace echo
#undef ECHO_HD
