# Recorded WAV audio inside the 2D FDTD room: feasibility and plan

**Status: design and numerical reasoning; NOT yet a wired audio-file player.**
The current application injects player clicks and short procedural object
chirps directly into CUDA FDTD. Arbitrary samples can replace that source
waveform without changing the wave equation. For another project, the
same physical listener and walls can therefore spatialize a recording.

## What the solver can reproduce

- **Floor plan:** current 500x500 grid at 1 cm spacing = 5x5 metres.
  A small 2x2m *planar* room could use a 200x200 grid at the same spacing,
  with about **6.25x fewer cell updates per simulated second**.
  A toilet-sized room is feasible in floor area; it is NOT a full
  bathroom-acoustics simulator because the ceiling, floor, height,
  furnishings, pinnae and 3D reflections are omitted.
- **Time resolution:** the present Courant timestep is approximately
  20.6 microseconds (about 48.5k FDTD steps/s). Audio output is
  resampled to the SDL device rate. Sampled WAV input should be
  similarly resampled onto the **FDTD timestep**, not simply indexed
  by 44.1kHz or 48kHz file frames.
- **Spatial resolution, not sample-rate Nyquist, limits fidelity:**
  1cm spatial cells and 343m/s sound speed provide ~10 grid cells
  per wavelength at 3.43kHz. Conservative demos should
  **low-pass to 2.5–3kHz before injection**; higher frequencies suffer
  numerical dispersion and direction-dependent artifacts.
  Speech/environmental sounds may remain identifiable, but music's
  upper frequencies and room-realistic acoustics are not preserved.
- **Positioning:** put a mono source at any allowed fluid position.
  The two existing continuously sampled virtual ears then hear the
  actual wave pressure including walls, echoes and diffraction.
  No ordinary game-engine stereo pan or separate left/right audio
  playback is necessary. Moving the source past the listener in the
  2D plane can change left/right pressure naturally, although true
  front/back elevation and individualized head localization need HRTFs
  or a 3D head/ear model.

## Efficient opt-in WAV demonstration (proposed, not implemented)

1. Load an explicitly supplied short WAV using SDL2's `SDL_LoadWAV`.
   Decode channels to a **single mono source** and convert to float32.
   A stereo recording is not two independent virtual ears; for a
   single spatial emitter, downmix or choose one channel deliberately.
2. Resample to the acoustic solver's fixed ~48.5kHz step frequency.
   Offline low-pass near 2.5–3kHz. Normalize conservatively, **fade
   in and out**, and cap input duration/sample memory.
3. Allocate **one GPU PCM buffer** and copy it once, not at each
   simulation step. Pass sample pointer, sample count, start step and
   a bilinear (2x2 cell) source position to the *existing*
   `updatePressure` CUDA kernel. At source cells only, add a
   safely scaled `pcm[step-start]` to `p_next`, subject to the
   same fluid/solid checks as the procedural click.
4. Use a continuously moving source coordinate in world metres or
   centimetres to demonstrate motion from **left of the head to
   right of the head**. Interpolate its location/weights over time:
   snapping to a different grid cell every video frame can generate
   false clicks. Never inject from inside an SVG hard wall or move
   the player's hard wedge mask.
5. Keep current two virtual microphone samples and the existing
   resampler/limiter; the room itself converts source audio to
   stereo. `J` direct-field subtraction should remain scoped to
   the player's clicks, not accidentally subtract the file.

**Performance expectation:** Reading one preloaded PCM value for a
single active emitter per step costs little *compared with updating
the entire grid*. With 500x500x48,500, the solver advances roughly
**12.1 billion cell-steps/s** before any source audio is added.
Two current CUDA launches per step correspond to roughly 97,000
kernel launches/s. Launch overhead, memory bandwidth, wall logic,
audio scheduling, and CPU/GPU synchronization could bottleneck long
before peak FLOPS. A short optional WAV source should **not** create
new whole-grid kernels; support many sources with a sparse GPU
source list/map and carefully benchmark its cost instead of a
loop over all files for each wave cell. CUDA Graphs or a more
substantially batched stencil are possible optimizations, but would
require separate correctness/performance validation.

The RTX 3090 drawing ~210W against a nominal 350W power limit
is **not** a reliable measurement of remaining simulation
throughput. Power draw does not measure kernel-launch bottlenecks,
driver scheduling, frame pacing or audio queue starvation.
Profile with GPU timings, Nsight Systems, buffer underrun counters,
and headphones on the target GPU and at least one weaker GPU.
`[AUDIO-DIAG]` in the current build can help distinguish high
pressure transients from empty SDL output queues.

**Suggested incremental experiment:** start with a 2x2m empty room,
one 2-second bandlimited mono voice or object recording, one stationary
source, and a head-local left-to-right sweep. Compare dry WAV,
FDTD microphones and the limiting/underrun diagnostics. Only then
try cluttered walls, sources in motion, multiple simultaneous
files, and more faithful 3D room simulations.

## References

- [SDL2 WAV loading](https://wiki.libsdl.org/SDL2/SDL_LoadWAV)
- [CUDA launch overhead and profiling](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)
- [NVIDIA CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- [2D FDTD numerical dispersion and spatial-resolution discussion](https://docs.jmrp.io/phonometry/simulation/fdtd-simulation/)
