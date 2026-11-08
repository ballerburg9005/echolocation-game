# Numerical model, assumptions, and validation

The project is an **experimental 2D scalar-pressure acoustics game**. It is not
a validated simulator of human echolocation or of 3D binaural hearing.

## Physics and time

- Grid: 500 × 500 cells, 0.01 m per cell: a 5 m square cross-section.
- Wave speed: 343 m/s. Second-order centered-space/centered-time update
  \(p^{n+1}=2p^n-p^{n-1}+(c\,dt/dx)^2\nabla_h^2 p^n\).
- Stable 2D Courant factor \(c\,dt/dx=1/\sqrt2\), so one simulation
  step is about 20.615 µs, i.e. about 48.508 kHz, **not 48 kHz**.
- GPU samples are interpolated onto exactly the negotiated 48-kHz (or other
  SDL-device-rate) output timeline with persistent resampler phase.
- The default emitted pulse is a 1.8 kHz, 2 ms Hann-tapered burst rather than
  a discontinuous 500 Hz quarter-cycle.
- Rigid walls use the zero-normal-gradient pressure boundary condition
  \(p_{ghost}=p_{fluid}\). For pressure reflection \(R<1\), a
  **centered-time, semi-implicit** local impedance approximation damps
  normal velocity at each neighboring wall. With \(n_w\) adjacent solid
  cells, \(q=n_w\,\mathrm{CFL}(1-R)/(2(1+R))\), and a rigid-wall
  Laplacian, the update is
  \(p^{n+1}=[2p^n-(1-q)p^{n-1}+\mathrm{CFL}^2\nabla_h^2p^n]/(1+q)\).
  This is a locally reacting approximation, not a measured wall model.
  An earlier backward-time ghost formula was discarded because it became
  unstable near \(R=0\). The full interval \(R=0..1\) is covered by a
  600-step CPU regression test.
- User-selectable per-step decay is artificial amplitude damping, not a
  physical frequency-dependent attenuation law. Walls are still represented
  by square grid cells, so corners are stair-stepped.

## Critical limitations

1. Pressure waves propagate in **2D**, where geometric spreading and room
   modes differ from the physical 3D world.
2. No HRTF, pinna, head shadowing, elevation, or ear-canal modeling is present.
   The "ears" sample pressure at two points; this is *not* physically correct
   binaural rendering or a validated echolocation stimulus.
3. Sources/walls are discretized to cells. Numerical dispersion varies with
   frequency and direction; do not extrapolate to frequencies near the grid
   Nyquist limit.
4. GPU launch cost may prevent real-time output on some hardware. The audio
   queue is paced by the real sample clock, and the game no longer modifies
   physical simulation time to hide underperformance.
5. SVG support is intentionally constrained; arbitrary paths/transforms and
   nontrivial SVG viewBox transformations are not rasterized.
6. Headphones are recommended for stereo localization, beginning at a **low
   listening volume**. Synthetic cues are not safe for real-world navigation.

## What is tested

The CPU-only tests check the Courant condition, unchanged uniform pressure,
long-run stability for reflection R=0..1, interpolated receiver placement,
tapered pulse endpoints, and 48-kHz resampling without drift in a 1-second
synthetic 1-kHz tone. Tests can run without a CUDA card:

    g++ -std=c++17 -O2 -Wall -Wextra -pedantic tests/test_acoustics.cpp -o /tmp/echo-test
    /tmp/echo-test

## Still needed for scientific credibility

- Compare simulated arrival times and impulse responses with analytic free
  space and reflection cases, and with reference wave solver outputs.
- Measure energy, stability, boundary reflections and dispersion as a function
  of grid spacing; test multiple obstacle shapes.
- Validate against measured room impulse responses and physical dummy-head
  recordings. Add calibrated HRTFs, scattering and frequency-dependent wall
  impedances in a future 3D solver if binaural fidelity is the goal.
- Conduct accessibility and usability trials with blind echolocators and
  non-echolocators; never infer effectiveness from the developer alone.

Changes here aim at **internal numerical correctness and usability**.
They do not establish perceptual fidelity.


## Gameplay and 3D addendum

The new first-person mode is a perspective rendering of walls extruded from
the 2D SVG raster mask; the numerical acoustic solver stays 2D. Enemies and
shards are stamped into the *dynamic* wave mask as special reflective cells.
Enemy reflection (0.82) and shard reflection (0.62) are approximate local
material boundary coefficients. Neither is a calibrated measurement.

Enemy sound attraction, stereo threat heartbeat, item/exit guidance beacons,
pickups and win jingles are game logic/audio cues. They are **not** modeled
as pressure-field sources or 3D binaural recordings. The manual-click F and
automatic-click T controls use the same physical timestep and are covered
by pulse scheduler tests. A dedicated 3D mesh test checks that each wall
surface originates from the actual mask and that solid adjacencies do not
generate interior faces.

Moving-mask artifacts, acoustic energy around moving objects, HRTF effects,
and real-time performance on NVIDIA hardware remain unmeasured. Do not
interpret success in the game as evidence of safe or effective physical
echolocation.


## Player-reported regressions: corrective changes and remaining uncertainty

Feedback from real hardware testing identified movement-related rattles, a
mirrored 3D heading, geometry lines, nonlocal wall-contact cues and persistent
synthetic marker beeps. The changes below are **hypotheses and numerical/UI
fixes awaiting renewed listening evaluation**, not proven cures:

1. **Moving pressure boundaries:** the prior implementation inserted the
   player's moving visual funnel into a hard 0/1 wall mask every frame.
   The wave solver set those newly blocked pressure cells to zero abruptly,
   generating nonphysical boundary discontinuities as the player moved.
   The funnel is now a visual overlay; the acoustic grid contains only the
   SVG walls and explicitly modeled moving arcade objects. Left/right
   microphone positions are interpolated over each GPU audio batch.
   **Trade-off:** removing the artificial funnel also removes its baffle
   directivity. The current source is approximately isotropic in the plane.
   A physically justified moving *directional source* or baffle deserves a
   separate numerical validation effort.
2. **3D presentation:** flip grid Y into world negative Z throughout the
   static mesh, camera and moving entities. Positive A/left rotation now
   turns left. Roof faces are separated slightly from the coplanar ceiling,
   avoiding depth-buffer z-fighting; sphere subdivision is increased and
   orb shading avoids per-pixel discontinuous procedural texture stripes.
3. **Contact feedback:** the contact point is chosen from an obstructing
   cell near the collision ring, instead of the old averaged collision normal.
   Synthetic stereo taps use ~16cm ear separation to derive ear-specific
   distance/gain and arrival-time differences. Rear contacts have a muted,
   lower-frequency spectral cue because simple stereo level/time cannot
   distinguish exact front from exact back. This is **not an HRTF**. Short
   event-triggered taps replace continuous 100ms 440Hz/330Hz collision and
   sliding tones that could sound everywhere.
4. **Close discovery:** O/I/W hints are gated to 24/30/35cm, straight-line
   unoccluded proximity, hysteresis on leaving and 4.5-second reminders.
   The opt-in F5/F6 UI beacons are off by default and constrained to close
   targets. Direct feedback remains game audio, not wave-simulated sound.

CPU regression tests exercise contact-point angle, left/right ear-level
difference, sample delays, rear distinctiveness, repeated collision gating,
short-range and occluded discovery, 3D handedness and ceiling separation.
**CUDA compilation alone cannot verify perceptual accuracy, acoustic
transient suppression or an audible difference.** Compare with previous
recordings and listen while moving and turning on a physical GPU before
accepting these changes.
