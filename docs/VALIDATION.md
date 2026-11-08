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
