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

Enemy sound-attraction decisions, optional stereo threat heartbeat/target
guidance and brief pickup/win feedback are game logic and UI cues, not
modeled pressure-field sources or 3D binaural recordings. However the
ghosts, shards, exit and SVG O/I/W markers now ALSO emit distinct soft
oscillating pressure sources **inside the actual FDTD solver**, so their
environmental sound does follow simulated 2D scattering and propagation. The manual-click F and
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
   **Trade-off:** removing the artificial funnel removes its hard-wall
   scattering. A phased two-point FDTD source creates forward directivity
   instead, but it is NOT a substitute for a rigorously simulated head
   with recessed, acoustically shielded microphones. The cyan wedge is
   now explicitly a non-solid aim indicator.
2. **3D presentation:** flip grid Y into world negative Z throughout the
   static mesh, camera and moving entities. Positive A/left rotation now
   turns left. Roof faces are separated slightly from the coplanar ceiling,
   avoiding depth-buffer z-fighting; sphere subdivision is increased and
   orb shading avoids per-pixel discontinuous procedural texture stripes.
3. **Contact feedback:** the contact point is chosen from an obstructing
   cell near the collision ring, instead of the old averaged collision normal.
   Synthetic stereo taps use 9cm ear separation, 6.5cm behind the pivot to derive ear-specific
   distance/gain and arrival-time differences. Rear contacts have a muted,
   lower-frequency spectral cue because simple stereo level/time cannot
   distinguish exact front from exact back. This is **not an HRTF**. Short
   event-triggered taps replace continuous 100ms 440Hz/330Hz collision and
   sliding tones that could sound everywhere.
4. **Close discovery:** the artificial O/I/W stereo chirps were removed,
   replaced by *FDTD-emitted* 85ms tapered notes with a hard eligibility
   condition based on close distance and static-wall line-of-sight.
   This gate is game logic, not predicted physical diffraction. The
   optional direct F5/F6 assists remain off by default. Outside of
   touch/score confirmations, the object's audio uses the wave field.

CPU regression tests exercise contact-point angle, left/right ear-level
difference, sample delays, rear distinctiveness, repeated collision gating,
short-range and occluded discovery, 3D handedness and ceiling separation.
**CUDA compilation alone cannot verify perceptual accuracy, acoustic
transient suppression or an audible difference.** Compare with previous
recordings and listen while moving and turning on a physical GPU before
accepting these changes.


## Forward FDTD sound, "echo only" and persistent arcade reflectors

The previous mouth wedge was a moving acoustic hard-wall baffle surrounding
the click source; it was intended to shelter the microphones but introduced
binary boundary jumps during locomotion. It is now visual only, narrowed
to a **32° full opening** and moved closer to the ears. The default player
click uses **three lateral FDTD columns**, -8/0/+8cm from the player's
forward axis, with front sources +6cm ahead of their delayed,
negative-pressure rear partner. Pair weights are 0.25/0.50/0.25.
Each of the six elements is an actual source in the pressure solver,
not a ray cone, rendering filter, acoustic shield, or rigid body.
The rear interference suppresses backwards radiation; the lateral
aperture suppresses sideways radiation. The old wedge is not a
physical object and therefore cannot leak through a 'solid' it no
longer represents.

In the prior iteration, a *single* front/back pair improved the rear
field but barely narrowed the sides. In a new dependency-free CPU
FDTD polar test 42cm from a straight-facing source, the six-emitter
array gave forward energy **0.7516**, side (90°) **0.00919**, and
rear **0.00254**, whereas the central 2-element endfire pair alone
gave side energy **0.4639**. Thus the sidelobe reduction is measured
in the simulated pressure field itself. These bench values depend
on the specific 1.8kHz source, numerical grid and probe geometry;
they are not a claimed real-world directivity index or proof of a
perfectly narrow beam in a reverberant 5×5m room.

The earlier microphones were lateral to the array and sometimes saw
MORE direct pressure than an isotropic source. They were first moved
8cm back and ±6cm sideways. At the user's request they now sit
**6.5cm back and ±4.5cm lateral**, only 9cm apart, with the visible
wedge tip 2cm in front of them. The microphones remain within the
player's 13cm collision radius. The incident-field J reference has
been updated to include all six emitters and the new microphone
positions. There is still NO impenetrable soundproof wedge or
physical head scattering: such moving solid geometry needs a
careful stable subcell boundary solver to avoid recreating whipping.

The optional **J** mode implements linear incident/scattered pressure
decomposition at the *receivers*. For each player pulse, it computes the
homogeneous free-field wave on a separate small CPU grid with the same
source shape and samples it at the two virtual ears. These samples are
subtracted from the main solver microphone samples before audio-rate
resampling. The main FDTD grid still propagates the entire direct wave,
so obstacles reflect normally. No physically impossible "ignore pressure
before reflection" algorithm is used. The reference only lasts around
3.3ms and can mismatch during position changes; longer free-field tails,
moving obstacles and GPU/CPU precision differences may leave direct
signal artifacts. Computing the reference on CPU at a pulse start may
delay frames, hence J is deliberately OFF by default.

Game acoustic models now use clearly visible **hexagonal 10cm-radius
ghosts** and **octagonal 7cm-radius shards**, with matching 3D models.
Ghosts now spin visually at about 4.8–6.2 radians/sec and choose
a new wandering heading every 0.20–0.55s. Alert lasts only 0.65–0.8s
following a nearby click, not 3–6s. Their FDTD mask retains a fixed
rotational orientation to avoid rapid, discontinuous moving-wall
pressure changes; 2D has a small spinning ornament inside the
reflective hexagon while 3D spins the actual rendered mesh.
The 2D renderer previously drew an additional 8px square GL_POINT on
top of each reflector, making coins appear misshapen. Those duplicate
squares are removed. Ghost hits now cause an at-most-1.5cm bump and a
tiny contact signal, with neither death nor score loss. The game
cannot lose lives or freeze its AI from repeated ghost contact.
Moving binary reflectors CAN STILL introduce pressure discontinuities
as their boundaries cross grid cells. Numerical moving-medium treatment,
subcell geometry/impedance interpolation and real hardware artifact
tests remain future work.

The previous **continuous oscillating sources** were a design regression:
with 2D geometric spreading and bounded-room resonances, they could
create a whole-level spaceship hum that barely changed through walls.
Quiet 85ms chirps now enter the SAME FDTD pressure update through a
sparse source map, but CPU gating permits their emission only within
48cm for ghosts, 42cm for shards, 50cm for the exit, and 35–50cm for
SVG marker types. A clear line of sight through the **static** wall
geometry is also required. Their amplitude ramps quadratically to
zero near the maximum distance. Physical 2D propagation, echoes
and occlusion still apply *after* these game-logic choices. This is
not an approximation of natural 3D attenuation and it will not model
diffraction around hidden corners correctly. Press N to A/B source
audio against echo-only exploration (game-play mode F4 remains
independent). Direct event feedback and wall-contact taps remain synthetic.

The 2D visualization uses bounded logarithmic exposure between
-0.45 and +0.45 so pressure values cannot overlap negative-coded
map object markers. Keys 9/0 adjust visualization only, not the
physical pressure state. Tests verify numeric bounds and monotonicity.


## Audio artifact diagnostics

Every five seconds the program logs `[AUDIO-DIAG]` with raw pre-limiter
headphone peak, sample count exceeding 0.95, and the number of rendered
frames where the SDL audio queue fell below 2ms. A large raw peak
with a healthy queue suggests pressure-grid artifacts; low-queue warnings
without pressure spikes suggest GPU/CPU scheduling and underruns. Neither
diagnostic alone diagnoses a rare whipping noise, but comparing B/N/F4/J
modes while capturing this line can narrow it down. All new numerical
checks are still CPU/compilation tests; no live NVIDIA listening result
is available from the automated runner.
