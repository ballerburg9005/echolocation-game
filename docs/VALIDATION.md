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
Enemy/coin rotating impedance and static wall reflection are local
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
3. **Contact feedback:** the contact point is selected from an obstructing
   cell near the 13cm collision circle; **the decorative wedge does not
   collide**. The same fundamental 820Hz short beep is used for all
   360 degrees. Ear gains and arrival times use 9cm interaural separation,
   6.5cm behind the pivot. A mild *continuous* spectral variation by
   cos(relative bearing) helps with the otherwise ambiguous front/back
   pair; there is no separate forward "wedge pang". World-space
   wall-bearing gating prevents new taps merely from turning at a
   stationary collision. This is **not an HRTF**. Tests check the
   entire 16-bearing compass, front/rear pitch consistency, and
   wedge geometry entirely within the 13cm circle.
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

Game acoustic models use **hexagonal 10cm-radius ghosts** and now
**octagonal 5.5cm-radius candies**, with matching spinning
3D models. The candy collection radius remains a forgiving 15cm.
Ghosts spin at about 4.8–6.2 radians/sec and choose a new
wandering heading every 0.20–0.55s. Alert lasts only 0.65–0.8s
following a nearby click. Their REAL FDTD reflector polygons
now rotate at the same rate as the visuals. Instead of writing a
binary wall-id 2 then zeroing every newly solid cell's pressure,
a separate 2×2 subcell coverage map records 0..1 material fraction
and an exponential ~55ms filter limits how much each cell changes
per display frame. Fractional reflectivity is now stored in a single existing CUDA
float mask per cell: negative values between -1 and 0 encode ghost
coverage, whereas values between -2 and -1 encode spinning candy
coverage. Static SVG walls stay positive hard materials. The CUDA
stencil blends fluid-neighbor coupling toward the ~98% reflecting
ghost condition, or the SAME ~98% reflecting candy condition,
continuously. This does not allocate an additional GPU grid buffer
or launch a separate device kernel. Two extra CPU-side target/smoothed
coverage arrays do add an O(grid size) update once per display frame. The field retains pressure history even *inside*
the ghost. This is NOT a fully physical moving rigid-boundary
solver: time-dependent impedance may still alter energy, so
listen for transients and benchmark numerical stability.
The 2D renderer previously drew an additional 8px square GL_POINT on
top of each reflector, making coins appear misshapen. Those duplicate
squares are removed. Ghost hits now cause an at-most-1.5cm bump and a
tiny contact signal, with neither death nor score loss. The game
cannot lose lives or freeze its AI from repeated ghost contact.
The candy octagons now rotate at a more visible 5.2–6.1rad/s,
with 8-fold symmetry, retaining their pressure history; collection
fades out soft occupancy. The FDTD reflectivity is the same as
ghosts', not reduced. The 11cm-diameter candies have bright
lime silhouettes and a white rotating marker in both 2D panels and
3D. Their infrequent, distance-gated in-FDTD notes have been
lowered to 330–587Hz with a moderate source-amplitude increase
from 0.15 to 1.2 pressure units at full strength. This is a deliberately
small, friendlier cue and may be less audible farther away than
the larger ghost scatterers. Numerical moving-medium momentum,
energy-conserving subcell impedance and hardware listening tests
remain future work.

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


## Rotating reflective ghosts and continuous 360-degree wall cue

The actual FDTD ghost reflector map now rotates with each ghost's visual
hexagon. A four-subcell coverage raster is filtered with a 55ms
exponential response, encoded as negative mask values, and represented
by symmetric fractional zero-gradient edge coupling in the wave stencil.
The player's collision circle and the SVG wall boundaries still use
hard blocking geometry; they are NEVER inferred from the cyan wedge.
This is preferable to rotating hard cell masks, which previously erased
pressure fields during movement, but still requires stability and
perceptual profiling on real GPU hardware. Unit tests demonstrate that
rotating the hexagon changes the resulting FDTD field and that the
fractional interface does not zero pressure in place.

The collision cue has a consistent 820Hz tone throughout 360 degrees.
Left/right source cues reflect each virtual ear's distance and arrival
time, while a subtle treble shelf varies *continuously* with azimuth
frontness. There is no loud forward-only pang. A full 16-point compass
regression tests contact localization, front/back pitch consistency,
and sound gating by world contact direction. This is not pinna/HRTF
rendering; ambiguous front/back localization may remain.


## Arbitrary recorded audio: possible, not yet implemented

A wave-source term is not limited to oscillators, clicks, or preset
frequency bands. Feed an arbitrary PCM sequence (for example decoded
from a WAV), resampled onto the ~48.5kHz simulation time grid and
bandlimited according to the 1cm **spatial** grid, directly into the
existing pressure kernel at a fluid source cell. The headphones
then sample that field at two independently located virtual ears.
Source motion from one side of the head to the other must be smoothed
and bilinearly distributed to nearby fluid cells to avoid new pops.
With 343m/s sound speed, 10 cells/wavelength corresponds to ~3.4kHz;
a conservative first test should low-pass to 2.5–3kHz. 2D propagation,
missing floor/ceiling and no HRTF prevent true room- or human-ear
fidelity, but the mechanism is physically meaningful as a 2D demo.
The 500² × ~48.5k pressure-cell updates/s are already the primary
simulation load; a *preloaded* short mono sample accessed at one
spatial source adds modest arithmetic without another full-grid
kernel, but **requires actual GPU profiling**. See
[WAV source design](RECORDED_AUDIO_FDTD.md).


## Angular-source anti-whip changes and distinctive room

The six pressure emitters were previously rounded to integer 1cm
grid positions at each video frame and their delayed negative
elements used rounded integer travel time. Turning through a cell
boundary could produce sudden differences in the direct wave.
The FDTD pressure update now injects the identical tapered
click using **normalized bilinear 2x2 source weights** at continuous
player-relative coordinates, and interpolates the rear anti-phase
sample at a fractional travel-time delay. These changes preserve
the same 1cm grid, Courant number, source spectrum, and reflected
pressure field. On CUDA, the computation is gated to within 1 grid
cell of each emitter so every other pressure cell avoids the expensive
floating-point work. The optional J-mode CPU direct reference
also uses exactly those fractional positions and delays.

The 360-degree source tests confirm continuous placement without
1cm teleporting. The CPU free-field FDTD sweep across 24 heading
angles observed near-ear peaks 0.185–0.192 (with 750-unit
input and 0.01 playback gain), rather than significant
orientation-dependent amplification. This does **not** prove all
hardware cracks are eliminated; GPU scheduling, spatial-microphone
motion or residual moving-impedance artifacts still need listening.
The output samples are NOT high-cut or muted to hide these sounds.
The diagnostic log now includes raw inter-sample MaxJump, SharpJumps
and the angle of the worst sample jump, alongside peak, limiter
and SDL underrun counts.

The new five-step zig-zag in the lower-right room is rasterized
as a set of static reflecting SVG rectangles. A CPU route test
uses the true 13cm player collision and checks that the corridor
remains traversable. The outer 2cm wall boundary is a separate
material (mask value 4) with fixed broad-band pressure-reflection
coefficient 0.32, using the same centered-time impedance boundary
calculation as inner walls. Maze walls retain adjustable reflection.
A material regression tests the loss difference and that zero-
velocity/uniform pressure fields stay unchanged without numerical
injection. The muted-blue outer ring is visually distinguishable
from inner walls in both 2D and 3D. This is rubber-*like loss*,
not a calibrated viscoelastic frequency-dependent surface.
