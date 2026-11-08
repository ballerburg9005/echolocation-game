# Echo Arcade: Ghost Circuit

This branch introduces a playable experiment around the original 2D CUDA wave
solver. It is a game, not a validated mobility or echolocation training tool.

## Quick start

On Linux with NVIDIA CUDA, SDL2, OpenGL 3.3, GLEW, tinyxml2 and GLM headers:

    make test
    make build
    ./build/echolocation_game echo_arena.svg

Existing SVG levels also work. Arcade generates up to 12 shards and an exit
inside reachable passages. An authored SVG W text marker overrides the
generated exit if its location is walkable.

## Mission

The maze has three wandering ghosts and up to twelve echo shards.
Ghost contact is nonlethal; there is no lives-based game-over state.
The room measures 5×5 metres (500×500 cells at 1 cm per cell).

- Find all shards. They are **5.5cm-radius eight-sided octagons**,
  still smaller than the original 7cm radius but easier to perceive
  than the previous 4cm experiment. They're vivid lime green in
  BOTH 2D views and 3D, with a small inset white marker visibly
  orbiting as they spin at 5.2–6.1rad/s. They rotate in the
  **actual FDTD impedance field** too, with exactly the
  ghosts' reflection strength and 55ms sub-cell smoothing.
  Their own 85ms FDTD notes now use more audible **330–587Hz**
  pentatonic pitches when within 42cm and unobstructed.
  The generous 15cm pickup radius is unchanged for blind play.
  Optional direct guidance is off by default.
- Ghosts are **nonlethal solid bumpers**, not hazards. Their 10cm-radius,
  six-sided reflective bodies change the wave mask and move through
  accessible paths. Close, unobstructed encounters trigger distinct
  short FDTD chirps, never a whole-room oscillator.
- Ghosts have distinct roles: a medium-speed violet **tracker**, a fast
  orange **sprinter**, and a slower blue **sentinel** with longer hearing
  range. Hearing follows reachable corridors rather than a direct radius
  through sealed walls. Ghosts pursue the **last audible source** instead of
  receiving omniscient updates about the player's position.
- Throw a **decoy** with G. A second tapered pulse is emitted from the
  projected safe cell inside the real CUDA solver, luring ghosts away from
  the player. It stops at static maze walls; the cooldown is 9 seconds.
- Ghosts wander with rapid changes of heading (every 0.20–0.55 seconds)
  until a nearby click alerts them. After **0.65–0.8 seconds**
  without another local click, they abandon pursuit and resume wandering.
  Their hexagons rotate rapidly (up to about 355° per second) in both
  the visuals AND the FDTD wave model. Every ghost has a rotating
  2×2-supersampled polygon impedance map, smoothed over ~55ms so
  turning never instantaneously creates hard zero-pressure cells.
  This is approximated moving acoustic reflectivity, not a measured
  rigid-body boundary condition. A contact gently displaces the player at most
  1.5cm and triggers one subdued cue with 0.9s cooldown. It does not
  deduct score/lives, end gameplay, freeze AI or disable pickups.
  Reflectors never teleport away as a penalty.
- Shards award 100 points each. An unlocked exit awards 500 and plays
  a victory sound. Press R to restart.
- Every four collected shards earn one **sonic overdrive charge** (up to
  three held). Press H to spend one. A normal physical FDTD ping is emitted,
  and ghosts become immobile and harmless for five seconds. Touch a frozen
  ghost to capture it for 200, then 400, then 800, then 1600 points within
  the same overdrive; a captured ghost temporarily disappears from the sound
  reflector mask and respawns when the effect ends. This is a game power-up,
  not a predicted consequence of sound exposure.

## Controls

| Input | Action |
|---|---|
| W/S or Up/Down | Walk forward and backward |
| A/D | Strafe left and right |
| Q/E or Left/Right | Turn left and right |
| F | Emit a manual echolocation click |
| T | Toggle automatic clicks; off means stealth/manual only |
| G | Throw a remote physical echo decoy, up to 85 cm ahead (9s cooldown) |
| C | Sweep a synthetic left/center/right stereo cane (100 cm range) |
| H | Spend one sonic overdrive charge to freeze and catch ghosts for five seconds |
| F2 | Toggle right 2D/3D viewport |
| F3 | Toggle immersive full-window 3D |
| F4 | Arcade on/off, preserving pure FDTD navigation mode |
| F5 | Opt into nearby directional enemy heartbeat (off initially) |
| F6 | Opt into nearby, unobstructed target discovery chime (off initially) |
| B | Toggle narrow six-source forward FDTD beam (default) / isotropic monopole |
| N | Toggle nearby, intermittent FDTD object/landmark chirps (on by default) |
| J | Toggle experimental echo-only receiver subtraction (off by default) |
| 9/0 | Lower/raise 2D wave-display exposure (does not alter physics) |
| R | Restart level and clear acoustic state |
| Space | Toggle object visibility in original 2D view |
| 1/2 | Pulse interval |
| 3/4 | Artificial wave damping per solver step |
| 5/6 | Approximate wall reflection |
| 7/8 | Wave playback gain |
| Escape | Quit |

### Sound types

**Physics-based:** the player's 1.8kHz tapered click excites the 2D FDTD
field. Default mode has **three separate phased front/back pairs**,
spaced -8, 0 and +8cm laterally, with weights 0.25, 0.50 and 0.25.
Each rear source emits an inverse click delayed by its source separation
divided by the speed of sound. The 16cm aperture suppresses sideways
energy; the delayed rear sources suppress backward energy in the same
actual FDTD pressure grid. No rays are discarded or wedge solids moved.
The B key selects a single isotropic source for comparison.
This is NOT an impenetrable head or a calibrated bat beam.

The tracker/sprinter/sentinel and candy have different *actual pressure
field emitters*, placed in nearby fluid cells just outside their
rotating reflective polygon outlines. The candy sound uses the same rotating-scatterer reflection strength
as ghosts but a smaller eight-sided geometry and a distinct, lower
330–587Hz pentatonic source note. Generated exits and SVG O/I/W markers also
have distinct frequencies. Emission is a short 85ms Hann-tapered chirp
rather than a continuously driven spaceship-like oscillator, spaced
approximately 2.15s apart. The CPU enables a source only while the
player is close with clear static-wall line of sight: ghosts 48cm,
shards 42cm, exits 50cm, SVG markers 35–50cm. Amplitude fades
quadratically near the maximum range. This range/occlusion *eligibility*
is an intentional gameplay heuristic: it does NOT simulate correct
3D distance attenuation or diffraction. Once emitted, the chirp
propagates and reflects in the genuine 2D FDTD pressure field.
N disables the optional room emitters to isolate physical clicks.
Static SVG walls remain hard mask materials; moving 6/8-sided
ghosts/coins are smoothed, fractionally reflective FDTD materials.
They do not abruptly zero pressure when rotating or being collected.

**Game-only synthetic audio:** brief score/pickup/touch/victory event
confirmations, optional F5/F6 UI assistance, C-key navigation cane, and
short directional wall-contact taps still bypass FDTD. O/I/W landmarks
NO LONGER generate separate proximity chimes: their gated intermittent
chirps are in the pressure field instead. F5 and F6 are OFF by default;
if enabled, F6 is silent beyond 35cm and around walls, and F5 beyond
55cm. A regular nonphysical whole-level beacon is not played.

**Wall contact:** a single short, gentle **820Hz beep** is used for every
wall touch around the full 360° of the player's *13cm collision circle*.
The actual contact-cell bearing drives left/right relative intensity
and sub-millisecond ear delay using the correct 9cm ear spacing,
6.5cm behind the pivot. A tiny continuous high-frequency shading
variation helps distinguish front from rear while preserving the SAME
basic beep—no separate special forward wedge "pang" sound.
Contacts are gated by the actual world-space wall direction, so holding
against a wall while turning cannot trigger fake new collisions.
These are game accessibility cues, not a measured HRTF; exact front/back
localization on stereo headphones remains approximate.

**Direct sound, head shielding, and echo-only monitoring:** the old
visual player wedge was a crude soundproof baffle intended to keep the
ears behind the outbound pulse. Its rotating hard mask generated abrupt
acoustic boundary changes (audible whipping/rattling), so it is no
longer stamped into the FDTD solver. Its visible 2D outline remains a muted-cyan **non-solid 32° aiming
guide**, clearly different from yellow reflective walls. Previously
the full visible opening was 76°. The wedge tip is 4.5cm behind the
pivot; the two virtual ears sit just 2cm behind it, at -6.5cm
longitudinal and ±4.5cm lateral, 9cm apart. Ears remain inside the
13cm collision radius.

**Near-ear source geometry matters:** the historical ears lay to the
sides of the original phased pair, where interference could increase
direct sound. Moving them behind the source improves suppression.
The new narrow FDTD beam further suppresses the sides.
A 42cm-radius CPU 2D pressure benchmark for a straight-ahead source
measured forward energy 0.752, side energy 0.00919 and rear 0.00254.
The old endfire pair gave 0.464 side energy in that same benchmark.
These are *not* calibrated directional hearing or a soundproof head;
the wedge is still only a visual aid. Restoring a fast-moving binary
hard wedge would recreate nonphysical pressure jumps.

J enables *experimental scattered-field monitoring*: a small free-field
CPU FDTD computes the identical incident pulse and subtracts it from
microphone samples on the numerical time grid before resampling. The
real world solver still evolves total pressure, so scattering from every
wall and object persists. Subtraction is only approximate if the player
moves, source quantization differs, or reference calculation stalls GPU
audio playback; keep J OFF unless evaluating it. This is mathematically
more coherent than simply suppressing all sound for the first 3ms, since
nearby reflections occur during that window.

**Stealth and decoys:** T disables automatic player pulses, not
numerical timekeeping. F triggers a player click on demand; G sends a
secondary physical click from a forward point not hidden inside an SVG
wall. A nearby enemy follows a path toward the last source it heard,
making diversion possible. This hearing check uses a coarse 5-cm
walkable-path approximation: it is a gameplay heuristic, not a validated
sound-pressure or diffraction calculation.

**Sonic overdrive:** the H-key activation ping uses the same FDTD player
source as an ordinary click. Ghost freezing, invulnerability, score combos
and respawning are explicitly game-only mechanics. Enemies are still
reflective while frozen, but captured enemies vanish from the wave mask
until overdrive ends.

**Cane:** the C-key probe measures exact collision-mask distances along
three rays (left, forward, right), then generates three successive synthetic
stereo taps. Their volume and pitch encode how close the obstacle is.
This is not a physical cane model, reflected sound, or haptic sensation.

Turn F5/F6 off when experimenting with reflection-only navigation, or leave
them on for more accessible gameplay. Start headphones at a low volume.
These cues are unsuitable for real-world navigation.

## 3D presentation

The historical experimental branch included a first-person 3D renderer.
This version restores the idea with a new smaller and testable mesh pipeline.

- Each wall is extruded from the same SVG rasterization used in collision
  and acoustic propagation. Walls therefore occupy the same planar cells.
- OpenGL 3.3 draws a right-handed perspective camera, directional
  lighting, subtle procedural materials and glowing 3D entities.
  Ghost meshes use obvious six-sided contours and the smaller candies
  use eight-sided contours; both spin with their planar FDTD
  reflector geometry.
  Grid +Y maps to GL -Z, so player turns match the 2D view. The
  ceiling is separated from obstacle roofs to prevent coplanar
  z-fighting stripes on round shapes; curved orbs use smoother shading.
- The original two-panel 2D diagnostic view remains available.
  Its signed wave colors now use bounded log exposure to avoid hard
  tanh saturation or collisions with negative glyph sentinel codes;
  9/0 only adjusts visualization, not pressure.
- This is **2.5D graphical extrusion, not 3D wave simulation**. No sound
  elevation, ear pinna filtering, head scattering or HRTFs are modeled.

## Tests

The standard C++17 tests validate wave math and resampling, stable
impedance boundaries, collision and sliding, SVG rasterization, generated
3D geometry, deterministic arcade spawns, score and nonlethal bumps,
enemy occupancy,
and stereo audio-cue direction. The CUDA Docker workflow compiles the full
application; it cannot exercise real GPU audio playback.

## Open questions

- Real-time throughput and actual audio-device latency need profiling on
  representative NVIDIA hardware.
- Ghosts and the smaller eight-sided candies use rotating
  **fractional impedance** instead of abrupt moving hard solids.
  Their 2x2 sub-cell coverage is smoothed over ~55ms, avoiding
  abrupt pressure resets on rotation and collection. Candy now has
  the SAME effective reflection strength as a ghost; the smaller
  geometry and lower emitted notes give it a different signature.
  This is not yet true rigid-body scattering or a conservative
  moving-boundary solver. The old moving wedge remains absent.
- The optional J incident-field reference uses CPU time per click. Under
  load it could cause audio underruns; A/B test with J off and on before
  considering a default-on echo-only mode.
- Ghost sound detection and 5 cm grid pathfinding are heuristics, not
  perceptually or scientifically validated acoustics.
- Decoy clicks run through the numerical solver, but decoy targeting and
  enemy detection are rules. The C-key cane is deliberately synthetic
  navigational sonification.
- Accessible onboarding, spoken instruction, remappable keys and evaluation
  with blind players are not implemented yet.


## Recorded audio inside FDTD (feasibility only)

The same pressure solver could inject arbitrary bandlimited samples
from an actual WAV recording instead of only synthesized pulses.
At 1cm cells, roughly ten spatial samples per wavelength imply an
upper reasonably resolved band near 3.4kHz, so a first experiment
would filter to 2.5–3kHz, preload mono samples into GPU memory,
move the source position smoothly, and read the resulting sound from
the existing two virtual microphones. Source motion left/right of
the listener would then be FDTD, not external stereo panning.
A 2x2m floor-plan room at 1cm cells would cost roughly one-sixth of
the current 5x5m planar grid. No ceiling, floor, real head scattering
or individualized HRTFs are modeled; power draw alone does not
validate performance on weaker GPUs. This is not implemented yet.
See [WAV-in-FDTD design](RECORDED_AUDIO_FDTD.md).


## Asymmetric room and outer-surface acoustics

The lower-right section of `echo_arena.svg` now contains a stepped
five-part zig-zag reflecting wall. This uses only supported SVG rectangles
and is a real 2D FDTD obstacle. A 5cm-spaced walkable-path regression uses
the actual 13cm-radius collider and confirms a route between the start
and the last room; a player never needs to squeeze through the zig-zag.

The outer **2cm ring** remains solid for movement and field occupancy,
but its wave interface uses a fixed lower pressure-reflection coefficient
`R=0.32` (broadband, impedance-loss approximation; not a measured rubber
frequency response). Interior SVG walls remain set by keys 5/6.
The outer wall is drawn muted blue in 2D and 3D, interior walls
remain their old color. This does not reduce the grid size, temporal
resolution, or interior reflection accuracy.

## Turning crackle diagnosis

All six player click sources now use continuously placed sub-cell
pressure injection with normalized bilinear weights and fractional
front/rear propagation delays. The earlier rounded integer emitters
changed placement/discrete delay abruptly as the player rotated.
The J experimental direct-reference model is matched to the new
in-field emitter geometry. This is NOT a blanket audio filter,
reduced resolution, or echo deletion. A free-field CPU sweep over
24 headings showed near-ear click peaks of ~0.185–0.192.
`[AUDIO-DIAG]` now reports raw **MaxJump, SharpJumps and JumpAngle**,
in addition to sample peaks, clipping counts and SDL queue underflows.
If loud whip noises continue, please compare T (auto-clicks off),
B (monopole versus beam) and N (object emitters off) and
share those diagnostic lines; audio underruns or geometry motion can
still cause other artifacts.
