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

- Find all shards. Their 7cm-radius **eight-sided** masks reflect wave
  energy in FDTD. They emit quiet 85ms chirps only while within 42cm
  in unobstructed maze space. Optional direct guidance is off by default.
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
- Ghosts wander until a nearby click alerts them, then pursue along a
  5cm-grid shortest path. A contact gently displaces the player at most
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
| A/D or Left/Right | Rotate |
| Q/E | Strafe |
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
| B | Toggle player FDTD click: forward phased pair (default) / isotropic monopole |
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
field. With B in its default forward mode, two pressure sources separated
by about 6cm emit equal and opposite pulses, with the rear negative
source delayed by the physical transit time. Far-field radiation behind
the player is consequently reduced by destructive interference, without
moving the wedge's hard binary boundary. This is NOT a perfect bat beam or
an impenetrable head. Both virtual microphones sample the FDTD pressure.

The tracker/sprinter/sentinel and candy have different *actual pressure
field emitters*, placed in nearby fluid cells just outside their
reflective polygon outlines. Generated exits and SVG O/I/W markers also
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
Static walls and 6/8-sided moving objects are genuine mask materials.

**Game-only synthetic audio:** brief score/pickup/touch/victory event
confirmations, optional F5/F6 UI assistance, C-key navigation cane, and
short directional wall-contact taps still bypass FDTD. O/I/W landmarks
NO LONGER generate separate proximity chimes: their gated intermittent
chirps are in the pressure field instead. F5 and F6 are OFF by default;
if enabled, F6 is silent beyond 35cm and around walls, and F5 beyond
55cm. A regular nonphysical whole-level beacon is not played.

**Wall contact:** collision taps are computed from the solid cell touched
on the player's 13cm collision circle, not a single averaged obstacle
bearing. Approximate 16cm ear spacing causes different arrival delays
and intensity at each ear, while a darker rear tap distinguishes front
and rear centre. Holding forward into a wall no longer repeats the beep;
there is no continuous sliding tone. These are short synthetic cues, not
a measured binaural or material-contact impulse response.

**Direct sound, head shielding, and echo-only monitoring:** the old
visual player wedge was a crude soundproof baffle intended to keep the
ears behind the outbound pulse. Its rotating hard mask generated abrupt
acoustic boundary changes (audible whipping/rattling), so it is no
longer stamped into the FDTD solver. Its visible 2D outline remains as a muted-cyan **non-solid aiming
guide**, clearly different from the yellow reflective walls.
Instead B selects a phased directional source or monopole.

**Actual ear geometry mattered:** the earlier microphones were
approximately level with the rear phased element but to its sides,
where interference could make the direct sound *stronger*. Each receiver
is now 8cm behind the pivot and 6cm to its corresponding side, fully
within the player's 13cm collision radius. A CPU test with these exact
positions measures phased-pair direct-click energy equal to 0.0775
times the single-emitter energy (about 92% lower at the virtual ears).
The earlier 98% figure measured a much more distant rear probe and
was not an honest prediction of ear shielding. The wedge still does
not physically block pressure in the room model; the source and
microphone geometry reduce the direct field without abruptly moving
a binary wall. This is *not* a calibrated scattering human head.

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
  Ghost meshes use obvious six-sided contours and candy eight-sided contours,
  matching their planar reflector mask facets.
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
- Moving solid enemy/shard 6/8-gon masks can still create pressure
  artifacts at interfaces; proper moving-boundary momentum treatment
  is future work. The old moving wedge mask has been removed.
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
