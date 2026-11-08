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

The maze has three enemies, three lives, and up to twelve echo shards.

- Find all shards. Their 5 cm radius masks reflect wave energy in the FDTD
  simulation, and optional guidance is available.
- Avoid enemies. Their 8 cm radius bodies change the wave mask and move through
  accessible paths. Each enemy has an approximate reflection coefficient.
- Ghosts have distinct roles: a medium-speed violet **tracker**, a fast
  orange **sprinter**, and a slower blue **sentinel** with longer hearing
  range. Hearing follows reachable corridors rather than a direct radius
  through sealed walls. Ghosts pursue the **last audible source** instead of
  receiving omniscient updates about the player's position.
- Throw a **decoy** with G. A second tapered pulse is emitted from the
  projected safe cell inside the real CUDA solver, luring ghosts away from
  the player. It stops at static maze walls; the cooldown is 9 seconds.
- Ghosts wander until a nearby click alerts them, then pursue via a 5 cm
  navigation-grid shortest-path field. An encounter removes one life,
  deducts up to 150 points, allows two seconds of invulnerability, and
  respawns enemies at their starting positions.
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
| F5 | Enable/disable directional enemy heartbeat |
| F6 | Enable/disable stereo next-target beacon |
| R | Restart level and clear acoustic state |
| Space | Toggle object visibility in original 2D view |
| 1/2 | Pulse interval |
| 3/4 | Artificial wave damping per solver step |
| 5/6 | Approximate wall reflection |
| 7/8 | Wave playback gain |
| Escape | Quit |

### Sound types

**Physics-based:** the 1.8 kHz tapered source excites the 2D FDTD field.
Static SVG walls, moving enemies and uncollected shards have real occupancy
in this simulation; two virtual microphones sample the resulting pressure.

**Game-only synthetic audio:** SVG O/I/W markers, item/damage/victory jingles,
enemy heartbeat (F5), and stereo target guidance (F6) do not enter the FDTD
field. They are deliberately synthesized and panned; they do not travel
through walls, apply HRTFs, or follow real room reverberation.

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
- OpenGL 3.3 draws a perspective camera, directional lighting, subtle
  procedural materials and glowing 3D entities.
- The original two-panel 2D diagnostic view remains available.
- This is **2.5D graphical extrusion, not 3D wave simulation**. No sound
  elevation, ear pinna filtering, head scattering or HRTFs are modeled.

## Tests

The standard C++17 tests validate wave math and resampling, stable
impedance boundaries, collision and sliding, SVG rasterization, generated
3D geometry, deterministic arcade spawns, score and lives, enemy occupancy,
and stereo audio-cue direction. The CUDA Docker workflow compiles the full
application; it cannot exercise real GPU audio playback.

## Open questions

- Real-time throughput and actual audio-device latency need profiling on
  representative NVIDIA hardware.
- Moving solid masks can produce pressure artifacts at interfaces and should
  be checked against a moving-boundary reference solver.
- Ghost sound detection and 5 cm grid pathfinding are heuristics, not
  perceptually or scientifically validated acoustics.
- Decoy clicks run through the numerical solver, but decoy targeting and
  enemy detection are rules. The C-key cane is deliberately synthetic
  navigational sonification.
- Accessible onboarding, spoken instruction, remappable keys and evaluation
  with blind players are not implemented yet.
