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
- Ghosts wander until a nearby click alerts them, then pursue via a 5 cm
  navigation-grid shortest-path field. An encounter removes one life,
  deducts up to 150 points, allows two seconds of invulnerability, and
  respawns enemies at their starting positions.
- Shards award 100 points each. An unlocked exit awards 500 and plays
  a victory sound. Press R to restart.

## Controls

| Input | Action |
|---|---|
| W/S or Up/Down | Walk forward and backward |
| A/D or Left/Right | Rotate |
| Q/E | Strafe |
| F | Emit a manual echolocation click |
| T | Toggle automatic clicks; off means stealth/manual only |
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

**Stealth:** T disables automatic pulses, not numerical timekeeping.
F triggers a click on demand. Enemy awareness uses a simplified game rule
that checks the local click position. It is not a validated acoustic
receiver and may hear through some obstacles.

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
- Accessible onboarding, spoken instruction, remappable keys and evaluation
  with blind players are not implemented yet.
